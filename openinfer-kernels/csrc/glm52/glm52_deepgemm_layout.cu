#include "../common.cuh"

#include <cuda.h>

namespace {

constexpr int kTrtllmGroupedOffsetAlignment = 32;
constexpr int kLayoutThreads = 256;

__device__ __host__ __forceinline__ int trtllm_grouped_offset_padded_row(
    int row, int problem_idx) {
  return ((row + problem_idx * (kTrtllmGroupedOffsetAlignment - 1)) /
          kTrtllmGroupedOffsetAlignment) *
         kTrtllmGroupedOffsetAlignment;
}

// Grouped-offset scale relayout. The per-expert (dst_start, dst_end,
// src_start) ranges are staged once per block in shared memory: with a
// capacity-sized launch (~100k threads at bound_rows 2080 x 48 cols) the
// original per-thread scan over `expert_offsets` was 2 x groups global loads
// per thread and dominated the kernel. The scan semantics (first matching
// valid expert wins; anything else writes the 0.0 TMA padding) are unchanged.
__global__ void deepgemm_grouped_offset_tma_aligned_f32_kernel(
    const float* __restrict__ input, const int64_t* __restrict__ expert_offsets,
    float* __restrict__ output, int m_capacity, int scale_cols, int groups,
    int padded_rows) {
  // 3 ints per expert: dst_start, dst_end, src_start (invalid -> empty range).
  extern __shared__ int seg[];
  for (int expert = threadIdx.x; expert < groups; expert += blockDim.x) {
    int64_t src_start_raw = expert_offsets[expert];
    int64_t src_end_raw = expert_offsets[expert + 1];
    int dst_start = 0, dst_end = 0, src_start = 0;
    if (src_start_raw >= 0 && src_end_raw >= src_start_raw &&
        src_end_raw <= m_capacity) {
      src_start = static_cast<int>(src_start_raw);
      dst_start = trtllm_grouped_offset_padded_row(src_start, expert);
      dst_end = dst_start + static_cast<int>(src_end_raw) - src_start;
    }
    seg[expert * 3] = dst_start;
    seg[expert * 3 + 1] = dst_end;
    seg[expert * 3 + 2] = src_start;
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = padded_rows * scale_cols;
  if (idx >= total) return;

  int dst_row = idx % padded_rows;
  int col = idx / padded_rows;
  float value = 0.0f;
  for (int expert = 0; expert < groups; ++expert) {
    int dst_start = seg[expert * 3];
    int dst_end = seg[expert * 3 + 1];
    if (dst_row >= dst_start && dst_row < dst_end) {
      int src_row = seg[expert * 3 + 2] + (dst_row - dst_start);
      value = input[src_row * scale_cols + col];
      break;
    }
  }
  output[idx] = value;
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

bool valid_grouped_offset_shape(int m_capacity, int scale_cols, int groups,
                                int padded_rows) {
  return m_capacity > 0 && scale_cols > 0 && groups > 0 &&
         padded_rows == trtllm_grouped_offset_padded_row(m_capacity, groups);
}

}  // namespace

extern "C" {

CUresult glm52_deepgemm_grouped_offset_tma_aligned_f32_cuda(
    const float* input, const int64_t* expert_offsets, float* output,
    int m_capacity, int scale_cols, int groups, int padded_rows,
    cudaStream_t stream) {
  if (input == nullptr || expert_offsets == nullptr || output == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_grouped_offset_shape(m_capacity, scale_cols, groups,
                                  padded_rows)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int total = padded_rows * scale_cols;
  int blocks = (total + kLayoutThreads - 1) / kLayoutThreads;
  size_t smem = static_cast<size_t>(groups) * 3 * sizeof(int);
  deepgemm_grouped_offset_tma_aligned_f32_kernel<<<blocks, kLayoutThreads, smem,
                                                   stream>>>(
      input, expert_offsets, output, m_capacity, scale_cols, groups,
      padded_rows);
  return consume_last_cuda_error();
}

}  // extern "C"
