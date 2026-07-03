#include "../common.cuh"

#include <cuda.h>

namespace {

constexpr int kTmaAlignmentBytes = 16;
constexpr int kF32Bytes = 4;
constexpr int kF32TmaRowAlignment = kTmaAlignmentBytes / kF32Bytes;
constexpr int kTrtllmGroupedOffsetAlignment = 32;
constexpr int kLayoutThreads = 256;

__global__ void deepgemm_mn_major_tma_aligned_f32_kernel(
    const float* __restrict__ input, float* __restrict__ output, int rows,
    int scale_cols, int aligned_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = aligned_rows * scale_cols;
  if (idx >= total) return;

  int row = idx % aligned_rows;
  int col = idx / aligned_rows;
  output[idx] = row < rows ? input[row * scale_cols + col] : 0.0f;
}

__device__ __host__ __forceinline__ int trtllm_grouped_offset_padded_row(
    int row, int problem_idx) {
  return ((row + problem_idx * (kTrtllmGroupedOffsetAlignment - 1)) /
          kTrtllmGroupedOffsetAlignment) *
         kTrtllmGroupedOffsetAlignment;
}

__global__ void deepgemm_grouped_offset_tma_aligned_f32_kernel(
    const float* __restrict__ input, const int64_t* __restrict__ expert_offsets,
    float* __restrict__ output, int m_capacity, int scale_cols, int groups,
    int padded_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = padded_rows * scale_cols;
  if (idx >= total) return;

  int dst_row = idx % padded_rows;
  int col = idx / padded_rows;
  float value = 0.0f;
  for (int expert = 0; expert < groups; ++expert) {
    int64_t src_start_raw = expert_offsets[expert];
    int64_t src_end_raw = expert_offsets[expert + 1];
    if (src_start_raw < 0 || src_end_raw < src_start_raw ||
        src_end_raw > m_capacity) {
      continue;
    }
    int src_start = static_cast<int>(src_start_raw);
    int src_end = static_cast<int>(src_end_raw);
    int dst_start = trtllm_grouped_offset_padded_row(src_start, expert);
    int dst_end = dst_start + (src_end - src_start);
    if (dst_row >= dst_start && dst_row < dst_end) {
      int src_row = src_start + (dst_row - dst_start);
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

bool valid_layout_shape(int rows, int scale_cols, int aligned_rows) {
  return rows > 0 && scale_cols > 0 && aligned_rows >= rows &&
         aligned_rows % kF32TmaRowAlignment == 0;
}

bool valid_grouped_offset_shape(int m_capacity, int scale_cols, int groups,
                                int padded_rows) {
  return m_capacity > 0 && scale_cols > 0 && groups > 0 &&
         padded_rows == trtllm_grouped_offset_padded_row(m_capacity, groups);
}

}  // namespace

extern "C" {

CUresult glm52_deepgemm_mn_major_tma_aligned_f32_cuda(
    const float* input, float* output, int rows, int scale_cols,
    int aligned_rows, cudaStream_t stream) {
  if (input == nullptr || output == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_layout_shape(rows, scale_cols, aligned_rows)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int total = aligned_rows * scale_cols;
  int blocks = (total + kLayoutThreads - 1) / kLayoutThreads;
  deepgemm_mn_major_tma_aligned_f32_kernel<<<blocks, kLayoutThreads, 0,
                                             stream>>>(
      input, output, rows, scale_cols, aligned_rows);
  return consume_last_cuda_error();
}

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
  deepgemm_grouped_offset_tma_aligned_f32_kernel<<<blocks, kLayoutThreads, 0,
                                                   stream>>>(
      input, expert_offsets, output, m_capacity, scale_cols, groups,
      padded_rows);
  return consume_last_cuda_error();
}

}  // extern "C"
