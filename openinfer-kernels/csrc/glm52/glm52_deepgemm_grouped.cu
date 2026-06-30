#include "../common.cuh"

#include <cuda.h>
#include <cstdint>

namespace {

constexpr int kKindW13 = 1;
constexpr int kKindW2 = 2;
// PP8 EP1: groups (256 local experts) and m_capacity (bs=1 = top_k*alignment) are
// RUNTIME args to the group-generic metadata kernel; only the 64-row expert
// alignment is a fixed design constant.
constexpr int kExpertAlignment = 64;
constexpr int kW13N = 4096;
constexpr int kW13K = 6144;
constexpr int kW13ScaleRows = 32;
constexpr int kW13ScaleCols = 48;
constexpr int kW2N = 6144;
constexpr int kW2K = 2048;
constexpr int kW2ScaleRows = 48;
constexpr int kW2ScaleCols = 16;
constexpr int kMetadataThreads = 32;

__device__ __forceinline__ int align_up_int(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ int clamp_nonnegative(int value) {
  return value < 0 ? 0 : value;
}

__device__ __forceinline__ int min_int(int lhs, int rhs) {
  return lhs < rhs ? lhs : rhs;
}

__global__ void deepgemm_grouped_fp8_metadata_kernel(
    const int* __restrict__ psum_expert,
    int64_t* __restrict__ expert_offsets,
    int* __restrict__ w13_problem_sizes,
    int* __restrict__ w2_problem_sizes, int groups, int m_capacity,
    int expert_alignment) {
  int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= groups) {
    return;
  }

  int previous_end =
      expert == 0 ? 0 : clamp_nonnegative(psum_expert[expert - 1]);
  int end = clamp_nonnegative(psum_expert[expert]);
  int start = expert == 0 ? 0 : align_up_int(previous_end, expert_alignment);
  int m = end >= start ? end - start : 0;
  if (start > m_capacity || end > m_capacity) {
    m = 0;
  }

  expert_offsets[expert] = static_cast<int64_t>(start);

  int* w13 = w13_problem_sizes + expert * 3;
  w13[0] = m;
  w13[1] = kW13N;
  w13[2] = kW13K;

  int* w2 = w2_problem_sizes + expert * 3;
  w2[0] = m;
  w2[1] = kW2N;
  w2[2] = kW2K;

  if (expert == groups - 1) {
    int expanded_rows = align_up_int(end, expert_alignment);
    expert_offsets[groups] =
        static_cast<int64_t>(min_int(expanded_rows, m_capacity));
  }
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

bool valid_common(int groups, int m_capacity, int psum_entries,
                  int expert_alignment, int activation_scale_tma_rows) {
  return groups > 0 && m_capacity > 0 && psum_entries == groups &&
         expert_alignment == kExpertAlignment &&
         activation_scale_tma_rows == m_capacity;
}

bool valid_w13(int n, int k, int weight_scale_rows, int weight_scale_cols,
               int activation_scale_cols) {
  return n == kW13N && k == kW13K && weight_scale_rows == kW13ScaleRows &&
         weight_scale_cols == kW13ScaleCols &&
         activation_scale_cols == kW13ScaleCols;
}

bool valid_w2(int n, int k, int weight_scale_rows, int weight_scale_cols,
              int activation_scale_cols) {
  return n == kW2N && k == kW2K && weight_scale_rows == kW2ScaleRows &&
         weight_scale_cols == kW2ScaleCols &&
         activation_scale_cols == kW2ScaleCols;
}

}  // namespace

extern "C" {

CUresult glm52_deepgemm_grouped_fp8_contract_cuda(
    int operand_kind, int groups, int m_capacity, int n, int k,
    int weight_scale_rows, int weight_scale_cols, int activation_scale_cols,
    int activation_scale_tma_rows, int psum_entries, int expert_alignment) {
  if (!valid_common(groups, m_capacity, psum_entries, expert_alignment,
                    activation_scale_tma_rows)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (operand_kind == kKindW13 &&
      valid_w13(n, k, weight_scale_rows, weight_scale_cols,
                activation_scale_cols)) {
    return CUDA_SUCCESS;
  }
  if (operand_kind == kKindW2 &&
      valid_w2(n, k, weight_scale_rows, weight_scale_cols,
               activation_scale_cols)) {
    return CUDA_SUCCESS;
  }
  return CUDA_ERROR_INVALID_VALUE;
}

CUresult glm52_deepgemm_grouped_fp8_metadata_cuda(
    const int* psum_expert, int64_t* expert_offsets, int* w13_problem_sizes,
    int* w2_problem_sizes, int groups, int m_capacity, int expert_alignment,
    cudaStream_t stream) {
  if (psum_expert == nullptr || expert_offsets == nullptr ||
      w13_problem_sizes == nullptr || w2_problem_sizes == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (groups <= 0 || m_capacity <= 0 || expert_alignment != kExpertAlignment) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int blocks = (groups + kMetadataThreads - 1) / kMetadataThreads;
  deepgemm_grouped_fp8_metadata_kernel<<<blocks, kMetadataThreads, 0, stream>>>(
      psum_expert, expert_offsets, w13_problem_sizes, w2_problem_sizes, groups,
      m_capacity, expert_alignment);
  return consume_last_cuda_error();
}

CUresult glm52_deepgemm_grouped_fp8_launch_cuda(
    int /*operand_kind*/, const unsigned char* /*a*/,
    const float* /*a_scale*/, const unsigned char* /*b*/,
    const float* /*b_scale*/, const int* /*psum_expert*/,
    unsigned short* /*out*/, int /*groups*/, int /*m_capacity*/, int /*n*/,
    int /*k*/, cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
