#include "../common.cuh"

#include <cuda.h>

namespace {

constexpr int kTmaAlignmentBytes = 16;
constexpr int kF32Bytes = 4;
constexpr int kF32TmaRowAlignment = kTmaAlignmentBytes / kF32Bytes;
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

}  // extern "C"
