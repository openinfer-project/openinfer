#include "common.cuh"

#include <stdint.h>

// FlashInfer stores recurrent state as [H, V, K], while OpenInfer's decode
// kernels use [H, K, V] with V contiguous.  Keep the conversion explicit at
// the backend boundary so a layout change cannot silently alter the model.
__global__ void transpose_gdn_state_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int32_t num_heads,
    int32_t key_dim,
    int32_t value_dim,
    bool to_flashinfer) {
  const int64_t total = static_cast<int64_t>(num_heads) * key_dim * value_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int64_t head_stride = static_cast<int64_t>(key_dim) * value_dim;
  const int64_t head = idx / head_stride;
  const int64_t rem = idx - head * head_stride;
  const int64_t row = rem / value_dim;
  const int64_t col = rem - row * value_dim;

  const int64_t src_idx = head * head_stride + row * value_dim + col;
  const int64_t dst_idx = head * head_stride + col * key_dim + row;
  if (to_flashinfer) {
    dst[dst_idx] = src[src_idx];
  } else {
    dst[src_idx] = src[dst_idx];
  }
}

extern "C" CUresult gated_delta_rule_state_transpose_cuda(
    const float* src,
    float* dst,
    int32_t num_heads,
    int32_t key_dim,
    int32_t value_dim,
    bool to_flashinfer,
    CUstream stream) {
  if (src == nullptr || dst == nullptr || num_heads <= 0 || key_dim <= 0 || value_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const int64_t total = static_cast<int64_t>(num_heads) * key_dim * value_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  transpose_gdn_state_kernel<<<blocks, threads, 0, stream>>>(
      src, dst, num_heads, key_dim, value_dim, to_flashinfer);
  return static_cast<CUresult>(cudaGetLastError());
}
