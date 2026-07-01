// GLM5.2 DSA indexer Hadamard rotate, naive in-place radix implementation.
//
// head_dim=128 -> 7 butterfly stages (log2(128)=7). Each stage applies the
// Hadamard pattern with scale = head_dim^-0.5 applied once at the end.
//
// This is NOT the Dao-AILab fast-hadamard-transform port. It is a naive
// correctness-first implementation. If ncu flags it as a decode TPOT
// bottleneck, replace with the Dao-AILab CUDA port (/tmp/fast-hadamard-transform,
// BSD-3-Clause, HEAD e7706fa, csrc/fast_hadamard_transform_cuda.cu).

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// Hadamard matrix entries are +1/-1. We use the recursive butterfly structure:
// H_2 = [[1,1],[1,-1]], H_{2n} = [[H_n, H_n],[H_n, -H_n]].
// For head_dim=128, 7 stages of pairwise butterfly with appropriate stride.
//
// For general power-of-2 head_dim, we compute the Hadamard sign for position
// (i,j) as: sign = popcount(i & j) & 1 ? -1 : 1.
// Naive: each thread computes one output element by iterating over all inputs.
// This is O(head_dim^2) work but correct and simple for head_dim=128.

constexpr int kHeadDim = 128;

__global__ void hadamard_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output, int tokens, int head_dim) {
  const int token_idx = blockIdx.x;
  const int out_idx = threadIdx.x;

  if (token_idx >= tokens || out_idx >= head_dim) return;

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const __nv_bfloat16* in = input + token_idx * head_dim;
  __nv_bfloat16* out = output + token_idx * head_dim;

  float sum = 0.0f;
  for (int j = 0; j < head_dim; ++j) {
    int sign = (__popc(out_idx & j) & 1) ? -1 : 1;
    sum += sign * __bfloat162float(in[j]);
  }
  out[out_idx] = __float2bfloat16(sum * scale);
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

extern "C" {

CUresult glm52_indexer_hadamard_bf16_cuda(
    const __nv_bfloat16* input, __nv_bfloat16* output, int tokens,
    int head_dim, cudaStream_t stream) {
  if (input == nullptr || output == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens <= 0 || head_dim != kHeadDim) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  dim3 grid(tokens);
  dim3 block(kHeadDim);
  hadamard_bf16_kernel<<<grid, block, 0, stream>>>(input, output, tokens,
                                                     head_dim);
  return map_cuda_error(cudaGetLastError());
}

}  // extern "C"
