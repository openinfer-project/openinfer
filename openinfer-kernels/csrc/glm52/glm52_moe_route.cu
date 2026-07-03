// GLM5.2 MoE local route/permute/combine glue for bs=1 decode (EP1: all 256
// routed experts resident on this stage; no DeepEP all-to-all). Three small
// kernels around the existing grouped FP8 GEMM:
//
//   1. route_offsets — top-k expert ids -> grouped-GEMM expert_offsets[E+1].
//      Builds the expert-major row layout directly (mirrors the metadata kernel
//      formula: each expert's block starts at align_up(running, 64)). For bs=1
//      each selected expert owns exactly one row.
//
//   2. scatter — replicate the single quantized hidden row (fp8 + per-group
//      scale) into the activation buffer at each selected expert's offset, and
//      build the expert-major per-row route weight for the weighted SwiGLU quant.
//
//   3. combine — weighted sum is already folded into the W2 input by the
//      weighted SwiGLU quant, so this just sums the selected experts' W2 output
//      rows back into the single token's routed output.
//
// The pad rows between experts (alignment slack) stay zero (the activation buffer
// is zero-initialised), so they contribute nothing and are never read by combine.

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

namespace {

// bs=1: each selected expert owns exactly one `alignment`-padded row block, so the
// running-total scan collapses to a closed form -- `expert_offsets[e] = alignment *
// (number of selected experts with id < e)`. top-k is distinct (the router selects
// distinct experts), so that count equals the number of `topk_idx` entries < e, and
// the final `expert_offsets[n_experts] = alignment * topk` (= m_capacity). One thread
// per expert id makes the n_experts+1 stores issue in parallel; the former <<<1,1>>>
// serialised them and exposed ~256x the global-store latency (~61us -> ~1us).
__global__ void glm52_moe_route_offsets_kernel(const int* __restrict__ topk_idx,
                                               long long* __restrict__ expert_offsets,
                                               int n_experts, int topk,
                                               int alignment) {
  extern __shared__ int s_topk[];
  for (int j = threadIdx.x; j < topk; j += blockDim.x) {
    s_topk[j] = topk_idx[j];
  }
  __syncthreads();
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e > n_experts) return;  // expert id in [0, n_experts]
  int rank = 0;
  for (int j = 0; j < topk; ++j) {
    rank += (s_topk[j] >= 0 && s_topk[j] < e) ? 1 : 0;
  }
  expert_offsets[e] = static_cast<long long>(alignment) * rank;
}

__global__ void glm52_moe_route_scatter_kernel(
    const unsigned char* __restrict__ hidden_fp8,  // [k]
    const float* __restrict__ hidden_scale,        // [k/128]
    const int* __restrict__ topk_idx,              // [topk]
    const float* __restrict__ topk_weight,         // [topk]
    const long long* __restrict__ expert_offsets,  // [n_experts+1]
    unsigned char* __restrict__ act,               // [m_capacity, k]
    float* __restrict__ act_scale,                 // [m_capacity, k/128]
    float* __restrict__ row_weight,                // [m_capacity]
    int k, int scale_cols) {
  const int j = blockIdx.x;  // one block per selected expert
  const long long row = expert_offsets[topk_idx[j]];
  unsigned char* dst = act + row * k;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    dst[i] = hidden_fp8[i];
  }
  float* dst_scale = act_scale + row * scale_cols;
  for (int i = threadIdx.x; i < scale_cols; i += blockDim.x) {
    dst_scale[i] = hidden_scale[i];
  }
  if (threadIdx.x == 0) {
    row_weight[row] = topk_weight[j];
  }
}

__global__ void glm52_moe_combine_kernel(
    const __nv_bfloat16* __restrict__ w2_out,      // [m_capacity, n]
    const int* __restrict__ topk_idx,              // [topk]
    const long long* __restrict__ expert_offsets,  // [n_experts+1]
    __nv_bfloat16* __restrict__ routed,            // [n]
    int n, int topk) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  float acc = 0.0f;
  for (int j = 0; j < topk; ++j) {
    const long long row = expert_offsets[topk_idx[j]];
    acc += __bfloat162float(w2_out[row * n + c]);
  }
  routed[c] = __float2bfloat16(acc);
}

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue) return CUDA_ERROR_INVALID_VALUE;
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

extern "C" {

CUresult glm52_moe_route_offsets_cuda(const int* topk_idx, long long* expert_offsets,
                                      int n_experts, int topk, int alignment,
                                      cudaStream_t stream) {
  if (topk_idx == nullptr || expert_offsets == nullptr || n_experts <= 0 ||
      topk <= 0 || alignment <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int total = n_experts + 1;  // expert ids [0, n_experts]
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  const size_t shmem = static_cast<size_t>(topk) * sizeof(int);
  glm52_moe_route_offsets_kernel<<<blocks, threads, shmem, stream>>>(
      topk_idx, expert_offsets, n_experts, topk, alignment);
  return consume_last_cuda_error();
}

CUresult glm52_moe_route_scatter_cuda(const unsigned char* hidden_fp8,
                                      const float* hidden_scale, const int* topk_idx,
                                      const float* topk_weight,
                                      const long long* expert_offsets, unsigned char* act,
                                      float* act_scale, float* row_weight, int topk,
                                      int k, int scale_cols, cudaStream_t stream) {
  if (hidden_fp8 == nullptr || hidden_scale == nullptr || topk_idx == nullptr ||
      topk_weight == nullptr || expert_offsets == nullptr || act == nullptr ||
      act_scale == nullptr || row_weight == nullptr || topk <= 0 || k <= 0 ||
      scale_cols <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_moe_route_scatter_kernel<<<topk, 256, 0, stream>>>(
      hidden_fp8, hidden_scale, topk_idx, topk_weight, expert_offsets, act, act_scale,
      row_weight, k, scale_cols);
  return consume_last_cuda_error();
}

CUresult glm52_moe_combine_cuda(const __nv_bfloat16* w2_out, const int* topk_idx,
                                const long long* expert_offsets, __nv_bfloat16* routed,
                                int n, int topk, cudaStream_t stream) {
  if (w2_out == nullptr || topk_idx == nullptr || expert_offsets == nullptr ||
      routed == nullptr || n <= 0 || topk <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  glm52_moe_combine_kernel<<<blocks, threads, 0, stream>>>(w2_out, topk_idx,
                                                           expert_offsets, routed, n, topk);
  return consume_last_cuda_error();
}

}  // extern "C"
