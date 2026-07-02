// GLM5.2 DSA indexer RoPE kernel.
//
// Interleaved RoPE for the DSA indexer q [n_heads, rope_dim] and k [rope_dim].
// Reuses the `rope_block` device function from glm52_mla_assembly.cu — the
// RoPE convention is identical (interleave-in / block-out, rope_dim=64,
// cos/sin=[32]). Only the launch shape differs: the MLA assembly kernel fuses
// RoPE into query-assemble / cache-pack; this kernel is standalone so the
// indexer can run it right after k_norm, before quant + cache write.
//
// Aligned to vllm DeepseekV32Indexer: RoPE is applied to the first
// `qk_rope_head_dim` (=64) elements of q (per-head) and k (single). The
// remaining `head_dim - qk_rope_head_dim` (=64) pass-through dimensions are
// copied unchanged.
//
// Source: vendored FlashInfer RoPE convention (rope_interleave=true). The
// `rope_block` device function is copied verbatim from glm52_mla_assembly.cu —
// it was oracle-validated bit-for-bit in PR1 (#477).

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

namespace {

constexpr int kRopeDim = 64;   // qk_rope_head_dim
constexpr int kRopeHalf = 32;  // cos/sin length
constexpr int kHeadDim = 128;  // index_head_dim

// Verbatim copy of rope_block from glm52_mla_assembly.cu — oracle-validated
// interleave RoPE (interleave-in / block-out). See PR1 for the validation.
__device__ __forceinline__ __nv_bfloat16 rope_block(const __nv_bfloat16* x, int r,
                                                     const __nv_bfloat16* cos,
                                                     const __nv_bfloat16* sin) {
  const int pair = r % kRopeHalf;
  const bool upper = r >= kRopeHalf;
  const float c = __bfloat162float(cos[pair]);
  const float s = __bfloat162float(sin[pair]);
  const float even = __bfloat162float(x[2 * pair]);
  const float odd = __bfloat162float(x[2 * pair + 1]);
  const float v = upper ? (odd * c + even * s) : (even * c - odd * s);
  return __float2bfloat16(v);
}

// One block per indexer q-head: applies RoPE to q[head, :64] and copies
// q[head, 64:128] (pass-through). Also handles k in the same launch via
// block 0 (k has no head dimension). We use a single kernel for both to
// avoid a second launch.
__global__ void glm52_indexer_rope_kernel(
    __nv_bfloat16* __restrict__ q,          // [n_heads, head_dim] (in-place)
    __nv_bfloat16* __restrict__ k,          // [head_dim] (in-place)
    int n_heads,
    const __nv_bfloat16* __restrict__ cos,  // [32]
    const __nv_bfloat16* __restrict__ sin)  // [32]
{
  const int head = blockIdx.x;
  const int tid = threadIdx.x;

  if (head < n_heads) {
    // q: [n_heads, head_dim] — RoPE on first 64, copy last 64.
    __nv_bfloat16* q_head = q + head * kHeadDim;
    // Use shared memory to avoid in-place aliasing (rope_block reads pairs).
    __shared__ __nv_bfloat16 q_buf[kHeadDim];
    for (int i = tid; i < kHeadDim; i += blockDim.x) {
      q_buf[i] = q_head[i];
    }
    __syncthreads();
    for (int r = tid; r < kRopeDim; r += blockDim.x) {
      q_head[r] = rope_block(q_buf, r, cos, sin);
    }
    // pass-through [64:128] is already in place — no copy needed.
  }

  // Block 0 also handles k (single vector, [head_dim]).
  if (head == 0) {
    __shared__ __nv_bfloat16 k_buf[kHeadDim];
    for (int i = tid; i < kHeadDim; i += blockDim.x) {
      k_buf[i] = k[i];
    }
    __syncthreads();
    for (int r = tid; r < kRopeDim; r += blockDim.x) {
      k[r] = rope_block(k_buf, r, cos, sin);
    }
  }
}

}  // namespace

extern "C" {

CUresult glm52_indexer_rope_cuda(__nv_bfloat16* q,      // [n_heads, head_dim]
                                __nv_bfloat16* k,       // [head_dim]
                                int n_heads,
                                const __nv_bfloat16* cos,
                                const __nv_bfloat16* sin,
                                cudaStream_t stream) {
  if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_indexer_rope_kernel<<<n_heads, 128, 0, stream>>>(q, k, n_heads, cos, sin);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue) return CUDA_ERROR_INVALID_VALUE;
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // extern "C"
