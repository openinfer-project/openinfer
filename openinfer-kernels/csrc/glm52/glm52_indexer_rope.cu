// GLM5.2 DSA indexer RoPE kernel.
//
// Non-interleaved (half-split / NeoX-style) RoPE for the DSA indexer
// q [n_heads, head_dim] and k [head_dim].
//
// The transformers reference (GlmMoeDsaIndexer.forward) uses
// `apply_rotary_pos_emb` which calls `rotate_half`:
//   rotate_half(x) = cat(-x[half:], x[:half])
//   q_embed = q * cos + rotate_half(q) * sin
// where cos/sin = cat(freqs, freqs).cos()/.sin(), shape [rope_dim].
//
// With rope_dim=64 and cos/sin=[32] (the unique freqs), this simplifies to:
//   q[j]        = q[j] * cos[j] - q[j+32] * sin[j]      (j in 0..31)
//   q[j+32]     = q[j+32] * cos[j] + q[j] * sin[j]
//
// RoPE is applied to the first `qk_rope_head_dim` (=64) elements of q
// (per-head) and k (single). The remaining 64 pass-through dimensions are
// left unchanged.

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

namespace {

constexpr int kRopeDim = 64;   // qk_rope_head_dim
constexpr int kRopeHalf = 32;  // cos/sin length
constexpr int kHeadDim = 128;  // index_head_dim

// Non-interleaved (half-split) RoPE: rotates x[r] using the pair
// (x[r % half], x[r % half + half]) and cos/sin[r % half].
__device__ __forceinline__ __nv_bfloat16 rope_half(const __nv_bfloat16* x, int r,
                                                    const __nv_bfloat16* cos,
                                                    const __nv_bfloat16* sin) {
  const int j = r % kRopeHalf;
  const bool upper = r >= kRopeHalf;
  const float c = __bfloat162float(cos[j]);
  const float s = __bfloat162float(sin[j]);
  const float a = __bfloat162float(x[j]);
  const float b = __bfloat162float(x[j + kRopeHalf]);
  const float v = upper ? (b * c + a * s) : (a * c - b * s);
  return __float2bfloat16(v);
}

// One block per indexer q-head: applies RoPE to q[head, :64] and copies
// q[head, 64:128] (pass-through). Also handles k in the same launch via
// block 0 (k has no head dimension).
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
    __nv_bfloat16* q_head = q + head * kHeadDim;
    __shared__ __nv_bfloat16 q_buf[kHeadDim];
    for (int i = tid; i < kHeadDim; i += blockDim.x) {
      q_buf[i] = q_head[i];
    }
    __syncthreads();
    for (int r = tid; r < kRopeDim; r += blockDim.x) {
      q_head[r] = rope_half(q_buf, r, cos, sin);
    }
  }

  if (head == 0) {
    __shared__ __nv_bfloat16 k_buf[kHeadDim];
    for (int i = tid; i < kHeadDim; i += blockDim.x) {
      k_buf[i] = k[i];
    }
    __syncthreads();
    for (int r = tid; r < kRopeDim; r += blockDim.x) {
      k[r] = rope_half(k_buf, r, cos, sin);
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
