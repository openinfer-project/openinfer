// GLM5.2 MLA decode "assembly" kernels: the glue between the projections and the
// FlashMLA sparse decode. Two pieces, both bs=1 decode (one new token):
//
//   1. query assemble — query[H,576] = [ ql_nope(512) | rope(q_pe)(64) ] per head.
//      The nope part is the absorbed q (copied through); the pe part is the
//      interleave RoPE applied to q_pe.
//
//   2. cache pack — one fp8_ds_mla 656-byte cache token =
//      [ 512 e4m3 ckv | 4 f32 group scales | 64 bf16 rope(k_pe) ]. The ckv fp8 +
//      scales come pre-computed from glm52_fp8_per_token_group_quant (amax/448,
//      the same convention the cache wants); this kernel lays them out and applies
//      interleave RoPE to the shared k_pe.
//
// RoPE is interleave-in / block-out (rope_interleave=True): the input pair lives at
// even/odd indices, the output at [i, i+rope/2]. cos/sin are the first half
// (rope_dim/2 = 32) of the position's rotary table. Mirrors the kimi_k2 rope_out
// device function; validated bit-for-bit against the HF oracle (q_rot/k_rot).

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int kHeads = 64;
constexpr int kQkNope = 512;      // absorbed ql_nope width
constexpr int kRopeDim = 64;      // q_pe / k_pe width
constexpr int kRopeHalf = 32;     // rope_dim / 2 = cos/sin length used
constexpr int kQueryDim = kQkNope + kRopeDim;  // 576
constexpr int kKvLora = 512;      // ckv width
constexpr int kCacheBytes = 656;  // 512 fp8 + 16 scale + 128 bf16 kpe
constexpr int kScaleOffset = 512;
constexpr int kKpeOffset = 528;

// Interleave RoPE of one rope_dim vector: out index r (0..63) reads the pair
// (even=2*pair, odd=2*pair+1), pair = r % 32; the lower half (r<32) is the real
// part, the upper half the imaginary part. cos/sin indexed by pair.
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

// q_pe lives at `q_pe_base[q_pe_offset + h*q_pe_head_stride + ...]`: contiguous
// [H,64] (offset 0, stride 64) when split out, or embedded in the q_b output
// [H,256] (offset 192, stride 256) in the fused forward.
__global__ void glm52_mla_query_assemble_kernel(
    const __nv_bfloat16* __restrict__ ql_nope,    // [H, 512]
    const __nv_bfloat16* __restrict__ q_pe_base,  // q_pe at offset/stride
    int q_pe_offset, int q_pe_head_stride,
    const __nv_bfloat16* __restrict__ cos,        // [32]
    const __nv_bfloat16* __restrict__ sin,        // [32]
    __nv_bfloat16* __restrict__ query) {          // [H, 576]
  const int h = blockIdx.x;
  if (h >= kHeads) return;
  const __nv_bfloat16* q_pe = q_pe_base + q_pe_offset + h * q_pe_head_stride;
  for (int i = threadIdx.x; i < kQueryDim; i += blockDim.x) {
    if (i < kQkNope) {
      query[h * kQueryDim + i] = ql_nope[h * kQkNope + i];
    } else {
      const int r = i - kQkNope;  // 0..63
      query[h * kQueryDim + i] = rope_block(q_pe, r, cos, sin);
    }
  }
}

__global__ void glm52_mla_cache_pack_kernel(
    const unsigned char* __restrict__ ckv_fp8,  // [512]
    const float* __restrict__ ckv_scales,        // [4]
    const __nv_bfloat16* __restrict__ k_pe,      // [64] pre-rope
    const __nv_bfloat16* __restrict__ cos,       // [32]
    const __nv_bfloat16* __restrict__ sin,       // [32]
    unsigned char* __restrict__ cache_token) {   // [656] (slot base)
  const int tid = threadIdx.x;
  // 512 e4m3 ckv
  for (int i = tid; i < kKvLora; i += blockDim.x) {
    cache_token[i] = ckv_fp8[i];
  }
  // 4 f32 group scales at byte 512 (slot base is 4-aligned: 656 % 4 == 0)
  if (tid < 4) {
    reinterpret_cast<float*>(cache_token + kScaleOffset)[tid] = ckv_scales[tid];
  }
  // 64 bf16 rope(k_pe) at byte 528
  __nv_bfloat16* kpe_out = reinterpret_cast<__nv_bfloat16*>(cache_token + kKpeOffset);
  for (int r = tid; r < kRopeDim; r += blockDim.x) {
    kpe_out[r] = rope_block(k_pe, r, cos, sin);
  }
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

CUresult glm52_mla_query_assemble_cuda(const __nv_bfloat16* ql_nope,
                                       const __nv_bfloat16* q_pe_base,
                                       int q_pe_offset, int q_pe_head_stride,
                                       const __nv_bfloat16* cos,
                                       const __nv_bfloat16* sin,
                                       __nv_bfloat16* query, cudaStream_t stream) {
  if (ql_nope == nullptr || q_pe_base == nullptr || cos == nullptr ||
      sin == nullptr || query == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_mla_query_assemble_kernel<<<kHeads, 192, 0, stream>>>(
      ql_nope, q_pe_base, q_pe_offset, q_pe_head_stride, cos, sin, query);
  return consume_last_cuda_error();
}

CUresult glm52_mla_cache_pack_cuda(const unsigned char* ckv_fp8,
                                   const float* ckv_scales,
                                   const __nv_bfloat16* k_pe,
                                   const __nv_bfloat16* cos,
                                   const __nv_bfloat16* sin,
                                   unsigned char* cache_token,
                                   cudaStream_t stream) {
  if (ckv_fp8 == nullptr || ckv_scales == nullptr || k_pe == nullptr ||
      cos == nullptr || sin == nullptr || cache_token == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_mla_cache_pack_kernel<<<1, 128, 0, stream>>>(ckv_fp8, ckv_scales, k_pe, cos,
                                                     sin, cache_token);
  return consume_last_cuda_error();
}

}  // extern "C"
