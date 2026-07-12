// GLM5.2 vLLM P/D restored-page RoPE layout fixup.
//
// vLLM (`is_neox_style=False`) rotates each RoPE pair (2i, 2i+1) in place, so
// the cache rows it stores keep the rotated pair interleaved. openinfer's
// kernels are interleave-in / block-out: the rotated pair lands at
// [i, i + rope/2] (glm52_mla_assembly.cu, glm52_indexer_rope.cu) — the same
// values, permuted dims. A page restored from a vLLM-written pegaflow
// namespace must therefore be deinterleaved exactly once, right after the
// H2D lands and before the page becomes readable, so it reads like a
// locally-written page:
//
//   out[i] = in[2*i],  out[i + 32] = in[2*i + 1]   (i in 0..32)
//
// kind 0 (MLA fp8_ds_mla, 656 B/token): permute bytes [528, 656) — the
//   64 bf16 rope-key dims. The fp8 ckv and its scales are rope-free.
// kind 1 (index-K, [64x128 fp8][64x4 f32 scale] per block): permute the
//   first 64 of each token's 128 fp8 key bytes (RoPE covers dims 0..64;
//   64..128 pass through). The scale is per token, so the in-token
//   permutation cannot cross a quantization group.

#include <cuda_bf16.h>

#include "../common.cuh"

namespace {

constexpr int kMlaRowBytes = 656;
constexpr int kMlaRopeOffsetBytes = 528;
constexpr int kIdxkKeyBytes = 128;
constexpr int kRopeDim = 64;
constexpr int kRopeHalf = 32;
constexpr int kRowsPerPage = 64;

__global__ void glm52_vllm_rope_fixup_kernel(unsigned char* base,
                                             long long block_stride_bytes,
                                             int kind,
                                             const int* __restrict__ pages) {
  const long long page = pages[blockIdx.x];
  const int row = threadIdx.x;  // one thread per token row

  if (kind == 0) {
    unsigned char* rope_bytes = base + page * block_stride_bytes +
                                (long long)row * kMlaRowBytes +
                                kMlaRopeOffsetBytes;
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(rope_bytes);
    __nv_bfloat16 tmp[kRopeDim];
#pragma unroll
    for (int i = 0; i < kRopeDim; ++i) tmp[i] = v[i];
#pragma unroll
    for (int i = 0; i < kRopeHalf; ++i) {
      v[i] = tmp[2 * i];
      v[i + kRopeHalf] = tmp[2 * i + 1];
    }
  } else {
    unsigned char* v =
        base + page * block_stride_bytes + (long long)row * kIdxkKeyBytes;
    unsigned char tmp[kRopeDim];
#pragma unroll
    for (int i = 0; i < kRopeDim; ++i) tmp[i] = v[i];
#pragma unroll
    for (int i = 0; i < kRopeHalf; ++i) {
      v[i] = tmp[2 * i];
      v[i + kRopeHalf] = tmp[2 * i + 1];
    }
  }
}

}  // namespace

extern "C" {

CUresult glm52_vllm_rope_fixup_cuda(unsigned char* arena_base,
                                    long long block_stride_bytes, int kind,
                                    const int* pages, int num_pages,
                                    cudaStream_t stream) {
  if (arena_base == nullptr || pages == nullptr || num_pages <= 0 ||
      block_stride_bytes <= 0 || (kind != 0 && kind != 1)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_vllm_rope_fixup_kernel<<<num_pages, kRowsPerPage, 0, stream>>>(
      arena_base, block_stride_bytes, kind, pages);
  return consume_last_cuda_error();
}

}  // extern "C"
