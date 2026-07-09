// GLM5.2 right-sized sparse MLA decode (M5b): replaces FlashMLA's sparse
// splitkv kernel on the attention-TP path (heads <= 16 real head slots).
//
// Why not FlashMLA: at the production decode shape (bucket-8, topk 2048) the
// FlashMLA kernel is ~80-95% fixed overhead — a persistent 132-CTA grid,
// prologue TMA/metadata, and a separate combine launch built for prefill-sized
// work. Measured 22.8 us on H200 where the DRAM floor is ~3.2 us (8 rows x
// 2048 tokens x 656 B).
//
// The main kernel is TileLang-generated Hopper wgmma AOT — see
// tools/tilelang/glm52/generate.py (12.8 us at the production shape). It
// emits, per (split, row) CTA, UNNORMALIZED f32 partials plus (m, l) in the
// log2 domain; this file owns the fixed-order combine that merges them and
// the naive f64 reference for the parity gate. Without an sm_90a build
// target the TileLang launcher is a NOT_SUPPORTED stub (build.rs).
//
// Cache token layout (fp8_ds_mla, 656 B, see glm52_mla_assembly.cu):
//   [ 512 e4m3 ckv | 4 f32 group scales (dim/128) | 64 bf16 rope(k_pe) ]
// Attention runs over 576 dims (512 dequantized nope + 64 bf16 rope); the
// value is the same dequantized 512-dim nope vector (MLA absorbed form).
//
// Head slots are a fixed 16-wide tile: the attention-TP shard's 8 real heads
// sit in slots 0..heads, pad slots compute on zero queries and are never
// written out. The EP8 (64-head) path stays on FlashMLA.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// TileLang-generated main kernel (or its NOT_SUPPORTED stub); returns a
// cudaError_t. Signature owned by tools/tilelang/glm52/generate.py.
extern "C" int glm52_tilelang_sparse_mla_decode(
    const void* q, const void* cache, const int* indices, float* o_part,
    float* ml, int batch, long long num_slots, int topk, int num_splits,
    int head_slots, cudaStream_t stream);

namespace {

// 16 splits measured best at every serving regime. TP8 always launches the
// full bucket (batch 8, idle rows padded with topk = -1), and pad rows are
// NOT free: they run the whole compute pipeline (only invalid-masked) and
// every CTA stages the full 64x576 Q tile, so raising splits multiplies
// fixed cost across all rows. A 32-split A/B lost everywhere despite
// halving the one real row's per-CTA gather (10k-ctx solo ITL p50
// 14.00 -> 14.17 ms, c8 20.92 -> 22.34 ms). The remaining solo-long-context
// gap vs FlashMLA (in-situ 28.5 us vs 18.0 at 10k ctx; flash's scheduler
// metadata gives the one real row the whole grid) needs dynamic per-row
// split planning, not a bigger static count.
// Changing kNumSplits requires the same value in the generator
// (generate.py NUM_SPLITS) and the Rust scratch sizing
// (GLM52_SPARSE_MLA_NUM_SPLITS); the chain is runtime-validated end to end
// (Rust -> entry -> generated launcher), so a lone edit fails loudly instead
// of writing o_part out of bounds. Same for kHeadSlots.
constexpr int kNumSplits = 16;
constexpr int kHeadSlots = 16;  // partial store width: real heads + zero pads
static_assert(kNumSplits >= 1 && kNumSplits <= 32,
              "combine owns one lane per split within a single warp");
constexpr unsigned kSplitLaneMask =
    kNumSplits == 32 ? 0xffffffffu : ((1u << kNumSplits) - 1u);
constexpr int kDqk = 576;
constexpr int kDv = 512;
constexpr int kCacheBytes = 656;
constexpr int kScaleOffset = 512;
constexpr int kKpeOffset = 528;
// The TileLang kernel bakes GLM5.2's softmax scale; reject anything else
// instead of silently attending with the wrong temperature.
constexpr float kSmScale = 0.0625f;

// Fixed-order split merge: deterministic by construction (split index
// ascending, f32). A row/head whose every split saw only invalid tokens
// (l == 0 across the board) produces zeros. Each thread merges four dims
// through float4 loads with the split loop unrolled — one-dim-per-thread
// scalar reads left too few bytes in flight to cover the split stride
// (measured 8.9 us, long-scoreboard 9.0). The (m, l) pairs and the
// derived weights are computed once per block instead of per thread.
__global__ void glm52_sparse_mla_combine_kernel(
    const float* __restrict__ o_part,   // [16, b, 16, 512]
    const float* __restrict__ ml_part,  // [16, b, 16, 2]
    __nv_bfloat16* __restrict__ latent, // [b, 64, 512]
    int batch) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;

  __shared__ float weights[kNumSplits];
  __shared__ float inv_l;
  if (threadIdx.x < kNumSplits) {
    // Lane s owns split s: one load each, two shuffle reductions. A single
    // serial thread here left 127 threads barrier-stalled behind the
    // dependent global loads (measured 6.4 us, stalled_barrier 5.4).
    const float2 pair = *reinterpret_cast<const float2*>(
        ml_part +
        ((static_cast<size_t>(threadIdx.x) * batch + row) * kHeadSlots + h) *
            2);
    float m_star = (pair.y > 0.f) ? pair.x : -INFINITY;
#pragma unroll
    for (int off = kNumSplits / 2; off > 0; off >>= 1)
      m_star = fmaxf(m_star, __shfl_xor_sync(kSplitLaneMask, m_star, off));
    const float w =
        (pair.y > 0.f && m_star != -INFINITY) ? exp2f(pair.x - m_star) : 0.f;
    weights[threadIdx.x] = w;
    float l_total = w * pair.y;
#pragma unroll
    for (int off = kNumSplits / 2; off > 0; off >>= 1)
      l_total += __shfl_xor_sync(kSplitLaneMask, l_total, off);
    if (threadIdx.x == 0) inv_l = (l_total > 0.f) ? 1.f / l_total : 0.f;
  }
  __syncthreads();

  const int dim = 4 * threadIdx.x;  // 128 threads x float4 = 512 dims
  float4 v{0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < kNumSplits; ++s) {
    const float4 part = *reinterpret_cast<const float4*>(
        o_part +
        ((static_cast<size_t>(s) * batch + row) * kHeadSlots + h) * kDv + dim);
    const float w = weights[s];
    v.x += w * part.x;
    v.y += w * part.y;
    v.z += w * part.z;
    v.w += w * part.w;
  }
  __nv_bfloat16* out = latent + (static_cast<size_t>(row) * 64 + h) * kDv + dim;
  const float inv = inv_l;
  out[0] = __float2bfloat16(v.x * inv);
  out[1] = __float2bfloat16(v.y * inv);
  out[2] = __float2bfloat16(v.z * inv);
  out[3] = __float2bfloat16(v.w * inv);
}

// Naive f64 reference for the parity gate: same dequant semantics, flat
// softmax. Test-only; runs everywhere (the main kernel is sm90a-only).
__global__ void glm52_sparse_mla_reference_kernel(
    const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ cache,
    const int* __restrict__ indices, __nv_bfloat16* __restrict__ latent,
    long long max_slots, int topk, float sm_scale) {
  extern __shared__ double scores[];  // [topk]
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int* row_indices = indices + static_cast<size_t>(row) * topk;
  const __nv_bfloat16* q_h = q + (static_cast<size_t>(row) * 64 + h) * kDqk;

  for (int j = threadIdx.x; j < topk; j += blockDim.x) {
    const int idx = row_indices[j];
    if (idx < 0) {
      scores[j] = -INFINITY;
      continue;
    }
    if (idx >= max_slots) __trap();
    const unsigned char* token = cache + static_cast<long long>(idx) * kCacheBytes;
    const float* scales = reinterpret_cast<const float*>(token + kScaleOffset);
    const __nv_bfloat16* kpe =
        reinterpret_cast<const __nv_bfloat16*>(token + kKpeOffset);
    double s = 0.0;
    for (int f = 0; f < kDv; ++f) {
      const double kv =
          static_cast<double>(
              __half2float(__nv_cvt_fp8_to_halfraw(token[f], __NV_E4M3))) *
          static_cast<double>(scales[f >> 7]);
      s += static_cast<double>(__bfloat162float(q_h[f])) * kv;
    }
    for (int f = 0; f < kDqk - kDv; ++f) {
      s += static_cast<double>(__bfloat162float(q_h[kDv + f])) *
           static_cast<double>(__bfloat162float(kpe[f]));
    }
    scores[j] = s * static_cast<double>(sm_scale);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double m = -INFINITY;
    for (int j = 0; j < topk; ++j) m = fmax(m, scores[j]);
    for (int j = 0; j < topk; ++j) {
      scores[j] = (scores[j] == -INFINITY) ? 0.0 : exp(scores[j] - m);
    }
  }
  __syncthreads();

  double l = 0.0;
  for (int j = 0; j < topk; ++j) l += scores[j];
  __nv_bfloat16* out = latent + (static_cast<size_t>(row) * 64 + h) * kDv;
  for (int d = threadIdx.x; d < kDv; d += blockDim.x) {
    double v = 0.0;
    for (int j = 0; j < topk; ++j) {
      if (scores[j] == 0.0) continue;
      const int idx = row_indices[j];
      const unsigned char* token =
          cache + static_cast<long long>(idx) * kCacheBytes;
      const float* scales = reinterpret_cast<const float*>(token + kScaleOffset);
      const double kv =
          static_cast<double>(
              __half2float(__nv_cvt_fp8_to_halfraw(token[d], __NV_E4M3))) *
          static_cast<double>(scales[d >> 7]);
      v += scores[j] * kv;
    }
    out[d] = __float2bfloat16((l > 0.0) ? v / l : 0.0);
  }
}

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue) return CUDA_ERROR_INVALID_VALUE;
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  return CUDA_ERROR_LAUNCH_FAILED;
}

CUresult map_launcher_error(int rc) {
  const cudaError_t err = static_cast<cudaError_t>(rc);
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue) return CUDA_ERROR_INVALID_VALUE;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

bool supported_topk(int topk) {
  // The TileLang main kernel is AOT-instantiated per topk
  // (tools/tilelang/glm52/generate.py TOPKS): production runs only the full
  // DSA topk — the 256 short tier was dropped (see the generator's note for
  // how to build it again).
  return topk == 2048;
}

}  // namespace

extern "C" {

CUresult glm52_sparse_mla_decode_cuda(const void* q, const void* cache,
                                      const int* indices, float* o_part,
                                      float* ml_part, void* latent, int batch,
                                      long long max_slots, int topk, int heads,
                                      int num_splits, int head_slots,
                                      float sm_scale, cudaStream_t stream) {
  // num_splits / head_slots are the Rust scratch-sizing constants; they must
  // match this file's combine layout AND the generated main kernel (which
  // re-validates them) — a mismatch means o_part indexing disagrees across
  // layers, so refuse to launch.
  if (num_splits != kNumSplits || head_slots != kHeadSlots) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (q == nullptr || cache == nullptr || indices == nullptr ||
      o_part == nullptr || ml_part == nullptr || latent == nullptr ||
      batch <= 0 || max_slots <= 0 || heads <= 0 || heads > kHeadSlots ||
      !supported_topk(topk) || fabsf(sm_scale - kSmScale) > 1e-6f) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const int rc = glm52_tilelang_sparse_mla_decode(
      q, cache, indices, o_part, ml_part, batch, max_slots, topk, kNumSplits,
      kHeadSlots, stream);
  if (rc != 0) return map_launcher_error(rc);

  const dim3 grid(batch, heads);
  glm52_sparse_mla_combine_kernel<<<grid, 128, 0, stream>>>(
      o_part, ml_part, static_cast<__nv_bfloat16*>(latent), batch);
  return consume_last_cuda_error();
}

CUresult glm52_sparse_mla_reference_cuda(const void* q, const void* cache,
                                         const int* indices, void* latent,
                                         int batch, long long max_slots,
                                         int topk, int heads, float sm_scale,
                                         cudaStream_t stream) {
  if (q == nullptr || cache == nullptr || indices == nullptr ||
      latent == nullptr || batch <= 0 || max_slots <= 0 || heads <= 0 ||
      heads > kHeadSlots || topk <= 0 || !(sm_scale > 0.f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t smem = static_cast<size_t>(topk) * sizeof(double);
  if (cudaFuncSetAttribute(glm52_sparse_mla_reference_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)) != cudaSuccess) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  const dim3 grid(batch, heads);
  glm52_sparse_mla_reference_kernel<<<grid, 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const unsigned char*>(cache), indices,
      static_cast<__nv_bfloat16*>(latent), max_slots, topk, sm_scale);
  return consume_last_cuda_error();
}

}  // extern "C"
