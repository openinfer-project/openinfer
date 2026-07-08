// GLM5.2 right-sized sparse MLA decode (M5b): replaces FlashMLA's sparse
// splitkv kernel on the attention-TP path (heads <= 16 real head slots).
//
// Why not FlashMLA: at the production decode shape (bucket-8, topk 2048) the
// FlashMLA kernel is ~80-95% fixed overhead — a persistent 132-CTA grid,
// prologue TMA/metadata, and a separate combine launch built for prefill-sized
// work. Measured 22.6 us where the DRAM floor is ~3.2 us (8 rows x 2048
// tokens x 656 B). This kernel is the split shape the work actually has:
// grid (16 splits, batch), one CTA per (row, split), each CTA gathering
// topk/16 tokens through a cp.async double buffer.
//
// Cache token layout (fp8_ds_mla, 656 B, see glm52_mla_assembly.cu):
//   [ 512 e4m3 ckv | 4 f32 group scales (dim/128) | 64 bf16 rope(k_pe) ]
// Attention runs over 576 dims (512 dequantized nope + 64 bf16 rope); the
// value is the same dequantized 512-dim nope vector (MLA absorbed form).
//
// Head slots are a fixed 16-wide m16 MMA tile: the attention-TP shard's 8
// real heads sit in slots 0..heads, pad slots compute on zero queries and
// are never written out. The EP8 (64-head) path stays on FlashMLA.
//
// Split partials are merged by a separate fixed-order combine kernel
// (deterministic: split index ascending, f32). A row whose indices are all
// -1 (want-mask pad rows carry arbitrary staged indices) combines to zero.

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <mutex>

namespace {

constexpr int kNumSplits = 16;
constexpr int kHeadSlots = 16;   // m16 MMA tile: real heads + zero pads
constexpr int kDqk = 576;
constexpr int kDv = 512;
constexpr int kCacheBytes = 656;
constexpr int kScaleOffset = 512;
constexpr int kKpeOffset = 528;
constexpr int kThreads = 256;    // 8 warps
constexpr int kChunksPerToken = kCacheBytes / 16;  // 41 x 16B cp.async
constexpr float kLog2e = 1.4426950408889634f;
// Padded row strides (elements). A 576-wide bf16 row is 1152 B = 9 x 128 B:
// stepping one row lands on the same shared-memory bank, so the PV stage's
// token-strided V reads (and the QK A-fragment's head-strided reads) serialize
// 8-way. +8 elements shifts each row by 4 banks.
constexpr int kQkStride = kDqk + 8;  // q and dequant tiles

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 bf16 f32-accumulate. Fragment coordinates (g = lane>>2,
// t = lane&3):
//   A (m16 x k16 row-major): a0/a1 = A[g][2t,2t+1]      a2/a3 = A[g+8][2t,2t+1]
//                            a4/a5 = A[g][2t+8,2t+9]    a6/a7 = A[g+8][..+8]
//   B (k16 x n8 col-major):  b0/b1 = B[2t,2t+1][g]      b2/b3 = B[2t+8,2t+9][g]
//   C (m16 x n8):            c0/c1 = C[g][2t,2t+1]      c2/c3 = C[g+8][2t,2t+1]
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_bf16_16x8x16(float c[4], const uint32_t a[4],
                                                 const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  __nv_bfloat162 pair{lo, hi};
  return *reinterpret_cast<const uint32_t*>(&pair);
}

// Row-major bf16 tile load of the A fragment at (row0, col0): rows row0/row0+8,
// cols col0 + {2t, 2t+1, 2t+8, 2t+9}. Callers keep col0 16-aligned so the
// paired loads are 4-byte aligned.
__device__ __forceinline__ void load_a_frag(uint32_t a[4], const __nv_bfloat16* tile,
                                            int stride, int row0, int col0, int lane) {
  const int g = lane >> 2;
  const int t = lane & 3;
  const __nv_bfloat16* r0 = tile + (row0 + g) * stride + col0 + 2 * t;
  const __nv_bfloat16* r1 = r0 + 8 * stride;
  a[0] = *reinterpret_cast<const uint32_t*>(r0);
  a[1] = *reinterpret_cast<const uint32_t*>(r1);
  a[2] = *reinterpret_cast<const uint32_t*>(r0 + 8);
  a[3] = *reinterpret_cast<const uint32_t*>(r1 + 8);
}

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
               "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Dynamic shared memory layout for the main kernel (BI = tokens per stage).
template <int BI>
struct MainSmem {
  static constexpr int kRowPad = 8;            // bank-shift, see kQkStride
  static constexpr int kSStride = BI + kRowPad;
  static constexpr int kPStride = BI + kRowPad;
  static constexpr int kQBytes = kHeadSlots * kQkStride * 2;
  static constexpr int kRawBytes = 2 * BI * kCacheBytes;
  static constexpr int kDeqBytes = BI * kQkStride * 2;
  static constexpr int kNTiles = BI / 8;       // n8 tiles per score row
  static constexpr int kKSplits = 8 / kNTiles; // warps stacked on k
  static constexpr int kPartialBytes = kKSplits * kHeadSlots * kSStride * 4;
  static constexpr int kPBytes = kHeadSlots * kPStride * 2;
  static constexpr int kBytes = kQBytes + kRawBytes + kDeqBytes + kPartialBytes +
                                kPBytes + 3 * kHeadSlots * 4 + 2 * BI;

  __device__ static __nv_bfloat16* q(char* base) {
    return reinterpret_cast<__nv_bfloat16*>(base);
  }
  __device__ static unsigned char* raw(char* base, int stage) {
    return reinterpret_cast<unsigned char*>(base + kQBytes) +
           stage * BI * kCacheBytes;
  }
  __device__ static __nv_bfloat16* deq(char* base) {
    return reinterpret_cast<__nv_bfloat16*>(base + kQBytes + kRawBytes);
  }
  // s partials: [kKSplits][kHeadSlots][BI] f32; slice 0 doubles as the reduced
  // score tile.
  __device__ static float* s_partial(char* base, int ks) {
    return reinterpret_cast<float*>(base + kQBytes + kRawBytes + kDeqBytes) +
           ks * kHeadSlots * kSStride;
  }
  __device__ static __nv_bfloat16* p(char* base) {
    return reinterpret_cast<__nv_bfloat16*>(base + kQBytes + kRawBytes +
                                            kDeqBytes + kPartialBytes);
  }
  __device__ static float* m_run(char* base) {
    return reinterpret_cast<float*>(base + kQBytes + kRawBytes + kDeqBytes +
                                    kPartialBytes + kPBytes);
  }
  __device__ static float* l_run(char* base) { return m_run(base) + kHeadSlots; }
  __device__ static float* alpha(char* base) { return m_run(base) + 2 * kHeadSlots; }
  __device__ static unsigned char* flags(char* base, int stage) {
    return reinterpret_cast<unsigned char*>(m_run(base) + 3 * kHeadSlots) +
           stage * BI;
  }
};

// Issue the cp.async gather for one stage: BI tokens x 41 16-byte chunks,
// chunk-strided across all threads. Invalid (-1) tokens issue nothing and set
// flags[t] = 0; the dequant stage zero-fills them (garbage e4m3 bytes can
// decode to NaN, which would poison the row max even behind a -inf mask).
template <int BI>
__device__ void issue_stage(char* smem, const unsigned char* cache,
                            const int* row_indices, long long max_slots,
                            int token_base, int stage) {
  unsigned char* raw = MainSmem<BI>::raw(smem, stage);
  unsigned char* flags = MainSmem<BI>::flags(smem, stage);
  for (int c = threadIdx.x; c < BI * kChunksPerToken; c += kThreads) {
    const int t = c / kChunksPerToken;
    const int part = c % kChunksPerToken;
    const int idx = row_indices[token_base + t];
    if (idx < 0) {
      if (part == 0) flags[t] = 0;
      continue;
    }
    if (idx >= max_slots) __trap();
    if (part == 0) flags[t] = 1;
    cp_async_16(raw + t * kCacheBytes + part * 16,
                cache + static_cast<long long>(idx) * kCacheBytes + part * 16);
  }
  cp_async_commit();
}

// Dequantize one landed stage into the bf16 [BI][576] tile: nope dims through
// f32(e4m3) * group scale (FlashMLA's fp8_ds_mla semantics), rope dims copied.
// Invalid tokens become all-zero rows (masked to -inf at the score stage).
template <int BI>
__device__ void dequant_stage(char* smem, int stage) {
  const unsigned char* raw = MainSmem<BI>::raw(smem, stage);
  const unsigned char* flags = MainSmem<BI>::flags(smem, stage);
  __nv_bfloat16* deq = MainSmem<BI>::deq(smem);
  constexpr int kTokensPerWarp = BI / 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int i = 0; i < kTokensPerWarp; ++i) {
    const int t = warp * kTokensPerWarp + i;
    const unsigned char* token = raw + t * kCacheBytes;
    __nv_bfloat16* out = deq + t * kQkStride;
    if (!flags[t]) {
      for (int f = lane; f < kDqk; f += 32) out[f] = __nv_bfloat16(0.f);
      continue;
    }
    const float* scales = reinterpret_cast<const float*>(token + kScaleOffset);
    for (int f = lane; f < kDv; f += 32) {
      const float v = __half2float(__nv_cvt_fp8_to_halfraw(token[f], __NV_E4M3));
      out[f] = __float2bfloat16(v * scales[f >> 7]);
    }
    const __nv_bfloat16* kpe =
        reinterpret_cast<const __nv_bfloat16*>(token + kKpeOffset);
    for (int f = lane; f < kDqk - kDv; f += 32) out[kDv + f] = kpe[f];
  }
}

// Main pass: one CTA per (split, row). Scores S[16, BI] = Q . K^T per stage,
// online softmax in the log2 domain, PV accumulated in registers
// (acc[16, 512] f32 spread as 8 warps x 64 dims x 32 f32/thread). Emits the
// unnormalized split partial + (m, l) for the fixed-order combine.
template <int BI>
__global__ void __launch_bounds__(kThreads) glm52_sparse_mla_main_kernel(
    const __nv_bfloat16* __restrict__ q,       // [b, 64, 576]
    const unsigned char* __restrict__ cache,   // [max_slots, 656]
    const int* __restrict__ indices,           // [b, topk]
    float* __restrict__ o_part,                // [16, b, 16, 512]
    float* __restrict__ ml_part,               // [16, b, 16, 2]
    long long max_slots, int topk, float scale_log2) {
  constexpr int kNTiles = MainSmem<BI>::kNTiles;
  constexpr int kKSplits = MainSmem<BI>::kKSplits;
  constexpr int kQkChunks = kDqk / 16;
  constexpr int kQkChunksPerWarp = kQkChunks / kKSplits;

  extern __shared__ char smem[];
  const int split = blockIdx.x;
  const int row = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t4 = lane & 3;

  const int tokens_per_cta = topk / kNumSplits;
  const int iters = tokens_per_cta / BI;
  const int* row_indices = indices + static_cast<size_t>(row) * topk;
  const int split_base = split * tokens_per_cta;

  // Stage q [16, 576] (real heads in slots 0..heads, pads already zero in the
  // full-width buffer) and init the running softmax state.
  __nv_bfloat16* q_tile = MainSmem<BI>::q(smem);
  {
    const __nv_bfloat16* q_row = q + static_cast<size_t>(row) * 64 * kDqk;
    for (int e = threadIdx.x; e < kHeadSlots * kDqk; e += kThreads) {
      q_tile[(e / kDqk) * kQkStride + e % kDqk] = q_row[e];
    }
    if (threadIdx.x < kHeadSlots) {
      MainSmem<BI>::m_run(smem)[threadIdx.x] = -INFINITY;
      MainSmem<BI>::l_run(smem)[threadIdx.x] = 0.f;
    }
  }

  float acc[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  issue_stage<BI>(smem, cache, row_indices, max_slots, split_base, 0);

  for (int it = 0; it < iters; ++it) {
    const int stage = it & 1;
    if (it + 1 < iters) {
      issue_stage<BI>(smem, cache, row_indices, max_slots,
                      split_base + (it + 1) * BI, (it + 1) & 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    dequant_stage<BI>(smem, stage);
    __syncthreads();

    // ---- QK: S[16, BI] = q_tile . deq^T, warps = kNTiles x kKSplits ----
    {
      const int nt = warp % kNTiles;
      const int ks = warp / kNTiles;
      const __nv_bfloat16* deq = MainSmem<BI>::deq(smem);
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      for (int kc = ks * kQkChunksPerWarp; kc < (ks + 1) * kQkChunksPerWarp;
           ++kc) {
        uint32_t a[4];
        load_a_frag(a, q_tile, kQkStride, 0, kc * 16, lane);
        // B(k=feature, n=token): feature pairs are contiguous per token row.
        const __nv_bfloat16* tok = deq + (nt * 8 + g) * kQkStride + kc * 16;
        uint32_t b[2];
        b[0] = *reinterpret_cast<const uint32_t*>(tok + 2 * t4);
        b[1] = *reinterpret_cast<const uint32_t*>(tok + 2 * t4 + 8);
        mma_bf16_16x8x16(c, a, b);
      }
      constexpr int kSStride = MainSmem<BI>::kSStride;
      float* part = MainSmem<BI>::s_partial(smem, ks);
      part[g * kSStride + nt * 8 + 2 * t4] = c[0];
      part[g * kSStride + nt * 8 + 2 * t4 + 1] = c[1];
      part[(g + 8) * kSStride + nt * 8 + 2 * t4] = c[2];
      part[(g + 8) * kSStride + nt * 8 + 2 * t4 + 1] = c[3];
    }
    __syncthreads();
    if (kKSplits > 1) {
      constexpr int kSStride = MainSmem<BI>::kSStride;
      float* s0 = MainSmem<BI>::s_partial(smem, 0);
      for (int e = threadIdx.x; e < kHeadSlots * BI; e += kThreads) {
        const int i = (e / BI) * kSStride + e % BI;
        float v = s0[i];
        for (int ks = 1; ks < kKSplits; ++ks)
          v += MainSmem<BI>::s_partial(smem, ks)[i];
        s0[i] = v;
      }
      __syncthreads();
    }

    // ---- online softmax over the stage, one head row per half-warp ----
    {
      const float* s0 = MainSmem<BI>::s_partial(smem, 0);
      const unsigned char* flags = MainSmem<BI>::flags(smem, stage);
      __nv_bfloat16* p = MainSmem<BI>::p(smem);
      float* m_run = MainSmem<BI>::m_run(smem);
      float* l_run = MainSmem<BI>::l_run(smem);
      float* alpha = MainSmem<BI>::alpha(smem);
      const int r = 2 * warp + (lane >> 4);  // head row
      const int sub = lane & 15;
      float mx = -INFINITY;
      for (int ccol = sub; ccol < BI; ccol += 16) {
        if (flags[ccol]) mx = fmaxf(mx, s0[r * MainSmem<BI>::kSStride + ccol] * scale_log2);
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off, 16));
      mx = __shfl_sync(0xffffffffu, mx, (lane & 16), 32);
      const float m_new = fmaxf(m_run[r], mx);
      float sum = 0.f;
      for (int ccol = sub; ccol < BI; ccol += 16) {
        float pv = 0.f;
        if (flags[ccol] && m_new != -INFINITY) {
          pv = exp2f(s0[r * MainSmem<BI>::kSStride + ccol] * scale_log2 - m_new);
        }
        p[r * MainSmem<BI>::kPStride + ccol] = __float2bfloat16(pv);
        sum += pv;
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off, 16);
      if (sub == 0) {
        const float a = (m_run[r] == -INFINITY) ? 0.f : exp2f(m_run[r] - m_new);
        l_run[r] = l_run[r] * a + sum;
        m_run[r] = m_new;
        alpha[r] = a;
      }
    }
    __syncthreads();

    // ---- PV: acc[16, 512] += P[16, BI] . V[BI, 512], warp = 64-dim slab ----
    {
      const float alpha_lo = MainSmem<BI>::alpha(smem)[g];
      const float alpha_hi = MainSmem<BI>::alpha(smem)[g + 8];
      const __nv_bfloat16* p = MainSmem<BI>::p(smem);
      const __nv_bfloat16* deq = MainSmem<BI>::deq(smem);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[j][0] *= alpha_lo;
        acc[j][1] *= alpha_lo;
        acc[j][2] *= alpha_hi;
        acc[j][3] *= alpha_hi;
      }
      for (int kc = 0; kc < BI / 16; ++kc) {
        uint32_t a[4];
        load_a_frag(a, p, MainSmem<BI>::kPStride, 0, kc * 16, lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          // B(k=token, n=dim): token pairs stride kDqk in the deq tile.
          const int dim = warp * 64 + j * 8 + g;
          const __nv_bfloat16* col = deq + (kc * 16 + 2 * t4) * kQkStride + dim;
          uint32_t b[2];
          b[0] = pack_bf16(col[0], col[kQkStride]);
          b[1] = pack_bf16(col[8 * kQkStride], col[9 * kQkStride]);
          mma_bf16_16x8x16(acc[j], a, b);
        }
      }
    }
    __syncthreads();  // raw/deq/p reused next iteration
  }

  // ---- epilogue: unnormalized split partial + (m, l) ----
  float* o_out = o_part + ((static_cast<size_t>(split) * gridDim.y + row) *
                           kHeadSlots) * kDv;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int dim = warp * 64 + j * 8 + 2 * t4;
    *reinterpret_cast<float2*>(o_out + g * kDv + dim) =
        make_float2(acc[j][0], acc[j][1]);
    *reinterpret_cast<float2*>(o_out + (g + 8) * kDv + dim) =
        make_float2(acc[j][2], acc[j][3]);
  }
  if (threadIdx.x < kHeadSlots) {
    float* ml = ml_part + ((static_cast<size_t>(split) * gridDim.y + row) *
                           kHeadSlots + threadIdx.x) * 2;
    ml[0] = MainSmem<BI>::m_run(smem)[threadIdx.x];
    ml[1] = MainSmem<BI>::l_run(smem)[threadIdx.x];
  }
}

// Fixed-order split merge: deterministic by construction (split index
// ascending, f32). A row/head whose every split saw only invalid tokens
// (l == 0 across the board) produces zeros. The dim axis is spread over
// grid.z (one thread per dim): with grid (b, heads) alone the launch is 64
// CTAs of latency-bound strided reads — too few warps to hide the 16
// split-strided loads (measured 9.7 us, long-scoreboard 11.3).
__global__ void glm52_sparse_mla_combine_kernel(
    const float* __restrict__ o_part,   // [16, b, 16, 512]
    const float* __restrict__ ml_part,  // [16, b, 16, 2]
    __nv_bfloat16* __restrict__ latent, // [b, 64, 512]
    int batch) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int dim = blockIdx.z * blockDim.x + threadIdx.x;

  float m_star = -INFINITY;
#pragma unroll
  for (int s = 0; s < kNumSplits; ++s) {
    const float* ml =
        ml_part + ((static_cast<size_t>(s) * batch + row) * kHeadSlots + h) * 2;
    if (ml[1] > 0.f) m_star = fmaxf(m_star, ml[0]);
  }
  float weights[kNumSplits];
  float l_total = 0.f;
#pragma unroll
  for (int s = 0; s < kNumSplits; ++s) {
    const float* ml =
        ml_part + ((static_cast<size_t>(s) * batch + row) * kHeadSlots + h) * 2;
    weights[s] = (ml[1] > 0.f) ? exp2f(ml[0] - m_star) : 0.f;
    l_total += weights[s] * ml[1];
  }
  const float inv = (l_total > 0.f) ? 1.f / l_total : 0.f;

  __nv_bfloat16* out = latent + (static_cast<size_t>(row) * 64 + h) * kDv;
  float v = 0.f;
#pragma unroll
  for (int s = 0; s < kNumSplits; ++s) {
    v += weights[s] *
         o_part[((static_cast<size_t>(s) * batch + row) * kHeadSlots + h) *
                    kDv + dim];
  }
  out[dim] = __float2bfloat16(v * inv);
}

// Naive f64 reference for the parity gate: same dequant semantics, flat
// softmax. Test-only; runs everywhere (the FlashMLA reference is sm90-only).
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

// BI=64 wants ~179 KB dynamic smem (Hopper's 227 KB opt-in); BI=16 fits the
// 99 KB consumer-part ceiling for the local dev loop.
template <int BI>
bool enable_main_smem() {
  static std::once_flag once;
  static bool ok = false;
  std::call_once(once, [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    int optin = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device) != cudaSuccess) {
      return;
    }
    if (optin < MainSmem<BI>::kBytes) return;
    ok = cudaFuncSetAttribute(glm52_sparse_mla_main_kernel<BI>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              MainSmem<BI>::kBytes) == cudaSuccess;
  });
  return ok;
}

template <int BI>
CUresult launch_main(const __nv_bfloat16* q, const unsigned char* cache,
                     const int* indices, float* o_part, float* ml_part,
                     int batch, long long max_slots, int topk, float sm_scale,
                     cudaStream_t stream) {
  const dim3 grid(kNumSplits, batch);
  glm52_sparse_mla_main_kernel<BI>
      <<<grid, kThreads, MainSmem<BI>::kBytes, stream>>>(
          q, cache, indices, o_part, ml_part, max_slots, topk,
          sm_scale * kLog2e);
  return consume_last_cuda_error();
}

}  // namespace

extern "C" {

CUresult glm52_sparse_mla_decode_cuda(const void* q, const void* cache,
                                      const int* indices, float* o_part,
                                      float* ml_part, void* latent, int batch,
                                      long long max_slots, int topk, int heads,
                                      float sm_scale, cudaStream_t stream) {
  if (q == nullptr || cache == nullptr || indices == nullptr ||
      o_part == nullptr || ml_part == nullptr || latent == nullptr ||
      batch <= 0 || max_slots <= 0 || heads <= 0 || heads > kHeadSlots ||
      topk <= 0 || topk % (kNumSplits * 16) != 0 || !(sm_scale > 0.f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const int tokens_per_cta = topk / kNumSplits;
  CUresult result;
  if (tokens_per_cta % 64 == 0 && enable_main_smem<64>()) {
    result = launch_main<64>(static_cast<const __nv_bfloat16*>(q),
                             static_cast<const unsigned char*>(cache), indices,
                             o_part, ml_part, batch, max_slots, topk, sm_scale,
                             stream);
  } else if (enable_main_smem<16>()) {
    result = launch_main<16>(static_cast<const __nv_bfloat16*>(q),
                             static_cast<const unsigned char*>(cache), indices,
                             o_part, ml_part, batch, max_slots, topk, sm_scale,
                             stream);
  } else {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (result != CUDA_SUCCESS) return result;

  const dim3 grid(batch, heads, kDv / 128);
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
