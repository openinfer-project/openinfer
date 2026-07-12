// GLM5.2 tensor-parallel MoE: one short chain of phase kernels per layer replacing the
// EP8 dispatch/grouped-GEMM/combine chain
// (docs/models/glm52/moe-tp8-low-latency.md).
//
// Topology: every rank holds a 1/kRanks intermediate-slice of ALL 257 experts
// (routed + shared folded in as expert index 256), and — replicated
// activations — every rank holds ALL kTokens rows' normed2 and routing
// locally (bit-identical redundant compute upstream), so there is NO
// allgather phase: the only wire traffic is the closing all-reduce of the
// per-rank expert-slice partials. Routing runs OUTSIDE this kernel on the
// production router (`glm52_router_noaux_tc`) over all rows; selection is
// byte-identical to the EP8 path and this kernel only consumes (idx, prob).
//
// Phases (device-side step epoch, per-layer slot regions with parity
// double-buffered LL packets, zero fences — the tag rides each 128-bit
// packet; graph replay never changes parameters):
//   U   active-expert union over all rows' topk (block 0, ballot compaction;
//       slot order = expert order, deterministic; u=0 is the shared expert,
//       prob 1, all tokens)
//   B   gate|up mma + fused SiLU over the union: fp8 w13 slice
//       [257,kSliceRows,6144] via m16n8k16.bf16 (sigma permutation,
//       fp8->bf16 lossless, f32 accum), per-CTA k-split partials reduced in
//       fixed ks order through shared -> ug[u][kTokens][kSliceI] bf16
//   C   down mma: w2 slice [257,6144,kSliceI], per-expert partials -> cpart
//   AR  out[j][h] = sum_u prob[u][j] * cpart[u][j][h] (fixed order); every
//       row's partial is LL-pushed to EVERY rank (rs_* names keep the wire
//       layout [row][src][hidden] of the former reduce-scatter; only the
//       destination set changed); every rank sums kRanks partials per row and
//       writes all kTokens rows of mlp_out (bf16) — bit-identical across
//       ranks (fixed source order). No residual here — the layer's closing
//       add consumes mlp_out exactly like the EP8 arm's combined+shared.
//
// Contracts (violations trapped or static_asserted):
//   - mma k-slices are multiples of 128 (the scale fold happens only at
//     128-column boundaries; a misaligned slice silently drops its tail)
//   - LL spins are capped and __trap on overrun (crash early, never ride
//     the ~100 s device timeout with a half-paired collective)
//   - phase ordering is kernel boundaries only; a spin may wait exclusively
//     on cross-rank packets (see the phase-ordering note below)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <mutex>

#include "glm52_tp_ll.cuh"

#if !defined(GLM52_TP_RANKS) || !defined(GLM52_TP_SLICE_ROWS) || \
    !defined(GLM52_TP_SLICE_I) || !defined(GLM52_TP_ABI_PREFIX) || \
    !defined(GLM52_TP_BLACKWELL_GEMM_B_BLOCKS_PER_SM) || \
    !defined(GLM52_TP_BLACKWELL_GEMM_C_BLOCKS_PER_SM)
#error "GLM5.2 TP implementation requires topology instantiation macros"
#endif

#define GLM52_TP_ABI_INNER(prefix, name) prefix##_##name
#define GLM52_TP_ABI_EXPAND(prefix, name) GLM52_TP_ABI_INNER(prefix, name)
#define GLM52_TP_ABI(name) GLM52_TP_ABI_EXPAND(GLM52_TP_ABI_PREFIX, name)

namespace {

constexpr int kHidden = 6144;
constexpr int kExperts = 256;      // routed
constexpr int kBankExperts = 257;  // + shared at index 256
constexpr int kTopk = 8;
constexpr int kRanks = GLM52_TP_RANKS;
constexpr int kTokens = 8;                    // lockstep global rows (replicated on every rank)
constexpr int kSliceRows = GLM52_TP_SLICE_ROWS;
constexpr int kSliceI = GLM52_TP_SLICE_I;
constexpr int kUnionMax = kTokens * (kTopk + 1);  // 72
constexpr int kThreads = 256;
constexpr int kKsplitB = 16;                  // w13 k-slice 6144/16 = 384
static_assert(kHidden % (kKsplitB * 128) == 0, "w13 kslice must be 128-aligned");

// LL packet primitives + the 16 B atomicity contract live in
// glm52_tp_ll.cuh (shared with the attention allreduce kernels).
__device__ __forceinline__ void st_ll(uint4* p, uint2 v, unsigned tag) {
  glm52_tp_st_ll(p, v, tag);
}
__device__ __forceinline__ void ll_wait(const uint4* p, unsigned tag, uint4* out) {
  glm52_tp_ll_wait(p, tag, out);
}

// Phase ordering is kernel boundaries, deliberately: the layer runs as a
// short chain of graph nodes and cross-phase visibility comes from the
// stream order, with ZERO device fences. A software grid barrier (and the
// cooperative grid.sync it replaced) compiles to a MEMBAR.SC.GPU +
// CCTL.IVALL + global-atomic cluster per site; 5 sites x every block x 8
// pilot layers invalidates L1 across the whole GPU thousands of times per
// step and taxes the memory-bound kernels on ALL layers (~+0.5 ms/step solo,
// measured — the TP8 kernel's own wall stays innocent at ~56 us). The fence
// flavor cannot be tamed (fence.acq_rel.gpu keeps CCTL.IVALL, ld.acquire
// adds more — SASS-verified); only removing intra-kernel cross-block
// dependencies does. Iron rule for these kernels: a spin may only wait on
// CROSS-RANK packets, never on data another block of the same kernel writes.

// fp8 e4m3 pair -> packed bf16x2, exact (e4m3 is representable in bf16).
__device__ __forceinline__ unsigned mma_cvt_pair(unsigned char b0, unsigned char b1) {
  __nv_fp8x2_e4m3 p;
  p.__x = (unsigned short)(b0 | (b1 << 8));
  __half2 h = static_cast<__half2>(p);
  float2 f = __half22float2(h);
  __nv_bfloat162 bb = __float22bfloat162_rn(f);
  return *reinterpret_cast<unsigned*>(&bb);
}

// One warp: NT consecutive 16-row x 8-token mma chains over a k-slice of a
// [rows, k] fp8 matrix with 128x128 BLOCK scales (all 16 rows of a tile share
// the row-block scale; tiles never straddle a /128 row boundary).
// acc[n][4] = {(gid,c0),(gid,c0+1),(gid+8,c0),(gid+8,c0+1)}, c0 = tid*2.
template <int NT>
__device__ __forceinline__ void mma_tiles_kslice(
    const unsigned char* W, const float* scale_rowblock0, int scale_cols,
    int row0, const __nv_bfloat16* act, int k, int k_begin, int k_slice,
    float acc[NT][4]) {
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2, tid = lane & 3;
  float cacc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    cacc[n][0] = cacc[n][1] = cacc[n][2] = cacc[n][3] = 0.f;
  }
  const unsigned char* w0[NT];
  const unsigned char* w1[NT];
  uint4 wp0[NT], wp1[NT];
  float sc[NT], nsc[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    w0[n] = W + (size_t)(row0 + n * 16 + gid) * k + k_begin + tid * 16;
    w1[n] = W + (size_t)(row0 + n * 16 + gid + 8) * k + k_begin + tid * 16;
    wp0[n] = __ldcs((const uint4*)w0[n]);
    wp1[n] = __ldcs((const uint4*)w1[n]);
    sc[n] = scale_rowblock0[(size_t)((row0 + n * 16) >> 7) * scale_cols +
                            (k_begin >> 7)];
    nsc[n] = sc[n];
  }
  for (int kk = k_begin; kk < k_begin + k_slice; kk += 64) {
    uint4 c0[NT], c1[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      c0[n] = wp0[n];
      c1[n] = wp1[n];
      w0[n] += 64;
      w1[n] += 64;
    }
    if (kk + 64 < k_begin + k_slice) {
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        wp0[n] = __ldcs((const uint4*)w0[n]);
        wp1[n] = __ldcs((const uint4*)w1[n]);
      }
      if (((kk + 64) & 127) == 0) {
#pragma unroll
        for (int n = 0; n < NT; ++n)
          nsc[n] = scale_rowblock0[(size_t)((row0 + n * 16) >> 7) * scale_cols +
                                   ((kk + 64) >> 7)];
      }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      // sigma: B slots (tid*2,+1,+8,+9) = columns tid*16 + 4s + {0..3}.
      const __nv_bfloat16* xrow = act + (size_t)gid * k + kk + tid * 16 + 4 * s;
      const uint2 bv = *reinterpret_cast<const uint2*>(xrow);
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* p0 = reinterpret_cast<const unsigned char*>(&c0[n]) + 4 * s;
        const unsigned char* p1 = reinterpret_cast<const unsigned char*>(&c1[n]) + 4 * s;
        unsigned a0 = mma_cvt_pair(p0[0], p0[1]);
        unsigned a1 = mma_cvt_pair(p1[0], p1[1]);
        unsigned a2 = mma_cvt_pair(p0[2], p0[3]);
        unsigned a3 = mma_cvt_pair(p1[2], p1[3]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(cacc[n][0]), "+f"(cacc[n][1]), "+f"(cacc[n][2]), "+f"(cacc[n][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bv.x), "r"(bv.y));
      }
    }
    if (((kk + 64) & 127) == 0) {
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        acc[n][0] += sc[n] * cacc[n][0];
        acc[n][1] += sc[n] * cacc[n][1];
        acc[n][2] += sc[n] * cacc[n][2];
        acc[n][3] += sc[n] * cacc[n][3];
        cacc[n][0] = cacc[n][1] = cacc[n][2] = cacc[n][3] = 0.f;
        sc[n] = nsc[n];
      }
    }
  }
}

struct Glm52MoeTpArgs {
  const __nv_bfloat16* normed2;  // [kTokens][H] all rows (replicated, bit-identical per rank)
  const int* topk_idx;           // [kTokens][8] production router output, all rows
  const float* topk_prob;        // [kTokens][8] renormalized x2.5 by the router
  const unsigned char* w13;      // [257, kSliceRows, 6144] fp8 slice
  const float* w13_scale;        // [257, kSliceRows/128, 48]
  const unsigned char* w2;       // [257, 6144, kSliceI] fp8 slice
  const float* w2_scale;         // [257, 48, kSliceI/128]
  __nv_bfloat16* mlp_out;        // [kTokens][H] all rows (routed + shared, no residual)
  // Want-mask: leading-active row count, read from device memory at kernel
  // time (staged identically on every rank by the host prologue — LL push/
  // wait symmetry needs all ranks to agree). Rows >= *active_rows are pads:
  // excluded from the union (their garbage routing never inflates UC),
  // never pushed, and zero-filled in mlp_out. nullptr = all rows active.
  const int* active_rows;
  // scratch arena (pointer-stable across capture/replay)
  int* guidx;                    // [kUnionMax]
  float* guprob;                 // [kUnionMax][kTokens]
  int* gucnt;                    // [1]
  int* gused;                    // [256]
  __nv_bfloat16* ug;             // [kUnionMax][kTokens][kSliceI]
  float* cpart;                  // [kUnionMax][kTokens][H]
  // LL comm (all device pointers; peer_rs pre-offset to THIS rank's src slot)
  uint4* rs_local;               // [2][row kTokens][src kRanks][H] own all-reduce buffer
  uint4* peer_rs[kRanks];        // peer p's rs buffer + myrank*H
  unsigned long long* epoch_dev;
  int nranks, myrank;
  // Which per-layer LL buffer slot region this layer uses (RS buffers are
  // sized slots x parity x row x src x packets; tag = step epoch is shared
  // by all layers of a step, so each layer needs its own region for the
  // parity double-buffer to alternate across steps).
  int layer_slot;
};

// Per-kernel preamble shared by every phase kernel: the step epoch (advanced
// once per step by the epoch_advance node) selects the tag and the parity
// half of this layer's slot region.
#define TP_PREAMBLE                                                          \
  const int t = threadIdx.x;                                                  \
  const int gt = blockIdx.x * kThreads + t;                                   \
  const int GT = gridDim.x * kThreads;                                        \
  const unsigned long long ep = *a.epoch_dev;                                 \
  const unsigned tag = (unsigned)ep;                                          \
  const int act = a.active_rows == nullptr                                    \
                      ? kTokens                                               \
                      : (*a.active_rows < kTokens ? *a.active_rows : kTokens);\
  const size_t rs_off =                                                       \
      ((size_t)a.layer_slot * 2 + (ep & 1)) * kTokens * kRanks * kHidden;     \
  (void)act;                                                                  \
  (void)tag;                                                                  \
  (void)gt;                                                                   \
  (void)GT;                                                                   \
  (void)rs_off;

// Step head: one node per step advances the shared epoch (all layers of the
// step share the tag; per-layer slot regions keep their parity alternating
// across steps). Launch-ahead replays advance it too — device-side only.
__global__ void tp_epoch_advance_kernel(unsigned long long* epoch_dev) {
  *epoch_dev += 1;
}

// ---- union: ballot compaction over all rows' local topk (single block;
// every rank computes the identical union from its replicated routing) ----
__global__ void __launch_bounds__(kThreads) tp_union_kernel(
    Glm52MoeTpArgs a) {
  TP_PREAMBLE
  __shared__ int wcnt[8];
  __shared__ int scomp[kExperts];
  const int w = t >> 5;
  {
    // Scratch reset lived in the deleted AG-push kernel; a single block
    // zeroes it here before the compaction (kExperts == kThreads).
    a.gused[t] = 0;
    for (int i = t; i < kUnionMax * kTokens; i += kThreads) a.guprob[i] = 0.f;
    __syncthreads();
    if (t < act * kTopk) {
      atomicOr(&a.gused[a.topk_idx[t]], 1);
    }
    __syncthreads();
    const int l = t & 31;
    int f = a.gused[t] != 0;
    unsigned bal = __ballot_sync(~0u, f);
    int before = __popc(bal & ((1u << l) - 1u));
    if (l == 0) wcnt[w] = __popc(bal);
    __syncthreads();
    int wbase = 0;
    for (int i = 0; i < w; i++) wbase += wcnt[i];
    int slot = 1 + wbase + before;  // u=0 fixed: shared expert
    scomp[t] = f ? slot : 0;
    if (f) a.guidx[slot] = t;
    if (t == 0) {
      int c = 1;
      for (int i = 0; i < 8; i++) c += wcnt[i];
      *a.gucnt = c;
      a.guidx[0] = kBankExperts - 1;  // shared expert bank index 256
    }
    __syncthreads();
    if (t < act) {
      a.guprob[(size_t)0 * kTokens + t] = 1.0f;
      for (int r = 0; r < kTopk; r++) {
        int e = a.topk_idx[t * kTopk + r];
        a.guprob[(size_t)scomp[e] * kTokens + t] = a.topk_prob[t * kTopk + r];
      }
    }
  }
}

// ---- B: fused gate|up mma + SiLU over the union. The former split (gemm_b
// k-split partials -> f32 bpart in global -> silu reduce kernel) paid a
// bpart write+read roundtrip and a graph node per layer. Here one 512-thread
// CTA owns a (u, 32-gate-row group) job: warp w runs k-slice w against the
// gate tile-pair and the matching up tile-pair (serial mma depth 2, the same
// ~2 slice-sweeps a warp of the old grid-stride gemm_b averaged), parks the
// per-slice f32 partials in shared, and 512 reduce lanes then sum their
// (half, row, token) position over ks IN ASCENDING ORDER — the exact
// two-level association the silu kernel used (per-slice mma sums, then the
// ks-ordered G/U sums), so ug is bit-identical to the unfused chain. SiLU
// math is unchanged (__expf).
//
// A warp's mma_tiles_kslice<2> tile is 32 rows x 8 tokens = 256 values, one
// full k-slice of the job — part[half][ks] holds exactly that, and the
// (lane, n, i) -> slot map is ks-invariant, so the cross-ks reduce at slot p
// always sums the same (row, token) position.
constexpr int kBGroups = kSliceI / 32;    // 32-row gate groups per expert
constexpr int kBSiluThreads = 2 * kThreads;  // one warp per k-slice
static_assert(kBSiluThreads / 32 == kKsplitB, "one warp per k-slice");
static_assert(kSliceI % 32 == 0, "gate/up groups must tile kSliceI");
__global__ void __launch_bounds__(kBSiluThreads) tp_gemm_b_silu_kernel(
    Glm52MoeTpArgs a) {
  TP_PREAMBLE
  const int UC = *a.gucnt;
  const int w = t >> 5;  // = this warp's k-slice
  const int lane = t & 31;
  const int kslice = kHidden / kKsplitB;
  __shared__ float part[2][kKsplitB][kThreads];  // [half][ks][position]
  __shared__ float sums[2][kThreads];            // [half][position]
  // Slot map (matches the mma C-fragment layout; ks-invariant):
  //   part[half][ks][lane*8 + n*4 + i] = acc[n][i] of the warp running ks.
  // Decode for the final store: position p carries
  //   row = ((p&7)>>2)*16 + (p>>5) + ((p&3)>=2 ? 8 : 0),
  //   token = ((p>>3)&3)*2 + (p&1).
  const int slot0 = lane * 8;
  const int rhalf = t >> 8, rpos = t & (kThreads - 1);
  for (int job = blockIdx.x; job < UC * kBGroups; job += gridDim.x) {
    const int u = job / kBGroups, tg = job % kBGroups;
    const int e = a.guidx[u];
    const unsigned char* W = a.w13 + (size_t)e * kSliceRows * kHidden;
    const float* S =
        a.w13_scale + (size_t)e * (kSliceRows / 128) * (kHidden / 128);
#pragma unroll
    for (int half = 0; half < 2; ++half) {  // 0 = gate rows, 1 = up rows
      const int row0 = half * kSliceI + tg * 32;
      float acc[2][4];
      mma_tiles_kslice<2>(W, S, kHidden / 128, row0, a.normed2, kHidden,
                          w * kslice, kslice, acc);
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) part[half][w][slot0 + n * 4 + i] = acc[n][i];
    }
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKsplitB; ++ks) s += part[rhalf][ks][rpos];  // silu's ks order
    sums[rhalf][rpos] = s;
    __syncthreads();
    if (t < kThreads) {
      const float G = sums[0][t], U = sums[1][t];
      const float sg = G / (1.f + __expf(-G));
      const int row = ((t & 7) >> 2) * 16 + (t >> 5) + ((t & 3) >= 2 ? 8 : 0);
      const int tok = (((t >> 3) & 3) * 2) + (t & 1);
      a.ug[((size_t)u * kTokens + tok) * kSliceI + tg * 32 + row] =
          __float2bfloat16(sg * U);
    }
    // No trailing barrier: the next iteration's mma stores are gated by the
    // sums barrier above, and silu lanes reread nothing the next job writes
    // before they pass it.
  }
}

// ---- C: down mma (k = 256, single slice) ----
__global__ void __launch_bounds__(kThreads) tp_gemm_c_kernel(
    Glm52MoeTpArgs a) {
  TP_PREAMBLE
  const int w = t >> 5;
  const int gw = blockIdx.x * (kThreads / 32) + w, TW = gridDim.x * (kThreads / 32);
  const int UC = *a.gucnt;
  for (int job = gw; job < UC * (kHidden / 16); job += TW) {
    int u = job / (kHidden / 16), tile = job % (kHidden / 16);
    int e = a.guidx[u];
    const unsigned char* W = a.w2 + (size_t)e * kHidden * kSliceI;
    const float* S = a.w2_scale + (size_t)e * (kHidden / 128) * (kSliceI / 128);
    float acc[1][4];
    mma_tiles_kslice<1>(W, S, kSliceI / 128, tile * 16,
                        a.ug + (size_t)u * kTokens * kSliceI, kSliceI, 0,
                        kSliceI, acc);
    const int lane = t & 31, gid = lane >> 2, c0 = (lane & 3) * 2;
    float* cp = a.cpart + (size_t)u * kTokens * kHidden;
    cp[(size_t)c0 * kHidden + tile * 16 + gid] = acc[0][0];
    cp[(size_t)(c0 + 1) * kHidden + tile * 16 + gid] = acc[0][1];
    cp[(size_t)c0 * kHidden + tile * 16 + gid + 8] = acc[0][2];
    cp[(size_t)(c0 + 1) * kHidden + tile * 16 + gid + 8] = acc[0][3];
  }
}

// ---- AR push: fixed-order prob-weighted sum of this rank's expert-slice
// partials, broadcast row j's partial to EVERY rank (each reduces all rows).
// The RS buffer keeps the former reduce-scatter's [row][src][hidden] layout —
// peer_rs[] is baked with the src offset (+myrank*kHidden), the row stride is
// added here. Egress is kRanks x the former per-row push (~6 MB/layer/rank at 8
// rows), spread evenly over the NVLink fabric — the concentrated span-owner
// ingress of the old mapping paid the same bytes on one rank. ----
__global__ void __launch_bounds__(kThreads) tp_rs_push_kernel(
    Glm52MoeTpArgs a) {
  TP_PREAMBLE
  const int UC = *a.gucnt;
  for (int i = gt; i < act * kHidden; i += GT) {
    int j = i / kHidden, h = i - j * kHidden;
    float s = 0;
    for (int u = 0; u < UC; ++u) {
      float p = a.guprob[(size_t)u * kTokens + j];
      if (p != 0.f) s += p * a.cpart[((size_t)u * kTokens + j) * kHidden + h];
    }
    const uint2 v = make_uint2(__float_as_uint(s), 0u);
#pragma unroll
    for (int dst = 0; dst < kRanks; ++dst) {
      st_ll(a.peer_rs[dst] + rs_off + (size_t)j * kRanks * kHidden + h, v, tag);
    }
  }
}

// ---- AR recv: every rank reduces ALL kTokens rows in fixed source order —
// the reduced mlp_out is bit-identical across ranks (the replicated-
// activations contract the next layer's redundant compute depends on). Self
// partials were pushed by the push node (already retired) — the spin only
// waits on cross-rank arrivals. ----
__global__ void __launch_bounds__(kThreads) tp_rs_recv_kernel(
    Glm52MoeTpArgs a) {
  TP_PREAMBLE
  for (int i = gt; i < kTokens * kHidden; i += GT) {
    const int j = i / kHidden, h = i - j * kHidden;
    if (j >= act) {
      // Pad rows never crossed the wire; zero-fill so no stale/NaN value
      // rides the pad row's residual stream into the next layer.
      a.mlp_out[(size_t)j * kHidden + h] = __float2bfloat16(0.f);
      continue;
    }
    float s = 0;
    for (int r = 0; r < a.nranks; r++) {
      uint4 q;
      ll_wait(a.rs_local + rs_off + ((size_t)j * kRanks + r) * kHidden + h,
              tag, &q);
      s += __uint_as_float(q.x);
    }
    a.mlp_out[(size_t)j * kHidden + h] = __float2bfloat16(s);
  }
}

// Occupancy-max grid for one phase kernel on the calling thread's device.
cudaError_t tp_occupancy_grid(const void* kernel_fn, int threads,
                              int* out_blocks) {
  int per_sm = 0, dev = 0, sms = 0;
  cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, kernel_fn, threads, 0);
  if (e == cudaSuccess) e = cudaGetDevice(&dev);
  if (e == cudaSuccess)
    e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e != cudaSuccess) return e;
  *out_blocks = sms * per_sm;
  return cudaSuccess;
}

cudaError_t tp_arch_grid_cap(int blocks_per_sm, int* blocks) {
  if (blocks_per_sm == 0) return cudaSuccess;
  int dev = 0, major = 0, sms = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e == cudaSuccess)
    e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (e == cudaSuccess && major == 10)
    e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e != cudaSuccess) return e;
  if (major == 10) {
    const int cap = sms * blocks_per_sm;
    if (*blocks > cap) *blocks = cap;
  }
  return cudaSuccess;
}

}  // namespace

extern "C" {

// Grid sizing for the fused gemm_b+silu kernel (occupancy-derived; no longer
// a correctness invariant — nothing grid-syncs anymore). gemm_c sizes itself
// inside the layer launcher: its lower register count fits more CTAs/SM.
int GLM52_TP_ABI(max_blocks_cuda)(int* out_blocks) {
  cudaError_t e = tp_occupancy_grid((const void*)tp_gemm_b_silu_kernel,
                                    kBSiluThreads, out_blocks);
  if (e == cudaSuccess)
    e = tp_arch_grid_cap(GLM52_TP_BLACKWELL_GEMM_B_BLOCKS_PER_SM, out_blocks);
  return (int)e;
}

// LL buffers are CUDA VMM allocations in the NCCL-window mapping topology:
// one physical allocation (cuMemCreate on the owner device), mapped once PER
// ACCESSOR at a fresh VA, each VA access-granted to exactly that one device.
// This is load-bearing. Any VA whose access list names a device other than
// (or in addition to) its single accessor taxes the memory-bound expert
// GEMMs on ALL layers — a flat ~0.85 ms/step on solo bucket-1 (8xH200),
// independent of kernel content (an empty node with the grant present pays
// it; no grant, no tax). Measured to pay the full tax: device-wide
// cudaDeviceEnablePeerAccess, pool-scoped cudaMemPoolSetAccess, and a single
// VMM range cuMemSetAccess'd to all 8 devices. Measured tax-free: this
// per-accessor form (19.04 ms vs the no-grant 19.12 ms control) — the same
// topology the running DeepEP/NCCL symmetric windows use.
// Zeroed so no stale word matches a live epoch tag.
namespace {
constexpr int kLlMaxDevices = 8;
struct LlVmmRecord {
  CUdeviceptr vas[kLlMaxDevices];  // vas[i] = accessor device_ordinals[i]'s VA
  int n_vas;
  size_t size;
  CUmemGenericAllocationHandle handle;
};
static LlVmmRecord g_ll_vmm[64] = {};
static int g_ll_vmm_count = 0;
static std::mutex g_ll_vmm_mu;  // rank threads allocate concurrently
}

int GLM52_TP_ABI(alloc_ll_cuda)(size_t bytes, const int* device_ordinals,
                                int n_devices, unsigned long long* out_vas) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  std::lock_guard<std::mutex> lock(g_ll_vmm_mu);
  if (n_devices <= 0 || n_devices > kLlMaxDevices || g_ll_vmm_count >= 64) {
    return (int)cudaErrorInvalidValue;
  }
  // The LL packet protocol needs aligned 16 B stores to cross the fabric as
  // one unit (see st_ll). NVLink provides that; PCIe P2P does not. Native
  // atomic support is the discriminating device attribute — refuse to build
  // LL buffers over links where a packet could tear.
  for (int i = 0; i < n_devices; ++i) {
    if (device_ordinals[i] == dev) continue;
    int native_atomics = 0;
    if (cuDeviceGetP2PAttribute(&native_atomics,
                                CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                                (CUdevice)dev,
                                (CUdevice)device_ordinals[i]) != CUDA_SUCCESS ||
        native_atomics == 0) {
      return (int)cudaErrorNotSupported;
    }
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  size_t gran = 0;
  if (cuMemGetAllocationGranularity(&gran, &prop,
                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
      CUDA_SUCCESS) {
    return (int)cudaErrorUnknown;
  }
  const size_t size = (bytes + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle handle = 0;
  if (cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS) {
    return (int)cudaErrorMemoryAllocation;
  }
  LlVmmRecord rec = {};
  rec.n_vas = n_devices;
  rec.size = size;
  rec.handle = handle;
  bool owner_seen = false;
  for (int i = 0; i < n_devices; ++i) {
    CUdeviceptr va = 0;
    if (cuMemAddressReserve(&va, size, 0, 0, 0) != CUDA_SUCCESS) goto fail;
    rec.vas[i] = va;
    if (cuMemMap(va, size, 0, handle, 0) != CUDA_SUCCESS) goto fail;
    {
      CUmemAccessDesc desc = {};
      desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      desc.location.id = device_ordinals[i];
      desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      if (cuMemSetAccess(va, size, &desc, 1) != CUDA_SUCCESS) goto fail;
    }
    if (device_ordinals[i] == dev) {
      owner_seen = true;
      if (cudaMemset((void*)va, 0, size) != cudaSuccess) goto fail;
    }
    out_vas[i] = (unsigned long long)va;
  }
  // The fleet must include the allocating device, or nothing zeroed the
  // physical pages and no VA is pollable locally.
  if (!owner_seen) goto fail;
  g_ll_vmm[g_ll_vmm_count++] = rec;
  return 0;

fail:
  for (int i = 0; i < n_devices; ++i) {
    if (rec.vas[i] != 0) {
      (void)cuMemUnmap(rec.vas[i], size);
      (void)cuMemAddressFree(rec.vas[i], size);
    }
  }
  (void)cuMemRelease(handle);
  return (int)cudaErrorMemoryAllocation;
}

int GLM52_TP_ABI(free_ll_cuda)(void* p) {
  std::lock_guard<std::mutex> lock(g_ll_vmm_mu);
  for (int i = 0; i < g_ll_vmm_count; ++i) {
    bool hit = false;
    for (int j = 0; j < g_ll_vmm[i].n_vas; ++j) {
      hit |= g_ll_vmm[i].vas[j] == (CUdeviceptr)p;
    }
    if (!hit) continue;
    for (int j = 0; j < g_ll_vmm[i].n_vas; ++j) {
      (void)cuMemUnmap(g_ll_vmm[i].vas[j], g_ll_vmm[i].size);
      (void)cuMemAddressFree(g_ll_vmm[i].vas[j], g_ll_vmm[i].size);
    }
    (void)cuMemRelease(g_ll_vmm[i].handle);
    g_ll_vmm[i] = g_ll_vmm[--g_ll_vmm_count];
    return 0;
  }
  return (int)cudaErrorInvalidValue;
}

int GLM52_TP_ABI(layer_launch_cuda)(
    const void* normed2, const void* topk_idx, const void* topk_prob,
    const void* w13, const void* w13_scale, const void* w2,
    const void* w2_scale, void* mlp_out, void* guidx, void* guprob,
    void* gucnt, void* gused, void* ug, void* cpart,
    void* rs_local, const void* const* peer_rs, void* epoch_dev,
    const void* active_rows, int layer_slot, int nranks, int myrank,
    int grid_blocks, cudaStream_t stream) {
  if (nranks != kRanks || myrank < 0 || myrank >= nranks) {
    return (int)cudaErrorInvalidValue;
  }
  Glm52MoeTpArgs a = {};
  a.normed2 = (const __nv_bfloat16*)normed2;
  a.topk_idx = (const int*)topk_idx;
  a.topk_prob = (const float*)topk_prob;
  a.w13 = (const unsigned char*)w13;
  a.w13_scale = (const float*)w13_scale;
  a.w2 = (const unsigned char*)w2;
  a.w2_scale = (const float*)w2_scale;
  a.mlp_out = (__nv_bfloat16*)mlp_out;
  a.guidx = (int*)guidx;
  a.guprob = (float*)guprob;
  a.gucnt = (int*)gucnt;
  a.gused = (int*)gused;
  a.ug = (__nv_bfloat16*)ug;
  a.cpart = (float*)cpart;
  a.rs_local = (uint4*)rs_local;
  for (int p = 0; p < kRanks; p++) {
    a.peer_rs[p] = (uint4*)peer_rs[p];
  }
  a.epoch_dev = (unsigned long long*)epoch_dev;
  a.active_rows = (const int*)active_rows;
  a.nranks = nranks;
  a.myrank = myrank;
  a.layer_slot = layer_slot;

  // One layer = a short chain of plain graph nodes; stream order is the only
  // cross-phase synchronization (see the phase-ordering note above). Spins
  // wait exclusively on cross-rank packets, so no grid size here is
  // deadlock-capable; grid_blocks (occupancy-derived) sizes gemm_b and silu.
  //
  // gemm_c gets its OWN occupancy grid: it needs ~48 registers vs gemm_b's
  // 84, so 5 CTAs/SM fit where gemm_b caps at 2 — sizing it by grid_blocks
  // (gemm_b's occupancy) leaves over half its warp slots empty. Measured on
  // the 8-row verify bucket (H200, UC=58, cold L2): 40.6 -> 35.9 us/layer.
  // Launches happen at graph-capture time only, so the occupancy query per
  // call costs nothing on the replay path.
  int gemm_c_blocks = 0;
  {
    cudaError_t e = tp_occupancy_grid((const void*)tp_gemm_c_kernel, kThreads,
                                      &gemm_c_blocks);
    if (e == cudaSuccess)
      e = tp_arch_grid_cap(GLM52_TP_BLACKWELL_GEMM_C_BLOCKS_PER_SM,
                           &gemm_c_blocks);
    if (e != cudaSuccess) return (int)e;
  }
  void* args[] = {&a};
  const int rs_blocks = (kHidden + kThreads - 1) / kThreads;
  struct Phase {
    const void* fn;
    int blocks;
    int threads;
  } phases[] = {
      {(const void*)tp_union_kernel, 1, kThreads},
      {(const void*)tp_gemm_b_silu_kernel, grid_blocks, kBSiluThreads},
      {(const void*)tp_gemm_c_kernel, gemm_c_blocks, kThreads},
      {(const void*)tp_rs_push_kernel, rs_blocks * kTokens, kThreads},
      {(const void*)tp_rs_recv_kernel, rs_blocks * kTokens, kThreads},
  };
  for (const Phase& ph : phases) {
    cudaError_t e = cudaLaunchKernel(ph.fn, dim3(ph.blocks), dim3(ph.threads),
                                     args, 0, stream);
    if (e != cudaSuccess) return (int)e;
  }
  return 0;
}

// Step-head epoch advance: exactly one node per replayed step (all TP
// layers of the step share the epoch; per-layer slot regions alternate
// parity across steps).
int GLM52_TP_ABI(epoch_advance_cuda)(void* epoch_dev, cudaStream_t stream) {
  void* args[] = {&epoch_dev};
  return (int)cudaLaunchKernel((const void*)tp_epoch_advance_kernel, dim3(1),
                               dim3(1), args, 0, stream);
}

}  // extern "C"

#undef GLM52_TP_ABI
#undef GLM52_TP_ABI_EXPAND
#undef GLM52_TP_ABI_INNER
