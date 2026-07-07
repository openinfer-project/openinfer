// GLM5.2 bucket-1 TP8 MoE: one short chain of phase kernels per layer
// replacing the EP8 dispatch/grouped-GEMM/combine chain
// (docs/models/glm52/moe-tp8-low-latency.md).
//
// Topology: every rank holds a 1/8 intermediate-slice of ALL 257 experts
// (routed + shared folded in as expert index 256). Routing runs OUTSIDE this
// kernel on the production router (`glm52_router_noaux_tc`) — each rank
// routes its own token, so selection is byte-identical to the EP8 path and
// this kernel only consumes (idx, prob) pairs.
//
// Phases (device-side step epoch, per-layer slot regions with parity
// double-buffered LL packets, zero fences — the tag rides each 128-bit
// packet; graph replay never changes parameters):
//   AG  own normed2 (bf16 [H]) + own topk (8 idx + 8 prob) pushed to every
//       rank's allgather slots; poll peers -> xg[8][H] + topk_all
//   U   active-expert union (block 0, ballot compaction; slot order = expert
//       order, deterministic; u=0 is the shared expert, prob 1, all tokens)
//   B   gate|up mma over the union: fp8 w13 slice [257,512,6144] via
//       m16n8k16.bf16 (sigma permutation, fp8->bf16 lossless, f32 accum),
//       k-split partials in fixed order, SiLU epilogue -> ug[u][8][256] bf16
//   C   down mma: w2 slice [257,6144,256], per-expert partials -> cpart
//   RS  out[j][h] = sum_u prob[u][j] * cpart[u][j][h] (fixed order); token
//       j's partial is LL-pushed only to rank j; receiver sums 8 partials
//       and writes mlp_out (bf16). No residual here — the layer's closing
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

namespace {

constexpr int kHidden = 6144;
constexpr int kExperts = 256;      // routed
constexpr int kBankExperts = 257;  // + shared at index 256
constexpr int kTopk = 8;
constexpr int kRanks = 8;
constexpr int kTokens = 8;                    // bucket-1 lockstep global tokens
constexpr int kSliceRows = 512;               // gate|up rows per expert per rank
constexpr int kSliceI = 256;                  // intermediate slice (2048/8)
constexpr int kUnionMax = kTokens * (kTopk + 1);  // 72
constexpr int kThreads = 256;
constexpr int kKsplitB = 16;                  // w13 k-slice 6144/16 = 384
static_assert(kHidden % (kKsplitB * 128) == 0, "w13 kslice must be 128-aligned");

// AG payload per rank: H bf16 (H/4 packets) + 8 idx (2 packets, 4 x i16
// each) + 8 prob (4 packets, 2 x f32 each — the LL payload is 8 bytes).
constexpr int kAgDataPackets = kHidden / 4;
constexpr int kAgPackets = kAgDataPackets + 6;

// LL packet contract: the tag rides in the 4th word of one 16 B aligned
// vector store, and readers treat tag-match as "payload words are valid".
// PTX only guarantees PER-ELEMENT atomicity for vector accesses; the
// tag-guards-payload protocol additionally assumes the interconnect delivers
// an aligned 16 B write as one observable unit. That holds on NVLink (the
// same assumption NCCL's LL128 protocol makes — NCCL disables LL128 on PCIe
// paths for exactly this reason) and is enforced at buffer alloc time via
// CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED (the NVLink-vs-PCIe
// discriminator): see glm52_moe_tp8_alloc_ll_cuda. On an unprobed PCIe P2P
// topology a torn packet would be SILENT data corruption.
__device__ __forceinline__ void st_ll(uint4* p, uint2 v, unsigned tag) {
  asm volatile("st.volatile.global.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(p),
                   "r"(v.x), "r"(v.y), "r"(0u), "r"(tag));
}
__device__ __forceinline__ uint4 ld_ll(const uint4* p) {
  uint4 q;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
               : "l"(p));
  return q;
}
__device__ __forceinline__ void ll_wait(const uint4* p, unsigned tag, uint4* out) {
  uint4 q;
  long c = 0;
  do {
    q = ld_ll(p);
    if (++c > 200000000L) __trap();
  } while (q.w != tag);
  *out = q;
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

struct Glm52MoeTp8Args {
  const __nv_bfloat16* normed2;  // [H] own token (post-attn layernorm)
  const int* topk_idx;           // [8] own token, production router output
  const float* topk_prob;        // [8] renormalized x2.5 by the router
  const unsigned char* w13;      // [257, 512, 6144] fp8 slice
  const float* w13_scale;        // [257, 4, 48]
  const unsigned char* w2;       // [257, 6144, 256] fp8 slice
  const float* w2_scale;         // [257, 48, 2]
  __nv_bfloat16* mlp_out;        // [H] own token (routed + shared, no residual)
  // scratch arena (pointer-stable across capture/replay)
  __nv_bfloat16* xg;             // [8][H] gathered normed2
  int* topk_all_idx;             // [8][8]
  float* topk_all_prob;          // [8][8]
  int* guidx;                    // [72]
  float* guprob;                 // [72][8]
  int* gucnt;                    // [1]
  int* gused;                    // [256]
  float* bpart;                  // [kKsplitB][72][8][512]
  __nv_bfloat16* ug;             // [72][8][256]
  float* cpart;                  // [72][8][H]
  // LL comm (all device pointers; peer_* pre-offset to THIS rank's slot)
  uint4* ag_local;               // [2][8][kAgPackets] own allgather buffer
  uint4* rs_local;               // [2][8][H] own reduce-scatter buffer
  uint4* peer_ag[kRanks];        // peer p's ag buffer + myrank*kAgPackets
  uint4* peer_rs[kRanks];        // peer p's rs buffer + myrank*H
  unsigned long long* epoch_dev;
  int nranks, myrank;
  // Which per-layer LL buffer slot region this layer uses (AG buffers are
  // sized slots x parity x src x packets, RS buffers slots x parity x row x
  // src x packets; tag = step epoch is shared by all layers of a step, so
  // each layer needs its own region for the parity double-buffer to
  // alternate across steps).
  int layer_slot;
  // Row-to-owner mapping. -1 = dp8: row i is rank i's token (normed2 /
  // topk_* / mlp_out are this rank's single row — the bucket-1 c8 shape).
  // >= 0 = span mode: ALL kTokens rows belong to rank span_owner (its
  // normed2/topk_*/mlp_out are [kTokens]-row arrays: 1 committed token + 7
  // speculative drafts); the other ranks contribute slice compute only and
  // zero-fill their pad mlp_out. Compute phases (union/B/SiLU/C) are
  // mapping-agnostic — only where rows come from (AG) and where results go
  // (RS) changes.
  int span_owner;
};

// Per-kernel preamble shared by every phase kernel: the step epoch (advanced
// once per step by the epoch_advance node) selects the tag and the parity
// half of this layer's slot region.
#define TP8_PREAMBLE                                                          \
  const int t = threadIdx.x;                                                  \
  const int gt = blockIdx.x * kThreads + t;                                   \
  const int GT = gridDim.x * kThreads;                                        \
  const unsigned long long ep = *a.epoch_dev;                                 \
  const unsigned tag = (unsigned)ep;                                          \
  const size_t ag_off =                                                       \
      ((size_t)a.layer_slot * 2 + (ep & 1)) * kRanks * kAgPackets;            \
  const size_t rs_off =                                                       \
      ((size_t)a.layer_slot * 2 + (ep & 1)) * kTokens * kRanks * kHidden;     \
  (void)tag;                                                                  \
  (void)gt;                                                                   \
  (void)GT;                                                                   \
  (void)ag_off;                                                               \
  (void)rs_off;

// Step head: one node per step advances the shared epoch (all layers of the
// step share the tag; per-layer slot regions keep their parity alternating
// across steps). Launch-ahead replays advance it too — device-side only.
__global__ void tp8_epoch_advance_kernel(unsigned long long* epoch_dev) {
  *epoch_dev += 1;
}

__global__ void __launch_bounds__(kThreads) tp8_ag_push_kernel(
    Glm52MoeTp8Args a) {
  TP8_PREAMBLE
  // ---- AG push: the rows THIS rank owns -> their src slots in every rank's
  // buffer (incl. self). dp8: one row into slot myrank. span: the owner
  // pushes all kTokens rows into slots 0..kTokens (8x egress, concentrated
  // on one rank — ~1.6 MB/layer, measured budget in the design doc); the
  // other ranks push nothing but still zero the scratch. peer_ag[] is baked
  // with +myrank*kAgPackets at exchange time, hence the (row - myrank)
  // rebase. ----
  for (int i = gt; i < kExperts; i += GT) a.gused[i] = 0;
  for (int i = gt; i < kUnionMax * kTokens; i += GT) a.guprob[i] = 0.f;
  const bool span = a.span_owner >= 0;
  const int rows_mine = span ? (a.myrank == a.span_owner ? kTokens : 0) : 1;
  for (int rp = gt; rp < rows_mine * a.nranks * kAgPackets; rp += GT) {
    const int row = span ? rp / (a.nranks * kAgPackets) : a.myrank;
    const int rem = rp % (a.nranks * kAgPackets);
    const int p = rem / kAgPackets, i = rem % kAgPackets;
    const size_t rbase = span ? (size_t)row : 0;  // row index into own arrays
    uint2 v;
    if (i < kAgDataPackets) {
      v = *reinterpret_cast<const uint2*>(a.normed2 + rbase * kHidden +
                                          (size_t)i * 4);
    } else {
      const int j = i - kAgDataPackets;  // 0,1: idx (4 x i16); 2..5: prob (2 x f32)
      const int* ti = a.topk_idx + rbase * kTopk;
      const float* tp = a.topk_prob + rbase * kTopk;
      if (j < 2) {
        v.x = (unsigned)ti[j * 4 + 0] | ((unsigned)ti[j * 4 + 1] << 16);
        v.y = (unsigned)ti[j * 4 + 2] | ((unsigned)ti[j * 4 + 3] << 16);
      } else {
        v.x = __float_as_uint(tp[(j - 2) * 2 + 0]);
        v.y = __float_as_uint(tp[(j - 2) * 2 + 1]);
      }
    }
    st_ll(a.peer_ag[p] + ag_off + (ptrdiff_t)(row - a.myrank) * kAgPackets + i,
          v, tag);
  }
}

__global__ void __launch_bounds__(kThreads) tp8_ag_recv_kernel(
    Glm52MoeTp8Args a) {
  TP8_PREAMBLE
  // ---- AG poll: assemble xg + topk_all (waits on cross-rank pushes only;
  // self packets were pushed by the ag_push node, already retired) ----
  for (int pk = gt; pk < kAgPackets * a.nranks; pk += GT) {
    const int src = pk / kAgPackets, i = pk % kAgPackets;
    uint4 q;
    ll_wait(a.ag_local + ag_off + (size_t)src * kAgPackets + i, tag, &q);
    if (i < kAgDataPackets) {
      *reinterpret_cast<uint2*>(a.xg + (size_t)src * kHidden + (size_t)i * 4) =
          make_uint2(q.x, q.y);
    } else {
      const int j = i - kAgDataPackets;
      if (j < 2) {
        a.topk_all_idx[src * kTopk + j * 4 + 0] = (int)(q.x & 0xffffu);
        a.topk_all_idx[src * kTopk + j * 4 + 1] = (int)(q.x >> 16);
        a.topk_all_idx[src * kTopk + j * 4 + 2] = (int)(q.y & 0xffffu);
        a.topk_all_idx[src * kTopk + j * 4 + 3] = (int)(q.y >> 16);
      } else {
        a.topk_all_prob[src * kTopk + (j - 2) * 2 + 0] = __uint_as_float(q.x);
        a.topk_all_prob[src * kTopk + (j - 2) * 2 + 1] = __uint_as_float(q.y);
      }
    }
  }
}

// ---- union: ballot compaction over gathered topk (single block) ----
__global__ void __launch_bounds__(kThreads) tp8_union_kernel(
    Glm52MoeTp8Args a) {
  TP8_PREAMBLE
  __shared__ int wcnt[8];
  __shared__ int scomp[kExperts];
  const int w = t >> 5;
  {
    if (t < kTokens * kTopk) {
      atomicOr(&a.gused[a.topk_all_idx[t]], 1);
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
    if (t < kTokens) {
      a.guprob[(size_t)0 * kTokens + t] = 1.0f;
      for (int r = 0; r < kTopk; r++) {
        int e = a.topk_all_idx[t * kTopk + r];
        a.guprob[(size_t)scomp[e] * kTokens + t] = a.topk_all_prob[t * kTopk + r];
      }
    }
  }
}

// ---- B: gate|up mma over the union (NT=2 chains) ----
__global__ void __launch_bounds__(kThreads) tp8_gemm_b_kernel(
    Glm52MoeTp8Args a) {
  TP8_PREAMBLE
  const int w = t >> 5;
  const int gw = blockIdx.x * (kThreads / 32) + w, TW = gridDim.x * (kThreads / 32);
  const int UC = *a.gucnt;
  {
    const int kslice = kHidden / kKsplitB;
    const int pairs = kSliceRows / 32;  // 16 double-tile jobs per expert
    const int jobsPerU = pairs * kKsplitB;
    for (int job = gw; job < UC * jobsPerU; job += TW) {
      int u = job / jobsPerU, r0 = job % jobsPerU;
      int tp = r0 / kKsplitB, ks = r0 % kKsplitB;
      int e = a.guidx[u];
      const unsigned char* W = a.w13 + (size_t)e * kSliceRows * kHidden;
      const float* S = a.w13_scale + (size_t)e * (kSliceRows / 128) * (kHidden / 128);
      float acc[2][4];
      mma_tiles_kslice<2>(W, S, kHidden / 128, tp * 32, a.xg, kHidden,
                          ks * kslice, kslice, acc);
      const int lane = t & 31, gid = lane >> 2, c0 = (lane & 3) * 2;
      float* bp = a.bpart + ((size_t)ks * kUnionMax + u) * kTokens * kSliceRows;
#pragma unroll
      for (int n = 0; n < 2; n++) {
        int row = tp * 32 + n * 16;
        bp[(size_t)c0 * kSliceRows + row + gid] = acc[n][0];
        bp[(size_t)(c0 + 1) * kSliceRows + row + gid] = acc[n][1];
        bp[(size_t)c0 * kSliceRows + row + gid + 8] = acc[n][2];
        bp[(size_t)(c0 + 1) * kSliceRows + row + gid + 8] = acc[n][3];
      }
    }
  }
}

// SiLU epilogue: gate rows [0,256), up rows [256,512), fixed-order k reduce.
__global__ void __launch_bounds__(kThreads) tp8_silu_kernel(Glm52MoeTp8Args a) {
  TP8_PREAMBLE
  const int UC = *a.gucnt;
  for (int i = gt; i < UC * kTokens * kSliceI; i += GT) {
    int u = i / (kTokens * kSliceI);
    int rem = i - u * kTokens * kSliceI;
    int j = rem / kSliceI, row = rem % kSliceI;
    float G = 0, U = 0;
#pragma unroll
    for (int ks = 0; ks < kKsplitB; ks++) {
      const float* bp = a.bpart +
                        ((size_t)ks * kUnionMax + u) * kTokens * kSliceRows +
                        (size_t)j * kSliceRows;
      G += bp[row];
      U += bp[kSliceI + row];
    }
    float sg = G / (1.f + __expf(-G));
    a.ug[((size_t)u * kTokens + j) * kSliceI + row] = __float2bfloat16(sg * U);
  }
}

// ---- C: down mma (k = 256, single slice) ----
__global__ void __launch_bounds__(kThreads) tp8_gemm_c_kernel(
    Glm52MoeTp8Args a) {
  TP8_PREAMBLE
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

// ---- RS push: fixed-order prob-weighted sum, push row j's partial to the
// rank that owns it (dp8: rank j; span: span_owner for every row). The RS
// buffer is [row][src][hidden] — peer_rs[] is baked with the src offset
// (+myrank*kHidden), the row stride is added here. Total push volume is
// mapping-independent; span mode only concentrates the destination. ----
__global__ void __launch_bounds__(kThreads) tp8_rs_push_kernel(
    Glm52MoeTp8Args a) {
  TP8_PREAMBLE
  const int UC = *a.gucnt;
  const bool span = a.span_owner >= 0;
  for (int i = gt; i < kTokens * kHidden; i += GT) {
    int j = i / kHidden, h = i - j * kHidden;
    float s = 0;
    for (int u = 0; u < UC; ++u) {
      float p = a.guprob[(size_t)u * kTokens + j];
      if (p != 0.f) s += p * a.cpart[((size_t)u * kTokens + j) * kHidden + h];
    }
    const int dst = span ? a.span_owner : j;
    st_ll(a.peer_rs[dst] + rs_off + (size_t)j * kRanks * kHidden + h,
          make_uint2(__float_as_uint(s), 0u), tag);
  }
}

// ---- RS recv: reduce the rows this rank owns (fixed source order). Self
// partials were pushed by the rs_push node (already retired) — the spin only
// waits on cross-rank arrivals. dp8: one row (row myrank). span owner: all
// kTokens rows (8x reduce work + concentrated ingress ~5.5 MB/layer,
// budgeted). span non-owners own nothing: their pad rows never entered the
// MoE, so mlp_out is zero-filled — deterministic, and no NaN can leak into
// the next layer's attention through a pad row. ----
__global__ void __launch_bounds__(kThreads) tp8_rs_recv_kernel(
    Glm52MoeTp8Args a) {
  TP8_PREAMBLE
  const bool span = a.span_owner >= 0;
  if (span && a.myrank != a.span_owner) {
    for (int i = gt; i < kTokens * kHidden; i += GT) {
      a.mlp_out[i] = __float2bfloat16(0.f);
    }
    return;
  }
  const int nrows = span ? kTokens : 1;
  for (int i = gt; i < nrows * kHidden; i += GT) {
    const int jr = i / kHidden, h = i - jr * kHidden;
    const int j = span ? jr : a.myrank;
    float s = 0;
    for (int r = 0; r < a.nranks; r++) {
      uint4 q;
      ll_wait(a.rs_local + rs_off + ((size_t)j * kRanks + r) * kHidden + h,
              tag, &q);
      s += __uint_as_float(q.x);
    }
    a.mlp_out[(size_t)jr * kHidden + h] = __float2bfloat16(s);
  }
}

}  // namespace

extern "C" {

// Grid sizing default for the mma phase kernels (occupancy-derived;
// no longer a correctness invariant — nothing grid-syncs anymore).
int glm52_moe_tp8_max_blocks_cuda(int* out_blocks) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  int sms = 0;
  e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e != cudaSuccess) return (int)e;
  int per_sm = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, tp8_gemm_b_kernel, kThreads, 0);
  if (e != cudaSuccess) return (int)e;
  *out_blocks = sms * per_sm;
  return 0;
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
LlVmmRecord g_ll_vmm[64] = {};
int g_ll_vmm_count = 0;
std::mutex g_ll_vmm_mu;  // 8 rank threads allocate concurrently
}

int glm52_moe_tp8_alloc_ll_cuda(size_t bytes, const int* device_ordinals,
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

int glm52_moe_tp8_free_ll_cuda(void* p) {
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

int glm52_moe_tp8_layer_launch_cuda(
    const void* normed2, const void* topk_idx, const void* topk_prob,
    const void* w13, const void* w13_scale, const void* w2,
    const void* w2_scale, void* mlp_out, void* xg, void* topk_all_idx,
    void* topk_all_prob, void* guidx, void* guprob, void* gucnt, void* gused,
    void* bpart, void* ug, void* cpart, void* ag_local, void* rs_local,
    const void* const* peer_ag, const void* const* peer_rs, void* epoch_dev,
    int layer_slot, int nranks, int myrank, int span_owner, int grid_blocks,
    cudaStream_t stream) {
  if (nranks != kRanks || myrank < 0 || myrank >= nranks ||
      span_owner >= nranks) {
    return (int)cudaErrorInvalidValue;
  }
  Glm52MoeTp8Args a = {};
  a.normed2 = (const __nv_bfloat16*)normed2;
  a.topk_idx = (const int*)topk_idx;
  a.topk_prob = (const float*)topk_prob;
  a.w13 = (const unsigned char*)w13;
  a.w13_scale = (const float*)w13_scale;
  a.w2 = (const unsigned char*)w2;
  a.w2_scale = (const float*)w2_scale;
  a.mlp_out = (__nv_bfloat16*)mlp_out;
  a.xg = (__nv_bfloat16*)xg;
  a.topk_all_idx = (int*)topk_all_idx;
  a.topk_all_prob = (float*)topk_all_prob;
  a.guidx = (int*)guidx;
  a.guprob = (float*)guprob;
  a.gucnt = (int*)gucnt;
  a.gused = (int*)gused;
  a.bpart = (float*)bpart;
  a.ug = (__nv_bfloat16*)ug;
  a.cpart = (float*)cpart;
  a.ag_local = (uint4*)ag_local;
  a.rs_local = (uint4*)rs_local;
  for (int p = 0; p < kRanks; p++) {
    a.peer_ag[p] = (uint4*)peer_ag[p];
    a.peer_rs[p] = (uint4*)peer_rs[p];
  }
  a.epoch_dev = (unsigned long long*)epoch_dev;
  a.nranks = nranks;
  a.myrank = myrank;
  a.layer_slot = layer_slot;
  a.span_owner = span_owner;

  // One layer = a short chain of plain graph nodes; stream order is the only
  // cross-phase synchronization (see the phase-ordering note above). Spins
  // wait exclusively on cross-rank packets, so no grid size here is
  // deadlock-capable; grid_blocks (occupancy-derived) sizes the mma kernels.
  void* args[] = {&a};
  const int ag_push_rows = span_owner >= 0 ? kTokens : 1;
  const int ag_push_blocks =
      (kAgPackets * kRanks * ag_push_rows + kThreads - 1) / kThreads;
  const int ag_blocks = (kAgPackets * kRanks + kThreads - 1) / kThreads;
  const int rs_recv_rows = span_owner >= 0 ? kTokens : 1;
  const int rs_blocks = (kHidden + kThreads - 1) / kThreads;
  struct Phase {
    const void* fn;
    int blocks;
  } phases[] = {
      {(const void*)tp8_ag_push_kernel, ag_push_blocks},
      {(const void*)tp8_ag_recv_kernel, ag_blocks},
      {(const void*)tp8_union_kernel, 1},
      {(const void*)tp8_gemm_b_kernel, grid_blocks},
      {(const void*)tp8_silu_kernel, grid_blocks},
      {(const void*)tp8_gemm_c_kernel, grid_blocks},
      {(const void*)tp8_rs_push_kernel, rs_blocks * kTokens},
      {(const void*)tp8_rs_recv_kernel, rs_blocks * rs_recv_rows},
  };
  for (const Phase& ph : phases) {
    cudaError_t e = cudaLaunchKernel(ph.fn, dim3(ph.blocks), dim3(kThreads),
                                     args, 0, stream);
    if (e != cudaSuccess) return (int)e;
  }
  return 0;
}

// Step-head epoch advance: exactly one node per replayed step (all TP8
// layers of the step share the epoch; per-layer slot regions alternate
// parity across steps).
int glm52_moe_tp8_epoch_advance_cuda(void* epoch_dev, cudaStream_t stream) {
  void* args[] = {&epoch_dev};
  return (int)cudaLaunchKernel((const void*)tp8_epoch_advance_kernel, dim3(1),
                               dim3(1), args, 0, stream);
}

}  // extern "C"
