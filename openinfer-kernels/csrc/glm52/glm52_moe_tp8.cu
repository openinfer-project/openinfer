// GLM5.2 bucket-1 TP8 MoE: one cooperative kernel per layer replacing the
// EP8 dispatch/grouped-GEMM/combine chain (docs/models/glm52/moe-tp8-low-latency.md).
//
// Topology: every rank holds a 1/8 intermediate-slice of ALL 257 experts
// (routed + shared folded in as expert index 256). Routing runs OUTSIDE this
// kernel on the production router (`glm52_router_noaux_tc`) — each rank
// routes its own token, so selection is byte-identical to the EP8 path and
// this kernel only consumes (idx, prob) pairs.
//
// Phases (device-side epoch, parity double-buffered LL packets, zero fences —
// the tag rides每个 128-bit packet; graph replay never changes parameters):
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
//   - no block barrier inside thread-divergent LL branches (a __syncthreads
//     there deadlocks against grid.sync — probe-proven)
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

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

// Software grid barrier (sense-reversing on a monotonic generation counter).
// Deliberately NOT a cooperative launch + cg::grid_group: a full-device
// cooperative kernel node makes every cudaGraphLaunch of the containing
// whole-step graph ~90 us more expensive on the host (measured pilot=1 vs
// pilot=0, graph-mode nsys), and the 8 rank threads' launches serialize on
// the driver — a flat ~0.8 ms/step tax that erased the per-layer win. The
// grid is sized to co-residency (occupancy API) and the stream is graph-
// serialized, so all blocks become resident and the barrier completes; the
// spin cap traps instead of hanging if that invariant is ever violated.
// `gen` is monotonic across launches and replays — no reset, same trick as
// the LL epoch. `count` self-resets each round.
__device__ __forceinline__ void grid_barrier(unsigned* count, unsigned* gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned g = *(volatile unsigned*)gen;
    if (atomicAdd(count, 1u) == gridDim.x - 1) {
      *count = 0u;
      __threadfence();
      atomicAdd(gen, 1u);
    } else {
      long c = 0;
      while (*(volatile unsigned*)gen == g) {
        if (++c > 2000000000L) __trap();
      }
    }
    __threadfence();
  }
  __syncthreads();
}

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
  unsigned* barrier_state;  // [2] = {count, generation}, zero-initialized once
  int nranks, myrank;
};

__global__ void __launch_bounds__(kThreads) glm52_moe_tp8_layer_kernel(
    Glm52MoeTp8Args a) {
  __shared__ int wcnt[8];
  __shared__ int scomp[kExperts];
  unsigned* const bar = a.barrier_state;
  const int t = threadIdx.x, w = t >> 5;
  const int gw = blockIdx.x * (kThreads / 32) + w, TW = gridDim.x * (kThreads / 32);
  const int gt = blockIdx.x * kThreads + t;
  const int GT = gridDim.x * kThreads;
  const unsigned long long ep = *a.epoch_dev;
  const unsigned tag = (unsigned)ep;
  const size_t ag_off = (size_t)(ep & 1) * kRanks * kAgPackets;
  const size_t rs_off = (size_t)(ep & 1) * kRanks * kHidden;

  // ---- AG push: own normed2 + topk to every rank's slot (incl. self) ----
  for (int i = gt; i < kExperts; i += GT) a.gused[i] = 0;
  for (int i = gt; i < kUnionMax * kTokens; i += GT) a.guprob[i] = 0.f;
  for (int pk = gt; pk < kAgPackets * a.nranks; pk += GT) {
    const int p = pk / kAgPackets, i = pk % kAgPackets;
    uint2 v;
    if (i < kAgDataPackets) {
      v = *reinterpret_cast<const uint2*>(a.normed2 + (size_t)i * 4);
    } else {
      const int j = i - kAgDataPackets;  // 0,1: idx (4 x i16); 2..5: prob (2 x f32)
      if (j < 2) {
        v.x = (unsigned)a.topk_idx[j * 4 + 0] | ((unsigned)a.topk_idx[j * 4 + 1] << 16);
        v.y = (unsigned)a.topk_idx[j * 4 + 2] | ((unsigned)a.topk_idx[j * 4 + 3] << 16);
      } else {
        v.x = __float_as_uint(a.topk_prob[(j - 2) * 2 + 0]);
        v.y = __float_as_uint(a.topk_prob[(j - 2) * 2 + 1]);
      }
    }
    st_ll(a.peer_ag[p] + ag_off + i, v, tag);
  }
  // ---- AG poll: assemble xg + topk_all ----
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
  grid_barrier(bar, bar + 1);

  // ---- union (block 0): ballot compaction over gathered topk ----
  if (blockIdx.x == 0) {
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
  grid_barrier(bar, bar + 1);
  const int UC = *a.gucnt;

  // ---- B: gate|up mma over the union (NT=2 chains) ----
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
  grid_barrier(bar, bar + 1);
  // SiLU epilogue: gate rows [0,256), up rows [256,512), fixed-order k reduce.
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
  grid_barrier(bar, bar + 1);

  // ---- C: down mma (k = 256, single slice) ----
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
  grid_barrier(bar, bar + 1);

  // ---- RS: fixed-order prob-weighted sum, push token j -> rank j ----
  for (int i = gt; i < kTokens * kHidden; i += GT) {
    int j = i / kHidden, h = i - j * kHidden;
    float s = 0;
    for (int u = 0; u < UC; ++u) {
      float p = a.guprob[(size_t)u * kTokens + j];
      if (p != 0.f) s += p * a.cpart[((size_t)u * kTokens + j) * kHidden + h];
    }
    st_ll(a.peer_rs[j] + rs_off + h, make_uint2(__float_as_uint(s), 0u), tag);
  }
  // receive own token: 8 partials, fixed source order
  for (int h = gt; h < kHidden; h += GT) {
    float s = 0;
    for (int r = 0; r < a.nranks; r++) {
      uint4 q;
      ll_wait(a.rs_local + rs_off + (size_t)r * kHidden + h, tag, &q);
      s += __uint_as_float(q.x);
    }
    a.mlp_out[h] = __float2bfloat16(s);
  }
  if (blockIdx.x == 0 && t == 0) *a.epoch_dev = ep + 1;
}

}  // namespace

extern "C" {

// Grid sizing: co-resident blocks for the cooperative launch.
int glm52_moe_tp8_max_blocks_cuda(int* out_blocks) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  int sms = 0;
  e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e != cudaSuccess) return (int)e;
  int per_sm = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, glm52_moe_tp8_layer_kernel, kThreads, 0);
  if (e != cudaSuccess) return (int)e;
  *out_blocks = sms * per_sm;
  return 0;
}

// LL buffers come from a dedicated per-device cudaMemPool whose access is
// granted to the peer devices with cudaMemPoolSetAccess. This is deliberate:
// cudaDeviceEnablePeerAccess is DEVICE-WIDE — it maps every current and
// future allocation of this device into the peers' address spaces, and the
// resulting page-table pressure measurably taxes the memory-bound expert
// GEMMs on ALL layers (solo bucket-1 paid a flat ~0.8 ms/step for it on
// 8xH200). Pool-scoped grants map only these few MB. NCCL/DeepEP never call
// cudaDeviceEnablePeerAccess either (they map their windows explicitly).
// Zeroed so no stale word matches a live epoch tag.
namespace {
cudaMemPool_t g_ll_pool[64] = {};
}

int glm52_moe_tp8_alloc_ll_cuda(size_t bytes, const int* device_ordinals,
                                int n_devices, void** out) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  if (dev < 0 || dev >= 64 || n_devices <= 0 || n_devices > 64) {
    return (int)cudaErrorInvalidValue;
  }
  if (!g_ll_pool[dev]) {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = dev;
    cudaMemPool_t pool = nullptr;
    e = cudaMemPoolCreate(&pool, &props);
    if (e != cudaSuccess) return (int)e;
    cudaMemAccessDesc desc[64];
    int n = 0;
    for (int i = 0; i < n_devices; ++i) {
      if (device_ordinals[i] == dev) continue;
      desc[n].location.type = cudaMemLocationTypeDevice;
      desc[n].location.id = device_ordinals[i];
      desc[n].flags = cudaMemAccessFlagsProtReadWrite;
      ++n;
    }
    if (n > 0) {
      e = cudaMemPoolSetAccess(pool, desc, n);
      if (e != cudaSuccess) {
        (void)cudaMemPoolDestroy(pool);
        return (int)e;
      }
    }
    g_ll_pool[dev] = pool;
  }
  e = cudaMallocFromPoolAsync(out, bytes, g_ll_pool[dev], (cudaStream_t)0);
  if (e != cudaSuccess) return (int)e;
  e = cudaMemsetAsync(*out, 0, bytes, (cudaStream_t)0);
  if (e != cudaSuccess) return (int)e;
  return (int)cudaStreamSynchronize((cudaStream_t)0);
}

int glm52_moe_tp8_free_ll_cuda(void* p) {
  cudaError_t e = cudaFreeAsync(p, (cudaStream_t)0);
  if (e != cudaSuccess) return (int)e;
  return (int)cudaStreamSynchronize((cudaStream_t)0);
}

int glm52_moe_tp8_layer_launch_cuda(
    const void* normed2, const void* topk_idx, const void* topk_prob,
    const void* w13, const void* w13_scale, const void* w2,
    const void* w2_scale, void* mlp_out, void* xg, void* topk_all_idx,
    void* topk_all_prob, void* guidx, void* guprob, void* gucnt, void* gused,
    void* bpart, void* ug, void* cpart, void* ag_local, void* rs_local,
    const void* const* peer_ag, const void* const* peer_rs, void* epoch_dev,
    void* barrier_state, int nranks, int myrank, int grid_blocks,
    cudaStream_t stream) {
  if (nranks != kRanks || myrank < 0 || myrank >= nranks) {
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
  a.barrier_state = (unsigned*)barrier_state;
  a.nranks = nranks;
  a.myrank = myrank;

  // Plain launch on purpose — see grid_barrier above for why the cooperative
  // attribute is banned here. grid_blocks must not exceed the occupancy
  // returned by glm52_moe_tp8_max_blocks_cuda (co-residency invariant).
  void* args[] = {&a};
  return (int)cudaLaunchKernel((const void*)glm52_moe_tp8_layer_kernel,
                               dim3(grid_blocks), dim3(kThreads), args, 0,
                               stream);
}

}  // extern "C"
