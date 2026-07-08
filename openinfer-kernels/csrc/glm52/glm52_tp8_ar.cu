// GLM5.2 TP8 attention allreduce: the o_proj epilogue
// collective for the attention-TP topology (docs/models/glm52/
// moe-tp8-low-latency.md, M4).
//
// Every rank computes a PARTIAL projection output for ALL bucket rows (its
// 8/64 heads' contribution, full hidden width); afterwards every rank must
// hold the identical full sum. At 8 rows x 12 KB the payload is byte-bound,
// not latency-bound, so this is a TWO-SHOT allreduce (reduce-scatter over
// hidden chunks, then broadcast), not the one-shot the R4 probe used for a
// single row: one-shot wire bytes are 7x payload per rank (measured 13.2
// us/layer standalone), two-shot is 2x (each element crosses the fabric
// once in, once out). Packets carry 12 B payload + tag (the 8 B LL form
// wastes a word).
//
// Wire layout per layer slot: [parity 2][stage 2][row kTokens][src kRanks]
// [kChunkPk] uint4 packets. Stage 0 = reduce-scatter (src = contributor,
// chunk = receiver's), stage 1 = broadcast (src = chunk owner). The tag is
// the shared device-side step epoch (same counter as the MoE chain; separate
// VMM region, tags never collide).
//
// Chain (kernel boundaries are the only cross-phase sync, zero fences; every
// spin waits exclusively on cross-rank packets from a PREVIOUS kernel):
//   push    own partial's chunk c -> rank c's stage-0 slots (all rows)
//   reduce  wait 8 contributors for MY chunk, fixed-order f32 sum ->
//           broadcast the reduced chunk to every rank's stage-1 slots
//           (fused: the spin waits on push packets, a previous kernel)
//   recv    wait 8 chunk owners' stage-1 packets -> assemble out rows
// Each element is summed exactly once, on exactly one rank, in a fixed src
// order — outputs are bit-identical across ranks, which the
// replicated-activation topology RELIES on (router/sampling run redundantly
// on all ranks downstream).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "glm52_tp8_ll.cuh"

namespace {

constexpr int kHidden = 6144;
constexpr int kRanks = 8;
constexpr int kTokens = 8;  // max bucket rows
constexpr int kThreads = 256;
// One rank's hidden chunk per row, packed 6 bf16 (12 B) per packet.
constexpr int kChunk = kHidden / kRanks;               // 768 bf16
constexpr int kChunkPk = kChunk * 2 / 12;              // 128 packets
static_assert(kChunk * 2 % 12 == 0, "chunk must pack into 12 B packets");
constexpr int kRowStride = kRanks * kChunkPk;          // [src] slots per row
constexpr int kStageStride = kTokens * kRowStride;     // one stage, all rows

struct Glm52Tp8ArArgs {
  const __nv_bfloat16* partial;  // [rows][H] this rank's partial, all rows
  __nv_bfloat16* out;            // [rows][H] reduced result (all ranks equal)
  uint4* ar_local;               // own AR buffer base
  uint4* peer_ar[kRanks];        // peer p's AR buffer + myrank*kChunkPk
  const unsigned long long* epoch_dev;
  // Want-mask: leading-active row count, read from device memory at kernel
  // time (staged by the host prologue like the step epoch, identically on
  // every rank — the tag discipline needs push/wait symmetry). Pad rows
  // (row >= *active_rows) push nothing, wait on nothing, and get zero-filled
  // outputs. nullptr = all `rows` active (oracle gates without staging).
  const int* active_rows;
  int layer_slot;
  int rows;
  int nranks, myrank;
};

#define TP8_AR_PREAMBLE                                                       \
  const int gt = blockIdx.x * kThreads + threadIdx.x;                         \
  const int GT = gridDim.x * kThreads;                                        \
  const unsigned long long ep = *a.epoch_dev;                                 \
  const unsigned tag = (unsigned)ep;                                          \
  const int act = a.active_rows == nullptr                                    \
                      ? a.rows                                                \
                      : (*a.active_rows < a.rows ? *a.active_rows : a.rows);  \
  (void)act;                                                                  \
  const size_t ar_off =                                                       \
      ((size_t)a.layer_slot * 2 + (ep & 1)) * 2 * kStageStride;

__device__ __forceinline__ void ld_payload12(const __nv_bfloat16* src,
                                             unsigned* x, unsigned* y,
                                             unsigned* z) {
  const unsigned* w = reinterpret_cast<const unsigned*>(src);
  *x = w[0];
  *y = w[1];
  *z = w[2];
}

__device__ __forceinline__ void st_payload12(__nv_bfloat16* dst, unsigned x,
                                             unsigned y, unsigned z) {
  unsigned* w = reinterpret_cast<unsigned*>(dst);
  w[0] = x;
  w[1] = y;
  w[2] = z;
}

__device__ __forceinline__ float2 bf2f(unsigned w) {
  const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(&w);
  return make_float2(__bfloat162float(p.x), __bfloat162float(p.y));
}

__device__ __forceinline__ unsigned f2bf(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&p);
}

// Stage 0: land this rank's partial chunk c in rank c's reduce slots
// (active rows only).
__global__ void __launch_bounds__(kThreads) tp8_ar_push_kernel(
    Glm52Tp8ArArgs a) {
  TP8_AR_PREAMBLE
  for (int rp = gt; rp < act * a.nranks * kChunkPk; rp += GT) {
    const int row = rp / (a.nranks * kChunkPk);
    const int rem = rp % (a.nranks * kChunkPk);
    const int c = rem / kChunkPk, i = rem % kChunkPk;
    unsigned x, y, z;
    ld_payload12(a.partial + (size_t)row * kHidden + (size_t)c * kChunk + i * 6,
                 &x, &y, &z);
    glm52_tp8_st_ll12(
        a.peer_ar[c] + ar_off + (size_t)row * kRowStride + i, x, y, z, tag);
  }
}

// Stage 1 (fused reduce + broadcast): wait the 8 contributors for MY chunk,
// sum in fixed src order, broadcast the result to every rank's stage-1
// slots. The spin waits on push packets — a previous kernel.
__global__ void __launch_bounds__(kThreads) tp8_ar_reduce_bcast_kernel(
    Glm52Tp8ArArgs a) {
  TP8_AR_PREAMBLE
  for (int rp = gt; rp < act * kChunkPk; rp += GT) {
    const int row = rp / kChunkPk, i = rp % kChunkPk;
    float2 a01 = make_float2(0.f, 0.f), a23 = a01, a45 = a01;
    for (int src = 0; src < a.nranks; ++src) {
      uint4 q;
      glm52_tp8_ll_wait(
          a.ar_local + ar_off + (size_t)row * kRowStride +
              (size_t)src * kChunkPk + i,
          tag, &q);
      const float2 p01 = bf2f(q.x), p23 = bf2f(q.y), p45 = bf2f(q.z);
      a01.x += p01.x; a01.y += p01.y;
      a23.x += p23.x; a23.y += p23.y;
      a45.x += p45.x; a45.y += p45.y;
    }
    const unsigned x = f2bf(a01.x, a01.y);
    const unsigned y = f2bf(a23.x, a23.y);
    const unsigned z = f2bf(a45.x, a45.y);
    for (int dst = 0; dst < a.nranks; ++dst) {
      glm52_tp8_st_ll12(
          a.peer_ar[dst] + ar_off + kStageStride + (size_t)row * kRowStride + i,
          x, y, z, tag);
    }
  }
}

// Stage 2: assemble the full rows from the 8 chunk owners' broadcasts.
// Pad rows never crossed the wire — zero-fill their output so no stale (or
// NaN) value leaks into the next layer through a pad row's residual.
__global__ void __launch_bounds__(kThreads) tp8_ar_recv_kernel(
    Glm52Tp8ArArgs a) {
  TP8_AR_PREAMBLE
  for (int rp = gt; rp < a.rows * a.nranks * kChunkPk; rp += GT) {
    const int row = rp / (a.nranks * kChunkPk);
    const int rem = rp % (a.nranks * kChunkPk);
    const int c = rem / kChunkPk, i = rem % kChunkPk;
    if (row >= act) {
      st_payload12(a.out + (size_t)row * kHidden + (size_t)c * kChunk + i * 6,
                   0u, 0u, 0u);
      continue;
    }
    uint4 q;
    glm52_tp8_ll_wait(
        a.ar_local + ar_off + kStageStride + (size_t)row * kRowStride +
            (size_t)c * kChunkPk + i,
        tag, &q);
    st_payload12(a.out + (size_t)row * kHidden + (size_t)c * kChunk + i * 6,
                 q.x, q.y, q.z);
  }
}

}  // namespace

extern "C" int glm52_tp8_ar_launch_cuda(
    const void* partial, void* out, void* ar_local,
    const void* const* peer_ar, const void* epoch_dev,
    const void* active_rows, int layer_slot, int rows, int nranks, int myrank,
    cudaStream_t stream) {
  if (nranks != kRanks || myrank < 0 || myrank >= nranks || rows < 1 ||
      rows > kTokens || layer_slot < 0) {
    return (int)cudaErrorInvalidValue;
  }
  Glm52Tp8ArArgs a = {};
  a.partial = (const __nv_bfloat16*)partial;
  a.out = (__nv_bfloat16*)out;
  a.ar_local = (uint4*)ar_local;
  for (int p = 0; p < kRanks; ++p) {
    a.peer_ar[p] = (uint4*)peer_ar[p];
  }
  a.epoch_dev = (const unsigned long long*)epoch_dev;
  a.active_rows = (const int*)active_rows;
  a.layer_slot = layer_slot;
  a.rows = rows;
  a.nranks = nranks;
  a.myrank = myrank;

  const int push_blocks = (rows * nranks * kChunkPk + kThreads - 1) / kThreads;
  const int reduce_blocks = (rows * kChunkPk + kThreads - 1) / kThreads;
  tp8_ar_push_kernel<<<push_blocks, kThreads, 0, stream>>>(a);
  tp8_ar_reduce_bcast_kernel<<<reduce_blocks, kThreads, 0, stream>>>(a);
  tp8_ar_recv_kernel<<<push_blocks, kThreads, 0, stream>>>(a);
  return (int)cudaGetLastError();
}
