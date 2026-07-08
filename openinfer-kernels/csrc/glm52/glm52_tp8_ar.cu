// GLM5.2 TP8 attention allreduce: the o_proj (and dense-MLP down) epilogue
// collective for the attention-TP topology (docs/models/glm52/
// moe-tp8-low-latency.md, M4).
//
// Every rank computes a PARTIAL projection output for ALL bucket rows (its
// 8/64 heads' contribution, full hidden width). This pair of kernels sums
// the 8 partials so every rank ends up with the identical full result —
// one-shot radix-8 over LL packets, the protocol the R4 probe measured at
// ~5.8 us/layer marginal on 8xH200.
//
// Wire layout per layer slot: [parity 2][row kTokens][src kRanks][kRowPk]
// uint4 packets, 4 bf16 payload + tag per packet. The tag is the shared
// device-side step epoch (same counter the MoE chain uses; the AR region is
// separate VMM memory, so tags never collide across the two collectives).
// Reduce order is a fixed src 0..7 sum on every rank — outputs are
// bit-identical across ranks, which the replicated-activation topology
// RELIES on (router/sampling run redundantly on all ranks downstream).
//
// Iron rule (same as the MoE chain): phase ordering is kernel boundaries
// only, zero device fences; a spin may wait exclusively on cross-rank
// packets written by kernels of a PREVIOUS phase.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "glm52_tp8_ll.cuh"

namespace {

constexpr int kHidden = 6144;
constexpr int kRanks = 8;
constexpr int kTokens = 8;  // max bucket rows; pads ride along (want-mask TBD)
constexpr int kThreads = 256;
// One packet carries 4 bf16 (8 B payload) of one row's hidden vector.
constexpr int kRowPk = kHidden / 4;
static_assert(kHidden % 4 == 0, "row payload must pack into 4-lane packets");

struct Glm52Tp8ArArgs {
  const __nv_bfloat16* partial;  // [rows][H] this rank's partial, all rows
  __nv_bfloat16* out;            // [rows][H] reduced result (all ranks equal)
  uint4* ar_local;               // own AR buffer base
  uint4* peer_ar[kRanks];        // peer p's AR buffer + myrank*kRowPk
  const unsigned long long* epoch_dev;
  int layer_slot;
  int rows;
  int nranks, myrank;
};

#define TP8_AR_PREAMBLE                                                       \
  const int gt = blockIdx.x * kThreads + threadIdx.x;                         \
  const int GT = gridDim.x * kThreads;                                        \
  const unsigned long long ep = *a.epoch_dev;                                 \
  const unsigned tag = (unsigned)ep;                                          \
  const size_t ar_off =                                                       \
      ((size_t)a.layer_slot * 2 + (ep & 1)) * kTokens * kRanks * kRowPk;

__global__ void __launch_bounds__(kThreads) tp8_ar_push_kernel(
    Glm52Tp8ArArgs a) {
  TP8_AR_PREAMBLE
  // Flattened (row, dst, packet) grid, mirroring tp8_ag_push: each thread
  // reads 4 bf16 of its own partial and lands them in dst's buffer at this
  // rank's src slot. The 8x re-read of the payload (one per dst) stays L2
  // resident.
  for (int rp = gt; rp < a.rows * a.nranks * kRowPk; rp += GT) {
    const int row = rp / (a.nranks * kRowPk);
    const int rem = rp % (a.nranks * kRowPk);
    const int dst = rem / kRowPk, i = rem % kRowPk;
    const uint2 v = *reinterpret_cast<const uint2*>(
        a.partial + (size_t)row * kHidden + (size_t)i * 4);
    glm52_tp8_st_ll(
        a.peer_ar[dst] + ar_off + (size_t)row * kRanks * kRowPk + i, v, tag);
  }
}

__global__ void __launch_bounds__(kThreads) tp8_ar_recv_kernel(
    Glm52Tp8ArArgs a) {
  TP8_AR_PREAMBLE
  for (int rp = gt; rp < a.rows * kRowPk; rp += GT) {
    const int row = rp / kRowPk, i = rp % kRowPk;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    // Fixed src order: the sum is deterministic and identical on every rank.
    for (int src = 0; src < a.nranks; ++src) {
      uint4 q;
      glm52_tp8_ll_wait(
          a.ar_local + ar_off + ((size_t)row * kRanks + src) * kRowPk + i,
          tag, &q);
      const __nv_bfloat162 p01 = *reinterpret_cast<const __nv_bfloat162*>(&q.x);
      const __nv_bfloat162 p23 = *reinterpret_cast<const __nv_bfloat162*>(&q.y);
      acc0 += __bfloat162float(p01.x);
      acc1 += __bfloat162float(p01.y);
      acc2 += __bfloat162float(p23.x);
      acc3 += __bfloat162float(p23.y);
    }
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(
        a.out + (size_t)row * kHidden + (size_t)i * 4);
    o[0] = __floats2bfloat162_rn(acc0, acc1);
    o[1] = __floats2bfloat162_rn(acc2, acc3);
  }
}

}  // namespace

extern "C" int glm52_tp8_ar_launch_cuda(
    const void* partial, void* out, void* ar_local,
    const void* const* peer_ar, const void* epoch_dev, int layer_slot,
    int rows, int nranks, int myrank, cudaStream_t stream) {
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
  a.layer_slot = layer_slot;
  a.rows = rows;
  a.nranks = nranks;
  a.myrank = myrank;

  const int push_blocks =
      (rows * nranks * kRowPk + kThreads - 1) / kThreads;
  const int recv_blocks = (rows * kRowPk + kThreads - 1) / kThreads;
  tp8_ar_push_kernel<<<push_blocks, kThreads, 0, stream>>>(a);
  tp8_ar_recv_kernel<<<recv_blocks, kThreads, 0, stream>>>(a);
  return (int)cudaGetLastError();
}
