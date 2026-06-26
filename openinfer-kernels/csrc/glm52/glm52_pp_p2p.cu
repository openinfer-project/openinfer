// GLM5.2 PP8 stage-boundary P2P handoff (Slice 0 runtime spine).
//
// Each pipeline stage runs a single-context single-stream CUDA graph:
//   stage0:        source_inject -> dummy_burn -> send_hidden(->stage1)
//   stage 1..n-2:  wait_hidden   -> dummy_burn -> send_hidden(->stage i+1)
//   stage n-1:     wait_hidden   -> dummy_burn          (sink; only acks)
// Stages are serialized by DEVICE-MEMORY FLAGS over NVLink P2P, never by
// stream/event edges. The protocol is lifted from the two-GPU L_send microbench
// (tilert_play/benchmarks/p2p_lsend) and hardened for an n-stage chain:
//
//   * epoch is DEVICE state -- wait/source atomicAdd it, send reads it -- so the
//     captured graph carries no per-step host immediate and replays unchanged.
//   * every gate that can observe a desync TRAPS in-kernel rather than consume
//     the wrong hidden: a missed deadline (peer stalled/dead) or a ring lap
//     (producer overwrote a slot the consumer had not read) -> sticky context
//     error -> the coordinator's next stream sync returns CUDA_ERROR_* -> crash.
//     The invariant lives in the kernel, not in a host-side check that a wrong
//     payload could slip past.
//
// Per-stage persistent buffers (allocated once on the stage's own context):
//   hidden_in_ring[R*words] bf16  peer-writable; upstream send remote-stores here
//   flag_ring[R]      u64          peer-writable; upstream send releases epoch
//   epoch_counter[1]  u64          local; wait/source atomicAdd, send reads
//   ack_ring[R]       u64          local; downstream wait remote-stores (reverse)
//   err_code[1]       u32          local; 0=ok (codes below)
// slot = epoch % R.  R>=2 double-buffers so a serial bs=1 hop never WAR-blocks.
//
// err_code: 1=wait deadline, 2=wait ring-lap, 3=send WAR deadline,
//           4=send forward-RTT deadline.

#include "../common.cuh"

#include <cuda.h>

namespace {

__device__ __forceinline__ unsigned long long pp_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

// Chain head: advance the local epoch with no inbound wait. The send that
// follows is paced against the next stage by its own WAR / RTT gate, so the
// source never races more than R epochs ahead.
__global__ void glm52_pp_source_inject_kernel(unsigned long long* epoch) {
  if (threadIdx.x == 0) {
    atomicAdd(epoch, 1ULL);
  }
}

// Acquire side: gate on the inbound flag, then ack upstream. The receiver does
// NOT copy -- the payload is already in its local hidden_in_ring slot, the
// upstream send remote-stored it before releasing the flag.
__global__ void glm52_pp_wait_hidden_kernel(
    const unsigned long long* __restrict__ my_flag,
    unsigned long long* __restrict__ epoch,
    unsigned long long* __restrict__ up_ack,  // upstream ack_ring (peer VA)
    unsigned int* __restrict__ err_code,
    unsigned long long deadline_ns,
    int ring) {
  __shared__ unsigned long long s_e;
  __shared__ int s_slot;
  if (threadIdx.x == 0) {
    const unsigned long long e = atomicAdd(epoch, 1ULL) + 1ULL;
    const int slot = (int)(e % (unsigned long long)ring);
    s_e = e;
    s_slot = slot;
    volatile unsigned long long* flag = (volatile unsigned long long*)my_flag;
    const unsigned long long start = pp_globaltimer();
    unsigned long long v;
    for (;;) {
      v = flag[slot];
      if (v >= e) break;
      if (pp_globaltimer() - start > deadline_ns) {
        *err_code = 1u;  // deadline: upstream stalled or dead
        __threadfence_system();
        __trap();
      }
    }
    if (v != e) {
      *err_code = 2u;  // ring lap: producer overwrote a slot we had not read
      __threadfence_system();
      __trap();
    }
  }
  __syncthreads();
  // Acquire: the in-stream stage compute that follows must observe the payload
  // bytes the upstream remote-stored before it released the flag.
  __threadfence_system();
  if (threadIdx.x == 0) {
    // Reverse ack -> upstream's WAR release + forward-RTT timer.
    // NOTE(Slice 7): this fires at flag-observe, BEFORE the stage compute reads
    // the payload. Once a real layer consumes the slot, move the ack to after
    // that read, else the R-deep WAR gate under-protects an in-use slot.
    ((volatile unsigned long long*)up_ack)[s_slot] = s_e;
    __threadfence_system();
  }
}

// Release side: remote-store the payload into the peer ring slot, fence, then
// release the flag. With deltas != null also measures the forward half-RTT
// (send -> downstream wait -> reverse ack) using the producer's own globaltimer,
// which is clock-skew-immune; with deltas == null (production) the RTT spin is
// skipped and the hop is fully async.
__global__ void glm52_pp_send_hidden_kernel(
    const int4* __restrict__ src_hidden,         // local hidden, 16B aligned
    int4* __restrict__ peer_hidden,              // downstream hidden_in_ring (peer VA)
    unsigned long long* __restrict__ peer_flag,  // downstream flag_ring (peer VA)
    const unsigned long long* __restrict__ epoch,  // local; wait/source advanced it
    const unsigned long long* __restrict__ down_ack,  // local ack_ring (downstream wrote)
    unsigned long long* __restrict__ deltas,     // optional RTT samples
    unsigned int* __restrict__ err_code,
    int words,                                   // bf16 elems this step, multiple of 8
    int ring,
    unsigned long long warmup,                   // deltas index base
    unsigned long long n_samples,                // deltas capacity
    unsigned long long deadline_ns) {
  __shared__ unsigned long long s_e;
  __shared__ int s_slot;
  __shared__ unsigned long long s_t0;
  if (threadIdx.x == 0) {
    const unsigned long long e = *epoch;  // wait/source already advanced it
    s_e = e;
    s_slot = (int)(e % (unsigned long long)ring);
    // WAR gate: the previous occupant of this slot (epoch e-ring) must have been
    // consumed downstream before we overwrite it. No-op at bs=1 (R>=2 never
    // blocks); real backpressure once microbatches pipeline.
    volatile unsigned long long* ack = (volatile unsigned long long*)down_ack;
    const unsigned long long start = pp_globaltimer();
    while (ack[s_slot] + (unsigned long long)ring < e) {
      if (pp_globaltimer() - start > deadline_ns) {
        *err_code = 3u;  // WAR deadline: downstream never freed the slot
        __threadfence_system();
        __trap();
      }
    }
    // Start the forward-RTT clock only after the WAR gate clears, so the recorded
    // delta is pure L_send and never absorbs single-buffer (R=1) backpressure.
    s_t0 = (deltas != nullptr) ? pp_globaltimer() : 0ULL;
  }
  __syncthreads();
  const unsigned long long e = s_e;
  const int slot = s_slot;

  const int words4 = words >> 3;  // 8 bf16 per int4 (128-bit remote store)
  int4* dst = peer_hidden + (long long)slot * (long long)words4;
  for (int i = threadIdx.x; i < words4; i += blockDim.x) {
    dst[i] = src_hidden[i];
  }
  // Each thread flushes its OWN payload stores to system scope, THEN the barrier
  // guarantees thread 0's flag release at line below follows every thread's
  // fence -- fencing after the barrier would only order thread 0's own stores.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ((volatile unsigned long long*)peer_flag)[slot] = e;  // release
    __threadfence_system();
    if (deltas != nullptr) {
      volatile unsigned long long* ack = (volatile unsigned long long*)down_ack;
      const unsigned long long start = pp_globaltimer();
      while (ack[slot] < e) {  // wait for downstream to ack THIS epoch
        if (pp_globaltimer() - start > deadline_ns) {
          *err_code = 4u;  // forward-RTT deadline
          __threadfence_system();
          __trap();
        }
      }
      const unsigned long long t1 = pp_globaltimer();
      if (e > warmup) {
        const unsigned long long idx = e - warmup - 1ULL;
        if (idx < n_samples) deltas[idx] = t1 - s_t0;
      }
    }
  }
}

// Models a stage's per-token compute time so the spine can sweep handoff cost
// against realistic stage latency. Latency-only (one warp spinning globaltimer).
__global__ void glm52_pp_dummy_burn_kernel(unsigned long long burn_ns) {
  if (threadIdx.x == 0 && burn_ns > 0ULL) {
    const unsigned long long start = pp_globaltimer();
    while (pp_globaltimer() - start < burn_ns) {
    }
  }
}

}  // namespace

extern "C" {

CUresult glm52_pp_source_inject(unsigned long long* epoch, cudaStream_t stream) {
  if (epoch == nullptr) return CUDA_ERROR_INVALID_VALUE;
  glm52_pp_source_inject_kernel<<<1, 1, 0, stream>>>(epoch);
  return consume_last_cuda_error();
}

CUresult glm52_pp_wait_hidden(
    const unsigned long long* my_flag, unsigned long long* epoch,
    unsigned long long* up_ack, unsigned int* err_code,
    unsigned long long deadline_ns, int ring, cudaStream_t stream) {
  if (my_flag == nullptr || epoch == nullptr || up_ack == nullptr ||
      err_code == nullptr || ring <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_pp_wait_hidden_kernel<<<1, 32, 0, stream>>>(
      my_flag, epoch, up_ack, err_code, deadline_ns, ring);
  return consume_last_cuda_error();
}

CUresult glm52_pp_send_hidden(
    const void* src_hidden, void* peer_hidden, unsigned long long* peer_flag,
    const unsigned long long* epoch, const unsigned long long* down_ack,
    unsigned long long* deltas, unsigned int* err_code, int words, int ring,
    unsigned long long warmup, unsigned long long n_samples,
    unsigned long long deadline_ns, cudaStream_t stream) {
  if (src_hidden == nullptr || peer_hidden == nullptr || peer_flag == nullptr ||
      epoch == nullptr || down_ack == nullptr || err_code == nullptr ||
      ring <= 0 || words <= 0 || (words & 7) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  glm52_pp_send_hidden_kernel<<<1, 256, 0, stream>>>(
      static_cast<const int4*>(src_hidden), static_cast<int4*>(peer_hidden),
      peer_flag, epoch, down_ack, deltas, err_code, words, ring, warmup,
      n_samples, deadline_ns);
  return consume_last_cuda_error();
}

CUresult glm52_pp_dummy_burn(unsigned long long burn_ns, cudaStream_t stream) {
  glm52_pp_dummy_burn_kernel<<<1, 32, 0, stream>>>(burn_ns);
  return consume_last_cuda_error();
}

}  // extern "C"
