// GLM5.2 bs=1 weight-only FP8 GEMV (bf16 activation, on-the-fly e4m3 block-scale
// weight dequant, f32 accumulate). Replaces the TRTLLM CUTLASS block-scale GEMM at
// M=1, where the M-tile pads 1->64 and runs compute-bound; this reads each weight
// exactly once and is weight-memory-bound. dequant matches the host reference
// glm52/src/mla_decode.rs:188-189: deq(W) = float(e4m3(W)) * weight_scale_inv.
//
// One __device__ row-tile core serves both:
//   - the routed grouped path: grid.x = slot in [0,topk), expert = topk_idx[slot];
//     computes exactly the topk live experts x 1 real row each (no 256-group, no
//     512-row pad). The topk multiplier is also the occupancy lever (topk x blocks).
//   - the plain linear path: groups=1, expert 0, broadcast activation.
//
// Activation is NOT staged in shared: it is read straight from global (L2-resident,
// reused across every block). Staging it cost a 32KB-shared prologue + __syncthreads +
// a shared-occupancy cap that throttled the long-K GEMVs hardest. Dropping the stage
// (H200 sm_90 microbench /tmp/gemv_grouped + /tmp/gemv_v2, bit-identical chksum):
//   W13 gate_up 67->83%, W2 down 63->75%, o_proj 54->73%, q_b 59->79% HBM BW.
// (Blog "Twelve Attempts": keep the reused vector hot in L2, not shared; shared only
// pays off for re-read data, and a streamed GEMV re-reads nothing.) rows/warp is the
// second lever: short-K amortises per-block overhead with 4 rows/warp, long-K fills the
// grid with 1 (grouped is topk-saturated -> always 4; plain dispatches on k).

#include "../common.cuh"  // warp_reduce_sum

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarpSize     = 32;
constexpr int kWarpsPerBlk  = 8;
constexpr int kBlockThreads = kWarpSize * kWarpsPerBlk;       // 256
constexpr int kVec          = 16;                             // 16 e4m3 = 128-bit LDG
constexpr int kFp8Block     = 128;
constexpr int kStep         = kWarpSize * kVec;               // 512 k per warp step

// The grouped (routed) path keeps 4 rows/warp: grid.x = topk already saturates the SMs,
// and at full occupancy more rows/warp amortise the per-block overhead (microbench W13
// RW1->RW4 at STAGE0: 74->83%). The plain path dispatches rows/warp on k (see the
// launcher): short-K (q_b k=2048) wants 4 to amortise; long-K (o_proj k=16384) wants 1
// to fill the single grid column. Both choices are bit-identical -- only the warp->row
// mapping changes, never a row's dot or its accumulation order.
constexpr int kRowsGrouped         = 4;
constexpr int kRowsPerBlockGrouped = kWarpsPerBlk * kRowsGrouped;  // 32
constexpr int kRowsPlainShortK     = 4;  // k <= 2048
constexpr int kRowsPlainLongK      = 1;  // k  > 2048

constexpr int kKindW13 = 1, kKindW13N = 4096, kKindW13K = 6144;
constexpr int kKindW2  = 2, kKindW2N  = 6144, kKindW2K  = 2048;

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer)
    return CUDA_ERROR_INVALID_VALUE;
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}
CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

// The activation is read with 16-byte float4/uint4 loads straight from global (no shared
// staging absorbs misalignment any more), so its base must be 16-byte aligned. CudaSlice
// allocation bases are 256B-aligned and satisfy this; guard here so a future misaligned
// sub-view crashes early at the boundary instead of faulting mid-kernel.
bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// One warp owns ROWS output rows starting at n0; sweeps the whole K of a single weight
// matrix (w_base/scale_base already resolved to this expert) against the bf16 activation
// row (xs, read straight from global/L2 -- not staged). No split-K, no cross-warp
// reduction. ROWS is the per-path tile (grouped 4; plain dispatched on k, see launcher).
template <int ROWS>
__device__ __forceinline__ void gemv_row_tile(
    const __nv_bfloat16* __restrict__ xs,        // activation [k], read from global/L2
    const unsigned char* __restrict__ w_base,    // fp8 e4m3 [n, k] for this expert
    const float* __restrict__ scale_base,        // f32 [n/128, k/128] for this expert
    __nv_bfloat16* __restrict__ out_row,         // [n] output for this slot
    int n, int k) {
  constexpr int kRowsPerBlock = kWarpsPerBlk * ROWS;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0   = blockIdx.y * kRowsPerBlock + warp * ROWS;
  const int scale_cols = k >> 7;                 // k/128
  // All ROWS rows share one weight-scale row: n0 is a multiple of ROWS and ROWS | 128,
  // so n0..n0+ROWS-1 never straddle a /128 boundary.
  const float* scale_row = scale_base + (size_t)(n0 >> 7) * scale_cols;
  const float4* xs4 = reinterpret_cast<const float4*>(xs);

  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

  for (int kk = lane * kVec; kk < k; kk += kStep) {
    const float scale = scale_row[kk >> 7];      // one f32 per (lane, k-chunk)
    // Keep the 16 activations as raw bf16 bits (8 regs), convert inline at use --
    // materialising float x[16] cost 16 regs and capped occupancy (ncu: reg-limited).
    float4 xv0 = xs4[(kk >> 3)];
    float4 xv1 = xs4[(kk >> 3) + 1];
    const __nv_bfloat16* xh0 = reinterpret_cast<const __nv_bfloat16*>(&xv0);
    const __nv_bfloat16* xh1 = reinterpret_cast<const __nv_bfloat16*>(&xv1);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      // 128-bit coalesced e4m3 weight load: 16 contiguous weights of row (n0+r).
      // __ldcs = cache-streaming (evict-first): the weight is read exactly once per
      // GEMV, so keeping it out of L1 stops one-shot data from evicting the staged
      // activation's working set. Microbench (long-K o_proj k=16384): 1.23x HBM BW,
      // neutral on short-K, zero accuracy/register cost. (KernelWiki: memory-bound
      // GEMV -> cache-policy differentiation is THE lever, not cp.async/ILP.)
      const uint4 wp = __ldcs(reinterpret_cast<const uint4*>(
          w_base + ((size_t)(n0 + r) * k) + kk));
      // Vectorised dequant: decode the 16 e4m3 weights two-at-a-time via the hardware
      // fp8x2->half2 path (8 cvts, not 16). e4m3 (3 mantissa bits) is exactly
      // representable in f16, so this is bit-identical to scalar `float(e4m3)` while
      // halving the decode-instruction count that caps this memory-bound GEMV below
      // peak BW. Activation stays inline bf16 (materialising float x[16] is reg-limited,
      // see above). Microbench grouped R4: +5-7% HBM BW, re 0.0000. (KernelWiki
      // contest-gpumode-p1: vectorise the low-precision dequant.)
      const __nv_fp8x2_e4m3* w2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp);
      float partial = 0.0f;
      // Per-term accumulation (NOT pair-grouped): preserves the original left-to-right
      // FMA association, so `partial` is bit-identical to the scalar path -- greedy
      // decode over 78 layers is sensitive to sub-ULP wobble and would otherwise
      // diverge. The only change vs scalar is the weight *decode* (fp8x2->half2), and
      // __low2float(h) == float(e4m3) exactly (e4m3 mantissa fits in f16).
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h = static_cast<__half2>(w2[j]);
        partial += __low2float(h) * __bfloat162float(xh0[2 * j]);
        partial += __high2float(h) * __bfloat162float(xh0[2 * j + 1]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h = static_cast<__half2>(w2[4 + j]);
        partial += __low2float(h) * __bfloat162float(xh1[2 * j]);
        partial += __high2float(h) * __bfloat162float(xh1[2 * j + 1]);
      }
      acc[r] += scale * partial;                 // block-scale hoisted (const over 128 cols)
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = warp_reduce_sum(acc[r]);           // common.cuh
    if (lane == 0) out_row[n0 + r] = __float2bfloat16(v);
  }
}

// Plain linear GEMV: single weight matrix, broadcast activation (groups=1). ROWS is
// dispatched on k by the launcher (short-K 4, long-K 1).
template <int ROWS>
__global__ void glm52_fp8_weight_only_gemv_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [k]
    const unsigned char* __restrict__ weight,      // [n, k] e4m3
    const float* __restrict__ weight_scale,        // [n/128, k/128]
    __nv_bfloat16* __restrict__ out,               // [n]
    int n, int k) {
  gemv_row_tile<ROWS>(activation, weight, weight_scale, out, n, k);  // x straight from L2
}

// Batched weight-stationary GEMV: one warp owns ONE output row n0 and sweeps K once,
// carrying BATCH accumulators — the weight is still read exactly once (the whole point
// of batching a weight-memory-bound GEMV), only the FMA count grows with BATCH.
// Each batch row's dot uses the same lane sweep, per-term order, and warp reduction
// as gemv_row_tile<1>, so every row is bit-identical to the m=1 kernel run alone.
// The BATCH activation rows ([BATCH, k] bf16, ≤ 256 KB at the largest K) stay
// L2-resident across the n/8 blocks, same as the single row does today.
template <int BATCH>
__device__ __forceinline__ void gemv_row_tile_batched(
    const __nv_bfloat16* __restrict__ xs,        // activation [BATCH, k]
    const unsigned char* __restrict__ w_base,    // fp8 e4m3 [n, k]
    const float* __restrict__ scale_base,        // f32 [n/128, k/128]
    __nv_bfloat16* __restrict__ out,             // [BATCH, n]
    int n, int k) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0   = blockIdx.y * kWarpsPerBlk + warp;  // 1 row/warp
  const int scale_cols = k >> 7;
  const float* scale_row = scale_base + (size_t)(n0 >> 7) * scale_cols;

  float acc[BATCH];
#pragma unroll
  for (int b = 0; b < BATCH; ++b) acc[b] = 0.0f;

  for (int kk = lane * kVec; kk < k; kk += kStep) {
    const float scale = scale_row[kk >> 7];
    const uint4 wp = __ldcs(reinterpret_cast<const uint4*>(
        w_base + ((size_t)n0 * k) + kk));
    const __nv_fp8x2_e4m3* w2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp);
#pragma unroll
    for (int b = 0; b < BATCH; ++b) {
      const float4* xs4 = reinterpret_cast<const float4*>(xs + (size_t)b * k);
      float4 xv0 = xs4[(kk >> 3)];
      float4 xv1 = xs4[(kk >> 3) + 1];
      const __nv_bfloat16* xh0 = reinterpret_cast<const __nv_bfloat16*>(&xv0);
      const __nv_bfloat16* xh1 = reinterpret_cast<const __nv_bfloat16*>(&xv1);
      float partial = 0.0f;
      // Same per-term left-to-right association as gemv_row_tile — bit parity
      // per row is the contract (see that kernel's comment).
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h = static_cast<__half2>(w2[j]);
        partial += __low2float(h) * __bfloat162float(xh0[2 * j]);
        partial += __high2float(h) * __bfloat162float(xh0[2 * j + 1]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h = static_cast<__half2>(w2[4 + j]);
        partial += __low2float(h) * __bfloat162float(xh1[2 * j]);
        partial += __high2float(h) * __bfloat162float(xh1[2 * j + 1]);
      }
      acc[b] += scale * partial;
    }
  }
#pragma unroll
  for (int b = 0; b < BATCH; ++b) {
    float v = warp_reduce_sum(acc[b]);
    if (lane == 0) out[(size_t)b * n + n0] = __float2bfloat16(v);
  }
}

template <int BATCH>
__global__ void glm52_fp8_weight_only_gemv_batched_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [BATCH, k]
    const unsigned char* __restrict__ weight,      // [n, k] e4m3
    const float* __restrict__ weight_scale,        // [n/128, k/128]
    __nv_bfloat16* __restrict__ out,               // [BATCH, n]
    int n, int k) {
  gemv_row_tile_batched<BATCH>(activation, weight, weight_scale, out, n, k);
}

// Routed grouped GEMV: blockIdx.x = slot in [0,topk); expert = topk_idx[slot].
__global__ void glm52_moe_fp8_weight_only_gemv_kernel(
    const __nv_bfloat16* __restrict__ activation,  // W13: [k]; W2: [topk, k]
    int act_row_stride,                            // 0 (W13 broadcast) | k (W2)
    const int* __restrict__ topk_idx,              // [topk]
    const unsigned char* __restrict__ weight,      // [experts, n, k] e4m3
    const float* __restrict__ weight_scale,        // [experts, n/128, k/128]
    __nv_bfloat16* __restrict__ out,               // [topk, n]
    int n, int k) {
  const int slot   = blockIdx.x;
  const int expert = topk_idx[slot];
  const __nv_bfloat16* act_row = activation + (size_t)slot * act_row_stride;
  const size_t scale_stride = (size_t)(n >> 7) * (k >> 7);    // (n/128)*(k/128) per expert
  gemv_row_tile<kRowsGrouped>(act_row,                        // x straight from L2
                              weight + ((size_t)expert * n) * k,
                              weight_scale + (size_t)expert * scale_stride,
                              out + (size_t)slot * n,
                              n, k);
}

// Weighted SiLU(gate)*up -> bf16, route weight folded per slot. Consumes the W13
// GEMV output [rows, 2*inter] (gate|up) and emits the bf16 W2 GEMV input
// [rows, inter]. bf16 companion of silu_and_mul_per_token_group_quant_bf16 with the
// fp8 quant dropped -- the W2 GEMV takes bf16 activation directly.
__global__ void glm52_silu_and_mul_weighted_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,    // [rows, 2*inter] (gate|up)
    const float* __restrict__ topk_weights,     // [rows] route weight per slot (or null)
    __nv_bfloat16* __restrict__ output,         // [rows, inter]
    int rows, int inter) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || col >= inter) return;
  const __nv_bfloat16* gate = input + (size_t)row * (2 * inter);
  const __nv_bfloat16* up = gate + inter;
  const float g = __bfloat162float(gate[col]);
  const float u = __bfloat162float(up[col]);
  const float sg = 1.0f / (1.0f + expf(-g));
  const float w = topk_weights == nullptr ? 1.0f : __ldg(topk_weights + row);
  output[(size_t)row * inter + col] = __float2bfloat16(g * sg * u * w);
}

// Combine the topk slot rows -> routed[n] (the route weight is already folded into
// the W2 input by the weighted SiLU, so this is a plain sum -- one row per slot, no
// expert_offsets indirection).
__global__ void glm52_moe_combine_slots_kernel(
    const __nv_bfloat16* __restrict__ w2_out,   // [topk, n]
    __nv_bfloat16* __restrict__ routed,         // [n]
    int n, int topk) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  float acc = 0.0f;
  for (int j = 0; j < topk; ++j) acc += __bfloat162float(w2_out[(size_t)j * n + c]);
  routed[c] = __float2bfloat16(acc);
}

bool valid_gemv_shape(int operand_kind, int n, int k) {
  if (operand_kind == kKindW13) return n == kKindW13N && k == kKindW13K;
  if (operand_kind == kKindW2)  return n == kKindW2N  && k == kKindW2K;
  return false;
}

// Kernel-coverage + scale-index invariants. k%128 keeps the scale column stride
// exact; n%rpb and k%512 keep the row-tile / lane sweep exact (no tail masking). n
// need NOT be %128: a warp owns ROWS consecutive rows from an rpb-row block based at a
// multiple of rpb (ROWS | 128), so its rows never straddle a /128 scale boundary -- the
// caller just sizes the scale buffer with div_ceil(n,128) rows (MLA kv_a n=576 -> 5
// rows). Encoding these turns any future off-shape into a crash, not a silent read.
bool valid_tiling(int n, int k, int rpb) {
  return n > 0 && k > 0 && k % kFp8Block == 0 && n % rpb == 0 && k % kStep == 0;
}

// rows/warp the plain launcher will use for this k (see kRowsPlainShortK/LongK).
int plain_rows_per_warp(int k) {
  return k <= 2048 ? kRowsPlainShortK : kRowsPlainLongK;
}

// The batched kernel's supported batch (one instantiation: the DP scheduler's
// fixed per-rank decode batch). Rust's GLM52_MAX_BATCH_PER_RANK must match —
// the launcher rejects any other batch, so a drift crashes at the boundary.
constexpr int kBatchedGemvBatch = 8;

// Whitelist of the model's linear shapes (shared by the plain and batched
// launchers; the tiling check differs — rpb is 8 or 32 for plain, 8 for
// batched — so it is applied by each launcher, not here).
bool whitelisted_linear_shape(int n, int k) {
  if (n == 2048  && k == 6144)  return true;  // q_a / shared gate,up
  if (n == 16384 && k == 2048)  return true;  // q_b
  if (n == 576   && k == 6144)  return true;  // kv_a + rope (partial-N: 576%128!=0)
  if (n == 6144  && k == 16384) return true;  // o_proj
  if (n == 24576 && k == 6144)  return true;  // dense gate|up (packed)
  if (n == 6144  && k == 12288) return true;  // dense down
  if (n == 4096  && k == 6144)  return true;  // shared gate|up (packed)
  if (n == 6144  && k == 2048)  return true;  // shared down
  if (n == 4096  && k == 2048)  return true;  // indexer wq_b
  if (n == 128   && k == 6144)  return true;  // indexer wk
  return false;
}

// Encoding these turns any future off-shape into a crash, not a silent read.
bool valid_linear_shape(int n, int k) {
  return valid_tiling(n, k, kWarpsPerBlk * plain_rows_per_warp(k)) &&
         whitelisted_linear_shape(n, k);
}

}  // namespace

extern "C" {

CUresult glm52_moe_fp8_weight_only_gemv_cuda(
    int operand_kind, const __nv_bfloat16* activation, int act_row_stride,
    const int* topk_idx, const unsigned char* weight, const float* weight_scale,
    __nv_bfloat16* out, int topk, int n, int k, cudaStream_t stream) {
  if (activation == nullptr || topk_idx == nullptr || weight == nullptr ||
      weight_scale == nullptr || out == nullptr || topk <= 0 ||
      !aligned16(activation) || !valid_gemv_shape(operand_kind, n, k) ||
      !valid_tiling(n, k, kRowsPerBlockGrouped)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid(topk, n / kRowsPerBlockGrouped, 1);
  glm52_moe_fp8_weight_only_gemv_kernel<<<grid, kBlockThreads, 0, stream>>>(
      activation, act_row_stride, topk_idx, weight, weight_scale, out, n, k);
  return consume_last_cuda_error();
}

CUresult glm52_fp8_weight_only_gemv_cuda(
    const __nv_bfloat16* activation, const unsigned char* weight,
    const float* weight_scale, __nv_bfloat16* out, int n, int k,
    cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || weight_scale == nullptr ||
      out == nullptr || !aligned16(activation) || !valid_linear_shape(n, k)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int rows = plain_rows_per_warp(k);
  const dim3 grid(1, n / (kWarpsPerBlk * rows), 1);
  if (rows == kRowsPlainShortK) {
    glm52_fp8_weight_only_gemv_kernel<kRowsPlainShortK>
        <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale, out, n, k);
  } else {
    glm52_fp8_weight_only_gemv_kernel<kRowsPlainLongK>
        <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale, out, n, k);
  }
  return consume_last_cuda_error();
}

// Batched plain GEMV: `out[b] = deq(weight) @ activation[b]` for b in [0, batch).
// batch == 1 routes to the m=1 kernel (identical result — the batched tile is
// per-row bit-identical to it by construction); the only other supported batch
// is kBatchedGemvBatch (the DP scheduler's fixed per-rank decode batch).
CUresult glm52_fp8_weight_only_gemv_batched_cuda(
    const __nv_bfloat16* activation, const unsigned char* weight,
    const float* weight_scale, __nv_bfloat16* out, int batch, int n, int k,
    cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || weight_scale == nullptr ||
      out == nullptr || !aligned16(activation) ||
      !whitelisted_linear_shape(n, k) || !valid_tiling(n, k, kWarpsPerBlk)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (batch == 1) {
    return glm52_fp8_weight_only_gemv_cuda(activation, weight, weight_scale, out,
                                           n, k, stream);
  }
  if (batch != kBatchedGemvBatch) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid(1, n / kWarpsPerBlk, 1);
  glm52_fp8_weight_only_gemv_batched_kernel<kBatchedGemvBatch>
      <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale, out, n, k);
  return consume_last_cuda_error();
}

CUresult glm52_silu_and_mul_weighted_bf16_cuda(
    const __nv_bfloat16* input, const float* topk_weights, __nv_bfloat16* output,
    int rows, int inter, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || inter <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int threads = 256;
  const dim3 grid(rows, (inter + threads - 1) / threads, 1);
  glm52_silu_and_mul_weighted_bf16_kernel<<<grid, threads, 0, stream>>>(
      input, topk_weights, output, rows, inter);
  return consume_last_cuda_error();
}

CUresult glm52_moe_combine_slots_cuda(const __nv_bfloat16* w2_out,
                                      __nv_bfloat16* routed, int n, int topk,
                                      cudaStream_t stream) {
  if (w2_out == nullptr || routed == nullptr || n <= 0 || topk <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  glm52_moe_combine_slots_kernel<<<blocks, threads, 0, stream>>>(w2_out, routed, n,
                                                                 topk);
  return consume_last_cuda_error();
}

}  // extern "C"
