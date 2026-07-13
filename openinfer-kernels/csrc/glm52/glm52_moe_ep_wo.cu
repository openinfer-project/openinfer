// GLM5.2 EP4 weight-only routed-expert chain: the Blackwell-native
// replacement for the sm_90a DeepGEMM masked grouped GEMM on the EP decode
// path. bf16 recv activations x on-the-fly e4m3 block-scale weight dequant,
// f32 accumulation on mma.m16n8k16.bf16 — the glm52_moe_gemv.cu batched-mma
// treatment applied per expert instead of per dense matrix.
//
// The chain works directly on the DeepEP aligned receive layout and never
// builds the masked slabs: every local expert's real rows are contiguous at
// a 64-aligned segment start, and 8 | 64, so an 8-row tile never straddles
// an expert boundary. A tiny metadata kernel scans psum_expert into a
// compact tile work list (expert id, aligned row base, live rows <= 8); the
// GEMMs and the SiLU read tiles, so no row_map, no fp8 activation re-quant,
// no masked->aligned remap. W2 writes its bf16 rows straight into the
// aligned slots decode_combine addresses.
//
// Shapes are host-quiet and CUDA-graph capturable: grids are sized at the
// host-known worst-case tile count; blocks at or past the device tile count
// retire on one 4-byte read. The metadata kernel device-traps on any
// cross-rank token-count disagreement (same contract as the EP8 metadata
// kernel).
//
// Numerics: fp8 e4m3 is exactly representable in bf16, so the weight decode
// is lossless; accumulation is f32 with a fixed per-shape order (mma slot
// order + per-128-column scale application), deterministic per bucket. The
// activations skip the EP8 chain's fp8 re-quant entirely.

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kExpertAlignment = 64;
constexpr int kTileRows = 8;
constexpr int kMmaWarps = 4;
constexpr int kNTiles = 2;  // 16-row n-tiles per warp -> 128 output rows/block
constexpr int kSiluThreads = 256;

__device__ __forceinline__ int align_up_int(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ int clamp_nonnegative(int value) {
  return value < 0 ? 0 : value;
}

// psum_expert (i32 aligned running ends, the DeepEP dispatch metadata) ->
// compact tile list. One block; per-expert bounds are checked exactly like
// the EP8 metadata kernel (a segment past m_capacity or masked_cap means the
// ranks disagreed about global_tokens — trap instead of multiplying stale
// rows into real outputs). tiles[t] = {aligned row base, expert | rows<<16}.
__global__ void moe_ep_wo_tiles_kernel(const int* __restrict__ psum_expert,
                                       int2* __restrict__ tiles,
                                       int* __restrict__ tile_count,
                                       int groups, int m_capacity,
                                       int masked_cap, int max_tiles) {
  if (threadIdx.x != 0) {
    return;
  }
  int count_out = 0;
  for (int expert = 0; expert < groups; ++expert) {
    const int previous_end =
        expert == 0 ? 0 : clamp_nonnegative(psum_expert[expert - 1]);
    const int end = clamp_nonnegative(psum_expert[expert]);
    const int start = expert == 0 ? 0 : align_up_int(previous_end, kExpertAlignment);
    const int count = end - start;
    if (start > m_capacity || align_up_int(end, kExpertAlignment) > m_capacity ||
        count < 0 || count > masked_cap) {
      __trap();
    }
    for (int r = 0; r < count; r += kTileRows) {
      const int rows = min(kTileRows, count - r);
      if (count_out >= max_tiles) {
        __trap();
      }
      tiles[count_out] = make_int2(start + r, expert | (rows << 16));
      ++count_out;
    }
  }
  *tile_count = count_out;
}

// fp8 e4m3 pair -> packed bf16x2, exact (mirrors glm52_moe_gemv.cu).
__device__ __forceinline__ unsigned mma_cvt_pair(unsigned char b0, unsigned char b1) {
  __nv_fp8x2_e4m3 p;
  p.__x = (unsigned short)(b0 | (b1 << 8));
  __half2 h = static_cast<__half2>(p);
  float2 f = __half22float2(h);
  __nv_bfloat162 bb = __float22bfloat162_rn(f);
  return *reinterpret_cast<unsigned*>(&bb);
}

// One expert tile (<= 8 aligned rows, one weight matrix) x one 128-row output
// chunk per block: the glm52_gemv_batched_mma_kernel structure with the
// weight/scale bases resolved per tile from the expert bank and the batch
// replaced by the tile's live rows. Full-K single pass (no k-split, no
// partial scratch): accumulation order is fixed per shape, and the store is
// the only global write. Blocks whose tile index is at or past *tile_count
// retire immediately (capacity-shaped grid, graph-stable).
__global__ __launch_bounds__(kMmaWarps* WARP_SIZE) void
moe_ep_wo_masked_mma_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [expanded, k] aligned rows
    const unsigned char* __restrict__ weight,      // [groups, n, k] e4m3 bank
    const float* __restrict__ weight_scale,        // [groups, n/128, k/128]
    const int2* __restrict__ tiles,
    const int* __restrict__ tile_count,
    const float* __restrict__ row_weights,  // per aligned row f32 scale, or null
    __nv_bfloat16* __restrict__ out,               // [expanded, n] aligned rows
    int n, int k) {
  if (static_cast<int>(blockIdx.y) >= *tile_count) {
    return;
  }
  const int2 tile = tiles[blockIdx.y];
  const int row_base = tile.x;
  const int expert = tile.y & 0xffff;
  const int live_rows = tile.y >> 16;

  const unsigned char* w_base = weight + (size_t)expert * n * k;
  const float* scale_base = weight_scale + (size_t)expert * (n >> 7) * (k >> 7);

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile0 = (blockIdx.x * kMmaWarps + warp) * kNTiles;  // 16-row n-tiles
  if (tile0 * 16 >= n) return;
  const int gid = lane >> 2, tid = lane & 3;
  const int scale_cols = k >> 7;

  float macc[kNTiles][4], cacc[kNTiles][4];
#pragma unroll
  for (int t = 0; t < kNTiles; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) { macc[t][i] = 0.f; cacc[t][i] = 0.f; }

  // Per chain: weight rows (gid, gid+8) of its 16-row tile; one 16B packet
  // per row per k64 super-chunk, 2-deep prefetch (see glm52_moe_gemv.cu for
  // the slot permutation argument).
  const unsigned char* w0[kNTiles];
  const unsigned char* w1[kNTiles];
#pragma unroll
  for (int t = 0; t < kNTiles; ++t) {
    const int n0 = (tile0 + t) * 16;
    w0[t] = w_base + (size_t)(n0 + gid) * k + tid * 16;
    w1[t] = w_base + (size_t)(n0 + gid + 8) * k + tid * 16;
  }

  uint4 wp0[kNTiles], wp1[kNTiles];
#pragma unroll
  for (int t = 0; t < kNTiles; ++t) {
    wp0[t] = __ldcs(reinterpret_cast<const uint4*>(w0[t]));
    wp1[t] = __ldcs(reinterpret_cast<const uint4*>(w1[t]));
  }
  for (int kk = 0; kk < k; kk += 64) {
    uint4 c0[kNTiles], c1[kNTiles];
#pragma unroll
    for (int t = 0; t < kNTiles; ++t) {
      c0[t] = wp0[t]; c1[t] = wp1[t];
      w0[t] += 64; w1[t] += 64;
    }
    if (kk + 64 < k) {
#pragma unroll
      for (int t = 0; t < kNTiles; ++t) {
        wp0[t] = __ldcs(reinterpret_cast<const uint4*>(w0[t]));
        wp1[t] = __ldcs(reinterpret_cast<const uint4*>(w1[t]));
      }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {  // four k16 mma steps per k64 super-chunk
      unsigned b01 = 0, b23 = 0;
      if (gid < live_rows) {
        const __nv_bfloat16* xrow =
            activation + (size_t)(row_base + gid) * k + kk + tid * 16 + 4 * s;
        const uint2 bv = *reinterpret_cast<const uint2*>(xrow);
        b01 = bv.x; b23 = bv.y;
      }
#pragma unroll
      for (int t = 0; t < kNTiles; ++t) {
        const unsigned char* p0 = reinterpret_cast<const unsigned char*>(&c0[t]) + 4 * s;
        const unsigned char* p1 = reinterpret_cast<const unsigned char*>(&c1[t]) + 4 * s;
        unsigned a0 = mma_cvt_pair(p0[0], p0[1]);
        unsigned a1 = mma_cvt_pair(p1[0], p1[1]);
        unsigned a2 = mma_cvt_pair(p0[2], p0[3]);
        unsigned a3 = mma_cvt_pair(p1[2], p1[3]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(cacc[t][0]), "+f"(cacc[t][1]), "+f"(cacc[t][2]), "+f"(cacc[t][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b01), "r"(b23));
      }
    }
    if (((kk + 64) & 127) == 0) {  // end of a 128-col scale group
#pragma unroll
      for (int t = 0; t < kNTiles; ++t) {
        // A 16-row weight tile never straddles a /128 scale-row boundary.
        const float scale =
            scale_base[(size_t)(((tile0 + t) * 16) >> 7) * scale_cols + (kk >> 7)];
#pragma unroll
        for (int i = 0; i < 4; ++i) { macc[t][i] += scale * cacc[t][i]; cacc[t][i] = 0.f; }
      }
    }
  }
  // C fragment: c0=(weight row gid, col tid*2) c1=(gid, +1) c2=(gid+8, tid*2)
  // c3=(gid+8, +1); cols index the tile's live activation rows. The optional
  // per-row weight (the dispatch route weight on W2) scales the f32
  // accumulator BEFORE the bf16 store — the same association as the oracle
  // reference's post-down multiply.
  const int col0 = tid * 2;
  float rw0 = 1.0f, rw1 = 1.0f;
  if (row_weights != nullptr) {
    if (col0 < live_rows) rw0 = __ldg(row_weights + row_base + col0);
    if (col0 + 1 < live_rows) rw1 = __ldg(row_weights + row_base + col0 + 1);
  }
#pragma unroll
  for (int t = 0; t < kNTiles; ++t) {
    const int n0 = (tile0 + t) * 16;
    if (col0 < live_rows)
      out[(size_t)(row_base + col0) * n + n0 + gid] = __float2bfloat16(rw0 * macc[t][0]);
    if (col0 + 1 < live_rows)
      out[(size_t)(row_base + col0 + 1) * n + n0 + gid] = __float2bfloat16(rw1 * macc[t][1]);
    if (col0 < live_rows)
      out[(size_t)(row_base + col0) * n + n0 + gid + 8] = __float2bfloat16(rw0 * macc[t][2]);
    if (col0 + 1 < live_rows)
      out[(size_t)(row_base + col0 + 1) * n + n0 + gid + 8] = __float2bfloat16(rw1 * macc[t][3]);
  }
}

// silu(gate) * up over the tile rows, bf16 out (no quant, no route weight —
// the weight applies to the f32 W2 output instead, matching the oracle
// reference's post-down association). gate|up layout matches the EP8 masked
// SiLU kernel; the tile list replaces row_map.
__global__ void moe_ep_wo_silu_kernel(
    const __nv_bfloat16* __restrict__ input,       // [expanded, 2*inter] aligned
    const int2* __restrict__ tiles,
    const int* __restrict__ tile_count,
    __nv_bfloat16* __restrict__ output,            // [expanded, inter] aligned
    int inter) {
  if (static_cast<int>(blockIdx.x) >= *tile_count) {
    return;
  }
  const int2 tile = tiles[blockIdx.x];
  const int live_rows = tile.y >> 16;
  if (static_cast<int>(blockIdx.y) >= live_rows) {
    return;
  }
  const int row = tile.x + blockIdx.y;
  const __nv_bfloat16* gate_row = input + (size_t)row * (inter * 2);
  const __nv_bfloat16* up_row = gate_row + inter;
  __nv_bfloat16* out_row = output + (size_t)row * inter;
  for (int col = threadIdx.x; col < inter; col += blockDim.x) {
    const float gate = __bfloat162float(gate_row[col]);
    const float up = __bfloat162float(up_row[col]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    out_row[col] = __float2bfloat16(gate * sigmoid_gate * up);
  }
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

}  // namespace

extern "C" {

CUresult glm52_moe_ep_wo_tiles_cuda(const int* psum_expert, int2* tiles,
                                    int* tile_count, int groups,
                                    int m_capacity, int masked_cap,
                                    int max_tiles, cudaStream_t stream) {
  if (psum_expert == nullptr || tiles == nullptr || tile_count == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (groups <= 0 || groups > 0xffff || m_capacity <= 0 || masked_cap <= 0 ||
      max_tiles <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  moe_ep_wo_tiles_kernel<<<1, WARP_SIZE, 0, stream>>>(
      psum_expert, tiles, tile_count, groups, m_capacity, masked_cap,
      max_tiles);
  return consume_last_cuda_error();
}

CUresult glm52_moe_ep_wo_masked_mma_cuda(
    const __nv_bfloat16* activation, const unsigned char* weight,
    const float* weight_scale, const int2* tiles, const int* tile_count,
    const float* row_weights, __nv_bfloat16* out, int n, int k, int max_tiles,
    cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || weight_scale == nullptr ||
      tiles == nullptr || tile_count == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // n covers whole 128-row blocks (scale rows); k covers whole 128-col scale
  // groups and the k64 super-chunk sweep.
  if (n <= 0 || k <= 0 || n % 128 != 0 || k % 128 != 0 || max_tiles <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int n_blocks = n / (16 * kNTiles * kMmaWarps);
  moe_ep_wo_masked_mma_kernel<<<dim3(n_blocks, max_tiles),
                                kMmaWarps * WARP_SIZE, 0, stream>>>(
      activation, weight, weight_scale, tiles, tile_count, row_weights, out,
      n, k);
  return consume_last_cuda_error();
}

CUresult glm52_moe_ep_wo_silu_cuda(const __nv_bfloat16* input,
                                   const int2* tiles, const int* tile_count,
                                   __nv_bfloat16* output, int inter,
                                   int max_tiles, cudaStream_t stream) {
  if (input == nullptr || tiles == nullptr || tile_count == nullptr ||
      output == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (inter <= 0 || max_tiles <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  moe_ep_wo_silu_kernel<<<dim3(max_tiles, kTileRows), kSiluThreads, 0,
                          stream>>>(input, tiles, tile_count, output, inter);
  return consume_last_cuda_error();
}

}  // extern "C"
