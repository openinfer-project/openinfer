// GLM5.2 DSA indexer cache + top-k slot conversion kernels.
//
// Cache insert: hand-written, cherry-picked from feat/glm52-dp8-ep8
// (commit 7e4200a). Memory-bound elementwise: fp8 per-128-group quant +
// scatter write into DeepGEMM block-split paged layout.
//
// local_topk_to_slots: hand-written, ported from TokenSpeed
// Triton `_local_topk_to_global_slots_kernel` (dsa_sparse_layout.py).
// int32 index-remap: page = block_table[t, off//bs]; slot = page*bs + off%bs.

#include "../common.cuh"
#include "glm52_min_gemv.cuh"

#include <cfloat>
#include <cuda.h>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kQuantBlockSize = 128;
constexpr int kScaleBytesPerToken = 4;
constexpr float kFp8ScaleDivisor = 448.0f;
constexpr float kFp8ScaleEps = 1.0e-4f;

__device__ __forceinline__ unsigned char quantize_e4m3(float value,
                                                       float scale) {
  float q = fminf(fmaxf(value / scale, -448.0f), 448.0f);
  return __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

__global__ void indexer_k_quant_and_cache_kernel(
    const __nv_bfloat16* __restrict__ k,
    unsigned char* __restrict__ indexer_cache,
    const int64_t* __restrict__ slot_mapping, int tokens, int cache_block_size,
    int64_t cache_block_stride_bytes) {
  constexpr int kVecSize = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx =
      (blockIdx.y * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
       threadIdx.x) *
      kVecSize;
  if (token_idx >= tokens || head_dim_idx >= kHeadDim) return;

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  float2 packed = reinterpret_cast<const float2*>(k)[(token_idx * kHeadDim +
                                                      head_dim_idx) /
                                                     kVecSize];
  __nv_bfloat16* values = reinterpret_cast<__nv_bfloat16*>(&packed);
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    amax = fmaxf(amax, fabsf(__bfloat162float(values[i])));
  }

  for (int mask = 16; mask > 0; mask /= 2) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
  }

  float scale = fmaxf(amax, kFp8ScaleEps) / kFp8ScaleDivisor;

  // vLLM cache_kernels.cu::indexer_k_quant_and_cache_kernel stores a block as:
  // [block_size * 128 fp8 values][block_size * 4 f32-scale bytes].
  const int64_t value_offset =
      block_idx * cache_block_stride_bytes + block_offset * kHeadDim +
      head_dim_idx;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    indexer_cache[value_offset + i] =
        quantize_e4m3(__bfloat162float(values[i]), scale);
  }

  if (threadIdx.x == 0) {
    const int64_t scale_offset =
        block_idx * cache_block_stride_bytes + cache_block_size * kHeadDim +
        (block_offset * kHeadDim + head_dim_idx) * kScaleBytesPerToken /
            kQuantBlockSize;
    reinterpret_cast<float*>(indexer_cache)[scale_offset / sizeof(float)] =
        scale;
  }
}

template <int kRowsPerBlock>
__global__ void indexer_gather_cache_kernel(
    const unsigned char* __restrict__ cache,
    unsigned char* __restrict__ gathered_k,
    float* __restrict__ gathered_scales,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_seq_lens,
    int requests, int tokens, int block_table_stride, int cache_block_size,
    int64_t cache_block_stride_bytes) {
  constexpr int kVecBytes = 16;
  const int token = blockIdx.x * blockDim.y + threadIdx.y;
  const int dim = (blockIdx.y * blockDim.x + threadIdx.x) * kVecBytes;
  __shared__ int request_for_row[kRowsPerBlock];
  if (threadIdx.x == 0) request_for_row[threadIdx.y] = -1;
  __syncthreads();
  for (int base = 0; base < requests; base += blockDim.x) {
    const int request = base + threadIdx.x;
    if (request < requests && token >= cu_seq_lens[request] &&
        token < cu_seq_lens[request + 1]) {
      request_for_row[threadIdx.y] = request;
    }
  }
  __syncthreads();

  const int request = request_for_row[threadIdx.y];
  if (token >= tokens || dim >= kHeadDim || request < 0) return;
  const int local = token - cu_seq_lens[request];
  const int page =
      block_table[request * block_table_stride + local / cache_block_size];
  const int in_page = local % cache_block_size;
  const int64_t src = static_cast<int64_t>(page) * cache_block_stride_bytes +
                      in_page * kHeadDim + dim;
  const int64_t dst = static_cast<int64_t>(token) * kHeadDim + dim;
  reinterpret_cast<float4*>(gathered_k)[dst / kVecBytes] =
      reinterpret_cast<const float4*>(cache)[src / kVecBytes];
  if (threadIdx.x == 0) {
    const int64_t scale =
        static_cast<int64_t>(page) * cache_block_stride_bytes +
        cache_block_size * kHeadDim + in_page * kScaleBytesPerToken;
    gathered_scales[token] =
        *reinterpret_cast<const float*>(cache + scale);
  }
}

// Convert local top-k offsets (within a sequence's KV cache) to global KV
// slot indices using the block table. Ported from TokenSpeed Triton
// `_local_topk_to_global_slots_kernel`. One block per token.
__global__ void local_topk_to_global_slots_kernel(
    int* __restrict__ global_slots, int* __restrict__ topk_lens,
    const int* __restrict__ local_topk_offsets,
    int local_topk_stride, const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    const int* __restrict__ row_table_ids, int block_table_stride,
    int block_table_cols, int block_size, int topk) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const int seq_len = seq_lens[token_idx];
  const int table_row = row_table_ids == nullptr ? token_idx
                                                  : row_table_ids[token_idx];

  int count = 0;
  for (int start = 0; start < topk; start += blockDim.x) {
    const int offset = start + tid;
    if (offset < topk) {
      const int local_idx = local_topk_offsets[token_idx * local_topk_stride + offset];
      bool valid = (local_idx >= 0) && (local_idx < seq_len);
      const int block_idx = local_idx / block_size;
      const int block_offset = local_idx % block_size;
      valid = valid && (block_idx >= 0) && (block_idx < block_table_cols);
      const int page = valid
                           ? block_table[table_row * block_table_stride + block_idx]
                           : 0;
      const int slot = page * block_size + block_offset;
      global_slots[token_idx * local_topk_stride + offset] =
          valid ? slot : -1;
      if (valid) {
        count += 1;
      }
    }
  }

  // Warp-reduce count, then block-reduce via shared memory.
  for (int mask = 16; mask > 0; mask /= 2) {
    count += __shfl_xor_sync(0xffffffff, count, mask);
  }

  __shared__ int warp_counts[32];
  const int warp = tid / 32;
  const int lane = tid % 32;
  if (lane == 0) {
    warp_counts[warp] = count;
  }
  __syncthreads();

  const int num_warps = blockDim.x / 32;
  int total = 0;
  for (int w = 0; w < num_warps; ++w) {
    total += warp_counts[w];
  }

  if (tid == 0) {
    topk_lens[token_idx] = total;
  }
}

// Fold the per-head weights_proj output with the per-head q quant scale and
// the two attention scale constants. Multiplication order matches the
// retired host-side fold bit-for-bit (left-to-right f32, no FMA — there is
// no add to contract into).
__global__ void indexer_weights_fold_kernel(
    const __nv_bfloat16* __restrict__ weights,  // [heads]
    const float* __restrict__ q_scale,          // [heads]
    float softmax_scale, float n_heads_scale,
    float* __restrict__ out,                    // [heads]
    int heads) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h < heads) {
    out[h] = __bfloat162float(weights[h]) * q_scale[h] * softmax_scale *
             n_heads_scale;
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

bool valid_cache_layout(int head_dim, int quant_block_size,
                        int cache_block_size,
                        int64_t cache_block_stride_bytes) {
  return head_dim == kHeadDim && quant_block_size == kQuantBlockSize &&
         cache_block_size > 0 &&
         cache_block_stride_bytes >=
             static_cast<int64_t>(cache_block_size) *
                 (kHeadDim + kScaleBytesPerToken);
}

}  // namespace

extern "C" {

CUresult glm52_indexer_k_quant_and_cache_cuda(
    const __nv_bfloat16* k, unsigned char* indexer_cache,
    const int64_t* slot_mapping, int tokens, int head_dim, int quant_block_size,
    int cache_block_size, int64_t cache_block_stride_bytes, cudaStream_t stream) {
  if (k == nullptr || indexer_cache == nullptr || slot_mapping == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (tokens <= 0 ||
      !valid_cache_layout(head_dim, quant_block_size, cache_block_size,
                          cache_block_stride_bytes)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int kVecSize = 4;
  dim3 grid(tokens,
            (kHeadDim + kQuantBlockSize * kVecSize - 1) /
                (kQuantBlockSize * kVecSize));
  dim3 block(32, kVecSize);
  indexer_k_quant_and_cache_kernel<<<grid, block, 0, stream>>>(
      k, indexer_cache, slot_mapping, tokens, cache_block_size,
      cache_block_stride_bytes);
  return consume_last_cuda_error();
}

CUresult glm52_indexer_gather_cache_cuda(
    const unsigned char* indexer_cache, unsigned char* gathered_k,
    float* gathered_scales, const int* block_table, const int* cu_seq_lens,
    int requests, int tokens, int block_table_stride, int cache_block_size,
    int64_t cache_block_stride_bytes, cudaStream_t stream) {
  if (indexer_cache == nullptr || gathered_k == nullptr ||
      gathered_scales == nullptr || block_table == nullptr ||
      cu_seq_lens == nullptr || requests <= 0 || tokens <= 0 ||
      block_table_stride <= 0 ||
      !valid_cache_layout(kHeadDim, kQuantBlockSize, cache_block_size,
                          cache_block_stride_bytes)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr int kRowsPerBlock = 8;
  constexpr int kThreadsX = 8;
  dim3 grid((tokens + kRowsPerBlock - 1) / kRowsPerBlock,
            (kHeadDim + kThreadsX * 16 - 1) / (kThreadsX * 16));
  dim3 block(kThreadsX, kRowsPerBlock);
  indexer_gather_cache_kernel<kRowsPerBlock><<<grid, block, 0, stream>>>(
      indexer_cache, gathered_k, gathered_scales, block_table, cu_seq_lens,
      requests, tokens, block_table_stride, cache_block_size,
      cache_block_stride_bytes);
  return consume_last_cuda_error();
}

// weights_proj [32, 6144] x hidden [tokens, 6144] -> bf16 [tokens, 32];
// min-latency GEMV replaces the cublas splitK plan (4 graph nodes/call).
CUresult glm52_indexer_weights_proj_cuda(const __nv_bfloat16* hidden,
                                         const __nv_bfloat16* weights_proj,
                                         __nv_bfloat16* out, int tokens,
                                         int heads, int hidden_dim,
                                         cudaStream_t stream) {
  if (hidden == nullptr || weights_proj == nullptr || out == nullptr ||
      heads != 32 || hidden_dim != glm52_min_gemv::kHidden) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return map_cuda_error(glm52_min_gemv::launch_tokens<32, __nv_bfloat16>(
      out, hidden, weights_proj, tokens, stream));
}

CUresult glm52_indexer_local_topk_to_slots_cuda(
    int* global_slots, int* topk_lens, const int* local_topk_offsets,
    int local_topk_stride, const int* seq_lens, const int* block_table,
    const int* row_table_ids, int block_table_stride, int block_table_cols,
    int block_size, int topk, int num_tokens, cudaStream_t stream) {
  if (global_slots == nullptr || topk_lens == nullptr ||
      local_topk_offsets == nullptr || seq_lens == nullptr ||
      block_table == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens <= 0 || topk <= 0 || block_size <= 0 ||
      block_table_cols <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int kBlockThreads = 256;
  dim3 grid(num_tokens);
  dim3 block(kBlockThreads);
  local_topk_to_global_slots_kernel<<<grid, block, 0, stream>>>(
      global_slots, topk_lens, local_topk_offsets, local_topk_stride,
      seq_lens, block_table, row_table_ids, block_table_stride, block_table_cols,
      block_size, topk);
  return consume_last_cuda_error();
}

CUresult glm52_indexer_weights_fold_cuda(const __nv_bfloat16* weights,
                                          const float* q_scale,
                                          float softmax_scale,
                                          float n_heads_scale, float* out,
                                          int heads, cudaStream_t stream) {
  if (weights == nullptr || q_scale == nullptr || out == nullptr ||
      heads <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr int kThreads = 256;
  indexer_weights_fold_kernel<<<(heads + kThreads - 1) / kThreads, kThreads, 0,
                                stream>>>(
      weights, q_scale, softmax_scale, n_heads_scale, out, heads);
  return consume_last_cuda_error();
}

}  // extern "C"
