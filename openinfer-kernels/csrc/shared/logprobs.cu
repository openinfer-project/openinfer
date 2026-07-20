// Batched GPU logprobs reduction (#719).
//
// Replaces the per-row CPU path (extract_vec + full-vocab D2H + stream sync +
// three O(V) host passes) with:
//   1. `logprobs_lse_bf16_cuda` — one block per scored row; two-pass online
//      log-sum-exp (f64 partial sums) plus the picked token's logprob.
//   2. `logprobs_topk_bf16_cuda` — vendored FlashInfer FilteredTopK with
//      deterministic smallest-index tie-break plus an index-sort /
//      stable-value-sort chain so output order exactly matches the host
//      reference `token_logprob_from_row` (value desc, index asc).
//   3. `logprobs_gather_rows_bf16_cuda` — row-index gather so sparse row
//      subsets can feed the contiguous FilteredTopK input layout.
//
// D2H per batch is O(rows * (k + 1)) instead of O(rows * V), with a single
// stream sync instead of one per row.

#include "common.cuh"
#include "ffi_guard.cuh"

#include <cstdio>

#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

#define LOGPROBS_BLOCK 256

// ---------------------------------------------------------------------------
// logsumexp + picked-token logprob
// ---------------------------------------------------------------------------

__global__ void logprobs_lse_kernel(const __nv_bfloat16* __restrict__ logits,
                                    const unsigned int* __restrict__ row_indices,
                                    const unsigned int* __restrict__ picked,
                                    int vocab_size, float* __restrict__ out_lse,
                                    float* __restrict__ out_picked_lp) {
  const int scored = blockIdx.x;
  const long long row = row_indices == nullptr ? scored : row_indices[scored];
  const __nv_bfloat16* x = logits + row * (long long)vocab_size;

  // Pass 1: block max.
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < vocab_size; i += LOGPROBS_BLOCK) {
    local_max = fmaxf(local_max, __bfloat162float(x[i]));
  }
  local_max = warp_reduce_max(local_max);

  __shared__ float warp_max[LOGPROBS_BLOCK / WARP_SIZE];
  __shared__ double warp_sum[LOGPROBS_BLOCK / WARP_SIZE];
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  if (lane == 0) {
    warp_max[warp] = local_max;
  }
  __syncthreads();

  float row_max = warp_max[0];
  for (int w = 1; w < LOGPROBS_BLOCK / WARP_SIZE; ++w) {
    row_max = fmaxf(row_max, warp_max[w]);
  }

  // Pass 2: f64 partial sums of exp(x - max), matching the f64 accumulation of
  // the host reference. Row is L2-hot from pass 1.
  double local_sum = 0.0;
  for (int i = threadIdx.x; i < vocab_size; i += LOGPROBS_BLOCK) {
    local_sum += (double)expf(__bfloat162float(x[i]) - row_max);
  }
  // f64 warp reduction via 64-bit shuffle.
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  if (lane == 0) {
    warp_sum[warp] = local_sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double total = 0.0;
    for (int w = 0; w < LOGPROBS_BLOCK / WARP_SIZE; ++w) {
      total += warp_sum[w];
    }
    const float lse = row_max + (float)log(total);
    out_lse[scored] = lse;
    const float picked_val = __bfloat162float(x[picked[scored]]);
    out_picked_lp[scored] = picked_val - lse;
  }
}

extern "C" int logprobs_lse_bf16_cuda(const __nv_bfloat16* logits,
                                      const unsigned int* row_indices,
                                      const unsigned int* picked, int num_rows,
                                      int vocab_size, float* out_lse,
                                      float* out_picked_lp, cudaStream_t stream) {
  OPENINFER_FFI_GUARD_BEGIN
  if (logits == nullptr || picked == nullptr || out_lse == nullptr ||
      out_picked_lp == nullptr) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  if (num_rows <= 0 || vocab_size <= 0) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  logprobs_lse_kernel<<<num_rows, LOGPROBS_BLOCK, 0, stream>>>(
      logits, row_indices, picked, vocab_size, out_lse, out_picked_lp);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "logprobs_lse_bf16_cuda: launch failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  return static_cast<int>(CUDA_SUCCESS);
  OPENINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Row-index gather (sparse subset -> contiguous [num_rows, vocab] bf16)
// ---------------------------------------------------------------------------

__global__ void logprobs_gather_rows_kernel(const uint4* __restrict__ logits,
                                            const unsigned int* __restrict__ row_indices,
                                            uint4* __restrict__ out, int vec4_per_row) {
  const long long row = row_indices[blockIdx.x];
  const uint4* src = logits + row * (long long)vec4_per_row;
  uint4* dst = out + (long long)blockIdx.x * vec4_per_row;
  for (int i = threadIdx.x; i < vec4_per_row; i += LOGPROBS_BLOCK) {
    dst[i] = src[i];
  }
}

extern "C" int logprobs_gather_rows_bf16_cuda(const __nv_bfloat16* logits,
                                              const unsigned int* row_indices,
                                              __nv_bfloat16* out, int num_rows,
                                              int vocab_size, cudaStream_t stream) {
  OPENINFER_FFI_GUARD_BEGIN
  if (logits == nullptr || row_indices == nullptr || out == nullptr) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  if (num_rows <= 0 || vocab_size <= 0 || vocab_size % 8 != 0) {
    // % 8: rows are copied as 16B uint4 chunks (8 bf16). Vocab sizes in-tree
    // are all multiples of 8; anything else keeps the caller on the CPU path.
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  logprobs_gather_rows_kernel<<<num_rows, LOGPROBS_BLOCK, 0, stream>>>(
      reinterpret_cast<const uint4*>(logits), row_indices,
      reinterpret_cast<uint4*>(out), vocab_size / 8);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "logprobs_gather_rows_bf16_cuda: launch failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  return static_cast<int>(CUDA_SUCCESS);
  OPENINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Deterministic top-k (FilteredTopK, smallest-index tie-break)
// ---------------------------------------------------------------------------

extern "C" int logprobs_topk_bf16_cuda(const __nv_bfloat16* logits, int num_rows,
                                       int vocab_size, int top_k,
                                       __nv_bfloat16* out_values, int* out_indices,
                                       cudaStream_t stream) {
  OPENINFER_FFI_GUARD_BEGIN
  if (logits == nullptr || out_values == nullptr || out_indices == nullptr) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  if (num_rows <= 0 || vocab_size <= 0 || top_k <= 0) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  // FilteredTopK needs ~128KB dynamic smem (Hopper+). Report unsupported so
  // the caller can fall back to the host path instead of crashing.
  if (!flashinfer::sampling::CanImplementFilteredTopK()) {
    return static_cast<int>(CUDA_ERROR_NOT_SUPPORTED);
  }

  auto* input = const_cast<__nv_bfloat16*>(logits);
  cudaError_t err = flashinfer::sampling::FilteredTopK<__nv_bfloat16, int>(
      input, out_indices, out_values, nullptr, static_cast<uint32_t>(num_rows),
      static_cast<uint32_t>(top_k), static_cast<uint32_t>(vocab_size),
      /*deterministic=*/true, flashinfer::sampling::TopKTieBreak::Small, stream,
      /*dsa_graph_safe=*/true);
  if (err != cudaSuccess) {
    fprintf(stderr, "logprobs_topk_bf16_cuda: FilteredTopK failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  // FilteredTopK emits >pivot winners in atomicAdd race order; only selection
  // (with smallest-index tie-break) is deterministic. Restore a canonical
  // (value desc, index asc) order to match the CPU reference exactly:
  // stable value-descending sort preceded by an index-ascending sort.
  err = flashinfer::sampling::LaunchSortTopKByIndex<
      flashinfer::sampling::FilteredTopKMode::Plain, __nv_bfloat16, int>(
      out_indices, out_values, nullptr, 0, nullptr, nullptr,
      static_cast<uint32_t>(num_rows), static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(vocab_size), stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "logprobs_topk_bf16_cuda: LaunchSortTopKByIndex failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  err = flashinfer::sampling::StableSortTopKByValue<__nv_bfloat16, int>(
      out_indices, out_values, static_cast<uint32_t>(num_rows),
      static_cast<uint32_t>(top_k), static_cast<uint32_t>(vocab_size), stream);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "logprobs_topk_bf16_cuda: StableSortTopKByValue failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  return static_cast<int>(CUDA_SUCCESS);
  OPENINFER_FFI_GUARD_END(-1)
}
