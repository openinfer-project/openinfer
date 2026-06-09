#include "common.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

#include <flashinfer/sampling.cuh>

namespace {

constexpr int SOFTMAX_BLOCK = 256;
constexpr int SOFTMAX_NUM_WARPS = SOFTMAX_BLOCK / WARP_SIZE;

__global__ void logits_to_probs_kernel(const __nv_bfloat16* __restrict__ logits,
                                       float* __restrict__ probs, int vocab_size,
                                       float inv_temperature) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  float local_max = -INFINITY;
  for (int i = tid; i < vocab_size; i += SOFTMAX_BLOCK) {
    float v = __bfloat162float(logits[i]) * inv_temperature;
    probs[i] = v;
    local_max = fmaxf(local_max, v);
  }

  local_max = warp_reduce_max(local_max);
  __shared__ float warp_vals[SOFTMAX_NUM_WARPS];
  if (lane_id == 0) {
    warp_vals[warp_id] = local_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float v = (lane_id < SOFTMAX_NUM_WARPS) ? warp_vals[lane_id] : -INFINITY;
    v = warp_reduce_max(v);
    if (lane_id == 0) {
      warp_vals[0] = v;
    }
  }
  __syncthreads();
  float global_max = warp_vals[0];

  float local_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += SOFTMAX_BLOCK) {
    float v = expf(probs[i] - global_max);
    probs[i] = v;
    local_sum += v;
  }

  local_sum = warp_reduce_sum(local_sum);
  if (lane_id == 0) {
    warp_vals[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float v = (lane_id < SOFTMAX_NUM_WARPS) ? warp_vals[lane_id] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane_id == 0) {
      warp_vals[0] = v;
    }
  }
  __syncthreads();

  float inv_sum = 1.0f / warp_vals[0];
  for (int i = tid; i < vocab_size; i += SOFTMAX_BLOCK) {
    probs[i] *= inv_sum;
  }
}

inline cudaError_t flashinfer_sample_from_probs(float* probs, uint8_t* valid_scratch, int* output,
                                                int vocab_size, int top_k, float top_p,
                                                uint64_t seed, cudaStream_t stream) {
  bool* valid = reinterpret_cast<bool*>(valid_scratch);
  constexpr bool deterministic = false;
  constexpr uint32_t batch_size = 1;
  constexpr uint64_t offset = 0;

  if (top_k > 0 && top_p < 1.0f) {
    return flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
        probs, nullptr, nullptr, output, valid, nullptr, batch_size, top_k, top_p, vocab_size,
        deterministic, nullptr, seed, nullptr, offset, stream);
  }
  if (top_k > 0) {
    return flashinfer::sampling::TopKSamplingFromProb<float, int>(
        probs, output, valid, nullptr, nullptr, batch_size, top_k, vocab_size, deterministic,
        nullptr, seed, nullptr, offset, stream);
  }
  if (top_p < 1.0f) {
    return flashinfer::sampling::TopPSamplingFromProb<float, int>(
        probs, output, valid, nullptr, nullptr, batch_size, top_p, vocab_size, deterministic,
        nullptr, seed, nullptr, offset, stream);
  }
  return flashinfer::sampling::SamplingFromProb<float, int>(
      probs, output, valid, nullptr, batch_size, vocab_size, deterministic, nullptr, seed,
      nullptr, offset, stream);
}

constexpr int GATHER_CAST_BLOCK = 256;
constexpr int GATHER_CAST_ELEMS_PER_THREAD = 8;

// Gather rows of a bf16 logits arena into a compact f32 buffer.
// grid = (ceil(vocab / (BLOCK * ELEMS)), n_rows); row_indices == nullptr means identity.
__global__ void gather_cast_logits_f32_kernel(const __nv_bfloat16* __restrict__ logits,
                                              const int* __restrict__ row_indices,
                                              float* __restrict__ out, int vocab_size) {
  int compact_row = blockIdx.y;
  int src_row = row_indices == nullptr ? compact_row : row_indices[compact_row];
  const __nv_bfloat16* src = logits + static_cast<size_t>(src_row) * vocab_size;
  float* dst = out + static_cast<size_t>(compact_row) * vocab_size;

  int base = (blockIdx.x * GATHER_CAST_BLOCK + threadIdx.x) * GATHER_CAST_ELEMS_PER_THREAD;
#pragma unroll
  for (int j = 0; j < GATHER_CAST_ELEMS_PER_THREAD; ++j) {
    int i = base + j;
    if (i < vocab_size) {
      dst[i] = __bfloat162float(src[i]);
    }
  }
}

}  // namespace

extern "C" void gpu_sample_flashinfer_cuda(const __nv_bfloat16* logits, float* probs_scratch,
                                            uint8_t* valid_scratch, int* output, int vocab_size,
                                            float inv_temperature, int top_k, float top_p,
                                            uint64_t seed, cudaStream_t stream) {
  logits_to_probs_kernel<<<1, SOFTMAX_BLOCK, 0, stream>>>(logits, probs_scratch, vocab_size,
                                                          inv_temperature);
  (void)flashinfer_sample_from_probs(probs_scratch, valid_scratch, output, vocab_size, top_k,
                                     top_p, seed, stream);
}

// Batched temperature/top-k/top-p sampling over a bf16 logits arena.
//
// Three launches for the whole batch: gather+cast (bf16 -> f32), FlashInfer
// OnlineSoftmax (per-row temperature, vocab-splitting strategy for the
// small-batch x large-vocab decode regime), FlashInfer TopKTopPSamplingFromProb
// (per-row top_k/top_p arrays, deterministic CDF scan). One philox seed per
// call; rows decorrelate through the philox subsequence (= row index), so the
// caller must supply a fresh seed per decode step.
//
// top_k_arr entries must be pre-clamped to [1, vocab_size] ("disabled" =
// vocab_size); temperature_arr entries must be > 0 — greedy rows belong on the
// argmax path, not here.
extern "C" int gpu_sample_batch_flashinfer_cuda(
    const __nv_bfloat16* logits, const int* row_indices, float* probs_scratch,
    const float* temperature_arr, const int* top_k_arr, const float* top_p_arr,
    uint8_t* valid_scratch, int* output, void* softmax_workspace,
    size_t softmax_workspace_bytes, int n_rows, int vocab_size, uint64_t seed, uint64_t offset,
    cudaStream_t stream) {
  dim3 gather_grid(
      (vocab_size + GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD - 1) /
          (GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD),
      n_rows);
  gather_cast_logits_f32_kernel<<<gather_grid, GATHER_CAST_BLOCK, 0, stream>>>(
      logits, row_indices, probs_scratch, vocab_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  // In-place: phase 2 of both OnlineSoftmax strategies is an elementwise
  // read-then-write of the same index.
  err = flashinfer::sampling::OnlineSoftmax<float>(
      probs_scratch, probs_scratch, n_rows, vocab_size, const_cast<float*>(temperature_arr),
      /*temperature_val=*/1.0f, softmax_workspace, softmax_workspace_bytes,
      /*enable_pdl=*/false, stream);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  err = flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
      probs_scratch, const_cast<int*>(top_k_arr), const_cast<float*>(top_p_arr), output,
      reinterpret_cast<bool*>(valid_scratch), /*indices=*/nullptr, n_rows, /*top_k_val=*/0,
      /*top_p_val=*/0.0f, vocab_size, /*deterministic=*/true, /*seed_arr=*/nullptr, seed,
      /*offset_arr=*/nullptr, offset, stream);
  return static_cast<int>(err);
}
