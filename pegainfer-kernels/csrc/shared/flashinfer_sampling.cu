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
