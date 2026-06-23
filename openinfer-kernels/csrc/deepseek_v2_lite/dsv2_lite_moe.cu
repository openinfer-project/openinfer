#include "../common.cuh"

#include <cuda.h>
#include <math_constants.h>

namespace {

constexpr int kMaxExperts = 64;
constexpr int kMaxTopk = 8;
constexpr int kRouterThreads = 128;
constexpr int kAccumThreads = 256;

__device__ __forceinline__ bool better_prob_choice(float value, int expert, float best_value,
                                                   int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

__global__ void router_softmax_topk_kernel(
    const __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ gate_weight,
    float *__restrict__ topk_weight,
    int *__restrict__ topk_idx,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= seq_len) return;

  __shared__ float logits[kMaxExperts];
  __shared__ float probs[kMaxExperts];
  __shared__ int selected_idx[kMaxTopk];
  __shared__ float selected_weight[kMaxTopk];

  if (tid < n_experts) {
    float acc = 0.0f;
    const int hidden_base = token * hidden_dim;
    const int weight_base = tid * hidden_dim;
    for (int dim = 0; dim < hidden_dim; ++dim) {
      acc = fmaf(
          __bfloat162float(hidden[hidden_base + dim]),
          __bfloat162float(gate_weight[weight_base + dim]),
          acc);
    }
    logits[tid] = acc;
  }
  __syncthreads();

  if (tid == 0) {
    // Probe-only fixed-topology router: keep the first version simple and
    // deterministic. Performance is not claimed for this diagnostic path.
    float max_score = -CUDART_INF_F;
    for (int expert = 0; expert < n_experts; ++expert) {
      max_score = fmaxf(max_score, logits[expert]);
    }

    float denom = 0.0f;
    for (int expert = 0; expert < n_experts; ++expert) {
      float value = expf(logits[expert] - max_score);
      probs[expert] = value;
      denom += value;
    }
    float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int expert = 0; expert < n_experts; ++expert) {
      probs[expert] *= inv_denom;
    }

    bool selected[kMaxExperts];
    for (int expert = 0; expert < kMaxExperts; ++expert) selected[expert] = false;
    for (int route = 0; route < topk; ++route) {
      int best_expert = n_experts;
      float best_value = -CUDART_INF_F;
      for (int expert = 0; expert < n_experts; ++expert) {
        if (selected[expert]) continue;
        float value = probs[expert];
        if (better_prob_choice(value, expert, best_value, best_expert)) {
          best_value = value;
          best_expert = expert;
        }
      }
      selected[best_expert] = true;
      selected_idx[route] = best_expert;
      selected_weight[route] = best_expert < n_experts ? probs[best_expert] : 0.0f;
    }

    for (int i = 1; i < topk; ++i) {
      int id = selected_idx[i];
      float weight = selected_weight[i];
      int j = i - 1;
      while (j >= 0 && selected_idx[j] > id) {
        selected_idx[j + 1] = selected_idx[j];
        selected_weight[j + 1] = selected_weight[j];
        --j;
      }
      selected_idx[j + 1] = id;
      selected_weight[j + 1] = weight;
    }

    int out_base = token * topk;
    for (int route = 0; route < topk; ++route) {
      topk_idx[out_base + route] = selected_idx[route];
      topk_weight[out_base + route] = selected_weight[route];
    }
  }
}

__global__ void accumulate_fixed_expert_kernel(
    const __nv_bfloat16 *__restrict__ expert_output,
    const float *__restrict__ topk_weight,
    const int *__restrict__ topk_idx,
    float *__restrict__ accum,
    int global_expert,
    int seq_len,
    int hidden_dim,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hidden_dim;
  if (idx >= total) return;

  int token = idx / hidden_dim;
  int route_base = token * topk;
  float weight = 0.0f;
  for (int route = 0; route < topk; ++route) {
    if (topk_idx[route_base + route] == global_expert) {
      weight += topk_weight[route_base + route];
    }
  }
  if (weight != 0.0f) {
    accum[idx] = fmaf(__bfloat162float(expert_output[idx]), weight, accum[idx]);
  }
}

CUresult map_cuda_error(cudaError_t err) {
  switch (err) {
    case cudaSuccess:
      return CUDA_SUCCESS;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevicePointer:
      return CUDA_ERROR_INVALID_VALUE;
    case cudaErrorInvalidDevice:
      return CUDA_ERROR_INVALID_DEVICE;
    case cudaErrorInvalidResourceHandle:
      return CUDA_ERROR_INVALID_HANDLE;
    case cudaErrorMemoryAllocation:
      return CUDA_ERROR_OUT_OF_MEMORY;
    case cudaErrorNotSupported:
      return CUDA_ERROR_NOT_SUPPORTED;
    case cudaErrorIllegalAddress:
      return CUDA_ERROR_ILLEGAL_ADDRESS;
    case cudaErrorLaunchOutOfResources:
      return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    case cudaErrorLaunchTimeout:
      return CUDA_ERROR_LAUNCH_TIMEOUT;
    case cudaErrorLaunchFailure:
      return CUDA_ERROR_LAUNCH_FAILED;
    case cudaErrorAssert:
      return CUDA_ERROR_ASSERT;
    case cudaErrorIllegalInstruction:
      return CUDA_ERROR_ILLEGAL_INSTRUCTION;
    case cudaErrorMisalignedAddress:
      return CUDA_ERROR_MISALIGNED_ADDRESS;
    case cudaErrorInvalidAddressSpace:
      return CUDA_ERROR_INVALID_ADDRESS_SPACE;
    case cudaErrorInvalidPc:
      return CUDA_ERROR_INVALID_PC;
    default:
      return CUDA_ERROR_UNKNOWN;
  }
}

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  return map_cuda_error(err);
}

}  // namespace

extern "C" {

CUresult dsv2_lite_router_softmax_topk_cuda(
    const __nv_bfloat16 *hidden,
    const __nv_bfloat16 *gate_weight,
    float *topk_weight,
    int *topk_idx,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    cudaStream_t stream) {
  if (hidden == nullptr || gate_weight == nullptr || topk_weight == nullptr ||
      topk_idx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (seq_len <= 0 || hidden_dim <= 0 || n_experts <= 0 || n_experts > kMaxExperts ||
      topk <= 0 || topk > kMaxTopk || topk > n_experts) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cudaGetLastError();
  router_softmax_topk_kernel<<<seq_len, kRouterThreads, 0, stream>>>(
      hidden, gate_weight, topk_weight, topk_idx, seq_len, hidden_dim, n_experts, topk);
  return consume_last_cuda_error();
}

CUresult dsv2_lite_accumulate_fixed_expert_cuda(
    const __nv_bfloat16 *expert_output,
    const float *topk_weight,
    const int *topk_idx,
    float *accum,
    int global_expert,
    int seq_len,
    int hidden_dim,
    int topk,
    cudaStream_t stream) {
  if (expert_output == nullptr || topk_weight == nullptr || topk_idx == nullptr ||
      accum == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (global_expert < 0 || seq_len <= 0 || hidden_dim <= 0 || topk <= 0 ||
      topk > kMaxTopk) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = seq_len * hidden_dim;
  int blocks = (total + kAccumThreads - 1) / kAccumThreads;
  cudaGetLastError();
  accumulate_fixed_expert_kernel<<<blocks, kAccumThreads, 0, stream>>>(
      expert_output, topk_weight, topk_idx, accum, global_expert, seq_len, hidden_dim, topk);
  return consume_last_cuda_error();
}

}  // extern "C"
