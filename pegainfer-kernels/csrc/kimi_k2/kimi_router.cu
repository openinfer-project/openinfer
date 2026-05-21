#include "../common.cuh"

#include <cublas_v2.h>
#include <cuda.h>
#include <math_constants.h>

extern thread_local cublasHandle_t g_cublas_handle;

namespace {

constexpr int kRouterThreads = 256;
constexpr int kRouterSelectThreads = 512;
constexpr int kKimiExperts = 384;
constexpr int kKimiTopk = 8;

__device__ __forceinline__ bool better_router_choice(float value, int expert, float best_value,
                                                     int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

__global__ void router_scores_kernel(const float *__restrict__ logits,
                                     const float *__restrict__ e_score_correction_bias,
                                     float *__restrict__ scores,
                                     float *__restrict__ choice_scores,
                                     int total_scores,
                                     int n_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_scores) return;

  int expert = idx % n_experts;
  float score = 1.0f / (1.0f + expf(-logits[idx]));
  scores[idx] = score;
  choice_scores[idx] = score + e_score_correction_bias[expert];
}

__global__ void router_topk_normalize_kernel(const float *__restrict__ scores,
                                             float *__restrict__ choice_scores,
                                             float *__restrict__ topk_weight,
                                             int *__restrict__ topk_idx,
                                             int active_tokens,
                                             int n_experts,
                                             int topk,
                                             float route_scale) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= active_tokens) return;

  extern __shared__ char shared[];
  float *reduce_values = reinterpret_cast<float *>(shared);
  int *reduce_indices = reinterpret_cast<int *>(reduce_values + blockDim.x);
  float *selected_scores = reinterpret_cast<float *>(reduce_indices + blockDim.x);

  const int row_base = token * n_experts;
  float selected_sum = 0.0f;
  for (int route = 0; route < topk; ++route) {
    const int expert = tid;
    reduce_values[tid] = expert < n_experts ? choice_scores[row_base + expert] : -CUDART_INF_F;
    reduce_indices[tid] = expert < n_experts ? expert : n_experts;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_value = reduce_values[tid + stride];
        const int other_idx = reduce_indices[tid + stride];
        if (better_router_choice(other_value, other_idx, reduce_values[tid], reduce_indices[tid])) {
          reduce_values[tid] = other_value;
          reduce_indices[tid] = other_idx;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int best_idx = reduce_indices[0];
      const float route_score = best_idx < n_experts ? scores[row_base + best_idx] : 0.0f;
      selected_scores[route] = route_score;
      topk_idx[token * topk + route] = best_idx;
      topk_weight[token * topk + route] = route_score;
      selected_sum += route_score;
      if (best_idx < n_experts) choice_scores[row_base + best_idx] = -CUDART_INF_F;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float scale = selected_sum > 0.0f ? route_scale / selected_sum : 0.0f;
    for (int route = 0; route < topk; ++route) {
      topk_weight[token * topk + route] = selected_scores[route] * scale;
    }
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

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  return map_cuda_error(err);
}

CUresult kimi_router_logits_gemm(
    const __nv_bfloat16 *hidden,
    const __nv_bfloat16 *gate_weight,
    float *logits,
    int padded_tokens,
    int hidden_dim,
    int n_experts,
    cudaStream_t stream) {
  if (g_cublas_handle == nullptr) return CUDA_ERROR_NOT_INITIALIZED;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return CUDA_ERROR_INVALID_HANDLE;
  status = cublasGemmEx(
      g_cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n_experts,
      padded_tokens,
      hidden_dim,
      &alpha,
      gate_weight,
      CUDA_R_16BF,
      hidden_dim,
      hidden,
      CUDA_R_16BF,
      hidden_dim,
      &beta,
      logits,
      CUDA_R_32F,
      n_experts,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

extern "C" {

CUresult kimi_k2_router_noaux_tc_cuda(
    const __nv_bfloat16 *hidden,
    const __nv_bfloat16 *gate_weight,
    const float *e_score_correction_bias,
    float *logits,
    float *scores,
    float *choice_scores,
    float *topk_weight,
    int *topk_idx,
    int active_tokens,
    int padded_tokens,
    int hidden_dim,
    int n_experts,
    int topk,
    float route_scale,
    cudaStream_t stream) {
  (void)stream;
  if (hidden == nullptr || gate_weight == nullptr || e_score_correction_bias == nullptr ||
      logits == nullptr || scores == nullptr || choice_scores == nullptr ||
      topk_weight == nullptr || topk_idx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens <= 0 || padded_tokens <= 0 || active_tokens > padded_tokens ||
      hidden_dim <= 0 || n_experts <= 0 || topk <= 0 || topk > n_experts ||
      n_experts != kKimiExperts || topk != kKimiTopk || !(route_scale > 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult result = kimi_router_logits_gemm(
      hidden, gate_weight, logits, padded_tokens, hidden_dim, n_experts, stream);
  if (result != CUDA_SUCCESS) return result;

  int total_scores = padded_tokens * n_experts;
  int blocks = (total_scores + kRouterThreads - 1) / kRouterThreads;
  router_scores_kernel<<<blocks, kRouterThreads, 0, stream>>>(
      logits, e_score_correction_bias, scores, choice_scores, total_scores, n_experts);
  result = consume_last_cuda_error();
  if (result != CUDA_SUCCESS) return result;

  size_t select_smem =
      static_cast<size_t>(kRouterSelectThreads) * (sizeof(float) + sizeof(int)) +
      static_cast<size_t>(topk) * sizeof(float);
  router_topk_normalize_kernel<<<active_tokens, kRouterSelectThreads, select_smem, stream>>>(
      scores, choice_scores, topk_weight, topk_idx, active_tokens, n_experts, topk, route_scale);
  result = consume_last_cuda_error();
  if (result != CUDA_SUCCESS) return result;

  return CUDA_SUCCESS;
}

}  // extern "C"
