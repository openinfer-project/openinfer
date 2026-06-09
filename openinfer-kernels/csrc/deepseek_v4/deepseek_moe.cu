#include "deepseek_common.cuh"

#include <mutex>

namespace {

constexpr int kMaxMoeScratchDevices = 16;

struct DeepseekMoeScratch {
  float* raw_scores = nullptr;
  size_t raw_score_elems = 0;
  cublasHandle_t handle = nullptr;
  std::mutex mutex;
};

DeepseekMoeScratch g_moe_scratch[kMaxMoeScratchDevices];

cudaError_t deepseek_ensure_f32_scratch(float** ptr, size_t* capacity, size_t required) {
  if (required <= *capacity) {
    return cudaSuccess;
  }
  if (*ptr) {
    cudaError_t err = cudaFree(*ptr);
    if (err != cudaSuccess) {
      return err;
    }
    *ptr = nullptr;
    *capacity = 0;
  }
  cudaError_t err = cudaMalloc(ptr, required * sizeof(float));
  if (err != cudaSuccess) {
    return err;
  }
  *capacity = required;
  return cudaSuccess;
}

cudaError_t deepseek_moe_scratch_for_device(DeepseekMoeScratch** out) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return err;
  }
  if (device < 0 || device >= kMaxMoeScratchDevices) {
    return cudaErrorInvalidDevice;
  }
  *out = &g_moe_scratch[device];
  return cudaSuccess;
}

cudaError_t deepseek_ensure_moe_cublas_handle(DeepseekMoeScratch& scratch) {
  if (scratch.handle != nullptr) {
    return cudaSuccess;
  }
  cublasStatus_t status = cublasCreate(&scratch.handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    scratch.handle = nullptr;
    return cudaErrorUnknown;
  }
  status = cublasSetMathMode(scratch.handle, CUBLAS_PEDANTIC_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(scratch.handle);
    scratch.handle = nullptr;
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

}  // namespace

__global__ void deepseek_hash_gate_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const long long *__restrict__ tid2eid,
    const unsigned int *__restrict__ token_ids,
    float *__restrict__ route_weights,
    int *__restrict__ route_indices,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    float route_scale) {
  int route = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  if (route >= topk || token >= seq_len) return;

  extern __shared__ float scratch[];
  unsigned int token_id = token_ids[token];
  long long expert_i64 = tid2eid[(size_t)token_id * topk + route];
  int expert = (int)expert_i64;

  float partial = 0.0f;
  if (expert >= 0 && expert < n_experts) {
    for (int k = tid; k < hidden_dim; k += blockDim.x) {
      float xv = __bfloat162float(x[token * hidden_dim + k]);
      float wv = __bfloat162float(gate_weight[expert * hidden_dim + k]);
      partial += xv * wv;
    }
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float score = scratch[0];
    float softplus = score > 20.0f ? score : log1pf(expf(score));
    float routed = sqrtf(softplus);
    route_weights[token * topk + route] = routed;
    route_indices[token * topk + route] = expert;
  }
}

__global__ void deepseek_route_normalize_kernel(
    float *__restrict__ route_weights,
    int seq_len,
    int topk,
    float route_scale) {
  int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= seq_len) return;

  float sum = 0.0f;
  for (int route = 0; route < topk; ++route) {
    sum += route_weights[token * topk + route];
  }
  float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
  for (int route = 0; route < topk; ++route) {
    route_weights[token * topk + route] =
        route_weights[token * topk + route] * inv_sum * route_scale;
  }
}

extern "C" {

cudaError_t deepseek_hash_gate_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *gate_weight,
    const long long *tid2eid,
    const unsigned int *token_ids,
    float *route_weights,
    int *route_indices,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    float route_scale,
    cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 score_grid(topk, seq_len);
  size_t shared_bytes = threads * sizeof(float);
  deepseek_hash_gate_kernel<<<score_grid, threads, shared_bytes, stream>>>(
      x, gate_weight, tid2eid, token_ids, route_weights, route_indices,
      seq_len, hidden_dim, n_experts, topk, route_scale);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (seq_len + threads - 1) / threads;
  deepseek_route_normalize_kernel<<<norm_blocks, threads, 0, stream>>>(
      route_weights, seq_len, topk, route_scale);
  return cudaGetLastError();
}

}  // extern "C"

__global__ void deepseek_score_gate_bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ out,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) out[idx] = __bfloat162float(x[idx]);
}

__global__ void deepseek_score_gate_select_kernel(
    const float *__restrict__ raw_scores,
    const float *__restrict__ gate_bias,
    float *__restrict__ original_scores_out,
    float *__restrict__ select_scores_out,
    float *__restrict__ route_weights,
    int *__restrict__ route_indices,
    int seq_len,
    int n_experts,
    int topk,
    float route_scale) {
  int token = blockIdx.x;
  int expert = threadIdx.x;
  if (token >= seq_len) return;

  extern __shared__ float scratch[];
  float *original_scores = scratch;
  float *select_scores = scratch + n_experts;
  float *reduce_scores = select_scores + n_experts;
  int *reduce_indices = reinterpret_cast<int *>(reduce_scores + blockDim.x);

  if (expert < n_experts) {
    float dot = raw_scores[token * n_experts + expert];
    float softplus = dot > 20.0f ? dot : log1pf(expf(dot));
    float score = sqrtf(softplus);
    original_scores[expert] = score;
    select_scores[expert] = score + gate_bias[expert];
    if (original_scores_out != nullptr) original_scores_out[token * n_experts + expert] = score;
    if (select_scores_out != nullptr) {
      select_scores_out[token * n_experts + expert] = score + gate_bias[expert];
    }
  }
  __syncthreads();

  float selected_sum = 0.0f;
  float w0 = 0.0f;
  float w1 = 0.0f;
  float w2 = 0.0f;
  float w3 = 0.0f;
  float w4 = 0.0f;
  float w5 = 0.0f;
  for (int route = 0; route < topk; ++route) {
    reduce_scores[expert] = expert < n_experts ? select_scores[expert] : -3.4028234663852886e38f;
    reduce_indices[expert] = expert < n_experts ? expert : 2147483647;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (expert < stride) {
        float other_score = reduce_scores[expert + stride];
        int other_idx = reduce_indices[expert + stride];
        float self_score = reduce_scores[expert];
        int self_idx = reduce_indices[expert];
        if (other_score > self_score || (other_score == self_score && other_idx < self_idx)) {
          reduce_scores[expert] = other_score;
          reduce_indices[expert] = other_idx;
        }
      }
      __syncthreads();
    }

    if (expert == 0) {
      int best_idx = reduce_indices[0];
      route_indices[token * topk + route] = best_idx;
      float route_weight = best_idx < n_experts ? original_scores[best_idx] : 0.0f;
      route_weights[token * topk + route] = route_weight;
      selected_sum = __fadd_rn(selected_sum, route_weight);
      if (topk == 6) {
        if (route == 0) w0 = route_weight;
        if (route == 1) w1 = route_weight;
        if (route == 2) w2 = route_weight;
        if (route == 3) w3 = route_weight;
        if (route == 4) w4 = route_weight;
        if (route == 5) w5 = route_weight;
      }
      if (best_idx < n_experts) select_scores[best_idx] = -3.4028234663852886e38f;
    }
    __syncthreads();
  }

  if (expert == 0) {
    if (topk == 6) {
      float left = __fadd_rn(__fadd_rn(w0, w4), w2);
      float right = __fadd_rn(__fadd_rn(w1, w5), w3);
      selected_sum = __fadd_rn(left, right);
    }

    for (int route = 0; route < topk; ++route) {
      float normalized =
          selected_sum > 0.0f ? (route_weights[token * topk + route] / selected_sum) : 0.0f;
      route_weights[token * topk + route] = __fmul_rn(normalized, route_scale);
    }
  }
}

extern "C" {

cudaError_t deepseek_score_gate_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *gate_weight,
    const float *gate_bias,
    float *route_weights,
    int *route_indices,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    float route_scale,
    cudaStream_t stream) {
  constexpr int threads = 256;
  DeepseekMoeScratch* scratch_ptr = nullptr;
  cudaError_t cuda_status = deepseek_moe_scratch_for_device(&scratch_ptr);
  if (cuda_status != cudaSuccess) return cuda_status;
  DeepseekMoeScratch& scratch = *scratch_ptr;
  std::lock_guard<std::mutex> lock(scratch.mutex);
  cuda_status = deepseek_ensure_f32_scratch(
      &scratch.raw_scores, &scratch.raw_score_elems, (size_t)seq_len * n_experts);
  if (cuda_status != cudaSuccess) return cuda_status;

  cuda_status = deepseek_ensure_moe_cublas_handle(scratch);
  if (cuda_status != cudaSuccess) return cuda_status;
  cublasStatus_t status = cublasSetStream(scratch.handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasGemmEx(
      scratch.handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n_experts,
      seq_len,
      hidden_dim,
      &alpha,
      gate_weight,
      CUDA_R_16BF,
      hidden_dim,
      x,
      CUDA_R_16BF,
      hidden_dim,
      &beta,
      scratch.raw_scores,
      CUDA_R_32F,
      n_experts,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  size_t shared_bytes = (2 * n_experts + threads) * sizeof(float) + threads * sizeof(int);
  deepseek_score_gate_select_kernel<<<seq_len, threads, shared_bytes, stream>>>(
      scratch.raw_scores, gate_bias, nullptr, nullptr, route_weights, route_indices,
      seq_len, n_experts, topk, route_scale);
  return cudaGetLastError();
}

cudaError_t deepseek_score_gate_debug_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *gate_weight,
    const float *gate_bias,
    float *raw_scores,
    float *original_scores,
    float *select_scores,
    float *route_weights,
    int *route_indices,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    float route_scale,
    cudaStream_t stream) {
  constexpr int threads = 256;
  float *x_f32 = nullptr;
  float *gate_f32 = nullptr;
  cudaError_t cuda_status = cudaMalloc(&x_f32, sizeof(float) * seq_len * hidden_dim);
  if (cuda_status != cudaSuccess) return cuda_status;
  cuda_status = cudaMalloc(&gate_f32, sizeof(float) * n_experts * hidden_dim);
  if (cuda_status != cudaSuccess) {
    cudaFree(x_f32);
    return cuda_status;
  }

  int x_total = seq_len * hidden_dim;
  int x_blocks = (x_total + threads - 1) / threads;
  deepseek_score_gate_bf16_to_f32_kernel<<<x_blocks, threads, 0, stream>>>(x, x_f32, x_total);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cudaFree(gate_f32);
    cudaFree(x_f32);
    return cuda_status;
  }
  int gate_total = n_experts * hidden_dim;
  int gate_blocks = (gate_total + threads - 1) / threads;
  deepseek_score_gate_bf16_to_f32_kernel<<<gate_blocks, threads, 0, stream>>>(
      gate_weight, gate_f32, gate_total);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cudaFree(gate_f32);
    cudaFree(x_f32);
    return cuda_status;
  }

  cublasHandle_t handle = nullptr;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFree(gate_f32);
    cudaFree(x_f32);
    return cudaErrorUnknown;
  }
  status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    cudaFree(gate_f32);
    cudaFree(x_f32);
    return cudaErrorUnknown;
  }
  status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    cudaFree(gate_f32);
    cudaFree(x_f32);
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (seq_len == 1) {
    status = cublasSgemv(
        handle,
        CUBLAS_OP_T,
        hidden_dim,
        n_experts,
        &alpha,
        gate_f32,
        hidden_dim,
        x_f32,
        1,
        &beta,
        raw_scores,
        1);
  } else {
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n_experts,
        seq_len,
        hidden_dim,
        &alpha,
        gate_f32,
        CUDA_R_32F,
        hidden_dim,
        x_f32,
        CUDA_R_32F,
        hidden_dim,
        &beta,
        raw_scores,
        CUDA_R_32F,
        n_experts,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT);
  }
  cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFree(gate_f32);
    cudaFree(x_f32);
    return cudaErrorUnknown;
  }

  size_t shared_bytes = (2 * n_experts + threads) * sizeof(float) + threads * sizeof(int);
  deepseek_score_gate_select_kernel<<<seq_len, threads, shared_bytes, stream>>>(
      raw_scores, gate_bias, original_scores, select_scores, route_weights, route_indices,
      seq_len, n_experts, topk, route_scale);
  cuda_status = cudaGetLastError();
  cudaFree(gate_f32);
  cudaFree(x_f32);
  return cuda_status;
}

}  // extern "C"

__global__ void deepseek_add_f32_bf16_to_bf16_kernel(
    const float *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = __float2bfloat16(a[idx] + __bfloat162float(b[idx]));
}

__global__ void deepseek_moe_clear_mapping_kernel(
    int *__restrict__ pos_to_token,
    int *__restrict__ pos_to_token_topk,
    int *__restrict__ token_topk_to_pos,
    int *__restrict__ expert_indptr,
    int *__restrict__ expert_cursor,
    int *__restrict__ local_count,
    int route_elems,
    int local_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < route_elems) {
    pos_to_token[idx] = -1;
    pos_to_token_topk[idx] = -1;
    token_topk_to_pos[idx] = -1;
  }
  if (idx < local_experts) {
    expert_indptr[idx] = 0;
    expert_cursor[idx] = 0;
  }
  if (idx == local_experts) {
    expert_indptr[local_experts] = 0;
  }
  if (idx == 0) {
    local_count[0] = 0;
  }
}

__global__ void deepseek_moe_local_mapping_kernel(
    const int *__restrict__ route_indices,
    int *__restrict__ pos_to_token,
    int *__restrict__ pos_to_token_topk,
    int *__restrict__ token_topk_to_pos,
    int *__restrict__ expert_indptr,
    int *__restrict__ expert_cursor,
    int seq_len,
    int topk,
    int global_start,
    int local_experts) {
  int route_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int route_elems = seq_len * topk;
  if (route_offset >= route_elems) return;

  int expert = route_indices[route_offset];
  if (expert < global_start || expert >= global_start + local_experts) {
    return;
  }
  int local_expert = expert - global_start;
  int pos = atomicAdd(&expert_cursor[local_expert], 1);
  if (pos >= route_elems) return;
  int token = route_offset / topk;
  pos_to_token[pos] = token;
  pos_to_token_topk[pos] = route_offset;
  token_topk_to_pos[route_offset] = pos;
}

__global__ void deepseek_moe_local_mapping_small_kernel(
    const int *__restrict__ route_indices,
    int *__restrict__ pos_to_token,
    int *__restrict__ pos_to_token_topk,
    int *__restrict__ token_topk_to_pos,
    int *__restrict__ expert_indptr,
    int *__restrict__ expert_cursor,
    int *__restrict__ local_count,
    int seq_len,
    int topk,
    int global_start,
    int local_experts) {
  const int tid = static_cast<int>(threadIdx.x);
  const int route_elems = seq_len * topk;

  for (int idx = tid; idx < route_elems; idx += blockDim.x) {
    pos_to_token[idx] = -1;
    pos_to_token_topk[idx] = -1;
    token_topk_to_pos[idx] = -1;
  }
  for (int idx = tid; idx <= local_experts; idx += blockDim.x) {
    expert_indptr[idx] = 0;
    if (idx < local_experts) {
      expert_cursor[idx] = 0;
    }
  }
  if (tid == 0) {
    local_count[0] = 0;
  }
  __syncthreads();

  for (int route_offset = tid; route_offset < route_elems; route_offset += blockDim.x) {
    int expert = route_indices[route_offset];
    if (expert >= global_start && expert < global_start + local_experts) {
      atomicAdd(&expert_indptr[expert - global_start + 1], 1);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int sum = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
      int count = expert_indptr[expert + 1];
      expert_indptr[expert] = sum;
      expert_cursor[expert] = sum;
      sum += count;
    }
    expert_indptr[local_experts] = sum;
    local_count[0] = sum;
  }
  __syncthreads();

  for (int route_offset = tid; route_offset < route_elems; route_offset += blockDim.x) {
    int expert = route_indices[route_offset];
    if (expert < global_start || expert >= global_start + local_experts) {
      continue;
    }
    int local_expert = expert - global_start;
    int pos = atomicAdd(&expert_cursor[local_expert], 1);
    if (pos >= route_elems) continue;
    int token = route_offset / topk;
    pos_to_token[pos] = token;
    pos_to_token_topk[pos] = route_offset;
    token_topk_to_pos[route_offset] = pos;
  }
}

__global__ void deepseek_moe_count_local_experts_kernel(
    const int *__restrict__ route_indices,
    int *__restrict__ expert_indptr,
    int seq_len,
    int topk,
    int global_start,
    int local_experts) {
  int route_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int route_elems = seq_len * topk;
  if (route_offset >= route_elems) return;
  int expert = route_indices[route_offset];
  if (expert < global_start || expert >= global_start + local_experts) {
    return;
  }
  atomicAdd(&expert_indptr[expert - global_start + 1], 1);
}

__global__ void deepseek_moe_prefix_local_experts_kernel(
    int *__restrict__ expert_indptr,
    int *__restrict__ expert_cursor,
    int *__restrict__ local_count,
    int local_experts) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int sum = 0;
  for (int expert = 0; expert < local_experts; ++expert) {
    int count = expert_indptr[expert + 1];
    expert_indptr[expert] = sum;
    expert_cursor[expert] = sum;
    sum += count;
  }
  expert_indptr[local_experts] = sum;
  local_count[0] = sum;
}

__global__ void deepseek_pplx_padded_expert_indptr_kernel(
    const int *__restrict__ recv_tokens_per_expert,
    int *__restrict__ expert_indptr,
    int local_experts,
    int expert_padding) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int sum = 0;
  for (int expert = 0; expert < local_experts; ++expert) {
    int count = recv_tokens_per_expert[expert];
    if (count < 0) count = 0;
    expert_indptr[expert] = sum;
    int padded = ((count + expert_padding - 1) / expert_padding) * expert_padding;
    sum += padded;
  }
  expert_indptr[local_experts] = sum;
}

__global__ void deepseek_moe_expand_to_fused_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const int *__restrict__ pos_to_token,
    __nv_bfloat16 *__restrict__ expanded,
    int hidden_dim,
    int num_expanded) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_expanded * hidden_dim;
  if (idx >= total) return;
  int pos = idx / hidden_dim;
  int dim = idx - pos * hidden_dim;
  int token = pos_to_token[pos];
  if (token < 0) {
    expanded[idx] = __float2bfloat16(0.0f);
  } else {
    expanded[idx] = x[token * hidden_dim + dim];
  }
}

__global__ void deepseek_moe_reduce_fused_f32_kernel(
    const __nv_bfloat16 *__restrict__ expanded,
    const float *__restrict__ route_weights,
    const int *__restrict__ token_topk_to_pos,
    float *__restrict__ out,
    int seq_len,
    int hidden_dim,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  int dim = idx - token * hidden_dim;

  float acc = 0.0f;
  for (int route = 0; route < topk; ++route) {
    int route_offset = token * topk + route;
    int pos = token_topk_to_pos[route_offset];
    if (pos >= 0) {
      acc += __bfloat162float(expanded[pos * hidden_dim + dim]) * route_weights[route_offset];
    }
  }
  out[idx] = acc;
}

extern "C" {

cudaError_t deepseek_moe_local_mapping_cuda(
    const int *route_indices,
    int *pos_to_token,
    int *pos_to_token_topk,
    int *token_topk_to_pos,
    int *expert_indptr,
    int *expert_cursor,
    int *local_count,
    int seq_len,
    int topk,
    int global_start,
    int local_experts,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int route_elems = seq_len * topk;
  int route_blocks = (route_elems + threads - 1) / threads;
  if (route_elems <= 1024 && local_experts <= 256) {
    deepseek_moe_local_mapping_small_kernel<<<1, threads, 0, stream>>>(
        route_indices, pos_to_token, pos_to_token_topk,
        token_topk_to_pos, expert_indptr, expert_cursor, local_count,
        seq_len, topk, global_start, local_experts);
    return cudaGetLastError();
  }

  int clear_elems = route_elems > local_experts + 1 ? route_elems : local_experts + 1;
  int clear_blocks = (clear_elems + threads - 1) / threads;
  cudaError_t err = cudaSuccess;
  deepseek_moe_clear_mapping_kernel<<<clear_blocks, threads, 0, stream>>>(
      pos_to_token, pos_to_token_topk, token_topk_to_pos,
      expert_indptr, expert_cursor, local_count, route_elems, local_experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  deepseek_moe_count_local_experts_kernel<<<route_blocks, threads, 0, stream>>>(
      route_indices, expert_indptr, seq_len, topk, global_start, local_experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  deepseek_moe_prefix_local_experts_kernel<<<1, 1, 0, stream>>>(
      expert_indptr, expert_cursor, local_count, local_experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  deepseek_moe_local_mapping_kernel<<<route_blocks, threads, 0, stream>>>(
      route_indices, pos_to_token, pos_to_token_topk,
      token_topk_to_pos, expert_indptr, expert_cursor,
      seq_len, topk, global_start, local_experts);
  return cudaGetLastError();
}

cudaError_t deepseek_moe_expand_to_fused_cuda(
    const __nv_bfloat16 *x,
    const int *pos_to_token,
    __nv_bfloat16 *expanded,
    int hidden_dim,
    int num_expanded,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = num_expanded * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_moe_expand_to_fused_kernel<<<blocks, threads, 0, stream>>>(
      x, pos_to_token, expanded, hidden_dim, num_expanded);
  return cudaGetLastError();
}

cudaError_t deepseek_pplx_padded_expert_indptr_cuda(
    const int *recv_tokens_per_expert,
    int *expert_indptr,
    int local_experts,
    int expert_padding,
    cudaStream_t stream) {
  if (recv_tokens_per_expert == nullptr || expert_indptr == nullptr ||
      local_experts <= 0 || expert_padding <= 0) {
    return cudaErrorInvalidValue;
  }
  deepseek_pplx_padded_expert_indptr_kernel<<<1, 1, 0, stream>>>(
      recv_tokens_per_expert, expert_indptr, local_experts, expert_padding);
  return cudaGetLastError();
}

cudaError_t deepseek_moe_reduce_fused_f32_cuda(
    const __nv_bfloat16 *expanded,
    const float *route_weights,
    const int *token_topk_to_pos,
    float *out,
    int seq_len,
    int hidden_dim,
    int topk,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = seq_len * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_moe_reduce_fused_f32_kernel<<<blocks, threads, 0, stream>>>(
      expanded, route_weights, token_topk_to_pos, out, seq_len, hidden_dim, topk);
  return cudaGetLastError();
}

cudaError_t deepseek_add_f32_bf16_to_bf16_cuda(
    const float *a,
    const __nv_bfloat16 *b,
    __nv_bfloat16 *out,
    int n,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  deepseek_add_f32_bf16_to_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
  return cudaGetLastError();
}

}  // extern "C"
