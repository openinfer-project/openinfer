#include "deepseek_common.cuh"

#include <mutex>

namespace {

constexpr int kMaxMoeScratchDevices = 16;

struct DeepseekMoeScratch {
  float* x_f32 = nullptr;
  size_t x_elems = 0;
  float* gate_f32 = nullptr;
  size_t gate_elems = 0;
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

__global__ void deepseek_swiglu_clamp_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n,
    float limit) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    float gate_value = __bfloat162float(gate[idx]);
    float up_value = __bfloat162float(up[idx]);

    if (limit > 0.0f) {
      gate_value = fminf(gate_value, limit);
      up_value = fminf(fmaxf(up_value, -limit), limit);
    }

    float silu_gate = gate_value / (1.0f + expf(-gate_value));
    out[idx] = __float2bfloat16(silu_gate * up_value);
  }
}

extern "C" {

cudaError_t deepseek_swiglu_clamp_cuda(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    __nv_bfloat16 *out,
    int n,
    float limit,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  deepseek_swiglu_clamp_kernel<<<blocks, threads, 0, stream>>>(
      gate, up, out, n, limit);
  return cudaGetLastError();
}

}  // extern "C"

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

  if (expert == 0) {
    float selected_sum = 0.0f;
    for (int route = 0; route < topk; ++route) {
      int best_idx = 0;
      float best_score = -3.4028234663852886e38f;
      for (int candidate = 0; candidate < n_experts; ++candidate) {
        float score = select_scores[candidate];
        if (score > best_score) {
          best_score = score;
          best_idx = candidate;
        }
      }
      route_indices[token * topk + route] = best_idx;
      float route_weight = original_scores[best_idx];
      route_weights[token * topk + route] = route_weight;
      selected_sum = __fadd_rn(selected_sum, route_weight);
      select_scores[best_idx] = -3.4028234663852886e38f;
    }

    if (topk == 6) {
      float w0 = route_weights[token * topk + 0];
      float w1 = route_weights[token * topk + 1];
      float w2 = route_weights[token * topk + 2];
      float w3 = route_weights[token * topk + 3];
      float w4 = route_weights[token * topk + 4];
      float w5 = route_weights[token * topk + 5];
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
      &scratch.x_f32, &scratch.x_elems, (size_t)seq_len * hidden_dim);
  if (cuda_status != cudaSuccess) return cuda_status;
  cuda_status = deepseek_ensure_f32_scratch(
      &scratch.gate_f32, &scratch.gate_elems, (size_t)n_experts * hidden_dim);
  if (cuda_status != cudaSuccess) return cuda_status;
  cuda_status = deepseek_ensure_f32_scratch(
      &scratch.raw_scores, &scratch.raw_score_elems, (size_t)seq_len * n_experts);
  if (cuda_status != cudaSuccess) return cuda_status;

  int x_total = seq_len * hidden_dim;
  int x_blocks = (x_total + threads - 1) / threads;
  deepseek_score_gate_bf16_to_f32_kernel<<<x_blocks, threads, 0, stream>>>(
      x, scratch.x_f32, x_total);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) return cuda_status;
  int gate_total = n_experts * hidden_dim;
  int gate_blocks = (gate_total + threads - 1) / threads;
  deepseek_score_gate_bf16_to_f32_kernel<<<gate_blocks, threads, 0, stream>>>(
      gate_weight, scratch.gate_f32, gate_total);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) return cuda_status;

  cuda_status = deepseek_ensure_moe_cublas_handle(scratch);
  if (cuda_status != cudaSuccess) return cuda_status;
  cublasStatus_t status = cublasSetStream(scratch.handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (seq_len == 1) {
    status = cublasSgemv(
        scratch.handle,
        CUBLAS_OP_T,
        hidden_dim,
        n_experts,
        &alpha,
        scratch.gate_f32,
        hidden_dim,
        scratch.x_f32,
        1,
        &beta,
        scratch.raw_scores,
        1);
  } else {
    status = cublasGemmEx(
        scratch.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n_experts,
        seq_len,
        hidden_dim,
        &alpha,
        scratch.gate_f32,
        CUDA_R_32F,
        hidden_dim,
        scratch.x_f32,
        CUDA_R_32F,
        hidden_dim,
        &beta,
        scratch.raw_scores,
        CUDA_R_32F,
        n_experts,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT);
  }
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  size_t shared_bytes = 2 * n_experts * sizeof(float);
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

  size_t shared_bytes = 2 * n_experts * sizeof(float);
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

__global__ void deepseek_moe_clear_i32_kernel(int* data, int n, int value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = value;
  }
}

__global__ void deepseek_moe_clear_bf16_kernel(__nv_bfloat16* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = __float2bfloat16(0.0f);
  }
}

__global__ void deepseek_moe_local_mapping_kernel(
    const int *__restrict__ route_indices,
    int *__restrict__ pos_to_expert,
    int *__restrict__ pos_to_token,
    int *__restrict__ pos_to_token_topk,
    int *__restrict__ token_topk_to_pos,
    int *__restrict__ expert_start,
    int *__restrict__ expert_end,
    int *__restrict__ num_tokens_per_expert,
    int seq_len,
    int topk,
    int global_start,
    int local_experts) {
  int local_expert = blockIdx.x;
  if (local_expert >= local_experts) return;

  int global_expert = global_start + local_expert;
  int cursor = local_expert * seq_len * topk;
  int count = 0;
  for (int token = 0; token < seq_len; ++token) {
    for (int route = 0; route < topk; ++route) {
      int route_offset = token * topk + route;
      if (route_indices[route_offset] == global_expert) {
        int pos = cursor + count;
        pos_to_expert[pos] = global_expert;
        pos_to_token[pos] = token;
        pos_to_token_topk[pos] = route_offset;
        token_topk_to_pos[route_offset] = pos;
        count += 1;
      }
    }
  }
  expert_start[local_expert] = cursor;
  expert_end[local_expert] = cursor + count;
  num_tokens_per_expert[local_expert] = count;
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

__global__ void deepseek_moe_accumulate_weighted_bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ expert_out,
    const float *__restrict__ route_weights,
    int route,
    float *__restrict__ out,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_dim) return;
  float weight = route_weights[route];
  out[idx] = __fadd_rn(out[idx], __bfloat162float(expert_out[idx]) * weight);
}

extern "C" {

cudaError_t deepseek_moe_local_mapping_cuda(
    const int *route_indices,
    int *pos_to_expert,
    int *pos_to_token,
    int *pos_to_token_topk,
    int *token_topk_to_pos,
    int *expert_start,
    int *expert_end,
    int *num_tokens_per_expert,
    int seq_len,
    int topk,
    int global_start,
    int local_experts,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int num_expanded = seq_len * topk * local_experts;
  int route_elems = seq_len * topk;
  int clear_blocks = (num_expanded + threads - 1) / threads;
  int route_blocks = (route_elems + threads - 1) / threads;
  deepseek_moe_clear_i32_kernel<<<clear_blocks, threads, 0, stream>>>(pos_to_expert, num_expanded, -1);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  deepseek_moe_clear_i32_kernel<<<clear_blocks, threads, 0, stream>>>(pos_to_token, num_expanded, -1);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  deepseek_moe_clear_i32_kernel<<<clear_blocks, threads, 0, stream>>>(pos_to_token_topk, num_expanded, -1);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  deepseek_moe_clear_i32_kernel<<<route_blocks, threads, 0, stream>>>(token_topk_to_pos, route_elems, -1);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  deepseek_moe_clear_i32_kernel<<<1, threads, 0, stream>>>(expert_start, local_experts, 0);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  deepseek_moe_clear_i32_kernel<<<1, threads, 0, stream>>>(expert_end, local_experts, 0);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  deepseek_moe_clear_i32_kernel<<<1, threads, 0, stream>>>(num_tokens_per_expert, local_experts, 0);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  deepseek_moe_local_mapping_kernel<<<local_experts, 1, 0, stream>>>(
      route_indices, pos_to_expert, pos_to_token, pos_to_token_topk,
      token_topk_to_pos, expert_start, expert_end, num_tokens_per_expert,
      seq_len, topk, global_start, local_experts);
  return cudaGetLastError();
}

cudaError_t deepseek_moe_expand_to_fused_cuda(
    const __nv_bfloat16 *x,
    const int *pos_to_token,
    __nv_bfloat16 *expanded,
    int seq_len,
    int hidden_dim,
    int topk,
    int local_experts,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int num_expanded = seq_len * topk * local_experts;
  int total = num_expanded * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_moe_expand_to_fused_kernel<<<blocks, threads, 0, stream>>>(
      x, pos_to_token, expanded, hidden_dim, num_expanded);
  return cudaGetLastError();
}

cudaError_t deepseek_moe_clear_bf16_cuda(
    __nv_bfloat16 *data,
    int n,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  deepseek_moe_clear_bf16_kernel<<<blocks, threads, 0, stream>>>(data, n);
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

cudaError_t deepseek_moe_accumulate_weighted_bf16_to_f32_cuda(
    const __nv_bfloat16 *expert_out,
    const float *route_weights,
    int route,
    float *out,
    int hidden_dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = (hidden_dim + threads - 1) / threads;
  deepseek_moe_accumulate_weighted_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
      expert_out, route_weights, route, out, hidden_dim);
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
