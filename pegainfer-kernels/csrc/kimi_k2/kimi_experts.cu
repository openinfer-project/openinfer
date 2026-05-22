#include "../common.cuh"

#include <cuda.h>
#include <stdint.h>

extern "C" {

namespace {

constexpr int kKimiHiddenDim = 7168;
constexpr int kKimiExpertIntermediateDim = 2048;
constexpr int kKimiLocalExperts = 48;
constexpr int kKimiInt4GroupSize = 32;
__device__ __forceinline__ bool kimi_shape_matches(
    const int32_t* __restrict__ shape,
    int expert,
    int out_dim,
    int in_dim) {
  return shape[expert * 2] == out_dim && shape[expert * 2 + 1] == in_dim;
}

__device__ __forceinline__ int kimi_round_up_to_block(int value, int block_size) {
  return ((value + block_size - 1) / block_size) * block_size;
}

__global__ void kimi_moe_local_route_clear_kernel(
    int* __restrict__ pos_to_token,
    int* __restrict__ token_topk_to_pos,
    uint32_t* __restrict__ expert_indptr,
    uint32_t* __restrict__ expert_cursor,
    uint32_t* __restrict__ local_count,
    int route_elems,
    int local_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < route_elems) {
    pos_to_token[idx] = -1;
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

__global__ void kimi_moe_count_local_route_kernel(
    const int* __restrict__ topk_idx,
    uint32_t* __restrict__ expert_indptr,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts) {
  int route_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int route_elems = active_tokens * topk;
  if (route_offset >= route_elems) return;

  int expert = topk_idx[route_offset];
  if (expert < global_start || expert >= global_start + local_experts) return;
  atomicAdd(&expert_indptr[expert - global_start + 1], 1u);
}

__global__ void kimi_moe_prefix_local_route_kernel(
    uint32_t* __restrict__ expert_indptr,
    uint32_t* __restrict__ expert_cursor,
    uint32_t* __restrict__ local_count,
    int local_experts) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t sum = 0;
  for (int expert = 0; expert < local_experts; ++expert) {
    uint32_t count = expert_indptr[expert + 1];
    expert_indptr[expert] = sum;
    expert_cursor[expert] = sum;
    sum += count;
  }
  expert_indptr[local_experts] = sum;
  local_count[0] = sum;
}

__global__ void kimi_moe_fill_local_route_kernel(
    const int* __restrict__ topk_idx,
    int* __restrict__ pos_to_token,
    int* __restrict__ token_topk_to_pos,
    uint32_t* __restrict__ expert_cursor,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts) {
  int route_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int route_elems = active_tokens * topk;
  if (route_offset >= route_elems) return;

  int expert = topk_idx[route_offset];
  if (expert < global_start || expert >= global_start + local_experts) return;
  int local_expert = expert - global_start;
  uint32_t pos = atomicAdd(&expert_cursor[local_expert], 1u);
  if (pos >= static_cast<uint32_t>(route_elems)) return;
  int token = route_offset / topk;
  pos_to_token[pos] = token;
  token_topk_to_pos[route_offset] = static_cast<int>(pos);
}

__global__ void kimi_moe_local_route_small_kernel(
    const int* __restrict__ topk_idx,
    int* __restrict__ pos_to_token,
    int* __restrict__ token_topk_to_pos,
    uint32_t* __restrict__ expert_indptr,
    uint32_t* __restrict__ expert_cursor,
    uint32_t* __restrict__ local_count,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts) {
  int tid = static_cast<int>(threadIdx.x);
  int route_elems = active_tokens * topk;

  for (int idx = tid; idx < route_elems; idx += blockDim.x) {
    pos_to_token[idx] = -1;
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
    int expert = topk_idx[route_offset];
    if (expert >= global_start && expert < global_start + local_experts) {
      atomicAdd(&expert_indptr[expert - global_start + 1], 1u);
    }
  }
  __syncthreads();

  if (tid == 0) {
    uint32_t sum = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
      uint32_t count = expert_indptr[expert + 1];
      expert_indptr[expert] = sum;
      expert_cursor[expert] = sum;
      sum += count;
    }
    expert_indptr[local_experts] = sum;
    local_count[0] = sum;
  }
  __syncthreads();

  for (int route_offset = tid; route_offset < route_elems; route_offset += blockDim.x) {
    int expert = topk_idx[route_offset];
    if (expert < global_start || expert >= global_start + local_experts) continue;
    int local_expert = expert - global_start;
    uint32_t pos = atomicAdd(&expert_cursor[local_expert], 1u);
    if (pos >= static_cast<uint32_t>(route_elems)) continue;
    int token = route_offset / topk;
    pos_to_token[pos] = token;
    token_topk_to_pos[route_offset] = static_cast<int>(pos);
  }
}

__global__ void kimi_moe_expand_to_expert_major_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ pos_to_token,
    __nv_bfloat16* __restrict__ expert_major_hidden,
    int hidden_dim,
    int routed_capacity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = routed_capacity * hidden_dim;
  if (idx >= total) return;
  int pos = idx / hidden_dim;
  int dim = idx - pos * hidden_dim;
  int token = pos_to_token[pos];
  if (token < 0) {
    expert_major_hidden[idx] = __float2bfloat16(0.0f);
  } else {
    expert_major_hidden[idx] = hidden[token * hidden_dim + dim];
  }
}

__global__ void kimi_moe_reduce_expert_major_f32_kernel(
    const __nv_bfloat16* __restrict__ expert_major_output,
    const float* __restrict__ topk_weight,
    const int* __restrict__ token_topk_to_pos,
    float* __restrict__ out,
    int active_tokens,
    int hidden_dim,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = active_tokens * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  int dim = idx - token * hidden_dim;

  float acc = 0.0f;
  for (int route = 0; route < topk; ++route) {
    int route_offset = token * topk + route;
    int pos = token_topk_to_pos[route_offset];
    if (pos >= 0) {
      acc += __bfloat162float(expert_major_output[pos * hidden_dim + dim]) *
             topk_weight[route_offset];
    }
  }
  out[idx] = acc;
}

__global__ void kimi_add_f32_bf16_to_bf16_kernel(
    const float* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = __float2bfloat16(a[idx] + __bfloat162float(b[idx]));
}

__global__ void kimi_scaled_add_f32_bf16_to_bf16_kernel(
    const float* __restrict__ a,
    float scale,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float scaled = __fmul_rn(a[idx], scale);
  float sum = __fadd_rn(scaled, __bfloat162float(b[idx]));
  out[idx] = __float2bfloat16(sum);
}

__global__ void kimi_moe_marlin_align_small_kernel(
    const int* __restrict__ topk_idx,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    uint32_t* __restrict__ expert_offsets,
    uint32_t* __restrict__ expert_cursor,
    int route_elems,
    int global_start,
    int local_experts,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks) {
  int tid = static_cast<int>(threadIdx.x);
  for (int idx = tid; idx < max_padded_tokens; idx += blockDim.x) {
    sorted_token_ids[idx] = route_elems;
  }
  for (int idx = tid; idx < max_m_blocks; idx += blockDim.x) {
    expert_ids[idx] = -1;
  }
  for (int idx = tid; idx <= local_experts; idx += blockDim.x) {
    expert_offsets[idx] = 0;
    if (idx < local_experts) {
      expert_cursor[idx] = 0;
    }
  }
  if (tid == 0) {
    num_tokens_post_padded[0] = 0;
  }
  __syncthreads();

  for (int route_offset = tid; route_offset < route_elems; route_offset += blockDim.x) {
    int expert = topk_idx[route_offset];
    if (expert >= global_start && expert < global_start + local_experts) {
      atomicAdd(&expert_offsets[expert - global_start + 1], 1u);
    }
  }
  __syncthreads();

  if (tid != 0) return;

  int total = 0;
  for (int expert = 0; expert < local_experts; ++expert) {
    int count = static_cast<int>(expert_offsets[expert + 1]);
    int padded = kimi_round_up_to_block(count, block_size);
    expert_offsets[expert] = static_cast<uint32_t>(total);
    expert_cursor[expert] = 0;
    for (int pos = total; pos < total + padded; pos += block_size) {
      expert_ids[pos / block_size] = expert;
    }
    total += padded;
  }
  expert_offsets[local_experts] = static_cast<uint32_t>(total);
  num_tokens_post_padded[0] = total;

  for (int route_offset = 0; route_offset < route_elems; ++route_offset) {
    int expert = topk_idx[route_offset];
    if (expert < global_start || expert >= global_start + local_experts) continue;
    int local_expert = expert - global_start;
    int pos = static_cast<int>(expert_offsets[local_expert] + expert_cursor[local_expert]);
    expert_cursor[local_expert] += 1;
    if (pos < max_padded_tokens) {
      sorted_token_ids[pos] = route_offset;
    }
  }
}

__global__ void kimi_moe_marlin_align_clear_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    uint32_t* __restrict__ expert_offsets,
    uint32_t* __restrict__ expert_cursor,
    int route_elems,
    int local_experts,
    int max_padded_tokens,
    int max_m_blocks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int pos = idx; pos < max_padded_tokens; pos += stride) {
    sorted_token_ids[pos] = route_elems;
  }
  for (int block = idx; block < max_m_blocks; block += stride) {
    expert_ids[block] = -1;
  }
  for (int expert = idx; expert <= local_experts; expert += stride) {
    expert_offsets[expert] = 0;
    if (expert < local_experts) {
      expert_cursor[expert] = 0;
    }
  }
  if (idx == 0) {
    num_tokens_post_padded[0] = 0;
  }
}

__global__ void kimi_moe_marlin_align_count_kernel(
    const int* __restrict__ topk_idx,
    uint32_t* __restrict__ expert_offsets,
    int route_elems,
    int global_start,
    int local_experts) {
  int route_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (route_offset >= route_elems) return;
  int expert = topk_idx[route_offset];
  if (expert >= global_start && expert < global_start + local_experts) {
    atomicAdd(&expert_offsets[expert - global_start + 1], 1u);
  }
}

__global__ void kimi_moe_marlin_align_prefix_kernel(
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    uint32_t* __restrict__ expert_offsets,
    uint32_t* __restrict__ expert_cursor,
    int local_experts,
    int block_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int total = 0;
  for (int expert = 0; expert < local_experts; ++expert) {
    int count = static_cast<int>(expert_offsets[expert + 1]);
    int padded = kimi_round_up_to_block(count, block_size);
    expert_offsets[expert] = static_cast<uint32_t>(total);
    expert_cursor[expert] = 0;
    for (int pos = total; pos < total + padded; pos += block_size) {
      expert_ids[pos / block_size] = expert;
    }
    total += padded;
  }
  expert_offsets[local_experts] = static_cast<uint32_t>(total);
  num_tokens_post_padded[0] = total;
}

__global__ void kimi_moe_marlin_align_fill_kernel(
    const int* __restrict__ topk_idx,
    int* __restrict__ sorted_token_ids,
    uint32_t* __restrict__ expert_offsets,
    uint32_t* __restrict__ expert_cursor,
    int route_elems,
    int global_start,
    int local_experts,
    int max_padded_tokens) {
  int route_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (route_offset >= route_elems) return;
  int expert = topk_idx[route_offset];
  if (expert < global_start || expert >= global_start + local_experts) return;
  int local_expert = expert - global_start;
  uint32_t rank = atomicAdd(&expert_cursor[local_expert], 1u);
  uint32_t pos = expert_offsets[local_expert] + rank;
  if (pos < static_cast<uint32_t>(max_padded_tokens)) {
    sorted_token_ids[pos] = route_offset;
  }
}

}  // namespace

CUresult kimi_int4_expert_metadata_probe_cuda(
    const int32_t* weight_shape,
    size_t weight_shape_entries,
    int local_experts,
    int in_dim,
    int out_dim,
    int group_size,
    cudaStream_t stream) {
  (void)stream;
  if (weight_shape == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (weight_shape_entries != static_cast<size_t>(local_experts * 2) ||
      local_experts != kKimiLocalExperts || in_dim <= 0 || out_dim <= 0 ||
      group_size != kKimiInt4GroupSize || (in_dim % group_size) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

CUresult kimi_moe_expert_major_route_cuda(
    const int* topk_idx,
    int* pos_to_token,
    int* token_topk_to_pos,
    uint32_t* expert_indptr,
    uint32_t* expert_cursor,
    uint32_t* local_count,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts,
    cudaStream_t stream) {
  if (topk_idx == nullptr || pos_to_token == nullptr || token_topk_to_pos == nullptr ||
      expert_indptr == nullptr || expert_cursor == nullptr || local_count == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens <= 0 || topk != 8 || global_start < 0 ||
      local_experts != kKimiLocalExperts) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int route_elems = active_tokens * topk;
  constexpr int threads = 256;
  int route_blocks = (route_elems + threads - 1) / threads;

  if (route_elems <= 1024) {
    kimi_moe_local_route_small_kernel<<<1, threads, 0, stream>>>(
        topk_idx, pos_to_token, token_topk_to_pos, expert_indptr, expert_cursor, local_count,
        active_tokens, topk, global_start, local_experts);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
  }

  int clear_elems = route_elems > local_experts + 1 ? route_elems : local_experts + 1;
  int clear_blocks = (clear_elems + threads - 1) / threads;
  kimi_moe_local_route_clear_kernel<<<clear_blocks, threads, 0, stream>>>(
      pos_to_token, token_topk_to_pos, expert_indptr, expert_cursor, local_count,
      route_elems, local_experts);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  kimi_moe_count_local_route_kernel<<<route_blocks, threads, 0, stream>>>(
      topk_idx, expert_indptr, active_tokens, topk, global_start, local_experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  kimi_moe_prefix_local_route_kernel<<<1, 1, 0, stream>>>(
      expert_indptr, expert_cursor, local_count, local_experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  kimi_moe_fill_local_route_kernel<<<route_blocks, threads, 0, stream>>>(
      topk_idx, pos_to_token, token_topk_to_pos, expert_cursor, active_tokens, topk,
      global_start, local_experts);
  err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

CUresult kimi_moe_marlin_align_block_size_cuda(
    const int* topk_idx,
    int* sorted_token_ids,
    int* expert_ids,
    int* num_tokens_post_padded,
    uint32_t* expert_offsets,
    uint32_t* expert_cursor,
    int active_tokens,
    int topk,
    int global_start,
    int local_experts,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks,
    cudaStream_t stream) {
  if (topk_idx == nullptr || sorted_token_ids == nullptr || expert_ids == nullptr ||
      num_tokens_post_padded == nullptr || expert_offsets == nullptr ||
      expert_cursor == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens <= 0 || topk != 8 || global_start < 0 ||
      local_experts != kKimiLocalExperts) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!(block_size == 8 || (block_size >= 16 && block_size <= 64 && block_size % 16 == 0))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int route_elems = active_tokens * topk;
  int required_padded = route_elems + local_experts * (block_size - 1);
  int required_blocks = (required_padded + block_size - 1) / block_size;
  if (max_padded_tokens < required_padded || max_m_blocks < required_blocks) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int threads = 256;
  if (route_elems < 1024) {
    kimi_moe_marlin_align_small_kernel<<<1, threads, 0, stream>>>(
        topk_idx, sorted_token_ids, expert_ids, num_tokens_post_padded, expert_offsets,
        expert_cursor, route_elems, global_start, local_experts, block_size, max_padded_tokens,
        max_m_blocks);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
  }

  int clear_elems = max_padded_tokens;
  if (max_m_blocks > clear_elems) clear_elems = max_m_blocks;
  if (local_experts + 1 > clear_elems) clear_elems = local_experts + 1;
  int clear_blocks = (clear_elems + threads - 1) / threads;
  kimi_moe_marlin_align_clear_kernel<<<clear_blocks, threads, 0, stream>>>(
      sorted_token_ids, expert_ids, num_tokens_post_padded, expert_offsets, expert_cursor,
      route_elems, local_experts, max_padded_tokens, max_m_blocks);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  int route_blocks = (route_elems + threads - 1) / threads;
  kimi_moe_marlin_align_count_kernel<<<route_blocks, threads, 0, stream>>>(
      topk_idx, expert_offsets, route_elems, global_start, local_experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  kimi_moe_marlin_align_prefix_kernel<<<1, 1, 0, stream>>>(
      expert_ids, num_tokens_post_padded, expert_offsets, expert_cursor, local_experts,
      block_size);
  err = cudaGetLastError();
  if (err != cudaSuccess) return CUDA_ERROR_LAUNCH_FAILED;

  kimi_moe_marlin_align_fill_kernel<<<route_blocks, threads, 0, stream>>>(
      topk_idx, sorted_token_ids, expert_offsets, expert_cursor, route_elems, global_start,
      local_experts, max_padded_tokens);
  err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

CUresult kimi_moe_expand_to_expert_major_cuda(
    const __nv_bfloat16* hidden,
    const int* pos_to_token,
    __nv_bfloat16* expert_major_hidden,
    int hidden_dim,
    int routed_capacity,
    cudaStream_t stream) {
  if (hidden == nullptr || pos_to_token == nullptr || expert_major_hidden == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (hidden_dim <= 0 || routed_capacity < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (routed_capacity == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int total = hidden_dim * routed_capacity;
  int blocks = (total + threads - 1) / threads;
  kimi_moe_expand_to_expert_major_kernel<<<blocks, threads, 0, stream>>>(
      hidden, pos_to_token, expert_major_hidden, hidden_dim, routed_capacity);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

CUresult kimi_moe_reduce_expert_major_f32_cuda(
    const __nv_bfloat16* expert_major_output,
    const float* topk_weight,
    const int* token_topk_to_pos,
    float* out,
    int active_tokens,
    int hidden_dim,
    int topk,
    cudaStream_t stream) {
  if (expert_major_output == nullptr || topk_weight == nullptr ||
      token_topk_to_pos == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens <= 0 || hidden_dim <= 0 || topk != 8) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr int threads = 256;
  int total = active_tokens * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  kimi_moe_reduce_expert_major_f32_kernel<<<blocks, threads, 0, stream>>>(
      expert_major_output, topk_weight, token_topk_to_pos, out, active_tokens, hidden_dim, topk);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

CUresult kimi_add_f32_bf16_to_bf16_cuda(
    const float* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || out == nullptr || n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kimi_add_f32_bf16_to_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

CUresult kimi_scaled_add_f32_bf16_to_bf16_cuda(
    const float* a,
    float scale,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || out == nullptr || n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kimi_scaled_add_f32_bf16_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
      a, scale, b, out, n);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

}  // extern "C"
