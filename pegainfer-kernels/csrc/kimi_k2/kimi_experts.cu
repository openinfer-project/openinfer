#include "../common.cuh"

#include <cuda.h>
#include <stdint.h>

extern "C" {

namespace {

constexpr int kKimiLocalExperts = 48;
__device__ __forceinline__ int kimi_round_up_to_block(int value, int block_size) {
  return ((value + block_size - 1) / block_size) * block_size;
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

__global__ void kimi_residual_add_scaled_f32_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ projected,
    const float* __restrict__ routed_f32,
    float scale,
    __nv_bfloat16* __restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  __nv_bfloat16 rounded = __float2bfloat16(
      __bfloat162float(hidden[idx]) + __bfloat162float(projected[idx]));
  float scaled = __fmul_rn(routed_f32[idx], scale);
  float sum = __fadd_rn(scaled, __bfloat162float(rounded));
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

CUresult kimi_residual_add_scaled_f32_cuda(
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* projected,
    const float* routed_f32,
    float scale,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream) {
  if (hidden == nullptr || projected == nullptr || routed_f32 == nullptr ||
      out == nullptr || n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kimi_residual_add_scaled_f32_kernel<<<blocks, threads, 0, stream>>>(
      hidden, projected, routed_f32, scale, out, n);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void kimi_residual_add_scaled_bf16_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ routed,
    float scale,
    __nv_bfloat16* __restrict__ out,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  __nv_bfloat16 rounded = __float2bfloat16(
      __bfloat162float(hidden[idx]) + __bfloat162float(projected[idx]));
  float scaled = __fmul_rn(__bfloat162float(routed[idx]), scale);
  float sum = __fadd_rn(scaled, __bfloat162float(rounded));
  out[idx] = __float2bfloat16(sum);
}

// BF16-routed sibling of kimi_residual_add_scaled_f32: the DeepEP combine
// reduces expert outputs to bf16, so the routed contribution arrives one
// bf16 rounding earlier than the old f32 PPLX combine did.
CUresult kimi_residual_add_scaled_bf16_cuda(
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* projected,
    const __nv_bfloat16* routed,
    float scale,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream) {
  if (hidden == nullptr || projected == nullptr || routed == nullptr ||
      out == nullptr || n < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n == 0) return CUDA_SUCCESS;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kimi_residual_add_scaled_bf16_kernel<<<blocks, threads, 0, stream>>>(
      hidden, projected, routed, scale, out, n);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

// Marlin routing metadata from a DeepEP post-epilogue expert prefix sum.
//
// DeepEP's dispatch writes psum_expert as the exclusive prefix sum of the
// per-local-expert ALIGNED counts (num_local_experts + 1 entries), and the
// copy epilogue's per-slot atomicAdd advances entry i from aligned_start_i
// to aligned_start_i + count_i. Entry [num_local_experts] is untouched and
// holds the total aligned expanded row count. Recover, for expert i:
//   start_i = round_up(psum[i-1], alignment)   (psum[-1] == 0)
//   count_i = psum[i] - start_i
// The expanded recv buffer is already expert-major and aligned, so
// sorted_token_ids is identity over real rows and sentinel over pad rows.
__global__ void kimi_deepep_build_marlin_routing_kernel(
    const int32_t* __restrict__ psum_expert,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_post_padded,
    int num_local_experts,
    int alignment,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks) {
  int tid = threadIdx.x;
  for (int block = tid; block < max_m_blocks; block += blockDim.x) {
    expert_ids[block] = 0;
  }
  __syncthreads();

  int sentinel = max_padded_tokens;
  if (tid < num_local_experts) {
    int start = 0;
    if (tid > 0) {
      int prev = psum_expert[tid - 1];
      start = ((prev + alignment - 1) / alignment) * alignment;
    }
    int count = psum_expert[tid] - start;
    if (count < 0) count = 0;
    int padded = ((count + alignment - 1) / alignment) * alignment;
    for (int j = 0; j < padded; j++) {
      int idx = start + j;
      if (idx < max_padded_tokens)
        sorted_token_ids[idx] = (j < count) ? idx : sentinel;
    }
    int block_start = start / block_size;
    int block_end = (start + padded) / block_size;
    for (int b = block_start; b < block_end && b < max_m_blocks; b++)
      expert_ids[b] = tid;
  }

  __syncthreads();
  if (tid == 0) {
    int total = psum_expert[num_local_experts];
    for (int j = total; j < max_padded_tokens; j++)
      sorted_token_ids[j] = sentinel;
    num_tokens_post_padded[0] = total;
  }
}

CUresult kimi_deepep_build_marlin_routing_on_stream(
    const int32_t* psum_expert,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* num_tokens_post_padded,
    int num_local_experts,
    int alignment,
    int block_size,
    int max_padded_tokens,
    int max_m_blocks,
    cudaStream_t stream) {
  kimi_deepep_build_marlin_routing_kernel<<<1, 64, 0, stream>>>(
      psum_expert,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      num_local_experts,
      alignment,
      block_size,
      max_padded_tokens,
      max_m_blocks);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

}  // extern "C"
