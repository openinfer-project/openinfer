#include "common.cuh"
#include <cuda.h>

// ============================================================================
// Element-wise add: out = a + b (bf16, computed in f32)
// ============================================================================

__global__ void add_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    float va = __bfloat162float(a[idx]);
    float vb = __bfloat162float(b[idx]);
    out[idx] = __float2bfloat16(va + vb);
  }
}

// ============================================================================
// Type conversion helpers for deterministic decode collectives.
// ============================================================================

__global__ void bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ input,
    float *__restrict__ output,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    output[idx] = __bfloat162float(input[idx]);
  }
}

__global__ void f32_to_bf16_kernel(
    const float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

__global__ void scale_f32_kernel(float *__restrict__ values, float scale, int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    values[idx] *= scale;
  }
}

__global__ void repeat_f32_rows_for_reduce_scatter_kernel(
    const float *__restrict__ local,
    float *__restrict__ repeated,
    int local_elems,
    int world_size) {
  int total = local_elems * world_size;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    repeated[idx] = local[idx % local_elems];
  }
}

// ============================================================================
// SiLU-mul from separate gate/up buffers: out = silu(gate) * up
// Matches Triton silu_mul_kernel rounding: silu computed in f32,
// cast to bf16, then multiplied with up in bf16→f32.
// ============================================================================

__global__ void silu_mul_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += gridDim.x * blockDim.x) {
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    // Match Triton rounding: silu result cast to bf16 before multiply
    out[idx] = __float2bfloat16(__bfloat162float(__float2bfloat16(silu_g)) * u);
  }
}

// ============================================================================
// Embedding lookup: out = embed[token_id, :]
// Reads token_id from token_id[0] (CUDA Graph safe).
// ============================================================================

__global__ void embedding_decode_kernel(
    const __nv_bfloat16 *__restrict__ embed,
    const uint32_t *__restrict__ token_id,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size) {
  uint32_t token_idx = __ldg(&token_id[0]);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < hidden_size;
       idx += gridDim.x * blockDim.x) {
    out[idx] = embed[(size_t)token_idx * hidden_size + idx];
  }
}

// ============================================================================
// Batched embedding lookup: out[:, i] = embed[token_ids[i], :]
// Column-major output: [hidden_size, seq_len].
// ============================================================================

__global__ void embedding_batched_kernel(
    const __nv_bfloat16 *__restrict__ embed,
    const uint32_t *__restrict__ token_ids,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size, int seq_len) {
  int total = hidden_size * seq_len;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token_offset = idx / hidden_size;
    int dim_offset = idx % hidden_size;
    uint32_t token_id = token_ids[token_offset];
    out[idx] = embed[(size_t)token_id * hidden_size + dim_offset];
  }
}

// ============================================================================
// Tensor-parallel vocab-sharded embedding lookup.
//
// Each rank owns [vocab_start, vocab_start + part_vocab_size). Tokens outside
// the local shard write zeros. An all-reduce over ranks recovers the full
// embedding result, matching the official ParallelEmbedding implementation.
// Output layout remains [seq_len, hidden_size].
// ============================================================================

__global__ void embedding_batched_vocab_shard_kernel(
    const __nv_bfloat16 *__restrict__ embed,
    const uint32_t *__restrict__ token_ids,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size, int seq_len, uint32_t vocab_start,
    uint32_t part_vocab_size) {
  int total = hidden_size * seq_len;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int token_offset = idx / hidden_size;
    int dim_offset = idx % hidden_size;
    uint32_t token_id = token_ids[token_offset];
    if (token_id >= vocab_start && token_id < vocab_start + part_vocab_size) {
      uint32_t local_token_id = token_id - vocab_start;
      out[idx] = embed[(size_t)local_token_id * hidden_size + dim_offset];
    } else {
      out[idx] = __float2bfloat16(0.0f);
    }
  }
}

extern "C" {

CUresult add_cuda(
    const __nv_bfloat16 *a, const __nv_bfloat16 *b,
    __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  add_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
  return (CUresult)cudaGetLastError();
}

CUresult bf16_to_f32_cuda(
    const __nv_bfloat16 *input, float *output, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  bf16_to_f32_kernel<<<grid, block, 0, stream>>>(input, output, n);
  return (CUresult)cudaGetLastError();
}

CUresult f32_to_bf16_cuda(
    const float *input, __nv_bfloat16 *output, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  f32_to_bf16_kernel<<<grid, block, 0, stream>>>(input, output, n);
  return (CUresult)cudaGetLastError();
}

CUresult scale_f32_cuda(float *values, float scale, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  scale_f32_kernel<<<grid, block, 0, stream>>>(values, scale, n);
  return (CUresult)cudaGetLastError();
}

CUresult repeat_f32_for_reduce_scatter_cuda(
    const float *local, float *repeated, int local_elems, int world_size,
    cudaStream_t stream) {
  int total = local_elems * world_size;
  int block = 256;
  int grid = (total + block - 1) / block;
  repeat_f32_rows_for_reduce_scatter_kernel<<<grid, block, 0, stream>>>(
      local, repeated, local_elems, world_size);
  return (CUresult)cudaGetLastError();
}

CUresult silu_mul_triton_aot_cuda(
    const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
    __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  silu_mul_kernel<<<grid, block, 0, stream>>>(gate, up, out, n);
  return (CUresult)cudaGetLastError();
}

CUresult embedding_decode_cuda(
    const __nv_bfloat16 *embed, const uint32_t *token_id,
    __nv_bfloat16 *out, int hidden_size, cudaStream_t stream) {
  int block = 256;
  int grid = (hidden_size + block - 1) / block;
  embedding_decode_kernel<<<grid, block, 0, stream>>>(embed, token_id, out, hidden_size);
  return (CUresult)cudaGetLastError();
}

CUresult embedding_batched_cuda(
    const __nv_bfloat16 *embed, const uint32_t *token_ids,
    __nv_bfloat16 *out, int hidden_size, int seq_len, cudaStream_t stream) {
  int total = hidden_size * seq_len;
  int block = 256;
  int grid = (total + block - 1) / block;
  embedding_batched_kernel<<<grid, block, 0, stream>>>(embed, token_ids, out, hidden_size, seq_len);
  return (CUresult)cudaGetLastError();
}

CUresult embedding_batched_vocab_shard_cuda(
    const __nv_bfloat16 *embed, const uint32_t *token_ids,
    __nv_bfloat16 *out, int hidden_size, int seq_len,
    uint32_t vocab_start, uint32_t part_vocab_size, cudaStream_t stream) {
  int total = hidden_size * seq_len;
  int block = 256;
  int grid = (total + block - 1) / block;
  embedding_batched_vocab_shard_kernel<<<grid, block, 0, stream>>>(
      embed, token_ids, out, hidden_size, seq_len, vocab_start, part_vocab_size);
  return (CUresult)cudaGetLastError();
}

} // extern "C"
