#include "deepseek_common.cuh"

#include <mutex>

namespace {

struct SparseAttentionScratch {
  __nv_bfloat16 *q_pad = nullptr;
  __nv_bfloat16 *out_pad = nullptr;
  float *sink_pad = nullptr;
  size_t q_pad_elems = 0;
  size_t out_pad_elems = 0;
  size_t sink_pad_elems = 0;
};

SparseAttentionScratch g_sparse_attention_scratch[16];
std::mutex g_sparse_attention_scratch_mutex[16];

cudaError_t deepseek_ensure_bf16_scratch(
    __nv_bfloat16 **ptr,
    size_t *capacity,
    size_t elems) {
  if (*capacity >= elems) return cudaSuccess;
  if (*ptr) {
    cudaError_t err = cudaFree(*ptr);
    if (err != cudaSuccess) return err;
    *ptr = nullptr;
    *capacity = 0;
  }
  cudaError_t err = cudaMalloc(ptr, elems * sizeof(__nv_bfloat16));
  if (err != cudaSuccess) return err;
  *capacity = elems;
  return cudaSuccess;
}

cudaError_t deepseek_ensure_f32_scratch(
    float **ptr,
    size_t *capacity,
    size_t elems) {
  if (*capacity >= elems) return cudaSuccess;
  if (*ptr) {
    cudaError_t err = cudaFree(*ptr);
    if (err != cudaSuccess) return err;
    *ptr = nullptr;
    *capacity = 0;
  }
  cudaError_t err = cudaMalloc(ptr, elems * sizeof(float));
  if (err != cudaSuccess) return err;
  *capacity = elems;
  return cudaSuccess;
}

}  // namespace

__global__ void deepseek_head_rms_norm_kernel(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int num_heads,
    int head_dim,
    float eps) {
  int token = blockIdx.x;
  int head = blockIdx.y;
  int tid = threadIdx.x;
  if (token >= seq_len || head >= num_heads) return;

  extern __shared__ float scratch[];
  int base = token * num_heads * head_dim + head * head_dim;
  float partial = 0.0f;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float value = __bfloat162float(x[base + dim]);
    partial += round_to_bf16_float(value * value);
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  float mean_square = round_to_bf16_float(scratch[0] / head_dim);
  float inv_rms = round_to_bf16_float(rsqrtf(round_to_bf16_float(mean_square + eps)));
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float value = __bfloat162float(x[base + dim]);
    out[base + dim] = __float2bfloat16(value * inv_rms);
  }
}

__device__ __forceinline__ void deepseek_apply_rope_pair(
    __nv_bfloat16 *x,
    int offset,
    float cos_value,
    float sin_value,
    bool inverse) {
  float x0 = __bfloat162float(x[offset]);
  float x1 = __bfloat162float(x[offset + 1]);
  float c = cos_value;
  float s = sin_value;
  if (inverse) s = -s;
  float out0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
  float out1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
  x[offset] = __float2bfloat16(out0);
  x[offset + 1] = __float2bfloat16(out1);
}

__global__ void deepseek_apply_rope_q_kv_kernel(
    __nv_bfloat16 *__restrict__ q,
    __nv_bfloat16 *__restrict__ kv,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int total_q_pairs = seq_len * local_heads * (rotary_dim / 2);
  int nope_dim = head_dim - rotary_dim;

  if (pair < total_q_pairs) {
    int rotary_pair = pair % (rotary_dim / 2);
    int tmp = pair / (rotary_dim / 2);
    int head = tmp % local_heads;
    int token = tmp / local_heads;
    int pos = start_pos + token;
    int q_offset = token * local_heads * head_dim + head * head_dim + nope_dim + 2 * rotary_pair;
    deepseek_apply_rope_pair(
        q, q_offset, cos_cache[pos * (rotary_dim / 2) + rotary_pair],
        sin_cache[pos * (rotary_dim / 2) + rotary_pair], false);
  }

  int kv_pair = pair - total_q_pairs;
  int total_kv_pairs = seq_len * (rotary_dim / 2);
  if (kv_pair >= 0 && kv_pair < total_kv_pairs) {
    int rotary_pair = kv_pair % (rotary_dim / 2);
    int token = kv_pair / (rotary_dim / 2);
    int pos = start_pos + token;
    int kv_offset = token * head_dim + nope_dim + 2 * rotary_pair;
    deepseek_apply_rope_pair(
        kv, kv_offset, cos_cache[pos * (rotary_dim / 2) + rotary_pair],
        sin_cache[pos * (rotary_dim / 2) + rotary_pair], false);
  }
}

__global__ void deepseek_fill_rope_cache_kernel(
    const float *__restrict__ inv_freq,
    float *__restrict__ cos_cache,
    float *__restrict__ sin_cache,
    int max_seq_len,
    int pairs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = max_seq_len * pairs;
  if (idx >= total) return;
  int pair = idx % pairs;
  int pos = idx / pairs;
  float angle = static_cast<float>(pos) * inv_freq[pair];
  sincosf(angle, &sin_cache[idx], &cos_cache[idx]);
}

__global__ void deepseek_pad_q_h8_to_h16_kernel(
    const __nv_bfloat16 *__restrict__ q,
    __nv_bfloat16 *__restrict__ q_pad,
    int seq_len,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * 16 * head_dim;
  if (idx >= total) return;
  int dim = idx % head_dim;
  int head = (idx / head_dim) % 16;
  int token = idx / (16 * head_dim);
  q_pad[idx] = head < 8 ? q[token * 8 * head_dim + head * head_dim + dim]
                        : __float2bfloat16(0.0f);
}

__global__ void deepseek_copy_o_h16_to_h8_kernel(
    const __nv_bfloat16 *__restrict__ out_pad,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * 8 * head_dim;
  if (idx >= total) return;
  int dim = idx % head_dim;
  int head = (idx / head_dim) % 8;
  int token = idx / (8 * head_dim);
  out[idx] = out_pad[token * 16 * head_dim + head * head_dim + dim];
}

__global__ void deepseek_pad_sink_h8_to_h16_kernel(
    const float *__restrict__ attn_sink,
    float *__restrict__ sink_pad) {
  int idx = threadIdx.x;
  if (idx < 16) sink_pad[idx] = idx < 8 ? attn_sink[idx] : 0.0f;
}

__global__ void deepseek_bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ input,
    float *__restrict__ output,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = __bfloat162float(input[idx]);
  }
}

__global__ void deepseek_f32_to_bf16_kernel(
    const float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

extern "C" int deepseek_tilelang_sparse_attn_local_h16_d512(
    const void* q,
    const void* kv,
    const void* attn_sink,
    const int* topk_idxs,
    void* out,
    int m,
    int n,
    int topk,
    cudaStream_t stream);

extern "C" {

cudaError_t deepseek_head_rms_norm_cuda(
    const __nv_bfloat16 *x,
    __nv_bfloat16 *out,
    int seq_len,
    int num_heads,
    int head_dim,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 512;
  dim3 grid(seq_len, num_heads);
  size_t shared_bytes = threads * sizeof(float);
  deepseek_head_rms_norm_kernel<<<grid, threads, shared_bytes, stream>>>(
      x, out, seq_len, num_heads, head_dim, eps);
  return cudaGetLastError();
}

cudaError_t deepseek_fill_rope_cache_cuda(
    const float *inv_freq,
    float *cos_cache,
    float *sin_cache,
    int max_seq_len,
    int pairs,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = max_seq_len * pairs;
  int blocks = (total + threads - 1) / threads;
  deepseek_fill_rope_cache_kernel<<<blocks, threads, 0, stream>>>(
      inv_freq, cos_cache, sin_cache, max_seq_len, pairs);
  return cudaGetLastError();
}

cudaError_t deepseek_apply_rope_q_kv_cuda(
    __nv_bfloat16 *q,
    __nv_bfloat16 *kv,
    const float *cos_cache,
    const float *sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total_pairs = seq_len * (local_heads + 1) * (rotary_dim / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  deepseek_apply_rope_q_kv_kernel<<<blocks, threads, 0, stream>>>(
      q, kv, cos_cache, sin_cache, seq_len, local_heads, head_dim, rotary_dim, start_pos);
  return cudaGetLastError();
}

cudaError_t deepseek_bf16_to_f32_cuda(
    const __nv_bfloat16 *input,
    float *output,
    int n,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  deepseek_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
  return cudaGetLastError();
}

cudaError_t deepseek_f32_to_bf16_cuda(
    const float *input,
    __nv_bfloat16 *output,
    int n,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  deepseek_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
  return cudaGetLastError();
}

cudaError_t deepseek_indexed_attention_prefill_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv,
    const float *attn_sink,
    const int *topk_idxs,
    __nv_bfloat16 *out,
    int seq_len,
    int kv_len,
    int local_heads,
    int head_dim,
    int topk,
    float softmax_scale,
    cudaStream_t stream) {
  if (seq_len <= 0 || kv_len <= 0 || topk <= 0) return cudaErrorInvalidValue;
  if (local_heads != 8 || head_dim != 512) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (device < 0 || device >= 16) return cudaErrorInvalidDevice;

  std::lock_guard<std::mutex> lock(g_sparse_attention_scratch_mutex[device]);
  SparseAttentionScratch &scratch = g_sparse_attention_scratch[device];
  size_t pad_elems = (size_t)seq_len * 16 * head_dim;
  err = deepseek_ensure_bf16_scratch(&scratch.q_pad, &scratch.q_pad_elems, pad_elems);
  if (err != cudaSuccess) return err;
  err = deepseek_ensure_bf16_scratch(&scratch.out_pad, &scratch.out_pad_elems, pad_elems);
  if (err != cudaSuccess) return err;
  err = deepseek_ensure_f32_scratch(&scratch.sink_pad, &scratch.sink_pad_elems, 16);
  if (err != cudaSuccess) return err;

  {
    constexpr int threads = 256;
    int q_total = seq_len * 16 * head_dim;
    int q_blocks = (q_total + threads - 1) / threads;
    deepseek_pad_q_h8_to_h16_kernel<<<q_blocks, threads, 0, stream>>>(
        q, scratch.q_pad, seq_len, head_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    deepseek_pad_sink_h8_to_h16_kernel<<<1, 16, 0, stream>>>(attn_sink, scratch.sink_pad);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  err = static_cast<cudaError_t>(deepseek_tilelang_sparse_attn_local_h16_d512(
      scratch.q_pad, kv, scratch.sink_pad, topk_idxs, scratch.out_pad, seq_len, kv_len, topk, stream));
  if (err != cudaSuccess) {
    return err;
  }

  {
    constexpr int threads = 256;
    int out_total = seq_len * 8 * head_dim;
    int out_blocks = (out_total + threads - 1) / threads;
    deepseek_copy_o_h16_to_h8_kernel<<<out_blocks, threads, 0, stream>>>(
        scratch.out_pad, out, seq_len, head_dim);
    err = cudaGetLastError();
  }

  return err == cudaSuccess ? cudaGetLastError() : err;
}

}  // extern "C"
