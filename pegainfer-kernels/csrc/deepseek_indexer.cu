#include "deepseek_common.cuh"

#include <cstdint>
#include <mutex>

struct DeepseekFp4QuantScratch {
  __nv_bfloat16 *rotated = nullptr;
  uint8_t *scales = nullptr;
  size_t rotated_elems = 0;
  size_t scale_bytes = 0;
};

static DeepseekFp4QuantScratch g_fp4_quant_scratch[16];
static std::mutex g_fp4_quant_scratch_mutex[16];

static cudaError_t deepseek_ensure_bf16_scratch(
    __nv_bfloat16 **ptr,
    size_t *capacity,
    size_t required) {
  if (*capacity >= required) return cudaSuccess;
  if (*ptr != nullptr) {
    cudaError_t err = cudaFree(*ptr);
    if (err != cudaSuccess) return err;
    *ptr = nullptr;
    *capacity = 0;
  }
  cudaError_t err = cudaMalloc(ptr, required * sizeof(__nv_bfloat16));
  if (err != cudaSuccess) return err;
  *capacity = required;
  return cudaSuccess;
}

static cudaError_t deepseek_ensure_byte_scratch(
    uint8_t **ptr,
    size_t *capacity,
    size_t required) {
  if (*capacity >= required) return cudaSuccess;
  if (*ptr != nullptr) {
    cudaError_t err = cudaFree(*ptr);
    if (err != cudaSuccess) return err;
    *ptr = nullptr;
    *capacity = 0;
  }
  cudaError_t err = cudaMalloc(ptr, required);
  if (err != cudaSuccess) return err;
  *capacity = required;
  return cudaSuccess;
}

extern "C" int deepseek_tilelang_fp4_quant_inplace_n128(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream);

__global__ void deepseek_indexer_scores_prefill_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv,
    const __nv_bfloat16 *__restrict__ weights,
    float *__restrict__ scores,
    int seq_len,
    int local_heads,
    int head_dim,
    int compressed_len,
    float score_scale) {
  int token = blockIdx.x;
  int compressed = blockIdx.y;
  int tid = threadIdx.x;
  if (token >= seq_len || compressed >= compressed_len) return;

  extern __shared__ float scratch[];
  float acc = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    float partial = 0.0f;
    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
      float qv = __bfloat162float(
          q[token * local_heads * head_dim + head * head_dim + dim]);
      float kvv = __bfloat162float(kv[compressed * head_dim + dim]);
      partial += qv * kvv;
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
      float dot = scratch[0];
      float weight = __bfloat162float(weights[token * local_heads + head]);
      acc += fmaxf(dot, 0.0f) * weight;
    }
    __syncthreads();
  }

  if (tid == 0) {
    scores[token * compressed_len + compressed] = acc * score_scale;
  }
}

__global__ void deepseek_indexer_scores_prefill_serial_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv,
    const __nv_bfloat16 *__restrict__ weights,
    float *__restrict__ scores,
    int seq_len,
    int local_heads,
    int head_dim,
    int compressed_len,
    float score_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * compressed_len;
  if (idx >= total) return;

  int token = idx / compressed_len;
  int compressed = idx - token * compressed_len;
  float acc = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    float dot = 0.0f;
    int q_base = token * local_heads * head_dim + head * head_dim;
    int kv_base = compressed * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
      float qv = __bfloat162float(q[q_base + dim]);
      float kvv = __bfloat162float(kv[kv_base + dim]);
      dot += qv * kvv;
    }
    float weight = __bfloat162float(weights[token * local_heads + head]);
    acc += fmaxf(dot, 0.0f) * weight;
  }

  scores[token * compressed_len + compressed] = acc * score_scale;
}

__global__ void deepseek_indexer_topk_prefill_kernel(
    const float *__restrict__ scores,
    int *__restrict__ topk_idxs,
    int seq_len,
    int compressed_len,
    int topk,
    int ratio,
    int offset) {
  int token = blockIdx.x;
  if (token >= seq_len) return;

  extern __shared__ float scratch[];
  float *select_scores = scratch;
  int valid = (token + 1) / ratio;
  for (int idx = threadIdx.x; idx < compressed_len; idx += blockDim.x) {
    select_scores[idx] =
        idx < valid ? scores[token * compressed_len + idx] : -3.4028234663852886e38f;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int route = 0; route < topk; ++route) {
      int best_idx = -1;
      float best_score = -3.4028234663852886e38f;
      for (int candidate = 0; candidate < compressed_len; ++candidate) {
        float score = select_scores[candidate];
        if (score > best_score) {
          best_score = score;
          best_idx = candidate;
        }
      }
      topk_idxs[token * topk + route] =
          best_idx >= 0 && best_score > -3.0e38f ? best_idx + offset : -1;
      if (best_idx >= 0) {
        select_scores[best_idx] = -3.4028234663852886e38f;
      }
    }
  }
}

__global__ void deepseek_indexer_scores_decode_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv,
    const __nv_bfloat16 *__restrict__ weights,
    float *__restrict__ scores,
    int local_heads,
    int head_dim,
    int compressed_len,
    float score_scale) {
  int compressed = blockIdx.x;
  int tid = threadIdx.x;
  if (compressed >= compressed_len) return;

  extern __shared__ float scratch[];
  float acc = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    float partial = 0.0f;
    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
      float qv = __bfloat162float(q[head * head_dim + dim]);
      float kvv = __bfloat162float(kv[compressed * head_dim + dim]);
      partial += qv * kvv;
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
      float dot = scratch[0];
      float weight = __bfloat162float(weights[head]);
      acc += fmaxf(dot, 0.0f) * weight;
    }
    __syncthreads();
  }

  if (tid == 0) {
    scores[compressed] = acc * score_scale;
  }
}

__global__ void deepseek_indexer_scores_decode_serial_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv,
    const __nv_bfloat16 *__restrict__ weights,
    float *__restrict__ scores,
    int local_heads,
    int head_dim,
    int compressed_len,
    float score_scale) {
  int compressed = blockIdx.x * blockDim.x + threadIdx.x;
  if (compressed >= compressed_len) return;

  float acc = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      float qv = __bfloat162float(q[head * head_dim + dim]);
      float kvv = __bfloat162float(kv[compressed * head_dim + dim]);
      dot += qv * kvv;
    }
    float weight = __bfloat162float(weights[head]);
    acc += fmaxf(dot, 0.0f) * weight;
  }

  scores[compressed] = acc * score_scale;
}

__global__ void deepseek_indexer_topk_decode_kernel(
    const float *__restrict__ scores,
    int *__restrict__ topk_idxs,
    int compressed_len,
    int topk,
    int offset) {
  extern __shared__ float select_scores[];
  for (int idx = threadIdx.x; idx < compressed_len; idx += blockDim.x) {
    select_scores[idx] = scores[idx];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int route = 0; route < topk; ++route) {
      int best_idx = -1;
      float best_score = -3.4028234663852886e38f;
      for (int candidate = 0; candidate < compressed_len; ++candidate) {
        float score = select_scores[candidate];
        if (score > best_score) {
          best_score = score;
          best_idx = candidate;
        }
      }
      topk_idxs[route] =
          best_idx >= 0 && best_score > -3.0e38f ? best_idx + offset : -1;
      if (best_idx >= 0) {
        select_scores[best_idx] = -3.4028234663852886e38f;
      }
    }
  }
}

__global__ void deepseek_concat_topk_indices_kernel(
    const int *__restrict__ a,
    const int *__restrict__ b,
    int *__restrict__ out,
    int seq_len,
    int a_topk,
    int b_topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * (a_topk + b_topk);
  if (idx >= total) return;
  int token = idx / (a_topk + b_topk);
  int route = idx % (a_topk + b_topk);
  if (route < a_topk) {
    out[idx] = a[token * a_topk + route];
  } else {
    out[idx] = b[token * b_topk + route - a_topk];
  }
}

__global__ void deepseek_hadamard_rotate_bf16_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int rows,
    int groups,
    int dim) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * groups;
  if (linear >= total || dim <= 0 || dim > 1024) return;

  int row = linear / groups;
  int group = linear - row * groups;
  int base = row * groups * dim + group * dim;
  float values[1024];
  float scale = rsqrtf((float)dim);

  for (int idx = 0; idx < dim; ++idx) {
    values[idx] = __bfloat162float(x[base + idx]) * scale;
  }

  for (int stride = 1; stride < dim; stride <<= 1) {
    for (int idx = 0; idx < dim; ++idx) {
      if ((idx & stride) == 0) {
        int other = idx | stride;
        float a = values[idx];
        float b = values[other];
        values[idx] = a + b;
        values[other] = a - b;
      }
    }
  }

  for (int idx = 0; idx < dim; ++idx) {
    out[base + idx] = __float2bfloat16(values[idx]);
  }
}

extern "C" {

cudaError_t deepseek_indexer_scores_prefill_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv,
    const __nv_bfloat16 *weights,
    float *scores,
    int seq_len,
    int local_heads,
    int head_dim,
    int compressed_len,
    float score_scale,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = seq_len * compressed_len;
  int blocks = (total + threads - 1) / threads;
  deepseek_indexer_scores_prefill_serial_kernel<<<blocks, threads, 0, stream>>>(
      q, kv, weights, scores, seq_len, local_heads, head_dim, compressed_len, score_scale);
  return cudaGetLastError();
}

cudaError_t deepseek_indexer_topk_prefill_cuda(
    const float *scores,
    int *topk_idxs,
    int seq_len,
    int compressed_len,
    int topk,
    int ratio,
    int offset,
    cudaStream_t stream) {
  constexpr int threads = 256;
  size_t shared_bytes = compressed_len * sizeof(float);
  deepseek_indexer_topk_prefill_kernel<<<seq_len, threads, shared_bytes, stream>>>(
      scores, topk_idxs, seq_len, compressed_len, topk, ratio, offset);
  return cudaGetLastError();
}

cudaError_t deepseek_indexer_scores_decode_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv,
    const __nv_bfloat16 *weights,
    float *scores,
    int local_heads,
    int head_dim,
    int compressed_len,
    float score_scale,
    cudaStream_t stream) {
  if (local_heads <= 0 || head_dim <= 0 || compressed_len <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int blocks = (compressed_len + threads - 1) / threads;
  deepseek_indexer_scores_decode_serial_kernel<<<blocks, threads, 0, stream>>>(
      q, kv, weights, scores, local_heads, head_dim, compressed_len, score_scale);
  return cudaGetLastError();
}

cudaError_t deepseek_indexer_topk_decode_cuda(
    const float *scores,
    int *topk_idxs,
    int compressed_len,
    int topk,
    int offset,
    cudaStream_t stream) {
  if (compressed_len <= 0 || topk <= 0 || topk > compressed_len) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  size_t shared_bytes = compressed_len * sizeof(float);
  deepseek_indexer_topk_decode_kernel<<<1, threads, shared_bytes, stream>>>(
      scores, topk_idxs, compressed_len, topk, offset);
  return cudaGetLastError();
}

cudaError_t deepseek_concat_topk_indices_cuda(
    const int *a,
    const int *b,
    int *out,
    int seq_len,
    int a_topk,
    int b_topk,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = seq_len * (a_topk + b_topk);
  int blocks = (total + threads - 1) / threads;
  deepseek_concat_topk_indices_kernel<<<blocks, threads, 0, stream>>>(
      a, b, out, seq_len, a_topk, b_topk);
  return cudaGetLastError();
}

cudaError_t deepseek_hadamard_fp4_quant_bf16_cuda(
    __nv_bfloat16 *x,
    int rows,
    int groups,
    int dim,
    cudaStream_t stream) {
  if (rows <= 0 || groups <= 0 || dim != 128) return cudaErrorInvalidValue;
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (device < 0 || device >= 16) return cudaErrorInvalidDevice;

  int m = rows * groups;
  std::lock_guard<std::mutex> lock(g_fp4_quant_scratch_mutex[device]);
  DeepseekFp4QuantScratch &scratch = g_fp4_quant_scratch[device];
  err = deepseek_ensure_bf16_scratch(&scratch.rotated, &scratch.rotated_elems, (size_t)m * dim);
  if (err != cudaSuccess) return err;
  err = deepseek_ensure_byte_scratch(&scratch.scales, &scratch.scale_bytes, (size_t)m * (dim / 32));
  if (err != cudaSuccess) return err;

  constexpr int threads = 256;
  int blocks = (m + threads - 1) / threads;
  deepseek_hadamard_rotate_bf16_serial_kernel<<<blocks, threads, 0, stream>>>(
      x, scratch.rotated, rows, groups, dim);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = static_cast<cudaError_t>(deepseek_tilelang_fp4_quant_inplace_n128(
      scratch.rotated, x, scratch.scales, m, stream));
  return err == cudaSuccess ? cudaGetLastError() : err;
}

}  // extern "C"
