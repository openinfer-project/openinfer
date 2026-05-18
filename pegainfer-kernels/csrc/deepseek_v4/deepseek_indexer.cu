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

extern "C" cudaError_t deepseek_cutedsl_indexer_dots_bf16_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv,
    float *dots,
    int rows,
    int compressed_len,
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

__global__ void deepseek_indexer_scores_epilogue_kernel(
    const float *__restrict__ dots,
    const __nv_bfloat16 *__restrict__ weights,
    float *__restrict__ scores,
    int seq_len,
    int local_heads,
    int compressed_len,
    float score_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * compressed_len;
  if (idx >= total) return;

  int token = idx / compressed_len;
  int compressed = idx - token * compressed_len;
  float acc = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    int dot_idx = (token * local_heads + head) * compressed_len + compressed;
    float dot = dots[dot_idx];
    float weight = __bfloat162float(weights[token * local_heads + head]);
    acc += fmaxf(dot, 0.0f) * weight;
  }
  scores[token * compressed_len + compressed] = acc * score_scale;
}

__global__ void deepseek_indexer_scores_decode_epilogue_kernel(
    const float *__restrict__ dots,
    const __nv_bfloat16 *__restrict__ weights,
    float *__restrict__ scores,
    int local_heads,
    int compressed_len,
    float score_scale) {
  int compressed = blockIdx.x * blockDim.x + threadIdx.x;
  if (compressed >= compressed_len) return;

  float acc = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    int dot_idx = head * compressed_len + compressed;
    float dot = dots[dot_idx];
    float weight = __bfloat162float(weights[head]);
    acc += fmaxf(dot, 0.0f) * weight;
  }
  scores[compressed] = acc * score_scale;
}

// One-block-per-token warp-shuffle top-K selector. Each lane keeps a
// register-resident sorted (score desc, candidate-index asc) list of size
// `DEEPSEEK_INDEXER_TOPK_MAX` entries (unused slots stay at the -INF
// sentinel). A single streaming pass over `scores` does sorted-insertion
// (strict `>`); a `topk`-round emit-merge then drains the winning lane each
// round via warp shuffles. No shared memory, no __syncthreads.
//
// All per-lane array indices are compile-time constants so the lists stay
// register-resident on sm_120. The runtime `topk` only bounds the emit
// loop; the streaming and bubble-up are always over the full size-32
// list. For `topk` < 32 this does strictly more insertion work than
// minimally required, but the final top-`topk` emit is bit-exact since
// top-K ⊂ top-32 under the same total order.
//
// Tie-break preserved bit-exact against the previous 256-thread shared-mem
// kernel: score desc, candidate-index asc (lower compressed index wins on
// equal score). Within a single lane the streamed `cand` is monotonically
// increasing so per-lane ties resolve naturally; the cross-lane comparator
// reasserts the (score desc, idx asc) order during the warp reduction.
constexpr int DEEPSEEK_INDEXER_TOPK_MAX = 32;

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

  constexpr float kNegInf = -3.4028234663852886e38f;
  constexpr float kNegInfThreshold = -3.0e38f;
  constexpr int K = DEEPSEEK_INDEXER_TOPK_MAX;

  float ls[K];
  int li[K];
  #pragma unroll
  for (int i = 0; i < K; ++i) {
    ls[i] = kNegInf;
    li[i] = -1;
  }

  int valid = (token + 1) / ratio;
  const float *row = scores + (size_t)token * compressed_len;

  // Streaming pass: bubble-up insertion (strict `>`). The candidate at
  // slot `K-1` is the smallest currently kept; only candidates that
  // strictly beat it are admitted.
  for (int cand = threadIdx.x; cand < compressed_len; cand += 32) {
    float s = (cand < valid) ? row[cand] : kNegInf;
    if (s <= ls[K - 1]) continue;
    ls[K - 1] = s;
    li[K - 1] = cand;
    #pragma unroll
    for (int i = K - 1; i > 0; --i) {
      if (ls[i] > ls[i - 1]) {
        float ts = ls[i];
        int ti = li[i];
        ls[i] = ls[i - 1];
        li[i] = li[i - 1];
        ls[i - 1] = ts;
        li[i - 1] = ti;
      }
    }
  }

  // Emit-merge: each round, every lane emits ls[0]; warp shuffles compute
  // the global winner; the winning lane shifts its list left and inserts
  // a sentinel at the tail.
  for (int route = 0; route < topk; ++route) {
    float best_s = ls[0];
    int best_i = li[0];
    int best_l = threadIdx.x;

    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      float o_s = __shfl_xor_sync(0xffffffff, best_s, stride);
      int o_i = __shfl_xor_sync(0xffffffff, best_i, stride);
      int o_l = __shfl_xor_sync(0xffffffff, best_l, stride);
      bool take = (o_s > best_s) ||
                  (o_s == best_s && o_i >= 0 && (best_i < 0 || o_i < best_i));
      if (take) {
        best_s = o_s;
        best_i = o_i;
        best_l = o_l;
      }
    }

    if (threadIdx.x == 0) {
      topk_idxs[(size_t)token * topk + route] =
          (best_i >= 0 && best_s > kNegInfThreshold) ? best_i + offset : -1;
    }

    if (threadIdx.x == best_l) {
      #pragma unroll
      for (int i = 0; i < K - 1; ++i) {
        ls[i] = ls[i + 1];
        li[i] = li[i + 1];
      }
      ls[K - 1] = kNegInf;
      li[K - 1] = -1;
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

__global__ void deepseek_indexer_scores_decode_batch_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv,
    const __nv_bfloat16 *__restrict__ weights,
    const int *__restrict__ compressed_len,
    const int *__restrict__ cache_base,
    float *__restrict__ scores,
    int batch,
    int local_heads,
    int head_dim,
    int max_compressed_len,
    float score_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * max_compressed_len;
  if (idx >= total) return;
  int token = idx / max_compressed_len;
  int compressed = idx - token * max_compressed_len;
  int valid = compressed_len[token];
  if (compressed >= valid) {
    scores[idx] = -3.4028234663852886e38f;
    return;
  }

  float acc = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    float dot = 0.0f;
    int q_base = token * local_heads * head_dim + head * head_dim;
    int kv_base = (cache_base[token] + compressed) * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
      float qv = __bfloat162float(q[q_base + dim]);
      float kvv = __bfloat162float(kv[kv_base + dim]);
      dot += qv * kvv;
    }
    float weight = __bfloat162float(weights[token * local_heads + head]);
    acc += fmaxf(dot, 0.0f) * weight;
  }

  scores[idx] = acc * score_scale;
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

__global__ void deepseek_ratio4_decode_topk_indices_batch_kernel(
    const float *__restrict__ scores,
    const int *__restrict__ start_pos,
    const int *__restrict__ window_base,
    const int *__restrict__ compressed_len,
    const int *__restrict__ compressed_base,
    int *__restrict__ topk_idxs,
    int batch,
    int window_size,
    int max_compressed_len,
    int index_topk) {
  int token = blockIdx.x;
  if (token >= batch) return;
  int tid = threadIdx.x;
  int out_topk = window_size + index_topk;
  int pos = start_pos[token];

  for (int route = tid; route < window_size; route += blockDim.x) {
    int logical = -1;
    if (pos >= 0) {
      if (pos >= window_size - 1) {
        int ring_pos = pos % window_size;
        int first_count = window_size - 1 - ring_pos;
        logical = route < first_count ? ring_pos + 1 + route : route - first_count;
      } else {
        logical = route <= pos ? route : -1;
      }
    }
    topk_idxs[token * out_topk + route] =
        logical >= 0 ? window_base[token] + logical : -1;
  }

  int compressed = compressed_len[token];
  extern __shared__ float select_scores[];
  for (int idx = tid; idx < max_compressed_len; idx += blockDim.x) {
    select_scores[idx] =
        idx < compressed ? scores[token * max_compressed_len + idx]
                         : -3.4028234663852886e38f;
  }
  __syncthreads();

  if (tid == 0) {
    for (int route = 0; route < index_topk; ++route) {
      int best_idx = -1;
      float best_score = -3.4028234663852886e38f;
      for (int candidate = 0; candidate < max_compressed_len; ++candidate) {
        float score = select_scores[candidate];
        if (score > best_score) {
          best_score = score;
          best_idx = candidate;
        }
      }
      topk_idxs[token * out_topk + window_size + route] =
          best_idx >= 0 && best_score > -3.0e38f ? compressed_base[token] + best_idx : -1;
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

__global__ void deepseek_window_topk_indices_kernel(
    int *__restrict__ out,
    int seq_len,
    int window_size,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * topk;
  if (idx >= total) return;
  int token = idx / topk;
  int route = idx - token * topk;
  int key_start = token - (window_size - 1);
  if (key_start < 0) key_start = 0;
  int key = key_start + route;
  out[idx] = key <= token ? key : -1;
}

__global__ void deepseek_window_topk_indices_decode_kernel(
    int *__restrict__ out,
    int start_pos,
    int window_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= window_size) return;
  if (start_pos >= window_size - 1) {
    int pos = start_pos % window_size;
    int first_count = window_size - 1 - pos;
    out[idx] = idx < first_count ? pos + 1 + idx : idx - first_count;
  } else {
    out[idx] = idx <= start_pos ? idx : -1;
  }
}

__global__ void deepseek_window_topk_indices_decode_batch_kernel(
    int *__restrict__ out,
    const int *__restrict__ start_pos,
    const int *__restrict__ cache_base,
    int batch,
    int window_size,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * topk;
  if (idx >= total) return;
  int token = idx / topk;
  int route = idx - token * topk;
  int pos = start_pos[token];
  if (pos < 0) {
    out[idx] = -1;
    return;
  }
  int logical = -1;
  if (pos >= window_size - 1) {
    int ring_pos = pos % window_size;
    int first_count = window_size - 1 - ring_pos;
    logical = route < first_count ? ring_pos + 1 + route : route - first_count;
  } else {
    logical = route <= pos ? route : -1;
  }
  out[idx] = logical >= 0 ? cache_base[token] + logical : -1;
}

__global__ void deepseek_compress_topk_indices_kernel(
    int *__restrict__ out,
    int seq_len,
    int compressed,
    int ratio,
    int offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * compressed;
  if (idx >= total) return;
  int token = idx / compressed;
  int block = idx - token * compressed;
  int valid = (token + 1) / ratio;
  out[idx] = block < valid ? offset + block : -1;
}

__global__ void deepseek_compress_topk_indices_decode_kernel(
    int *__restrict__ out,
    int compressed,
    int offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= compressed) return;
  out[idx] = offset + idx;
}

__global__ void deepseek_compress_topk_indices_decode_batch_kernel(
    int *__restrict__ out,
    const int *__restrict__ compressed_len,
    const int *__restrict__ cache_base,
    int batch,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * topk;
  if (idx >= total) return;
  int token = idx / topk;
  int route = idx - token * topk;
  int compressed = compressed_len[token];
  out[idx] = route < compressed ? cache_base[token] + route : -1;
}

__global__ void deepseek_window_and_compress_topk_indices_kernel(
    int *__restrict__ out,
    int seq_len,
    int window_size,
    int window_topk,
    int compressed,
    int ratio,
    int compress_offset,
    int topk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * topk;
  if (idx >= total) return;
  int token = idx / topk;
  int route = idx - token * topk;
  if (route < window_topk) {
    int key_start = token - (window_size - 1);
    if (key_start < 0) key_start = 0;
    int key = key_start + route;
    out[idx] = key <= token ? key : -1;
  } else {
    int block = route - window_topk;
    int valid = (token + 1) / ratio;
    out[idx] = block < compressed && block < valid ? compress_offset + block : -1;
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
  if (seq_len <= 0 || local_heads <= 0 || head_dim <= 0 || compressed_len <= 0) {
    return cudaErrorInvalidValue;
  }

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
  if (topk <= 0 || topk > DEEPSEEK_INDEXER_TOPK_MAX) {
    return cudaErrorInvalidValue;
  }
  deepseek_indexer_topk_prefill_kernel<<<seq_len, 32, 0, stream>>>(
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
  if (q == nullptr || kv == nullptr || weights == nullptr || scores == nullptr ||
      local_heads <= 0 || head_dim <= 0 || compressed_len <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr int threads = 256;
  int blocks = (compressed_len + threads - 1) / threads;
  deepseek_indexer_scores_decode_serial_kernel<<<blocks, threads, 0, stream>>>(
      q, kv, weights, scores, local_heads, head_dim, compressed_len, score_scale);
  return cudaGetLastError();
}

cudaError_t deepseek_indexer_scores_decode_batch_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv,
    const __nv_bfloat16 *weights,
    const int *compressed_len,
    const int *cache_base,
    float *scores,
    int batch,
    int local_heads,
    int head_dim,
    int max_compressed_len,
    float score_scale,
    cudaStream_t stream) {
  if (q == nullptr || kv == nullptr || weights == nullptr || compressed_len == nullptr ||
      cache_base == nullptr || scores == nullptr || batch <= 0 || local_heads <= 0 ||
      head_dim <= 0 || max_compressed_len <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int total = batch * max_compressed_len;
  int blocks = (total + threads - 1) / threads;
  deepseek_indexer_scores_decode_batch_kernel<<<blocks, threads, 0, stream>>>(
      q, kv, weights, compressed_len, cache_base, scores, batch, local_heads, head_dim,
      max_compressed_len, score_scale);
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

cudaError_t deepseek_ratio4_decode_topk_indices_batch_cuda(
    const float *scores,
    const int *start_pos,
    const int *window_base,
    const int *compressed_len,
    const int *compressed_base,
    int *topk_idxs,
    int batch,
    int window_size,
    int max_compressed_len,
    int index_topk,
    cudaStream_t stream) {
  if (scores == nullptr || start_pos == nullptr || window_base == nullptr ||
      compressed_len == nullptr || compressed_base == nullptr || topk_idxs == nullptr ||
      batch <= 0 || window_size <= 0 || max_compressed_len <= 0 || index_topk <= 0 ||
      index_topk > max_compressed_len) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  size_t shared_bytes = max_compressed_len * sizeof(float);
  deepseek_ratio4_decode_topk_indices_batch_kernel<<<batch, threads, shared_bytes, stream>>>(
      scores, start_pos, window_base, compressed_len, compressed_base, topk_idxs, batch,
      window_size, max_compressed_len, index_topk);
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

cudaError_t deepseek_window_topk_indices_cuda(
    int *out,
    int seq_len,
    int window_size,
    int topk,
    cudaStream_t stream) {
  if (seq_len <= 0 || window_size <= 0 || topk <= 0) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int total = seq_len * topk;
  int blocks = (total + threads - 1) / threads;
  deepseek_window_topk_indices_kernel<<<blocks, threads, 0, stream>>>(
      out, seq_len, window_size, topk);
  return cudaGetLastError();
}

cudaError_t deepseek_window_topk_indices_decode_cuda(
    int *out,
    int start_pos,
    int window_size,
    cudaStream_t stream) {
  if (start_pos < 0 || window_size <= 0) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int blocks = (window_size + threads - 1) / threads;
  deepseek_window_topk_indices_decode_kernel<<<blocks, threads, 0, stream>>>(
      out, start_pos, window_size);
  return cudaGetLastError();
}

cudaError_t deepseek_window_topk_indices_decode_batch_cuda(
    int *out,
    const int *start_pos,
    const int *cache_base,
    int batch,
    int window_size,
    int topk,
    cudaStream_t stream) {
  if (batch <= 0 || window_size <= 0 || topk <= 0 || start_pos == nullptr ||
      cache_base == nullptr) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int total = batch * topk;
  int blocks = (total + threads - 1) / threads;
  deepseek_window_topk_indices_decode_batch_kernel<<<blocks, threads, 0, stream>>>(
      out, start_pos, cache_base, batch, window_size, topk);
  return cudaGetLastError();
}

cudaError_t deepseek_compress_topk_indices_cuda(
    int *out,
    int seq_len,
    int compressed,
    int ratio,
    int offset,
    cudaStream_t stream) {
  if (seq_len <= 0 || compressed <= 0 || ratio <= 0) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int total = seq_len * compressed;
  int blocks = (total + threads - 1) / threads;
  deepseek_compress_topk_indices_kernel<<<blocks, threads, 0, stream>>>(
      out, seq_len, compressed, ratio, offset);
  return cudaGetLastError();
}

cudaError_t deepseek_compress_topk_indices_decode_cuda(
    int *out,
    int compressed,
    int offset,
    cudaStream_t stream) {
  if (compressed < 0) return cudaErrorInvalidValue;
  if (compressed == 0) return cudaSuccess;
  constexpr int threads = 256;
  int blocks = (compressed + threads - 1) / threads;
  deepseek_compress_topk_indices_decode_kernel<<<blocks, threads, 0, stream>>>(
      out, compressed, offset);
  return cudaGetLastError();
}

cudaError_t deepseek_compress_topk_indices_decode_batch_cuda(
    int *out,
    const int *compressed_len,
    const int *cache_base,
    int batch,
    int topk,
    cudaStream_t stream) {
  if (batch <= 0 || topk < 0 || compressed_len == nullptr || cache_base == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (topk == 0) return cudaSuccess;
  constexpr int threads = 256;
  int total = batch * topk;
  int blocks = (total + threads - 1) / threads;
  deepseek_compress_topk_indices_decode_batch_kernel<<<blocks, threads, 0, stream>>>(
      out, compressed_len, cache_base, batch, topk);
  return cudaGetLastError();
}

cudaError_t deepseek_window_and_compress_topk_indices_cuda(
    int *out,
    int seq_len,
    int window_size,
    int window_topk,
    int compressed,
    int ratio,
    int compress_offset,
    int topk,
    cudaStream_t stream) {
  if (seq_len <= 0 || window_size <= 0 || window_topk <= 0 ||
      compressed <= 0 || ratio <= 0 || topk <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int total = seq_len * topk;
  int blocks = (total + threads - 1) / threads;
  deepseek_window_and_compress_topk_indices_kernel<<<blocks, threads, 0, stream>>>(
      out, seq_len, window_size, window_topk, compressed, ratio, compress_offset, topk);
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
