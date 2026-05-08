#include "deepseek_common.cuh"

#include <mutex>

namespace {

constexpr int kMaxCompressorScratchDevices = 16;

struct DeepseekCompressorScratch {
  cublasHandle_t bf16_linear_handle = nullptr;
  std::mutex mutex;
};

DeepseekCompressorScratch g_compressor_scratch[kMaxCompressorScratchDevices];

cudaError_t deepseek_compressor_scratch_for_device(DeepseekCompressorScratch** out) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return err;
  }
  if (device < 0 || device >= kMaxCompressorScratchDevices) {
    return cudaErrorInvalidDevice;
  }
  *out = &g_compressor_scratch[device];
  return cudaSuccess;
}

cudaError_t deepseek_ensure_compressor_bf16_linear_handle(
    DeepseekCompressorScratch& scratch) {
  if (scratch.bf16_linear_handle != nullptr) {
    return cudaSuccess;
  }
  cublasStatus_t status = cublasCreate(&scratch.bf16_linear_handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    scratch.bf16_linear_handle = nullptr;
    return cudaErrorUnknown;
  }
  status = cublasSetMathMode(scratch.bf16_linear_handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(scratch.bf16_linear_handle);
    scratch.bf16_linear_handle = nullptr;
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

}  // namespace

static __device__ __forceinline__ void deepseek_apply_rope_pair(
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

__global__ void deepseek_apply_rope_hidden_kernel(
    __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  if (pair >= total_pairs) return;

  int rotary_pair = pair % (rotary_dim / 2);
  int tmp = pair / (rotary_dim / 2);
  int head = tmp % local_heads;
  int token = tmp / local_heads;
  int nope_dim = head_dim - rotary_dim;
  int pos = start_pos + token;
  int offset = token * local_heads * head_dim + head * head_dim + nope_dim + 2 * rotary_pair;
  deepseek_apply_rope_pair(
      x, offset, cos_cache[pos * (rotary_dim / 2) + rotary_pair],
      sin_cache[pos * (rotary_dim / 2) + rotary_pair], inverse != 0);
}

__global__ void deepseek_apply_rope_hidden_strided_kernel(
    __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int position_stride,
    int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  if (pair >= total_pairs) return;

  int rotary_pair = pair % (rotary_dim / 2);
  int tmp = pair / (rotary_dim / 2);
  int head = tmp % local_heads;
  int token = tmp / local_heads;
  int nope_dim = head_dim - rotary_dim;
  int pos = start_pos + token * position_stride;
  int offset = token * local_heads * head_dim + head * head_dim + nope_dim + 2 * rotary_pair;
  deepseek_apply_rope_pair(
      x, offset, cos_cache[pos * (rotary_dim / 2) + rotary_pair],
      sin_cache[pos * (rotary_dim / 2) + rotary_pair], inverse != 0);
}

__global__ void deepseek_compressor_nonoverlap_weighted_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ weighted,
    int compressed_len,
    int hidden_dim,
    int head_dim,
    int ratio) {
  int dim = blockIdx.x;
  int compressed = blockIdx.y;
  int tid = threadIdx.x;
  if (dim >= head_dim || compressed >= compressed_len) return;

  extern __shared__ float scratch[];
  float *score_scratch = scratch;
  float *kv_scratch = scratch + blockDim.x;
  float *scores = scratch + 2 * blockDim.x;
  float *values = scores + ratio;

  for (int route = 0; route < ratio; ++route) {
    int token = compressed * ratio + route;
    float score_partial = 0.0f;
    float kv_partial = 0.0f;
    for (int k = tid; k < hidden_dim; k += blockDim.x) {
      float x_value = __bfloat162float(x[token * hidden_dim + k]);
      score_partial += x_value * __bfloat162float(wgate[dim * hidden_dim + k]);
      kv_partial += x_value * __bfloat162float(wkv[dim * hidden_dim + k]);
    }
    score_scratch[tid] = score_partial;
    kv_scratch[tid] = kv_partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        score_scratch[tid] += score_scratch[tid + stride];
        kv_scratch[tid] += kv_scratch[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      scores[route] = score_scratch[0] + ape[route * head_dim + dim];
      values[route] = kv_scratch[0];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float max_score = -3.4028234663852886e38f;
    for (int route = 0; route < ratio; ++route) {
      max_score = fmaxf(max_score, scores[route]);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (int route = 0; route < ratio; ++route) {
      float prob = expf(scores[route] - max_score);
      denom += prob;
      acc += prob * values[route];
    }
    weighted[compressed * head_dim + dim] = acc / denom;
  }
}

__global__ void deepseek_compressor_nonoverlap_weighted_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ weighted,
    int compressed_len,
    int hidden_dim,
    int head_dim,
    int ratio) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = compressed_len * head_dim;
  if (idx >= total) return;

  int dim = idx % head_dim;
  int compressed = idx / head_dim;
  float max_score = -3.4028234663852886e38f;
  float scores[128];
  float values[128];

  for (int route = 0; route < ratio; ++route) {
    int token = compressed * ratio + route;
    float score_sum = 0.0f;
    float kv_sum = 0.0f;
    for (int k = 0; k < hidden_dim; ++k) {
      float x_value = __bfloat162float(x[token * hidden_dim + k]);
      score_sum += x_value * __bfloat162float(wgate[dim * hidden_dim + k]);
      kv_sum += x_value * __bfloat162float(wkv[dim * hidden_dim + k]);
    }
    scores[route] = score_sum + ape[route * head_dim + dim];
    values[route] = kv_sum;
    max_score = fmaxf(max_score, scores[route]);
  }

  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < ratio; ++route) {
    float prob = expf(scores[route] - max_score);
    denom += prob;
    acc += prob * values[route];
  }
  weighted[compressed * head_dim + dim] = acc / denom;
}

__global__ void deepseek_compressor_nonoverlap_norm_kernel(
    const float *__restrict__ weighted,
    const __nv_bfloat16 *__restrict__ norm,
    __nv_bfloat16 *__restrict__ out,
    int compressed_len,
    int head_dim,
    float eps) {
  int compressed = blockIdx.x;
  int tid = threadIdx.x;
  if (compressed >= compressed_len) return;

  extern __shared__ float scratch[];
  float partial = 0.0f;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float value = weighted[compressed * head_dim + dim];
    partial += value * value;
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(scratch[0] / head_dim + eps);
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float value = weighted[compressed * head_dim + dim] * inv_rms *
                  __bfloat162float(norm[dim]);
    out[compressed * head_dim + dim] = __float2bfloat16(value);
  }
}

__global__ void deepseek_compressor_norm_serial_kernel(
    const float *__restrict__ weighted,
    const __nv_bfloat16 *__restrict__ norm,
    __nv_bfloat16 *__restrict__ out,
    int compressed_len,
    int head_dim,
    float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = compressed_len * head_dim;
  if (idx >= total) return;

  int dim = idx % head_dim;
  int compressed = idx / head_dim;
  float sum_sq = 0.0f;
  for (int k = 0; k < head_dim; ++k) {
    float value = weighted[compressed * head_dim + k];
    sum_sq += value * value;
  }
  float inv_rms = rsqrtf(sum_sq / head_dim + eps);
  float value = weighted[compressed * head_dim + dim] * inv_rms *
                __bfloat162float(norm[dim]);
  out[compressed * head_dim + dim] = __float2bfloat16(value);
}

__global__ void deepseek_compressor_overlap_weighted_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ weighted,
    int compressed_len,
    int hidden_dim,
    int head_dim) {
  int dim = blockIdx.x;
  int compressed = blockIdx.y;
  int tid = threadIdx.x;
  if (dim >= head_dim || compressed >= compressed_len) return;

  constexpr int ratio = 4;
  constexpr int routes = 8;
  extern __shared__ float scratch[];
  float *score_scratch = scratch;
  float *kv_scratch = scratch + blockDim.x;
  float *scores = scratch + 2 * blockDim.x;
  float *values = scores + routes;

  for (int route = 0; route < routes; ++route) {
    bool valid = true;
    int token = 0;
    int out_dim = 0;
    int ape_dim = 0;
    if (route < ratio) {
      valid = compressed > 0;
      token = (compressed - 1) * ratio + route;
      out_dim = dim;
      ape_dim = route * (2 * head_dim) + dim;
    } else {
      int local_route = route - ratio;
      token = compressed * ratio + local_route;
      out_dim = head_dim + dim;
      ape_dim = local_route * (2 * head_dim) + head_dim + dim;
    }

    float score_partial = 0.0f;
    float kv_partial = 0.0f;
    if (valid) {
      for (int k = tid; k < hidden_dim; k += blockDim.x) {
        float x_value = __bfloat162float(x[token * hidden_dim + k]);
        score_partial += x_value * __bfloat162float(wgate[out_dim * hidden_dim + k]);
        kv_partial += x_value * __bfloat162float(wkv[out_dim * hidden_dim + k]);
      }
    }
    score_scratch[tid] = score_partial;
    kv_scratch[tid] = kv_partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        score_scratch[tid] += score_scratch[tid + stride];
        kv_scratch[tid] += kv_scratch[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      scores[route] = valid ? score_scratch[0] + ape[ape_dim] : -3.4028234663852886e38f;
      values[route] = valid ? kv_scratch[0] : 0.0f;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float max_score = -3.4028234663852886e38f;
    for (int route = 0; route < routes; ++route) {
      max_score = fmaxf(max_score, scores[route]);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (int route = 0; route < routes; ++route) {
      float prob = expf(scores[route] - max_score);
      denom += prob;
      acc += prob * values[route];
    }
    weighted[compressed * head_dim + dim] = acc / denom;
  }
}

__global__ void deepseek_compressor_overlap_weighted_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ weighted,
    int compressed_len,
    int hidden_dim,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = compressed_len * head_dim;
  if (idx >= total) return;

  int dim = idx % head_dim;
  int compressed = idx / head_dim;
  constexpr int ratio = 4;
  constexpr int routes = 8;
  float scores[routes];
  float values[routes];

  for (int route = 0; route < routes; ++route) {
    bool valid = true;
    int token = 0;
    int out_dim = 0;
    int ape_dim = 0;
    if (route < ratio) {
      valid = compressed > 0;
      token = (compressed - 1) * ratio + route;
      out_dim = dim;
      ape_dim = route * (2 * head_dim) + dim;
    } else {
      int local_route = route - ratio;
      token = compressed * ratio + local_route;
      out_dim = head_dim + dim;
      ape_dim = local_route * (2 * head_dim) + head_dim + dim;
    }

    if (valid) {
      float score_sum = 0.0f;
      float kv_sum = 0.0f;
      for (int k = 0; k < hidden_dim; ++k) {
        float x_value = __bfloat162float(x[token * hidden_dim + k]);
        score_sum += x_value * __bfloat162float(wgate[out_dim * hidden_dim + k]);
        kv_sum += x_value * __bfloat162float(wkv[out_dim * hidden_dim + k]);
      }
      scores[route] = score_sum + ape[ape_dim];
      values[route] = kv_sum;
    } else {
      scores[route] = -3.4028234663852886e38f;
      values[route] = 0.0f;
    }
  }

  float max_score = -3.4028234663852886e38f;
  for (int route = 0; route < routes; ++route) {
    max_score = fmaxf(max_score, scores[route]);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < routes; ++route) {
    float prob = expf(scores[route] - max_score);
    denom += prob;
    acc += prob * values[route];
  }
  weighted[compressed * head_dim + dim] = acc / denom;
}

__global__ void deepseek_compressor_decode_project_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    int start_pos,
    int hidden_dim,
    int out_dim,
    int ratio,
    int state_offset) {
  int dim = blockIdx.x;
  int tid = threadIdx.x;
  if (dim >= out_dim) return;

  extern __shared__ float scratch[];
  float *kv_scratch = scratch;
  float *score_scratch = scratch + blockDim.x;
  float kv_partial = 0.0f;
  float score_partial = 0.0f;
  for (int k = tid; k < hidden_dim; k += blockDim.x) {
    float xv = __bfloat162float(x[k]);
    kv_partial += xv * __bfloat162float(wkv[dim * hidden_dim + k]);
    score_partial += xv * __bfloat162float(wgate[dim * hidden_dim + k]);
  }
  kv_scratch[tid] = kv_partial;
  score_scratch[tid] = score_partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      kv_scratch[tid] += kv_scratch[tid + stride];
      score_scratch[tid] += score_scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int local_pos = start_pos % ratio;
    int state_row = state_offset + local_pos;
    kv_state[state_row * out_dim + dim] = kv_scratch[0];
    score_state[state_row * out_dim + dim] =
        score_scratch[0] + ape[local_pos * out_dim + dim];
  }
}

__global__ void deepseek_compressor_decode_project_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    int start_pos,
    int hidden_dim,
    int out_dim,
    int ratio,
    int state_offset) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= out_dim) return;

  float kv_sum = 0.0f;
  float score_sum = 0.0f;
  for (int k = 0; k < hidden_dim; ++k) {
    float xv = __bfloat162float(x[k]);
    kv_sum += xv * __bfloat162float(wkv[dim * hidden_dim + k]);
    score_sum += xv * __bfloat162float(wgate[dim * hidden_dim + k]);
  }

  int local_pos = start_pos % ratio;
  int state_row = state_offset + local_pos;
  kv_state[state_row * out_dim + dim] = kv_sum;
  score_state[state_row * out_dim + dim] = score_sum + ape[local_pos * out_dim + dim];
}

__global__ void deepseek_compressor_nonoverlap_decode_weighted_kernel(
    const float *__restrict__ kv_state,
    const float *__restrict__ score_state,
    float *__restrict__ weighted,
    int head_dim,
    int ratio) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;

  float max_score = -3.4028234663852886e38f;
  for (int route = 0; route < ratio; ++route) {
    max_score = fmaxf(max_score, score_state[route * head_dim + dim]);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < ratio; ++route) {
    float prob = expf(score_state[route * head_dim + dim] - max_score);
    denom += prob;
    acc += prob * kv_state[route * head_dim + dim];
  }
  weighted[dim] = acc / denom;
}

__global__ void deepseek_compressor_overlap_decode_weighted_kernel(
    const float *__restrict__ kv_state,
    const float *__restrict__ score_state,
    float *__restrict__ weighted,
    int head_dim) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;

  constexpr int ratio = 4;
  constexpr int routes = 8;
  int state_dim = 2 * head_dim;
  float route_scores[routes];
  float route_values[routes];
  for (int route = 0; route < routes; ++route) {
    if (route < ratio) {
      route_scores[route] = score_state[route * state_dim + dim];
      route_values[route] = kv_state[route * state_dim + dim];
    } else {
      int local = route - ratio;
      route_scores[route] = score_state[(ratio + local) * state_dim + head_dim + dim];
      route_values[route] = kv_state[(ratio + local) * state_dim + head_dim + dim];
    }
  }

  float max_score = -3.4028234663852886e38f;
  for (int route = 0; route < routes; ++route) {
    max_score = fmaxf(max_score, route_scores[route]);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < routes; ++route) {
    float prob = expf(route_scores[route] - max_score);
    denom += prob;
    acc += prob * route_values[route];
  }
  weighted[dim] = acc / denom;
}

__global__ void deepseek_compressor_overlap_shift_kernel(
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    int state_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 4 * state_dim;
  if (idx >= total) return;
  kv_state[idx] = kv_state[total + idx];
  score_state[idx] = score_state[total + idx];
}

__global__ void deepseek_concat_seq_bf16_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    int a_seq_len,
    int b_seq_len,
    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (a_seq_len + b_seq_len) * hidden_dim;
  if (idx >= total) return;
  int token = idx / hidden_dim;
  int dim = idx % hidden_dim;
  if (token < a_seq_len) {
    out[idx] = a[token * hidden_dim + dim];
  } else {
    int b_token = token - a_seq_len;
    out[idx] = b[b_token * hidden_dim + dim];
  }
}

__global__ void deepseek_bf16_linear_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int in_dim,
    int out_dim) {
  int out_col = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  if (out_col >= out_dim || token >= seq_len) return;

  extern __shared__ float scratch[];
  float partial = 0.0f;
  for (int k = tid; k < in_dim; k += blockDim.x) {
    float x_value = __bfloat162float(x[token * in_dim + k]);
    float w_value = __bfloat162float(weight[out_col * in_dim + k]);
    partial += x_value * w_value;
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
    out[token * out_dim + out_col] = __float2bfloat16(scratch[0]);
  }
}

__global__ void deepseek_bf16_linear_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int in_dim,
    int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * out_dim;
  if (idx >= total) return;

  int token = idx / out_dim;
  int out_col = idx - token * out_dim;
  float sum = 0.0f;
  for (int k = 0; k < in_dim; ++k) {
    float x_value = __bfloat162float(x[token * in_dim + k]);
    float w_value = __bfloat162float(weight[out_col * in_dim + k]);
    sum += x_value * w_value;
  }
  out[token * out_dim + out_col] = __float2bfloat16(sum);
}

extern "C" {

cudaError_t deepseek_apply_rope_hidden_cuda(
    __nv_bfloat16 *x,
    const float *cos_cache,
    const float *sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int inverse,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  deepseek_apply_rope_hidden_kernel<<<blocks, threads, 0, stream>>>(
      x, cos_cache, sin_cache, seq_len, local_heads, head_dim, rotary_dim, start_pos, inverse);
  return cudaGetLastError();
}

cudaError_t deepseek_apply_rope_hidden_strided_cuda(
    __nv_bfloat16 *x,
    const float *cos_cache,
    const float *sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int position_stride,
    int inverse,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  deepseek_apply_rope_hidden_strided_kernel<<<blocks, threads, 0, stream>>>(
      x, cos_cache, sin_cache, seq_len, local_heads, head_dim, rotary_dim, start_pos,
      position_stride, inverse);
  return cudaGetLastError();
}

cudaError_t deepseek_bf16_linear_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,
    __nv_bfloat16 *out,
    int seq_len,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  DeepseekCompressorScratch* scratch_ptr = nullptr;
  cudaError_t cuda_status = deepseek_compressor_scratch_for_device(&scratch_ptr);
  if (cuda_status != cudaSuccess) return cuda_status;
  DeepseekCompressorScratch& scratch = *scratch_ptr;
  std::lock_guard<std::mutex> lock(scratch.mutex);
  cuda_status = deepseek_ensure_compressor_bf16_linear_handle(scratch);
  if (cuda_status != cudaSuccess) return cuda_status;
  cublasStatus_t status = cublasSetStream(scratch.bf16_linear_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasGemmEx(
      scratch.bf16_linear_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      out_dim,
      seq_len,
      in_dim,
      &alpha,
      weight,
      CUDA_R_16BF,
      in_dim,
      x,
      CUDA_R_16BF,
      in_dim,
      &beta,
      out,
      CUDA_R_16BF,
      out_dim,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
  return cudaGetLastError();
}

cudaError_t deepseek_compressor_nonoverlap_prefill_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *weighted,
    __nv_bfloat16 *out,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int ratio,
    float eps,
    cudaStream_t stream) {
  if (ratio <= 0 || ratio > 128 || seq_len < ratio) return cudaErrorInvalidValue;
  int compressed_len = seq_len / ratio;
  constexpr int threads = 256;
  dim3 weighted_grid(head_dim, compressed_len);
  size_t weighted_shared = (2 * threads + 2 * ratio) * sizeof(float);
  deepseek_compressor_nonoverlap_weighted_kernel<<<weighted_grid, threads, weighted_shared, stream>>>(
      x, wkv, wgate, ape, weighted, compressed_len, hidden_dim, head_dim, ratio);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_total = compressed_len * head_dim;
  int norm_blocks = (norm_total + threads - 1) / threads;
  deepseek_compressor_norm_serial_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, compressed_len, head_dim, eps);
  return cudaGetLastError();
}

cudaError_t deepseek_compressor_overlap_prefill_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *weighted,
    __nv_bfloat16 *out,
    int seq_len,
    int hidden_dim,
    int head_dim,
    float eps,
    cudaStream_t stream) {
  if (seq_len < 4) return cudaErrorInvalidValue;
  int compressed_len = seq_len / 4;
  constexpr int threads = 256;
  dim3 weighted_grid(head_dim, compressed_len);
  constexpr int routes = 8;
  size_t weighted_shared = (2 * threads + 2 * routes) * sizeof(float);
  deepseek_compressor_overlap_weighted_kernel<<<weighted_grid, threads, weighted_shared, stream>>>(
      x, wkv, wgate, ape, weighted, compressed_len, hidden_dim, head_dim);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_total = compressed_len * head_dim;
  int norm_blocks = (norm_total + threads - 1) / threads;
  deepseek_compressor_norm_serial_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, compressed_len, head_dim, eps);
  return cudaGetLastError();
}

cudaError_t deepseek_compressor_nonoverlap_decode_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *kv_state,
    float *score_state,
    float *weighted,
    __nv_bfloat16 *out,
    int start_pos,
    int hidden_dim,
    int head_dim,
    int ratio,
    float eps,
    cudaStream_t stream) {
  if (start_pos < 0 || hidden_dim <= 0 || head_dim <= 0 || ratio <= 1 || ratio > 128) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  size_t project_shared = 2 * threads * sizeof(float);
  deepseek_compressor_decode_project_kernel<<<head_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, start_pos, hidden_dim, head_dim, ratio, 0);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  bool should_compress = ((start_pos + 1) % ratio) == 0;
  if (!should_compress) return cudaSuccess;
  if (weighted == nullptr || out == nullptr) return cudaErrorInvalidValue;

  int blocks = (head_dim + threads - 1) / threads;
  deepseek_compressor_nonoverlap_decode_weighted_kernel<<<blocks, threads, 0, stream>>>(
      kv_state, score_state, weighted, head_dim, ratio);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (head_dim + threads - 1) / threads;
  deepseek_compressor_norm_serial_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, 1, head_dim, eps);
  return cudaGetLastError();
}

cudaError_t deepseek_compressor_overlap_decode_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *kv_state,
    float *score_state,
    float *weighted,
    __nv_bfloat16 *out,
    int start_pos,
    int hidden_dim,
    int head_dim,
    float eps,
    cudaStream_t stream) {
  if (start_pos < 0 || hidden_dim <= 0 || head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int ratio = 4;
  constexpr int threads = 256;
  int state_dim = 2 * head_dim;
  size_t project_shared = 2 * threads * sizeof(float);
  deepseek_compressor_decode_project_kernel<<<state_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, start_pos, hidden_dim, state_dim, ratio, ratio);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  bool should_compress = ((start_pos + 1) % ratio) == 0;
  if (!should_compress) return cudaSuccess;
  if (weighted == nullptr || out == nullptr) return cudaErrorInvalidValue;

  int blocks = (head_dim + threads - 1) / threads;
  deepseek_compressor_overlap_decode_weighted_kernel<<<blocks, threads, 0, stream>>>(
      kv_state, score_state, weighted, head_dim);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (head_dim + threads - 1) / threads;
  deepseek_compressor_norm_serial_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, 1, head_dim, eps);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int shift_total = ratio * state_dim;
  int shift_blocks = (shift_total + threads - 1) / threads;
  deepseek_compressor_overlap_shift_kernel<<<shift_blocks, threads, 0, stream>>>(
      kv_state, score_state, state_dim);
  return cudaGetLastError();
}

cudaError_t deepseek_concat_seq_bf16_cuda(
    const __nv_bfloat16 *a,
    const __nv_bfloat16 *b,
    __nv_bfloat16 *out,
    int a_seq_len,
    int b_seq_len,
    int hidden_dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = (a_seq_len + b_seq_len) * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_concat_seq_bf16_kernel<<<blocks, threads, 0, stream>>>(
      a, b, out, a_seq_len, b_seq_len, hidden_dim);
  return cudaGetLastError();
}

}  // extern "C"
