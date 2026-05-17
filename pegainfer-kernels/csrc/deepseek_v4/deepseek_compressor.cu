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

__global__ void deepseek_apply_rope_hidden_batch_kernel(
    __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    const int *__restrict__ start_pos,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  if (pair >= total_pairs) return;

  int rotary_pair = pair % (rotary_dim / 2);
  int tmp = pair / (rotary_dim / 2);
  int head = tmp % local_heads;
  int token = tmp / local_heads;
  int nope_dim = head_dim - rotary_dim;
  int pos = start_pos[token];
  if (pos < 0) return;
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

// Fused overlap-compressor prefill epilogue. Consumes per-token (score, value)
// FP32 tensors produced by two upstream X @ W^T BF16->FP32 cuBLAS GEMMs (gate
// scores, kv values), and for each compressed row:
//   1) gathers the 8 ratio-4 overlap routes per (compressed, dim) cell,
//   2) adds the per-route APE bias and runs numerically-stable softmax,
//   3) writes the FP32 weighted output,
//   4) reduces the row's sum-of-squares to inv_rms and emits the BF16 RMSNormed
//      output -- all in one launch, without round-tripping `weighted` through
//      HBM between softmax and RMSNorm.
//
// scores_in / values_in are row-major (seq_len, 2*head_dim) FP32 tensors emitted
// by cuBLAS via the swap-and-transpose trick used in deepseek_bf16_linear_cuda:
// cuBLAS sees A=W with OP_T and B=X with OP_N, so its column-major (n, seq_len)
// result is bit-equal to the row-major (seq_len, n) buffer we read here.
// sv_n_stride is the M-row stride, equal to 2 * head_dim.
//
// Launch: 1 block per compressed position, blockDim.x covers head_dim with a
// strided loop. Block must be a multiple of warpSize and at most 1024 threads.
__global__ void deepseek_compressor_overlap_fused_epilogue_kernel(
    const float *__restrict__ scores_in,
    const float *__restrict__ values_in,
    const float *__restrict__ ape,
    const __nv_bfloat16 *__restrict__ norm,
    float *__restrict__ weighted,
    __nv_bfloat16 *__restrict__ out,
    int compressed_len,
    int head_dim,
    int sv_n_stride,
    float eps) {
  int c = blockIdx.x;
  if (c >= compressed_len) return;
  int tid = threadIdx.x;
  int n_block = blockDim.x;

  constexpr int ratio = 4;
  constexpr int routes = 8;
  constexpr float neg_inf = -3.4028234663852886e38f;

  float sum_sq_local = 0.0f;

  for (int d = tid; d < head_dim; d += n_block) {
    float scores[routes];
    float values[routes];
#pragma unroll
    for (int r = 0; r < routes; ++r) {
      bool valid;
      int token;
      int out_dim;
      int ape_dim;
      if (r < ratio) {
        valid = c > 0;
        token = (c - 1) * ratio + r;
        out_dim = d;
        ape_dim = r * (2 * head_dim) + d;
      } else {
        int lr = r - ratio;
        valid = true;
        token = c * ratio + lr;
        out_dim = head_dim + d;
        ape_dim = lr * (2 * head_dim) + head_dim + d;
      }
      if (valid) {
        int offset = token * sv_n_stride + out_dim;
        scores[r] = scores_in[offset] + ape[ape_dim];
        values[r] = values_in[offset];
      } else {
        scores[r] = neg_inf;
        values[r] = 0.0f;
      }
    }

    float m = scores[0];
#pragma unroll
    for (int r = 1; r < routes; ++r) m = fmaxf(m, scores[r]);

    float denom = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < routes; ++r) {
      float p = __expf(scores[r] - m);
      denom += p;
      acc += p * values[r];
    }
    float w = acc / denom;
    weighted[c * head_dim + d] = w;
    sum_sq_local += w * w;
  }

  // Block-wide reduction of sum_sq via warp-shfl + smem.
  __shared__ float warp_sums[32];
  int lane = tid & 31;
  int warp = tid >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum_sq_local += __shfl_down_sync(0xffffffffu, sum_sq_local, off);
  }
  if (lane == 0) warp_sums[warp] = sum_sq_local;
  __syncthreads();

  int n_warps = (n_block + 31) >> 5;
  float total = (tid < n_warps) ? warp_sums[tid] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, off);
    }
  }
  __shared__ float total_sum;
  if (tid == 0) total_sum = total;
  __syncthreads();

  float inv_rms = rsqrtf(total_sum / static_cast<float>(head_dim) + eps);

  for (int d = tid; d < head_dim; d += n_block) {
    float w = weighted[c * head_dim + d];
    float ns = __bfloat162float(norm[d]);
    out[c * head_dim + d] = __float2bfloat16(w * inv_rms * ns);
  }
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

cudaError_t deepseek_apply_rope_hidden_batch_cuda(
    __nv_bfloat16 *x,
    const float *cos_cache,
    const float *sin_cache,
    const int *start_pos,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int inverse,
    cudaStream_t stream) {
  if (x == nullptr || cos_cache == nullptr || sin_cache == nullptr || start_pos == nullptr ||
      seq_len <= 0 || local_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || (rotary_dim % 2) != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  deepseek_apply_rope_hidden_batch_kernel<<<blocks, threads, 0, stream>>>(
      x, cos_cache, sin_cache, start_pos, seq_len, local_heads, head_dim, rotary_dim, inverse);
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

// Two BF16 x BF16 -> FP32 cuBLAS GEMMs (X @ Wgate^T for scores, X @ Wkv^T for
// values) into a shared FP32 scratch, then one fused epilogue kernel that
// gathers the 8 overlap routes, softmaxes, writes `weighted`, and RMSNorms in
// place to BF16 `out`. The single scratch alloc (covering both score + value
// buffers) avoids paying cudaMallocAsync overhead twice.
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
  if (x == nullptr || wkv == nullptr || wgate == nullptr || ape == nullptr ||
      norm == nullptr || weighted == nullptr || out == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (seq_len < 4 || (seq_len & 3) != 0 || hidden_dim <= 0 || head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  const int compressed_len = seq_len / 4;
  const int n = 2 * head_dim;

  DeepseekCompressorScratch *scratch_ptr = nullptr;
  cudaError_t cuda_status = deepseek_compressor_scratch_for_device(&scratch_ptr);
  if (cuda_status != cudaSuccess) return cuda_status;
  DeepseekCompressorScratch &scratch = *scratch_ptr;
  std::lock_guard<std::mutex> lock(scratch.mutex);
  cuda_status = deepseek_ensure_compressor_bf16_linear_handle(scratch);
  if (cuda_status != cudaSuccess) return cuda_status;
  cublasStatus_t status = cublasSetStream(scratch.bf16_linear_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  const size_t sv_elems = static_cast<size_t>(seq_len) * n;
  float *sv_buf = nullptr;
  cuda_status = cudaMallocAsync(
      reinterpret_cast<void **>(&sv_buf), 2 * sv_elems * sizeof(float), stream);
  if (cuda_status != cudaSuccess) return cuda_status;
  float *scores_buf = sv_buf;
  float *values_buf = sv_buf + sv_elems;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  // C = X @ W^T at row-major (seq_len, 2*head_dim) via cuBLAS column-major
  // swap-and-transpose: A=W with OP_T and B=X with OP_N, then C in cuBLAS
  // column-major (n, seq_len) is bit-for-bit the row-major (seq_len, n)
  // buffer the epilogue reads. Output dtype is CUDA_R_32F to preserve the
  // FP32 tensor-core accumulator -- a BF16 output here loses ~3 orders of
  // magnitude of precision in the per-route scores before softmax.
  status = cublasGemmEx(
      scratch.bf16_linear_handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      n, seq_len, hidden_dim,
      &alpha,
      wgate, CUDA_R_16BF, hidden_dim,
      x, CUDA_R_16BF, hidden_dim,
      &beta,
      scores_buf, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFreeAsync(sv_buf, stream);
    return cudaErrorUnknown;
  }
  status = cublasGemmEx(
      scratch.bf16_linear_handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      n, seq_len, hidden_dim,
      &alpha,
      wkv, CUDA_R_16BF, hidden_dim,
      x, CUDA_R_16BF, hidden_dim,
      &beta,
      values_buf, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFreeAsync(sv_buf, stream);
    return cudaErrorUnknown;
  }

  // Block size: round head_dim up to a multiple of warpSize, capped at 1024.
  int epilogue_threads = head_dim;
  if (epilogue_threads < 32) epilogue_threads = 32;
  if (epilogue_threads > 1024) epilogue_threads = 1024;
  epilogue_threads = (epilogue_threads + 31) & ~31;
  deepseek_compressor_overlap_fused_epilogue_kernel<<<compressed_len, epilogue_threads, 0, stream>>>(
      scores_buf, values_buf, ape, norm, weighted, out,
      compressed_len, head_dim, n, eps);
  cuda_status = cudaGetLastError();
  cudaFreeAsync(sv_buf, stream);
  return cuda_status;
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

cudaError_t deepseek_compressor_nonoverlap_decode_at_cuda(
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
    int state_offset,
    float eps,
    cudaStream_t stream) {
  if (start_pos < 0 || hidden_dim <= 0 || head_dim <= 0 || ratio <= 1 || ratio > 128 ||
      state_offset < 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  size_t project_shared = 2 * threads * sizeof(float);
  deepseek_compressor_decode_project_kernel<<<head_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, start_pos, hidden_dim, head_dim, ratio,
      state_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  bool should_compress = ((start_pos + 1) % ratio) == 0;
  if (!should_compress) return cudaSuccess;
  if (weighted == nullptr || out == nullptr) return cudaErrorInvalidValue;

  int blocks = (head_dim + threads - 1) / threads;
  float *kv_slot = kv_state + state_offset * head_dim;
  float *score_slot = score_state + state_offset * head_dim;
  deepseek_compressor_nonoverlap_decode_weighted_kernel<<<blocks, threads, 0, stream>>>(
      kv_slot, score_slot, weighted, head_dim, ratio);
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

cudaError_t deepseek_compressor_overlap_decode_at_cuda(
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
    int state_offset,
    float eps,
    cudaStream_t stream) {
  if (start_pos < 0 || hidden_dim <= 0 || head_dim <= 0 || state_offset < 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int ratio = 4;
  constexpr int threads = 256;
  int state_dim = 2 * head_dim;
  size_t project_shared = 2 * threads * sizeof(float);
  deepseek_compressor_decode_project_kernel<<<state_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, start_pos, hidden_dim, state_dim, ratio,
      state_offset + ratio);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  bool should_compress = ((start_pos + 1) % ratio) == 0;
  if (!should_compress) return cudaSuccess;
  if (weighted == nullptr || out == nullptr) return cudaErrorInvalidValue;

  int blocks = (head_dim + threads - 1) / threads;
  float *kv_slot = kv_state + state_offset * state_dim;
  float *score_slot = score_state + state_offset * state_dim;
  deepseek_compressor_overlap_decode_weighted_kernel<<<blocks, threads, 0, stream>>>(
      kv_slot, score_slot, weighted, head_dim);
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
      kv_slot, score_slot, state_dim);
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
