#include "deepseek_common.cuh"

#include <mutex>

namespace {

constexpr int kMaxHcScratchDevices = 16;

struct DeepseekHcScratch {
  float* x_f32 = nullptr;
  size_t x_elems = 0;
  cublasHandle_t handle = nullptr;
  std::mutex mutex;
};

DeepseekHcScratch g_hc_scratch[kMaxHcScratchDevices];

cudaError_t deepseek_ensure_hc_f32_scratch(float** ptr, size_t* capacity, size_t required) {
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

cudaError_t deepseek_hc_scratch_for_device(DeepseekHcScratch** out) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return err;
  }
  if (device < 0 || device >= kMaxHcScratchDevices) {
    return cudaErrorInvalidDevice;
  }
  *out = &g_hc_scratch[device];
  return cudaSuccess;
}

cudaError_t deepseek_ensure_hc_cublas_handle(DeepseekHcScratch& scratch) {
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

__global__ void deepseek_hc_bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ input,
    float *__restrict__ output,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = __bfloat162float(input[idx]);
  }
}

__global__ void deepseek_hc_expand_kernel(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int hc,
    int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hc * dim;
  if (idx >= total) return;
  int dim_idx = idx % dim;
  int token = idx / (hc * dim);
  out[idx] = x[token * dim + dim_idx];
}

__global__ void deepseek_hc_mixes_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ hc_fn,
    float *__restrict__ mixes,
    int seq_len,
    int hc,
    int dim,
    int mix_hc,
    float eps) {
  int mix = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  int hc_dim = hc * dim;
  if (mix >= mix_hc || token >= seq_len) return;

  extern __shared__ float scratch[];
  float dot = 0.0f;
  float sumsq = 0.0f;
  for (int k = tid; k < hc_dim; k += blockDim.x) {
    float value = __bfloat162float(x[token * hc_dim + k]);
    dot += value * hc_fn[mix * hc_dim + k];
    sumsq += value * value;
  }
  scratch[tid] = dot;
  scratch[blockDim.x + tid] = sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
      scratch[blockDim.x + tid] += scratch[blockDim.x + tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float rsqrt = rsqrtf(scratch[blockDim.x] / hc_dim + eps);
    mixes[token * mix_hc + mix] = scratch[0] * rsqrt;
  }
}

__global__ void deepseek_hc_scale_mixes_kernel(
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ mixes,
    int seq_len,
    int hc_dim,
    int mix_hc,
    float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * mix_hc;
  if (idx >= total) return;
  int token = idx / mix_hc;

  float sumsq = 0.0f;
  for (int k = 0; k < hc_dim; ++k) {
    float value = __bfloat162float(x[token * hc_dim + k]);
    sumsq += value * value;
  }
  float scale = rsqrtf(sumsq / hc_dim + eps);
    mixes[idx] *= scale;
}

__global__ void deepseek_hc_scale_mixes_block_kernel(
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ mixes,
    float *__restrict__ rms_scales,
    int seq_len,
    int hc_dim,
    int mix_hc,
    float eps) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= seq_len) return;

  extern __shared__ float scratch[];
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int vec_end = hc_dim / 4;
  for (int vec = tid; vec < vec_end; vec += blockDim.x) {
    int base = token * hc_dim + vec * 4;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      float value = __bfloat162float(x[base + lane]);
      float square = __fmul_rn(value, value);
      sums[lane] = __fadd_rn(sums[lane], square);
    }
  }
  float sumsq = sums[0];
  sumsq = __fadd_rn(sumsq, sums[1]);
  sumsq = __fadd_rn(sumsq, sums[2]);
  sumsq = __fadd_rn(sumsq, sums[3]);
  if (tid == 0) {
    for (int k = vec_end * 4; k < hc_dim; ++k) {
      float value = __bfloat162float(x[token * hc_dim + k]);
      float square = __fmul_rn(value, value);
      sumsq = __fadd_rn(sumsq, square);
    }
  }
  scratch[tid] = sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] = __fadd_rn(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }

  float mean = __fmul_rn(scratch[0], 1.0f / static_cast<float>(hc_dim));
  float scale = rsqrtf(__fadd_rn(mean, eps));
  if (tid == 0 && rms_scales != nullptr) {
    rms_scales[token] = scale;
  }
  for (int mix = tid; mix < mix_hc; mix += blockDim.x) {
    mixes[token * mix_hc + mix] *= scale;
  }
}

__device__ __forceinline__ float deepseek_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

extern "C" int deepseek_tilelang_hc_split_sinkhorn_hc4_i20(
    const float* mixes,
    const float* hc_scale,
    const float* hc_base,
    float* pre,
    float* post,
    float* comb,
    int n,
    cudaStream_t stream);

__global__ void deepseek_hc_pre_output_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ pre,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int hc,
    int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * dim;
  if (idx >= total) return;
  int dim_idx = idx % dim;
  int token = idx / dim;
  float sum = 0.0f;
  for (int h = 0; h < hc; ++h) {
    sum += pre[token * hc + h] * __bfloat162float(x[(token * hc + h) * dim + dim_idx]);
  }
  out[idx] = __float2bfloat16(sum);
}

__global__ void deepseek_hc_head_pre_kernel(
    const float *__restrict__ mixes,
    const float *__restrict__ hc_scale,
    const float *__restrict__ hc_base,
    float *__restrict__ pre,
    int seq_len,
    int hc,
    float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hc;
  if (idx >= total) return;
  int h = idx % hc;
  pre[idx] = deepseek_sigmoid(mixes[idx] * hc_scale[0] + hc_base[h]) + eps;
}

__global__ void deepseek_hc_post_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const float *__restrict__ post,
    const float *__restrict__ comb,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int hc,
    int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hc * dim;
  if (idx >= total) return;
  int dim_idx = idx % dim;
  int h_out = (idx / dim) % hc;
  int token = idx / (hc * dim);
  float residual_sum = 0.0f;
  if (hc == 4) {
    float term0 = __fmul_rn(
        comb[(token * hc + 0) * hc + h_out],
        __bfloat162float(residual[(token * hc + 0) * dim + dim_idx]));
    float term1 = __fmul_rn(
        comb[(token * hc + 1) * hc + h_out],
        __bfloat162float(residual[(token * hc + 1) * dim + dim_idx]));
    float term2 = __fmul_rn(
        comb[(token * hc + 2) * hc + h_out],
        __bfloat162float(residual[(token * hc + 2) * dim + dim_idx]));
    float term3 = __fmul_rn(
        comb[(token * hc + 3) * hc + h_out],
        __bfloat162float(residual[(token * hc + 3) * dim + dim_idx]));
    residual_sum = __fadd_rn(__fadd_rn(__fadd_rn(term0, term1), term2), term3);
  } else {
    for (int h_in = 0; h_in < hc; ++h_in) {
      float term = __fmul_rn(
          comb[(token * hc + h_in) * hc + h_out],
          __bfloat162float(residual[(token * hc + h_in) * dim + dim_idx]));
      residual_sum = __fadd_rn(residual_sum, term);
    }
  }
  float post_term =
      __fmul_rn(post[token * hc + h_out], __bfloat162float(x[token * dim + dim_idx]));
  float sum = __fadd_rn(post_term, residual_sum);
  out[idx] = __float2bfloat16(sum);
}

__global__ void deepseek_last_token_bf16_logits_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    float *__restrict__ out,
    int seq_len,
    int dim,
    int vocab_size) {
  int vocab = blockIdx.x;
  int tid = threadIdx.x;
  if (vocab >= vocab_size || seq_len <= 0) return;

  extern __shared__ float scratch[];
  int token_base = (seq_len - 1) * dim;
  float partial = 0.0f;
  for (int k = tid; k < dim; k += blockDim.x) {
    partial += __bfloat162float(x[token_base + k]) *
               __bfloat162float(weight[vocab * dim + k]);
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
    out[vocab] = scratch[0];
  }
}

extern "C" {

cudaError_t deepseek_hc_expand_cuda(
    const __nv_bfloat16 *x,
    __nv_bfloat16 *out,
    int seq_len,
    int hc,
    int dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = seq_len * hc * dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_hc_expand_kernel<<<blocks, threads, 0, stream>>>(x, out, seq_len, hc, dim);
  return cudaGetLastError();
}

cudaError_t deepseek_hc_mixes_cuda(
    const __nv_bfloat16 *x,
    const float *hc_fn,
    float *mixes,
    float *raw_mixes,
    float *rms_scales,
    int seq_len,
    int hc,
    int dim,
    int mix_hc,
    float eps,
  cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int scale_threads = 512;
  int hc_dim = hc * dim;
  DeepseekHcScratch* scratch_ptr = nullptr;
  cudaError_t cuda_status = deepseek_hc_scratch_for_device(&scratch_ptr);
  if (cuda_status != cudaSuccess) return cuda_status;
  DeepseekHcScratch& scratch = *scratch_ptr;
  std::lock_guard<std::mutex> lock(scratch.mutex);
  cuda_status = deepseek_ensure_hc_f32_scratch(
      &scratch.x_f32, &scratch.x_elems, (size_t)seq_len * hc_dim);
  if (cuda_status != cudaSuccess) return cuda_status;

  int total = seq_len * hc_dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_hc_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(x, scratch.x_f32, total);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return cuda_status;
  }

  cuda_status = deepseek_ensure_hc_cublas_handle(scratch);
  if (cuda_status != cudaSuccess) return cuda_status;
  cublasStatus_t status = cublasSetStream(scratch.handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (seq_len == 1) {
    status = cublasSgemv(
        scratch.handle,
        CUBLAS_OP_T,
        hc_dim,
        mix_hc,
        &alpha,
        hc_fn,
        hc_dim,
        scratch.x_f32,
        1,
        &beta,
        mixes,
        1);
  } else {
    status = cublasGemmEx(
        scratch.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        mix_hc,
        seq_len,
        hc_dim,
        &alpha,
        hc_fn,
        CUDA_R_32F,
        hc_dim,
        scratch.x_f32,
        CUDA_R_32F,
        hc_dim,
        &beta,
        mixes,
        CUDA_R_32F,
        mix_hc,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }
  if (raw_mixes != nullptr) {
    cuda_status = cudaMemcpyAsync(
        raw_mixes,
        mixes,
        sizeof(float) * seq_len * mix_hc,
        cudaMemcpyDeviceToDevice,
        stream);
    if (cuda_status != cudaSuccess) {
      return cuda_status;
    }
  }

  deepseek_hc_scale_mixes_block_kernel<<<seq_len, scale_threads, scale_threads * sizeof(float), stream>>>(
      x, mixes, rms_scales, seq_len, hc_dim, mix_hc, eps);
  cuda_status = cudaGetLastError();
  return cuda_status;
}

cudaError_t deepseek_hc_split_sinkhorn_cuda(
    const float *mixes,
    const float *hc_scale,
    const float *hc_base,
    float *pre,
    float *post,
    float *comb,
    int seq_len,
    int hc,
    int sinkhorn_iters,
    float eps,
    cudaStream_t stream) {
  if (hc != 4 || sinkhorn_iters != 20 || fabsf(eps - 1.0e-6f) > 1.0e-12f) {
    return cudaErrorInvalidValue;
  }
  cudaError_t err = static_cast<cudaError_t>(deepseek_tilelang_hc_split_sinkhorn_hc4_i20(
      mixes, hc_scale, hc_base, pre, post, comb, seq_len, stream));
  return err;
}

cudaError_t deepseek_hc_pre_output_cuda(
    const __nv_bfloat16 *x,
    const float *pre,
    __nv_bfloat16 *out,
    int seq_len,
    int hc,
    int dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = seq_len * dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_hc_pre_output_kernel<<<blocks, threads, 0, stream>>>(x, pre, out, seq_len, hc, dim);
  return cudaGetLastError();
}

cudaError_t deepseek_hc_head_pre_cuda(
    const float *mixes,
    const float *hc_scale,
    const float *hc_base,
    float *pre,
    int seq_len,
    int hc,
    float eps,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = seq_len * hc;
  int blocks = (total + threads - 1) / threads;
  deepseek_hc_head_pre_kernel<<<blocks, threads, 0, stream>>>(
      mixes, hc_scale, hc_base, pre, seq_len, hc, eps);
  return cudaGetLastError();
}

cudaError_t deepseek_hc_post_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *residual,
    const float *post,
    const float *comb,
    __nv_bfloat16 *out,
    int seq_len,
    int hc,
    int dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  int total = seq_len * hc * dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_hc_post_kernel<<<blocks, threads, 0, stream>>>(
      x, residual, post, comb, out, seq_len, hc, dim);
  return cudaGetLastError();
}

cudaError_t deepseek_last_token_bf16_logits_cuda(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,
    float *out,
    int seq_len,
    int dim,
    int vocab_size,
    cudaStream_t stream) {
  constexpr int threads = 256;
  float *x_f32 = nullptr;
  float *weight_f32 = nullptr;
  cudaError_t cuda_status = cudaMalloc(&x_f32, sizeof(float) * dim);
  if (cuda_status != cudaSuccess) return cuda_status;
  cuda_status = cudaMalloc(&weight_f32, sizeof(float) * vocab_size * dim);
  if (cuda_status != cudaSuccess) {
    cudaFree(x_f32);
    return cuda_status;
  }

  const __nv_bfloat16 *last_x = x + (seq_len - 1) * dim;
  int x_blocks = (dim + threads - 1) / threads;
  deepseek_hc_bf16_to_f32_kernel<<<x_blocks, threads, 0, stream>>>(last_x, x_f32, dim);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cudaFree(weight_f32);
    cudaFree(x_f32);
    return cuda_status;
  }
  int weight_total = vocab_size * dim;
  int weight_blocks = (weight_total + threads - 1) / threads;
  deepseek_hc_bf16_to_f32_kernel<<<weight_blocks, threads, 0, stream>>>(
      weight, weight_f32, weight_total);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cudaFree(weight_f32);
    cudaFree(x_f32);
    return cuda_status;
  }

  cublasHandle_t handle = nullptr;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFree(weight_f32);
    cudaFree(x_f32);
    return cudaErrorUnknown;
  }
  status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    cudaFree(weight_f32);
    cudaFree(x_f32);
    return cudaErrorUnknown;
  }
  status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    cudaFree(weight_f32);
    cudaFree(x_f32);
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasSgemv(
      handle,
      CUBLAS_OP_T,
      dim,
      vocab_size,
      &alpha,
      weight_f32,
      dim,
      x_f32,
      1,
      &beta,
      out,
      1);
  cublasDestroy(handle);
  cudaFree(weight_f32);
  cudaFree(x_f32);
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
  return cudaSuccess;
}

}  // extern "C"
