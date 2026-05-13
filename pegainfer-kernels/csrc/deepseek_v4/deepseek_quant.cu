#include "deepseek_common.cuh"

#include <flashinfer/gemm/gemm_groupwise_sm120.cuh>

#include <mutex>

namespace {

constexpr size_t kFlashInferFp8WorkspaceBytes = 32ull * 1024ull * 1024ull;
constexpr int kMaxQuantScratchDevices = 16;

struct DeepseekQuantScratch {
  unsigned char* act = nullptr;
  size_t act_bytes = 0;
  unsigned char* act_scale = nullptr;
  size_t act_scale_bytes = 0;
  std::mutex mutex;
};

DeepseekQuantScratch g_quant_scratch[kMaxQuantScratchDevices];

cudaError_t deepseek_ensure_byte_scratch(
    unsigned char** ptr,
    size_t* capacity,
    size_t required) {
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
  cudaError_t err = cudaMalloc(ptr, required);
  if (err != cudaSuccess) {
    return err;
  }
  *capacity = required;
  return cudaSuccess;
}

__global__ void deepseek_fill_f32_kernel(float* data, int n, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = value;
  }
}

constexpr int kSwigluQuantInterDim = 2048;
constexpr int kSwigluQuantGroupSize = 128;
constexpr int kSwigluQuantRowsPerBlock = 4;
constexpr int kSwigluQuantWarpSize = 32;
constexpr int kSwigluQuantScaleCols = kSwigluQuantInterDim / kSwigluQuantGroupSize;

__global__ void deepseek_swiglu_clamp_act_quant_k2048_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ out,
    unsigned char* __restrict__ scales,
    int rows,
    float limit) {
  const int row_block = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) / kSwigluQuantWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kSwigluQuantWarpSize;
  const int row = row_block * kSwigluQuantRowsPerBlock + warp;

  float activated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float amax = 0.0f;
  if (row < rows) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * kSwigluQuantWarpSize;
      const int idx = row * kSwigluQuantInterDim + group * kSwigluQuantGroupSize + col;
      float gate_value = __bfloat162float(gate[idx]);
      float up_value = __bfloat162float(up[idx]);
      if (limit > 0.0f) {
        gate_value = fminf(gate_value, limit);
        up_value = fminf(fmaxf(up_value, -limit), limit);
      }
      const float silu_gate = gate_value / (1.0f + expf(-gate_value));
      activated[item] = round_to_bf16_float(silu_gate * up_value);
      amax = fmaxf(amax, fabsf(activated[item]));
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
  }
  const float rounded_amax = fmaxf(__shfl_sync(0xffffffff, amax, 0), 1.0e-4f);
  const unsigned char scale_e8m0 = float_to_e8m0(rounded_amax / 448.0f);
  const float scale = e8m0_to_float(scale_e8m0);

  if (row < rows) {
    if (lane == 0) {
      scales[row * kSwigluQuantScaleCols + group] = scale_e8m0;
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * kSwigluQuantWarpSize;
      const float q = fminf(fmaxf(activated[item] / scale, -448.0f), 448.0f);
      out[row * kSwigluQuantInterDim + group * kSwigluQuantGroupSize + col] =
          float_to_fp8_e4m3(q);
    }
  }
}

static cudaError_t deepseek_swiglu_clamp_act_quant_k2048_cuda(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    unsigned char* out,
    unsigned char* scales,
    int rows,
    float limit,
    cudaStream_t stream) {
  if (rows < 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (gate == nullptr || up == nullptr || out == nullptr || scales == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  dim3 grid((rows + kSwigluQuantRowsPerBlock - 1) / kSwigluQuantRowsPerBlock,
            kSwigluQuantScaleCols,
            1);
  deepseek_swiglu_clamp_act_quant_k2048_kernel<<<grid, 128, 0, stream>>>(
      gate, up, out, scales, rows, limit);
  return cudaGetLastError();
}

__global__ void deepseek_fp8_quantize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ out,
    float* __restrict__ scales,
    int seq_len,
    int padded_seq_len,
    int hidden_dim,
    int scale_cols) {
  int group = blockIdx.x;
  int token = blockIdx.y;
  if (group >= scale_cols || token >= seq_len) return;

  int k_start = group * 128;
  int k_end = min(k_start + 128, hidden_dim);
  float amax = 0.0f;
  for (int k = k_start; k < k_end; ++k) {
    amax = fmaxf(amax, fabsf(__bfloat162float(x[token * hidden_dim + k])));
  }
  float scale_float = fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f);
  unsigned char scale_e8m0 = __nv_cvt_float_to_e8m0(scale_float, __NV_SATFINITE, cudaRoundPosInf);
  __nv_bfloat16_raw scale_raw = __nv_cvt_e8m0_to_bf16raw(scale_e8m0);
  __nv_bfloat16 scale_bf16(scale_raw);
  float scale = __bfloat162float(scale_bf16);
  scales[token * scale_cols + group] = scale;

  for (int k = k_start; k < k_end; ++k) {
    float value = __bfloat162float(x[token * hidden_dim + k]);
    out[token * hidden_dim + k] = __nv_cvt_float_to_fp8(value / scale, __NV_SATFINITE, __NV_E4M3);
  }
}

__global__ void deepseek_e8m0_scales_to_f32_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    __nv_bfloat16_raw raw = __nv_cvt_e8m0_to_bf16raw(input[idx]);
    __nv_bfloat16 value(raw);
    output[idx] = __bfloat162float(value);
  }
}

}  // namespace

__global__ void deepseek_fp8_linear_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const unsigned char *__restrict__ weight,
    const unsigned char *__restrict__ weight_scale,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int in_dim,
    int out_dim) {
  int out_col = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  if (out_col >= out_dim || token >= seq_len) return;

  extern __shared__ float scratch[];
  float sum = 0.0f;
  const int scale_cols = (in_dim + 127) / 128;
  const int weight_scale_row = out_col / 128;

  for (int group = 0; group < scale_cols; ++group) {
    int k_start = group * 128;
    int k_end = min(k_start + 128, in_dim);

    float amax = 0.0f;
    for (int k = k_start + tid; k < k_end; k += blockDim.x) {
      float v = fabsf(__bfloat162float(x[token * in_dim + k]));
      amax = fmaxf(amax, v);
    }
    scratch[tid] = amax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
      }
      __syncthreads();
    }

    float act_scale_float = fmaxf(scratch[0], 1.0e-4f) * (1.0f / 448.0f);
    unsigned char act_scale_e8m0 = float_to_e8m0(act_scale_float);
    float act_scale = e8m0_to_float(act_scale_e8m0);
    float w_scale = e8m0_to_float(weight_scale[weight_scale_row * scale_cols + group]);

    float partial = 0.0f;
    for (int k = k_start + tid; k < k_end; k += blockDim.x) {
      float x_value = __bfloat162float(x[token * in_dim + k]);
      float q_value = fp8_e4m3_to_float(float_to_fp8_e4m3(x_value / act_scale));
      float w_value = fp8_e4m3_to_float(weight[out_col * in_dim + k]);
      partial += q_value * w_value * act_scale * w_scale;
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
      sum += scratch[0];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[token * out_dim + out_col] = __float2bfloat16(sum);
  }
}

__global__ void deepseek_fp8_linear_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const unsigned char *__restrict__ weight,
    const unsigned char *__restrict__ weight_scale,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int in_dim,
    int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * out_dim;
  if (idx >= total) return;

  int token = idx / out_dim;
  int out_col = idx - token * out_dim;
  int scale_cols = (in_dim + 127) / 128;
  int weight_scale_row = out_col / 128;
  float sum = 0.0f;

  for (int group = 0; group < scale_cols; ++group) {
    int k_start = group * 128;
    int k_end = min(k_start + 128, in_dim);

    float amax = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
      float v = fabsf(__bfloat162float(x[token * in_dim + k]));
      amax = fmaxf(amax, v);
    }

    float act_scale_float = fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f);
    unsigned char act_scale_e8m0 = float_to_e8m0(act_scale_float);
    float act_scale = e8m0_to_float(act_scale_e8m0);
    float w_scale = e8m0_to_float(weight_scale[weight_scale_row * scale_cols + group]);

    for (int k = k_start; k < k_end; ++k) {
      float x_value = __bfloat162float(x[token * in_dim + k]);
      float q_value = fp8_e4m3_to_float(float_to_fp8_e4m3(x_value / act_scale));
      float w_value = fp8_e4m3_to_float(weight[out_col * in_dim + k]);
      sum += q_value * w_value * act_scale * w_scale;
    }
  }

  out[token * out_dim + out_col] = __float2bfloat16(sum);
}

__global__ void deepseek_fp8_act_quant_nope_bf16_kernel(
    __nv_bfloat16 *__restrict__ x,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int block_size) {
  int token = blockIdx.x;
  int head = blockIdx.y;
  int group = blockIdx.z;
  int tid = threadIdx.x;
  int nope_dim = head_dim - rotary_dim;
  if (token >= seq_len || head >= local_heads || group * block_size >= nope_dim) return;

  int start = group * block_size;
  int end = min(start + block_size, nope_dim);
  int base = token * local_heads * head_dim + head * head_dim;

  extern __shared__ float scratch[];
  float amax = 0.0f;
  for (int dim = start + tid; dim < end; dim += blockDim.x) {
    amax = fmaxf(amax, fabsf(__bfloat162float(x[base + dim])));
  }
  scratch[tid] = amax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }

  float scale_float = fmaxf(scratch[0], 1.0e-4f) * (1.0f / 448.0f);
  unsigned char scale_e8m0 = float_to_e8m0(scale_float);
  float scale = e8m0_to_float(scale_e8m0);
  for (int dim = start + tid; dim < end; dim += blockDim.x) {
    float value = __bfloat162float(x[base + dim]);
    float clamped = fminf(fmaxf(value / scale, -448.0f), 448.0f);
    float quantized = round_to_bf16_float(clamped) * scale;
    x[base + dim] = __float2bfloat16(quantized);
  }
}

__global__ void deepseek_bf16_copy_rows_kernel(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    int hidden_dim,
    int rows,
    int src_start_row,
    int dst_start_row) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = hidden_dim * rows;
  if (idx >= total) return;
  int row = idx / hidden_dim;
  int col = idx - row * hidden_dim;
  dst[(dst_start_row + row) * hidden_dim + col] =
      src[(src_start_row + row) * hidden_dim + col];
}

__global__ void deepseek_bf16_copy_rows_indexed_kernel(
    const __nv_bfloat16 *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    const int *__restrict__ src_rows,
    const int *__restrict__ dst_rows,
    int hidden_dim,
    int rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = hidden_dim * rows;
  if (idx >= total) return;
  int row = idx / hidden_dim;
  int col = idx - row * hidden_dim;
  int src_row = src_rows[row];
  int dst_row = dst_rows[row];
  if (src_row < 0 || dst_row < 0) return;
  dst[dst_row * hidden_dim + col] = src[src_row * hidden_dim + col];
}

extern "C" int deepseek_tilelang_act_quant_k4096(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_act_quant_k2048(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_act_quant_k1024(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp8_gemm_n512_k4096(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp8_gemm_n1024_k4096(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp8_gemm_n2048_k4096(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp8_w13_gemm_n2048_k4096(
    const void* a,
    const void* w1,
    const void* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* scales_w1,
    const void* scales_w3,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp8_gemm_n4096_k1024(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp8_gemm_n1024_k1024(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp8_gemm_n4096_k2048(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_gemm_n2048_k4096(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_gemm_n4096_k2048(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_gemm_n2048_k4096(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096(
    const void* a,
    const void* const* w1,
    const void* const* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* const* scales_w1,
    const void* const* scales_w3,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_gemm_n4096_k2048(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

using DeepseekTilelangActQuantFn = int (*)(
    const void*, void*, void*, int, cudaStream_t);
using DeepseekTilelangFp8GemmFn = int (*)(
    const void*, const void*, void*, const void*, const void*, int, cudaStream_t);
using DeepseekTilelangFp4GemmFn = int (*)(
    const void*, const void*, void*, const void*, const void*, int, cudaStream_t);
using DeepseekTilelangGroupedFp4GemmFn = int (*)(
    const void*, const void* const*, void*, const void*, const void* const*,
    const int*, int, int, cudaStream_t);

static bool deepseek_tilelang_fp8_linear_fns(
    int in_dim,
    int out_dim,
    DeepseekTilelangActQuantFn* act_fn,
    DeepseekTilelangFp8GemmFn* gemm_fn) {
  *act_fn = nullptr;
  *gemm_fn = nullptr;
  if (in_dim == 4096) {
    *act_fn = deepseek_tilelang_act_quant_k4096;
    if (out_dim == 512) {
      *gemm_fn = deepseek_tilelang_fp8_gemm_n512_k4096;
    } else if (out_dim == 1024) {
      *gemm_fn = deepseek_tilelang_fp8_gemm_n1024_k4096;
    } else if (out_dim == 2048) {
      *gemm_fn = deepseek_tilelang_fp8_gemm_n2048_k4096;
    }
  } else if (in_dim == 2048) {
    *act_fn = deepseek_tilelang_act_quant_k2048;
    if (out_dim == 4096) {
      *gemm_fn = deepseek_tilelang_fp8_gemm_n4096_k2048;
    }
  } else if (in_dim == 1024) {
    *act_fn = deepseek_tilelang_act_quant_k1024;
    if (out_dim == 4096) {
      *gemm_fn = deepseek_tilelang_fp8_gemm_n4096_k1024;
    } else if (out_dim == 1024) {
      *gemm_fn = deepseek_tilelang_fp8_gemm_n1024_k1024;
    }
  }
  return *act_fn != nullptr && *gemm_fn != nullptr;
}

static bool deepseek_tilelang_grouped_fp4_linear_fns(
    int in_dim,
    int out_dim,
    DeepseekTilelangActQuantFn* act_fn,
    DeepseekTilelangGroupedFp4GemmFn* gemm_fn) {
  *act_fn = nullptr;
  *gemm_fn = nullptr;
  if (in_dim == 4096 && out_dim == 2048) {
    *act_fn = deepseek_tilelang_act_quant_k4096;
    *gemm_fn = deepseek_tilelang_fp4_grouped_gemm_n2048_k4096;
  } else if (in_dim == 2048 && out_dim == 4096) {
    *act_fn = deepseek_tilelang_act_quant_k2048;
    *gemm_fn = deepseek_tilelang_fp4_grouped_gemm_n4096_k2048;
  }
  return *act_fn != nullptr && *gemm_fn != nullptr;
}

static cudaError_t deepseek_moe_fp4_grouped_w1_w3_workspace_cuda(
    const __nv_bfloat16 *x,
    const unsigned char *const *w1_weights,
    const unsigned char *const *w1_scales,
    const unsigned char *const *w3_weights,
    const unsigned char *const *w3_scales,
    const int *expert_indptr,
    __nv_bfloat16 *gate_out,
    __nv_bfloat16 *up_out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int rows,
    int in_dim,
    int out_dim,
    int local_experts,
    cudaStream_t stream) {
  if (rows < 0 || in_dim <= 0 || out_dim <= 0 || local_experts <= 0) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) return cudaSuccess;
  if (x == nullptr || w1_weights == nullptr || w1_scales == nullptr ||
      w3_weights == nullptr || w3_scales == nullptr || expert_indptr == nullptr ||
      gate_out == nullptr || up_out == nullptr || act == nullptr || act_scale == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  DeepseekTilelangActQuantFn act_fn = nullptr;
  DeepseekTilelangGroupedFp4GemmFn gemm_fn = nullptr;
  if (!deepseek_tilelang_grouped_fp4_linear_fns(in_dim, out_dim, &act_fn, &gemm_fn)) {
    return cudaErrorNotSupported;
  }
  if (in_dim != 4096 || out_dim != 2048) {
    return cudaErrorNotSupported;
  }

  const int scale_cols = (in_dim + 127) / 128;
  const size_t required_act_bytes = (size_t)rows * (size_t)in_dim;
  const size_t required_act_scale_bytes = (size_t)rows * (size_t)scale_cols;
  if (act_bytes < required_act_bytes || act_scale_bytes < required_act_scale_bytes) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = static_cast<cudaError_t>(
      act_fn(x, act, act_scale, rows, stream));
  if (err != cudaSuccess) return err;

  err = static_cast<cudaError_t>(deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096(
      act,
      reinterpret_cast<const void* const*>(w1_weights),
      reinterpret_cast<const void* const*>(w3_weights),
      gate_out,
      up_out,
      act_scale,
      reinterpret_cast<const void* const*>(w1_scales),
      reinterpret_cast<const void* const*>(w3_scales),
      expert_indptr,
      rows,
      local_experts,
      stream));
  return err == cudaSuccess ? cudaGetLastError() : err;
}

static cudaError_t deepseek_moe_fp4_grouped_w2_swiglu_workspace_cuda(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    const unsigned char *const *weights,
    const unsigned char *const *scales,
    const int *expert_indptr,
    __nv_bfloat16 *out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int rows,
    int in_dim,
    int out_dim,
    int local_experts,
    float limit,
    cudaStream_t stream) {
  if (rows < 0 || in_dim <= 0 || out_dim <= 0 || local_experts <= 0) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) return cudaSuccess;
  if (gate == nullptr || up == nullptr || weights == nullptr || scales == nullptr ||
      expert_indptr == nullptr || out == nullptr || act == nullptr || act_scale == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (in_dim != 2048 || out_dim != 4096) {
    return cudaErrorNotSupported;
  }

  DeepseekTilelangActQuantFn act_fn = nullptr;
  DeepseekTilelangGroupedFp4GemmFn gemm_fn = nullptr;
  if (!deepseek_tilelang_grouped_fp4_linear_fns(in_dim, out_dim, &act_fn, &gemm_fn)) {
    return cudaErrorNotSupported;
  }

  const int scale_cols = (in_dim + 127) / 128;
  const size_t required_act_bytes = (size_t)rows * (size_t)in_dim;
  const size_t required_act_scale_bytes = (size_t)rows * (size_t)scale_cols;
  if (act_bytes < required_act_bytes || act_scale_bytes < required_act_scale_bytes) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = deepseek_swiglu_clamp_act_quant_k2048_cuda(
      gate, up, act, act_scale, rows, limit, stream);
  if (err != cudaSuccess) return err;

  err = static_cast<cudaError_t>(gemm_fn(
      act,
      reinterpret_cast<const void* const*>(weights),
      out,
      act_scale,
      reinterpret_cast<const void* const*>(scales),
      expert_indptr,
      rows,
      local_experts,
      stream));
  return err == cudaSuccess ? cudaGetLastError() : err;
}

static cudaError_t deepseek_fp8_w1_w3_workspace_cuda(
    const __nv_bfloat16 *x,
    const unsigned char *w1_weight,
    const unsigned char *w1_scale,
    const unsigned char *w3_weight,
    const unsigned char *w3_scale,
    __nv_bfloat16 *gate_out,
    __nv_bfloat16 *up_out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int seq_len,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  if (seq_len < 0 || in_dim <= 0 || out_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) return cudaSuccess;
  if (x == nullptr || w1_weight == nullptr || w1_scale == nullptr ||
      w3_weight == nullptr || w3_scale == nullptr || gate_out == nullptr ||
      up_out == nullptr || act == nullptr || act_scale == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  DeepseekTilelangActQuantFn act_fn = nullptr;
  DeepseekTilelangFp8GemmFn gemm_fn = nullptr;
  if (!deepseek_tilelang_fp8_linear_fns(in_dim, out_dim, &act_fn, &gemm_fn)) {
    return cudaErrorNotSupported;
  }

  const int scale_cols = (in_dim + 127) / 128;
  const size_t required_act_bytes = (size_t)seq_len * (size_t)in_dim;
  const size_t required_act_scale_bytes = (size_t)seq_len * (size_t)scale_cols;
  if (act_bytes < required_act_bytes || act_scale_bytes < required_act_scale_bytes) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = static_cast<cudaError_t>(
      act_fn(x, act, act_scale, seq_len, stream));
  if (err != cudaSuccess) return err;

  if (in_dim == 4096 && out_dim == 2048) {
    err = static_cast<cudaError_t>(deepseek_tilelang_fp8_w13_gemm_n2048_k4096(
        act,
        w1_weight,
        w3_weight,
        gate_out,
        up_out,
        act_scale,
        w1_scale,
        w3_scale,
        seq_len,
        stream));
    return err == cudaSuccess ? cudaGetLastError() : err;
  }

  err = static_cast<cudaError_t>(
      gemm_fn(act, w1_weight, gate_out, act_scale, w1_scale, seq_len, stream));
  if (err != cudaSuccess) return err;

  err = static_cast<cudaError_t>(
      gemm_fn(act, w3_weight, up_out, act_scale, w3_scale, seq_len, stream));
  return err == cudaSuccess ? cudaGetLastError() : err;
}

static cudaError_t deepseek_fp8_w2_swiglu_workspace_cuda(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    const unsigned char *weight,
    const unsigned char *weight_scale,
    __nv_bfloat16 *out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int seq_len,
    int in_dim,
    int out_dim,
    float limit,
    cudaStream_t stream) {
  if (seq_len < 0 || in_dim <= 0 || out_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) return cudaSuccess;
  if (gate == nullptr || up == nullptr || weight == nullptr || weight_scale == nullptr ||
      out == nullptr || act == nullptr || act_scale == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  if (in_dim != 2048 || out_dim != 4096) {
    return cudaErrorNotSupported;
  }

  DeepseekTilelangActQuantFn act_fn = nullptr;
  DeepseekTilelangFp8GemmFn gemm_fn = nullptr;
  if (!deepseek_tilelang_fp8_linear_fns(in_dim, out_dim, &act_fn, &gemm_fn)) {
    return cudaErrorNotSupported;
  }

  const int scale_cols = (in_dim + 127) / 128;
  const size_t required_act_bytes = (size_t)seq_len * (size_t)in_dim;
  const size_t required_act_scale_bytes = (size_t)seq_len * (size_t)scale_cols;
  if (act_bytes < required_act_bytes || act_scale_bytes < required_act_scale_bytes) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = deepseek_swiglu_clamp_act_quant_k2048_cuda(
      gate, up, act, act_scale, seq_len, limit, stream);
  if (err != cudaSuccess) return err;

  err = static_cast<cudaError_t>(
      gemm_fn(act, weight, out, act_scale, weight_scale, seq_len, stream));
  return err == cudaSuccess ? cudaGetLastError() : err;
}

static cudaError_t deepseek_fp8_linear_tilelang_cuda(
    const __nv_bfloat16 *x,
    const unsigned char *weight,
    const unsigned char *weight_scale,
    __nv_bfloat16 *out,
    int seq_len,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  DeepseekTilelangActQuantFn act_fn = nullptr;
  DeepseekTilelangFp8GemmFn gemm_fn = nullptr;
  if (!deepseek_tilelang_fp8_linear_fns(in_dim, out_dim, &act_fn, &gemm_fn)) {
    return cudaErrorNotSupported;
  }
  const int scale_cols = (in_dim + 127) / 128;
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (device < 0 || device >= kMaxQuantScratchDevices) return cudaErrorInvalidDevice;

  DeepseekQuantScratch& scratch = g_quant_scratch[device];
  std::lock_guard<std::mutex> lock(scratch.mutex);
  err = deepseek_ensure_byte_scratch(
      &scratch.act, &scratch.act_bytes, (size_t)seq_len * in_dim);
  if (err != cudaSuccess) return err;
  err = deepseek_ensure_byte_scratch(
      &scratch.act_scale, &scratch.act_scale_bytes, (size_t)seq_len * scale_cols);
  if (err != cudaSuccess) return err;

  err = static_cast<cudaError_t>(
      act_fn(x, scratch.act, scratch.act_scale, seq_len, stream));
  if (err != cudaSuccess) return err;

  err = static_cast<cudaError_t>(
      gemm_fn(scratch.act, weight, out, scratch.act_scale, weight_scale, seq_len, stream));
  return err == cudaSuccess ? cudaGetLastError() : err;
}

extern "C" {

cudaError_t deepseek_fp8_linear_cuda(
    const __nv_bfloat16 *x,
    const unsigned char *weight,
    const unsigned char *weight_scale,
    __nv_bfloat16 *out,
    int seq_len,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  DeepseekTilelangActQuantFn tilelang_act_fn = nullptr;
  DeepseekTilelangFp8GemmFn tilelang_gemm_fn = nullptr;
  if (deepseek_tilelang_fp8_linear_fns(
          in_dim, out_dim, &tilelang_act_fn, &tilelang_gemm_fn)) {
    return deepseek_fp8_linear_tilelang_cuda(
        x, weight, weight_scale, out, seq_len, in_dim, out_dim, stream);
  }

  constexpr int threads = 128;
  int scale_cols = (in_dim + 127) / 128;
  int out_scale_rows = (out_dim + 127) / 128;
  int padded_seq_len = ((seq_len + 3) / 4) * 4;

  unsigned char* act = nullptr;
  float* act_scale = nullptr;
  float* weight_scale_f32 = nullptr;
  __nv_bfloat16* out_temp = nullptr;
  void* workspace = nullptr;

  cudaError_t err = cudaMalloc(&act, (size_t)padded_seq_len * in_dim);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&act_scale, (size_t)padded_seq_len * scale_cols * sizeof(float));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&weight_scale_f32, (size_t)out_scale_rows * scale_cols * sizeof(float));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&out_temp, (size_t)padded_seq_len * out_dim * sizeof(__nv_bfloat16));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&workspace, kFlashInferFp8WorkspaceBytes);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemsetAsync(act, 0, (size_t)padded_seq_len * in_dim, stream);
  if (err != cudaSuccess) goto cleanup;
  {
    int scale_total = padded_seq_len * scale_cols;
    int blocks = (scale_total + threads - 1) / threads;
    deepseek_fill_f32_kernel<<<blocks, threads, 0, stream>>>(act_scale, scale_total, 1.0f);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
  }
  {
    dim3 quant_grid(scale_cols, seq_len);
    deepseek_fp8_quantize_bf16_kernel<<<quant_grid, 1, 0, stream>>>(
        x, act, act_scale, seq_len, padded_seq_len, in_dim, scale_cols);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
  }
  {
    int scale_total = out_scale_rows * scale_cols;
    int blocks = (scale_total + threads - 1) / threads;
    deepseek_e8m0_scales_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        weight_scale, weight_scale_f32, scale_total);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
  }

  err = flashinfer::gemm::CutlassGroupwiseScaledGEMMSM120<
      1,
      128,
      128,
      true,
      cutlass::float_e4m3_t,
      cutlass::bfloat16_t>(
      workspace,
      kFlashInferFp8WorkspaceBytes,
      reinterpret_cast<cutlass::float_e4m3_t*>(act),
      reinterpret_cast<cutlass::float_e4m3_t*>(const_cast<unsigned char*>(weight)),
      act_scale,
      weight_scale_f32,
      reinterpret_cast<cutlass::bfloat16_t*>(out_temp),
      padded_seq_len,
      out_dim,
      in_dim,
      1,
      stream);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      out,
      out_temp,
      (size_t)seq_len * out_dim * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToDevice,
      stream);

cleanup:
  if (workspace) cudaFree(workspace);
  if (out_temp) cudaFree(out_temp);
  if (weight_scale_f32) cudaFree(weight_scale_f32);
  if (act_scale) cudaFree(act_scale);
  if (act) cudaFree(act);
  return err == cudaSuccess ? cudaGetLastError() : err;
}

cudaError_t deepseek_fp8_w1_w3_with_workspace_cuda(
    const __nv_bfloat16 *x,
    const unsigned char *w1_weight,
    const unsigned char *w1_scale,
    const unsigned char *w3_weight,
    const unsigned char *w3_scale,
    __nv_bfloat16 *gate_out,
    __nv_bfloat16 *up_out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int seq_len,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  return deepseek_fp8_w1_w3_workspace_cuda(
      x, w1_weight, w1_scale, w3_weight, w3_scale, gate_out, up_out,
      act, act_bytes, act_scale, act_scale_bytes, seq_len, in_dim, out_dim, stream);
}

cudaError_t deepseek_fp8_w2_swiglu_with_workspace_cuda(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    const unsigned char *weight,
    const unsigned char *weight_scale,
    __nv_bfloat16 *out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int seq_len,
    int in_dim,
    int out_dim,
    float limit,
    cudaStream_t stream) {
  return deepseek_fp8_w2_swiglu_workspace_cuda(
      gate, up, weight, weight_scale, out, act, act_bytes, act_scale,
      act_scale_bytes, seq_len, in_dim, out_dim, limit, stream);
}

cudaError_t deepseek_fp8_act_quant_nope_bf16_cuda(
    __nv_bfloat16 *x,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int block_size,
    cudaStream_t stream) {
  if (seq_len <= 0 || local_heads <= 0 || head_dim <= 0 ||
      rotary_dim < 0 || rotary_dim >= head_dim || block_size <= 0) {
    return cudaErrorInvalidValue;
  }
  int nope_dim = head_dim - rotary_dim;
  if (nope_dim % block_size != 0) return cudaErrorInvalidValue;
  constexpr int threads = 128;
  dim3 grid(seq_len, local_heads, nope_dim / block_size);
  size_t shared_bytes = threads * sizeof(float);
  deepseek_fp8_act_quant_nope_bf16_kernel<<<grid, threads, shared_bytes, stream>>>(
      x, seq_len, local_heads, head_dim, rotary_dim, block_size);
  return cudaGetLastError();
}

cudaError_t deepseek_bf16_copy_rows_cuda(
    const __nv_bfloat16 *src,
    __nv_bfloat16 *dst,
    int hidden_dim,
    int rows,
    int src_start_row,
    int dst_start_row,
    cudaStream_t stream) {
  if (hidden_dim <= 0 || rows < 0 || src_start_row < 0 || dst_start_row < 0) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) return cudaSuccess;
  constexpr int threads = 256;
  int total = hidden_dim * rows;
  int blocks = (total + threads - 1) / threads;
  deepseek_bf16_copy_rows_kernel<<<blocks, threads, 0, stream>>>(
      src, dst, hidden_dim, rows, src_start_row, dst_start_row);
  return cudaGetLastError();
}

cudaError_t deepseek_bf16_copy_rows_indexed_cuda(
    const __nv_bfloat16 *src,
    __nv_bfloat16 *dst,
    const int *src_rows,
    const int *dst_rows,
    int hidden_dim,
    int rows,
    cudaStream_t stream) {
  if (hidden_dim <= 0 || rows < 0 || src_rows == nullptr || dst_rows == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) return cudaSuccess;
  constexpr int threads = 256;
  int total = hidden_dim * rows;
  int blocks = (total + threads - 1) / threads;
  deepseek_bf16_copy_rows_indexed_kernel<<<blocks, threads, 0, stream>>>(
      src, dst, src_rows, dst_rows, hidden_dim, rows);
  return cudaGetLastError();
}

}  // extern "C"

__global__ void deepseek_fp4_linear_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const unsigned char *__restrict__ weight,
    const unsigned char *__restrict__ weight_scale,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int in_dim,
    int out_dim) {
  int out_col = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  if (out_col >= out_dim || token >= seq_len) return;

  extern __shared__ float scratch[];
  float sum = 0.0f;
  const int act_scale_cols = (in_dim + 127) / 128;
  const int weight_scale_cols = in_dim / 32;

  for (int group = 0; group < act_scale_cols; ++group) {
    int k_start = group * 128;
    int k_end = min(k_start + 128, in_dim);

    float amax = 0.0f;
    for (int k = k_start + tid; k < k_end; k += blockDim.x) {
      float v = fabsf(__bfloat162float(x[token * in_dim + k]));
      amax = fmaxf(amax, v);
    }
    scratch[tid] = amax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
      }
      __syncthreads();
    }

    float act_scale_float = fmaxf(scratch[0], 1.0e-4f) * (1.0f / 448.0f);
    unsigned char act_scale_e8m0 = float_to_e8m0(act_scale_float);
    float act_scale = e8m0_to_float(act_scale_e8m0);

    float partial = 0.0f;
    for (int k = k_start + tid; k < k_end; k += blockDim.x) {
      float x_value = __bfloat162float(x[token * in_dim + k]);
      float q_value = fp8_e4m3_to_float(float_to_fp8_e4m3(x_value / act_scale));
      unsigned char packed = weight[out_col * (in_dim / 2) + (k / 2)];
      unsigned char nibble = (k & 1) == 0 ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
      float w_value = fp4_e2m1_to_float(nibble);
      float w_scale = e8m0_to_float(weight_scale[out_col * weight_scale_cols + (k / 32)]);
      partial += q_value * w_value * act_scale * w_scale;
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
      sum += scratch[0];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[token * out_dim + out_col] = __float2bfloat16(sum);
  }
}

__global__ void deepseek_fp4_linear_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const unsigned char *__restrict__ weight,
    const unsigned char *__restrict__ weight_scale,
    __nv_bfloat16 *__restrict__ out,
    int seq_len,
    int in_dim,
    int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * out_dim;
  if (idx >= total) return;

  int token = idx / out_dim;
  int out_col = idx - token * out_dim;
  const int act_scale_cols = (in_dim + 127) / 128;
  const int weight_scale_cols = in_dim / 32;
  float sum = 0.0f;

  for (int group = 0; group < act_scale_cols; ++group) {
    int k_start = group * 128;
    int k_end = min(k_start + 128, in_dim);
    float amax = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
      float v = fabsf(__bfloat162float(x[token * in_dim + k]));
      amax = fmaxf(amax, v);
    }

    float act_scale_float = fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f);
    unsigned char act_scale_e8m0 = float_to_e8m0(act_scale_float);
    float act_scale = e8m0_to_float(act_scale_e8m0);

    for (int k = k_start; k < k_end; ++k) {
      float x_value = __bfloat162float(x[token * in_dim + k]);
      float q_value = fp8_e4m3_to_float(float_to_fp8_e4m3(x_value / act_scale));
      unsigned char packed = weight[out_col * (in_dim / 2) + (k / 2)];
      unsigned char nibble = (k & 1) == 0 ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
      float w_value = fp4_e2m1_to_float(nibble);
      float w_scale = e8m0_to_float(weight_scale[out_col * weight_scale_cols + (k / 32)]);
      sum += q_value * w_value * act_scale * w_scale;
    }
  }

  out[token * out_dim + out_col] = __float2bfloat16(sum);
}

extern "C" {

cudaError_t deepseek_fp4_linear_cuda(
    const __nv_bfloat16 *x,
    const unsigned char *weight,
    const unsigned char *weight_scale,
    __nv_bfloat16 *out,
    int seq_len,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  DeepseekTilelangActQuantFn act_fn = nullptr;
  DeepseekTilelangFp4GemmFn gemm_fn = nullptr;
  if (in_dim == 4096 && out_dim == 2048) {
    act_fn = deepseek_tilelang_act_quant_k4096;
    gemm_fn = deepseek_tilelang_fp4_gemm_n2048_k4096;
  } else if (in_dim == 2048 && out_dim == 4096) {
    act_fn = deepseek_tilelang_act_quant_k2048;
    gemm_fn = deepseek_tilelang_fp4_gemm_n4096_k2048;
  }
  if (act_fn != nullptr && gemm_fn != nullptr) {
    const int scale_cols = (in_dim + 127) / 128;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    if (device < 0 || device >= kMaxQuantScratchDevices) return cudaErrorInvalidDevice;

    DeepseekQuantScratch& scratch = g_quant_scratch[device];
    std::lock_guard<std::mutex> lock(scratch.mutex);
    err = deepseek_ensure_byte_scratch(
        &scratch.act, &scratch.act_bytes, (size_t)seq_len * in_dim);
    if (err != cudaSuccess) return err;
    err = deepseek_ensure_byte_scratch(
        &scratch.act_scale, &scratch.act_scale_bytes, (size_t)seq_len * scale_cols);
    if (err != cudaSuccess) return err;

    err = static_cast<cudaError_t>(
        act_fn(x, scratch.act, scratch.act_scale, seq_len, stream));
    if (err != cudaSuccess) return err;

    err = static_cast<cudaError_t>(
        gemm_fn(scratch.act, weight, out, scratch.act_scale, weight_scale, seq_len, stream));
    return err == cudaSuccess ? cudaGetLastError() : err;
  }

  constexpr int threads = 256;
  int total = seq_len * out_dim;
  int blocks = (total + threads - 1) / threads;
  deepseek_fp4_linear_serial_kernel<<<blocks, threads, 0, stream>>>(
      x, weight, weight_scale, out, seq_len, in_dim, out_dim);
  return cudaGetLastError();
}

cudaError_t deepseek_moe_fp4_grouped_w1_w3_with_workspace_cuda(
    const __nv_bfloat16 *x,
    const unsigned char *const *w1_weights,
    const unsigned char *const *w1_scales,
    const unsigned char *const *w3_weights,
    const unsigned char *const *w3_scales,
    const int *expert_indptr,
    __nv_bfloat16 *gate_out,
    __nv_bfloat16 *up_out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int rows,
    int in_dim,
    int out_dim,
    int local_experts,
    cudaStream_t stream) {
  return deepseek_moe_fp4_grouped_w1_w3_workspace_cuda(
      x, w1_weights, w1_scales, w3_weights, w3_scales, expert_indptr,
      gate_out, up_out, act, act_bytes, act_scale, act_scale_bytes,
      rows, in_dim, out_dim, local_experts, stream);
}

cudaError_t deepseek_moe_fp4_grouped_w2_swiglu_with_workspace_cuda(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    const unsigned char *const *weights,
    const unsigned char *const *scales,
    const int *expert_indptr,
    __nv_bfloat16 *out,
    unsigned char *act,
    size_t act_bytes,
    unsigned char *act_scale,
    size_t act_scale_bytes,
    int rows,
    int in_dim,
    int out_dim,
    int local_experts,
    float limit,
    cudaStream_t stream) {
  return deepseek_moe_fp4_grouped_w2_swiglu_workspace_cuda(
      gate, up, weights, scales, expert_indptr, out,
      act, act_bytes, act_scale, act_scale_bytes,
      rows, in_dim, out_dim, local_experts, limit, stream);
}

}  // extern "C"
