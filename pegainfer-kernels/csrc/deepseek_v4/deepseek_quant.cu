#include "deepseek_common.cuh"

// task #57 (#58 root-cause): the CUTLASS 3.x grouped block-scaled GEMM
// machinery used by `deepseek_moe_mxfp4_grouped_mixed_gemm_cuda` below MUST
// be included before `<flashinfer/gemm/gemm_groupwise_sm120.cuh>`. The
// FlashInfer groupwise SM120 header transitively pulls a CUTLASS state that
// makes the grouped mainloop instantiate `fp4_shift_B` with a const Tensor
// where the lambda needs `RegisterTypeB&` (instantiation tail
// `SM120_16x8x32_TN_VS<e4m3, e2m1, f32, ue8m0, VS=32>`). Reversing the order
// is a pure build-compat fix; no runtime / type / API change.
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

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

// task #57: Expand a per-128-element UE8M0 activation scale tensor to its
// per-32-element form by replicating each input byte 4 times in place along
// the scale-column axis. Required because the existing `act_fn` (TileLang
// AOT) emits 128-block scales while the CUTLASS mixed MX FP8 x FP4 grouped
// GEMM consumes 32-block scales. The transform is mathematically lossless:
// applying the same scalar `s` to each 32-element subblock recovers the
// original 128-block dequant. Input and output buffers must not alias.
__global__ void deepseek_mxfp4_scale_reformat_128_to_32_kernel(
    const unsigned char* __restrict__ scale_in,    // [rows, k_div_128]
    unsigned char* __restrict__ scale_out,         // [rows, 4 * k_div_128]
    int rows,
    int k_div_128) {
  int col_in = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (col_in >= k_div_128 || row >= rows) return;
  unsigned char s = scale_in[row * k_div_128 + col_in];
  int k_div_32 = k_div_128 * 4;
  int col_out_base = col_in * 4;
  unsigned char* row_out = scale_out + row * k_div_32 + col_out_base;
  row_out[0] = s;
  row_out[1] = s;
  row_out[2] = s;
  row_out[3] = s;
}

static cudaError_t deepseek_mxfp4_scale_reformat_128_to_32(
    const unsigned char* scale_in,
    unsigned char* scale_out,
    int rows,
    int in_dim,
    cudaStream_t stream) {
  if (rows <= 0 || in_dim <= 0 || (in_dim % 128) != 0) {
    return cudaErrorInvalidValue;
  }
  if (scale_in == nullptr || scale_out == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }
  int k_div_128 = in_dim / 128;
  constexpr int threads = 128;
  dim3 grid((k_div_128 + threads - 1) / threads, rows);
  deepseek_mxfp4_scale_reformat_128_to_32_kernel<<<grid, threads, 0, stream>>>(
      scale_in, scale_out, rows, k_div_128);
  return cudaGetLastError();
}

// task #57: CUTLASS SM120 grouped mixed-MX GEMM specialization.
// A is MX-FP8 (E4M3 elements + UE8M0 scale per 32-element block),
// B is MX-FP4 (E2M1 elements + UE8M0 scale per 32-element block),
// D is bf16 (no block-scale output).
// All experts share in_dim/out_dim; only rows-per-expert vary, so the GEMM
// runs in `kGrouped` mode with a single launcher.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
namespace deepseek_mxfp4_grouped_mixed {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag*, AlignmentC,
    ElementD, LayoutDTag*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag*, AlignmentA,
    ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using ElementAData = typename ElementA::DataType;          // float_e4m3_t
using ElementBData = typename ElementB::DataType;          // float_e2m1_t
using ElementSF    = typename ElementA::ScaleFactorType;   // float_ue8m0_t (shared by A & B)

using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

// Tiny launcher kernel that materializes per-group problem shapes, A/SFA
// pointer arrays, strides, and SF layouts from the device-side `expert_indptr`
// CSR. Output dim and input dim are uniform across groups; only the M extent
// changes per group.
__global__ void deepseek_mxfp4_grouped_mixed_setup_kernel(
    const int* __restrict__ expert_indptr,
    int local_experts,
    int in_dim,
    int out_dim,
    const ElementAData* __restrict__ a_base,           // FP8 act, [rows, in_dim] row-major
    const ElementSF* __restrict__ sfa_base,            // SFA, [rows, in_dim/32]
    const ElementBData* const* __restrict__ b_ptrs,    // per-expert B (FP4 weight) pointers
    const ElementSF* const* __restrict__ sfb_ptrs,    // per-expert SFB pointers
    ElementD* __restrict__ d_base,                     // bf16 out, [rows, out_dim]
    UnderlyingProblemShape* __restrict__ problems,
    const ElementAData** __restrict__ ptr_a,
    const ElementBData** __restrict__ ptr_b,
    const ElementSF** __restrict__ ptr_sfa,
    const ElementSF** __restrict__ ptr_sfb,
    ElementD** __restrict__ ptr_d,
    StrideA* __restrict__ stride_a,
    StrideB* __restrict__ stride_b,
    StrideD* __restrict__ stride_d,
    LayoutSFA* __restrict__ layout_sfa,
    LayoutSFB* __restrict__ layout_sfb) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= local_experts) return;
  int row_start = expert_indptr[g];
  int row_end = expert_indptr[g + 1];
  int m = row_end - row_start;
  int n = out_dim;
  int k = in_dim;
  problems[g] = cute::make_shape(m, n, k);

  ptr_a[g] = a_base + static_cast<long long>(row_start) * in_dim;
  // SFA layout assumes row-major scale tensor with `in_dim/32` columns.
  ptr_sfa[g] = sfa_base + static_cast<long long>(row_start) * (in_dim / 32);
  ptr_b[g] = b_ptrs[g];
  ptr_sfb[g] = sfb_ptrs[g];
  ptr_d[g] = d_base + static_cast<long long>(row_start) * out_dim;

  stride_a[g] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  stride_b[g] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  stride_d[g] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  layout_sfa[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  layout_sfb[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

}  // namespace deepseek_mxfp4_grouped_mixed

static cudaError_t deepseek_moe_mxfp4_grouped_mixed_gemm_cuda(
    const uint8_t* x_fp8,                  // [rows, in_dim] row-major FP8 E4M3
    const uint8_t* x_scale_32,             // [rows, in_dim/32] row-major UE8M0
    const uint8_t* const* w_e2m1,          // per-expert col-major FP4 weights (nibble-packed)
    const uint8_t* const* w_scale_32,      // per-expert UE8M0 weight scales
    const int* expert_indptr,              // device [local_experts+1] CSR
    __nv_bfloat16* d_bf16,                 // [rows, out_dim] row-major
    int rows,
    int in_dim,
    int out_dim,
    int local_experts,
    cudaStream_t stream) {
  using namespace deepseek_mxfp4_grouped_mixed;

  if (rows < 0 || in_dim <= 0 || out_dim <= 0 || local_experts <= 0)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if ((in_dim % 32) != 0) return cudaErrorInvalidValue;
  if (x_fp8 == nullptr || x_scale_32 == nullptr || w_e2m1 == nullptr ||
      w_scale_32 == nullptr || expert_indptr == nullptr || d_bf16 == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  // One contiguous device scratch for all per-group arrays.
  size_t bytes_problems   = sizeof(UnderlyingProblemShape) * local_experts;
  size_t bytes_ptr_a      = sizeof(const ElementAData*) * local_experts;
  size_t bytes_ptr_b      = sizeof(const ElementBData*) * local_experts;
  size_t bytes_ptr_sfa    = sizeof(const ElementSF*)    * local_experts;
  size_t bytes_ptr_sfb    = sizeof(const ElementSF*)    * local_experts;
  size_t bytes_ptr_d      = sizeof(ElementD*)           * local_experts;
  size_t bytes_stride_a   = sizeof(StrideA)   * local_experts;
  size_t bytes_stride_b   = sizeof(StrideB)   * local_experts;
  size_t bytes_stride_d   = sizeof(StrideD)   * local_experts;
  size_t bytes_layout_sfa = sizeof(LayoutSFA) * local_experts;
  size_t bytes_layout_sfb = sizeof(LayoutSFB) * local_experts;

  auto align = [](size_t s) -> size_t { return (s + 15ull) & ~15ull; };
  size_t total_bytes =
      align(bytes_problems) + align(bytes_ptr_a) + align(bytes_ptr_b) +
      align(bytes_ptr_sfa) + align(bytes_ptr_sfb) + align(bytes_ptr_d) +
      align(bytes_stride_a) + align(bytes_stride_b) + align(bytes_stride_d) +
      align(bytes_layout_sfa) + align(bytes_layout_sfb);

  uint8_t* scratch = nullptr;
  cudaError_t err = cudaMallocAsync(&scratch, total_bytes, stream);
  if (err != cudaSuccess) return err;

  uint8_t* cursor = scratch;
  auto carve = [&](size_t s) -> uint8_t* {
    uint8_t* p = cursor;
    cursor += align(s);
    return p;
  };

  auto* problems   = reinterpret_cast<UnderlyingProblemShape*>(carve(bytes_problems));
  auto* ptr_a      = reinterpret_cast<const ElementAData**>(carve(bytes_ptr_a));
  auto* ptr_b      = reinterpret_cast<const ElementBData**>(carve(bytes_ptr_b));
  auto* ptr_sfa    = reinterpret_cast<const ElementSF**>(carve(bytes_ptr_sfa));
  auto* ptr_sfb    = reinterpret_cast<const ElementSF**>(carve(bytes_ptr_sfb));
  auto* ptr_d      = reinterpret_cast<ElementD**>(carve(bytes_ptr_d));
  auto* stride_a   = reinterpret_cast<StrideA*>(carve(bytes_stride_a));
  auto* stride_b   = reinterpret_cast<StrideB*>(carve(bytes_stride_b));
  auto* stride_d   = reinterpret_cast<StrideD*>(carve(bytes_stride_d));
  auto* layout_sfa = reinterpret_cast<LayoutSFA*>(carve(bytes_layout_sfa));
  auto* layout_sfb = reinterpret_cast<LayoutSFB*>(carve(bytes_layout_sfb));

  {
    constexpr int threads = 32;
    int blocks = (local_experts + threads - 1) / threads;
    deepseek_mxfp4_grouped_mixed_setup_kernel<<<blocks, threads, 0, stream>>>(
        expert_indptr,
        local_experts,
        in_dim,
        out_dim,
        reinterpret_cast<const ElementAData*>(x_fp8),
        reinterpret_cast<const ElementSF*>(x_scale_32),
        reinterpret_cast<const ElementBData* const*>(w_e2m1),
        reinterpret_cast<const ElementSF* const*>(w_scale_32),
        reinterpret_cast<ElementD*>(d_bf16),
        problems,
        ptr_a, ptr_b, ptr_sfa, ptr_sfb, ptr_d,
        stride_a, stride_b, stride_d,
        layout_sfa, layout_sfb);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFreeAsync(scratch, stream);
      return err;
    }
  }

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {local_experts, problems, /*host_problem_shapes=*/nullptr},
      {ptr_a, stride_a, ptr_b, stride_b, ptr_sfa, layout_sfa, ptr_sfb, layout_sfb},
      {{/*alpha=*/1.0f, /*beta=*/0.0f},
       /*ptr_C=*/nullptr, /*stride_C=*/nullptr,
       /*ptr_D=*/ptr_d,  /*stride_D=*/stride_d}};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  uint8_t* gemm_workspace = nullptr;
  if (workspace_size > 0) {
    err = cudaMallocAsync(&gemm_workspace, workspace_size, stream);
    if (err != cudaSuccess) {
      cudaFreeAsync(scratch, stream);
      return err;
    }
  }

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    if (gemm_workspace) cudaFreeAsync(gemm_workspace, stream);
    cudaFreeAsync(scratch, stream);
    return cudaErrorNotSupported;
  }

  status = gemm.initialize(arguments, gemm_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    if (gemm_workspace) cudaFreeAsync(gemm_workspace, stream);
    cudaFreeAsync(scratch, stream);
    return cudaErrorUnknown;
  }

  status = gemm.run(stream);
  if (gemm_workspace) cudaFreeAsync(gemm_workspace, stream);
  cudaFreeAsync(scratch, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}
#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED

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

  // task #57: scratch holds two co-resident scale tensors:
  //   [0 .. rows*scale_cols_128)              -> per-128-block UE8M0 (act_fn output)
  //   [rows*scale_cols_128 .. rows*(128+32))  -> per-32-block UE8M0 (CUTLASS input)
  const int scale_cols_128 = (in_dim + 127) / 128;
  const int scale_cols_32  = (in_dim + 31) / 32;
  const size_t required_act_bytes = (size_t)rows * (size_t)in_dim;
  const size_t required_act_scale_bytes =
      (size_t)rows * (size_t)(scale_cols_128 + scale_cols_32);
  if (act_bytes < required_act_bytes || act_scale_bytes < required_act_scale_bytes) {
    return cudaErrorInvalidValue;
  }
  unsigned char* act_scale_128 = act_scale;
  unsigned char* act_scale_32  = act_scale + (size_t)rows * (size_t)scale_cols_128;

  cudaError_t err = static_cast<cudaError_t>(
      act_fn(x, act, act_scale_128, rows, stream));
  if (err != cudaSuccess) return err;

  err = deepseek_mxfp4_scale_reformat_128_to_32(
      act_scale_128, act_scale_32, rows, in_dim, stream);
  if (err != cudaSuccess) return err;

  // W13 is two grouped GEMMs sharing the FP8 activation tensor; the TileLang
  // path fused them, but for the CUTLASS mixed E4M3 x E2M1 + UE8M0 GroupedGemm
  // it is cleaner (and within budget) to run them as two back-to-back launches.
  err = deepseek_moe_mxfp4_grouped_mixed_gemm_cuda(
      act, act_scale_32,
      w1_weights, w1_scales,
      expert_indptr,
      gate_out,
      rows, in_dim, out_dim, local_experts, stream);
  if (err != cudaSuccess) return err;

  err = deepseek_moe_mxfp4_grouped_mixed_gemm_cuda(
      act, act_scale_32,
      w3_weights, w3_scales,
      expert_indptr,
      up_out,
      rows, in_dim, out_dim, local_experts, stream);
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
  (void)act_fn;
  (void)gemm_fn;

  // task #57: see W13 wrapper for the dual-scale workspace layout.
  const int scale_cols_128 = (in_dim + 127) / 128;
  const int scale_cols_32  = (in_dim + 31) / 32;
  const size_t required_act_bytes = (size_t)rows * (size_t)in_dim;
  const size_t required_act_scale_bytes =
      (size_t)rows * (size_t)(scale_cols_128 + scale_cols_32);
  if (act_bytes < required_act_bytes || act_scale_bytes < required_act_scale_bytes) {
    return cudaErrorInvalidValue;
  }
  unsigned char* act_scale_128 = act_scale;
  unsigned char* act_scale_32  = act_scale + (size_t)rows * (size_t)scale_cols_128;

  cudaError_t err = deepseek_swiglu_clamp_act_quant_k2048_cuda(
      gate, up, act, act_scale_128, rows, limit, stream);
  if (err != cudaSuccess) return err;

  err = deepseek_mxfp4_scale_reformat_128_to_32(
      act_scale_128, act_scale_32, rows, in_dim, stream);
  if (err != cudaSuccess) return err;

  err = deepseek_moe_mxfp4_grouped_mixed_gemm_cuda(
      act, act_scale_32,
      weights, scales,
      expert_indptr,
      out,
      rows, in_dim, out_dim, local_experts, stream);
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
