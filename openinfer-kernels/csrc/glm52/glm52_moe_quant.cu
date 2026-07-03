#include "../common.cuh"

#include <cuda.h>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace {

constexpr int kGroupSize = 128;
constexpr float kFp8Min = -448.0f;
constexpr float kFp8Max = 448.0f;
constexpr float kPerTokenGroupQuantEps = 1.0e-10f;
constexpr float kMinSiluScale = 1.0f / (kFp8Max * 512.0f);

__device__ __forceinline__ unsigned char quantize_e4m3(float value,
                                                       float scale) {
  float q = fminf(fmaxf(value / scale, kFp8Min), kFp8Max);
  return __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

__global__ void fp8_per_token_group_quant_bf16_k128_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output, float* __restrict__ scales, int rows,
    int hidden_dim, const long long* __restrict__ row_bound) {
  const int row = blockIdx.x;
  // Optional device-side row bound (the grouped-GEMM aligned segment end):
  // the grid is capacity-sized for CUDA-graph pointer/shape stability, but
  // rows past the real aligned end are never read by any consumer — retire
  // their blocks immediately instead of quantizing garbage.
  if (row_bound != nullptr && row >= *row_bound) return;
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  const int group_start = group * kGroupSize;
  const int col = group_start + tid;
  const int scale_cols = hidden_dim / kGroupSize;

  __shared__ float shared[kGroupSize];
  float value = 0.0f;
  if (row < rows && col < hidden_dim) {
    value = __bfloat162float(input[row * hidden_dim + col]);
  }
  shared[tid] = fabsf(value);
  __syncthreads();

#pragma unroll
  for (int stride = kGroupSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    shared[0] = fmaxf(shared[0], kPerTokenGroupQuantEps) / kFp8Max;
    scales[row * scale_cols + group] = shared[0];
  }
  __syncthreads();

  if (row < rows && col < hidden_dim) {
    output[row * hidden_dim + col] = quantize_e4m3(value, shared[0]);
  }
}

__global__ void silu_and_mul_per_token_group_quant_bf16_k128_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ topk_weights, unsigned char* __restrict__ output,
    float* __restrict__ scales, int rows, int hidden_size,
    const long long* __restrict__ row_bound) {
  const int row = blockIdx.x;
  if (row_bound != nullptr && row >= *row_bound) return;  // see quant kernel note
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  const int group_start = group * kGroupSize;
  const int col = group_start + tid;
  const int input_stride = hidden_size * 2;
  const int scale_cols = hidden_size / kGroupSize;

  __shared__ float shared[kGroupSize];
  float activated = 0.0f;
  if (row < rows && col < hidden_size) {
    const __nv_bfloat16* token_gate = input + row * input_stride + group_start;
    const __nv_bfloat16* token_up = token_gate + hidden_size;
    float gate = __bfloat162float(token_gate[tid]);
    float up = __bfloat162float(token_up[tid]);
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    const float route_weight =
        topk_weights == nullptr ? 1.0f : __ldg(topk_weights + row);
    activated = gate * sigmoid_gate * up * route_weight;
  }
  shared[tid] = fabsf(activated);
  __syncthreads();

#pragma unroll
  for (int stride = kGroupSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    shared[0] = fmaxf(shared[0] / kFp8Max, kMinSiluScale);
    scales[row * scale_cols + group] = shared[0];
  }
  __syncthreads();

  if (row < rows && col < hidden_size) {
    output[row * hidden_size + col] = quantize_e4m3(activated, shared[0]);
  }
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

bool valid_quant_shape(int rows, int width, int group_size) {
  return rows > 0 && width > 0 && group_size == kGroupSize &&
         width % kGroupSize == 0;
}

}  // namespace

extern "C" {

CUresult glm52_fp8_per_token_group_quant_bf16_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_dim, int group_size, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_dim, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  dim3 grid(rows, hidden_dim / kGroupSize, 1);
  fp8_per_token_group_quant_bf16_k128_kernel<<<grid, kGroupSize, 0, stream>>>(
      input, output, scales, rows, hidden_dim, nullptr);
  return consume_last_cuda_error();
}

CUresult glm52_fp8_per_token_group_quant_bf16_bounded_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_dim, int group_size, const long long* row_bound,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      row_bound == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_dim, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(rows, hidden_dim / kGroupSize, 1);
  fp8_per_token_group_quant_bf16_k128_kernel<<<grid, kGroupSize, 0, stream>>>(
      input, output, scales, rows, hidden_dim, row_bound);
  return consume_last_cuda_error();
}

CUresult glm52_silu_and_mul_per_token_group_quant_bf16_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_size, int group_size, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_size, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  dim3 grid(rows, hidden_size / kGroupSize, 1);
  silu_and_mul_per_token_group_quant_bf16_k128_kernel<<<grid, kGroupSize, 0,
                                                        stream>>>(
      input, nullptr, output, scales, rows, hidden_size, nullptr);
  return consume_last_cuda_error();
}

CUresult glm52_silu_and_mul_weighted_per_token_group_quant_bf16_cuda(
    const __nv_bfloat16* input, const float* topk_weights,
    unsigned char* output, float* scales, int rows, int hidden_size,
    int group_size, cudaStream_t stream) {
  if (input == nullptr || topk_weights == nullptr || output == nullptr ||
      scales == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_size, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Mirrors DeepGEMM third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py:
  // y = silu(gate) * up * topk_weight, then per-token/per-channel FP8 quant.
  dim3 grid(rows, hidden_size / kGroupSize, 1);
  silu_and_mul_per_token_group_quant_bf16_k128_kernel<<<grid, kGroupSize, 0,
                                                        stream>>>(
      input, topk_weights, output, scales, rows, hidden_size, nullptr);
  return consume_last_cuda_error();
}

CUresult glm52_silu_and_mul_weighted_per_token_group_quant_bf16_bounded_cuda(
    const __nv_bfloat16* input, const float* topk_weights,
    unsigned char* output, float* scales, int rows, int hidden_size,
    int group_size, const long long* row_bound, cudaStream_t stream) {
  if (input == nullptr || topk_weights == nullptr || output == nullptr ||
      scales == nullptr || row_bound == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_size, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(rows, hidden_size / kGroupSize, 1);
  silu_and_mul_per_token_group_quant_bf16_k128_kernel<<<grid, kGroupSize, 0,
                                                        stream>>>(
      input, topk_weights, output, scales, rows, hidden_size, row_bound);
  return consume_last_cuda_error();
}

}  // extern "C"
