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

// Grid-strided over rows: the row grid is capped (kMaxRowBlocks) and each
// block loops rows with stride gridDim.x up to the effective end. At the MoE
// recv capacity (2080 rows x 48 groups) a one-block-per-(row,group) grid is
// ~100k tiny blocks whose SCHEDULING alone costs ~60 us — far more than the
// actual quant work — and a device-bound early-return does not help because
// retired blocks are still scheduled. The device-side `row_bound` (the
// grouped-GEMM aligned segment end) instead bounds the loop, so the work AND
// the scheduling scale with the real row count while the launch shape stays
// fixed (CUDA-graph stable). Per-row math is unchanged (bit-identical).
//
// kMasked redirects the writes for the DeepGEMM masked grouped layout: the
// loop space stays the aligned recv rows, row_map[row] gives the masked slot
// (g*masked_cap + r_local; -1 on alignment-gap rows, which are skipped), the
// value goes to the fixed-stride masked row and the scale to the mn-major
// TMA layout [g, scale_cols, masked_cap] the GEMM's SFA descriptor reads —
// no separate scale-relayout kernel.
template <bool kMasked>
__global__ void fp8_per_token_group_quant_bf16_k128_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output, float* __restrict__ scales, int rows,
    int hidden_dim, const long long* __restrict__ row_bound,
    const int* __restrict__ row_map, int masked_cap) {
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  const int group_start = group * kGroupSize;
  const int col = group_start + tid;
  const int scale_cols = hidden_dim / kGroupSize;
  int end = rows;
  if (row_bound != nullptr) {
    const long long b = *row_bound;
    if (b < end) end = static_cast<int>(b < 0 ? 0 : b);
  }

  __shared__ float shared[kGroupSize];
  for (int row = blockIdx.x; row < end; row += gridDim.x) {
    int out_row = row;
    if constexpr (kMasked) {
      out_row = row_map[row];
      if (out_row < 0) continue;
    }
    float value = 0.0f;
    if (col < hidden_dim) {
      value = __bfloat162float(input[(size_t)row * hidden_dim + col]);
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
      if constexpr (kMasked) {
        const int g = out_row / masked_cap;
        const int r_local = out_row % masked_cap;
        scales[((size_t)g * scale_cols + group) * masked_cap + r_local] =
            shared[0];
      } else {
        scales[(size_t)row * scale_cols + group] = shared[0];
      }
    }
    __syncthreads();

    if (col < hidden_dim) {
      output[(size_t)out_row * hidden_dim + col] = quantize_e4m3(value, shared[0]);
    }
    __syncthreads();
  }
}

// Grid-strided over aligned receive rows. The gate|up input rows are already
// in the masked layout written by W13; route weights stay indexed by aligned
// receive row.
__global__ void silu_and_mul_per_token_group_quant_bf16_k128_masked_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ topk_weights, unsigned char* __restrict__ output,
    float* __restrict__ scales, int rows, int hidden_size,
    const long long* __restrict__ row_bound,
    const int* __restrict__ row_map, int masked_cap) {
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  const int group_start = group * kGroupSize;
  const int col = group_start + tid;
  const int input_stride = hidden_size * 2;
  const int scale_cols = hidden_size / kGroupSize;
  int end = rows;
  if (row_bound != nullptr) {
    const long long b = *row_bound;
    if (b < end) end = static_cast<int>(b < 0 ? 0 : b);
  }

  __shared__ float shared[kGroupSize];
  for (int row = blockIdx.x; row < end; row += gridDim.x) {
    const int data_row = row_map[row];
    if (data_row < 0) continue;
    float activated = 0.0f;
    if (col < hidden_size) {
      const __nv_bfloat16* token_gate =
          input + (size_t)data_row * input_stride + group_start;
      const __nv_bfloat16* token_up = token_gate + hidden_size;
      float gate = __bfloat162float(token_gate[tid]);
      float up = __bfloat162float(token_up[tid]);
      float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
      const float route_weight = __ldg(topk_weights + row);
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
      const int g = data_row / masked_cap;
      const int r_local = data_row % masked_cap;
      scales[((size_t)g * scale_cols + group) * masked_cap + r_local] = shared[0];
    }
    __syncthreads();

    if (col < hidden_size) {
      output[(size_t)data_row * hidden_size + col] =
          quantize_e4m3(activated, shared[0]);
    }
    __syncthreads();
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

// Row-grid cap for the grid-strided quant kernels: enough blocks to fill the
// SMs at 128 threads/block, small enough that a capacity-sized (2080-row)
// launch does not pay ~100k block-schedules for ~400 real rows.
constexpr int kMaxRowBlocks = 256;
int row_grid(int rows) { return rows < kMaxRowBlocks ? rows : kMaxRowBlocks; }

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

  dim3 grid(row_grid(rows), hidden_dim / kGroupSize, 1);
  fp8_per_token_group_quant_bf16_k128_kernel<false>
      <<<grid, kGroupSize, 0, stream>>>(input, output, scales, rows,
                                        hidden_dim, nullptr, nullptr, 0);
  return consume_last_cuda_error();
}

CUresult glm52_fp8_per_token_group_quant_bf16_masked_cuda(
    const __nv_bfloat16* input, unsigned char* output, float* scales, int rows,
    int hidden_dim, int group_size, const long long* row_bound,
    const int* row_map, int masked_cap, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      row_bound == nullptr || row_map == nullptr || masked_cap <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_dim, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(row_grid(rows), hidden_dim / kGroupSize, 1);
  fp8_per_token_group_quant_bf16_k128_kernel<true>
      <<<grid, kGroupSize, 0, stream>>>(input, output, scales, rows,
                                        hidden_dim, row_bound, row_map,
                                        masked_cap);
  return consume_last_cuda_error();
}

CUresult glm52_silu_and_mul_weighted_per_token_group_quant_bf16_masked_cuda(
    const __nv_bfloat16* input, const float* topk_weights,
    unsigned char* output, float* scales, int rows, int hidden_size,
    int group_size, const long long* row_bound, const int* row_map,
    int masked_cap, cudaStream_t stream) {
  if (input == nullptr || topk_weights == nullptr || output == nullptr ||
      scales == nullptr || row_bound == nullptr || row_map == nullptr ||
      masked_cap <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_quant_shape(rows, hidden_size, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(row_grid(rows), hidden_size / kGroupSize, 1);
  silu_and_mul_per_token_group_quant_bf16_k128_masked_kernel
      <<<grid, kGroupSize, 0, stream>>>(input, topk_weights, output, scales,
                                        rows, hidden_size, row_bound, row_map,
                                        masked_cap);
  return consume_last_cuda_error();
}

}  // extern "C"
