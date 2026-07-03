// FlashInfer-backed norm kernels.
//
// Provides the same extern "C" API surface as our hand-written norm.cu,
// but delegates to FlashInfer's header-only RMSNorm / FusedAddRMSNorm /
// GemmaRMSNorm / GemmaFusedAddRMSNorm templates.
//
// Semantic adapter for FusedAddRMSNorm:
//   Our API:       hidden += residual; out = norm(hidden, weight)
//   FlashInfer:    residual_arg += input_arg; input_arg = norm(residual_arg, weight)
//
//   To bridge the gap we memcpy residual → out, then call FlashInfer with
//   (input=out, residual=hidden). After the call:
//     hidden = hidden + out(=residual)   ← what we want
//     out    = norm(hidden)              ← what we want
//   The memcpy is ≤14 KB per row (hidden_size=3584 × 2 bytes) and negligible.

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <numeric>

#include <flashinfer/norm.cuh>

using DType = __nv_bfloat16;

namespace openinfer {
namespace norm {

// Exact-preserving variant for the decode pattern:
//   hidden = bf16(hidden + residual)
//   out = RMSNorm(hidden, weight)
//
// FlashInfer's FusedAddRMSNorm keeps the pre-BF16-round add value in shared
// memory for the RMS reduction. Kimi token correctness currently depends on
// the separate add kernel's BF16 rounding boundary, so this kernel mirrors the
// FlashInfer reduction/order but feeds it the rounded BF16 sum.
template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormRoundKernel(T* __restrict__ hidden,
                                           const T* __restrict__ residual,
                                           T* __restrict__ weight,
                                           T* __restrict__ out,
                                           const uint32_t d,
                                           const uint32_t stride_hidden,
                                           const uint32_t stride_residual,
                                           const uint32_t stride_out,
                                           float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + flashinfer::ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> hidden_vec;
    flashinfer::vec_t<T, VEC_SIZE> residual_vec;
    flashinfer::vec_t<float, VEC_SIZE> rounded_vec;
    hidden_vec.fill(0.f);
    residual_vec.fill(0.f);
    rounded_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      hidden_vec.load(hidden + bx * stride_hidden + elem);
      residual_vec.load(residual + bx * stride_residual + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      T rounded = static_cast<T>(float(hidden_vec[j]) + float(residual_vec[j]));
      float x = float(rounded);
      hidden_vec[j] = rounded;
      rounded_vec[j] = x;
      sum_sq += x * x;
    }
    if (elem < d) {
      hidden_vec.store(hidden + bx * stride_hidden + elem);
      rounded_vec.store(smem_x + elem);
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = flashinfer::math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> weight_vec;
    flashinfer::vec_t<float, VEC_SIZE> rounded_vec;
    flashinfer::vec_t<T, VEC_SIZE> out_vec;
    weight_vec.fill(0.f);
    rounded_vec.fill(0.f);
    out_vec.fill(0.f);
    const uint32_t elem = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if (elem < d) {
      weight_vec.load(weight + elem);
      rounded_vec.load(smem_x + elem);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      out_vec[j] = rounded_vec[j] * rms_rcp * float(weight_vec[j]);
    }
    if (elem < d) {
      out_vec.store(out + bx * stride_out + elem);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T>
cudaError_t FusedAddRMSNormRound(T* hidden, const T* residual, T* weight, T* out,
                                 uint32_t batch_size, uint32_t d,
                                 uint32_t stride_hidden, uint32_t stride_residual,
                                 uint32_t stride_out, float eps, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = (flashinfer::ceil_div(num_warps, 4) * 4 + d) * sizeof(float);

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = false;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormRoundKernel<VEC_SIZE, T>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, hidden, residual, weight, out, d,
                                            stride_hidden, stride_residual, stride_out, eps));
  });
  return cudaSuccess;
}

}  // namespace norm
}  // namespace openinfer

__global__ void rms_norm_batched_serial_kernel(const DType *x, const DType *weight, DType *out,
                                               int hidden_dim, int seq_len, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_dim * seq_len;
    if (idx >= total) return;

    int dim = idx % hidden_dim;
    int row = idx / hidden_dim;
    const DType *row_x = x + row * hidden_dim;
    float sum_sq = 0.0f;
    for (int k = 0; k < hidden_dim; ++k) {
        float value = __bfloat162float(row_x[k]);
        sum_sq += value * value;
    }
    float inv_rms = rsqrtf(sum_sq / hidden_dim + eps);
    float value = __bfloat162float(row_x[dim]) * inv_rms * __bfloat162float(weight[dim]);
    out[row * hidden_dim + dim] = __float2bfloat16(value);
}

extern "C" {

// ============================================================================
// RMSNorm (single vector, decode path)
// ============================================================================
void rms_norm_cuda(const DType *x, const DType *weight, DType *out,
                   int n, float eps, cudaStream_t stream) {
    flashinfer::norm::RMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        1, n, n, n, eps, false, stream);
}

// ============================================================================
// RMSNorm batched (prefill path, one block per token)
// ============================================================================
void rms_norm_batched_cuda(const DType *x, const DType *weight, DType *out,
                           int hidden_dim, int seq_len,
                           float eps, cudaStream_t stream) {
    flashinfer::norm::RMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        seq_len, hidden_dim, hidden_dim, hidden_dim, eps, false, stream);
}

// ============================================================================
// Fused Add + RMSNorm (single vector, decode path)
//   hidden += residual; out = norm(hidden, weight)
// ============================================================================
void fused_add_rms_norm_cuda(DType *hidden, const DType *residual,
                             const DType *weight, DType *out,
                             int n, float eps, cudaStream_t stream) {
    // Copy residual → out so FlashInfer can read it as the "input" addend.
    cudaMemcpyAsync(out, residual, static_cast<size_t>(n) * sizeof(DType),
                    cudaMemcpyDeviceToDevice, stream);

    // FlashInfer: hidden(=residual_arg) += out(=input_arg); out = norm(hidden)
    flashinfer::norm::FusedAddRMSNorm<DType>(
        /*input=*/out, /*residual=*/hidden, const_cast<DType*>(weight),
        /*batch_size=*/1, /*d=*/static_cast<uint32_t>(n),
        /*stride_input=*/static_cast<uint32_t>(n),
        /*stride_residual=*/static_cast<uint32_t>(n),
        eps, /*enable_pdl=*/false, stream);
}

// ============================================================================
// Fused Add + RMSNorm batched (prefill path)
// ============================================================================
void fused_add_rms_norm_batched_cuda(DType *hidden, const DType *residual,
                                     const DType *weight, DType *out,
                                     int hidden_dim, int batch_size,
                                     float eps, cudaStream_t stream) {
    size_t total_bytes = static_cast<size_t>(hidden_dim) * batch_size * sizeof(DType);
    cudaMemcpyAsync(out, residual, total_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    flashinfer::norm::FusedAddRMSNorm<DType>(
        /*input=*/out, /*residual=*/hidden, const_cast<DType*>(weight),
        /*batch_size=*/static_cast<uint32_t>(batch_size),
        /*d=*/static_cast<uint32_t>(hidden_dim),
        /*stride_input=*/static_cast<uint32_t>(hidden_dim),
        /*stride_residual=*/static_cast<uint32_t>(hidden_dim),
        eps, /*enable_pdl=*/false, stream);
}

CUresult fused_add_rms_norm_round_batched_cuda(DType *hidden, const DType *residual,
                                               const DType *weight, DType *out,
                                               int hidden_dim, int batch_size,
                                               float eps, cudaStream_t stream) {
    cudaError_t err = openinfer::norm::FusedAddRMSNormRound<DType>(
        hidden, residual, const_cast<DType*>(weight), out,
        /*batch_size=*/static_cast<uint32_t>(batch_size),
        /*d=*/static_cast<uint32_t>(hidden_dim),
        /*stride_hidden=*/static_cast<uint32_t>(hidden_dim),
        /*stride_residual=*/static_cast<uint32_t>(hidden_dim),
        /*stride_out=*/static_cast<uint32_t>(hidden_dim),
        eps, stream);
    return static_cast<CUresult>(err);
}

// ============================================================================
// (1+weight) RMSNorm — Qwen3.5 / Gemma style
// ============================================================================
void rms_norm_offset_cuda(const DType *x, const DType *weight, DType *out,
                          int n, float eps, cudaStream_t stream) {
    flashinfer::norm::GemmaRMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        /*batch_size=*/1, /*d=*/static_cast<uint32_t>(n),
        /*stride_input=*/static_cast<uint32_t>(n),
        /*stride_output=*/static_cast<uint32_t>(n),
        eps, /*enable_pdl=*/false, stream);
}

// ============================================================================
// Batched (1+weight) RMSNorm
// ============================================================================
void rms_norm_batched_offset_cuda(const DType *x, const DType *weight, DType *out,
                                  int hidden_dim, int seq_len,
                                  float eps, cudaStream_t stream) {
    flashinfer::norm::GemmaRMSNorm<DType>(
        const_cast<DType*>(x), const_cast<DType*>(weight), out,
        /*batch_size=*/static_cast<uint32_t>(seq_len),
        /*d=*/static_cast<uint32_t>(hidden_dim),
        /*stride_input=*/static_cast<uint32_t>(hidden_dim),
        /*stride_output=*/static_cast<uint32_t>(hidden_dim),
        eps, /*enable_pdl=*/false, stream);
}

// ============================================================================
// LayerNorm (with bias) — GLM5.2 DSA indexer k_norm.
// HAND-WRITTEN: FlashInfer's generalLayerNorm template depends on
// tensorrt_llm::common::packed_as / num_elems traits that are not available in
// this build's include path. This kernel is a simple single-token LayerNorm
// (mean + variance + affine with bias), memory-bound elementwise — same
// pattern as the hand-written rms_norm_batched_serial_kernel above.
// eps=1e-6, with bias (unlike RMSNorm which has no bias).
// Aligned to vllm DeepseekV32Indexer: nn.LayerNorm(head_dim, eps=1e-6).
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

__global__ void layer_norm_kernel(const DType *x, const float *gamma, const float *beta,
                                   DType *out, int n, float eps) {
    int tid = threadIdx.x;
    extern __shared__ float smem[];  // [n] for val, reused for partial sums

    // Phase 1: load + mean (warp shuffle reduction).
    float val = 0.0f;
    if (tid < n) {
        val = __bfloat162float(x[tid]);
    }
    float sum = warp_reduce_sum(val);

    // Cross-warp reduction via shared memory (only lane 0 of each warp writes).
    int lane = tid % 32;
    int warp = tid / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            smem[0] = sum;
        }
    }
    __syncthreads();
    float mean = smem[0] / n;

    // Phase 2: variance (same reduction pattern).
    float diff_sum = 0.0f;
    if (tid < n) {
        float diff = val - mean;
        diff_sum = diff * diff;
    }
    float var_sum = warp_reduce_sum(diff_sum);
    if (lane == 0) {
        smem[warp] = var_sum;
    }
    __syncthreads();
    if (warp == 0) {
        var_sum = (lane < num_warps) ? smem[lane] : 0.0f;
        var_sum = warp_reduce_sum(var_sum);
        if (lane == 0) {
            smem[0] = var_sum;
        }
    }
    __syncthreads();
    float rstd = rsqrtf(smem[0] / n + eps);

    // Phase 3: output = (x - mean) * rstd * gamma + beta.
    if (tid < n) {
        float normalized = (val - mean) * rstd;
        out[tid] = __float2bfloat16(normalized * gamma[tid] + beta[tid]);
    }
}

CUresult layer_norm_cuda(const DType *x, const float *gamma, const float *beta,
                         DType *out, int n, float eps, cudaStream_t stream) {
    if (x == nullptr || gamma == nullptr || beta == nullptr || out == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int block_size = std::min(n, 1024);
    block_size = 32 * ((block_size + 31) / 32);
    size_t shmem_size = std::min(block_size / 32, (n + 31) / 32) * sizeof(float);
    layer_norm_kernel<<<1, block_size, shmem_size, stream>>>(x, gamma, beta, out, n, eps);
    cudaError_t err = cudaGetLastError();
    return static_cast<CUresult>(err);
}

// ============================================================================
// Fused Add + (1+weight) RMSNorm — Qwen3.5 / Gemma style
// ============================================================================
void fused_add_rms_norm_offset_cuda(DType *hidden, const DType *residual,
                                    const DType *weight, DType *out,
                                    int n, float eps, cudaStream_t stream) {
    cudaMemcpyAsync(out, residual, static_cast<size_t>(n) * sizeof(DType),
                    cudaMemcpyDeviceToDevice, stream);

    flashinfer::norm::GemmaFusedAddRMSNorm<DType>(
        /*input=*/out, /*residual=*/hidden, const_cast<DType*>(weight),
        /*batch_size=*/1, /*d=*/static_cast<uint32_t>(n),
        /*stride_input=*/static_cast<uint32_t>(n),
        /*stride_residual=*/static_cast<uint32_t>(n),
        eps, /*enable_pdl=*/false, stream);
}

} // extern "C"
