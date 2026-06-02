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

namespace pegainfer {
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
}  // namespace pegainfer

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
    cudaError_t err = pegainfer::norm::FusedAddRMSNormRound<DType>(
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
