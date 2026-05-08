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
#include <cuda_bf16.h>
#include <cstdint>

#include <flashinfer/norm.cuh>

using DType = __nv_bfloat16;

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
