#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

extern thread_local cublasHandle_t g_cublas_handle;

namespace {

using DType = __nv_bfloat16;
using IdType = int32_t;
using PrefillParamsT = flashinfer::SinglePrefillParams<DType, DType, DType>;
using MlaDecodeParamsT = flashinfer::BatchDecodeParamsMLA<DType, DType, DType, IdType>;
using Variant = flashinfer::DefaultAttention</*custom_mask=*/false,
                                           /*sliding_window=*/false,
                                           /*logits_soft_cap=*/false,
                                           /*alibi=*/false>;

constexpr int kKvLoraRank = 512;
constexpr int kRopeDim = 64;
constexpr int kNopeDim = 128;
constexpr int kQHeadDim = 192;
constexpr int kVHeadDim = 128;
constexpr int kKvBHeadDim = kNopeDim + kVHeadDim;
constexpr int kLocalHeads = 8;
constexpr int kQLoraRank = 1536;
constexpr int kQkvAOut = kQLoraRank + kKvLoraRank + kRopeDim;

flashinfer::paged_kv_mla_t<DType, IdType> make_paged_kv_mla(void* ckv_cache,
                                                             void* kpe_cache,
                                                             IdType* page_indices,
                                                             IdType* page_indptr,
                                                             IdType* last_page_len,
                                                             int64_t ckv_stride_page,
                                                             int64_t ckv_stride_n,
                                                             int64_t kpe_stride_page,
                                                             int64_t kpe_stride_n,
                                                             int page_size,
                                                             int batch_size) {
  int64_t ckv_strides[2] = {ckv_stride_page, ckv_stride_n};
  int64_t kpe_strides[2] = {kpe_stride_page, kpe_stride_n};
  return flashinfer::paged_kv_mla_t<DType, IdType>(
      static_cast<uint32_t>(page_size),
      static_cast<uint32_t>(kKvLoraRank),
      static_cast<uint32_t>(kRopeDim),
      static_cast<uint32_t>(batch_size),
      reinterpret_cast<DType*>(ckv_cache),
      ckv_strides,
      reinterpret_cast<DType*>(kpe_cache),
      kpe_strides,
      page_indices,
      page_indptr,
      last_page_len,
      /*rope_pos_offset=*/nullptr);
}

__global__ void split_qkv_a_kernel(const DType* __restrict__ qkv_a,
                                   DType* __restrict__ q_a,
                                   DType* __restrict__ compressed,
                                   DType* __restrict__ k_rope,
                                   int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * kQkvAOut;
  if (idx >= total) {
    return;
  }
  int token = idx / kQkvAOut;
  int dim = idx - token * kQkvAOut;
  if (dim < kQLoraRank) {
    q_a[token * kQLoraRank + dim] = qkv_a[idx];
  } else if (dim < kQLoraRank + kKvLoraRank) {
    compressed[token * kKvLoraRank + (dim - kQLoraRank)] = qkv_a[idx];
  } else {
    k_rope[token * kRopeDim + (dim - kQLoraRank - kKvLoraRank)] = qkv_a[idx];
  }
}

__device__ __forceinline__ DType rope_out(DType even, DType odd, DType cos_v,
                                          DType sin_v, bool upper) {
  float x_even = __bfloat162float(even);
  float x_odd = __bfloat162float(odd);
  float c = __bfloat162float(cos_v);
  float s = __bfloat162float(sin_v);
  float value = upper ? (x_odd * c + x_even * s) : (x_even * c - x_odd * s);
  return __float2bfloat16(value);
}

__global__ void rope_assemble_prefill_kernel(const DType* __restrict__ q_proj,
                                             const DType* __restrict__ k_rope,
                                             const DType* __restrict__ kv_b,
                                             const DType* __restrict__ cos,
                                             const DType* __restrict__ sin,
                                             DType* __restrict__ q_attn,
                                             DType* __restrict__ k_cache,
                                             DType* __restrict__ v_cache,
                                             int seq_len,
                                             int local_heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * local_heads * kQHeadDim;
  if (idx >= total) {
    return;
  }

  int dim = idx % kQHeadDim;
  int head_token = idx / kQHeadDim;
  int head = head_token % local_heads;
  int token = head_token / local_heads;
  int q_base = token * local_heads * kQHeadDim + head * kQHeadDim;

  if (dim < kNopeDim) {
    q_attn[idx] = q_proj[idx];
    int k_dst = head * seq_len * kQHeadDim + token * kQHeadDim + dim;
    int kv_src = token * local_heads * kKvBHeadDim + head * kKvBHeadDim + dim;
    k_cache[k_dst] = kv_b[kv_src];
    return;
  }

  int rope_dim = dim - kNopeDim;
  int pair = rope_dim % (kRopeDim / 2);
  bool upper = rope_dim >= (kRopeDim / 2);
  int q_even_idx = q_base + kNopeDim + pair * 2;
  int q_odd_idx = q_even_idx + 1;
  int k_even_idx = token * kRopeDim + pair * 2;
  int k_odd_idx = k_even_idx + 1;
  // Kimi's HF code first converts adjacent RoPE pairs from
  // [x0, x1, x2, x3, ...] to split-half [x0, x2, ..., x1, x3, ...]
  // with view(..., d/2, 2).transpose(...).  The rotated Q/K tail is
  // intentionally written in that split-half layout.
  int rope_cache_idx = token * kRopeDim + rope_dim;
  DType cos_v = cos[rope_cache_idx];
  DType sin_v = sin[rope_cache_idx];

  q_attn[idx] = rope_out(q_proj[q_even_idx], q_proj[q_odd_idx], cos_v, sin_v, upper);
  int k_dst = head * seq_len * kQHeadDim + token * kQHeadDim + dim;
  k_cache[k_dst] = rope_out(k_rope[k_even_idx], k_rope[k_odd_idx], cos_v, sin_v, upper);
}

__global__ void assemble_v_cache_kernel(const DType* __restrict__ kv_b,
                                        DType* __restrict__ v_cache,
                                        int seq_len,
                                        int local_heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * local_heads * kVHeadDim;
  if (idx >= total) {
    return;
  }
  int dim = idx % kVHeadDim;
  int head_token = idx / kVHeadDim;
  int head = head_token % local_heads;
  int token = head_token / local_heads;
  int src = token * local_heads * kKvBHeadDim + head * kKvBHeadDim + kNopeDim + dim;
  int dst = head * seq_len * kVHeadDim + token * kVHeadDim + dim;
  v_cache[dst] = kv_b[src];
}

__global__ void rope_split_decode_kernel(const DType* __restrict__ q_proj,
                                         const DType* __restrict__ k_rope,
                                         const DType* __restrict__ cos,
                                         const DType* __restrict__ sin,
                                         const IdType* __restrict__ positions,
                                         DType* __restrict__ q_nope,
                                         DType* __restrict__ q_pe,
                                         DType* __restrict__ append_kpe,
                                         int batch_size,
                                         int local_heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * local_heads * kQHeadDim;
  if (idx >= total) {
    return;
  }

  int dim = idx % kQHeadDim;
  int head_token = idx / kQHeadDim;
  int head = head_token % local_heads;
  int token = head_token / local_heads;
  int q_base = token * local_heads * kQHeadDim + head * kQHeadDim;

  if (dim < kNopeDim) {
    int dst = token * local_heads * kNopeDim + head * kNopeDim + dim;
    q_nope[dst] = q_proj[idx];
    return;
  }

  int rope_dim = dim - kNopeDim;
  int pair = rope_dim % (kRopeDim / 2);
  bool upper = rope_dim >= (kRopeDim / 2);
  int position = positions[token];
  int rope_cache_idx = position * kRopeDim + rope_dim;
  DType cos_v = cos[rope_cache_idx];
  DType sin_v = sin[rope_cache_idx];

  int q_even_idx = q_base + kNopeDim + pair * 2;
  int q_odd_idx = q_even_idx + 1;
  int q_dst = token * local_heads * kRopeDim + head * kRopeDim + rope_dim;
  q_pe[q_dst] = rope_out(q_proj[q_even_idx], q_proj[q_odd_idx], cos_v, sin_v, upper);

  if (head == 0) {
    int k_even_idx = token * kRopeDim + pair * 2;
    int k_odd_idx = k_even_idx + 1;
    append_kpe[token * kRopeDim + rope_dim] =
        rope_out(k_rope[k_even_idx], k_rope[k_odd_idx], cos_v, sin_v, upper);
  }
}

__global__ void rope_apply_kpe_kernel(const DType* __restrict__ k_rope,
                                      const DType* __restrict__ cos,
                                      const DType* __restrict__ sin,
                                      const IdType* __restrict__ positions,
                                      DType* __restrict__ append_kpe,
                                      int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * kRopeDim;
  if (idx >= total) {
    return;
  }

  int rope_dim = idx % kRopeDim;
  int token = idx / kRopeDim;
  int pair = rope_dim % (kRopeDim / 2);
  bool upper = rope_dim >= (kRopeDim / 2);
  int position = positions[token];
  int rope_cache_idx = position * kRopeDim + rope_dim;
  DType cos_v = cos[rope_cache_idx];
  DType sin_v = sin[rope_cache_idx];
  int k_even_idx = token * kRopeDim + pair * 2;
  int k_odd_idx = k_even_idx + 1;
  append_kpe[idx] = rope_out(k_rope[k_even_idx], k_rope[k_odd_idx], cos_v, sin_v, upper);
}

}  // namespace

extern "C" {

cudaError_t kimi_mla_split_qkv_a_cuda(const DType* qkv_a,
                                      DType* q_a,
                                      DType* compressed,
                                      DType* k_rope,
                                      int seq_len,
                                      cudaStream_t stream) {
  if (seq_len <= 0) {
    return cudaErrorInvalidValue;
  }
  int total = seq_len * kQkvAOut;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  split_qkv_a_kernel<<<blocks, threads, 0, stream>>>(qkv_a, q_a, compressed, k_rope, seq_len);
  return cudaGetLastError();
}

cudaError_t kimi_mla_rope_assemble_prefill_cuda(const DType* q_proj,
                                                const DType* k_rope,
                                                const DType* kv_b,
                                                const DType* cos,
                                                const DType* sin,
                                                DType* q_attn,
                                                DType* k_cache,
                                                DType* v_cache,
                                                int seq_len,
                                                int local_heads,
                                                cudaStream_t stream) {
  if (seq_len <= 0 || local_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  int threads = 256;
  int qk_total = seq_len * local_heads * kQHeadDim;
  int qk_blocks = (qk_total + threads - 1) / threads;
  rope_assemble_prefill_kernel<<<qk_blocks, threads, 0, stream>>>(
      q_proj, k_rope, kv_b, cos, sin, q_attn, k_cache, v_cache, seq_len, local_heads);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  int v_total = seq_len * local_heads * kVHeadDim;
  int v_blocks = (v_total + threads - 1) / threads;
  assemble_v_cache_kernel<<<v_blocks, threads, 0, stream>>>(kv_b, v_cache, seq_len, local_heads);
  return cudaGetLastError();
}

cudaError_t kimi_mla_rope_split_decode_cuda(const DType* q_proj,
                                            const DType* k_rope,
                                            const DType* cos,
                                            const DType* sin,
                                            const IdType* positions,
                                            DType* q_nope,
                                            DType* q_pe,
                                            DType* append_kpe,
                                            int batch_size,
                                            int local_heads,
                                            cudaStream_t stream) {
  if (batch_size <= 0 || local_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  int total = batch_size * local_heads * kQHeadDim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  rope_split_decode_kernel<<<blocks, threads, 0, stream>>>(
      q_proj, k_rope, cos, sin, positions, q_nope, q_pe, append_kpe, batch_size, local_heads);
  return cudaGetLastError();
}

cudaError_t kimi_mla_rope_apply_kpe_cuda(const DType* k_rope,
                                         const DType* cos,
                                         const DType* sin,
                                         const IdType* positions,
                                         DType* append_kpe,
                                         int seq_len,
                                         cudaStream_t stream) {
  if (seq_len <= 0) {
    return cudaErrorInvalidValue;
  }
  int total = seq_len * kRopeDim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  rope_apply_kpe_kernel<<<blocks, threads, 0, stream>>>(
      k_rope, cos, sin, positions, append_kpe, seq_len);
  return cudaGetLastError();
}

int kimi_flashinfer_single_prefill_mla_cuda(void* q,
                                            void* output,
                                            void* k_cache,
                                            void* v_cache,
                                            int local_heads,
                                            int seq_len,
                                            float sm_scale,
                                            cudaStream_t stream) {
  if (local_heads <= 0 || seq_len <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  PrefillParamsT params;
  params.q = reinterpret_cast<DType*>(q);
  params.k = reinterpret_cast<DType*>(k_cache);
  params.v = reinterpret_cast<DType*>(v_cache);
  params.maybe_custom_mask = nullptr;
  params.o = reinterpret_cast<DType*>(output);
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.group_size = flashinfer::uint_fastdiv(1);
  params.qo_len = static_cast<uint32_t>(seq_len);
  params.kv_len = static_cast<uint32_t>(seq_len);
  params.num_qo_heads = static_cast<uint32_t>(local_heads);
  params.num_kv_heads = static_cast<uint32_t>(local_heads);
  params.q_stride_n = static_cast<uint32_t>(local_heads * kQHeadDim);
  params.q_stride_h = static_cast<uint32_t>(kQHeadDim);
  params.k_stride_n = static_cast<uint32_t>(kQHeadDim);
  params.k_stride_h = static_cast<uint32_t>(seq_len * kQHeadDim);
  params.v_stride_n = static_cast<uint32_t>(kVHeadDim);
  params.v_stride_h = static_cast<uint32_t>(seq_len * kVHeadDim);
  params.head_dim = static_cast<uint32_t>(kQHeadDim);
  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;
  params.rope_rcp_scale = 1.0f;
  params.rope_rcp_theta = 1.0e-6f;
  params.partition_kv = false;

  return static_cast<int>(
      flashinfer::SinglePrefillWithKVCacheDispatched<
          /*HEAD_DIM_QK=*/kQHeadDim,
          /*HEAD_DIM_VO=*/kVHeadDim,
          flashinfer::PosEncodingMode::kNone,
          /*USE_FP16_QK_REDUCTION=*/false,
          flashinfer::MaskMode::kCausal,
          Variant,
          PrefillParamsT>(params, /*tmp=*/nullptr, stream));
}

int kimi_mla_absorb_q_nope_cuda(const DType* kv_b_proj,
                                const DType* q_nope,
                                DType* q_abs_nope,
                                int batch_size,
                                cudaStream_t stream) {
  if (batch_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (g_cublas_handle == nullptr) {
    return static_cast<int>(cudaErrorInitializationError);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }

  // kv_b_proj is row-major [local_heads, k_nope + v, kv_lora_rank].
  // Row-major [k_nope, kv_lora_rank] is the same memory as column-major
  // [kv_lora_rank, k_nope], which is exactly W_UK_T for q absorption.
  status = cublasGemmStridedBatchedEx(
      g_cublas_handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      /*m=*/kKvLoraRank,
      /*n=*/batch_size,
      /*k=*/kNopeDim,
      &alpha,
      kv_b_proj,
      CUDA_R_16BF,
      /*lda=*/kKvLoraRank,
      /*strideA=*/static_cast<long long>(kKvBHeadDim) * kKvLoraRank,
      q_nope,
      CUDA_R_16BF,
      /*ldb=*/kLocalHeads * kNopeDim,
      /*strideB=*/kNopeDim,
      &beta,
      q_abs_nope,
      CUDA_R_16BF,
      /*ldc=*/kLocalHeads * kKvLoraRank,
      /*strideC=*/kKvLoraRank,
      /*batchCount=*/kLocalHeads,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

int kimi_mla_v_up_cuda(const DType* kv_b_proj,
                       const DType* latent,
                       DType* output,
                       int batch_size,
                       cudaStream_t stream) {
  if (batch_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (g_cublas_handle == nullptr) {
    return static_cast<int>(cudaErrorInitializationError);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }

  const DType* w_uv = kv_b_proj + static_cast<int64_t>(kNopeDim) * kKvLoraRank;
  status = cublasGemmStridedBatchedEx(
      g_cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      /*m=*/kVHeadDim,
      /*n=*/batch_size,
      /*k=*/kKvLoraRank,
      &alpha,
      w_uv,
      CUDA_R_16BF,
      /*lda=*/kKvLoraRank,
      /*strideA=*/static_cast<long long>(kKvBHeadDim) * kKvLoraRank,
      latent,
      CUDA_R_16BF,
      /*ldb=*/kLocalHeads * kKvLoraRank,
      /*strideB=*/kKvLoraRank,
      &beta,
      output,
      CUDA_R_16BF,
      /*ldc=*/kLocalHeads * kVHeadDim,
      /*strideC=*/kVHeadDim,
      /*batchCount=*/kLocalHeads,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

int kimi_mla_paged_kv_append_cuda(void* ckv_cache,
                                  void* kpe_cache,
                                  IdType* page_indices,
                                  IdType* page_indptr,
                                  IdType* last_page_len,
                                  void* append_ckv,
                                  void* append_kpe,
                                  IdType* batch_indices,
                                  IdType* positions,
                                  int nnz,
                                  int64_t ckv_stride_page,
                                  int64_t ckv_stride_n,
                                  int64_t kpe_stride_page,
                                  int64_t kpe_stride_n,
                                  int page_size,
                                  int batch_size,
                                  cudaStream_t stream) {
  if (nnz <= 0 || page_size <= 0 || batch_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto paged_kv =
      make_paged_kv_mla(ckv_cache, kpe_cache, page_indices, page_indptr, last_page_len,
                        ckv_stride_page, ckv_stride_n, kpe_stride_page, kpe_stride_n, page_size,
                        batch_size);
  return static_cast<int>(
      flashinfer::AppendPagedKVMlaCache<DType, IdType>(
          paged_kv,
          reinterpret_cast<DType*>(append_ckv),
          reinterpret_cast<DType*>(append_kpe),
          batch_indices,
          positions,
          static_cast<uint32_t>(nnz),
          /*append_ckv_stride_n=*/kKvLoraRank,
          /*append_kpe_stride_n=*/kRopeDim,
          stream));
}

int kimi_flashinfer_batch_decode_mla_cuda(void* q_nope,
                                          void* q_pe,
                                          void* output,
                                          void* ckv_cache,
                                          void* kpe_cache,
                                          IdType* page_indices,
                                          IdType* page_indptr,
                                          IdType* last_page_len,
                                          IdType* request_indices,
                                          IdType* kv_tile_indices,
                                          IdType* kv_chunk_size_ptr,
                                          int num_qo_heads,
                                          int64_t ckv_stride_page,
                                          int64_t ckv_stride_n,
                                          int64_t kpe_stride_page,
                                          int64_t kpe_stride_n,
                                          int page_size,
                                          int batch_size,
                                          float sm_scale,
                                          cudaStream_t stream) {
  if (num_qo_heads <= 0 || page_size <= 0 || batch_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto paged_kv =
      make_paged_kv_mla(ckv_cache, kpe_cache, page_indices, page_indptr, last_page_len,
                        ckv_stride_page, ckv_stride_n, kpe_stride_page, kpe_stride_n, page_size,
                        batch_size);
  MlaDecodeParamsT params(
      reinterpret_cast<DType*>(q_nope),
      reinterpret_cast<DType*>(q_pe),
      /*q_rope_offset=*/nullptr,
      paged_kv,
      reinterpret_cast<DType*>(output),
      /*lse=*/nullptr,
      static_cast<uint32_t>(num_qo_heads),
      /*window_left=*/-1,
      /*logits_soft_cap=*/0.0f,
      sm_scale,
      /*rope_scale=*/1.0f,
      /*rope_theta=*/1.0e6f);

  params.padded_batch_size = static_cast<uint32_t>(batch_size);
  params.request_indices = request_indices;
  params.kv_tile_indices = kv_tile_indices;
  params.o_indptr = nullptr;
  params.kv_chunk_size_ptr = kv_chunk_size_ptr;
  params.block_valid_mask = nullptr;
  params.partition_kv = false;

  return static_cast<int>(
      flashinfer::BatchDecodeWithPagedKVCacheDispatchedMLA<
          /*HEAD_DIM_CKV=*/kKvLoraRank,
          /*HEAD_DIM_KPE=*/kRopeDim,
          Variant,
          MlaDecodeParamsT>(
          params,
          /*tmp_v=*/nullptr,
          /*tmp_s=*/nullptr,
          /*enable_pdl=*/false,
          stream));
}

}  // extern "C"
