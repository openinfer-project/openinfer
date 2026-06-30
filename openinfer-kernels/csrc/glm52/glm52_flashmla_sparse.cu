#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <exception>

#include "params.h"
#include "sm90/decode/sparse_fp8/splitkv_mla.cuh"
#include "smxx/decode/combine/combine.cu"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu"

namespace {

constexpr int kBatchCapacity = 128;
constexpr int kSq = 1;
constexpr int kHeads = 64;
constexpr int kKvHeads = 1;
constexpr int kQkDim = 576;
constexpr int kVDim = 512;
constexpr int kPageSize = 64;
constexpr int kBytesPerToken = 656;
constexpr int kTopk = 2048;
constexpr int kSchedMetaInts = sizeof(DecodingSchedMeta) / sizeof(int);
constexpr int kMaxSmParts = 160;
constexpr int kBlockSizeTopk = 64;
constexpr int kFixedOverheadBlocks = 5;
constexpr float kLog2E = 1.4426950408889634f;

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

template <typename Fn>
CUresult run_flashmla_host(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return CUDA_ERROR_INVALID_VALUE;
  } catch (...) {
    return CUDA_ERROR_LAUNCH_FAILED;
  }
  return consume_last_cuda_error();
}

CUresult current_sm90_num_sm_parts(int* num_sm_parts) {
  if (num_sm_parts == nullptr) return CUDA_ERROR_INVALID_VALUE;

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return map_cuda_error(err);

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) return map_cuda_error(err);
  if (prop.major != 9 || prop.minor != 0) return CUDA_ERROR_NOT_SUPPORTED;

  int parts = std::max(prop.multiProcessorCount / kSq / (kHeads / 64), 1);
  if (parts > kMaxSmParts) return CUDA_ERROR_NOT_SUPPORTED;
  *num_sm_parts = parts;
  return CUDA_SUCCESS;
}

bool valid_common_shape(int batch_size, int num_blocks, int topk,
                        int num_sm_parts) {
  return batch_size > 0 && batch_size <= kBatchCapacity && num_blocks > 0 &&
         topk == kTopk && num_sm_parts > 0 && num_sm_parts <= kMaxSmParts;
}

}  // namespace

extern "C" CUresult glm52_flashmla_sparse_decode_num_sm_parts_cuda(
    int* num_sm_parts) {
  return current_sm90_num_sm_parts(num_sm_parts);
}

extern "C" CUresult glm52_flashmla_sparse_decode_metadata_cuda(
    int* tile_scheduler_metadata, int* num_splits, int batch_size, int topk,
    int num_sm_parts, cudaStream_t stream) {
  if (tile_scheduler_metadata == nullptr || num_splits == nullptr ||
      !valid_common_shape(batch_size, 1, topk, num_sm_parts)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  GetDecodeSchedMetaParams params{
      batch_size,
      kSq,
      kBlockSizeTopk,
      kFixedOverheadBlocks,
      topk,
      0,
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata),
      num_splits,
      num_sm_parts,
      stream,
  };
  return run_flashmla_host(
      [&] { smxx::decode::run_get_decoding_sched_meta_kernel(params); });
}

extern "C" CUresult glm52_flashmla_sparse_decode_launch_cuda(
    const void* q, const void* packed_kv_cache, const int* topk_indices,
    const int* tile_scheduler_metadata, const int* num_splits, void* out_latent,
    float* lse, float* lse_accum, float* o_accum, int batch_size,
    int num_blocks, int topk, int num_sm_parts, float sm_scale,
    cudaStream_t stream) {
  if (q == nullptr || packed_kv_cache == nullptr || topk_indices == nullptr ||
      tile_scheduler_metadata == nullptr || num_splits == nullptr ||
      out_latent == nullptr || lse == nullptr || lse_accum == nullptr ||
      o_accum == nullptr ||
      !valid_common_shape(batch_size, num_blocks, topk, num_sm_parts)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  SparseAttnDecodeParams params{};
  params.b = batch_size;
  params.s_q = kSq;
  params.h_q = kHeads;
  params.h_kv = kKvHeads;
  params.d_qk = kQkDim;
  params.d_v = kVDim;
  params.sm_scale = sm_scale;
  params.sm_scale_div_log2 = sm_scale * kLog2E;
  params.num_blocks = num_blocks;
  params.page_block_size = kPageSize;
  params.topk = topk;
  params.model_type = ModelType::V32;

  params.q = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(q));
  params.kv =
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(packed_kv_cache));
  params.indices = const_cast<int*>(topk_indices);
  params.topk_length = nullptr;
  params.attn_sink = nullptr;
  params.lse = lse;
  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out_latent);

  params.extra_num_blocks = 0;
  params.extra_page_block_size = 0;
  params.extra_topk = 0;
  params.extra_kv = nullptr;
  params.extra_indices = nullptr;
  params.extra_topk_length = nullptr;

  params.stride_q_b = kSq * kHeads * kQkDim;
  params.stride_q_s_q = kHeads * kQkDim;
  params.stride_q_h_q = kQkDim;
  params.stride_kv_block = kPageSize * kBytesPerToken;
  params.stride_kv_row = kBytesPerToken;
  params.stride_indices_b = kSq * topk;
  params.stride_indices_s_q = topk;
  params.stride_lse_b = kSq * kHeads;
  params.stride_lse_s_q = kHeads;
  params.stride_o_b = kSq * kHeads * kVDim;
  params.stride_o_s_q = kHeads * kVDim;
  params.stride_o_h_q = kVDim;
  params.stride_extra_kv_block = 0;
  params.stride_extra_kv_row = 0;
  params.stride_extra_indices_b = 0;
  params.stride_extra_indices_s_q = 0;
  params.stream = stream;

  params.lse_accum = lse_accum;
  params.o_accum = o_accum;
  params.stride_lse_accum_split = kSq * kHeads;
  params.stride_lse_accum_s_q = kHeads;
  params.stride_o_accum_split = kSq * kHeads * kVDim;
  params.stride_o_accum_s_q = kHeads * kVDim;
  params.stride_o_accum_h_q = kVDim;
  params.tile_scheduler_metadata_ptr =
      reinterpret_cast<DecodingSchedMeta*>(const_cast<int*>(tile_scheduler_metadata));
  params.num_splits_ptr = const_cast<int*>(num_splits);
  params.num_sm_parts = num_sm_parts;

  CUresult result = run_flashmla_host([&] {
    sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<
        ModelType::V32, kHeads>(params);
  });
  if (result != CUDA_SUCCESS) return result;

  CombineParams combine_params{};
  combine_params.b = batch_size;
  combine_params.s_q = kSq;
  combine_params.h_q = kHeads;
  combine_params.d_v = kVDim;
  combine_params.lse = lse;
  combine_params.out = out_latent;
  combine_params.stride_lse_b = params.stride_lse_b;
  combine_params.stride_lse_s_q = params.stride_lse_s_q;
  combine_params.stride_o_b = params.stride_o_b;
  combine_params.stride_o_s_q = params.stride_o_s_q;
  combine_params.stride_o_h_q = params.stride_o_h_q;
  combine_params.lse_accum = lse_accum;
  combine_params.o_accum = o_accum;
  combine_params.stride_lse_accum_split = params.stride_lse_accum_split;
  combine_params.stride_lse_accum_s_q = params.stride_lse_accum_s_q;
  combine_params.stride_o_accum_split = params.stride_o_accum_split;
  combine_params.stride_o_accum_s_q = params.stride_o_accum_s_q;
  combine_params.stride_o_accum_h_q = params.stride_o_accum_h_q;
  combine_params.tile_scheduler_metadata_ptr =
      params.tile_scheduler_metadata_ptr;
  combine_params.num_splits_ptr = params.num_splits_ptr;
  combine_params.num_sm_parts = num_sm_parts;
  combine_params.attn_sink = nullptr;
  combine_params.stream = stream;

  return run_flashmla_host([&] {
    smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(
        combine_params);
  });
}
