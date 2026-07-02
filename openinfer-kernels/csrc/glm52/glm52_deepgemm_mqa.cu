// GLM5.2 DSA indexer: DeepGEMM paged MQA logits C ABI wrapper (torch-free).
//
// Calls SM90FP8PagedMQALogitsRuntime + SM90PagedMQALogitsMetadataRuntime directly
// with raw device pointers and manually-constructed TMA descriptors, bypassing
// the torch::Tensor-based free functions in sm90_fp8_mqa_logits.hpp.
//
// This is the first real DeepGEMM JIT kernel call in the codebase.
// DG_NO_TORCH is defined via build.rs (-DDG_NO_TORCH).

#include "../common.cuh"

#include <cuda.h>
#include <cstdint>
#include <cmath>
#include <memory>
#include <mutex>
#include <filesystem>

#include <jit_kernels/impls/sm90_fp8_mqa_logits.hpp>

namespace {

constexpr int kSM90SmemCapacity = 232448;
constexpr int kSplitKv = 256;
constexpr int kMmaM = 64;
constexpr int kNumSpecializedThreads = 128;
constexpr int kNumQStages = 3;
constexpr int kNumKVStages = 3;

std::once_flag g_dg_init_flag;
CUresult g_dg_init_result = CUDA_SUCCESS;

CUresult ensure_dg_runtime_init() {
    std::call_once(g_dg_init_flag, []() {
        const char* dg_root = std::getenv("OPENINFER_DEEPGEMM_ROOT");
        const char* cuda_home = std::getenv("CUDA_HOME");
        if (!dg_root || !cuda_home) {
            fprintf(stderr, "glm52_deepgemm_mqa: OPENINFER_DEEPGEMM_ROOT and CUDA_HOME must be set\n");
            g_dg_init_result = CUDA_ERROR_INVALID_VALUE;
            return;
        }
        deep_gemm::Compiler::prepare_init(dg_root, cuda_home);
        deep_gemm::KernelRuntime::prepare_init(cuda_home);
        deep_gemm::IncludeParser::prepare_init(dg_root);
    });
    return g_dg_init_result;
}

} // namespace anon

extern "C" {

CUresult glm52_deepgemm_paged_mqa_metadata_cuda(
    int* context_lens,
    int* schedule_metadata,
    int batch_size,
    int next_n,
    int block_kv,
    int num_sms,
    bool is_context_lens_2d,
    bool is_varlen,
    const int* indices_ptr,
    cudaStream_t stream
) {
    if (!context_lens || !schedule_metadata || batch_size <= 0 || block_kv <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (kSplitKv % block_kv != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    CUresult init_err = ensure_dg_runtime_init();
    if (init_err != CUDA_SUCCESS) {
        return init_err;
    }

    constexpr int num_threads = 32;
    const int aligned_batch_size = deep_gemm::align(batch_size, 32);
    const int num_smem_ints = is_varlen ? 3 * aligned_batch_size + 1 : aligned_batch_size;
    const int smem_size = num_smem_ints * static_cast<int>(sizeof(int));
    if (smem_size > kSM90SmemCapacity) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const deep_gemm::SM90PagedMQALogitsMetadataRuntime::Args args = {
        .aligned_batch_size = aligned_batch_size,
        .split_kv = kSplitKv,
        .num_sms = num_sms,
        .is_varlen = is_varlen,
        .batch_size = batch_size,
        .next_n = next_n,
        .is_context_lens_2d = is_context_lens_2d,
        .context_lens = context_lens,
        .indices = const_cast<int*>(indices_ptr),
        .schedule_metadata = schedule_metadata,
        .launch_args = deep_gemm::LaunchArgs(1, num_threads, smem_size)
    };

    try {
        const auto code = deep_gemm::SM90PagedMQALogitsMetadataRuntime::generate(args);
        const auto runtime = deep_gemm::compiler->build("sm90_paged_mqa_logits_metadata", code);
        deep_gemm::SM90PagedMQALogitsMetadataRuntime::launch(runtime, args, stream);
    } catch (const std::exception& e) {
        fprintf(stderr, "glm52_deepgemm_paged_mqa_metadata: %s\n", e.what());
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    return CUDA_SUCCESS;
}

CUresult glm52_deepgemm_paged_mqa_logits_cuda(
    const void* q,
    const void* kv_cache,
    int64_t kv_cache_stride_bytes,
    const void* weights,
    const int* context_lens,
    void* logits,
    const int* block_table,
    const int* indices,
    int* schedule_meta,
    int batch_size,
    int next_n,
    int num_heads,
    int head_dim,
    int num_kv_blocks,
    int block_kv,
    bool is_context_lens_2d,
    bool is_varlen,
    int logits_stride,
    int block_table_stride,
    int num_sms,
    int q_elem_size,
    int kv_elem_size,
    int weights_elem_size,
    int kv_scales_elem_size,
    cudaStream_t stream
) {
    if (!q || !kv_cache || !weights || !context_lens ||
        !logits || !block_table || !schedule_meta || batch_size <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (head_dim != 128 || block_kv <= 0 || num_heads <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (128 % num_heads != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (next_n != 1 && next_n != 2) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Indexer cache layout: [block_kv * head_dim fp8 | block_kv * 4 f32] per block.
    // The stride must accommodate both regions.
    const int64_t min_stride = static_cast<int64_t>(block_kv) * (head_dim + 4);
    if (kv_cache_stride_bytes < min_stride) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int split_kv = kSplitKv;
    if (split_kv % kMmaM != 0 || logits_stride % split_kv != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int num_math_warp_groups = split_kv / kMmaM;
    const int num_math_threads = num_math_warp_groups * 128;

    CUresult init_err = ensure_dg_runtime_init();
    if (init_err != CUDA_SUCCESS) {
        return init_err;
    }

    try {
        const int next_n_atom = (is_varlen || next_n >= 2) ? 2 : 1;

        // TMA descriptor for q: [batch_size * next_n * num_heads, head_dim]
        // gmem: inner=head_dim, outer=batch_size*next_n*num_heads
        // smem: inner=next_n_atom*num_heads (overwritten by swizzle), outer=...
        // gmem_outer_stride = head_dim (row stride of q in elements)
        // swizzle_mode = head_dim (128)
        const auto tensor_map_q = deep_gemm::make_tma_2d_desc_raw(
            const_cast<void*>(q), q_elem_size, deep_gemm::DgDtype::Float8_e4m3,
            head_dim, batch_size * next_n * num_heads,
            next_n_atom * num_heads, 1,
            head_dim,
            head_dim);

        // Indexer cache layout (from glm52_indexer.cu::indexer_k_quant_and_cache_kernel):
        // Each block is [block_kv * head_dim fp8 values][block_kv * 4 f32 scales],
        // blocks strided by kv_cache_stride_bytes. The scales region starts at
        // byte offset block_kv * head_dim within each block. We compute the
        // scales pointer from the kv_cache base + that offset — no separate
        // scales buffer needed (matches vllm's decode-path API).
        const float* kv_cache_scales = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(kv_cache) +
            static_cast<size_t>(block_kv) * head_dim);

        // TMA descriptor for kv_cache: [head_dim, block_kv, num_kv_blocks] (3D)
        // gstride0 = head_dim (token stride within a block — fp8 values are
        //   packed as [block_kv, head_dim] contiguous)
        // gstride1 = kv_cache_stride_bytes / kv_elem_size (block stride —
        //   jumps over the trailing scales region of each block)
        const auto tensor_map_kv = deep_gemm::make_tma_3d_desc_raw(
            const_cast<void*>(kv_cache), kv_elem_size, deep_gemm::DgDtype::Float8_e4m3,
            head_dim, block_kv, num_kv_blocks,
            head_dim, block_kv, 1,
            head_dim,
            static_cast<int>(kv_cache_stride_bytes / kv_elem_size),
            head_dim);

        // TMA descriptor for kv_cache_scales: [block_kv, num_kv_blocks] (2D, f32)
        // The scales pointer is an offset into kv_cache (start of scale region
        // in block 0). Within each block, scales are [block_kv] f32 contiguous.
        // gstride0 = kv_cache_stride_bytes / kv_scales_elem_size (block stride)
        const int aligned_block_kv = deep_gemm::get_tma_aligned_size(block_kv, kv_scales_elem_size);
        const auto tensor_map_kv_scales = deep_gemm::make_tma_2d_desc_raw(
            const_cast<void*>(static_cast<const void*>(kv_cache_scales)),
            kv_scales_elem_size, deep_gemm::DgDtype::Float,
            aligned_block_kv, num_kv_blocks,
            block_kv, 1,
            static_cast<int>(kv_cache_stride_bytes / kv_scales_elem_size),
            0);

        // TMA descriptor for weights: [batch_size * next_n, num_heads] (2D)
        // gmem: inner=num_heads, outer=batch_size*next_n
        // smem: inner=num_heads (overwritten by swizzle=0, so stays), outer=next_n_atom
        // gmem_outer_stride = weights.stride(0) = num_heads
        // swizzle_mode = 0
        // weights are f32 (per-head scaling factors folded with q_scale).
        const auto tensor_map_weights = deep_gemm::make_tma_2d_desc_raw(
            const_cast<void*>(weights), weights_elem_size, deep_gemm::DgDtype::Float,
            num_heads, batch_size * next_n,
            num_heads, next_n_atom,
            num_heads,
            0);

        // smem size calculation (mirrors the original sm90_fp8_paged_mqa_logits)
        const int swizzle_alignment = head_dim * 8;
        const int smem_q_size_per_stage = next_n * num_heads * head_dim * q_elem_size;
        const int aligned_smem_weight_size_per_stage = deep_gemm::align(
            next_n * num_heads * weights_elem_size, swizzle_alignment);
        const int smem_q_pipe_size = kNumQStages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage)
                                     + deep_gemm::align(kNumQStages * 8 * 2, swizzle_alignment);
        const int smem_kv_size_per_stage = block_kv * head_dim * kv_elem_size;
        const int aligned_smem_kv_scale_size_per_stage = deep_gemm::align(
            block_kv * kv_scales_elem_size, swizzle_alignment);
        const int smem_kv_pipe_size = kNumKVStages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage)
                                     + deep_gemm::align(kNumKVStages * 8 * 2, swizzle_alignment);
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;
        const int smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size
                             + smem_umma_barriers + smem_tmem_ptr;
        if (smem_size > kSM90SmemCapacity) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        const deep_gemm::SM90FP8PagedMQALogitsRuntime::Args args = {
            .batch_size = batch_size,
            .next_n = next_n,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .block_kv = block_kv,
            .is_context_lens_2d = is_context_lens_2d,
            .is_varlen = is_varlen,
            .block_table_stride = block_table_stride,
            .logits_stride = logits_stride,
            .num_q_stages = kNumQStages,
            .num_kv_stages = kNumKVStages,
            .split_kv = split_kv,
            .context_lens = const_cast<int*>(context_lens),
            .logits = logits,
            .block_table = const_cast<int*>(block_table),
            .indices = is_varlen ? const_cast<int*>(indices) : nullptr,
            .schedule_meta = schedule_meta,
            .tensor_map_q = tensor_map_q,
            .tensor_map_kv = tensor_map_kv,
            .tensor_map_kv_scales = tensor_map_kv_scales,
            .tensor_map_weights = tensor_map_weights,
            .logits_dtype = deep_gemm::DgDtype::BFloat16,
            .num_specialized_threads = kNumSpecializedThreads,
            .num_math_threads = num_math_threads,
            .launch_args = deep_gemm::LaunchArgs(num_sms,
                                                  kNumSpecializedThreads + num_math_threads,
                                                  smem_size)
        };

        const auto code = deep_gemm::SM90FP8PagedMQALogitsRuntime::generate(args);
        const auto runtime = deep_gemm::compiler->build("sm90_fp8_paged_mqa_logits", code);
        deep_gemm::SM90FP8PagedMQALogitsRuntime::launch(runtime, args, stream);
    } catch (const std::exception& e) {
        fprintf(stderr, "glm52_deepgemm_paged_mqa_logits: %s\n", e.what());
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    return CUDA_SUCCESS;
}

} // extern "C"
