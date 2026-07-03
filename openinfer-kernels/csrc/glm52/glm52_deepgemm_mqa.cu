// GLM5.2 DSA indexer: DeepGEMM paged MQA logits, AOT-instantiated (no JIT).
//
// The decode path's codegen parameters are all compile-time constants
// (next_n=1, 32 heads, head_dim 128, block_kv 64, split_kv 256, bf16 logits,
// batch <= 32, 132 SMs), so both kernels are instantiated here directly from
// DeepGEMM's device headers and launched with cudaLaunchKernelExC. This
// removes DeepGEMM's runtime JIT entirely — its compiler, include parser and
// launch-config helpers keep unsynchronized global state that the DP8
// coordinator's 8 concurrent rank threads corrupt (include-parser
// assertions, per-context CUfunction handles, a shared static attrs array),
// and the per-launch codegen + code hashing cost ~0.4 ms per serialized
// call. It also drops the OPENINFER_DEEPGEMM_ROOT / CUDA_HOME runtime
// requirements — nothing is compiled at runtime anymore.
//
// Requires sm_90a (build.rs promotes sm_90 targets). DG_NO_TORCH is defined
// via build.rs.

#include "../common.cuh"

#include <cuda.h>
#include <cstdint>
#include <cstdio>

// Without an sm_90a nvcc target the wgmma device code cannot be assembled;
// build.rs then omits this define and both entry points compile as
// NOT_SUPPORTED stubs (GLM5.2 decode is Hopper-only today).
#ifdef GLM52_DEEPGEMM_MQA_SM90A

#include <jit_kernels/impls/runtime_utils.hpp>

#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm90_paged_mqa_logits.cuh>

namespace {

constexpr int kSM90SmemCapacity = 232448;
constexpr int kSplitKv = 256;
constexpr int kMmaM = 64;
constexpr int kNumSpecializedThreads = 128;
constexpr int kNumQStages = 3;
constexpr int kNumKVStages = 3;

// AOT instantiation shape: the GLM5.2 DSA decode indexer. The C ABI keeps
// the general parameters and fail-closes on anything the instantiation does
// not cover.
constexpr int kAotNextN = 1;
constexpr int kAotNumHeads = 32;
constexpr int kAotHeadDim = 128;
constexpr int kAotBlockKv = 64;
constexpr int kAotNumSms = 132;
constexpr int kAotAlignedBatchSize = 32;
constexpr int kAotNumMathThreads = kSplitKv / kMmaM * 128;

const auto kMetadataKernel = &deep_gemm::sched::sm90_paged_mqa_logits_metadata<
    kAotAlignedBatchSize, kSplitKv, kAotNumSms, /*kIsVarlen=*/false>;

const auto kLogitsKernel = &deep_gemm::sm90_fp8_paged_mqa_logits<
    kAotNextN, kAotNumHeads, kAotHeadDim, kAotBlockKv,
    /*kIsContextLens2D=*/false, /*kIsVarlen=*/false,
    kNumQStages, kNumKVStages, kSplitKv,
    kNumSpecializedThreads, kAotNumMathThreads, cutlass::bfloat16_t>;

// Both kernels use programmatic dependent launch intrinsics
// (cudaGridDependencySynchronize), so every launch carries the PDL
// attribute — mirrors DeepGEMM's own launch config. The attrs array is
// per-call stack storage (the vendored helper's static array is one of the
// thread-safety hazards this file avoids).
CUresult launch_aot(const void* func, dim3 grid_dim, dim3 block_dim, int smem_size,
                    cudaStream_t stream, void** args) {
    if (smem_size > 0) {
        const cudaError_t attr_err = cudaFuncSetAttribute(
            func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "glm52_deepgemm_mqa: cudaFuncSetAttribute failed: %s\n",
                    cudaGetErrorString(attr_err));
            return CUDA_ERROR_LAUNCH_FAILED;
        }
    }

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = static_cast<size_t>(smem_size);
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;

    const cudaError_t err = cudaLaunchKernelExC(&config, func, args);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm52_deepgemm_mqa: launch failed: %s\n", cudaGetErrorString(err));
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    return CUDA_SUCCESS;
}

} // namespace

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
    // AOT instantiation bounds.
    if (batch_size > kAotAlignedBatchSize || num_sms != kAotNumSms || is_varlen) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int smem_size = kAotAlignedBatchSize * static_cast<int>(sizeof(int));
    static_assert(kAotAlignedBatchSize * sizeof(int) <= kSM90SmemCapacity);

    const uint32_t arg_batch_size = static_cast<uint32_t>(batch_size);
    const uint32_t arg_next_n = static_cast<uint32_t>(next_n);
    const bool arg_is_2d = is_context_lens_2d;
    const uint32_t* arg_context_lens = reinterpret_cast<const uint32_t*>(context_lens);
    const uint32_t* arg_indices = reinterpret_cast<const uint32_t*>(indices_ptr);
    uint32_t* arg_schedule = reinterpret_cast<uint32_t*>(schedule_metadata);
    void* args[] = {
        const_cast<uint32_t*>(&arg_batch_size),
        const_cast<uint32_t*>(&arg_next_n),
        const_cast<bool*>(&arg_is_2d),
        &arg_context_lens,
        &arg_indices,
        &arg_schedule,
    };
    return launch_aot(reinterpret_cast<const void*>(kMetadataKernel),
                      dim3(1, 1, 1), dim3(32, 1, 1), smem_size, stream, args);
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
    // AOT instantiation bounds.
    if (next_n != kAotNextN || num_heads != kAotNumHeads || block_kv != kAotBlockKv ||
        num_sms != kAotNumSms || is_context_lens_2d || is_varlen) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Indexer cache layout: [block_kv * head_dim fp8 | block_kv * 4 f32] per block.
    // The stride must accommodate both regions.
    const int64_t min_stride = static_cast<int64_t>(block_kv) * (head_dim + 4);
    if (kv_cache_stride_bytes < min_stride) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // Weights are f32 (per-head scaling factors folded with q_scale).
    if (weights_elem_size != static_cast<int>(sizeof(float))) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int split_kv = kSplitKv;
    if (split_kv % kMmaM != 0 || logits_stride % split_kv != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int num_math_warp_groups = split_kv / kMmaM;
    const int num_math_threads = num_math_warp_groups * 128;

    const int next_n_atom = 1;

    // TMA descriptor for q: [batch_size * next_n * num_heads, head_dim] (2D)
    // gmem: inner=head_dim, outer=batch_size*next_n*num_heads
    // smem: inner=head_dim, outer=next_n_atom*num_heads (must cover the
    //   [kHeadDim, kNextN*kNumHeads] tile that tma::copy loads)
    // gmem_outer_stride = head_dim (row stride of q in elements)
    // swizzle_mode = head_dim (128)
    const auto tensor_map_q = deep_gemm::make_tma_2d_desc_raw(
        const_cast<void*>(q), q_elem_size, deep_gemm::DgDtype::Float8_e4m3,
        head_dim, batch_size * next_n * num_heads,
        head_dim, next_n_atom * num_heads,
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

    const uint32_t arg_batch_size = static_cast<uint32_t>(batch_size);
    const uint32_t arg_logits_stride = static_cast<uint32_t>(logits_stride);
    const uint32_t arg_block_table_stride = static_cast<uint32_t>(block_table_stride);
    const uint32_t* arg_context_lens = reinterpret_cast<const uint32_t*>(context_lens);
    cutlass::bfloat16_t* arg_logits = static_cast<cutlass::bfloat16_t*>(logits);
    const uint32_t* arg_block_table = reinterpret_cast<const uint32_t*>(block_table);
    // Non-varlen instantiation: the kernel ignores indices.
    const uint32_t* arg_indices = nullptr;
    const uint32_t* arg_schedule = reinterpret_cast<const uint32_t*>(schedule_meta);
    void* args[] = {
        const_cast<uint32_t*>(&arg_batch_size),
        const_cast<uint32_t*>(&arg_logits_stride),
        const_cast<uint32_t*>(&arg_block_table_stride),
        &arg_context_lens,
        &arg_logits,
        &arg_block_table,
        &arg_indices,
        &arg_schedule,
        const_cast<CUtensorMap*>(&tensor_map_q),
        const_cast<CUtensorMap*>(&tensor_map_kv),
        const_cast<CUtensorMap*>(&tensor_map_kv_scales),
        const_cast<CUtensorMap*>(&tensor_map_weights),
    };
    (void)indices;
    (void)is_varlen;
    return launch_aot(reinterpret_cast<const void*>(kLogitsKernel),
                      dim3(static_cast<unsigned>(num_sms), 1, 1),
                      dim3(static_cast<unsigned>(kNumSpecializedThreads + num_math_threads), 1, 1),
                      smem_size, stream, args);
}

} // extern "C"

#else // !GLM52_DEEPGEMM_MQA_SM90A

extern "C" {

CUresult glm52_deepgemm_paged_mqa_metadata_cuda(
    int*, int*, int, int, int, int, bool, bool, const int*, cudaStream_t) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult glm52_deepgemm_paged_mqa_logits_cuda(
    const void*, const void*, int64_t, const void*, const int*, void*,
    const int*, const int*, int*, int, int, int, int, int, int, bool, bool,
    int, int, int, int, int, int, int, cudaStream_t) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

} // extern "C"

#endif // GLM52_DEEPGEMM_MQA_SM90A
