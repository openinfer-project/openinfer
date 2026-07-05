// GLM5.2 EP8 routed-expert GEMM: DeepGEMM SM90 MGroupedMasked fp8 blockscale,
// AOT-instantiated from the vendored device headers (no JIT, no torch) — the
// glm52_deepgemm_mqa.cu pattern. Replaces the TRTLLM grouped GEMM whose
// 64-row M-tile against ~1-8 real rows/expert measured 1.5-1.9x slower on the
// same data/distribution (jz-38 H200 standalone A/B, 2026-07-05), and whose
// activation-scale TMA relayout kernel the masked layout makes unnecessary.
//
// Masked layout contract (matches DeepGEMM's m_grouped masked host wrapper):
//   activation  [kMaskedGroups, kMaskedCap, k]           fp8, fixed stride
//   act scales  [kMaskedGroups, k/128, kMaskedCap]       f32, mn-major
//   weight      [kMaskedGroups, n, k]                    fp8 (bank as-is)
//   wt scales   [kMaskedGroups, n/128, k/128]            f32 (checkpoint as-is)
//   masked_m    i32[kMaskedGroups]                       real rows per expert
//   out         [kMaskedGroups, kMaskedCap, n]           bf16
//
// The metadata kernel bridges the DeepEP aligned-segment recv layout to the
// masked layout: alongside the segment-start offsets it emits masked_m and a
// row_map (aligned recv row -> masked slot, -1 on alignment-gap rows) that the
// masked quant/SiLU kernels and the out-remap kernel index through.
//
// Instantiation configs are the ones vLLM's DeepGEMM JIT picked for these
// exact shapes on H200 (read from its kernel cache): BLOCK_M=64, W13
// BLOCK_N=128/8 stages, W2 BLOCK_N=192/6 stages, TMA multicast 2 on B,
// 132 persistent SMs. Requires sm_90a; without it the GEMM entry compiles as
// a NOT_SUPPORTED stub (metadata/remap stay real — plain CUDA).

#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kExpertAlignment = 64;
constexpr int kMetadataThreads = 32;

// EP8 masked-layout constants (baked into the AOT instantiation).
constexpr int kMaskedGroups = 32;
constexpr int kMaskedCap = 64;

constexpr int kKindW13 = 1;
constexpr int kKindW2 = 2;
constexpr int kW13N = 4096;
constexpr int kW13K = 6144;
constexpr int kW2N = 6144;
constexpr int kW2K = 2048;

__device__ __forceinline__ int align_up_int(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ int clamp_nonnegative(int value) {
  return value < 0 ? 0 : value;
}

__global__ void deepgemm_grouped_fp8_metadata_kernel(
    const int* __restrict__ psum_expert,
    int64_t* __restrict__ expert_offsets, int* __restrict__ masked_m,
    int* __restrict__ row_map, int groups, int m_capacity,
    int expert_alignment, int masked_cap) {
  int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= groups) {
    return;
  }

  int previous_end =
      expert == 0 ? 0 : clamp_nonnegative(psum_expert[expert - 1]);
  int end = clamp_nonnegative(psum_expert[expert]);
  int start = expert == 0 ? 0 : align_up_int(previous_end, expert_alignment);
  int count = end - start;

  // m_capacity is the host-derived row bound (from the coordinator's global
  // token count): the quant kernels covered exactly [0, m_capacity). A
  // segment past it means the ranks disagreed about the token count — the
  // grouped GEMM would multiply stale activations from the previous layer
  // into real outputs with no error anywhere downstream. Crash instead.
  // Likewise a segment longer than the masked capacity would alias the next
  // expert's masked rows.
  if (start > m_capacity || align_up_int(end, expert_alignment) > m_capacity ||
      count < 0 || count > masked_cap) {
    __trap();
  }

  expert_offsets[expert] = static_cast<int64_t>(start);
  if (expert == groups - 1) {
    expert_offsets[groups] =
        static_cast<int64_t>(align_up_int(end, expert_alignment));
  }
  masked_m[expert] = count;

  // Aligned recv row -> masked slot for this expert's segment, -1 across the
  // trailing alignment gap (rows the quant kernels must skip).
  for (int r = 0; r < count; ++r) {
    row_map[start + r] = expert * masked_cap + r;
  }
  int gap_end = align_up_int(end, expert_alignment);
  for (int r = end; r < gap_end; ++r) {
    row_map[r] = -1;
  }
}

// Masked GEMM output -> the DeepEP aligned-segment slots decode_combine
// addresses. Capacity-shaped grid (graph-stable); blocks past a segment's
// real row count retire immediately.
__global__ void masked_out_to_aligned_kernel(
    const __nv_bfloat16* __restrict__ masked_out,
    const int* __restrict__ masked_m, const int64_t* __restrict__ offsets,
    __nv_bfloat16* __restrict__ aligned_out, int n) {
  const int g = blockIdx.x;
  const int r = blockIdx.y;
  if (r >= masked_m[g]) {
    return;
  }
  const uint2* src = reinterpret_cast<const uint2*>(
      masked_out + ((size_t)g * kMaskedCap + r) * n);
  uint2* dst = reinterpret_cast<uint2*>(
      aligned_out + ((size_t)offsets[g] + r) * n);
  const int words = n / 4;  // n is a multiple of 4 (6144)
  for (int i = threadIdx.x; i < words; i += blockDim.x) {
    dst[i] = src[i];
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

}  // namespace

extern "C" {

CUresult glm52_deepgemm_grouped_fp8_metadata_cuda(
    const int* psum_expert, int64_t* expert_offsets, int* masked_m,
    int* row_map, int groups, int m_capacity, int expert_alignment,
    int masked_cap, cudaStream_t stream) {
  if (psum_expert == nullptr || expert_offsets == nullptr ||
      masked_m == nullptr || row_map == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (groups <= 0 || m_capacity <= 0 || expert_alignment != kExpertAlignment ||
      masked_cap != kMaskedCap) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int blocks = (groups + kMetadataThreads - 1) / kMetadataThreads;
  deepgemm_grouped_fp8_metadata_kernel<<<blocks, kMetadataThreads, 0, stream>>>(
      psum_expert, expert_offsets, masked_m, row_map, groups, m_capacity,
      expert_alignment, masked_cap);
  return consume_last_cuda_error();
}

CUresult glm52_deepgemm_masked_out_to_aligned_cuda(
    const __nv_bfloat16* masked_out, const int* masked_m,
    const int64_t* expert_offsets, __nv_bfloat16* aligned_out, int n,
    cudaStream_t stream) {
  if (masked_out == nullptr || masked_m == nullptr ||
      expert_offsets == nullptr || aligned_out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (n <= 0 || n % 4 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  masked_out_to_aligned_kernel<<<dim3(kMaskedGroups, kMaskedCap), 256, 0,
                                 stream>>>(masked_out, masked_m,
                                           expert_offsets, aligned_out, n);
  return consume_last_cuda_error();
}

}  // extern "C"

#ifdef GLM52_DEEPGEMM_GROUPED_SM90A

#include <jit_kernels/impls/runtime_utils.hpp>

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

namespace {

constexpr int kNumSms = 132;
constexpr int kSmemCapacity = 232448;

// Both instantiations share BLOCK_M=64/BLOCK_K=128, 128B swizzles, 128+128
// threads, TMA multicast 2 on B — only BLOCK_N and the stage count differ
// (each shape's max stages within the 228KB smem budget).
template <uint32_t N, uint32_t K, uint32_t BN, uint32_t STAGES>
struct MaskedGemmAot {
  static constexpr uint32_t kBlockN = BN;
  static constexpr uint32_t kShapeN = N;
  static constexpr uint32_t kShapeK = K;
  static constexpr auto kKernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
      cute::UMMA::Major::K,
      /*SHAPE_M=*/0, N, K,
      kMaskedGroups,
      /*BLOCK_M=*/64, BN, /*BLOCK_K=*/128,
      /*swizzle A/B/D=*/128, 128, 128,
      STAGES,
      /*TMA threads=*/128, /*math threads=*/128,
      /*multicast=*/2, /*multicast on A=*/false,
      kNumSms, deep_gemm::GemmType::MGroupedMasked,
      cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity>;

  // Mirrors SM90ArchSpec::get_pipeline_config for this shape.
  static constexpr int smem_size() {
    const int smem_cd = deep_gemm::align<int>(64 * BN * 2, 1024);
    const int smem_barriers = 16 * 8 * 2;
    const int per_stage = 64 * 128 + BN * 128 + deep_gemm::align<int>(64 * 4, 128);
    const int use_uniform_sfb = (128 % BN == 0) ? 1 : 2;
    const int smem_extra_sfb =
        deep_gemm::align<int>((K / 128) * 4 * use_uniform_sfb, 8);
    return smem_cd + smem_barriers + smem_extra_sfb +
           static_cast<int>(STAGES) * per_stage;
  }
};

using MaskedW13 = MaskedGemmAot<kW13N, kW13K, 128, 8>;
using MaskedW2 = MaskedGemmAot<kW2N, kW2K, 192, 6>;
static_assert(MaskedW13::smem_size() <= kSmemCapacity);
static_assert(MaskedW2::smem_size() <= kSmemCapacity);

template <typename Gemm>
CUresult launch_masked_aot(const unsigned char* a, const float* a_scale,
                           const unsigned char* b, const float* b_scale,
                           const int* masked_m, unsigned short* out,
                           cudaStream_t stream) {
  const auto func = reinterpret_cast<const void*>(Gemm::kKernel);
  const int smem_size = Gemm::smem_size();
  const cudaError_t attr_err = cudaFuncSetAttribute(
      func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if (attr_err != cudaSuccess) {
    return map_cuda_error(attr_err);
  }

  const uint32_t n = Gemm::kShapeN, k = Gemm::kShapeK;

  // TMA descriptors mirror DeepGEMM's sm90_m_grouped_fp8_gemm_masked_1d2d
  // host wrapper. Built per launch on the host — a whole-step graph capture
  // bakes them into the recorded node params; pointers are the persistent
  // per-rank state buffers, so replay stays valid.
  const auto tma_a = deep_gemm::make_tma_2d_desc_raw(
      const_cast<unsigned char*>(a), 1, deep_gemm::DgDtype::Float8_e4m3,
      k, kMaskedCap * kMaskedGroups, 128, 64, k, 128);
  const auto tma_b = deep_gemm::make_tma_2d_desc_raw(
      const_cast<unsigned char*>(b), 1, deep_gemm::DgDtype::Float8_e4m3,
      k, n * kMaskedGroups, 128, Gemm::kBlockN, k, 128);
  const auto tma_d = deep_gemm::make_tma_2d_desc_raw(
      out, 2, deep_gemm::DgDtype::BFloat16,
      n, kMaskedCap * kMaskedGroups, 64, 64, n, 128);
  const auto tma_sfa = deep_gemm::make_tma_2d_desc_raw(
      const_cast<float*>(a_scale), 4, deep_gemm::DgDtype::Float,
      kMaskedCap, (k / 128) * kMaskedGroups, 64, 1, kMaskedCap, 0);

  // Cluster 2 (the B-side TMA multicast pair) + PDL, per DeepGEMM's own
  // launch config. The attrs array is per-call stack storage.
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(kNumSms, 1, 1);
  config.blockDim = dim3(128 + 128, 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(smem_size);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = 2;

  uint32_t shape_m = kMaskedCap, shape_n = n, shape_k = k;
  float* sfb = const_cast<float*>(b_scale);
  int* grouped_layout = const_cast<int*>(masked_m);
  void* args[] = {
      &sfb, &grouped_layout, &shape_m, &shape_n, &shape_k,
      const_cast<CUtensorMap*>(&tma_a), const_cast<CUtensorMap*>(&tma_b),
      const_cast<CUtensorMap*>(&tma_d), const_cast<CUtensorMap*>(&tma_sfa),
  };
  return map_cuda_error(cudaLaunchKernelExC(&config, func, args));
}

}  // namespace

extern "C" {

CUresult glm52_deepgemm_masked_grouped_fp8_launch_cuda(
    int operand_kind, const unsigned char* a, const float* a_scale,
    const unsigned char* b, const float* b_scale, const int* masked_m,
    unsigned short* out, int n, int k, cudaStream_t stream) {
  if (a == nullptr || a_scale == nullptr || b == nullptr ||
      b_scale == nullptr || masked_m == nullptr || out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (operand_kind == kKindW13 && n == kW13N && k == kW13K) {
    return launch_masked_aot<MaskedW13>(a, a_scale, b, b_scale, masked_m, out,
                                        stream);
  }
  if (operand_kind == kKindW2 && n == kW2N && k == kW2K) {
    return launch_masked_aot<MaskedW2>(a, a_scale, b, b_scale, masked_m, out,
                                       stream);
  }
  return CUDA_ERROR_INVALID_VALUE;
}

}  // extern "C"

#else  // !GLM52_DEEPGEMM_GROUPED_SM90A

extern "C" {

CUresult glm52_deepgemm_masked_grouped_fp8_launch_cuda(
    int /*operand_kind*/, const unsigned char* /*a*/, const float* /*a_scale*/,
    const unsigned char* /*b*/, const float* /*b_scale*/,
    const int* /*masked_m*/, unsigned short* /*out*/, int /*n*/, int /*k*/,
    cudaStream_t /*stream*/) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"

#endif  // GLM52_DEEPGEMM_GROUPED_SM90A
