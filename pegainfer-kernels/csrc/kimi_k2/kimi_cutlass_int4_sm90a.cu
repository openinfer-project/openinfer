#include "../common.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace pegainfer_kimi_cutlass_int4 {

constexpr int kKimiHiddenDim = 7168;
constexpr int kKimiExpertIntermediateDim = 2048;
constexpr int kKimiW1W3OutputDim = 2 * kKimiExpertIntermediateDim;
constexpr int kKimiLocalExperts = 48;
constexpr int kKimiInt4GroupSize = 32;
constexpr size_t kKimiCutlassWorkspaceAlignment = 128;

struct KimiCutlassInt4GroupedWorkspaceSizes {
  size_t problem_sizes_bytes;
  size_t ptr_arrays_bytes;
  size_t stride_arrays_bytes;
  size_t layout_arrays_bytes;
  size_t cutlass_workspace_bytes;
  size_t total_bytes;
  size_t alignment;
};

struct KimiCutlassInt4GroupedLaunchParams {
  const __nv_bfloat16* input;
  const uint8_t* weight_packed_reordered;
  const __nv_bfloat16* weight_scale;
  const uint32_t* expert_indptr;
  __nv_bfloat16* output;
  void* workspace;
  size_t workspace_bytes;
  int routed_tokens;
  int in_dim;
  int out_dim;
  int local_experts;
  int group_size;
  int sm_count;
};

bool kimi_cutlass_common_shape_ok(
    int routed_tokens,
    int in_dim,
    int out_dim,
    int local_experts,
    int group_size) {
  return routed_tokens >= 0 && in_dim > 0 && out_dim > 0 &&
         local_experts == kKimiLocalExperts && group_size == kKimiInt4GroupSize &&
         (in_dim % group_size) == 0;
}

constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

CUresult kimi_cutlass_status_to_cuda(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess:
      return CUDA_SUCCESS;
    case cutlass::Status::kErrorWorkspaceNull:
    case cutlass::Status::kErrorInternal:
    case cutlass::Status::kErrorInvalidProblem:
    case cutlass::Status::kInvalid:
      return CUDA_ERROR_INVALID_VALUE;
    case cutlass::Status::kErrorNotSupported:
      return CUDA_ERROR_NOT_SUPPORTED;
    default:
      return CUDA_ERROR_UNKNOWN;
  }
}

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::int4b_t;
using ElementScale = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using LayoutScale = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

using ValueShuffle = Layout<Shape<_2, _4>, Stride<_4, _1>>;
int constexpr NumShuffleAtoms = 1;
using MmaAtomShape = Layout<Shape<_1, Int<NumShuffleAtoms>>>;
using LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
using LayoutB_Reordered =
    decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int, int, Int<1>>, StrideB>{}));

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128, _16, cute::Int<TileShapeK>>;
using ClusterShape = Shape<_1, _1, _1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
    AlignmentC,
    ElementD,
    typename cutlass::layout::LayoutTranspose<LayoutD>::type*,
    AlignmentD,
    EpilogueSchedule>::CollectiveOp;

using CollectiveMainloopScaleOnlyShuffled =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        cute::tuple<QuantType, ElementScale>,
        LayoutB_Reordered*,
        AlignmentB,
        MmaType,
        LayoutA_Transpose*,
        AlignmentA,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using CollectiveMainloopScaleOnlyStride =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        cute::tuple<QuantType, ElementScale>,
        LayoutB_Transpose*,
        AlignmentB,
        MmaType,
        LayoutA_Transpose*,
        AlignmentA,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernelScaleOnlyShuffled = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopScaleOnlyShuffled,
    CollectiveEpilogue>;

using KimiInt4GroupedGemm =
    cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnlyShuffled>;

using StrideC = typename GemmKernelScaleOnlyShuffled::InternalStrideC;
using StrideD = typename GemmKernelScaleOnlyShuffled::InternalStrideD;
using StrideS = typename CollectiveMainloopScaleOnlyStride::StrideScale;

static_assert(cute::is_same_v<ElementC, cutlass::bfloat16_t>);
static_assert(cute::is_same_v<ElementD, cutlass::bfloat16_t>);
static_assert(kKimiW1W3OutputDim == 4096);

struct KimiCutlassInt4GroupedWorkspaceView {
  typename ProblemShape::UnderlyingProblemShape* problem_sizes;
  const MmaType** ptr_a;
  const QuantType** ptr_b;
  const ElementScale** ptr_scale;
  const ElementC** ptr_c;
  ElementD** ptr_d;
  StrideA* stride_a;
  LayoutB_Reordered* layout_b;
  StrideC* stride_c;
  StrideD* stride_d;
  StrideS* stride_s;
  void* cutlass_workspace;
  size_t cutlass_workspace_bytes;
};

size_t kimi_cutlass_internal_workspace_bytes(int local_experts) {
  KimiInt4GroupedGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {local_experts, nullptr, nullptr},
      {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, kKimiInt4GroupSize},
      {},
      {}};
  return KimiInt4GroupedGemm::get_workspace_size(arguments);
}

KimiCutlassInt4GroupedWorkspaceSizes kimi_cutlass_workspace_sizes_host(int local_experts) {
  KimiCutlassInt4GroupedWorkspaceSizes sizes{};
  sizes.problem_sizes_bytes =
      align_up(sizeof(typename ProblemShape::UnderlyingProblemShape) * local_experts,
               kKimiCutlassWorkspaceAlignment);
  sizes.ptr_arrays_bytes =
      align_up((sizeof(MmaType*) + sizeof(QuantType*) + sizeof(ElementScale*) +
                sizeof(ElementC*) + sizeof(ElementD*)) *
                   local_experts,
               kKimiCutlassWorkspaceAlignment);
  sizes.stride_arrays_bytes =
      align_up((sizeof(StrideA) + sizeof(StrideC) + sizeof(StrideD) + sizeof(StrideS)) *
                   local_experts,
               kKimiCutlassWorkspaceAlignment);
  sizes.layout_arrays_bytes =
      align_up(sizeof(LayoutB_Reordered) * local_experts, kKimiCutlassWorkspaceAlignment);
  sizes.cutlass_workspace_bytes =
      align_up(kimi_cutlass_internal_workspace_bytes(local_experts),
               kKimiCutlassWorkspaceAlignment);
  sizes.total_bytes = sizes.problem_sizes_bytes + sizes.ptr_arrays_bytes +
                      sizes.stride_arrays_bytes + sizes.layout_arrays_bytes +
                      sizes.cutlass_workspace_bytes;
  sizes.alignment = kKimiCutlassWorkspaceAlignment;
  return sizes;
}

KimiCutlassInt4GroupedWorkspaceView kimi_cutlass_workspace_view(
    void* workspace,
    int local_experts) {
  auto sizes = kimi_cutlass_workspace_sizes_host(local_experts);
  uintptr_t cursor = reinterpret_cast<uintptr_t>(workspace);
  KimiCutlassInt4GroupedWorkspaceView view{};
  view.problem_sizes = reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(cursor);
  cursor += sizes.problem_sizes_bytes;

  view.ptr_a = reinterpret_cast<const MmaType**>(cursor);
  cursor += sizeof(MmaType*) * local_experts;
  view.ptr_b = reinterpret_cast<const QuantType**>(cursor);
  cursor += sizeof(QuantType*) * local_experts;
  view.ptr_scale = reinterpret_cast<const ElementScale**>(cursor);
  cursor += sizeof(ElementScale*) * local_experts;
  view.ptr_c = reinterpret_cast<const ElementC**>(cursor);
  cursor += sizeof(ElementC*) * local_experts;
  view.ptr_d = reinterpret_cast<ElementD**>(cursor);
  cursor += sizeof(ElementD*) * local_experts;
  cursor += sizes.ptr_arrays_bytes -
            (sizeof(MmaType*) + sizeof(QuantType*) + sizeof(ElementScale*) +
             sizeof(ElementC*) + sizeof(ElementD*)) *
                local_experts;

  view.stride_a = reinterpret_cast<StrideA*>(cursor);
  cursor += sizeof(StrideA) * local_experts;
  view.stride_c = reinterpret_cast<StrideC*>(cursor);
  cursor += sizeof(StrideC) * local_experts;
  view.stride_d = reinterpret_cast<StrideD*>(cursor);
  cursor += sizeof(StrideD) * local_experts;
  view.stride_s = reinterpret_cast<StrideS*>(cursor);
  cursor += sizeof(StrideS) * local_experts;
  cursor += sizes.stride_arrays_bytes -
            (sizeof(StrideA) + sizeof(StrideC) + sizeof(StrideD) + sizeof(StrideS)) *
                local_experts;

  view.layout_b = reinterpret_cast<LayoutB_Reordered*>(cursor);
  cursor += sizes.layout_arrays_bytes;

  view.cutlass_workspace = reinterpret_cast<void*>(cursor);
  view.cutlass_workspace_bytes = sizes.cutlass_workspace_bytes;
  return view;
}

__global__ void kimi_cutlass_prepare_grouped_projection_kernel(
    KimiCutlassInt4GroupedWorkspaceView view,
    KimiCutlassInt4GroupedLaunchParams params) {
  int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= params.local_experts) return;

  uint32_t start = params.expert_indptr[expert];
  uint32_t end = params.expert_indptr[expert + 1];
  int tokens = static_cast<int>(end - start);
  int scale_k = params.in_dim / params.group_size;

  view.problem_sizes[expert] =
      typename ProblemShape::UnderlyingProblemShape{params.out_dim, tokens, params.in_dim};
  view.ptr_a[expert] =
      reinterpret_cast<const MmaType*>(params.input + static_cast<size_t>(start) * params.in_dim);
  view.ptr_b[expert] = reinterpret_cast<const QuantType*>(
      params.weight_packed_reordered +
      static_cast<size_t>(expert) * params.out_dim * ((params.in_dim + 1) / 2));
  view.ptr_scale[expert] =
      reinterpret_cast<const ElementScale*>(params.weight_scale) +
      static_cast<size_t>(expert) * params.out_dim * scale_k;
  view.ptr_c[expert] =
      reinterpret_cast<const ElementC*>(params.output + static_cast<size_t>(start) * params.out_dim);
  view.ptr_d[expert] =
      reinterpret_cast<ElementD*>(params.output + static_cast<size_t>(start) * params.out_dim);

  view.stride_a[expert] =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(tokens, params.in_dim, 1));
  view.layout_b[expert] = cute::tile_to_shape(
      LayoutAtomQuant{}, cute::make_shape(params.out_dim, params.in_dim, Int<1>{}));
  view.stride_c[expert] =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(params.out_dim, tokens, 1));
  view.stride_d[expert] =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(params.out_dim, tokens, 1));
  view.stride_s[expert] =
      cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(params.out_dim, scale_k, 1));
}

KimiInt4GroupedGemm::Arguments kimi_cutlass_arguments(
    const KimiCutlassInt4GroupedWorkspaceView& view,
    const KimiCutlassInt4GroupedLaunchParams& params) {
  KimiInt4GroupedGemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  int current_device = 0;
  cudaGetDevice(&current_device);
  hw_info.device_id = current_device;
  hw_info.sm_count = params.sm_count > 0
                         ? params.sm_count
                         : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                               hw_info.device_id);

  arguments = KimiInt4GroupedGemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {params.local_experts, view.problem_sizes, nullptr},
      {view.ptr_b,
       view.layout_b,
       view.ptr_a,
       view.stride_a,
       view.ptr_scale,
       view.stride_s,
       params.group_size},
      {fusion_args, view.ptr_c, view.stride_c, view.ptr_d, view.stride_d},
      hw_info};
  return arguments;
}

__global__ void kimi_cutlass_xor_int4_offset_binary_kernel(uint8_t* data, size_t bytes) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (; idx < bytes; idx += stride) {
    data[idx] ^= 0x88u;
  }
}

__global__ void kimi_cutlass_reorder_scale_kernel(
    const __nv_bfloat16* scale_checkpoint,
    __nv_bfloat16* scale_reordered,
    int out_dim,
    int scale_k,
    size_t total_elements) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  size_t elements_per_expert = static_cast<size_t>(out_dim) * scale_k;
  for (; idx < total_elements; idx += stride) {
    size_t in_expert = idx % elements_per_expert;
    size_t expert_base = idx - in_expert;
    int row = static_cast<int>(in_expert / scale_k);
    int group = static_cast<int>(in_expert - static_cast<size_t>(row) * scale_k);
    scale_reordered[expert_base + static_cast<size_t>(group) * out_dim + row] =
        scale_checkpoint[idx];
  }
}

#endif

__device__ __forceinline__ int kimi_marlin_scale_perm_64(int offset) {
  return (offset / 8) + 8 * (offset % 8);
}

__global__ void kimi_marlin_reorder_scale_kernel(
    const __nv_bfloat16* scale_checkpoint,
    __nv_bfloat16* scale_marlin,
    int out_dim,
    int scale_k,
    size_t total_elements) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  size_t elements_per_expert = static_cast<size_t>(out_dim) * scale_k;
  for (; idx < total_elements; idx += stride) {
    size_t in_expert = idx % elements_per_expert;
    size_t expert_base = idx - in_expert;
    size_t block = in_expert / 64;
    int offset = static_cast<int>(in_expert % 64);
    size_t transposed = block * 64 + static_cast<size_t>(kimi_marlin_scale_perm_64(offset));
    int group = static_cast<int>(transposed / out_dim);
    int row = static_cast<int>(transposed - static_cast<size_t>(group) * out_dim);
    scale_marlin[idx] = scale_checkpoint[expert_base + static_cast<size_t>(row) * scale_k + group];
  }
}

}  // namespace pegainfer_kimi_cutlass_int4

using namespace pegainfer_kimi_cutlass_int4;

extern "C" {

CUresult kimi_cutlass_int4_sm90a_support_cuda() {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  return CUDA_SUCCESS;
#else
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

CUresult kimi_cutlass_int4_sm90a_support_probe_cuda(
    int* supported,
    int* sm_major,
    int* sm_minor) {
  if (supported == nullptr || sm_major == nullptr || sm_minor == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *supported = 0;
  *sm_major = 0;
  *sm_minor = 0;
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;
  *sm_major = props.major;
  *sm_minor = props.minor;
  *supported = props.major == 9 && props.minor == 0;
  return *supported ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
#else
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

CUresult kimi_cutlass_int4_grouped_workspace_sizes_sm90a_cuda(
    int max_routed_tokens,
    int in_dim,
    int out_dim,
    int local_experts,
    int group_size,
    KimiCutlassInt4GroupedWorkspaceSizes* sizes) {
  if (sizes == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *sizes = KimiCutlassInt4GroupedWorkspaceSizes{};
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (!kimi_cutlass_common_shape_ok(max_routed_tokens, in_dim, out_dim, local_experts, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *sizes = kimi_cutlass_workspace_sizes_host(local_experts);
  return CUDA_SUCCESS;
#else
  (void)max_routed_tokens;
  (void)in_dim;
  (void)out_dim;
  (void)local_experts;
  (void)group_size;
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

CUresult kimi_cutlass_int4_grouped_prepare_sm90a_cuda(
    KimiCutlassInt4GroupedLaunchParams params,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (params.input == nullptr || params.weight_packed_reordered == nullptr ||
      params.weight_scale == nullptr || params.expert_indptr == nullptr ||
      params.output == nullptr || params.workspace == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_cutlass_common_shape_ok(
          params.routed_tokens, params.in_dim, params.out_dim, params.local_experts,
          params.group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto sizes = kimi_cutlass_workspace_sizes_host(params.local_experts);
  if (params.workspace_bytes < sizes.total_bytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (params.routed_tokens == 0) return CUDA_SUCCESS;

  auto view = kimi_cutlass_workspace_view(params.workspace, params.local_experts);
  dim3 block(128);
  dim3 grid((params.local_experts + block.x - 1) / block.x);
  kimi_cutlass_prepare_grouped_projection_kernel<<<grid, block, 0, stream>>>(view, params);
  cudaError_t err = cudaPeekAtLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
#else
  (void)params;
  (void)stream;
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

CUresult kimi_cutlass_int4_grouped_launch_sm90a_cuda(
    KimiCutlassInt4GroupedLaunchParams params,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (params.input == nullptr || params.weight_packed_reordered == nullptr ||
      params.weight_scale == nullptr || params.expert_indptr == nullptr ||
      params.output == nullptr || params.workspace == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_cutlass_common_shape_ok(
          params.routed_tokens, params.in_dim, params.out_dim, params.local_experts,
          params.group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto sizes = kimi_cutlass_workspace_sizes_host(params.local_experts);
  if (params.workspace_bytes < sizes.total_bytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (params.routed_tokens == 0) return CUDA_SUCCESS;

  auto view = kimi_cutlass_workspace_view(params.workspace, params.local_experts);
  auto arguments = kimi_cutlass_arguments(view, params);
  size_t required_workspace = KimiInt4GroupedGemm::get_workspace_size(arguments);
  if (required_workspace > view.cutlass_workspace_bytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  KimiInt4GroupedGemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return kimi_cutlass_status_to_cuda(status);
  }
  status = gemm.initialize(arguments, view.cutlass_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return kimi_cutlass_status_to_cuda(status);
  }
  status = gemm.run(stream);
  return kimi_cutlass_status_to_cuda(status);
#else
  (void)params;
  (void)stream;
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

CUresult kimi_cutlass_int4_reorder_weight_sm90a_cuda(
    const uint8_t* weight_packed_offset_binary,
    uint8_t* weight_packed_reordered,
    int in_dim,
    int out_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (weight_packed_offset_binary == nullptr || weight_packed_reordered == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_cutlass_common_shape_ok(0, in_dim, out_dim, local_experts, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto shape_B = cute::make_shape(out_dim, in_dim, Int<1>{});
  auto layout_B_checkpoint =
      make_layout(shape_B, cute::make_stride(in_dim, Int<1>{}, Int<0>{}));
  auto layout_B_reordered = tile_to_shape(LayoutAtomQuant{}, shape_B);
  size_t bytes_per_expert = static_cast<size_t>(out_dim) * ((static_cast<size_t>(in_dim) + 1) / 2);

  for (int expert = 0; expert < local_experts; ++expert) {
    auto src = reinterpret_cast<const QuantType*>(
        weight_packed_offset_binary + static_cast<size_t>(expert) * bytes_per_expert);
    auto dst = reinterpret_cast<QuantType*>(
        weight_packed_reordered + static_cast<size_t>(expert) * bytes_per_expert);
    // Kimi compressed-tensors weights arrive in checkpoint/vLLM row-major
    // order: [out_dim, in_dim / 2]. CUTLASS example 69 consumes a reordered B
    // operand, so this step changes layout first and the bytewise xor below
    // converts offset-binary nibbles into CUTLASS int4b_t storage.
    cutlass::reorder_tensor(src, layout_B_checkpoint, dst, layout_B_reordered);
  }

  size_t total_bytes = bytes_per_expert * static_cast<size_t>(local_experts);
  dim3 block(256);
  dim3 grid(static_cast<unsigned>((total_bytes + block.x - 1) / block.x));
  kimi_cutlass_xor_int4_offset_binary_kernel<<<grid, block, 0, stream>>>(
      weight_packed_reordered, total_bytes);
  cudaError_t err = cudaPeekAtLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
#else
  (void)weight_packed_offset_binary;
  (void)weight_packed_reordered;
  (void)in_dim;
  (void)out_dim;
  (void)local_experts;
  (void)group_size;
  (void)stream;
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

CUresult kimi_cutlass_int4_reorder_scale_sm90a_cuda(
    const __nv_bfloat16* weight_scale_checkpoint,
    __nv_bfloat16* weight_scale_reordered,
    int in_dim,
    int out_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (weight_scale_checkpoint == nullptr || weight_scale_reordered == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_cutlass_common_shape_ok(0, in_dim, out_dim, local_experts, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int scale_k = in_dim / group_size;
  size_t total_elements =
      static_cast<size_t>(local_experts) * static_cast<size_t>(out_dim) * scale_k;
  dim3 block(256);
  dim3 grid(static_cast<unsigned>((total_elements + block.x - 1) / block.x));
  kimi_cutlass_reorder_scale_kernel<<<grid, block, 0, stream>>>(
      weight_scale_checkpoint, weight_scale_reordered, out_dim, scale_k, total_elements);
  cudaError_t err = cudaPeekAtLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
#else
  (void)weight_scale_checkpoint;
  (void)weight_scale_reordered;
  (void)in_dim;
  (void)out_dim;
  (void)local_experts;
  (void)group_size;
  (void)stream;
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

CUresult kimi_marlin_int4_reorder_scale_cuda(
    const __nv_bfloat16* weight_scale_checkpoint,
    __nv_bfloat16* weight_scale_marlin,
    int in_dim,
    int out_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
  if (weight_scale_checkpoint == nullptr || weight_scale_marlin == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_cutlass_common_shape_ok(0, in_dim, out_dim, local_experts, group_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int scale_k = in_dim / group_size;
  size_t elements_per_expert = static_cast<size_t>(out_dim) * scale_k;
  if ((elements_per_expert % 64) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t total_elements = static_cast<size_t>(local_experts) * elements_per_expert;
  dim3 block(256);
  dim3 grid(static_cast<unsigned>((total_elements + block.x - 1) / block.x));
  kimi_marlin_reorder_scale_kernel<<<grid, block, 0, stream>>>(
      weight_scale_checkpoint, weight_scale_marlin, out_dim, scale_k, total_elements);
  cudaError_t err = cudaPeekAtLastError();
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult kimi_cutlass_int4_grouped_w1_w3_sm90a_cuda(
    const __nv_bfloat16* expert_major_hidden,
    const uint8_t* w1_packed,
    const __nv_bfloat16* w1_scale,
    const uint8_t* w3_packed,
    const __nv_bfloat16* w3_scale,
    const uint32_t* expert_indptr,
    __nv_bfloat16* gate_out,
    __nv_bfloat16* up_out,
    int routed_tokens,
    int hidden_dim,
    int intermediate_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
  (void)stream;
  if (expert_major_hidden == nullptr || w1_packed == nullptr || w1_scale == nullptr ||
      w3_packed == nullptr || w3_scale == nullptr || expert_indptr == nullptr ||
      gate_out == nullptr || up_out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_cutlass_common_shape_ok(
          routed_tokens, hidden_dim, 2 * intermediate_dim, local_experts, group_size) ||
      hidden_dim != kKimiHiddenDim || intermediate_dim != kKimiExpertIntermediateDim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (routed_tokens == 0) return CUDA_SUCCESS;

  // This translation unit owns the CUTLASS C++ AOT type instantiation. The
  // runtime launcher still needs the graph-resident ptr/stride/problem scratch
  // package from the Kimi rank weight loader before it may run a grouped GEMM.
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult kimi_cutlass_int4_grouped_w2_sm90a_cuda(
    const __nv_bfloat16* activated,
    const uint8_t* w2_packed,
    const __nv_bfloat16* w2_scale,
    const uint32_t* expert_indptr,
    __nv_bfloat16* expert_output,
    int routed_tokens,
    int intermediate_dim,
    int hidden_dim,
    int local_experts,
    int group_size,
    cudaStream_t stream) {
  (void)stream;
  if (activated == nullptr || w2_packed == nullptr || w2_scale == nullptr ||
      expert_indptr == nullptr || expert_output == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!kimi_cutlass_common_shape_ok(
          routed_tokens, intermediate_dim, hidden_dim, local_experts, group_size) ||
      intermediate_dim != kKimiExpertIntermediateDim || hidden_dim != kKimiHiddenDim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (routed_tokens == 0) return CUDA_SUCCESS;

  return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
