#include "../common.cuh"

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>

#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu"
#include "cpp/common/stringUtils.cpp"
#include "cpp/common/tllmException.cpp"
#include "cpp/common/logger.cpp"

namespace {

namespace trtllm_fp8 =
    tensorrt_llm::kernels::fp8_blockscale_gemm;

using Glm52TrtllmGroupedRunner =
    trtllm_fp8::CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3,
                                               __nv_fp8_e4m3,
                                               __nv_bfloat16>;

constexpr int kKindW13 = 1;
constexpr int kKindW2 = 2;
constexpr int kW13N = 4096;
constexpr int kW13K = 6144;
constexpr int kW13WeightScaleRows = 32;
constexpr int kW13ScaleCols = 48;
constexpr int kW2N = 6144;
constexpr int kW2K = 2048;
constexpr int kW2WeightScaleRows = 48;
constexpr int kW2ScaleCols = 16;

Glm52TrtllmGroupedRunner& runner_for_thread() {
  thread_local Glm52TrtllmGroupedRunner runner;
  return runner;
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

bool valid_w13(int n, int k) { return n == kW13N && k == kW13K; }

bool valid_w2(int n, int k) { return n == kW2N && k == kW2K; }

bool valid_shape(int operand_kind, int groups, int m_capacity, int n, int k) {
  if (groups <= 0 || m_capacity <= 0) {
    return false;
  }
  if (operand_kind == kKindW13) return valid_w13(n, k);
  if (operand_kind == kKindW2) return valid_w2(n, k);
  return false;
}

int div_up_int(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

CUresult workspace_size_checked(int operand_kind, int groups, int m_capacity,
                                int n, int k, size_t* workspace_bytes) {
  if (workspace_bytes == nullptr ||
      !valid_shape(operand_kind, groups, m_capacity, n, k)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  try {
    auto& runner = runner_for_thread();
    *workspace_bytes = runner.getWorkspaceSizeBase(
        static_cast<size_t>(m_capacity), static_cast<size_t>(n),
        static_cast<size_t>(k), static_cast<size_t>(groups));
    return CUDA_SUCCESS;
  } catch (const std::exception&) {
    return CUDA_ERROR_NOT_SUPPORTED;
  } catch (...) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
}

}  // namespace

extern "C" {

CUresult glm52_trtllm_grouped_fp8_workspace_size_cuda(
    int operand_kind, int groups, int m_capacity, int n, int k,
    size_t* workspace_bytes) {
  return workspace_size_checked(operand_kind, groups, m_capacity, n, k,
                                workspace_bytes);
}

CUresult glm52_trtllm_grouped_fp8_launch_cuda(
    int operand_kind, const unsigned char* a, const float* a_scale_trtllm,
    const unsigned char* b, const float* b_scale,
    const int64_t* expert_offsets, unsigned short* out, void* workspace,
    size_t workspace_bytes, int groups, int m_capacity, int n, int k,
    cudaStream_t stream) {
  if (a == nullptr || a_scale_trtllm == nullptr || b == nullptr ||
      b_scale == nullptr || expert_offsets == nullptr || out == nullptr ||
      !valid_shape(operand_kind, groups, m_capacity, n, k)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t required_workspace = 0;
  CUresult workspace_status =
      workspace_size_checked(operand_kind, groups, m_capacity, n, k,
                             &required_workspace);
  if (workspace_status != CUDA_SUCCESS) {
    return workspace_status;
  }
  if (required_workspace != 0 &&
      (workspace == nullptr || workspace_bytes < required_workspace)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    auto& runner = runner_for_thread();
    runner.configureWorkspace(reinterpret_cast<char*>(workspace));
    runner.moeGemm(reinterpret_cast<void*>(out),
                   reinterpret_cast<const void*>(a),
                   reinterpret_cast<const void*>(b), expert_offsets,
                   static_cast<size_t>(groups), static_cast<size_t>(n),
                   static_cast<size_t>(k), stream, a_scale_trtllm, b_scale);
    return consume_last_cuda_error();
  } catch (const std::exception&) {
    return CUDA_ERROR_NOT_SUPPORTED;
  } catch (...) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
}

}  // extern "C"
