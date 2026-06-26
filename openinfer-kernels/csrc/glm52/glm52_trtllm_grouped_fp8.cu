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
constexpr int kLocalExperts = 32;
constexpr int kMCapacity = 10240;
constexpr int kTrtllmOffsetScaleRows = 11232;
constexpr int kW13N = 4096;
constexpr int kW13K = 6144;
constexpr int kW13WeightScaleRows = 32;
constexpr int kW13ScaleCols = 48;
constexpr int kW2N = 6144;
constexpr int kW2K = 2048;
constexpr int kW2WeightScaleRows = 48;
constexpr int kW2ScaleCols = 16;
constexpr int kLinearBatchCapacity = 128;

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
  if (groups != kLocalExperts || m_capacity != kMCapacity) {
    return false;
  }
  if (operand_kind == kKindW13) return valid_w13(n, k);
  if (operand_kind == kKindW2) return valid_w2(n, k);
  return false;
}

int div_up_int(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

bool valid_glm52_linear_shape(int n, int k) {
  if (n <= 0 || k <= 0 || k % 128 != 0) return false;
  // Attention projections.
  if (n == 2048 && k == 6144) return true;    // q_a / shared gate/up
  if (n == 65536 && k == 2048) return true;   // q_b
  if (n == 576 && k == 6144) return true;     // kv_a + rope
  if (n == 114688 && k == 512) return true;   // kv_b
  if (n == 6144 && k == 16384) return true;   // o_proj
  if (n == 128 && k == 6144) return true;     // indexer wk
  if (n == 4096 && k == 2048) return true;    // indexer wq_b
  // Dense MLP projections.
  if (n == 12288 && k == 6144) return true;   // dense gate/up
  if (n == 6144 && k == 12288) return true;   // dense down
  // Shared expert projections.
  if (n == 6144 && k == 2048) return true;    // shared down
  return false;
}

bool valid_linear_contract(int m, int n, int k, int weight_scale_rows,
                           int weight_scale_cols, int activation_scale_cols) {
  return m > 0 && m <= kLinearBatchCapacity && valid_glm52_linear_shape(n, k) &&
         weight_scale_rows == div_up_int(n, 128) &&
         weight_scale_cols == div_up_int(k, 128) &&
         activation_scale_cols == div_up_int(k, 128);
}

bool valid_contract(int operand_kind, int groups, int m_capacity, int n, int k,
                    int weight_scale_rows, int weight_scale_cols,
                    int activation_scale_cols,
                    int activation_scale_trtllm_rows) {
  if (!valid_shape(operand_kind, groups, m_capacity, n, k) ||
      activation_scale_trtllm_rows != kTrtllmOffsetScaleRows) {
    return false;
  }
  if (operand_kind == kKindW13) {
    return weight_scale_rows == kW13WeightScaleRows &&
           weight_scale_cols == kW13ScaleCols &&
           activation_scale_cols == kW13ScaleCols;
  }
  return weight_scale_rows == kW2WeightScaleRows &&
         weight_scale_cols == kW2ScaleCols &&
         activation_scale_cols == kW2ScaleCols;
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

CUresult glm52_trtllm_grouped_fp8_contract_cuda(
    int operand_kind, int groups, int m_capacity, int n, int k,
    int weight_scale_rows, int weight_scale_cols, int activation_scale_cols,
    int activation_scale_trtllm_rows) {
  if (!valid_contract(operand_kind, groups, m_capacity, n, k,
                      weight_scale_rows, weight_scale_cols,
                      activation_scale_cols, activation_scale_trtllm_rows)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

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

CUresult glm52_trtllm_fp8_linear_contract_cuda(
    int m, int n, int k, int weight_scale_rows, int weight_scale_cols,
    int activation_scale_cols) {
  if (!valid_linear_contract(m, n, k, weight_scale_rows, weight_scale_cols,
                             activation_scale_cols)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

CUresult glm52_trtllm_fp8_linear_workspace_size_cuda(int m, int n, int k,
                                                     size_t* workspace_bytes) {
  if (workspace_bytes == nullptr ||
      !valid_linear_contract(m, n, k, div_up_int(n, 128), div_up_int(k, 128),
                             div_up_int(k, 128))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  try {
    auto& runner = runner_for_thread();
    *workspace_bytes =
        runner.getWorkspaceSizeBase(static_cast<size_t>(m), static_cast<size_t>(n),
                                    static_cast<size_t>(k), 1);
    return CUDA_SUCCESS;
  } catch (const std::exception&) {
    return CUDA_ERROR_NOT_SUPPORTED;
  } catch (...) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
}

CUresult glm52_trtllm_fp8_linear_launch_cuda(
    const unsigned char* a, const float* a_scale, const unsigned char* b,
    const float* b_scale, unsigned short* out, void* workspace,
    size_t workspace_bytes, int m, int n, int k, cudaStream_t stream) {
  if (a == nullptr || a_scale == nullptr || b == nullptr || b_scale == nullptr ||
      out == nullptr || !valid_glm52_linear_shape(n, k) || m <= 0 ||
      m > kLinearBatchCapacity) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t required_workspace = 0;
  CUresult workspace_status =
      glm52_trtllm_fp8_linear_workspace_size_cuda(m, n, k, &required_workspace);
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
    runner.gemm(reinterpret_cast<const __nv_fp8_e4m3*>(a), k,
                reinterpret_cast<const __nv_fp8_e4m3*>(b), k,
                reinterpret_cast<__nv_bfloat16*>(out), n, m, n, k, a_scale,
                b_scale, stream);
    return consume_last_cuda_error();
  } catch (const std::exception&) {
    return CUDA_ERROR_NOT_SUPPORTED;
  } catch (...) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
}

}  // extern "C"
