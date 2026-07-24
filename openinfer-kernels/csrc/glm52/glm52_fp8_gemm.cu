#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#ifdef GLM52_FP8_GEMM_SM100A
#include <flashinfer/gemm/gemm_groupwise_sm100.cuh>
#endif

namespace {

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

extern "C" CUresult glm52_fp8_groupwise_gemm_sm100_cuda(
    const unsigned char* activation, const float* activation_scale,
    const unsigned char* weight, const float* weight_scale,
    __nv_bfloat16* output, void* workspace, size_t workspace_bytes, int m,
    int n, int k, CUstream stream) {
  if (!activation || !activation_scale || !weight || !weight_scale || !output ||
      !workspace || workspace_bytes == 0 || m <= 0 || n <= 0 || k <= 0 ||
      (m % 4) != 0 || (k % 128) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  OPENINFER_FFI_GUARD_BEGIN
#ifdef GLM52_FP8_GEMM_SM100A
  auto status =
      flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100<
          1, 128, 128, true, 2, cutlass::float_e4m3_t,
          cutlass::bfloat16_t>(
          workspace, workspace_bytes,
          reinterpret_cast<cutlass::float_e4m3_t*>(
              const_cast<unsigned char*>(activation)),
          reinterpret_cast<cutlass::float_e4m3_t*>(
              const_cast<unsigned char*>(weight)),
          const_cast<float*>(activation_scale),
          const_cast<float*>(weight_scale),
          reinterpret_cast<cutlass::bfloat16_t*>(output), m, n, k, 1,
          reinterpret_cast<cudaStream_t>(stream));
  return map_cuda_error(status);
#else
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

extern "C" CUresult glm52_fp8_groupwise_batched_gemm_sm100_cuda(
    const unsigned char* activation, const float* activation_scale,
    const unsigned char* weight, const float* weight_scale,
    __nv_bfloat16* output, void* workspace, size_t workspace_bytes, int m,
    int n, int k, int batch, CUstream stream) {
  if (!activation || !activation_scale || !weight || !weight_scale || !output ||
      !workspace || workspace_bytes == 0 || m <= 0 || n <= 0 || k <= 0 ||
      batch <= 0 || (m % 4) != 0 || (k % 128) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  OPENINFER_FFI_GUARD_BEGIN
#ifdef GLM52_FP8_GEMM_SM100A
  auto status =
      flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100<
          1, 128, 128, true, 2, cutlass::float_e4m3_t,
          cutlass::bfloat16_t>(
          workspace, workspace_bytes,
          reinterpret_cast<cutlass::float_e4m3_t*>(
              const_cast<unsigned char*>(activation)),
          reinterpret_cast<cutlass::float_e4m3_t*>(
              const_cast<unsigned char*>(weight)),
          const_cast<float*>(activation_scale),
          const_cast<float*>(weight_scale),
          reinterpret_cast<cutlass::bfloat16_t*>(output), m, n, k, batch,
          reinterpret_cast<cudaStream_t>(stream));
  return map_cuda_error(status);
#else
  return CUDA_ERROR_NOT_SUPPORTED;
#endif
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}
