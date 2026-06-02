#include <cuda_bf16.h>
#include <cublas_v2.h>

static constexpr int CUBLAS_STATUS_ERROR_OFFSET = 100000;

static int cublas_status_to_error(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaSuccess);
  }
  return CUBLAS_STATUS_ERROR_OFFSET + static_cast<int>(status);
}

// cuBLAS handle management.
// Make handles thread-local so each TP rank thread can bind a handle to its own
// CUDA context/device without racing on a process-global singleton.
thread_local cublasHandle_t g_cublas_handle = nullptr;
thread_local cublasHandle_t g_cublas_prefill_handle = nullptr;
thread_local void *g_cublas_workspace = nullptr;
static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB

extern "C" {

int cuda_set_device(int device_ordinal) { return static_cast<int>(cudaSetDevice(device_ordinal)); }

void cublas_init() {
  if (g_cublas_handle == nullptr) {
    cublasCreate(&g_cublas_handle);
    cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
  }
  if (g_cublas_prefill_handle == nullptr) {
    cublasCreate(&g_cublas_prefill_handle);
    cublasSetMathMode(g_cublas_prefill_handle, CUBLAS_TENSOR_OP_MATH);
    cudaMalloc(&g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
    cublasSetWorkspace(g_cublas_prefill_handle, g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
  }
}

void cublas_destroy() {
  if (g_cublas_handle != nullptr) {
    cublasDestroy(g_cublas_handle);
    g_cublas_handle = nullptr;
  }
  if (g_cublas_prefill_handle != nullptr) {
    cublasDestroy(g_cublas_prefill_handle);
    g_cublas_prefill_handle = nullptr;
  }
  if (g_cublas_workspace != nullptr) {
    cudaFree(g_cublas_workspace);
    g_cublas_workspace = nullptr;
  }
}


// General GEMM: Y = W @ X where W is [M, K] row-major, X is [K, N] col-major, Y is [M, N] col-major
// N=1 is equivalent to GEMV. N>1 enables batched prefill.
// Uses prefill handle (with workspace) — only called from prefill path, never under CUDA Graphs.
int gemm_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
              int M, int N, int K, cudaStream_t stream) {
  if (g_cublas_prefill_handle == nullptr) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_prefill_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  status = cublasGemmEx(g_cublas_prefill_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        M, N, K,
                        &h_alpha,
                        W, CUDA_R_16BF, K,
                        X, CUDA_R_16BF, K,
                        &h_beta,
                        Y, CUDA_R_16BF, M,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

// Graph-safe GEMM: same math as gemm_cuda but uses the workspace-free handle.
// Safe for CUDA Graph capture and decode path.
int gemm_graphsafe_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
                         int M, int N, int K, cudaStream_t stream) {
  if (g_cublas_handle == nullptr) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  status = cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        M, N, K,
                        &h_alpha,
                        W, CUDA_R_16BF, K,
                        X, CUDA_R_16BF, K,
                        &h_beta,
                        Y, CUDA_R_16BF, M,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

// Batched per-token GEMM: each row is computed as the same N=1 GEMM used by
// the decode path. This preserves row-wise numerical parity with the serial
// prompt_len=1 path while keeping a batch-shaped Rust API.
int gemm_per_token_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X,
                                 __nv_bfloat16 *Y, int M, int batch, int K,
                                 cudaStream_t stream) {
  if (g_cublas_handle == nullptr) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }
  if (M <= 0 || batch <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  for (int row = 0; row < batch; ++row) {
    status = cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          M, 1, K,
                          &h_alpha,
                          W, CUDA_R_16BF, K,
                          X + static_cast<int64_t>(row) * K, CUDA_R_16BF, K,
                          &h_beta,
                          Y + static_cast<int64_t>(row) * M, CUDA_R_16BF, M,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return cublas_status_to_error(status);
    }
  }
  return static_cast<int>(cudaPeekAtLastError());
}

} // extern "C"
