#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <array>
#include <cstddef>
#include <map>

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
thread_local std::map<int, cublasHandle_t> g_cublas_handles_by_device;
thread_local std::map<int, cublasHandle_t> g_cublas_prefill_handles_by_device;
thread_local std::map<int, void *> g_cublas_workspaces_by_device;
static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB

// cublasLt path for small-N decode GEMMs. cuBLAS's default heuristic leaves
// 4-6% bandwidth on the table for these shapes; gemm_lt_tune_cuda times every
// heuristic candidate once at startup and caches the winner per (M, N, K).
struct LtGemmPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

thread_local cublasLtHandle_t g_lt_handle = nullptr;
thread_local void *g_lt_workspace = nullptr;
thread_local std::map<std::array<int, 3>, LtGemmPlan> g_lt_plans;
static const size_t LT_WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB

// Pin-path workspace, separate from the 32MB default. Arch-dependent, with 128MB kept as margin
// for the tested decode buckets. Allocated lazily on first Pin use.
thread_local void *g_lt_pin_workspace = nullptr;
static const size_t LT_PIN_WORKSPACE_SIZE = 128 * 1024 * 1024;
// Tuner pref kept smaller than the buffer: cuBLASLt picks a larger-workspace algo given a larger
// budget, so 64MB pref forces a small-workspace algo and the 128MB buffer stays margin.
static const size_t LT_PIN_TUNER_PREF = 64 * 1024 * 1024;
// gemm_lt_cuda returns this when the calling thread has no tuned plan for the
// shape; callers fall back to the cublasGemmEx paths so untuned models keep
// their existing kernel selection and capture behavior.
static constexpr int GEMM_LT_UNTUNED = -1;

// Keyed on {M,K} only (g_lt_plans uses {M,N,K}): one cublasLt algo chosen at rep_n, reused for every N.
static constexpr int GEMM_LT_PIN_UNTUNED = -1;     // no pinned plan for {M,K}
static constexpr int GEMM_LT_PIN_UNSUPPORTED = -2; // pinned algo cannot serve this N
struct LtPinPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr; // [K, M], independent of N
  cublasLtMatmulAlgo_t algo{};
};
thread_local std::map<std::array<int, 2>, LtPinPlan> g_lt_pin_plans;

static void lt_plan_destroy(LtGemmPlan &plan) {
  if (plan.c != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.c);
  }
  if (plan.b != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.b);
  }
  if (plan.a != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.a);
  }
  if (plan.op != nullptr) {
    cublasLtMatmulDescDestroy(plan.op);
  }
  plan = LtGemmPlan{};
}

static void lt_pin_destroy(LtPinPlan &plan) {
  if (plan.a != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.a);
  }
  if (plan.op != nullptr) {
    cublasLtMatmulDescDestroy(plan.op);
  }
  plan = LtPinPlan{};
}

// Lazily create this thread's cublasLt handle + 32MB workspace (shared by tuner and pin paths).
static int ensure_lt_resources() {
  if (g_lt_handle == nullptr) {
    cublasStatus_t status = cublasLtCreate(&g_lt_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      g_lt_handle = nullptr;
      return cublas_status_to_error(status);
    }
  }
  if (g_lt_workspace == nullptr) {
    cudaError_t status = cudaMalloc(&g_lt_workspace, LT_WORKSPACE_SIZE);
    if (status != cudaSuccess) {
      g_lt_workspace = nullptr;
      return static_cast<int>(status);
    }
  }
  return static_cast<int>(cudaSuccess);
}

static int ensure_lt_pin_workspace() {
  if (g_lt_pin_workspace == nullptr) {
    cudaError_t status = cudaMalloc(&g_lt_pin_workspace, LT_PIN_WORKSPACE_SIZE);
    if (status != cudaSuccess) {
      g_lt_pin_workspace = nullptr;
      return static_cast<int>(status);
    }
  }
  return static_cast<int>(cudaSuccess);
}

// Op descriptor + A layout [K,M] for the pinned path; B/C rebuilt per call from N.
static cublasStatus_t lt_pin_desc_create(LtPinPlan &plan, int M, int K) {
  cublasStatus_t status = cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                          sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                          sizeof(transb));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  return cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, K, M, K);
}

// Same math as gemm_cuda: Y[M,N] = W[M,K]^T-layout @ X[K,N], all bf16/FP32 compute.
static cublasStatus_t lt_plan_create(LtGemmPlan &plan, int M, int N, int K) {
  cublasStatus_t status = cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                          sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                          sizeof(transb));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, K, M, K);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, K, N, K);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  return cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16BF, M, N, M);
}

static cublasStatus_t lt_plan_heuristics(const LtGemmPlan &plan,
                                         cublasLtMatmulHeuristicResult_t *results,
                                         int max_results, int *returned) {
  cublasLtMatmulPreference_t pref = nullptr;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&pref);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }
  size_t ws = LT_WORKSPACE_SIZE;
  status = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &ws, sizeof(ws));
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(g_lt_handle, plan.op, plan.a, plan.b, plan.c,
                                            plan.c, pref, max_results, results, returned);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  if (status == CUBLAS_STATUS_SUCCESS && *returned == 0) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return status;
}


extern "C" {

int cuda_set_device(int device_ordinal) { return static_cast<int>(cudaSetDevice(device_ordinal)); }

void cublas_init() {
  int device = 0;
  cudaGetDevice(&device);

  auto handle_it = g_cublas_handles_by_device.find(device);
  if (handle_it == g_cublas_handles_by_device.end()) {
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    handle_it = g_cublas_handles_by_device.emplace(device, handle).first;
  }
  g_cublas_handle = handle_it->second;

  auto prefill_it = g_cublas_prefill_handles_by_device.find(device);
  if (prefill_it == g_cublas_prefill_handles_by_device.end()) {
    cublasHandle_t handle = nullptr;
    void *workspace = nullptr;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cudaMalloc(&workspace, CUBLAS_WORKSPACE_SIZE);
    cublasSetWorkspace(handle, workspace, CUBLAS_WORKSPACE_SIZE);
    prefill_it = g_cublas_prefill_handles_by_device.emplace(device, handle).first;
    g_cublas_workspaces_by_device.emplace(device, workspace);
  }
  g_cublas_prefill_handle = prefill_it->second;
  g_cublas_workspace = g_cublas_workspaces_by_device[device];
}

int cublas_activate_device_handles() {
  int device = 0;
  cudaError_t cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }

  auto handle_it = g_cublas_handles_by_device.find(device);
  auto prefill_it = g_cublas_prefill_handles_by_device.find(device);
  auto workspace_it = g_cublas_workspaces_by_device.find(device);
  if (handle_it == g_cublas_handles_by_device.end() ||
      prefill_it == g_cublas_prefill_handles_by_device.end() ||
      workspace_it == g_cublas_workspaces_by_device.end()) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }

  g_cublas_handle = handle_it->second;
  g_cublas_prefill_handle = prefill_it->second;
  g_cublas_workspace = workspace_it->second;
  return static_cast<int>(cudaSuccess);
}

void cublas_destroy() {
  for (auto &entry : g_cublas_handles_by_device) {
    if (entry.second != nullptr) {
      cublasDestroy(entry.second);
    }
  }
  g_cublas_handles_by_device.clear();
  for (auto &entry : g_cublas_prefill_handles_by_device) {
    if (entry.second != nullptr) {
      cublasDestroy(entry.second);
    }
  }
  g_cublas_prefill_handles_by_device.clear();
  for (auto &entry : g_cublas_workspaces_by_device) {
    if (entry.second != nullptr) {
      cudaFree(entry.second);
    }
  }
  g_cublas_workspaces_by_device.clear();
  g_cublas_handle = nullptr;
  g_cublas_prefill_handle = nullptr;
  g_cublas_workspace = nullptr;
  for (auto &entry : g_lt_plans) {
    lt_plan_destroy(entry.second);
  }
  g_lt_plans.clear();
  for (auto &entry : g_lt_pin_plans) {
    lt_pin_destroy(entry.second);
  }
  g_lt_pin_plans.clear();
  if (g_lt_handle != nullptr) {
    cublasLtDestroy(g_lt_handle);
    g_lt_handle = nullptr;
  }
  if (g_lt_workspace != nullptr) {
    cudaFree(g_lt_workspace);
    g_lt_workspace = nullptr;
  }
  if (g_lt_pin_workspace != nullptr) {
    cudaFree(g_lt_pin_workspace);
    g_lt_pin_workspace = nullptr;
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

// Decode GEMM through the cublasLt plan tuned by gemm_lt_tune_cuda. Returns
// GEMM_LT_UNTUNED when this thread holds no plan for (M, N, K) — the tuned
// kernel was already executed during tuning, so replaying it inside a CUDA
// Graph capture is safe.
int gemm_lt_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
                 int M, int N, int K, cudaStream_t stream) {
  if (g_lt_handle == nullptr || g_lt_workspace == nullptr) {
    return GEMM_LT_UNTUNED;
  }
  auto it = g_lt_plans.find(std::array<int, 3>{M, N, K});
  if (it == g_lt_plans.end()) {
    return GEMM_LT_UNTUNED;
  }
  const LtGemmPlan &plan = it->second;
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(g_lt_handle, plan.op, &h_alpha,
                                         W, plan.a,
                                         X, plan.b,
                                         &h_beta,
                                         Y, plan.c,
                                         Y, plan.c,
                                         &plan.algo, g_lt_workspace, LT_WORKSPACE_SIZE, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

// Time every heuristic candidate for (M, N, K) and cache the winner for
// gemm_lt_cuda. `Ws` holds several same-shaped weight pointers (different
// layers); rotating them keeps the timing loop out of L2 so the ranking
// matches steady-state decode, where each weight is read cold once per step.
// Must run on the executor thread before graph capture; not capture-safe.
int gemm_lt_tune_cuda(const __nv_bfloat16 *const *Ws, int num_ws, int M, int N, int K,
                      cudaStream_t stream) {
  if (num_ws <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  // Lt resources are created here rather than in cublas_init so only threads
  // that actually tune (model executor threads) pay the 32MB workspace.
  int lt_rc = ensure_lt_resources();
  if (lt_rc != static_cast<int>(cudaSuccess)) {
    return lt_rc;
  }

  const std::array<int, 3> key{M, N, K};
  if (g_lt_plans.find(key) != g_lt_plans.end()) {
    return static_cast<int>(cudaSuccess);
  }

  LtGemmPlan plan;
  cublasStatus_t status = lt_plan_create(plan, M, N, K);
  cublasLtMatmulHeuristicResult_t results[16];
  int returned = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = lt_plan_heuristics(plan, results, 16, &returned);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    lt_plan_destroy(plan);
    return cublas_status_to_error(status);
  }

  __nv_bfloat16 *x = nullptr;
  __nv_bfloat16 *y = nullptr;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  cudaError_t cuda_status = cudaMalloc(&x, static_cast<size_t>(K) * N * sizeof(__nv_bfloat16));
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemset(x, 0, static_cast<size_t>(K) * N * sizeof(__nv_bfloat16));
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMalloc(&y, static_cast<size_t>(M) * N * sizeof(__nv_bfloat16));
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventCreate(&begin);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventCreate(&end);
  }

  int best = -1;
  float best_ms = 0.0f;
  if (cuda_status == cudaSuccess) {
    const float h_alpha = 1.0f;
    const float h_beta = 0.0f;
    const int warmup = 3;
    const int iters = 20;
    for (int i = 0; i < returned; ++i) {
      bool ok = true;
      for (int j = 0; j < warmup + iters && ok; ++j) {
        if (j == warmup) {
          ok = cudaEventRecord(begin, stream) == cudaSuccess;
          if (!ok) {
            break;
          }
        }
        ok = cublasLtMatmul(g_lt_handle, plan.op, &h_alpha,
                            Ws[j % num_ws], plan.a,
                            x, plan.b,
                            &h_beta,
                            y, plan.c,
                            y, plan.c,
                            &results[i].algo, g_lt_workspace, LT_WORKSPACE_SIZE,
                            stream) == CUBLAS_STATUS_SUCCESS;
      }
      if (!ok || cudaEventRecord(end, stream) != cudaSuccess ||
          cudaEventSynchronize(end) != cudaSuccess) {
        continue;
      }
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, begin, end) != cudaSuccess) {
        continue;
      }
      if (best < 0 || ms < best_ms) {
        best = i;
        best_ms = ms;
      }
    }
  }

  if (begin != nullptr) {
    cudaEventDestroy(begin);
  }
  if (end != nullptr) {
    cudaEventDestroy(end);
  }
  if (x != nullptr) {
    cudaFree(x);
  }
  if (y != nullptr) {
    cudaFree(y);
  }
  if (cuda_status != cudaSuccess) {
    lt_plan_destroy(plan);
    return static_cast<int>(cuda_status);
  }
  if (best < 0) {
    lt_plan_destroy(plan);
    return cublas_status_to_error(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  plan.algo = results[best].algo;
  g_lt_plans.emplace(key, plan);
  return static_cast<int>(cudaSuccess);
}

// Pin one cublasLt algo for (M,K) at rep_n (heuristic top, no timing → deterministic), keyed {M,K}.
int gemm_lt_pin_tune_cuda(int M, int rep_n, int K) {
  if (M <= 0 || rep_n <= 0 || K <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int rc = ensure_lt_resources();
  if (rc == static_cast<int>(cudaSuccess)) {
    rc = ensure_lt_pin_workspace();
  }
  if (rc != static_cast<int>(cudaSuccess)) {
    return rc;
  }

  LtPinPlan plan;
  cublasStatus_t status = lt_pin_desc_create(plan, M, K);
  cublasLtMatrixLayout_t b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t results[16];
  int returned = 0;
  size_t ws = LT_PIN_TUNER_PREF;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, rep_n, K);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, M, rep_n, M);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulPreferenceCreate(&pref);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulAlgoGetHeuristic(g_lt_handle, plan.op, plan.a, b, c, c, pref, 16, results,
                                            &returned);
  if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
  if (status == CUBLAS_STATUS_SUCCESS && returned == 0) status = CUBLAS_STATUS_NOT_SUPPORTED;
  if (c != nullptr) cublasLtMatrixLayoutDestroy(c);
  if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
  if (status != CUBLAS_STATUS_SUCCESS) {
    lt_pin_destroy(plan);
    return cublas_status_to_error(status);
  }

  plan.algo = results[0].algo;
  const std::array<int, 2> key{M, K};
  auto existing = g_lt_pin_plans.find(key);
  if (existing != g_lt_pin_plans.end()) {
    lt_pin_destroy(existing->second);
    g_lt_pin_plans.erase(existing);
  }
  g_lt_pin_plans.emplace(key, plan);
  return static_cast<int>(cudaSuccess);
}

// Run or check the pinned (M,K) algo at an arbitrary N (rebuilds only B/C). `check_only` skips the
// matmul (W/X/Y/stream untouched) and returns 0 when the algo serves N. The serviceability tests
// (AlgoCheck + workspace-over-budget) live here ONCE so the boot self-check is never more permissive
// than production. Returns PIN_UNTUNED (no plan) or PIN_UNSUPPORTED (algo can't serve this N).
static int lt_pin_run(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y, int M, int N,
                      int K, cudaStream_t stream, bool check_only) {
  if (g_lt_handle == nullptr || g_lt_pin_workspace == nullptr) {
    return GEMM_LT_PIN_UNTUNED;
  }
  auto it = g_lt_pin_plans.find(std::array<int, 2>{M, K});
  if (it == g_lt_pin_plans.end()) {
    return GEMM_LT_PIN_UNTUNED;
  }
  LtPinPlan &plan = it->second;

  cublasLtMatrixLayout_t b = nullptr, c = nullptr;
  cublasStatus_t status = cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, N, K);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, M, N, M);
  }
  int result;
  if (status != CUBLAS_STATUS_SUCCESS) {
    result = cublas_status_to_error(status);
  } else {
    cublasLtMatmulHeuristicResult_t check{};
    cublasStatus_t check_status =
        cublasLtMatmulAlgoCheck(g_lt_handle, plan.op, plan.a, b, c, c, &plan.algo, &check);
    if (check_status != CUBLAS_STATUS_SUCCESS) {
      result = GEMM_LT_PIN_UNSUPPORTED;
    } else if (check.workspaceSize > LT_PIN_WORKSPACE_SIZE) {
      result = GEMM_LT_PIN_UNSUPPORTED;
    } else if (check_only) {
      result = 0;
    } else {
      const float h_alpha = 1.0f;
      const float h_beta = 0.0f;
      cublasStatus_t mm =
          cublasLtMatmul(g_lt_handle, plan.op, &h_alpha, W, plan.a, X, b, &h_beta, Y, c, Y, c,
                         &plan.algo, g_lt_pin_workspace, LT_PIN_WORKSPACE_SIZE, stream);
      result = (mm == CUBLAS_STATUS_SUCCESS) ? static_cast<int>(cudaPeekAtLastError())
                                             : cublas_status_to_error(mm);
    }
  }
  if (c != nullptr) cublasLtMatrixLayoutDestroy(c);
  if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
  return result;
}

int gemm_lt_pin_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y, int M, int N,
                     int K, cudaStream_t stream) {
  return lt_pin_run(W, X, Y, M, N, K, stream, /*check_only=*/false);
}

// Boot self-check: thin check-only wrapper over lt_pin_run.
int gemm_lt_pin_check_cuda(int M, int N, int K) {
  return lt_pin_run(nullptr, nullptr, nullptr, M, N, K, /*stream=*/nullptr, /*check_only=*/true);
}

// Batched per-token GEMM: each row is computed as the same N=1 GEMM used by
// decode, preserving row-wise numerical parity while keeping a batch-shaped
// Rust API.
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
