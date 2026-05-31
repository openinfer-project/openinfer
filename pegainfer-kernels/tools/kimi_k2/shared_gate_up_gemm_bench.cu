#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Build:
//   nvcc -O3 -std=c++17 -lcublas -lcublasLt \
//     -o /tmp/shared_gate_up_gemm_bench \
//     pegainfer-kernels/tools/kimi_k2/shared_gate_up_gemm_bench.cu
//
// Default shape mirrors Kimi-K2 TP1 DP8 PPLX shared_gate_up:
//   Y[M, N] = W[M, K] @ X[K, N]
//   M = 4096 shared gate/up output, N = active decode rows, K = 7168 hidden.

namespace {

enum class Backend {
  Cublas,
  CublasLt,
  Both,
};

const char *cublas_status_name(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "CUBLAS_STATUS_UNKNOWN";
  }
}

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

void check_cublas(cublasStatus_t status, const char *expr, const char *file,
                  int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expr,
                 cublas_status_name(status));
    std::exit(1);
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

int parse_int(const char *value, const char *name) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
    throw std::runtime_error(std::string("invalid positive integer for ") + name +
                             ": " + value);
  }
  return static_cast<int>(parsed);
}

int parse_nonnegative_int(const char *value, const char *name) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
    throw std::runtime_error(std::string("invalid nonnegative integer for ") +
                             name + ": " + value);
  }
  return static_cast<int>(parsed);
}

double parse_double(const char *value, const char *name) {
  char *end = nullptr;
  double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0' || parsed <= 0.0) {
    throw std::runtime_error(std::string("invalid positive number for ") + name +
                             ": " + value);
  }
  return parsed;
}

std::vector<int> parse_csv_ints(const char *value, const char *name) {
  std::vector<int> out;
  const char *start = value;
  while (*start != '\0') {
    const char *comma = std::strchr(start, ',');
    std::string token =
        comma == nullptr ? std::string(start) : std::string(start, comma - start);
    if (token.empty()) {
      throw std::runtime_error(std::string("empty item in ") + name);
    }
    out.push_back(parse_int(token.c_str(), name));
    if (comma == nullptr) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

struct Args {
  int device = 0;
  int m = 4096;
  int k = 7168;
  int layers = 60;
  int warmup_steps = 10;
  int measure_steps = 100;
  int workspace_mb = 0;
  double peak_tflops = 148.0;
  double peak_gbps = 4800.0;
  bool reuse_weight = false;
  bool csv = false;
  Backend backend = Backend::Cublas;
  int lt_requested_algos = 32;
  int lt_tune_iters = 5;
  std::vector<int> n_list = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128};
};

const char *backend_name(Backend backend) {
  switch (backend) {
  case Backend::Cublas:
    return "cublas";
  case Backend::CublasLt:
    return "cublaslt";
  case Backend::Both:
    return "both";
  }
  return "unknown";
}

Backend parse_backend(const char *value) {
  if (std::strcmp(value, "cublas") == 0) {
    return Backend::Cublas;
  }
  if (std::strcmp(value, "cublaslt") == 0) {
    return Backend::CublasLt;
  }
  if (std::strcmp(value, "both") == 0) {
    return Backend::Both;
  }
  throw std::runtime_error(std::string("invalid backend: ") + value);
}

void print_help(const char *argv0) {
  std::printf(
      "Usage: %s [options]\n"
      "\n"
      "Options:\n"
      "  --device N             CUDA device ordinal, default 0\n"
      "  --m N                  GEMM M / output rows, default 4096\n"
      "  --k N                  GEMM K / input hidden, default 7168\n"
      "  --n-list CSV           GEMM N / active rows sweep, default "
      "1,2,3,4,5,6,7,8,16,32,64,128\n"
      "  --layers N             distinct layer weights per measured step, default 60\n"
      "  --warmup N             warmup steps, each step runs --layers GEMMs, default 10\n"
      "  --iters N              measured steps, each step runs --layers GEMMs, default 100\n"
      "  --workspace-mb N       call cublasSetWorkspace with this many MiB, default 0\n"
      "  --backend NAME         cublas, cublaslt, or both; default cublas\n"
      "  --lt-algos N           cuBLASLt heuristic candidates, default 32\n"
      "  --lt-tune-iters N      cuBLASLt candidate timing iterations, default 5\n"
      "  --reuse-weight         use one W matrix for every layer instead of distinct W\n"
      "  --peak-tflops N        BF16 peak for %%peak calculation, default 148\n"
      "  --peak-gbps N          HBM peak GB/s for %%peak calculation, default 4800\n"
      "  --csv                  print CSV instead of a table\n"
      "  --help                 show this help\n",
      argv0);
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value after ") + name);
      }
      return argv[++i];
    };
    if (std::strcmp(arg, "--help") == 0) {
      print_help(argv[0]);
      std::exit(0);
    } else if (std::strcmp(arg, "--device") == 0) {
      args.device = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--m") == 0) {
      args.m = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--k") == 0) {
      args.k = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--n-list") == 0) {
      args.n_list = parse_csv_ints(need_value(arg), arg);
    } else if (std::strcmp(arg, "--layers") == 0) {
      args.layers = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--warmup") == 0) {
      args.warmup_steps = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--iters") == 0) {
      args.measure_steps = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--workspace-mb") == 0) {
      args.workspace_mb = parse_nonnegative_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--backend") == 0) {
      args.backend = parse_backend(need_value(arg));
    } else if (std::strcmp(arg, "--lt-algos") == 0) {
      args.lt_requested_algos = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--lt-tune-iters") == 0) {
      args.lt_tune_iters = parse_int(need_value(arg), arg);
    } else if (std::strcmp(arg, "--peak-tflops") == 0) {
      args.peak_tflops = parse_double(need_value(arg), arg);
    } else if (std::strcmp(arg, "--peak-gbps") == 0) {
      args.peak_gbps = parse_double(need_value(arg), arg);
    } else if (std::strcmp(arg, "--reuse-weight") == 0) {
      args.reuse_weight = true;
    } else if (std::strcmp(arg, "--csv") == 0) {
      args.csv = true;
    } else {
      throw std::runtime_error(std::string("unknown argument: ") + arg);
    }
  }
  if (args.n_list.empty()) {
    throw std::runtime_error("--n-list must not be empty");
  }
  return args;
}

struct LtAlgoInfo {
  int id = -1;
  int tile = -1;
  int split_k = -1;
  int reduction = -1;
  int swizzle = -1;
  int custom = -1;
  std::size_t workspace_bytes = 0;
  float tune_us = 0.0f;
};

struct Measurement {
  float ms = 0.0f;
  LtAlgoInfo lt;
};

struct LtPlan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algo{};
  LtAlgoInfo info;
  bool has_algo = false;

  ~LtPlan() {
    if (d != nullptr) {
      cublasLtMatrixLayoutDestroy(d);
    }
    if (c != nullptr) {
      cublasLtMatrixLayoutDestroy(c);
    }
    if (b != nullptr) {
      cublasLtMatrixLayoutDestroy(b);
    }
    if (a != nullptr) {
      cublasLtMatrixLayoutDestroy(a);
    }
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
    }
  }

  LtPlan(const LtPlan &) = delete;
  LtPlan &operator=(const LtPlan &) = delete;
  LtPlan() = default;

  LtPlan(LtPlan &&other) noexcept
      : operation(other.operation), a(other.a), b(other.b), c(other.c),
        d(other.d), algo(other.algo), info(other.info),
        has_algo(other.has_algo) {
    other.operation = nullptr;
    other.a = nullptr;
    other.b = nullptr;
    other.c = nullptr;
    other.d = nullptr;
    other.has_algo = false;
  }

  LtPlan &operator=(LtPlan &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (d != nullptr) {
      cublasLtMatrixLayoutDestroy(d);
    }
    if (c != nullptr) {
      cublasLtMatrixLayoutDestroy(c);
    }
    if (b != nullptr) {
      cublasLtMatrixLayoutDestroy(b);
    }
    if (a != nullptr) {
      cublasLtMatrixLayoutDestroy(a);
    }
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
    }
    operation = other.operation;
    a = other.a;
    b = other.b;
    c = other.c;
    d = other.d;
    algo = other.algo;
    info = other.info;
    has_algo = other.has_algo;
    other.operation = nullptr;
    other.a = nullptr;
    other.b = nullptr;
    other.c = nullptr;
    other.d = nullptr;
    other.has_algo = false;
    return *this;
  }
};

void run_one_gemm(cublasHandle_t handle, const __nv_bfloat16 *w,
                  const __nv_bfloat16 *x, __nv_bfloat16 *y, int m, int n,
                  int k) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                            w, CUDA_R_16BF, k, x, CUDA_R_16BF, k, &beta, y,
                            CUDA_R_16BF, m, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

LtAlgoInfo get_lt_algo_info(const cublasLtMatmulHeuristicResult_t &result,
                            float tune_us) {
  LtAlgoInfo info;
  info.workspace_bytes = result.workspaceSize;
  info.tune_us = tune_us;
  std::size_t written = 0;
  (void)cublasLtMatmulAlgoConfigGetAttribute(
      &result.algo, CUBLASLT_ALGO_CONFIG_ID, &info.id, sizeof(info.id),
      &written);
  (void)cublasLtMatmulAlgoConfigGetAttribute(
      &result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &info.tile,
      sizeof(info.tile), &written);
  (void)cublasLtMatmulAlgoConfigGetAttribute(
      &result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &info.split_k,
      sizeof(info.split_k), &written);
  (void)cublasLtMatmulAlgoConfigGetAttribute(
      &result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &info.reduction,
      sizeof(info.reduction), &written);
  (void)cublasLtMatmulAlgoConfigGetAttribute(
      &result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &info.swizzle,
      sizeof(info.swizzle), &written);
  (void)cublasLtMatmulAlgoConfigGetAttribute(
      &result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &info.custom,
      sizeof(info.custom), &written);
  return info;
}

cublasStatus_t run_one_gemm_lt(cublasLtHandle_t handle, const LtPlan &plan,
                               const __nv_bfloat16 *w,
                               const __nv_bfloat16 *x, __nv_bfloat16 *y,
                               void *workspace, std::size_t workspace_bytes,
                               cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasLtMatmul(handle, plan.operation, &alpha, w, plan.a, x, plan.b,
                        &beta, y, plan.c, y, plan.d, &plan.algo, workspace,
                        workspace_bytes, stream);
}

float tune_lt_candidate_us(cublasLtHandle_t handle, const LtPlan &plan,
                           const cublasLtMatmulHeuristicResult_t &candidate,
                           const __nv_bfloat16 *w, const __nv_bfloat16 *x,
                           __nv_bfloat16 *y, void *workspace,
                           std::size_t workspace_bytes, cudaStream_t stream,
                           int tune_iters) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (candidate.workspaceSize > workspace_bytes) {
    return std::numeric_limits<float>::infinity();
  }

  cublasStatus_t status =
      cublasLtMatmul(handle, plan.operation, &alpha, w, plan.a, x, plan.b,
                     &beta, y, plan.c, y, plan.d, &candidate.algo, workspace,
                     workspace_bytes, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return std::numeric_limits<float>::infinity();
  }
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    (void)cudaGetLastError();
    return std::numeric_limits<float>::infinity();
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < tune_iters; ++i) {
    status = cublasLtMatmul(handle, plan.operation, &alpha, w, plan.a, x,
                            plan.b, &beta, y, plan.c, y, plan.d,
                            &candidate.algo, workspace, workspace_bytes,
                            stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      return std::numeric_limits<float>::infinity();
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  cuda_status = cudaEventSynchronize(stop);
  if (cuda_status != cudaSuccess) {
    (void)cudaGetLastError();
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return std::numeric_limits<float>::infinity();
  }

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.0f / static_cast<float>(tune_iters);
}

LtPlan make_lt_plan(cublasLtHandle_t handle, const __nv_bfloat16 *w,
                    const __nv_bfloat16 *x, __nv_bfloat16 *y, int m, int n,
                    int k, void *workspace, std::size_t workspace_bytes,
                    cudaStream_t stream, const Args &args) {
  LtPlan plan;
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F,
                                        CUDA_R_32F));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, k, m, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16BF, m, n, m));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.d, CUDA_R_16BF, m, n, m));

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes)));

  std::vector<cublasLtMatmulHeuristicResult_t> candidates(
      static_cast<std::size_t>(args.lt_requested_algos));
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      handle, plan.operation, plan.a, plan.b, plan.c, plan.d, preference,
      args.lt_requested_algos, candidates.data(), &returned));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));

  float best_us = std::numeric_limits<float>::infinity();
  int best_idx = -1;
  for (int i = 0; i < returned; ++i) {
    const float us =
        tune_lt_candidate_us(handle, plan, candidates[static_cast<std::size_t>(i)],
                             w, x, y, workspace, workspace_bytes, stream,
                             args.lt_tune_iters);
    if (us < best_us) {
      best_us = us;
      best_idx = i;
    }
  }

  if (best_idx < 0) {
    throw std::runtime_error("cuBLASLt found no runnable algorithm");
  }
  const auto &best = candidates[static_cast<std::size_t>(best_idx)];
  plan.algo = best.algo;
  plan.info = get_lt_algo_info(best, best_us);
  plan.has_algo = true;
  return plan;
}

Measurement measure_ms_cublas(cublasHandle_t handle, const __nv_bfloat16 *w,
                              const __nv_bfloat16 *x, __nv_bfloat16 *y,
                              const Args &args, int n,
                              cudaStream_t stream) {
  const std::size_t weight_stride =
      static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.k);
  const int weight_layers = args.reuse_weight ? 1 : args.layers;

  for (int step = 0; step < args.warmup_steps; ++step) {
    for (int layer = 0; layer < args.layers; ++layer) {
      const int weight_idx = layer % weight_layers;
      run_one_gemm(handle, w + weight_stride * weight_idx, x, y, args.m, n,
                   args.k);
    }
  }
  CUDA_CHECK(cudaPeekAtLastError());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int step = 0; step < args.measure_steps; ++step) {
    for (int layer = 0; layer < args.layers; ++layer) {
      const int weight_idx = layer % weight_layers;
      run_one_gemm(handle, w + weight_stride * weight_idx, x, y, args.m, n,
                   args.k);
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaPeekAtLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return Measurement{ms, {}};
}

Measurement measure_ms_cublaslt(cublasLtHandle_t handle, const __nv_bfloat16 *w,
                                const __nv_bfloat16 *x, __nv_bfloat16 *y,
                                const Args &args, int n, void *workspace,
                                std::size_t workspace_bytes,
                                cudaStream_t stream) {
  const std::size_t weight_stride =
      static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.k);
  const int weight_layers = args.reuse_weight ? 1 : args.layers;
  LtPlan plan =
      make_lt_plan(handle, w, x, y, args.m, n, args.k, workspace,
                   workspace_bytes, stream, args);

  for (int step = 0; step < args.warmup_steps; ++step) {
    for (int layer = 0; layer < args.layers; ++layer) {
      const int weight_idx = layer % weight_layers;
      CUBLAS_CHECK(run_one_gemm_lt(handle, plan, w + weight_stride * weight_idx,
                                   x, y, workspace, workspace_bytes, stream));
    }
  }
  CUDA_CHECK(cudaPeekAtLastError());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int step = 0; step < args.measure_steps; ++step) {
    for (int layer = 0; layer < args.layers; ++layer) {
      const int weight_idx = layer % weight_layers;
      CUBLAS_CHECK(run_one_gemm_lt(handle, plan, w + weight_stride * weight_idx,
                                   x, y, workspace, workspace_bytes, stream));
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaPeekAtLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return Measurement{ms, plan.info};
}

void print_result(const Args &args, const char *backend, int n,
                  const Measurement &measurement, double ridge) {
  const double calls =
      static_cast<double>(args.measure_steps) * static_cast<double>(args.layers);
  const double per_call_us = static_cast<double>(measurement.ms) * 1000.0 / calls;
  const double step_ms =
      per_call_us * static_cast<double>(args.layers) / 1000.0;
  const double flops =
      2.0 * static_cast<double>(args.m) * static_cast<double>(n) *
      static_cast<double>(args.k);
  const double bytes =
      2.0 * (static_cast<double>(args.m) * static_cast<double>(args.k) +
             static_cast<double>(args.k) * static_cast<double>(n) +
             static_cast<double>(args.m) * static_cast<double>(n));
  const double sec = per_call_us * 1.0e-6;
  const double ai = flops / bytes;
  const double tflops = flops / sec / 1.0e12;
  const double tbps = bytes / sec / 1.0e12;
  const double compute_pct = tflops / args.peak_tflops * 100.0;
  const double memory_pct = tbps * 1000.0 / args.peak_gbps * 100.0;
  const double mem_roof_us = bytes / (args.peak_gbps * 1.0e9) * 1.0e6;
  const double compute_roof_us =
      flops / (args.peak_tflops * 1.0e12) * 1.0e6;
  const char *bound = ai < ridge ? "Memory" : "Compute";

  if (!args.csv) {
    const LtAlgoInfo &lt = measurement.lt;
    const double workspace_mib =
        static_cast<double>(lt.workspace_bytes) / 1024.0 / 1024.0;
    std::printf("%8s %5d %12.3f %12.3f %11.3f %9.2f %10.2f %10.3f %8.2f%% %8.2f%% %8s %7d %7d %7d %9.3f %8.2f\n",
                backend, n, per_call_us, step_ms, mem_roof_us, ai, tflops,
                tbps, compute_pct, memory_pct, bound, lt.id, lt.tile,
                lt.split_k, lt.tune_us, workspace_mib);
  } else {
    const LtAlgoInfo &lt = measurement.lt;
    std::printf("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%d,%d,%d,%d,%d,%d,%.6f,%zu\n",
                backend, n, per_call_us, step_ms, mem_roof_us,
                compute_roof_us, ai, tflops, tbps, compute_pct, memory_pct,
                bound, lt.id, lt.tile, lt.split_k, lt.reduction, lt.swizzle,
                lt.custom, lt.tune_us, lt.workspace_bytes);
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    const int max_n = *std::max_element(args.n_list.begin(), args.n_list.end());
    const int weight_layers = args.reuse_weight ? 1 : args.layers;
    const std::size_t w_elems = static_cast<std::size_t>(weight_layers) *
                                static_cast<std::size_t>(args.m) *
                                static_cast<std::size_t>(args.k);
    const std::size_t x_elems =
        static_cast<std::size_t>(args.k) * static_cast<std::size_t>(max_n);
    const std::size_t y_elems =
        static_cast<std::size_t>(args.m) * static_cast<std::size_t>(max_n);
    const std::size_t w_bytes = w_elems * sizeof(__nv_bfloat16);
    const std::size_t x_bytes = x_elems * sizeof(__nv_bfloat16);
    const std::size_t y_bytes = y_elems * sizeof(__nv_bfloat16);

    __nv_bfloat16 *w = nullptr;
    __nv_bfloat16 *x = nullptr;
    __nv_bfloat16 *y = nullptr;
    CUDA_CHECK(cudaMalloc(&w, w_bytes));
    CUDA_CHECK(cudaMalloc(&x, x_bytes));
    CUDA_CHECK(cudaMalloc(&y, y_bytes));
    CUDA_CHECK(cudaMemset(w, 0x3f, w_bytes));
    CUDA_CHECK(cudaMemset(x, 0x3f, x_bytes));
    CUDA_CHECK(cudaMemset(y, 0, y_bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t handle = nullptr;
    cublasLtHandle_t lt_handle = nullptr;
    const bool use_cublas =
        args.backend == Backend::Cublas || args.backend == Backend::Both;
    const bool use_cublaslt =
        args.backend == Backend::CublasLt || args.backend == Backend::Both;
    if (use_cublas) {
      CUBLAS_CHECK(cublasCreate(&handle));
      CUBLAS_CHECK(cublasSetStream(handle, stream));
      CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }
    if (use_cublaslt) {
      CUBLAS_CHECK(cublasLtCreate(&lt_handle));
    }

    void *workspace = nullptr;
    std::size_t workspace_bytes = 0;
    if (args.workspace_mb > 0) {
      workspace_bytes = static_cast<std::size_t>(args.workspace_mb) * 1024 * 1024;
      CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
      if (use_cublas) {
        CUBLAS_CHECK(cublasSetWorkspace(handle, workspace, workspace_bytes));
      }
    }

    const double ridge = args.peak_tflops * 1000.0 / args.peak_gbps;
    if (!args.csv) {
      std::printf("device=%s sm=%d%d\n", prop.name, prop.major, prop.minor);
      std::printf("gemm=shared_gate_up opT/opN compute=f32 output=bf16 backend=%s\n",
                  backend_name(args.backend));
      std::printf("M=%d K=%d layers=%d weight_layers=%d warmup=%d iters=%d workspace=%dMiB lt_algos=%d lt_tune_iters=%d\n",
                  args.m, args.k, args.layers, weight_layers,
                  args.warmup_steps, args.measure_steps, args.workspace_mb,
                  args.lt_requested_algos, args.lt_tune_iters);
      std::printf("peaks=%.2f TFLOP/s %.2f GB/s ridge=%.2f flop/byte\n\n",
                  args.peak_tflops, args.peak_gbps, ridge);
      std::printf("%8s %5s %12s %12s %11s %9s %10s %10s %9s %9s %8s %7s %7s %7s %9s %8s\n",
                  "backend", "N", "per_call_us", "step_ms", "mem_roof",
                  "AI", "TFLOP/s", "TB/s", "%compute", "%memory", "bound",
                  "algo", "tile", "splitK", "tune_us", "ws_MiB");
      std::printf("%8s %5s %12s %12s %11s %9s %10s %10s %9s %9s %8s %7s %7s %7s %9s %8s\n",
                  "--------", "-----", "------------", "------------",
                  "-----------", "---------", "----------", "----------",
                  "---------", "---------", "--------", "-------", "-------",
                  "-------", "---------", "--------");
    } else {
      std::printf("backend,n,per_call_us,step_ms,mem_roof_us,compute_roof_us,ai,tflops,tbps,compute_pct,memory_pct,bound,algo_id,tile,split_k,reduction,swizzle,custom,tune_us,workspace_bytes\n");
    }

    for (int n : args.n_list) {
      if (use_cublas) {
        const Measurement measurement =
            measure_ms_cublas(handle, w, x, y, args, n, stream);
        print_result(args, "cublas", n, measurement, ridge);
      }
      if (use_cublaslt) {
        const Measurement measurement =
            measure_ms_cublaslt(lt_handle, w, x, y, args, n, workspace,
                                workspace_bytes, stream);
        print_result(args, "cublaslt", n, measurement, ridge);
      }
    }

    if (workspace != nullptr) {
      CUDA_CHECK(cudaFree(workspace));
    }
    if (lt_handle != nullptr) {
      CUBLAS_CHECK(cublasLtDestroy(lt_handle));
    }
    if (handle != nullptr) {
      CUBLAS_CHECK(cublasDestroy(handle));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(w));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    return 0;
  } catch (const std::exception &err) {
    std::fprintf(stderr, "error: %s\n", err.what());
    return 1;
  }
}
