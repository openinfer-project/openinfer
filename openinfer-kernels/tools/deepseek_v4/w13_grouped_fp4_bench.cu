#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" int deepseek_tilelang_act_quant_k4096(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_gemm_n2048_k4096(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_act_quant_k2048(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_gemm_n4096_k2048(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

extern "C" int deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096(
    const void* a,
    const void* const* w1,
    const void* const* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* const* scales_w1,
    const void* const* scales_w3,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream);

extern "C" __global__ void deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096_kernel(
    const void* a,
    const void* const* w1,
    const void* const* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* const* scales_w1,
    const void* const* scales_w3,
    const int* expert_indptr,
    int m);

extern "C" __global__ void deepseek_tilelang_fp4_grouped_gemm_n4096_k2048_kernel(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m);

namespace {

constexpr int kInDim = 4096;
constexpr int kOutDim = 2048;
constexpr int kActScaleCols = kInDim / 128;
constexpr int kWeightScaleCols = kInDim / 32;
constexpr int kW2InDim = 2048;
constexpr int kW2OutDim = 4096;
constexpr int kW2ActScaleCols = kW2InDim / 128;
constexpr int kW2WeightScaleCols = kW2InDim / 32;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(_err));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define TK_CHECK(expr)                                                         \
  do {                                                                         \
    int _err = (expr);                                                         \
    if (_err != 0) {                                                           \
      std::fprintf(stderr, "TileLang launcher error %s:%d: %d\n", __FILE__,    \
                   __LINE__, _err);                                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Args {
  int rows = 128;
  int experts = 8;
  int active_experts = 0;
  int rows_per_active = 0;
  int warmup = 20;
  int iters = 200;
  int seed = 42;
  int shared_bytes = 0;
  int capacity_launch_rows = 0;
  int compact_launch_rows = 0;
  std::vector<int> counts;
};

Args parse_args(int argc, char** argv) {
  Args args;
  auto parse_counts = [](const char* text) {
    std::vector<int> counts;
    const char* cursor = text;
    while (*cursor != '\0') {
      char* end = nullptr;
      long value = std::strtol(cursor, &end, 10);
      if (end == cursor || value < 0 || value > 1'000'000) {
        std::fprintf(stderr, "invalid --counts entry near '%s'\n", cursor);
        std::exit(2);
      }
      counts.push_back(static_cast<int>(value));
      cursor = end;
      if (*cursor == ',') {
        ++cursor;
      } else if (*cursor != '\0') {
        std::fprintf(stderr, "invalid --counts separator near '%s'\n", cursor);
        std::exit(2);
      }
    }
    return counts;
  };
  for (int i = 1; i < argc; ++i) {
    auto read_int = [&](const char* name, int* out) {
      if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        *out = std::atoi(argv[++i]);
        return true;
      }
      return false;
    };
    if (read_int("--rows", &args.rows) || read_int("--experts", &args.experts) ||
        read_int("--active-experts", &args.active_experts) ||
        read_int("--rows-per-active", &args.rows_per_active) ||
        read_int("--warmup", &args.warmup) || read_int("--iters", &args.iters) ||
        read_int("--seed", &args.seed) || read_int("--shared-bytes", &args.shared_bytes) ||
        read_int("--capacity-launch-rows", &args.capacity_launch_rows) ||
        read_int("--compact-launch-rows", &args.compact_launch_rows)) {
      continue;
    }
    if (std::strcmp(argv[i], "--counts") == 0 && i + 1 < argc) {
      args.counts = parse_counts(argv[++i]);
      continue;
    }
    std::fprintf(stderr,
                 "usage: %s [--rows N] [--experts N] [--warmup N] [--iters N] "
                 "[--seed N] [--active-experts N] [--rows-per-active N] "
                 "[--shared-bytes N] [--capacity-launch-rows N] "
                 "[--compact-launch-rows N] [--counts c0,c1,...]\n",
                 argv[0]);
    std::exit(2);
  }
  if (!args.counts.empty()) {
    if (args.active_experts != 0 || args.rows_per_active != 0) {
      std::fprintf(stderr, "--counts cannot be combined with active prefix mode\n");
      std::exit(2);
    }
    args.experts = static_cast<int>(args.counts.size());
    args.rows = 0;
    for (int count : args.counts) {
      args.rows += count;
      if (count > 0) {
        ++args.active_experts;
        args.rows_per_active = std::max(args.rows_per_active, count);
      }
    }
    if (args.rows == 0 || args.active_experts == 0) {
      std::fprintf(stderr, "--counts needs at least one nonzero expert\n");
      std::exit(2);
    }
  }
  if ((args.active_experts == 0) != (args.rows_per_active == 0)) {
    std::fprintf(stderr, "active mode needs both --active-experts and --rows-per-active\n");
    std::exit(2);
  }
  if (args.active_experts > 0 && args.counts.empty()) {
    args.rows = args.active_experts * args.rows_per_active;
  }
  if (args.rows <= 0 || args.experts <= 0 || args.warmup < 0 || args.iters <= 0 ||
      args.shared_bytes < 0 || args.capacity_launch_rows < 0 ||
      args.compact_launch_rows < 0) {
    std::fprintf(stderr, "invalid arguments\n");
    std::exit(2);
  }
  if (args.active_experts < 0 || args.rows_per_active < 0 ||
      args.active_experts > args.experts) {
    std::fprintf(stderr, "invalid active expert arguments\n");
    std::exit(2);
  }
  if (args.compact_launch_rows > 0 && args.active_experts == 0) {
    std::fprintf(stderr, "--compact-launch-rows requires active mode\n");
    std::exit(2);
  }
  return args;
}

std::vector<int> make_indptr(int rows, int experts) {
  std::vector<int> counts(experts, 0);
  int remaining = rows;
  for (int e = 0; e < experts; ++e) {
    int left = experts - e;
    int count = (e % 5 == 0) ? 0 : std::max(1, remaining / left);
    count = std::min(count, remaining);
    counts[e] = count;
    remaining -= count;
  }
  counts.back() += remaining;

  std::vector<int> indptr(experts + 1, 0);
  for (int e = 0; e < experts; ++e) {
    indptr[e + 1] = indptr[e] + counts[e];
  }
  return indptr;
}

std::vector<int> make_active_prefix_indptr(int active_experts, int rows_per_active) {
  std::vector<int> indptr(active_experts + 1, 0);
  for (int e = 0; e < active_experts; ++e) {
    indptr[e + 1] = indptr[e] + rows_per_active;
  }
  return indptr;
}

std::vector<int> make_indptr_from_counts(const std::vector<int>& counts) {
  std::vector<int> indptr(counts.size() + 1, 0);
  for (size_t e = 0; e < counts.size(); ++e) {
    indptr[e + 1] = indptr[e] + counts[e];
  }
  return indptr;
}

std::vector<int> make_active_prefix_full_indptr(
    int experts,
    int active_experts,
    int rows_per_active) {
  std::vector<int> indptr(experts + 1, active_experts * rows_per_active);
  for (int e = 0; e <= active_experts; ++e) {
    indptr[e] = e * rows_per_active;
  }
  return indptr;
}

std::vector<int> make_active_indices(const std::vector<int>& counts) {
  std::vector<int> indices;
  for (int e = 0; e < static_cast<int>(counts.size()); ++e) {
    if (counts[e] > 0) {
      indices.push_back(e);
    }
  }
  return indices;
}

std::vector<int> compact_counts(const std::vector<int>& counts, const std::vector<int>& indices) {
  std::vector<int> compact;
  compact.reserve(indices.size());
  for (int expert : indices) {
    compact.push_back(counts[expert]);
  }
  return compact;
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return ptr;
}

void fill_ptrs(
    unsigned char* base,
    size_t stride,
    int experts,
    void*** out_device_ptrs) {
  std::vector<const void*> host(experts);
  for (int e = 0; e < experts; ++e) {
    host[e] = base + e * stride;
  }
  void** device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, experts * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), experts * sizeof(void*), cudaMemcpyHostToDevice));
  *out_device_ptrs = device;
}

void fill_selected_ptrs(
    unsigned char* base,
    size_t stride,
    const std::vector<int>& experts,
    void*** out_device_ptrs) {
  std::vector<const void*> host;
  host.reserve(experts.size());
  for (int expert : experts) {
    host.push_back(base + static_cast<size_t>(expert) * stride);
  }
  void** device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(void*), cudaMemcpyHostToDevice));
  *out_device_ptrs = device;
}

float time_ms(cudaStream_t stream, int iters, const std::function<void()>& fn) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

int launch_w13_raw(
    const void* a,
    const void* const* w1,
    const void* const* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* const* scales_w1,
    const void* const* scales_w3,
    const int* expert_indptr,
    int m,
    int local_experts,
    int shared_bytes,
    cudaStream_t stream) {
  constexpr int kThreads = 128;
  cudaError_t err = cudaFuncSetAttribute(
      deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue && err != cudaErrorNotSupported) {
    return static_cast<int>(err);
  }
  dim3 grid(32, (m + 31) / 32, local_experts);
  deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096_kernel<<<
      grid, kThreads, shared_bytes, stream>>>(
      a,
      w1,
      w3,
      gate_out,
      up_out,
      scales_a,
      scales_w1,
      scales_w3,
      expert_indptr,
      m);
  return static_cast<int>(cudaGetLastError());
}

int launch_w2_raw(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m,
    int local_experts,
    int shared_bytes,
    cudaStream_t stream) {
  constexpr int kThreads = 128;
  cudaError_t err = cudaFuncSetAttribute(
      deepseek_tilelang_fp4_grouped_gemm_n4096_k2048_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue) {
    return static_cast<int>(err);
  }
  dim3 grid(32, (m + 31) / 32, local_experts);
  deepseek_tilelang_fp4_grouped_gemm_n4096_k2048_kernel<<<
      grid, kThreads, shared_bytes, stream>>>(
      a,
      b,
      c,
      scales_a,
      scales_b,
      expert_indptr,
      m);
  return static_cast<int>(cudaGetLastError());
}

int compare_u16(
    const std::vector<uint16_t>& expected,
    const std::vector<uint16_t>& got,
    const char* name) {
  int mismatches = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != got[i]) {
      if (mismatches < 8) {
        std::fprintf(stderr, "%s mismatch[%zu]: expected=0x%04x got=0x%04x\n",
                     name, i, expected[i], got[i]);
      }
      ++mismatches;
    }
  }
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> x_dist(-3.0f, 3.0f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> scale_dist(120, 132);

  const size_t x_elems = static_cast<size_t>(args.rows) * kInDim;
  const size_t act_bytes = x_elems;
  const size_t act_scale_bytes = static_cast<size_t>(args.rows) * kActScaleCols;
  const size_t weight_bytes_per_expert = static_cast<size_t>(kOutDim) * kInDim / 2;
  const size_t weight_scale_bytes_per_expert = static_cast<size_t>(kOutDim) * kWeightScaleCols;
  const size_t out_elems = static_cast<size_t>(args.rows) * kOutDim;
  const size_t w2_x_elems = static_cast<size_t>(args.rows) * kW2InDim;
  const size_t w2_act_bytes = w2_x_elems;
  const size_t w2_act_scale_bytes = static_cast<size_t>(args.rows) * kW2ActScaleCols;
  const size_t w2_weight_bytes_per_expert = static_cast<size_t>(kW2OutDim) * kW2InDim / 2;
  const size_t w2_weight_scale_bytes_per_expert =
      static_cast<size_t>(kW2OutDim) * kW2WeightScaleCols;
  const size_t w2_out_elems = static_cast<size_t>(args.rows) * kW2OutDim;

  std::vector<__nv_bfloat16> x_host(x_elems);
  for (auto& value : x_host) {
    value = __float2bfloat16(x_dist(rng));
  }
  auto* x = device_copy(x_host);

  unsigned char* act = nullptr;
  unsigned char* act_scale = nullptr;
  CUDA_CHECK(cudaMalloc(&act, act_bytes));
  CUDA_CHECK(cudaMalloc(&act_scale, act_scale_bytes));
  TK_CHECK(deepseek_tilelang_act_quant_k4096(x, act, act_scale, args.rows, stream));

  const size_t all_weight_bytes = weight_bytes_per_expert * args.experts;
  const size_t all_scale_bytes = weight_scale_bytes_per_expert * args.experts;
  std::vector<unsigned char> w1_host(all_weight_bytes);
  std::vector<unsigned char> w3_host(all_weight_bytes);
  std::vector<unsigned char> s1_host(all_scale_bytes);
  std::vector<unsigned char> s3_host(all_scale_bytes);
  for (auto& value : w1_host) value = static_cast<unsigned char>(byte_dist(rng));
  for (auto& value : w3_host) value = static_cast<unsigned char>(byte_dist(rng));
  for (auto& value : s1_host) value = static_cast<unsigned char>(scale_dist(rng));
  for (auto& value : s3_host) value = static_cast<unsigned char>(scale_dist(rng));

  auto* w1 = device_copy(w1_host);
  auto* w3 = device_copy(w3_host);
  auto* s1 = device_copy(s1_host);
  auto* s3 = device_copy(s3_host);
  void** w1_ptrs = nullptr;
  void** w3_ptrs = nullptr;
  void** s1_ptrs = nullptr;
  void** s3_ptrs = nullptr;
  fill_ptrs(w1, weight_bytes_per_expert, args.experts, &w1_ptrs);
  fill_ptrs(w3, weight_bytes_per_expert, args.experts, &w3_ptrs);
  fill_ptrs(s1, weight_scale_bytes_per_expert, args.experts, &s1_ptrs);
  fill_ptrs(s3, weight_scale_bytes_per_expert, args.experts, &s3_ptrs);

  const size_t all_w2_weight_bytes = w2_weight_bytes_per_expert * args.experts;
  const size_t all_w2_scale_bytes = w2_weight_scale_bytes_per_expert * args.experts;
  std::vector<unsigned char> w2_host(all_w2_weight_bytes);
  std::vector<unsigned char> s2_host(all_w2_scale_bytes);
  for (auto& value : w2_host) value = static_cast<unsigned char>(byte_dist(rng));
  for (auto& value : s2_host) value = static_cast<unsigned char>(scale_dist(rng));
  auto* w2 = device_copy(w2_host);
  auto* s2 = device_copy(s2_host);
  void** w2_ptrs = nullptr;
  void** s2_ptrs = nullptr;
  fill_ptrs(w2, w2_weight_bytes_per_expert, args.experts, &w2_ptrs);
  fill_ptrs(s2, w2_weight_scale_bytes_per_expert, args.experts, &s2_ptrs);

  const bool counts_mode = !args.counts.empty();
  const bool active_mode = args.active_experts > 0;
  const std::vector<int> active_indices =
      counts_mode ? make_active_indices(args.counts) : std::vector<int>{};
  std::vector<int> indptr_host = counts_mode
      ? make_indptr_from_counts(args.counts)
      : (active_mode
             ? make_active_prefix_full_indptr(args.experts, args.active_experts,
                                              args.rows_per_active)
             : make_indptr(args.rows, args.experts));
  auto* indptr = device_copy(indptr_host);
  void** w1_compact_ptrs = nullptr;
  void** w3_compact_ptrs = nullptr;
  void** s1_compact_ptrs = nullptr;
  void** s3_compact_ptrs = nullptr;
  void** w2_compact_ptrs = nullptr;
  void** s2_compact_ptrs = nullptr;
  int* compact_indptr = nullptr;
  std::vector<int> compact_indptr_host;
  if (active_mode) {
    if (counts_mode) {
      fill_selected_ptrs(w1, weight_bytes_per_expert, active_indices, &w1_compact_ptrs);
      fill_selected_ptrs(w3, weight_bytes_per_expert, active_indices, &w3_compact_ptrs);
      fill_selected_ptrs(s1, weight_scale_bytes_per_expert, active_indices, &s1_compact_ptrs);
      fill_selected_ptrs(s3, weight_scale_bytes_per_expert, active_indices, &s3_compact_ptrs);
      fill_selected_ptrs(w2, w2_weight_bytes_per_expert, active_indices, &w2_compact_ptrs);
      fill_selected_ptrs(s2, w2_weight_scale_bytes_per_expert, active_indices, &s2_compact_ptrs);
      compact_indptr_host = make_indptr_from_counts(compact_counts(args.counts, active_indices));
    } else {
      fill_ptrs(w1, weight_bytes_per_expert, args.active_experts, &w1_compact_ptrs);
      fill_ptrs(w3, weight_bytes_per_expert, args.active_experts, &w3_compact_ptrs);
      fill_ptrs(s1, weight_scale_bytes_per_expert, args.active_experts, &s1_compact_ptrs);
      fill_ptrs(s3, weight_scale_bytes_per_expert, args.active_experts, &s3_compact_ptrs);
      fill_ptrs(w2, w2_weight_bytes_per_expert, args.active_experts, &w2_compact_ptrs);
      fill_ptrs(s2, w2_weight_scale_bytes_per_expert, args.active_experts, &s2_compact_ptrs);
      compact_indptr_host = make_active_prefix_indptr(args.active_experts, args.rows_per_active);
    }
    compact_indptr = device_copy(compact_indptr_host);
  }

  __nv_bfloat16* gate_ref = nullptr;
  __nv_bfloat16* up_ref = nullptr;
  __nv_bfloat16* gate_w13 = nullptr;
  __nv_bfloat16* up_w13 = nullptr;
  __nv_bfloat16* gate_compact = nullptr;
  __nv_bfloat16* up_compact = nullptr;
  CUDA_CHECK(cudaMalloc(&gate_ref, out_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&up_ref, out_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gate_w13, out_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&up_w13, out_elems * sizeof(__nv_bfloat16)));
  if (active_mode) {
    CUDA_CHECK(cudaMalloc(&gate_compact, out_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&up_compact, out_elems * sizeof(__nv_bfloat16)));
  }
  CUDA_CHECK(cudaMemsetAsync(gate_ref, 0x11, out_elems * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(up_ref, 0x22, out_elems * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(gate_w13, 0x33, out_elems * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(up_w13, 0x44, out_elems * sizeof(__nv_bfloat16), stream));
  if (active_mode) {
    CUDA_CHECK(cudaMemsetAsync(gate_compact, 0x55, out_elems * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(up_compact, 0x66, out_elems * sizeof(__nv_bfloat16), stream));
  }

  auto run_baseline = [&]() {
    TK_CHECK(deepseek_tilelang_fp4_grouped_gemm_n2048_k4096(
        act, reinterpret_cast<const void* const*>(w1_ptrs), gate_ref, act_scale,
        reinterpret_cast<const void* const*>(s1_ptrs), indptr, args.rows, args.experts, stream));
    TK_CHECK(deepseek_tilelang_fp4_grouped_gemm_n2048_k4096(
        act, reinterpret_cast<const void* const*>(w3_ptrs), up_ref, act_scale,
        reinterpret_cast<const void* const*>(s3_ptrs), indptr, args.rows, args.experts, stream));
  };
  const int compact_launch_rows =
      args.compact_launch_rows > 0 ? args.compact_launch_rows : args.rows;
  auto run_w13 = [&]() {
    if (args.shared_bytes > 0 || args.capacity_launch_rows > 0) {
      const int launch_rows =
          args.capacity_launch_rows > 0 ? args.capacity_launch_rows : args.rows;
      const int shared_bytes = args.shared_bytes > 0 ? args.shared_bytes : 32768;
      TK_CHECK(launch_w13_raw(
          act, reinterpret_cast<const void* const*>(w1_ptrs),
          reinterpret_cast<const void* const*>(w3_ptrs), gate_w13, up_w13, act_scale,
          reinterpret_cast<const void* const*>(s1_ptrs),
          reinterpret_cast<const void* const*>(s3_ptrs), indptr, launch_rows, args.experts,
          shared_bytes, stream));
    } else {
      TK_CHECK(deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096(
          act, reinterpret_cast<const void* const*>(w1_ptrs),
          reinterpret_cast<const void* const*>(w3_ptrs), gate_w13, up_w13, act_scale,
          reinterpret_cast<const void* const*>(s1_ptrs),
          reinterpret_cast<const void* const*>(s3_ptrs), indptr, args.rows, args.experts, stream));
    }
  };
  auto run_w13_compact = [&]() {
    if (args.shared_bytes > 0) {
      TK_CHECK(launch_w13_raw(
          act, reinterpret_cast<const void* const*>(w1_compact_ptrs),
          reinterpret_cast<const void* const*>(w3_compact_ptrs), gate_compact, up_compact,
          act_scale, reinterpret_cast<const void* const*>(s1_compact_ptrs),
          reinterpret_cast<const void* const*>(s3_compact_ptrs), compact_indptr, compact_launch_rows,
          args.active_experts, args.shared_bytes, stream));
    } else {
      TK_CHECK(deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096(
          act, reinterpret_cast<const void* const*>(w1_compact_ptrs),
          reinterpret_cast<const void* const*>(w3_compact_ptrs), gate_compact, up_compact,
          act_scale, reinterpret_cast<const void* const*>(s1_compact_ptrs),
          reinterpret_cast<const void* const*>(s3_compact_ptrs), compact_indptr, compact_launch_rows,
          args.active_experts, stream));
    }
  };

  __nv_bfloat16* w2_x = nullptr;
  unsigned char* w2_act = nullptr;
  unsigned char* w2_act_scale = nullptr;
  __nv_bfloat16* w2_capacity = nullptr;
  __nv_bfloat16* w2_compact = nullptr;
  if (active_mode) {
    std::vector<__nv_bfloat16> w2_x_host(w2_x_elems);
    for (auto& value : w2_x_host) {
      value = __float2bfloat16(x_dist(rng));
    }
    w2_x = device_copy(w2_x_host);
    CUDA_CHECK(cudaMalloc(&w2_act, w2_act_bytes));
    CUDA_CHECK(cudaMalloc(&w2_act_scale, w2_act_scale_bytes));
    CUDA_CHECK(cudaMalloc(&w2_capacity, w2_out_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&w2_compact, w2_out_elems * sizeof(__nv_bfloat16)));
    TK_CHECK(deepseek_tilelang_act_quant_k2048(w2_x, w2_act, w2_act_scale, args.rows, stream));
    CUDA_CHECK(cudaMemsetAsync(w2_capacity, 0x77, w2_out_elems * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(w2_compact, 0x88, w2_out_elems * sizeof(__nv_bfloat16), stream));
  }
  auto run_w2_capacity = [&]() {
    if (args.shared_bytes > 0 || args.capacity_launch_rows > 0) {
      const int launch_rows =
          args.capacity_launch_rows > 0 ? args.capacity_launch_rows : args.rows;
      const int shared_bytes = args.shared_bytes > 0 ? args.shared_bytes : 32768;
      TK_CHECK(launch_w2_raw(
          w2_act, reinterpret_cast<const void* const*>(w2_ptrs), w2_capacity, w2_act_scale,
          reinterpret_cast<const void* const*>(s2_ptrs), indptr, launch_rows, args.experts,
          shared_bytes, stream));
    } else {
      TK_CHECK(deepseek_tilelang_fp4_grouped_gemm_n4096_k2048(
          w2_act, reinterpret_cast<const void* const*>(w2_ptrs), w2_capacity, w2_act_scale,
          reinterpret_cast<const void* const*>(s2_ptrs), indptr, args.rows, args.experts, stream));
    }
  };
  auto run_w2_compact = [&]() {
    if (args.shared_bytes > 0) {
      TK_CHECK(launch_w2_raw(
          w2_act, reinterpret_cast<const void* const*>(w2_compact_ptrs), w2_compact,
          w2_act_scale, reinterpret_cast<const void* const*>(s2_compact_ptrs), compact_indptr,
          compact_launch_rows, args.active_experts, args.shared_bytes, stream));
    } else {
      TK_CHECK(deepseek_tilelang_fp4_grouped_gemm_n4096_k2048(
          w2_act, reinterpret_cast<const void* const*>(w2_compact_ptrs), w2_compact, w2_act_scale,
          reinterpret_cast<const void* const*>(s2_compact_ptrs), compact_indptr, compact_launch_rows,
          args.active_experts, stream));
    }
  };

  run_baseline();
  run_w13();
  if (active_mode) run_w13_compact();
  if (active_mode) {
    run_w2_capacity();
    run_w2_compact();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<uint16_t> gate_ref_host(out_elems);
  std::vector<uint16_t> up_ref_host(out_elems);
  std::vector<uint16_t> gate_w13_host(out_elems);
  std::vector<uint16_t> up_w13_host(out_elems);
  std::vector<uint16_t> gate_compact_host(out_elems);
  std::vector<uint16_t> up_compact_host(out_elems);
  CUDA_CHECK(cudaMemcpy(gate_ref_host.data(), gate_ref, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(up_ref_host.data(), up_ref, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gate_w13_host.data(), gate_w13, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(up_w13_host.data(), up_w13, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  if (active_mode) {
    CUDA_CHECK(cudaMemcpy(gate_compact_host.data(), gate_compact, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(up_compact_host.data(), up_compact, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  }
  std::vector<uint16_t> w2_capacity_host(w2_out_elems);
  std::vector<uint16_t> w2_compact_host(w2_out_elems);
  if (active_mode) {
    CUDA_CHECK(cudaMemcpy(w2_capacity_host.data(), w2_capacity, w2_out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w2_compact_host.data(), w2_compact, w2_out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  }

  int gate_mismatches = compare_u16(gate_ref_host, gate_w13_host, "gate");
  int up_mismatches = compare_u16(up_ref_host, up_w13_host, "up");
  if (active_mode) {
    gate_mismatches += compare_u16(gate_ref_host, gate_compact_host, "gate_compact");
    up_mismatches += compare_u16(up_ref_host, up_compact_host, "up_compact");
    gate_mismatches += compare_u16(w2_capacity_host, w2_compact_host, "w2_compact");
  }
  if (gate_mismatches || up_mismatches) {
    std::fprintf(stderr, "FUZZ FAIL gate_mismatches=%d up_mismatches=%d\n",
                 gate_mismatches, up_mismatches);
    return 1;
  }

  for (int i = 0; i < args.warmup; ++i) {
    run_baseline();
    run_w13();
    if (active_mode) run_w13_compact();
    if (active_mode) {
      run_w2_capacity();
      run_w2_compact();
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float baseline_ms = time_ms(stream, args.iters, run_baseline);
  float w13_ms = time_ms(stream, args.iters, run_w13);
  float compact_w13_ms = active_mode ? time_ms(stream, args.iters, run_w13_compact) : 0.0f;
  float w2_capacity_ms = active_mode ? time_ms(stream, args.iters, run_w2_capacity) : 0.0f;
  float compact_w2_ms = active_mode ? time_ms(stream, args.iters, run_w2_compact) : 0.0f;

  std::printf("W13 grouped FP4 fuzz: PASS rows=%d experts=%d seed=%d\n",
              args.rows, args.experts, args.seed);
  if (args.shared_bytes > 0) {
    std::printf("raw_launch_shared_bytes=%d\n", args.shared_bytes);
  }
  if (args.capacity_launch_rows > 0) {
    std::printf("capacity_launch_rows=%d\n", args.capacity_launch_rows);
  }
  std::printf("expert_indptr:");
  for (int value : indptr_host) std::printf(" %d", value);
  std::printf("\n");
  if (counts_mode) {
    std::printf("counts:");
    for (int value : args.counts) std::printf(" %d", value);
    std::printf("\n");
    std::printf("active_indices:");
    for (int value : active_indices) std::printf(" %d", value);
    std::printf("\n");
  }
  std::printf("baseline_two_gemm_ms=%.6f\n", baseline_ms);
  std::printf("w13_one_gemm_ms=%.6f\n", w13_ms);
  std::printf("speedup=%.3fx\n", baseline_ms / w13_ms);
  if (active_mode) {
    std::printf("compact_active_experts=%d\n", args.active_experts);
    std::printf("compact_rows_per_active=%d\n", args.rows_per_active);
    std::printf("compact_launch_rows=%d\n", compact_launch_rows);
    std::printf("compact_expert_indptr:");
    for (int value : compact_indptr_host) std::printf(" %d", value);
    std::printf("\n");
    std::printf("compact_w13_one_gemm_ms=%.6f\n", compact_w13_ms);
    std::printf("compact_vs_capacity_speedup=%.3fx\n", w13_ms / compact_w13_ms);
    std::printf("w2_capacity_gemm_ms=%.6f\n", w2_capacity_ms);
    std::printf("compact_w2_gemm_ms=%.6f\n", compact_w2_ms);
    std::printf("compact_w2_vs_capacity_speedup=%.3fx\n", w2_capacity_ms / compact_w2_ms);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
