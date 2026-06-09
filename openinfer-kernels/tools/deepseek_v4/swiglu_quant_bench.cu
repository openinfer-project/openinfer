#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

constexpr int kInterDim = 2048;
constexpr int kGroupSize = 128;
constexpr int kRowsPerBlock = 4;
constexpr int kWarpSize = 32;
constexpr int kScaleCols = kInterDim / kGroupSize;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(_err));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Args {
  int rows = 128;
  int warmup = 20;
  int iters = 200;
  int seed = 42;
  float limit = 7.0f;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto read_int = [&](const char* name, int* out) {
      if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        *out = std::atoi(argv[++i]);
        return true;
      }
      return false;
    };
    auto read_float = [&](const char* name, float* out) {
      if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
        *out = std::strtof(argv[++i], nullptr);
        return true;
      }
      return false;
    };
    if (read_int("--rows", &args.rows) || read_int("--warmup", &args.warmup) ||
        read_int("--iters", &args.iters) || read_int("--seed", &args.seed) ||
        read_float("--limit", &args.limit)) {
      continue;
    }
    std::fprintf(stderr,
                 "usage: %s [--rows N] [--warmup N] [--iters N] "
                 "[--seed N] [--limit F]\n",
                 argv[0]);
    std::exit(2);
  }
  if (args.rows <= 0 || args.warmup < 0 || args.iters <= 0) {
    std::fprintf(stderr, "invalid arguments\n");
    std::exit(2);
  }
  return args;
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return ptr;
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

int compare_u8(
    const std::vector<unsigned char>& expected,
    const std::vector<unsigned char>& got,
    const char* name) {
  int mismatches = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != got[i]) {
      if (mismatches < 8) {
        std::fprintf(stderr, "%s mismatch[%zu]: expected=0x%02x got=0x%02x\n",
                     name, i, expected[i], got[i]);
      }
      ++mismatches;
    }
  }
  return mismatches;
}

__device__ __forceinline__ float round_to_bf16_float(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ unsigned char float_to_e8m0(float value) {
  return __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf);
}

__device__ __forceinline__ float e8m0_to_float(unsigned char value) {
  __nv_bfloat16_raw raw = __nv_cvt_e8m0_to_bf16raw(value);
  __nv_bfloat16 bf16_value(raw);
  return __bfloat162float(bf16_value);
}

__device__ __forceinline__ unsigned char float_to_fp8_e4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__global__ void swiglu_clamp_act_quant_k2048_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ out,
    unsigned char* __restrict__ scales,
    int rows,
    float limit) {
  const int row_block = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int row = row_block * kRowsPerBlock + warp;

  float activated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float amax = 0.0f;
  if (row < rows) {
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * kWarpSize;
      const int idx = row * kInterDim + group * kGroupSize + col;
      float gate_value = __bfloat162float(gate[idx]);
      float up_value = __bfloat162float(up[idx]);
      if (limit > 0.0f) {
        gate_value = fminf(gate_value, limit);
        up_value = fminf(fmaxf(up_value, -limit), limit);
      }
      const float silu_gate = gate_value / (1.0f + expf(-gate_value));
      activated[item] = round_to_bf16_float(silu_gate * up_value);
      amax = fmaxf(amax, fabsf(activated[item]));
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
  }
  const float rounded_amax = fmaxf(__shfl_sync(0xffffffff, amax, 0), 1.0e-4f);
  const unsigned char scale_e8m0 = float_to_e8m0(rounded_amax / 448.0f);
  const float scale = e8m0_to_float(scale_e8m0);

  if (row < rows) {
    if (lane == 0) {
      scales[row * kScaleCols + group] = scale_e8m0;
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * kWarpSize;
      const float q = fminf(fmaxf(activated[item] / scale, -448.0f), 448.0f);
      out[row * kInterDim + group * kGroupSize + col] = float_to_fp8_e4m3(q);
    }
  }
}

__global__ void materialize_accumulators_to_bf16_kernel(
    const float* __restrict__ gate_acc,
    const float* __restrict__ up_acc,
    __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ up,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    gate[idx] = __float2bfloat16(gate_acc[idx]);
    up[idx] = __float2bfloat16(up_acc[idx]);
  }
}

__global__ void accumulator_swiglu_act_quant_k2048_kernel(
    const float* __restrict__ gate_acc,
    const float* __restrict__ up_acc,
    unsigned char* __restrict__ out,
    unsigned char* __restrict__ scales,
    int rows,
    float limit) {
  const int row_block = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int row = row_block * kRowsPerBlock + warp;

  float activated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float amax = 0.0f;
  if (row < rows) {
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * kWarpSize;
      const int idx = row * kInterDim + group * kGroupSize + col;
      float gate_value = round_to_bf16_float(gate_acc[idx]);
      float up_value = round_to_bf16_float(up_acc[idx]);
      if (limit > 0.0f) {
        gate_value = fminf(gate_value, limit);
        up_value = fminf(fmaxf(up_value, -limit), limit);
      }
      const float silu_gate = gate_value / (1.0f + expf(-gate_value));
      activated[item] = round_to_bf16_float(silu_gate * up_value);
      amax = fmaxf(amax, fabsf(activated[item]));
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
  }
  const float rounded_amax = fmaxf(__shfl_sync(0xffffffff, amax, 0), 1.0e-4f);
  const unsigned char scale_e8m0 = float_to_e8m0(rounded_amax / 448.0f);
  const float scale = e8m0_to_float(scale_e8m0);

  if (row < rows) {
    if (lane == 0) {
      scales[row * kScaleCols + group] = scale_e8m0;
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * kWarpSize;
      const float q = fminf(fmaxf(activated[item] / scale, -448.0f), 448.0f);
      out[row * kInterDim + group * kGroupSize + col] = float_to_fp8_e4m3(q);
    }
  }
}

void swiglu_clamp_act_quant_k2048(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    unsigned char* out,
    unsigned char* scales,
    int rows,
    float limit,
    cudaStream_t stream) {
  dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock, kScaleCols, 1);
  swiglu_clamp_act_quant_k2048_kernel<<<grid, kGroupSize, 0, stream>>>(
      gate, up, out, scales, rows, limit);
}

void materialize_accumulators_to_bf16(
    const float* gate_acc,
    const float* up_acc,
    __nv_bfloat16* gate,
    __nv_bfloat16* up,
    int n,
    cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  materialize_accumulators_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
      gate_acc, up_acc, gate, up, n);
}

void accumulator_swiglu_act_quant_k2048(
    const float* gate_acc,
    const float* up_acc,
    unsigned char* out,
    unsigned char* scales,
    int rows,
    float limit,
    cudaStream_t stream) {
  dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock, kScaleCols, 1);
  accumulator_swiglu_act_quant_k2048_kernel<<<grid, kGroupSize, 0, stream>>>(
      gate_acc, up_acc, out, scales, rows, limit);
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> value_dist(-10.0f, 10.0f);

  const size_t elems = static_cast<size_t>(args.rows) * kInterDim;
  const size_t scale_elems = static_cast<size_t>(args.rows) * kScaleCols;

  std::vector<__nv_bfloat16> gate_host(elems);
  std::vector<__nv_bfloat16> up_host(elems);
  for (auto& value : gate_host) {
    value = __float2bfloat16(value_dist(rng));
  }
  for (auto& value : up_host) {
    value = __float2bfloat16(value_dist(rng));
  }
  auto* gate = device_copy(gate_host);
  auto* up = device_copy(up_host);

  unsigned char* fused_q = nullptr;
  unsigned char* fused_scale = nullptr;
  float* gate_acc = nullptr;
  float* up_acc = nullptr;
  unsigned char* accum_q = nullptr;
  unsigned char* accum_scale = nullptr;
  CUDA_CHECK(cudaMalloc(&fused_q, elems));
  CUDA_CHECK(cudaMalloc(&fused_scale, scale_elems));
  CUDA_CHECK(cudaMalloc(&gate_acc, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&up_acc, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&accum_q, elems));
  CUDA_CHECK(cudaMalloc(&accum_scale, scale_elems));

  std::vector<float> gate_acc_host(elems);
  std::vector<float> up_acc_host(elems);
  for (size_t i = 0; i < elems; ++i) {
    gate_acc_host[i] = __bfloat162float(gate_host[i]);
    up_acc_host[i] = __bfloat162float(up_host[i]);
  }
  CUDA_CHECK(cudaMemcpy(gate_acc, gate_acc_host.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(up_acc, up_acc_host.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

  auto run_fused = [&]() {
    swiglu_clamp_act_quant_k2048(gate, up, fused_q, fused_scale, args.rows, args.limit, stream);
    CUDA_CHECK(cudaGetLastError());
  };
  auto run_accumulator_upper_bound = [&]() {
    accumulator_swiglu_act_quant_k2048(
        gate_acc, up_acc, accum_q, accum_scale, args.rows, args.limit, stream);
    CUDA_CHECK(cudaGetLastError());
  };
  auto run_materialized_accumulator = [&]() {
    materialize_accumulators_to_bf16(
        gate_acc, up_acc, gate, up, static_cast<int>(elems), stream);
    CUDA_CHECK(cudaGetLastError());
    swiglu_clamp_act_quant_k2048(gate, up, fused_q, fused_scale, args.rows, args.limit, stream);
    CUDA_CHECK(cudaGetLastError());
  };

  run_fused();
  run_accumulator_upper_bound();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<unsigned char> fused_q_host(elems);
  std::vector<unsigned char> fused_scale_host(scale_elems);
  std::vector<unsigned char> accum_q_host(elems);
  std::vector<unsigned char> accum_scale_host(scale_elems);
  CUDA_CHECK(cudaMemcpy(fused_q_host.data(), fused_q, elems, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(fused_scale_host.data(), fused_scale, scale_elems, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(accum_q_host.data(), accum_q, elems, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(accum_scale_host.data(), accum_scale, scale_elems, cudaMemcpyDeviceToHost));

  const int q_mismatches = compare_u8(fused_q_host, accum_q_host, "accum_fp8");
  const int scale_mismatches =
      compare_u8(fused_scale_host, accum_scale_host, "accum_scale");
  if (q_mismatches || scale_mismatches) {
    std::fprintf(stderr, "FUZZ FAIL q_mismatches=%d scale_mismatches=%d\n",
                 q_mismatches, scale_mismatches);
    return 1;
  }

  for (int i = 0; i < args.warmup; ++i) {
    run_fused();
    run_accumulator_upper_bound();
    run_materialized_accumulator();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float fused_ms = time_ms(stream, args.iters, run_fused);
  float materialized_accumulator_ms = time_ms(stream, args.iters, run_materialized_accumulator);
  float accumulator_upper_bound_ms = time_ms(stream, args.iters, run_accumulator_upper_bound);

  std::printf("SwiGLU+act_quant fuzz: PASS rows=%d seed=%d limit=%.3f\n",
              args.rows, args.seed, args.limit);
  std::printf("fused_swiglu_act_quant_ms=%.6f\n", fused_ms);
  std::printf("materialized_accumulator_to_gate_up_plus_quant_ms=%.6f\n",
              materialized_accumulator_ms);
  std::printf("accumulator_direct_swiglu_quant_upper_bound_ms=%.6f\n",
              accumulator_upper_bound_ms);
  std::printf("accumulator_direct_speedup=%.3fx\n",
              materialized_accumulator_ms / accumulator_upper_bound_ms);

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
