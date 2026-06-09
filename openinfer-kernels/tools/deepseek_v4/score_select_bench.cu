#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>

#define TK_CHECK(expr)                                                            \
  do {                                                                            \
    cudaError_t _err = (expr);                                                    \
    if (_err != cudaSuccess) {                                                    \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(_err));                                     \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

__global__ void score_select_serial_kernel(
    const float* __restrict__ raw_scores,
    const float* __restrict__ gate_bias,
    float* __restrict__ route_weights,
    int* __restrict__ route_indices,
    int seq_len,
    int n_experts,
    int topk,
    float route_scale) {
  int token = blockIdx.x;
  int expert = threadIdx.x;
  if (token >= seq_len) return;

  extern __shared__ float scratch[];
  float* original_scores = scratch;
  float* select_scores = scratch + n_experts;

  if (expert < n_experts) {
    float dot = raw_scores[token * n_experts + expert];
    float softplus = dot > 20.0f ? dot : log1pf(expf(dot));
    float score = sqrtf(softplus);
    original_scores[expert] = score;
    select_scores[expert] = score + gate_bias[expert];
  }
  __syncthreads();

  if (expert == 0) {
    float selected_sum = 0.0f;
    for (int route = 0; route < topk; ++route) {
      int best_idx = 0;
      float best_score = -3.4028234663852886e38f;
      for (int candidate = 0; candidate < n_experts; ++candidate) {
        float score = select_scores[candidate];
        if (score > best_score) {
          best_score = score;
          best_idx = candidate;
        }
      }
      route_indices[token * topk + route] = best_idx;
      float route_weight = original_scores[best_idx];
      route_weights[token * topk + route] = route_weight;
      selected_sum = __fadd_rn(selected_sum, route_weight);
      select_scores[best_idx] = -3.4028234663852886e38f;
    }

    if (topk == 6) {
      float w0 = route_weights[token * topk + 0];
      float w1 = route_weights[token * topk + 1];
      float w2 = route_weights[token * topk + 2];
      float w3 = route_weights[token * topk + 3];
      float w4 = route_weights[token * topk + 4];
      float w5 = route_weights[token * topk + 5];
      float left = __fadd_rn(__fadd_rn(w0, w4), w2);
      float right = __fadd_rn(__fadd_rn(w1, w5), w3);
      selected_sum = __fadd_rn(left, right);
    }

    for (int route = 0; route < topk; ++route) {
      float normalized =
          selected_sum > 0.0f ? (route_weights[token * topk + route] / selected_sum) : 0.0f;
      route_weights[token * topk + route] = __fmul_rn(normalized, route_scale);
    }
  }
}

__global__ void score_select_parallel_kernel(
    const float* __restrict__ raw_scores,
    const float* __restrict__ gate_bias,
    float* __restrict__ route_weights,
    int* __restrict__ route_indices,
    int seq_len,
    int n_experts,
    int topk,
    float route_scale) {
  int token = blockIdx.x;
  int expert = threadIdx.x;
  if (token >= seq_len) return;

  extern __shared__ float scratch[];
  float* original_scores = scratch;
  float* select_scores = scratch + n_experts;
  float* reduce_scores = select_scores + n_experts;
  int* reduce_indices = reinterpret_cast<int*>(reduce_scores + blockDim.x);

  if (expert < n_experts) {
    float dot = raw_scores[token * n_experts + expert];
    float softplus = dot > 20.0f ? dot : log1pf(expf(dot));
    float score = sqrtf(softplus);
    original_scores[expert] = score;
    select_scores[expert] = score + gate_bias[expert];
  }
  __syncthreads();

  float selected_sum = 0.0f;
  float w0 = 0.0f;
  float w1 = 0.0f;
  float w2 = 0.0f;
  float w3 = 0.0f;
  float w4 = 0.0f;
  float w5 = 0.0f;
  for (int route = 0; route < topk; ++route) {
    reduce_scores[expert] = expert < n_experts ? select_scores[expert] : -3.4028234663852886e38f;
    reduce_indices[expert] = expert < n_experts ? expert : 2147483647;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (expert < stride) {
        float other_score = reduce_scores[expert + stride];
        int other_idx = reduce_indices[expert + stride];
        float self_score = reduce_scores[expert];
        int self_idx = reduce_indices[expert];
        if (other_score > self_score || (other_score == self_score && other_idx < self_idx)) {
          reduce_scores[expert] = other_score;
          reduce_indices[expert] = other_idx;
        }
      }
      __syncthreads();
    }

    if (expert == 0) {
      int best_idx = reduce_indices[0];
      route_indices[token * topk + route] = best_idx;
      float route_weight = best_idx < n_experts ? original_scores[best_idx] : 0.0f;
      route_weights[token * topk + route] = route_weight;
      selected_sum = __fadd_rn(selected_sum, route_weight);
      if (topk == 6) {
        if (route == 0) w0 = route_weight;
        if (route == 1) w1 = route_weight;
        if (route == 2) w2 = route_weight;
        if (route == 3) w3 = route_weight;
        if (route == 4) w4 = route_weight;
        if (route == 5) w5 = route_weight;
      }
      if (best_idx < n_experts) select_scores[best_idx] = -3.4028234663852886e38f;
    }
    __syncthreads();
  }

  if (expert == 0) {
    if (topk == 6) {
      float left = __fadd_rn(__fadd_rn(w0, w4), w2);
      float right = __fadd_rn(__fadd_rn(w1, w5), w3);
      selected_sum = __fadd_rn(left, right);
    }
    for (int route = 0; route < topk; ++route) {
      float normalized =
          selected_sum > 0.0f ? (route_weights[token * topk + route] / selected_sum) : 0.0f;
      route_weights[token * topk + route] = __fmul_rn(normalized, route_scale);
    }
  }
}

static float time_kernel(
    void (*kernel)(const float*, const float*, float*, int*, int, int, int, float),
    const float* raw,
    const float* bias,
    float* weights,
    int* indices,
    int seq_len,
    int n_experts,
    int topk,
    float route_scale,
    size_t shared_bytes,
    int iters) {
  cudaEvent_t start, stop;
  TK_CHECK(cudaEventCreate(&start));
  TK_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 100; ++i) {
    kernel<<<seq_len, 256, shared_bytes>>>(
        raw, bias, weights, indices, seq_len, n_experts, topk, route_scale);
  }
  TK_CHECK(cudaGetLastError());
  TK_CHECK(cudaDeviceSynchronize());
  TK_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    kernel<<<seq_len, 256, shared_bytes>>>(
        raw, bias, weights, indices, seq_len, n_experts, topk, route_scale);
  }
  TK_CHECK(cudaGetLastError());
  TK_CHECK(cudaEventRecord(stop));
  TK_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  TK_CHECK(cudaEventElapsedTime(&ms, start, stop));
  TK_CHECK(cudaEventDestroy(start));
  TK_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(iters);
}

int main() {
  constexpr int n_experts = 256;
  constexpr int topk = 6;
  constexpr float route_scale = 2.5f;
  constexpr int iters = 20000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> raw_dist(-4.0f, 4.0f);
  std::uniform_real_distribution<float> bias_dist(-0.2f, 0.2f);

  std::vector<float> bias_host(n_experts);
  for (float& v : bias_host) v = bias_dist(rng);

  float* raw = nullptr;
  float* bias = nullptr;
  float* serial_w = nullptr;
  float* parallel_w = nullptr;
  int* serial_i = nullptr;
  int* parallel_i = nullptr;
  TK_CHECK(cudaMalloc(&bias, n_experts * sizeof(float)));
  TK_CHECK(cudaMemcpy(bias, bias_host.data(), n_experts * sizeof(float), cudaMemcpyHostToDevice));

  for (int seq_len : {1, 8, 16, 32}) {
    std::vector<float> raw_host(seq_len * n_experts);
    for (float& v : raw_host) v = raw_dist(rng);
    TK_CHECK(cudaMalloc(&raw, raw_host.size() * sizeof(float)));
    TK_CHECK(cudaMalloc(&serial_w, seq_len * topk * sizeof(float)));
    TK_CHECK(cudaMalloc(&parallel_w, seq_len * topk * sizeof(float)));
    TK_CHECK(cudaMalloc(&serial_i, seq_len * topk * sizeof(int)));
    TK_CHECK(cudaMalloc(&parallel_i, seq_len * topk * sizeof(int)));
    TK_CHECK(cudaMemcpy(raw, raw_host.data(), raw_host.size() * sizeof(float), cudaMemcpyHostToDevice));

    size_t serial_shared = 2 * n_experts * sizeof(float);
    size_t parallel_shared = (2 * n_experts + 256) * sizeof(float) + 256 * sizeof(int);
    score_select_serial_kernel<<<seq_len, 256, serial_shared>>>(
        raw, bias, serial_w, serial_i, seq_len, n_experts, topk, route_scale);
    score_select_parallel_kernel<<<seq_len, 256, parallel_shared>>>(
        raw, bias, parallel_w, parallel_i, seq_len, n_experts, topk, route_scale);
    TK_CHECK(cudaGetLastError());
    TK_CHECK(cudaDeviceSynchronize());

    std::vector<float> sw(seq_len * topk);
    std::vector<float> pw(seq_len * topk);
    std::vector<int> si(seq_len * topk);
    std::vector<int> pi(seq_len * topk);
    TK_CHECK(cudaMemcpy(sw.data(), serial_w, sw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    TK_CHECK(cudaMemcpy(pw.data(), parallel_w, pw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    TK_CHECK(cudaMemcpy(si.data(), serial_i, si.size() * sizeof(int), cudaMemcpyDeviceToHost));
    TK_CHECK(cudaMemcpy(pi.data(), parallel_i, pi.size() * sizeof(int), cudaMemcpyDeviceToHost));

    int idx_mismatches = 0;
    int weight_mismatches = 0;
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < si.size(); ++i) {
      if (si[i] != pi[i]) ++idx_mismatches;
      float diff = std::abs(sw[i] - pw[i]);
      max_abs_diff = std::max(max_abs_diff, diff);
      if (diff != 0.0f) ++weight_mismatches;
    }

    float serial_ms = time_kernel(
        score_select_serial_kernel, raw, bias, serial_w, serial_i,
        seq_len, n_experts, topk, route_scale, serial_shared, iters);
    float parallel_ms = time_kernel(
        score_select_parallel_kernel, raw, bias, parallel_w, parallel_i,
        seq_len, n_experts, topk, route_scale, parallel_shared, iters);

    std::printf(
        "seq_len=%d serial_ms=%.6f parallel_ms=%.6f speedup=%.3fx idx_mismatches=%d weight_mismatches=%d max_abs_diff=%.9g\n",
        seq_len, serial_ms, parallel_ms, serial_ms / parallel_ms,
        idx_mismatches, weight_mismatches, max_abs_diff);

    TK_CHECK(cudaFree(raw));
    TK_CHECK(cudaFree(serial_w));
    TK_CHECK(cudaFree(parallel_w));
    TK_CHECK(cudaFree(serial_i));
    TK_CHECK(cudaFree(parallel_i));
  }

  TK_CHECK(cudaFree(bias));
  return 0;
}
