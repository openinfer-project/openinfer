#include "../common.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// GLM5.2 is FP8 weights + FlashMLA: nothing below Hopper can run the model,
// so a sub-sm_90 compilation target is a build misconfiguration, not a
// support tier. Fail here instead of shipping kernels with the PDL calls
// silently compiled out.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#error "GLM5.2 kernels require sm_90+ (check OPENINFER_CUDA_SM / detected GPU targets)"
#endif

namespace {

constexpr int kRouterSelectThreads = 512;
constexpr int kGlm52Experts = 256;
constexpr int kGlm52Topk = 8;
constexpr int kGlm52Hidden = 6144;
// Production row bound: GLM52_MAX_BATCH_PER_RANK (= the largest decode
// bucket; prefill and dspark verify spans ride the decode buckets, so no
// call path exceeds it).
constexpr int kRouterGemvMaxTokens = 8;

__device__ __forceinline__ bool better_router_choice(float value, int expert,
                                                     float best_value,
                                                     int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

__global__ void router_scores_topk_normalize_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ e_score_correction_bias,
    float* __restrict__ topk_weight, int* __restrict__ topk_idx,
    int active_tokens, int padded_tokens, int n_experts, int topk,
    float route_scale) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= padded_tokens) return;

  extern __shared__ char shared[];
  float* scores = reinterpret_cast<float*>(shared);
  float* choice_scores = scores + blockDim.x;
  float* selected_scores = choice_scores + blockDim.x;

  const int expert = tid;
  if (expert < n_experts) {
    float score = 1.0f / (1.0f + expf(-logits[token * n_experts + expert]));
    scores[tid] = score;
    choice_scores[tid] = score + e_score_correction_bias[expert];
  } else {
    scores[tid] = 0.0f;
    choice_scores[tid] = -CUDART_INF_F;
  }
  if (token >= active_tokens) return;
  __syncthreads();

  // Single-pass rank-count selection. `better_router_choice` is a strict total order over
  // (value desc, index asc) — `choice_scores` is always finite here (sigmoid in (0,1) plus a
  // finite bias), so no NaN can break the ordering — and expert indices are unique, so each
  // expert's rank — the count of strictly-better experts — is a bijection with the
  // sequential-argmax pick order. That makes
  // the selected idx/weight bit-identical to the old 8 masked full-block reductions (the
  // descending route order, hence the f32 `selected_sum` order, is preserved) while collapsing
  // ~8*log(blockDim) __syncthreads to two. For bs=1 this is one block over 256 experts — an
  // irreducibly tiny grid (cf. SGLang's warp-per-token moe_fused_gate, the bs>1 form); the only
  // lever at bs=1 is removing the serial structure, which this does (ncu 12.2 -> 8.2 us/call).
  if (expert < n_experts) {
    const float cv = choice_scores[tid];
    int rank = 0;
    for (int j = 0; j < n_experts; ++j) {
      if (better_router_choice(choice_scores[j], j, cv, expert)) ++rank;
    }
    if (rank < topk) {
      topk_idx[token * topk + rank] = expert;
      selected_scores[rank] = scores[expert];
    }
  }
  __syncthreads();

  if (tid == 0) {
    float selected_sum = 0.0f;
    for (int route = 0; route < topk; ++route) selected_sum += selected_scores[route];
    const float scale = selected_sum > 0.0f ? route_scale / selected_sum : 0.0f;
    for (int route = 0; route < topk; ++route) {
      topk_weight[token * topk + route] = selected_scores[route] * scale;
    }
  }
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

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  return map_cuda_error(err);
}

// Router logits GEMV, adapted from TensorRT-LLM's dsv3MinLatencyKernels
// dsv3RouterGemm.cu (Apache-2.0) via SGLang sgl-kernel / vLLM. One block per
// expert row; f32 accumulation in a fixed order (per-thread serial over the
// k-chunks -> warp butterfly -> cross-warp smem sum), so logits are
// deterministic run-to-run. Replaces cublasGemmEx, whose splitK plan cost 4
// whole-step-graph nodes per layer (GEMM + splitKreduce + workspace
// alloc/free) at decode row counts.
template <int kNumTokens>
__global__ __launch_bounds__(128, 1) void glm52_router_logits_gemv_kernel(
    float* __restrict__ out, const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_weight) {
  constexpr int kBlockSize = 128;
  constexpr int kVpt = 8;  // bf16 per uint4 vector load
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int kElemsPerIter = kVpt * kBlockSize;
  constexpr int kIters = kGlm52Hidden / kElemsPerIter;
  static_assert(kGlm52Hidden % kElemsPerIter == 0);

  const int expert = blockIdx.x;
  const int tid = threadIdx.x;
  const __nv_bfloat16* w_row =
      gate_weight + static_cast<size_t>(expert) * kGlm52Hidden;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  cudaGridDependencySynchronize();

  for (int ki = 0; ki < kIters; ++ki) {
    const int k_base = ki * kElemsPerIter + tid * kVpt;
    const uint4 w_vec = *reinterpret_cast<const uint4*>(w_row + k_base);
    const __nv_bfloat16* w_bf16 = reinterpret_cast<const __nv_bfloat16*>(&w_vec);
#pragma unroll
    for (int m = 0; m < kNumTokens; ++m) {
      const uint4 h_vec = *reinterpret_cast<const uint4*>(
          hidden + static_cast<size_t>(m) * kGlm52Hidden + k_base);
      const __nv_bfloat16* h_bf16 =
          reinterpret_cast<const __nv_bfloat16*>(&h_vec);
#pragma unroll
      for (int k = 0; k < kVpt; ++k) {
        acc[m] += __bfloat162float(h_bf16[k]) * __bfloat162float(w_bf16[k]);
      }
    }
  }

  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
#pragma unroll
  for (int m = 0; m < kNumTokens; ++m) {
    float sum = acc[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    if (lane_id == 0) sm_reduction[m][warp_id] = sum;
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; ++m) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; ++w) final_sum += sm_reduction[m][w];
      out[m * kGlm52Experts + expert] = final_sum;
    }
  }
  // No fence before the trigger, matching TRT-LLM/FlashInfer upstream: the
  // trigger only lets a PDL-opted dependent grid LAUNCH early; that grid may
  // not consume our stores until its cudaGridDependencySynchronize, which
  // orders on this grid's full completion (memory visibility included).
  cudaTriggerProgrammaticLaunchCompletion();
}

template <int kNumTokens>
cudaError_t launch_router_logits_gemv(float* logits,
                                      const __nv_bfloat16* hidden,
                                      const __nv_bfloat16* gate_weight,
                                      cudaStream_t stream) {
  // PSS needs cc >= 9.0, which the #error guard above makes a build
  // invariant — no host-side fallback (unlike argmax.cu, whose kernels also
  // serve pre-Hopper models).
  cudaLaunchConfig_t config = {};
  config.gridDim = kGlm52Experts;
  config.blockDim = 128;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  return cudaLaunchKernelEx(&config,
                            glm52_router_logits_gemv_kernel<kNumTokens>,
                            logits, hidden, gate_weight);
}

CUresult glm52_router_logits_gemm(const __nv_bfloat16* hidden,
                                  const __nv_bfloat16* gate_weight,
                                  float* logits, int padded_tokens,
                                  int hidden_dim, int n_experts,
                                  cudaStream_t stream) {
  if (hidden_dim != kGlm52Hidden || n_experts != kGlm52Experts ||
      padded_tokens < 1 || padded_tokens > kRouterGemvMaxTokens) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cudaError_t err = cudaSuccess;
  switch (padded_tokens) {
#define GLM52_ROUTER_GEMV_CASE(N)                                     \
  case N:                                                             \
    err = launch_router_logits_gemv<N>(logits, hidden, gate_weight, stream); \
    break;
    GLM52_ROUTER_GEMV_CASE(1)
    GLM52_ROUTER_GEMV_CASE(2)
    GLM52_ROUTER_GEMV_CASE(3)
    GLM52_ROUTER_GEMV_CASE(4)
    GLM52_ROUTER_GEMV_CASE(5)
    GLM52_ROUTER_GEMV_CASE(6)
    GLM52_ROUTER_GEMV_CASE(7)
    GLM52_ROUTER_GEMV_CASE(8)
#undef GLM52_ROUTER_GEMV_CASE
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
  return map_cuda_error(err);
}

}  // namespace

extern "C" {

CUresult glm52_router_noaux_tc_cuda(
    const __nv_bfloat16* hidden, const __nv_bfloat16* gate_weight,
    const float* e_score_correction_bias, float* logits, float* topk_weight,
    int* topk_idx, int active_tokens, int padded_tokens, int hidden_dim,
    int n_experts, int topk, float route_scale, cudaStream_t stream) {
  if (hidden == nullptr || gate_weight == nullptr ||
      e_score_correction_bias == nullptr || logits == nullptr ||
      topk_weight == nullptr || topk_idx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_tokens <= 0 || padded_tokens <= 0 ||
      active_tokens > padded_tokens || hidden_dim <= 0 || n_experts <= 0 ||
      topk <= 0 || topk > n_experts || n_experts != kGlm52Experts ||
      topk != kGlm52Topk || !(route_scale > 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult result = glm52_router_logits_gemm(
      hidden, gate_weight, logits, padded_tokens, hidden_dim, n_experts, stream);
  if (result != CUDA_SUCCESS) return result;

  size_t select_smem =
      static_cast<size_t>(kRouterSelectThreads) * (2 * sizeof(float)) +
      static_cast<size_t>(topk) * sizeof(float);
  router_scores_topk_normalize_kernel<<<padded_tokens, kRouterSelectThreads,
                                        select_smem, stream>>>(
      logits, e_score_correction_bias, topk_weight, topk_idx, active_tokens,
      padded_tokens, n_experts, topk, route_scale);
  result = consume_last_cuda_error();
  if (result != CUDA_SUCCESS) return result;

  return CUDA_SUCCESS;
}

}  // extern "C"
