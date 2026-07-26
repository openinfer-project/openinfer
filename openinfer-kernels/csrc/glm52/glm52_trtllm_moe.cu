#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tensorrt_llm/kernels/quantization.h"

// The generic runner keeps an NVFP4 branch in the same host function even
// though this backend is fixed to FP8. Avoid linking FlashInfer's large
// all-format quantization translation unit solely for that unreachable call.
template <>
void tensorrt_llm::kernels::invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
    uint32_t, uint32_t, const __nv_bfloat16*, float, int32_t*, uint8_t*,
    uint8_t*, float*, flashinfer::QuantizationSFLayout, cudaStream_t) {
  throw std::runtime_error("GLM5.2 fused MoE is fixed to FP8");
}

namespace moe::dev::routing::routingLlama4 {
void run(Data const&, void*) {
  throw std::runtime_error("GLM5.2 fused MoE does not support Llama4 routing");
}
}  // namespace moe::dev::routing::routingLlama4

extern "C" CUresult glm52_fp8_per_token_group_quant_bf16_trtllm_cuda(
    const uint16_t* input, uint8_t* output, float* scales, int rows,
    int hidden_dim, int group_size, cudaStream_t stream);

namespace {

namespace moe = tensorrt_llm::kernels::trtllmgen_moe;
namespace btg = batchedGemm::trtllm::gen;

constexpr int kHidden = 6144;
constexpr int kIntermediate = 512;
constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kGroups = 8;
constexpr int kTopKGroups = 4;
constexpr int kTileTokens = 64;
constexpr float kRouteScale = 2.5f;

struct Allocation {
  void* ptr = nullptr;

  Allocation() = default;
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;

  ~Allocation() {
    if (ptr != nullptr) cudaFree(ptr);
  }

  void reserve(size_t bytes) {
    if (bytes != 0) {
      auto status = cudaMalloc(&ptr, bytes);
      if (status != cudaSuccess) throw std::bad_alloc();
    }
  }

  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr);
  }
};

struct Glm52TrtllmMoe {
  int device;
  int capacity;
  int max_padded;
  int max_ctas;
  std::unique_ptr<moe::MoE::Runner> runner;

  Allocation num_tokens_per_expert;
  Allocation total_num_padded_tokens;
  Allocation expanded_to_permuted;
  Allocation permuted_to_token;
  Allocation expert_weights;
  Allocation expert_indices;
  Allocation expert_histogram;
  Allocation cta_to_batch;
  Allocation cta_to_limit;
  Allocation non_exiting_ctas;
  Allocation hidden_fp8;
  Allocation hidden_scale;
  Allocation gemm1_output;
  Allocation gemm1_output_scale;
  Allocation activation_output;
  Allocation activation_output_scale;
  Allocation gemm2_output;
  Allocation workspace_fc1;
  Allocation workspace_fc2;

  explicit Glm52TrtllmMoe(int max_tokens, int device_ordinal)
      : device(device_ordinal),
        capacity(max_tokens),
        max_padded(moe::Routing::getMaxPermutedPaddedCount(
            max_tokens, kTopK, kExperts, kTileTokens)),
        max_ctas(moe::Routing::getMaxNumCtasInBatchDim(
            max_tokens, kTopK, kExperts, kTileTokens)) {
    if (max_tokens <= 0) throw std::invalid_argument("max_tokens");

    runner = std::make_unique<moe::MoE::Runner>(
        btg::Dtype::E4m3, true, kTileTokens, false,
        batchedGemm::gemm::MatrixLayout::MajorK, false, false, false, false);

    num_tokens_per_expert.reserve(kExperts * sizeof(int32_t));
    total_num_padded_tokens.reserve(sizeof(int32_t));
    expanded_to_permuted.reserve(
        static_cast<size_t>(capacity) * kTopK * sizeof(int32_t));
    permuted_to_token.reserve(static_cast<size_t>(max_padded) * sizeof(int32_t));
    expert_weights.reserve(
        static_cast<size_t>(capacity) * kTopK * sizeof(uint16_t));
    expert_indices.reserve(
        static_cast<size_t>(capacity) * kTopK * sizeof(int32_t));
    expert_histogram.reserve(512 * sizeof(int32_t));
    cta_to_batch.reserve(static_cast<size_t>(max_ctas) * sizeof(int32_t));
    cta_to_limit.reserve(static_cast<size_t>(max_ctas) * sizeof(int32_t));
    non_exiting_ctas.reserve(sizeof(int32_t));
    hidden_fp8.reserve(static_cast<size_t>(capacity) * kHidden);
    hidden_scale.reserve(
        static_cast<size_t>(kHidden / 128) * capacity * sizeof(float));

    const int gemm1_rows = moe::Routing::maybeGetMinTokenCount(
        max_padded, 2 * kIntermediate, 8);
    const int gemm2_rows =
        moe::Routing::maybeGetMinTokenCount(max_padded, kHidden, 16);
    gemm1_output.reserve(
        static_cast<size_t>(gemm1_rows) * 2 * kIntermediate);
    gemm1_output_scale.reserve(
        static_cast<size_t>(2 * kIntermediate / 128) * max_padded *
        sizeof(float));
    activation_output.reserve(
        static_cast<size_t>(gemm1_rows) * kIntermediate);
    activation_output_scale.reserve(
        static_cast<size_t>(kIntermediate / 128) * gemm1_rows * sizeof(float));
    gemm2_output.reserve(
        static_cast<size_t>(gemm2_rows) * kHidden * sizeof(uint16_t));

    moe::MoE::MoERunnerArgs args;
    populate_shape(args, capacity);
    size_t fc1_bytes = 0;
    size_t fc2_bytes = 0;
    for (int tokens : {std::min(capacity, 512), std::min(capacity, 4096),
                       capacity}) {
      populate_shape(args, tokens);
      for (const auto config : runner->getValidConfigIndices(
               kTopK, kHidden, kIntermediate, kExperts, tokens)) {
        const auto [fc1, fc2] = runner->getWorkspaceSizeInBytes(args, config);
        fc1_bytes = std::max(fc1_bytes, static_cast<size_t>(fc1));
        fc2_bytes = std::max(fc2_bytes, static_cast<size_t>(fc2));
      }
    }
    workspace_fc1.reserve(fc1_bytes);
    workspace_fc2.reserve(fc2_bytes);
  }

  ~Glm52TrtllmMoe() = default;

  static void populate_shape(moe::MoE::MoERunnerArgs& args, int tokens) {
    args.num_tokens = tokens;
    args.num_experts = kExperts;
    args.hidden_size = kHidden;
    args.top_k = kTopK;
    args.n_group = kGroups;
    args.topk_group = kTopKGroups;
    args.routed_scaling_factor = kRouteScale;
    args.intermediate_size = kIntermediate;
    args.local_expert_offset = 0;
    args.local_num_experts = kExperts;
    args.mDtypeElt = btg::Dtype::E4m3;
    args.mDtypeExpW = btg::Dtype::Bfloat16;
    args.mDtypeOut = btg::Dtype::Bfloat16;
    args.mUseDeepSeekFp8 = true;
    args.activation_type = moe::MoE::ActivationType::Swiglu;
    args.do_finalize = true;
  }

  CUresult run(const uint16_t* hidden, const float* routing_logits,
               const float* routing_bias,
               const uint8_t* w13, const float* w13_scale, const uint8_t* w2,
               const float* w2_scale, uint16_t* output, int tokens,
               cudaStream_t stream) {
    if (!hidden || !routing_logits || !routing_bias || !w13 ||
        !w13_scale || !w2 || !w2_scale || !output || tokens <= 0 ||
        tokens > capacity) {
      return CUDA_ERROR_INVALID_VALUE;
    }

    moe::MoE::MoERunnerArgs args;
    populate_shape(args, tokens);
    CUresult quant_status = glm52_fp8_per_token_group_quant_bf16_trtllm_cuda(
        hidden, hidden_fp8.as<uint8_t>(), hidden_scale.as<float>(), tokens,
        kHidden, 128, stream);
    if (quant_status != CUDA_SUCCESS) return quant_status;
    args.routing_logits = const_cast<float*>(routing_logits);
    args.routing_bias = const_cast<float*>(routing_bias);
    args.hidden_states = hidden_fp8.ptr;
    args.hidden_states_scale = hidden_scale.ptr;
    args.gemm1_weights = const_cast<uint8_t*>(w13);
    args.gemm1_weights_scale = const_cast<float*>(w13_scale);
    args.gemm2_weights = const_cast<uint8_t*>(w2);
    args.gemm2_weights_scale = const_cast<float*>(w2_scale);
    args.output = output;

    moe::MoE::MoEWorkspace workspace;
    workspace.routing_expert_indexes = expert_indices.as<int32_t>();
    workspace.permuted_idx_size = total_num_padded_tokens.as<int32_t>();
    workspace.total_num_padded_tokens =
        total_num_padded_tokens.as<int32_t>();
    workspace.total_max_padded_tokens = max_padded;
    workspace.expanded_idx_to_permuted_idx =
        expanded_to_permuted.as<int32_t>();
    workspace.permuted_idx_to_token_idx = permuted_to_token.as<int32_t>();
    workspace.expert_weights = expert_weights.ptr;
    workspace.cta_idx_xy_to_batch_idx = cta_to_batch.as<int32_t>();
    workspace.cta_idx_xy_to_mn_limit = cta_to_limit.as<int32_t>();
    workspace.num_non_exiting_ctas = non_exiting_ctas.as<int32_t>();
    workspace.gemm1_output = gemm1_output.ptr;
    workspace.gemm1_output_scale = gemm1_output_scale.as<float>();
    workspace.activation_output = activation_output.ptr;
    workspace.activation_output_scale = activation_output_scale.as<float>();
    workspace.gemm2_output = gemm2_output.ptr;
    workspace.bmm1_workspace = workspace_fc1.ptr;
    workspace.bmm2_workspace = workspace_fc2.ptr;

    moe::Routing::Runner routing(kTileTokens);
    routing.run(
        args.routing_logits, args.routing_bias, tokens, kExperts, kTopK,
        kGroups, kTopKGroups, 0, kExperts, kRouteScale,
        workspace.routing_expert_indexes, expert_histogram.as<int32_t>(),
        workspace.total_num_padded_tokens,
        workspace.expanded_idx_to_permuted_idx, nullptr,
        workspace.permuted_idx_to_token_idx, nullptr, workspace.expert_weights,
        num_tokens_per_expert.as<int32_t>(), workspace.cta_idx_xy_to_batch_idx,
        workspace.cta_idx_xy_to_mn_limit, workspace.num_non_exiting_ctas,
        btg::Dtype::E4m3, btg::Dtype::Fp32, false, true,
        moe::Routing::RoutingMethodType::DeepSeekV3, stream, btg::Dtype::Fp32,
        true, nullptr
#ifndef GLM52_TRTLLM_LEGACY_ABI
        , true
#endif
        );

    const int64_t config = runner->getDefaultValidConfigIndex(
        kTopK, kHidden, kIntermediate, kExperts, tokens);
    runner->run(args, workspace, device, stream, config, true);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
      openinfer_ffi_set_last_error(cudaGetErrorString(error));
      return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
  }
};

}  // namespace

#define GLM52_MOE_EXPORT extern "C" __attribute__((visibility("default")))

GLM52_MOE_EXPORT CUresult glm52_trtllm_moe_create(int max_tokens, int device,
                                                   void** out) {
  if (!out || max_tokens <= 0 || device < 0) return CUDA_ERROR_INVALID_VALUE;
  OPENINFER_FFI_GUARD_BEGIN
  *out = new Glm52TrtllmMoe(max_tokens, device);
  return CUDA_SUCCESS;
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

GLM52_MOE_EXPORT CUresult glm52_trtllm_moe_destroy(void* handle) {
  OPENINFER_FFI_GUARD_BEGIN
  delete static_cast<Glm52TrtllmMoe*>(handle);
  return CUDA_SUCCESS;
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

GLM52_MOE_EXPORT CUresult glm52_trtllm_moe_launch(
    void* handle, const uint16_t* hidden, const float* routing_logits,
    const float* routing_bias, const uint8_t* w13,
    const float* w13_scale, const uint8_t* w2, const float* w2_scale,
    uint16_t* output, int tokens, cudaStream_t stream) {
  if (!handle) return CUDA_ERROR_INVALID_HANDLE;
  OPENINFER_FFI_GUARD_BEGIN
  return static_cast<Glm52TrtllmMoe*>(handle)->run(
      hidden, routing_logits, routing_bias, w13, w13_scale, w2,
      w2_scale, output, tokens, stream);
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}
