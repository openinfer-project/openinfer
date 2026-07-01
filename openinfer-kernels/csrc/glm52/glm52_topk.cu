// GLM5.2 FlashInfer deterministic top-k K=2048 wrapper.
//
// Wraps vendored FlashInfer `TopKDispatch` (topk.cuh:3342) for the DSA decode
// indexer. Matches TokenSpeed's `deterministic_decode_topk` contract:
// deterministic=true, TopKTieBreak::Small, dsa_graph_safe=true, K=2048.
//
// Pattern follows csrc/shared/flashinfer_top1.cu (K=1 wrapper).

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

extern "C" size_t glm52_flashinfer_topk_2048_row_states_bytes_cuda() {
  int device = 0;
  int sm_count = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  }
  size_t groups = err == cudaSuccess && sm_count > 0
                      ? static_cast<size_t>(sm_count)
                      : static_cast<size_t>(
                            flashinfer::sampling::RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP);
  size_t radix_bytes =
      groups * (sizeof(flashinfer::sampling::RadixRowState) +
                sizeof(flashinfer::sampling::RadixDeterministicCollectScratch));
  return std::max<size_t>(1024 * 1024, radix_bytes);
}

extern "C" int glm52_flashinfer_topk_2048_cuda(
    const float* logits, int* output_indices, float* output_values,
    const int* lengths, int num_rows, int top_k, int max_len,
    uint8_t* row_states_scratch, cudaStream_t stream) {
  if (logits == nullptr || output_indices == nullptr || lengths == nullptr ||
      row_states_scratch == nullptr) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }
  if (num_rows <= 0 || top_k <= 0 || max_len <= 0) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }

  auto* row_states =
      reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states_scratch);
  auto* input = const_cast<float*>(logits);

  cudaError_t err = flashinfer::sampling::TopKDispatch<float, int>(
      input, output_indices, output_values, num_rows, static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(max_len), row_states,
      /*sorted_output=*/false, /*deterministic=*/true,
      flashinfer::sampling::TopKTieBreak::Small, stream,
      /*dsa_graph_safe=*/true);
  if (err != cudaSuccess) {
    fprintf(stderr, "glm52_flashinfer_topk_2048_cuda: TopKDispatch failed: %s\n",
            cudaGetErrorString(err));
    return static_cast<int>(CUDA_ERROR_LAUNCH_FAILED);
  }
  return static_cast<int>(CUDA_SUCCESS);
}
