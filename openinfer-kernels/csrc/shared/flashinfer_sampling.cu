#include "common.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

#include <flashinfer/sampling.cuh>

namespace {

constexpr int GATHER_CAST_BLOCK = 256;
constexpr int GATHER_CAST_ELEMS_PER_THREAD = 8;

// Gather rows of a bf16 logits arena into a compact f32 buffer.
// grid = (ceil(vocab / (BLOCK * ELEMS)), n_rows); row_indices == nullptr means identity.
__global__ void gather_cast_logits_f32_kernel(const __nv_bfloat16* __restrict__ logits,
                                              const int* __restrict__ row_indices,
                                              float* __restrict__ out, int vocab_size) {
  int compact_row = blockIdx.y;
  int src_row = row_indices == nullptr ? compact_row : row_indices[compact_row];
  const __nv_bfloat16* src = logits + static_cast<size_t>(src_row) * vocab_size;
  float* dst = out + static_cast<size_t>(compact_row) * vocab_size;

  int base = (blockIdx.x * GATHER_CAST_BLOCK + threadIdx.x) * GATHER_CAST_ELEMS_PER_THREAD;
#pragma unroll
  for (int j = 0; j < GATHER_CAST_ELEMS_PER_THREAD; ++j) {
    int i = base + j;
    if (i < vocab_size) {
      dst[i] = __bfloat162float(src[i]);
    }
  }
}

}  // namespace

// Batched temperature/top-k/top-p sampling over a bf16 logits arena.
//
// Three launches for the whole batch: gather+cast (bf16 -> f32), FlashInfer
// OnlineSoftmax (per-row temperature, vocab-splitting strategy for the
// small-batch x large-vocab decode regime), then one FlashInfer sampling kernel
// (Sampling/TopP/TopKTopP depending on the row params). One philox seed per
// call; rows decorrelate through the philox subsequence (= row index), so the
// caller must supply a fresh seed per decode step.
//
// top_k_arr entries must be pre-clamped to [1, vocab_size] when top-k is used;
// temperature_arr entries must be > 0 — greedy rows belong on the argmax path,
// not here.
extern "C" int gpu_sample_batch_flashinfer_cuda(
    const __nv_bfloat16* logits, const int* row_indices, float* probs_scratch,
    const float* temperature_arr, const int* top_k_arr, const float* top_p_arr,
    uint8_t* valid_scratch, int* output, void* softmax_workspace,
    size_t softmax_workspace_bytes, int n_rows, int vocab_size, int has_top_k_filter,
    int has_top_p_filter, uint64_t seed, uint64_t offset, cudaStream_t stream) {
  dim3 gather_grid(
      (vocab_size + GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD - 1) /
          (GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD),
      n_rows);
  gather_cast_logits_f32_kernel<<<gather_grid, GATHER_CAST_BLOCK, 0, stream>>>(
      logits, row_indices, probs_scratch, vocab_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  // In-place: phase 2 of both OnlineSoftmax strategies is an elementwise
  // read-then-write of the same index.
  err = flashinfer::sampling::OnlineSoftmax<float>(
      probs_scratch, probs_scratch, n_rows, vocab_size, const_cast<float*>(temperature_arr),
      /*temperature_val=*/1.0f, softmax_workspace, softmax_workspace_bytes,
      /*enable_pdl=*/false, stream);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  bool* valid = reinterpret_cast<bool*>(valid_scratch);
  if (has_top_k_filter) {
    err = flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
        probs_scratch, const_cast<int*>(top_k_arr), const_cast<float*>(top_p_arr), output, valid,
        /*indices=*/nullptr, n_rows, /*top_k_val=*/0, /*top_p_val=*/0.0f, vocab_size,
        /*deterministic=*/true, /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset,
        stream);
  } else if (has_top_p_filter) {
    err = flashinfer::sampling::TopPSamplingFromProb<float, int>(
        probs_scratch, output, valid, /*indices=*/nullptr, const_cast<float*>(top_p_arr), n_rows,
        /*top_p_val=*/1.0f, vocab_size, /*deterministic=*/true, /*seed_arr=*/nullptr, seed,
        /*offset_arr=*/nullptr, offset, stream);
  } else {
    err = flashinfer::sampling::SamplingFromProb<float, int>(
        probs_scratch, output, valid, /*indices=*/nullptr, n_rows, vocab_size,
        /*deterministic=*/true, /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset,
        stream);
  }
  return static_cast<int>(err);
}
