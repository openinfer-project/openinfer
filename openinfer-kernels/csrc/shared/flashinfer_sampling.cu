#include "ffi_guard.cuh"

#include "common.cuh"
#include "flashinfer_radix_scratch.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

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
// Workspace for the radix top-k renorm used on the min_p pipeline; same
// layout contract as flashinfer_top1_row_states_bytes_cuda.
extern "C" size_t gpu_sample_topk_renorm_row_states_bytes_cuda() {
  return flashinfer_radix_row_states_bytes();
}

// min_p_arr enables the min_p pipeline: (optional) top-k renorm, (optional)
// top-p renorm, then FlashInfer's MinPSamplingFromProb with the per-row
// thresholds. min_p_arr == nullptr keeps the original fused single-kernel
// paths bit-for-bit (the fast path).
//
// Per-request seeds deliberately do NOT go through FlashInfer's `seed_arr`:
// these kernels read `seed_arr[0]` (one seed for the whole batch) and fold
// `blockIdx.x` into the philox subsequence, so a request's stream would
// change with its position in the batch. Seeded rows are instead sampled as
// their own n_rows=1 calls by the Rust layer, with the request seed and step
// mixed into `seed` — blockIdx is then always 0 and replay is independent of
// batch composition.
// One-hot proposal distributions for a greedy proposer: draft_probs is
// zeroed and draft_probs[i * vocab + draft_token_ids[i]] = 1.0 for each of
// the batch*K proposed tokens. A deterministic (argmax) draft is the
// degenerate proposal q(x) = delta(x - draft), under which chain rejection
// sampling accepts with min(1, p_target(draft)) and resamples from the
// target with the draft token's mass removed — still distribution-exact.
__global__ void onehot_rows_kernel(float* probs, const int* token_ids, int vocab) {
  const int row = blockIdx.x;
  const int id = token_ids[row];
  if (id >= 0 && id < vocab && threadIdx.x == 0) {
    probs[static_cast<size_t>(row) * vocab + id] = 1.0f;
  }
}

// Chain speculative (rejection) sampling over one verify span per row:
// accepts each draft token with min(1, p_target/q_draft), resamples the first
// rejection from relu(target - draft) renormalized, appends the bonus token
// on full acceptance, and -1-fills the tail. accepted/emitted counters are
// ACCUMULATED by the kernel, so the caller must zero them first.
extern "C" int gpu_chain_speculative_sampling_cuda(
    float* draft_probs, const int* draft_token_ids, float* target_probs,
    int* output_token_ids, int* output_accepted_num, int* output_emitted_num,
    int batch_size, int num_speculative_tokens, int vocab_size, int onehot_draft,
    uint64_t seed, uint64_t offset, cudaStream_t stream) {
  if (onehot_draft) {
    const size_t total = static_cast<size_t>(batch_size) * num_speculative_tokens;
    cudaError_t err = cudaMemsetAsync(
        draft_probs, 0, total * vocab_size * sizeof(float), stream);
    if (err != cudaSuccess) {
      return static_cast<int>(err);
    }
    onehot_rows_kernel<<<static_cast<unsigned>(total), 32, 0, stream>>>(
        draft_probs, draft_token_ids, vocab_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      return static_cast<int>(err);
    }
  }
  cudaError_t err = flashinfer::sampling::ChainSpeculativeSampling<float, int>(
      draft_probs, const_cast<int*>(draft_token_ids), target_probs, output_token_ids,
      output_accepted_num, output_emitted_num, batch_size, num_speculative_tokens,
      vocab_size, /*deterministic=*/true, /*seed_arr=*/nullptr, seed,
      /*offset_arr=*/nullptr, offset, stream);
  return static_cast<int>(err);
}

// The verify-side half of the batched sampling pipeline: gather + softmax +
// top-k/top-p renorm, WITHOUT the terminal sampling kernel — the renormalized
// probabilities stay in probs_out for a downstream consumer (speculative
// rejection sampling needs the target distribution itself, filtered tokens as
// exact zeros). Distribution-equivalent to the fused sampling fast path: the
// rejection sampler filters at draw time, the renorm filters then draws — the
// resulting law over tokens is identical.
extern "C" int gpu_verify_probs_flashinfer_cuda(
    const __nv_bfloat16* logits, const int* row_indices, float* probs_out,
    const float* temperature_arr, const int* top_k_arr, const float* top_p_arr,
    uint8_t* topk_row_states_scratch, void* softmax_workspace,
    size_t softmax_workspace_bytes, int n_rows, int vocab_size, int has_top_k_filter,
    int has_top_p_filter, cudaStream_t stream) {
  dim3 gather_grid(
      (vocab_size + GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD - 1) /
          (GATHER_CAST_BLOCK * GATHER_CAST_ELEMS_PER_THREAD),
      n_rows);
  gather_cast_logits_f32_kernel<<<gather_grid, GATHER_CAST_BLOCK, 0, stream>>>(
      logits, row_indices, probs_out, vocab_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = flashinfer::sampling::OnlineSoftmax<float>(
      probs_out, probs_out, n_rows, vocab_size, const_cast<float*>(temperature_arr),
      /*temperature_val=*/1.0f, softmax_workspace, softmax_workspace_bytes,
      /*enable_pdl=*/false, stream);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  if (has_top_k_filter) {
    auto* row_states =
        reinterpret_cast<flashinfer::sampling::RadixRowState*>(topk_row_states_scratch);
    if (row_states == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    err = flashinfer::sampling::RadixTopKRenormProbMultiCTA<float, int>(
        probs_out, probs_out, const_cast<int*>(top_k_arr), n_rows,
        /*top_k_val=*/0, vocab_size, row_states, stream);
    if (err != cudaSuccess) {
      return static_cast<int>(err);
    }
  }
  if (has_top_p_filter) {
    err = flashinfer::sampling::TopPRenormProb<float>(
        probs_out, probs_out, const_cast<float*>(top_p_arr), n_rows,
        /*top_p_val=*/0.0f, vocab_size, stream);
  }
  return static_cast<int>(err);
}

extern "C" int gpu_sample_batch_flashinfer_cuda(
    const __nv_bfloat16* logits, const int* row_indices, float* probs_scratch,
    const float* temperature_arr, const int* top_k_arr, const float* top_p_arr,
    const float* min_p_arr, uint8_t* topk_row_states_scratch,
    uint8_t* valid_scratch, int* output, void* softmax_workspace,
    size_t softmax_workspace_bytes, int n_rows, int vocab_size, int has_top_k_filter,
    int has_top_p_filter, uint64_t seed, uint64_t offset, cudaStream_t stream) {
  OPENINFER_FFI_GUARD_BEGIN
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
  if (min_p_arr != nullptr) {
    if (has_top_k_filter) {
      auto* row_states =
          reinterpret_cast<flashinfer::sampling::RadixRowState*>(topk_row_states_scratch);
      if (row_states == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
      }
      // In-place: the renorm kernels reduce a threshold first, then rewrite
      // each element from its own index.
      err = flashinfer::sampling::RadixTopKRenormProbMultiCTA<float, int>(
          probs_scratch, probs_scratch, const_cast<int*>(top_k_arr), n_rows,
          /*top_k_val=*/0, vocab_size, row_states, stream);
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (has_top_p_filter) {
      err = flashinfer::sampling::TopPRenormProb<float>(
          probs_scratch, probs_scratch, const_cast<float*>(top_p_arr), n_rows,
          /*top_p_val=*/0.0f, vocab_size, stream);
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    err = flashinfer::sampling::MinPSamplingFromProb<float, int>(
        probs_scratch, const_cast<float*>(min_p_arr), output, valid,
        /*indices=*/nullptr, n_rows, /*min_p_val=*/0.0f, vocab_size,
        /*deterministic=*/true, /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset,
        stream);
    return static_cast<int>(err);
  }
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
  OPENINFER_FFI_GUARD_END(-1)
}
