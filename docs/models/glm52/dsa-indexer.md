# GLM5.2 DSA Indexer (PR2)

> **TL;DR:** DSA indexer chain kernels landed: 3 hand-written CUDA (cache insert+gather cherry-picked, local_topk_to_slots ported from TokenSpeed Triton, naive Hadamard), 1 FlashInfer `TopKDispatch` K=2048 wrapper (vendored). All 5 smoke tests pass on H200 (jz38). DeepGEMM paged MQA logits C ABI wrapper (highest risk — first DeepGEMM JIT call in the codebase) is the remaining piece.
>
> **Last touched:** 2026-07

## What this PR adds

- `indexer.rs` (model crate) — `Glm52IndexerLayerWeights` (indexer q/k projections + Hadamard + RoPE) + `glm52_indexer_forward(...)` producing `topk_indices[2048]`.
- Kernel ops (see inventory below) in `openinfer-kernels/src/ops/glm52/` + `csrc/glm52/`.

## Kernel inventory

| op | file | backend | hand-written CUDA? |
|---|---|---|---|
| `glm52_indexer_k_quant_and_cache` | `ops/glm52/indexer.rs` + `csrc/glm52/glm52_indexer.cu` (quant+cache insert half) | **hand-written** (258 lines, cherry-pick) | **yes** |
| `glm52_indexer_k_gather_quant_cache` | same file (gather half) | **hand-written** (same file) | **yes** |
| `glm52_deepgemm_paged_mqa_logits` | `ops/glm52/deepgemm_mqa.rs` + `csrc/glm52/glm52_deepgemm_mqa.cu` (new) | vendored DeepGEMM `sm90_fp8_paged_mqa_logits` | no (vendored, new C ABI wrapper) |
| `glm52_flashinfer_topk_2048` | `ops/glm52/topk.rs` + `csrc/glm52/glm52_topk.cu` (new) | vendored FlashInfer `TopKDispatch` | no (vendored, new C wrapper) |
| `glm52_indexer_local_topk_to_slots` | `ops/glm52/indexer.rs` + `csrc/glm52/glm52_indexer.cu` (new kernel) | **hand-written** (new, ported from TokenSpeed Triton) | **yes** |
| `glm52_indexer_hadamard_bf16` | `ops/glm52/hadamard.rs` + `csrc/glm52/glm52_hadamard.cu` (new) | **hand-written** (new, naive radix) | **yes** |

## Hand-written CUDA perf debt

Three files are hand-written (not vendored from FlashInfer/TRTLLM/DeepGEMM/cuBLAS):

| file | lines | what |
|---|---|---|
| `csrc/glm52/glm52_indexer.cu` (cache insert + gather) | 258 (cherry-pick) | fp8 per-128-group quant + scatter write / gather read into DeepGEMM block-split paged layout: `[block_size * 128 fp8][block_size * 4 f32 scale]` per block. Memory-bound elementwise. |
| `csrc/glm52/glm52_indexer.cu` (local_topk_to_slots) | ~80 (new) | int32 index-remap: `page = block_table[t, off//bs]; slot = page*bs + off%bs`. Ported from TokenSpeed Triton `_local_topk_to_global_slots_kernel`. |
| `csrc/glm52/glm52_hadamard.cu` | ~60 (new) | naive in-place radix Hadamard for head_dim=128 (7 stages). Not the Dao-AILab `fast-hadamard-transform` port — that is a follow-up if ncu flags it. |

All three are correct (cache kernels validated in the prototype branch against HF oracle; topk_to_slots and Hadamard are simple enough to unit-test) but **not tuned**: single-issue-per-element, no vectorized load/store, no occupancy targeting. First ncu candidates when decode TPOT is measured.

## Vendored wrapper notes

### DeepGEMM paged MQA logits (main engineering risk)

`sm90_fp8_paged_mqa_logits` (vendored at `third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp:308`) is a JIT-compiled kernel launched through DeepGEMM's `device_runtime` / `compiler` / `launch_kernel` path with `torch::Tensor` TMA descriptors. The new C ABI wrapper must replicate the TMA descriptor construction (`make_tma_2d_desc` / `make_tma_3d_desc` in `runtime_utils.hpp:113/152`) from raw device pointers + shape + strides, without torch (option (a) in the plan). Fail-closed if the JIT runtime is not initialized.

This is the first real DeepGEMM JIT kernel call in the codebase — main's `glm52_deepgemm_grouped` only does metadata, compute returns `NOT_SUPPORTED`. Two concrete sub-risks:
1. TMA descriptor construction must move off `torch::Tensor` to raw `CUtensorMap` via `cuTensorMapEncode*Tiled` driver API.
2. JIT compiler + `device_runtime` are `LazyInit` globals; first call triggers `cuLibraryLoadData` compile+load. `build.rs` already links `libcuda`, but nobody has driven this path yet.

If this blocks, PR2 can be split: `feat/glm52-dsa-indexer-cache` (cache + topk + slots + hadamard) lands first, `feat/glm52-deepgemm-mqa` lands the DeepGEMM wrapper separately.

### FlashInfer deterministic top-k K=2048

`TopKDispatch` (vendored at `third_party/flashinfer/include/flashinfer/topk.cuh:3342`) natively supports K=2048 (`FILTERED_TOPK_MAX_K=2048`), `deterministic=true`, `TopKTieBreak::Small`, `dsa_graph_safe=true` — exactly TokenSpeed's `deterministic_decode_topk` contract. The existing `csrc/shared/flashinfer_top1.cu` (72 lines, K=1) is the pattern to extend. Pure C ABI, no torch.

## Do NOT cherry-pick

The old branch's `glm52_indexer_topk_2048_cuda` was a **stub** returning `NOT_SUPPORTED`:
```
CUresult glm52_indexer_topk_2048_cuda(...) {
  (void)...;  // all params unused
  return CUDA_ERROR_NOT_SUPPORTED;
}
```
PR2 replaces it with the FlashInfer `TopKDispatch` K=2048 wrapper. Do not cherry-pick the stub.

## Gap-doc cross-reference

- `glm52_deepgemm_paged_mqa_logits` -> gap-doc `DSA decode indexer logits` (P0 #3).
- `glm52_flashinfer_topk_2048` -> gap-doc `Decode deterministic top-k` (P0 #4).
- `glm52_indexer_local_topk_to_slots` -> gap-doc `top-k offset to KV slot` (P0 #5).
- `glm52_indexer_k_quant_and_cache` -> gap-doc `DSA index-K cache set/gather` (P0 #2).
- Hadamard rotate -> gap-doc `DSA Hadamard rotate` (P1 #4). PR2 includes a naive GPU Hadamard for correctness; the Dao-AILab `fast-hadamard-transform` CUDA port (`/tmp/fast-hadamard-transform`, HEAD `e7706fa`, BSD-3-Clause, 441-line launcher) is a follow-up if the naive version is a measured bottleneck.

## Not in PR2

- Blackwell TRTLLM sparse MLA (gap-doc P0 #6) — PR2 still uses the SM90 FlashMLA sparse path from PR1, now fed with real sparse top-k instead of full top-k.
- Prefill indexer logits (contiguous MQA logits, `fp8_mqa_logits` non-paged) — decode-first; prefill rides the decode path token-by-token.

## Oracle gate — deferred

Same blocker as PR1: the prototype's fixture pipeline (HF forward dump → `layer0.npz` → probe bins → Rust test) was not self-contained. The oracle gate is deferred to a follow-up that designs a self-contained fixture pipeline.

## Build

Same as PR1 — requires SM90a GPU (H200), CUDA 12.6+, NCCL 2.30.4+. Testing on `jz38` (Hopper dev box).

## Execution Log

### Step 1: Cherry-pick cache kernels + new hand-written kernels
- Cherry-picked `glm52_indexer.cu` cache insert + gather from `feat/glm52-dp8-ep8` (commit `7e4200a`). Dropped the old `glm52_indexer_topk_2048_cuda` stub (returned `NOT_SUPPORTED`).
- Added hand-written `local_topk_to_global_slots_kernel` to the same file, ported from TokenSpeed Triton `_local_topk_to_global_slots_kernel` (`dsa_sparse_layout.py:205`). ~80 lines: int32 block-table lookup + `topk_lens` warp+block reduce.
- Added `glm52_topk.cu`: FlashInfer `TopKDispatch` K=2048 wrapper (vendored, ~70 lines). Modeled on `csrc/shared/flashinfer_top1.cu` (K=1). Uses `deterministic=true`, `TopKTieBreak::Small`, `dsa_graph_safe=true`.
- Added `glm52_hadamard.cu`: naive in-place radix Hadamard for head_dim=128 (~60 lines). O(n²) naive approach, not the Dao-AILab port.
- Rust ops + FFI for all four new modules (`indexer`, `topk`, `hadamard`).
- `build.rs`: added `glm52_topk` to FlashInfer include list.
- Result: compiles clean on sm_90 (local cross-compile + jz38 H200).

### Step 2: Smoke tests on jz38 (H200)
- 5 tests, all pass:
  - `indexer_cache_round_trip`: quant+pack → gather, 4 scales all positive ✅
  - `local_topk_to_slots_basic`: `offsets=[0,1,2,3]` → `slots=[20,21,40,41]` ✅
  - `local_topk_to_slots_invalid`: `-1` offset → `-1` slot, `topk_lens=1` ✅
  - `hadamard_correctness`: all-ones input, `output[0]=11.3125` (expected √128≈11.3137), `output[1]=0.0` ✅
  - `flashinfer_topk_basic`: K=4 from 2048 ascending logits → indices `[2044,2045,2046,2047]` ✅

### Step 3: DeepGEMM paged MQA logits wrapper — NOT YET DONE
- This is the remaining piece and the highest-risk item (first real DeepGEMM JIT kernel call in the codebase).
- Two sub-risks identified: (a) TMA descriptor construction must move off `torch::Tensor` to raw `CUtensorMap` via `cuTensorMapEncode*Tiled`, (b) JIT compiler + `device_runtime` LazyInit globals trigger `cuLibraryLoadData` on first call.
- If this blocks, PR2 can be split: current work lands as `feat/glm52-dsa-indexer-cache`, DeepGEMM wrapper lands separately as `feat/glm52-deepgemm-mqa`.

## Debrief

- **Outcome**: 5 of 6 PR2 kernel ops landed and smoke-tested on H200. DeepGEMM paged MQA logits wrapper (highest risk) remains.
- **Pitfalls encountered**:
  - Local machine (sm_120/RTX 5090) can't build the `glm52` feature because `glm52` → `moe` → DeepEP → NCCL ≥ 2.30.4, and the system NCCL was 2.28.9. Fixed by `uv pip install nvidia-nccl-cu13>=2.30.4` and setting `OPENINFER_NCCL_ROOT`.
  - Pre-commit `clippy-kernels-kimi` hook triggers on any `openinfer-kernels/` file change and requires NCCL — must export `OPENINFER_NCCL_ROOT` before `git commit`.
  - jz38 system NCCL is 2.29.7 (also < 2.30.4) — same pip NCCL workaround needed there.
- **Lessons learned**:
  - FlashInfer `TopKDispatch` with `dsa_graph_safe=true` forces `VEC_SIZE=1` and the `FilteredTopK` path — needs 128KB dynamic shared memory. H200 supports it (228KB per SM); verified by the passing test.
  - The naive Hadamard (O(n²) per token, 128 threads per token) is fine for correctness but will likely show up on ncu as a bottleneck for long-context decode. The Dao-AILab `fast-hadamard-transform` port (`/tmp/fast-hadamard-transform`, 441-line launcher, O(n log n) butterfly) is the follow-up.
- **Follow-ups**:
  - DeepGEMM paged MQA logits C ABI wrapper (the remaining PR2 piece).
  - Model crate `indexer.rs` forward wiring (depends on DeepGEMM wrapper).
  - Oracle gate — deferred, same fixture-pipeline blocker as PR1.

- **Read**:
  - `docs/models/glm52/dp1-ep8-decode-plan.md` — the 5-PR roadmap this PR belongs to.
  - `docs/models/glm52/mla-decode-brick.md` — PR1 dev doc, the pattern to follow.
  - `tokenspeed-kernel-gap.md` (local-only, recovered from git history at `71b9d18`) — full TokenSpeed GLM5.2 kernel DAG with source tags.
  - `openinfer-kernels/csrc/shared/flashinfer_top1.cu` — existing `TopKDispatch` K=1 wrapper, the pattern to extend for K=2048.
  - `openinfer-kernels/third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp` — vendored paged MQA logits entry.
  - `openinfer-kernels/third_party/flashinfer/include/flashinfer/topk.cuh` — vendored `TopKDispatch` template.
  - `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/dsa_sparse_layout.py` — TokenSpeed Triton source for `local_topk_to_global_slots`.
  - `/tmp/fast-hadamard-transform/csrc/fast_hadamard_transform_cuda.cu` — Dao-AILab upstream Hadamard (follow-up reference, not used in PR2).
- **Relevant history**:
  - `feat/glm52-dp8-ep8` old branch (commit `7e4200a`) — cherry-pick source for cache kernels. Its top-k was a stub; do not cherry-pick that.
  - PR1 (#477) — established the `fp8.rs` / `mla_decode.rs` / kernel ops layout this PR mirrors.
- **Plan**:
  1. Cherry-pick `glm52_indexer.cu` cache kernels (quant_and_cache + gather_quant_cache) from old branch — CUDA + ops + FFI. Drop the top-k stub.
  2. Write FlashInfer `TopKDispatch` K=2048 wrapper (`csrc/glm52/glm52_topk.cu` + `ops/glm52/topk.rs`), modeled on `flashinfer_top1.cu`.
  3. Write hand-written `local_topk_to_slots` kernel, ported from TokenSpeed Triton `_local_topk_to_global_slots_kernel`.
  4. Write hand-written naive Hadamard for head_dim=128.
  5. Write DeepGEMM paged MQA logits C ABI wrapper (`csrc/glm52/glm52_deepgemm_mqa.cu` + `ops/glm52/deepgemm_mqa.rs`) — TMA descriptor construction off torch, JIT runtime init.
  6. Wire `indexer.rs` model crate forward.
  7. Build + test on jz38 (Hopper).
- **Risks / open questions**:
  - DeepGEMM JIT runtime init is the first-of-its-kind call in this codebase — may need a separate spike to validate TMA descriptor construction + `cuLibraryLoadData` path before the full wrapper.
  - FlashInfer `TopKDispatch` with `dsa_graph_safe=true` forces `VEC_SIZE=1` and `FilteredTopK` path — needs 128KB dynamic shared memory; verify jz38 H200 supports it.
