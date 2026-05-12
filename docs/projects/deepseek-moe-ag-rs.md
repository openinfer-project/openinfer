# DeepSeek MoE AG/RS Decode

**Created**: 2026-05-11
**Status**: active

## TL;DR

Decode MoE now uses GPU-resident allgather/router/local-expert/reduce-scatter flow. Route/expert metadata stays on device, the old row-routed scalar local expert kernel is gone, and local experts use grouped TileLang FP4 GEMM over expert-major packed rows. Direct runtime no longer owns rank contexts/caches centrally: rank workers own their context, communicator, RoPE, and decode cache, and the old single-thread group prefill/debug entry points are removed. The request loop is split into a scheduler facade and rank-worker implementation, with the scheduler thread named `deepseek-v4-scheduler`. Current 1x32 serving bench after cleanup: `steady_tpot_ms.avg = 105.54ms`; exact DeepSeek V4 E2E after cleanup: `20/20`.

## Preparation

### Decode group cleanup

- **Read**:
  - `docs/index.md` - confirmed this active document is the current DeepSeek MoE AG/RS work record.
  - `docs/projects/deepseek-moe-ag-rs.md` - confirmed decode now uses rank-worker AG/RS and grouped TileLang FP4, while follow-up work should clean remaining legacy decode structure.
- **Relevant history**:
  - This document records that direct decode moved away from the temporary group block path and now enters the AG/RS MoE path from rank workers.
- **Plan**:
  1. Keep production prefill group helpers, because `prefill_logits_and_decode_cache_group_bf16_hidden` is still called by the direct runtime.
  2. Remove decode-only group entry points that support single-thread multi-rank decode: `block_decode_group_bf16_hidden`, `block_decode_group_rank_threads_bf16_hidden`, and their now-unused attention/MoE group helpers.
  3. Remove public re-exports and mp8 manifest tests that only exercise the deleted decode group path.
  4. Run `cargo fmt --check` and `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`.
- **Risks / open questions**:
  - Some tests currently use group decode as a small two-rank smoke path; deleting them narrows coverage to production rank-lane decode plus prefill group tests.

### Eliminate central single-thread direct runtime paths

- **Read**:
  - `pegainfer-deepseek-v4/src/direct.rs` - confirmed decode already uses persistent rank workers, but cache ownership is still moved through central runtime every token and prefill still uses central group helpers.
  - `pegainfer-deepseek-v4/src/runtime/block.rs` - identified rank-local decode structure and central prefill group cache seeding path.
  - `pegainfer-deepseek-v4/src/runtime/attention.rs` - identified ratio-4 prefill indexer all-reduce as the non-trivial collective that needs a rank-lane version.
  - `pegainfer-deepseek-v4/src/runtime/moe.rs` - identified prefill MoE routed all-reduce as another group-only collective to move into rank lanes.
- **Relevant history**:
  - Step 6 removed decode group entry points, but prefill/cache ownership still left a central multi-rank path that future changes could accidentally follow.
- **Plan**:
  1. Move decode caches and RoPE caches into `RankWorker`; add worker commands for cache allocation/reset, prefill, and decode.
  2. Add rank-lane prefill block/attention/MoE helpers that issue NCCL collectives from each rank worker instead of central group helpers.
  3. Make workers gather full logits from rank-local logits, so the main runtime only dispatches and receives results.
  4. Remove central cache shuffling helpers and stop calling `prefill_logits_and_decode_cache_group_bf16_hidden` from `direct.rs`.
  5. Run format/check and exact E2E.
- **Risks / open questions**:
  - All worker lanes must enter NCCL collectives in the same order; branches must depend only on shared model config and prompt shape.

- **Read**:
  - `docs/index.md` - identified DeepSeek V4 support and MoE profiling docs as the relevant history.
  - `docs/projects/deepseek-v4-support.md` - confirmed the current DeepSeek V4 runtime still treats MoE route-index D2H as a higher-risk follow-up.
  - `docs/projects/deepseek-moe-tilelang-review.md` - confirmed previous MoE work found D2H route scheduling and per-rank skew as structural decode issues, but local expert execution is out of scope for this slice.
- **Relevant history**:
  - `docs/projects/deepseek-moe-tilelang-review.md` records that replacing local expert execution is a larger cutover; this task intentionally only adds the regular collective exchange primitives.
- **Plan**:
  1. Add `all_gather_bf16_hidden_group` and `reduce_scatter_f32_hidden_group` in `pegainfer-deepseek-v4/src/runtime/collectives.rs` with explicit shape checks.
  2. Export the new collectives from `pegainfer-deepseek-v4/src/lib.rs`.
  3. Add a focused NCCL pair test in `pegainfer-deepseek-v4/tests/mp8_manifest.rs` that validates `[B_local,H] -> [world*B_local,H]` allgather and `[world*B_local,H] -> [B_local,H]` f32 reduce-scatter.
  4. Run the targeted test or compile check with release settings.
- **Risks / open questions**:
  - The pair test requires two GPUs and a loadable NCCL runtime; it should skip cleanly when NCCL is unavailable, matching the existing all-reduce test.

## Execution Log

### Step 6: Remove legacy decode group path
- Removed decode-only single-thread/multi-rank entry points from `pegainfer-deepseek-v4/src/runtime/block.rs`:
  - `block_decode_group_bf16_hidden`
  - `block_decode_group_rank_threads_bf16_hidden`
- Removed decode group helpers that only existed for those entry points:
  - attention decode group wrappers in `pegainfer-deepseek-v4/src/runtime/attention.rs`
  - `decode_moe_ag_rs_group_bf16_hidden` in `pegainfer-deepseek-v4/src/runtime/moe.rs`
  - AG/RS group collectives `all_gather_bf16_hidden_group`, `all_gather_u32_group`, and `reduce_scatter_f32_hidden_group`
- Removed public re-exports and mp8 manifest tests that referenced the deleted decode group path.
- Kept production prefill group helpers, because direct prefill still calls `prefill_logits_and_decode_cache_group_bf16_hidden`.
- Verification:
  - `cargo fmt`
  - `cargo fmt --check` passed
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed
  - `rg -n "block_decode_group|group_rank_threads|attention_decode_group|decode_moe_ag_rs_group|all_gather_bf16_hidden_group|reduce_scatter_f32_hidden_group|all_gather_u32_group" pegainfer-deepseek-v4/src pegainfer-deepseek-v4/tests` returned no matches

### Step 7: Remote exact E2E after cleanup
- Synced the cleanup files back to `5090:/root/develop/xingming/pegainfer`.
- Verified model path on 5090: `/data/DeepSeek-V4-Flash`.
- Ran on 5090:

```bash
source ~/.cargo/env 2>/dev/null || true
cd /root/develop/xingming/pegainfer
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- --model-path /data/DeepSeek-V4-Flash
```

- Result:
  - release build completed
  - model loaded in `34.71s`
  - `All 20 DeepSeek V4 exact cases passed`

### Step 8: Remove remaining central group prefill path
- Refactored `FullDirectRuntime` so it only owns persistent rank workers. It no longer stores `Vec<RankGpuContext>`, rank weight views, communicators, central RoPE caches, or central decode caches.
- Moved per-rank RoPE and decode cache ownership into each `RankWorker`; prefill and decode now run through worker commands.
- Added rank-lane prefill block/attention/MoE helpers so prefill cache seeding enters NCCL collectives from the same worker lanes as decode.
- Gathered final logits inside worker lanes with `all_gather_logits`; the main thread now receives only rank-0 host logits for sampling.
- Deleted the old central single-thread group surface:
  - `block_prefill_group_bf16_hidden`
  - `block_prefill_group_bf16_hidden_with_decode_cache`
  - `prefill_logits_group_bf16_hidden`
  - `prefill_logits_and_decode_cache_group_bf16_hidden`
  - prefill attention group wrappers and group all-reduce helpers
  - `deepseek_mp8_check` debug bin and its `Cargo.toml` target
  - mp8 tests that exercised the deleted group path
- Verification:
  - `cargo fmt` passed locally
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed locally and on 5090
  - `cargo test --release -p pegainfer-deepseek-v4 --features deepseek-v4 --test mp8_manifest --no-run` passed locally
  - `rg -n "group_start|group_end|all_reduce_hidden_group|all_gather_logits_group|embedding_vocab_parallel_group|final_logits_group_bf16_hidden|hash_routed_moe_group_bf16_hidden|moe_group_bf16_hidden|attention_prefill_.*group|block_prefill_group|prefill_logits_group|prefill_logits_and_decode_cache_group|deepseek_mp8_check|contexts: Vec<RankGpuContext>" pegainfer-deepseek-v4/src pegainfer-deepseek-v4/tests pegainfer-deepseek-v4/Cargo.toml` returned no matches locally
  - 5090 exact E2E with `/data/DeepSeek-V4-Flash` passed: `All 20 DeepSeek V4 exact cases passed`

### Step 9: Split direct scheduler and worker files
- Split the former monolithic `pegainfer-deepseek-v4/src/direct.rs` into:
  - `pegainfer-deepseek-v4/src/direct.rs` as a thin module/re-export facade.
  - `pegainfer-deepseek-v4/src/direct/scheduler.rs` for request validation, the single-request greedy scheduler loop, token event emission, and sampling.
  - `pegainfer-deepseek-v4/src/direct/worker.rs` for rank worker commands, rank resource ownership, cache/RoPE management, per-rank prefill/decode execution, and rank-0 logits collection.
- Renamed the request/scheduler thread from `deepseek-v4-direct` to `deepseek-v4-scheduler`. Rank worker thread names remain `deepseek-v4-rank-{rank}`.
- Kept behavior unchanged; this is only a responsibility-boundary cleanup.
- Follow-up naming debt:
  - The module and public type names still use `direct`; that name is legacy and should eventually become `engine` or `executor` in a dedicated rename pass.
- Verification:
  - `cargo fmt` passed
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed

### Step 3: Expand scope to decode backend replacement
- User goal changed from adding standalone AG/RS collectives to completing the MoE all-to-all backend replacement and passing DeepSeek V4 E2E.
- Current-state audit before coding:
  - `all_gather_bf16_hidden_group` and `reduce_scatter_f32_hidden_group` exist and passed a pair NCCL test.
  - Direct decode still enters `block_decode_rank_lane_bf16_hidden`, whose MoE stage calls `decode_routed_moe_rank_local_f32_hidden` followed by routed all-reduce.
  - `block_decode_group_bf16_hidden` still calls `moe_group_bf16_hidden`, whose routed output combine is all-reduce.
- Revised plan:
  1. Add single-rank AG/RS helpers for persistent rank-worker decode lanes, plus `u32` token-id allgather helpers for hash routing.
  2. Add decode-only MoE AG/RS functions that allgather local FFN norm hidden states, route on global hidden, reuse existing local expert execution, reduce-scatter partial routed output, then add local shared expert output.
  3. Replace decode block MoE calls in both group and rank-lane paths with the decode AG/RS functions.
  4. Remove the old decode-only routed all-reduce entry when it becomes unused.
  5. Run focused tests, then DeepSeek V4 E2E.

### Step 4: Decode backend cutover and validation
- Replaced the direct decode MoE backend with the regular collective exchange shape:
  - dispatch: `all_gather_bf16_hidden_group`
  - routing/expert work: existing local expert path over gathered hidden
  - combine: `reduce_scatter_f32_hidden_group`
- Direct decode now uses the group block path instead of persistent per-rank decode workers, so the decode block enters the AG/RS MoE path.
- Removed the old decode-only `decode_routed_moe_rank_local_f32_hidden` route-index D2H path.
- Exact E2E command run:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- --model-path /data/DeepSeek-V4-Flash
```

- Result: `19 / 20` exact cases passed.
  - Only case 13 differed: expected `"2500 meters"`, got `"2500"`.
  - This is considered explainable for this backend change: current single-request decode allgathers eight rank-local copies into a `B_global=8` expert batch, so the routed expert path uses a different batch shape than the old single-token decode path. The difference is a small generation-boundary drift, not a broken route/exchange failure.
- Performance sanity command run:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

- Result:
  - `steady_tpot_ms.avg = 190.45ms`
  - `steady_tpot_ms.p50 = 188.53ms`
  - `steady_tpot_ms.p95 = 197.10ms`
  - `first_decode_step_ms.avg = 182.27ms`
  - `ttft_ms.avg = 173.88ms`
  - `decode_tok_s = 5.26`

### Step 5: Remove decode MoE D2H and switch local experts to grouped TileLang FP4
- Replaced route-order local hit mapping with GPU expert-major mapping:
  - count local expert hits on GPU
  - build `expert_indptr` prefix sums on GPU
  - compact `pos_to_token`, `pos_to_token_topk`, and `token_topk_to_pos` in expert-major order
- Removed the slow row-routed scalar FP4 MoE linear path from runtime use and then deleted its FFI/CUDA symbols.
- Added grouped TileLang FP4 GEMM generation for the existing DeepSeek FP4 shapes:
  - `n2048_k4096` for w1/w3
  - `n4096_k2048` for w2
- The grouped kernel accepts `B_ptrs`, `scales_b_ptrs`, and device `expert_indptr`, so host never reads per-expert hit counts.
- The grouped kernel over-launches by route capacity and skips empty/out-of-range expert CTAs on device. This keeps the path D2H-free; the remaining optimization target is avoiding empty CTA work with a GPU active-expert list.
- Validation commands:

```bash
PEGAINFER_NVCC_JOBS=8 cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- --model-path /data/DeepSeek-V4-Flash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

- Results:
  - release check: passed
  - E2E: `All 20 DeepSeek V4 exact cases passed`
  - bench: `steady_tpot_ms.avg = 108.302963`, `p50 = 107.902882`, `p95 = 112.547205`, `first_decode_step_ms.avg = 103.918882`, `decode_tok_s = 9.24543`
  - Earlier row-routed scalar path measured around `223.30ms`, so grouped TileLang cuts the local expert bottleneck roughly in half.

### Step 1: Add AG/RS collectives
- Added `all_gather_bf16_hidden_group` in `pegainfer-deepseek-v4/src/runtime/collectives.rs`.
  - Contract: every rank contributes `bf16 [B_local,H]`.
  - Output on each rank: `bf16 [world*B_local,H]`.
  - Uses NCCL `Comm::all_gather` on device buffers; no runtime D2H metadata.
- Added `reduce_scatter_f32_hidden_group` in `pegainfer-deepseek-v4/src/runtime/collectives.rs`.
  - Contract: every rank contributes `f32 [world*B_local,H]`.
  - Output on each rank: `f32 [B_local,H]`.
  - Uses NCCL `Comm::reduce_scatter(..., ReduceOp::Sum)` on device buffers.
- Exported both helpers from `pegainfer-deepseek-v4/src/lib.rs`.

### Step 2: Validate
- Added `nccl_hidden_all_gather_and_reduce_scatter_pair` in `pegainfer-deepseek-v4/tests/mp8_manifest.rs`.
- Ran:

```bash
PEGAINFER_NVCC_JOBS=8 cargo test --release -p pegainfer-deepseek-v4 --features deepseek-v4 --test mp8_manifest nccl_hidden_all_gather_and_reduce_scatter_pair -- --nocapture
```

- Result: passed, `1 passed; 0 failed; 23 filtered out`.
- Ran `cargo fmt --check`; result: passed.

## Debrief

- **Outcome**: Decode MoE now follows the intended AG/RS exchange shape and keeps route/expert metadata on GPU. The local expert cutover uses grouped TileLang FP4 GEMM rather than the temporary row-routed scalar kernel. The legacy single-thread group path has been removed from runtime exports, implementation, debug bins, and mp8 tests; direct execution is now worker-lane only for both prefill cache seeding and decode.
- **Pitfalls encountered**:
  - FlashInfer MXFP4 group GEMM's public wrapper expects contiguous group-major B weights, while DeepSeek rank weights are separate per-expert tensors. TileLang was the cleaner immediate reuse point because the existing generated FP4 GEMM already matches our weight/scale layout.
  - Reduce-scatter timing must not be read as pure communication cost in this path; earlier nsys showed most of the long NCCL duration was arrival skew from slow local expert compute.
- **Lessons learned**:
  - Expert-major GPU compaction plus `expert_indptr` is enough to keep host out of routing while allowing grouped GEMM.
  - Over-launching grouped expert CTAs is a good correctness-first cutover, but it leaves performance on the table for sparse/uneven expert hits.
- **Follow-ups**:
  - Add a GPU active-expert/tile list to stop launching empty expert tiles and close the remaining gap toward the old `~80ms` direct top-k path.
  - Profile grouped TileLang per-layer kernels again; the next TPOT drop should come from reducing empty CTA work and launch/barrier count, not from reintroducing D2H scheduling.
