# Qwen3.5 DFlash Speculative Decoding

> **TL;DR:** Qwen3.5 DFlash is an opt-in, single-active greedy speculative path behind `--dflash-draft-model-path`; it is correctness-gated and does not claim concurrent throughput uplift yet.

Last touched: 2026-07

## What This Adds

Qwen3.5 can load a DFlash draft model alongside the target model and use it for speculative decoding when one greedy request is active. The default engine path is unchanged unless the draft path is provided.

The implementation treats speculation as a transaction:

1. capture target hidden features during prompt prefill;
2. run the DFlash draft model to propose a short token span;
3. verify the span with the Qwen3.5 target model;
4. accept the longest greedy prefix plus one target bonus token;
5. restore draft state and commit the accepted target span.

Qwen3.5 needs a stricter transaction than Qwen3 because it has both full-attention KV state and recurrent/conv state. The current implementation uses the conservative commit path: restore canonical state, verify target tokens through the normal decode path, then replay the accepted span.

## How To Enable

```bash
cargo run --release --features qwen35-4b -- \
  --model-path <Qwen3.5-4B> \
  --dflash-draft-model-path <Qwen3.5-DFlash>
```

## Verified Scope

Supported:

- single active request;
- greedy sampling;
- `logprobs=0`;
- Qwen3.5 single-GPU launch;
- prompt-prefill hidden capture before speculative decode;
- fallback to normal target decode when several requests are active.

Unsupported or rejected:

- LoRA with DFlash;
- KV offload with DFlash;
- tensor parallel launch with DFlash;
- decode-overlap with DFlash;
- concurrent speculative verify/commit;
- serving-throughput or vLLM parity claims.

## Validation

Correctness gates:

```bash
cargo fmt --all --check
git diff --check
OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
OPENINFER_DFLASH_TEST_MODEL_PATH=<Qwen3.5-DFlash> \
cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test dflash_speculative_gate -- --nocapture --test-threads=1
OPENINFER_TEST_MODEL_PATH=<Qwen3.5-4B> \
cargo test --release -p openinfer-qwen35-4b --features qwen35-4b \
  --test speculative_verify -- --nocapture --test-threads=1
```

The scheduler gate compares single-active DFlash output tokens with the plain greedy scheduler and checks that multi-active/logprobs fallback requests still finish through the normal path.

The single-active test also enables `OPENINFER_QWEN35_DFLASH_REQUIRE_SPEC=1` during DFlash generation. That diagnostic mode fails the request if a greedy single-active request is expected to speculate but cannot enter the DFlash path, so the test cannot pass by silently using normal decode.

RTX 5090 validation after the fallback-gating fix:

- `cargo fmt --all --check`: pass;
- `speculative_verify`: 6 passed;
- `dflash_speculative_gate`: 2 passed;
- `e2e_scheduler`: 1 passed.

The same run checked `bench_serving` in-process A/B at prompt 1 / output 64 and prompt 1 / output 256, c1/c4/c8/c16, with CUDA Graph enabled. Enabling the DFlash flag changed fallback TPOT by at most +0.51% in that smoke, so the remaining multi-active path is treated as normal-decode fallback evidence, not a throughput claim. Concurrent hash drift seen at c4/c16 also reproduced in repeated baseline runs.

## Remaining Risks And PR Boundary

This PR should fix risks that can break the opt-in single-active DFlash MVP or silently perturb the default path. The DFlash KV reservation, verify-span validation, and fallback prefill-capture gate are in that category and are handled here.

The following risks are real, but should stay out of this PR unless they start failing the single-active correctness gate:

- Multi-active DFlash throughput is not implemented. c4/c8/c16 currently use normal decode fallback, so batched verify, verify CUDA Graph capture, and prefix-state checkpoint commit belong in a follow-up performance PR.
- Concurrent Qwen3.5 hash drift is not unique to DFlash. Repeated baseline c4/c16 runs also change some hashes, so a deterministic near-tie oracle or batch-invariant sampling investigation should be tracked separately from this DFlash MVP.
- HTTP serving A/B is not closed by this document. The measured numbers are in-process `bench_serving` evidence, not OpenAI-compatible endpoint evidence.
- Single-active DFlash speedup is not claimed. The conservative verify and replay path is correctness-first; optimizing acceptance overhead and commit cost belongs after this opt-in path lands.

## Current Claim Boundary

This change proves that Qwen3.5 can run an opt-in DFlash path without changing the default engine path. It does not claim c4/c8/c16 throughput improvement. Multi-active batched verify, CUDA Graph capture for verify, and benchmark/profile closure remain follow-up work.
