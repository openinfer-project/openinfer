# Simulated Inference Engine

**Created**: 2026-05-16
**Status**: ready for PR review
**TL;DR**: `pegainfer-sim` is a CPU-only simulated model crate that serves through the existing vLLM/OpenAI frontend with configurable TTFT/TPOT. It is a benchmark and frontend validation harness, not a real-model performance path.

## Scope

Issue #125 needs a server path that can run `vllm bench serve` without GPU or model weights while still exercising the same HTTP frontend used by real pegainfer models.

This PR keeps that boundary narrow:

- Add `pegainfer-engine` for the lightweight `EngineHandle`, `GenerateRequest`, `TokenEvent`, and `SamplingParams` contract.
- Re-export that contract from `pegainfer-core` so existing model crates keep their current imports.
- Move the vLLM bridge into `pegainfer-vllm-frontend`, leaving `pegainfer-server/src/vllm_frontend.rs` as a compatibility re-export.
- Add `pegainfer-sim` as an independently maintained model crate with a thin CLI binary.

Out of scope:

- No CUDA, kernel, KV-cache, or real model execution changes.
- No claim about real model serving throughput.
- No jitter, tail-latency distribution, or batching realism beyond fixed TTFT/TPOT timing.

## Behavior

`pegainfer-sim` exposes CLI knobs for model id, port, max model length, base TTFT, prefill throughput, TPOT, and fallback token id.

The timing model is intentionally simple: TTFT is `base_ttft_ms + prompt_len / prefill_tokens_per_ms`, and TPOT is a fixed delay between generated tokens. Output token ids cycle through the prompt tokens, using the fallback id for empty prompts.

The frontend still needs tokenizer/model metadata, but the simulator never loads model weights.

## Implementation Details

- `pegainfer-engine` owns the shared engine contract, while `pegainfer-core` only re-exports it for existing model crates.
- `pegainfer-vllm-frontend` owns the bridge logic; `pegainfer-server/src/vllm_frontend.rs` stays as a compatibility re-export.
- `pegainfer-sim` is kept as a separate model crate so future simulation changes do not have to live inside the real model crates.

## Future Plans

Frontend e2e and CPU profiling are the next useful follow-ups for this crate boundary.

If reviewers want richer simulation, add jitter, tail distributions, and batching behavior in follow-up PRs after this crate boundary lands.
