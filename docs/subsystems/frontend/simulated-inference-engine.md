# Simulated Inference Engine

**Created**: 2026-05-16
**Status**: ready for PR review
**Last touched**: 2026-05
**TL;DR**: `openinfer-sim` is a CPU-only simulated model crate that serves through the existing vLLM/OpenAI frontend with configurable TTFT/TPOT and lightweight HTTP e2e coverage. It is a benchmark and frontend validation harness, not a real-model performance path.

## Scope

Issue #125 needs a server path that can run `vllm bench serve` without GPU or model weights while still exercising the same HTTP frontend used by real openinfer models.

This PR keeps that boundary narrow:

- Add `openinfer-engine` for the lightweight `EngineHandle`, `GenerateRequest`, `TokenEvent`, and `SamplingParams` contract.
- Re-export that contract from `openinfer-core` so existing model crates keep their current imports.
- Move the vLLM bridge into `openinfer-vllm-frontend`, leaving `openinfer-server/src/vllm_frontend.rs` as a compatibility re-export.
- Add `openinfer-sim` as an independently maintained model crate with a thin CLI binary.

Out of scope:

- No CUDA, kernel, KV-cache, or real model execution changes.
- No claim about real model serving throughput.
- No jitter, tail-latency distribution, or batching realism beyond fixed TTFT/TPOT timing.

## Behavior

`openinfer-sim` exposes CLI knobs for model id, port, max model length, base TTFT, prefill throughput, TPOT, and fallback token id.

The timing model is intentionally simple: TTFT is `base_ttft_ms + prompt_len / prefill_tokens_per_ms`, and TPOT is a fixed delay between generated tokens. Output token ids cycle through the prompt tokens, using the fallback id for empty prompts.

The frontend still needs tokenizer/model metadata, but the simulator never loads model weights.

## Frontend Metadata Contract

`openinfer-sim` does not load model weights, but serving it through the
vLLM/OpenAI frontend still constructs the normal text/chat backend. That
frontend path requires enough local model metadata to initialize tokenization
and detokenization.

For CPU-only tests that do not intend to exercise tokenizer encoding, use
token-id prompts. Generated token ids still pass through detokenization, so the
test fixture must provide at least a tokenizer source such as `tokenizer.json`.
`tokenizer_config.json` and `config.json` are useful for EOS and context-window
metadata, but no weight files are required.

Chat-completions tests also need a `chat_template` in
`tokenizer_config.json`. Keep the minimal template deterministic and ensure it
renders at least one token that the simulated engine can stream as observable
content; otherwise response-shape tests can pass without exercising
`delta.content`.

## Implementation Details

- `openinfer-engine` owns the shared engine contract, while `openinfer-core` only re-exports it for existing model crates.
- `openinfer-vllm-frontend` owns the bridge logic; `openinfer-server/src/vllm_frontend.rs` stays as a compatibility re-export.
- `openinfer-sim` is kept as a separate model crate so future simulation changes do not have to live inside the real model crates.

## Future Plans

Frontend e2e and CPU profiling are the next useful follow-ups for this crate boundary.

If reviewers want richer simulation, add jitter, tail distributions, and batching behavior in follow-up PRs after this crate boundary lands.
