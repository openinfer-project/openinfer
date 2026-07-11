# vLLM model-based speculative decoding × P/D behavior audit

> **TL;DR:** For model-based speculators that own KV (EAGLE/EAGLE3, DFlash/DSpark, and applicable MTP variants), vLLM `0206f10871` runs the drafter on P and transfers draft KV alongside target KV; on a full remote prompt hit, D restores both and replays only the final prompt token for logits. Methods without drafter KV do not add this payload. The checked-in P/D gates cover EAGLE3/MTP but not DSpark/GLM5.2, so OpenInfer should treat DSpark suffix-only cold-start as an explicit measured alternative, never as valid full-history draft state.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — routed the investigation to the GLM5.2 model domain.
  - `docs/models/glm52/pegaflow-offload-pd.md` — OpenInfer M2 currently plans to transfer target MLA + index-K arenas from vLLM prefill to OpenInfer decode; partial tail blocks remain local work on D.
  - `docs/models/glm52/dspark-mtp.md` — OpenInfer DSpark owns separate draft KV and consumes five target aux-hidden captures; the current prefix-hit boundary is unsupported.
  - `docs/models/glm52/serving-status.md` — DSpark × prefix cache is the ordered non-perf follow-up after P/D M2.
- **Relevant history**:
  - `docs/models/qwen3/pd-disaggregation-m2.md` — existing P/D path transfers content-addressed target KV through PegaFlow and explicitly falls back to local prefill on remote-fetch failure.
- **Plan**:
  1. Record the sibling vLLM checkout identity, then locate its speculative-decoding implementation for DSpark/DFlash and its KV-connector v1 P/D receive path.
  2. Trace the exact tensors and metadata exported by prefill and installed by decode, distinguishing target MLA/indexer caches from drafter KV, aux hidden, and proposal state.
  3. Trace decode startup after an external KV load: whether vLLM reruns a suffix, performs one ordinary target step, reconstructs draft state, suppresses speculation, or rejects the combination.
  4. Compare the discovered contract with OpenInfer #590 and update this document plus `pegaflow-offload-pd.md` only where the vLLM evidence changes the design.
- **Risks / open questions**:
  - The audited revision may implement generic speculative decoding but not the external `vllm-project/speculators` DSpark plugin; absence will be reported as absence, not inferred behavior.
  - vLLM's connector may deliberately treat draft models as separate KV groups without a GLM5.2-specific integration; generic connector behavior and model-specific support must be distinguished.
  - The checkout is at `0206f10871`, newer than the `cdab28319` source snapshot cited by the existing GLM5.2 layout survey.

## Execution Log

- Audited vLLM revision `0206f10871` on `main`.
- Located the P/D + speculative-decoding fix in vLLM commit `90f3c01fa4dfc00d13beb8ae758d43365f7ba91f` ([vLLM #35158](https://github.com/vllm-project/vllm/pull/35158)). The target forward now defers KV-connector finalization until after the draft forward specifically so the producer can save draft-model KV as well as target-model KV.
- Confirmed the current path still implements that contract: `gpu_model_runner.py` finalizes the connector after the speculator runs; `get_kv_cache_spec()` and connector registration enumerate all attention layers; draft attention layers are separate KV groups initialized by the speculator. NIXL registers every resulting cache tensor, with no target-only filter.
- Traced the model-based speculator hierarchy: EAGLE uses `AutoRegressiveSpeculator`, while DFlash/DSpark use the parallel-context path; all inherit `DraftModelSpeculator`, discover their own attention layers, and allocate separate KV groups. For DFlash/DSpark, P consumes target aux-hidden captures to materialize drafter context KV before proposing. Aux hidden and transient proposal state are not connector payloads; the resulting draft K/V pages are.
- Checked the NIXL acceptance gate: the producer and consumer both load the same speculator; the producer uses one speculative token, while the consumer uses the serving width (three for the EAGLE3 case and one for the MTP case). The gate compares disaggregated acceptance length with standalone and treats a drop as evidence that drafter KV was not transferred. The checked-in matrix covers EAGLE3 and MTP, not DSpark/GLM5.2, so this proves the generic mechanism rather than GLM-specific interoperability.
- Pinned the PR lineage: [vLLM #35158](https://github.com/vllm-project/vllm/pull/35158) / `90f3c01fa4d` introduced the generic fix and EAGLE3 acceptance gate; [vLLM #41869](https://github.com/vllm-project/vllm/pull/41869) / `24337fb860a8` added Qwen3.5 GDN state support to NIXL; [vLLM #42677](https://github.com/vllm-project/vllm/pull/42677) / `129019f3342f` added the Qwen3.5 MTP row to the two-GPU P/D acceptance sweep. That MTP row explicitly enables the same MTP model on both P and D; there is no P-without-MTP/D-with-MTP case.
- Statically traced that asymmetric Qwen3.5 case through NIXL: enabling MTP on D adds cache regions that a target-only P does not register. Handshake validation requires matching region counts, so this configuration fails KV loading. The default `kv_load_failure_policy` is `fail`; `recompute` can recover by rebuilding the prefix locally on D, but forfeits that request's disaggregated-prefill benefit. Proposal widths may differ, as long as P and D load the same cache-owning speculator. This asymmetric configuration was not run by the checked-in gate.
- Traced the request boundary: the toy proxy sends P a copy with `max_tokens=1`, discards P's sampled text, and sends D the original prompt plus transfer metadata. D accounts the loaded prompt as externally computed tokens; after a full hit, `_update_waiting_for_remote_kv()` changes `num_computed_tokens` from `N` to `N-1` so D runs exactly the final prompt token and obtains first-token logits. Historical drafter KV is not reconstructed on D; the boundary draft row is overwritten/appended by the normal proposal step.
- Confirmed DSpark support is newer than the P/D fix: DSpark landed on 2026-07-01 in `f5a8d73377` / [vLLM #46995](https://github.com/vllm-project/vllm/pull/46995) and is forced onto Model Runner V2. V2 loads the speculator before collecting cache specs, calls connector `post_forward()` only after `speculator.propose()`, and registers the resulting target and draft cache tensors together. Its DSpark e2e gate is Qwen3 standalone, not P/D or GLM5.2.
- Compared state sizes for the OpenInfer GLM5.2 contract: existing target state is `78 × 656 + 21 × 132 = 53,940 B/token` (about 52.7 KiB); five-layer BF16 draft K/V is `5 × 2 × 4096 × 2 = 81,920 B/token` (80 KiB). Stateful target+draft transfer is therefore about 2.52× target-only. Five target aux-hidden captures would be 60 KiB/token but require D-side draft context precompute; vLLM does not use that representation.
- Updated `pegaflow-offload-pd.md` and `serving-status.md`: “M2 only needs hashing” now applies only to target KV. DSpark has two explicit paths—suffix-only cold-start with acceptance measurement, or vLLM-style draft-KV transfer with an independent layout gate.

## Debrief

### Result

For any model-based speculator with its own attention cache, vLLM's answer is state transfer, not drafter recovery. Its logical checkpoint contains all attention cache groups owned by the request: target state and the speculative model's K/V. P must load and execute the same speculator so those draft pages exist before connector finalization. Transient draft tokens and auxiliary hidden tensors are not handed off. Lookup-only methods such as ngram/suffix have no drafter KV to transfer; the rule is about state ownership, not the marketing name of the speculation algorithm.

On vLLM NIXL's full-prompt-hit path, D still performs one target forward at the boundary. This is the ordinary full-cache-hit rule needed to regenerate logits from the last prompt token; it does not rebuild draft history. Running one more target step cannot recover old aux hidden from target KV, because the target KV representation has discarded those intermediate residual streams. OpenInfer's current sealed-block PegaFlow protocol instead leaves a 1–64-token suffix for D to prefill locally.

P and D therefore need compatible state ownership, not necessarily identical proposal widths. Static inspection of Qwen3.5 MTP shows that P-without-MTP and D-with-MTP have different KV region counts and fail NIXL handshake validation. Falling back to D-side recomputation is possible only when configured explicitly, and turns that request back into local prefill.

For OpenInfer, “give DSpark bad KV and let verification reject it” is the wrong state model. Verification protects output tokens from a low-quality but well-formed proposal; it does not make stale pages, holes marked committed, or mismatched RoPE positions valid. A target-only compatibility implementation must reset draft KV and describe its context as suffix-only. Current `Glm52DsparkSlotState` couples absolute anchor position to `committed_len + pending_len`, so #590 needs an explicit absolute-position base or equivalent compact-history mapping before that cold-start is structurally valid.

### Verification

- Static call-chain audit against sibling vLLM `0206f10871`, including the original #35158 diff and current Model Runner V1/V2, scheduler, NIXL worker, DSpark/DFlash speculator, proxy, and acceptance tests.
- Confirmed no DSpark or GLM5.2 entry exists in the checked-in NIXL P/D acceptance matrix; no model-specific result is inferred from the generic path.
- `git diff --check` passes for the OpenInfer documentation edits. No GLM5.2/DSpark runtime benchmark was run as part of this source audit, so no suffix-cold-start acceptance or transfer-performance claim is made.

### Follow-up gates

1. Target-only M2: hash compatibility plus byte parity for the 99 target arenas.
2. #590 cold-start probe: explicit empty draft cache, absolute positions preserved, current 1–64-token uncached suffix prefill, then acceptance split into first round and steady state versus full-context standalone.
3. Only if that loss is unacceptable: define a draft-KV namespace/layout contract and byte-compare vLLM P pages with OpenInfer D pages before measuring the 2.52× payload path.
