# Qwen3-4B-DFlash model

**TL;DR**: `openinfer-qwen3-4b-dflash` supports only the `z-lab/Qwen3-4B-DFlash-b16` model. It now has two draft-only execution surfaces: the original bs1 transformers-parity forward path, and an internal exact-shape batch runner/scheduler that batches already-prepared `noise_embedding`, selected target hidden states, and `position_ids`. The forward gate currently measures mean delta `0.034243`, p99 `0.125000`, max `0.500000` over 7,680 output values for uncached, unified-cache one-shot, and first-step draft-cache paths; batch-vs-single and executor request-tag smoke extend that gate. Cache control APIs are fail-closed for unknown request ids. The scheduler thread now joins on handle drop (mirrors `EngineHandle`) and resident draft caches are bounded by `max_caches` with an explicit `drop_cache` retirement path (mirrors Qwen3 `drop_request`); over-cap admission fails closed. The batch K/V concatenation now uses a fused `strided_segment_copy` kernel instead of a per-request `memcpy_dtod` loop, lifting bs32 draft throughput from ~42K to ~63K tok/s (1.5x) with zero accuracy drift. Target verification, acceptance, fallback token selection, and OpenAI serving remain out of scope.

Last touched: 2026-06

## Boundary

This task is model-specific. The boundary is:

| Crate | Owns |
| --- | --- |
| `openinfer-qwen3-4b-dflash` | `Qwen3-4B-DFlash-b16` config, weights, draft forward, draft-only batch executor/scheduler, model-specific kernels/wrappers, and transformers parity tests |
| `openinfer-qwen3-4b` | Unchanged existing Qwen3 target serving, scheduler, KV, LoRA/offload/TP policy, and HF logits gate |

Out of scope for this task: generic speculative decoding, a generic DFlash abstraction, OpenAI/server flags, LoRA/TP/KV-offload interactions, target verification, acceptance-length calculation, fallback token selection, and target hidden extraction from Qwen3.

## Reference Model

The authoritative reference is the Hugging Face repo `z-lab/Qwen3-4B-DFlash-b16`, not an inferred architecture from the target Qwen3 crate. The model card uses:

```python
transformers==4.57.3
AutoModel.from_pretrained(..., trust_remote_code=True)
draft.spec_generate(target, input_ids, ...)
```

The local checkpoint at `/home/hezhaozhao/models/Qwen3-4B-DFlash-b16` contains the same remote-code shape:

| Field | Value |
| --- | --- |
| `architectures` | `DFlashDraftModel` |
| draft layers | `5` |
| target layers | `36` |
| hidden size | `2560` |
| intermediate size | `9728` |
| attention heads / KV heads | `32 / 8` |
| head dim | `128` |
| block size | `16` |
| mask token | `151669` |
| target hidden layers | `[1, 9, 17, 25, 33]` |
| vocab size | `151936` |

Checkpoint keys are unprefixed relative to a target `model.` namespace: `layers.*`, `fc.weight`, `hidden_norm.weight`, and `norm.weight`. `fc.weight` is `[2560, 12800]`, i.e. one hidden-sized projection from five concatenated target hidden states.

## Draft Forward

The draft forward is not target Qwen3 attention with a different checkpoint. Its attention is dense and non-causal:

1. `target_hidden = hidden_norm(fc(concat(selected target hidden states)))`
2. `hidden_states = noise_embedding`
3. for each of the five draft layers:
   - RMSNorm `hidden_states`
   - Q comes from normalized noise hidden
   - K/V come from `cat(target_hidden, hidden_states)`
   - Q/K get Qwen3 head RMSNorm and RoPE
   - attention is non-causal over the whole `target_hidden + noise_hidden` span
   - residual add
   - post-attention RMSNorm + Qwen3 MLP + residual add
4. final `norm(hidden_states)`

The crate should expose draft-model primitives, not speculative serving:

```rust
pub struct DFlashDraftModel { ... }

impl DFlashDraftModel {
    pub fn load(model_path: &Path, device_ordinal: usize) -> anyhow::Result<Self>;
    pub fn config(&self) -> &DFlashConfig;
    pub fn target_layer_ids(&self) -> &[usize];
    pub fn forward(
        &self,
        noise_embedding: &HiddenStates,
        selected_target_hidden: &DFlashTargetHidden,
        position_ids: &[i32],
    ) -> anyhow::Result<HiddenStates>;
}
```

The first version takes already-selected target hidden states as input and returns the final draft hidden states. Extracting those hidden states from `openinfer-qwen3-4b`, target verification, acceptance length calculation, and KV cropping are not part of this model implementation.

## Draft-Only Batch Runner

The batch path is intentionally internal. It is not an OpenAI-compatible text
generation surface because the DFlash draft model does not consume prompt token
ids and does not own a language-model head. Callers must provide device
`HiddenStates` for:

| Input | Shape |
| --- | --- |
| `noise_embedding` | `[q_len, hidden_size]` |
| `target_hidden` | `[ctx_len, target_layer_count * hidden_size]` |
| `position_ids` | `ctx_len + q_len` host positions |

The runner groups only exact-shape requests. The batch key is
`(q_len, ctx_len, past_len, cache_mode)`. `NoCache` requests use the real
batched path: compact D2D input staging, batched FC/context projection, batched
per-layer Q/K/V and MLP GEMMs, and FlashInfer
`BatchPrefillWithRaggedKVCache` in non-causal mode for attention. `DraftCache`
requests keep the same `DFlashDraftCache` lifecycle and are executed serially
inside the GPU owner thread in this step; cross-request draft-cache batching
needs a compact past-K/V layout and should be added with the target
verification loop.

The public Rust surface is crate-local serving infrastructure, not server API:

```rust
pub struct DFlashDraftHostRequest { ... }
pub struct DFlashDraftHostResponse { ... }
pub struct DFlashExecutor { ... }
pub struct DFlashSchedulerHandle { ... }
```

`DFlashSchedulerHandle` is a single-thread GPU owner with FCFS exact-shape
batching, a small `max_wait` coalescing window, and `max_total_tokens`
admission over `(ctx_len + q_len + past_len)` for each candidate batch. Its
public `submit` boundary uses host bf16 buffers and returns host bf16 output so
CUDA device tensors do not cross thread/context ownership boundaries. It also
owns per-request draft cache state through `reset_cache`, `crop_cache`,
`cache_seq_len`, and `drop_cache`, and the cache-reading calls error on unknown
request ids instead of silently treating them as empty state; `drop_cache` is
idempotent (a missing cache is not an error) so callers can retire a request
from any lifecycle state. Resident caches are bounded by `max_caches`
(`DFlashExecutorOptions`, default 64); exceeding it fails closed until a
retired request's cache is dropped — this mirrors Qwen3's per-request block
accounting under the fixed `KvCacheManager` pool and prevents the unbounded
GPU-memory leak the old grow-only `HashMap` had. The handle joins the scheduler
thread on drop (the last clone closes the channel and joins, mirroring
`EngineHandle`), so dropping the handle without an explicit shutdown no longer
leaks the GPU-owner thread. `NoCache` requests use the real batched path, while
host `DraftCache` requests run serially until compact past-K/V batching lands.
The executor also exposes a borrowed compact batch view for same-thread
controller experiments.

## Draft Cache

Do not maintain separate public cache concepts for this crate. The reference
Python uses one `past_key_values_draft = DynamicCache()` in `spec_generate`,
then calls the drafter with:

```python
position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size]
past_key_values=past_key_values_draft
use_cache=True
past_key_values_draft.crop(start)
```

OpenInfer mirrors that boundary with one `DFlashDraftCache`:

| State | Meaning |
| --- | --- |
| `prepare_step_context(...)` | Projects the current selected target hidden states and prepares per-layer context `K/V`; this replaces the old standalone `prepare_context_cache(...)` wording. |
| `forward_with_draft_cache(...)` | Runs one draft block, appends step context `K/V` and noise-token `K/V` to each layer's draft past state, and advances `seq_len`. |
| `crop(seq_len)` / `reset()` | Matches the reference `DynamicCache.crop(start)` lifecycle after target verification decides how far the draft state remains valid. |

The first-step cached path is numerically identical to the standalone HF
remote-code forward because there is no existing past yet. Cross-step cached
parity must be validated only after the target verification/controller is added;
without the target loop, a second cached draft step is not the same numerical
problem as the old no-draft-cache substitution probe.

## Correctness Gate

The accuracy bar is transformers parity. For the draft crate that means:

| Gate | Purpose |
| --- | --- |
| config/loader shape test | Reject wrong checkpoint layout early: `target_layer_ids`, `block_size`, `mask_token_id`, `fc.weight`, layer count, and attention/MLP shapes |
| draft-forward smoke | Load `/home/hezhaozhao/models/Qwen3-4B-DFlash-b16`, run a tiny GPU block with synthetic `noise_embedding`, selected target hidden states, and position ids, and catch shape/kernel failures |
| transformers forward parity | Compare the standalone draft forward against the HF remote-code model for fixed synthetic `noise_embedding`, selected target hidden states, and position ids |
| batch-vs-single parity | Compare two exact-shape batched rows against the bs1 forward output under the same DFlash tolerance |
| executor smoke | Submit request-tagged exact-shape `NoCache` requests and assert output shape/request ids |
| scheduler cache smoke | Submit host `DraftCache` request, then assert scheduler-owned `cache_seq_len`, `crop_cache`, and `reset_cache` behavior; also checks control messages preserve FIFO ordering behind pending submits |
| cache control rejection | `reset_cache` / `crop_cache` / `cache_seq_len` fail closed on unknown request ids; `drop_cache` is idempotent (retiring an unknown id is not an error) |
| drafter generation parity | Run a greedy bs1 transformers target loop twice, once with the HF drafter and once with the OpenInfer drafter, then compare generated token ids/text and acceptance lengths |

Do not use `Qwen3-4B-Instruct-2507` as a correctness baseline for this model. The checkpoint is documented for `Qwen/Qwen3-4B`, but this task's gate is the DFlash draft model's own transformers forward, not target acceptance rate.

## Kernel Notes

Existing Qwen3 target attention is causal/paged and does not match `Qwen3-4B-DFlash-b16` draft attention. The draft kernel path should follow vLLM/FlashAttention semantics where possible: Q/K/V in head-major logical shape, GQA expansion by `q_head / (num_q_heads / num_kv_heads)`, RoPE on Q and K, softmax over all context+draft keys, and no causal mask.

The reference implementation to mirror is vLLM's attention stack, especially `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend` and `vllm.v1.attention.backends.flashinfer.FlashInferBackend`: both explicitly support `supports_non_causal()`, and their prefill/decode planners expose the causal flag and varlen context shape that DFlash needs.

The batch runner uses FlashInfer `BatchPrefillWithRaggedKVCache` with
`MaskMode::kNone` for compact non-causal attention. That keeps the DFlash batch
path close to vLLM's varlen/non-causal attention semantics instead of looping
over single-request prefill.

## Accuracy Scripts

The DFlash scripts intentionally mirror the rest of the repository:

| Script | Output | Use |
| --- | --- | --- |
| `tools/accuracy/dump_qwen3_4b_dflash_hf_golden.py` | `test_data/qwen3-4b-dflash-hf-golden.safetensors` | Offline transformers remote-code forward oracle for the Rust gate |
| `openinfer-qwen3-4b-dflash/tests/hf_golden_gate.rs` | test pass/fail plus delta distribution | Release Rust gate that replays the stored oracle without Python |
| `tools/accuracy/compare_qwen3_4b_dflash_drafter_generation.py` | `target/accuracy/qwen3-dflash/drafter-generation.json` | End-to-end drafter-substitution evidence: same transformers target loop, HF drafter vs OpenInfer drafter |
| `tools/accuracy/bench_qwen3_4b_dflash_forward.py` + `qwen3_dflash_forward_bench` | `target/benchmarks/qwen3-dflash/forward.json` | Standalone forward latency comparison: transformers remote-code vs OpenInfer forward on the same synthetic fixture |
| `qwen3_dflash_batch_bench` | stdout JSON / redirected benchmark artifact | Draft-only batch sweep over bs `1,2,4,8,16,32`, reporting req/s, draft tok/s, and latency percentiles |
| `openinfer-qwen3-4b-dflash/src/bin/qwen3_dflash_forward_fixture.rs` | safetensors with `openinfer_output` | Bridge used by the generation comparison script to call the Rust drafter from Python |

The forward golden is generated by:

```bash
.venv/bin/python tools/accuracy/dump_qwen3_4b_dflash_hf_golden.py \
  --model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
  --out test_data/qwen3-4b-dflash-hf-golden.safetensors
```

The Rust gate is:

```bash
OPENINFER_DFLASH_TEST_MODEL_PATH=/home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
cargo test --release -p openinfer-qwen3-4b-dflash --test hf_golden_gate -- --nocapture
```

The DFlash gate intentionally uses `OPENINFER_DFLASH_TEST_MODEL_PATH` rather
than the generic `OPENINFER_TEST_MODEL_PATH`, because the latter usually points
at the normal Qwen3 target checkpoint. The test also checks that
`config.json.architectures` contains `DFlashDraftModel` before running.

The batch throughput probe is:

```bash
cargo run --release -p openinfer-qwen3-4b-dflash --bin qwen3_dflash_batch_bench -- \
  --model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
  --ctx-len 2 \
  --q-len 16 \
  --batch-sizes 1,2,4,8,16,32 \
  --warmup 5 \
  --iters 30
```

Observed local batch runner sweep on the same WSL/CUDA `sm_120` setup,
`ctx_len=2`, `q_len=16`, warmup `5`, iters `30`:

| Batch | mean ms | p50 ms | p90 ms | p99 ms | draft tok/s | req/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.065 | — | — | — | 7,748 | — |
| 2 | 2.154 | — | — | — | 14,856 | — |
| 4 | 3.118 | — | — | — | 20,525 | — |
| 8 | 3.335 | — | — | — | 38,382 | — |
| 16 | 4.699 | — | — | — | 54,476 | — |
| 32 | 8.178 | — | — | — | 62,611 | — |

The batch path now improves draft-token throughput by `8.1x` from bs1 to bs32.
The bs16/bs32 step gained ~1.5x after replacing the per-request `compact_kv`
memcpy loop (`2 * batch_size` `memcpy_dtod` calls per K/V tensor per layer)
with a single fused `strided_segment_copy` CUDA kernel — one launch copies the
entire batch's ctx segment, another the noise segment, collapsing 128
launches/layer at bs32 into 4. This is draft-model throughput only; it does not
include target hidden production, verification, acceptance, or fallback-token
work.

On the local WSL setup used for the first run, the workspace-level vLLM git dependency and empty FlashInfer submodule required a narrower temporary workspace plus:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib \
OPENINFER_FLASHINFER_INCLUDE=/home/hezhaozhao/openinfer/.venv/lib/python3.12/site-packages/flashinfer/data/include \
cargo test --release -p openinfer-qwen3-4b-dflash --test hf_golden_gate -- --nocapture
```

Observed result after the unified cache change:

```text
dflash HF golden deltas: mean=0.034243, p99=0.125000, max=0.500000, n=7680
dflash unified-cache one-shot HF golden deltas: mean=0.034243, p99=0.125000, max=0.500000, n=7680
dflash draft-cache HF golden deltas: mean=0.034243, p99=0.125000, max=0.500000, n=7680
test dflash_forward_matches_hf_remote_code ... ok
```

The drafter-substitution generation probe is:

```bash
cargo build --release -p openinfer-qwen3-4b-dflash --bin qwen3_dflash_forward_fixture

.venv/bin/python tools/accuracy/compare_qwen3_4b_dflash_drafter_generation.py \
  --target-model-path /path/to/Qwen3-4B \
  --draft-model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
  --openinfer-bin target/release/qwen3_dflash_forward_fixture \
  --out target/accuracy/qwen3-dflash/drafter-generation.json
```

The JSON report records each prompt's generated token ids/text, token/text hashes,
first mismatch if any, acceptance lengths, and optional OpenInfer-vs-HF draft
hidden deltas. It exits non-zero unless every case is `all_token_text_exact`.
This is the DFlash analogue of the DeepSeek-V2-Lite same-host generation
comparison, but scoped to the current standalone drafter boundary.

For performance, use the same synthetic fixture on both sides:

```bash
cargo build --release -p openinfer-qwen3-4b-dflash --bin qwen3_dflash_forward_bench

.venv/bin/python tools/accuracy/bench_qwen3_4b_dflash_forward.py \
  --draft-model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
  --openinfer-bin target/release/qwen3_dflash_forward_bench \
  --out target/benchmarks/qwen3-dflash/forward.json
```

The benchmark report includes transformers latency stats and OpenInfer latency
stats for the same bf16 fixture. It is a standalone draft-forward measurement,
not a full speculative-decoding throughput claim.

Observed local benchmark on RTX 5070 Ti, WSL, CUDA `sm_120`, `ctx_len=2`,
`q_len=16`, warmup `5`, iters `30`, same generated bf16 fixture:

| Engine | mean ms | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| transformers remote-code | 4.294 | 3.612 | 5.067 | 15.360 |
| OpenInfer DFlash | 2.285 | 2.195 | 2.659 | 2.895 |

OpenInfer is `1.65x` faster at p50 and `1.88x` faster by mean for this
standalone forward shape. The transformers p99 includes a single 15.36 ms tail
in this short run, so p99 should not be over-interpreted without a longer sweep.
The measured artifact is `target/benchmarks/qwen3-dflash/forward.json`.

First optimization pass: `DFlashForwardScratch` reuses the forward buffer set
across repeated calls. The HF forward gate stayed identical:
`mean=0.034243`, `p99=0.125000`, `max=0.500000`, `n=7680`. The same forward
benchmark wrote `target/benchmarks/qwen3-dflash/forward-final.json`:

| OpenInfer path | mean ms | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| allocate buffers per forward | 2.285 | 2.195 | 2.659 | 2.895 |
| reuse `DFlashForwardScratch` | 2.125 | 2.035 | 2.410 | 2.936 |

This pass improved OpenInfer p50 by `1.08x`. It is a necessary cleanup for the
future decode loop, but not enough by itself to prove DFlash value.

A follow-up attempt to move the cloned input hidden state into reusable scratch
was not kept: the current fused residual+RMSNorm op mutates the residual hidden
state in place, so separating input/output ping-pong buffers correctly requires
reworking that layer boundary rather than a local buffer-only patch.

Second optimization pass: `DFlashForwardScratch` gained an explicit draft-side
target-hidden context K/V cache. `prepare_context_cache(...)` computes
`target_normed` plus each layer's context `K/V` and K norm+RoPE once; repeated
`forward_with_context_cache(...)` calls then only compute the noise-token K/V and
concat cached context with the current draft block. The HF gate now checks both
uncached and cached paths, and both stayed identical:
`mean=0.034243`, `p99=0.125000`, `max=0.500000`, `n=7680`.

Cached benchmark artifact: `target/benchmarks/qwen3-dflash/forward-context-cache.json`.
The reported latency excludes the one-time `prepare_context_cache(...)` call,
matching the intended loop shape where context cache is updated explicitly when
target hidden changes.

| OpenInfer path | mean ms | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| allocate buffers per forward | 2.285 | 2.195 | 2.659 | 2.895 |
| reuse `DFlashForwardScratch` | 2.125 | 2.035 | 2.410 | 2.936 |
| reuse scratch + context K/V cache | 1.863 | 1.831 | 2.001 | 2.301 |

The context cache improves p50 by `1.11x` over scratch-only and `1.20x` over the
initial implementation for this small `ctx_len=2`, `q_len=16` fixture.

Third pass: the public cache shape was unified as `DFlashDraftCache`. The old
"context cache" is now just the step-context part of the same object, and the
cache also owns per-layer draft past K/V buffers plus `seq_len`, `crop`, and
`reset` state. The HF gate checks uncached, unified-cache one-shot, and first-step
draft-cache paths; all three retain the same delta distribution:
`mean=0.034243`, `p99=0.125000`, `max=0.500000`, `n=7680`.

The cache internals now follow the `openinfer-kv-cache` separation more closely
without directly adopting its paged block manager: `DFlashDraftState` owns the
long-lived draft past K/V and sequence length, `DFlashStepContext` owns the
current target-hidden context K/V, and `ForwardBuffers` remains transient
scratch. The public object is still a single `DFlashDraftCache`, but a prepared
step is consumed by `forward_with_draft_cache(...)`; callers must prepare the
next step explicitly after `crop(start)`, mirroring the reference `DynamicCache`
lifecycle.

The corresponding benchmark artifact is
`target/benchmarks/qwen3-dflash/forward-draft-cache.json`. This benchmark uses
the more honest `prepare_step_context + forward_with_draft_cache` timing inside
each measured iteration, so it should not be compared directly against the
previous context-cache number that excluded prepare time:

| Engine/path | mean ms | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| transformers remote-code | 5.564 | 4.429 | 9.078 | 18.713 |
| OpenInfer `DFlashDraftCache` first-step path | 2.311 | 2.209 | 2.479 | 3.519 |

After the internal state/step/scratch refactor, the same benchmark wrote
`target/benchmarks/qwen3-dflash/forward-draft-cache-refactor.json` with no
accuracy change and no performance regression:

| Engine/path | mean ms | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| transformers remote-code | 4.242 | 3.861 | 5.616 | 6.922 |
| OpenInfer `DFlashDraftCache` refactor path | 2.228 | 2.155 | 2.454 | 2.541 |

## Current Implementation

The crate now exists as a standalone model implementation with config parsing, exact-key safetensor loading, a block draft forward, unified draft cache state, a tiny local GPU smoke test, and a HF remote-code golden gate. The attention path uses the existing Qwen3 Q/K RMSNorm+RoPE kernel and a FlashInfer single-prefill wrapper with `MaskMode::kNone`; context K currently reuses the Q/K kernel with a throwaway Q scratch buffer, so a future cleanup can split a K-only norm+RoPE helper without changing semantics.

The local `.venv` uses `torch==2.9.0+cu129`, `transformers==4.57.3`, `safetensors`, `accelerate`, and `datasets` because the HF remote code imports `datasets` via `utils.py`. The generated fixture stores seed-pinned synthetic `noise_embedding`, selected `target_hidden`, `position_ids`, and HF final `output`; `openinfer-qwen3-4b-dflash/tests/hf_golden_gate.rs` replays those tensors through the Rust forward and compares deltas.

An additional end-to-end generation probe used the same transformers target
model for verification and swapped only the drafter:

| Prompt | Result |
| --- | --- |
| `Hello, my name is` | identical token ids/text; acceptance `[1, 2, 1, 2, 1, 1]` |
| `The capital of France is` | identical token ids/text; acceptance `[2, 1, 2, 2, 2]` |
| `Qwen is a language model that` | identical token ids/text; acceptance `[2, 2, 1, 1, 1, 1]` |
| `1, 1, 2, 3, 5,` | identical token ids/text; acceptance `[4, 1, 2, 2]` |

The probe intentionally used a no-draft-cache loop on both sides because it
predates `DFlashDraftCache` and because `openinfer-qwen3-4b-dflash` still does
not own the target verification/controller. Within that older boundary,
OpenInfer DFlash produces the same greedy generation tokens as the transformers
DFlash drafter when the target/verification path is held fixed. The next
meaningful generation probe should use the real target loop and exercise
`DFlashDraftCache.crop(start)` after acceptance calculation.

## 2026-06-18 Batch Bench

The current Codex runner needed an explicit runtime library path to see the WSL
CUDA driver:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib \
OPENINFER_FLASHINFER_INCLUDE=/home/hezhaozhao/openinfer/.venv/lib/python3.12/site-packages/flashinfer/data/include \
cargo run --release -p openinfer-qwen3-4b-dflash --bin qwen3_dflash_batch_bench -- \
  --model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
  --ctx-len 2 \
  --q-len 16 \
  --batch-sizes 1,2,4,8 \
  --warmup 2 \
  --iters 5
```

Observed result on the RTX 5070 Ti host:

| Batch | mean ms | draft tok/s | req/s |
| ---: | ---: | ---: | ---: |
| 1 | 2.052 | 7,796 | 487 |
| 2 | 2.303 | 13,893 | 868 |
| 4 | 3.532 | 18,121 | 1,133 |
| 8 | 4.364 | 29,333 | 1,833 |

This confirms the draft-only batch path still scales after the fail-closed
cache fix. It is draft throughput only; it does not include target hidden
production, verification, acceptance, or fallback-token work.
