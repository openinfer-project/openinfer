# Qwen3-4B n-gram speculative decoding (design)

**TL;DR**: Draft-model-free speculative decoding for Qwen3-4B. An n-gram
(prompt-lookup) proposer suggests up to `K` continuation tokens from the
request's running context; the target model verifies them in one forward pass
and commits the longest greedy-agreed prefix plus one model token. Greedy
verification is **lossless** — output is bit-for-bit identical to plain greedy
decode, just fewer forward passes on repetitive / structured text. It is a second
*propose* implementation on the shared speculative core (#436/#442): verify,
greedy acceptance, and the KV transaction are reused unchanged; only *propose*
differs. An **acceptance gate** closes speculation when recent draft acceptance
is too low to pay for the verify forward, so low-acceptance traffic (prose) falls
back to plain decode instead of regressing — see *Serving A/B*.

Last touched: 2026-06 · Enabled with `--ngram-speculative` (off by default).
Proposer, host-side propose, the shared verify/accept/KV transaction, scheduler
wiring, and the acceptance gate are landed and GPU-validated lossless.

## Shape: one transaction, only *propose* varies

Speculative decoding is a single optimistic transaction; the only method-specific
part is where the drafts come from. The loaded method is one enum on the
executor (`openinfer-qwen3-4b/src/executor.rs`):

```rust
enum SpeculativeMethod {
    Dflash(DFlashMeta),     // worker-side draft-model forward, captures hidden state
    Ngram(NgramRuntime),    // host-side prompt-lookup, no draft model
}
// Qwen3Executor::speculative_method: Option<SpeculativeMethod>  — single source of truth
```

This is the pre-trait consolidation #436 deferred. A full `SpeculativeProposer`
trait object is **not** introduced: DFlash's propose is a worker RPC (needs
`&mut executor` + lane), so it can't be a self-contained proposer without borrow
gymnastics. Enum dispatch is the honest shape until a third method (EAGLE, #325)
makes a shared trait pay off.

Per step (`scheduler/plan.rs`, `ExecutionPlan::SpeculativeDecode`):

1. **Propose** — produce up to `K` candidate tokens per request.
2. **Verify** — one target forward over each request's `K + 1` span
   `[current, draft_1..draft_K]`, argmax at every position.
3. **Accept** — `accept_greedy` keeps the longest prefix matching the target
   argmax, then appends the target's own token at the first mismatch (always
   commits `1..=K + 1` tokens).
4. **Commit / roll back** — accepted KV committed; the unused draft tail is
   LIFO-released by kvbm (`RequestKv::apply_speculative`).

Because the draft↔verify boundary is a pure token span, `accept_greedy` /
`build_verify_results`, the verify transaction
(`executor/spec.rs::execute_speculative_verify_impl`), and the
`schedule/apply/revert_speculative` KV lifecycle are **reused unchanged** from
the DFlash core. Only two things are method-specific: where drafts come from, and
whether verify captures hidden states.

## What n-gram adds

- **Proposer** (`ngram.rs`, `NgramProposer`) — stateless longest-suffix
  prompt-lookup over a request's own token history: tries the longest configured
  suffix first, falls back to shorter. Unit-tested in isolation.
- **`NgramRuntime`** (`ngram.rs`) — owns the method's entire state: the proposer,
  the acceptance gate (`NgramGate`), and the per-request running context map. The
  executor drives it through intent-level calls — `seed`, `append_committed`,
  `drop_request`, `is_ready`, `propose_step`, `record_verified` — instead of
  maintaining three parallel fields across every execution path.
- **Host-side propose** — `Qwen3Executor::execute_speculative_draft` calls
  `NgramRuntime::propose_step` on the executor thread: no worker forward, no draft
  model, no readiness handshake. (DFlash instead dispatches
  `StepCommand::SpeculativeDraft` to the lane.)
- **No-capture verify** — `LocalQwen3Lane::execute_ngram_verify` reuses
  `batch_prefill_into` with **zero capture layers** (a token-only proposer keeps
  no model state to seed), then argmax + `build_verify_results`. The worker routes
  verify by `lane.dflash.is_some()`; the methods are mutually exclusive.
- **Running context** (`NgramRuntime.ctx`) — per-request token context, seeded at
  prefill (`prompt + first_token`), appended at every commit path (verify / plain
  decode / fused unified decode, plus a defensive overlap branch), dropped on
  retire. Always ends with the request's dangling token so the next draft
  continues from it.

## Enabling it

`--ngram-speculative` (server flag) → `Qwen3LaunchOptions.ngram_speculative` →
`scheduler::start_qwen3` → `Qwen3Executor::load_ngram_drafter(NgramConfig::from_env())`.

- Single-GPU greedy only; **requires `--tp-size=1`**; mutually exclusive with
  `--dflash-draft-model-path`, `--enable-lora`, `--kv-offload`, and
  `--decode-overlap` (validated in `lib.rs::launch` and `server/src/main.rs`).
- Forces the prefix cache off — the verify forward writes each request's own
  speculative KV span, so cross-request prefix reuse is unsafe here.

Proposer knobs (`NgramConfig::from_env`):

- `OPENINFER_QWEN3_NGRAM_TOKENS=K` — draft count (default 4).
- `OPENINFER_QWEN3_NGRAM_MAX_NGRAM=N` — longest matched suffix (default 3).
- `OPENINFER_QWEN3_NGRAM_ACCEPT_THRESHOLD=f` — acceptance gate threshold (mean
  accepted draft tokens/step below which speculation falls back to plain decode;
  default 0.3, `0` disables the gate). See *Serving A/B*.

**Per-request eligibility.** `should_speculative_decode` takes the speculative
path only when *every* active request is greedy (`SamplingParams::is_greedy()`),
asks for no decode logprobs (`logprobs == 0`), and is non-LoRA — a single
ineligible request falls the whole batch back to plain decode for that step. So
enabling speculation never changes a sampled request's output or strips requested
logprobs. A miss (length-1 verify span) already degrades to a plain single-token
decode in `build_verify_results`, so it needs no special fallback.

## Why it is lossless

`accept_greedy` keeps only the prefix where `draft[i] == target_argmax[i]`, then
appends one of the target's own tokens, so the committed sequence is exactly what
plain greedy decode would have produced one token at a time. This is a built-in
oracle: **spec-on must equal spec-off** token-for-token under greedy params
(modulo the benign prefill-vs-decode bf16 tie flip). The gate only changes
*whether* we speculate, never the committed tokens, so it preserves this.

GPU-validated by `tests/ngram_speculative_gate.rs` (3 tests, real Qwen3-4B):
greedy-matches-plain (bs=1), concurrent heterogeneous (bs>1 verify-span path),
and mixed greedy+sampling. The proposer and gate also have isolated unit tests in
`ngram.rs`.

## Serving A/B (`vllm bench serve`) and the acceptance gate

Greedy speculation can only win when enough drafts are accepted to offset the
verify forward (which costs ~`K + 1`× a plain decode). Single RTX 4090, Qwen3-4B,
`temperature=0`, `--ignore-eos`, `--no-prefix-cache`, 12 prompts × 128 output
tokens, sonnet (550/128, prefix 200) and random (550/128). `Δ` is n-gram vs plain
decode on the **same** engine; vLLM (0.21 / 0.23, ngram `K=4`,
`prompt_lookup_max=3`) is the reference.

| dataset / conc | acceptance | vLLM 0.23 Δ | openinfer (no gate) Δ | **openinfer (gate) Δ** |
| --- | --- | --- | --- | --- |
| sonnet c1 | ~5% | −5.8% | −26.2% | **+0.6%** |
| sonnet c4 | ~7% | −5.8% | −40.0% | **−2.2%** |
| random c1 | ~33% | +19.3% | −10.3% | **+21.8%** |
| random c4 | ~37% | +17.0% | −33.8% | **−20.0%** |

Reading this:

- **Prose (sonnet) acceptance is ~5–7%** — prompt-lookup drafts are almost never
  the model's greedy continuation, so *every* engine loses here (vLLM included).
  Without the gate the loss is severe (−40% at c4) because each step pays the
  verify forward (a `K+1`-wide prefill-kernel pass) plus, when only some requests
  matched, a second plain-decode pass — both wasted. The gate detects the low
  acceptance and falls back to plain decode, recovering the regression to ≈0 and
  **beating vLLM**, which keeps speculating and eats its −6%.
- **random c1 matches vLLM** (+21.8% vs +19.3%): low concurrency leaves the GPU
  compute-idle, so the gate stays open and the ~33% acceptance is a free win.
- **random c4 is the open gap**: vLLM **+17%** vs openinfer **−20%** at the *same*
  ~37% acceptance (proven by the matching c1 numbers). This is purely the verify
  path — see below. The gate can't help: acceptance is high, so it correctly
  stays open; the loss is in *how* we verify, not *whether* we should.

**Root cause of the random-c4 gap (nsys, `--cuda-graph-trace=node`).** At c4 the
GPU is already ~92% busy on plain decode. Under n-gram the same 22 s window shows
attention running through the **prefill** kernel (`BatchPrefill`, ~23 µs) instead
of the decode kernel (`BatchDecode`, ~15 µs), ~989 verify forwards **plus** ~887
decode forwards (the batch splits two ways every mixed step), and ~43% more GEMM.
vLLM instead runs **one fused forward**: draft tokens are extra rows in the same
batch (a missed request is just `query_len=1`), verified by one rejection step —
no second pass, no prefill-kernel penalty, no ragged-span CUDA-graph thrash.
Folding our verify into that single unified forward is the way to turn random-c4
positive; tracked as the follow-up below, not in this PR.

## The acceptance gate (`NgramGate`)

`ngram.rs`, owned by `NgramRuntime`. An engine-wide EWMA of accepted draft tokens
per drafted step. `should_draft` opens while in warmup, while the EWMA clears
`accept_threshold`, or on a periodic probe; otherwise `propose_step` produces no
drafts for the step and the existing draft/undraft partition in `plan.rs` routes
the whole batch to plain decode (no verify forward). The probe (every
`PROBE_INTERVAL` steps) re-opens it so a shift into repetitive text is picked back
up. `record_verified` (called from the verify transaction in `spec.rs`) feeds the
step's mean acceptance back in. Lossless either way. Unit-tested (`gate_*` in
`ngram.rs`); the engine-level losslessness gate still passes with the gate live.

## Remaining work / follow-up

1. **Single fused verify forward** (the random-c4 fix, separate PR): fold the
   verify tokens into one unified batched forward (FlashInfer varlen) with a GPU
   rejection step — a missed request becomes a `query_len=1` row in the same
   batch — instead of the current `batch_prefill_into`-based verify plus the
   second plain-decode pass for undrafted requests. The *Serving A/B* above
   quantifies the prize: closing the ~37-point random-c4 gap to vLLM
   (+17% vs −20%).

## Scope / deferred

Greedy, single-GPU, no LoRA. Deferred: sampling (non-greedy) acceptance (needs
draft distributions + a rejection sampler, not argmax), the unified fused verify
forward above, and a general `SpeculativeProposer` trait (revisit when EAGLE
#325 lands a third method).

## Prior art

- vLLM V1 n-gram / prompt-lookup spec decode (`NgramProposer` in the runner, GPU
  rejection sampler, scheduler reserves KV for `k` draft tokens). Its single
  fused forward is what the follow-up above borrows.
- kvbm `SchedulableSequence` speculative lifecycle (`schedule_speculative` /
  `apply_speculative`, LIFO block release), reused unchanged here.
