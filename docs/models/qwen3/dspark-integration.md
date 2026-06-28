# DSpark Integration (Qwen3-4B)

**TL;DR:** DeepSeek's **DSpark** (paper + DeepSpec repo, Jun 2026) is **our DFlash drafter plus two bolt-ons**: (1) a *semi-autoregressive* **Markov head** — a rank-256 logit bias added in a short sequential loop over the block so each draft token conditions on the previously sampled one; this is the whole draft-quality win (+16–18% accepted length over DFlash in their offline table, paper Table 1), and (2) a **confidence head** (tiny linear → per-position survival probability) feeding a **hardware-aware prefix scheduler** that trims verify length under load — the production throughput/Pareto win. The released checkpoint `dspark_qwen3_4b_block7` has an **identical backbone to our `Qwen3-4B-DFlash-b16`** (same hidden 2560 / 5 layers / `target_layer_ids=[1,9,17,25,33]` / KV-injection) — it differs only by `block_size=7`, +4 tensors (`markov_w1 [V,256]`, `markov_w2 [V,256]`, `confidence_head.proj.weight [1,2816]`, `.bias [1]`), and an "anchor-is-the-first-prediction" block-layout tweak. **Our verify/accept/KV-transaction stack is reused unchanged** — DSpark only changes the *propose* step. Recommended phasing: **Phase 1 = Markov head** (lossless-by-construction in greedy, reuses the losslessness gate, ~all the algorithmic win, small change to one function); **Phase 2 = confidence head + static-threshold draft truncation** (small); **Phase 3 = hardware-aware scheduler** (large; fights our bucketed CUDA-graph verify — needs its own design). Decisions settled: **block_size = 7**, **reuse the target head** (DSpark's `embed_tokens`/`lm_head` are byte-identical to target Qwen3-4B's — skip loading them). **Phase 1 delta over DFlash is small: config + 2 tensor loads + one sequential Markov loop in `execute_dflash_draft`; no new verify/KV/graph.**

Last touched: 2026-06

## Implementation status — Phase 1 built & lossless

**Phase 1 (Markov head) is implemented and passes the losslessness gate.** Greedy DSpark output is byte-for-byte identical to plain greedy on the 5070 Ti dev box:

```
OPENINFER_TEST_MODEL_PATH=/data/models/Qwen3-4B \
OPENINFER_DFLASH_TEST_MODEL_PATH=/data/models/dspark_qwen3_4b_block7 \
cargo test --release -p openinfer-qwen3-4b --test dflash_speculative_gate
# → 4 passed; all prompts 100% lossless. (The "kernel-gap flip … Not a spec bug"
#   diagnostics are the pre-existing DFlash prefill/decode numeric gap, unrelated.)
```

What was actually built (the new DSpark code lives in its own module `openinfer-qwen3-4b/src/dspark.rs`):

- **`dspark.rs`** — `MarkovHead { w1, w2 }` + `MarkovScratch` + `sample_block()` (the sequential semi-autoregressive loop, batched across requests) + `reservation_bytes()`. This is the home of all DSpark-specific logic; `dflash.rs` only holds an `Option<MarkovHead>` and an `Option<MarkovScratch>` and exposes `uses_markov_head()` / `markov_draft_tokens()` / `verify_span()`.
- **Custom CUDA kernel** `markov_step_argmax` (`openinfer-kernels/csrc/shared/argmax.cu` + FFI + `ops::markov_step_argmax_into`). **We chose to add one kernel** — the earlier "no new kernel needed" prediction below was revised: the base block logits are request-major `[N·block, V]`, so step `k` needs a *strided* argmax over row `i·block+k` plus a *per-request* bias row `i`. Composing that from existing ops would mean slicing/re-batching one column per step (messier and slower than a single strided argmax-with-bias kernel). The kernel reads `base[(i·block+k)·V+v] + bias[i·V+v]` and writes the chosen token id as `u32` so it feeds straight back as the next step's prev-token lookup. Two-stage (partial+finalize), reusing the existing batched-argmax tiling.
- **Per-step math** = `embedding_batch(markov_w1, prev) → gemm(markov_w2) → markov_step_argmax`, looped `block_size` times; matches `VanillaMarkov.sample_block_tokens` exactly.
- **Dual-schema config** — `DFlashConfig::from_file` now resolves both our nested `b16` schema (`dflash_config:{…}`, flat `rope_theta`) and DeepSpec's flat schema (`mask_token_id`/`target_layer_ids` flat, `rope_theta` under `rope_parameters`). `markov_rank == 0` ⇒ plain DFlash. The struct was flattened (no more `dflash_config` nesting).
- **Anchor-first span** — DSpark proposes all `block_size` drafts (span `block_size+1`); DFlash drops position 0 (span `block_size`). `verify_span()` sizes the verify CUDA-graph buffers accordingly. The confidence head is parsed but **not loaded/used** in Phase 1 (logged at load).

**Next:** perf A/B vs DFlash on the company 5090 (`vllm bench serve`, c1/c4/c8 × datasets) — see [Benchmark plan](#benchmark-plan--5090) at the bottom. The 5070 Ti is dev/compile/correctness only; perf runs on the 5090.

### Released checkpoint tensor layout (`dspark_qwen3_4b_block7`, single `model.safetensors`)

5 backbone layers (`layers.0..4`), all BF16. Top-level (non-layer) tensors:

| tensor | shape | use |
| --- | --- | --- |
| `markov_head.markov_w1.weight` | `[151936, 256]` | prev-token embedding lookup (`r=256`) — **loaded** |
| `markov_head.markov_w2.weight` | `[151936, 256]` | `Linear(256→V)` bias projection — **loaded** |
| `fc.weight` | `[2560, 12800]` | KV-injection context projection (`12800 = 2560×5` target layers) |
| `hidden_norm.weight` / `norm.weight` | `[2560]` | context RMSNorm / final RMSNorm |
| `confidence_head.proj.weight` / `.bias` | `[1, 2816]` / `[1]` | Phase 2 only — **skipped** |
| `embed_tokens.weight` / `lm_head.weight` | `[151936, 2560]` | byte-identical to target — **skipped**, reuse target head |

Per layer: `self_attn.{q,k,v,o}_proj` (`q 4096`, `kv 1024`, head_dim 128, 32q/8kv), `self_attn.{q,k}_norm [128]`, `mlp.{gate,up,down}_proj` (inter 9728), `input_layernorm` / `post_attention_layernorm [2560]`. Config: `block_size=7`, `markov_rank=256`, `markov_head_type="vanilla"`, `enable_confidence_head=true`, `target_layer_ids=[1,9,17,25,33]`, `mask_token_id=151669`, `rope_theta=1e6` (under `rope_parameters`).

## Sources (cloned / downloaded, local)

- Paper PDF: `../DeepSpec/DSpark_paper.pdf` — *"DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"* (Cheng et al., DeepSeek-AI / PKU, 2026). Converted to text with `uv run --with pymupdf` and read in full.
- Training/eval repo: `../DeepSpec` (`github.com/deepseek-ai/DeepSpec`). Reference modeling for the draft side:
  - `deepspec/modeling/dspark/qwen3/modeling.py` — backbone + heads (the canonical inference math).
  - `deepspec/modeling/dspark/markov_head.py` — `VanillaMarkov` / `GatedMarkovHead` / `RNNHead`.
  - `deepspec/eval/dspark/draft_ops.py` — the inference draft loop (`forward_dspark_draft_block` → `build_dspark_proposal`), the file to mirror.
  - `config/dspark/dspark_qwen3_4b.py` — the released checkpoint's training config.
- Released draft weights: `/data/models/dspark_qwen3_4b_block7` (downloaded). Target stays `/data/models/Qwen3-4B`.

DeepSpec also re-trains **DFlash** and **Eagle3** in the same framework; their DFlash config matches our checkpoint's geometry, which is why the backbones line up.

## What DSpark actually contributes (paper)

Per-token spec-decode latency is `L = (T_draft + T_verify) / τ` (τ = accepted tokens/round). DSpark attacks two of the three levers:

### 1. Semi-autoregressive generation → raises τ (draft quality)

Parallel drafters (DFlash) emit all block logits in one pass, so every position **marginalizes over all possible predecessors** instead of conditioning on the one actually sampled → "multi-modal collision" (e.g. "of problem" / "no course") → acceptance decays along the block. Their position-wise analysis (Fig 2): DFlash's *conditional* acceptance falls from ~0.87→0.78 (code) and 0.72→0.63 (chat) across the block; an autoregressive drafter stays flat/rises but starts lower (shallow net). DSpark wants both: the deep **parallel backbone** for the high-leverage first token, plus a **lightweight sequential head** for suffix coherence.

The sequential head adds a prefix-dependent **bias** to the parallel base logits and factorizes the block autoregressively (paper Eq. 4):

```
p_k(v | x0, x_<k) ∝ exp( U_k(v) + B_k(x0, x_<k, v) )
```

`U_k` = parallel backbone base logit at position k; `B_k` = the sequential bias. Two instantiations; **DSpark default = Markov head** (RNN head gives only marginal extra gain at longer blocks and is harder to deploy — paper §4.3.2):

- **Markov head** (first-order, low-rank): `B(x_{k-1}, ·) = W1[x_{k-1}] · W2`, `W1 ∈ R^{V×r}` (embedding lookup), `W2 ∈ R^{r×V}` (logit projection), `r = 256`. Cheap: per step it's a gather + one rank-256 GEMV over the vocab. Once position 1 samples "of", the head boosts "course" / suppresses "problem" at position 2.

The backbone change vs DFlash is **minor** (paper §3.1): instead of "anchor + γ masks, predict the γ mask positions", DSpark treats the **anchor itself as the first prediction position** → γ inputs (anchor + γ−1 masks) yield γ draft logits. Slightly less compute, same quality.

**Result (paper Table 1, offline, scheduler OFF, τ = accepted length incl. bonus):** on Qwen3-4B, DSpark beats DFlash by **+16.3% macro-avg** (e.g. GSM8K 5.40→6.11, MBPP 4.40→5.13, MT-Bench 3.07→3.64) and beats Eagle3 by +30.9%. A **2-layer** DSpark already beats a **5-layer** DFlash (Fig 3 — the sequential head is very parameter-efficient). Latency overhead of the sequential loop at batch 128 is **+0.2–1.3%** per round because target verify dominates (Fig 4).

### 2. Confidence-scheduled verification → cuts effective T_verify (systems)

Verifying the *whole* block wastes target batch capacity on tokens that will be rejected — the cost depends on domain (code accepts more than chat) **and** on live engine load. Two pieces:

- **Confidence head:** `c_k = σ( w·[h_k ; W1[x_{k-1}]] )` ∈ (0,1) = P(draft k survives | all earlier accepted). Trained against the analytic per-step acceptance `c*_k = 1 − ½‖p_draft − p_target‖₁` (TV distance). Post-hoc **Sequential Temperature Scaling (STS)** calibrates the cumulative product (raw head is overconfident, ECE 3–8% → ~1%).
- **Hardware-aware prefix scheduler** (paper Alg. 1): per request the prefix-survival prob is `a_{r,j} = Π_{i≤j} c_{r,i}`; globally sort all `(r,j)` by `a_{r,j}`, greedily admit tokens, and pick the verify lengths that maximize `Θ = τ · SPS(B)` where `SPS(B)` is a **once-profiled** "steps/sec vs batch-size" cost table. Early-stop on Θ drop keeps it lossless (non-anticipating). Production adaptations (§5.2): async/ZOS-compatible (decide truncation length from confidence two steps prior → "dynamic top-K"), variable-length verify kernels.

**Result (paper §5.4, live DeepSeek-V4 traffic vs MTP-1):** +60–85% tok/s/user (Flash) / +57–78% (Pro) at matched throughput, and it preserves serving capacity under strict-SLA regimes where the baseline collapses. Verify budget auto-expands to ~4–6 tokens under light load and shrinks under heavy load (Fig 8).

**The split that matters for us:** the **draft-quality win (1) is local and self-contained**; the **systems win (2) is a scheduler/engine change**. They are independently shippable, and (1) is most of the per-request speedup at the low/medium concurrency our single-card serving lives in.

## How DSpark maps onto our DFlash

Our DFlash drafter (`openinfer-qwen3-4b/src/dflash.rs`, lane in `executor/dflash_lane.rs`) already **is** the DSpark parallel backbone. Side-by-side of the released `dspark_qwen3_4b_block7` config vs our `Qwen3-4B-DFlash-b16`:

| | DFlash-b16 (ours) | DSpark-block7 (released) |
| --- | --- | --- |
| hidden / layers / heads | 2560 / 5 / 32q-8kv | **identical** |
| `target_layer_ids` | `[1,9,17,25,33]` | **identical** |
| KV-injection context (`fc`) | `[2560, 12800]` (5×2560) | **identical** |
| `mask_token_id` | 151669 | 151669 |
| `block_size` | 16 | **7** |
| extra tensors | — | `markov_w1 [151936,256]`, `markov_w2 [151936,256]`, `confidence_head.proj.weight [1,2816]`, `.bias [1]` |
| block→draft mapping | predict block, **drop position 0**, use `block[1..]` (K=block−1) | **anchor is position 0's prediction**, use all (K=block) |

So the backbone weight loader, the `fc`+`hidden_norm` context projection, the per-layer KV-injection attention, the verify forward, `accept_greedy`, and the KV transaction are **all reusable as-is**. The draft↔verify boundary is already a pure token span (see `dflash-speculative-decoding.md` — "propose differs, verify is shared"), which is exactly the seam DSpark slots into.

### The one function that changes (Phase 1)

`execute_dflash_draft` (`executor/dflash_lane.rs:212`) today does: `draft_logits_batched` → `select_step_tokens` (independent greedy argmax per row) → split into per-request blocks. DSpark replaces the **argmax step** with the sequential Markov loop. Mirroring `markov_head.sample_block_tokens` + `build_dspark_proposal`:

```
# base_logits[N, block, V] from draft_logits_batched (unchanged)
for each request block (anchor = current_token):
    prev = anchor
    for k in 0..block_size:
        bias  = markov_w2 @ markov_w1[prev]     # gather [256] then rank-256 GEMV → [V]
        tok_k = argmax(base_logits[k] + bias)   # greedy; our engine is greedy spec
        prev  = tok_k
    drafts = [tok_0 .. tok_{block-1}]           # anchor-first: keep ALL positions
```

`base_logits` come from the **untouched** batched backbone forward. The loop is `block_size` sequential steps, each a `[N,256]·[256,V]` GEMV + per-row argmax — small compute, but it serializes drafting (see Risks). Output feeds the existing verify span builder `[current_token, draft_1, …]` and everything downstream is unchanged.

**Losslessness is free in greedy:** the Markov head only changes *which* tokens we propose; verify still checks each draft against the target's own argmax, so `accept_greedy` keeps losslessness regardless of how drafts were produced. A better proposal can only raise accept length, never corrupt output. Our existing `dflash_speculative_gate.rs` covers it unchanged.

## Integration plan (phased)

### Phase 1 — Markov head (the draft-quality win) — recommended first

1. **Config:** extend `DFlashConfig` (`config.rs:42`) with `markov_rank: usize`, `markov_head_type: String`, and the `enable_confidence_head` / `confidence_head_with_markov` flags (defaulting off so existing DFlash configs still parse). A `markov_rank == 0` config = today's DFlash, so this is a superset, not a fork.
2. **Weights:** load `markov_head.markov_w1.weight` ([V,256] embedding) and `markov_head.markov_w2.weight` ([V,256], the `r→V` projection — `bias = w1[prev] @ w2ᵀ`) in `dflash/loading.rs`.
3. **Propose:** add the sequential loop above. Cleanest kernel shape: a `markov_bias` op (gather rows of `markov_w1` for the current `prev` tokens of all N requests → `[N,256]`, GEMV against `markov_w2` → `[N,V]`, add into the k-th logit slice), then reuse our batched argmax for that one column. Repeat `block_size` times. The base-logit GEMM stays a single batched pass; only the bias+argmax is sequential.
4. **Block layout:** switch the draft span to anchor-first (keep all `block_size` outputs) to match how the checkpoint was trained. Keep the DFlash path (`block[1..]`) selectable by config so we don't regress the b16 drafter.
5. **Gate:** run `dflash_speculative_gate.rs` (losslessness) + measure accept length and single-stream/concurrent A/B vs the DFlash-b16 baseline on this box. The bar is the paper's +16% accept on Qwen3-4B (offline), translated to our greedy/serving setting — **measure before claiming the win** (CLAUDE.md Performance Work rule).

**Decided: `block_size = 7`** — the checkpoint's trained width (matches the paper). Smaller blocks than DFlash-b16 but higher per-position accept; A/B compares block7-DSpark vs block16-DFlash on accept length **and** tok/s, not accept alone.

### Phase 2 — confidence head + static-threshold draft truncation (optional, small)

Load `confidence_head.proj.{weight,bias}` (Linear `2816→1`, `2816 = 2560 + 256`). After sampling, compute `c_k = σ(proj([h_k ; markov_w1[prev_k]]))` and truncate the proposed block at the first `c_k < threshold` (mirrors `_confident_prefix_length` in `draft_ops.py`). This shortens the verify span on low-confidence suffixes — a cheap, per-request, **single-card-friendly** slice of the systems win, with no scheduler. Lossless (truncating drafts only shortens the span; verify still corrects).

### Phase 3 — hardware-aware prefix scheduler (the systems/throughput win) — larger, design-first

The full Alg. 1 scheduler needs: an `SPS(B)` profiling table at engine init, batch-global survival-prob sorting each step, and **variable verify length per request**. Our verify path is **greedy, single-path, and CUDA-graph bucketed** (`verify_graph.rs`, captured per batch×span bucket — see `dflash-speculative-decoding.md`), so per-request variable span **fights the captured-shape invariant** (`total_tokens == batch_size × span`). This is a real engine change (likely: dynamic top-K truncation to a few discrete span buckets + an eager fallback), justified only at the high concurrency where verify waste bites. Defer until Phase 1/2 land and we have a concurrency regime that needs it. Note this is also where the EAGLE **proposer trait** discussion reconnects: DSpark is a second concrete proposer, so it's the forcing function for the trait that DFlash alone didn't justify.

## Risks / open questions

- **Sequential-loop latency at small batch.** The paper's +0.2–1.3% overhead is at batch 128 where target verify dominates. At our single-stream regime (where DFlash gives 1.56–1.82×) the `block_size` sequential bias+argmax launches are more exposed. The bias is a full-vocab (V=151936) materialization per step × block_size. Mitigations to evaluate: fuse gather+GEMV+add+argmax into one kernel per step; or only materialize top-k of the base logit before adding bias. **Measure draft-step time before/after**, like the DFlash batching A/B did.
- **Greedy vs sampling.** Our engine is greedy-lossless spec; the paper evaluates at temperature 1.0 with rejection sampling. Markov bias works identically under greedy (argmax of base+bias). The confidence head's TV-distance training target assumes sampling, but as a *learned acceptance ranker* it still works for Phase 2 truncation; STS calibration only matters for Phase 3's throughput math.
- **`tie_word_embeddings` — RESOLVED, reuse target.** Byte-compared (sha256): DSpark's `embed_tokens.weight` *and* `lm_head.weight` are **both identical** to target Qwen3-4B's `model.embed_tokens.weight` (`eabe5625…`; target is tied, no separate `lm_head`). They're just the frozen target head serialized into the checkpoint. So we do exactly what DFlash does today — reuse the target's head via `compute_logits_with_target_head_into`, **skip loading DSpark's `embed_tokens`/`lm_head`** (saves ~1.5 GiB). Zero implementation cost: it's the existing path.
- **RNN head — skip.** Marginal gain, harder to deploy (paper §4.3.2). Markov only.
- **CUDA-graph draft.** The draft-side piecewise graph (tracked in the DFlash doc) now has to also cover the sequential Markov steps; the eager-attention boundary story is unchanged but the bias loop is new capture surface.

## Benchmark plan — 5090

Phase 1 is built and lossless (see top). The deliverable A/B runs on the company **5090** (`ssh 5090`, models under `/data`), the 5070 Ti is dev/compile/correctness only.

- **Contenders (same DeepSpec block7 recipe, isolates the Markov head):** DSpark `/data/dspark_qwen3_4b_block7` vs DFlash `/data/dflash_qwen3_4b_block7`, both against target `/data/Qwen3-4B`. (Optionally 8B: `/data/dspark_qwen3_8b_block7` vs `/data/dflash_qwen3_8b_block7` vs `/data/Qwen3-8B`.)
- **Load:** `vllm bench serve` with `--ignore-eos` (openinfer rejects `min_tokens`), across datasets × concurrency **c1 / c4 / c8**.
- **Metrics:** output tok/s, TTFT/TPOT, and cumulative accept rate / accepted length per round (engine logs `cumulative_accept_rate`). The paper's bar is +16% accepted length on Qwen3-4B (offline); we report the greedy/serving translation. **Measure before claiming the win** (CLAUDE.md).
- Results land in a "Results — 5090" section here once collected.

## Integration delta — what's new on top of DFlash (Phase 1)

We already support DFlash, so almost everything is reused. The *only* new surface:

| Area | File | Change | Effort |
| --- | --- | --- | --- |
| **Config schema** | `config.rs` | DSpark `config.json` is **flat** (`mask_token_id` / `target_layer_ids` / `block_size` / `markov_rank` / `markov_head_type` / `enable_confidence_head` at top level), unlike our DFlash config's nested `dflash_config: {…}`. Add the flat fields (or a small adapter); `markov_rank` default 0 = legacy DFlash, so it's a superset not a fork. | trivial |
| **Weight load** | `dflash/loading.rs` | Load 2 tensors: `markov_head.markov_w1.weight [V,256]`, `markov_head.markov_w2.weight [V,256]` (~156 MiB total). **Do not** load `embed_tokens`/`lm_head` (reuse target — proven identical). | trivial |
| **Propose loop** | `dspark.rs` (`MarkovHead::sample_block`), called from `executor/dflash_lane.rs` (`execute_dflash_draft`) | `block_size`-step Markov loop, batched across requests: per step `embedding_batch(markov_w1, prev)→[N,256]`, `gemm(markov_w2)→[N,V]` bias, then `markov_step_argmax` over the step-`k` base-logit rows + bias → next prev. | the only real work |
| **Custom kernel** | `openinfer-kernels/csrc/shared/argmax.cu` | `markov_step_argmax` — strided argmax over `base[(i·block+k)·V+v] + bias[i·V+v]`, writes `u32` token. + FFI decl + `ops::markov_step_argmax_into`. | small |
| **Block layout** | `executor/dflash_lane.rs` + `dflash.rs::verify_span()` | Anchor-first: keep all `block_size` sampled tokens as drafts (DFlash drops position 0 / uses `block[1..]`), span `block_size+1`. Gated by `uses_markov_head()` so DFlash-b16 is untouched. | small |
| **Memory reservation** | `dflash/reservation.rs` | Add `MarkovHead::reservation_bytes` (2 markov tensors ~156 MiB + sample scratch) to the fixed term. | trivial |

Reused **unchanged**: backbone forward (`draft_logits_batched`), `fc`/`hidden_norm` context projection, KV-injection attention, verify forward + `verify_graph.rs`, `accept_greedy` / `build_verify_results`, KV transaction, prefill capture, the losslessness gate and perf harness.

**One new CUDA kernel was added** (revising the original prediction — see Implementation status at top). The Markov bias is `gather + GEMM(M=N,K=256,N=V) + add + argmax`; the gather and GEMM reuse existing ops (`embedding_batch` on `markov_w1`, `gemm_into`), but the final **strided argmax-with-bias** over request-major `[N·block,V]` base logits warranted one purpose-built kernel rather than per-step column slicing.

**Difficulty: low, as predicted.** Small PR — one sequential loop + one focused kernel + config/loader plumbing, no verify/KV/graph changes. The thing still to *watch* (Risks §) is sequential-loop latency at small batch: the per-step `[N,V]` bias materialization × `block_size` is the exposed cost; measure draft-step time on the 5090 and fuse gather+gemm+argmax only if it shows.
