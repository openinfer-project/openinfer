# DeepSeek-V2-Lite EP2 HF Accuracy Gate

> **TL;DR:** HF comparison gate for issue #135 after PR #149 and PR #150. The remaining correctness question was not NCCL performance; it was whether the existing DeepSeek-V2-Lite EP=2 baseline matches Hugging Face `generate(use_cache=true)` greedy decode for `prompt="Hello"`, `batch=1`, `output_len=16`.
>
> **Status:** Passing for the covered shape. The latest run is token-exact and text-exact across HF `generate(use_cache=true)` greedy, pegainfer host-staged EP2, and pegainfer NCCL EP2.

## Scope

In scope:

- HF truth: `AutoTokenizer` and `AutoModelForCausalLM` with `trust_remote_code=True`, `torch_dtype=torch.bfloat16`, `model.eval()`, and `torch.no_grad()`.
- Generation shape: batch `1`, prompt `Hello`, prompt token ids `[17464]`, output length `16`, greedy argmax only.
- Pegainfer paths: default host-staged EP2 backend and explicit `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl`.
- Result comparison: generated token ids, generated text, token sha256, text sha256, and first different generated-token index.

Out of scope:

- Performance claims.
- Sparse dispatch or production EP backend work.
- Generic EP topology, multi-node support, larger prompts, or batch > 1.
- Any NCCL runtime-path change when host-staged and NCCL still match each other.

## Issue #135 Coverage Map

| Issue / maintainer requirement | Covered by | Evidence |
| --- | --- | --- |
| DeepSeek-V2-Lite config loads independently from DeepSeek V4 assumptions. | PR #149 | Dedicated `pegainfer-deepseek-v2-lite` config/weight/model crate. |
| Single-node `ep_size=2` validates rank, expert ownership, and local expert count. | PR #149 | EP layout is fixed to rank 0 experts `0..31` and rank 1 experts `32..63`, with load-time validation. |
| Each rank only loads its owned 32 routed experts. | PR #149 | Driver rank loads rank 0 experts; expert rank loads only rank 1 routed experts. |
| Unsupported backend/topology reports explicit errors. | PR #149 / #150 | Unsupported device count, duplicate devices, cuda_graph, and backend names fail closed. |
| Minimal dispatch/combine path exists for the first correctness gate. | PR #149 | Host-staged dispatch/combine path remains the default baseline. |
| Maintainer-requested naive NCCL backend exists before pegainfer-comm/NVLink work. | PR #150 | `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl` path passes the same EP2 greedy E2E as host-staged. |
| HF ground-truth accuracy comparison exists. | This gate | HF `generate(use_cache=true)` greedy, host-staged EP2, and NCCL EP2 are token/text exact for the covered shape. |

Together with PR #149 and PR #150, this gate covers issue #135's correctness-first acceptance surface for the narrow EP=2 milestone. Follow-up work should be tracked separately for sparse/GPU dispatch, pegainfer-comm/NVLink integration, performance evidence, long context, and broader prompts/batches.

## Commands

Run all three outputs from the same model snapshot:

```bash
mkdir -p target/accuracy/dsv2-lite-ep2

python tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py \
  --model-path models/DeepSeek-V2-Lite \
  --prompt Hello \
  --output-len 16 \
  --out target/accuracy/dsv2-lite-ep2/hf.json

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/host-staged.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/nccl.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

python tools/accuracy/compare_dsv2_lite_ep2_outputs.py \
  --hf target/accuracy/dsv2-lite-ep2/hf.json \
  --host-staged target/accuracy/dsv2-lite-ep2/host-staged.json \
  --nccl target/accuracy/dsv2-lite-ep2/nccl.json \
  --out target/accuracy/dsv2-lite-ep2/comparison.json \
  --require-all-exact
```

Omit `--require-all-exact` only when intentionally collecting mismatch diagnostics.

On Blackwell-class GPUs, make sure the selected NCCL runtime supports the device. Older NCCL runtimes may fail communicator initialization before the model-level comparison runs.

## Interpretation

- `all_token_text_exact`: HF, host-staged, and NCCL agree on generated token ids and generated text.
- `pegainfer_baseline_accuracy_gap`: host-staged and NCCL match each other, but both differ from HF. Treat this as a pegainfer baseline accuracy problem before touching NCCL transport.
- `nccl_transport_regression`: host-staged and NCCL differ. Debug the NCCL path before drawing any HF parity conclusion.

## Latest Evidence

2026-05-27, single-node 2 GPU validation with the same `models/DeepSeek-V2-Lite` snapshot for all three outputs. The model snapshot metadata recorded commit `604d5664dddd88a0433dbae533b7fe9472482de0`. The HF truth source used `AutoModelForCausalLM.generate(..., do_sample=false, use_cache=true)` with `torch==2.7.0+cu128` and `transformers==4.40.2`:

The earlier 2026-05-26 hash is superseded for the active gate because it was produced from a different local model snapshot. Treat only the hash below as the active oracle for snapshot `604d5664dddd88a0433dbae533b7fe9472482de0`. This refresh does not claim a model-runtime improvement, a manual-loop root cause, or a transport issue.

| Source | Backend | Tokens | Token SHA256 | Text SHA256 | Text |
| --- | --- | ---: | --- | --- | --- |
| HF | `generate(use_cache=true)` | 16 | `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225` | `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347` | `, I am a 19 year old girl from the UK. I am` |
| pegainfer | host-staged | 16 | `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225` | `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347` | `, I am a 19 year old girl from the UK. I am` |
| pegainfer | NCCL | 16 | `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225` | `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347` | `, I am a 19 year old girl from the UK. I am` |

Classification: `all_token_text_exact`.

- HF vs host-staged: token-exact and text-exact; no first different token.
- HF vs NCCL: token-exact and text-exact; no first different token.
- Host-staged vs NCCL: token-exact and text-exact; this run does not show an NCCL transport regression.

Accuracy fixes covered by this gate:

- DeepSeek-V2 RoPE host path now matches HF's pair permutation and bf16 multiply/add materialization.
- YaRN inv-frequency and `mscale_all_dim` attention softmax scaling are applied in the host attention path.
- Host attention now rounds attention scores/probabilities through bf16 at the HF materialization points.
- DeepSeek-V2 RMSNorm now rounds the normalized hidden to bf16 before multiplying the bf16 norm weight, matching the HF module.
- MoE gate logits now use the HF fp32 gate projection, and selected experts are accumulated in deterministic expert-id order after top-k selection.
- MoE routed expert output is materialized before adding shared experts, matching HF's `moe_infer(...).to(bf16) + shared_experts(...)` structure.
- Fused `silu_mul` now matches the existing non-fused `silu_mul` bf16 behavior by rounding `SiLU(gate)` before multiplying by `up`.
