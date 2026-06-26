# GLM5.2 Support

> **TL;DR:** GLM5.2 support is in bring-up. The `openinfer-glm52` crate and server/bench feature surface now exist: `glm_moe_dsa` config probing, stop-token loading, safetensor manifest parsing, all-rank EP8 load-plan validation, rank0 dtype/shape header validation, rank-worker raw H2D weight loading, a non-expert weight contract covering attention/dense/router/shared tensors plus `666` non-expert FP8 projections per rank (`432` attention/indexer, `9` dense, `225` shared-expert), streaming FP8 routed-expert packaging into per-layer expert-major per-expert `[gate; up]` `W13`/`W2` buffers with raw expert tensor cleanup, a GLM-specific DeepEP shape contract, persistent decode-bucket arena allocation with full-layer scratch (norm/FP8-linear input, attention/DSA indexer, dense/shared, logits) plus separate DeepEP recv, W13, W2 input, W2 output, and combine buffers, and a fail-closed `DP8 TP1 EP8`/DeepEP `EngineHandle` with 1 coordinator + 8 rank workers. `openinfer-kernels` also has a single GLM `ops::glm52`/`ffi::glm52` module owning router, DeepEP decode, DeepGEMM grouped-FP8 ABI contract plus a fail-closed launch boundary with Rust-side buffer checks, graph-captured DeepEP-`psum_expert` -> grouped-GEMM metadata generation with DeepGEMM `BLOCK_M` psum-compatibility audit fields, MoE activation-quant substrates including a DeepGEMM TileLang-source weighted W2-input SwiGLU quant path, a DeepGEMM F32 scale-layout substrate, FlashInfer/TRTLLM `GroupedWithOffset` activation-scale relayout, a FlashInfer/TRTLLM grouped-FP8 raw-runner ABI/workspace gate, and a FlashInfer/TRTLLM plain FP8 blockscale linear ABI for GLM non-expert projection shapes. The router copies Kimi's complete noaux_tc semantics for GLM's BF16 `[256,6144]` gate, the DeepEP ABI bakes GLM dimensions (`hidden=6144`, `experts=256`, `local=32`) and aligns expanded expert segments to 64 rows with `10240` worst-case rows, the MoE quant kernels copy/adapt vLLM CUDA/C++ semantics for BF16 -> FP8 per-token/per-128 W13 input and SiLU*up -> FP8 per-token/per-128 W2 input, the weighted W2-input variant mirrors DeepGEMM TileLang `swiglu_apply_weight_to_fp8`, the TRTLLM grouped-offset relayout scatters F32 activation scales into the `11232`-row 32-aligned offset space, and the TRTLLM wrappers link the vendored `CutlassFp8BlockScaleGemmRunner<fp8,fp8,bf16>` for fixed GLM W13/W2 plus plain q/dense/shared projection shapes. Startup now runs an 8-rank batched decode validation sequence: synthetic all-EP route dispatch/combine, real-router route smoke using checkpoint router weights plus typed MoE psum-layout report, actual DeepEP recv-row W13 quant and grouped-GEMM metadata validation, non-expert FP8 projection contract validation, real checkpoint plain FP8 projection smokes for q_a/q_b/kv_a/kv_b/o_proj/indexer_wk/indexer_wq_b (`rows=128`, `workspace=0`, valid activation scales, nonzero outputs on all ranks), MoE quant smoke validating F32 scales/nonzero FP8 output/DeepGEMM scale-layout conversion/TRTLLM grouped-offset scale relayout, all-layer MoE GEMM contract validation over sorted routed layers `3..=77`, real TRTLLM W13/W2 grouped-FP8 decode smoke, and a fixed 128-token decode-substrate CUDA Graph capture+replay smoke over router + DeepEP dispatch + grouped-GEMM metadata + W13 quant + TRTLLM W13 grouped FP8 + weighted W2 quant + TRTLLM W2 grouped FP8 + DeepEP combine. That MoE decode-substrate chain is now a shared call surface for eager startup smoke and graph capture, not duplicated smoke-only code. The GLM test surface keeps only the node38 checkpoint integration test; synthetic probe fixtures were deleted after the real checkpoint IT covered config, stop tokens, manifest, rank loading, and startup smokes. The first runnable GLM engine is explicitly decode-only for future P/D split: decode requests are assumed to arrive with prefilled KV/page state, and the GLM engine must not compute prompt prefill in this phase; the Rust DeepEP safe wrapper exposes only decode dispatch/combine. Forward execution is still pending: scheduler-owned batched decode from a real prefilled handoff, typed norm/residual/eager-forward projection sequencing, attention/KV decode, dense/shared/residual integration, logits/sampling, and full-forward CUDA Graph integration. Prefill, unified mixed batches, MTP, per-token prefill loops, and per-request bs=1 decode loops are out of scope/prohibited. Kernel coverage is tracked below; search FlashInfer, DeepGEMM, FlashMLA, DeepEP, and sibling `../vllm` before writing kernels. Handwritten CUDA is last resort, mainly when vLLM only has Triton, and needs local NCU evidence.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routed this work to a new `models/glm52` document and identified Kimi/Qwen/kernel/correctness docs.
  - `docs/playbooks/model-optimization-pipeline.md` - doc structure should track E2E dashboard, architecture/operator coverage, and append-only optimization attempts.
  - `docs/models/kimi-k2/dp-design.md` - `TP1 DP8 EP8` means each DP rank is an independent engine; MoE EP all-to-all is the natural synchronization point.
  - `docs/models/kimi-k2/deepep-migration.md` - Kimi's current MoE backend is DeepEP with AOT shim and host-quiet decode; this is the communication template for GLM5.2.
  - `docs/models/kimi-k2/source-layout.md` - Kimi crate split gives the source-layout target for a new model crate.
  - `docs/models/qwen3/model-crate.md` - model crates expose `launch/start_engine` and keep runtime internals out of the server.
  - `docs/subsystems/frontend/simulated-inference-engine.md` - frontend/server boundary is the generic `EngineHandle` contract.
  - `docs/subsystems/correctness/logits-golden-gate.md` - correctness gate should be logits-based and tolerance-calibrated, not exact text/hash.
  - `openinfer-kimi-k2/src/runner/bringup.rs` - Kimi has explicit `(8,1)` NCCL and `(1,8)` DeepEP branches; GLM5.2 should start with only `(1,8)`.
  - `openinfer-kimi-k2/src/runner/scheduler/dp.rs` and `openinfer-kimi-k2/src/runner/executor/tp1_dp8.rs` - DP coordinator, per-rank executors, and request lifecycle are the scheduler template.
  - `openinfer-kernels/csrc/deepep/deepep_shim.cu` and `openinfer-kernels/src/ops/deepep.rs` - current DeepEP shim bakes Kimi dimensions; GLM5.2 needs its own baked config or a parameterized shim.
  - `/data/models/GLM-5.2-0614-Provider-FP8/config.json` on jiuzhang node 38 - authoritative model dimensions and quantization metadata.
- **Relevant history**:
  - `docs/models/kimi-k2/deepep-migration.md` - adapter kernels that scale with worst-case capacity can erase the comm win; every GLM adapter kernel must be checked against actual row count, not capacity.
  - `docs/lessons/moe-bench-prompt-diversity.md` - MoE decode benches need prompt diversity; identical prompts can hide routing cost.
  - `docs/subsystems/correctness/logits-golden-gate.md` - first correctness gate should compare logits drift against an HF/vLLM reference, not free-running text.
- **Plan**:
  1. Finish model probing on node 38: summarize `config.json`, `generation_config.json`, weight key names, safetensor shapes, and installed reference runtime availability.
  2. Add `openinfer-glm52` as a feature-gated model crate with config probing, stop-token loading, weight-manifest parsing, and a loud `DP8 TP1 EP8` launch contract.
  3. Wire server and `bench_serving` model detection to GLM5.2 without changing existing defaults.
  4. Build the runtime by reusing Kimi's DP coordinator/DeepEP structure and Qwen3-4B's frontend `EngineHandle`/server path; first runnable execution is decode-only for future P/D split, so the GLM engine must reject or bypass prompt prefill rather than computing it, must not add unified mixed-batch scheduling or MTP, and must not implement decode as host-side per-request bs=1 kernel launches.
  5. Fill the kernel ledger before implementing each operator: search FlashInfer, DeepGEMM, FlashMLA, DeepEP, sibling `../vllm`, then cuBLASLt before writing code. If vLLM has reusable CUDA/C++ for a GLM5.2-shaped operator, copy/adapt that instead of writing our own. Handwritten CUDA is allowed only when vLLM's relevant path is Triton-only or otherwise not reusable, and it needs NCU evidence on jiuzhang H200.
  6. Verify progressively: local metadata/format/checks, remote release build on node 38, then single-request decode smoke from prefilled state, `bs > 1` decode smoke, decode CUDA Graph smoke, and finally logits-golden/e2e gates.
- **Risks / open questions**:
  - Kimi's DeepEP shim is Kimi-specific (`hidden=7168`, `experts=384`, `topk=8`); GLM5.2 now has a separate compile-verified shim, but runtime context bootstrap and MoE call sites still need to be wired before dispatch/combine can run.
  - GLM5.2 uses MLA/DSA-style attention plus indexer fields; FlashInfer/FlashMLA/vLLM appear to have relevant decode pieces, but exact Rust FFI coverage and KV handoff shape are not wired yet.
  - The model is FP8 with dynamic activation and block weight scales; weight layout must be proven from safetensor shapes before copying Kimi's INT4 Marlin path.

## E2E Dashboard

GPU: jiuzhang `host-172-31-13-38`, 8x H200, model `/data/models/GLM-5.2-0614-Provider-FP8`, target topology `DP8 TP1 EP8`.

| Profile | Metric | openinfer | Reference | Delta |
| --- | --- | ---: | ---: | ---: |
| prefilled-decode single stream | TPOT median | not running yet | pending | pending |
| prefilled-decode single stream | TPOT p99 | not running yet | pending | pending |
| decode-heavy (1,128) | TPOT median | not running yet | pending | pending |
| decode-heavy (1,128) | TPOT p99 | not running yet | pending | pending |
| decode-heavy bs>1 | TPOT median | not running yet | pending | pending |

## Architecture

Facts from `config.json` on node 38:

| Field | Value |
| --- | --- |
| `model_type` | `glm_moe_dsa` |
| Architecture | `GlmMoeDsaForCausalLM` |
| dtype / quantization | bf16 activations, FP8 `e4m3`, dynamic activation, block size `[128,128]` |
| Layers | 78 total; first 3 dense MLP, remaining sparse MoE |
| Hidden / vocab | `hidden_size=6144`, `vocab_size=154880` |
| Attention | `num_attention_heads=64`, `num_key_value_heads=64`, `head_dim=192` |
| MLA ranks/dims | `q_lora_rank=2048`, `kv_lora_rank=512`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256` |
| MoE | `n_routed_experts=256`, `n_shared_experts=1`, `num_experts_per_tok=8`, `moe_intermediate_size=2048` |
| Routing | `topk_method=noaux_tc`, `scoring_func=sigmoid`, `norm_topk_prob=true`, `routed_scaling_factor=2.5` |
| Context / RoPE | `max_position_embeddings=1048576`, `rope_theta=8000000`, interleaved RoPE |
| Indexer | `index_topk=2048`, `index_topk_freq=4`, `index_head_dim=128`, `index_n_heads=32` |

### Weight Layout Probe

Representative safetensor header facts from node 38:

| Tensor | dtype | Shape | Notes |
| --- | --- | ---: | --- |
| `model.embed_tokens.weight` | BF16 | `[154880, 6144]` | full vocab embedding |
| `lm_head.weight` | BF16 | `[154880, 6144]` | separate tensor despite equal shape |
| `model.norm.weight` | BF16 | `[6144]` | final RMSNorm |
| `model.layers.0.self_attn.q_a_proj.weight` | FP8 E4M3 | `[2048, 6144]` | scale `[16,48]` |
| `model.layers.0.self_attn.q_b_proj.weight` | FP8 E4M3 | `[65536, 2048]` | `64 * 1024`; scale `[512,16]` |
| `model.layers.0.self_attn.kv_a_proj_with_mqa.weight` | FP8 E4M3 | `[576, 6144]` | `kv_lora_rank 512 + rope 64`; scale `[5,48]` |
| `model.layers.0.self_attn.kv_b_proj.weight` | FP8 E4M3 | `[114688, 512]` | scale `[896,4]`; split interpretation still to map |
| `model.layers.0.self_attn.o_proj.weight` | FP8 E4M3 | `[6144, 16384]` | `64 * v_head_dim 256`; scale `[48,128]` |
| `model.layers.0.self_attn.indexer.wk.weight` | FP8 E4M3 | `[128, 6144]` | indexer key projection |
| `model.layers.0.self_attn.indexer.wq_b.weight` | FP8 E4M3 | `[4096, 2048]` | `32 * 128`; scale `[32,16]` |
| `model.layers.0.self_attn.indexer.weights_proj.weight` | BF16 | `[32, 6144]` | indexer scoring weights |
| `model.layers.0.mlp.gate_proj.weight` | FP8 E4M3 | `[12288, 6144]` | dense layer, scale `[96,48]` |
| `model.layers.0.mlp.up_proj.weight` | FP8 E4M3 | `[12288, 6144]` | dense layer, scale `[96,48]` |
| `model.layers.0.mlp.down_proj.weight` | FP8 E4M3 | `[6144, 12288]` | dense layer, scale `[48,96]` |
| `model.layers.3.mlp.gate.weight` | BF16 | `[256, 6144]` | router weight |
| `model.layers.3.mlp.gate.e_score_correction_bias` | F32 | `[256]` | noaux correction bias |
| `model.layers.3.mlp.shared_experts.gate_proj.weight` | FP8 E4M3 | `[2048, 6144]` | shared expert, scale `[16,48]` |
| `model.layers.3.mlp.shared_experts.down_proj.weight` | FP8 E4M3 | `[6144, 2048]` | shared expert, scale `[48,16]` |
| `model.layers.3.mlp.experts.0.gate_proj.weight` | FP8 E4M3 | `[2048, 6144]` | routed expert, same shape as shared expert |
| `model.layers.3.mlp.experts.0.down_proj.weight` | FP8 E4M3 | `[6144, 2048]` | routed expert, same down shape |

Count facts:

| Item | Value |
| --- | ---: |
| Total tensors in index | `118629` |
| Dense layer tensor count (`layer0`) | `27` |
| Sparse shared-indexer layer tensor count (`layer3`) | `1558` |
| Sparse full-indexer layer tensor count (`layer6`) | `1565` |
| Sparse layer routed expert tensors | `1536 = 256 experts * 3 projections * (weight + scale)` |
| Tensor names containing `mtp`/`next` | `0` |
| First-cut runtime layers | `model.layers.0..77`; `model.layers.78.*` has `1569` tensors and is treated as GLM's built-in next-token/MTP head, deferred |

Sparse layers differ by DSA indexer ownership. Runtime layers with full indexer tensors are `0,1,2,6,10,...,74` (`21` layers). Shared-indexer sparse layers do not have `self_attn.indexer.*` tensors; their indexer work must share the previous full indexer state instead of loading per-layer indexer weights.

### Required Runtime Surface

| Requirement | First-cut status |
| --- | --- |
| Prefill | out of scope for first runnable engine; P/D split will hand prefilled KV/page state to decode |
| Decode | required; engine-level forward path is decode-only in this phase |
| `bs > 1` | required as real batched execution, not a loop over bs=1 kernels |
| `DP8 TP1 EP8` | required and only supported topology initially |
| Decode CUDA Graph | required; follow Kimi DeepEP decode graph contract |
| Unified mixed prefill+decode batch | explicitly out of scope initially; reject loudly if exposed |
| MTP / next-token prediction heads | explicitly out of scope initially; GLM5.2 has `num_nextn_predict_layers=1`, but first cut serves normal causal decoding only |
| DeepEP communication | required for MoE EP |
| Frontend | copy Qwen3-4B's OpenAI-compatible `EngineHandle` path and request/error semantics |

### Reuse-First Policy

This model line should be assembled from existing proven paths wherever possible. New code needs a GLM-specific reason.

| Area | Copy/reuse source | Rule |
| --- | --- | --- |
| Frontend/server launch | Qwen3-4B | Copy model detection, `launch`/`start_engine`, `EngineHandle`, sampling parameter handling, stop-token behavior, and error semantics. |
| DP scheduler shape | Kimi-K2 TP1/DP8 | Copy coordinator/rank-worker split and lifecycle structure. |
| MoE EP communication | Kimi-K2 DeepEP | Copy DeepEP bootstrap and host-quiet decode contract; add GLM dimensions rather than inventing another comm path. |
| Sampling/logprobs | shared `openinfer-sample` + Qwen/Kimi usage | Reuse shared sampler/logprob helpers. |
| KV/cache ownership | Kimi paged MLA + Qwen/KV block patterns | Adapt only where GLM DSA/MLA layout differs. |
| Kernel selection | FlashInfer / DeepGEMM / FlashMLA / DeepEP / `../vllm` / cuBLASLt first | Do not handwrite kernels if an existing CUDA/C++ operator can be reused or adapted. When vLLM has CUDA/C++ for the shape, copy/adapt vLLM. Only write local CUDA for a vLLM Triton-only gap or a proven missing operator, and attach NCU evidence. |

### Operator Source Inventory

Search these sources before writing or adapting any GLM5.2 kernel:

| Source | Path | Current commit | GLM5.2 relevance |
| --- | --- | --- | --- |
| FlashInfer | `openinfer-kernels/third_party/flashinfer` | `d768c14` (`v0.6.12`) | Existing paged attention, MLA, GLM5 prefill shape checks, GLM router GEMM candidates. |
| DeepGEMM | `openinfer-kernels/third_party/DeepGEMM` | `54e2261` (`v2.1.1.post3-19-g54e2261`) | H200 first path is SM90 FP8 E4M3 grouped GEMM (`m_grouped_fp8_gemm_nt_contiguous`) plus F32 scale layout; MegaMoE is useful as a design reference but its public API dispatches only to arch major 10 and is not part of first runnable H200 decode. |
| FlashMLA | `openinfer-kernels/third_party/FlashMLA` | `9241ae3` (`heads/main`) | Dense/sparse MLA decode/prefill; SM90 sparse decode/prefill and dense decode are relevant to H200. |
| DeepEP | `openinfer-kernels/third_party/DeepEP` | `d4f41e4` (`v1.2.1-32-gd4f41e4`) | EP dispatch/combine substrate. Kimi and GLM now have separate baked shims; GLM's `openinfer-kernels` `glm52` shim is compile-verified but not wired into runtime forward yet. |
| vLLM sibling checkout | `../vllm` | `4d3b4b9b0` (`main`, behind origin by 7 at probe time) | Reference runtime/model/kernel semantics. Prefer copying CUDA/C++ source or launch contracts; Triton-only pieces may need local CUDA rewrite plus NCU. |

DeepGEMM/FlashMLA are source references and operator candidates, not automatic build dependencies yet. If we wire either into `openinfer-kernels`, record the exact wrapper, build requirements, benchmark, and NCU evidence here.

Detailed vLLM kernel classification now lives in `docs/models/glm52/vllm-kernel-reference.md`. Focused MoE/FP8 backend mapping now lives in `docs/models/glm52/vllm-moe-fp8-kernels.md`; use it for FlashInfer CUTLASS, DeepGEMM, and vLLM CUTLASS decisions instead of re-reading the same backend oracle. The table below stays as the bring-up summary; use the reference docs before selecting a new vLLM-derived kernel or route-weight contract.

vLLM-specific findings from the current checkout:

| Path | Finding | GLM5.2 action |
| --- | --- | --- |
| `../vllm/vllm/model_executor/models/glm4_moe.py` | GLM-4.x MoE model code, not full GLM5 DSA, but it matches GLM router semantics: sigmoid scoring, noaux correction bias, top-k renormalization, routed scale, and F32 router logits. | Use as model/load semantics reference only; do not assume it covers DSA/indexer attention. |
| `../vllm/csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu` + `dsv3_router_gemm_float_out.cu` | CUDA/C++ router GEMM already has GLM-5 instantiations for `hidden_dim=6144`, `num_experts=256`, `num_tokens=1..16`, SM90+. | Primary router source to copy/adapt for decode small batches before writing any local router kernel. |
| `../vllm/vllm/v1/attention/backends/mla/indexer.py` | DSA indexer metadata supports chunked prefill, paged decode, non-MTP decode as `(B,1)` sequence lengths, and `AttentionCGSupport.UNIFORM_BATCH`. | Use the metadata shape as the decode-graph and no-per-token-prefill reference. |
| `../vllm/vllm/model_executor/layers/sparse_attn_indexer.py` | Prefill uses chunk-level `fp8_fp4_mqa_logits`; decode uses paged `fp8_fp4_paged_mqa_logits`; top-k uses CUDA ops when available. | Copy the chunk/decode operator sequence. The chunk loop is acceptable; a token loop is not. |
| `../vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py` | vLLM's FP8 E4M3 path uses `m_grouped_fp8_gemm_nt_contiguous`; MXFP8/FP4 paths use the `fp8_fp4` alias and are gated to Blackwell-family devices. | GLM H200 decode should start from grouped FP8, not `m_grouped_fp8_fp4_gemm_nt_contiguous` or MegaMoE. |
| `../vllm/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | vLLM DeepGEMM permutes into DeepGEMM layout, runs W13/W2 grouped FP8 GEMMs under `mk_alignment_scope(align_used)`, then applies router weights in `deepgemm_unpermute_and_reduce` by gathering each top-k output and accumulating `expert_output * topk_weight`. This path is Triton/Python, not reusable CUDA/C++ source as-is. | Do not add a local standalone route-weight kernel as the default. First prefer an equivalent fused/source-backed route: Kimi-style expert-kernel weighting or DeepGEMM TileLang `swiglu_apply_weight_to_fp8`; only rewrite the vLLM gather/reduce semantics in CUDA after proving no existing source path fits, with NCU evidence. |
| `../vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py` | Low-latency DeepEP finalize delegates weight application and reduction to `buffer.low_latency_combine(..., topk_weights, ...)`. | This explains why vLLM has no extra route-weight kernel in the EPLL path, but GLM's current DeepEP elastic expanded layout cannot pass weights into combine. |
| `../vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py` and `deepep_ht.py` | v2/HT paths apply weight-and-reduce before combine for contiguous BF16 expert outputs, then call `buffer.combine(..., topk_weights=None)`. Decode v2 uses `do_expand=False` and `do_cpu_sync=False` for CUDA Graph. | A non-expanded/contiguous GLM path could copy this contract later. The current GLM substrate is expanded `psum_expert`, so copying this means changing layout, not adding a small local multiply. |
| `../vllm/csrc/libtorch_stable/cache_kernels.cu` | Contains CUDA `indexer_k_quant_and_cache` and `cp_gather_indexer_k_quant_cache` kernels for sparse indexer KV/cache movement. | Candidate source for GLM indexer cache insert/gather. |
| `../vllm/csrc/libtorch_stable/sampler.cu` and top-k CUDA ops | Contains `top_k_per_row_prefill` and `top_k_per_row_decode`; vLLM may also route through cooperative/persistent top-k. | Candidate source for DSA top-k if FlashInfer/FlashMLA do not supply the complete path. |
| vLLM Triton helpers around packing/unpacking | Decode padding uses Triton pack/unpack helpers in edge cases. | Rewrite locally only if needed; any CUDA rewrite needs NCU profile data before it can be called ready. |

### DP Scheduler/Threading Contract

GLM5.2 should follow Kimi's current `TP1 DP8` architecture, not invent a new scheduler shape.

| Component | Count | Responsibility |
| --- | ---: | --- |
| DP coordinator / scheduler thread | 1 | Owns request admission, DP rank selection, lifecycle, token events, and per-rank batch formation. Mirrors Kimi `DpCoordinator`. |
| Rank worker threads | 8 | One independent worker per DP rank/GPU. Each owns CUDA context, weights, KV/cache arena, DeepEP context, and forward execution. Mirrors Kimi `KimiRankWorker`. |
| Total model runtime threads | at least 9 | Excludes Tokio/frontend threads. This is the minimum server-side model runtime shape for `DP8 TP1 EP8`. |

Per-rank state ownership should stay local to the rank: request slots, KV pages, decode buffers, graph captures, and rank-local expert weights. The coordinator can route and batch rows, but it should not own GPU tensors or act as a central forward executor.

### Prefill / P-D Handoff Contract

The first runnable GLM5.2 engine does not compute prefill. The decode engine is entered only after another component has materialized the request's KV/page state. This keeps the first implementation focused on graphable decode and prevents a temporary prompt path from becoming the serving contract.

| Contract | Rule |
| --- | --- |
| No GLM engine prefill | A request that still needs prompt compute must be rejected or held for a future P/D handoff path; do not add a hidden prefill fallback. |
| No host per-token kernel loop | No `for token in prompt { launch_kernel(token) }` path may enter the GLM engine, including temporary debug code that outlives the experiment. |
| Handoff state | Decode needs a typed handoff for token ids, positions, KV pages, request slots, and any DSA/indexer cache state. |
| Future prefill implementation | When prefill returns to GLM, it must be chunked/batched; this document keeps the red line but does not track that work in the first runnable decode. |

### Decode Batch Contract

Decode must keep `bs > 1` as a first-class tensor/kernel shape. The scheduler may build per-row metadata, but the GPU work cannot be a host loop over requests.

| Contract | Rule |
| --- | --- |
| No host per-request bs=1 kernel loop | No `for req in batch { launch_decode_bs1(req) }` path for attention, router, MoE, sampling, or logits. |
| Batched attention decode | Use FlashInfer/MLA batch decode or an equivalent batched kernel over all active rows. |
| Batched router/top-k | Route all active rows together; do not call router once per request. |
| Batched DeepEP | Dispatch/combine uses the active row count for the DP rank. Empty ranks still participate; active ranks do not split rows into singleton launches. |
| Batched sampling/logits | Use shared batched sampler/logprob helpers where possible. |
| Evidence | Nsys/NCU should show per-step batched kernels, not one kernel chain per request. |

### Decode CUDA Graph Contract

Kimi already proves this direction is viable for `TP1 DP8 EP8`: DeepEP decode is host-quiet, persistent-buffer, fixed worst-case shape, and graph replay is enabled for full decode buckets. GLM5.2 should meet the same contract before claiming graph support:

| Contract | GLM5.2 design rule |
| --- | --- |
| Fixed decode bucket | Capture per bucket; partial buckets may run eager until proven safe. |
| No host sync in graph region | Router, DeepEP dispatch/combine, expert GEMM, attention decode, sampling inputs must be device-driven. |
| Persistent buffers | Page tables, token ids, positions, top-k routes, DeepEP scratch, attention scratch, and logits buffers allocated before capture. |
| Stable kernel sequence | Every rank calls the same MoE/attention kernel sequence for the bucket; no rank skips EP. |
| Numerics gate | Compare graph vs eager logits/tokens before serving graph by default. |

### MoE Decode Layout Contract

This is the contract between router, DeepEP, activation quantization, DeepGEMM, route weighting, and combine. The first MoE integration should encode these facts in types before adding more call sites.

| Boundary | Contract |
| --- | --- |
| Decode bucket | Per-rank active rows are capped by the fixed decode bucket (`128` today). The graph path uses stable worst-case allocations even when fewer requests are active. |
| DeepEP dispatch | All ranks participate every step. Empty ranks/experts are legal and must still follow the same kernel sequence; graph code must not branch around EP by rank. |
| Receive layout | `deepep_recv_x` is BF16 `[recv_rows, hidden=6144]`; `recv_topk_weights` carries the route weights for the expanded rows. |
| Grouped layout | DeepEP expand-mode `psum_expert` is the grouped-layout tensor. Expert `i` owns rows `[align(psum_expert[i - 1], 64), psum_expert[i])`; expert `0` starts at `0`; gaps are padding, not valid tokens. |
| Capacity | Worst-case expanded rows are `10240` for `topk=8`, `local_experts=32`, `expert_alignment=64`, and per-rank decode cap `128`. Persistent W13/W2 buffers are sized to that, not to the current batch. |
| DeepGEMM path | H200 first path is DeepGEMM SM90 grouped FP8 E4M3 (`m_grouped_fp8_gemm_nt_contiguous`) with F32 activation/weight scales and runtime `mk_alignment=64`. Do not use `m_grouped_fp8_fp4_gemm_nt_contiguous` or MegaMoE for GLM H200 first cut. |
| Scale layout | W13/W2 activation scales are converted to DeepGEMM SM90 MN-major/TMA-aligned F32 layout before GEMM. The FlashInfer/TRTLLM `GroupedWithOffset` fallback has a separate persistent 32-row-offset scale layout: `10240` expanded rows become `11232` padded scale rows. Device `psum_expert` stays device-resident in the forward path; D2H layout validation is only for startup/IT smoke. |
| Route weights | The current expanded DeepEP combine cannot accept `topk_weights`; DeepEP elastic rejects weights when `use_expanded_layout=true`. GLM now applies `recv_topk_weight` to the W2 input during SiLU*up -> FP8 quant, following DeepGEMM TileLang `swiglu_apply_weight_to_fp8`. DeepEP combine will therefore reduce already weighted rows once W2 GEMM writes `moe_w2_output_bf16`. Do not add a standalone local route-weight multiply; switching to vLLM gather/reduce or DeepEP LL/v2 weighting would require a layout-contract change and new profile evidence. |
| Combine source | DeepEP combine consumes `moe_w2_output_bf16`, not `deepep_recv_x`. The current smoke already uses this buffer to preserve the future call-site shape. |

## Kernel Ledger

Every operator must stay in this table. "Handwritten" means we own CUDA/C++ code in this repo; before claiming it is ready, attach NCU evidence and why none of FlashInfer, DeepGEMM, FlashMLA, DeepEP, `../vllm`, or cuBLASLt supplied a usable CUDA/C++ path. If vLLM has only Triton for an operator, say that explicitly before writing local CUDA.

| Area | Operator | Candidate source | Status | Evidence / next action |
| --- | --- | --- | --- | --- |
| Token embedding | gather embedding rows | existing model code / simple CUDA if needed | pending | Check Qwen3/Kimi embedding path before adding code. |
| RMSNorm | input/post/final norm | existing openinfer kernel or FlashInfer | pending | Prefer shared norm wrapper; no handwritten kernel until coverage confirmed. |
| Dense GEMM | q/lora/dense/shared projections | DeepGEMM FP8 GEMM / cuBLASLt / `../vllm` | attention/indexer projection smoke landed; dense/shared layer integration pending | Weights are FP8 E4M3 with F32 block `weight_scale_inv`; check DeepGEMM SM90 NT path and vLLM quant path before custom wrappers. Decode arena now owns persistent per-rank FP8 activation input scratch sized to the largest non-MoE projection input (`16384`) plus scale cols `128`; the shared plain-FP8 helper validates q_a/q_b/kv_a/kv_b/o_proj/indexer_wk/indexer_wq_b on node38, but typed norm/residual/eager-forward call sites and dense/shared projections are not wired yet. |
| Attention prefill | GLM5 MLA/DSA prefill | FlashInfer `trtllm_ragged_attention_deepseek` / FlashMLA sparse prefill / `../vllm` | deferred by decode-only first cut | Keep source notes for future P/D worker, but do not wire this into the first GLM engine. |
| Attention decode | GLM5 MLA decode | FlashMLA dense/sparse decode / FlashInfer MLA / `../vllm` | arena substrate landed; kernel wrappers pending | FlashMLA dense decode supports SM90 BF16; sparse decode supports SM90 FP8 KV. Decode arena now owns persistent q/kv/indexer/output scratch (`q_a=2048`, `q_b=65536`, `kv_a=576`, `kv_lora=512`, `k_rope=64`, `kv_b=114688`, `o_proj_in=16384`, indexer `32/128/4096/topk=2048`), but GLM cache layout and DSA decode wrappers are still unmapped. |
| RoPE / KPE assembly | interleaved partial RoPE and MLA cache write | Kimi MLA helpers if shape-compatible, otherwise FlashInfer | pending | Compare GLM q/k/v projection layout against Kimi. |
| Indexer logits | DSA indexer MQA logits | DeepGEMM FP8 MQA/indexer logits through `../vllm` sparse indexer / FlashMLA sparse indices | pending | vLLM uses chunked prefill `fp8_fp4_mqa_logits` and decode `fp8_fp4_paged_mqa_logits`; copy that sequence before attempting local CUDA. |
| Indexer cache | quant/cache and gather indexer K | `../vllm/csrc/libtorch_stable/cache_kernels.cu` | pending | Candidate CUDA source for `indexer_k_quant_and_cache` and `cp_gather_indexer_k_quant_cache`; map GLM FP8 cache layout first. |
| Indexer top-k | prefill/decode sparse top-k | `../vllm` CUDA `top_k_per_row_*`, cooperative/persistent top-k / FlashMLA | pending | Need `index_topk=2048` coverage. If the only usable path is Triton, rewrite locally with NCU profile evidence. |
| Router | sigmoid noaux top-k over 256 experts | Kimi noaux_tc router adapted for GLM / `../vllm` `dsv3_router_gemm_*` / FlashInfer `mm_M1_16_K6144_N256` | substrate landed; perf specialization pending | `openinfer-kernels/csrc/glm52/glm52_router.cu` copies Kimi's complete router contract: BF16 cuBLAS GEMM to F32 logits, sigmoid, correction-bias top-k, unbiased-score normalization, route scale `1.0` with GLM's `2.5` routed residual scale reserved for later layer integration. Node38 release checkpoint smoke runs it over 8 rows from real layer-3 router weights and validates top-k range/uniqueness and per-row weight sum. vLLM's `dsv3_router_gemm_*` remains the decode small-batch GEMM candidate before claiming performance; FlashInfer candidate currently marks supported CC `[100,103]`, so H200 fallback/port must be measured. |
| EP dispatch/combine | MoE all-to-all | DeepEP | startup decode substrate landed; real W13/W2 MoE substrate call sites landed; layer integration pending | `openinfer-glm52/src/deepep.rs` records GLM's `hidden=6144`, `experts=256`, `local=32`, `topk=8`, and decode cap 128. Prefill is not part of the GLM engine runtime; the Rust safe wrapper in `openinfer-kernels/src/ops/glm52/deepep.rs` exposes only decode dispatch/combine even though the copied C shim still carries historical prefill symbols. `openinfer-kernels/csrc/glm52_deepep/` provides a separate GLM ABI under the kernel crate `glm52` feature. Node38 release checkpoint smoke creates 8 DeepEP contexts, runs synthetic all-EP dispatch/combine, then runs router-produced routes through dispatch/combine. The smoke now validates expand-mode `psum_expert` using the DeepGEMM slice rule: expert `i` rows are `[align(psum[i-1]), psum[i])`, and empty ranks/experts are legal. The MoE GEMM and graph smokes consume DeepEP dispatch output through TRTLLM W13/W2 and combine weighted W2 output back to token rows. Next: dense/shared/residual integration, attention/KV, logits/sampling, and scheduler handoff. |
| MoE activation quant | W13 input BF16 -> FP8 group-128; W13 output SiLU*up*route_weight -> W2 FP8 group-128 | `../vllm/csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`, `../vllm/csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu`, and DeepGEMM `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` | decode substrate landed; weighted W2-input quant landed and profiled on the old capacity; TRTLLM GEMM smoke consumes it | `openinfer-kernels/csrc/glm52/glm52_moe_quant.cu` copies/adapts vLLM CUDA/C++ semantics into raw-pointer GLM ABI: row-major BF16 input, FP8 E4M3 `u8` output, F32 scales, group size 128 only. The weighted W2-input variant follows DeepGEMM TileLang: `y = silu(gate) * up * topk_weight`, then per-token/per-128 FP8 quant. Decode arena owns persistent worst-case buffers for `deepep_recv_x` -> `moe_w13_input_fp8/scale`, `moe_w13_output_bf16` -> `moe_w2_input_fp8/scale`, and `moe_w2_output_bf16` -> DeepEP combine. Node38 release checkpoint smoke validates finite positive scales, nonzero FP8 output, and `swiglu_weighted_scale_valid=true` for routed DeepEP recv rows and decode graph replay. Historical H200 profile for the pre-Step-40 shape `rows=8416, hidden=2048`: CUDA event `97.88us`, NCU full replay `122.43us`, SM `82.19%`, L1/TEX `75.56%`, DRAM `13.51%`; artifacts in `profile/glm52_weighted_swiglu_quant_20260626/`. The current `rows=10240` shape must be re-profiled before claiming performance. |
| MoE scale layout | activation F32 scales -> DeepGEMM MN-major/TMA-aligned F32 scale layout and FlashInfer/TRTLLM grouped-offset scale layout | DeepGEMM `get_mn_major_tma_aligned_tensor` / `get_tma_aligned_size`; FlashInfer/TRTLLM `compute_padded_offset` | decode substrate landed; TRTLLM grouped-offset relayout landed and profiled; TRTLLM grouped-FP8 smoke landed | `openinfer-kernels/csrc/glm52/glm52_deepgemm_layout.cu` copies DeepGEMM's SM90 F32 scale layout contract into a raw-pointer GLM ABI: input row-major `[rows, scale_cols]`, output logical shape `[rows, scale_cols]` with strides `(1, aligned_rows)`, and `aligned_rows=align(rows, 16 / sizeof(f32))`. The same file also owns the FlashInfer/TRTLLM `GroupedWithOffset` scale relayout: GLM's `10240` expanded rows and 32 local experts produce `11232` padded scale rows through the TRTLLM 32-row offset formula. Decode arena owns persistent worst-case DeepGEMM and TRTLLM scale buffers for W13 and W2 input scales; node38 release checkpoint smoke validates both layouts for all 8 ranks, and decode graph replay validates the TRTLLM relayout in-graph. H200 NCU evidence for the local relayout kernel lives in `profile/glm52_trtllm_offset_scale_20260626/`: standalone W13-scale shape takes `22.46-23.04us`, SM compute about `52%`, DRAM throughput under `2%` peak, and the main rule-engine note is a partial-wave/tail effect. This is layout evidence, not a GEMM performance claim. |
| Routed expert GEMM | FP8 routed experts | DeepGEMM grouped FP8 GEMM / FlashInfer TRTLLM grouped-offset runner / FlashInfer fused MoE / cuBLASLt / `../vllm` | weight package + activation quant + scale-layout + graph-captured grouped-layout metadata + DeepGEMM package-plan + weighted W2-input + all-layer GEMM C ABI contract + fail-closed DeepGEMM launch boundary + TRTLLM grouped-offset scale relayout + TRTLLM raw-runner ABI/workspace gate + real W13/W2 TRTLLM smoke and graph capture landed; full layer integration/perf pending | Kimi Marlin INT4 path is not applicable as a GEMM backend, but its DeepEP contract is directly relevant: expanded dispatch feeds expert-major rows, W2 applies per-slot router weight, and combine reduces already weighted outputs. GLM now streams local routed experts into per-layer expert-major `W13=[local,4096,6144]` and `W2=[local,6144,2048]` FP8 packages with F32 checkpoint block scales, deleting the raw routed expert tensors after each layer is packed so resident memory does not double. W13 package order is DeepGEMM-compatible per-expert `[gate; up]`: `[expert0 gate, expert0 up, expert1 gate, expert1 up, ...]`, with scales in the same order. `Glm52DeepGemmMGroupedFp8WeightPlan` validates the exact DeepGEMM `[G,N,K]` / `[G,N/128,K/128]` package view: W13 is `[32,4096,6144]` and down is `[32,6144,2048]` per rank. Startup now validates every rank has exactly 75 sorted routed layers `3..=77`, graph-stable arena capacity `m_capacity=10240`, DeepEP `psum_expert` entries `32`, `expert_alignment=64`, W13 operand `G=32,N=4096,K=6144`, and W2 operand `G=32,N=6144,K=2048`; the report passes through `openinfer-kernels/src/ops/glm52/deepgemm_grouped.rs` / `glm52_deepgemm_grouped_fp8_contract_cuda` and `openinfer-kernels/src/ops/glm52/trtllm_grouped.rs` / `glm52_trtllm_grouped_fp8_workspace_size_cuda` before the server starts. The grouped metadata path now consumes the real `Glm52DeepEpDispatchScratch::psum_expert` produced by DeepEP dispatch, not an arena shadow buffer, then writes persistent arena-owned `expert_offsets[int64]`, W13 problem sizes `[m,4096,6144]`, and W2 problem sizes `[m,6144,2048]`; the CUDA kernel is captured in the decode-substrate graph and validated on node38 against router-produced layouts including empty ranks. Step 40 moved DeepEP expansion to 64-row expert alignment so this psum layout can feed DeepGEMM SM90 `MGroupedContiguousWithPsumLayout` under runtime `mk_alignment=64` without a post-DeepEP repack. The same Rust module also exposes `glm52_deepgemm_grouped_fp8_launch`, which validates activation, scale, weight, psum, and output buffer lengths before entering the C ABI; the C launch intentionally returns `CUDA_ERROR_NOT_SUPPORTED` until a real raw DeepGEMM runtime exists. The TRTLLM wrapper exposes `glm52_trtllm_grouped_fp8_launch` over the vendored `CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3,__nv_fp8_e4m3,__nv_bfloat16>` and validates GLM W13/W2 FP8 buffers, `expert_offsets`, `11232`-row activation scales, and zero workspace; startup evidence on node38 reports `trtllm_workspace_bytes=0` for W13 and W2 on all ranks. H200 first path is currently the expanded-row TRTLLM grouped FP8 route, not the SM100/MXFP FP8xFP4 alias. Route weighting uses DeepGEMM TileLang `swiglu_apply_weight_to_fp8` semantics during W2 input quant, so DeepEP combine sees already weighted W2 outputs in the MoE GEMM smoke and graph smoke. DeepGEMM MegaMoE PR-304 fuses dispatch/linear1/SwiGLU/linear2/combine but its public API dispatches only to arch major 10, so it is a design reference, not the GLM H200 route. Still pending: residual/layer integration, attention/KV, logits/sampling, measured DeepGEMM-vs-TRTLLM backend choice, H200 NCU/perf evidence, and full forward graph integration. |
| Shared expert | dense SwiGLU expert | DeepGEMM FP8 GEMM / cuBLASLt / `../vllm` | arena substrate landed; layer integration pending | Shapes match routed expert; likely share projection wrappers. Decode arena now owns shared gate/up `[bs,4096]` and activation `[bs,2048]` scratch, but shared expert compute and residual composition are not wired. |
| Sampling | greedy and non-greedy token selection | `openinfer-sample` / FlashInfer sampling | pending | Reuse shared sampler; no model-local sampler. |
| KV cache | MLA paged cache | Kimi paged MLA cache shape adapted | pending | GLM `kv_lora_rank=512`, `qk_rope_head_dim=64`, `v_head_dim=256`; page layout must be documented. |
| Decode graph capture | full decode step | Kimi DeepEP graph pattern | full-layer arena substrate + MoE decode-substrate graph landed; full forward graph pending | Startup now captures and replays a fixed 128-token decode-substrate graph on all 8 ranks: checkpoint router, DeepEP dispatch, grouped-GEMM metadata, W13 input quant, TRTLLM W13 grouped FP8, weighted W2 input quant, TRTLLM W2 grouped FP8, and DeepEP combine. The arena now also pre-allocates full-layer decode scratch so attention/dense/shared/logits can enter the same fixed-bucket contract later without CUDA allocation in the graph region. This is not yet the complete decode forward graph because attention/KV, dense/shared/residuals, logits, and sampling kernels are still absent. |
| MTP | next-token prediction layer | none in first cut | deferred | GLM5.2 config has `num_nextn_predict_layers=1`; first cut does not serve MTP. |
| Per-token prefill loop | none | prohibited | rejected | First decode-only runtime must not grow a hidden prompt path; any future prefill worker that launches kernels per prompt token fails the contract. |
| Per-request decode loop | none | prohibited | rejected | Any design that launches bs=1 kernel chains for each active request fails the decode batch contract. |
| DP runtime shape | scheduler + rank workers | Kimi `DpCoordinator` + 8 worker threads | required | At least 9 independent model runtime threads for `DP8 TP1 EP8`; do not collapse into one host executor loop. |

## Execution Log

### Step 1: Machine and model probe

- Confirmed `xingming-dev` is active on jiuzhang node 38 (`host-172-31-13-38`) and holds 8 H200 GPUs with a balloon pod; host GPUs are idle.
- Confirmed GLM5.2 model path exists: `/data/models/GLM-5.2-0614-Provider-FP8`.
- Confirmed model files include `config.json`, `generation_config.json`, tokenizer files, `model.safetensors.index.json`, and 144 safetensor shards.
- Updated first-cut scope: MTP is not supported initially; decode CUDA Graph is a goal.
- Added prefill red line: no host-side per-token kernel-launch loop; prefill must use batched/chunked operators.
- Added decode red line: `bs > 1` must be real batched execution, not a host loop over bs=1 kernel chains.
- Added DP runtime shape: follow Kimi's 1 coordinator + 8 rank-worker architecture, at least 9 model runtime threads.
- Added reuse-first policy: frontend follows Qwen3-4B, DP/MoE follows Kimi-K2, and new code needs a GLM-specific reason.
- Parsed representative safetensor headers without loading weights; recorded FP8/BF16 shapes and tensor counts above.

### Step 2: Branch and repo prep

- Started from clean local `main`.
- Ran `git pull --ff-only`, fast-forwarding to `1ee1319`.
- Created branch `feat/glm52-dp8-ep8`.

### Step 3: Operator source setup

- Confirmed existing submodules:
  - FlashInfer at `openinfer-kernels/third_party/flashinfer` (`d768c14`, `v0.6.12`).
  - DeepEP at `openinfer-kernels/third_party/DeepEP` (`d4f41e4`, `v1.2.1-32-gd4f41e4`).
- Added requested submodules:
  - DeepGEMM at `openinfer-kernels/third_party/DeepGEMM` (`54e2261`, `v2.1.1.post3-19-g54e2261`).
  - FlashMLA at `openinfer-kernels/third_party/FlashMLA` (`9241ae3`, `heads/main`).
- Confirmed sibling vLLM checkout:
  - `../vllm` -> `/data/code/workspace-rustllm/vllm`, commit `4d3b4b9b0`, branch `main` behind `origin/main` by 7 at probe time, plus one unrelated untracked draft file.
- Updated kernel policy: search FlashInfer, DeepGEMM, FlashMLA, DeepEP, `../vllm`, then cuBLASLt before writing local CUDA. When vLLM has CUDA/C++ for a GLM-shaped operator, copy/adapt it; handwritten CUDA requires a vLLM Triton-only or missing-operator reason plus NCU evidence.
- Found vLLM router source for GLM5.2 decode-sized routing: `dsv3_router_gemm_entry.cu` checks `hidden_dim=6144` and `num_experts=256`, and `dsv3_router_gemm_float_out.cu` instantiates bf16-input/F32-output kernels for `num_tokens=1..16` on SM90+.
- Found vLLM DSA indexer path: metadata chunks prefill by request/query ranges, decode metadata supports non-MTP `(B,1)` sequence lengths and `UNIFORM_BATCH` CUDA Graph support, and sparse indexer execution uses DeepGEMM MQA logits plus CUDA cache/top-k helpers.
- Recorded Triton boundary: vLLM has Triton helpers in the indexer padding/unpadding path; only rewrite those locally when GLM needs them, and treat that as a handwritten kernel requiring NCU proof.

### Step 4: Crate and frontend surface

- Added workspace crate `openinfer-glm52`.
- Added GLM5.2 config probe constants and validation for the provider shape:
  - `model_type=glm_moe_dsa`, `GlmMoeDsaForCausalLM`, bf16 activations, FP8 E4M3 dynamic activation, `[128,128]` weight blocks.
  - 78 layers, first 3 dense and remaining sparse, `hidden=6144`, `vocab=154880`, `num_attention_heads=64`, `q_lora_rank=2048`, `kv_lora_rank=512`, `qk_nope=192`, `qk_rope=64`, `v_head_dim=256`.
  - MoE/router shape `n_routed_experts=256`, `topk=8`, `n_shared_experts=1`, `noaux_tc`, sigmoid, normalized top-k, routed scale 2.5.
  - DSA indexer shape `index_topk=2048`, `index_topk_freq=4`, `index_head_dim=128`, `index_n_heads=32`.
- Added `load_stop_token_ids`, preferring `generation_config.json` and falling back to `config.json`; provider stop tokens are `[154820, 154827, 154829]`.
- Added `Glm52ParallelShape::tp1_dp8()` so local expert ownership is explicit: `EP8` gives `32` routed experts per rank.
- Added `Glm52LaunchOptions`, `launch`, and `start_engine`. The path validates config, stop tokens, `model.safetensors.index.json`, `TP1/DP8/EP8`, device count, and `--ep-backend=deepep`, then enters the GLM `EngineHandle` path. Until forward lands, the coordinator schedules each request and returns an explicit rejection: no fake token generation and no silent fallback.
- Wired `openinfer-server`:
  - Added optional feature `glm52`.
  - `detect_model_type` now recognizes `glm_moe_dsa` before the Qwen fallback; feature-disabled builds report `--features glm52` instead of misclassifying GLM as Qwen.
  - `load_engine` forwards Qwen-style frontend requests into `openinfer_glm52::launch`.
  - `bench_serving` detects GLM5.2 and records `tp/dp/ep` shape metadata, but still fails at model load until runtime execution exists.
- Fixed an existing `bench_serving` Qwen3 compile drift exposed by the `glm52` check: `start_engine_with_offload` now receives the current memory/overlap/batch-invariant/dflash/KV-event arguments.
- Local verification:
  - `cargo fmt`
  - `cargo check -p openinfer-server --features glm52` -> passed
  - `cargo check -p openinfer-server` -> passed

### Step 5: Safetensor manifest and EP8 rank plans

- Added GLM5.2 safetensor manifest parsing in `openinfer-glm52/src/weights.rs`.
- The manifest parses `model.safetensors.index.json` without loading tensor data and validates:
  - top weights: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`.
  - runtime layers `0..77` only.
  - dense layers `0..2` with FP8 gate/up/down projections and scale tensors.
  - MoE layers `3..77` with router weight, router correction bias, shared expert FP8 projections, and all `256` routed experts.
  - full-indexer layers `0,1,2,6,10,...,74`; shared-indexer layers intentionally have no per-layer `self_attn.indexer.*` tensors.
  - deferred `model.layers.78.*` nextn/MTP tensors: provider index has `1569` of them.
- Added rank plan derivation for first-cut `TP1 DP8 EP8`:
  - EP8 assigns `32` routed experts per rank.
  - rank0 local experts `0..32`, rank7 local experts `224..256`.
  - each rank load plan has `16260` tensors: replicated non-expert runtime weights plus local routed expert weights.
  - load specs are full-tensor loads for now because first cut is TP1; no TP row/column slicing is implemented or exposed.
- `start_engine` now validates the real safetensor manifest, builds all 8 EP-rank tensor/load plans, checks their tensor counts agree, and mmap-validates rank0 safetensor headers before failing closed, so a malformed GLM5.2 directory fails before runtime work begins.
- Rank0 header validation checks every selected tensor is present in its shard as a full-tensor load and matches the observed provider dtype/shape contract:
  - BF16 top weights and norms: embedding/head `[154880,6144]`, final norm `[6144]`.
  - Attention FP8 projections: `q_a [2048,6144]`, `q_b [65536,2048]`, `kv_a [576,6144]`, `kv_b [114688,512]`, `o_proj [6144,16384]`, plus F32 block scales.
  - Indexer tensors: BF16 `weights_proj [32,6144]`, FP8 `wk [128,6144]`, FP8 `wq_b [4096,2048]`, plus F32 block scales.
  - Dense and MoE FP8 MLP tensors: dense intermediate `12288`, routed/shared expert intermediate `2048`, router BF16 `[256,6144]`, correction bias F32 `[256]`.
- Local verification after manifest work:
  - `cargo fmt --check`
  - `cargo test -p openinfer-glm52` -> `2` public API integration tests passed; crate-internal unit tests are intentionally absent.
  - `cargo check -p openinfer-server --features glm52` -> passed
  - `cargo check -p openinfer-server` -> passed

### Step 6: Feature and test boundary cleanup

- Removed the internal `glm52` feature from `openinfer-glm52`. The model crate now compiles as one coherent GLM crate; `openinfer-server` still uses its `glm52` feature to include or exclude the dependency.
- Deleted synthetic crate-internal unit tests that mostly duplicated private manifest construction and tensor-shape contracts. Keeping them would have inflated test surface without materially improving confidence.
- The early synthetic probe integration tests were later deleted in Step 46 after the real checkpoint IT covered the same public config/stop-token path plus manifest, rank loading, and startup smokes. Manifest/header confidence comes from fail-closed startup validation against real checkpoint files, not a giant synthetic fixture.

### Step 7: All-rank startup validation

- Extended startup validation from rank0-only plan construction to all `EP8` ranks:
  - every rank now builds a `Glm52RankLoadBundle` containing `rank_plan` and `rank_sliced_load_plan`.
  - every rank must have matching plan/load tensor counts.
  - all ranks must have the same tensor count (`16260` for the provider manifest shape), catching asymmetric expert partition mistakes before worker bring-up.
- Added device ordinal uniqueness validation for the first-cut `DP8 TP1 EP8` launch contract. Duplicate ordinals now fail before any runtime worker setup.
- Kept safetensor dtype/shape header validation on rank0 for now. Rank0 exercises the full replicated and local-expert contract without repeatedly deserializing large shard metadata eight times during this fail-closed phase; full data reads belong in the real GPU loader.
- Probed jiuzhang node 38 for an existing pegainfer checkout under `/root/develop` and `/data/code`; no checkout was found. Real checkpoint validation on node 38 needs a remote worktree first, preferably by branch/remote rather than copying a local tree.
- Local verification:
  - `cargo fmt --check`
  - `cargo test -p openinfer-glm52` -> `2` public API integration tests passed; crate-internal unit tests remain absent.
  - `cargo check -p openinfer-server --features glm52` -> passed.

### Step 8: Raw GPU weight loader

- Added GLM5.2 CUDA rank context and raw safetensor H2D loader:
  - `Glm52RankGpuContext` mirrors Kimi's rank-owned CUDA context/stream boundary and initializes cuBLAS for the worker thread.
  - `load_rank_sliced_weights_to_gpu` consumes `Glm52RankLoadBundle`, mmap-deserializes each shard, validates dtype/shape against the provider contract, and copies raw tensor bytes to the target GPU.
  - FP8 weights/scales, BF16 norms/top weights, and F32 router correction tensors are resident as raw byte tensors for now; typed views and kernel-specific packing are still the next layer.
- Added `Glm52RankWorker`, modeled after Kimi's rank worker boundary:
  - `start_engine` spawns 8 rank worker threads for `DP8 TP1 EP8`.
  - each worker owns its CUDA context and holds its raw GPU weights after load.
  - loading is dispatched asynchronously to all workers and reports per-rank tensor count/bytes before the final fail-closed forward-runtime error.
- This is intentionally real loading, not a mock path: a GLM start on node 38 should now validate CUDA context creation and H2D copies before accepting requests through the normal server path.
- Added the first DP coordinator lifecycle:
  - `start_engine` now returns an `EngineHandle` after loading rank weights.
  - the coordinator thread owns the 8 loaded rank workers, preserving rank CUDA contexts and resident raw weights until engine shutdown.
  - every request receives `Scheduled` followed by `Rejected` with the missing forward surface named explicitly.
  - this completes the thread/lifecycle skeleton for 1 coordinator + 8 rank workers; the forward executor is still pending.
- Local verification:
  - `cargo fmt`
  - `cargo test -p openinfer-glm52` -> `2` public API integration tests passed.
  - `cargo check -p openinfer-server --features glm52` -> passed.

### Step 9: Test boundary

- Applied the test rule from `docs/conventions/coding-style.md`: keep a test only if deleting it would lower correctness confidence.
- The GLM crate still has no crate-internal unit tests. The two default public integration tests are:
  - provider-shaped model probing plus stop-token loading.
  - provider-shape drift rejection through the public config probe.
- Added `openinfer-glm52/tests/checkpoint.rs` as an ignored integration test, not an env-gated path. It hardcodes the jiuzhang checkpoint path and must be run explicitly with `-- --ignored`.

### Step 10: Jiuzhang checkpoint startup smoke

- Built and ran the ignored checkpoint IT on `jz-38` against `/data/models/GLM-5.2-0614-Provider-FP8`:
  - command shape: `cargo test -p openinfer-glm52 --test checkpoint --release -- --ignored --nocapture`.
  - release compile finished in `44.12s`.
  - test passed in `318.12s`.
  - every H200 rank reached roughly `92-94 GiB` resident memory mid-load, proving real CUDA context creation and raw H2D weight residency rather than a metadata-only path.
  - after load, the test submits one request through `EngineHandle::submit` and verifies `Scheduled` followed by `Rejected`, so the loaded workers stay owned by the coordinator and no fake forward tokens are emitted.
  - GPUs returned to `0 MiB` after test exit, so worker teardown/drop released device memory.
- The first server release attempt on node38 failed before GLM runtime with `openssl-sys` missing system OpenSSL development metadata (`openssl.pc`). Step 11 fixes this with vendored OpenSSL and validates the full server path.

### Step 11: Jiuzhang server HTTP smoke

- Fixed the node38 release server build by making `openinfer-server` depend on vendored OpenSSL. The vLLM frontend dependency chain pulls `native-tls` through HF/download clients, so release builds should not require host `libssl-dev`/`openssl.pc`.
- Fixed GLM5.2 chat template loading in the vLLM Rust frontend by detecting `model_type=glm_moe_dsa`, reading `chat_template.jinja`, and applying the Minijinja-compatible `m.content[0].type` form. This keeps the override model-specific and does not add a runtime environment switch.
- Tightened the clap feature boundary:
  - GLM launch uses existing CLI args: `--model-path`, `--served-model-name`, `--tp-size`, `--dp-size`, `--ep-backend`, `--cuda-graph`.
  - Qwen-only clap fields (`--gpu-memory-utilization`, `--decode-overlap`, `--batch-invariant`, etc.) no longer require the Qwen crate in a GLM-only build.
  - `cargo check -p openinfer-server --no-default-features --features glm52` now passes locally.
- Changed the local vLLM bridge mapping for `TokenEvent::Rejected`: real `TokenEvent::Error` still maps to `EngineCoreFinishReason::Error`, but request-level rejection maps to `Stop` with `stop_reason=message` because the current vLLM Rust server drops `stop_reason` on the `Error` branch and returns a generic `Internal server error`.
- Remote release build on `jz-38`:
  - command shape: `cargo build --release -p openinfer-server --features glm52`.
  - result: passed in `13.27s` after the small resync rebuild.
- Remote OpenAI HTTP smoke on `jz-38`:
  - launch shape: `target/release/openinfer --model-path /data/models/GLM-5.2-0614-Provider-FP8 --served-model-name glm52-smoke --tp-size 1 --dp-size 8 --ep-backend deepep --port 18080`.
  - `/v1/models` returned `glm52-smoke` with `max_model_len=1048576`.
  - `/v1/completions` for prompt `hello`, `max_tokens=1` returned HTTP `200` with empty completion, `finish_reason="stop"`, and a request-time rejection message in `stop_reason`. The message has since been updated to say the pending runtime is decode-only and needs prefilled KV handoff, batched decode `bs > 1`, DeepEP MoE, and decode CUDA Graph.
  - server log showed startup validation after `235606ms`, rank tensor counts `[16260; 8]`, rank GPU bytes `[122540541504; 8]`, and `nextn_tensors=1569`.
  - GPUs returned to `0 MiB` after the smoke script cleaned up the server process.
- Local verification after this step:
  - `cargo fmt --check` -> passed.
  - `cargo test -p openinfer-vllm-frontend rejected_request_preserves_rejection_message` -> passed.
  - `cargo test -p openinfer-glm52` -> `2` integration tests passed; ignored checkpoint test remained ignored by default.
  - `cargo check -p openinfer-server --features glm52` -> passed.
  - `cargo check -p openinfer-server` -> passed.
  - `cargo check -p openinfer-server --no-default-features --features glm52` -> passed.
  - `git diff --check` -> passed.

### Step 12: Typed rank weight views

- Added typed rank weight names and resident GPU views in `openinfer-glm52/src/weights/view.rs`, following Kimi's manifest-name -> raw GPU tensor -> typed view pattern but keeping GLM's FP8 `weight + weight_scale_inv` layout instead of Kimi's INT4/Marlin packaging.
- Each `Glm52RankLoadBundle` now carries:
  - the EP rank plan (`tp_rank`, `ep_rank`, vocab range, local expert range).
  - the raw safetensor load plan.
  - the typed names for top weights, per-layer attention, optional full-indexer tensors, dense MLP, router, shared experts, and the rank-local routed experts.
- After H2D load, the rank worker now builds a typed resident view and verifies it covers exactly the same tensor count and bytes as the raw resident map:
  - `78` runtime layers.
  - `3` dense layers and `75` MoE layers.
  - `21` full-indexer layers.
  - `32` local routed experts per MoE layer under `EP8`.
  - provider rank tensor count remains `16260`.
- This does not implement kernels yet. It removes the next forward-layer hazard where attention/MoE code would otherwise repeatedly fish tensors out of a string-keyed map.
- Local verification:
  - `cargo fmt -p openinfer-glm52`
  - `cargo check -p openinfer-glm52` -> passed.
  - `cargo test -p openinfer-glm52` -> `2` integration tests passed; ignored checkpoint test remained ignored by default.
  - `cargo check -p openinfer-server --features glm52` -> passed.
- Remote verification on `jz-38`:
  - synced `openinfer-glm52/src/weights.rs`, `openinfer-glm52/src/weights/load.rs`, and `openinfer-glm52/src/weights/view.rs` into `/root/develop/xingming/pegainfer-glm52`.
  - command: `/root/.cargo/bin/cargo test -p openinfer-glm52 --test checkpoint -- --ignored --nocapture`.
  - result: passed in `223.65s` against `/data/models/GLM-5.2-0614-Provider-FP8`.
  - post-test `nvidia-smi` showed all 8 H200 GPUs at `0 MiB`, and no checkpoint/server process remained.

### Step 13: Decode persistent arena

- Added `openinfer-glm52/src/arena.rs` and allocate one decode arena per rank immediately after raw weight H2D + typed-view validation.
- The arena is a fixed `batch_capacity=128` bucket for the future decode CUDA Graph path; it does not run kernels yet and does not change the current fail-closed request behavior.
- Per-rank persistent decode arena shape:
  - hidden input/output buffers: `[128, 6144]` BF16.
  - router logits: `[128, 256]` F32.
  - top-k idx/weights: `[128, 8]` I32/F32.
  - DeepEP dispatch scratch: `rank_count=1056`, `dst_slot=1024`, `psum_rank=8`, `psum_expert=33`.
  - DeepEP decode worst receive rows: `1024 = 8 ranks * 128 tokens`.
  - DeepEP expanded rows: `8416`, aligned to 8 rows per local expert segment for `32` local experts.
  - DeepEP source metadata entries: `10240 = 1024 * (topk 8 + 2)`.
  - arena bytes: `106,783,908` bytes (`~101.84 MiB`) per rank.
- This is the first concrete decode-graph substrate: buffers are pointer-stable, rank-local, and allocated before requests. It still needs the GLM DeepEP shim/config, router kernel, expert GEMM wrappers, attention cache, and graph capture before serving.
- Local verification:
  - `cargo fmt -p openinfer-glm52`
  - `cargo check -p openinfer-glm52` -> passed without warnings.
  - `cargo test -p openinfer-glm52` -> `2` integration tests passed; ignored checkpoint test remained ignored by default.
  - `cargo check -p openinfer-server --features glm52` -> passed.
  - `git diff --check` -> passed.
- Remote verification on `jz-38`:
  - synced `openinfer-glm52/Cargo.toml`, `openinfer-glm52/src/lib.rs`, `openinfer-glm52/src/arena.rs`, `openinfer-glm52/src/runner.rs`, and `openinfer-glm52/src/weights/context.rs`.
  - command: `/root/.cargo/bin/cargo test -p openinfer-glm52 --test checkpoint -- --ignored --nocapture`.
  - result: passed in `208.96s` against `/data/models/GLM-5.2-0614-Provider-FP8`.
  - post-test `nvidia-smi` showed all 8 H200 GPUs at `0 MiB`, and no checkpoint/server process remained.

### Step 14: GLM DeepEP shape contract

- Added `openinfer-glm52/src/deepep.rs` as the GLM-specific DeepEP contract instead of reusing the Kimi-baked shim dimensions by accident.
- The contract records the exact first-cut GLM topology and H200 capacity assumptions:
  - `EP8`, `hidden=6144`, `routed_experts=256`, `local_experts=32`, `topk=8`.
  - expert segment alignment `8`, device SM count `132`.
  - decode cap `128` tokens per rank.
- Decode capacity now comes from this contract and is consumed by the persistent decode arena:
  - worst receive rows `1024 = 8 * 128`.
  - worst expanded rows `8416`.
  - source metadata entries `10240`.
  - rank-count scratch `1056 = 132 * 8`.
- Prefill capacity is not part of the GLM engine contract. The first runnable runtime receives prefilled KV/page state from a later P/D handoff path.
- Startup validation computes and logs only the GLM DeepEP decode capacity. At this step it was not a DeepEP backend install yet; Step 15 adds the GLM kernel ABI substrate, while runtime worker integration is still pending.
- Local verification:
  - `cargo fmt --check` -> passed.
  - `cargo test -p openinfer-glm52` -> `2` integration tests passed; ignored checkpoint test remained ignored by default.
  - `cargo check -p openinfer-server --features glm52` -> passed.
  - `cargo check -p openinfer-server --no-default-features --features glm52` -> passed.
  - `git diff --check` -> passed.
- Remote verification on `jz-38`:
  - synced `openinfer-glm52/src/deepep.rs`, `openinfer-glm52/src/arena.rs`, `openinfer-glm52/src/lib.rs`, `docs/models/glm52/support.md`, and `docs/index.md`.
  - command: `/root/.cargo/bin/cargo test -p openinfer-glm52 --test checkpoint -- --ignored --nocapture`.
  - result: passed in `201.89s` against `/data/models/GLM-5.2-0614-Provider-FP8`.
  - post-test `nvidia-smi` showed all 8 H200 GPUs at `0 MiB`, and no checkpoint/server process remained.

### Step 15: GLM DeepEP kernel substrate

- Added an `openinfer-kernels` feature named `glm52` for the GLM-specific DeepEP shim. This is deliberately separate from the server/model `glm52` feature, so ordinary GLM server builds still do not require NCCL.
- Added GLM-specific DeepEP CUDA/C ABI sources:
  - `openinfer-kernels/csrc/glm52_deepep/glm52_deepep.h`
  - `openinfer-kernels/csrc/glm52_deepep/glm52_deepep_config.cuh`
  - `openinfer-kernels/csrc/glm52_deepep/glm52_deepep_shim.cu`
- Added matching Rust FFI/safe wrappers:
  - `openinfer-kernels/src/ffi/glm52/deepep.rs`
  - `openinfer-kernels/src/ops/glm52/deepep.rs`
- The GLM shim bakes the first-cut H200 topology and dimensions instead of reusing Kimi's:
  - `EP8`, `hidden=6144`, `routed_experts=256`, `local_experts=32`, `topk=8`.
  - decode cap `128`, device SM count `132`, expert alignment `8`.
  - exported C symbols are `glm52_deepep_*`, so Kimi's `deepep_*` ABI cannot be accidentally linked for GLM.
- Updated `openinfer-kernels/build.rs` so:
  - Kimi DeepEP sources compile only with `kimi-k2`.
  - GLM DeepEP sources compile only with kernel feature `glm52`.
  - both DeepEP shims share the existing NCCL build hook and DeepEP nvcc flags.
  - `openinfer-glm52` directly depends on `openinfer-kernels` with the `glm52` kernel feature, so GLM server builds include the GLM DeepEP shim rather than exposing an optional GLM no-DeepEP path.
  - no GLM runtime environment switch was added; server shape remains clap-driven (`--tp-size`, `--dp-size`, `--ep-backend`).
- Local verification:
  - `cargo fmt`
  - `cargo check -p openinfer-server --features glm52` -> passed.
  - `cargo check -p openinfer-server --no-default-features --features glm52` -> passed.
  - `cargo fmt --check` -> passed.
  - `git diff --check` -> passed.
- Remote verification on `jz-38`:
  - system NCCL is `2.29.7`, which is too old for the DeepEP device API requirement (`>=2.30.4`).
  - installed `nvidia-nccl-cu13==2.30.7` into `/tmp/openinfer-nccl-cu13` as a temporary build-only root.
  - command shape: `OPENINFER_NCCL_ROOT=/tmp/openinfer-nccl-cu13/nvidia/nccl /root/.cargo/bin/cargo check -p openinfer-kernels --features glm52`.
  - result: passed in `42.49s`, compiling the GLM shim for `sm_90`.
  - post-check `nvidia-smi` showed all 8 H200 GPUs at `0 MiB`, and no openinfer/checkpoint/cargo validation process remained.

### Step 16: GLM DeepEP worker install and startup timing

- Wired GLM rank workers to install a GLM-specific DeepEP context after all rank weights and decode arenas are resident:
  - rank 0 creates a `glm52_deepep_unique_id`.
  - all 8 workers receive `EnableDeepEp` before the coordinator waits on any one response, matching Kimi's collective bootstrap shape.
  - every worker validates the GLM DeepEP ABI (`hidden=6144`, `experts=256`, `local=32`, `topk=8`, decode cap `128`) before creating its context.
- Fixed worker teardown order for DeepEP: shutdown is now requested on every rank before joining any rank. A previous sequential drop could let one rank enter DeepEP destroy while peers were still blocked elsewhere, producing barrier timeout noise after an otherwise passing test.
- Added logforth initialization to the ignored checkpoint IT and added Kimi-style top-level GLM startup timing logs.
- Remote release checkpoint IT on `jz-38`:
  - command shape: `cargo test --release -p openinfer-glm52 --test checkpoint -- --ignored --nocapture`.
  - result: passed; cargo test runtime `201.05s`.
  - rank worker spawn cost `4.42s`.
  - all-rank weight load cost `188.12s`.
  - GLM DeepEP context install cost `4.05s`.
  - every rank loaded `16260` tensors and retained `122,647,325,412` bytes (raw weights plus persistent decode arena) on its H200.
  - request behavior remains fail-closed: `Scheduled` then `Rejected`, no forward tokens.
- Warm rerun on the already-built release binary showed the load time is not mostly cargo or first-touch noise:
  - total test runtime `196.13s`, all-rank weight load `185.07s`, DeepEP install `3.30s`.
  - node38 had `489GiB` page cache available before the rerun, while the GLM checkpoint directory is `715G`; the full working set does not fit in cache.
  - `iostat -dx 5` showed `/data` device `nvme2n1` reading around `2.4-2.5GB/s` at `98-99%` utilization during the load, so the current raw loader is disk/page-cache bound as well as doing about `981GB` aggregate H2D across 8 ranks.
- Reduced node38 2MiB hugepages from `768000` (`1500GiB`) to `512000` (`1000GiB`) by setting each NUMA node to `256000`; all hugepages were free before the change (`HugePages_Rsvd=0`), so this did not evict a hugepage user.
- Post-hugepage timing:
  - first rerun after freeing memory: total test runtime `143.95s`, all-rank weight load `130.44s`, DeepEP install `3.74s`; page cache grew from about `508GB` to `741GB`.
  - second rerun with the checkpoint fully warm in page cache: total test runtime `62.59s`, all-rank weight load `51.08s`, DeepEP install `3.75s`.
  - `iostat -dx 5` on the second rerun showed `/data` `nvme2n1` at `0 r/s` throughout the captured load window, so the remaining `~51s` is loader-side work over cached pages: safetensor deserialize/lookups, per-tensor allocation/copy, typed view validation, and aggregate H2D.
- Added a GLM DeepEP decode dispatch/combine startup smoke:
  - after context install, every rank seeds a batched 8-row route tensor with top-k experts `[0,32,64,96,128,160,192,224]`, so all 8 EP ranks participate.
  - the smoke calls GLM `decode_dispatch` then `decode_combine` through the same rank-owned DeepEP context and persistent decode arena buffers intended for real decode.
  - hidden states are zero and the smoke asserts the combined output remains all-zero, so it validates the ABI/collective/buffer contract without claiming model numerics.
  - node38 release checkpoint rerun passed in `57.65s` with warm page cache: weight load `45.45s`, DeepEP install `3.93s`, DeepEP decode smoke `<1ms`, all 8 smoke reports `combined_zero=true`, and GPUs returned to `0MiB`.

### Step 17: GLM router decode smoke

- Added a GLM router CUDA/FFI/ops substrate under the `openinfer-kernels` `glm52` feature:
  - CUDA source: `openinfer-kernels/csrc/glm52/glm52_router.cu`.
  - Rust FFI/wrapper: `openinfer-kernels/src/ffi/glm52.rs` and `openinfer-kernels/src/ops/glm52/router.rs`.
  - The implementation copies Kimi's complete noaux_tc router semantics and specializes validation to GLM's BF16 gate shape `[256,6144]`, `topk=8`.
  - It intentionally keeps router weights normalized with scale `1.0`; GLM's routed scaling factor `2.5` belongs at the later residual/layer boundary, matching Kimi's rounding boundary.
- Wired startup smoke so each rank now does two batched 8-row decode checks:
  - synthetic route smoke still covers all 8 EP ranks with experts `[0,32,64,96,128,160,192,224]`.
  - real-router smoke seeds nonzero BF16 hidden rows, runs the layer-3 router gate/bias from the real checkpoint, validates top-k index range/uniqueness and per-row normalized weights, then zeros hidden and runs DeepEP dispatch/combine with those router-produced routes.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed, proving default Qwen build path does not compile GLM feature code.
  - remote `jz-38`: `OPENINFER_NCCL_ROOT=/tmp/openinfer-nccl-cu13/nvidia/nccl cargo check -p openinfer-glm52` -> passed.
  - remote `jz-38` release checkpoint IT: `RC=0`, external elapsed `100.26s` including a `43.67s` release rebuild; test body `55.26s`, rank worker spawn `3.84s`, weight load `43.91s`, DeepEP install `3.61s`, combined synthetic+router DeepEP decode smoke `0.09s`; all 8 reports had `router_routes_valid=true`, `router_weights_normalized=true`, and `combined_zero=true`.
  - post-test `nvidia-smi` showed all 8 H200 GPUs at `0MiB`; hugepages remain `512000` total/free and page cache about `742GB`.
  - follow-up page-cache rerun after idempotently setting each NUMA node to `256000` 2MiB hugepages (`512000` total/free, `HugePages_Rsvd=0`): pre-run `Cached=744066140 kB`, `MemAvailable=1036229780 kB`, all GPUs `0MiB`; release build reused existing artifacts (`0.18s`), checkpoint IT passed in `56.11s` test time / `57s` external elapsed, weight load `44.07s`, DeepEP install `3.67s`, DeepEP decode smoke `0.08s`; `iostat -dx 5 nvme2n1` showed `0 r/s` for the captured load windows after the first historical-average line, so this run is page-cache/H2D/loader-side rather than disk-read bound.

### Step 18: Streaming FP8 routed-expert packages

- Added `openinfer-glm52/src/weights/package.rs`, modeled on Kimi's load-time expert package path but specialized to GLM's FP8 E4M3 block-scale experts instead of Kimi's INT4 Marlin layout.
- Each rank now packs every MoE layer's 32 local routed experts into owned expert-major buffers:
  - `W13` weight layout: `[local_experts, 2 * 2048, 6144]` with checkpoint order `[gate; up]`.
  - `W13` scale layout: checkpoint F32 block scale `[local_experts, 32, 48]`.
  - `W2/down` weight layout: `[local_experts, 6144, 2048]`.
  - `W2/down` scale layout: checkpoint F32 block scale `[local_experts, 48, 16]`.
- The loader attempts packaging after each safetensor shard, and deletes a layer's raw routed expert tensors as soon as that layer is packaged. This follows Kimi's memory shape: the future kernel package is resident, but the raw expert map no longer keeps a second copy of all local experts.
- `Glm52RankGpuWeights::validate_non_expert_weight_contract` replaces the old full-raw view check after packaging. Router smoke now asks for `first_moe_router` directly, so it remains valid after routed expert raw tensors are removed.
- This is not a GEMM implementation yet. The package is a substrate for DeepGEMM/FlashInfer/vLLM-style grouped MoE: activation quantization, routed row compaction metadata, scale-layout conversion, SwiGLU, `W2`, routed weight application, and residual integration are still pending.
- Validation:
  - local `cargo fmt -p openinfer-glm52` -> passed.
  - local `cargo check -p openinfer-server` -> passed, proving default Qwen build path still does not compile GLM feature code.
  - remote `jz-38`: `OPENINFER_NCCL_ROOT=/tmp/openinfer-nccl-cu13/nvidia/nccl cargo check -p openinfer-glm52` -> passed.
  - remote `jz-38` release checkpoint IT: release build reused existing artifacts (`0.18s`), test body `54.99s` / external elapsed `57s`, rank worker spawn `4.24s`, weight load plus streaming pack `43.08s`, DeepEP install `3.77s`, DeepEP decode smoke `0.08s`; all 8 reports stayed `router_routes_valid=true`, `router_weights_normalized=true`, and `combined_zero=true`.
  - resident bytes remained `122647325412` per rank after adding the package, because raw routed expert tensors are removed as packages are created. GPUs returned to `0MiB` after test exit; hugepages stayed `512000` total/free and page cache stayed about `744GB`.

### Step 19: MoE activation FP8 quant substrate

- Added GLM-specific raw-pointer wrappers for the two activation quantization steps needed before grouped FP8 expert GEMM:
  - `glm52_fp8_per_token_group_quant_bf16_cuda`: BF16 row-major input -> FP8 E4M3 row-major output plus F32 per-token/per-128 scales. This follows vLLM's `per_token_group_quant_fp8` CUDA/C++ semantics (`amax / fp8_max`, default eps) without importing torch dispatch.
  - `glm52_silu_and_mul_per_token_group_quant_bf16_cuda`: BF16 `[gate; up]` rows -> SiLU(gate) * up -> FP8 E4M3 output plus F32 per-token/per-128 scales. This follows vLLM's `silu_and_mul_per_block_quant` CUDA/C++ kernel shape: one block per `(token, group)`, `group_size=128`, and minimum safe scale.
- Added safe Rust wrappers in `openinfer-kernels/src/ops/glm52/moe_quant.rs`. The wrapper accepts only positive row counts, row-major contiguous buffers, and `group_size=128`; other group sizes fail early instead of drifting into a generic path.
- Extended `Glm52DecodeArena` with persistent worst-case decode buffers:
  - W13 input quant: `deepep_worst_expanded_tokens * 6144` FP8 bytes and `* 48` F32 scales.
  - W13 output staging: `deepep_worst_expanded_tokens * 4096` BF16 values.
  - W2 input quant: `deepep_worst_expanded_tokens * 2048` FP8 bytes and `* 16` F32 scales.
  - The rank resident byte count rose from `122647325412` to `122787367652` (`+140042240` bytes/rank), matching these persistent buffers; this is expected and keeps decode-graph pointer stability.
- Added `SmokeMoeQuantDecode` rank-worker command and startup smoke. Each of the 8 rank workers seeds deterministic BF16 rows, runs both quant kernels for 8 rows, syncs, then checks finite positive scales and nonzero FP8 output. This is a substrate smoke only; the real routed expert GEMM, W2 GEMM, route weighting, and residual integration remain pending.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `git diff --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed, so default Qwen build path stays clean.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed.
  - remote `jz-38` release checkpoint IT: release build `43.62s`, test body `55.06s`, rank worker spawn `3.82s`, weight load plus streaming pack `43.44s`, DeepEP install `3.85s`, DeepEP decode smoke `0.04s`, MoE quant decode smoke `<1ms`; all 8 reports had `hidden_quant_valid=true` and `swiglu_quant_valid=true`.
  - Pre-run machine state after reducing hugepages to 1TiB: all 8 GPUs `0MiB`, `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`, `Cached=744097932 kB`. GPUs returned to `0MiB` after the test.

### Step 20: DeepEP grouped-layout metadata for DeepGEMM

- Investigated the DeepGEMM/vLLM grouped MoE path before writing more kernels:
  - DeepGEMM's public C++ API is still a PyTorch extension/JIT layer, not a raw C ABI; direct Rust integration needs an explicit wrapper around the JIT/runtime boundary rather than a small FFI call.
  - DeepGEMM contiguous grouped FP8 GEMM accepts `use_psum_layout=true`; tests read each expert slice as `start = align(grouped_layout[i - 1])`, `end = grouped_layout[i]`.
  - DeepEP's expand-mode handle has the same contract: `psum_expert[i]` equals aligned previous-expert start plus the current expert's real token count. Empty experts and even empty ranks are legal.
  - vLLM's DeepGEMM MoE path additionally transforms activation/weight scales into the layout DeepGEMM expects; this remains the next wrapper problem after metadata.
- Added `Glm52GroupedLayoutSmokeReport` to the existing DeepEP decode smoke. It D2H-validates the real `Glm52MoeDeepEpState` scratch after dispatch, checking:
  - `psum_rank` is monotonic and within `deepep_worst_recv_tokens`.
  - `psum_expert` can be interpreted with the DeepGEMM slice rule.
  - aligned expanded rows fit the persistent decode arena.
  - zero-route ranks are valid, because real router top-k does not guarantee every EP rank receives rows.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `git diff --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - local `cargo check -p openinfer-glm52` is intentionally not a valid local gate unless `OPENINFER_NCCL_ROOT` points at NCCL >= 2.30.4; the default local failure is the GLM DeepEP build requirement, not a Rust type error.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed.
  - remote `jz-38` release checkpoint IT: release rebuild `2.27s`, test body `53.85s`, rank worker spawn `3.83s`, weight load plus streaming pack `42.18s`, DeepEP install `3.93s`, DeepEP decode smoke `0.04s`, MoE quant decode smoke `<1ms`.
  - The grouped-layout report showed valid non-empty and empty ranks in the real-router smoke, e.g. rank0 `recv_tokens=64`, `active_experts=3`, `expanded_rows=168`, while rank2/rank4 had `recv_tokens=0`, `active_experts=0`, `expanded_rows=0`; all reports had `grouped_layout_valid=true`.
  - Post-run machine state: all 8 GPUs `0MiB`, `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`, `Cached=744197080 kB`, `MemAvailable=1036134264 kB`.
  - Follow-up rerun after confirming the machine still has `512000` free 2MiB hugepages (`1TiB`) and about `744GB` page cache: release build reused existing artifacts (`0.18s`), checkpoint IT passed in `54.68s`, rank worker spawn `4.37s`, weight load plus streaming pack `42.45s`, DeepEP install `3.96s`, DeepEP decode smoke `0.04s`, MoE quant decode smoke `<1ms`; post-run all GPUs were `0MiB`, hugepages remained `512000` total/free, `Cached=744201392 kB`, `MemAvailable=1036226576 kB`. Current warm-cache load baseline is therefore about `42-45s`, not the historical `185-188s` cold/page-cache-limited runs.

### Step 21: DeepGEMM activation scale layout substrate

- Added a GLM-specific DeepGEMM scale-layout kernel:
  - CUDA source: `openinfer-kernels/csrc/glm52/glm52_deepgemm_layout.cu`.
  - Rust FFI/wrapper: `openinfer-kernels/src/ffi/glm52.rs` plus `openinfer-kernels/src/ops/glm52/deepgemm_layout.rs`.
  - The wrapper follows DeepGEMM's `get_mn_major_tma_aligned_tensor` contract for SM90 F32 scales: input row-major `[rows, scale_cols]`, output logical `[rows, scale_cols]` with strides `(1, aligned_rows)`, and `aligned_rows=align(rows, 16 / sizeof(f32))`.
- Extended the persistent decode arena with DeepGEMM-ready W13/W2 activation-scale buffers:
  - `moe_w13_input_scale_tma`: `align(deepep_worst_expanded_tokens, 4) * 48` F32 values.
  - `moe_w2_input_scale_tma`: `align(deepep_worst_expanded_tokens, 4) * 16` F32 values.
  - The per-rank resident byte count rose from `122787367652` to `122789522148` (`+2154496` bytes/rank), matching the two persistent scale-layout buffers.
- MoE quant startup smoke now launches the layout conversion immediately after W13/W2 activation quant and D2H-validates the DeepGEMM layout:
  - W13/W2 row-major F32 scale values must appear at `row + col * aligned_rows`.
  - any padding rows in the aligned layout must be zero.
  - all 8 rank reports expose `hidden_scale_layout_valid` and `swiglu_scale_layout_valid`.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `git diff --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed, proving the default Qwen build still ignores GLM feature code.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed in `41.91s`, compiling the GLM CUDA sources for `sm_90`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed in `42.83s`.
  - remote `jz-38` release checkpoint IT: release rebuild `43.81s`, test body `57.10s`, rank worker spawn `3.79s`, weight load plus streaming pack `45.52s`, DeepEP install `3.44s`, DeepEP decode smoke `0.04s`, MoE quant + DeepGEMM scale-layout smoke `<1ms`.
  - The MoE quant report showed all 8 ranks with `hidden_quant_valid=true`, `swiglu_quant_valid=true`, `hidden_scale_layout_valid=true`, `swiglu_scale_layout_valid=true`, and `scale_layout_aligned_rows=8`.
  - Pre-run machine state: all 8 GPUs `0MiB`, `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`, `Cached=744200976 kB`, `MemAvailable=1036095952 kB`. Post-run state stayed clean: all 8 GPUs `0MiB`, hugepages still `512000` total/free, `Cached=744205768 kB`, `MemAvailable=1036053076 kB`.

### Step 22: W13 package order aligned with DeepGEMM

- While mapping DeepGEMM grouped GEMM's `B=[G,N,K]` contract, found the W13 package implementation was projection-major (`all gate experts` then `all up experts`) even though the design contract required each expert's fused W13 row block to be `[gate; up]`.
- Fixed the loader to build W13 directly from raw per-expert gate/up tensors in DeepGEMM order: `[expert0 gate, expert0 up, expert1 gate, expert1 up, ...]`. W13 scale tensors are packed in the same order.
- Added package length invariants to `Glm52RankExpertFp8Weights::validate`:
  - W13 weight length must be `local_experts * 2 * 2048 * 6144`; W13 scale length must be `local_experts * 2 * 16 * 48`.
  - W2 weight length must be `local_experts * 6144 * 2048`; W2 scale length must be `local_experts * 48 * 16`.
- Existing startup smokes did not catch this ordering bug because no actual expert GEMM consumed W13 yet; they covered residency, raw tensor cleanup, activation quant, scale layout, and grouped-layout metadata.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `git diff --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - local `cargo check -p openinfer-glm52` still requires `OPENINFER_NCCL_ROOT` for NCCL >= 2.30.4; without it, the build guard fails before Rust type checking, as expected for this workstation.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed in `0.49s`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed in `1.06s`.
  - remote `jz-38` release checkpoint IT: release rebuild `2.22s`, test body `55.80s`, rank worker spawn `3.81s`, weight load plus streaming pack `44.62s`, DeepEP install `3.47s`, DeepEP decode smoke `0.04s`, MoE quant + DeepGEMM scale-layout smoke `<1ms`.
  - Pre-run machine state after reducing hugepages to 1TiB: all 8 GPUs `0MiB`, `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`, `Cached=744210512 kB`, `MemAvailable=1036196056 kB`.
  - Post-run state stayed clean: all 8 GPUs `0MiB`, hugepages still `512000` total/free, `Cached=744213048 kB`, `MemAvailable=1036129312 kB`. Warm-cache load remains about `44-46s`; the historical `185-188s` load time was page-cache/disk limited.

### Step 23: W2 output buffer split for MoE forward

- Aligned the GLM decode arena with Kimi's MoE buffer lifecycle before wiring GEMM:
  - `deepep_recv_x`: DeepEP-dispatched hidden rows.
  - `moe_w13_output_bf16`: W13 `[gate; up]` output staging.
  - `moe_w2_input_fp8/scale`: SiLU(gate) * up activation quant output.
  - `moe_w2_output_bf16`: future W2/down GEMM output, now the source for DeepEP combine.
  - `deepep_combined`: rank-local post-combine output.
- The previous DeepEP smoke combined from `deepep_recv_x`, which was acceptable for an all-zero smoke but not the forward shape. The smoke now combines from `moe_w2_output_bf16`, so adding W2 GEMM later can fill the existing buffer without changing the combine call site.
- Added `moe_w2_output_bf16` as a persistent worst-case decode buffer sized `deepep_worst_expanded_tokens * hidden`, preserving future decode CUDA Graph pointer stability. The per-rank resident byte increase is exactly `8416 * 6144 * sizeof(bf16) = 103415808` bytes.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `git diff --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed in `0.49s`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed in `1.07s`.
  - remote `jz-38` release checkpoint IT: release rebuild `2.21s`, test body `56.81s`, rank worker spawn `3.83s`, weight load plus streaming pack plus arena allocation `44.85s`, DeepEP install `3.75s`, DeepEP decode smoke `0.04s`, MoE quant + DeepGEMM scale-layout smoke `<1ms`.
  - Resident bytes per rank rose from `122789522148` to `122892937956`, matching the new `103415808`-byte W2 output buffer. Post-run machine state stayed clean: all 8 GPUs `0MiB`, `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`, `Cached=744215348 kB`, `MemAvailable=1036118600 kB`.

### Step 24: Decode-only scope and MoE layout contract

- Updated the first runnable GLM scope to decode-only for future P/D split:
  - the GLM engine should only enter forward when KV/page state has already been materialized by another component.
  - prompt prefill is out of scope for this engine phase; no hidden prefill fallback should be added.
  - unified mixed prefill+decode batches remain out of scope.
- Added the MoE decode layout contract:
  - DeepEP runs with fixed decode bucket capacity and every rank participates every step, including empty ranks.
  - `deepep_recv_x` holds BF16 expanded hidden rows; `psum_expert` is the device grouped-layout tensor for expert slices; padding gaps are not valid tokens.
  - W13/W2 buffers stay persistent at worst-case capacity so decode graph capture can keep pointer stability.
  - W2 output must be route-weighted before combine unless a later GEMM wrapper proves an equivalent fused epilogue.
- Re-explored DeepGEMM and vLLM MoE sources for H200 route selection:
  - vLLM's FP8 E4M3 expert path uses `m_grouped_fp8_gemm_nt_contiguous`.
  - vLLM/DeepGEMM FP8xFP4/MXFP paths are Blackwell-family paths, not the first GLM H200 path.
  - DeepGEMM MegaMoE PR-304 fuses dispatch, linear1, SwiGLU, linear2, and combine, but `csrc/apis/mega.hpp` dispatches only when `arch_major == 10`; for H200 it is a design reference, not an implementation target.
- Sources read for this step:
  - `openinfer-kernels/third_party/DeepGEMM/csrc/apis/mega.hpp`
  - `openinfer-kernels/third_party/DeepGEMM/csrc/apis/layout.hpp`
  - `openinfer-kernels/third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`
  - `../vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`
  - KernelWiki pages `sources/prs/DeepGEMM/PR-304.md`, `wiki/kernels/deepgemm.md`, `wiki/kernels/grouped-gemm.md`, and `sources/prs/flashinfer/PR-1819.md`
- No code was changed in this step.

### Step 25: Typed MoE psum layout and actual recv-row quant smoke

- Added a typed MoE layout boundary in `openinfer-glm52/src/moe_deepep.rs`:
  - `Glm52MoePsumLayout` is the forward-side device view over DeepEP `psum_rank`/`psum_expert`.
  - `Glm52MoePsumLayoutSnapshot` is the startup/IT-only D2H validation path.
  - `Glm52MoePsumLayoutReport` records `recv_tokens`, `active_experts`, `expanded_rows`, `empty_rank`, and the grouped-layout validity bit.
- Changed the real-router DeepEP smoke to quantize actual DeepEP recv rows:
  - router uses checkpoint BF16 gate weights and real hidden rows.
  - DeepEP dispatch fills `deepep_recv_x` and device `psum_expert`.
  - non-empty ranks run W13 input BF16 -> FP8 quant over `expanded_rows`; empty ranks return an explicit `quant_ran=false` report instead of pretending to launch a 0-row kernel.
  - W2 input quant still uses deterministic seeded W13 output until the W13 grouped GEMM wrapper lands.
- Kept the separate synthetic MoE quant smoke for the all-rank non-empty path.
- Updated request-time rejection text to name the decode-only pending runtime and prefilled-KV handoff rather than prefill execution.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `git diff --check` for touched GLM/doc files -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed in `0.62s`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed in `1.33s`.
  - first remote release checkpoint IT exposed a gate bug, not a kernel bug: rank0 actual recv quant returned `rows=168`, all quant/layout booleans true, but the runner check had `quant_ran` polarity inverted, so the test panicked and DeepEP teardown printed barrier noise. The gate was fixed.
  - second remote release checkpoint IT passed: release rebuild `2.23s`, test body `54.38s`, rank worker spawn `3.78s`, weight load `43.53s`, DeepEP install `3.67s`, DeepEP decode smoke `0.04s`, MoE quant smoke `<1ms`.
  - Real-router DeepEP layout in the passing IT included empty ranks: rank2 and rank4 had `recv_tokens=0`, `expanded_rows=0`, `empty_rank=true`, `quant_ran=false`; non-empty ranks quantized actual expanded rows (`168`, `80`, `24`, `168`, `8`, `64`) with W13/W2 quant and DeepGEMM scale-layout checks all true.
  - Post-run machine state stayed clean: all 8 GPUs `0MiB`; `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`; `Cached=749682032 kB`, `MemAvailable=1036416564 kB`.

### Step 26: Decode-only DeepEP API boundary and DeepGEMM wrapper exploration

- Narrowed the GLM engine API to match the P/D split decision:
  - `openinfer-glm52/src/deepep.rs` now records only the decode DeepEP contract.
  - startup validation/logging no longer computes or reports prefill capacity.
  - `Glm52DeepEpEnableReport` records only rank count and decode cap.
  - `openinfer-kernels/src/ops/glm52/deepep.rs` no longer exposes `new_prefill`, prefill dispatch, prefill wait-count, prefill recv, or prefill combine wrappers.
- The copied C shim still contains prefill symbols and info fields from the DeepEP source shape, but GLM Rust code has no safe path to call them. The first runnable engine should accept only decode work whose KV/page state already exists.
- Re-read DeepGEMM and vLLM grouped-MoE sources for the next wrapper:
  - DeepGEMM `m_grouped_fp8_fp4_gemm_nt_contiguous` aliases to the public `m_grouped_fp8_gemm_nt_contiguous` name for FP8 E4M3, and on H200 dispatches SM90 `1d2d` when activation scales are F32.
  - `use_psum_layout=true` is supported and passes `GemmType::MGroupedContiguousWithPsumLayout`; that matches the DeepEP `psum_expert` layout already validated by startup smoke.
  - vLLM wraps grouped GEMM in `mk_alignment_scope(align_used)` because DeepGEMM's scheduler can otherwise read the wrong expert id under CUDA Graph replay. GLM should make this a fixed arena/graph contract, not a per-step dynamic setting.
  - DeepGEMM's public C++ API is still torch/pybind-oriented: it uses torch tensor shape/stride checks, torch current stream, torch CUDA workspace tensors, and JIT compiler/runtime init. A production OpenInfer wrapper needs a raw-pointer JIT/runtime boundary or a different CUDA/C++ source; directly linking libtorch into `openinfer-kernels` would violate the current engine shape.
  - MegaMoE remains out of first H200 scope: `csrc/apis/mega.hpp` dispatches only for `arch_major == 10`.
- Validation so far:
  - local `cargo fmt --check` -> passed before the docs edit.
  - local `cargo check -p openinfer-server` -> passed.
  - local `git diff --check` on touched GLM/doc files -> passed.
  - local `cargo check -p openinfer-glm52` cannot run without a GLM NCCL root; it fails in build.rs requesting `OPENINFER_NCCL_ROOT`, which is expected for the GLM DeepEP feature. Node38 validation is the feature gate.
  - remote `jz-38`: `cargo fmt --check` -> passed.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed in `3.45s`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed in `2.47s`.
  - remote `jz-38` release checkpoint IT passed: release rebuild `3.21s`, test body `54.99s`, rank worker spawn `3.82s`, weight load `43.35s`, DeepEP install `3.46s`, DeepEP decode smoke `0.04s`, MoE quant smoke `<1ms`.
  - The passing startup log now reports only `deepep_decode_recv=1024` and `deepep_decode_expanded=8416`; no GLM startup prefill capacity is reported.
  - Post-run machine state stayed clean: all 8 GPUs `0MiB`; `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`; `Cached=767845156 kB`, `MemAvailable=1036111032 kB`.

### Step 27: Decode-substrate CUDA Graph smoke

- Added a fixed-bucket graph smoke for the decode substrate:
  - `Glm52MoeDeepEpState::decode_graph_smoke_roundtrip` seeds the persistent decode arena, captures one 128-token CUDA Graph, launches the captured graph once, then replays it once.
  - The captured sequence is host-quiet: checkpoint router, DeepEP decode dispatch, W13 input FP8 quant, W13 scale-layout conversion, W2 SiLU*up FP8 quant, W2 scale-layout conversion, and DeepEP decode combine.
  - The smoke uses the same fixed `GLM52_DEEPEP_DECODE_BATCH_CAP=128` bucket and `deepep_worst_expanded_tokens=8416` arena rows that the decode path will use. Partial decode buckets still need an eager path or separate graph policy later.
  - This is a substrate graph smoke, not a full decode forward graph: routed expert GEMM, route weighting before combine, attention/KV decode, residual path, logits, and sampling are not in the captured sequence yet.
- Kept the engine boundary decode-only:
  - no prompt prefill work is added to the GLM engine.
  - future P/D split is expected to hand off prefilled KV/page state; decode receives work that is already ready for decode.
  - DeepEP remains the only GLM EP communication path; no NCCL all-to-all backend is introduced.
- Validation:
  - local `cargo fmt --check` -> passed after formatting.
  - local `cargo check -p openinfer-server` -> passed in `2.30s`.
  - local `git diff --check` for touched GLM source files -> passed.
  - remote `jz-38`: `cargo fmt --check` -> passed.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed in `0.87s`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed in `2.22s`.
  - remote `jz-38` release checkpoint IT passed: release rebuild `2.44s`, test body `56.82s`, rank worker spawn `3.81s`, weight load `45.32s`, DeepEP install `3.81s`, DeepEP decode smoke `0.04s`, MoE quant smoke `<1ms`, decode CUDA Graph smoke `0.36s`.
  - Graph smoke reports were valid on all 8 ranks: `num_tokens=128`, `fixed_bucket_tokens=128`, `worst_expanded_rows=8416`, router routes normalized, grouped layout valid, combined output zero, capture+first launch OK, replay OK.
  - Post-run machine state stayed clean: all 8 GPUs `0MiB`; `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`; `Cached=767852180 kB`, `MemAvailable=1036049476 kB`.

### Step 28: vLLM/Kimi route-weight contract mapping

- Stopped the standalone route-weight-kernel direction and re-read the source contracts first:
  - vLLM DeepGEMM FP8 path (`deep_gemm_moe.py` + `deep_gemm_utils.py`) permutes inputs, runs W13/W2 grouped FP8 GEMMs under `mk_alignment_scope(align_used)`, then applies router weights in Triton `deepgemm_unpermute_and_reduce` by gathering each top-k output and accumulating `expert_output * topk_weight`.
  - vLLM DeepEP low-latency finalize (`deepep_ll.py`) says weight application and reduction happen in `low_latency_combine`; it passes `combine_topk_weights` to DeepEP instead of launching a separate multiply.
  - vLLM DeepEP v2/HT (`deepep_v2.py`, `deepep_ht.py`) first applies a `TopKWeightAndReduce` implementation to contiguous BF16 expert outputs, then calls DeepEP combine with `topk_weights=None`. Decode v2 is graph-oriented through `do_expand=False` and `do_cpu_sync=False`.
  - DeepEP elastic combine rejects `topk_weights` when `use_expanded_layout=true`; its host checks and kernel assert both encode that expanded-mode reduction should already be weighted before combine.
  - Kimi's current DeepEP path is the closest OpenInfer precedent for expanded layout: DeepEP expanded dispatch produces expert-major rows, W2/Marlin consumes `recv_topk_weight` and multiplies inside the W2 expert kernel, then DeepEP combine reduces already weighted BF16 outputs.
  - DeepGEMM has a source-backed candidate in `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py`, which multiplies `topk_weight` during SiLU*up FP8 activation quant before W2. This is TileLang/Python source, so using it in OpenInfer still needs a deliberate wrapper/codegen plan rather than an ad hoc CUDA multiply.
- Consequence for GLM:
  - The current GLM substrate is DeepEP elastic expanded `psum_expert`; it cannot copy vLLM EPLL combine weighting without changing the DeepEP layout/API.
  - The next MoE implementation should copy one source-backed contract: either Kimi-style expert-kernel weighting, DeepGEMM TileLang activation-quant weighting, or a CUDA/C++ rewrite of vLLM's gather/reduce semantics after proving no existing source path fits.
  - Do not add a standalone local route-weight kernel as the next step. If a local CUDA rewrite becomes unavoidable because the reusable source is Triton/TileLang-only, it needs jiuzhang H200 NCU evidence before it can be called ready.
- No code changed in this step.

### Step 29: DeepGEMM expert package plan

- Encoded the DeepGEMM grouped FP8 weight package view in `openinfer-glm52/src/weights/package.rs` without adding a new kernel:
  - `Glm52DeepGemmMGroupedFp8WeightPlan` records the source-backed `m_grouped_fp8_gemm_nt_contiguous` contract as `[G,N,K]` FP8 weights plus F32 block scales `[G,N/128,K/128]`.
  - W13 package validation now proves each rank's package is `G=32`, `N=4096`, `K=6144`, scales `[32,32,48]`.
  - W2/down package validation now proves each rank's package is `G=32`, `N=6144`, `K=2048`, scales `[32,48,16]`.
- This deliberately stops at the contract boundary. It does not link DeepGEMM JIT, does not add a raw GEMM wrapper, and does not choose a route-weight implementation.
- Validation:
  - local `cargo fmt --check` passed.
  - local `cargo check -p openinfer-server` passed; default Qwen build path remains isolated from GLM feature code.
  - local `git diff --check -- openinfer-glm52/src/weights/package.rs` passed.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root passed in `0.86s`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root passed in `2.19s`.
  - remote `jz-38` release checkpoint IT passed against `/data/models/GLM-5.2-0614-Provider-FP8`: release rebuild `2.48s`, test body `57.12s`, rank worker spawn `4.27s`, weight load plus streaming pack and DeepGEMM package-plan validation `45.62s`, DeepEP install `3.34s`, DeepEP decode smoke `0.04s`, MoE quant smoke `<1ms`, decode CUDA Graph smoke `0.36s`.
  - The release IT retained `122892937956` bytes per rank and ran all 8-rank smokes successfully; post-run machine state was clean: all GPUs `0MiB`, `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`, `Cached=767860476 kB`, `MemAvailable=1035955688 kB`.

### Step 30: Route-weight contract audit

- Re-checked the current GLM call surface before choosing a route-weight implementation:
  - `openinfer-glm52/src/moe_deepep.rs::launch_moe_quant_substrate` still calls `glm52_silu_and_mul_per_token_group_quant_bf16_launch`, so the validated Step 29 smoke path is unweighted W2-input activation quant.
  - At audit time, `openinfer-kernels` had an unvalidated `glm52_silu_and_mul_weighted_per_token_group_quant_bf16_*` symbol and Rust wrapper. It was not wired into GLM runtime or startup smokes, had no NCU profile, and must not be treated as the selected route-weight contract. Step 31 removes it.
  - Local default `cargo check -p openinfer-server` passed in `52.80s`, proving this half-added GLM symbol does not leak into the default Qwen/server build surface. GLM feature validation was not rerun in this audit.
- Re-read the closest source contracts:
  - Kimi TP1/DP8 DeepEP is the closest layout match to GLM's current substrate: expanded DeepEP dispatch produces expert-major rows, W2/Marlin consumes `recv_topk_weight` with `mul_topk_weights=true`, and DeepEP combine only reduces already weighted BF16 outputs.
  - vLLM DeepGEMM FP8 applies route weights in `deepgemm_unpermute_and_reduce` after W2 grouped GEMM. That path assumes a permute/unpermute index contract; copying it would require GLM to add an equivalent gather/reduce layout rather than only toggling activation quant.
  - vLLM DeepEP v2 decode graph uses `do_expand=false` and does contiguous `TopKWeightAndReduce` before combine. This is not the same layout as GLM's current expanded `psum_expert` substrate.
  - vLLM DeepEP low-latency combine takes `topk_weights`, but DeepEP elastic expanded combine rejects weights when `use_expanded_layout=true`; GLM cannot copy that API without changing the DeepEP layout.
  - DeepGEMM TileLang `swiglu_apply_weight_to_fp8.py` is a source-backed candidate for weighting during SiLU*up FP8 quant, but it is TileLang/Python source. A local CUDA adaptation needs an explicit source note, jiuzhang H200 NCU evidence, and correctness evidence before being called ready.
- Consequence:
  - For the current expanded DeepEP layout, the default next implementation should follow Kimi's contract: W2/GEMM-side weighting or an equivalent fused expert path writes already weighted BF16 rows into `moe_w2_output_bf16`, then DeepEP combine performs reduction only.
  - The half-added weighted SiLU quant path should either be removed before the next implementation step, or promoted deliberately as the DeepGEMM TileLang-source candidate with profile and correctness evidence. It should not become a silent fallback; Step 31 chose removal.
- No implementation code was changed in this audit. The only output is this project-doc update.

### Step 31: Remove unvalidated weighted activation-quant path

- Removed the half-added weighted SiLU activation-quant surface from `openinfer-kernels`:
  - Deleted the `route_weights` branch from `silu_and_mul_per_token_group_quant_bf16_k128_kernel`.
  - Deleted C ABI export `glm52_silu_and_mul_weighted_per_token_group_quant_bf16_cuda`.
  - Deleted Rust FFI and safe wrapper `glm52_silu_and_mul_weighted_per_token_group_quant_bf16_launch`.
  - Deleted the wrapper-only `validate_silu_quant_buffers` helper.
- The validated GLM substrate remains unchanged at the call site:
  - `launch_moe_quant_substrate` still runs W13 input BF16 -> FP8 and unweighted SiLU*up -> W2 input FP8.
  - `moe_w2_output_bf16` remains the future source for DeepEP combine.
  - Route weighting is still pending and should be implemented through the source-backed expanded-layout contract, preferably Kimi-style W2/GEMM-side weighting unless the DeepGEMM wrapper proves a better equivalent fused contract.
- This is a cleanup step, not a performance or correctness claim. It intentionally removes an unprofiled local CUDA variant rather than expanding the implementation surface.
- Validation:
  - local `cargo fmt` -> passed.
  - local `rg` over GLM kernel/FFI/ops/runtime code finds no remaining `glm52_silu_and_mul_weighted`, `route_weights`, or `validate_silu_quant_buffers` references outside this document.
  - local `git diff --check -- openinfer-kernels/csrc/glm52/glm52_moe_quant.cu openinfer-kernels/src/ffi/glm52.rs openinfer-kernels/src/ops/glm52/moe_quant.rs docs/models/glm52/support.md` -> passed.
  - local `cargo check -p openinfer-server` -> passed in `52.87s`; default Qwen/server build remains isolated from GLM feature code.
  - synced the touched GLM kernel/FFI/ops/doc files to `jz-38:/root/develop/xingming/pegainfer-glm52`.
  - remote `jz-38`: `cargo fmt --check` -> passed.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the NCCL root -> passed in `40.87s`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the NCCL root -> passed in `41.41s`.
  - remote `jz-38` release checkpoint IT passed against `/data/models/GLM-5.2-0614-Provider-FP8`: release rebuild `43.52s`, test body `56.95s`, rank worker spawn `4.29s`, weight load `45.02s`, DeepEP install `3.28s`, DeepEP decode smoke `0.04s`, MoE quant smoke `<1ms`, decode CUDA Graph smoke `0.36s`.
  - Post-run machine state stayed clean: all 8 GPUs `0MiB`, `HugePages_Total=512000`, `HugePages_Free=512000`, `HugePages_Rsvd=0`, `Cached=767855028 kB`, `MemAvailable=1035956104 kB`.

### Step 32: vLLM kernel reference doc

- Added `docs/models/glm52/vllm-kernel-reference.md` so vLLM source lookups are no longer rediscovered ad hoc during GLM52 implementation.
- The reference doc classifies GLM52-relevant vLLM paths into direct CUDA/C++ copy candidates, contract-only paths, Triton/Python rewrites, and deferred hardware-mismatched paths:
  - direct-copy candidates include GLM-5 router GEMM, grouped noaux top-k, MoE unpermute/reduce, indexer cache helpers, sparse-indexer top-k, and FP8 quantization helpers.
  - DeepGEMM MoE and DeepEP LL/v2 are contract-first because much of their route layout/weighting glue is Python, Triton, or third-party API composition.
  - attention/indexer decode should follow vLLM's non-MTP `(B,1)` metadata and sparse-indexer decode sequence, while prefill chunking stays future P-worker work.
- Updated `docs/index.md` with a GLM52 routing row for the new reference doc.
- Validation:
  - local `git diff --check -- docs/models/glm52/support.md docs/models/glm52/vllm-kernel-reference.md docs/index.md` -> passed.
  - local scan found no stale-status enum field or banned communication terms in the doc/index edits.

### Step 33: Source-backed weighted W2-input SwiGLU quant

- Reintroduced route weighting deliberately as a source-backed W2-input activation-quant contract, not as a standalone post-W2 multiply:
  - CUDA/ABI source: `openinfer-kernels/csrc/glm52/glm52_moe_quant.cu`.
  - Rust FFI/wrapper: `openinfer-kernels/src/ffi/glm52.rs` and `openinfer-kernels/src/ops/glm52/moe_quant.rs`.
  - Runtime use: `openinfer-glm52/src/moe_deepep.rs`.
  - Validation helper/report plumbing: `openinfer-glm52/src/arena.rs` and `openinfer-glm52/src/runner.rs`.
- Source contract:
  - DeepGEMM `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` computes `y = silu(gate) * up * topk_weight`, then emits FP8 plus per-token/per-channel scales.
  - GLM maps this to W2-input quant over expanded DeepEP recv rows: `recv_topk_weight[row]` is consumed while quantizing `moe_w13_output_bf16 -> moe_w2_input_fp8`.
  - Standalone `decode_moe_quant_smoke` remains unweighted; only routed DeepEP recv rows and the decode graph smoke use `recv_topk_weight`.
- Correctness checks added to startup:
  - `route_weights_applied=true` is required for non-empty routed DeepEP recv quant reports.
  - `swiglu_weighted_scale_valid=true` recomputes the first positive weighted row on CPU and checks the produced W2 input scale against `silu(gate) * up * topk_weight`.
  - The decode graph replay validates the same weighted scale after capture/replay.
- Validation:
  - local `cargo fmt` and `cargo fmt --check` -> passed.
  - local `git diff --check -- openinfer-kernels/csrc/glm52/glm52_moe_quant.cu openinfer-kernels/src/ffi/glm52.rs openinfer-kernels/src/ops/glm52/moe_quant.rs openinfer-glm52/src/moe_deepep.rs openinfer-glm52/src/arena.rs openinfer-glm52/src/runner.rs` -> passed.
  - local `cargo check -p openinfer-glm52` is still not a valid local GLM feature gate without the node38 NCCL root; it fails in `openinfer-kernels/build.rs` requesting NCCL >= 2.30.4, before Rust type-checking.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the node38 NCCL root -> passed in `41.44s`.
  - remote `jz-38` release checkpoint IT passed against `/data/models/GLM-5.2-0614-Provider-FP8`: release compile `43.51s`, test body `55.15s`, rank weight load `43.27s`, DeepEP install `3.54s`, DeepEP decode smoke `0.04s`, decode CUDA Graph smoke `0.36s`.
  - The routed DeepEP smoke reported `route_weights_applied=true` and `swiglu_weighted_scale_valid=true` for non-empty ranks; empty ranks remain valid with `rows=0`.
  - The standalone MoE quant smoke kept `route_weights_applied=false`, proving the weighted path is not a hidden default for synthetic rows.
  - The decode graph smoke reported all 8 ranks with `worst_expanded_rows=8416`, `router_routes_valid=true`, `router_weights_normalized=true`, `route_weights_applied=true`, `swiglu_weighted_scale_valid=true`, `grouped_layout_valid=true`, `capture_and_first_launch_ok=true`, and `replay_ok=true`.
- NCU evidence:
  - Full checkpoint IT under NCU hit DeepEP NVLink barrier timeouts because profiler replay delayed collective progress; the normal release checkpoint IT immediately before it passed, so this was treated as profiler interference rather than a code regression.
  - Added a profile-only harness under `profile/glm52_weighted_swiglu_quant_20260626/harness/weighted_swiglu_quant_bench.cu` to isolate the weighted SwiGLU quant kernel from DeepEP.
  - Shape: `rows=8416`, `hidden=2048`, `group_size=128`, grid `(8416,16,1)`, block `(128,1,1)`.
  - CUDA event timing: `0.097880 ms` over 500 iterations after 50 warmups.
  - NCU full report: `profile/glm52_weighted_swiglu_quant_20260626/reports/weighted_swiglu_full.ncu-rep`.
  - NCU full replay metrics: duration `122.43us`, SM throughput `82.19%`, L1/TEX throughput `75.56%`, L2 throughput `21.76%`, DRAM throughput `13.51%`, memory throughput `663.57 GB/s`, achieved occupancy `88.05%`, registers/thread `19`, issue slots busy `83.53%`.
  - NCU SourceCounters report: `profile/glm52_weighted_swiglu_quant_20260626/reports/weighted_swiglu_source.ncu-rep`; branch efficiency `92.62%`.
  - The profile conclusion is diagnostic only: this kernel is mainly SM/L1/TEX pressured by SiLU, FP8 conversion, and group reduction, not HBM-bound. No end-to-end optimization win is claimed until W13/W2 GEMM and full decode are running.

### Step 34: vLLM MoE/FP8 kernel map

- Added `docs/models/glm52/vllm-moe-fp8-kernels.md` so the GLM52 routed-expert GEMM discussion has one source map for vLLM's FP8 MoE backend choices.
- Source snapshots recorded:
  - `../vllm` at `4d3b4b9b01efbca77872e3d4a568b273c7a245a7`.
  - FlashInfer submodule at `d768c14e7cf5dd5df45a8a1de78ae815879f108a`.
  - DeepGEMM submodule at `54e22612409371d6364144b69086735beb54e98b`.
- Findings:
  - vLLM's Hopper `SM90` + block-FP8 + EP policy moves `FLASHINFER_CUTLASS` to the front, so FlashInfer is present. If GLM52 does not use it immediately, the recorded blockers are ABI, layout, graph workspace ownership, and missing OpenInfer H200 A/B evidence.
  - DeepGEMM standard grouped FP8 is still the best near-term layout oracle for current GLM expanded DeepEP `psum_expert` work, but vLLM reaches it through Python/JIT bindings, not a ready Rust/C ABI.
  - vLLM CUTLASS 3.x grouped GEMM is the closest self-contained CUDA/C++ source candidate, but its public wrapper uses `torch::stable::Tensor` and per-call pointer/workspace allocation; OpenInfer would need fixed arena pointer arrays and a raw C ABI before decode graph capture.
  - Immediate order remains DeepGEMM grouped FP8 wrapper first, vLLM CUTLASS raw port as fallback, FlashInfer CUTLASS once the weight transform/TVM FFI/raw wrapper/graph workspace boundary is mapped.
- Validation:
  - local `git diff --check -- docs/models/glm52/vllm-moe-fp8-kernels.md docs/models/glm52/support.md docs/models/glm52/vllm-kernel-reference.md docs/index.md` -> passed.
  - local source sanity check matched the recorded revisions for `../vllm`, FlashInfer, and DeepGEMM, and key vLLM/FlashInfer MoE FP8 files existed at the documented paths.
  - local scan found no stale-status enum field in the new doc.

### Step 35: MoE GEMM contract and routed-layer order

- Added `openinfer-glm52/src/moe_gemm.rs` as the first real routed-expert GEMM boundary, but deliberately stopped at a contract check rather than pretending the GEMM wrapper exists.
- The contract validates the operand shapes that the first DeepGEMM grouped FP8 wrapper must consume:
  - W13: `G=32`, `M_capacity=8416`, `N=4096`, `K=6144`, activation scale columns `48`, TMA-aligned scale rows `8416`.
  - W2: `G=32`, `M_capacity=8416`, `N=6144`, `K=2048`, activation scale columns `16`, TMA-aligned scale rows `8416`.
  - Group metadata: DeepEP expand-mode `psum_expert` has 32 local-expert entries and expert alignment 8; the arena pointers are persistent and graph-stable.
- Tightened the routed expert package invariant:
  - `load_sliced_rank_weights_to_gpu` sorts streamed expert packages by `layer_idx` after shard loading, because shard completion order is not model-layer order.
  - `Glm52RankExpertFp8Weights::validate` now requires exactly 75 routed layers in order `3..=77`; a package ordered by shard stream completion fails before any GEMM wrapper can read it.
- Startup now sends `ValidateMoeGemmContract` to all 8 rank workers after the DeepEP and activation-quant smokes, and fails closed unless every rank reports the same sorted layer range and W13/W2 contract.
- Source decision:
  - DeepGEMM SM90 m-grouped contiguous FP8 is still the target for the first H200 route, and its public shape contract is the source of the `[G,N,K]` / grouped-`M` boundary.
  - vLLM is still useful as an implementation donor, but its DeepGEMM path is Python/JIT composition and its CUTLASS grouped path still needs an arena-backed raw ABI before it is graph-safe here.
  - No performance win is claimed by this step; it is a launch-time invariant and wrapper precondition.
- Validation:
  - local `cargo fmt` -> passed.
  - local `git diff --check -- docs/models/glm52/support.md docs/index.md openinfer-glm52/src/moe_gemm.rs openinfer-glm52/src/lib.rs openinfer-glm52/src/runner.rs openinfer-glm52/src/weights.rs openinfer-glm52/src/weights/load.rs openinfer-glm52/src/weights/package.rs` -> passed.
  - local `cargo check -p openinfer-server` -> passed in `0.49s`.
  - local `cargo check -p openinfer-glm52` remains expected to fail before Rust type-checking without the node38 NCCL root; `openinfer-kernels/build.rs` asks for NCCL >= 2.30.4.
  - remote `jz-38`: `cargo check -p openinfer-glm52` with the node38 NCCL root -> passed in `0.86s` after refreshing the changed source mtimes.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed in `1.47s`.
  - remote `jz-38` release checkpoint IT passed against `/data/models/GLM-5.2-0614-Provider-FP8`: release compile `2.67s`, test body `57.10s`, rank weight load `44.74s`, DeepEP install `3.62s`, DeepEP decode smoke `0.04s`, MoE GEMM contract validation `0.00s`, decode CUDA Graph smoke `0.36s`.
  - The contract report was correct on all 8 ranks: `layer_count=75`, `first_layer_idx=3`, `last_layer_idx=77`, W13 `G=32,N=4096,K=6144`, W2 `G=32,N=6144,K=2048`, and `graph_stable_arena=true`.
- Operational note:
  - One remote run still reported stale order `first_layer_idx=10,last_layer_idx=9` after `rsync -avR`; the remote source had the new content, but the source mtimes were older than the release artifact. Touching the changed GLM files forced Cargo to recompile `openinfer-glm52`, and the next release IT passed. For future remote syncs, verify either rebuild output or changed-file mtimes when a report contradicts visible source.

### Step 36: GLM kernel module owner and DeepGEMM ABI boundary

- Collapsed GLM kernel wrappers into one feature-owned module instead of keeping many top-level `glm52_*` modules:
  - `openinfer-kernels/src/ops.rs` now has one `#[cfg(feature = "glm52")] mod glm52;`.
  - `openinfer-kernels/src/ops/glm52.rs` owns submodules `deepep`, `deepgemm_grouped`, `deepgemm_layout`, `moe_quant`, and `router`, then re-exports the public functions/types so existing callers keep using `openinfer_kernels::ops::*`.
  - `openinfer-kernels/src/ffi.rs` now has one `#[cfg(feature = "glm52")] mod glm52;`; DeepEP raw symbols moved under `openinfer-kernels/src/ffi/glm52/deepep.rs`, while the other GLM raw symbols stay in `openinfer-kernels/src/ffi/glm52.rs`.
- Added the first DeepGEMM grouped FP8 ABI file without pretending GEMM compute is wired:
  - CUDA source: `openinfer-kernels/csrc/glm52/glm52_deepgemm_grouped.cu`.
  - Rust wrapper: `openinfer-kernels/src/ops/glm52/deepgemm_grouped.rs`.
  - `glm52_deepgemm_grouped_fp8_contract_cuda` validates the fixed GLM H200 W13/W2 grouped-FP8 contract from the kernel crate side.
  - `glm52_deepgemm_grouped_fp8_launch` is the safe Rust launch boundary: it validates activation, activation-scale, weight, weight-scale, `psum_expert`, and output buffer sizes before entering C.
  - `glm52_deepgemm_grouped_fp8_launch_cuda` currently returns `CUDA_ERROR_NOT_SUPPORTED`; real launch will require the raw DeepGEMM JIT/runtime split, tensor-map construction over raw pointers, explicit `cudaStream_t`, and startup warmup before graph capture.
- Routed expert startup validation now goes through both layers:
  - `openinfer-glm52/src/moe_gemm.rs` still checks sorted 75-layer packages and arena capacity.
  - It then calls `openinfer_kernels::ops::glm52_deepgemm_grouped_fp8_contract_validate` for W13 and W2, proving the GLM feature build includes the C ABI symbol and that model-side assumptions match the kernel crate boundary.
- Validation so far:
  - local `cargo fmt` -> passed.
  - local `git diff --check -- openinfer-kernels/src/ops.rs openinfer-kernels/src/ops/glm52.rs openinfer-kernels/src/ops/glm52 openinfer-kernels/src/ffi.rs openinfer-kernels/src/ffi/glm52.rs openinfer-kernels/src/ffi/glm52 openinfer-glm52/src/moe_gemm.rs openinfer-kernels/csrc/glm52/glm52_deepgemm_grouped.cu` -> passed.
  - local `cargo check -p openinfer-server` -> passed in `53.52s` after rebuilding `openinfer-kernels`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed in `42.86s` after rebuilding `openinfer-kernels`.
  - remote `jz-38` release checkpoint IT passed against `/data/models/GLM-5.2-0614-Provider-FP8`: release rebuild `43.82s`, test body `54.47s`, rank worker spawn `3.84s`, weight load `43.02s`, DeepEP install `3.64s`, DeepEP decode smoke `0.05s`, MoE GEMM contract validation `0.00s`, decode CUDA Graph smoke `0.37s`.
  - The release IT contract report still had all 8 ranks at `layer_count=75`, `first_layer_idx=3`, `last_layer_idx=77`, W13 `G=32,N=4096,K=6144`, W2 `G=32,N=6144,K=2048`, and `graph_stable_arena=true`, proving the new kernel C ABI contract agrees with the model-side report.
  - After adding the safe launch boundary, local `cargo fmt`, local `git diff --check -- openinfer-kernels/...`, and local `cargo check -p openinfer-server` passed; the local check took `0.50s`.
  - The updated module files were synced to node38 with `rsync -avR`; `find openinfer-kernels/src/{ops,ffi} -maxdepth 3 -type f | sort | grep glm52` confirmed only `ops/glm52.rs`, `ops/glm52/*`, `ffi/glm52.rs`, and `ffi/glm52/*` remain for GLM Rust modules.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the node38 NCCL root passed again after rebuilding `openinfer-kernels`; the run took `41.85s`.
- Sync note:
  - A remote `rsync` command without `-R` briefly copied `ffi.rs`, `glm52.rs`, `ops.rs`, `moe_gemm.rs`, and `glm52_deepgemm_grouped.cu` into the remote repo root. Those root files were immediately removed, the files were resent with `-R`, old remote `ops/glm52_*` and `ffi/glm52_deepep.rs` paths were removed, and `find openinfer-kernels/src/{ops,ffi}` confirmed only the new GLM module layout remains.

### Step 37: DeepGEMM JIT integration plan

- Re-read the DeepGEMM grouped-FP8 host path instead of guessing at the JIT boundary:
  - `openinfer-kernels/third_party/DeepGEMM/csrc/apis/gemm.hpp` exposes `m_grouped_fp8_gemm_nt_contiguous` through `torch::Tensor` arguments, shape/layout checks, scale-layout transforms, and `device_runtime`.
  - `openinfer-kernels/third_party/DeepGEMM/csrc/jit/device_runtime.hpp` depends on ATen CUDA context and torch-owned cuBLASLt workspace.
  - `openinfer-kernels/third_party/DeepGEMM/csrc/jit/compiler.hpp` and `jit/kernel_runtime.hpp` are controlled by `DG_JIT_*` environment knobs in upstream DeepGEMM; GLM main code must not inherit those as serving configuration.
  - The SM90 grouped kernel implementation eventually reaches raw launch arguments and `CUtensorMap` descriptors in `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`, including `sm90_m_grouped_fp8_gemm_contiguous_1d2d`.
- Conclusion:
  - Do not call DeepGEMM through its Python/torch extension from `openinfer-glm52`.
  - Keep the OpenInfer boundary as raw pointers, explicit `cudaStream_t`, fixed GLM W13/W2 shapes, and persistent arena-owned buffers.
  - The first implementation path should split or adapt the DeepGEMM JIT/runtime pieces needed for fixed SM90 grouped FP8 without ATen, then warm the exact W13/W2 kernels before decode graph capture.
  - If that raw split grows too invasive, fall back to a raw vLLM CUTLASS grouped-FP8 port with the same `glm52_deepgemm_grouped_fp8_launch` Rust/C ABI, so the model-side call contract does not churn.

### Step 38: Graph-captured MoE GEMM metadata from DeepEP psum

- Fixed the MoE GEMM metadata ownership boundary before adding real W13/W2 compute:
  - The only authoritative grouped-layout input is now `Glm52DeepEpDispatchScratch::psum_expert`, populated by the DeepEP dispatch call.
  - Removed the unused arena-owned DeepEP scratch shadow buffers from `Glm52DecodeArena`; keeping a second `deepep_psum_expert` buffer would let a later GEMM launch silently consume zeros.
  - Added persistent arena-owned GEMM metadata buffers:
    - `moe_gemm_expert_offsets: [i64; local_experts + 1]`
    - `moe_w13_problem_sizes: [i32; local_experts * 3]`
    - `moe_w2_problem_sizes: [i32; local_experts * 3]`
- Added `glm52_deepgemm_grouped_fp8_metadata_cuda` under the existing GLM grouped-GEMM ABI file:
  - CUDA source: `openinfer-kernels/csrc/glm52/glm52_deepgemm_grouped.cu`.
  - Rust wrapper: `openinfer-kernels/src/ops/glm52/deepgemm_grouped.rs::glm52_deepgemm_grouped_fp8_metadata_launch`.
  - Source semantics follow vLLM CUTLASS MoE metadata rules from `../vllm/csrc/libtorch_stable/quantization/w8a8/cutlass/moe/moe_data.cu`: grouped GEMM consumes per-expert `expert_offsets` plus `[M,N,K]` problem-size triples. GLM skips vLLM's `moe_permute` path because DeepEP already produced expert-major expanded rows; it derives `M` from DeepEP's expand-mode rule `expert i = [align(psum[i-1], 8), psum[i])`.
  - W13 metadata is `[m,4096,6144]`; W2 metadata is `[m,6144,2048]`; `expert_offsets[local_experts]` is the final aligned expanded row count.
- Wired metadata generation into the real decode substrate:
  - `decode_router_smoke_roundtrip` launches metadata after router-produced DeepEP dispatch and validates offsets/problem sizes against the same host snapshot used for grouped-layout validation.
  - `decode_graph_smoke_roundtrip` captures the metadata launch inside the fixed 128-token CUDA Graph, before activation quant/combine.
  - `Glm52DeepEpSmokeReport` and `Glm52DecodeGraphSmokeReport` now fail closed on `moe_gemm_metadata_valid`.
- Validation:
  - local `cargo fmt` -> passed.
  - local `cargo check -p openinfer-server` -> passed after rebuilding `openinfer-kernels`; default feature build took `52.54s`.
  - local `cargo check -p openinfer-server --features glm52` still stops at the expected local NCCL-root guard before Rust type-checking: `OPENINFER_NCCL_ROOT` is required for the DeepEP shim.
  - local `git diff --check` -> passed.
  - Synced the changed files to `jz-38:/root/develop/xingming/pegainfer-glm52` with `rsync -avR`.
  - remote `jz-38`: `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed in `42.39s` after rebuilding `openinfer-kernels`.
  - remote `jz-38` release checkpoint IT passed against `/data/models/GLM-5.2-0614-Provider-FP8`: release rebuild `43.70s`, test body `56.92s`, rank worker spawn `3.80s`, weight load `45.02s`, DeepEP install `3.73s`, DeepEP route smoke `0.04s`, MoE GEMM contract validation `0.00s`, decode CUDA Graph smoke `0.36s`.
  - The real-router DeepEP smoke reported `gemm_metadata=Some(...)` with `offsets_valid=true`, `w13_problem_sizes_valid=true`, and `w2_problem_sizes_valid=true` on all 8 ranks, including empty ranks.
  - The decode CUDA Graph smoke reported `moe_gemm_metadata_valid=true`, `capture_and_first_launch_ok=true`, and `replay_ok=true` on all 8 ranks.
- Current end-to-end gap:
  - This step prepares the actual W13/W2 GEMM call metadata; it does not compute GEMM. `glm52_deepgemm_grouped_fp8_launch_cuda` still returns `CUDA_ERROR_NOT_SUPPORTED` until the raw DeepGEMM or vLLM CUTLASS grouped-FP8 runtime is wired.

### Step 39: DeepGEMM psum compatibility audit

- Re-read the relevant backend source before choosing the next GEMM implementation path:
  - DeepGEMM SM90 `MGroupedContiguousWithPsumLayout` treats `grouped_layout[i]` as an end offset, but when advancing groups it starts the next group at `align(previous_end, BLOCK_M)`, not at GLM DeepEP's `expert_alignment=8`.
  - vLLM's CUTLASS grouped-FP8 wrapper is closer to a raw CUDA/C++ port, but its `ScaledEpilogueArray` path is per-token/per-output-channel scale, and its torch wrapper allocates pointer arrays/workspace per call. It is not a direct match for GLM checkpoint block scales or CUDA Graph capture without an arena-backed ABI.
  - vLLM now prefers FlashInfer CUTLASS for Hopper block-FP8 MoE; FlashInfer has an SM90 DeepSeek FP8 block-scale path, but the public path is a fused-MoE/JIT module with TensorRT-LLM/CUTLASS runner pieces rather than the current DeepEP-expanded-row W13/W2 substrate.
- Added non-gating audit fields to `Glm52MoeGemmMetadataSmokeReport`:
  - `deepgemm_block_m64_psum_compatible`
  - `deepgemm_block_m128_psum_compatible`
- These fields check the current router-produced DeepEP `psum_expert` against DeepGEMM's scheduler rule. They do not change startup acceptance yet; they exist to prove whether direct raw DeepGEMM can consume the current layout or whether the backend must change layout/metadata before real W13/W2 GEMM lands.
- Source implication:
  - Step 40 supersedes the `expert_alignment=8` layout by moving the current GLM DeepEP expanded rows to 64-row expert alignment.
  - Direct DeepGEMM psum layout needs evidence that actual `psum_expert` groups are compatible with the selected DeepGEMM runtime alignment; with the Step 40 contract, `deepgemm_block_m64_psum_compatible` should become the gating evidence and `m128` is no longer required for the chosen H200 path.
  - FlashInfer should stay in the MoE backend decision tree because it has the relevant SM90 DeepSeek FP8 block-scale source, but plugging it into GLM likely means a fused-MoE/raw-runner integration rather than a narrow replacement for `glm52_deepgemm_grouped_fp8_launch_cuda`.

### Step 40: DeepEP alignment for DeepGEMM psum

- Re-read DeepGEMM SM90 grouped-FP8 heuristics before changing the layout:
  - `MGroupedContiguous` and `MGroupedContiguousWithPsumLayout` use `heuristics_runtime->get_mk_alignment_for_contiguous_layout()` as the `BLOCK_M` candidate.
  - DeepGEMM's contiguous psum tests generate group starts as `align(previous_end, get_mk_alignment_for_contiguous_layout())`.
  - Therefore GLM can choose a fixed DeepGEMM runtime alignment of 64 and make DeepEP's expanded layout match that contract directly.
- Changed the current GLM DeepEP/DeepGEMM shape contract:
  - DeepEP expert segment alignment: `8 -> 64`.
  - Worst-case expanded rows: `8416 -> 10240`.
  - Grouped-GEMM ABI capacity: `m_capacity=10240`, with activation-scale TMA rows also `10240`.
  - Persistent W2 output buffer shape grows to `10240 * 6144 * sizeof(bf16) = 125829120` bytes per rank.
- Rationale:
  - This avoids a post-DeepEP repack kernel before W13/W2 GEMM.
  - It keeps `Glm52DeepEpDispatchScratch::psum_expert` as the single source of grouped-layout truth.
  - It lets the first raw DeepGEMM wrapper target SM90 `m_grouped_fp8_gemm_nt_contiguous` with `MGroupedContiguousWithPsumLayout` and runtime alignment 64.
- Validation:
  - local `cargo fmt` -> passed.
  - local `git diff --check` for the touched GLM DeepEP/DeepGEMM files -> passed.
  - local `cargo check -p openinfer-server` -> passed after rebuilding `openinfer-kernels`.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed in `41.66s` after rebuilding `openinfer-kernels`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `44.19s`, test body `56.52s`, rank worker spawn `3.81s`, weight load `44.41s`, DeepEP install `3.80s`, DeepEP decode smoke `0.05s`, decode CUDA Graph smoke `0.44s`.
  - Startup log reported `deepep_decode_recv=1024`, `deepep_decode_expanded=10240`, and resident bytes `122968582728` per rank.
  - Real-router DeepEP smoke reported `expert_alignment=64`, grouped-layout valid, metadata offsets/problem sizes valid, and `deepgemm_block_m64_psum_compatible=true` on all 8 ranks. `deepgemm_block_m128_psum_compatible=false` appears on several non-empty ranks and is acceptable for the chosen runtime alignment.
  - All-layer MoE GEMM contract reported `layer_count=75`, `first_layer_idx=3`, `last_layer_idx=77`, backend `ExpandedDeepEpGroupedFp8Contract`, `m_capacity=10240`, `activation_scale_tma_rows=10240`, and `expert_alignment=64` on all 8 ranks.
  - Decode CUDA Graph smoke reported `worst_expanded_rows=10240`, `moe_gemm_metadata_valid=true`, `grouped_layout_valid=true`, `capture_and_first_launch_ok=true`, and `replay_ok=true` on all 8 ranks.
  - Post-test machine state stayed clean: all 8 H200 GPUs at `0MiB`; hugepages `512000` total/free, `HugePages_Rsvd=0`; page cache about `768GB`.
- This is a layout compatibility change, not a compute implementation. `glm52_deepgemm_grouped_fp8_launch_cuda` still intentionally returns `CUDA_ERROR_NOT_SUPPORTED` until the raw DeepGEMM or fallback CUTLASS grouped-FP8 runtime is wired.

### Step 41: FlashInfer/TRTLLM grouped-offset audit

- Re-read the vendored FlashInfer TensorRT-LLM DeepGEMM path before treating it as a raw JIT shortcut:
  - `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh` exposes `moeGemm(... problem_m_offsets ...)` and dispatches Hopper block-FP8 grouped GEMM through `deep_gemm::GemmType::GroupedWithOffset`.
  - `csrc/nv_internal/tensorrt_llm/deep_gemm/scheduler.cuh` consumes `problem_m_offsets[e]..problem_m_offsets[e + 1]` for each expert and separately maps activation scale rows through `compute_padded_offset(offset, problem_idx)`.
  - `compute_padded_offset` uses fixed 32-row padding: `(offset + problem_idx * 31) / 32 * 32`.
- Added non-gating startup diagnostics to `Glm52MoeGemmMetadataSmokeReport`:
  - `trtllm_grouped_offset_scale_rows_required`
  - `trtllm_grouped_offset_scale_rows_covered`
- Current implication:
  - GLM's existing `expert_offsets[int64; 33]` are a useful input shape for FlashInfer/TRTLLM `GroupedWithOffset`, but the current DeepGEMM MN-major activation scale layout is not enough for that route.
  - Fixed-bucket GLM capacity is `10240` expanded rows and `32` local experts. FlashInfer/TRTLLM's scale-row formula requires `compute_padded_offset(10240, 32) = 11232` activation scale rows.
  - Current GLM scale buffers are `10240` rows because they target DeepGEMM SM90 `MGroupedContiguousWithPsumLayout`, so `trtllm_grouped_offset_scale_rows_covered=false` is expected.
  - Therefore FlashInfer/TRTLLM GroupedWithOffset is still a serious fallback source, but not drop-in: it needs either a separate activation-scale relayout/scatter into the padded offset space or a deliberate scheduler/runtime adaptation. Do not wire it by only passing `expert_offsets`.
- This narrows the GEMM decision:
  - Primary route remains upstream DeepGEMM SM90 psum layout with current `psum_expert` and 64-row DeepEP alignment.
  - FlashInfer/TRTLLM raw runner fallback is now blocked specifically on activation-scale layout ownership, not on missing grouped offsets.
- Validation:
  - local `cargo fmt` -> passed.
  - local `cargo check -p openinfer-server` -> passed in `0.51s`.
  - local `git diff --check` for the touched GLM files and docs -> passed.
  - Synced `openinfer-glm52/src/{arena.rs,moe_deepep.rs}` plus the GLM docs to node38 with `rsync -avR`, then touched the changed Rust files to force recompilation.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed in `1.27s` after rebuilding `openinfer-glm52`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `2.64s`, test body `56.32s`, rank worker spawn `3.81s`, weight load `44.43s`, DeepEP install `4.16s`, DeepEP route smoke `0.04s`, MoE GEMM contract validation `0.00s`, decode CUDA Graph smoke `0.44s`.
  - Real-router DeepEP smoke now reports `trtllm_grouped_offset_scale_rows_required=11232` and `trtllm_grouped_offset_scale_rows_covered=false` on all 8 ranks, while keeping `offsets_valid=true`, `w13_problem_sizes_valid=true`, `w2_problem_sizes_valid=true`, and `deepgemm_block_m64_psum_compatible=true`.
  - Post-test machine state stayed clean: all 8 H200 GPUs at `0MiB`; hugepages `512000` total/free, `HugePages_Rsvd=0`; page cache about `768GB`.

### Step 42: Gate DeepGEMM m64 psum compatibility

- Step 40 made `expert_alignment=64` the contract for the primary DeepGEMM SM90 psum route, but the code still treated `deepgemm_block_m64_psum_compatible` as a report-only field.
- Promoted that evidence into startup validation:
  - `validate_moe_gemm_metadata_smoke` now fails closed unless router-produced `psum_expert` is compatible with DeepGEMM's `BLOCK_M=64` slice rule.
  - Decode CUDA Graph smoke also includes `deepgemm_block_m64_psum_compatible` in `moe_gemm_metadata_valid`.
  - `deepgemm_block_m128_psum_compatible` remains diagnostic only; false is acceptable for the chosen H200 route.
- This keeps the primary W13/W2 GEMM path honest: a future DeepEP or metadata change cannot preserve valid offsets/problem sizes while silently breaking the DeepGEMM psum scheduler assumption.
- Validation:
  - local `cargo fmt` -> passed.
  - local `cargo check -p openinfer-server` -> passed in `0.46s`.
  - local `git diff --check` for `openinfer-glm52/src/moe_deepep.rs` and this doc -> passed.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed in `1.08s` after rebuilding `openinfer-glm52`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `2.67s`, test body `57.03s`, rank worker spawn `3.85s`, weight load `44.59s`, DeepEP install `3.67s`, DeepEP route smoke `0.05s`, decode CUDA Graph smoke `0.50s`.
  - The real-router DeepEP smoke still reports `deepgemm_block_m64_psum_compatible=true` on all 8 ranks; decode graph smoke still reports `moe_gemm_metadata_valid=true` on all 8 ranks.
  - Post-test machine state stayed clean: all 8 H200 GPUs at `0MiB`; hugepages `512000` total/free, `HugePages_Rsvd=0`; page cache about `768GB`.

### Step 43: TRTLLM grouped-offset scale relayout

- Added the missing activation-scale substrate for the vendored FlashInfer/TRTLLM `GroupedWithOffset` raw runner candidate:
  - CUDA source: `openinfer-kernels/csrc/glm52/glm52_deepgemm_layout.cu`.
  - Rust wrapper: `openinfer-kernels/src/ops/glm52/deepgemm_layout.rs::glm52_deepgemm_grouped_offset_tma_aligned_f32_launch`.
  - ABI: `glm52_deepgemm_grouped_offset_tma_aligned_f32_cuda(input, expert_offsets, output, m_capacity, scale_cols, groups, padded_rows, stream)`.
  - Layout rule copies the TRTLLM scheduler formula: `compute_padded_offset(row, problem_idx) = ((row + problem_idx * 31) / 32) * 32`.
- Decode arena now owns separate persistent TRTLLM offset-scale buffers:
  - W13 scale output: `11232 * 48 * sizeof(f32)`.
  - W2 scale output: `11232 * 16 * sizeof(f32)`.
  - These are separate from the DeepGEMM `10240`-row MN-major scale buffers because the two backend candidates have different scale-row addressing.
- Startup validation now fails closed on the new layout when requested:
  - `trtllm_grouped_offset_scale_rows_required=11232`.
  - `trtllm_grouped_offset_scale_rows_covered=true`.
  - Real-router DeepEP MoE quant smoke reports `trtllm_offset_scale_layout_ran=true`, `trtllm_offset_scale_layout_valid=true`, and `trtllm_offset_scale_rows=11232` on non-empty ranks.
  - Decode CUDA Graph smoke captures the W13/W2 grouped-offset relayout kernels and reports `trtllm_offset_scale_layout_valid=true` on all 8 ranks.
- Validation:
  - local `cargo fmt` -> passed.
  - local `git diff --check` for touched files -> passed.
  - local `cargo check -p openinfer-server` -> passed after rebuilding `openinfer-kernels` in `52.40s`.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed after rebuilding `openinfer-kernels` in `43.45s`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `44.90s`, test body `110.09s`, rank worker spawn `3.74s`, weight load `86.42s`, DeepEP install `6.91s`, DeepEP route smoke `0.07s`, decode CUDA Graph smoke `0.47s`.
  - The `86.42s` weight-load time is slower than the warm `~44-51s` range from earlier page-cache runs and is treated as IO/page-cache noise, not a relayout-code regression.
- NCU evidence:
  - Full-checkpoint NCU around the GLM IT failed because NCU replay/attach disrupted the DeepEP NVLink barrier before the target kernel; DeepEP reported a barrier timeout and then `glm52_deepep_decode_dispatch: cudaFuncSetAttribute: unspecified launch failure`.
  - A standalone H200 harness was used instead: `profile/glm52_trtllm_offset_scale_20260626/harness/trtllm_offset_scale_harness.cu`.
  - NCU full report: `profile/glm52_trtllm_offset_scale_20260626/reports/grouped_offset_scale_full.ncu-rep`.
  - Profile report: `profile/glm52_trtllm_offset_scale_20260626/REPORT.md`.
  - Measured W13-scale shape (`m_capacity=10240`, `groups=32`, `scale_cols=48`, `padded_rows=11232`): two NCU launches took `23.04us` and `22.46us`; achieved occupancy `68.80%`/`69.53%`; SM compute throughput `52.03%`/`52.12%`; DRAM throughput `1.74%`/`1.79%`; memory throughput `85.52`/`87.72 GB/s`.
  - NCU rule engine reports one full wave plus a partial wave of `1050` blocks, with an estimated `50%` tail-effect upper bound under uniform block duration assumptions. The relayout is therefore not DRAM-bandwidth-bound; it is a layout substrate to keep the FlashInfer/TRTLLM fallback viable, not a MoE performance win.

### Step 44: FlashInfer/TRTLLM grouped-FP8 raw runner ABI

- Added the first raw-runner boundary for the vendored FlashInfer/TRTLLM grouped block-FP8 GEMM path:
  - CUDA source: `openinfer-kernels/csrc/glm52/glm52_trtllm_grouped_fp8.cu`.
  - Rust wrapper: `openinfer-kernels/src/ops/glm52/trtllm_grouped.rs`.
  - C ABI:
    - `glm52_trtllm_grouped_fp8_contract_cuda`
    - `glm52_trtllm_grouped_fp8_workspace_size_cuda`
    - `glm52_trtllm_grouped_fp8_launch_cuda`
  - Backend source: FlashInfer vendored TensorRT-LLM `CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>` from `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm`.
- Contract encoded in the wrapper:
  - W13: `groups=32`, `m_capacity=10240`, `N=4096`, `K=6144`, weight scales `[32,32,48]`, activation scale columns `48`, TRTLLM activation scale rows `11232`.
  - W2: `groups=32`, `m_capacity=10240`, `N=6144`, `K=2048`, weight scales `[32,48,16]`, activation scale columns `16`, TRTLLM activation scale rows `11232`.
  - FP8+FP8 path must report `workspace_bytes=0`; the wrapper still calls `getWorkspaceSizeBase(...)` before launch because that initializes the runner's internal expected/padded M state.
  - `glm52_trtllm_grouped_fp8_launch` validates activation, activation-scale, weight, weight-scale, `expert_offsets`, and output buffer lengths before entering the C ABI. Step 45 wires this launch into the startup MoE GEMM smoke and fixed-bucket graph smoke; Step 44 used only the workspace query as the runner-link/shape-init gate.
- Build/link lessons:
  - Including the TRTLLM runner `.cu` pulls in DeepGEMM JIT host code even when the runtime compiler is invalid and the static CUTLASS fallback is expected.
  - The first release IT linked-failed on `Logger::getLogger`, `TllmException`, `fmtstr_`, and CUDA driver `cuLibrary*` symbols.
  - Fix: compile the vendored TRTLLM common cpp sources into the GLM52 TU (`stringUtils.cpp`, `tllmException.cpp`, `logger.cpp`) and link `libcuda` when the `glm52` kernel feature is enabled.
  - Do not set DeepGEMM/JIT include dirs in OpenInfer main code. Leaving the global compiler invalid keeps TRTLLM on the static CUTLASS fallback and avoids runtime `nvcc`/environment coupling.
- Startup integration:
  - `Glm52MoeGemmOperandContract` now records `activation_scale_trtllm_rows` and `trtllm_workspace_bytes`.
  - `validate_moe_gemm_contracts` fails closed unless W13/W2 both report `activation_scale_trtllm_rows=11232` and `trtllm_workspace_bytes=0` on every rank.
- Validation:
  - local `cargo fmt --check && cargo check -p openinfer-server` -> passed; default check rebuilt `openinfer-kernels` in `53.22s`.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed after rebuilding `openinfer-kernels` in `1m16s`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `1m14s`, test body `87.64s`, rank worker spawn `3.90s`, weight load `75.24s`, DeepEP install `4.08s`, DeepEP decode smoke `0.09s`, MoE GEMM contract validation `0.00s`, decode CUDA Graph smoke `0.51s`.
  - Contract report on all 8 ranks: W13 `G=32,N=4096,K=6144,activation_scale_tma_rows=10240,activation_scale_trtllm_rows=11232,trtllm_workspace_bytes=0`; W2 `G=32,N=6144,K=2048,activation_scale_tma_rows=10240,activation_scale_trtllm_rows=11232,trtllm_workspace_bytes=0`.
  - Post-test node38 state stayed clean: all 8 H200 GPUs at `0MiB`, and no leftover `checkpoint`/`openinfer`/`cargo test` process other than the probing shell.

### Step 45: TRTLLM W13/W2 launch and graph capture

- Wired the FlashInfer/TRTLLM grouped-FP8 runner into the GLM MoE decode smoke:
  - `launch_trtllm_w13_grouped_fp8` consumes DeepEP recv-row W13 FP8 activations, `11232`-row TRTLLM activation scales, local W13 FP8 expert packages, W13 weight scales, and `expert_offsets`.
  - `launch_trtllm_w2_grouped_fp8` consumes the weighted W2-input FP8 activations, `11232`-row TRTLLM activation scales, local W2 FP8 expert packages, W2 weight scales, and the same offsets.
  - The smoke runs router -> DeepEP dispatch -> grouped metadata -> W13 input quant -> W13 GEMM -> weighted W2 input quant -> W2 GEMM -> DeepEP combine, then validates nonzero W13 output, W2 output, and combined BF16 output on all non-empty ranks.
- Fixed the Hopper WGMMA build target for this TU:
  - node38 CUDA 12.8's `nvcc --list-gpu-arch` does not list `compute_90a`, but a real probe compile with `-gencode arch=compute_90a,code=sm_90a` succeeds.
  - `openinfer-kernels/build.rs` now probes the gencode target and compiles only `glm52_trtllm_grouped_fp8.cu` as `sm_90a`; the rest of the CUDA build keeps the detected `sm_90` target.
  - The first release IT after the launch wiring failed in FlashInfer/TRTLLM with `wgmma is only available on SM90a`; after the probe-based target selection the same path enters real GEMM.
- Upgraded decode CUDA Graph smoke to include real W13/W2:
  - The captured fixed-bucket sequence is now router, DeepEP dispatch, grouped-GEMM metadata, W13 input quant, TRTLLM W13 grouped FP8, weighted W2 input quant, TRTLLM W2 grouped FP8, and DeepEP combine.
  - The graph smoke validates `w13_output_nonzero=true`, `w2_output_nonzero=true`, `combined_nonzero=true`, `capture_and_first_launch_ok=true`, and `replay_ok=true` on all 8 ranks.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed and reported `Compiling GLM5.2 TRTLLM grouped FP8 for nvcc targets: sm_90a`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `3.12s`, test body `68.69s`, rank worker spawn `3.91s`, weight load `56.63s`, DeepEP install `3.74s`, DeepEP decode smoke `0.08s`, MoE GEMM decode smoke `0.01s`, decode CUDA Graph smoke `0.51s`.
  - MoE GEMM smoke report: all 8 ranks had `w13_output_nonzero=true`, `w2_output_nonzero=true`, and `combined_nonzero=true`; the heaviest observed rank in this smoke had `active_experts=9`, `expanded_rows=3904`.
  - Decode CUDA Graph smoke report: all 8 ranks had `fixed_bucket_tokens=128`, `worst_expanded_rows=10240`, `moe_gemm_metadata_valid=true`, `trtllm_offset_scale_layout_valid=true`, nonzero W13/W2/combined outputs, capture success, and replay success.
  - Post-test node38 state stayed clean: all 8 H200 GPUs returned to `0MiB`.

### Step 46: MoE substrate call surface and test cleanup

- Refactored the GLM MoE decode path so the real decode-substrate chain is no longer duplicated between eager smoke and graph smoke:
  - New private call surface: `Glm52MoeDeepEpState::launch_decode_moe_layer_substrate`.
  - Both `decode_moe_gemm_smoke_roundtrip` and `decode_graph_smoke_kernels` now call the same router -> DeepEP dispatch -> grouped metadata -> W13 quant -> TRTLLM W13 grouped FP8 -> weighted W2 quant -> TRTLLM W2 grouped FP8 -> DeepEP combine sequence.
  - Smoke-specific D2H validation remains outside that call surface, so a later full decode layer can reuse the substrate without inheriting test-only checks.
- Split report/layout data structures out of the execution-heavy file:
  - `openinfer-glm52/src/moe_deepep.rs` shrank from `1137` lines to `913`.
  - `openinfer-glm52/src/moe_deepep/types.rs` now owns psum layout snapshots, smoke reports, and the DeepGEMM psum compatibility helper.
- Cleaned the GLM test surface:
  - Deleted `openinfer-glm52/tests/probe.rs` because it was a 103-line synthetic provider config fixture and the node38 checkpoint IT now covers config probing, stop-token loading, manifest/header validation, rank worker loading, DeepEP install, MoE substrate smokes, graph replay, and fail-closed request behavior against the real checkpoint.
  - Removed the `tempfile` dev dependency from `openinfer-glm52`; `Cargo.lock` still contains `tempfile` for other workspace packages, but GLM no longer depends on it.
  - The only GLM test file is now `openinfer-glm52/tests/checkpoint.rs`, intentionally ignored by default and run explicitly on node38.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - local `cargo test -p openinfer-glm52` is still not a valid local gate without NCCL >= 2.30.4; it fails in `openinfer-kernels/build.rs` before Rust type-checking, as expected for the GLM DeepEP feature.
  - node38 `cargo fmt --check` -> passed.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed and reported `Compiling GLM5.2 TRTLLM grouped FP8 for nvcc targets: sm_90a`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `2.88s`, test body `117.19s`, rank worker spawn `3.15s`, weight load `98.95s`, DeepEP install `6.72s`, DeepEP decode smoke `0.10s`, MoE GEMM decode smoke `0.01s`, decode CUDA Graph smoke `0.52s`.
  - MoE GEMM smoke still reported `w13_output_nonzero=true`, `w2_output_nonzero=true`, and `combined_nonzero=true` on all 8 ranks; decode graph smoke still reported nonzero W13/W2/combined outputs, capture success, and replay success on all 8 ranks.
  - Post-test node38 state stayed clean: all 8 H200 GPUs returned to `0MiB`, and the remote GLM test directory contains only `checkpoint.rs`.

### Step 47: Non-expert FP8 projection contract cleanup

- Deleted the stale full-rank typed-view scaffold from `openinfer-glm52/src/weights/view.rs`:
  - The removed path still assumed raw routed expert tensors stayed resident, but the loader now streams routed experts into per-layer FP8 packages and removes the raw expert tensors from the resident map.
  - The live weight validation surface is now `validate_non_expert_weight_contract`, which covers exactly the tensors that remain resident after expert packaging.
- Added a checkpoint-backed non-expert FP8 projection contract report:
  - Each rank must expose `666` non-expert FP8 projections: `432` attention/indexer, `9` dense MLP, and `225` shared-expert projections.
  - The contract validates FP8 weight dtype, F32 `weight_scale_inv` dtype, and `[ceil(out/128), ceil(in/128)]` scale grids for every non-expert projection.
  - The max non-expert projection columns are fail-closed as `max_out=114688`, `max_in=16384`, with max scale grid `896x128`; these are independent maxima from `kv_b_proj` output rows and `o_proj` input columns.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - node38 `cargo fmt --check` -> passed.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed and reported `Compiling GLM5.2 TRTLLM grouped FP8 for nvcc targets: sm_90a`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `2.97s`, test body `68.79s`, rank worker spawn `3.83s`, weight load `56.60s`, DeepEP install `3.97s`, DeepEP decode smoke `0.07s`, MoE GEMM decode smoke `0.01s`, decode CUDA Graph smoke `0.49s`.
  - Startup report showed all 8 ranks at `non_expert_fp8_projections=[666, ...]` and `attention/dense/shared=[(432, 9, 225), ...]`.
  - Post-test node38 state stayed clean: all 8 H200 GPUs returned to `0MiB`.

### Step 48: Full-layer decode arena substrate and cleanup

- Added the persistent fixed-bucket scratch that full decode needs before it can enter one CUDA Graph:
  - generic layer state: `hidden`, `normed`, FP8 linear input, FP8 linear scales.
  - attention scratch: `q_a=2048`, `q_b=65536`, `kv_a=576`, `kv_lora=512`, `k_rope=64`, `kv_b=114688`, `attention_out/o_proj_in=16384`.
  - DSA/indexer scratch: scores `32`, `wk=128`, `wq_b=4096`, top-k ids/weights `2048`.
  - dense/shared/logits scratch: dense gate/up `24576`, dense activated `12288`, shared gate/up `4096`, shared activated `2048`, logits `154880`.
  - non-expert FP8 activation quant scratch is sized to `linear_quant_max_in=16384` with `128` scale columns.
- Kept the arena contract honest:
  - `Glm52DecodeArenaPlan::validate` now fail-closes on the full-layer dimensions, not just MoE/DeepEP capacity.
  - `openinfer-glm52/src/arena/validation.rs` owns the allocation shape and byte-accounting checks so `arena.rs` stays under the 1k-line source limit.
  - Deleted the generic `projected` scratch because it had no call site and no distinct shape contract; attention/dense/shared already have typed output buffers.
- This is still substrate, not forward execution:
  - No non-expert FP8 GEMM wrapper is wired yet.
  - No attention/KV/indexer decode wrapper is wired yet.
  - No dense/shared/residual/logits/sampling path is wired yet.
  - The value here is graph-safe pointer stability and fixed bucket shape before those kernels land.
- Validation:
  - local `cargo fmt --check` -> passed.
  - local `cargo check -p openinfer-server` -> passed.
  - local source shape check: `arena.rs` is `930` lines, `arena/validation.rs` is `136` lines, and the GLM checkpoint IT remains `58` lines.
  - node38 `cargo fmt --check` -> passed.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed and reported `Compiling GLM5.2 TRTLLM grouped FP8 for nvcc targets: sm_90a`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: release rebuild `3.06s`, test body `64.98s`, rank worker spawn `3.86s`, weight load `52.50s`, DeepEP install `4.15s`, DeepEP decode smoke `0.07s`, MoE GEMM decode smoke `0.01s`, decode CUDA Graph smoke `0.49s`.
  - Startup report still showed all 8 ranks at `non_expert_fp8_projections=[666, ...]` and `attention/dense/shared=[(432, 9, 225), ...]`.
  - Post-test node38 state stayed clean: all 8 H200 GPUs were `0MiB` before and after the IT.

### Step 49: Generic non-expert FP8 projection smoke

- Replaced the q_a-only startup smoke with a shared projection smoke that covers q_a, q_b, kv_a, kv_b, o_proj, indexer_wk, and indexer_wq_b for layer 0.
- Validation:
  - local `cargo fmt --check` -> passed.
  - node38 `cargo fmt --check` -> passed.
  - node38 `cargo check -p openinfer-server --features glm52` with the node38 NCCL root -> passed in `1.91s`.
  - node38 release checkpoint IT against `/data/models/GLM-5.2-0614-Provider-FP8` -> passed: test body `68.30s`, rank worker spawn `2.58s`, weight load `57.93s`, non-expert projection smoke `0.03s`, DeepEP install `3.33s`, DeepEP decode smoke `0.08s`, MoE GEMM decode smoke `0.01s`, decode CUDA Graph smoke `0.50s`.
  - Projection smoke evidence: all 8 ranks reported `workspace_bytes=0`, `activation_quant_valid=true`, and `output_nonzero=true` for q_a, q_b, kv_a, kv_b, o_proj, indexer_wk, and indexer_wq_b.

## Blocks

- Runtime forward execution is not implemented yet. The current `glm52` server path intentionally rejects requests after scheduling them, while keeping the loaded 8-rank worker runtime alive.
- The biggest known design blocker is full decode integration shape: FlashInfer, DeepGEMM, FlashMLA, and `../vllm` all appear relevant, but exact H200 performance, DeepGEMM-vs-TRTLLM backend choice, KV handoff shape, attention/indexer decode wrappers, and graph-safe full-forward composition are unproven. TRTLLM W13/W2 shape/workspace, actual launch, and graph capture are now proven for the fixed GLM bucket.
- DeepEP context creation, synthetic all-EP decode dispatch/combine, real-router decode route smoke, typed MoE psum-layout validation, graph-captured MoE GEMM metadata from the real DeepEP dispatch scratch, actual DeepEP recv-row W13 quant, activation FP8 quant smoke, DeepGEMM F32 scale-layout smoke, FlashInfer/TRTLLM grouped-offset scale relayout smoke, FlashInfer/TRTLLM grouped-FP8 runner workspace gate, weighted W2-input quant, real TRTLLM W13/W2 grouped-FP8 launch, all-layer MoE GEMM contract validation, 64-row DeepGEMM psum compatibility, decode-substrate CUDA Graph capture/replay, and separate W2 output/combine residency are wired into GLM rank workers. The real MoE substrate sequence is now shared by eager and graph smoke call sites. Still pending: dense/shared/residual integration, attention/KV/indexer decode, logits/sampling, real scheduler handoff, and full-forward graph integration. Prefill dispatch/recv/combine is not a first runnable engine target. There is no GLM NCCL all-to-all backend.
- Weighted W2-input activation quant has landed with DeepGEMM TileLang source backing, startup correctness checks, decode-graph replay validation, and H200 NCU evidence. The real MoE substrate now has W13/W2 grouped FP8 GEMM; the full forward still needs layer integration and an end-to-end logits gate.
- Rank weights are copied to GPU and retained by rank workers as non-expert raw tensors plus per-layer FP8 routed-expert packages; the non-expert weight contract validates attention/dense/router/shared-expert coverage plus `666` FP8 projections per rank; decode-bucket buffers are allocated persistently per rank with full-layer scratch plus distinct DeepEP recv, W13 output, W2 input/output, and combine buffers. Alternative DeepGEMM/FlashInfer-fused/vLLM GEMM routes, KV/cache arenas, and actual forward execution are not wired yet.

## Optimization Log

### #0 Baseline

Not collected yet. Baseline must include:

| Profile | Required evidence |
| --- | --- |
| Reference vLLM/SGLang decode-heavy | TPOT median/p99 plus version/command |
| openinfer first runnable decode-only | Same metrics, same prompts, same topology, with prefilled KV/page-state contract stated |
| openinfer decode CUDA Graph | eager vs graph correctness and TPOT delta |
| handwritten kernel NCU | Kernel name, launch shape, achieved occupancy, SM %, memory throughput, top bottleneck |
