# GLM5.2 vLLM Kernel Reference

> **TL;DR:** vLLM already provides several GLM5.2-relevant kernel contracts. Direct-copy candidates are the GLM-5 router GEMM, grouped noaux top-k, MoE permute/unpermute reduction, indexer cache insert/gather, `concat_mla_q`, sparse-indexer top-k, and some FP8 quantization kernels. DeepGEMM MoE and DeepEP paths are mostly Python/Triton/third-party API composition, so copy their contracts first and only rewrite missing pieces after source-backed shape checks and NCU evidence. Current OpenInfer GLM52 uses DeepGEMM TileLang `swiglu_apply_weight_to_fp8` semantics for weighted W2-input activation quant; it does not use a standalone route-weight multiply.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes this as a GLM52 model reference doc.
  - `docs/models/glm52/support.md` - current GLM52 constraints: decode-only first, `DP8 TP1 EP8`, real `bs > 1`, decode graph, DeepEP, no MTP, no hidden prefill path.
  - `docs/playbooks/model-optimization-pipeline.md` - keep kernel/operator evidence in a per-model doc instead of rediscovering paths.
  - `docs/models/kimi-k2/deepep-migration.md` - route weights and DeepEP graph readiness must be contract-backed and measured.
  - `docs/models/kimi-k2/vllm-path-comparison.md` - Kimi already used vLLM path comparison as a durable operator map.
  - `../vllm/vllm/model_executor/models/deepseek_v2.py` - `GlmMoeDsaForCausalLM` reuses the DeepSeek-V2/V3 model path; MoE uses `GateLinear`, `FusedMoE`, noaux correction bias, grouped top-k, and routed scaling.
  - `../vllm/vllm/model_executor/models/glm4_moe.py` - GLM MoE semantics reference: F32 router, sigmoid, correction bias, grouped top-k, shared expert, routed scale.
  - `../vllm/csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu` and siblings - CUDA router GEMM has GLM-5 `hidden=6144`, `experts=256`, `tokens=1..16` instantiations.
  - `../vllm/csrc/libtorch_stable/moe/grouped_topk_kernels.cu` - CUDA noaux grouped top-k implements sigmoid, correction-bias ranking, unbiased weight normalization, and routed scaling.
  - `../vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py` and `deep_gemm_utils.py` - DeepGEMM grouped FP8 MoE layout, padding, activation quant, and route-weight gather/reduce contracts.
  - `../vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py` and `deepep_v2.py` - DeepEP low-latency/v2 prepare-finalize contracts and where route weights are applied.
  - `../vllm/csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu` and `permute_unpermute_kernel.inl` - CUDA route reduction multiplies top-k weights while unpermuting expanded expert rows.
  - `../vllm/vllm/v1/attention/backends/mla/indexer.py` and `sparse_attn_indexer.py` - DSA indexer decode metadata, paged MQA logits, and top-k sequence.
  - `../vllm/csrc/libtorch_stable/cache_kernels.cu`, `cache_kernels_fused.cu`, and `sampler.cu` - CUDA cache/indexer helpers and sparse top-k helpers.
  - `../vllm/vllm/v1/attention/backends/mla/flashmla.py`, `flashmla_sparse.py`, and `ops/flashmla.py` - FlashMLA dense/sparse support and fp8 cache layouts.
- **Relevant history**:
  - `docs/models/kimi-k2/deepep-migration.md` - capacity-sized adapter kernels can erase DeepEP wins; GLM wrappers must use actual row counts or device counts.
  - `docs/models/glm52/support.md` - a local weighted activation-quant wrapper was first removed because it lacked source backing and NCU evidence, then deliberately reintroduced as a DeepGEMM TileLang-backed W2-input quant path with checkpoint IT and NCU evidence.
- **Plan**:
  1. Keep a stable source map for vLLM paths that matter to GLM52.
  2. Classify each path as direct CUDA/C++ copy, contract-only, Triton/Python rewrite, or deferred.
  3. Feed the GLM52 kernel ledger from this doc before implementing new kernels.

## Source Snapshot

| Source | Value |
| --- | --- |
| vLLM checkout | `../vllm` |
| Commit | `4d3b4b9b01efbca77872e3d4a568b273c7a245a7` |
| Branch | `main` |
| Local note | one unrelated untracked draft file was present: `pr-draft-runtime-lora-rust.md` |
| GLM mapping | `GlmMoeDsaForCausalLM` is registered to `deepseek_v2.py`, not a separate GLM5 model file |

## How To Use This Doc

Before adding a GLM52 kernel or wrapper, check the table below in this order:

1. **Direct copy/adapt** means vLLM has CUDA/C++ source with a usable ABI idea. Start there.
2. **Contract copy** means the implementation is Python/Triton/third-party glue; preserve shapes, layout, and weight semantics, then choose an OpenInfer-native substrate.
3. **Deferred** means it is outside first decode-only GLM52 or is hardware-mismatched.

If an operator falls through all known sources and becomes local CUDA, attach NCU evidence to `docs/models/glm52/support.md` before calling it ready.

## vLLM GLM52 Decode Shape

vLLM does not implement GLM5.2 as an isolated stack. `GlmMoeDsaForCausalLM` inherits `DeepseekV2ForCausalLM`, so the relevant execution shape is DeepSeek-style MLA/DSA attention plus `DeepseekV2MoE`:

| Stage | vLLM source contract | GLM52 implication |
| --- | --- | --- |
| Router logits | `GateLinear(hidden=6144, experts=256)` with optional F32 `e_score_correction_bias` under `topk_method=noaux_tc`. | Keep router logits F32 and noaux correction-bias semantics. |
| Top-k | `FusedMoE(... use_grouped_topk=True, scoring_func=sigmoid, renormalize=norm_topk_prob, routed_scaling_factor=...)`. | Do not rank by normalized weights; rank by biased sigmoid score, then normalize unbiased sigmoid weights. |
| Expert parallel | `FusedMoE` picks a prepare/finalize backend such as DeepEP LL/v2 or no-DP permute path. | For OpenInfer `DP8 TP1 EP8`, Kimi-style DeepEP owns communication; route-weight placement must match the chosen finalize contract. |
| Attention/indexer | DSA indexer metadata builds decode as `(B,1)` for non-MTP, with `AttentionCGSupport.UNIFORM_BATCH`. | This aligns with decode CUDA Graph and no MTP in first GLM52 cut. |

## Kernel Matrix

| Area | vLLM path | Classification | What to copy | GLM52 action |
| --- | --- | --- | --- | --- |
| GLM model semantics | `vllm/model_executor/models/deepseek_v2.py`, `glm4_moe.py` | Contract copy | F32 router, noaux bias, sigmoid, top-k renorm, routed scale, shared+routed expert structure. | Use as semantic truth source; OpenInfer runtime shape still follows Kimi DP8. |
| Router GEMM | `csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu`, `dsv3_router_gemm_float_out.cu`, `dsv3_router_gemm_bf16_out.cu` | Direct copy/adapt | GLM-5 instantiations for `hidden=6144`, `experts=256`, `tokens=1..16`, SM90+, BF16 input/weight, F32 or BF16 output. | This is the first performance candidate for small decode batches; existing GLM router should be compared against it before further tuning. |
| Router top-k | `csrc/libtorch_stable/moe/grouped_topk_kernels.cu` | Direct copy/adapt | Sigmoid/no-score modes, correction-bias ranking, group selection, unbiased weight output, renorm, routed scale, PDL hooks. | Candidate to replace or validate GLM router top-k. Keep Kimi-compatible noaux semantics if not copied immediately. |
| Generic top-k | `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`, `topk_softplus_sqrt_kernels.cu` | Direct copy/adapt | Softmax/sigmoid top-k kernels for non-grouped variants. | Not the first GLM route because GLM uses noaux grouped top-k. Keep as fallback reference. |
| DeepEP low-latency | `fused_moe/prepare_finalize/deepep_ll.py` | Contract copy | Supported hidden sizes include `6144`; `low_latency_dispatch`; `low_latency_combine(fused_expert_output, topk_ids, topk_weights, handle, out=output)` applies route weights in combine. | Good explanation for "why no standalone weight kernel" in vLLM LL. Not directly compatible with current GLM expanded elastic layout. |
| DeepEP v2 decode | `fused_moe/prepare_finalize/deepep_v2.py` | Contract copy | Decode mode uses `do_expand=False`, `do_cpu_sync=False`, bounded `num_max_tokens_per_rank`, globalizes local top-k ids on device, applies weight/reduce before `combine(topk_weights=None)`. | Candidate future layout if we switch away from current expanded `psum_expert` layout. It changes the layout contract, so do not mix it halfway with existing GLM substrate. |
| DeepGEMM grouped FP8 MoE | `fused_moe/experts/deep_gemm_moe.py` | Contract copy | `m_grouped_fp8_gemm_nt_contiguous` for W13/W2, `mk_alignment_scope(align_used)`, FP8 E4M3 static-128 weights + dynamic-128 activations, H200 supported. | First H200 routed expert GEMM target. Need OpenInfer wrapper around DeepGEMM, not a Python dependency in runtime. |
| DeepGEMM scatter/gather | `fused_moe/deep_gemm_utils.py` | Triton/Python rewrite | `compute_aligned_M_and_alignment`, per-expert aligned slices, `expert_ids=-1` for invalid rows, `inv_perm`, and `ep_gather` route-weight accumulation. | Current GLM DeepEP `psum_expert` already provides aligned expert-major rows; use this as a layout oracle, not direct code. |
| Route-weight reduce | `deep_gemm_utils.py::ep_gather`, `topk_weight_and_reduce.py`, `moe_permute_unpermute_op.cu` | Mixed: contract plus direct CUDA candidate | vLLM either weights inside DeepEP LL combine, weights+reduces contiguous expert output before combine, or uses CUDA `finalizeMoeRoutingKernel` for expanded rows. | Prefer source-backed route weighting through W2/GEMM/finalize/combine contract. A standalone local multiply is not the default answer. |
| W13 input quant | `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`, `quantization/input_quant_fp8.py` | Direct copy/adapt | BF16/FP16 -> FP8 E4M3 per-token group quant, group size 128, F32 or UE8M0 scale formats, PDL-capable packed path. | Existing GLM W13 quant substrate copied vLLM semantics; next step is NCU before performance claims. |
| W2 activation quant | `csrc/libtorch_stable/quantization/activation_kernels.cu`, `fp8_utils.py`, DeepGEMM `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` | Mixed | CUDA DeepGEMM activation kernel exists for batched expert format; Python/Triton/TileLang paths also define layout and route-weight semantics. | GLM now folds `recv_topk_weight` into W2-input SiLU*up FP8 quant using DeepGEMM TileLang semantics and profiles the local CUDA adaptation on H200. W13 input quant remains unweighted. |
| MoE permute/unpermute | `csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu`, `permute_unpermute_kernel.inl` | Direct copy/adapt | `moe_unpermute` multiplies top-k weights and reduces expanded expert rows back to original token rows. | Useful if GLM keeps an expanded-row output and needs a CUDA source-backed weight+reduce finalize. Need adapt to DeepEP expanded layout and graph buffers. |
| Marlin WNA16 MoE | `csrc/libtorch_stable/moe/marlin_moe_wna16/*` | Deferred for GLM FP8 | INT4/WNA16 expert path with in-kernel top-k weights. | Relevant as Kimi route-weight precedent only; GLM FP8 expert weights should use DeepGEMM or FP8 kernels. |
| MLA dense decode | `v1/attention/backends/mla/flashmla.py`, `v1/attention/ops/flashmla.py` | Use third-party wrapper/contract | FlashMLA dense supports Hopper, uniform decode metadata, fp8 dense helper exists. | Candidate for dense MLA portions if GLM cache layout matches. Need map GLM `kv_lora_rank=512`, `qk_rope_head_dim=64`, `v_head_dim=256`. |
| Sparse MLA decode | `v1/attention/backends/mla/flashmla_sparse.py`, `flashinfer_mla_sparse.py` | Use third-party wrapper/contract | Sparse MLA uses indexer top-k, `fp8_ds_mla` cache layouts, FlashMLA or FlashInfer sparse decode. | Candidate for GLM DSA decode. Prefer FlashMLA/FlashInfer before local attention kernels. |
| DSA indexer metadata | `v1/attention/backends/mla/indexer.py` | Contract copy | Non-MTP decode normalized to `(B,1)`, uniform batch CUDA graph support, block table/seq lens expansion, DeepGEMM paged MQA scheduler metadata. | Copy metadata shape into GLM decode graph design. Do not add token-by-token prefill paths. |
| DSA indexer logits | `model_executor/layers/sparse_attn_indexer.py`, `vllm/utils/deep_gemm.py` | Contract copy | `fp8_fp4_paged_mqa_logits` for decode, `fp8_fp4_mqa_logits` for prefill chunks, optional cooperative/persistent top-k. | Decode first: map q/indexer cache layout to DeepGEMM/FlashMLA source path. Prefill chunk path is for future P worker. |
| Indexer cache insert/gather | `csrc/libtorch_stable/cache_kernels.cu`, `cache_kernels_fused.cu` | Direct copy/adapt | `indexer_k_quant_and_cache`, `cp_gather_indexer_k_quant_cache`, `concat_and_cache_mla`, `concat_and_cache_mla_rope_fused`, `concat_mla_q`. | Strong CUDA source candidates for GLM indexer/MLA cache plumbing. Need verify cache byte layout against GLM52. |
| Fused QNorm/RoPE/KV insert | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | Direct copy/adapt with caution | DeepSeek V4 hard-coded constants: head dim 512, rope 64, NoPE 448, UE8M0 FP8 cache. Also has FlashInfer full-cache sibling. | Not directly GLM52-shaped if GLM uses `kv_lora_rank=512` plus `rope=64`; inspect before copying. Useful as a fusion pattern, not a drop-in. |
| Sparse indexer top-k | `csrc/libtorch_stable/sampler.cu`, `cooperative_topk.cu`, `persistent_topk` call sites | Direct copy/adapt | `top_k_per_row_decode` accepts 2D `(B,next_n)` seq lens; cooperative/persistent paths cover top-k 512/1024/2048 on CUDA. | GLM index top-k is `2048`; this is a strong source candidate before local top-k code. |
| Sampling | vLLM sampler paths and OpenInfer `openinfer-sample` | Reuse OpenInfer first | vLLM has rich sampler, but OpenInfer already has shared sampler/logprob helpers. | Reuse `openinfer-sample`; only inspect vLLM if GLM-specific logits/sampling behavior differs. |
| MegaMoE | `vllm/models/deepseek_v4/nvidia/model.py`, DeepGEMM MegaMoE APIs | Deferred | DeepGEMM MegaMoE requires SM100 and transforms FP8/FP4 scales/weights for fused dispatch+GEMM+combine. | Not first GLM52 H200 path. Keep as design reference for later Blackwell work. |

## Route-Weight Decision Boundary

Do not implement route weighting as an isolated local multiply unless all source-backed paths fail and the rewrite is profiled.

The vLLM evidence says route weights belong in one of these contracts:

| Contract | vLLM evidence | Fit for current GLM52 |
| --- | --- | --- |
| Combine applies weights | DeepEP LL `low_latency_combine(... topk_weights ...)` | Not compatible with current GLM expanded elastic combine; useful if GLM changes backend/layout. |
| Expert output is weighted+reduced before combine | DeepEP v2 `_finalize` uses `TopKWeightAndReduceContiguous`, then `combine(topk_weights=None)` | Compatible only if GLM uses contiguous non-expanded decode layout or adds a matching finalize step. |
| Gather/unpermute applies weights | DeepGEMM `ep_gather` and CUDA `moe_unpermute` multiply weights while reducing expanded rows | Best current source-backed candidate if GLM keeps expanded expert-major rows. Need adapt to `psum_expert` and persistent graph buffers. |
| W2 input quant applies weights | DeepGEMM TileLang `swiglu_apply_weight_to_fp8.py`, plus Kimi's expanded-layout precedent that expert compute emits already weighted rows | Current GLM52 choice for expanded DeepEP layout. Startup validates `recv_topk_weight` in routed rows and the decode graph replay; H200 NCU evidence lives in `profile/glm52_weighted_swiglu_quant_20260626/`. |

Current GLM52 direction: keep W13 input quant unweighted, apply route weights during W2-input SiLU*up FP8 quant using the DeepGEMM TileLang contract, run W2 grouped FP8 GEMM, then let DeepEP combine reduce rows that are already weighted. vLLM gather/reduce and DeepEP LL/v2 weighting remain reference contracts for a future layout change, not the current expanded `psum_expert` path.

## Attention / Indexer Decode Boundary

First GLM52 runtime is decode-only. For vLLM attention/indexer paths, keep this split:

| Path | Use now? | Notes |
| --- | --- | --- |
| `DeepseekV32IndexerMetadataBuilder` decode metadata | Yes | Non-MTP decode becomes `(B,1)` context lengths and supports uniform CUDA graph batches. |
| `sparse_attn_indexer` decode branch | Yes | Calls `fp8_fp4_paged_mqa_logits`, then cooperative/persistent/top-k decode. |
| `indexer_k_quant_and_cache` | Yes, if decode handoff writes indexer cache locally | Direct CUDA source candidate. |
| FlashMLA/FlashInfer sparse decode | Yes | Prefer these before local attention kernels. |
| Prefill chunk metadata and `fp8_fp4_mqa_logits` | Not in first runtime | Keep for future P worker. Chunk loops are acceptable in prefill workers; token loops are not. |
| Native MTP decode | Not in first runtime | vLLM supports 2D next-n shapes; GLM first cut should force next_n=1. |

## Copy Priority

1. Router GEMM/top-k CUDA: compare existing GLM router to vLLM `dsv3_router_gemm` + `grouped_topk`.
2. DeepGEMM grouped FP8 wrapper: expose `m_grouped_fp8_gemm_nt_contiguous` with existing GLM weight packages, activation scale layout, and weighted W2-input quant.
3. MoE route finalize: keep CUDA `moe_unpermute`/DeepGEMM `ep_gather` as fallback references if the expanded-layout W2-input weighting contract fails a future logits gate.
4. Indexer/cache helpers: copy/adapt `indexer_k_quant_and_cache`, `cp_gather_indexer_k_quant_cache`, and `concat_mla_q` before writing local indexer cache kernels.
5. Sparse MLA decode: wire FlashMLA/FlashInfer source paths once cache layout is proven.

## Open Questions

| Question | Why it matters | Next evidence |
| --- | --- | --- |
| Can GLM52 use vLLM's `fp8_ds_mla` cache byte layout as-is? | Determines whether FlashMLA sparse decode and fused cache insert are drop-in. | Compare GLM config/weights with vLLM cache assumptions: NoPE/RoPE split, scale bytes, block size, `v_head_dim`. |
| Should GLM keep current expanded DeepEP layout or switch to vLLM DeepEP v2 decode layout? | Determines route-weight and GEMM layout. | Prototype only after documenting exact buffer shapes and graph constraints; do not mix contracts. |
| Is vLLM router GEMM faster than current GLM router for buckets >16? | vLLM router GEMM only instantiates `tokens=1..16`; GLM decode bucket is 128. | Need microbench/NCU: bucket slicing may lose to current batched path. |
| Does DeepGEMM grouped FP8 beat FlashInfer/CUTLASS for GLM H200 routed experts? | Kernel choice should be measured in this repo. | Build one wrapper and run H200 NCU/bench before claiming a win. |
| Which sparse top-k path wins for `index_topk=2048`? | Indexer top-k can dominate sparse decode. | Compare vLLM CUDA top-k, cooperative/persistent top-k, and any FlashMLA/FlashInfer-provided path. |

## Debrief

- **Outcome**: vLLM GLM/MoE/DSA kernel paths are now indexed in one GLM52 doc, with direct-copy candidates separated from Python/Triton/third-party contracts.
- **Pitfalls encountered**:
  - `GlmMoeDsaForCausalLM` has almost no code of its own; the real source is the DeepSeek-V2/V3 stack.
  - DeepGEMM MoE looks like one backend from the outside, but route layout and weighting are mostly Python/Triton glue around third-party GEMM calls.
  - DeepEP LL, DeepEP v2, and current OpenInfer GLM expanded DeepEP have different route-weight contracts.
- **Follow-ups**:
  - Use `docs/models/glm52/vllm-moe-fp8-kernels.md` for the narrower FlashInfer CUTLASS vs DeepGEMM vs vLLM CUTLASS FP8 MoE decision.
  - Feed this table back into `docs/models/glm52/support.md` when selecting the first real routed expert GEMM/finalize path.
  - Add NCU-backed rows only after the corresponding OpenInfer wrapper exists and runs on jiuzhang H200.
