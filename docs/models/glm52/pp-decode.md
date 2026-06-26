# GLM5.2 PP8 Decode Exploration

> **TL;DR:** PP8 TP1 EP1 decode is now the **committed** GLM5.2 low-latency path; DP8/EP8 DeepEP is being dropped. 8 stages, 1 H200/stage, each stage owns a contiguous layer slice + all 256 routed experts for its sparse layers; stage boundary = BF16 hidden `[1,6144]` over a **graph-internal NVLink P2P handoff** (device-flag serialized, no stream/event edges). **Target: bs=1, TPOT < 10 ms** — purely HBM-bound (~42-50 GiB active/token / ~4.0-4.8 TB/s of one H200), so the budget is per-stage FP8-GEMM HBM efficiency, not communication (7-hop handoff ~16 us = noise). The **Build Plan** below is authoritative and supersedes the older hypothesis sections; it sequences work as Slices P->0->7. Two governing facts: **(1) Slice 0 = the PP runtime spine** (novel, load-bearing, independent of forward math); **(2) #1 correctness blocker = the MLA `q_b`/`kv_b` head-count factorization** (input side N=256, o_proj side N=64 -- a 4:1 fold the contract mis-states) -- no attention code until a node38 vLLM dump resolves it (Slice 1). bs=1 collapses MoE to top-8 active experts (G=8 grouped GEMM, no 256-group permute / no alignment padding), so DP/EP's Risk 6 alignment-blowup vanishes.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes this as a GLM52 model-line exploration doc.
  - `docs/models/glm52/support.md` - current bring-up target is decode-only `DP8 TP1 EP8` with real batched decode, full decode CUDA Graph, no prefill fallback, no MTP first cut, and no host loops over prompt tokens or requests.
  - `docs/models/glm52/vllm-kernel-reference.md` - GLM52 operator source map: router/top-k, DeepEP/DeepGEMM route contracts, DSA indexer/cache/top-k, and FlashMLA/FlashInfer decode candidates.
  - `docs/models/glm52/vllm-moe-fp8-kernels.md` - FP8 MoE backend map: FlashInfer CUTLASS, DeepGEMM, vLLM CUTLASS, and the current TRTLLM grouped-offset substrate.
  - `docs/playbooks/model-optimization-pipeline.md` - keep roofline/e2e/profiling evidence in the per-model doc, and do not claim wins before A/B data.
  - `docs/models/kimi-k2/dp-design.md` - current DP runtime precedent uses a coordinator plus per-rank engines; PP should preserve the same "GPU-owning worker thread" discipline, but stage dependency replaces DP independence.
  - `/data/code/tilert_play/glm5_tpot_pp_tp_估算.md` - copied the user's PP/TP roofline and H200 P2P handoff measurements into this project doc.
- **Relevant history**:
  - `docs/models/glm52/support.md` - current GLM52 work already rejected hidden prefill paths, `for token in prompt` loops, and `for req in bs1` decode loops; PP must keep the same red lines.
  - `docs/models/kimi-k2/dp-design.md` - Kimi's TP1/DP8 design removed layer-level TP all-reduce by making ranks independent; PP8 removes collectives differently, by serial layer partitioning plus small hidden transfers.
  - `docs/lessons/kimi-bringup-numerics.md` - MoE+TP correctness is sensitive to reduce precision and finalize placement; PP's local MoE path still needs a logits gate before serving.
- **Plan**:
  1. Preserve the user's roofline and H200 P2P experiment data in this doc as the baseline reasoning for PP8.
  2. Add the current OpenInfer GLM52 implementation state and missing kernel list, separated from the PP hypothesis.
  3. Define the first PP8 decode shape: stage-owned layer slices, fixed graph buffers, graph-internal P2P hidden handoff, no engine-side prefill, no MTP initially.
  4. Classify existing GLM52 kernels by whether they migrate unchanged, require shape/layout parameterization, or need new PP-specific kernels.
  5. Record the next measurements needed before committing to PP as a production direction.
- **Risks / open questions**:
  - PP8 low latency can still lose if single-stage HBM reads dominate and TP's graph-fused communication edge cost is already small.
  - The "all 256 experts per stage layer" plan must be checked against H200/B200 memory with real GLM52 FP8 weights and KV budget.
  - Local MoE finalize is a new correctness surface even though W13/W2 GEMM kernels mostly carry over.

## Scope

This doc tracks a low-latency **PP8 decode** branch for GLM5.2. It is not replacing the current `DP8 TP1 EP8` DeepEP bring-up until measurements say it should.

The shared per-layer decode DAG, tensor shapes, source maps, and first implementation split now live in `docs/models/glm52/decode-forward-contract.md`. PP8 should reuse that math contract and change only placement/runtime: stage-local layer slices, all 256 experts per stage layer, local MoE finalize, and graph-internal hidden handoff.

| Item | PP8 first-cut rule |
| --- | --- |
| Objective | Single-request / small-batch decode latency, especially TPOT without DFlash/MTP. |
| Parallelism | `PP8 TP1 EP1` candidate: 8 pipeline stages, one GPU per stage, no NCCL all-to-all, no tensor-parallel all-reduce. |
| Expert placement | Each stage holds all 256 routed experts for the sparse layers assigned to that stage. |
| Prefill | Out of scope inside GLM engine. Decode receives prefilled KV/page/indexer state from future P/D handoff. |
| MTP | Out of scope for the first PP decode branch; GLM5.2 has built-in MTP, but base decode must be measured first. |
| Batch shape | `bs > 1` stays real batched tensor work. No host loop over bs=1 requests. |
| CUDA Graph | Required: per-stage fixed-buffer graph replay plus graph-internal P2P hidden handoff. |
| Frontend | Reuse Qwen3/Kimi request semantics; PP affects model runtime, not OpenAI API behavior. |


## Build Plan (authoritative -- 2026-06-26)

Synthesized from a 6-agent code-grounded grounding pass (rust + kernels inventory, forward-op ABI verification, sibling reuse, p2p protocol). This **supersedes** the former *Current State / Missing Work / PP8 Shape / Migration Map / First Measurements* hypothesis sections (removed). The roofline/`L_send` reasoning that motivated PP is retained as the appendix below.

### bs=1 reconciliation (READ FIRST -- this doc targets bs=1; the plan tables below are bs=128-framed)

The active goal fixes **bs=1, TPOT < 10 ms**. Three deltas override the bs=128 assumptions baked into the plan's MoE/arena rows and Risk 6:

- **MoE collapses to top-8 active.** At bs=1 the router selects 8 of 256 experts, each seeing exactly 1 token. No 256-group permute, no `psum_expert`, no per-expert alignment padding. Slice 5 becomes: router top-8 -> **G=8 grouped FP8 GEMM whose 8 problems point at the 8 selected experts' resident weights** (no weight copy -- the 256-expert weights stay resident, only 8 are *read*) -> 8-row weighted reduce into `[1,6144]`. **Risk 6 (alignment 32/64, m_capacity blowup) is a bs=128 problem and does not apply.** Expert *residency* (memory, 256/stage) is unchanged; only the per-token *read set* shrinks to 8.
- **Arena is `[1, ...]` everywhere.** `batch_cap` 128->1; MoE expanded buffers size to 8 active rows, not 10240/8960; `decode_meta` geometry is one row.
- **Scheduler is a single-request epoch loop.** No bucket, no padding, no active-mask, no per-rank row routing. Coordinator admits 1 request, advances epoch `t` across the 8 stage graphs, stage7 samples token `t+1`, loops to EOS/max_tokens. Prefilled KV/indexer state seeded from a fixture or a one-shot prefill (out of the hot loop) for bring-up.

Everything else in the plan (KEEP/ADAPT/CLEAN inventory, Slice 0 spine spec, the MLA/indexer/cache risks) is bs-independent and stands as written. Where a plan row cites a bs=128 capacity (`m_capacity=8960`, `expert_offsets len 257`), read it as the bs=1 equivalent (8 active rows, 8 problems).

**Initial static layer partition** (balance by active bytes; honor the indexer-boundary constraint; retune from Slice 7 per-stage time -- Risk 3). Full-indexer layers are 0,1,2,6,10,...,74; **prefer stage boundaries that do not split a shared-indexer layer from its full-indexer owner**, else the handoff must also carry indexer cache state.

| stage | layers | count | extra load |
|---|---|---:|---|
| 0 | 0-10  | 11 | embed + dense 0-2 |
| 1 | 11-20 | 10 | |
| 2 | 21-30 | 10 | |
| 3 | 31-40 | 10 | |
| 4 | 41-50 | 10 | |
| 5 | 51-60 | 10 | |
| 6 | 61-70 | 10 | |
| 7 | 71-77 | 7  | final_norm + lm_head (~1.9 GiB ~ +3 sparse-layer-equiv) |

## GLM5.2 PP8 TP1 EP1 Decode — Build Plan

This is the authoritative build plan synthesized from the five grounding reports (rust inventory, kernels inventory, forward-op sources, sibling reuse, p2p protocol). It supersedes the PP8 *hypothesis* section of this doc: PP8 is now the committed path and DP8/EP8 DeepEP is being dropped.

Two facts govern everything below:
- **The novel, load-bearing architecture is the PP runtime spine** (8 single-stream CudaGraphs serialized by *device-memory flags*, not stream/event edges). It is independent of forward correctness, so it is **Slice 0**.
- **The #1 correctness blocker is the MLA head-count factorization** (`q_b`/`kv_b` imply N=256 heads on the input side but N=64 on the o_proj side — a 4:1 fold the contract does not capture). No attention code may be written until a node38 vLLM dump resolves it (Slice 1).

---

### 1. KEEP / ADAPT / CLEAN — consolidated inventory

`CLEAN` = delete with DP8/EP8. `ADAPT` = keep the unit, make the cited edit. `KEEP` = carries to PP unchanged. `NEW` = must be written (listed for completeness; detail in §2/§3).

| Subsystem | Unit (file:line / symbol) | Verdict | Concrete action |
|---|---|---|---|
| **lib.rs** | `validate_startup` (`lib.rs:131`) asserts `tp1&&dp8` (`:141`), `ep_backend==DeepEp` (`:148`), `device_count==ep_world` (`:151`) | ADAPT | Replace all three asserts with "8 devices, one per stage; stage_count==device_count". Drop DeepEP requirement. |
| lib.rs | `Glm52LaunchOptions{tp,dp,ep_backend}` (`:56`), `launch` (`:64`), `probe_model` (`:33`) | ADAPT | Keep EngineHandle/probe surface; replace shape with `Glm52PipelineShape` (stage_count + per-stage layer slice). |
| lib.rs | `shape_from_parallel` (`:293`), `StartupValidation.deepep_decode_worst_*` (`:116-117`), 6 `runner::smoke_*` calls (`:268-274`), `ep_backend==DeepEp` plumbing | CLEAN | Pure DP/EP-shaped; delete. |
| lib.rs | `load_rank_weights_to_gpu` spawn+async-load+join (`:202`), `run_rejecting_dp_coordinator` spawn (`:100`) | ADAPT | Skeleton is the stage-worker bring-up shape; body rewritten (see runner). |
| **config.rs** | all model-dim consts, `probe_config_json` + `ensure_*` (`:48-324`) | KEEP | Parallelism-independent checkpoint validation. |
| config.rs | `Glm52ParallelShape::new` (`:343`) | ADAPT/reuse | `new(1,1)` already yields `local_experts=256` (EP1, G=256) — reuse for the per-stage MoE shape. |
| config.rs | `Glm52ParallelShape::tp1_dp8` (`:338`) | CLEAN | DP-shaped. |
| config.rs | — | **NEW** | Add `Glm52PipelineShape{ stage_count=8, stage_layer_ranges:[(start,end);8], experts_per_stage=256, owns_embedding(stage0), owns_head(stage7), per-stage indexer-layer set }`. Stage0 slice **must** contain dense layers 0..3. |
| **deepep.rs** (model crate) | `GLM52_EP_WORLD=8` (`:12`), `GLM52_LOCAL_EXPERTS=32` (`:13`), `Glm52DeepEpShape`, `decode_capacity`/`worst_recv_tokens`/`src_metadata_len`/`rank_count_len` (`:68-96`) | CLEAN | All-to-all shape contract; delete. |
| deepep.rs | `GLM52_DEEPEP_EXPERT_ALIGNMENT=64` (`:14`), `_DEVICE_SMS=132` (`:15`), `_DECODE_BATCH_CAP=128` (`:16`), `expanded_rows_for_recv_tokens` (`:80-86`) | ADAPT | Rename (drop "DEEPEP"). Keep as the decode bucket/alignment + *starting point* for local-permute capacity — **re-derive, do not port** (slack blows up at G=256, see arena). |
| **arena.rs** | group-A general scratch (`hidden`,`normed`,`q_a..attention_out`,`indexer_*`,`shared_*`,`router_*`,`topk_*`) | KEEP | Sized by `batch_cap=128`+model dims; per-stage trim only (no `logits` except stage7, no `dense_*` except stage0). |
| arena.rs | group-B MoE-expanded buffers (`deepep_recv_x`,`moe_w13_*`,`moe_w2_*`,`moe_gemm_expert_offsets`,`moe_*_problem_sizes`,`deepep_recv_topk_weight`,`deepep_combined`) (`:320-346`) | ADAPT | Resize to PP capacity; `expert_offsets`→len 257, `problem_sizes`→len 768; `moe_trtllm_grouped_offset_rows = glm52_trtllm_grouped_offset_padded_rows(pp_m_capacity, 256)`. |
| arena.rs | `deepep_recv_src_metadata` (`:345`) + `deepep_src_metadata_len` plan field | CLEAN | DeepEP combine source map; replaced by local permute index map (`src_token`). |
| arena.rs | `Glm52DecodeArenaPlan::tp1_dp8` (`:56`), all `seed_*`/`validate_*_smoke` (`:363-784`) | CLEAN/ADAPT | Plan builder → pipeline builder; `validate`/`validate_allocations`/`allocated_bytes` track resized buffers; smokes deleted; keep `align_up`, `deepgemm_scale_layout_valid`, `trtllm_grouped_offset_scale_layout_valid` as references. |
| **runner.rs** | `Glm52RankWorker::spawn` ctx-bind + handshake + `Drop` (`:86,92,104,261`) | KEEP (discipline) | Thread-per-GPU, bind-before-touch, crash-early handshake. Rename rank→stage. |
| runner.rs | `Glm52RankCommand` DeepEP variants (`:50-84`), `install_deepep_backends`/`unique_id` (`:521`), all `smoke_*`/`validate_moe_gemm_contracts` (`:357-911`) | CLEAN | DeepEP/NCCL bootstrap + smokes; delete. |
| runner.rs | `run_rejecting_dp_coordinator` (`:913-928`) | ADAPT (rewrite) | DP premise (8 *symmetric independent* engines) → PP (8 *asymmetric* serial stages). Coordinator owns no GPU tensors; `send_scheduled` (`:930`) keeps. |
| runner.rs | `Glm52RankThreadState`/`LoadedState` (`:267,275`), `deepep` field (`:272`) | ADAPT | State container keeps; drop `deepep`; arena → per-stage. |
| **decode_meta.rs** | `Glm52DecodeBatchGeometry/HostMetadata/Metadata` (`:60,185,252`) | KEEP | Per-stage paged-decode contract for bs=128. Rename `GLM52_DEEPEP_*` consts (`:21`). |
| **moe_deepep.rs** | `launch_decode_moe_layer_substrate` (`:800-877`) — router/quant/W13/W2 steps 1,3,4,5,6 | KEEP | The MoE GEMM core between the two seams is reused verbatim. |
| moe_deepep.rs | `self.ep.decode_dispatch` (`:829`) | **NEW (replace)** | Local route-permute kernel emitting `psum_expert` directly (§ kernels). |
| moe_deepep.rs | `self.ep.decode_combine` (`:865`) | **NEW (replace)** | Local weighted gather/reduce (weights already folded in W2 quant). |
| moe_deepep.rs | `Glm52MoeDeepEpState`, `new()` NCCL handshake, `decode_*_smoke_roundtrip` | CLEAN | Delete; keep `run_or_capture` graph-wrap pattern (`:714-730`). |
| **moe_deepep/types.rs** | `Glm52MoePsumLayout/Snapshot`, all `*SmokeReport`, `Glm52DeepEpEnableReport` | CLEAN | DeepEP recv-layout; delete. Keep `deepgemm_psum_compatible`, `align_up`. |
| **moe_gemm.rs** | `validate_w13/w2_contract` (`:131,175`), `launch_trtllm_w13/w2_grouped_fp8` (`:260,286`) | ADAPT | Flip `local_experts` 32→256; `m_capacity`→PP capacity; rename `ExpandedDeepEpGroupedFp8Contract`. **Kernels are already G-parameterized** — Rust-constant + buffer-size change only. |
| **linear.rs** | `launch_projection_smoke`/`projection_contract` (`:138,215`) | KEEP | Plain FP8 linear from per-projection `[n,k]`; zero DP/EP coupling. Used by q_a/q_b/kv_a/kv_b/o_proj/dense/shared. |
| linear.rs | `decode_attention_projection_smoke` (`:34`), `seed_linear_smoke_input` | CLEAN | Smoke harness. |
| **weights/{package,view,load}.rs** | `local_expert_range` slicing (`weights.rs:143,307`; `view.rs:188`) | ADAPT | **Invert**: expert-range → `stage_layer_range`; load all 256 experts for the stage's sparse layers only. |
| weights | `rank_tensor_names` (`weights.rs:319`), `validate_non_expert_weight_contract` (`view.rs:216`) | ADAPT | Iterate only the stage's layer slice; embedding only stage0 (`:321`), final_norm+lm_head only stage7 (`:322-323`); per-stage expected tensor counts. |
| weights | `all_rank_load_bundles` cross-rank equal-count assert (`weights.rs:394-399`); `validate` asserts `layers.len()==GLM52_MOE_LAYERS` (`package.rs:648`, `load.rs:144`) | ADAPT (**remove the equality**) | PP stages are deliberately asymmetric (different layer counts, indexer cadence). This assert crashes the *correct* PP plan — delete it with the slicing inversion. |
| weights | `pack_expert_major_w13/projection_fp8_buffers` (`package.rs:410,334`), `context.rs Glm52RankGpuContext` | KEEP | Packing is G-agnostic (packs whatever experts it gets, contiguous); `local_experts=projections.len()`→256. Context = cleanest reuse, PP-agnostic. |
| weights | manifest parsing (`from_index_json`, `attention_manifest`, `moe_layer_manifest`, `expected_tensor_contract`) | KEEP | Pure checkpoint→manifest; `with_parallel_shape` (`:248`) carries pipeline shape. |
| **Kernels — KEEP unchanged** | `glm52_router.cu` (256-expert, top-8), `glm52_moe_quant.cu` W13/W2 quant, `glm52_deepgemm_layout.cu:13` MN-major, `glm52_trtllm_grouped_fp8.cu:231` plain FP8 linear, `glm52_flashmla_sparse.cu`, `glm52_indexer.cu` | KEEP | No G/capacity/EP coupling. W2 quant folds `topk_weight` (`glm52_moe_quant.cu:80`) — rewire its weight *source* to the local-permute weight array. |
| **Kernels — ADAPT (G=256)** | `glm52_deepgemm_grouped.cu` host guard `groups==32&&m_cap==10240` (`:141`,`valid_common :86`); `glm52_trtllm_grouped_fp8.cu` `valid_shape/valid_contract` (`:62,101`); `deepgemm_grouped_offset` relayout call (G=256); Rust consts `GLM52_*_{LOCAL_EXPERTS,M_CAPACITY,OFFSET_ROWS}` (`deepgemm_grouped.rs:10`, `trtllm_grouped.rs:10-12`) | ADAPT | ~9 magic-number sites: relax 32→256, recompute m_capacity/offset_rows (formula in §4 risk 6). Kernel *bodies* are G-generic. |
| **Kernels — CLEAN** | `csrc/glm52_deepep/*` (3 files), `src/ops/glm52/deepep.rs`, `src/ffi/glm52/deepep.rs` + re-exports (`ops/glm52.rs:11-13`, `ffi/glm52.rs:4,7`) | CLEAN | Delete the DeepEP shim entirely. |
| **build.rs** | `is_glm52_deepep_source` (`:408`) + uses (`:1290,1399`); `deepep_enabled = kimi||glm52` (`:1237`); `deepep_nccl_root`/`link_deepep_nccl` (`:435,491,1719`); DeepEP include + `-DEP_NUM_TOPK_IDX_BITS=32` (`:1399-1423`) | ADAPT | Set `deepep_enabled = kimi_k2_enabled` only → glm52 build becomes **NCCL-free, DeepEP-free, no `OPENINFER_NCCL_ROOT`**. **KEEP `cargo:rustc-link-lib=cuda`** (`:1713`) — TRTLLM/router need `CUresult` and P2P needs the driver peer API. |
| **Kernels — NEW** | `glm52_route_local.cu` (permute), `glm52_combine_local.cu` (reduce), `glm52_pp_p2p.cu` (send/wait/dummy_burn) | NEW | Land in `csrc/glm52/` → auto-compiled `--std=c++17`, no NCCL (`build.rs:1523`). |
| **Forward glue — NEW** | 656-byte cache append (`concat_and_cache_mla`), indexer 132 cache (`indexer_k_quant_and_cache`), indexer paged-MQA logits (DeepGEMM — risk), indexer top-2048 (`persistent_topk`), RoPE interleaved θ=8e6, v_up bmm, q-absorb | NEW | Port from vLLM `cache_kernels.cu`/`topk.cu` + FlashMLA. Detail in §2 Slices 3–4. |
| **Sibling reuse — KEEP** | `norm.rs:143 rms_norm_batch_into` (untyped, width-generic); `cuda_graph.rs:35 run_or_capture`; `linear.rs:327 gemm_graphsafe_into_checked` (lm_head); `openinfer-sample:114 select_batch`; `EngineHandle`/`TokenEvent` (`openinfer-engine/src/engine.rs:333,138`) | KEEP | Use *untyped core* wrappers, **not** Kimi const-generic `typed_ops` (no `DIM=6144`). `select_batch` replaces Kimi's cross-rank top-1 (PP stage7 holds whole vocab). |

---

### 2. PP8 build sequence (dependency-ordered)

DAG: **P → {0, 1, 2}** can all start immediately and run in parallel. **3** needs {1,2}; **4** needs {1,2,3}; **5,6** need {2}; **7** needs {0,3,4,5,6}.

#### Slice P — DP/EP deletion-only prep (do first, recommended separate commit)
- **Deliverable:** glm52 builds NCCL-free and DeepEP-free; `cargo build -p openinfer-glm52 --features glm52` green; no DeepEP symbols remain.
- **Files:** delete `csrc/glm52_deepep/*`, `ops/glm52/deepep.rs`, `ffi/glm52/deepep.rs` + re-exports; strip DeepEP from `deepep.rs`/`moe_deepep.rs`/`moe_deepep/types.rs`/`runner.rs`/`lib.rs`/`arena.rs`; `build.rs:1237` → `kimi_k2_enabled`.
- **Reuses:** nothing new — pure subtraction.
- **node38 verify:** clippy + build green; `grep -rn "deepep\|DeepEp\|GLM52_EP_WORLD\|decode_dispatch" openinfer-glm52 openinfer-kernels/src/{ops,ffi}/glm52*` returns only renamed-survivor consts. (Per memory: test-removal needs no `cargo test`; clippy is the gate.)
- **Blocker:** none. **Caveat:** stub the two MoE all-to-all seams (`moe_deepep.rs:829,865`) with `unimplemented!()` so the file compiles while unreachable (coordinator still rejects). **Keep** the router/quant/GEMM helper fns. Do **not** put any G=256/capacity/slicing edits here — those ship with their feature slices.

#### Slice 0 — PP runtime spine (novel architecture; independent of forward math)
- **Deliverable:** 8 stage-worker threads + coordinator; per-stage single-stream CudaGraph containing `wait → dummy_burn → send`; device-flag-serialized cycle `0→…→7→0`; crash-early on timeout/lap.
- **Files:** `openinfer-glm52/src/pp/{peer.rs,stage_graph.rs,runtime.rs}`, `csrc/glm52/glm52_pp_p2p.cu`, `ffi/glm52.rs`, `ops/glm52/pp_p2p.rs`, `tests/pp_p2p_spine.rs`. Full spec in §3.
- **Reuses:** `runner.rs:86` spawn/handshake/`Drop` spine; `weights/context.rs Glm52RankGpuContext`; `cuda_graph.rs:103` capture (with the `capture_only` fix); cudarc 0.19.3 driver API (`cuCtxEnablePeerAccess`, `device_ptr→u64`).
- **node38 verify:** `pp_p2p_spine.rs` prints per-hop p50/p99/p999 CSV. **Pass bar:** 7-hop (pp=8) 12KB handoff ≈16us total, p99 within a few % of p50, **zero samples >100us over ≥50k iters**, 500us-dummy cells show handoff cost unchanged (hidden behind compute).
- **Blocker:** node38 8×H200 access; assert `cuDeviceCanAccessPeer==1` + UVA at setup (NV18 always 1, but assert).

#### Slice 1 — MLA layout probe resolution (a measurement, not code; gates all attention)
- **Deliverable:** the head-count + page-format contract nailed: real `num_heads`, `W_UK_T.shape`, `W_UV.shape`, and the `q.shape` (h_q) FlashMLA actually receives; the 256→64 o_proj fold mechanism. Correct `decode-forward-contract.md` (its `q[B,1,64,576]` and `W_UV[64,512,256]` are **wrong** — input side proves N=256).
- **Files:** node38 dump script only (load layer 0 via vLLM `GlmMoeDsaForCausalLM`, print `mla_attn.impl.num_heads` + the three shapes + the tensor passed to `flash_mla_sparse_fwd`). Update the contract doc.
- **node38 verify:** the dump *is* the artifact. Cross-check against `mla_layout_probe.rs:75-96` (independently found 4:1 expansion over 64 config heads).
- **Blocker:** vLLM env + checkpoint on node38. Runs fully parallel with Slice 0/P.

#### Slice 2 — Weights load-plan inversion (layer-slice sharding; gates 3/5/6)
- **Deliverable:** each stage loads its contiguous layer slice + **all 256 experts** for its sparse layers; stage0 adds embedding+dense, stage7 adds final_norm+lm_head.
- **Files:** `config.rs` (`Glm52PipelineShape`), `weights.rs`/`view.rs`/`package.rs`/`load.rs` (invert `local_expert_range`→`stage_layer_range`, drop cross-rank equality assert `:394-399`, per-stage expected counts).
- **Reuses:** manifest parsing, `pack_expert_major_*` (G-agnostic), `context.rs`, the async-load+pack-and-evict pattern (`load.rs:39`).
- **node38 verify:** load all 8 stages; assert per-stage tensor counts; **measure per-stage residency** (256 experts × ~9-10 layers; W13 ≈ 6.4 GB/stage + scales — this is the flagged must-measure). Memory must fit one H200 per stage with KV + arena headroom.
- **Blocker:** stage→layer table decided (§4 risk 3).

#### Slice 3 — Per-stage MLA attention forward, fixture indices (needs 1,2)
- **Deliverable:** one sparse-MLA decode layer end-to-end on **fixture top-k indices** + seeded 656 cache, producing o_proj output matching vLLM.
- **Operators:** 656-byte cache append (`concat_and_cache_mla`, `cache_kernels.cu:842` — layout proven, scale arg unused in ds_mla path); RoPE interleaved θ=8e6 (new kernel, **interleaved not neox**); FlashMLA sparse decode (`sparse_decode.h:183` → raw `SparseAttnDecodeParams` launcher, pre-allocate 6 scratch tensors in arena); q-absorb (`q_nope×W_UK_T`) + v_up bmm (`latent[B,N,512]×W_UV→[B,N,256]`, cuBLASLt strided-batched); o_proj (plain FP8 linear, `linear.rs` KEEP).
- **Reuses:** `rms_norm_batch_into` (q_a/kv_a/post norms); Kimi rope *apply* as algorithm template (but GLM head_dim=256 ≠ Kimi 192 → GLM split variant); Kimi `build_slot_page_table` host logic.
- **node38 verify:** seed identical 656 cache + identical fixture indices in both engines; diff FlashMLA latent `[B,N,512]` (atol ~1e-2 bf16); byte-compare one packed 656 entry; diff v_up vs `torch.bmm`.
- **Blocker:** Slice 1 head count (gates h_q=64 vs 128×2-loop, W_UV shape, o_proj assembly); cache-format decision (§4 risk 2 — packed-656).

#### Slice 4 — Indexer / real sparse selection (needs 1,2,3)
- **Deliverable:** replace fixture indices with real DSA top-2048.
- **Operators:** indexer k_quant+132-cache (`indexer_k_quant_and_cache`, `cache_kernels.cu:1461`); indexer k_norm = **LayerNorm(+bias, eps 1e-6)** — `view.rs:111` carries `k_norm_bias`, do **not** route through RMSNorm; indexer RoPE (interleaved, 64-dim); **indexer paged-MQA logits** (`fp8_fp4_paged_mqa_logits` — the DeepGEMM hard gap, §4 risk 5); top-2048 (`persistent_topk`, `topk.cu:233`, bs=128>32 so persistent not cooperative).
- **node38 verify:** top-2048 **set-equality** vs `persistent_topk` (ties may reorder); logits diff vs `fp8_fp4_paged_mqa_logits` on identical q_fp8/cache/weights (this diff is also the DeepGEMM build-feasibility test).
- **Blocker:** the (g) backend decision (raw-DeepGEMM-JIT vs local-CUDA-rewrite) — **decide early**; Slice 3's fixture path decouples attention bring-up from it.

#### Slice 5 — MoE local route permute/combine (replaces DeepEP; needs 2, parallel to 3/4)
- **Deliverable:** stage-local MoE over 256 experts: permute → metadata → W13 → SwiGLU+W2 quant → W2 → combine, replacing both `self.ep.*` seams.
- **Operators:** `glm52_route_local_permute` (emits `psum_expert` byte-compatible with `glm52_deepgemm_grouped.cu:46-55` + `src_token` map); `glm52_combine_local_reduce` (pure scatter-add — weights already folded by W2 quant). Keep router/metadata/W13/W2 GEMM untouched. Apply G=256 + capacity recompute (arena, moe_gemm, kernel guards).
- **Reuses:** router, both quant kernels, both TRTLLM grouped GEMMs, `run_or_capture` graph wrap.
- **node38 verify:** seed hidden + router weights; diff one MoE layer output vs vLLM; measure realized expanded-row count vs `m_capacity` (validate the alignment choice, §4 risk 6).
- **Blocker:** `expert_alignment` / `m_capacity` decision (32 vs 64).

#### Slice 6 — Dense + head + sampler glue (needs 2, parallel)
- **Deliverable:** stage0 dense layers (0..3, SiLU*up=weight-1 variant of W2 quant + FP8 down_proj); stage7 final_norm + lm_head + sampler.
- **Operators:** dense SiLU*up (reuse MoE quant w/ weight=1); `gemm_graphsafe_into_checked` lm_head `[6144→154880]`; `select_batch` greedy.
- **Reuses:** all four are direct sibling reuse (untyped core wrappers + `openinfer-sample`).
- **node38 verify:** stage0 dense-layer diff; stage7 logits + greedy token vs vLLM; logprobs D2H stays graph-after.
- **Blocker:** none beyond Slice 2.

#### Slice 7 — Full PP8 graph + TPOT (needs 0,3,4,5,6)
- **Deliverable:** real stage layers replace `dummy_burn`; full cycle decodes a seeded prefill; golden-logits gate across all 78 layers; TPOT measured.
- **node38 verify:** end-to-end golden-logits gate vs vLLM `GlmMoeDsaForCausalLM` (mean + p99 |Δlogprob|, per the qwen3/kimi methodology); TPOT p50/p99; per-stage time calibration to retune the layer-slice table (§4 risk 3).
- **Blocker:** all prior slices.

---

### 3. Slice 0 — concrete spec (build this first)

#### Files / symbols to create
| Artifact | Path | Contents |
|---|---|---|
| Kernels | `openinfer-kernels/csrc/glm52/glm52_pp_p2p.cu` | `glm52_pp_send_hidden`, `glm52_pp_wait_hidden`, `glm52_pp_dummy_burn` (auto-compiled `--std=c++17`, no NCCL) |
| FFI | `openinfer-kernels/src/ffi/glm52.rs` | `unsafe extern "C"` block (mirror existing `:10-30`) |
| Op wrappers | `openinfer-kernels/src/ops/glm52/pp_p2p.rs` | `device_ptr`/`cu_stream` plumbing (mirror `router.rs:142-185`); re-export from `ops/glm52.rs` |
| Peer setup + rings | `openinfer-glm52/src/pp/peer.rs` | `cuCtxEnablePeerAccess`, VA table, persistent ring buffers |
| Stage graph | `openinfer-glm52/src/pp/stage_graph.rs` | **`capture_only`** CudaGraph variant (GLM-local copy of `CudaGraphState`) |
| Runtime spine | `openinfer-glm52/src/pp/runtime.rs` | clone `runner.rs` thread-per-GPU spine; cycle ordering |
| Measurement IT | `openinfer-glm52/tests/pp_p2p_spine.rs` | node38-gated; CSV output |

#### Per-stage persistent buffers (allocated once on the stage's own context, never freed)
`hidden_in_ring: CudaSlice<bf16>` `[R·128·6144]` (peer-writable input) · `flag_ring: CudaSlice<u64>` `[R]` (pad each to 16 u64, false-sharing) · `epoch_counter: CudaSlice<u64>` `[1]` · `ack_ring: CudaSlice<u64>` `[R]` (reverse edge, WAR gate + RTT) · `err_code: CudaSlice<u32>` `[1]` (0=ok) · `deltas: CudaSlice<u64>` `[N_iters]` (globaltimer samples). **R=2** (double-buffer; serial bs=1 never blocks). `slot = epoch % R`.

#### Peer-access enable (coordinator-orchestrated, after all 8 contexts+buffers exist)
For each edge `i→i+1`, its reverse, and the cycle edge `7→0`: on the **producer** thread with its context current — assert `cuDeviceCanAccessPeer==1` and `UNIFIED_ADDRESSING==1` (else `bail!`), then `cuCtxEnablePeerAccess(ctx[neighbor].cu_ctx(), 0)` accepting `SUCCESS | PEER_ACCESS_ALREADY_ENABLED`. Read each peer's `hidden_in_ring`/`flag_ring`/`ack_ring` base VA once via `device_ptr(&stream)→u64`; store in a per-stage `Glm52PeerEdge{ down_hidden, down_flag, up_flag, down_ack: u64 }`. VAs are allocation-stable → safe to bake as captured-graph kernel immediates. (First peer-access user in the repo — lift the protocol from `/data/code/tilert_play/benchmarks/p2p_lsend/p2p_lsend.cu`, driver-API not runtime-API.)

#### Protocol
**Send** `glm52_pp_send_hidden <<<1,256>>>(src_hidden, peer_hidden, peer_flag, my_epoch, words, slot)`:
1. `e = *my_epoch` (thread0, broadcast via `__syncthreads`).
2. (WAR gate, for R≥2 pipelining) spin `while(down_ack[slot] < e)` — non-blocking at bs=1.
3. Vectorized peer store: cast to `int4` (128-bit, 8 bf16/store), strided `peer_hidden4[k]=src_hidden4[k]`. Measure 64-bit vs 128-bit (§ sweep) — 128-bit is the proposed upgrade over the harness's volatile 64-bit.
4. `__syncthreads(); __threadfence_system();` (payload globally visible before flag).
5. thread0: `peer_flag[slot]=e; __threadfence_system();` (release).

**Wait** `glm52_pp_wait_hidden <<<1,32>>>(my_flag, my_epoch, up_ack, err_code, deadline_ns, slot)` — receiver does **not** copy (payload already in its local ring slot), only gates + acquire-fences:
1. thread0: `e = atomicAdd(my_epoch,1)+1` (implicit epoch = replay count; stages i/i+1 stay lockstep).
2. spin `while((v=my_flag[slot]) < e)`; if `globaltimer()-start > deadline_ns` → `err_code=1; __trap()`.
3. `if (v != e) { err_code=2; __trap(); }` (producer lapped consumer → ring overwrite).
4. `__threadfence_system();` (acquire — subsequent in-stream RMSNorm read sees payload).
5. thread0: `up_ack[slot]=e; __threadfence_system();` (reverse ack → RTT + WAR release).

`__trap()` → sticky context error → coordinator's next `stream.synchronize()` returns `CUDA_ERROR_*` → engine crashes. **No silent wrong-hidden consumption** (invariant encoded in-kernel, not host-side).

#### Capture
Each stage captures an **independent single-context single-stream graph**; the inter-stage dependency lives in the device flag, *never* in stream/event edges (no cross-device `cudaStreamWaitEvent` — that is the fragile path, avoid entirely). **Critical fix:** `CudaGraphState::run_or_capture` does a live `cuGraphLaunch` at capture end (`cuda_graph.rs:143`) — for a spin-wait graph that first launch *hangs* (upstream hasn't sent epoch-0). Add a **`capture_only`** path: `BeginCapture → record kernels → EndCapture → Instantiate`, **skip the bundled first launch**. Land it GLM-local (no other model needs spin-wait-in-graph today). On every (re)build, `cuMemsetD8Async` all `flag_ring`/`ack_ring`/`epoch_counter`/`err_code = 0` across all 8 stages so they restart at epoch 0 in lockstep. Epoch is device state (`atomicAdd`), not a host immediate → graph is epoch-agnostic, no per-step host write.

#### Cycle ordering (coordinator drives, owns no GPU tensors)
stage0: wait-on-(stage7 token-ready flag for `t`) → embed `TOKEN_OUT[t-1]` → dummy_burn → `send→stage1`. middle `i`: `wait_hidden(up_flag≥t)` → dummy_burn → `send→stage(i+1)`. stage7: `wait_hidden` → dummy_burn → write `TOKEN_OUT[t]` → remote-store token id + bump stage0's token-ready flag (cycle edge `7→0`) + bump a host-pinned completion flag. Coordinator tells all 8 threads "replay step t"; each issues async `cuGraphLaunch` and returns — device flags serialize. bs=1 has **no compute overlap** (expected; PP cannot beat single-card BW for one token — the ring exists for later microbatch/MTP).

#### Measurement plan (node38, 8×H200 NV18)
**Primary (clock-skew-immune):** producer records `t0=globaltimer()` before `send`, spins on the reverse `up_ack`, `t1=globaltimer()`, stores `t1-t0` into `deltas`; D2H → p50/p90/p99/p999/max + count(>10us) + count(>100us). **Cross-check:** coordinator wall-clock over N full-cycle replays via the host-pinned completion flag.
**Sweep (mandatory cells):** pp_size {2,4,8} × payload {12KB(words=6144,bs1), 48KB(words=24576,bs4)} × dummy_burn {0,50,100,500 us} × store-width {64,128} × ring {R=1,R=2}.
**Pass bar:** pp=8 7-hop 12KB ≈16us (per the doc roofline), p99 within a few % of p50, **zero >100us over ≥50k iters**, 500us-dummy handoff cost unchanged → `L_send` confirmed not the PP risk; next slice replaces `dummy_burn` with real layers.
**Open in Slice 0:** (a) does one stage's `__trap` cleanly crash the coordinator vs leave 7 peers spinning to deadline — resolve by fault-injection (kill one flag, measure teardown); (b) B200/B300 re-measurement still required (NVLink gen + fence cost differ) — node38 H200 validates correctness + microsecond order only.

---

### 4. Open decisions / risks (need user or measurement before committing)

**Risk 1 — MLA `q_b`/`kv_b` factorization / page-format (BLOCKER, resolve in Slice 1).** Checkpoint shapes (`q_b=[65536,2048]`, `kv_b=[114688,512]`, `o_proj=[6144,16384]`) imply **N=256 heads on the input side but N=64 on the o_proj side** — a 4:1 fold the contract's `q[B,1,64,576]` / `W_UV[64,512,256]` do not capture; `mla_layout_probe.rs:75-96` independently found this 4× expansion, and vanilla vLLM (`num_heads=64`) cannot even load this checkpoint. **Do NOT hard-code `[64,512,256]`.** The split *rule* (per-head `[qk_nope | v_head_dim]`, `W_UK_T=[N,192,512]`, `W_UV=[N,512,256]`) is right; **N and the 256→64 reduction are unproven.** Resolve with one node38 vLLM dump (Slice 1) — it gates FlashMLA h_q, v_up shape, and o_proj assembly. Gate every attention op on it.

**Risk 2 — packed-vs-separated KV cache (DECIDE: packed-656).** FlashMLA sparse (the DSA mainline) is layout-locked to the **packed 656-byte FP8** token (`512 fp8 nope + 16 scale + 128 rope`; append `concat_and_cache_mla` and decode read are self-consistent and verified). Kimi's **separated BF16** ckv/kpe pool reuses cleanly *only* for dense MLA — which is **wrong for DSA sparse**. **Recommendation: commit to packed-656 FP8.** Consequence: do not reuse Kimi's separated pool for the main cache (reuse only its host page-table logic); the indexer 132-byte cache is a *separate* pool. Resolve fully in Slice 3 before any higher layer — wrong page format decodes plausible-looking garbage.

**Risk 3 — dense-stage layer-slice imbalance (config, measure-to-tune).** Stage0 carries embedding + dense layers 0..3 (dense_intermediate=12288, heavier MLP, no routing); stage7 carries final_norm + lm_head (154880-vocab GEMM) + sampler. **Equal 9-10-layer split will be imbalanced and the slowest stage sets TPOT.** Recommendation: `stage_layer_ranges` chosen by **measured per-stage decode time** (give stage0/stage7 fewer transformer layers to offset their fixed extra cost), not equal count. It is a cheap `Glm52PipelineShape` table; ship an initial guess, retune from Slice 7 calibration.

**Risk 4 — DP/EP cleanup as a separate prep commit (RECOMMEND: YES).** Make Slice P a **deletion-only** commit before Slice 0: it is mechanical, reviewable in isolation, makes the glm52 build NCCL-free (faster node38 iteration), and removes the symmetric-DP coordinator premise Slice 0 replaces. **Keep all ADAPT edits out of it** (G=256 capacity, slicing inversion, kernel guard relaxations ship with Slices 2/5). Stub the two MoE all-to-all seams with `unimplemented!()` so the crate compiles unreachable.

**Risk 5 — indexer paged-MQA logits is DeepGEMM-JIT-only (HARDEST gap, decide early).** `fp8_fp4_paged_mqa_logits` has **no copyable C++ and no TRTLLM/FlashInfer substitute** — vLLM JITs it via DeepGEMM's CuTe-DSL, and this repo deliberately has no raw-DeepGEMM runtime (`grouped launch returns CUDA_ERROR_NOT_SUPPORTED`). **Decision needed: (i) integrate a real raw-DeepGEMM-JIT runtime in the pure-Rust build, or (ii) local CUDA rewrite of `sm90_fp8_paged_mqa_logits.cuh` with NCU evidence.** Flag for the user. Mitigation: Slice 3 brings up attention on **fixture top-k indices**, decoupling the whole MLA lane from this gap until Slice 4.

**Risk 6 — `expert_alignment` 32 vs 64 (capacity blowup; tie to grouped-backend choice).** EP1 scatters 128 tokens thinly across 256 experts (~4 tokens/expert), so 64-row alignment is catastrophic: real 1024 rows pad to ~16384 (16× waste). `m_capacity = align_up(bs·topk + (A−1)·G, A)` → **A=64 → 17152** (`offset_rows` 24320), **A=32 → 8960** (`offset_rows` 16896). 64 exists only to feed DeepGEMM SM90 without a repack — a constraint PP **does not have**. **Recommendation: drop to 32-row TRTLLM-native alignment (`m_capacity=8960`) unless the DeepGEMM SM90 grouped path is measured as the PP MoE winner.** This is the single biggest sizing decision and it sets the W13/W2 output buffers — decide it with the grouped-backend choice (TRTLLM CUTLASS vs DeepGEMM SM90), measure in Slice 5. Note `local_experts` is currently overloaded ("experts this rank owns" *and* "MoE group G") — **split the concept** in `Glm52PipelineShape` or the arena/package/gemm contracts will silently agree on a wrong G.

**Risk 7 — RoPE YaRN-vs-plain (silent-garbage, cheap to resolve).** Contract says θ=8e6, interleaved, but does not confirm YaRN. **Do not reuse Kimi `build_yarn_rope_cache`** until GLM's HF `rope_scaling` is read — a stray YaRN ramp/mscale corrupts positions. Both main q_pe/k_pe (64-dim) and indexer q/k (64-dim) are interleaved (not neox) — keep them as two kernels until both cache layouts agree.


## Imported Roofline / Experiment Notes

The rest of this file is copied from `/data/code/tilert_play/glm5_tpot_pp_tp_估算.md` and lightly integrated as the baseline reasoning for the PP8 branch.

## 0. 结论先行

这个问题的核心不是 40B active params 怎么除,而是 **这一 token 的 active weight bytes 是否真的同时使用了 8 张卡的 HBM 带宽**。

如果按 8 卡 B200 聚合峰值算:

```text
42 GB / 64 TB/s ~= 0.66 ms/token ~= 1520 token/s
```

如果按 B200 单卡峰值算:

```text
42 GB / 8 TB/s ~= 5.25 ms/token ~= 190 token/s
```

如果按 7-8 ms TPOT 反推有效带宽:

```text
42 GB / 7.6 ms ~= 5.5 TB/s
```

所以不开 DFlash/MTP 时看到 7-8 ms TPOT,并不一定是数错了。它更像说明实际 critical path 没有吃到 8 卡聚合 HBM,而是在接近"单个 HBM 域的有效带宽"上运行。这里的 `5.5 TB/s` 不是 B200 峰值规格,而是由 7-8 ms 现象反推出来的有效带宽。

这对评估 PP 很关键: **PP 通信轮次少,但 bs=1 自回归 decode 的单 token 延迟通常也只能吃到单 stage/单卡带宽;TP 通信轮次多,但它的价值是让同一层的权重读取并行化,吃聚合 HBM。**

## 1. 估算口径

符号:

| 符号 | 含义 |
|---|---|
| `W_active` | 单 token 实际激活参数字节数,按 GLM5/GLM-5.1 文档取 40-42 GB |
| `B_gpu` | 单卡有效 HBM 带宽 |
| `N` | 参与同一 token 同一层计算的 GPU 数 |
| `B_agg` | 聚合 HBM 带宽,近似 `N * B_gpu` |

最简单的 memory-bound roofline:

```text
TPOT_compute ~= W_active / B_effective
```

其中 `B_effective` 不是机器总带宽,而是 critical path 上真实并发参与读 active weights 的带宽。

| 口径 | 公式 | TPOT | TPS |
|---|---:|---:|---:|
| 8 卡 B200 聚合峰值 | `42 GB / 64 TB/s` | `~0.66 ms` | `~1520 token/s` |
| B200 单卡峰值 | `42 GB / 8.0 TB/s` | `~5.25 ms` | `~190 token/s` |
| 7-8 ms 反推有效带宽 | `42 GB / 5.25-6.0 TB/s` | `~7-8 ms` | `~125-143 token/s` |

因此用户观察的"不开 DFlash 大概 130 TPS / 7-8 ms"和单卡有效带宽口径是自洽的,但低于 B200 单卡峰值 roofline。它不支持"base decode 已接近 8 卡聚合 bandwidth roofline"这个说法。

## 2. PP 与 TP 的本质差别

### TP: 通信多,但吃聚合 HBM

TP 把同一层的权重和计算切到多卡。对 bs=1 单 token decode,它的价值是让同一 token 的同一层在多张卡上并行读权重。

理想情况下:

```text
TPOT_compute_tp ~= W_active / (tp_size * B_gpu)
```

代价是每层都有同步点。按常见 transformer TP:

| 阶段 | 通信 |
|---|---|
| attention `o_proj` 后 | 1 次 allreduce 或等价 reduce/scatter |
| FFN/MoE down 后 | 1 次 allreduce 或等价 reduce/scatter |
| logits/top1/top-p | 末尾还可能有一次跨卡规约或采样通信 |

对 GLM5 DSA 路径还要加:

| 阶段 | 通信 |
|---|---|
| GPU0 sparse index 到 GPU1-7 | 每层 1 次 selected-index P2P packet |
| attention output | 每层 1 次 fused allreduce |
| FFN/MoE down | 每层 1 次 fused allreduce |

所以粗略轮次:

```text
heavy collective rounds ~= 2 * num_layers
selected-index packet rounds ~= num_layers
tail sampling/logits ~= O(1)
```

GLM5 `num_layers = 78` 时:

```text
heavy allreduce ~= 156 次/token
selected-index P2P ~= 78 次/token
```

通信量本身不一定大,因为 hidden 激活很小:

```text
hidden bytes ~= 6144 * 2 = 12 KB/token
```

但轮次很多,每层都是 dependency edge。TileRT 把 allreduce/P2P 融进 graph 内 physical op,主要是在砍这些同步和 launch 边界。

### PP: 通信少,但 bs=1 单 token 吃不到聚合 HBM

PP 把层切到不同 stage。单 token forward 只需要在 stage 之间传 hidden:

```text
pp_comm_rounds ~= pp_size - 1
pp_comm_bytes ~= (pp_size - 1) * hidden_bytes
```

例如 `pp_size=8`:

```text
rounds ~= 7
bytes ~= 7 * 12 KB ~= 84 KB/token
```

这比 TP 每层 allreduce 少很多。这里先不讨论 pipeline 是否填满,因为目标是单请求 bs=1 的极致 TPOT,不是多请求 throughput。对 TPOT 来说,关键是同一个 token 在 PP stage 之间存在严格依赖:

```text
stage i+1 必须等 stage i 输出 hidden
```

因此单 token critical path 是所有 PP stage 串行相加:

```text
stage0 跑 token t 的前几层
stage1 跑 token t 的中间层
...
last stage 采样出 token t+1
stage0 才能开始 token t+1
```

```text
TPOT_compute_pp ~= sum_i W_stage_i / B_gpu
                 ~= W_active / B_gpu
```

这会落回单卡带宽 roofline:按 B200 峰值约 5.25 ms,按用户观察反推的有效带宽约 7-8 ms。

所以 PP 的优点是:

| 优点 | 说明 |
|---|---|
| 通信轮次少 | `pp_size - 1` 个 activation send,不是每层 2 次 allreduce |
| 通信体积小 | hidden 只有十几 KB/token |
| 工程上更少 collective pressure | 不需要 156 个 layer-level allreduce |

PP 的缺点是:

| 缺点 | 说明 |
|---|---|
| 单 token latency 不吃聚合 HBM | stage 串行,critical path 接近单卡读完整 active weights |
| 不能靠 stage overlap 降低同一 token TPOT | stage 间是数据依赖,不是可并行分支 |
| stage 负载必须极准 | 最慢 stage 直接决定局部瓶颈,层数/MoE active bytes 不均会放大尾部 |

## 3. PP 是否还有可能

PP 不是没价值,但目标要分清。

如果目标是 **单条请求 bs=1 极致 TPOT**:

```text
TP 的优势:同一层并行读权重,compute roofline 更低。
PP 的优势:通信轮次少很多,可以砍掉 TP 的 per-layer collective gap。
```

所以 PP 是否可能赢,不是看 pipeline occupancy,而是看这个不等式:

```text
TPOT_pp ~= W_active / B_gpu_eff + (pp_size - 1) * L_send

TPOT_tp ~= W_active / (tp_size * B_gpu_eff)
          + num_layers * (L_attn_comm + L_ffn_comm + L_selected_index)
          + graph/runtime gap

PP win iff TP 的通信/gap > PP 丢掉聚合 HBM 带来的 compute 增量。
```

按 B200 峰值粗算,PP 相对 8 卡聚合 TP 丢掉的 compute roofline 约:

```text
42 GB / 8 TB/s - 42 GB / 64 TB/s
~= 5.25 ms - 0.66 ms
~= 4.6 ms
```

如果按 7-8 ms 单卡有效带宽口径,这个差距会更大。因此 PP 要在单请求 TPOT 上赢,TP 那边每 token 的 156 次 allreduce + 78 次 selected-index P2P + runtime gap 必须吃掉数毫秒级预算。

### `L_send` 在 NVLink 下怎么估

PP stage boundary 传的是 hidden,量级非常小:

```text
hidden_bytes = 6144 * sizeof(bf16) = 12,288 B ~= 12 KB
```

B200/DGX B200 的 NVLink 口径:

```text
DGX B200 aggregate NVLink bandwidth = 14.4 TB/s
per GPU bidirectional NVLink ~= 1.8 TB/s
per GPU one-way 粗略按 ~= 0.9 TB/s
```

所以只看带宽搬运时间:

```text
12 KB / 1.8 TB/s ~= 0.007 us
12 KB / 0.9 TB/s ~= 0.014 us
```

即使 `seq_len=4` 一次传 4 个 hidden:

```text
48 KB / 0.9 TB/s ~= 0.055 us
```

因此 `L_send` 不能按 payload/bandwidth 估成主项。NVLink 上 12KB hidden send 的主项是固定开销:

```text
L_send ~= L_enqueue_or_graph_node
        + L_remote_store_or_copy_setup
        + L_visibility_fence
        + L_receiver_wait
        + payload_bytes / B_nvlink
```

估算时建议用三档:

| 实现方式 | 单次 `L_send` 估计 | 7-stage PP 合计 |
|---|---:|---:|
| graph 内自研 P2P packet/store + flag | `~1-3 us` | `~7-21 us` |
| `cudaMemcpyPeerAsync`/小 kernel copy + event | `~5-10 us` | `~35-70 us` |
| NCCL send/recv 或通用 runtime 路径 | `~10-20+ us` | `~70-140+ us` |

这个量级和 PP/TP 的 compute roofline 差距相比很小。按 B200 峰值,PP 相对 8 卡 TP 丢掉的并行读权重收益约 `4.6 ms`;7 次 PP send 即使用 `20 us` 估也只有 `0.14 ms`。

因此对 bs=1 极致 TPOT,PP 的 stage 间 send 大概率不是瓶颈。真正要比较的是:

```text
PP 丢掉 8 卡并行读权重的 4-7 ms
vs
TP 每 token 156 次 allreduce + 78 次 selected-index P2P + runtime gap
```

倒过来算,如果 PP 要靠少通信赢回 B200 峰值口径的 `~4.6 ms`,TP 侧每个 layer-level 通信/gap edge 的平均成本要达到:

```text
4.6 ms / (156 + 78) ~= 20 us / edge
```

如果 TileRT 的 fused P2P/allreduce 已经把每个 edge 压到个位数微秒,PP 不一定赢。如果现有 TP 实现用通用 NCCL/event/runtime,每个 edge 接近十几到几十微秒,PP 就可能有空间。

### PP 能不能 mega 化

可以。PP mega 化的目标不是填满 pipeline,而是把 stage boundary 从 host/runtime/NCCL 小消息路径里拿掉,变成 graph 内 P2P handoff。

理想形态:

```text
prepare:
  每个 stage 分配固定 input/output hidden buffer
  cudaDeviceEnablePeerAccess
  下游 input buffer 地址写进上游 resource table
  每个 stage capture 一张 CUDA Graph

decode:
  stage0 graph replay
    -> 最后一个 kernel/packet kernel 通过 NVLink P2P 写 stage1 input buffer
    -> system-scope store 写 flag epoch
  stage1 graph 内 receive/wait flag
    -> 读 input hidden
    -> 跑本 stage layers
    -> P2P 写 stage2
  ...
  last stage head/sample 写 TOKEN_OUT
```

这和 TileRT 现在 selected-index 的 P2P packet 思路很像,只是 payload 从 `IDX_SELECTS` 换成 hidden activation。

关键点:

| 组件 | 作用 |
|---|---|
| peer buffer | 每个 stage 暴露下一 stage 的 input hidden 地址 |
| packet/copy kernel | 上游用 remote store 或 P2P copy 写下一卡显存 |
| flag/epoch | 下游 graph 内 spin/wait,保证读到本轮 hidden |
| static buffer | graph replay 时地址不变,只改 epoch/position |
| per-stage graph | host 只触发 replay,stage 间依赖在 device/NVLink 上解决 |

这样可以把 `L_send` 从通用 runtime 路径压到 graph 内 P2P packet 的固定延迟区间。对 12KB hidden,带宽项几乎为零,主项就是 flag 和同步。

但它不能改变这个事实:

```text
stage1 必须等 stage0 hidden
stage2 必须等 stage1 hidden
```

所以 PP mega 化能消掉 stage boundary gap,不能把同一个 token 的多个 stage 变成并行计算。它的收益上限大概是:

```text
省掉 PP stage 间 host/event/NCCL 小消息开销
```

不是:

```text
吃到 8 卡聚合 HBM 读同一层权重
```

### NVLink P2P 是什么

NVLink P2P 指的是 GPU 之间可以直接访问彼此显存。常见层级:

| 路径 | 说明 | 适合度 |
|---|---|---|
| `cudaMemcpyPeerAsync` | runtime 发起 P2P copy,走 NVLink/NVSwitch | 能用,但小消息固定开销偏大 |
| P2P kernel remote load/store | kernel 直接 `LDG/STG` peer GPU 地址 | PP mega 化最适合,可放进 graph |
| packet + flag | payload 和 epoch 一起写,下游轮询 expected flag | 最像 TileRT selected-index |
| NCCL send/recv | 通用通信库路径 | 可靠但对 12KB hidden 可能太重 |

PP hidden handoff 最应该用第三种:

```text
上游: STG.E.STRONG.SYS 写 peer input buffer + flag
下游: LDG.E.STRONG.SYS 轮询 flag,再读 hidden
```

这需要处理几个细节:

| 问题 | 处理 |
|---|---|
| 可见性 | system-scope store/load 或合适 fence |
| buffer 覆盖 | double buffer 或 epoch ring,避免下一轮覆盖上一轮 |
| graph 静态地址 | input/output buffer 地址 prepare 阶段固定 |
| 死等 | expected epoch 必须单调,reset 时清状态 |
| 多 stage | 每条边一套 peer buffer + flag |

所以答案是: **NVLink P2P 正是 PP mega 化的实现工具**。它能把 PP 的 `L_send` 压到很低;但 PP 是否赢,仍要看 TP 的 per-layer allreduce/P2P/gap 是否真的大到超过 PP 丢失聚合 HBM 的 4-7ms。

如果目标是 **多请求吞吐**:

```text
PP 可以用多个 sequence 填 pipeline,通信压力比 TP 小,可能更好。
```

如果目标是 **结合 speculative/MTP**:

```text
PP 可能重新有空间,因为一次 verify 有 seq_len=2/4 或更多 draft token,
可以给 pipeline 更多并发工作。
```

但这已经不是纯 bs=1 单 token PP,而是:

```text
PP + speculative window
PP + 多请求 microbatch
PP + stage 内 TP
```

更现实的混合形态可能是:

| 并行方式 | 作用 |
|---|---|
| stage 内 TP | 保留同一层的聚合 HBM |
| stage 间 PP | 减少全模型范围的 collective,把通信变成少数 activation send |
| speculative window | 给 PP 填 pipeline 的 token 级并发 |

## 4. 当前判断

`42 GB / 64 TB/s ~= 0.66 ms` 这种 8 卡 Blackwell 聚合 roofline 只有在一个前提下成立:

```text
同一 token 的 active weights 被 8 张卡并行读取,且通信/调度 gap 足够小。
```

`7-8 ms` 这个 TPOT 则说明实际更接近:

```text
同一 token critical path 只吃到单卡级有效带宽,而且没有达到 B200 单卡峰值。
```

因此,评估 PP 时不能只看"PP 通信次数更少"。PP 的通信确实少,但它用通信少换掉了 TP 最重要的东西:同一 token 同一层的 HBM 并行读权重。

下一步需要实测确认的不是"PP 通信少不少",而是:

1. base decode 不开 DFlash/MTP 的真实 TPOT。
2. 每层/每阶段 GPU active 时间是否重叠。
3. HBM throughput 是接近单卡还是 8 卡聚合。
4. TP allreduce/P2P 的 round-trip latency 占比。
5. MTP verify step latency 和 accepted length 的乘积收益。

只有这些数据齐了,才能判断 PP、TP、PP+TP、PP+MTP 哪个方向值得写。

## 5. DFlash/MTP 为什么会显得很重要

DFlash/MTP 本质上不是把一次 base replay 变成 1 ms,而是让一次较贵的 verify/replay 产出多个 accepted token。这个结论应该后置到 PP/TP roofline 之后看:如果 base decode 的 critical path 接近单卡有效带宽,那 speculative 接受长度就会成为有效 TPS 的主要放大器。

如果不开 DFlash:

```text
step_tpot ~= 7.5 ms
accepted ~= 1
effective_tpot ~= 7.5 ms
effective_tps ~= 133
```

如果 MTP 平均接受长度 `a = 3.2`:

```text
effective_tpot ~= step_tpot / a
```

例子:

| verify step latency | accepted | effective TPOT | effective TPS |
|---:|---:|---:|---:|
| 8.0 ms | 3.2 | 2.50 ms | 400 |
| 7.0 ms | 3.2 | 2.19 ms | 457 |
| 6.0 ms | 3.2 | 1.88 ms | 533 |

所以如果 base decode 只有 100 多 TPS,最终 500 TPS 级别很可能确实主要来自 speculative/MTP 接受长度,再叠加 verify 图内部的优化。

这里需要实测拆分:

| 项 | 状态 |
|---|---|
| 不开 DFlash/MTP 的 base TPOT | 用户观察约 7-8 ms,需要固定配置复测 |
| MTP verify step latency | 待测 |
| 平均 accepted length | 文档里有约 3.2 的说法,需要同上下文长度复测 |
| effective TPS | 不能直接和 base TPOT 混算,要按 accepted token 归一 |

## 6. 带宽来源

NVIDIA DGX B200 官方规格:8x Blackwell GPUs, GPU memory `1,440 GB total`, `64 TB/s HBM3e bandwidth`。因此本文把 B200 单卡峰值按 `64 / 8 = 8 TB/s` 估算。

历史博客里常见的 `38 TB/s` 不是本文 Blackwell/B300 对比的目标口径。评估当前 B200/B300 机器时,优先使用 Blackwell 8 卡 `64 TB/s` 级别聚合峰值,再用实测 TPOT 反推有效带宽。

## 7. NVL72 相比普通 8x B300 的 NVLink 有没有更快

本文只比较 Blackwell/Blackwell Ultra 内部形态: **NVL72 vs 普通 8x B200/B300 NVSwitch 节点**。结论是:NVL72 的 NVLink domain 更大,但单 GPU 注入带宽没有比普通 8 卡 B200/B300 NVSwitch 节点更快,仍是约 `1.8 TB/s/GPU bidirectional`。

官方规格口径:

| 系统 | GPU 数 | HBM 带宽 | NVLink 带宽 |
|---|---:|---:|---:|
| DGX B200 | 8 | `64 TB/s` total | `14.4 TB/s` aggregate |
| DGX B300 | 8 | B300 单卡仍按 `~8 TB/s` 量级看 | `14.4 TB/s` aggregate |
| GB200 NVL72 | 72 | `576 TB/s` total | `130 TB/s` total,`3.6 TB/s` per Grace Blackwell Superchip |
| GB300 NVL72 | 72 | `576 TB/s` total 量级 | `130 TB/s` total 量级 |

这些数除一下会发现同一代 Blackwell/Blackwell Ultra 的 per-GPU NVLink 注入带宽基本还是:

```text
14.4 TB/s / 8  ~= 1.8 TB/s per GPU
130 TB/s / 72 ~= 1.8 TB/s per GPU
```

所以如果只取 NVL72 里的 4 张或 8 张 GPU 做一个 bs=1 low-latency request,它不会因为在 NVL72 rack 里就获得比普通 8x B300 更高的 per-GPU NVLink 注入带宽。NVL72 的新增价值是:

```text
把 72 张 GPU 放进同一个 NVLink/NVSwitch domain,
避免 8-GPU node 之间掉到 InfiniBand/Ethernet 级别。
```

这对这些场景有价值:

| 场景 | NVL72 价值 |
|---|---|
| 模型/KV/cache 必须跨超过 8 GPU | 很大,因为仍在 NVLink domain 内 |
| 大 batch / 多请求吞吐 | 很大,可以把更多 GPU 当一个 rack-scale pool |
| 训练或大规模 all-to-all | 很大 |
| 单请求 bs=1,只需要 4/8 GPU | 小,甚至可能不如普通 8-GPU 节点划算 |

对本文目标,也就是 **bs=1 极致 TPOT**,NVL72 的 72 卡互联不是直接收益。最优策略一般是:

```text
用尽可能少的 GPU 覆盖模型 active working set,
并保证这些 GPU 在同一个低延迟 NVLink island 内。
```

如果 4/8 张 B300 已经能放下 active weights/KV,并且 PP mega 化只需要 stage 间传 12KB hidden,那么 NVL72 的 72-GPU domain 不会明显降低 `L_send`;`L_send` 已经由 fixed latency 主导,不是带宽主导。NVL72 真正避免的是"超过 8 GPU 后跨节点通信变慢"这个问题。

### Hopper 开发口径

可以先用 Hopper/H100/H200 开发 mega PP 的 P2P handoff。CUDA 接口是同一套:

```text
cudaDeviceCanAccessPeer
cudaDeviceEnablePeerAccess
UVA peer pointer
kernel remote store/load peer allocation
__threadfence_system + flag/epoch
```

需要注意的是性能口径不同:

| GPU 代际 | NVLink per GPU bidirectional | 对 12KB hidden 的影响 |
|---|---:|---|
| Hopper H100/H200 | `~900 GB/s` | `12KB / 900GB/s ~= 0.014 us` |
| Blackwell B200/B300 | `~1.8 TB/s` | `12KB / 1.8TB/s ~= 0.007 us` |

payload 带宽项在两代上都远小于微秒,所以 Hopper 可以很好地验证:

```text
P2P peer access 是否通
remote store + flag 协议是否正确
fixed latency 是 1us / 5us / 10us 哪个量级
多 stage 串起来有没有长尾
```

最终 B200/B300 上仍要复测,因为 NVLink 代际、NVSwitch、GPU clocks、driver 和 system-scope memory ordering 开销都会影响尾延迟。

## 8. 如何测 mega PP 的 `L_send`

NVIDIA/生态里有现成 baseline 工具,但不能直接替代 mega PP microbench。

| 工具 | 测到什么 | 用途 |
|---|---|---|
| CUDA sample `p2pBandwidthLatencyTest` | GPU-GPU P2P bandwidth/latency | 确认 peer access/NVLink 拓扑是否正常 |
| NVIDIA `nvbandwidth` | GPU/CPU/GPU 间 bandwidth 与 latency | 更系统地扫 NVLink/PCIe/内存路径 |
| `nccl-tests` | NCCL allreduce/sendrecv 等 collective/P2P 路径 | 测通用通信库小消息 latency,作为反例或 baseline |
| 自写 packet+flag microbench | remote store + fence + flag + receiver wait | 这才是 mega PP 的 `L_send` |

### 8.1 先测机器 baseline

先确认拓扑:

```bash
nvidia-smi topo -m
```

再跑 NVIDIA/CUDA baseline:

```bash
# CUDA samples
./p2pBandwidthLatencyTest

# NVIDIA nvbandwidth
./nvbandwidth

# NCCL 小消息路径,扫 8B 到 1MB
./sendrecv_perf -b 8 -e 1M -f 2 -g 2
./all_reduce_perf -b 8 -e 1M -f 2 -g 8
```

这些数只回答:

```text
这台机器的 P2P/NVLink/NCCL 有没有坏;
通用 runtime/NCCL 小消息大概多慢。
```

它们不回答:

```text
graph 内 remote store hidden + flag handoff 到底几微秒。
```

### 8.2 测 `cudaMemcpyPeerAsync` 小消息

第二层测 memcpy peer:

```text
for size in {4KB, 12KB, 48KB, 64KB, 256KB}:
  src stream:
    cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, size)
  event timing over many iterations
```

再测 captured graph 版本:

```text
capture:
  cudaMemcpyPeerAsync(...)
instantiate graph
for many iterations:
  cudaGraphLaunch(graph)
sync
```

这个给出:

```text
runtime P2P copy 小消息固定开销
graph replay 是否明显降低 launch/enqueue 开销
```

但它仍然不是最理想的 mega PP,因为 hidden handoff 可以不走 memcpy node,而是直接由上游 kernel remote store 到下游 buffer。

### 8.3 真正要测:kernel remote store + flag ping-pong

mega PP 的 stage boundary 应该这样测:

```text
GPU0: write 12KB hidden to GPU1 peer buffer
GPU0: system-scope fence/store flag epoch
GPU1: wait flag epoch
GPU1: read hidden / optional checksum
GPU1: write ack back to GPU0
GPU0: wait ack
```

用 round-trip ping-pong 的原因:跨 GPU 时钟不一定可靠同步。让 GPU0 自己测:

```text
t0 = GPU0 globaltimer
GPU0 -> GPU1 hidden + flag
GPU1 -> GPU0 ack
t1 = GPU0 globaltimer
RTT = t1 - t0
one_way ~= RTT / 2
```

这个 microbench 最接近实际 `L_send`,因为它包含:

```text
remote store payload
system-scope visibility
flag epoch
receiver polling
ack path
```

建议扫这些变量:

| 变量 | 取值 |
|---|---|
| payload | `4KB, 12KB, 24KB, 48KB, 64KB` |
| buffer | single buffer vs double buffer/ring |
| store width | scalar / 64-bit / 128-bit vectorized store |
| flag | same cacheline vs separate cacheline |
| wait | spin load frequency, backoff/no backoff |
| path | adjacent GPU pairs, all pair matrix |
| mode | standalone kernel loop vs captured graph |

输出至少要有:

```text
p50 / p90 / p99 one-way latency
payload bandwidth
pair matrix
flag wait retries
```

### 8.4 最终端到端测法

最后要测一个最小 PP graph:

```text
GPU0 graph:
  dummy layer kernel, burn X us
  p2p_send_hidden_kernel

GPU1 graph:
  wait_hidden_kernel
  dummy layer kernel, burn Y us
  p2p_send_hidden_kernel

...
```

扫 `pp_size = 2/4/8`,payload = `12KB/48KB`,dummy compute = `0/50/100/500us`。这样能看出:

```text
stage handoff 是否真的只有个位数 us;
多个 stage 串起来是否有长尾;
graph replay + P2P flag 会不会出现抖动;
```

如果这个最小 PP graph 的 7 次 handoff 仍只有几十微秒,那 `L_send` 就不是 PP 的主要风险。下一步就该测真实 layer stage 的 HBM throughput 和 stage balance。

### 8.5 jiuzhang H200 node37 实测 baseline

测试位置:

```text
cluster: jiuzhang
node: host-172-31-13-37 / 172.31.13.37
GPU: 8x H200, GPU-GPU topo = NV18
binary: /tmp/p2p_lsend/p2p_lsend
build: /usr/local/cuda-12.8/bin/nvcc -arch=sm_90
```

关键命令:

```bash
./p2p_lsend --src 0 --dst 1 --scan --touch-payload --iters 50000 --warmup 5000
./p2p_lsend --src 0 --dst 4 --bytes 12288 --touch-payload --iters 50000 --warmup 5000
```

`--touch-payload` 表示接收端 wait flag 后实际读取 payload,更接近下游 stage 读 hidden 的场景。

结果摘要:

| pair | bytes | rtt p50 | rtt p99 | rtt p999 | half-rtt avg | 备注 |
|---|---:|---:|---:|---:|---:|---|
| GPU0→GPU1 | 0 | `2.43 us` | `2.66 us` | `3.30 us` | `1.25 us` | flag+ack baseline |
| GPU0→GPU1 | 12KB | `4.42 us` | `4.61 us` | `5.12 us` | `2.23 us` | GLM5 bf16 hidden |
| GPU0→GPU1 | 48KB | `9.12 us` | `9.28 us` | `10.02 us` | `4.47 us` | seq_len=4 hidden |
| GPU0→GPU4 | 12KB | `4.45 us` | `4.64 us` | `5.22 us` | `2.25 us` | 跨 NUMA 组,仍是 NV18 |

完整 scan(`GPU0→GPU1`, touch payload):

| bytes | rtt avg | rtt p50 | rtt p90 | rtt p99 | rtt p999 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `2.490` | `2.432` | `2.624` | `2.656` | `3.296` | `3.616` |
| 4096 | `3.272` | `3.328` | `3.360` | `3.392` | `4.064` | `4.288` |
| 12288 | `4.461` | `4.416` | `4.576` | `4.608` | `5.120` | `5.504` |
| 24576 | `5.803` | `5.824` | `5.856` | `6.016` | `6.560` | `6.752` |
| 49152 | `8.935` | `9.120` | `9.184` | `9.280` | `10.016` | `10.848` |
| 65536 | `11.004` | `11.104` | `11.136` | `11.264` | `11.968` | `12.768` |
| 262144 | `34.429` | `34.528` | `34.560` | `34.752` | `35.424` | `36.352` |

解释:

```text
12KB hidden handoff 的 one-way proxy ~= half RTT ~= 2.2-2.3 us
48KB seq_len=4 hidden handoff 的 one-way proxy ~= 4.5 us
7-stage PP 的 12KB handoff 合计 ~= 7 * 2.3 us ~= 16 us
```

所以在 H200/NV18 上,mega PP 的 stage-boundary P2P handoff 本身已经是十几微秒总量级,不是毫秒级风险。B200/B300 上需要复测,但由于 12KB payload 仍由 fixed latency 主导,预期不会比这个更差到改变 PP/TP 的毫秒级判断。
