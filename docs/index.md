# docs index

Organized by domain (model line / subsystem / playbook / lesson) instead of by lifecycle stage. A doc's freshness is recorded in its own header (TL;DR, and `Last touched` for active areas), not by which directory it lives in.

| Where it lives | What it is |
| --- | --- |
| `roadmap/` | Strategic plans and milestones — quarterly direction, product positioning. |
| `models/<line>/` | Per-model living docs: design, accuracy, perf, refactor records, gotchas. |
| `subsystems/<area>/` | Cross-cutting components (runtime / scheduler / frontend / kernels). |
| `playbooks/` | Reusable how-to: benching, profiling, accuracy debugging, onboarding. |
| `lessons/` | Tribal knowledge from research / other projects worth keeping. |
| `benchmarks/` | Standalone benchmark snapshots and eval reports. |
| `conventions/` | Ongoing standards (bench regression policy, coding style). |
| `private/` | Local-only notes (gitignored). |

## roadmap

| Path | TL;DR |
| --- | --- |
| `roadmap/roadmap-2026-h2.md` | 2026-H2 plan, supersedes issue #203. Now: website as product surface (recipes-style pages, no runtime Docker image), model maturity tiers (Qwen3-4B first Stable), observability wiring, GLM5.2 large-MoE mainline. Later: NIXL-compatible P/D, design-first. |
| `roadmap/direction.md` | One size can't fit all. Shared infrastructure (frontend, runtime primitives, kernels, data plane) + per-model engines with their own scheduler/kernel DAG/state. Long-term loop: kernel ledger → simulator → request tracing. |
| `roadmap/execution.md` | Current state and immediate next steps. No timeline — entries move through In progress → Next → Open. Covers cross-model infrastructure (kernel ledger, simulator, tracing, frontend polish) and per-model active work (DeepSeek V4, Qwen3.5, Qwen3). |

## models / qwen3

| Path | TL;DR |
| --- | --- |
| `models/qwen3/serving-performance.md` | **Authoritative Qwen3 serving perf numbers** (4B + 8B QPS sweep vs vLLM 0.24.0, footprint, DSpark/DFlash spec decode, warm prefix-cache TTFT, KV offload). All data reproducible via `tools/bench/run_serving_bench.sh`. |
| `models/qwen3/serving-perf-5090.md` | Tuning history behind the serving numbers: unified-step attention fusion, batched step tail (#345), chunked prefill, cuBLAS 12.9 N=1025 cliff, cublasLt per-shape tuning, split-KV ≤bs32. Latest data lives in `serving-performance.md`. |
| `models/qwen3/decode-attention.md` | Decode attention path (`NonPartition` vs `SplitKv`) is chosen by **batch (CTA-vs-SM), not context**: the old `max_seq_len>=1024` gate stranded bs=1 mid-context decode on the SM-starved NonPartition kernel — a tpot hump peaking ~ctx800, cliff-dropping at ctx1024. Removing it flattens bs=1 tpot (5090 −16% / 5070 Ti −7.5% @ctx800); kept `padded_bs<=32` (bs≤8 wins big, bs16 even, bs32 <1% loss). Also records the SplitKv chunk-size/grid policy (`Tuned` adaptive vs `Pin`/`PerToken` fixed-split batch-invariance, #435/#438). Two-card A/B + CUDA-graph capture + golden-gate verified. |
| `models/qwen3/green-ctx-sm-partition.md` | Green Context SM partition (`OPENINFER_SM_PARTITION=20`) runs prefill/decode on disjoint SMs so decode stops stalling behind co-scheduled prefill: 5090 mid-band ITL p99 ~halved, TPOT down (−22% @QPS12), but TTFT 2–4× worse (prefill deferred + fewer SMs) — a TTFT↔ITL/TPOT trade, not a free win. Two-graph change (decode CUDA graph captured on the green decode stream) adds ~5% ITL p99 / 1–4% TPOT on top. Mechanism, A/B table, Xid-31/gemm_lt pitfalls. |
| `models/qwen3/roadmap.md` | Qwen3-4B roadmap (2026-06 review): line is the maturity bar; #220 RoPE OOB, batched greedy sampling (#307), mixed greedy/non-greedy sampling (#284), and pegaflow KV offload (#316) are landed; open set is zero TP coverage, zero-adapter-only LoRA gate, dropped prefix-cache observability, stale docs, and YaRN #8 follow-up. |
| `models/qwen3/model-crate.md` | `openinfer-qwen3` owns Qwen3 config/weights/executor/scheduler/tests/kernel plan; root sees generic `EngineHandle`; split-K decode gated on `padded_bs<=32` (64-token `Tuned` floor, cap 64 chunks; `Pin`/`PerToken` fixed 160-token split), 4k/64 serving TPOT p50 `6.46ms` on RTX 5090. |
| `models/qwen3/prefix-cache.md` | Prefix caching on by default for Qwen3-4B: full-block kvbm radix matching at the executor, suffix-only prefill. Repeated ~1900-token prompt TTFT 141.8 → 16.3ms p50 (8.7×); warm TTFT ≈ TPOT + ~5ms setup. Includes the RoPE scalar-path corruption fix and the drain-the-stream TTFT measurement pitfall. |
| `models/qwen3/dspark-integration.md` | DeepSeek **DSpark** Phase 1 is implemented for Qwen3-4B: DFlash backbone + rank-256 Markov head, anchor-first DeepSpec layout, one strided argmax-with-bias kernel, PDL polish, and one D2H per draft block. Greedy losslessness passes; 5090 block7 A/B vs matched DFlash shows DSpark +3.6% geomean output tok/s overall (+3–16% on text/code, random synthetic exception) and better accepted-draft distribution (2.52 vs 2.30 draft tokens/round). |
| `models/qwen3/dflash-speculative-decoding.md` | DFlash speculative decoding behind `--dflash-draft-model-path`, modelled as an optimistic transaction (propose K → verify K+1 span → accept longest argmax prefix + 1 bonus → commit/roll back KV). Lossless up to bf16 tie-flips (bit-identical multi-token accepts; lm-eval gsm8k strict-match identical spec on/off). Single-stream decode 1.82× on 5070 Ti, 1.56× on 5090. Concurrent throughput fixed by batching the draft forward, then a piecewise verify CUDA Graph (dense ops captured, attention eager) closed single-stream: 5090 greedy c1 274 ≈ vLLM 278, c8 1525 > 1240, c16 1834 ≈ 1846 — all batch sizes now ≥ vLLM. Accept measured equal (9.1% vs 8.85%, same drafter); draft-side piecewise graph tracked next. Proposer trait deferred to EAGLE. |
| `models/qwen3/accuracy-gate.md` | Qwen3-4B instance of the logits golden gate (`tests/hf_golden_gate.rs`): 48 teacher-forced sequences / 816 positions vs a stored HF bf16 golden, replayed over bs=1 / batched eager / CUDA-graph. Strict guards: regret check + mean ≤ 0.06 + p99 ≤ 0.20; absolute max printed but not asserted (coverage-unstable). Methodology in `subsystems/correctness/`. |
| `models/qwen3/kernels-crate.md` | Phase 1 split implemented and 5090-verified: Qwen3-4B kernel surface lives in `openinfer-kernels`; release build, test-target compile, accuracy gate, and bench snapshot pass. |
| `models/qwen3/tp-design.md` | Qwen3 tensor-parallel design: `TP=2` milestone scope plus the controller/worker broadcast execution model, request identity, and coarse-grained step protocol for future TP/MoE work. |
| `models/qwen3/kv-pressure-hang.md` | Issue #85 Qwen3-4B KV pressure hang fixed by full-lifetime scheduler KV admission, waiting-queue deferral, cleanup on disconnect/error, impossible-request errors, scheduler/bridge gates, and real `vllm bench serve` QPS=2 `500/500` pass with post-pressure completion healthy. |
| `models/qwen3/pd-disaggregation-m2.md` | P/D 分离 M2 **已端到端验证**（单机 2×H200 + 400G IB）：Qwen3-8B 1P+1D，KV 经 pegaflow metaserver P2P（RDMA READ）流转，greedy 输出与单实例逐 token 一致，杀 metaserver/P 优雅退化；多轮并发压测已过（含 router `max_completion_tokens` 坑）。openinfer `feat/pd-pegaflow-p2p` + pegaflow PR #381。RemoteFetch 状态机单测欠账；M3 layer-wise push 延后。 |

## models / qwen35

| Path | TL;DR |
| --- | --- |
| `models/qwen35/roadmap.md` | Qwen3.5-4B roadmap (2026-06 review): decode-tuning refresh improves direct TPOT by 2-3%, while vLLM still leads 1024/256 HTTP decode and high-concurrency throughput. Open items: HND prefill staging, prefix-cache design, serving concurrency. |
| `models/qwen35/kv-admission.md` | Issue #254 complete: Qwen3.5 now uses full-lifetime KV admission, deferred pressure handling, impossible-request rejection, explicit error semantics, direct rejection-event coverage, RTX 5090 e2e, and real HTTP pressure/post-pressure validation. |
| `models/qwen35/optimization.md` | Hybrid 24 linear + 8 full attn optimization ledger. Decode-tuning refresh fuses MLP gate/up and tunes decode cublasLt buckets, improving direct TPOT by 2-3%; vLLM still leads 1024/256 HTTP decode. |
| `models/qwen35/accuracy.md` | Qwen3.5 HF bf16 logits goldens, size-keyed (4b, 9b committed; 27b once dumped), through `past_key_values`: short replay covers sequential graph, bucket-straddling batched graph, and slot-compaction; long replay covers 4097/8192-token prompts; full GSM8K 8-shot now matches the HF baseline within 0.15 percentage points. |
| `models/qwen35/model-crate.md` | `openinfer-qwen35-4b` owns Qwen3.5 model/scheduler/recurrent ops/tests/benches; feature-gated behind `qwen35-4b` (Triton AOT is the only Python build dependency); root loads it through `EngineHandle`. Build/check/clippy, root bench sanity check, historical Qwen3.5 e2e, and scheduler e2e records live here. |
| `models/qwen35/kernel-plan.md` | Qwen3.5-4B has a `openinfer_qwen35_4b::kernel_plan()` static descriptor mirroring the qwen3 module — enumerates every prefill/decode/unified op with its Rust call site, backend, and notes, so you can dump the active kernel mix without reading call sites. Pure refactor (issue #256), no kernel behavior change. |
| `models/qwen35/batched-step-tail.md` | Qwen3.5 issue #353 implementation record: final prefill tail is batched, decode/unified sample from batched logits, host full-vocab copies are logprobs-only, HF + scheduler e2e pass, and final serving A/B supports only the first-token/short-output TTFT claim. |

## models / glm52

| Path | TL;DR |
| --- | --- |
| `models/glm52/dp16-smr-coordinator.md` | Design sketch (no code): cross-node DP16 via a **replicated deterministic coordinator** (SMR/Calvin shape) instead of hub-and-spoke RPC — sequencer stamps a total order on admissions/cancels, per-step output tokens allgathered (≤8 u32/rank), every replica derives byte-identical step shapes locally. Known determinism violation to fix first: `token_tx.is_closed()` disconnect probes → sequenced cancel events. Fallback: framed-TCP hub-and-spoke (<1% overhead at 20–46 ms steps). Cheapest first step stands alone: (admission, outputs) replay journal for desync forensics. |
| `models/glm52/dspark-mtp.md` | DSpark greedy speculative decoding design (community `RedHatAI/GLM-5.2-speculator.dspark`, not native MTP): qwen3-arch 5-layer draft at hidden 6144 + rank-256 Markov head; verify span 8 = one bucket-8 step with all rows on one slot (D2.5 infra reused). Checkpoint verified on jz-38, embed/lm_head byte-identical to target → skip. Milestones: M1 span-steps for prompt ingest (~8× fewer prompt steps), M2 draft lane, M3 round loop + A/B. |
| `models/glm52/continuous-batching.md` | D2 + D2.5 execution record: multi-slot admission (8 requests/rank, least-loaded first) + {1,2,4,8} batch-bucket graphs (smallest bucket covering the fullest rank, per-bucket `Glm52BucketState`). Solo 22.4 ms/step; D2's c9 cliff killed (47.1 → 31.8 ms/step, 171 → 254 tok/s); poisson soaks clean; pinned slot-3/7 parity PASS. Known: buckets are distinct FP associations (bucket-crossing requests can greedy-diverge at near-ties); open anomaly: one-off silent request drop (#551). |
| `models/glm52/batched-decode.md` | Dynamic-batching D1 execution record: fixed 8-row decode batch per rank (pad rows in free slots, per-slot KV regions, global tokens 8→64, zero DeepEP shim changes). jz-38: oracle gates green, solo output byte-identical to the PR5c record, all e2e gates PASS; pad-row cost measured 22.5→39.2 ms/step ⇒ D2 must add {1,8} batch-bucket graphs. |
| `models/glm52/dp1-ep8-decode-plan.md` | Five sub-PRs from the PR #476 load-weight scaffold to DP1/EP8 DSA decode serving: PR1 MLA projection/absorb/cache brick (full top-k, short-context DSA-equivalent), PR2 DSA indexer chain (DeepGEMM paged MQA logits + FlashInfer deterministic top-k=2048 + slot conversion), PR3 EP1 full forward (dense+MoE+bookends), PR4 DeepEP EP8 MoE all-to-all, PR5 scheduler+CUDA Graph. Flags `glm52_moe_quant.cu`, `glm52_mla_assembly.cu`, `glm52_indexer.cu` as hand-written perf debt. |
| `models/glm52/mla-decode-brick.md` | PR1 dev doc: build instructions, kernel inventory, hand-written CUDA perf-debt flags. Oracle gate deferred — fixture pipeline was not self-contained. |
| `models/glm52/oracle-harness.md` | Self-contained accuracy oracle: `tools/accuracy/glm52_oracle.py` (pinned transformers 5.12.1 official `glm_moe_dsa`, fp8-precision-emulated) emits hardcodable Rust probe constants; `oracle/mla.rs` replays the seeded input and asserts. MLA gate green on jz38 (64/64 probes, diff RMS 1.8e-5), negative controls red. No MB fixtures in git. |
| `models/glm52/dsa-indexer.md` | PR2 dev doc: DSA indexer chain that replaces PR1's full top-k with sparse top-k=2048. Five kernel ops — two cherry-picked hand-written cache kernels, FlashInfer `TopKDispatch` K=2048 wrapper (vendored), hand-written `local_topk_to_slots` (ported from TokenSpeed Triton), hand-written naive Hadamard, DeepGEMM paged MQA logits C ABI wrapper (vendored JIT, highest risk). |
| `models/glm52/indexer-forward.md` | PR2 model-crate doc (supersedes the plan's PR2 section): `glm52_indexer_forward` composing the #489 kernel ops, aligned to vllm's `DeepseekV32Indexer` — k_norm is LayerNorm-with-bias, weights-fold into DeepGEMM, no Hadamard. Merged as #521; oracle gate green (2013/2048 slot overlap). |
| `models/glm52/ep8-deepep-moe.md` | PR4: GLM-baked DeepEP v2 shim instantiation replaces PR3's local scatter/combine; loader places experts into their packed layout at H2D time (post-load repack cannot fit HBM); rank 0 runs the full 78-layer spine + bs=1 greedy coordinator, ranks 1..7 replay the 75 MoE collectives per step. Gates: EP8 layer-6 oracle 62/64 (same outliers as EP1), full-model e2e generation. |
| `models/glm52/ep1-forward.md` | PR3 built + all gates green on jz-38 H200 (2026-07-03): MoE/dense/bookend bricks (cherry-picked from the PP8 branch, re-gated via the #499 harness) + decoder-layer composition with cross-layer top-k sharing. MoE chain shaped to the DeepEP v2 elastic shim contract, Grouped + GEMV expert paths behind one signature; graph capturability as the bar. Gates: bookend exact, layer-0 dense 64/64, layer-6 MoE 62/64 both paths (measured router near-ties, bounded allowance). |
| `models/glm52/bs1-decode-serial-overhead.md` | PR5a perf pass on the PR4 bring-up path: 101–103 → 46–50 ms/step (~2.2×) at bs=1, output byte-identical, all gates green. Fixes: quant/SiLU/GEMM rows bounded by the coordinator token count (device trap on violation), persistent MoE workspace (was ~11.6k allocs/step), FlashMLA sched metadata hoisted to build. Remaining gap = launch overhead → PR5c graph target. |
| `models/glm52/dp8-scheduler.md` | PR5b: DP8 lock-step scheduler — non-expert stack replicated to every rank, one request per rank, padding tokens on idle ranks (`GLM52_DECODE_GLOBAL_TOKENS` 1→8). All 8 jz-38 gates green, outputs byte-identical to PR5a; retired the DeepGEMM MQA runtime JIT (3 latent bugs) via build-time AOT. Known: fixed ~200 ms step cost (8-way concurrency free) → PR5c graph. |
| `models/glm52/whole-step-decode-graph.md` | Whole-step decode graph execution record: 200 → **19.6 ms/step** on jz-38 (below the vLLM 20.0 reference) via CUDA graph + weight-only fp8 GEMV (tensor-core mma at batch 4/8) + DeepGEMM masked grouped expert GEMM (the earlier "swapAB" attribution retracted inside). Remaining lever = collective wait structure (#542); indexer oracle reference drift (#541). |

## models / deepseek-v4

| Path | TL;DR |
| --- | --- |
| `models/deepseek-v4/support.md` | Initial DeepSeek V4 support PR record: native MP8 engine, official-style TileLang build-time kernels, exact E2E, HTTP validation, nsys-guided speed fixes, prefill RoPE reuse, sync removal, scratch reuse, and GPU index generation. |
| `models/deepseek-v4/decode-performance.md` | Fixed long decode is retained sub-30 with exact E2E `20/20` and hash `6346f03343d75a65`; stable sub-25 remains open. |
| `models/deepseek-v4/serving-baseline.md` | Serving baseline gate: HTTP single-request smoke and direct TPOT/hash regression available; bs>1 serving, continuous batching, and service-level KV management remain follow-up. |
| `models/deepseek-v4/http-serving-benchmark.md` | HTTP serving benchmark gate: streaming `/v1/completions` load records QPS, TTFT, TPOT/ITL, latency percentiles, error rate, and output hashes without using direct bench as serving evidence. |
| `models/deepseek-v4/online-throughput.md` | Latest-main DSV4 online throughput baseline: direct/HTTP/mixed 5090 results, input/output tok/s, bs>1 operator coverage, CUDA Graph blockers, and next task routing. |
| `models/deepseek-v4/prefix-paged-kv-pd-handoff.md` | Prefix/paged KV and P-D handoff design contract: evolves slot-owned direct KV leases into page ownership, prefix cache, allocator telemetry, and transport-agnostic handoff handles. |
| `models/deepseek-v4/moe-ag-rs.md` | Decode MoE now uses GPU AG/RS, GPU route compaction, and grouped TileLang FP4 local experts; no route/expert D2H in hot path. Current 1x32 TPOT avg `105.54ms`, exact E2E `20/20`. |
| `models/deepseek-v4/moe-tilelang-review.md` | Persistent rank workers + decode-only direct top-k MoE cut 1x32 steady TPOT to `80.49ms/token`; remaining cost is rank arrival skew before `107` f32 collectives/token. |
| `models/deepseek-v4/pplx-ep-integration.md` | DeepSeek V4 PPLX EP integration: pplx-garden decode MoE path, EP8 bootstrap, common NUMA rank-slice placement, and H200 steady TPOT p50 `66.65ms`. |
| `models/deepseek-v4/kernel-paths.md` | DeepSeek V4 CUDA sources, TileLang generator path, and `openinfer-kernels/KERNELS.md` routing index are organized. |

## models / deepseek-v2-lite

| Path | TL;DR |
| --- | --- |
| `models/deepseek-v2-lite/status.md` | DeepSeek-V2-Lite EP2 status ledger: HF/host-staged/NCCL exactness, mixed-request serving, HTTP trace evidence, and retained vLLM TP2/EP2 benchmark boundaries; no vLLM parity or production serving claim. |
| `models/deepseek-v2-lite/hf-accuracy-gate.md` | DeepSeek-V2-Lite EP2 HF accuracy gate after PR #149/#150/#274: HF `generate(use_cache=true)`, host-staged EP2, and NCCL EP2 are compared across the committed small case set. |
| `models/deepseek-v2-lite/decode-attribution-gate.md` | DeepSeek-V2-Lite EP2 decode attribution gate for `Hello`/16-token batch sizes 1/4/8: structured JSON with accuracy hashes, timing/counters, separated NCCL all-reduce smoke, and fail-closed full-decode graph probe evidence for the retained batch-1 shape. |
| `models/deepseek-v2-lite/source-layout.md` | DeepSeek-V2-Lite runtime layout refactor: `runtime.rs` split by responsibility, HF/host-staged/NCCL EP2 E2E exact on 2x RTX 5090; NCCL CUDA Graph smoke remains a diagnostic blocker on that host, independent of the passed correctness gate. |
| `models/deepseek-v2-lite/device-resident-nccl-combine.md` | Issue #275 record: NCCL decode combine uses reusable device-resident f32 scratch; current NCCL graph-readiness blockers live in `status.md`. |

## models / kimi-k2

| Path | TL;DR |
| --- | --- |
| `models/kimi-k2/roadmap.md` | Cross-cutting Kimi-K2 plan, re-verified 2026-06-08 on 8×H200. Decode leads vLLM on the active TP1/DP8 **DeepEP** line (bs64 graph TPOT `26.3 ms` p50 / `30.5` p99); M1 serving contract (sampling/EOS/admission) + M2 accuracy gate shipped and green teacher-forced. Live frontier = serving perf: the "+51% HTTP" (#225) was a **bench/metric artifact** (measured: identical prompts under-measure decode ~7–15% via the Marlin expert GEMM; transport ≈0) — floor ~34 ms, a2a ~30% GPU (#228); TTFT 4.5×/31× behind vLLM (#224). Open correctness debt: tests (#222), concurrent mispick (#286), graph-replay gate (#300). |
| `models/kimi-k2/accuracy-gate.md` | vLLM-golden accuracy gate (#223)：`tests/vllm_golden_gate.rs` + committed K2.6 fixture，teacher-forced regret sweep + free-greedy decode parity，走真实 serving path（TP1/DP8/EP8 PPLX）；两档 regret 规则（自信位 0.30 / 平分布位 1.25 且每 pass 限 2 个），缺模型/fixture 显式 fail。 |
| `models/kimi-k2/deepep-migration.md` | PPLX→DeepEP 迁移已实现：kimi 路径 PPLX 全删（moe_pplx.rs 没了，kimi crate 不再依赖 openinfer-comm）；decode `expand=true`+`cpu_sync=false` 零 host 同步/分配（graph-ready，#227 capture 仍关）；Marlin 原地消费 recv buffer（alignment 8 == block size，identity routing + sentinel）；router scale 在 residual 处应用，combine 提前一步 bf16 取整。待 8×H200 数值 gate + serving bench。 |
| `models/kimi-k2/sampling.md` | Sampling param surface + design (#237)：TP1/DP8 上 temperature/top_k/top_p 经单次 batched FlashInfer pass 生效（greedy 行保持 in-graph argmax，零开销），TP8 显式拒绝非 greedy；OpenAI 参数表逐项标注 honored/rejected/ignored，无静默路径；8×H200 已验证 e2e + TPOT 无回归。 |
| `models/kimi-k2/kv-cache-design.md` | KV cache 接入 qwen3 paged 栈 (#239→#230/#231)，单 PR 落地：kimi kernel 层本就 paged，kernel 零改动；kvbm `BlockPool` per rank 取代静态 slot→pages 映射，full-lifetime reservation admission + 超界显式 Rejected，per-request cap 2048→8192（DP prompt 仍 ≤2048，PPLX fabric buffer 约束）；#230/#231 的 substrate，8×H200 验证待做。 |
| `models/kimi-k2/optimization.md` | Kimi-K2 model card + decode 优化主线。Active mainline 是 TP1+DP8+EP8 PPLX（decode batch cap 64，buckets `[1,2,4,8,16,32,64]`，bs64 output `1336 tok/s`）；下半篇的 TP8+EP8 NCCL bs4 graph TPOT `14.39ms` 路径是历史 bring-up 记录，保留以解释 MLA/MoE/collective kernel 结构。 |
| `models/kimi-k2/bringup-history.md` | Kimi-K2 text-only bring-up 压缩史（合并自旧 support-analysis/changelog/operator-todo trio）：HF probe → 文本 manifest → TP8/EP8 sliced loader → MLA + Marlin WNA16 routed expert → NCCL bridge → bs4 wave decode → 整段 CUDA Graph → vLLM top-20 gate。持有 still-load-bearing 的 checkpoint/INT4/Marlin layout facts 与 #234 tombstone（expert-major CUTLASS 删除、weight_shape 不再加载、bs4 cap → 64）。 |
| `models/kimi-k2/vllm-path-comparison.md` | Kimi-K2 decode 路径对照：vLLM-style fused qkv_a、MoE shared/routed compute overlap、shared/dense gate-up fusion、routed scaled-add 和 bridge microbench 已过 H20 gate；output64 avg/p50/p99 均在 `15ms` 内，vLLM TP-only MoE final all-reduce BF16/F32 两版均慢于当前 RS bridge。 |
| `models/kimi-k2/vllm-h20-baseline.md` | vLLM 0.19.0 H20 ×8 TP1+DP8+EP8 decode-heavy baseline：bs 1..256 扫描，bs=8 拐点 TPOT med `26.4ms` / aggregate `308 tok/s`，bs=256 拉到 `1131 tok/s`；同 client 下 openinfer TP8+EP8 bs=4 TPOT `19.13ms` 比 vLLM 低 23%，但 HTTP 口径比 in-process 高 33%，frontend overhead 待查。 |
| `models/kimi-k2/pplx-ep-decode.md` | PPLX EP decode bs=1 TPOT 37ms → 17.94ms（−52%），超过 NCCL no-graph 18.52ms。根因是 expert_padding=64 导致 Marlin 98% 计算浪费 + <<<1,1>>> 串行 routing kernel。含完整优化 log、failed approaches、nsys 对比数据。 |
| `models/kimi-k2/pplx-ep-correctness.md` | TP8/EP8 PPLX correctness baseline：H20 64-token token trace 与 TP8/EP8 NCCL 完全一致，hash `4920f088c2338236`；记录 recv capacity、routed-row top-k weight、F32 combine 边界。 |
| `models/kimi-k2/tp1-dp8-ep8-performance.md` | TP1 DP8 EP8 性能优化 ledger：O1 prompt_len1 decode admission 过 vLLM bs64 gate；O2 落地 5 个 decode kernel cherry-pick（cuBLASLt fixed-shape GEMM、argmax split、router fusion），精度由 base-vs-opt prefill logits A/B 压在 bf16 ULP 底，PPLX Marlin small-N tile 因 `-inf`/SIGSEGV 被定性为原分支精度破坏点并拒绝；bs64 TPOT 噪声内持平（p50 `40.58→40.09ms`）。 |
| `models/kimi-k2/source-layout.md` | Kimi-K2 source files over 1k lines were split by responsibility; the largest Rust file under `openinfer-kimi-k2/src` is now `layers/attention.rs` at 950 lines. |
| `models/kimi-k2/dp-design.md` | TP×DP 可配置并行：每 DP rank 是独立 decode engine，EP all-to-all 天然 sync，轻量 load balancer 做 request 路由。首批 TP1×DP8 + TP8×DP1。 |

## subsystems / runtime

| Path | TL;DR |
| --- | --- |
| `subsystems/router/kv-aware-routing.md` | Dynamo KV-aware routing on 8×Qwen3-4B (RTX 5090): cache-affinity routing keeps a multi-turn conversation on its home worker, so follow-up-turn TTFT stays flat ~45ms vs round-robin 160–170ms / random 165–180ms (all-turns p50 3.3–3.8× lower). Router prefix overlap 0.72 under KV, 0 under stateless policies; `kv_hit_rate>0` is the gate that the worker↔router block-hash bridge is actually matching. Includes the per-response `prompt_tokens_details.cached_tokens` signal. |
| `subsystems/runtime/runtime.md` | Runtime complexity is controlled by a shared `openinfer-core` that owns the generation contract and orchestration; per-model crates implement `ModelForward` so prefill/decode and hybrid attention stay hidden from the caller. State (`&mut`) is separated from weights (`&self`) for future bs > 1. |
| `subsystems/runtime/kv-cache-design.md` | Dynamo 式 logical/physical 分层 KV cache：BlockManager 管 block 生命周期和 admission，PhysicalBackend trait 管 GPU 内存和布局（FullAttention / MLA）。支持 TP / DP。基于 vLLM/Dynamo/pegaflow 调研。 |
| `subsystems/runtime/pegaflow-offload-integration.md` | 把 `pegaflow-core` 当进程内 Rust 库做 KV 卸载物理后端（HBM→DRAM/SSD/RDMA），补 kvbm 没写的卸载层。**Qwen3-4B full-attn 首发，端到端已在真实 GPU 跑通并验证**（async SAVE+LOAD 接进 executor/scheduler，纯 CPU-hit 与 GPU+CPU 组合 hit 恢复后 logits 与冷算一致）。pegaflow 经 git rev pin（#331+#333）。默认关，server CLI 已接（#316：`--kv-offload`/`--no-prefix-cache`，plain+LoRA）。linear 排除，sparse 暂缓。 |

## subsystems / scheduler

| Path | TL;DR |
| --- | --- |
| `subsystems/scheduler/scheduler.md` | Single dedicated thread owns GPU; FCFS prefill-priority, paged KV, bucket CUDA Graphs, unified forward for mixed prefill+decode. Qwen3-4B at QPS=2 is within 2% of vLLM throughput while winning TTFT (-16%), TPOT (-3%), and latency stability. Open: ITL p99 tail, Qwen3.5 full-paged prefill, and high-concurrency wedge triage. |
| `subsystems/scheduler/output-dispatch.md` | GPU bubble study + token-dispatch redesign (**landed 2026-06**). Single-thread CPU↔GPU(sync) alternation idles the GPU through scheduling; bubble ≈3µs×batch (bs=128 → ~380µs, 2% of an 18ms step on 5070 Ti), dominated by N per-request `token_tx.send` wakeups. Fix shipped: `token_tx` is a `TokenSink` drop-in over one request-tagged channel + one bridge demux loop (N→1 wakeups/tasks/ZMQ msgs); cancellation rides an `Arc<AtomicBool>` flag, not a separate channel. Bubble target ~150µs (exec_cpu floor). Trigger: fast GPUs (→10–15%) or N≫128. |
| `subsystems/scheduler/qwen-batched-sampling.md` | Issue #284 record: Qwen3/Qwen3.5 mixed greedy/non-greedy token selection compacts non-greedy rows into one batched FlashInfer sampling call per step, with greedy rows staying on indexed batched argmax. |

## subsystems / sampling

| Path | TL;DR |
| --- | --- |
| `subsystems/sampling/openinfer-sample.md` | `openinfer-sample` is the one crate every model routes through for batched token selection (`select_batch`) and host logprobs (`token_logprob_from_row`, generic over f32/bf16). Replaces `core::ops::select_batch_tokens_into` + three copies of the logprob math. Kimi keeps its sharded-vocab greedy argmax (a DP concern the whole-vocab `select_batch` can't express) but shares the non-greedy sampler and the logprob math. |

## subsystems / frontend

| Path | TL;DR |
| --- | --- |
| `subsystems/frontend/simulated-inference-engine.md` | CPU-only simulated model crate for vLLM/OpenAI frontend and `vllm bench serve` validation without CUDA, real model weights, or real-model performance claims. |
| `subsystems/frontend/cpu-profiling-baseline.md` | Frontend CPU profiling baseline using `openinfer-sim` with fixed TTFT=5ms/TPOT=12ms: 200 req / concurrency=16 shows ~150ms TTFT overhead (no dominant hotspot), heap allocation ~10%, stream polling ~7.5%, IPC ~1%; reproducible benchmark command and perf evidence documented. |
| `subsystems/frontend/startup-time.md` | Qwen3-4B warm startup-to-ready 3.25s → ~1.45s: frontend tokenizer load runs concurrently with the engine load (HTTP still binds only after the engine registers), and the source safetensors mmap is kept alive to dodge ~0.4s of munmap stalling the next cudaMalloc. |

## subsystems / correctness

| Path | TL;DR |
| --- | --- |
| `subsystems/correctness/logits-golden-gate.md` | Reusable pattern for guarding a model's logits against an HF bf16 golden without binding to one GPU's bits: teacher-force fixed sequences, assert a structural regret check on the argmax + mean/p99 of the logprob delta at the bf16 floor (never the absolute max — it grows with coverage). Replay bs=1 / batched eager / CUDA-graph for determinism / cross-request / padding surfaces. Qwen3-4B is the reference impl. |

## subsystems / kernels

| Path | TL;DR |
| --- | --- |
| `subsystems/kernels/openinfer-kernels-boundary.md` | Architecture decision: reusable frontend/runtime/data-plane layers plus per-model engines; `openinfer-kernels` keeps shared MoE/MLA substrate (`moe`: DeepEP/DeepGEMM/FlashMLA) separate from model-local surfaces such as the narrow GLM5.2 DeepGEMM/FlashMLA wrappers. |
| `subsystems/kernels/kernel-op-reports.md` | Qwen3 kernel/report tooling is feature-gated: `qwen3_kernel_report` covers per-op kernel reports, and `qwen3_model_report` emits runtime-traced eager-DAG decode operator rollups with TensorSpec `KernelCall`s, latency stats, tables, and Graphviz DOT; measured FA2 `CTA_TILE_Q=64` prefill default in place. |
| `subsystems/kernels/typed-forward-pipeline.md` | Reusable typed tensor pipeline macro in `openinfer-kernels` so model crates can express common `typed_ops` chains without model-specific wrapper macros. |
| `subsystems/kernels/tvm-ffi-mvp.md` | Optional `tvm-ffi-triton-cubin` bridge in `openinfer-kernels` plus a packed TVM wrapper for the Qwen3.5 GDR solve Triton AOT CUBIN launcher. |

## playbooks

| Path | TL;DR |
| --- | --- |
| `playbooks/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, quick benchmark validation. |
| `playbooks/bench-vs-vllm.md` | openinfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas. |
| `playbooks/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log. |
| `playbooks/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons. |
| `playbooks/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands. |

## lessons

| Path | TL;DR |
| --- | --- |
| `lessons/moe-bench-prompt-diversity.md` | MoE decode TPOT is routing-diversity-dependent: identical concurrent prompts route greedy streams to the same experts and under-measure decode TPOT by **~7–15%** (measured via a `--distinct-prompts` sweep, not the ~30% first claimed). Bench MoE+EP with seeded distinct prompts. nsys kernel diff proves the whole delta is the **Marlin expert GEMM** (per-launch ~2× K=1→64); the DeepEP all-to-all is flat → lever is grouped-GEMM tile efficiency, not a2a overlap (#228). Transport ≈0. |
| `lessons/profile-diff-before-blaming-transport.md` | Profiling discipline from the #225 misfire: when two profiles of one workload differ in wall-time, **diff `cuda_gpu_kern_sum` first** — transport can't change GPU kernel time, so a kernel delta means compute/data, full stop. I nsys'd both paths and missed a +15.6% Marlin delta in plain view. Also: pin the same metric both sides; chase tails, don't annotate them; a root cause without a number is a hypothesis. |
| `lessons/moe-dplb-decode-imbalance.md` | DPLB lesson for future PegaFlow/WiDeep MoE+EP serving: decode-side DP imbalance is a sticky KV-state problem; engines should emit raw progress while external router/proxy derive load and routing. |
| `lessons/moe-zero-prefill-long-prefill.md` | ZeRO-Prefill lesson for future long-prefill MoE serving: once a router selects long-P work, maximize batch throughput by preserving compute-bound execution, hiding expert-weight movement, respecting KV handoff boundaries, and measuring bottlenecks before committing to an AsyncEP-style backend. |
| `lessons/exact-match-gate-thread-cublas.md` | Two durable lessons from a Qwen3.5 e2e gibberish bug: worker threads that run a model must rebind the CUDA context and init thread-local cuBLAS handles, and exact-match greedy gates are sensitive to equal-logit top1 choices (keep a single FlashInfer selector). |
| `lessons/kimi-bringup-numerics.md` | Three MoE+TP greedy-parity / reporting lessons from Kimi-K2 bring-up, reusable on any MoE+TP decode engine gated on token-id parity: reduce hidden states in F32 not BF16 (BF16 bulk all-reduce silently breaks greedy); don't merge shared+routed expert reduce into one collective (breaks cold-batch greedy); always report p50+p99, never just mean (tail dominates on barrier-synced MoE+EP decode). |
| `lessons/cuda-green-contexts.md` | Local mirror of NVIDIA CUDA 13.1+ Green Contexts guide (§4.6): static SM/workqueue partitioning via runtime execution contexts; host-only changes, no kernel edits. Generated by `scripts/html_to_md.py`. |

## benchmarks

| Path | TL;DR |
| --- | --- |
| `benchmarks/qwen3-4b-serving-vllm-rtx5090.md` | **Deleted** — superseded by `models/qwen3/serving-performance.md`. |
| `benchmarks/deepseek-v2-lite-vllm-tp2-ep2-2026-06.md` | DeepSeek-V2-Lite EP2 2026-06-28 snapshot: OpenInfer host-staged/NCCL passed correctness, direct diagnostics, HTTP pressure, and trace rows; stock vLLM TP2/TP2+EP2 are retained as FlashInfer SM120/CUDA 12.8 setup failures, with a separate FlashInfer-fixed vLLM validation and no parity claim. |
| `benchmarks/qwen35-4b-serving-vllm-rtx5090.md` | Qwen3.5-4B TP1 vs vLLM 0.23.0 on RTX 5090: latest direct OpenInfer A/B improves TPOT by 2-3%; HTTP `vllm bench serve` shows prompt-len-1 decode close, but vLLM still leads 1024/256 TPOT and high-concurrency output tok/s. Includes Nsight Systems direct/HTTP gap notes. |
| `benchmarks/qwen-mixed-sampling-http.md` | Issue #412 HTTP mixed-sampling evidence: Qwen3-4B `/v1/completions` completed 64/64 with 32 greedy + 32 sampled requests, failed=0/timeouts=0, TTFT/TPOT/ITL/output tok/s retained; Qwen3.5-4B passed the same workload as supplemental evidence. |
| `benchmarks/bs1-4k64-vllm-openinfer.md` | RTX 5090 single-concurrency probe: `input_len=4096`, `output_len=64`, no vLLM prefix cache. OpenInfer TTFT median `177ms` vs vLLM `198ms`; TPOT median `6.47ms` vs `6.36ms`; corrected output throughput `+6%` for OpenInfer. |
| `benchmarks/mixed-load-itl.md` | Qwen3-4B + Qwen3.5 mixed-load ITL (#244, #375): chunking-off sweeps (qps × prompt × prefix) via `bench_serving mixed`. Both lines freeze every active decode for the full prefill in the unified step (8k≈0.9–1.2s, 12k≈1.4–2.2s). Qwen3 p99 blows up with prompt (8k 1161, 12k 3270ms at qps≥0.5); Qwen3.5's freeze lands just under the 1% p99 knee (shows in `max`, not p99 — measurement artifact of the 1024-tok background, **not** immunity). Prefix reuse defeats it. #375 chunked prefill caps the per-step freeze. |
| `benchmarks/accuracy-eval-results.md` | Phase 1 GSM8K: Qwen3-4B PASS (openinfer 85.37% vs HF 85.82%, delta -0.45 pp). Qwen3.5-4B historical FAIL recovered by #250 (strict 79.38%, flexible 79.30% vs HF 79.45%). |
| `benchmarks/qwen3-8b-pd-vs-mix-h200.md` | Qwen3-8B 多轮负载三方 A/B（2×H200）：P/D 1P+1D vs mixed×2（会话亲和 LB）vs mixed×1。吞吐持平（47.8k vs 47.0k tok/s），P/D 赢在 decode 稳定性（TPOT p99 10.08 vs 12.77ms，turn2+ TTFT 恒定 ~107ms vs 爬升 71→132ms），冷 turn1 多付 ~200ms（M3 目标）。含 vllm-bench 命令与 `max_completion_tokens` 坑。 |
| `benchmarks/pplx-ep-a2a-h20-nvlink.md` | pplx EP all-to-all latency on 8× H20 NV18 NVLink: DSV4 & Kimi-K2 shapes, tok=1..256. tok=1 p50 ~82μs, tok=256 p50 ~204/303μs. |
| `benchmarks/deepep-v2-vs-pplx-moe-backend.md` | H20 x8 DeepEP V2 vs current OpenInfer PPLX EP backend comparison: ElasticBuffer/NCCL Gin shows a directional 2.5x-5.3x paired-run ratio on tested DSV4 and Kimi-K2 MoE exchange shapes, with dtype, correctness, harness, and PPLX baseline-drift caveats recorded. |

## conventions

| Path | TL;DR |
| --- | --- |
| `conventions/bench-regression.md` | Benchmark regression tracking: one snapshot per model, git-tracked history, TPOT >2% / TTFT >3% thresholds. |
| `conventions/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches. |
