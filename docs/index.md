# docs index

Organized by domain (model line / subsystem / playbook / lesson) instead of by lifecycle stage. A doc's freshness is recorded in its own header (TL;DR / Status), not by which directory it lives in.

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
| `roadmap/direction.md` | One size can't fit all. Shared infrastructure (frontend, runtime primitives, kernels, data plane) + per-model engines with their own scheduler/kernel DAG/state. Long-term loop: kernel ledger → simulator → request tracing. |
| `roadmap/execution.md` | Current state and immediate next steps. No timeline — entries move through In progress → Next → Open. Covers cross-model infrastructure (kernel ledger, simulator, tracing, frontend polish) and per-model active work (DeepSeek V4, Qwen3.5, Qwen3). |

## models / qwen3

| Path | TL;DR |
| --- | --- |
| `models/qwen3/roadmap.md` | Qwen3-4B roadmap (2026-06 review): line is the maturity bar; open set is #220 RoPE OOB, per-row batch sampling, zero TP coverage, zero-adapter-only LoRA gate, dropped prefix-cache observability, stale docs. Sequenced Now/Next/Later + cleanup ledger. |
| `models/qwen3/model-crate.md` | `pegainfer-qwen3-4b` owns Qwen3 config/weights/executor/scheduler/tests/kernel plan; root sees generic `EngineHandle`; split-K retuned to `256/64`, with 4k/64 serving TPOT p50 at `6.46ms` on RTX 5090. |
| `models/qwen3/prefix-cache.md` | Prefix caching on by default for Qwen3-4B: full-block kvbm radix matching at the executor, suffix-only prefill. Repeated ~1900-token prompt TTFT 141.8 → 16.3ms p50 (8.7×); warm TTFT ≈ TPOT + ~5ms setup. Includes the RoPE scalar-path corruption fix and the drain-the-stream TTFT measurement pitfall. |
| `models/qwen3/accuracy-gate.md` | Qwen3-4B instance of the logits golden gate (`tests/hf_golden_gate.rs`): 48 teacher-forced sequences / 816 positions vs a stored HF bf16 golden, replayed over bs=1 / batched eager / CUDA-graph. Strict guards: regret check + mean ≤ 0.06 + p99 ≤ 0.20; absolute max printed but not asserted (coverage-unstable). Methodology in `subsystems/correctness/`. |
| `models/qwen3/kernels-crate.md` | Phase 1 split implemented and 5090-verified: Qwen3-4B kernel surface lives in `pegainfer-kernels`; release build, test-target compile, accuracy gate, and bench snapshot pass. |
| `models/qwen3/tp-design.md` | Qwen3 tensor-parallel design: `TP=2` milestone scope plus the controller/worker broadcast execution model, request identity, and coarse-grained step protocol for future TP/MoE work. |
| `models/qwen3/kv-pressure-hang.md` | Issue #85 Qwen3-4B KV pressure hang fixed by full-lifetime scheduler KV admission, waiting-queue deferral, cleanup on disconnect/error, impossible-request errors, scheduler/bridge gates, and real `vllm bench serve` QPS=2 `500/500` pass with post-pressure completion healthy. |

## models / qwen35

| Path | TL;DR |
| --- | --- |
| `models/qwen35/roadmap.md` | Qwen3.5-4B roadmap (2026-06 review): fast and decode-correct, #186 adds the teacher-forced HF logits gate, and #250 covers the old 4096 RoPE boundary with a 4097/8192-token HF logits replay plus a recovered GSM8K 8-shot score; ~640MB HND prefill staging remains, pre-#85 admission semantics still open, and current scheduler admission/slot/compaction policy is now CPU-tested. |
| `models/qwen35/optimization.md` | Hybrid 24 linear + 8 full attn. At parity with vLLM: TTFT 225ms, TPOT 11.81ms (+1%). Post-accuracy-fix GDR decode kernel restore (#9). |
| `models/qwen35/accuracy.md` | Qwen3.5-4B HF bf16 logits goldens through `past_key_values`: short replay covers sequential graph, bucket-straddling batched graph, and slot-compaction; long replay covers 4097/8192-token prompts; full GSM8K 8-shot now matches the HF baseline within 0.15 percentage points. |
| `models/qwen35/model-crate.md` | `pegainfer-qwen35-4b` owns Qwen3.5 model/scheduler/recurrent ops/tests/benches; root loads it through `EngineHandle`. Build/check/clippy, root bench sanity check, historical Qwen3.5 e2e, and scheduler e2e records live here. |
| `models/qwen35/e2e-gibberish.md` | Historical Qwen3.5 exact-text e2e debug record: scheduler threads now bind CUDA context and initialize thread-local cuBLAS handles. The exact-text gate is retired by the HF logits gate. |

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
| `models/deepseek-v4/kernel-paths.md` | DeepSeek V4 CUDA sources, TileLang generator path, and `pegainfer-kernels/KERNELS.md` routing index are organized. |

## models / deepseek-v2-lite

| Path | TL;DR |
| --- | --- |
| `models/deepseek-v2-lite/status.md` | DeepSeek-V2-Lite EP2 model status and benchmark ledger: HF/host-staged/NCCL are exact for the narrow greedy gate; direct same-prompt batch and manual vLLM snapshots are diagnostic evidence, not production serving parity. |
| `models/deepseek-v2-lite/hf-accuracy-gate.md` | DeepSeek-V2-Lite EP2 HF accuracy gate after PR #149/#150: HF incremental greedy, host-staged EP2, and NCCL EP2 are token/text exact for `Hello`, output_len=16. |
| `models/deepseek-v2-lite/decode-attribution-gate.md` | DeepSeek-V2-Lite EP2 decode attribution gate for `Hello`/16-token batch sizes 1/4/8: structured JSON with accuracy hashes, CPU-side timing, selected CUDA event/NVTX attribution, host-staged/NCCL EP counts, and explicit no-throughput claim boundary. |

## models / kimi-k2

| Path | TL;DR |
| --- | --- |
| `models/kimi-k2/roadmap.md` | Cross-cutting Kimi-K2 plan, verified against code 2026-06: decode beats vLLM (bs64 `1336 vs 594 tok/s`) but serving contract is broken (sampling params silently ignored, no EOS stop, 2048-token arena overrun) and no accuracy gate is clone-reproducible. Sequence: correctness + accuracy-gate-in-git → TTFT/HTTP overhead → batching/prefix-cache and PPLX-throughput chains → cleanup ledger. |
| `models/kimi-k2/optimization.md` | Kimi-K2 model card + optimization log：61 层 MLA + Marlin WNA16 MoE，H20 ×8 当前 TP8/EP8。重点是 decode：bs4 graph TPOT `14.39ms`（≈`278 tok/s`），目标 `> 300 tok/s`；下一阶段迁到 TP1+DP8+EP8（PPLX）。Prefill 优先级低。 |
| `models/kimi-k2/support-analysis.md` | Kimi-K2 text-only bring-up：Marlin WNA16 routed expert、MLA prompt、全 61 层 prompt forward、多 prompt vLLM top-20 gate 已过；bs4 wave decode 已接入，Marlin atomic split-K row-state bug 修复后 output16 row diff 清零，正在撤掉 decode 诊断负担并回到 bs4 性能主线。 |
| `models/kimi-k2/operator-todo.md` | Kimi-K2 算子清单：MLA + Marlin WNA16 routed expert + NCCL RS bridge 主链；CUDA Graph 覆盖整段 decode，synthetic output64 avg `14.39ms` / p99 `14.83ms`；CUTLASS INT4 后端与 decode row-diff 诊断已下线，详见 changelog。 |
| `models/kimi-k2/changelog.md` | Kimi-K2 算子工作日志：Execution Log / Rejected / Debrief / 经验迁移 全部归档，按原始顺序保留；当前状态见 operator-todo。 |
| `models/kimi-k2/vllm-path-comparison.md` | Kimi-K2 decode 路径对照：vLLM-style fused qkv_a、MoE shared/routed compute overlap、shared/dense gate-up fusion、routed scaled-add 和 bridge microbench 已过 H20 gate；output64 avg/p50/p99 均在 `15ms` 内，vLLM TP-only MoE final all-reduce BF16/F32 两版均慢于当前 RS bridge。 |
| `models/kimi-k2/vllm-h20-baseline.md` | vLLM 0.19.0 H20 ×8 TP1+DP8+EP8 decode-heavy baseline：bs 1..256 扫描，bs=8 拐点 TPOT med `26.4ms` / aggregate `308 tok/s`，bs=256 拉到 `1131 tok/s`；同 client 下 pegainfer TP8+EP8 bs=4 TPOT `19.13ms` 比 vLLM 低 23%，但 HTTP 口径比 in-process 高 33%，frontend overhead 待查。 |
| `models/kimi-k2/pplx-ep-decode.md` | PPLX EP decode bs=1 TPOT 37ms → 17.94ms（−52%），超过 NCCL no-graph 18.52ms。根因是 expert_padding=64 导致 Marlin 98% 计算浪费 + <<<1,1>>> 串行 routing kernel。含完整优化 log、failed approaches、nsys 对比数据。 |
| `models/kimi-k2/pplx-ep-correctness.md` | TP8/EP8 PPLX correctness baseline：H20 64-token token trace 与 TP8/EP8 NCCL 完全一致，hash `4920f088c2338236`；记录 recv capacity、routed-row top-k weight、F32 combine 边界。 |
| `models/kimi-k2/dp1-tp8-ep8-performance.md` | DP1 TP8 EP8 性能优化 ledger：从 correctness baseline `72c770b` 起步，目标 bs64 超过 vLLM baseline output `583.9 tok/s` / TPOT median `109.00ms`，每个优化必须带正确性 gate 和 commit。 |
| `models/kimi-k2/tp1-dp8-ep8-performance.md` | TP1 DP8 EP8 性能优化 ledger：O1 prompt_len1 decode admission 过 vLLM bs64 gate；O2 落地 5 个 decode kernel cherry-pick（cuBLASLt fixed-shape GEMM、argmax split、router fusion），精度由 base-vs-opt prefill logits A/B 压在 bf16 ULP 底，PPLX Marlin small-N tile 因 `-inf`/SIGSEGV 被定性为原分支精度破坏点并拒绝；bs64 TPOT 噪声内持平（p50 `40.58→40.09ms`）。 |
| `models/kimi-k2/source-layout.md` | Kimi-K2 source files over 1k lines were split by responsibility; the largest Rust file under `pegainfer-kimi-k2/src` is now `layers/attention.rs` at 950 lines. |
| `models/kimi-k2/dp-design.md` | TP×DP 可配置并行：每 DP rank 是独立 decode engine，EP all-to-all 天然 sync，轻量 load balancer 做 request 路由。首批 TP1×DP8 + TP8×DP1。 |

## subsystems / runtime

| Path | TL;DR |
| --- | --- |
| `subsystems/runtime/runtime.md` | Runtime complexity is controlled by a shared `pegainfer-core` that owns the generation contract and orchestration; per-model crates implement `ModelForward` so prefill/decode and hybrid attention stay hidden from the caller. State (`&mut`) is separated from weights (`&self`) for future bs > 1. |
| `subsystems/runtime/kv-cache-design.md` | Dynamo 式 logical/physical 分层 KV cache：BlockManager 管 block 生命周期和 admission，PhysicalBackend trait 管 GPU 内存和布局（FullAttention / MLA）。支持 TP / DP。基于 vLLM/Dynamo/pegaflow 调研。 |

## subsystems / scheduler

| Path | TL;DR |
| --- | --- |
| `subsystems/scheduler/scheduler.md` | Single dedicated thread owns GPU; FCFS prefill-priority, paged KV, bucket CUDA Graphs, unified forward for mixed prefill+decode. Qwen3-4B at QPS=2 is within 2% of vLLM throughput while winning TTFT (-16%), TPOT (-3%), and latency stability. Open: ITL p99 tail, Qwen3.5 full-paged prefill, batched per-row sampling redesign. |

## subsystems / frontend

| Path | TL;DR |
| --- | --- |
| `subsystems/frontend/simulated-inference-engine.md` | CPU-only simulated model crate for vLLM/OpenAI frontend and `vllm bench serve` validation without CUDA, real model weights, or real-model performance claims. |
| `subsystems/frontend/cpu-profiling-baseline.md` | Frontend CPU profiling baseline using `pegainfer-sim` with fixed TTFT=5ms/TPOT=12ms: 200 req / concurrency=16 shows ~150ms TTFT overhead (no dominant hotspot), heap allocation ~10%, stream polling ~7.5%, IPC ~1%; reproducible benchmark command and perf evidence documented. |

## subsystems / correctness

| Path | TL;DR |
| --- | --- |
| `subsystems/correctness/logits-golden-gate.md` | Reusable pattern for guarding a model's logits against an HF bf16 golden without binding to one GPU's bits: teacher-force fixed sequences, assert a structural regret check on the argmax + mean/p99 of the logprob delta at the bf16 floor (never the absolute max — it grows with coverage). Replay bs=1 / batched eager / CUDA-graph for determinism / cross-request / padding surfaces. Qwen3-4B is the reference impl. |

## subsystems / kernels

| Path | TL;DR |
| --- | --- |
| `subsystems/kernels/pegainfer-kernels-boundary.md` | Architecture decision: pegainfer should use reusable frontend/runtime/data-plane layers plus per-model engines; kernels become first-class assets through a ledger, simulator, and request tracing. |
| `subsystems/kernels/kernel-op-reports.md` | Qwen3 kernel/report tooling is feature-gated: `qwen3_kernel_report` covers per-op kernel reports, and `qwen3_model_report` emits runtime-traced eager-DAG decode operator rollups with TensorSpec `KernelCall`s, latency stats, tables, and Graphviz DOT; measured FA2 `CTA_TILE_Q=64` prefill default in place. |
| `subsystems/kernels/typed-forward-pipeline.md` | Reusable typed tensor pipeline macro in `pegainfer-kernels` so model crates can express common `typed_ops` chains without model-specific wrapper macros. |

## playbooks

| Path | TL;DR |
| --- | --- |
| `playbooks/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, quick benchmark validation. |
| `playbooks/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas. |
| `playbooks/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log. |
| `playbooks/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons. |
| `playbooks/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands. |

## lessons

| Path | TL;DR |
| --- | --- |
| `lessons/moe-dplb-decode-imbalance.md` | DPLB lesson for future PegaFlow/WiDeep MoE+EP serving: decode-side DP imbalance is a sticky KV-state problem; engines should emit raw progress while external router/proxy derive load and routing. |
| `lessons/moe-zero-prefill-long-prefill.md` | ZeRO-Prefill lesson for future long-prefill MoE serving: once a router selects long-P work, maximize batch throughput by preserving compute-bound execution, hiding expert-weight movement, respecting KV handoff boundaries, and measuring bottlenecks before committing to an AsyncEP-style backend. |

## benchmarks

| Path | TL;DR |
| --- | --- |
| `benchmarks/bs1-4k64-vllm-pegainfer.md` | RTX 5090 single-concurrency probe: `input_len=4096`, `output_len=64`, no vLLM prefix cache. PegaInfer TTFT median `177ms` vs vLLM `198ms`; TPOT median `6.47ms` vs `6.36ms`; corrected output throughput `+6%` for PegaInfer. |
| `benchmarks/accuracy-eval-results.md` | Phase 1 GSM8K: Qwen3-4B PASS (pegainfer 85.37% vs HF 85.82%, delta -0.45 pp). Qwen3.5-4B historical FAIL recovered by #250 (strict 79.38%, flexible 79.30% vs HF 79.45%). |
| `benchmarks/pplx-ep-a2a-h20-nvlink.md` | pplx EP all-to-all latency on 8× H20 NV18 NVLink: DSV4 & Kimi-K2 shapes, tok=1..256. tok=1 p50 ~82μs, tok=256 p50 ~204/303μs. |
| `benchmarks/deepep-v2-vs-pplx-moe-backend.md` | H20 x8 DeepEP V2 vs current PegaInfer PPLX EP backend comparison: ElasticBuffer/NCCL Gin shows a directional 2.5x-5.3x paired-run ratio on tested DSV4 and Kimi-K2 MoE exchange shapes, with dtype, correctness, harness, and PPLX baseline-drift caveats recorded. |

## conventions

| Path | TL;DR |
| --- | --- |
| `conventions/bench-regression.md` | Benchmark regression tracking: one snapshot per model, git-tracked history, TPOT >2% / TTFT >3% thresholds. |
| `conventions/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches. |
