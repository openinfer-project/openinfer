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
| `models/qwen3/serving-perf-5090.md` | Qwen3-4B vs vLLM 0.22.1 tuning record: beats vLLM at **every measured QPS point** after unified-step attention fusion (decode rows as qo_len=1 prefill-plan entries + cta_tile_q dispatch fix). Also: batched step tail (#345), chunked prefill (default 1024), **cuBLAS 12.9 N=1025 cliff (build with CUDA ≥ 13)**, cublasLt per-shape tuning (buckets 8/16 restored), split-KV ≤bs32, two-stage argmax. |
| `models/qwen3/roadmap.md` | Qwen3-4B roadmap (2026-06 review): line is the maturity bar; #220 RoPE OOB now fixed (sized cache + admission guard + kernel trap, gated by reject + in-window ITs); open set is per-row batch sampling, zero TP coverage, zero-adapter-only LoRA gate, dropped prefix-cache observability, stale docs, YaRN #8 follow-up. Sequenced Now/Next/Later + cleanup ledger. |
| `models/qwen3/model-crate.md` | `openinfer-qwen3-4b` owns Qwen3 config/weights/executor/scheduler/tests/kernel plan; root sees generic `EngineHandle`; split-K retuned to `256/64`, with 4k/64 serving TPOT p50 at `6.46ms` on RTX 5090. |
| `models/qwen3/prefix-cache.md` | Prefix caching on by default for Qwen3-4B: full-block kvbm radix matching at the executor, suffix-only prefill. Repeated ~1900-token prompt TTFT 141.8 → 16.3ms p50 (8.7×); warm TTFT ≈ TPOT + ~5ms setup. Includes the RoPE scalar-path corruption fix and the drain-the-stream TTFT measurement pitfall. |
| `models/qwen3/accuracy-gate.md` | Qwen3-4B instance of the logits golden gate (`tests/hf_golden_gate.rs`): 48 teacher-forced sequences / 816 positions vs a stored HF bf16 golden, replayed over bs=1 / batched eager / CUDA-graph. Strict guards: regret check + mean ≤ 0.06 + p99 ≤ 0.20; absolute max printed but not asserted (coverage-unstable). Methodology in `subsystems/correctness/`. |
| `models/qwen3/kernels-crate.md` | Phase 1 split implemented and 5090-verified: Qwen3-4B kernel surface lives in `openinfer-kernels`; release build, test-target compile, accuracy gate, and bench snapshot pass. |
| `models/qwen3/tp-design.md` | Qwen3 tensor-parallel design: `TP=2` milestone scope plus the controller/worker broadcast execution model, request identity, and coarse-grained step protocol for future TP/MoE work. |
| `models/qwen3/kv-pressure-hang.md` | Issue #85 Qwen3-4B KV pressure hang fixed by full-lifetime scheduler KV admission, waiting-queue deferral, cleanup on disconnect/error, impossible-request errors, scheduler/bridge gates, and real `vllm bench serve` QPS=2 `500/500` pass with post-pressure completion healthy. |

## models / qwen35

| Path | TL;DR |
| --- | --- |
| `models/qwen35/roadmap.md` | Qwen3.5-4B roadmap (2026-06 review): fast and decode-correct, #186 adds the teacher-forced HF logits gate, and #250 covers the old 4096 RoPE boundary with a 4097/8192-token HF logits replay plus a recovered GSM8K 8-shot score; ~640MB HND prefill staging remains, with the 20k cap now fail-closed, pre-#85 admission semantics still open, and current scheduler admission/slot/compaction policy is now CPU-tested. |
| `models/qwen35/kv-admission.md` | Issue #254 complete: Qwen3.5 now uses full-lifetime KV admission, deferred pressure handling, impossible-request rejection, explicit error semantics, direct rejection-event coverage, RTX 5090 e2e, and real HTTP pressure/post-pressure validation. |
| `models/qwen35/optimization.md` | Hybrid 24 linear + 8 full attn. At parity with vLLM: TTFT 234ms (+2%), TPOT 11.77ms (+1%). Post-accuracy-fix GDR decode kernel restore (#9). |
| `models/qwen35/accuracy.md` | Qwen3.5-4B HF bf16 logits goldens through `past_key_values`: short replay covers sequential graph, bucket-straddling batched graph, and slot-compaction; long replay covers 4097/8192-token prompts; full GSM8K 8-shot now matches the HF baseline within 0.15 percentage points. |
| `models/qwen35/model-crate.md` | `openinfer-qwen35-4b` owns Qwen3.5 model/scheduler/recurrent ops/tests/benches; feature-gated behind `qwen35-4b` (Triton AOT is the only Python build dependency); root loads it through `EngineHandle`. Build/check/clippy, root bench sanity check, historical Qwen3.5 e2e, and scheduler e2e records live here. |
| `models/qwen35/kernel-plan.md` | Qwen3.5-4B has a `openinfer_qwen35_4b::kernel_plan()` static descriptor mirroring the qwen3 module — enumerates every prefill/decode/unified op with its Rust call site, backend, and notes, so you can dump the active kernel mix without reading call sites. Pure refactor (issue #256), no kernel behavior change. |

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
| `models/deepseek-v2-lite/status.md` | DeepSeek-V2-Lite EP2 model status and benchmark ledger: HF/host-staged/NCCL use a committed small correctness case set; direct batch, OpenInfer HTTP pressure, and vLLM TP2 / TP2+EP2 rows from a documented validation environment remain diagnostic evidence, not production serving parity. |
| `models/deepseek-v2-lite/hf-accuracy-gate.md` | DeepSeek-V2-Lite EP2 HF accuracy gate after PR #149/#150/#274: HF `generate(use_cache=true)`, host-staged EP2, and NCCL EP2 are compared across the committed small case set. |
| `models/deepseek-v2-lite/decode-attribution-gate.md` | DeepSeek-V2-Lite EP2 decode attribution gate for `Hello`/16-token batch sizes 1/4/8: structured JSON with accuracy hashes, CPU-side timing, selected CUDA event/NVTX attribution, host-staged/NCCL EP counts, and explicit no-throughput claim boundary. |
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
| `subsystems/runtime/runtime.md` | Runtime complexity is controlled by a shared `openinfer-core` that owns the generation contract and orchestration; per-model crates implement `ModelForward` so prefill/decode and hybrid attention stay hidden from the caller. State (`&mut`) is separated from weights (`&self`) for future bs > 1. |
| `subsystems/runtime/kv-cache-design.md` | Dynamo 式 logical/physical 分层 KV cache：BlockManager 管 block 生命周期和 admission，PhysicalBackend trait 管 GPU 内存和布局（FullAttention / MLA）。支持 TP / DP。基于 vLLM/Dynamo/pegaflow 调研。 |
| `subsystems/runtime/pegaflow-offload-integration.md` | 把 `pegaflow-core` 当进程内 Rust 库做 KV 卸载物理后端（HBM→DRAM/SSD/RDMA），补 kvbm 没写的卸载层。**Qwen3-4B full-attn 首发，端到端已在真实 GPU 跑通并验证**（async SAVE+LOAD 接进 executor/scheduler，纯 CPU-hit 与 GPU+CPU 组合 hit 恢复后 logits 与冷算一致）。pegaflow 经 git rev pin（#331+#333）。默认关，server CLI 已接（#316：`--kv-offload`/`--no-prefix-cache`，plain+LoRA）。linear 排除，sparse 暂缓。 |

## subsystems / scheduler

| Path | TL;DR |
| --- | --- |
| `subsystems/scheduler/scheduler.md` | Single dedicated thread owns GPU; FCFS prefill-priority, paged KV, bucket CUDA Graphs, unified forward for mixed prefill+decode. Qwen3-4B at QPS=2 is within 2% of vLLM throughput while winning TTFT (-16%), TPOT (-3%), and latency stability. Open: ITL p99 tail, Qwen3.5 full-paged prefill, batched per-row sampling redesign. |

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
| `subsystems/kernels/openinfer-kernels-boundary.md` | Architecture decision: openinfer should use reusable frontend/runtime/data-plane layers plus per-model engines; kernels become first-class assets through a ledger, simulator, and request tracing. |
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

## benchmarks

| Path | TL;DR |
| --- | --- |
| `benchmarks/qwen3-4b-serving-vllm-rtx5090.md` | Qwen3-4B TP1 vs vLLM 0.22.1 on RTX 5090 (2026-06-13, @`0b42ed3`), README source: footprint ~5× smaller (771 MB vs 3.8 GB) + ~3 s vs 70 s cold start; warm prefix-cache TTFT leads all lengths, 3.6× at 16k (26.3 vs 95.6 ms); QPS sweep comparable at low load, vLLM TPOT edge mid-load, openinfer edges ahead at saturation (1794 vs 1692) post-#362; pegaflow KV-offload host restore 2.6–9.1×. |
| `benchmarks/qwen35-4b-serving-vllm-rtx5090.md` | Qwen3.5-4B TP1 vs vLLM 0.23.0 on RTX 5090 (2026-06-15, @`f3dcdf4`), README source: fixed HTTP `vllm bench serve` n=30/warm1; openinfer reports lower TTFT p50 under the fixed-client contract, but prompt-token counts differ (1,944 vs 2,048 and 971 vs 1,024/request); vLLM leads decode TPOT (6.32 vs 7.31 ms) and output tok/s (152.3 vs 133.6). |
| `benchmarks/bs1-4k64-vllm-openinfer.md` | RTX 5090 single-concurrency probe: `input_len=4096`, `output_len=64`, no vLLM prefix cache. OpenInfer TTFT median `177ms` vs vLLM `198ms`; TPOT median `6.47ms` vs `6.36ms`; corrected output throughput `+6%` for OpenInfer. |
| `benchmarks/mixed-load-itl.md` | Qwen3-4B mixed-load ITL (#244): long prompts into steady-state decode via `bench_serving mixed`. A prefill freezes every active decode for its full duration (4k→~490ms, 10k→~2730ms); reaches ITL p99 only when stall-gap fraction >~1%. Low-QPS moderate-prompt profile p99 baseline-order (~15–19ms); 1 req/s or 10k prompt → 487/2818ms. Chunked prefill = conditional no-go. |
| `benchmarks/accuracy-eval-results.md` | Phase 1 GSM8K: Qwen3-4B PASS (openinfer 85.37% vs HF 85.82%, delta -0.45 pp). Qwen3.5-4B historical FAIL recovered by #250 (strict 79.38%, flexible 79.30% vs HF 79.45%). |
| `benchmarks/pplx-ep-a2a-h20-nvlink.md` | pplx EP all-to-all latency on 8× H20 NV18 NVLink: DSV4 & Kimi-K2 shapes, tok=1..256. tok=1 p50 ~82μs, tok=256 p50 ~204/303μs. |
| `benchmarks/deepep-v2-vs-pplx-moe-backend.md` | H20 x8 DeepEP V2 vs current OpenInfer PPLX EP backend comparison: ElasticBuffer/NCCL Gin shows a directional 2.5x-5.3x paired-run ratio on tested DSV4 and Kimi-K2 MoE exchange shapes, with dtype, correctness, harness, and PPLX baseline-drift caveats recorded. |

## conventions

| Path | TL;DR |
| --- | --- |
| `conventions/bench-regression.md` | Benchmark regression tracking: one snapshot per model, git-tracked history, TPOT >2% / TTFT >3% thresholds. |
| `conventions/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches. |
