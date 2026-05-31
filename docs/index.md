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
| `models/qwen3/model-crate.md` | `pegainfer-qwen3-4b` owns Qwen3 config/weights/executor/scheduler/tests/kernel plan; root sees generic `EngineHandle`; split-K retuned to `256/64`, with 4k/64 serving TPOT p50 at `6.46ms` on RTX 5090. |
| `models/qwen3/accuracy-gate.md` | Qwen3-4B instance of the logits golden gate (`tests/hf_golden_gate.rs`): 48 teacher-forced sequences / 816 positions vs a stored HF bf16 golden, replayed over bs=1 / batched eager / CUDA-graph. Strict guards: regret check + mean ≤ 0.06 + p99 ≤ 0.20; absolute max printed but not asserted (coverage-unstable). Methodology in `subsystems/correctness/`. |
| `models/qwen3/kernels-crate.md` | Phase 1 split implemented and 5090-verified: Qwen3-4B kernel surface lives in `pegainfer-kernels`; release build, test-target compile, accuracy gate, and bench snapshot pass. |
| `models/qwen3/tp-design.md` | Qwen3 tensor-parallel design: `TP=2` milestone scope plus the controller/worker broadcast execution model, request identity, and coarse-grained step protocol for future TP/MoE work. |
| `models/qwen3/kv-pressure-hang.md` | Issue #85 Qwen3-4B KV pressure hang fixed by full-lifetime scheduler KV admission, waiting-queue deferral, cleanup on disconnect/error, impossible-request errors, scheduler/bridge gates, and real `vllm bench serve` QPS=2 `500/500` pass with post-pressure completion healthy. |

## models / qwen35

| Path | TL;DR |
| --- | --- |
| `models/qwen35/optimization.md` | Hybrid 24 linear + 8 full attn. At parity with vLLM: TTFT 225ms, TPOT 11.81ms (+1%). Post-accuracy-fix GDR decode kernel restore (#9). |
| `models/qwen35/accuracy.md` | Qwen3.5-4B HF parity work: major decode-state bugs are fixed, `conv1d` now matches HF's bf16 pre-`SiLU` rounding, exact HF matches improved to 11/13, and only two small-logit-drift cases remain. |
| `models/qwen35/model-crate.md` | `pegainfer-qwen35-4b` owns Qwen3.5 model/scheduler/recurrent ops/tests/benches; root loads it through `EngineHandle`. Build/check/clippy, root bench sanity check, Qwen3.5 e2e, and scheduler e2e pass. |
| `models/qwen35/e2e-gibberish.md` | Qwen3.5 e2e gibberish fixed: scheduler threads now bind CUDA context and initialize thread-local cuBLAS handles; Qwen3.5 greedy stays on FlashInfer top1, and the default e2e remains an exact golden-text regression. |

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
| `models/deepseek-v2-lite/hf-accuracy-gate.md` | DeepSeek-V2-Lite EP2 HF accuracy gate after PR #149/#150: HF incremental greedy, host-staged EP2, and NCCL EP2 are token/text exact for `Hello`, output_len=16. |
| `models/deepseek-v2-lite/decode-attribution-gate.md` | DeepSeek-V2-Lite EP2 decode attribution gate for `Hello`/16-token batch sizes 1/4/8: structured JSON with accuracy hashes, CPU-side timing, selected CUDA event/NVTX attribution, host-staged/NCCL EP counts, and explicit no-throughput claim boundary. |

## models / kimi-k2

| Path | TL;DR |
| --- | --- |
| `models/kimi-k2/optimization.md` | Kimi-K2 model card + optimization log：61 层 MLA + Marlin WNA16 MoE，H20 ×8 当前 TP8/EP8。重点是 decode：bs4 graph TPOT `14.39ms`（≈`278 tok/s`），目标 `> 300 tok/s`；下一阶段迁到 TP1+DP8+EP8（PPLX）。Prefill 优先级低。 |
| `models/kimi-k2/support-analysis.md` | Kimi-K2 text-only bring-up：Marlin WNA16 routed expert、MLA prompt、全 61 层 prompt forward、多 prompt vLLM top-20 gate 已过；bs4 wave decode 已接入，Marlin atomic split-K row-state bug 修复后 output16 row diff 清零，正在撤掉 decode 诊断负担并回到 bs4 性能主线。 |
| `models/kimi-k2/operator-todo.md` | Kimi-K2 算子清单：MLA + Marlin WNA16 routed expert + NCCL RS bridge 主链；CUDA Graph 覆盖整段 decode，synthetic output64 avg `14.39ms` / p99 `14.83ms`；CUTLASS INT4 后端与 decode row-diff 诊断已下线，详见 changelog。 |
| `models/kimi-k2/changelog.md` | Kimi-K2 算子工作日志：Execution Log / Rejected / Debrief / 经验迁移 全部归档，按原始顺序保留；当前状态见 operator-todo。 |
| `models/kimi-k2/vllm-path-comparison.md` | Kimi-K2 decode 路径对照：vLLM-style fused qkv_a、MoE shared/routed compute overlap、shared/dense gate-up fusion、routed scaled-add 和 bridge microbench 已过 H20 gate；output64 avg/p50/p99 均在 `15ms` 内，vLLM TP-only MoE final all-reduce BF16/F32 两版均慢于当前 RS bridge。 |
| `models/kimi-k2/vllm-h20-baseline.md` | vLLM 0.19.0 H20 ×8 TP1+DP8+EP8 decode-heavy baseline：bs 1..256 扫描，bs=8 拐点 TPOT med `26.4ms` / aggregate `308 tok/s`，bs=256 拉到 `1131 tok/s`；同 client 下 pegainfer TP8+EP8 bs=4 TPOT `19.13ms` 比 vLLM 低 23%，但 HTTP 口径比 in-process 高 33%，frontend overhead 待查。 |
| `models/kimi-k2/pplx-ep-decode.md` | PPLX EP decode bs=1 TPOT 37ms → 17.94ms（−52%），超过 NCCL no-graph 18.52ms。根因是 expert_padding=64 导致 Marlin 98% 计算浪费 + <<<1,1>>> 串行 routing kernel。含完整优化 log、failed approaches、nsys 对比数据。 |
| `models/kimi-k2/pplx-ep-correctness.md` | TP8/EP8 PPLX correctness baseline：H20 64-token token trace 与 TP8/EP8 NCCL 完全一致，hash `4920f088c2338236`；记录 recv capacity、routed-row top-k weight、F32 combine 边界。 |
| `models/kimi-k2/dp1-tp8-ep8-performance.md` | DP1 TP8 EP8 性能优化 ledger：从 correctness baseline `72c770b` 起步，目标 bs64 超过 vLLM baseline output `583.9 tok/s` / TPOT median `109.00ms`，每个优化必须带正确性 gate 和 commit。 |
| `models/kimi-k2/tp1-dp8-ep8-performance.md` | TP1 DP8 EP8 性能优化 ledger：目标 H20 bs64 超过 vLLM TP1 DP8 EP8 baseline；记录 vLLM DPLB/CUDA Graph bucket cliff（8x8 `48ms`，9/8 skew `96ms`），统一压测/profile 命令，并按 profile → 动机预期收益 → microbench → correctness → performance 记录每个优化。 |
| `models/kimi-k2/tp1-dp8-ep8-decode-optimization-master.md` | Kimi-K2 TP1 DP8 EP8 decode 优化 master ledger：H20 per-rank bs=8/global bs≈64 全 decode 算子表、roofline 分类、peak gap、fusion 扫描队列和单 kernel report 队列；KV append provider 已补 production page metadata 并有 H20 latency，production NCU 等 `ncu` 恢复；PPLX Marlin local compute 使用 runtime route-hist replay p95 baseline。 |
| `models/kimi-k2/tp1-dp8-ep8-fusion-scan.md` | Phase 2 fusion scan for TP1 DP8 EP8 decode 已完成：H20 NCU 覆盖 `shared_gate_up -> shared_swiglu`、`attention input_norm -> qkv_a`、`qkv_a_split_norm -> q_b`；qkv_a cuBLASLt 和 stock CUTLASS gated-dual GEMM 均已拒绝，无 accepted fusion，剩余方向转 Phase 3 custom/single-kernel work。 |
| `models/kimi-k2/embedding_report.md` | `decode.embedding` H20 report：TP1 vocab-sharded embedding lookup 为 control/lookup，`6.83-7.24us`、`224` CTAs、`31.7-33.6GB/s` payload，停止 standalone 调参。 |
| `models/kimi-k2/dense_gate_up_report.md` | `decode.dense.gate_up` H20 report：dense layer0 gate/up BF16 GEMM 为 memory-bound，`147.96us` / `3.58TB/s` / `74.5%` H20 HBM；单层单次调用，停止 standalone 调参。 |
| `models/kimi-k2/dense_swiglu_report.md` | `decode.dense.swiglu` H20 report：dense layer0 SwiGLU 为 control/elementwise，`7.79us`、`576` CTAs、`113.6GB/s` payload，停止 standalone 调参。 |
| `models/kimi-k2/dense_down_report.md` | `decode.dense.down` H20 report：dense layer0 down BF16 GEMM 为 memory-bound，`85.48us` / `3.10TB/s` / `64.5%` H20 HBM；单层单次调用，停止 standalone 调参。 |
| `models/kimi-k2/dense_residual_add_report.md` | `decode.dense.residual_add` H20 report：dense layer0 residual add 为 control/elementwise，`6.81-7.51us`、`224` CTAs、`45.8-50.5GB/s` payload，停止 standalone 调参。 |
| `models/kimi-k2/pplx_residual_add_scaled_report.md` | `decode.moe.residual_add_scaled` H20 report：PPLX post-combine scaled residual add 为 control/elementwise，`408.3-410.1us/step`、`6.81-6.83us/call`、`224` CTAs、`~84GB/s` payload，停止 standalone 调参。 |
| `models/kimi-k2/kimi_router_noaux_tc_report.md` | `decode.moe.router` H20 report：control/small-grid limited；保留 pedantic logits GEMM，采纳 post-GEMM score/topk fusion，TP1 PPLX `bs=8,ctx=1` 从 `3.655ms` 到 `3.514ms`；fast tensor-op logits GEMM 因 TP1 DP8 bs64/o5 token trace `30/64` mismatch 仍拒绝。 |
| `models/kimi-k2/shared_gate_up_report.md` | `decode.moe.shared_gate_up` H20 report：memory-bound BF16 skinny GEMM，采用 Kimi 专用 cuBLASLt exact-shape path，TP1 PPLX `bs=8,ctx=1` 从 `1.818ms` 到 `1.505ms` per 60 MoE layers。 |
| `models/kimi-k2/shared_swiglu_report.md` | `decode.moe.shared_swiglu` H20 report：standalone shared SwiGLU 是 tiny-grid/latency limited（`64` CTAs、`0.10` waves/SM、`0.51%` DRAM read），停止单独优化，只保留 row21/22 或 row22/23 custom fusion 方向。 |
| `models/kimi-k2/shared_down_report.md` | `decode.moe.shared_down` H20 report：memory-bound BF16 skinny GEMM，TP1 PPLX `bs=8,ctx=1` 当前 `897.1us` per 60 MoE layers；standalone cuBLASLt `11.000us -> 10.995us` 无有效收益，后续只看 row 22/23 fusion。 |
| `models/kimi-k2/attention_o_proj_report.md` | `decode.attention.o_proj` H20 report：memory-bound BF16 skinny GEMM，采用 Kimi TP1 cuBLASLt exact-shape path，TP1 PPLX `bs=8,ctx=1` 从 `2.715ms` 到 `2.374ms` per 61 attention layers。 |
| `models/kimi-k2/qkv_a_proj_report.md` | `decode.attention.qkv_a` H20 report：memory-bound BF16 skinny GEMM，TP1 PPLX `bs=8,ctx=1` 当前 `1.245ms` per 61 attention layers；standalone cuBLASLt 只有 `0.8-1.7%`，后续只看 row 6/7 RMSNorm-prologue fusion。 |
| `models/kimi-k2/attention_absorb_q_nope_report.md` | `decode.attention.absorb_q_nope` H20 report：TP1 MLA strided-batched BF16 GEMM，采用 cuBLASLt path，TP1 PPLX `bs=8,ctx=1` 从 `973.6us` 到 `748.5us` per 61 attention layers。 |
| `models/kimi-k2/attention_v_up_report.md` | `decode.attention.v_up` H20 report：TP1 MLA strided-batched BF16 GEMM，采用 cuBLASLt path，TP1 PPLX `bs=8,ctx=1` 从 `781.0us` 到 `738.5us` per 61 attention layers。 |
| `models/kimi-k2/q_b_proj_report.md` | `decode.attention.q_b` H20 report：memory-bound skinny BF16 GEMM，cuBLASLt exact-shape sweep 在目标 `batch_size=8` 只有 `8.899us -> 8.746us` (`1.0175x`)，已拒绝 standalone 替换；后续只看 row 8/9 fusion。 |
| `models/kimi-k2/attention_input_norm_report.md` | `decode.attention.input_norm` H20 report：FlashInfer RMSNorm 是 tiny-grid/launch limited（`8` CTAs、`0.05` waves/SM、`0.70-0.74%` DRAM），停止 standalone RMSNorm，后续只看 RMSNorm → qkv_a prologue/custom GEMM。 |
| `models/kimi-k2/attention_qkv_a_split_norm_report.md` | `decode.attention.qkv_a_split_norm` H20 report：Kimi split/norm helper 是 tiny-grid/launch limited（`8` CTAs、`0.01` waves/SM、`0.19-0.20%` DRAM），停止 standalone row8，后续只看 row8 → q_b prologue/custom GEMM。 |
| `models/kimi-k2/attention_rope_split_report.md` | `decode.attention.rope_split` H20 report：Kimi RoPE split helper 是 control/elementwise（`441.8us/step`、`7.24us/call`、`~54GB/s` payload），production NCU 因 `h20-100` ncu timeout 暂缺，停止 standalone 调参。 |
| `models/kimi-k2/attention_paged_kv_append_report.md` | `decode.attention.paged_kv_append` report：provider 已改用 production page metadata 并标为 control；H20 `ctx=1` 为 `7.07us/call`，早期 compact-metadata H20 NCU 仅作方向参考，production NCU 等 `ncu` 恢复。 |
| `models/kimi-k2/attention_flashinfer_mla_decode_report.md` | `decode.attention.flashinfer_mla_decode` H20 report：ctx-sensitive；`ctx=1` 为 `624.6us/step`，`ctx=8192` 为 `103.5ms/step` 且约 `2.85TB/s` payload，production NCU 因 `h20-100` ncu timeout 暂缺。 |
| `models/kimi-k2/attention_post_attn_add_norm_report.md` | `decode.attention.post_attn_add_norm` H20 report：exact-preserving fused add + RMSNorm round 是 tiny-grid/control（`8` CTAs、`896` threads/CTA、`8.65-8.69us/call`），停止 standalone 调参。 |
| `models/kimi-k2/final_norm_report.md` | `decode.final.norm` H20 report：最终 FlashInfer RMSNorm 与 attention input norm 同 shape，`8.01us/call`、`57.3GB/s` payload；同 shape NCU 为 `8` CTAs / `0.05` waves/SM / `<1%` DRAM，停止 standalone 调参。 |
| `models/kimi-k2/final_lm_head_report.md` | `decode.final.lm_head` H20 report：BF16 full-vocab GEMM 已到 `542.7us` / `4.33TB/s` / `90.3%` H20 HBM，停止 standalone 优化，后续只看量化/FP8 格式级变化。 |
| `models/kimi-k2/final_argmax_report.md` | `decode.final.argmax` H20 report：把 one-CTA-per-row BF16 top1 改成 split-vocab partial reduction，TP1 PPLX `bs=8,ctx=1` 从 `125.3us` 到 `12.7us`，TP1 DP8 bs64/o5 token A/B `0/64` mismatch。 |
| `models/kimi-k2/pplx_build_marlin_routing_report.md` | `decode.moe.pplx_build_marlin_routing` H20 report：PPLX Marlin routing metadata builder 为 one-block control row，trace replay p95 `9.87us/call`、`592.3us/step`，NCU `1 x 64` / `0.04%` DRAM，停止 standalone 调参。 |
| `models/kimi-k2/pplx_marlin_compute_report.md` | PPLX routed local compute H20 report：synthetic stress provider 与 trace-driven replay provider 均已覆盖 routing/W13/SwiGLU/W2；已采纳 small-N Marlin tile，p95 W13/W2 `250.64/138.51us -> 161.45/98.14us`，后续只看 route-aware grouped/persistent 设计。 |
| `models/kimi-k2/tp1-pplx-decode-bench.md` | Implemented `kimi_tp1_pplx_decode_bench` and `kimi_pplx_marlin_replay`: TP1/PPLX operator bench with roofline fields, NCU filters, production-metadata MLA KV append provider, synthetic PPLX providers, runtime route histogram trace, trace-driven local Marlin replay, and p95 NCU isolation. |
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
| `benchmarks/accuracy-eval-results.md` | Phase 1 GSM8K: Qwen3-4B PASS (pegainfer 85.37% vs HF 85.82%, delta -0.45%). Qwen3.5-4B FAIL — long-prompt prefill quality divergence on 8-shot. |
| `benchmarks/pplx-ep-a2a-h20-nvlink.md` | pplx EP all-to-all latency on 8× H20 NV18 NVLink: DSV4 & Kimi-K2 shapes, tok=1..256. tok=1 p50 ~82μs, tok=256 p50 ~204/303μs. |
| `benchmarks/deepep-v2-vs-pplx-moe-backend.md` | H20 x8 DeepEP V2 vs current PegaInfer PPLX EP backend comparison: ElasticBuffer/NCCL Gin shows a directional 2.5x-5.3x paired-run ratio on tested DSV4 and Kimi-K2 MoE exchange shapes, with dtype, correctness, harness, and PPLX baseline-drift caveats recorded. |

## conventions

| Path | TL;DR |
| --- | --- |
| `conventions/bench-regression.md` | Benchmark regression tracking: one snapshot per model, git-tracked history, TPOT >2% / TTFT >3% thresholds. |
| `conventions/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches. |
