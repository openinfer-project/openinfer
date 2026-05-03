# docs index

| Path | TL;DR |
| --- | --- |
| `projects/batch-optimization.md` | Realistic benchmark: within 2% of vLLM throughput, TTFT −16%, TPOT −1.6%. Decode TPOT beats vLLM at all concurrencies. Dynamic KV cache (85% free VRAM). Remaining: ITL p99 tail (chunked prefill). |
| `projects/continuous-batching.md` | Phase 1-2 done. Scheduler thread with prefill-priority, batch decode, channel-based streaming. Next: multi-request throughput testing |
| `projects/vllm-frontend-rs-integration.md` | vLLM frontend replacement complete: pegainfer serves through vllm-server with a local engine-core bridge; old HTTP/tokenizer modules are removed; Qwen3 release build/e2e and vLLM models/completions/chat smoke pass on 5090 |
| `projects/q2-2026-plan.md` | Q2 plan: W1 harden batching, W2 PegaInfer+PegaFlow native, W3 differentiation. Competitive intel: Qwen3.5 is both competitors' Achilles' heel, startup time 215s vs seconds, observability as product moat. MTP deferred to Q3 |
| `projects/nonstandard-attention-milestone.md` | Milestone direction: pegainfer focuses on non-standard attention models, with emphasis on model-family readiness, service experience, framework debt repayment, and disciplined evaluation |
| `projects/qwen3-tp-design.md` | Qwen3 tensor-parallel design merged into one doc: `TP=2` milestone scope plus the controller/worker broadcast execution model, request identity, and coarse-grained step protocol for future TP/MoE work |
| `projects/qwen3-kernels-crate.md` | Phase 1 split implemented and 5090-verified: Qwen3-4B kernel surface lives in `crates/pegainfer-kernels`; release build, test-target compile, Qwen3 e2e, and bench snapshot pass |
| `projects/core-entry-crate.md` | Core entry split ready for diff review: `pegainfer-core` owns shared runtime/API pieces, root keeps compatibility re-exports, trace is out of the active entry, and full 5090 build/clippy/e2e/bench passes |
| `projects/qwen3-model-crate.md` | Qwen3 model crate split ready for diff review: `pegainfer-qwen3-4b` owns Qwen3 config/weights/executor/scheduler/tests/benches/kernel plan; root sees generic `EngineHandle`; model benches use `runtime::Qwen3Executor` instead of `ModelForward`; full 5090 clippy/e2e/bench snapshot passes |
| `projects/flashinfer-sampling-benchmark.md` | FlashInfer runtime sampling and greedy top-1 path are in place, but the work is blocked on restoring batched decode token selection instead of per-request sampling inside batch decode |
| `projects/pegainfer-kernels-boundary.md` | Architecture decision: pegainfer should use reusable frontend/runtime/data-plane layers plus per-model engines; kernels become first-class assets through a ledger, simulator, and request tracing |
| `projects/qwen35-4b-accuracy.md` | Qwen3.5-4B HF parity work: major decode-state bugs are fixed, `conv1d` now matches HF's bf16 pre-`SiLU` rounding, exact HF matches improved to 11/13, and only two small-logit-drift cases remain |
| `projects/qwen35-4b-optimization.md` | Hybrid 24 linear + 8 full attn. At parity with vLLM: TTFT 225ms, TPOT 11.81ms (+1%). Post-accuracy-fix GDR decode kernel restore (#9) |
| `projects/model-forward-trait.md` | ModelForward trait extraction: weights/state separation, shared generation loop, designed for bs > 1 |
| `projects/runtime-complexity-paydown.md` | Project to reduce model-specific runtime fragmentation; focus shifting to architecture-level abstraction (ModelForward trait) |
| `archives/pure-gpu-decode-loop.md` | Concluded: CPU overhead is ~0.6% of TPOT (~77μs/token). Batch launch saves ~1ms/128tok. Not worth further investment — TPOT is GPU-compute bound |
| `archives/qwen3-4b-optimization.md` | Dense-attention Qwen3-4B optimization record; archived as reference material after pegainfer led the measured RTX 5070 Ti workloads |
| `archives/qwen35-gdr-chunkwise-plan.md` | Qwen3.5 chunk-wise GDR plan and validation history; archived after the plan landed in the real runtime and rolled into the broader Qwen3.5 optimization record |
| `areas/bench-regression.md` | Benchmark regression tracking: one snapshot per model, git-tracked history, TPOT >2% / TTFT >3% thresholds |
| `resources/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands |
| `resources/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, benchmark smoke test |
| `resources/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons |
| `resources/bench-vs-vllm.md` | pegainfer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas |
| `resources/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log |
| `resources/kernel-technology-reference.md` | Kernel tech reference: current stack, ecosystem survey (Triton/Gluon/CUTLASS/ThunderKittens/FlashAttention/FlashInfer), decision framework, source-level lessons, and operator policy |
| `resources/flashinfer-reference.md` | FlashInfer map: official docs structure, operator families, major features, and which source areas matter beyond the docs index |
| `resources/5090.md` | 5090 dev box workflow: SSH alias, repo/model paths, SM target, rsync workflow, temporary validation worktree, build/e2e/bench commands |
| `areas/coding-style.md` | Testing principle: prefer integration tests, don't test what E2E catches |
