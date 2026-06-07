# Qwen3.5 Kernel Plan

> **TL;DR:** Qwen3.5-4B has a `pegainfer_qwen35_4b::kernel_plan()` static descriptor mirroring the qwen3-4b module — enumerates every prefill / decode / unified op with its Rust call site, backend, and notes, so you can dump the active kernel mix without reading call sites.
>
> **Last touched:** 2026-06

## Why

Qwen3-4B centralized runtime kernel selection in a `kernel_plan` module — one place that decides which kernel variant serves each op, plus a dump facility that prints the active plan. This is invaluable when debugging perf or accuracy differences across GPUs (you can `diff` plans between runs).

Qwen3.5-4B historically hardwired kernel choices at call sites. There was no way to see which variants a given run used without reading code.

The refactor (issue #256) adds a qwen35 counterpart: a `src/kernel_plan.rs` module that mirrors the qwen3 structure and is exposed via `pegainfer_qwen35_4b::kernel_plan()`.

## What the plan covers

Three phases, matching the qwen3 layout:

- **prefill** — embedding, full-attention prefill (Q/K/V gemm, QK-norm+RoPE, paged-kv scatter, paged prefill attention, attention gate, O proj), linear-attention prefill (4 GEMMs: in_proj_qkv fused q+k+v + z + b + a, conv1d, GDR chunkwise Triton AOT, gated RMSNorm, out-proj), shared MLP/norm/residual (RMSNorm via FlashInfer GemmaRMSNorm), final norm (FlashInfer), LM head.
- **decode** — embedding, RMSNorm (FlashInfer GemmaRMSNorm), full-attention decode (Q/K/V gemm, QK-norm+RoPE, paged decode attention, attention gate, O proj), linear-attention decode (4 GEMMs: in_proj_qkv fused q+k+v + z + b + a, conv1d, GDR per-slot, gated RMSNorm, out-proj), shared MLP/norm/residual, final norm (FlashInfer), LM head, sampling (FlashInfer stochastic / CUDA argmax greedy).
- **unified** — `unified_step` (mixed prefill+decode) and per-request logit extraction (D2D memcpy, not a kernel launch).

Each `KernelOp` records the Rust call site (path through the crate), the runtime backend (CUDA, cuBLAS, FlashInfer, Triton AOT, …), and a free-form note explaining why that kernel.

## How to use it

```rust
use pegainfer_qwen35_4b::kernel_plan;

let plan = kernel_plan();
println!("model: {}", plan.model);
for phase in plan.phases {
    println!("[{}] {} ops", phase.name, phase.ops.len());
    for op in phase.ops {
        println!("  - {:<40} {:<12} {}", op.id, op.backend, op.notes);
    }
}
```

Or, for JSON, walk the structure and serialize. (No built-in JSON helper; the data is plain `&'static` so adding one is a 5-line method.)

## What's NOT in scope (yet)

- **No `qwen35_kernel_report.rs` bin.** The qwen3 counterpart (`pegainfer-qwen3-4b/src/bin/qwen3_kernel_report.rs`) is a CUPTI-driven per-op microbench with manifest-driven variant sweeps. That's a much larger piece of work — out of scope for the "pure refactor, no kernel behavior change" boundary in #256.
- **No `kernel_manifests/qwen35-4b.toml` either**, since no kernel_report bin consumes it.
- **No actual selection logic.** Like qwen3, the plan is descriptive only — it documents the call sites, it doesn't dispatch between them. If/when a kernel variant choice depends on shape (e.g., CTA size for prefill attention), that decision still happens at the call site. The plan is the **observability** layer, not a policy engine.

## Future work

- (Optional) Add a `qwen35_kernel_report` bin once there's a concrete need for kernel regression tracking.
- (Optional) Hook `kernel_plan()` into the startup banner so the active plan prints when the server starts.
- (Optional) Compare qwen3 and qwen35 plans side-by-side in `docs/benchmarks/` to highlight where Qwen3.5 differs from Qwen3 (e.g., GDR chunkwise path, dual-mode attention).

## See also

- `pegainfer-qwen3-4b/src/kernel_plan.rs` — the reference implementation this is modeled on.
- `pegainfer-qwen3-4b/src/bin/qwen3_kernel_report.rs` — full CUPTI kernel report runner (future work, not in this refactor).
- `pegainfer-qwen3-4b/kernel_manifests/qwen3-4b.toml` — manifest consumed by the qwen3 report runner.
- Issue #256 — "qwen35: no kernel_plan — decode kernel picks are hardwired".
