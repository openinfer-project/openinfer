# Kimi-K2 TP1 PPLX Decode Bench

> **TL;DR:** Implemented `kimi_tp1_pplx_decode_bench`: a TP1 DP8 PPLX decode operator bench that reports per-op shape, latency, bound class, FLOPs, bytes, achieved TFLOP/s, and achieved bandwidth across active-row and ctx-len sweeps; PPLX comm and PPLX routed Marlin stages are explicit estimate-only rows until an all-rank harness exists.
>
> **Last touched:** 2026-05

## Preparation

- **Read**:
  - `docs/index.md` — confirmed Kimi-K2 domain docs and existing report/perf docs.
  - `docs/models/kimi-k2/tp1-dp8-ep8-performance.md` — TP1 DP8 target shape is per-rank batch 8 with PPLX EP and H20 serving gates.
  - `docs/models/kimi-k2/pplx-ep-decode.md` — PPLX decode bottlenecks were expert padding, Marlin work granularity, routing kernel launch cost, and avoiding D2H in the decode loop.
  - `docs/models/kimi-k2/operator-todo.md` — current Kimi decode operator surface: MLA, dense/shared MLP, Marlin WNA16 routed experts, final norm/lm_head/top1.
  - `pegainfer-kimi-k2/src/bin/kimi_kernel_report.rs` — existing per-op report runner selects one op from the decode DAG and measures it with `kernel_report::measure_call`.
  - `pegainfer-kimi-k2/src/bin/kimi_model_report.rs` — existing model report folds the same DAG into op/call-site rollups.
  - `pegainfer-kimi-k2/src/batch_decode_trace.rs` — current trace is TP8-shaped (`TP_WORLD_SIZE = 8`) and models NCCL/RS-style MoE, not TP1 PPLX.
  - `pegainfer-kimi-k2/src/runner/worker/forward.rs` — actual TP1 decode forward path calls attention, dense layer, and PPLX MoE call sites.
  - `pegainfer-kimi-k2/src/runner/moe_pplx.rs` — actual TP1 PPLX MoE path uses dispatch/recv, PPLX routing, Marlin PPLX W13/W2, combine, and scaled residual add.
  - `pegainfer-kimi-k2/src/kernel_report.rs` — reusable single-op providers exist for many local compute kernels, but not for TP1-specific shape cataloging or PPLX comm.
- **Relevant history**:
  - `docs/models/kimi-k2/support-analysis.md` — existing report tooling was useful for rank0 local compute, but explicitly did not cover full-rank EP imbalance.
  - `docs/models/kimi-k2/changelog.md` — report tooling grew from TP8/NCCL decode composition and should not be treated as TP1 PPLX coverage without changing the schedule.
  - `docs/benchmarks/pplx-ep-a2a-h20-nvlink.md` — PPLX all-to-all has a separate benchmark baseline that can inform comm-stage expectations.
- **Plan**:
  1. Add a new binary `pegainfer-kimi-k2/src/bin/kimi_tp1_pplx_decode_bench.rs` behind `kernel-report` or a new narrow feature if needed.
  2. Define a TP1 PPLX operator manifest from current code: per-rank `arena_rows=8`, active rows `1..=8`, TP1 full vocab, `local_heads=64`, `local_experts=48`, and PPLX receive capacity derived from `KimiMoePplxScratch::new_decode`.
  3. Reuse existing `kernel_report` providers for local compute where shapes match, and add explicit providers only for missing TP1/PPLX operators such as PPLX routing, PPLX Marlin W13/W2, PPLX SwiGLU, and PPLX comm placeholders or harness calls.
  4. Support a bench matrix instead of a single point:
     - active rows: default `1,2,4,8`; optional explicit `--active-rows`.
     - context lengths: default `1,128,1024,4096,8192`; optional explicit `--ctx-lens`.
     - arena rows: fixed `8` for TP1 DP8, reported separately from active rows.
     - PPLX receive capacity: computed from arena rows, not active rows, while actual routed rows are stage-specific metadata.
  5. Emit JSON and text tables with `op`, `stage`, `active_rows`, `arena_rows`, `ctx_len`, shape, latency, bound class, FLOPs, bytes, achieved TFLOP/s, achieved GB/s, and notes.
  6. Use sub-agents with disjoint analysis scopes before or during implementation:
     - MLA/final agent: attention, lm_head, top1 shape and roofline formulas.
     - MoE compute agent: router, shared expert, PPLX routing, Marlin PPLX W13/W2, residual add formulas.
     - PPLX comm agent: dispatch/combine shape, byte movement, and comm-bound reporting boundary.
  7. Verify with `cargo fmt --all --check` and `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench`.
- **Risks / open questions**:
  - PPLX comm stages require all ranks and may need a dedicated H20 harness; the first version may classify them as `comm` with shape/byte accounting while local compute providers are timed.
  - Existing providers use TP8 constants for several MLA helpers; TP1 `local_heads=64` may require runtime-dim wrappers instead of reusing TP8 typed providers directly.
  - Bound classification can start rule-based, but any final claim should compare arithmetic intensity with hardware peak assumptions recorded in the output.
  - Context length only affects a subset of kernels, mostly MLA decode and KV metadata/cache movement; the report must avoid implying ctx sensitivity for GEMM-only stages.

## Execution Log

### Step 1: Split manifest ownership
- Spawned three sub-agents with disjoint file ownership:
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench/attention.rs`
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench/moe_compute.rs`
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench/pplx_comm.rs`
- Parent-owned files:
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench.rs`
  - `pegainfer-kimi-k2/src/bin/kimi_tp1_pplx_decode_bench.rs`
  - `pegainfer-kimi-k2/src/lib.rs`
  - `pegainfer-kimi-k2/Cargo.toml`
- Result: common `BenchSpec` contract and CLI aggregation skeleton added.

### Step 2: Integrate manifest and binary
- Added `pegainfer-kimi-k2/src/tp1_pplx_decode_bench.rs` as the shared manifest contract and aggregator.
- Added the split manifest files:
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench/attention.rs`
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench/moe_compute.rs`
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench/pplx_comm.rs`
- Added `pegainfer-kimi-k2/src/bin/kimi_tp1_pplx_decode_bench.rs` and registered it in `pegainfer-kimi-k2/Cargo.toml`.
- The binary supports:
  - `--active-rows` CSV, default `1,2,4,8`.
  - `--ctx-lens` CSV, default `1,128,1024,4096,8192`.
  - `--iters`, `--format text|json`, `--out`, and `--measure true|false`.
- Result: one binary combines the sub-agent manifests and emits text plus JSON rows with `spec`, `measured`, `total_mean_us`, `achieved_tflops`, and `achieved_gbps`.

### Step 3: Add local measurement adapters
- Extended `pegainfer-kimi-k2/src/kernel_report.rs` with providers needed by TP1 shapes:
  - runtime-dim MLA decode providers (`*_rt`) for TP1 `local_heads=64`.
  - typed DM/HS GEMM providers used by q_b/o_proj, dense MLP, and shared expert.
  - fused HS SiLU-mul and batch BF16 argmax provider.
- Kept PPLX comm and PPLX routed Marlin compute as `estimate_only`, because a single-rank provider would misrepresent full EP behavior.
- Result: local compute rows can be timed through the existing CUDA-event `measure_loop`, while missing multi-rank pieces are called out in output instead of hidden.

### Step 4: Verification
- Ran `cargo fmt --all`.
- Ran:
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench`
  - Result: passed. Build warnings were the existing CUDA target/build-script warnings (`sm_120`, retired legacy units, Triton/FlashInfer notes).
- Ran manifest-only smoke:
  - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --active-rows 1 --ctx-lens 1 --measure false --format text --out target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-smoke.json`
  - Result: passed, emitted 32 rows for `bs=1, ctx=1`.
- Ran measured smoke:
  - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --active-rows 1 --ctx-lens 1 --iters 1 --format text --out target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-measure-smoke.json`
  - Result: passed. Local compute rows reported latency; PPLX comm/routed PPLX Marlin rows reported estimate-only reasons.

### Step 5: H20-100 run
- Confirmed `h20-100:/root/develop/xingming/pegainfer` was at local commit `3bec64f173b8cffdb9cbf378d124e34c723a9dcf` and clean before sync.
- Synced only the bench-related files with `rsync`.
- Remote build/check:
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench`
  - Result: passed on H20 (`sm_90`).
- Remote bench:
  - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --iters 32 --format text --out target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`
  - Result: passed, emitted 640 rows for active rows `1,2,4,8` and ctx lens `1,128,1024,4096,8192`.
- Copied the JSON back locally to `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json` for analysis.
- H20 summary:
  - 460 measured rows and 180 estimate-only rows.
  - Local measured subtotal for `bs=8`: `17.48ms` at ctx `1`, `18.94ms` at ctx `128`, `30.09ms` at ctx `1024`, `70.46ms` at ctx `4096`, and `121.94ms` at ctx `8192`.
  - At `bs=8, ctx=8192`, `kimi_flashinfer_batch_decode_mla_rt` alone was `103.50ms`, so long-context local measured time is dominated by MLA decode cache traffic.

### Step 6: shared_gate_up backend check and optimization
- `shared_gate_up` maps to `pegainfer-kernels/csrc/linear.cu` and uses `cublasGemmEx(... CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`.
- Standalone same-shape cuBLAS harness on H20 with `M=4096,K=7168,N=8,layers=60` measured `~22us` per call, or `~1.32ms` for 60 calls.
- NCU shows the cuBLAS path is memory-bound and under-occupies H20 (`64` blocks for `78` SMs, low L2 hit rate, split-K reduce overhead), but it is not trivially replaceable.
- KernelWiki's closest SM90 lead was FlashInfer `tinygemm2`. The repo-local FlashInfer submodule has only Python/JIT exposure plus an internal `.cu` launcher, not a stable public C++ header. A direct C++ smoke using the internal launcher measured roughly `31-33us` for `N=1,2,4` and `30.6us` for `N=8`, slower than cuBLAS.
- Post-baseline cuBLASLt candidate evidence is better for this shape than both generic cuBLAS and tinygemm smoke:
  - standalone `N=8`: `18.673us` per call, `1.120ms` for 60 calls, zero workspace.
  - TP1 PPLX bench provider after wiring Kimi path: `bs=8,ctx=1` shared_gate_up is `1.505ms` for 60 calls, versus old generic-cuBLAS report row `1.854ms`.
  - Non-power-of-two active batches are valid: `bs=3,ctx=1` measured `1.524ms` and the row op is `kimi_shared_gate_up_cublaslt`.
- Queue for next commit: put a Kimi-specific cuBLASLt wrapper under `pegainfer-kernels/src/ops/kimi_k2/shared_gate_up.rs` plus `pegainfer-kernels/csrc/kimi_k2/kimi_shared_gate_up.cu`, gated by exact shape `M=4096,K=7168,batch_size=1..=64`. The old typed GEMM remains fallback for other shapes.

### Unexpected
- `--measure false` initially failed because clap's default bool flag handling did not accept an explicit value. Fixed by using `ArgAction::Set`.
- `Option<Vec<usize>>` with a CSV parser caused a clap downcast panic. Fixed by accepting raw strings and parsing CSV in the binary.
- The existing Kimi kernel report providers were TP8-shaped for MLA decode internals. Added runtime-dim TP1 provider paths instead of reusing TP8 constants.
- FlashInfer is repo-local at `pegainfer-kernels/third_party/flashinfer`; using an external checkout can hide source-layout and API-boundary differences. `AGENTS.md` now records this submodule path.
- cuBLASLt candidate note: active rows are batch size, not graph bucket. The queued implementation must prebuild plans for every `1..=64`, so `bs=3` does not fall back to generic cuBLAS.

## Debrief

- **Outcome**: Dedicated TP1 DP8 PPLX decode bench binary is implemented and checked. It covers embedding, dense layer0 MLP, 61-layer attention aggregate, final norm/lm_head/top1, MoE router/shared expert, PPLX routed compute accounting, and PPLX comm accounting across active batch sizes and context lengths.
- **Pitfalls encountered**:
  - CLI value parsing needed explicit owned strings to avoid clap's Vec parser mismatch.
  - TP1 MLA must use runtime-dim `_rt` providers; old TP8 typed providers would make the bench look valid while measuring the wrong local-head shape.
  - FlashInfer `tinygemm2` is a useful lead from KernelWiki, but the H20 smoke for this exact shared dense shape lost to cuBLAS, so it should not be wired into the production path without a stronger kernel result.
  - PPLX routed Marlin and comm still need a true all-rank harness for timing.
- **Lessons learned**:
  - Bench rows should distinguish `arena_rows` from `active_rows`; the current TP1 path uses arena rows for attention/final and active rows after `set_moe_seq_len(active_len)` for MoE/top1.
  - Estimate-only rows are useful when they are explicit: they preserve the operator inventory without fabricating single-rank numbers for EP collectives.
- **Follow-ups**:
  - Add an all-rank PPLX harness for dispatch/combine timing and PPLX routed Marlin providers once the harness can create real recv counts/topk weights.
