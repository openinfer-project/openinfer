# Kimi-K2 TP1 PPLX Decode Bench

> **TL;DR:** Implemented `kimi_tp1_pplx_decode_bench`: a TP1 DP8 PPLX decode operator bench with per-op roofline fields and `--ops` / `--labels` filters for NCU isolation. Current accepted Kimi paths cover shared_gate_up and attention o_proj cuBLASLt for batch_size `1..=64`, TP1 MLA absorb/v_up cuBLASLt for `local_heads=64,batch_size<=8`, final argmax split-vocab reduction, router post-GEMM score/topk fusion, MLA paged-KV append provider coverage with production page metadata, synthetic expected-local-route PPLX Marlin compute providers, runtime TP1/DP8/PPLX route histogram tracing with deterministic varied prompt ids, and `kimi_pplx_marlin_replay` for trace-driven local W13/SwiGLU/W2 measurements plus p95 NCU isolation; H20 `bs=8,ctx=1` accepted rows are tracked in the decode optimization master.
>
> **Last touched:** 2026-06

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
- cuBLASLt first heuristic is better for this shape than both generic cuBLAS and tinygemm smoke:
  - standalone `N=8`: `18.673us` per call, `1.120ms` for 60 calls, zero workspace.
  - TP1 PPLX bench provider after wiring Kimi path: `bs=8,ctx=1` shared_gate_up is `1.505ms` for 60 calls, versus the Phase 1 baseline row `1.818ms`.
  - Non-power-of-two active batches are valid: `bs=3,ctx=1` measured `1.524ms` and the row op is `kimi_shared_gate_up_cublaslt`.
- Production decision: put a Kimi-specific cuBLASLt wrapper under `pegainfer-kernels/src/ops/kimi_k2/shared_gate_up.rs` plus `pegainfer-kernels/csrc/kimi_k2/kimi_shared_gate_up.cu`, gated by exact shape `M=4096,K=7168,batch_size=1..=64`. The old typed GEMM remains fallback for other shapes.

### Step 7: Add row filters for NCU isolation
- Updated attention/final manifest labels so repeated providers are distinguishable:
  - `decode.attention.input_norm`
  - `decode.attention.qkv_a`
  - `decode.attention.qkv_a_split_norm`
  - `decode.attention.q_b`
  - `decode.attention.rope_split`
  - `decode.attention.absorb_q_nope`
  - `decode.attention.paged_kv_append`
  - `decode.attention.flashinfer_mla_decode`
  - `decode.attention.v_up`
  - `decode.attention.o_proj`
  - `decode.attention.post_attn_add_norm`
  - `decode.final.norm`
  - `decode.final.lm_head`
  - `decode.final.argmax`
- Added CLI filters:
  - `--ops <csv>` filters by provider/op name.
  - `--labels <csv>` filters by unique manifest label.
  - Empty filter result fails early with `filters matched no TP1 PPLX decode bench rows`.
- Verified locally:
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench`
  - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --active-rows 8 --ctx-lens 1 --measure false --format json --labels decode.attention.input_norm,decode.attention.qkv_a --out target/kernel_reports/kimi-k2/tp1-pplx-decode-filter-smoke.json`
  - Result: passed; the JSON contained exactly the two requested rows with shapes `elems=57344` and `rows=8,out=2112,in=7168`.
- Verified on `h20-100` with the same label filter before collecting row 6/7 NCU artifacts under `profile/kimi-attention-row6-row7-h20-baseline/`.

### Step 8: shared_down isolated profile
- Ran a filtered H20 bench for `decode.moe.shared_down`:
  - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --active-rows 8 --ctx-lens 1 --labels decode.moe.shared_down --iters 256 --format json --out target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-shared-down-bs8.json`
  - Result: `14.9519us/call`, `897.112us` per 60 MoE layers, `15.709 TF/s`, `1.974 TB/s`, `41.115%` HBM roofline.
- Ran NCU full profile on the same filtered row:
  - Main cuBLAS kernel: `nvjet_tst_128x8_64x12_4x1_v_bz_TNT`.
  - Main duration: `10.78us`; grid `56` blocks; block `384` threads; `0.93` waves/SM.
  - Memory throughput: `2.73 TB/s`; DRAM throughput `55.94%`; SM throughput `15.74%`; achieved occupancy `14.25%`; L2 hit rate `2.37%`; no eligible `82.37%`.
- Recorded the conclusion in `shared_down_report.md`: the row is memory-bound and small-grid limited, but exact-shape cuBLASLt replacement was already measured as a no-op (`11.000us -> 10.995us`), so the standalone provider swap is rejected.

### Step 9: PPLX Marlin local compute providers
- Added measured providers for the non-communication PPLX local compute rows:
  - `decode.moe.pplx_build_marlin_routing`
  - `decode.moe.pplx_marlin_w13`
  - `decode.moe.pplx_swiglu`
  - `decode.moe.pplx_marlin_w2`
- The provider models the target `bs=8/rank, global~=64` load as `64` expected local routes per EP rank, `400` expected padded work rows, and `recv_capacity=848`; it does not time EP dispatch/combine and does not claim all-rank route imbalance.
- H20 filtered bench:
  - routing: `9.489us/call`, `569.3us` per 60 MoE layers.
  - W13: `436.432us/call`, `26.186ms` per 60 MoE layers.
  - PPLX SwiGLU: `14.135us/call`, `848.1us` per 60 MoE layers.
  - W2: `236.797us/call`, `14.208ms` per 60 MoE layers.
- NCU artifacts are under `profile/kimi-pplx-marlin-compute-h20-baseline/`; W13/W2 Marlin kernels run with `234` CTAs, `1` wave/SM, `56.8-58.7%` SM throughput, and `32.6-34.7%` DRAM throughput.
- Verified the PPLX Marlin provider across `active_rows=1,2,4,8` with `iters=4`; W13/W2 roofline percentages now stay below `39%` after using expected padded work rows and active-expert weight bytes instead of full recv-capacity bytes.

### Step 10: Runtime PPLX route histogram trace
- Extended the runtime trace entry point so `kimi_kernel_report` and `kimi_model_report` can use non-default parallelism and EP backend selection instead of being fixed to TP8/DP1/NCCL.
- Added `kimi_pplx_route_histogram` trace rows immediately after PPLX `dispatch_recv` and Marlin routing construction. Each row records rank, layer, active rows, local expert range, `recv_counts`, total received routes, active local expert count, max per-expert count, host-computed padded rows, device `num_tokens_post_padded`, receive capacity, expert padding, and routing block size.
- This is diagnostic infrastructure for replacing synthetic PPLX Marlin shapes with real TP1/DP8/PPLX histograms. It does not time or optimize EP communication kernels.
- Local verification:
  - `cargo fmt --all -- --check`
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_kernel_report`
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_model_report`
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench`

### Step 11: H20 route histogram artifact
- Changed runtime trace prompts from all-zero token ids to deterministic varied token ids. All-zero prompts made the router collapse onto a few experts and produced a misleading `max_count_per_expert=63` pattern.
- H20 verification:
  - `cargo check --release -p pegainfer-kimi-k2 --features kimi-k2,kernel-report --bin kimi_kernel_report`
  - Runtime TP1/DP8/PPLX trace wrote `target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json`.
- Artifact summary:
  - `8008` total trace calls, `1920` `kimi_pplx_route_histogram` calls.
  - Two admission waves had `active_rows=1`; two near-target waves had rank0 `active_rows=7` and ranks1-7 `active_rows=8`, for `504` routed tokens per wave.
  - active8 rank rows: `padded_rows` p50/p95/max `80/216/336`, `recv_total_routes` p50/p95/max `63/161/282`, active local experts p50/p95/max `3/24/32`.
- Decision at this step: keep the synthetic PPLX Marlin latency rows until a replay provider or cleaner steady trace can use these histograms directly. Step 12 supersedes this by adding trace replay and moving the master table to replay p95 rows.

### Step 12: PPLX Marlin trace replay
- Extended the PPLX providers in `pegainfer-kimi-k2/src/kernel_report.rs` to accept a `pplx_recv_counts` attr. When absent, they keep using the existing synthetic expected-local-route counts, so the original bench rows remain runnable.
- Added `pegainfer-kimi-k2/src/bin/kimi_pplx_marlin_replay.rs`. It reads runtime trace JSON, filters non-empty `kimi_pplx_route_histogram` rows with `active_rows>=7`, selects p0/p50/p90/p95/p99/p100 by padded rows, and replays routing/W13/SwiGLU/W2 against the local providers.
- Local verification:
  - `cargo fmt --all -- --check`
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_pplx_marlin_replay`
  - `cargo clippy -p pegainfer-kimi-k2 --no-deps --release --features kimi-k2,kernel-report --all-targets -- -D warnings`
  - Smoke replay with `--iters 1` on `target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json`.
- H20 verification:
  - `PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python /root/.cargo/bin/cargo check --release -p pegainfer-kimi-k2 --features kimi-k2,kernel-report --bin kimi_pplx_marlin_replay`
  - `PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python /root/.cargo/bin/cargo run --release -p pegainfer-kimi-k2 --features kimi-k2,kernel-report --bin kimi_pplx_marlin_replay -- --trace target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json --iters 16 --format text --out target/kernel_reports/kimi-k2/pplx-marlin-replay-bs64-kv2-varied.json`
- H20 replay summary:
  - p50 `recv=56,padded=96,active_experts=8`: W13 `114.52us`, W2 `66.39us`.
  - p95 `recv=67,padded=224,active_experts=28`: W13 `250.64us`, W2 `138.51us`.
  - p100 `recv=207,padded=336,active_experts=26`: W13 `368.57us`, W2 `200.31us`.
- Decision: update the master table to use trace replay p95 for PPLX local compute rows. No `opt(...)` commit: this is measurement infrastructure and baseline correction, not a faster kernel.

### Step 13: PPLX Marlin p95 NCU isolation
- Added replay filters:
  - `--quantiles p95,p100` selects the already-ranked histogram samples instead of remeasuring every quantile.
  - `--ops w13,w2` selects local replay providers without relying on NCU launch-skip over unrelated ops.
- H20 filtered replay smoke:
  - `PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python /root/.cargo/bin/cargo run --release -p pegainfer-kimi-k2 --features kimi-k2,kernel-report --bin kimi_pplx_marlin_replay -- --trace target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json --iters 2 --quantiles p95 --ops w13,w2 --format text --out target/kernel_reports/kimi-k2/pplx-marlin-replay-filter-h20-smoke.json`
  - Result: W13 `250.96us`, W2 `139.70us`, matching the unfiltered p95 replay.
- NCU full/source reports:
  - Run directory: `profile/kimi-pplx-marlin-replay-p95-h20/`.
  - Reports: `w13_p95_full.ncu-rep`, `w13_p95_source.ncu-rep`, `w2_p95_full.ncu-rep`, `w2_p95_source.ncu-rep`.
  - Parsed metrics: `analysis/p95_full_metrics.json`, `analysis/p95_source_metrics.json`.
- NCU summary:
  - W13 p95: `265.6us`, grid `234 x 128`, `1.0` waves/SM, `76,458 B` dynamic shared memory, theoretical/achieved occupancy `18.75%/17.50%`, SM throughput `58.07%`, tensor pipe `14.08%`, DRAM read `1.75 TB/s`, L2 hit `4.45%`.
  - W2 p95: `144.3us`, grid `234 x 128`, `1.0` waves/SM, `76,458 B` dynamic shared memory, theoretical/achieved occupancy `18.75%/17.67%`, SM throughput `55.94%`, tensor pipe `13.00%`, DRAM read `1.60 TB/s`, L2 hit `5.07%`.
  - Source-counter reports have no file/line mapping because the release Marlin TU was not built with CUDA line info. Aggregate counters still show W13 `11,360` excessive shared wavefronts and W2 `61,552` excessive shared wavefronts.
- Decision: PPLX Marlin p95 is not at H20 bandwidth or tensor-core limits. Only pursue variants that improve one-wave/smem/route-grouping behavior; otherwise stop this kernel.

### Step 14: MLA paged KV append provider
- Added a real provider for `decode.attention.paged_kv_append` instead of leaving the row estimate-only:
  - `pegainfer-kimi-k2/src/tp1_pplx_decode_bench/attention.rs` now marks the row measured.
  - `pegainfer-kimi-k2/src/bin/kimi_tp1_pplx_decode_bench.rs` emits a `kimi_mla_paged_kv_append` `KernelCall` with `append_ckv`, `append_kpe`, and `kv_len`.
  - `pegainfer-kimi-k2/src/kernel_report.rs` builds synthetic paged MLA cache metadata and times `kimi_mla_paged_kv_append`.
- Local verification:
  - `cargo fmt --all --check`
  - `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench`
  - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --active-rows 8 --ctx-lens 1 --labels decode.attention.paged_kv_append --iters 2 --format json --out target/kernel_reports/kimi-k2/tp1-pplx-decode-kv-append-local-smoke.json`
- Local smoke result on the development GPU: `6.688us/call`, `407.97us` per 61 attention layers. This is not a H20 baseline and must not be promoted into the master table.
- H20 verification:
  - First filtered H20 run used compact page metadata and is no longer a production baseline after review. It measured `7.342us/call`, `447.9us` per 61 layers; NCU was directionally tiny-grid/control limited (`78 x 256`, `0.12` waves/SM, DRAM `0.09%`, no eligible `97.90%`).
  - Review fixes:
    - manifest row now uses `BoundKind::Control`, so JSON/text output does not compute an HBM peak percentage.
    - provider now uses production decode arena metadata: `page_size=16`, `128` pages/request, and page base `request_idx * 128`.
  - Local production-metadata smoke passed:
    - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --active-rows 8 --ctx-lens 1 --labels decode.attention.paged_kv_append --iters 2 --format json --out target/kernel_reports/kimi-k2/tp1-pplx-decode-kv-append-local-production-metadata-smoke.json`
    - Result: `6.256us/call`, `381.62us/step`, `roofline_bound=control`, `roofline_peak_pct=null`.
  - Local default-ctx filtered sweep passed:
    - `cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench -- --active-rows 8 --labels decode.attention.paged_kv_append --iters 1 --format json --out target/kernel_reports/kimi-k2/tp1-pplx-decode-kv-append-default-ctx-local.json`
    - `ctx=4096/8192` rows report `supported=false` with `kv_len exceeds decode arena capacity 2048` instead of aborting the run.
  - H20 production-metadata bench passed after rebuilding the target binary with `cargo +nightly build`:
    - command shape: `--active-rows 8 --ctx-lens 1,128,1024,2048 --labels decode.attention.paged_kv_append --iters 128 --format json --out /tmp/kimi_kv_h20_prod.json`
    - `ctx=1`: `7.066us/call`, `431.03us/step`, `achieved_gbps=2.63`, `roofline_bound=control`, `roofline_peak_pct=null`.
    - `ctx=128/1024/2048`: `7.233/7.245/7.358us/call`; the row stays control/tiny-grid across valid arena lengths.
    - `ctx=4096/8192` is invalid for this provider because the represented production decode arena has `128 * 16 = 2048` tokens/request; the default sweep keeps those rows but reports `supported=false` with an explicit capacity reason instead of measuring an invalid page table.
  - NCU production-metadata rerun is still pending. `/usr/local/cuda/bin/ncu --version` currently times out on `h20-100`; `--set full` did not produce a usable report. The compact-metadata NCU remains directional evidence for the control/tiny-grid diagnosis, not a promoted production NCU report.
- Decision: promote the H20 production-metadata latency into the master table, but do not claim production-metadata NCU coverage yet.

### Step 15: Attention input_norm report
- Added `docs/models/kimi-k2/attention_input_norm_report.md` to close the standalone Phase 3 direction for `decode.attention.input_norm`.
- Evidence reused from `profile/kimi-attention-row6-row7-h20-baseline/`:
  - Event timing: `8.008us/call`, `488.5us/step` for `61` layers at `bs=8,ctx=1`.
  - NCU: FlashInfer `RMSNormKernel<8,bf16>`, `8 x 896` launch, `0.05` waves/SM, `0.70-0.74%` DRAM, `60-61%` scheduler no eligible.
- Decision: stop standalone RMSNorm tuning. Future work should only revisit this row as RMSNorm -> qkv_a prologue/custom skinny-GEMM fusion, and only if the full TP1 PPLX bench beats the current cuBLAS qkv_a path by more than noise.

### Step 16: Attention qkv_a_split_norm report
- Added `docs/models/kimi-k2/attention_qkv_a_split_norm_report.md` to close the standalone Phase 3 direction for `decode.attention.qkv_a_split_norm`.
- Evidence reused from `profile/kimi-attention-row8-row9-h20-baseline/`:
  - Event timing: `8.217us/call`, `501.2us/step` for `61` layers at `bs=8,ctx=1`.
  - NCU: `split_qkv_a_norm_kernel`, `8 x 192` launch, `0.01` waves/SM, `0.19-0.20%` DRAM, `93.4-93.7%` scheduler no eligible.
- Decision: stop standalone row-8 tuning. Future work should only revisit this row as row8 -> q_b prologue/custom skinny-GEMM fusion that preserves `ckv_normed` and `k_rope`.

### Step 17: Shared SwiGLU report
- Added `docs/models/kimi-k2/shared_swiglu_report.md` to close the standalone Phase 3 direction for `decode.moe.shared_swiglu`.
- Evidence reused from `profile/kimi-shared-swiglu-h20-baseline/`, and parsed with the `ncu-report-skill` helper using the local Nsight Compute Python module:
  - Full TP1 PPLX bench artifacts: `410.2-473.3us/step` for `60` calls at `bs=8,ctx=1`.
  - Standalone event timing: `202.2us/step`, `3.37us/call`.
  - NCU: `silu_mul_fused_kernel`, `64 x 256` launch, `0.10` waves/SM, `0.51%` DRAM read, `2.53%` SM, `93.39%` scheduler no eligible.
- Decision: reclassify row 22 as `control/tiny-grid` in the master table and stop standalone SwiGLU tuning. Future work should only revisit this row as row21 -> row22 gated-dual GEMM or row22 -> row23 activation-prologue fusion, with full TP1 PPLX bench proof.

### Step 18: FlashInfer MLA decode ctx sweep report
- Added `docs/models/kimi-k2/attention_flashinfer_mla_decode_report.md` for `decode.attention.flashinfer_mla_decode`.
- Evidence reused from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`:
  - At `bs=8,ctx=1`: `624.6us/step`, `10.24us/call`, `211GB/s` payload-equivalent.
  - At `bs=8,ctx=8192`: `103.50ms/step`, `1.697ms/call`, `2.85TB/s` payload-equivalent, about `59%` of the H20 HBM roofline.
  - `active_rows=1,2,4,8` are nearly identical for this row because attention uses fixed `arena_rows=8`; MoE rows are the active-row-sensitive part of the bench.
- KernelWiki points to FlashInfer MLA fast decode plan, Hopper backend selection, and FP8 KV cache as plausible directions.
- NCU status: `h20-100` is reachable, but `/usr/local/cuda-12.9/bin/ncu --version` currently times out. Decision: do not adopt a code change from event timing alone; this row remains active pending production NCU.

### Step 19: Final lm_head report
- Added `docs/models/kimi-k2/final_lm_head_report.md` for `decode.final.lm_head`.
- Evidence reused from H20 TP1 PPLX bench artifacts:
  - `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json`: `542.68us`, `34.63TF/s`, `4.333TB/s`, `90.28%` H20 HBM.
  - `tp1-pplx-decode-bench-h20-100.json`: active rows `1,2,4,8` all measure the same fixed `arena_rows=8` final row shape at `541.95-542.69us`.
- Decision: stop standalone BF16 LM-head tuning. Future work only makes sense with NCU-backed evidence of a real bottleneck, a library upgrade beating `542.7us` by `>3%`, or a quantized/FP8 LM-head format change with correctness gates.

### Step 20: Attention rope_split report
- Added `docs/models/kimi-k2/attention_rope_split_report.md` for `decode.attention.rope_split`.
- Evidence reused from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json` and the source launch in `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu`:
  - target shape: `batch_size=8`, `local_heads=64`, `q_head_dim=192`, launch `384 x 256`.
  - `bs=8,ctx=1`: `441.76us/step`, `7.24us/call`, `0.027TF/s`, `54.44GB/s` payload-equivalent.
  - `ctx=128/1024/4096/8192` stays in the same `~421-544us/step` band; this row only indexes a different RoPE cache position, so long-context cost belongs to MLA decode, not this helper.
- NCU status: `/usr/local/cuda-12.9/bin/ncu --version` still times out on `h20-100`, so the report does not claim stall breakdown.
- Decision: reclassify the master row from memory-bound to `control/elementwise` and stop standalone tuning. Reopen only for a launch-removing MLA prep fusion or a production NCU result with a concrete `>3%` full-bench path.

### Step 21: Final norm report
- Added `docs/models/kimi-k2/final_norm_report.md` for `decode.final.norm`.
- Evidence reused from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` and the same-shape row-6 NCU in `profile/kimi-attention-row6-row7-h20-baseline/`:
  - final norm shape: `rows=8`, `hidden=7168`, BF16, one `rms_norm_batch` call before `lm_head`.
  - H20 final norm: `8.01us/call`, `57.27GB/s` payload-equivalent, `1.19%` H20 HBM on the bench model.
  - same-shape FlashInfer RMSNorm NCU: `8` CTAs, `0.05` waves/SM, `0.70-0.74%` DRAM, `60-61%` scheduler no eligible.
- Decision: reclassify final norm as `control/tiny-grid` and stop standalone tuning. Also corrected master row 6 (`decode.attention.input_norm`) to the same control/tiny-grid classification already documented in `attention_input_norm_report.md`.

### Step 22: Attention post_attn_add_norm report
- Added `docs/models/kimi-k2/attention_post_attn_add_norm_report.md` for `decode.attention.post_attn_add_norm`.
- Evidence reused from H20 bench artifacts and source launch geometry in `pegainfer-kernels/csrc/flashinfer_norm.cu`:
  - target shape: `rows=8`, `hidden=7168`, BF16 hidden/residual/output plus BF16 norm weight.
  - source launch: `8` CTAs x `896` threads, `28,784B` dynamic shared memory per CTA.
  - H20 timing: `527.74-530.03us/step`, `8.65-8.69us/call`, `~79GB/s` payload-equivalent.
- NCU status: fresh production NCU is still unavailable because `ncu --version` times out on `h20-100`.
- Decision: reclassify row 16 from memory to `control/tiny-grid` and stop standalone tuning. The only plausible follow-up is a downstream prologue fusion that preserves Kimi's BF16 rounding boundary and passes the full TP1 PPLX gate.

### Unexpected
- `--measure false` initially failed because clap's default bool flag handling did not accept an explicit value. Fixed by using `ArgAction::Set`.
- `Option<Vec<usize>>` with a CSV parser caused a clap downcast panic. Fixed by accepting raw strings and parsing CSV in the binary.
- The existing Kimi kernel report providers were TP8-shaped for MLA decode internals. Added runtime-dim TP1 provider paths instead of reusing TP8 constants.
- FlashInfer is repo-local at `pegainfer-kernels/third_party/flashinfer`; using an external checkout can hide source-layout and API-boundary differences. Keep this path in the repo instructions so future kernel work starts from the submodule.
- The first cuBLASLt implementation incorrectly treated active batch as graph bucket and only supported `1,2,4,8,16,32,64`. Fixed to name the dimension `batch_size` and prebuild plans for every `1..=64`, so `bs=3` does not fall back to generic cuBLAS.

## Debrief

- **Outcome**: Dedicated TP1 DP8 PPLX decode bench binary is implemented and checked. It covers embedding, dense layer0 MLP, 61-layer attention aggregate including MLA paged-KV append provider coverage, final norm/lm_head/top1, MoE router/shared expert, PPLX routed compute accounting, PPLX comm accounting across active batch sizes and context lengths, runtime PPLX route histograms, trace-driven PPLX Marlin replay, and NCU-isolated p95 replay profiling.
- **Pitfalls encountered**:
  - CLI value parsing needed explicit owned strings to avoid clap's Vec parser mismatch.
  - TP1 MLA must use runtime-dim `_rt` providers; old TP8 typed providers would make the bench look valid while measuring the wrong local-head shape.
  - FlashInfer `tinygemm2` is a useful lead from KernelWiki, but the H20 smoke for this exact shared dense shape lost to cuBLAS, so it should not be wired into the production path without a stronger kernel result.
  - PPLX routed Marlin and comm still need a true all-rank harness for timing.
- **Lessons learned**:
  - Bench rows should distinguish `arena_rows` from `active_rows`; the current TP1 path uses arena rows for attention/final and active rows after `set_moe_seq_len(active_len)` for MoE/top1.
  - Estimate-only rows are useful when they are explicit: they preserve the operator inventory without fabricating single-rank numbers for EP collectives.
- **Follow-ups**:
  - Run NCU on trace replay p95/max W13/W2 before changing Marlin scheduling or tiling.
  - Add an all-rank PPLX harness for dispatch/combine timing; trace replay covers local compute only and intentionally excludes EP transport.
