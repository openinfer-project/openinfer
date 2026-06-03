# Kimi-K2 TP1 PPLX Decode Bench

> **TL;DR:** Implemented `kimi_tp1_pplx_decode_bench`: a TP1 DP8 PPLX decode operator bench with per-op roofline fields and `--ops` / `--labels` filters for NCU isolation. Current accepted Kimi paths cover shared_gate_up and attention o_proj cuBLASLt for batch_size `1..=64`, TP1 MLA absorb/v_up cuBLASLt for `local_heads=64,batch_size<=8`, final argmax split-vocab reduction, router post-GEMM score/topk fusion, MLA paged-KV append provider coverage with production page metadata, synthetic expected-local-route PPLX Marlin compute providers, runtime TP1/DP8/PPLX route histogram tracing with deterministic varied prompt ids, and `kimi_pplx_marlin_replay` for trace-driven local W13/SwiGLU/W2 measurements plus p95 NCU isolation. A bench-scoped FlashInfer MLA `partition_kv` probe showed H20 synthetic ctx4096/8192 near-2x latency reduction; under the current production cap, graph-safe fixed-grid p32 improved ctx1024/2048 from `13.37/26.24ms` to `7.31/13.85ms` with synthetic output-equivalence passing. Production PPLX wiring was tried and rejected because global bs64 runtime gates failed before the partition branch could be validated; the unused partition kernel/FFI/bench code was removed and no `opt(...)` commit is appropriate.
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
  - Local measured subtotal for per-rank `active_rows=8`: `17.48ms` at ctx `1`, `18.94ms` at ctx `128`, `30.09ms` at ctx `1024`, `70.46ms` at ctx `4096`, and `121.94ms` at ctx `8192`.
  - At per-rank `active_rows=8, ctx=8192`, `kimi_flashinfer_batch_decode_mla_rt` alone was `103.50ms`, so long-context local measured time is dominated by MLA decode cache traffic.

### Step 6: shared_gate_up backend check and optimization
- `shared_gate_up` maps to `pegainfer-kernels/csrc/linear.cu` and uses `cublasGemmEx(... CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`.
- Standalone same-shape cuBLAS harness on H20 with `M=4096,K=7168,N=8,layers=60` measured `~22us` per call, or `~1.32ms` for 60 calls.
- NCU shows the cuBLAS path is memory-bound and under-occupies H20 (`64` blocks for `78` SMs, low L2 hit rate, split-K reduce overhead), but it is not trivially replaceable.
- KernelWiki's closest SM90 lead was FlashInfer `tinygemm2`. The repo-local FlashInfer submodule has only Python/JIT exposure plus an internal `.cu` launcher, not a stable public C++ header. A direct C++ smoke using the internal launcher measured roughly `31-33us` for `N=1,2,4` and `30.6us` for `N=8`, slower than cuBLAS.
- cuBLASLt first heuristic is better for this shape than both generic cuBLAS and tinygemm smoke:
  - standalone `N=8`: `18.673us` per call, `1.120ms` for 60 calls, zero workspace.
  - TP1 PPLX bench provider after wiring Kimi path: per-rank `active_rows=8,ctx=1` shared_gate_up is `1.505ms` for 60 calls, versus the Phase 1 baseline row `1.818ms`.
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
- Added `attention_input_norm_report.md` to close the standalone Phase 3 direction for `decode.attention.input_norm`.
- Evidence reused from `profile/kimi-attention-row6-row7-h20-baseline/`:
  - Event timing: `8.008us/call`, `488.5us/step` for `61` layers at per-rank `active_rows=8,ctx=1`.
  - NCU: FlashInfer `RMSNormKernel<8,bf16>`, `8 x 896` launch, `0.05` waves/SM, `0.70-0.74%` DRAM, `60-61%` scheduler no eligible.
- Decision: stop standalone RMSNorm tuning. Future work should only revisit this row as RMSNorm -> qkv_a prologue/custom skinny-GEMM fusion, and only if the full TP1 PPLX bench beats the current cuBLAS qkv_a path by more than noise.

### Step 16: Attention qkv_a_split_norm report
- Added `attention_qkv_a_split_norm_report.md` to close the standalone Phase 3 direction for `decode.attention.qkv_a_split_norm`.
- Evidence reused from `profile/kimi-attention-row8-row9-h20-baseline/`:
  - Event timing: `8.217us/call`, `501.2us/step` for `61` layers at per-rank `active_rows=8,ctx=1`.
  - NCU: `split_qkv_a_norm_kernel`, `8 x 192` launch, `0.01` waves/SM, `0.19-0.20%` DRAM, `93.4-93.7%` scheduler no eligible.
- Decision: stop standalone row-8 tuning. Future work should only revisit this row as row8 -> q_b prologue/custom skinny-GEMM fusion that preserves `ckv_normed` and `k_rope`.

### Step 17: Shared SwiGLU report
- Added `shared_swiglu_report.md` to close the standalone Phase 3 direction for `decode.moe.shared_swiglu`.
- Evidence reused from `profile/kimi-shared-swiglu-h20-baseline/`, and parsed with the `ncu-report-skill` helper using the local Nsight Compute Python module:
  - Full TP1 PPLX bench artifacts: `410.2-473.3us/step` for `60` calls at per-rank `active_rows=8,ctx=1`.
  - Standalone event timing: `202.2us/step`, `3.37us/call`.
  - NCU: `silu_mul_fused_kernel`, `64 x 256` launch, `0.10` waves/SM, `0.51%` DRAM read, `2.53%` SM, `93.39%` scheduler no eligible.
- Decision: reclassify row 22 as `control/tiny-grid` in the master table and stop standalone SwiGLU tuning. Future work should only revisit this row as row21 -> row22 gated-dual GEMM or row22 -> row23 activation-prologue fusion, with full TP1 PPLX bench proof.

### Step 18: FlashInfer MLA decode ctx sweep report
- Added `attention_flashinfer_mla_decode_report.md` for `decode.attention.flashinfer_mla_decode`.
- Evidence reused from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json`:
  - At per-rank `active_rows=8,ctx=1`: `624.6us/step`, `10.24us/call`, `211GB/s` payload-equivalent.
  - At per-rank `active_rows=8,ctx=8192`: `103.50ms/step`, `1.697ms/call`, `2.85TB/s` payload-equivalent, about `59%` of the H20 HBM roofline.
  - `active_rows=1,2,4,8` are nearly identical for this row because attention uses fixed `arena_rows=8`; MoE rows are the active-row-sensitive part of the bench.
- KernelWiki points to FlashInfer MLA fast decode plan, Hopper backend selection, and FP8 KV cache as plausible directions.
- NCU status: `h20-100` is reachable, but `/usr/local/cuda-12.9/bin/ncu --version` currently times out. Decision: do not adopt a code change from event timing alone; this row remains active pending production NCU.

### Step 19: Final lm_head report
- Added `final_lm_head_report.md` for `decode.final.lm_head`.
- Evidence reused from H20 TP1 PPLX bench artifacts:
  - `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json`: `542.68us`, `34.63TF/s`, `4.333TB/s`, `90.28%` H20 HBM.
  - `tp1-pplx-decode-bench-h20-100.json`: active rows `1,2,4,8` all measure the same fixed `arena_rows=8` final row shape at `541.95-542.69us`.
- Decision: stop standalone BF16 LM-head tuning. Future work only makes sense with NCU-backed evidence of a real bottleneck, a library upgrade beating `542.7us` by `>3%`, or a quantized/FP8 LM-head format change with correctness gates.

### Step 20: Attention rope_split report
- Added `attention_rope_split_report.md` for `decode.attention.rope_split`.
- Evidence reused from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-h20-100.json` and the source launch in `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu`:
  - target shape: `batch_size=8`, `local_heads=64`, `q_head_dim=192`, launch `384 x 256`.
  - Per-rank `active_rows=8,ctx=1`: `441.76us/step`, `7.24us/call`, `0.027TF/s`, `54.44GB/s` payload-equivalent.
  - `ctx=128/1024/4096/8192` stays in the same `~421-544us/step` band; this row only indexes a different RoPE cache position, so long-context cost belongs to MLA decode, not this helper.
- NCU status: `/usr/local/cuda-12.9/bin/ncu --version` still times out on `h20-100`, so the report does not claim stall breakdown.
- Decision: reclassify the master row from memory-bound to `control/elementwise` and stop standalone tuning. Reopen only for a launch-removing MLA prep fusion or a production NCU result with a concrete `>3%` full-bench path.

### Step 21: Final norm report
- Added `final_norm_report.md` for `decode.final.norm`.
- Evidence reused from `target/kernel_reports/kimi-k2/tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` and the same-shape row-6 NCU in `profile/kimi-attention-row6-row7-h20-baseline/`:
  - final norm shape: `rows=8`, `hidden=7168`, BF16, one `rms_norm_batch` call before `lm_head`.
  - H20 final norm: `8.01us/call`, `57.27GB/s` payload-equivalent, `1.19%` H20 HBM on the bench model.
  - same-shape FlashInfer RMSNorm NCU: `8` CTAs, `0.05` waves/SM, `0.70-0.74%` DRAM, `60-61%` scheduler no eligible.
- Decision: reclassify final norm as `control/tiny-grid` and stop standalone tuning. Also corrected master row 6 (`decode.attention.input_norm`) to the same control/tiny-grid classification already documented in `attention_input_norm_report.md`.

### Step 22: Attention post_attn_add_norm report
- Added `attention_post_attn_add_norm_report.md` for `decode.attention.post_attn_add_norm`.
- Evidence reused from H20 bench artifacts and source launch geometry in `pegainfer-kernels/csrc/flashinfer_norm.cu`:
  - target shape: `rows=8`, `hidden=7168`, BF16 hidden/residual/output plus BF16 norm weight.
  - source launch: `8` CTAs x `896` threads, `28,784B` dynamic shared memory per CTA.
  - H20 timing: `527.74-530.03us/step`, `8.65-8.69us/call`, `~79GB/s` payload-equivalent.
- NCU status: fresh production NCU is still unavailable because `ncu --version` times out on `h20-100`.
- Decision: reclassify row 16 from memory to `control/tiny-grid` and stop standalone tuning. The only plausible follow-up is a downstream prologue fusion that preserves Kimi's BF16 rounding boundary and passes the full TP1 PPLX gate.

### Step 23: Dense layer0 GEMM reports
- Added `dense_gate_up_report.md` for `decode.dense.gate_up`.
  - H20 evidence: `147.96us`, `28.57TF/s`, `3.58TB/s`, `74.5%` H20 HBM payload model.
  - Decision: stop standalone tuning because it is one call per decode step and already a high-bandwidth BF16 skinny GEMM.
- Added `dense_down_report.md` for `decode.dense.down`.
  - H20 evidence: `85.48us`, `24.73TF/s`, `3.10TB/s`, `64.5%` H20 HBM payload model.
  - Decision: stop standalone tuning unless production NCU identifies a concrete cuBLAS scheduling gap, or a dense down+residual fusion clears the full-bench gate.
- Both rows now explicitly say production NCU is pending because `h20-100` still times out on `ncu --version`.

### Step 24: Embedding and dense elementwise reports
- Added `embedding_report.md` for `decode.embedding`.
  - H20 evidence: `6.83-7.24us`, `31.7-33.6GB/s` payload-equivalent; source launch `224 x 256`.
  - Decision: classify as `control/lookup` and stop standalone tuning.
- Added `dense_swiglu_report.md` for `decode.dense.swiglu`.
  - H20 evidence: `7.79us`, `113.6GB/s` payload-equivalent; source launch `576 x 256`.
  - Decision: classify as `control/elementwise` and stop standalone tuning; future only as dense MLP fusion.
- Added `dense_residual_add_report.md` for `decode.dense.residual_add`.
  - H20 evidence: `6.81-7.51us`, `45.8-50.5GB/s` payload-equivalent; source launch `224 x 256`.
  - Decision: classify as `control/elementwise` and stop standalone tuning; future only as down-GEMM epilogue fusion.

### Step 25: PPLX residual scaled-add report
- Added `pplx_residual_add_scaled_report.md` for `decode.moe.residual_add_scaled`.
- Evidence reused from H20 bench artifacts and source launch geometry in `pegainfer-kernels/csrc/kimi_k2/kimi_experts.cu`:
  - target shape: `rows=8`, `hidden=7168`, BF16 hidden + BF16 projected + F32 routed + BF16 output, scale `2.827`.
  - source launch: `224 x 256` for `57344` elements per MoE layer.
  - H20 timing: `408.3-410.1us/step`, `6.81-6.83us/call`, `~84GB/s` payload-equivalent.
- NCU status: fresh production NCU is still unavailable because `ncu --version` times out on `h20-100`.
- Decision: reclassify row 28 from memory to `control/elementwise` and stop standalone tuning. Future work only makes sense as a launch-removing fusion that preserves the current BF16 rounding boundary after `hidden + projected`.

### Step 26: PPLX Marlin routing report
- Added `pplx_build_marlin_routing_report.md` for `decode.moe.pplx_build_marlin_routing`.
- Evidence reused from `pplx_marlin_compute_report.md`, `profile/kimi-pplx-marlin-compute-h20-baseline/`, and trace replay artifacts:
  - source launch: `1 x 64` in `pegainfer-kernels/csrc/kimi_k2/kimi_experts.cu`.
  - NCU: `5.28-5.31us`, `0.00` waves/SM, `0.04%` DRAM, `87-88%` scheduler no eligible.
  - H20 replay p95: `recv=67`, `padded=224`, active experts `28`, `9.87us/call`, `592.3us/step`.
- Decision: keep row 24 as `control`, stop standalone routing metadata tuning, and only revisit it as part of route-aware Marlin scheduling or launch-removing fusion.

### Step 27: PPLX SwiGLU report
- Added `pplx_swiglu_report.md` for `decode.moe.pplx_swiglu`.
- Evidence reused from `pplx_marlin_compute_report.md`, `profile/kimi-pplx-marlin-compute-h20-baseline/`, and trace replay artifacts:
  - source launch at p95: `recv_capacity=848`, `intermediate=2048`, so `6784 x 256`; actual p95 work is `224 * 2048` elements read from `num_tokens_post_padded[0]`.
  - NCU: `10.62us`, DRAM `6.32%`, SM throughput `55.40%`, occupancy `76.05%`, scheduler no eligible `34.20%`.
  - H20 replay p95: `12.66us/call`, `759.7us/step`, `217.4GB/s` payload-equivalent.
- Decision: reclassify row 26 from memory to `compute/elementwise` and stop standalone activation tuning. Future work needs W13/W2 fusion or a tighter route-aware launch bound.

### Step 28: Master row 8 consistency fix
- Corrected `tp1-dp8-ep8-decode-optimization-master.md` row 8 for `decode.attention.qkv_a_split_norm`.
- Evidence was already in `attention_qkv_a_split_norm_report.md`: H20 NCU reports `8` CTAs, `0.01` waves/SM, `0.19-0.20%` DRAM, `0.37-0.39%` compute, and `93.4-93.7%` scheduler no eligible.
- Decision: classify row 8 as `control/tiny-grid` with payload-equivalent throughput instead of the stale memory-bound `0.3% / gap 99.7%` entry.

### Step 29: NCU priority audit and row 13 collection attempt
- Spawned a read-only audit sub-agent to rank remaining NCU gaps across master rows 1-28. The recommended order is now recorded in `tp1-dp8-ep8-decode-optimization-master.md`: row 13 `flashinfer_mla_decode`, row 16 `post_attn_add_norm`, row 10 `rope_split`, row 12 `paged_kv_append`, then row 28 `residual_add_scaled`.
- Attempted to start `decode.attention.flashinfer_mla_decode` `ctx=8192` NCU collection under `profile/kimi-flashinfer-mla-decode-ctx8192-h20/`.
- Remote state observed:
  - `target/release/kimi_tp1_pplx_decode_bench` exists on `h20-100` and supports `--labels`.
  - `/usr/local/cuda-12.9/bin/ncu --version` returned `Version 2025.2.0.0` once during this session.
  - Running through `/root/.cargo/bin/cargo` without a CUDA PATH failed because `pegainfer-comm-a2a-kernels` could not find `nvcc`; the next attempt should use the existing release binary directly or set `PATH=/usr/local/cuda-12.9/bin:$PATH`.
  - Existing-binary smoke passed at `ctx=8192`: `1693.95us/call`, `103.33ms/step`, `2.85TB/s` payload-equivalent.
  - Full NCU collection completed for `flashinfer::BatchDecodeWithPagedKVCacheKernelMLA<...>` and wrote `/root/develop/xingming/pegainfer/profile/kimi-flashinfer-mla-decode-ctx8192-h20/reports/ctx8192_full.ncu-rep`.
  - The NCU-run event row under profiler overhead was `1793.02us/call`, `109.37ms/step`, `2.70TB/s` payload-equivalent.
  - SourceCounters collection did not finish after several minutes and was killed from the local SSH side.
  - Pulling/listing the remote profile directory later timed out on `h20-100`, so the full NCU report has not been retrieved or parsed locally yet.
- Decision: keep row 13 active. The next concrete action is to retrieve and parse `ctx8192_full.ncu-rep`; do not change FlashInfer MLA decode code from event timing or kernel-name evidence alone.

### Step 30: Row 13 selected NCU metrics and FlashInfer split-K audit
- Ran a narrower NCU stdout metrics pass for `decode.attention.flashinfer_mla_decode` at per-rank `active_rows=8,ctx=8192` and parsed it locally into `profile/kimi-flashinfer-mla-decode-ctx8192-h20/analysis/ctx8192_metrics_summary.json`.
- Key metrics:
  - Kernel: `BatchDecodeWithPagedKVCacheKernelMLA<2,16,2,32,8,1,2,...>`.
  - Launch: grid `(8,4,1)` = `32` CTAs, block `(32,8,1)` = `256` threads, `0.41` waves/SM on H20.
  - Resource pressure: `254` registers/thread, `22,528B` dynamic shared memory/block, `12.50%` active warps.
  - Throughput counters: `28.77%` SM throughput, `22.74%` compute-memory throughput, `0.87%` DRAM throughput, `48.22%` L2 sector hit rate.
- Applied the ncu-report-skill diagnosis playbook: `launch__waves_per_multiprocessor < 0.5` and grid smaller than SM count match the small-grid / SM-idle pattern; the first candidate fix is splitting long KV work across more CTAs.
- Queried KernelWiki with `uv run --with pyyaml --no-project python scripts/query.py ...`; relevant hits were `wiki/patterns/low-sm-utilization.md`, `pr-flashinfer-2530`, `pr-flashinfer-844`, and `pr-vllm-34597`.
- Inspected the repo-local FlashInfer submodule:
  - `BatchDecodeWithPagedKVCacheDispatchedMLA` in `include/flashinfer/attention/decode.cuh` forces `partition_kv=false` when `tmp_v == nullptr`.
  - FlashInfer's TVM-FFI path runs `BatchDecodeWithPagedKVCachePlanMLA`, then passes planned `request_indices`, `kv_tile_indices`, `o_indptr`, `kv_chunk_size_ptr`, `tmp_v`, and `tmp_s`.
  - The current Kimi wrapper in `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu` sets `o_indptr=nullptr` and passes `tmp_v/tmp_s=nullptr`, so it cannot enter FlashInfer's split-K branch.
- Decision: update `attention_flashinfer_mla_decode_report.md` and the master ledger. The next code experiment should be a bench-scoped planned `partition_kv` path; no production code change is adopted yet because no speedup has been measured.

### Step 31: Bench-scoped MLA partition-KV probe
- Added a local WIP probe behind `kimi_tp1_pplx_decode_bench --mla-decode-partition-pages <pages>`; this code was removed in Step 41 after the production adoption was rejected, so the artifacts below are historical evidence rather than current CLI support:
  - CUDA wrapper: `kimi_flashinfer_batch_decode_mla_partitioned_cuda`.
  - Rust wrapper: `kimi_flashinfer_batch_decode_mla_partitioned_rt`.
  - Bench provider: `kimi_flashinfer_batch_decode_mla_partitioned_rt` is selected only when the new CLI flag is present; the default baseline path is unchanged.
  - Metadata is generated with FlashInfer-style `request_indices`, `kv_tile_indices`, `o_indptr`, one `kv_chunk_size`, and `tmp_v/tmp_s`.
- Local verification:
  - `PEGAINFER_CUDA_SM=90 cargo build --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench` passed.
  - `cargo fmt --all --check` and `git diff --check` passed.
  - Local same-shape sanity at per-rank `active_rows=8,ctx=8192,iters=2`:
    - baseline: `1517.904us/call`, `92.592ms/step`.
    - `--mla-decode-partition-pages 256`: `770.496us/call`, `47.000ms/step`.
    - `--mla-decode-partition-pages 128`: `773.664us/call`, `47.194ms/step`.
- H20 status:
  - Original workspace writes on `h20-100:/root/develop/xingming/pegainfer` still hang intermittently, so the WIP source was packaged locally as `/tmp/pegainfer-kimi-partition-src.tar.zst` and built from `/dev/shm/pegainfer-kimi-partition-src`.
  - Remote build needed `HOME=/dev/shm/pegainfer-cargo-home` and direct nightly toolchain binaries; `/root/.cargo/bin/cargo` hangs through the rustup shim in the default HOME/cwd environment.
  - H20 build passed from tmpfs:
    - `PATH=/root/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:$PATH HOME=/dev/shm/pegainfer-cargo-home CARGO_HOME=/root/.cargo TMPDIR=/dev/shm/pegainfer-tmp RUSTC=/root/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc PEGAINFER_CUDA_SM=90 PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python CARGO_TARGET_DIR=/dev/shm/pegainfer-kimi-partition-target /root/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/cargo build --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench`
  - H20 bench artifacts pulled locally:
    - `target/kernel_reports/kimi-k2/kimi_mla_baseline_h20.json`
    - `target/kernel_reports/kimi-k2/kimi_mla_p256_h20.json`
    - `target/kernel_reports/kimi-k2/kimi_mla_p128_h20.json`
    - `target/kernel_reports/kimi-k2/kimi_mla_p64_h20.json`
    - `target/kernel_reports/kimi-k2/kimi_mla_p128_confirm_h20.json`
    - `target/kernel_reports/kimi-k2/kimi_mla_p128_ncu_ctx8192_metrics.csv`
    - `target/kernel_reports/kimi-k2/kimi_mla_p128_check4096_h20.json`
    - `target/kernel_reports/kimi-k2/kimi_mla_p128_check8192_h20.json`
  - H20 `iters=8` sweep:
    - `ctx=1024`: baseline `13.379ms`; p256/p128/p64 all `13.566-13.567ms`, a small regression.
    - `ctx=4096`: baseline `51.975ms`; p128 `26.631ms` (`1.95x`), p64 `26.807ms` (`1.94x`), p256 no win.
    - `ctx=8192`: baseline `103.269ms`; p256 `52.355ms` (`1.97x`), p128 `52.517ms` (`1.97x`), p64 `52.851ms` (`1.95x`).
  - H20 `iters=16` p128 confirmation reproduced the signal: baseline `51.983/103.336ms`, p128 `26.664/52.535ms` at `ctx=4096/8192`.
  - Selected NCU for p128 `ctx=8192` with `/usr/local/cuda/bin/ncu`:
    - decode kernel grid `(32,4,1)` = `128` CTAs, `1.64` waves/SM, `254` regs/thread, `57.29%` SM throughput, `3.44%` DRAM throughput.
    - merge kernel grid `(546,1,1)`, `0.70` waves/SM, `48` regs/thread, `39.85%` SM throughput, `7.29%` DRAM throughput.
  - Added `--mla-decode-partition-check` to compare baseline and partition outputs on deterministic BF16 q/cache data before measuring:
    - Local `ctx=1024,pages=128`: `max_abs=0`, `mean_abs=0`.
    - H20 `ctx=4096,pages=128`: `max_abs=0.001953`, `mean_abs=0.000170`.
    - H20 `ctx=8192,pages=128`: `max_abs=0.001953`, `mean_abs=0.000101`.
- Decision: do not commit the code as an optimization yet. The H20 speedup is real for the bench path and synthetic output-equivalence passes, but the probe still lacks production CUDA-graph metadata/temp-buffer ownership and an end-to-end decode/token gate.
- Production caveat found while reading the runner:
  - `pegainfer-kimi-k2/src/runner/worker.rs` fixes `KIMI_DECODE_PAGE_SIZE=16` and `KIMI_DECODE_PAGES_PER_REQUEST=128`, so current production decode has a `2048` token/request arena.
  - The H20 `ctx=4096/8192` measurements are still valid kernel-bench evidence, but not a direct production-runner shape.
  - H20 production-cap sweep landed for baseline vs `64/32/16` page chunks at `ctx=1024/2048`:

    | Config | ctx1024 step | ctx2048 step | Check result |
    |---|---:|---:|---|
    | baseline | `13.371ms` | `26.236ms` | baseline |
    | p64 | `13.572ms` (`0.99x`) | `13.747ms` (`1.91x`) | max_abs `0/0.001953`, mean_abs `0/0.000161` |
    | p32 | `7.284ms` (`1.84x`) | `13.837ms` (`1.90x`) | max_abs `0.001953/0.001953`, mean_abs `0.000360/0.000245` |
    | p16 | `7.424ms` (`1.80x`) | `14.123ms` (`1.86x`) | max_abs `0.001953/0.001953`, mean_abs `0.000222/0.000250` |

  - Decision from the production-cap sweep: p32 is the current candidate because it wins both `ctx=1024` and `ctx=2048`. p64 is slightly faster at `ctx=2048` alone, but it regresses `ctx=1024`.
  - Added graph-safe fixed-grid metadata support to the bench path:
    - CLI flag: `--mla-decode-partition-max-pages-per-request`.
    - CUDA wrapper now accepts `block_valid_mask`.
    - Rust wrapper accepts `Option<&CudaSlice<u8>>` and validates mask/temp sizes.
    - Bench metadata pads to `batch * ceil(max_pages_per_request / kv_chunk_pages)` and masks invalid chunks.
  - Local fixed-grid p32 smoke passed at `ctx=1024/2048` with `max_abs=0.001953`.
  - H20 fixed-grid p32 artifact: `target/kernel_reports/kimi-k2/kimi_mla_p32_fixed_ctx2048_h20.json`.
    - `--mla-decode-partition-pages 32 --mla-decode-partition-max-pages-per-request 128 --iters 8`.
    - `ctx=1024`: `7.309ms/step`, `1.83x` vs baseline `13.371ms`.
    - `ctx=2048`: `13.854ms/step`, `1.89x` vs baseline `26.236ms`.
    - Equivalence: both rows `max_abs=0.001953`, mean abs `0.000362/0.000245`.
  - Decision: fixed-grid p32 keeps the real-padded p32 gain while matching the CUDA Graph launch-shape requirement. The next code step is production arena ownership and full decode/token validation, not more bench-only tuning.

### Step 32: Production PPLX MLA partition rejection
- Wired the fixed-grid p32 plan into the production PPLX non-graph decode path, including arena-owned metadata/temp buffers and a `.partitioned_actual` `kernel-call-trace` marker.
- Verification:
  - Local `cargo fmt --all --check` passed.
  - Local `cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_tp1_pplx_decode_bench` passed.
  - Local `cargo check --release -p pegainfer-kimi-k2 --features kimi-k2,kernel-report --bin kimi_kernel_report` passed.
  - H20 `cargo check --release -p pegainfer-kimi-k2 --features kimi-k2,kernel-report --bin kimi_kernel_report` passed from `/dev/shm/pegainfer-kimi-partition-src`.
  - H20 runtime gate failed: global bs64/kv513 long-prefill trace hit rank7 non-finite top logit `NaN`; trace-only decode growth to kv513 hit prompt_len1 decode non-finite top logit `-inf`; minimal global bs64/kv2 PPLX trace also hit prompt_len1 decode `-inf`.
- Decision: reject and remove the production runner wiring from this WIP. The bench artifacts remain useful evidence, but this optimization is not accepted and must not be committed.

### Step 33: Post-attention add+RMSNorm NCU stop
- Moved to row 16 `decode.attention.post_attn_add_norm` after rejecting production MLA partition adoption.
- H20 NCU command:

  ```bash
  /usr/local/cuda-12.9/bin/ncu --target-processes all \
    --kernel-name-base demangled --print-kernel-base demangled --set full \
    -k regex:FusedAddRMSNormRoundKernel \
    -o /dev/shm/kimi-post-attn-add-norm-ncu/reports/post_attn_add_norm_full \
    --force-overwrite /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
    --active-rows 8 --ctx-lens 1 --iters 1 --format text \
    --labels decode.attention.post_attn_add_norm \
    --out /dev/shm/kimi-post-attn-add-norm-ncu/post_attn_add_norm_ncu.json
  ```

- Result:
  - NCU report: `/dev/shm/kimi-post-attn-add-norm-ncu/reports/post_attn_add_norm_full.ncu-rep`.
  - Key counters: `8` CTAs, `0.05` waves/SM, `32` regs/thread, `28.78KiB` dynamic shared memory/block, `2.29%` SM throughput, `1.11%` DRAM throughput, `64.78%` no-eligible scheduler cycles.
  - NCU rule engine flags the launch as too small (`8` blocks on `78` SMs). Shared-memory conflicts show a local `~15%` hint, but that is not enough to clear the full-bench bar without deleting the launch.
- Decision: stop standalone tuning for `post_attn_add_norm`. Only a downstream prologue fusion that preserves the BF16 rounding boundary is worth reopening. Updated `attention_post_attn_add_norm_report.md`, the master ledger, and `profile/kimi-attention-post-attn-add-norm-h20/REPORT.md`.

### Step 34: Attention rope_split NCU stop
- Moved to row 10 `decode.attention.rope_split`.
- H20 filtered bench sanity on the existing tmpfs target binary passed:

  ```bash
  /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
    --active-rows 8 --ctx-lens 1 --iters 4 --format text \
    --labels decode.attention.rope_split \
    --out /dev/shm/kimi_rope_probe.json
  ```

  The sanity run measured `483.61us/step`; the master baseline remains the earlier `441.8us/step` artifact because this was a short NCU-prep probe on a bench-scoped tmpfs binary.
- H20 selected NCU command:

  ```bash
  /usr/local/cuda/bin/ncu --target-processes all \
    --kernel-name-base demangled --print-kernel-base demangled \
    --section LaunchStats --section Occupancy --section SpeedOfLight \
    --section SchedulerStats --section WarpStateStats \
    --section MemoryWorkloadAnalysis \
    --launch-skip 3 --launch-count 1 \
    -k regex:rope_split_decode_kernel \
    -o /dev/shm/kimi-rope-split-ncu/reports/rope_split_selected \
    --force-overwrite /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
    --active-rows 8 --ctx-lens 1 --iters 1 --format text \
    --labels decode.attention.rope_split \
    --out /dev/shm/kimi-rope-split-ncu/rope_split_ncu.json
  ```

- Result:
  - NCU report: `profile/kimi-attention-rope-split-h20/reports/rope_split_selected.ncu-rep`.
  - Parsed details: `profile/kimi-attention-rope-split-h20/analysis/rope_split_details.csv`.
  - Key counters: `384` CTAs, `0.62` waves/SM, `22` regs/thread, `0B` dynamic shared memory, `10.51%` SM throughput, `1.27%` DRAM throughput, `61.96GB/s` memory throughput, `48.67%` achieved occupancy, and `77.03%` no-eligible scheduler cycles.
  - The NCU rule engine reports the launch has only `0.6` full waves across SMs; the bench JSON row under NCU replay reports `73s` and must not be used as latency evidence.
- Decision: stop standalone tuning for `rope_split`. The only remaining direction is launch-removing MLA-prep fusion that preserves `q_nope`, `q_pe`, and `append_kpe`. Updated `attention_rope_split_report.md`, the master ledger, and `profile/kimi-attention-rope-split-h20/REPORT.md`.

### Step 35: MLA paged KV append production-metadata NCU stop
- Moved to row 12 `decode.attention.paged_kv_append`.
- H20 selected NCU command:

  ```bash
  /usr/local/cuda/bin/ncu --target-processes all \
    --kernel-name-base demangled --print-kernel-base demangled \
    --section LaunchStats --section Occupancy --section SpeedOfLight \
    --section SchedulerStats --section WarpStateStats \
    --section MemoryWorkloadAnalysis \
    --launch-skip 3 --launch-count 1 \
    -k regex:AppendPagedKVMlaCacheKernel \
    -o /dev/shm/kimi-kv-append-prod-ncu/reports/kv_append_selected \
    --force-overwrite /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
    --active-rows 8 --ctx-lens 1 --iters 1 --format text \
    --labels decode.attention.paged_kv_append \
    --out /dev/shm/kimi-kv-append-prod-ncu/kv_append_ncu.json
  ```

- Result:
  - Remote NCU report: `/dev/shm/kimi-kv-append-prod-ncu/reports/kv_append_selected.ncu-rep`.
  - Parsed details copied locally: `profile/kimi-mla-paged-kv-append-prod-h20/analysis/kv_append_details.csv`.
  - Key counters: `78` CTAs, `0.12` waves/SM, `28` regs/thread, `0B` dynamic shared memory, `1.50%` SM throughput, `0.09%` DRAM throughput, `4.40GB/s` memory throughput, `8.71%` achieved occupancy, and `97.63%` no-eligible scheduler cycles.
  - The NCU rule engine reports only `0.1` full waves across SMs. The bench JSON row under NCU replay reports `75s` and must not be used as latency evidence.
  - Repeated rsync attempts for the `.ncu-rep` timed out; the exported details CSV and remote report path are recorded in `profile/kimi-mla-paged-kv-append-prod-h20/REPORT.md`.
- Decision: stop standalone tuning for `paged_kv_append`. Keep the production-metadata provider and only reopen if MLA cache-prep fusion can remove the launch. Updated `attention_paged_kv_append_report.md`, the master ledger, and `profile/kimi-mla-paged-kv-append-prod-h20/REPORT.md`.

### Step 36: PPLX residual_add_scaled NCU stop
- Moved to row 28 `decode.moe.residual_add_scaled`.
- H20 selected NCU command:

  ```bash
  /usr/local/cuda/bin/ncu --target-processes all \
    --kernel-name-base demangled --print-kernel-base demangled \
    --section LaunchStats --section Occupancy --section SpeedOfLight \
    --section SchedulerStats --section WarpStateStats \
    --section MemoryWorkloadAnalysis \
    --launch-skip 3 --launch-count 1 \
    -k regex:kimi_residual_add_scaled_f32_kernel \
    -o /dev/shm/kimi-residual-add-scaled-ncu/reports/residual_add_scaled_selected \
    --force-overwrite /dev/shm/pegainfer-kimi-partition-target/release/kimi_tp1_pplx_decode_bench \
    --active-rows 8 --ctx-lens 1 --iters 1 --format text \
    --labels decode.moe.residual_add_scaled \
    --out /dev/shm/kimi-residual-add-scaled-ncu/residual_add_scaled_ncu.json
  ```

- Result:
  - Remote NCU report: `/dev/shm/kimi-residual-add-scaled-ncu/reports/residual_add_scaled_selected.ncu-rep`.
  - Parsed details copied locally: `profile/kimi-pplx-residual-add-scaled-h20/analysis/residual_add_scaled_details.csv`.
  - Key counters: `224` CTAs, `0.36` waves/SM, `16` regs/thread, `0B` dynamic shared memory, `8.37%` SM throughput, `3.33%` DRAM throughput, `162.16GB/s` memory throughput, `34.03%` achieved occupancy, and `91.25%` no-eligible scheduler cycles.
  - The NCU rule engine reports only `0.4` full waves across SMs. The bench JSON row under NCU replay reports `72s` and must not be used as latency evidence.
- Decision: stop standalone tuning for `residual_add_scaled`. Keep the exact-preserving post-combine helper and reopen only for a launch-removing fusion across the PPLX combine boundary. Updated `pplx_residual_add_scaled_report.md`, the master ledger, and `profile/kimi-pplx-residual-add-scaled-h20/REPORT.md`.

### Step 37: CUDA Tile C++ probe
- Investigated CUDA 13.3's CUDA Tile C++ surface for a possible PPLX Marlin successor prototype.
- Local headers:
  - `/usr/local/cuda-13.3/targets/x86_64-linux/include/cuda_tile.h` is the public include.
  - `/usr/local/cuda-13.3/targets/x86_64-linux/include/crt/cuda_tile.h` owns the `cuda::tiles::inline __1` API.
  - `/usr/local/cuda-13.3/bin/tileiras` is the Tile IR optimizing assembler.
- Toolchain facts:
  - `#include <cuda_tile.h>` requires `nvcc -std=c++20 -enable-tile`; without `-enable-tile`, tile annotations are ignored and the header fails in host parsing.
  - `__tile_global__` kernels launch with ordinary CUDA launch syntax. The local probe used `<<<grid, 1>>>`.
- Scratch probes under `target/cuda_tile_probe/`:
  - `tile_add.cu`: `shape<128>` + `iota/load_masked/store_masked`; local `sm_120` run passed.
  - `tile_matmul.cu`: `partition_view + matmul` for `16x16` FP32; local `sm_120` run passed.
  - `tile_bf16_matmul.cu`: `16x16` BF16 matmul; local `sm_120` run passed, but `tileiras` reported Tensor Core failure with MMA shape `[1,1,1]`, so it lowered to scalar `FMUL/FADD`.
  - `tile_bf16_matmul64.cu`: `64x64` BF16 matmul; local `sm_120` run passed, and `sm_90 -tilecubin` `tileiras` reported Tensor Core success with shape `[64,64,16]`; SASS contains `HGMMA.64x64x16.F32.BF16`.
  - `tile_i8_matmul64.cu`: `64x64` INT8 matmul; `sm_90 -tilecubin` `tileiras` reported Tensor Core success with shape `[64,64,32]`; SASS contains `IGMMA.64x64x32.S8.S8.SAT`.
- Negative finding:
  - The CUDA 13.3 header has no `int4`, `uint4`, `fp4`, `nvfp4`, `e2m1`, `nibble`, or subbyte element type exposed through `cuda::tiles::matmul`.
  - The current Marlin WNA16 packed INT4 path cannot be expressed as a direct CUDA Tile matmul without custom unpack/dequant or a weight-format change.
- H20 note:
  - `sm_90 -tilecubin` generation works locally, but `h20-100` SSH commands were timing out during this probe, so no H20 runtime number was recorded.
- Decision: CUDA Tile C++ is viable for a small standalone BF16/INT8 tile prototype and useful for understanding the Tile compiler, but it is not a direct replacement for packed W4A16 Marlin. Any Marlin experiment should start with a trace-replay harness and an explicit packed-int4 unpack/dequant design; otherwise keep optimizing current CUDA Marlin scheduling.

### Step 38: Local SM120 Marlin baseline correction
- Re-centered the CUDA Tile investigation on the real target: Kimi WNA16 uses BF16 activations with packed INT4 weights. BF16/INT8 CUDA Tile probes only prove Tile compiler behavior; they are not an apples-to-apples comparison with Marlin.
- Confirmed local machine:
  - GPU: RTX 5070 Ti, compute capability `12.0`.
  - Build target: `PEGAINFER_CUDA_SM=120`; build.rs warns that nvcc does not list `compute_120f`, so kernels compile for raw `sm_120`.
- Ran current PPLX Marlin providers locally:

  ```bash
  PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-kimi-k2 \
    --features kernel-report --bin kimi_tp1_pplx_decode_bench -- \
    --active-rows 8 --ctx-lens 1 \
    --labels decode.moe.pplx_marlin_w13,decode.moe.pplx_marlin_w2 \
    --iters 16 --format text \
    --out target/kernel_reports/kimi-k2/tp1-pplx-marlin-sm120-local.json
  ```

  Result on the synthetic stress provider (`rows=400`, `recv_capacity=848`):
  - W13: `15.436ms/step` across 60 layers, `91.30 TF/s`, `3.116 TB/s`.
  - W2: `8.825ms/step` across 60 layers, `79.85 TF/s`, `2.745 TB/s`.
- Ran trace-replay p95 locally:

  ```bash
  PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-kimi-k2 \
    --features kimi-k2,kernel-report --bin kimi_pplx_marlin_replay -- \
    --trace target/kernel_reports/kimi-k2/tp1-dp8-pplx-route-hist-bs64-kv2-varied.json \
    --iters 16 --quantiles p95 --ops w13,w2 --format text \
    --out target/kernel_reports/kimi-k2/pplx-marlin-replay-p95-sm120-local.json
  ```

  Result for H20-derived p95 route shape (`recv=67`, `padded=224`, active experts `28`):
  - W13: `149.82us`, `3.120 TB/s`, `65.01%` of the local assumed `4.8TB/s` roofline.
  - W2: `89.53us`, `2.629 TB/s`, `54.76%` of the local assumed `4.8TB/s` roofline.
  - Relative to the H20 accepted p95 rows (`161.45us` / `98.14us`), local `sm_120` is only about `1.08x` / `1.10x` faster for this Marlin path.
- KernelWiki check:
  - `kernel-grouped-gemm` points to grouped/persistent scheduling as the right MoE direction, with small-M and load imbalance as the practical bottlenecks.
  - `pr-cutlass-3091` is about Hopper CuTe DSL grouped GEMM examples, not packed W4A16 Marlin.
  - `pr-cutlass-2865` notes SM120 interactions with SM90 mainloops / SM100 scheduler plumbing; useful context for local `sm_120` checks, but not an INT4 Tile answer.
- Decision: local Marlin can and does run. CUDA Tile has no direct packed-INT4 matmul surface, so the next valid CUDA Tile comparison would be an explicit W4A16 toy: load packed nibbles, dequant to a Tensor-Core-friendly BF16/INT8 tile, then compare against Marlin on the same replay shape. Until that exists, the current Marlin numbers above are the real local INT4 baseline.

### Step 39: Local CUDA Tile W4A16 toy vs Marlin
- Added a scratch-only CUDA Tile W4A16 microbench under `target/cuda_tile_probe/tile_w4a16_perf.cu`; this file is ignored and is not a production candidate.
- The toy uses plain row-major packed nibbles, not Marlin's production layout:
  - activation: BF16 `[M,K]`;
  - weight: packed INT4 `[K,N/2]`, two output-column weights per byte;
  - kernel: unpack nibble, convert `int4 -> float -> BF16`, then call `cuda::tiles::mma` over `64x64` tiles.
- Built and ran locally on RTX 5070 Ti / `sm_120`:

  ```bash
  /usr/local/cuda-13.3/bin/nvcc -std=c++20 -enable-tile -arch=sm_120 \
    target/cuda_tile_probe/tile_w4a16_perf.cu -lcublas \
    -o target/cuda_tile_probe/tile_w4a16_w13_perf_sm120

  /usr/local/cuda-13.3/bin/nvcc -std=c++20 -enable-tile -arch=sm_120 \
    -DPROBLEM_NAME='"w2"' -DPROBLEM_M=224 -DPROBLEM_K=2048 -DPROBLEM_N=7168 \
    -DMARLIN_PREV_US=89.53f \
    target/cuda_tile_probe/tile_w4a16_perf.cu -lcublas \
    -o target/cuda_tile_probe/tile_w4a16_w2_perf_sm120

  ./target/cuda_tile_probe/tile_w4a16_w13_perf_sm120 --iters 30
  ./target/cuda_tile_probe/tile_w4a16_w2_perf_sm120 --iters 30
  ```

  Results:

  | Problem | Shape | CUDA Tile W4A16 toy | CUDA Tile BF16-weight toy | cuBLAS BF16-weight | current Marlin local p95 |
  |---|---:|---:|---:|---:|---:|
  | W13 | `M=224,K=7168,N=4096` | `1609.62us` / `8.17 TF/s` | `1664.38us` / `7.90 TF/s` | `160.29us` / `82.06 TF/s` | `149.82us` |
  | W2 | `M=224,K=2048,N=7168` | `821.88us` / `8.00 TF/s` | `809.68us` / `8.12 TF/s` | `90.13us` / `72.97 TF/s` | `89.53us` |

- Compiler evidence:
  - `-tilecubin -Xtileiras --remarks=all,...` says the toy `mma` optimized to Tensor Cores.
  - On local `sm_120`, the reported instruction family is `Tensor-core SM80` with shape `[16,8,16]`, and SASS contains `HMMA.16816.F32.BF16`, not a Blackwell-specific packed-INT4 instruction.
- Interpretation:
  - The naive CUDA Tile W4A16 toy is about `10.7x` slower than Marlin for W13 and `9.2x` slower for W2.
  - The BF16-weight CUDA Tile variant is almost as slow as the W4 unpack variant, so the main gap is the generated Tile kernel shape/scheduling, not only nibble unpack.
  - cuBLAS BF16 at the same shapes is close to current local Marlin (`160.29us` vs `149.82us` W13, `90.13us` vs `89.53us` W2), which gives a useful sanity baseline for the local GPU.
- Decision: do not pursue this CUDA Tile C++ path as a Marlin replacement. A serious successor still needs a route-aware grouped/persistent design or a library/kernel path that directly supports packed W4A16/NVFP4-style operands. CUDA Tile remains useful for small compiler probes, not for the current Kimi Marlin hot path.

### Step 40: Profile doc consolidation
- Created `docs/models/kimi-k2/profiles/` as the Kimi-K2 decode profiling home.
- Moved the TP1/DP8/EP8 master ledger, TP1 PPLX bench log, fusion scan, and every Kimi kernel `*_report.md` into that directory.
- Added `README.md` as the local entry point and updated `docs/index.md` so profile docs route through `profiles/`.
- Kept root-level Kimi docs for architecture, correctness, serving/performance ledgers, and model-wide context.

### Step 41: Remove rejected partition-KV kernel code
- Removed the unused FlashInfer MLA partition probe from live code after review:
  - CUDA symbol `kimi_flashinfer_batch_decode_mla_partitioned_cuda`.
  - FFI binding `kimi_flashinfer_batch_decode_mla_partitioned_cuda`.
  - Rust runtime wrapper `kimi_flashinfer_batch_decode_mla_partitioned_rt`.
  - `kimi_tp1_pplx_decode_bench` partition CLI flags and measurement/check adapters.
- Kept the H20 artifacts and conclusions in this doc and `attention_flashinfer_mla_decode_report.md`.
- Result: production and bench binaries now expose only the accepted non-partitioned MLA decode path; reopening split-K requires reintroducing it as a fresh experiment after the global-bs64 runtime gate is healthy.

### Unexpected
- `--measure false` initially failed because clap's default bool flag handling did not accept an explicit value. Fixed by using `ArgAction::Set`.
- `Option<Vec<usize>>` with a CSV parser caused a clap downcast panic. Fixed by accepting raw strings and parsing CSV in the binary.
- The existing Kimi kernel report providers were TP8-shaped for MLA decode internals. Added runtime-dim TP1 provider paths instead of reusing TP8 constants.
- FlashInfer is repo-local at `pegainfer-kernels/third_party/flashinfer`; using an external checkout can hide source-layout and API-boundary differences. Keep this path in the repo instructions so future kernel work starts from the submodule.
- The first cuBLASLt implementation incorrectly treated active batch as graph bucket and only supported `1,2,4,8,16,32,64`. Fixed to name the dimension `batch_size` and prebuild plans for every `1..=64`, so `bs=3` does not fall back to generic cuBLAS.
- Runtime trace and operator bench use different batch terms: `kimi_tp1_pplx_decode_bench --active-rows 8` is a single DP-rank shape, while `kimi_kernel_report trace --batch-size 64 --tp-world 1 --dp-world 8` is the matching global-bs64 production trace. A `--batch-size 8` runtime trace is global bs8 and must not be used as the target-load gate.
- The FlashInfer MLA `partition_kv` production attempt failed before reaching the partition threshold. Record it as a rejected production adoption, not as a kernel win; unused probe code should not stay in the live kernel surface.

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
  - Kimi decode profile evidence belongs under `docs/models/kimi-k2/profiles/`; root-level model docs should stay as higher-level routing and design context.
- **Follow-ups**:
  - Run NCU on trace replay p95/max W13/W2 before changing Marlin scheduling or tiling.
  - Add an all-rank PPLX harness for dispatch/combine timing; trace replay covers local compute only and intentionally excludes EP transport.
  - For row 13, reopen FlashInfer MLA `partition_kv` only after the base TP1/DP8/PPLX global-bs64 runtime trace has a passing token gate; then reintroduce a fresh bench path before touching production decode.
