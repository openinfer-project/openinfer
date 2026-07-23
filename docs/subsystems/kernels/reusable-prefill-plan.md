# Reusable PrefillPagedPlan capacity and validation

> **TL;DR:** Issue #711 / PR #714 removes per-step uncompiled-GQA plan allocation for Qwen3-14B and Qwen3.5-27B. Allocation and accounting share one checked layout, the scheduler-derived logical page-table bound covers shared prefixes, device pointers remain stable across changing shapes, and both production checkpoints pass their runtime gates. Five Qwen3-14B Nsight A/B pairs remove 22 allocation/free calls per decode step with a paired-median 19.27 us/step reduction in those CUDA API calls and no change in metadata uploads or kernels.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` - routed the cross-model attention-plan work to the kernels subsystem.
  - `docs/models/qwen3/serving-perf-5090.md` - unified steps place prefill and decode rows in one `PrefillPagedPlan`.
  - `docs/models/qwen3/prefix-cache.md` - prefix hits reuse physical KV blocks while each request retains its own logical page list.
  - `docs/models/qwen3/model-crate.md` - Qwen3 owns scheduler and executor bounds while the reusable plan implementation lives in the shared kernel layer.
  - `docs/models/qwen35/model-crate.md` - Qwen3.5 owns a separate decode graph and startup memory budget.
  - `docs/subsystems/scheduler/scheduler.md` - admission bounds active plus prefilling requests by decode capacity and rejects sequences beyond the model context window.
  - `docs/conventions/bench-regression.md` - performance evidence needs repeated runs; p99 is inspected rather than inferred from a five-iteration sample.
  - `docs/conventions/coding-style.md` - tricky capacity invariants merit focused unit coverage, while runtime behavior should use an integration gate.
- **Relevant history**:
  - `docs/models/qwen3/serving-perf-5090.md` records the original unified-attention fusion and why decode rows carry full logical page lists.
  - PR #714's first RTX 5090 run reduced `cuMemAllocAsync` and `cuMemFreeAsync` calls from 708 to 642, but review found that physical pool size is not a valid logical page-table bound under shared prefixes.
- **Plan**:
  1. Add a checked `PrefillPagedPlan` capacity value derived from maximum step tokens, request rows, model context, page size, and GQA group size.
  2. Use `max_batch * ceil(max_context / page_size)` logical page references for Qwen3 and Qwen3.5, preserving startup memory accounting.
  3. Pass the same capacity and allocation layout through profiling, byte accounting, allocation, and serving ownership.
  4. Add CPU capacity coverage and a GPU gate that changes batch/context/page shapes while checking all eleven device pointers.
  5. Repeat the targeted tail benchmark and collect production Qwen3-14B allocation/API traces.
  6. Run Qwen3-14B golden and prefix-cache mixed gates plus Qwen3.5-27B short/long golden and scheduler E2E.
- **Acceptance risks**:
  - Prefix sharing can make logical page references exceed unique physical pages. The capacity therefore uses the scheduler/model logical bound, not KV pool size.
  - Persistent plan bytes compete with KV capacity. Every reserve is checked for overflow and charged before KV allocation.
  - An allocation-count win could hide changed work. The production traces pin identical decode steps, metadata uploads, and kernel counts.

## Execution Log

### 1. Derive the reachable logical capacity

Qwen3 admission subtracts both `active.len()` and `prefilling.len()` from `max_decode_batch_size`. Qwen3.5 likewise reserves prefilling promotion slots before admitting more work. Both schedulers reject request lifetimes beyond `max_position_embeddings`.

The safe logical page-reference bound is therefore:

```text
max_batch * ceil(max_position_embeddings / page_size)
```

This includes the worst shared-prefix case: every request keeps its own logical page list even when those lists refer to the same physical prefix pages.

### 2. Share layout, allocation, and accounting

- `PrefillPagedPlanCapacity` validates token, logical-page, batch, tile, and i32 ABI limits once.
- `PrefillPagedPlanLayout` is the single source for the eleven allocation lengths and their exact byte footprint.
- Qwen3 passes one capacity through temporary profile allocation, profile-byte subtraction, startup accounting, and the serving lane.
- Qwen3.5 charges the same capacity before KV allocation and passes it to `BatchDecodeGraphState`.
- Qwen3 and Qwen3.5 reserve arithmetic uses checked multiplication/addition through one checked total reused by admission and subtraction.
- Runtime updates return dimension-specific errors; they never fall back to the allocating constructor.

The CPU layout tests cover the shared-prefix logical bound, invalid capacity, exact allocation layout, and i32 overflow. The GPU gate updates different page, token, and batch shapes, confirms all eleven device pointers remain unchanged, and checks page/token/batch/tile capacity errors.

### 3. Validate shared-prefix mixed execution

The Qwen3 group-size-5 fixture ran the complete `prefix_cache_behavior` integration test. It covered repeated warm prefixes, cold plus warm multi-request batching, the full-block prefix cap, and warm prefill plus active decode in one unified step. The test passed `1/1`.

The final production rerun used Qwen3-14B itself and passed the same gate. Its mixed-batch rows were bit-identical to the cold reference; the unified-plan warm row stayed within the gate (`mean 0.0335`, `max 0.0592`).

### 4. Repeat the targeted tail comparison

The original five-iteration batch-16 result moved p99 from `1.384` to `1.737 ms`. The longer rerun fixed context at 512, batch at 16, decode at 256 steps, warmup at 32 steps, and distinct prompts at 16. CUDA Graphs were disabled. Five interleaved base/PR pairs each ran 100 iterations, producing 356,800 TPOT samples per run.

The retained JSON records base `41b77566e48d46673cc0d8c7f279f6dbe275f4f5` and PR implementation `59a74963c5bf845befd55b8bd92b55681f1e72c1`. The collection loop was:

```bash
run_decode() {
  revision=$1
  binary=$2
  seed=$3
  "$binary" \
    --model-path /root/autodl-tmp/models/Qwen3-Group5-1L-fixture \
    --cuda-graph false \
    --format json \
    --label "${revision}-100iter-s${seed}" \
    --out "/root/autodl-tmp/pr714-100iter/${revision}-s${seed}.json" \
    decode \
    --ctxs 512 \
    --batches 16 \
    --decode-steps 256 \
    --warmup-steps 32 \
    --distinct-prompts 16 \
    --iters 100 \
    --seed "$seed"
}

BASE_BIN=/root/autodl-tmp/target-base/release/bench_serving
PR_BIN=/root/autodl-tmp/target-pr714/release/bench_serving

run_decode base "$BASE_BIN" 47; run_decode pr "$PR_BIN" 47
run_decode pr "$PR_BIN" 48; run_decode base "$BASE_BIN" 48
run_decode base "$BASE_BIN" 49; run_decode pr "$PR_BIN" 49
run_decode pr "$PR_BIN" 50; run_decode base "$BASE_BIN" 50
run_decode base "$BASE_BIN" 51; run_decode pr "$PR_BIN" 51
```

| Seed | Average TPOT base -> PR | Delta | p99 TPOT base -> PR | Delta | Throughput base -> PR | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 47 | 0.867688 -> 0.916159 ms | +5.59% | 1.013080 -> 1.964354 ms | +93.90% | 18,546.04 -> 18,153.58 tok/s | -2.12% |
| 48 | 0.864537 -> 0.852398 ms | -1.40% | 1.021063 -> 1.005956 ms | -1.48% | 18,581.68 -> 18,765.62 tok/s | +0.99% |
| 49 | 0.923673 -> 0.859485 ms | -6.95% | 1.867574 -> 1.003935 ms | -46.24% | 17,911.38 -> 18,711.22 tok/s | +4.47% |
| 50 | 0.917316 -> 0.858558 ms | -6.41% | 1.833100 -> 0.994906 ms | -45.73% | 18,023.70 -> 18,709.42 tok/s | +3.80% |
| 51 | 0.866872 -> 0.863014 ms | -0.45% | 1.033162 -> 1.061248 ms | +2.72% | 18,720.28 -> 18,778.51 tok/s | +0.31% |

| Metric | Base median | PR median | Paired-delta median |
| --- | ---: | ---: | ---: |
| Average TPOT | 0.867688 ms | 0.859485 ms | -1.40% |
| p99 TPOT | 1.033162 ms | 1.005956 ms | -1.48% |
| Decode throughput | 18,546.04 tok/s | 18,711.22 tok/s | +0.99% |

The isolated p99 spikes occur on both revisions. The repeated, order-interleaved medians do not reproduce the original short-run regression.

### 5. Measure production allocation and plan API time

The production A/B used Qwen3-14B group 5 on one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), driver `580.119.02`, CUDA 12.8, `OPENINFER_CUDA_SM=120`, Rust nightly 2026-07-10, and Nsight Systems exporter 2024.6.2. The model revision was `40c069824f4251a91eefaf281ebe4c544efd3e18`; all eight weight shards matched the official Hugging Face SHA-256 values.

The retained binaries are independently identifiable:

```text
base a62416f12c6384fae2f4ce81474fb2f50a772df219808caf7462840b8432373a
PR   fa60e425a59fef53066aa3ef3dc9fdac5b13a8390127f1b394d3c573376dee5a
```

Each trace captured exactly 32 steady eager decode steps at context 512 and batch 16. This is the exact pair-1 command preserved in the trace metadata; pairs 2-5 changed only the output pair number and selected base/PR binary:

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --export=sqlite \
  -o /root/autodl-tmp/pr714-nsys-qwen3-14b/pair1-base \
  /root/autodl-tmp/pr714-ab/bin/qwen3_decode_context-base \
  --mode profile \
  --model-path /root/autodl-tmp/models/Qwen3-14B \
  --contexts 512 \
  --batch-size 16 \
  --profile-steps 32 \
  --capture-range \
  --disable-cuda-graph
```

CUDA Driver API counts and CPU durations were queried directly from the exported CUPTI table:

```sql
SELECT
    s.value AS api,
    COUNT(*) AS calls,
    ROUND(SUM(r.end - r.start) / 1000000.0, 6) AS cpu_api_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
JOIN StringIds AS s ON s.id = r.nameId
WHERE s.value IN (
    'cuMemAllocAsync',
    'cuMemFreeAsync',
    'cuMemcpyHtoDAsync_v2',
    'cuEventSynchronize'
)
GROUP BY s.value
ORDER BY s.value;
```

Kernel and device-copy work were checked separately:

```sql
SELECT
    (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL) AS kernels,
    (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY) AS device_copies,
    ROUND((SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY) / 1000000.0, 6)
        AS device_copy_ms;
```

| Pair | Alloc+free calls base -> PR | Alloc+free API ms base -> PR | Saved per step | Alloc+free+H2D API ms base -> PR | Profile TPOT base -> PR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,728 -> 1,024 | 2.448186 -> 2.037476 | 12.83 us | 4.567320 -> 4.402517 | 24.5670 -> 24.6304 ms |
| 2 | 1,728 -> 1,024 | 2.887641 -> 2.253055 | 19.83 us | 5.270461 -> 4.902352 | 24.6975 -> 24.5400 ms |
| 3 | 1,728 -> 1,024 | 2.387319 -> 1.770831 | 19.27 us | 4.416813 -> 3.958116 | 24.6127 -> 24.5858 ms |
| 4 | 1,728 -> 1,024 | 2.750674 -> 2.764723 | -0.44 us | 5.269879 -> 6.438497 | 24.1020 -> 24.3742 ms |
| 5 | 1,728 -> 1,024 | 4.199736 -> 2.456119 | 54.49 us | 8.314309 -> 5.859004 | 24.8021 -> 24.5328 ms |

Each API individually changes from 864 to 512 calls: `-352`, or exactly `-11` calls per step, for both allocation and free. The paired-median allocation/free API saving is `0.616488 ms / 32 = 19.265 us/step`. Including the unchanged H2D API calls, paired-median host API saving is `0.368109 ms / 32 = 11.503 us/step`.

Every trace contains 448 `cuMemcpyHtoDAsync_v2` calls, 480 device-copy activities, 24,512 kernels, and 32 `cuEventSynchronize` calls on both revisions. Device-copy time stays within `0.172-0.181 ms` per trace. The optimization therefore removes allocation/free and pointer churn while retaining metadata uploads and computational work. Profile TPOT has a `24.6127 -> 24.5400 ms` median and a `-0.11%` paired-delta median, so the production claim is reduced plan-preparation API work, not a material TPOT speedup.

### 6. Run production correctness and scheduler gates

The production-gate source was rebased on `origin/main` `90440c5` and synced byte-for-byte to the GPU host. The final submission was then rebased onto `cb41b07` (#745); `git range-diff` shows that this only narrows the Qwen3.5 capacity field from `pub(super)` to private, leaving the runtime diff unchanged. Qwen3-14B used revision `40c069824f4251a91eefaf281ebe4c544efd3e18`. Qwen3.5-27B used revision `fc05daec18b0a78c049392ed2e771dde82bdf654`; all eleven weight shards and its tokenizer matched the official Hugging Face SHA-256 values.

Environment and commands:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export OPENINFER_CUDA_SM=120
export OPENINFER_NVCC_JOBS=2
export OPENINFER_TRITON_PYTHON=/root/miniconda3/bin/python
export CARGO_TARGET_DIR=/root/autodl-tmp/target-pr714
export RUSTUP_TOOLCHAIN=nightly-2026-07-10-x86_64-unknown-linux-gnu

cargo fmt --all -- --check
cargo check --release --locked -p openinfer-server --all-targets
cargo clippy --release --locked \
  -p openinfer-server -p openinfer-qwen3 -p openinfer-core \
  -p openinfer-kernels -p openinfer-kv-cache -p openinfer-kv-offload \
  -p openinfer-sample -p openinfer-bench \
  --all-targets -- -D warnings
cargo check --release --locked -p openinfer-qwen35-4b \
  --all-targets --features qwen35-4b

cargo test --release --locked -p openinfer-core --lib \
  paged_plan::tests -- --nocapture
cargo test --release --locked -p openinfer-kernels --lib \
  preallocated_footprint -- --nocapture
cargo test --release --locked -p openinfer-kernels --lib \
  preallocated_update_preserves_all_pointers_and_reports_capacities \
  -- --ignored --nocapture

OPENINFER_TEST_MODEL_PATH=/root/autodl-tmp/models/Qwen3-14B \
  cargo test --release --locked -p openinfer-qwen3 \
  --test hf_golden_gate -- --nocapture
OPENINFER_TEST_MODEL_PATH=/root/autodl-tmp/models/Qwen3-14B \
  cargo test --release --locked -p openinfer-qwen3 \
  --test prefix_cache -- --nocapture

export OPENINFER_TEST_MODEL_PATH=/root/autodl-tmp/models/Qwen3.5-27B
export OPENINFER_TEST_MODEL_REVISION=fc05daec18b0a78c049392ed2e771dde82bdf654
cargo test --release --locked -p openinfer-qwen35-4b --features qwen35-4b \
  --test hf_golden_gate \
  pega_logprobs_match_hf_golden_within_qwen35_tolerance -- --nocapture
cargo test --release --locked -p openinfer-qwen35-4b --features qwen35-4b \
  --test hf_golden_gate \
  pega_logprobs_match_hf_long_golden_within_qwen35_tolerance -- --nocapture
cargo test --release --locked -p openinfer-qwen35-4b --features qwen35-4b \
  --test e2e_scheduler test_e2e_qwen35_scheduler -- --nocapture
```

Final receipts:

| Gate | Result | Key receipt |
| --- | --- | --- |
| Capacity/layout CPU tests | pass, 4 tests | shared-prefix bound, invalid page size, exact layout, i32 overflow |
| GPU pointer/capacity gate | pass, 1 test | all eleven pointers stable across changing shapes; dimension errors checked |
| Qwen3-14B HF golden | pass, 1 test | sequential mean `0.0291`, p99 `0.1087`; cached replay mean `0.0281`, p99 `0.1046` |
| Qwen3-14B prefix-cache mixed | pass, 1 test | warm/cold mixed and unified-plan paths pass |
| Qwen3.5-27B short golden | pass, 1 test | sequential mean `0.0201`, p99 `0.0694`; batch-5 p99 `0.0651`; slot-compaction p99 `0.0755` |
| Qwen3.5-27B long golden | pass, 1 test | prompt lengths 4097/8192; mean `0.0218`, p99 `0.0627` |
| Qwen3.5-27B scheduler E2E | pass, 1 test | full TP1 request-flow/liveness suite, `60.47 s` |

## Debrief

- **Outcome**: Both production uncompiled-GQA owners reuse fixed-capacity plan storage. Capacity follows reachable logical request state, allocation and byte accounting share one layout, startup arithmetic is checked end to end, and runtime overflow fails with a dimension-specific error.
- **Performance**: Targeted repeated medians show no tail regression. Production Nsight traces remove exactly eleven allocations and eleven frees per decode step, preserve every metadata upload and kernel, and reduce the paired-median allocation/free API time by 19.27 us/step.
- **Correctness**: Shared-prefix mixed execution, actual GPU pointer stability, Qwen3-14B golden/prefix-cache, and Qwen3.5-27B short/long golden plus scheduler E2E all pass on the final rebased source.
