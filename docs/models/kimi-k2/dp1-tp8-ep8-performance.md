# Kimi-K2 DP1 TP8 EP8 Performance

> **TL;DR:** DP1 TP8 EP8 的性能主线从 correctness baseline
> `72c770b` 开始。目标是在 H20 ×8、bs64、decode-heavy 服务口径下超过
> vLLM `0.19.0` 的 bs64 baseline：output `583.9 tok/s`，TPOT median
> `109.00ms`。
>
> **Status:** Project doc opened. No performance optimization is accepted here
> until it has a correctness gate and its own commit.

## Target

| Item | Target / baseline |
| --- | --- |
| Machine | `h20-100`, 8× NVIDIA H20 |
| Model | `/data/models/Kimi-K2.5` |
| Shape | DP1 TP8 EP8 |
| Primary workload | `input_len=1`, `output_len=128`, `ignore-eos`, `bs=64` |
| vLLM baseline | TP1 DP8 EP8, `vllm bench serve`, output `583.9 tok/s`, TPOT median `109.00ms`, TPOT p99 `109.76ms` |
| PegaInfer goal | output tok/s `> 583.9` at bs64. Exact-token baselines remain the default; any accepted drift must be marked and counted. |

The comparison target comes from [vllm-h20-baseline.md](vllm-h20-baseline.md).
The correctness ground truth starts from
[pplx-ep-correctness.md](pplx-ep-correctness.md): TP8 PPLX must be token-trace
exact against TP8 NCCL under the same bs64 active-decode schedule.

Default policy is exact token-trace parity. For large-batch performance work,
an entry may be kept only as a drift-recorded performance baseline when the
mismatch count, hash distribution, exact reference point, and revert line are
all recorded. Such an entry does not replace the exact-token reference.

## Gate Rules

Every kept optimization needs all of these recorded before commit:

| Gate | Requirement |
| --- | --- |
| Profile | Start from an observed profile or benchmark delta. Record the command, output path, and the measured bottleneck or symptom. |
| Motivation / expected gain | State why the change should help and the expected size/direction of the win before implementing it. |
| Microbench | Add or run the smallest probe that isolates the changed subsystem when practical. If no microbench is practical, record why and use the closest lower-level measurement. |
| Correctness | Record the exact command, output file, token hash, and comparison target. For TP8/PPLX changes, compare against the TP8 NCCL baseline unless a stronger reference is documented. |
| Drift exception | Exact parity is required by default. A drift-recorded performance baseline must say which exact reference remains authoritative and record mismatch counts plus hash distribution. |
| Performance | Record bs64 service numbers and the lower-level in-process probe that explains the delta. |
| Scope | State whether the optimization targets frontend/scheduler, CUDA graph, collectives, MLA, MoE, or sampling. |
| Revert line | Record the measurable regression that would make the change revert-worthy. |
| Commit | Commit the code and this doc update together. |

No optimization is accepted on performance numbers alone.

Preferred entry shape:

```text
Profile:
  <command + report path + bottleneck>
Motivation / expected gain:
  <why this change should move bs64, and by roughly how much>
Microbench:
  <isolated probe, or the reason a subsystem-only probe is not available>
Correctness gate:
  <hash / trace / reference path>
Performance gate:
  <bs64 service number + supporting in-process/profile number>
Decision:
  <keep/reject/defer + commit>
```

This is a discipline, not a rigid template. The important part is that future
readers can reconstruct why an optimization was attempted, what it was expected
to buy, and which evidence made it worth keeping.

## Canonical Bs64 Pressure Test

Use this exact service pressure-test shape for bs64 comparisons. This is the
single project-wide pressure command for Kimi-K2 bs64 reports. Do not change
prompt/output length, request count, request rate, concurrency, percentiles,
streaming mode, or `ignore-eos` when reporting numbers against the vLLM bs64
baseline.

Server:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep -- \
  --model-path /data/models/Kimi-K2.5 \
  --port 8124 \
  --cuda-graph true
```

Client:

```bash
cd /root/develop/xingming/pegainfer
COMMIT=$(git rev-parse --short HEAD)
mkdir -p /tmp/kimi-bs64-baseline
source /root/develop/xingming/vllm_test/.venv/bin/activate
vllm bench serve \
  --backend openai \
  --model /data/models/Kimi-K2.5 \
  --tokenizer /data/models/Kimi-K2.5 \
  --trust-remote-code \
  --base-url http://127.0.0.1:8124 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 256 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/kimi-bs64-baseline \
  --result-filename pegainfer_tp8_pplx_bs64_${COMMIT}.json \
  2>&1 | tee /tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_${COMMIT}.log
```

Required report fields:

| Field | Value |
| --- | --- |
| `--random-input-len` | `1` |
| `--random-output-len` | `128` |
| `--random-range-ratio` | `0` |
| `--num-prompts` | `256` |
| `--max-concurrency` | `64` |
| `--request-rate` | `inf` |
| `--ignore-eos` | enabled |
| `--percentile-metrics` | `ttft,tpot,itl` |
| `--metric-percentiles` | `50,95,99` |

Supporting in-process probe:

Use this command when a change needs a lower-level bs64 number without HTTP,
SSE, and vLLM bridge overhead. It is not a replacement for the canonical
service pressure test; it exists to explain service deltas with a stable
engine-side shape.

```bash
cd /root/develop/xingming/pegainfer
COMMIT=$(git rev-parse --short HEAD)
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep \
  --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_micro_bs64_o128_${COMMIT}.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

## Correctness Probe

Run this before accepting a performance change, and compare it with the TP8 NCCL
reference from [pplx-ep-correctness.md](pplx-ep-correctness.md):

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_correctness64.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

For math-kernel changes, also run the stronger output128 in-process trace
comparison. The output5 probe catches early PPLX/NCCL path drift, but R5 showed
router GEMM precision changes can pass output5 and diverge later.

```bash
cd /root/develop/xingming/pegainfer
COMMIT=$(git rev-parse --short HEAD)
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_math_correctness_bs64_o128_${COMMIT}.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

## Optimization Ledger

| ID | Date | Commit | Area | Change | Correctness gate | bs64 result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 2026-05-25 | `72c770b` | correctness | TP8 PPLX baseline fixed; no performance claim | TP8 NCCL/PPLX 64-token hash `4920f088c2338236` | Not measured | Keep as ground truth |
| B1 | 2026-05-25 | `d639e55` code, `df1cd18` command doc | scheduler / service profile | Canonical bs64 pressure baseline before performance work | No code change after B0; PPLX correctness baseline remains `4920f088c2338236` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json`: output `137.51 tok/s`, TPOT p50/p95/p99 `26.40/28.13/28.46ms`, TTFT p50/p99 `54.76/58.68s`, 256/256 success | Keep as profile baseline; first optimization should address 4-row scheduling/admission before kernel work |
| O1 | 2026-05-25 | this commit | scheduler / decode arena | Raise DP1 TP8 admission to bs64; allocate decode arenas lazily in `1/2/4/8/16/32/64` buckets; preflight arena allocation on all TP ranks before prefill collectives | `/tmp/kimi_pplx_tp8_correctness64_o1_bucket.json`: TP8 PPLX 64-token hash `4920f088c2338236` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.json`: output `145.18 tok/s`, TPOT p50/p95/p99 `195.07/221.08/224.72ms`, TTFT p50/p99 `31.00/35.76s`, 256/256 success | Keep as bs64 enabling baseline; not enough for vLLM target, next profile must attack bs64 kernel/communication cost |
| C1 | 2026-05-25 | this commit | correctness / PPLX MoE | Align TP8 PPLX with TP8 NCCL for active bs64 decode: active MoE rows, TP8-only duplicate-source canonicalization, NCCL-layout local expert compute before PPLX combine | `/tmp/kimi_pplx_tp8_active64_o5_after_review.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 per-index token mismatches; both paths hash counter `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` | Not a performance optimization; PPLX correctness probe TPOT p50 `110.14ms` vs NCCL `97.53ms`; rerun canonical bs64 pressure after this correctness commit | Keep as the new correctness baseline before further optimization |
| P1 | 2026-05-25 | documentation only | service / scheduler profile | Profile `00b3f1f` after C1 with the canonical bs64 command and an in-process bs64/output128 microbench | No code change after C1; C1 correctness baseline remains the gate | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.json`: output `353.91 tok/s`, TPOT p50/p95/p99 `146.15/172.83/175.10ms`, TTFT p50/p99 `4.58/10.24s`, 256/256 success; in-process warm1 steady TPOT p50 `107.76ms` | Keep as profile baseline; next optimization should target serial first-token prefill without changing token trace |
| O2 | 2026-05-25 | this commit | scheduler / MLA prefill | Replace prompt_len=1 first-token MLA attention with the exact single-token V path; keep microbatch at 1 because seq_len>1 drifted | `/tmp/kimi_pplx_tp8_c1fast_mb1_o5.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 mismatches; hash counter `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fastmb1_candidate.json`: output `414.28 tok/s`, TPOT p50/p95/p99 `133.36/147.74/149.42ms`, TTFT p50/p99 `2.76/6.90s`, 256/256 success | Keep as an incremental first-token optimization; still below vLLM, next work must make batch>1 prompt_len=1 prefill correct or reduce PPLX TPOT |
| O3 | 2026-05-25 | this commit | scheduler / prompt_len=1 prefill | Reuse prompt_len=1 dense/shared/router/Marlin scratch for the single-row prefill path, and widen the fixed admission coalesce window to `100ms` so bs64 pressure is admitted as one wave | `/tmp/kimi_pplx_tp8_o3_scratch_coalesce_o5.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 mismatches; hash counter `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o3_scratch_coalesce_candidate.json`: output `492.34 tok/s`, TPOT p50/p95/p99 `121.05/124.99/125.58ms`, TTFT p50/p99 `0.67/3.96s`, 256/256 success | Keep as a measured bs64 improvement; still below vLLM, next work should attack service TPOT/ITL and PPLX steady decode |
| O5 | 2026-05-25 | this commit | PPLX / MoE stream overlap | Start the PPLX decode router on the aux stream immediately after RMSNorm, matching the NCCL decode overlap window instead of waiting for shared expert/all-reduce | `/tmp/kimi_pplx_tp8_o5_router_overlap_o5.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 mismatches; hash counter `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o5_router_overlap_candidate.json`: output `509.89 tok/s`, TPOT p50/p95/p99 `116.53/120.45/121.44ms`, TTFT p50/p99 `0.67/3.95s`, 256/256 success | Keep as a measured PPLX decode improvement; still below vLLM, next work should remove PPLX TP8 dispatch/copy overhead |
| O7 | 2026-05-25 | this commit | PPLX / dispatch recv | Add a metadata/counts-only `dispatch_recv` path for TP8 decode, where local experts no longer consume the dispatched hidden payload | `/tmp/kimi_pplx_tp8_counts_recv_short_v2.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 mismatches; hash counter `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_counts_recv_v2.json`: output `511.78 tok/s`, TPOT p50/p95/p99 `115.83/120.26/121.26ms`, TTFT p50/p99 `0.67/4.07s`, 256/256 success; in-process bs64/o128 reached `589.98 tok/s`, TPOT p50 `101.58ms` | Keep as a measured PPLX decode improvement; still below vLLM service target, next work should remove dispatch-send payload or compact scatter |
| O8 | 2026-05-25 | this commit | compute / RMSNorm fusion | Fuse attention residual add with post-attention RMSNorm using a Kimi-specific kernel that first materializes the BF16-rounded residual sum | `/tmp/kimi_pplx_tp8_fused_addrms_round_bs64_o128.json` vs `/tmp/kimi_pplx_tp8_counts_recv_micro_bs64_o128_warm1_v2.json`: 0 output128 mismatches; hash counter `32x 82a791616c737442`, `16x 4ae8834e96c7d195`, `16x 24b2b3856ac0ea3a` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fused_addrms_round.json`: output `516.44 tok/s`, TPOT p50/p95/p99 `114.92/118.95/119.57ms`, TTFT p50/p95/p99 `0.66/3.81/3.97s`, 256/256 success; in-process bs64/o128 steady TPOT p50 `101.57ms` | Keep. This recovers the add+rms launch/memory win that R4 attempted, without changing token traces. |
| O9 | 2026-05-25 | this commit | scheduler / prompt_len=1 prefill | Let prompt_len=1 first-token prefill run in microbatch `2`, using row-wise `per_token` GEMM/router/all-reduce boundaries that preserve the decode math order | `/tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb2_bs64_o5_probe.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 mismatches; `/tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb2_bs64_o128_probe.json` vs O8/O7 output128: 0 mismatches | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_per_token_mb2_candidate.json`: output `523.88 tok/s`, TPOT p50/p95/p99 `113.88/117.51/118.37ms`, TTFT p50/p95/p99 `0.50/3.75/3.86s`, 256/256 success; in-process bs64/o128 steady TPOT p50 `101.68ms` | Keep. This is a small service win that preserves token traces; larger microbatches remain rejected until layer parity is proven. |
| O10 | 2026-05-25 | this commit | scheduler / prompt_len=1 prefill | Allocate bs64 decode arena and warm prompt_len=1 before the service is ready, then run prompt_len=1 at microbatch `64` under the recorded large-batch drift gate | Final candidate records drift: o5 `48/64` mismatches vs TP8 NCCL; o128 `64/64` mismatches vs O8/O7. Exact reference point: mb8 was `0/64` on o5 and o128 but slower. | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb64_candidate.json`: output `557.71 tok/s`, TPOT p50/p95/p99 `110.95/115.30/115.34ms`, TTFT p50/p95/p99 `473.94/504.25/506.18ms`, 256/256 success; in-process bs64/o128 first decode p50 `104.61ms` | Keep as a drift-recorded performance baseline, not an exact-token baseline; still below vLLM output `583.9 tok/s`, so the next work must target steady TPOT/ITL. |
| O11 | 2026-05-25 | this commit | PPLX / dispatch send | Add TP route-only `dispatch_send` for the PPLX counts-only TP8 decode path, preserving route metadata and worker sync while skipping the unused hidden payload copy | No extra drift over O10: pending-guard route-only o5 vs O10 `0/64`; route-only o128 vs O10 `0/64`. Drift remains recorded as o5 `48/64` vs TP8 NCCL and o128 `64/64` vs O8. | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb64_route_send_pending_guard.json`: output `564.49 tok/s`, TPOT p50/p95/p99 `110.54/111.60/111.60ms`, TTFT p50/p95/p99 `472.88/500.05/501.35ms`, 256/256 success; A2A split p50 `158.6us -> 135.1us` | Keep as a measured PPLX decode improvement; still below vLLM output `583.9 tok/s`, so continue with the remaining combine/scatter/GEMM profile items. |

### O11 PPLX Route-Only Dispatch Send

Profile:

```text
/tmp/kimi-profile/o10-steady/bs64_o64_graph_nodes.nsys-rep
/tmp/kimi-profile/o10-steady/bs64_o64_graph_nodes_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
```

The O10 steady profile shows `a2a_dispatch_send_kernel` as a visible PPLX
decode cost: `60,480` instances, total `4.40s`, avg `72.72us`, median
`71.07us`. The surrounding PPLX costs were `a2a_combine_recv_kernel` avg
`128.37us`, Marlin routed expert avg `59.47us`, `a2a_combine_send_kernel` avg
`32.68us`, compact scatter avg `30.89us`, and `dispatch_recv_counts` avg
`21.69us`.

Motivation / expected gain:

After O7, TP8/DP1 decode uses `dispatch_recv_counts` and then computes local
experts from the post-collective hidden state on each rank. The dispatched
hidden payload is no longer consumed on this TP path; only route metadata and
the PPLX worker protocol are required. A route-only send should remove the BF16
hidden payload copies while preserving route counters, `dispatch_send_done`,
and the node sync flags. Expected gain is one small but direct PPLX split
reduction, with no token-trace change relative to O10.

Safety constraints are part of the optimization, not a caller convention:
`dispatch_send_route_only` is intra-node only, rejects a following full
`dispatch_recv`, and route-only pending state allows only
`dispatch_recv_counts` before the next stage. The bench CLI also rejects
`--dispatch-send-route-only` unless `--dispatch-recv-counts-only` is present.
The Kimi caller gates the route-only path on TP8/DP1 duplicate-source
canonicalization.

Microbench:

```bash
target/release/pplx_a2a_bench \
  --n-experts 384 --topk 8 --hidden-dim 7168 --world-size 8 \
  --max-num-tokens 64 --max-private-tokens 64 --expert-padding 8 \
  --nets-per-gpu 1 --warmup 20 --repeats 100 \
  --dispatch-recv-counts-only --canonicalize-duplicate-sources

target/release/pplx_a2a_bench \
  --n-experts 384 --topk 8 --hidden-dim 7168 --world-size 8 \
  --max-num-tokens 64 --max-private-tokens 64 --expert-padding 8 \
  --nets-per-gpu 1 --warmup 20 --repeats 100 \
  --dispatch-recv-counts-only --dispatch-send-route-only \
  --canonicalize-duplicate-sources
```

Results:

- Full send: `/tmp/kimi-route-only/pplx_a2a_kimi_bs64_full_send_counts_recv.txt`;
  `dispatch_send` p50/p95/p99 `73.25/78.30/82.21us`, max-rank split
  p50/p95/p99 `158.6/173.2/179.7us`.
- Route-only send:
  `/tmp/kimi-route-only/pplx_a2a_kimi_bs64_route_send_counts_recv.txt`;
  `dispatch_send` p50/p95/p99 `46.11/49.38/52.58us`, max-rank split
  p50/p95/p99 `136.3/154.0/171.8us`.
- Guarded repeat:
  `/tmp/kimi-route-only/pplx_a2a_kimi_bs64_route_send_counts_recv_guarded.txt`;
  `dispatch_send` p50/p95/p99 `45.60/49.09/51.42us`, max-rank split
  p50/p95/p99 `135.1/149.1/159.3us`.

Correctness gate:

```text
/tmp/kimi-route-only/kimi_pplx_tp8_route_send_mb64_bs64_o5_probe.json
/tmp/kimi-route-only/kimi_pplx_tp8_route_send_mb64_bs64_o128_probe.json
/tmp/kimi-route-only/kimi_pplx_tp8_route_send_guarded_mb64_bs64_o5_probe.json
/tmp/kimi-route-only/kimi_pplx_tp8_route_send_pending_guard_mb64_bs64_o5_probe.json
```

Recorded comparison counts:

- o5 vs O10: `0/64` mismatches.
- guarded o5 vs O10: `0/64` mismatches.
- pending-guard o5 vs O10: `0/64` mismatches.
- o5 vs TP8 NCCL: `48/64` mismatches.
- guarded o5 vs TP8 NCCL: `48/64` mismatches.
- pending-guard o5 vs TP8 NCCL: `48/64` mismatches.
- o128 vs O10: `0/64` mismatches.
- o128 vs O8: `64/64` mismatches.

The mismatch counts match O10's drift envelope, so route-only send adds no new
token drift.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb64_route_send_pending_guard.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb64_route_send_pending_guard.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `564.49 tok/s` vs O10 `557.71 tok/s`.
- TTFT p50/p95/p99: `472.88/500.05/501.35ms`.
- TPOT p50/p95/p99: `110.54/111.60/111.60ms`.
- ITL p50/p95/p99: `110.15/112.81/115.35ms`.

Decision:

Keep. The gain is modest but the subsystem microbench confirms the skipped work,
service throughput moves in the expected direction, token drift is unchanged
from the O10 drift-recorded baseline, and the API now rejects the unsafe
route-only/full-recv pairing. Revert if this path causes any non-O10 token
mismatch or if repeat canonical bs64 service drops below O10 throughput with
the same test shape.

### B1 Profile Notes

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json
```

Observed:

- bs64 output throughput is `137.51 tok/s`, far below vLLM bs64 `583.9 tok/s`.
- TPOT p50 is `26.40ms`, much lower than vLLM bs64 TPOT p50 `109.00ms`.
- TTFT p50 is `54.76s`, showing requests are queued in long waves.
- Current TP8 scheduler cap is still `KIMI_RUNNER_MAX_BATCH = 4`, so bs64 service
  pressure effectively runs as repeated 4-row decode waves.

Motivation / expected gain:

Raising the DP1 TP8 admission/arena path beyond 4 rows should attack the main
service-throughput gap directly. If per-token TPOT stayed near the B1 value,
bs64 throughput would have roughly 4x headroom before kernel scaling becomes the
dominant limit. The actual gain must be measured because MLA/MoE kernels,
collectives, scratch size, and graph capture may scale nonlinearly with batch.

Microbench:

B1 is a service profile, not an optimization. The next optimization must add a
lower-level probe for the changed layer, for example an in-process bs sweep or a
decode arena/scheduler trace that confirms active rows > 4 before rerunning the
canonical bs64 pressure command.

### O1 Lazy Bucketed Bs64 Decode Arenas

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json
```

Observed:

- Canonical bs64 service throughput was only `137.51 tok/s`, with TTFT p50
  `54.76s`.
- TPOT p50 was `26.40ms`, which was good for each 4-row wave but did not
  translate into bs64 service throughput.
- Code profile: TP8 scheduler admitted at most `KIMI_RUNNER_MAX_BATCH = 4`,
  and worker startup allocated all decode arenas eagerly up to the worker cap.

Motivation / expected gain:

Raising the scheduler and worker cap to 64 removes the obvious admission limit.
Decode arenas are allocated lazily in power-of-two buckets so canonical bs64
uses one bs64 KV/scratch/graph arena without allocating every size from 1 to 64.
The rank preflight makes allocation failure happen before prefill/decode
collectives, avoiding a partial-rank failure mode. Expected direction: much
lower bs64 TTFT and enough active rows to expose the real bs64 kernel and PPLX
communication cost.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_o1_bucket_micro_bs64.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 0 --iters 1
Result:

- Output path: `/tmp/kimi_pplx_tp8_o1_bucket_micro_bs64.json`.
- Workload confirmed `concurrency=64`, `output_len=128`, all `64` traces had
  length `128`.
- In-process wall throughput, computed as `64 * 128 / max_e2e`, was about
  `226.9 tok/s` (`max_e2e=36.108s`).
- Steady TPOT p50/p95/p99 was `178.35/201.96/218.85ms`.

Correctness gate:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_correctness64_o1_bucket.json \
  request --output-len 64 --warmup 0 --iters 1
```

Result: generated-token hash `4920f088c2338236`, matching the TP8 NCCL/PPLX
baseline.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `145.18 tok/s` vs B1 `137.51 tok/s`.
- Peak output throughput: `504.00 tok/s` vs B1 `168.00 tok/s`.
- TTFT p50/p95/p99: `31.00/35.23/35.76s` vs B1 p50/p99
  `54.76/58.68s`.
- TPOT p50/p95/p99: `195.07/221.08/224.72ms`.

Decision:

Keep. O1 preserves token correctness and turns bs64 into real 64-row decode
waves, but the service output throughput is still far below the vLLM `583.9
tok/s` target. The next accepted optimization needs a profile of the bs64
decode step itself, especially PPLX MoE routing/combine, MLA decode, and TP
collectives.

### P1 Post-C1 Bs64 Profile

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.json
```

Observed:

- Canonical bs64 service result at `00b3f1f`: `256/256` success, output
  throughput `353.91 tok/s`, request throughput `2.76 req/s`.
- TTFT p50/p95/p99: `4.58/9.04/10.24s`.
- TPOT p50/p95/p99: `146.15/172.83/175.10ms`.
- ITL p50/p95/p99: `116.62/119.65/122.74ms`.
- Peak output-token bucket from vLLM bench result:
  `max_output_tokens_per_s=640.0`.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_00b3f1f_micro_bs64_o128_warm1.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Result:

- Output path:
  `/tmp/kimi_pplx_tp8_00b3f1f_micro_bs64_o128_warm1.json`.
- All `64` traces have length `128`.
- Hash counter: `32x 82a791616c737442`, `16x 4ae8834e96c7d195`,
  `16x 24b2b3856ac0ea3a`.
- Steady TPOT p50/p95/p99: `107.76/109.06/110.45ms`, equivalent to about
  `593.9 tok/s` for a 64-row decode step.
- End-to-end p50/max: `20.81/20.81s`, equivalent to about `393.6 tok/s`
  over `64 * 128` output tokens.

Motivation / expected gain:

The steady decode step is already in the vLLM target range, while request e2e is
not. `bench_serving` measures `first_decode_step_ms` as the interval from the
first emitted token to the second emitted token; it is not the first kernel
duration. Code inspection shows the Kimi scheduler coalesces 64 requests, then
runs `prefill_request` one slot at a time before entering batched decode. The
observed wall time is consistent with:

```text
64 serial prompt_len=1 first-token forwards + 127 batched decode steps
```

An accepted fix needs to make the first-token path batched while preserving the
C1 TP8 NCCL token trace. Replacing prompt prefill with decode is not sufficient;
see the rejected item below.

### O2 Prompt-Len-1 Single-Row Fast Prefill

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.json
```

Observed:

- P1 showed canonical bs64 output throughput `353.91 tok/s`, with TTFT p50/p99
  `4.58/10.24s` and TPOT p50/p95/p99 `146.15/172.83/175.10ms`.
- The in-process bs64/output128 probe showed steady TPOT p50 `107.76ms`, so the
  remaining service gap was dominated by first-token work and serving cadence,
  not only steady decode.
- Code inspection confirmed `64` serial `prefill_request` calls before batched
  decode. For `prompt_len=1`, causal MLA attention has exactly one key, so the
  attention output should equal the V slice produced by `kv_b_proj`.

Motivation / expected gain:

Avoid the Q branch, temporary K/V cache assembly, and FlashInfer single-prefill
call for each one-token prompt. The change keeps the original prefill semantics:
embedding and residual all-reduces remain BF16 NCCL, KV is still appended at
position 0, and TP8 prompt MoE remains the NCCL path. Expected gain is lower
TTFT and modest service throughput improvement while preserving the C1 token
trace.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_c1fast_mb1_o5.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

Result:

- TP8 NCCL fast path:
  `/tmp/kimi_nccl_tp8_c1fast_mb1_o5.json`, TTFT p50/p99
  `4.71/6.32s`, e2e p50 `6.92s`, steady TPOT p50 `97.81ms`.
- TP8 PPLX fast path:
  `/tmp/kimi_pplx_tp8_c1fast_mb1_o5.json`, TTFT p50/p99
  `5.27/7.35s`, e2e p50 `8.07s`, steady TPOT p50 `110.45ms`.
- Both files match `/tmp/kimi_nccl_tp8_active64_o5_final.json` exactly:
  0 per-index mismatches and hash counter `32x 7c4c5d83355198fd`,
  `32x 9eecc1ca6fb3409d`.

Correctness gate:

```bash
uv run --no-project python - <<'PY'
import collections, json, subprocess
old=json.loads(subprocess.check_output(
    ['ssh','h20-100','cat','/tmp/kimi_nccl_tp8_active64_o5_final.json']))
new=json.loads(subprocess.check_output(
    ['ssh','h20-100','cat','/tmp/kimi_pplx_tp8_c1fast_mb1_o5.json']))
mis=[i for i,(a,b) in enumerate(zip(
    old['metrics']['generated_token_traces'],
    new['metrics']['generated_token_traces'])) if a['prefix'] != b['prefix']]
print(collections.Counter(t['hash'] for t in new['metrics']['generated_token_traces']))
print('mismatches', len(mis))
PY
```

Observed output: `Counter({'7c4c5d83355198fd': 32, '9eecc1ca6fb3409d': 32})`,
`mismatches 0`.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fastmb1_candidate.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fastmb1_candidate.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `414.28 tok/s` vs P1 `353.91 tok/s`.
- Peak output throughput: `704.00 tok/s` vs P1 `640.00 tok/s`.
- TTFT p50/p95/p99: `2.76/6.26/6.90s`.
- TPOT p50/p95/p99: `133.36/147.74/149.42ms`.
- ITL p50/p95/p99: `117.15/120.31/126.00ms`.

Decision:

Keep. The change preserves the TP8 NCCL/PPLX correctness baseline and improves
canonical bs64 output throughput by about `17%`. It does not reach vLLM
`583.9 tok/s`; the next optimization should either make prompt_len=1 batch>1
prefill trace-exact, or reduce the PPLX steady TPOT gap.

### O3 Prompt-Len-1 Scratch Reuse And Admission Coalesce

Profile:

```text
/tmp/kimi_pplx_tp8_o2_micro_bs64_o128_warm1.json
/tmp/kimi_pplx_tp8_o3_scratch_micro_bs64_o128_warm1.json
/tmp/kimi_pplx_tp8_o3_scratch_coalesce_micro_bs64_o128_warm1.json
```

Observed:

- O2 in-process bs64/output128 was `458.5 tok/s` by `64 * 128 / max_e2e`,
  with TTFT p50/p99 `2173.49/4127.33ms`, first-decode p50/p99
  `2198.80/4171.37ms`, steady TPOT p50 `107.47ms`.
- Code profile showed that the accepted prompt_len=1 path still allocated
  dense MLP, shared expert, router, Marlin route/workspace, Marlin outputs, and
  routed F32 buffers per MoE layer and per request.
- The first scratch-only probe improved the first wave but split bs64 admission
  into `40 + 24` requests:
  `/tmp/kimi_pplx_tp8_o3_scratch_micro_bs64_o128_warm1.json` had
  `max_e2e=25.594s`, about `320.1 tok/s`.
- The split wave showed the fixed `20ms` coalesce window was too short for this
  pressure shape. After widening it to `100ms`, the same in-process probe
  admitted all `64` requests in one wave.

Motivation / expected gain:

The prompt_len=1 path is still intentionally serial at microbatch `1` because
batch>1 trace parity is not proven. Reusing the existing decode arena scratch
removes repeated GPU allocations without changing the math boundary: BF16 TP
all-reduces stay BF16, routed MoE all-reduce stays F32, Marlin uses the same
block size `8` as `kimi_marlin_block_size(1)`, and token trace remains gated
against TP8 NCCL. The coalesce change trades up to `80ms` extra admission wait
for avoiding a second full decode wave, which is worth seconds at bs64.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep \
  --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_o3_scratch_coalesce_micro_bs64_o128_warm1.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Result:

- Output path:
  `/tmp/kimi_pplx_tp8_o3_scratch_coalesce_micro_bs64_o128_warm1.json`.
- In-process wall throughput: `557.70 tok/s`
  (`64 * 128 / 14.688772613s`).
- TTFT p50/p95/p99: `505.53/927.55/957.74ms`.
- First-decode p50/p95/p99: `597.01/1022.10/1052.56ms`.
- Steady TPOT p50/p95/p99: `107.81/109.32/110.64ms`.
- E2E p50/p95/p99/max: `14.686/14.689/14.689/14.689s`.

Correctness gate:

```text
/tmp/kimi_pplx_tp8_o3_scratch_coalesce_o5.json
/tmp/kimi_nccl_tp8_active64_o5_final.json
```

Observed:

- Per-index generated-token trace mismatches: `0/64`.
- Hash counter on both files: `32x 7c4c5d83355198fd`,
  `32x 9eecc1ca6fb3409d`.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o3_scratch_coalesce_candidate.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o3_scratch_coalesce_candidate.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `492.34 tok/s` vs O2 `414.28 tok/s`.
- Peak output throughput: `592.00 tok/s`.
- TTFT p50/p95/p99: `0.67/3.80/3.96s`.
- TPOT p50/p95/p99: `121.05/124.99/125.58ms`.
- ITL p50/p95/p99: `116.64/120.13/124.76ms`.

Decision:

Keep. O3 improves canonical bs64 output throughput by about `18.8%` over O2
while preserving the TP8 NCCL/PPLX token trace gate. Revert this change if the
canonical bs64 output throughput falls below O2's `414.28 tok/s`, if bs64
admission again splits under the documented pressure command, or if the TP8
NCCL/PPLX short-trace gate shows any mismatch.

### O5 PPLX Decode Router Overlap

Profile:

```text
/tmp/kimi_pplx_tp8_o3_scratch_coalesce_micro_bs64_o128_warm1.json
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o3_scratch_coalesce_candidate.json
```

Observed:

- O3 in-process bs64/output128 steady TPOT p50/p95/p99 was
  `107.81/109.32/110.64ms`, while canonical service TPOT p50/p95/p99 was
  `121.05/124.99/125.58ms`.
- Code inspection showed the NCCL decode MoE path records `norm_ready`
  immediately after RMSNorm, then runs shared expert on the main stream while
  router/routed work proceeds on the aux stream.
- The PPLX decode MoE path recorded `norm_ready` only after shared expert and
  its TP all-reduce, so the aux-stream router could not overlap the shared
  expert window.

Motivation / expected gain:

Router uses only the post-attention normed hidden state, independent of shared
expert output. Starting the PPLX router right after RMSNorm preserves the same
math and stream dependency boundary as the NCCL decode path, while exposing
more overlap before `dispatch_send`. Expected gain: a few milliseconds per
steady bs64 decode step, with no token-trace change.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep \
  --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_o5_router_overlap_micro_bs64_o128_warm1.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Result:

- Output path:
  `/tmp/kimi_pplx_tp8_o5_router_overlap_micro_bs64_o128_warm1.json`.
- In-process wall throughput: `582.9 tok/s`
  (`64 * 128 / 14.054035966s`).
- TTFT p50/p95/p99: `504.81/925.43/955.54ms`.
- First-decode p50/p95/p99: `590.84/1015.02/1045.38ms`.
- Steady TPOT p50/p95/p99: `102.84/104.09/105.48ms`.
- This improves O3 in-process steady TPOT p50 from `107.81ms` to
  `102.84ms`.

Correctness gate:

```text
/tmp/kimi_pplx_tp8_o5_router_overlap_o5.json
/tmp/kimi_nccl_tp8_active64_o5_final.json
```

Observed:

- Per-index generated-token trace mismatches: `0/64`.
- Hash counter on both files: `32x 7c4c5d83355198fd`,
  `32x 9eecc1ca6fb3409d`.
- Short-probe steady TPOT p50: `105.53ms`.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o5_router_overlap_candidate.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o5_router_overlap_candidate.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `509.89 tok/s` vs O3 `492.34 tok/s`.
- Peak output throughput: `708.00 tok/s`.
- TTFT p50/p95/p99: `0.67/3.80/3.95s`.
- TPOT p50/p95/p99: `116.53/120.45/121.44ms`.
- ITL p50/p95/p99: `112.12/115.18/119.38ms`.

Decision:

Keep. O5 preserves the TP8 NCCL/PPLX token trace gate and improves canonical
bs64 output throughput by about `3.6%` over O3. Revert this change if the
canonical bs64 output throughput falls below O3's `492.34 tok/s`, if the
in-process steady TPOT p50 regresses above O3's `107.81ms`, or if the TP8
NCCL/PPLX short-trace gate shows any mismatch.

### R1 Rejected Full Routed Aux-Stream PPLX Decode

Profile:

```text
/tmp/kimi-profile/f779a66/nsys_o5_bs64_o128/inproc_bs64_o128.sqlite
/tmp/kimi-profile/f779a66/nsys_o5_bs64_o128/cuda_gpu_kern_sum.txt
/tmp/kimi-profile/f779a66/nsys_o5_bs64_o128/tail.md
/tmp/kimi-profile/f779a66/pplx_a2a_kimi_tok64.log
```

Observed:

- O5 nsys shows PPLX decode still spends time after the router in dispatch,
  local routed expert work, compact scatter, combine send, and combine recv.
- `pplx_a2a_bench` at Kimi tok64 shape reports dispatch+combine p50 around
  `151.58us` per rank, with `a2a_dispatch_send_kernel` p50 `73.31us` and
  `a2a_combine_recv_kernel` p50 `121.09us` visible in the O5 nsys report.

Motivation / expected gain:

Move the full routed expert/PPLX decode path to the aux stream after the router,
leaving shared expert and TP all-reduce on the main stream. The change is math
equivalent because both paths consume the same RMSNorm output and only join
before final residual add. Expected direction: reduce steady TPOT if PPLX
communication/local routed work can overlap with shared expert and all-reduce.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep \
  --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_o6_aux_routed_micro_bs64_o128_warm1.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Result:

- Output path:
  `/tmp/kimi_pplx_tp8_o6_aux_routed_micro_bs64_o128_warm1.json`.
- In-process wall throughput: `580.65 tok/s`
  (`64 * 128 / 14.108380907s`), below O5 `582.89 tok/s`.
- TTFT p50/p95/p99: `505.43/927.11/957.29ms`, slightly worse than O5
  `504.81/925.43/955.54ms`.
- First-decode p50/p95/p99: `592.74/1017.34/1047.71ms`, worse than O5
  `590.84/1015.02/1045.38ms`.
- Steady TPOT p50/p95/p99: `103.21/104.60/106.05ms`, worse than O5
  `102.84/104.09/105.48ms`.

Correctness gate:

```text
/tmp/kimi_pplx_tp8_o6_aux_routed_short.json
/tmp/kimi_nccl_tp8_active64_o5_final.json
```

Observed:

- Per-index generated-token trace mismatches: `0/64`.
- Hash counter on both files: `32x 7c4c5d83355198fd`,
  `32x 9eecc1ca6fb3409d`.
- Short-probe steady TPOT p50: `105.91ms`.

Performance gate:

Not run as service pressure. The supporting in-process probe regressed versus
O5, so the candidate was stopped before the canonical service command.

Decision:

Reject and revert. The idea is correctness-preserving, but the measured bs64
microbench says the extra aux-stream boundary and concurrent PPLX/main-stream
competition cost more than the overlap buys. Keep O5's router-only overlap and
focus next on removing PPLX dispatch/copy work or compact scatter rather than
moving the whole routed path to a separate stream.

### O7 PPLX Decode Counts-Only Dispatch Recv

Profile:

```text
/tmp/kimi-profile/0ba76a6-counts-recv/pplx_a2a_normal_tok64.log
/tmp/kimi-profile/0ba76a6-counts-recv/pplx_a2a_counts_only_tok64.log
/tmp/kimi-profile/0ba76a6-counts-recv-v2/pplx_a2a_normal_tok64_canon.log
/tmp/kimi-profile/0ba76a6-counts-recv-v2/pplx_a2a_counts_only_tok64_canon.log
/tmp/kimi-profile/f779a66/nsys_o5_bs64_o128/cuda_gpu_kern_sum.txt
```

Observed:

- O5 nsys and `pplx_a2a_bench` both show decode still spends time in
  `dispatch_recv` even though the TP8 correctness path computes local experts
  from NCCL-layout `scratch.mla.normed`, not from `pplx_recv_hidden`.
- The normal Kimi tok64 A2A probe measured `dispatch_recv_us` p50/p95/p99
  `22.14/29.31/35.39us` and `max_rank_split_us` p50/p95/p99
  `163.0/174.7/181.2us`.
- The final v2 A2A probe uses `canonicalize_duplicate_sources=true`, matching
  the Kimi TP8 production PPLX bootstrap.

Motivation / expected gain:

For `comm.is_some()` TP8 decode, PPLX dispatch is still needed to build and
exchange routing metadata (`num_routed`, worker offsets, `padded_index`,
`combine_send_offset`) and to keep the protocol state machine intact. The
received hidden payload and dispatch weight buffer are not consumed on this
path. A counts-only `dispatch_recv` should preserve protocol flags and local
expert counts while skipping the unused hidden copy. Expected gain: about
`10us` on the A2A split and a sub-millisecond but measurable bs64 TPOT win.

Microbench:

This is an isolated A2A timing shape with `max_private_tokens=64`, not the full
production bootstrap sizing (`max_num_tokens=2048`, derived private capacity).
It isolates the copied payload cost; protocol equivalence is accepted by the
TP8 NCCL/PPLX token-trace gate below.

Normal:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
target/release/pplx_a2a_bench \
  --n-experts 384 --topk 8 --hidden-dim 7168 --world-size 8 \
  --max-num-tokens 64 --expert-padding 8 --warmup 20 --repeats 100 \
  --canonicalize-duplicate-sources
```

Counts-only:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
target/release/pplx_a2a_bench \
  --n-experts 384 --topk 8 --hidden-dim 7168 --world-size 8 \
  --max-num-tokens 64 --expert-padding 8 --warmup 20 --repeats 100 \
  --canonicalize-duplicate-sources \
  --dispatch-recv-counts-only
```

Result:

- Normal `dispatch_recv_us` p50/p95/p99: `22.14/29.50/43.42us`.
- Counts-only `dispatch_recv_us` p50/p95/p99: `11.07/18.14/24.29us`.
- Normal `split_sum_us` p50/p95/p99: `154.11/171.81/190.34us`.
- Counts-only `split_sum_us` p50/p95/p99: `141.92/159.07/167.04us`.
- Normal `max_rank_split_us` p50/p95/p99: `164.7/181.7/201.2us`.
- Counts-only `max_rank_split_us` p50/p95/p99: `151.0/167.0/177.1us`.

Correctness gate:

```text
/tmp/kimi_pplx_tp8_counts_recv_short_v2.json
/tmp/kimi_nccl_tp8_active64_o5_final.json
```

Observed:

- Per-index generated-token trace mismatches: `0/64`.
- Hash counter on both files: `32x 7c4c5d83355198fd`,
  `32x 9eecc1ca6fb3409d`.
- Short-probe steady TPOT p50: `104.19ms`.

Performance gate:

Supporting in-process probe:

```text
/tmp/kimi_pplx_tp8_counts_recv_micro_bs64_o128_warm1_v2.json
```

Observed:

- In-process wall throughput: `589.98 tok/s`
  (`64 * 128 / 13.885239306s`) vs O5 `582.89 tok/s`.
- TTFT p50/p95/p99: `501.68/920.89/950.53ms`.
- First-decode p50/p95/p99: `586.66/1009.15/1039.25ms`.
- Steady TPOT p50/p95/p99: `101.58/102.58/103.92ms` vs O5
  `102.84/104.09/105.48ms`.

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_counts_recv_v2.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_counts_recv_v2.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `511.78 tok/s` vs O5 `509.89 tok/s`.
- Peak output throughput: `706.00 tok/s`.
- TTFT p50/p95/p99: `0.67/3.92/4.07s`.
- TPOT p50/p95/p99: `115.83/120.26/121.26ms`.
- ITL p50/p95/p99: `111.24/115.38/118.36ms`.

Decision:

Keep. The optimization is TP8-specific, uses the same NVLink sync-slot
orientation as the full `dispatch_recv`, and has positive A2A, in-process, and
canonical service deltas without changing token traces.
Revert this change if canonical bs64 output falls below O5's `509.89 tok/s`, if
in-process steady TPOT p50 regresses above O5's `102.84ms`, or if the TP8
NCCL/PPLX short-trace gate shows any mismatch.

### O8 Exact Attention Residual + Post-Attention RMSNorm Fusion

Profile:

```text
/tmp/kimi_fusion_model_report_static_bs64_kv128_o7_baseline.json
/tmp/kimi_fused_addrms_round_model_report_static_bs64_kv128.json
```

Observed:

- O7 static bs64/kv128 local compute report had `61` attention residual
  `add_batch` calls and `61` following post-attention `rms_norm_batch` calls.
- The targeted separate slice was about `1002us` per rank/step:
  attention residual add `504.35us`, dense/MoE post-attention norm
  `497.64us`.
- R4 proved the existing FlashInfer fused adapter cannot be used directly:
  it lets the FP32 add value participate in RMSNorm and changed `32/64` short
  token traces.

Motivation / expected gain:

The current decode loop always performs:

```text
projected all-reduce -> hidden = bf16(hidden + projected)
post-attention normed = rms_norm(hidden)
```

The fusion should remove one kernel launch and one global-memory read per
layer. To preserve correctness, the fused kernel first materializes the
BF16-rounded residual sum and then computes RMSNorm from that rounded value,
matching the separate `add_batch` then `rms_norm_batch` boundary.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
/root/.cargo/bin/cargo run --release -p pegainfer-kimi-k2 \
  --features kernel-report,pplx-ep --bin kimi_model_report -- \
  decode --source static --batch-size 64 --kv-len 128 --iters 32 \
  --format json --out /tmp/kimi_fused_addrms_round_model_report_static_bs64_kv128.json
```

Result:

- Schedule calls: `1765` -> `1704`.
- `fused_add_rms_norm_round_batch`: `61` calls, total `533.57us`,
  per-call `8.75us`.
- Removed targeted separate calls: about `1001.99us`; measured local-compute
  saving is about `468us` per rank/step.
- Total measured local compute: `262.005ms` -> `261.310ms`.

Correctness gate:

```text
/tmp/kimi_pplx_tp8_fused_addrms_round_bs64_o128.json
/tmp/kimi_pplx_tp8_counts_recv_micro_bs64_o128_warm1_v2.json
```

Observed:

- Strong bs64/output128 trace mismatches versus the O7 baseline: `0/64`.
- Hash counter on both files: `32x 82a791616c737442`,
  `16x 4ae8834e96c7d195`, `16x 24b2b3856ac0ea3a`.
- Candidate in-process steady TPOT p50/p95/p99:
  `101.57/102.86/103.95ms`.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fused_addrms_round.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fused_addrms_round.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `516.44 tok/s` vs O7 `511.78 tok/s`.
- Peak output throughput: `640.00 tok/s`.
- TTFT p50/p95/p99: `0.66/3.81/3.97s`.
- TPOT p50/p95/p99: `114.92/118.95/119.57ms`.
- ITL p50/p95/p99: `110.76/114.13/118.88ms`.

Decision:

Keep. This accepts the same optimization target as R4 only after preserving the
BF16 residual-sum boundary and passing the strong output128 token-trace gate.
Revert this change if output128 trace parity fails, or if canonical bs64 output
falls below O7's `511.78 tok/s`.

### O9 Prompt-Len1 Per-Token Microbatch 2

Profile:

```text
/tmp/kimi_pplx_tp8_prompt1_per_token_mb1_bs64_o5_probe.json
/tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb2_bs64_o5_probe.json
/tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb64_bs64_o5_probe.json
```

Observed:

- O8 still admitted bs64 prompt_len=1 as one scheduler wave, but
  `KIMI_PROMPT_LEN1_PREFILL_MICROBATCH=1` forced the first-token path through
  64 worker forwards.
- Row-wise per-token math at microbatch `1` preserved the short trace, with
  TTFT p50 `3582.56ms`, first decode p50 `797.43ms`, steady TPOT p50
  `104.19ms`.
- The same row-wise math at microbatch `2` also preserved the short trace, with
  TTFT p50 `3358.80ms`, first decode p50 `659.31ms`, steady TPOT p50
  `104.03ms`.
- Microbatch `64` reduced first decode p50 to `306.47ms`, but changed `48/64`
  short token traces, so it is rejected.

Motivation / expected gain:

A one-token prompt should share more of the decode-shaped batch path: KV append,
router, shared expert, routed expert, final norm, and sampling all operate on
one row per request. Earlier attempts failed because regular batched cuBLAS and
bulk collectives changed the row boundary. This version exposes `per_token`
helpers that keep each row's decode GEMM/router/all-reduce boundary while still
letting the scheduler group two active prompt rows.

Expected gain is modest: less first-token scheduling/worker overhead and a
shorter first-decode tail, without changing steady decode kernels.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/.cargo/bin/cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb2_bs64_o128_probe.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Result:

- In-process bs64/output128 TTFT p50/p95/p99:
  `377.73/681.77/703.48ms`.
- First decode p50/p95/p99: `455.65/763.08/785.15ms`.
- Steady TPOT p50/p95/p99: `101.68/102.82/103.99ms`.

Correctness gate:

```text
/tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb2_bs64_o5_probe.json
/tmp/kimi_nccl_tp8_active64_o5_final.json
/tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb2_bs64_o128_probe.json
/tmp/kimi_pplx_tp8_fused_addrms_round_bs64_o128.json
/tmp/kimi_pplx_tp8_counts_recv_micro_bs64_o128_warm1_v2.json
```

Observed:

- Short output5 trace versus TP8 NCCL: `0/64` mismatches; hash counter
  `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d`.
- Strong output128 trace versus O8 and O7: `0/64` mismatches; hash counter
  `32x 82a791616c737442`, `16x 4ae8834e96c7d195`,
  `16x 24b2b3856ac0ea3a`.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_per_token_mb2_candidate.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_per_token_mb2_candidate.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `523.88 tok/s` vs O8 `516.44 tok/s`.
- Peak output throughput: `656.00 tok/s`.
- TTFT p50/p95/p99: `0.50/3.75/3.86s`.
- TPOT p50/p95/p99: `113.88/117.51/118.37ms`.
- ITL p50/p95/p99: `110.55/113.76/118.04ms`.

Decision:

Keep. The public helper names use `per_token` instead of exposing the rejected
`n1`/`strided_batched` implementation detail, and the scheduler only raises the
prompt_len=1 microbatch to the largest trace-exact value currently proven.
Revert this change if output128 parity fails, if canonical bs64 output falls
below O8's `516.44 tok/s`, or if a future mb4+ parity probe shows that this
row-wise boundary is hiding a correctness issue.

### O10 Warmed Prompt-Len1 Microbatch 64

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_per_token_mb2_candidate.json
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb8_candidate.json
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb64_candidate.json
/tmp/kimi_pplx_tp8_prompt1_warm_mb8_bs64_o128_probe.json
/tmp/kimi_pplx_tp8_prompt1_warm_mb64_bs64_o5_probe.json
/tmp/kimi_pplx_tp8_prompt1_warm_mb64_bs64_o128_probe.json
```

Observed:

- O9's canonical bs64 result still had TTFT p95/p99 `3.75/3.86s`; detailed
  `ttfts` showed the first 64-request service wave at `3.2-3.9s`, while later
  waves were in the sub-second range. The remaining large tail was cold
  bs64/prompt_len1 first use, not the benchmark's frontend accounting.
- Allocating only the decode arena before serving was not enough:
  `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_mb2_warm_arena_candidate.json`
  reported output `522.16 tok/s`, TTFT p50/p95/p99
  `503/3342/3453ms`, TPOT p50/p95/p99 `115.06/118.10/118.80ms`.
- Warming the prompt_len1 bs64 path before the service is marked ready
  removed the first-wave cold tail. With mb8, canonical service reached output
  `546.50 tok/s`, TTFT p50/p95/p99 `392.95/701.16/732.46ms`, TPOT
  p50/p95/p99 `114.95/117.78/118.22ms`.
- The mb8 service still showed per-wave TTFT spread: the four 64-request
  chunks had medians `406.52/383.20/381.30/383.58ms` but max
  `752.86/723.44/722.95/723.41ms`, consistent with prompt_len1
  stair-stepping inside the wave.

Motivation / expected gain:

For the `prompt_len=1` workload, the first token is decode-shaped: each request
has one active row, one KV append, one routed MoE decision, one sampled token,
and then immediately enters regular decode. Allocating the bs64 decode arena
and running prompt_len1 in the service warmup avoids charging one-time
CUDA/PPLX/kernel setup to TTFT. The warmup calls the real prompt_len1 path on
slots `0..63`, writing position-0 KV; the serving scheduler also writes
position 0 for real prompt_len1/prefill requests before any decode append, so
the warm token is overwritten on this path.

Raising the hard-coded prompt_len1 microbatch from `2` to `64` trades exact
trace parity for tighter service TTFT tail and a much lower first-decode step:
mb64 improves service TTFT p95/p99 over mb8, while mb8 keeps a better TTFT p50
and remains the exact-token reference.

Expected gain was lower TTFT tail and service throughput closer to vLLM's
`583.9 tok/s` baseline. The risk is the recorded large-batch trace drift, so
this entry is kept as a drift-recorded performance baseline rather than an
exact-token correctness baseline.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
/root/develop/xingming/pegainfer/target/release/bench_serving \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_prompt1_warm_mb64_bs64_o128_probe.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Sweep result:

- mb4: `0/64` short mismatches, TTFT p50 `323.62ms`, first decode p50
  `612.89ms`, steady TPOT p50 `104.06ms`.
- mb8: `0/64` short mismatches and `0/64` output128 mismatches versus O8/O7,
  TTFT p50 `318.15ms`, first decode p50 `354.89ms`, steady TPOT p50
  `101.65ms`.
- mb16: `64/64` short mismatches, TTFT p50 `361.52ms`, first decode p50
  `544.75ms`, steady TPOT p50 `106.64ms`.
- mb32: `54/64` short mismatches, TTFT p50 `459.60ms`, first decode p50
  `533.74ms`, steady TPOT p50 `106.58ms`.
- mb64 final candidate: `48/64` short mismatches, output128 `64/64`
  mismatches versus O8/O7; TTFT p50/p95/p99 `456.50/459.28/459.52ms`, first
  decode p50/p95/p99 `104.61/104.63/104.63ms`, steady TPOT p50/p95/p99
  `104.01/104.72/105.47ms`.

Correctness / drift record:

```text
/tmp/kimi_pplx_tp8_prompt1_warm_mb8_bs64_o128_probe.json
/tmp/kimi_pplx_tp8_prompt1_warm_mb64_bs64_o5_probe.json
/tmp/kimi_pplx_tp8_prompt1_warm_mb64_bs64_o128_probe.json
/tmp/kimi_nccl_tp8_active64_o5_final.json
/tmp/kimi_pplx_tp8_fused_addrms_round_bs64_o128.json
/tmp/kimi_pplx_tp8_counts_recv_micro_bs64_o128_warm1_v2.json
```

Observed:

- mb8 exact reference point: output128 `0/64` mismatches versus O8 and O7;
  hash counter `32x 82a791616c737442`, `16x 4ae8834e96c7d195`,
  `16x 24b2b3856ac0ea3a`.
- mb64 short drift versus TP8 NCCL: `48/64` mismatches; hash counter
  `16x a1c5655be80ec3b6`, `16x 7c4c5d83355198fd`,
  `8x 38fec438cee33079`, `8x ace6ba1ebbd24e18`,
  `8x 82dd9581ecb6c1e4`, `8x 9eecc1ca6fb3409d`.
- mb64 output128 drift versus O8 and O7: `64/64` mismatches; hash counter
  `8x 10752d4e5eaa4735`, `8x cd003a7698315eb7`,
  `8x 0406ab8af0aa7077`, `8x 33e0619511d08262`,
  `8x dd0bcd04bc691b1d`, `8x 26f87f078f3baef6`,
  `8x e4c3e9cd67571817`, `8x 92ba3a6008989001`.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb64_candidate.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_prompt1_warm_forward_mb64_candidate.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `557.71 tok/s` vs O9 `523.88 tok/s`, mb8 warm-forward
  `546.50 tok/s`, vLLM `583.9 tok/s`.
- Peak output throughput: `640.00 tok/s`.
- TTFT p50/p95/p99: `473.94/504.25/506.18ms`.
- TPOT p50/p95/p99: `110.95/115.30/115.34ms`.
- ITL p50/p95/p99: `111.03/115.40/117.89ms`.

Decision:

Keep as the current drift-recorded service-performance baseline with the
mismatch counts recorded. This is not an exact-token baseline: mb8 remains the
exact reference point for this path, while mb64 is the faster serving candidate
accepted under the current large-batch drift policy. Revert to mb8 if exact
trace parity becomes mandatory again, or revert below O9 if canonical bs64
output falls under `523.88 tok/s`.

## R4: Fused Attention Residual + Post-Attention RMSNorm

### Profile

The O7 bs64 nsys attempt at
`/tmp/kimi-profile/4131e17-compute-o7/nsys_bs64_o128/` exceeded its `15m`
timeout and was stopped without a usable `.nsys-rep`. The fallback static
operator report is:

```text
/tmp/kimi_fusion_model_report_static_bs64_kv128_o7_baseline.json
```

Baseline static bs64/kv128 local-compute report:

- `add_batch`: `122` calls, total `1008.696us`, per call `8.268us`.
- `rms_norm_batch`: `245` calls, total `1923.680us`, per call `7.852us`.
- The fusion candidate only targets the attention residual add and the next
  post-attention RMSNorm, so the removable slice is `61` add calls plus `61`
  RMSNorm calls, about `0.98ms` per rank in this synthetic report.

### Motivation / Expected Gain

Each decode layer currently materializes `hidden + attention_out` and then
immediately RMS-normalizes that value before dense/shared/router compute. The
existing FlashInfer-backed `fused_add_rms_norm` kernel can express this data
edge as `hidden += residual; normed = rms_norm(hidden)`, potentially removing
one launch and one BF16 write/read pair per layer. Expected gain was small but
real: below `1ms` per bs64 decode step before collective and graph effects.

### Microbench

Candidate static report:

```text
/tmp/kimi_fusion_model_report_static_bs64_kv128_fused_addrms_candidate.json
```

Measured local-compute delta:

- `add_batch`: `61` calls, total `494.466us`.
- `rms_norm_batch`: `184` calls, total `1446.646us`.
- `fused_add_rms_norm_batch`: `61` calls, total `793.854us`, per call
  `13.014us`.
- The measured local-compute slice changed from about `2932us` to about
  `2735us`, only `~0.20ms` per rank. The fused kernel's device-to-device copy
  absorbs most of the expected launch/data-movement win.

### Correctness

Short TP8 PPLX probe:

```text
/tmp/kimi_pplx_tp8_fused_addrms_short.json
/tmp/kimi_nccl_tp8_active64_o5_final.json
```

Observed comparison output:

```text
old Counter({'7c4c5d83355198fd': 32, '9eecc1ca6fb3409d': 32})
new Counter({'7c4c5d83355198fd': 64})
mismatches 32
first_mismatch [16, 17, 18, 19, 20]
```

### Decision

Reject and revert. The candidate is not token-trace preserving, likely because
FlashInfer fused add+RMSNorm does not match the current `add_batch` BF16
materialization followed by RMSNorm exactly enough for the TP8 baseline. It also
only showed a `~0.20ms` local-compute microbench gain, so writing a bespoke
bit-exact fused kernel is not justified before higher-share compute operators
such as router and Marlin are addressed.

## R5: Router GEMM cuBLAS Compute Mode

### Profile

Static bs64/kv128 model report identified the largest local-compute call sites:

```text
/tmp/kimi_fusion_model_report_static_bs64_kv128_o7_baseline.json
```

Top call sites:

- `layer.*.moe.marlin_w13`: `127.33ms`, `48.60%`.
- `layer.*.moe.marlin_w2`: `67.98ms`, `25.94%`.
- `layer.*.moe.router`: `45.84ms`, `17.50%`, per call `764.052us`.

A short static nsys report split the router body:

```text
/tmp/kimi-profile/26ec1e7-compute-static-report/static_bs64_kv128.nsys-rep
/tmp/kimi-profile/26ec1e7-compute-static-report/cuda_gpu_kern_sum_cuda_gpu_kern_sum.txt
```

Router internals from the nsys kernel summary:

- Router GEMM (`magma_sgemmEx_kernel`): `7` instances, avg `762.32us`.
- `router_topk_normalize_kernel`: `7` instances, avg `8.35us`.
- `router_scores_kernel`: `7` instances, avg `1.98us`.

### Motivation / Expected Gain

Router post-processing is not the bottleneck; the BF16 router GEMM dominates.
The current code uses `CUBLAS_COMPUTE_32F_PEDANTIC`. Two temporary candidates
were measured:

- `CUBLAS_COMPUTE_32F_FAST_16BF`
- `CUBLAS_COMPUTE_32F`

The expected benefit was large because replacing the pedantic GEMM path could
remove most of the `45.84ms` local-compute router slice.

### Microbench

Candidate static reports:

```text
/tmp/kimi_router_fast16bf_model_report_static_bs64_kv128.json
/tmp/kimi_router_compute32f_model_report_static_bs64_kv128.json
```

Measured results:

- Baseline `kimi_router_noaux_tc`: `60` calls, total `45843.120us`, per call
  `764.052us`.
- `FAST_16BF`: `60` calls, total `1785.480us`, per call `29.758us`.
- `COMPUTE_32F`: `60` calls, total `1817.280us`, per call `30.288us`.
- Static local-compute total moved from `262.005ms` to about `218.0ms`.

### Correctness / Performance Gates

Short output5 TP8 PPLX gate passed for `FAST_16BF`:

```text
/tmp/kimi_pplx_tp8_router_fast16bf_short.json
old Counter({'7c4c5d83355198fd': 32, '9eecc1ca6fb3409d': 32})
new Counter({'7c4c5d83355198fd': 32, '9eecc1ca6fb3409d': 32})
mismatches 0
```

The stronger output128 in-process gate failed against the O7 baseline:

```text
/tmp/kimi_pplx_tp8_counts_recv_micro_bs64_o128_warm1_v2.json
/tmp/kimi_router_fast16bf_micro_bs64_o128_warm1.json
/tmp/kimi_router_compute32f_micro_bs64_o128_warm1.json
```

`FAST_16BF` output128:

```text
base Counter({'82a791616c737442': 32, '4ae8834e96c7d195': 16, '24b2b3856ac0ea3a': 16})
new Counter({'4ae8834e96c7d195': 32, 'f8484b874a3f0572': 16, '82a791616c737442': 16})
mismatches 32
steady TPOT p50/p95/p99 43.996/44.869/46.312ms
```

`COMPUTE_32F` output128:

```text
base Counter({'82a791616c737442': 32, '4ae8834e96c7d195': 16, '24b2b3856ac0ea3a': 16})
new Counter({'4ae8834e96c7d195': 32, 'f8484b874a3f0572': 16, '82a791616c737442': 16})
mismatches 32
steady TPOT p50/p95/p99 44.266/45.709/48.028ms
```

### Decision

Reject and revert both compute-mode variants. The performance win is large, but
the long token trace changes after the short output5 gate. This is an important
profile result: router GEMM is a real bs64 bottleneck, but changing cuBLAS
compute mode is not precision-preserving under the current baseline. A future
router optimization needs either a bit-equivalent faster GEMM path or a stronger
model-level reference that explicitly accepts this numerical boundary change.

## Candidate Queue

| Priority | Area | Hypothesis | Correctness risk |
| --- | --- | --- | --- |
| P0 | PPLX / MoE / scheduler | Recover the remaining gap from O10 `557.71 tok/s` to vLLM `583.9 tok/s` through steady TPOT/ITL work rather than more prompt_len1 admission changes. First profile PPLX dispatch/combine, indexed compact scatter, TP all-reduce, and graph replay at the mb64 candidate. | High: O10 already records mb64 drift (`48/64` o5, `64/64` o128); further math or routing changes need their own mismatch count and hash distribution. |
| P0 | PPLX / MoE | O7 removed the unused TP8 `dispatch_recv` hidden copy, but `dispatch_send` still moves a full hidden payload only to build metadata. Prototype route-only dispatch send and measure `pplx_a2a_bench` / nsys before changing model code. | High: dispatch still builds `token_offset`, `expert_offsets`, `padded_index`, and `combine_send_offset`; compare these hashes plus token trace. |
| P0 | PPLX / MoE | TP8 PPLX scatters Marlin output into a compact PPLX buffer before `combine_send`. Add an indexed combine-send path that reads NCCL-layout rows through `routing.sorted_token_ids`, then verify that `kimi_scatter_marlin_routes_to_compact_kernel` disappears in nsys. | High: duplicate-source canonicalization and BF16 row order must remain trace-exact. |
| P1 | CUDA Graph | Reduce bs64 first-step graph capture/replay and metadata overhead after kernel profile identifies host or graph-node cost. | Medium: graph replay must preserve per-row metadata and PPLX participation. |
| P1 | frontend | Measure HTTP/streaming overhead separately from in-process TPOT. | Low for model math, medium for serving semantics. |
| P1 | collectives | Profile TP all-reduce and routed combine tail at bs64. | Medium: BF16/F32 collective boundary is correctness-sensitive. |
| P2 | MLA/MoE | Retune batch-shape kernels only after scheduler and graph bottlenecks are visible. | High: routed expert and MLA cache layout are easy to perturb. |

## Rejected / Deferred

| Date | Idea | Reason |
| --- | --- | --- |
| 2026-05-25 | Use TP1/DP8 correctness as the baseline for this doc | Deferred. TP1/DP8 matched short probes but diverged at 32 tokens, so DP1 TP8 work uses TP8 NCCL/PPLX baseline first. |
| 2026-05-25 | Use the batch decode kernel as the `prompt_len=1` first-token path | Rejected. New TP8 NCCL and PPLX matched each other (`/tmp/kimi_nccl_tp8_single_prefill_batch_o2_o5.json` vs `/tmp/kimi_pplx_tp8_single_prefill_batch_o2_o5.json`: 0 mismatches), but both changed `32/64` per-index traces compared with the C1 TP8 NCCL ground truth `/tmp/kimi_nccl_tp8_active64_o5_final.json`. Hash counter changed from `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` to `48x 9eecc1ca6fb3409d`, `16x f45b2f0248e7059d`; this is not correctness-preserving. |
| 2026-05-25 | Run the older exact prompt_len=1 fast prefill path with regular batched GEMM at microbatch `2` or larger | Rejected and superseded by O9's row-wise `per_token` variant. The full-batch probe `/tmp/kimi_nccl_tp8_c1batch_o5.json` produced `42/64` mismatches and hash counter `40x 7c4c5d83355198fd`, `18x f45b2f0248e7059d`, `6x 9eecc1ca6fb3409d`. A block-size-8 A/B still failed (`/tmp/kimi_nccl_tp8_c1batch_block8_o5.json`). The old scheduler microbatch=2 candidate `/tmp/kimi_nccl_tp8_c1micro2_o5.json` had `37/64` mismatches because it did not preserve the row-wise decode math boundary. |
| 2026-05-25 | Use strided-batched per-token GEMM/router logits, or widen the O9 row-wise prompt_len=1 path directly to microbatch `64` as an exact-token optimization | Rejected as an exact-token optimization. Strided-batched per-token GEMM changed the short trace (`/tmp/kimi_pplx_tp8_prompt1_n1batched_bs64_o5_probe.json`: `64/64` mismatches; `/tmp/kimi_pplx_tp8_prompt1_per_token_mb2_bs64_o5_probe.json`: `32/64` mismatches). Row-wise GEMM fixed microbatch `2`, but row-wise microbatch `64` changed `/tmp/kimi_pplx_tp8_prompt1_per_token_loop_mb64_bs64_o5_probe.json` by `48/64` traces. O10 later keeps mb64 only under the explicit large-batch drift record. |
| 2026-05-25 | Opportunistically coalesce multiple `EngineCoreOutputs` in `pegainfer-vllm-frontend` before msgpack/ZMQ send | Rejected after service pressure test. The protocol can carry many `EngineCoreOutput` values per message, and the candidate preserved request order/final outputs in unit tests, but the canonical bs64 service result regressed from O3 `492.34 tok/s` to `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o4_output_coalesce_candidate.json` output `487.70 tok/s`, TPOT p50/p95/p99 `122.29/126.70/127.57ms`. This indicates the remaining service gap is not dominated by one-msgpack-per-token-output framing. |
| 2026-05-25 | Move the full routed expert/PPLX decode path to the aux stream after router | Rejected after correctness and microbench. The candidate preserved TP8 NCCL/PPLX short-token trace (`/tmp/kimi_pplx_tp8_o6_aux_routed_short.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 mismatches), but regressed the bs64/o128 in-process probe from O5 `582.89 tok/s`, TPOT p50/p95/p99 `102.84/104.09/105.48ms` to `/tmp/kimi_pplx_tp8_o6_aux_routed_micro_bs64_o128_warm1.json` `580.65 tok/s`, TPOT p50/p95/p99 `103.21/104.60/106.05ms`; service pressure was skipped because the lower-level gate already lost. |
| 2026-05-25 | Fuse attention residual add with post-attention RMSNorm using the existing FlashInfer fused add+rmsnorm adapter | Rejected after microbench and correctness, not because the win was too small. Static bs64/kv128 operator reports changed the targeted local-compute slice from about `2932us` to `2735us`, only `~0.20ms` per rank, and `/tmp/kimi_pplx_tp8_fused_addrms_short.json` mismatched `/tmp/kimi_nccl_tp8_active64_o5_final.json` on `32/64` generated-token traces. The likely issue is different BF16 materialization/rounding than the current `add_batch` then `rms_norm_batch` sequence; O8 keeps an exact BF16-rounded variant instead. |
| 2026-05-25 | Change router GEMM from `CUBLAS_COMPUTE_32F_PEDANTIC` to `FAST_16BF` or `COMPUTE_32F` | Rejected after a stronger output128 correctness gate. Static microbench improved router from `764us` to about `30us` per MoE layer and in-process TPOT p50 improved to `~44ms`, but both variants changed `32/64` output128 token traces versus `/tmp/kimi_pplx_tp8_counts_recv_micro_bs64_o128_warm1_v2.json`. The short output5 gate was too weak to catch this drift. |
