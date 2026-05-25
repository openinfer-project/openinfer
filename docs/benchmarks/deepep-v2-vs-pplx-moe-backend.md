# DeepEP V2 vs PegaInfer PPLX EP on H20 x8

> **TL;DR** On an 8x H20 node, DeepEP V2 ElasticBuffer/NCCL Gin is clearly ahead of the current PegaInfer PPLX EP microbenchmark on the tested MoE exchange shapes. In the paired run here, the directional dispatch+combine ratio is about 2.5x to 5.3x; against the earlier PPLX snapshot, it is about 2.4x to 4.5x. This is a backend direction check, not a dtype-identical replacement gate.

Last touched: 2026-05-25

## Revisions

| Component | Revision |
| --- | --- |
| PegaInfer paired run | `f071baa` |
| PegaInfer historical PPLX snapshot | `ec514ef` |
| DeepEP | `723716f` |

## Hardware And Software

| Component | Value |
| --- | --- |
| GPU | 8x NVIDIA H20-3e |
| Driver | 575.57.08 |
| CUDA toolchain | 12.9 |
| DeepEP Python stack | PyTorch 2.10.0+cu128, NCCL 2.30.4 cu12 |

## Method

PegaInfer paired-run command:

```bash
cargo run -r -p pegainfer-comm --bin pplx_a2a_bench -- --sweep --warmup 20 --repeats 100
```

DeepEP V2 command template:

```bash
python tests/elastic/test_ep.py \
  --num-processes 8 \
  --skip-check \
  --test-first-only \
  --num-experts <experts> \
  --num-topk <topk> \
  --hidden <hidden> \
  --num-tokens <tokens>
```

Sweep inputs:

| Shape | Experts | Top-k | Hidden | Tokens per rank |
| --- | ---: | ---: | ---: | --- |
| DSV4 | 256 | 6 | 4096 | 1, 4, 8, 32, 128, 256 |
| Kimi-K2 | 384 | 8 | 7168 | 1, 4, 8, 32, 128, 256 |

DeepEP was run with `--test-first-only`, so the measured case is the first elastic EP case: copy enabled, expert alignment 128, FP8 dispatch enabled, BF16 combine, no previous event, synchronous path. Correctness checks were skipped with `--skip-check`; this run is latency-only.

PegaInfer reports event-timed `max_rank_split_sum_us` for the full dispatch_send -> dispatch_recv -> combine_send -> combine_recv cycle. DeepEP reports profiler averages for ordinary dispatch and ordinary combine. For comparison, this note takes the ordinary dispatch line and ordinary combine line, sums dispatch+combine by rank, and reports both the worst rank and the mean rank. Because these are not identical timing harnesses, all ratios below are directional.

## Results

| Config | PegaInfer paired p50 us | PegaInfer paired mean us | DeepEP V2 worst-rank sum us | DeepEP V2 mean-rank sum us | Directional ratio vs PegaInfer p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsv4/tok=1 | 87.5 | 91.0 | 23.815 | 23.632 | 3.7x |
| dsv4/tok=4 | 95.9 | 97.4 | 24.094 | 23.801 | 4.0x |
| dsv4/tok=8 | 100.3 | 101.1 | 24.232 | 23.895 | 4.1x |
| dsv4/tok=32 | 146.8 | 149.1 | 27.637 | 27.218 | 5.3x |
| dsv4/tok=128 | 139.9 | 141.2 | 44.639 | 44.432 | 3.1x |
| dsv4/tok=256 | 205.0 | 207.2 | 67.089 | 66.826 | 3.1x |
| kimi-k2/tok=1 | 95.0 | 96.1 | 24.613 | 24.403 | 3.9x |
| kimi-k2/tok=4 | 96.4 | 97.2 | 24.974 | 24.619 | 3.9x |
| kimi-k2/tok=8 | 99.9 | 101.3 | 25.173 | 24.992 | 4.0x |
| kimi-k2/tok=32 | 122.1 | 123.4 | 33.889 | 33.613 | 3.6x |
| kimi-k2/tok=128 | 188.0 | 190.0 | 74.572 | 74.409 | 2.5x |
| kimi-k2/tok=256 | 312.1 | 313.7 | 122.617 | 122.463 | 2.5x |

### DeepEP V2 Dispatch/Combine Split

| Config | Max dispatch us | Max combine us | Worst-rank sum us |
| --- | ---: | ---: | ---: |
| dsv4/tok=1 | 13.972 | 9.866 | 23.815 |
| dsv4/tok=4 | 14.086 | 10.013 | 24.094 |
| dsv4/tok=8 | 14.180 | 10.052 | 24.232 |
| dsv4/tok=32 | 14.886 | 12.751 | 27.637 |
| dsv4/tok=128 | 20.780 | 23.859 | 44.639 |
| dsv4/tok=256 | 28.311 | 38.905 | 67.089 |
| kimi-k2/tok=1 | 14.621 | 10.025 | 24.613 |
| kimi-k2/tok=4 | 14.891 | 10.083 | 24.974 |
| kimi-k2/tok=8 | 14.749 | 10.467 | 25.173 |
| kimi-k2/tok=32 | 17.272 | 16.684 | 33.889 |
| kimi-k2/tok=128 | 32.042 | 42.672 | 74.572 |
| kimi-k2/tok=256 | 50.735 | 71.921 | 122.617 |

## PegaInfer Baseline Drift

The table above uses the PegaInfer run taken in the same benchmarking session as the DeepEP run. The earlier PPLX benchmark snapshot in `docs/benchmarks/pplx-ep-a2a-h20-nvlink.md` was captured at `ec514ef`. Those two PegaInfer snapshots differ enough that the comparison should not pretend to be a precise speedup gate.

Positive delta means the paired run here is slower than the historical snapshot.

| Config | Historical p50 us | Paired-run p50 us | Delta |
| --- | ---: | ---: | ---: |
| dsv4/tok=1 | 79.9 | 87.5 | +10% |
| dsv4/tok=4 | 90.9 | 95.9 | +6% |
| dsv4/tok=8 | 109.0 | 100.3 | -8% |
| dsv4/tok=32 | 94.0 | 146.8 | +56% |
| dsv4/tok=128 | 128.1 | 139.9 | +9% |
| dsv4/tok=256 | 204.3 | 205.0 | +0% |
| kimi-k2/tok=1 | 82.2 | 95.0 | +16% |
| kimi-k2/tok=4 | 84.1 | 96.4 | +15% |
| kimi-k2/tok=8 | 88.1 | 99.9 | +13% |
| kimi-k2/tok=32 | 103.4 | 122.1 | +18% |
| kimi-k2/tok=128 | 181.2 | 188.0 | +4% |
| kimi-k2/tok=256 | 302.8 | 312.1 | +3% |

Using the historical PPLX p50s instead of the paired run gives a directional ratio range of about 2.4x to 4.5x. The largest baseline drift is DSV4 tok=32; that case alone pushes the paired-run ratio up to 5.3x.

## Interpretation Guardrails

- DeepEP V2 was measured through the elastic EP path: ElasticBuffer with the NCCL Gin backend. The repository still builds legacy NVSHMEM pieces, but this V2 path is the one relevant to the current comparison.
- The measured DeepEP V2 case uses FP8 dispatch and BF16 combine. PegaInfer PPLX currently benchmarks a BF16 payload. Treat the table as a backend signal, not an exact dtype-to-dtype gate.
- DeepEP correctness checks were skipped in this latency run. A replacement decision needs a correctness run in the integrated PegaInfer path.
- DeepEP `num_tokens` is a max-per-rank input; the test uses slightly different actual token counts across ranks. PegaInfer uses the fixed max token count per rank.
- DeepEP numbers are profiler kernel averages. PegaInfer numbers are CUDA event timings around the benchmark cycle. The delta is large enough to be actionable, but integration work should add one apples-to-apples harness before replacing backend policy.

## Read

DeepEP V2 is especially strong at low token counts: the tested DSV4 and Kimi-K2 shapes sit around 24-34 us for tok <= 32, while the paired PegaInfer PPLX path is roughly 96-147 us. At larger payloads, DeepEP still holds about a 2.5x to 3.1x directional advantage in the paired run.

The next useful gate is a strict integration benchmark with the same payload dtype, token distribution, correctness checks, and PegaInfer scheduler-facing API cost included.
