# Embedding Report

> **TL;DR:** `decode.embedding` is the TP1 vocab-sharded embedding lookup `embedding_batch_vocab_shard`, filling the fixed decode arena (`rows=8, hidden=7168`, BF16). H20 TP1 PPLX `bs=8,ctx=1` measures `6.83-7.24us` and only `31.7-33.6GB/s` payload-equivalent throughput. Source launch geometry is `224` CTAs x `256` threads for `57,344` elements; this row is a small lookup/control row, not an H20 bandwidth target. Stop standalone tuning.
>
> **Last touched:** 2026-06

## KernelWiki Conclusion

Relevant KernelWiki references:

| Page | Relevant conclusion | Application to this row |
|---|---|---|
| `wiki/patterns/memory-bound.md` (`pattern-memory-bound`) | Memory-bound diagnosis requires high measured DRAM throughput, not just low arithmetic intensity. | This row reports `<1%` H20 HBM by payload model, so calling it a bandwidth target would be misleading. |
| `wiki/patterns/low-sm-utilization.md` (`pattern-low-sm-utilization`) | Low utilization can come from insufficient useful work or scheduling overhead; profile before adding persistent scheduling. | The row is one small lookup launch per decode step. There is no evidence for a standalone scheduling rewrite. |

Practical conclusion: embedding lookup is path coverage, not an optimization target. The only meaningful future direction would be launch removal as part of a larger decode graph fusion, which is not currently plausible for the first row of the step.

## NCU Conclusion

Fresh production NCU is currently unavailable on `h20-100`:

```bash
timeout 20s ssh -o ConnectTimeout=5 h20-100 '/usr/local/cuda-12.9/bin/ncu --version'
# exits 124 with no output
```

Source geometry:

| Source fact | Value |
|---|---:|
| CUDA entry | `embedding_batched_vocab_shard_cuda` |
| Kernel | `embedding_batched_vocab_shard_kernel` in `pegainfer-kernels/csrc/elementwise.cu` |
| Elements | `rows * hidden = 8 * 7168 = 57344` |
| Launch | `224` CTAs x `256` threads |

No production NCU counters are claimed here. The event timing is already enough to stop standalone work because the row costs about `7us` total.

## Bench Evidence

| Artifact | Step latency | Payload GB/s | H20 HBM pct |
|---|---:|---:|---:|
| `tp1-pplx-decode-bench-h20-100.json` | `7.24us` | `31.70` | inferred `0.66%` |
| `tp1-pplx-decode-bench-o-proj-cublaslt-bs8.json` | `6.83us` | `33.58` | `0.70%` |

## Attempts

| Attempt | Result | Decision |
|---|---|---|
| Current vocab-sharded embedding lookup | `6.83-7.24us` per decode step. | Current baseline. |
| Standalone rewrite | Not attempted. The row is too small and does not show a bandwidth ceiling. | Stop standalone direction. |

## Final Conclusion

Keep the current `embedding_batch_vocab_shard` provider and classify `decode.embedding` as `control/lookup`. Reopen only if a larger graph-level change can remove the launch without changing TP1 vocab-shard semantics.
