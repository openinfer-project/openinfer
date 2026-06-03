# Kimi-K2 Profiles

> **TL;DR:** Kimi-K2 TP1/DP8/EP8 decode profiling evidence lives here: start with the master ledger, then the TP1 PPLX bench log, then per-kernel `*_report.md` files.
>
> **Last touched:** 2026-06

This directory owns the H20 TP1/DP8/EP8 decode profiling and optimization records:

| Entry | Purpose |
|---|---|
| `tp1-dp8-ep8-decode-optimization-master.md` | Master table covering every decode-path operator, roofline class, accepted/rejected optimization state, and next action. |
| `tp1-pplx-decode-bench.md` | Project log for the bench binaries, route replay, NCU collection, CUDA Tile probes, and measurement caveats. |
| `tp1-dp8-ep8-fusion-scan.md` | Phase 2 fusion scan record; current conclusion is no accepted fusion for the measured baseline. |
| `*_report.md` | Per-kernel reports with KernelWiki notes, NCU evidence, attempts, and final stop/adopt conclusion. |

Root-level Kimi docs keep model architecture, serving/performance ledgers, and correctness notes. Profile artifacts and kernel-specific evidence should be added here so the decode optimization trail stays in one place.
