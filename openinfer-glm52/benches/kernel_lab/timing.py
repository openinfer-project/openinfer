"""torch.cuda.Event timing: warmup + rounds x inner; median/p50/p99/mean.

The SM clock is RECORDED (via nvidia-smi), never enforced — locked clocks are
the bench owner's discipline. torch is imported lazily (bench path only).
"""
from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass


@dataclass
class BenchStats:
    samples_us: list[float]  # one per round, inner-averaged
    clocks_sm_mhz: int | None = None

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us)

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.samples_us)

    @property
    def p50_us(self) -> float:
        return self.percentile(50)

    @property
    def p99_us(self) -> float:
        return self.percentile(99)

    def percentile(self, pct: int) -> float:
        """Nearest-rank percentile (ceil), robust for small round counts."""
        s = sorted(self.samples_us)
        if not s:
            raise ValueError("no samples")
        rank = -(-pct * len(s) // 100)
        return s[min(rank, len(s)) - 1]

    def summary(self) -> dict:
        return {
            "median_us": round(self.median_us, 3),
            "p50_us": round(self.p50_us, 3),
            "p99_us": round(self.p99_us, 3),
            "mean_us": round(self.mean_us, 3),
            "rounds": len(self.samples_us),
        }


def _measure_once(call, inner: int) -> float:
    from kernel_lab.loader import require_torch

    torch = require_torch()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / inner  # ms -> us


def bench(call, warmup: int, rounds: int, inner: int) -> BenchStats:
    from kernel_lab.loader import require_torch

    torch = require_torch()
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = [_measure_once(call, inner) for _ in range(rounds)]
    return BenchStats(samples_us=samples, clocks_sm_mhz=sm_clocks_mhz())


def interleaved(call_a, call_b, warmup: int, rounds: int, inner: int) -> tuple[BenchStats, BenchStats]:
    """Same-session A/B: alternate one round of A with one of B (swapping the
    order every round) so clock/thermal drift hits both sides equally."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    for _ in range(warmup):
        call_a()
        call_b()
    torch.cuda.synchronize()
    samples_a: list[float] = []
    samples_b: list[float] = []
    for r in range(rounds):
        if r % 2 == 0:
            samples_a.append(_measure_once(call_a, inner))
            samples_b.append(_measure_once(call_b, inner))
        else:
            samples_b.append(_measure_once(call_b, inner))
            samples_a.append(_measure_once(call_a, inner))
    clocks = sm_clocks_mhz()
    return BenchStats(samples_a, clocks), BenchStats(samples_b, clocks)


def sm_clocks_mhz() -> int | None:
    """Record-only SM clock probe; None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
