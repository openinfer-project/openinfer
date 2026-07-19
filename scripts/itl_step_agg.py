#!/usr/bin/env python3
"""Aggregate OPENINFER_ITL_DEBUG `ITL_STEP` scheduler logs (issue #470).

The Qwen3.5 scheduler, when run with `OPENINFER_ITL_DEBUG` set, emits one line
per executed step:

    ITL_STEP mono_us=<u> epoch_us=<u> plan=unified|prefill|decode \
        prefill_tok=<n> prefill_reqs=<n> decode_n=<n> dur_us=<n>

`dur_us` is the step's CPU wall-time, which — because decode sampling forces a
D2H sync — closely tracks the GPU time the active decode rows actually wait on
this step. A step with `prefill_tok > 0` is the only kind that freezes active
decodes behind real prefill work; a `plan=decode` step (`prefill_tok=0`) is a
genuine steady gap.

This script reports the *true per-step stall* distribution straight from the
scheduler, independent of the mixed-load bench's coarse `[submit, last-token]`
window bucketing (which can fold admission/queue latency into the stall bucket
and is apples-to-oranges across chunk on/off). Compare the `unified` rows across
a chunk-ON vs chunk-OFF log to see chunking bound the per-step freeze.

Usage:
    python3 scripts/itl_step_agg.py LOG [LOG ...]
    python3 scripts/itl_step_agg.py --label on canonical_on.log --label off canonical_off.log
"""

import argparse
import re
import sys
from collections import Counter

STEP_RE = re.compile(
    r"ITL_STEP\s+"
    r"mono_us=(?P<mono>\d+)\s+"
    r"epoch_us=(?P<epoch>\d+)\s+"
    r"plan=(?P<plan>\w+)\s+"
    r"prefill_tok=(?P<ptok>\d+)\s+"
    r"prefill_reqs=(?P<preqs>\d+)\s+"
    r"decode_n=(?P<dn>\d+)\s+"
    r"dur_us=(?P<dur>\d+)"
)


def percentile(sorted_vals, pct):
    """Nearest-rank-ish percentile matching the Rust bench's convention:
    idx = round(pct/100 * (n-1))."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = round((pct / 100.0) * (n - 1))
    return sorted_vals[idx]


def summarize_us(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return {
        "n": n,
        "avg_ms": (sum(s) / n) / 1000.0,
        "p50_ms": percentile(s, 50) / 1000.0,
        "p95_ms": percentile(s, 95) / 1000.0,
        "p99_ms": percentile(s, 99) / 1000.0,
        "max_ms": s[-1] / 1000.0,
    }


def parse(path):
    steps = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                steps.append(
                    {
                        "plan": m.group("plan"),
                        "ptok": int(m.group("ptok")),
                        "preqs": int(m.group("preqs")),
                        "dn": int(m.group("dn")),
                        "dur": int(m.group("dur")),
                    }
                )
    return steps


def fmt(stats):
    if stats is None:
        return "  (none)"
    return (
        "  n={n}  p50={p50_ms:.2f}ms  p95={p95_ms:.2f}ms  "
        "p99={p99_ms:.2f}ms  max={max_ms:.2f}ms  avg={avg_ms:.2f}ms".format(**stats)
    )


def report(label, path):
    steps = parse(path)
    total = len(steps)
    plan_counts = Counter(s["plan"] for s in steps)

    # Steps that actually ran prefill work.
    prefill_steps = [s for s in steps if s["ptok"] > 0]
    # Steps that froze *active decodes* behind a prefill chunk (the real ITL
    # stall the background streams experience): prefill_tok>0 AND decode_n>0.
    stall_steps = [s for s in prefill_steps if s["dn"] > 0]
    # Pure decode steps (steady gaps).
    decode_steps = [s for s in steps if s["ptok"] == 0 and s["dn"] > 0]

    print(f"=== {label}  ({path}) ===")
    print(f"total ITL_STEP: {total}  " + "  ".join(f"{k}={v}" for k, v in sorted(plan_counts.items())))
    print(f"prefill-executing steps (prefill_tok>0): {len(prefill_steps)}")
    print("  per-step dur_us [ALL prefill steps]:")
    print(fmt(summarize_us([s["dur"] for s in prefill_steps])))
    print(f"stall steps (prefill_tok>0 AND decode_n>0) — TRUE per-step decode stall: {len(stall_steps)}")
    print(fmt(summarize_us([s["dur"] for s in stall_steps])))
    print("  steady decode steps (prefill_tok=0, decode_n>0):")
    print(fmt(summarize_us([s["dur"] for s in decode_steps])))
    ptok_dist = Counter(s["ptok"] for s in prefill_steps)
    print("  prefill_tok distribution (chunk sizes actually forwarded):")
    for tok, cnt in sorted(ptok_dist.items()):
        print(f"    prefill_tok={tok}: {cnt} step(s)")
    dn_dist = Counter(s["dn"] for s in stall_steps)
    if dn_dist:
        print("  frozen decode width during stall steps (decode_n):")
        for dn, cnt in sorted(dn_dist.items()):
            print(f"    decode_n={dn}: {cnt} step(s)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", action="append", default=[], help="label for the next log path")
    ap.add_argument("logs", nargs="+", help="ITL_STEP log file(s)")
    args = ap.parse_args()

    labels = args.label
    for i, path in enumerate(args.logs):
        label = labels[i] if i < len(labels) else path
        try:
            report(label, path)
        except FileNotFoundError:
            print(f"!! missing log: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
