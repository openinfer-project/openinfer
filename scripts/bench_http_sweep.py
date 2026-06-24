#!/usr/bin/env python3
"""Run a reproducible HTTP serving sweep over concurrency and max_tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH = SCRIPT_DIR / "bench_http_serving.py"


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    parsed = [int(item) for item in items]
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def request_hashes(report: dict[str, Any], sampling_label: str | None = None) -> list[str]:
    return [
        request["output_hash"]
        for request in report["requests"]
        if request["ok"]
        and (sampling_label is None or request.get("sampling_label") == sampling_label)
    ]


def successful_requests_have_output_evidence(report: dict[str, Any]) -> bool:
    for request in report["requests"]:
        if not request["ok"]:
            continue
        if request.get("max_tokens", 0) <= 0:
            continue
        if not request.get("output_hash") or request.get("output_chunks", 0) <= 0:
            return False
    return True


def sampling_profiles(args: argparse.Namespace) -> dict[str, dict[str, float | int | str]]:
    if args.sampling_mode == "mixed-greedy-sampled":
        return {
            "greedy": {
                "label": "greedy",
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
            },
            "sampled": {
                "label": "sampled",
                "temperature": args.sample_temperature,
                "top_k": args.sample_top_k,
                "top_p": args.sample_top_p,
            },
        }
    return {
        "single": {
            "label": "single",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
    }


def sampling_cli_args(args: argparse.Namespace) -> list[str]:
    return [
        "--sampling-mode",
        args.sampling_mode,
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--sample-temperature",
        str(args.sample_temperature),
        "--sample-top-k",
        str(args.sample_top_k),
        "--sample-top-p",
        str(args.sample_top_p),
    ]


def run_one(
    args: argparse.Namespace,
    prompt_words: int,
    concurrency: int,
    max_tokens: int,
    repeat: int,
) -> dict[str, Any]:
    out = args.out_dir / f"pw{prompt_words}_c{concurrency}_mt{max_tokens}_r{repeat}.json"
    cmd = [
        sys.executable,
        str(BENCH),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--num-requests",
        str(args.num_requests),
        "--concurrency",
        str(concurrency),
        "--warmup",
        str(args.warmup),
        "--prompt-words",
        str(prompt_words),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(args.temperature),
        "--timeout",
        str(args.timeout),
        "--out",
        str(out),
    ]
    cmd.extend(sampling_cli_args(args))
    if args.no_ignore_eos:
        cmd.append("--no-ignore-eos")
    if args.server_log:
        cmd.extend(["--server-log", str(args.server_log)])
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def build_summary(args: argparse.Namespace, reports: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for report in reports:
        workload = report["workload"]
        key = (workload["prompt_words"], workload["concurrency"], workload["max_tokens"])
        cells.setdefault(key, []).append(report)

    rows = []
    correctness_ok = True
    hash_stability_required = args.sampling_mode == "single"
    greedy_hash_stability_required = args.sampling_mode == "mixed-greedy-sampled"
    sampled_hash_presence_required = args.sampling_mode == "mixed-greedy-sampled"
    for (prompt_words, concurrency, max_tokens), cell_reports in sorted(cells.items()):
        baseline = request_hashes(cell_reports[0])
        baseline_greedy = request_hashes(cell_reports[0], sampling_label="greedy")
        zero_failures = True
        hashes_stable = True
        greedy_hashes_stable = True
        greedy_hashes_present = bool(baseline_greedy)
        sampled_hashes_present = bool(request_hashes(cell_reports[0], sampling_label="sampled"))
        output_evidence_present = True
        for report in cell_reports:
            summary = report["summary"]
            hashes = request_hashes(report)
            greedy_hashes = request_hashes(report, sampling_label="greedy")
            sampled_hashes = request_hashes(report, sampling_label="sampled")
            if summary["failed"] or summary["timeouts"]:
                zero_failures = False
            if hashes != baseline:
                hashes_stable = False
            if not greedy_hashes:
                greedy_hashes_present = False
            if not sampled_hashes:
                sampled_hashes_present = False
            if greedy_hashes != baseline_greedy:
                greedy_hashes_stable = False
            if not successful_requests_have_output_evidence(report):
                output_evidence_present = False
        cell_ok = (
            zero_failures
            and output_evidence_present
            and (hashes_stable if hash_stability_required else True)
            and (
                (greedy_hashes_present and greedy_hashes_stable)
                if greedy_hash_stability_required
                else True
            )
            and (sampled_hashes_present if sampled_hash_presence_required else True)
        )
        correctness_ok = correctness_ok and cell_ok
        rows.append(
            {
                "concurrency": concurrency,
                "prompt_words": prompt_words,
                "max_tokens": max_tokens,
                "repeats": len(cell_reports),
                "passed": cell_ok,
                "output_evidence_present": output_evidence_present,
                "hash_stability_checked": hash_stability_required,
                "stable_per_request_hashes": hashes_stable,
                "greedy_hash_stability_checked": greedy_hash_stability_required,
                "stable_greedy_hashes": greedy_hashes_stable,
                "greedy_hashes_present": greedy_hashes_present,
                "sampled_hashes_present": sampled_hashes_present,
                "combined_output_hashes": [
                    report["summary"]["combined_output_hash"] for report in cell_reports
                ],
                "qps": [report["summary"]["qps"] for report in cell_reports],
                "input_tokens_per_s": [
                    report["summary"]["input_tokens_per_s"] for report in cell_reports
                ],
                "output_tokens_per_s": [
                    report["summary"]["output_tokens_per_s"] for report in cell_reports
                ],
                "ttft_avg_ms": [report["metrics"]["ttft"]["avg_ms"] for report in cell_reports],
                "tpot_avg_ms": [report["metrics"]["tpot"]["avg_ms"] for report in cell_reports],
                "itl_avg_ms": [report["metrics"]["itl"]["avg_ms"] for report in cell_reports],
                "failed": [report["summary"]["failed"] for report in cell_reports],
                "timeouts": [report["summary"]["timeouts"] for report in cell_reports],
                "trace_phases_avg_ms": [
                    {
                        phase: stats["avg_ms"]
                        for phase, stats in report["server_trace"]["phases_ms"].items()
                    }
                    for report in cell_reports
                ],
            }
        )

    return {
        "schema_version": 1,
        "kind": "openai_http_completions_sweep",
        "base_url": args.base_url,
        "model": args.model,
        "workload": {
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "prompt_words": args.prompt_words,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "sampling_mode": args.sampling_mode,
            "sampling_profiles": sampling_profiles(args),
            "ignore_eos": not args.no_ignore_eos,
            "timeout_s": args.timeout,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
        },
        "correctness_gate": {
            "passed": correctness_ok,
            "rule": (
                "single mode: failed=0, timeout=0, and per-request output_hash list stable "
                "across repeats for each prompt_words/concurrency/max_tokens cell; mixed-greedy-sampled "
                "mode: failed=0, timeout=0, successful requests have output evidence, and "
                "both greedy and sampled requests are present; greedy output_hash lists stay "
                "stable across repeats, while sampled output hashes are reported but not required stable"
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prompt-words", type=parse_int_list, default=[16])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sampling-mode",
        choices=["single", "mixed-greedy-sampled"],
        default="single",
    )
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--no-ignore-eos", action="store_true")
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=parse_int_list, default=[16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise SystemExit("--top-p must be in (0, 1]")
    if args.sample_top_p <= 0.0 or args.sample_top_p > 1.0:
        raise SystemExit("--sample-top-p must be in (0, 1]")
    if args.sampling_mode == "mixed-greedy-sampled" and args.sample_temperature <= 0.0:
        raise SystemExit("--sample-temperature must be positive in mixed-greedy-sampled mode")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for prompt_words in args.prompt_words:
        for max_tokens in args.max_tokens:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    reports.append(run_one(args, prompt_words, concurrency, max_tokens, repeat))

    summary = build_summary(args, reports)
    (args.out_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["correctness_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
