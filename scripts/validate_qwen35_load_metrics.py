#!/usr/bin/env python3
"""Qwen3.5 NVIDIA end-to-end gate for scheduler metrics.

Run this against an already-running OpenInfer server started with a constrained
Qwen3.5 decode capacity (normally ``--max-batch 1``). The gate drives real
``/v1/completions`` traffic, scrapes ``/metrics`` throughout the workload, and
retains both a JSON artifact and paste-ready Markdown evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import shlex
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_http_common import (
    artifact_command,
    current_commit,
    detect_hardware_toolchain,
    model_fingerprint,
    write_json,
)


METRIC_FIELDS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
}
PROMETHEUS_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[-+]?Inf)"
    r"(?:\s+\d+)?$"
)
PROMETHEUS_LABEL_RE = re.compile(
    r'(?:^|,)\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"'
)
PROMPT_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


@dataclass
class MetricSnapshot:
    observed_at_unix_s: float
    elapsed_ms: float
    running: float
    waiting: float
    kv_usage: float
    raw_lines: list[str]


@dataclass
class RequestResult:
    request_id: str
    ok: bool
    status: int | None
    finish_reason: str | None
    completion_tokens: int | None
    latency_ms: float
    error: str | None
    output_hash: str = ""


def parse_prometheus_labels(raw: str | None) -> dict[str, str]:
    if raw is None or raw == "":
        return {}
    labels: dict[str, str] = {}
    covered = []
    for match in PROMETHEUS_LABEL_RE.finditer(raw):
        name = match.group("name")
        try:
            value = json.loads(f'"{match.group("value")}"')
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Prometheus label value for {name}: {exc}") from exc
        labels[name] = value
        covered.append(match.group(0).lstrip(",").strip())
    if ",".join(covered).replace(" ", "") != raw.replace(" ", ""):
        raise ValueError(f"could not parse Prometheus labels: {raw}")
    return labels


def parse_metric_snapshot(
    metrics: str,
    *,
    model_name: str,
    engine: str,
    observed_at_unix_s: float,
    elapsed_ms: float,
) -> MetricSnapshot:
    matches: dict[str, tuple[float, str]] = {}
    wanted_names = set(METRIC_FIELDS.values())
    for raw_line in metrics.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE_RE.fullmatch(line)
        if match is None or match.group("name") not in wanted_names:
            continue
        labels = parse_prometheus_labels(match.group("labels"))
        if labels.get("model_name") != model_name or labels.get("engine") != engine:
            continue
        metric_name = match.group("name")
        if metric_name in matches:
            raise ValueError(
                f"multiple {metric_name} series matched "
                f"model_name={model_name!r}, engine={engine!r}"
            )
        value = float(match.group("value"))
        if not math.isfinite(value):
            raise ValueError(f"{metric_name} returned non-finite value {value}")
        matches[metric_name] = (value, line)

    missing = [name for name in METRIC_FIELDS.values() if name not in matches]
    if missing:
        raise ValueError(
            "required scheduler metrics were not found for "
            f"model_name={model_name!r}, engine={engine!r}: {', '.join(missing)}"
        )

    running = matches[METRIC_FIELDS["running"]][0]
    waiting = matches[METRIC_FIELDS["waiting"]][0]
    kv_usage = matches[METRIC_FIELDS["kv_usage"]][0]
    if running < 0 or waiting < 0:
        raise ValueError(
            "request gauges must be non-negative, "
            f"got running={running}, waiting={waiting}"
        )
    if kv_usage < 0 or kv_usage > 1.0 + 1e-9:
        raise ValueError(f"KV usage must be in [0, 1], got {kv_usage}")

    return MetricSnapshot(
        observed_at_unix_s=observed_at_unix_s,
        elapsed_ms=elapsed_ms,
        running=running,
        waiting=waiting,
        kv_usage=kv_usage,
        raw_lines=[matches[name][1] for name in METRIC_FIELDS.values()],
    )


def snapshot_to_json(snapshot: MetricSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def snapshot_is_zero(snapshot: MetricSnapshot) -> bool:
    return (
        abs(snapshot.running) < 1e-12
        and abs(snapshot.waiting) < 1e-12
        and abs(snapshot.kv_usage) < 1e-12
    )


def request_succeeded(result: RequestResult, expected_tokens: int) -> bool:
    return (
        result.ok
        and result.status == 200
        and result.finish_reason == "length"
        and result.completion_tokens == expected_tokens
    )


def evaluate_acceptance(
    *,
    baseline: MetricSnapshot,
    traffic_samples: list[MetricSnapshot],
    drained: MetricSnapshot,
    requests: list[RequestResult],
    recovery: RequestResult,
    post_recovery: MetricSnapshot,
    expected_concurrency: int,
    pressure_max_tokens: int,
    recovery_max_tokens: int,
) -> list[str]:
    failures: list[str] = []
    if not snapshot_is_zero(baseline):
        failures.append("initial scheduler metrics were not all zero")
    if not traffic_samples:
        failures.append("no metric samples were collected during traffic")
    elif max(sample.running for sample in traffic_samples) <= 0:
        failures.append("running requests never became non-zero")
    if not traffic_samples or max(sample.waiting for sample in traffic_samples) <= 0:
        failures.append("waiting requests never became non-zero")
    if not traffic_samples or max(sample.kv_usage for sample in traffic_samples) <= 0:
        failures.append("KV cache usage never became non-zero")
    if len(requests) != expected_concurrency:
        failures.append(
            f"expected {expected_concurrency} pressure results, collected {len(requests)}"
        )
    for result in requests:
        if not request_succeeded(result, pressure_max_tokens):
            failures.append(
                f"pressure request {result.request_id} failed: "
                f"status={result.status}, finish_reason={result.finish_reason!r}, "
                f"completion_tokens={result.completion_tokens}, error={result.error!r}"
            )
    if not snapshot_is_zero(drained):
        failures.append("scheduler metrics did not return to zero after pressure traffic")
    if not request_succeeded(recovery, recovery_max_tokens):
        failures.append(
            "follow-up recovery request failed: "
            f"status={recovery.status}, finish_reason={recovery.finish_reason!r}, "
            f"completion_tokens={recovery.completion_tokens}, error={recovery.error!r}"
        )
    if not snapshot_is_zero(post_recovery):
        failures.append("scheduler metrics did not return to zero after recovery traffic")
    return failures


def select_evidence_samples(
    baseline: MetricSnapshot,
    traffic_samples: list[MetricSnapshot],
    drained: MetricSnapshot,
) -> dict[str, MetricSnapshot]:
    if traffic_samples:
        active = next(
            (
                sample
                for sample in traffic_samples
                if sample.running > 0 and sample.kv_usage > 0
            ),
            max(traffic_samples, key=lambda sample: (sample.running, sample.kv_usage)),
        )
        pressure = max(traffic_samples, key=lambda sample: sample.waiting)
    else:
        active = baseline
        pressure = baseline
    return {
        "baseline": baseline,
        "active": active,
        "pressure": pressure,
        "drained": drained,
    }


def make_prompt(index: int, words: int = 32) -> str:
    return " ".join(
        PROMPT_WORDS[(index + offset) % len(PROMPT_WORDS)] for offset in range(words)
    )


def request_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_text(url: str, *, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def assert_served_model(base_url: str, model: str, timeout: float) -> None:
    payload = request_json(f"{base_url}/v1/models", timeout=timeout)
    model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    if model not in model_ids:
        raise RuntimeError(f"model {model!r} not found at /v1/models; available={model_ids}")


def fetch_metric_snapshot(
    base_url: str,
    *,
    model: str,
    engine: str,
    timeout: float,
    started_monotonic: float,
) -> MetricSnapshot:
    observed_at = time.time()
    metrics = request_text(f"{base_url}/metrics", timeout=timeout)
    return parse_metric_snapshot(
        metrics,
        model_name=model,
        engine=engine,
        observed_at_unix_s=observed_at,
        elapsed_ms=(time.monotonic() - started_monotonic) * 1000.0,
    )


def wait_for_idle(
    base_url: str,
    *,
    model: str,
    engine: str,
    scrape_timeout: float,
    idle_timeout: float,
    sample_interval_s: float,
    started_monotonic: float,
) -> MetricSnapshot:
    deadline = time.monotonic() + idle_timeout
    last_snapshot: MetricSnapshot | None = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            last_snapshot = fetch_metric_snapshot(
                base_url,
                model=model,
                engine=engine,
                timeout=scrape_timeout,
                started_monotonic=started_monotonic,
            )
            if snapshot_is_zero(last_snapshot):
                return last_snapshot
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(sample_interval_s)
    if last_snapshot is not None:
        return last_snapshot
    raise TimeoutError(
        f"no valid scheduler metric snapshot was collected within {idle_timeout}s: {last_error!r}"
    )


def run_completion(
    base_url: str,
    *,
    model: str,
    request_id: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    start_barrier: threading.Barrier | None = None,
) -> RequestResult:
    if start_barrier is not None:
        start_barrier.wait(timeout=timeout)
    started = time.monotonic()
    status: int | None = None
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": False,
        "request_id": request_id,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = json.load(response)
        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        usage = payload.get("usage") or {}
        text = choice.get("text") or ""
        finish_reason = choice.get("finish_reason")
        completion_tokens = usage.get("completion_tokens")
        ok = (
            status == 200
            and finish_reason == "length"
            and completion_tokens == max_tokens
        )
        error = None if ok else f"unexpected completion payload: {payload}"
        return RequestResult(
            request_id=request_id,
            ok=ok,
            status=status,
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
            latency_ms=(time.monotonic() - started) * 1000.0,
            error=error,
            output_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )
    except urllib.error.HTTPError as exc:
        status = exc.code
        detail = exc.read(4096).decode("utf-8", errors="replace")
        error = f"HTTP {status}: {detail}"
    except (OSError, TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return RequestResult(
        request_id=request_id,
        ok=False,
        status=status,
        finish_reason=None,
        completion_tokens=None,
        latency_ms=(time.monotonic() - started) * 1000.0,
        error=error,
    )


def render_community_evidence(report: dict[str, Any]) -> str:
    hardware = report.get("hardware_toolchain") or {}
    gpu_lines = hardware.get("gpu") or ["not detected"]
    nvcc = hardware.get("nvcc_version") or "not detected"
    workload = report["workload"]
    summary = report["summary"]
    fingerprint = json.dumps(report.get("model_fingerprint") or {}, sort_keys=True)
    lines = [
        "## Qwen3.5 scheduler metrics NVIDIA validation",
        "",
        f"Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "### Environment",
        "",
        f"- OpenInfer commit: `{report.get('commit') or 'unknown'}`",
        f"- GPU: `{' | '.join(gpu_lines)}`",
        f"- CUDA compiler: `{str(nvcc).replace(chr(10), ' | ')}`",
        f"- Model: `{report.get('model')}`",
        f"- Model revision: `{report.get('model_revision')}`",
        f"- Model fingerprint: `{fingerprint}`",
        "",
        "### Commands",
        "",
        "```bash",
        report.get("server_command") or "# server command not recorded",
        report.get("runner_command") or "# runner command not recorded",
        "```",
        "",
        "### Workload",
        "",
        f"- Concurrency: {workload['concurrency']}",
        f"- Server max batch: {workload['server_max_batch']}",
        f"- Pressure output tokens/request: {workload['pressure_max_tokens']}",
        f"- Sampling interval: {workload['sample_interval_ms']} ms",
        f"- Completed requests: {summary['completed_requests']}",
        f"- Failed requests: {summary['failed_requests']}",
        f"- Peak running: {summary['max_running']}",
        f"- Peak waiting: {summary['max_waiting']}",
        f"- Peak KV usage: {summary['max_kv_usage']}",
        f"- Follow-up recovery: {'PASS' if summary['recovery_succeeded'] else 'FAIL'}",
        "",
        "### Metric output",
        "",
    ]
    for name in ("baseline", "active", "pressure", "drained", "post_recovery"):
        sample = report.get("evidence", {}).get(name)
        if sample is None:
            continue
        lines.extend(
            [
                f"#### {name.replace('_', ' ').title()} ({sample['elapsed_ms']:.1f} ms)",
                "",
                "```text",
                *sample["raw_lines"],
                "```",
                "",
            ]
        )
    if report.get("failures"):
        lines.extend(["### Failures", ""])
        lines.extend(f"- {failure}" for failure in report["failures"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_live(args: argparse.Namespace, runner_command: str) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"invalid --base-url: {args.base_url}")
    started_monotonic = time.monotonic()
    sample_interval_s = args.sample_interval_ms / 1000.0

    assert_served_model(base_url, args.model, args.scrape_timeout)
    baseline = wait_for_idle(
        base_url,
        model=args.model,
        engine=args.engine,
        scrape_timeout=args.scrape_timeout,
        idle_timeout=args.idle_timeout,
        sample_interval_s=sample_interval_s,
        started_monotonic=started_monotonic,
    )

    traffic_samples: list[MetricSnapshot] = []
    scrape_errors: list[str] = []
    barrier = threading.Barrier(args.concurrency + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_completion,
                base_url,
                model=args.model,
                request_id=f"qwen35-load-pressure-{index}",
                prompt=make_prompt(index),
                max_tokens=args.pressure_max_tokens,
                timeout=args.request_timeout,
                start_barrier=barrier,
            )
            for index in range(args.concurrency)
        ]
        barrier.wait(timeout=args.request_timeout)
        sampling_deadline = time.monotonic() + args.request_timeout
        while not all(future.done() for future in futures):
            if time.monotonic() >= sampling_deadline:
                scrape_errors.append("timed out while sampling pressure traffic")
                break
            try:
                traffic_samples.append(
                    fetch_metric_snapshot(
                        base_url,
                        model=args.model,
                        engine=args.engine,
                        timeout=args.scrape_timeout,
                        started_monotonic=started_monotonic,
                    )
                )
            except (OSError, ValueError, urllib.error.URLError) as exc:
                scrape_errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(sample_interval_s)
        requests = [future.result() for future in futures]

    drained = wait_for_idle(
        base_url,
        model=args.model,
        engine=args.engine,
        scrape_timeout=args.scrape_timeout,
        idle_timeout=args.idle_timeout,
        sample_interval_s=sample_interval_s,
        started_monotonic=started_monotonic,
    )
    recovery = run_completion(
        base_url,
        model=args.model,
        request_id="qwen35-load-recovery",
        prompt="Hello",
        max_tokens=args.recovery_max_tokens,
        timeout=args.request_timeout,
    )
    post_recovery = wait_for_idle(
        base_url,
        model=args.model,
        engine=args.engine,
        scrape_timeout=args.scrape_timeout,
        idle_timeout=args.idle_timeout,
        sample_interval_s=sample_interval_s,
        started_monotonic=started_monotonic,
    )

    failures = evaluate_acceptance(
        baseline=baseline,
        traffic_samples=traffic_samples,
        drained=drained,
        requests=requests,
        recovery=recovery,
        post_recovery=post_recovery,
        expected_concurrency=args.concurrency,
        pressure_max_tokens=args.pressure_max_tokens,
        recovery_max_tokens=args.recovery_max_tokens,
    )
    evidence = select_evidence_samples(baseline, traffic_samples, drained)
    evidence["post_recovery"] = post_recovery
    completed = sum(
        request_succeeded(result, args.pressure_max_tokens) for result in requests
    )
    commit = current_commit()
    fingerprint = model_fingerprint(str(args.model_path))
    hardware_toolchain = detect_hardware_toolchain()
    if not hardware_toolchain.get("gpu"):
        failures.append("no NVIDIA GPU was detected by nvidia-smi")
    if commit is None:
        failures.append("OpenInfer git commit could not be detected")
    if not fingerprint:
        failures.append("model fingerprint is empty")
    report = {
        "schema_version": 1,
        "kind": "qwen35_load_snapshot_nvidia_e2e",
        "passed": not failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "base_url": base_url,
        "model": args.model,
        "model_revision": args.model_revision,
        "engine": args.engine,
        "model_fingerprint": fingerprint,
        "hardware_toolchain": hardware_toolchain,
        "server_command": args.server_command,
        "runner_command": runner_command,
        "metrics": METRIC_FIELDS,
        "workload": {
            "concurrency": args.concurrency,
            "server_max_batch": args.server_max_batch,
            "pressure_max_tokens": args.pressure_max_tokens,
            "recovery_max_tokens": args.recovery_max_tokens,
            "sample_interval_ms": args.sample_interval_ms,
            "request_timeout_s": args.request_timeout,
            "scrape_timeout_s": args.scrape_timeout,
            "idle_timeout_s": args.idle_timeout,
        },
        "summary": {
            "completed_requests": completed,
            "failed_requests": len(requests) - completed,
            "sample_count": len(traffic_samples),
            "max_running": max((sample.running for sample in traffic_samples), default=0.0),
            "max_waiting": max((sample.waiting for sample in traffic_samples), default=0.0),
            "max_kv_usage": max((sample.kv_usage for sample in traffic_samples), default=0.0),
            "drained_to_zero": snapshot_is_zero(drained),
            "recovery_succeeded": request_succeeded(recovery, args.recovery_max_tokens),
            "post_recovery_zero": snapshot_is_zero(post_recovery),
        },
        "failures": failures,
        "scrape_errors": scrape_errors,
        "requests": [asdict(result) for result in requests],
        "recovery": asdict(recovery),
        "traffic_samples": [snapshot_to_json(sample) for sample in traffic_samples],
        "evidence": {
            name: snapshot_to_json(sample) for name, sample in evidence.items()
        },
    }
    return report


def command_option(command: str, option: str) -> str | None:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == option:
            return tokens[index + 1] if index + 1 < len(tokens) else None
        prefix = f"{option}="
        if token.startswith(prefix):
            return token.removeprefix(prefix)
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True, help="Exact id returned by /v1/models")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Checkpoint revision, commit, or immutable local snapshot id",
    )
    parser.add_argument(
        "--server-command",
        required=True,
        help="Exact server launch command to retain in the community evidence",
    )
    parser.add_argument("--engine", default="0")
    parser.add_argument("--server-max-batch", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--pressure-max-tokens", type=int, default=512)
    parser.add_argument("--recovery-max-tokens", type=int, default=8)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--scrape-timeout", type=float, default=5.0)
    parser.add_argument("--idle-timeout", type=float, default=60.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("target/qwen35-load-metrics.json"),
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        help="Markdown path; defaults to --out with a .md suffix",
    )
    args = parser.parse_args(argv)
    if not args.model_path.is_dir():
        parser.error(f"--model-path is not a directory: {args.model_path}")
    if args.server_max_batch <= 0:
        parser.error("--server-max-batch must be positive")
    command_max_batch = command_option(args.server_command, "--max-batch")
    if command_max_batch != str(args.server_max_batch):
        parser.error(
            "--server-command must contain the same --max-batch value as "
            f"--server-max-batch ({args.server_max_batch})"
        )
    if args.concurrency <= args.server_max_batch:
        parser.error("--concurrency must exceed --server-max-batch to create slot pressure")
    if args.pressure_max_tokens <= 0 or args.recovery_max_tokens <= 0:
        parser.error("token counts must be positive")
    if args.sample_interval_ms < 10:
        parser.error("--sample-interval-ms must be at least 10")
    if args.request_timeout <= 0 or args.scrape_timeout <= 0 or args.idle_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.evidence_out is None:
        args.evidence_out = args.out.with_suffix(".md")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    invocation = argv if argv is not None else sys.argv[1:]
    runner_command = artifact_command(["python3", str(Path(__file__).resolve()), *invocation])
    report = run_live(args, runner_command)
    write_json(args.out, report)
    evidence = render_community_evidence(report)
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_text(evidence, encoding="utf-8")
    print(evidence, end="")
    print(f"JSON artifact: {args.out}")
    print(f"Markdown evidence: {args.evidence_out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
