#!/usr/bin/env python3
"""OpenAI-compatible HTTP serving benchmark for openinfer.

The harness intentionally talks to /v1/completions over HTTP instead of using
the in-process bench_serving binary. It records streaming TTFT/ITL/TPOT,
request latency, QPS, error rate, timeout rate, and deterministic output hashes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import re
import socket
import statistics
import time
import urllib.parse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


@dataclass
class RequestResult:
    index: int
    request_id: str
    prompt_words: int
    max_tokens: int
    ok: bool
    status: int | None
    error: str | None
    timed_out: bool
    start_s: float
    start_wall_s: float
    first_token_s: float | None
    first_token_wall_s: float | None
    end_s: float
    end_wall_s: float
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    itl_ms: list[float]
    output_chunks: int
    output_chars: int
    output_hash: str
    text_prefix: str
    sampling_label: str = "single"
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    server_trace: dict[str, Any] | None = None


@dataclass(frozen=True)
class SamplingProfile:
    label: str
    temperature: float
    top_k: int
    top_p: float


def arg_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def sampling_mode(args: argparse.Namespace) -> str:
    return arg_value(args, "sampling_mode", "single")


def single_profile(args: argparse.Namespace) -> SamplingProfile:
    return SamplingProfile(
        label="single",
        temperature=float(arg_value(args, "temperature", 0.0)),
        top_k=int(arg_value(args, "top_k", -1)),
        top_p=float(arg_value(args, "top_p", 1.0)),
    )


def greedy_profile() -> SamplingProfile:
    return SamplingProfile(label="greedy", temperature=0.0, top_k=-1, top_p=1.0)


def sampled_profile(args: argparse.Namespace) -> SamplingProfile:
    return SamplingProfile(
        label="sampled",
        temperature=float(arg_value(args, "sample_temperature", 0.8)),
        top_k=int(arg_value(args, "sample_top_k", 40)),
        top_p=float(arg_value(args, "sample_top_p", 0.95)),
    )


def sampling_profile_for(args: argparse.Namespace, global_index: int) -> SamplingProfile:
    if sampling_mode(args) == "mixed-greedy-sampled":
        return greedy_profile() if global_index % 2 == 0 else sampled_profile(args)
    return single_profile(args)


def sampling_profiles_for_report(args: argparse.Namespace) -> dict[str, dict[str, float | int | str]]:
    if sampling_mode(args) == "mixed-greedy-sampled":
        profiles = [greedy_profile(), sampled_profile(args)]
    else:
        profiles = [single_profile(args)]
    return {profile.label: asdict(profile) for profile in profiles}


def count_sampling(results: list[RequestResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.sampling_label] = counts.get(result.sampling_label, 0) + 1
    return counts


def wire_top_k(top_k: int) -> int:
    return 0 if top_k <= 0 else top_k


def validate_top_p(name: str, value: float) -> None:
    if value <= 0.0 or value > 1.0:
        raise SystemExit(f"{name} must be in (0, 1]")


def validate_sampling_args(args: argparse.Namespace) -> None:
    validate_top_p("--top-p", args.top_p)
    validate_top_p("--sample-top-p", args.sample_top_p)
    if args.sampling_mode == "mixed-greedy-sampled" and args.sample_temperature <= 0.0:
        raise SystemExit("--sample-temperature must be positive in mixed-greedy-sampled mode")


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = round((pct / 100.0) * (len(sorted_values) - 1))
    return sorted_values[idx]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "samples": 0,
        }
    sorted_values = sorted(values)
    return {
        "avg_ms": statistics.fmean(sorted_values),
        "p50_ms": percentile(sorted_values, 50),
        "p95_ms": percentile(sorted_values, 95),
        "p99_ms": percentile(sorted_values, 99),
        "max_ms": sorted_values[-1],
        "samples": len(sorted_values),
    }


def summarize_counts(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {
            "min": None,
            "max": None,
            "total": None,
            "samples": 0,
        }
    return {
        "min": min(values),
        "max": max(values),
        "total": sum(values),
        "samples": len(values),
    }


def summarize_trace_ms(measured: list[RequestResult]) -> dict[str, Any]:
    fields = [
        "frontend_to_queue_ms",
        "admission_queue_ms",
        "queue_wait_ms",
        "prefill_ms",
        "first_decode_ms",
        "decode_mean_ms",
        "decode_total_ms",
        "scheduled_to_first_token_ms",
        "scheduled_to_terminal_ms",
        "stream_flush_ms",
    ]
    phase_summary: dict[str, Any] = {}
    for field in fields:
        values = [
            float(result.server_trace[field])
            for result in measured
            if result.server_trace is not None and isinstance(result.server_trace.get(field), (int, float))
        ]
        phase_summary[field] = summarize(values)
    traced = [result for result in measured if result.server_trace is not None]
    active_set_sizes = [
        int(result.server_trace["active_set_size"])
        for result in traced
        if isinstance(result.server_trace.get("active_set_size"), int)
    ]
    decode_batch_sizes = [
        int(result.server_trace["decode_batch_size_max"])
        for result in traced
        if isinstance(result.server_trace.get("decode_batch_size_max"), int)
    ]
    decode_step_counts: list[int] = []
    decode_step_breakdowns: list[tuple[int, int] | None] = []
    saw_decode_step_fields = False
    for result in traced:
        trace = result.server_trace
        assert trace is not None
        count = (
            int(trace["decode_step_count"])
            if isinstance(trace.get("decode_step_count"), int)
            else None
        )
        batched = (
            int(trace["batch_decode_steps"])
            if isinstance(trace.get("batch_decode_steps"), int)
            else None
        )
        singleton = (
            int(trace["singleton_decode_steps"])
            if isinstance(trace.get("singleton_decode_steps"), int)
            else None
        )
        if count is not None:
            decode_step_counts.append(count)
        if count is None and batched is None and singleton is None:
            decode_step_breakdowns.append(None)
            continue

        saw_decode_step_fields = True
        if batched is None and count is not None and singleton is not None:
            batched = count - singleton
        if singleton is None and count is not None and batched is not None:
            singleton = count - batched
        if (
            batched is not None
            and singleton is not None
            and batched >= 0
            and singleton >= 0
            and (count is None or batched + singleton == count)
        ):
            decode_step_breakdowns.append((batched, singleton))
        else:
            decode_step_breakdowns.append(None)

    complete_breakdowns = [
        breakdown for breakdown in decode_step_breakdowns if breakdown is not None
    ]
    breakdown_complete = (
        saw_decode_step_fields and len(complete_breakdowns) == len(traced)
    )
    if breakdown_complete:
        batched_total = sum(batched for batched, _ in complete_breakdowns)
        singleton_total = sum(singleton for _, singleton in complete_breakdowns)
        total_decode_steps = batched_total + singleton_total
        batched_share = (
            batched_total / total_decode_steps if total_decode_steps else None
        )
    else:
        batched_total = None
        singleton_total = None
        batched_share = None
    return {
        "source": "server log lines matching `openinfer_http_trace`; frontend_to_queue includes HTTP ingress, tokenization, and vLLM submit before engine queue",
        "traced_requests": len(traced),
        "missing_traces": [result.request_id for result in measured if result.server_trace is None],
        "phases_ms": phase_summary,
        "active_set_size_max": max(active_set_sizes) if active_set_sizes else None,
        "decode_batch_size_max": max(decode_batch_sizes) if decode_batch_sizes else None,
        "decode_steps": {
            "per_request": summarize_counts(decode_step_counts),
            "batched_total": batched_total,
            "singleton_total": singleton_total,
            "batched_share": batched_share,
        },
    }


def make_prompt(index: int, prompt_words: int) -> str:
    words = [
        DEFAULT_PROMPT_WORDS[(index + offset) % len(DEFAULT_PROMPT_WORDS)]
        for offset in range(prompt_words)
    ]
    return " ".join(words)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def single_or_list(values: list[int]) -> int | list[int]:
    return values[0] if len(values) == 1 else values


def workload_shapes(prompt_words: list[int], max_tokens: list[int]) -> list[tuple[int, int]]:
    return list(product(prompt_words, max_tokens))


def parse_sse_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def parse_sse_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    finish_reason = choices[0].get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def parse_sse_error(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(error, sort_keys=True)
    return None


def request_once(
    index: int,
    request_id: str,
    url: urllib.parse.ParseResult,
    model: str,
    prompt_words: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    ignore_eos: bool,
    top_k: int = -1,
    top_p: float = 1.0,
    sampling_label: str = "single",
) -> RequestResult:
    start = time.perf_counter()
    start_wall = time.time()
    first_token: float | None = None
    first_token_wall: float | None = None
    last_token: float | None = None
    inter_token_ms: list[float] = []
    chunks: list[str] = []
    status: int | None = None

    try:
        conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        port = url.port
        conn = conn_cls(url.hostname, port=port, timeout=timeout)
        path = (url.path.rstrip("/") or "") + "/v1/completions"
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": wire_top_k(top_k),
            "top_p": top_p,
            "stream": True,
            "ignore_eos": ignore_eos,
            "request_id": request_id,
        }
        conn.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status = response.status
        if status != 200:
            error_body = response.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status}: {error_body}")

        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            stream_error = parse_sse_error(payload)
            if stream_error is not None:
                raise RuntimeError(f"SSE error: {stream_error}")
            finish_reason = parse_sse_finish_reason(payload)
            if finish_reason == "error":
                raise RuntimeError("SSE finish_reason=error")
            text = parse_sse_text(payload)
            if not text:
                continue
            now = time.perf_counter()
            if first_token is None:
                first_token = now
                first_token_wall = time.time()
            if last_token is not None:
                inter_token_ms.append((now - last_token) * 1000.0)
            last_token = now
            chunks.append(text)
        conn.close()
        if max_tokens > 0 and not chunks:
            raise RuntimeError("stream completed without streamed text chunks")
        end = time.perf_counter()
        end_wall = time.time()
        text = "".join(chunks)
        output_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        latency_ms = (end - start) * 1000.0
        ttft_ms = None if first_token is None else (first_token - start) * 1000.0
        tpot_ms = None
        if first_token is not None and last_token is not None and len(chunks) > 1:
            tpot_ms = (last_token - first_token) * 1000.0 / (len(chunks) - 1)
        return RequestResult(
            index=index,
            request_id=request_id,
            prompt_words=prompt_words,
            max_tokens=max_tokens,
            ok=True,
            status=status,
            error=None,
            timed_out=False,
            start_s=start,
            start_wall_s=start_wall,
            first_token_s=first_token,
            first_token_wall_s=first_token_wall,
            end_s=end,
            end_wall_s=end_wall,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            itl_ms=inter_token_ms,
            output_chunks=len(chunks),
            output_chars=len(text),
            output_hash=output_hash,
            text_prefix=text[:80],
            sampling_label=sampling_label,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    except (TimeoutError, socket.timeout) as exc:
        end = time.perf_counter()
        return failed_result(
            index,
            request_id,
            prompt_words,
            max_tokens,
            sampling_label,
            temperature,
            top_k,
            top_p,
            status,
            start,
            start_wall,
            end,
            str(exc),
            timed_out=True,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark reports the error string.
        end = time.perf_counter()
        return failed_result(
            index,
            request_id,
            prompt_words,
            max_tokens,
            sampling_label,
            temperature,
            top_k,
            top_p,
            status,
            start,
            start_wall,
            end,
            str(exc),
            timed_out=False,
        )


def failed_result(
    index: int,
    request_id: str,
    prompt_words: int,
    max_tokens: int,
    sampling_label: str,
    temperature: float,
    top_k: int,
    top_p: float,
    status: int | None,
    start: float,
    start_wall: float,
    end: float,
    error: str,
    timed_out: bool,
) -> RequestResult:
    end_wall = time.time()
    return RequestResult(
        index=index,
        request_id=request_id,
        prompt_words=prompt_words,
        max_tokens=max_tokens,
        ok=False,
        status=status,
        error=error,
        timed_out=timed_out,
        start_s=start,
        start_wall_s=start_wall,
        first_token_s=None,
        first_token_wall_s=None,
        end_s=end,
        end_wall_s=end_wall,
        latency_ms=(end - start) * 1000.0,
        ttft_ms=None,
        tpot_ms=None,
        itl_ms=[],
        output_chunks=0,
        output_chars=0,
        output_hash="",
        text_prefix="",
        sampling_label=sampling_label,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


TRACE_RE = re.compile(r"openinfer_http_trace\s+(\{.*\})")
STREAM_ERROR_RE = re.compile(r'request failed .*self\.request_id="([^"]+)"')


def load_server_traces(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    traces: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stream_error_match = STREAM_ERROR_RE.search(line)
        if stream_error_match:
            request_id = stream_error_match.group(1)
            traces.setdefault(request_id, {"request_id": request_id})["server_error"] = line.strip()
            continue
        match = TRACE_RE.search(line)
        if not match:
            continue
        try:
            trace = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        request_id = trace.get("request_id")
        if isinstance(request_id, str):
            traces[request_id] = trace
    return traces


def attach_server_traces(results: list[RequestResult], traces: dict[str, dict[str, Any]]) -> None:
    for result in results:
        trace = find_server_trace(result.request_id, result.start_wall_s, traces)
        if trace is None:
            continue
        result.server_trace = trace
        if result.ok and result.first_token_wall_s is not None:
            emit_at = trace.get("first_token_emit_unix_s")
            if isinstance(emit_at, (int, float)):
                trace["stream_flush_ms"] = max(0.0, (result.first_token_wall_s - float(emit_at)) * 1000.0)
            queued_at = trace.get("queued_at_unix_s")
            if isinstance(queued_at, (int, float)):
                trace["frontend_to_queue_ms"] = max(0.0, (float(queued_at) - result.start_wall_s) * 1000.0)
            scheduled_at = trace.get("scheduled_at_unix_s")
            if isinstance(queued_at, (int, float)) and isinstance(scheduled_at, (int, float)):
                trace["admission_queue_ms"] = max(0.0, (float(scheduled_at) - float(queued_at)) * 1000.0)
        apply_server_error_gate(result)


def apply_server_error_gate(result: RequestResult) -> None:
    if not result.ok or result.server_trace is None:
        return
    server_error = result.server_trace.get("server_error")
    if isinstance(server_error, str):
        result.ok = False
        result.error = f"server generation error: {server_error}"
        return
    finish_reason = result.server_trace.get("finish_reason")
    if finish_reason == "error":
        result.ok = False
        result.error = "server generation error: finish_reason=error"
        return
    completion_tokens = result.server_trace.get("completion_tokens")
    if result.max_tokens > 0 and completion_tokens == 0:
        result.ok = False
        result.error = "server generation error: completion_tokens=0"


def find_server_trace(
    request_id: str,
    start_wall_s: float,
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    prefix = f"cmpl-{request_id}-"
    matches = [
        trace
        for trace_id, trace in traces.items()
        if trace_id == request_id or trace_id == f"cmpl-{request_id}" or trace_id.startswith(prefix)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        timed_matches = [
            trace
            for trace in matches
            if isinstance(trace.get("queued_at_unix_s"), (int, float))
        ]
        if timed_matches:
            return min(
                timed_matches,
                key=lambda trace: abs(float(trace["queued_at_unix_s"]) - start_wall_s),
            )
    return None


def run_batch(args: argparse.Namespace, measured: bool) -> tuple[list[RequestResult], float]:
    url = urllib.parse.urlparse(args.base_url)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise SystemExit(f"invalid --base-url: {args.base_url}")

    offset = args.warmup if measured else 0
    count = args.num_requests if measured else args.warmup
    label = "measured" if measured else "warmup"
    shapes = workload_shapes(args.prompt_words, args.max_tokens)
    workloads = []
    for idx in range(count):
        global_index = offset + idx
        prompt_words, max_tokens = shapes[global_index % len(shapes)]
        workloads.append(
            (
                global_index,
                prompt_words,
                max_tokens,
                make_prompt(global_index, prompt_words),
                sampling_profile_for(args, global_index),
            )
        )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                idx,
                f"openinfer-bench-{label}-{offset + idx}",
                url,
                args.model,
                prompt_words,
                prompt,
                max_tokens,
                profile.temperature,
                args.timeout,
                args.ignore_eos,
                top_k=profile.top_k,
                top_p=profile.top_p,
                sampling_label=profile.label,
            )
            for idx, (_global_index, prompt_words, max_tokens, prompt, profile) in enumerate(workloads)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    ended = time.perf_counter()
    results.sort(key=lambda result: result.index)
    return results, ended - started


def build_report(args: argparse.Namespace, measured: list[RequestResult], wall_s: float) -> dict[str, Any]:
    successes = [result for result in measured if result.ok]
    failures = [result for result in measured if not result.ok]
    latencies = [result.latency_ms for result in successes]
    ttfts = [result.ttft_ms for result in successes if result.ttft_ms is not None]
    tpots = [result.tpot_ms for result in successes if result.tpot_ms is not None]
    itls: list[float] = []
    output_chunks = [result.output_chunks for result in successes]
    output_chars = [result.output_chars for result in successes]
    hashes = [result.output_hash for result in successes]
    input_tokens = [
        int(result.server_trace["prompt_tokens"])
        if result.server_trace is not None and isinstance(result.server_trace.get("prompt_tokens"), int)
        else result.prompt_words
        for result in successes
    ]
    output_tokens = [
        int(result.server_trace["completion_tokens"])
        if result.server_trace is not None and isinstance(result.server_trace.get("completion_tokens"), int)
        else (result.max_tokens if args.ignore_eos else result.output_chunks)
        for result in successes
    ]
    shape_counts: dict[str, int] = {}
    for result in measured:
        key = f"prompt_words={result.prompt_words},max_tokens={result.max_tokens}"
        shape_counts[key] = shape_counts.get(key, 0) + 1

    for result in successes:
        itls.extend(result.itl_ms)

    return {
        "schema_version": 1,
        "kind": "openai_http_completions_stream_benchmark",
        "base_url": args.base_url,
        "model": args.model,
        "workload": {
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "prompt_words": single_or_list(args.prompt_words),
            "max_tokens": single_or_list(args.max_tokens),
            "mixed_shapes": shape_counts,
            "temperature": args.temperature,
            "top_k": int(arg_value(args, "top_k", -1)),
            "top_p": float(arg_value(args, "top_p", 1.0)),
            "sampling_mode": sampling_mode(args),
            "sampling_profiles": sampling_profiles_for_report(args),
            "sampling_counts": count_sampling(measured),
            "ignore_eos": args.ignore_eos,
            "timeout_s": args.timeout,
        },
        "summary": {
            "wall_s": wall_s,
            "completed": len(successes),
            "failed": len(failures),
            "timeouts": sum(1 for result in failures if result.timed_out),
            "sampling_mode": sampling_mode(args),
            "completed_sampling_counts": count_sampling(successes),
            "failed_sampling_counts": count_sampling(failures),
            "qps": len(successes) / wall_s if wall_s > 0 else 0.0,
            "input_tokens_total": sum(input_tokens),
            "output_tokens_total": sum(output_tokens),
            "input_tokens_per_s": sum(input_tokens) / wall_s if wall_s > 0 else 0.0,
            "output_tokens_per_s": sum(output_tokens) / wall_s if wall_s > 0 else 0.0,
            "error_rate": len(failures) / args.num_requests if args.num_requests else 0.0,
            "timeout_rate": (
                sum(1 for result in failures if result.timed_out) / args.num_requests
                if args.num_requests
                else 0.0
            ),
            "output_chunks_total": sum(output_chunks),
            "output_chars_total": sum(output_chars),
            "unique_output_hashes": len(set(hashes)),
            "combined_output_hash": hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()[:16],
        },
        "metrics": {
            "latency": summarize(latencies),
            "ttft": summarize(ttfts),
            "tpot": summarize(tpots),
            "itl": summarize(itls),
        },
        "server_trace": summarize_trace_ms(measured),
        "requests": [asdict(result) for result in measured],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--prompt-words",
        type=parse_int_list,
        default=[16],
        help="Prompt word count, or comma-separated counts for a mixed workload.",
    )
    parser.add_argument(
        "--max-tokens",
        type=parse_int_list,
        default=[16],
        help="Completion token count, or comma-separated counts for a mixed workload.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sampling-mode",
        choices=["single", "mixed-greedy-sampled"],
        default="single",
        help=(
            "single uses --temperature/--top-k/--top-p for every request; "
            "mixed-greedy-sampled alternates greedy and sampled profiles by global request index."
        ),
    )
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--server-log",
        type=Path,
        help="Optional openinfer server log containing openinfer_http_trace lines for TTFT phase attribution.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be positive")
    validate_sampling_args(args)
    if args.warmup > 0:
        warmup_results, _ = run_batch(args, measured=False)
        failed = [result for result in warmup_results if not result.ok]
        if failed:
            raise SystemExit(f"warmup failed: {failed[0].error}")

    measured, wall_s = run_batch(args, measured=True)
    attach_server_traces(measured, load_server_traces(args.server_log))
    report = build_report(args, measured, wall_s)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
