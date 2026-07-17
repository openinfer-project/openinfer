#!/usr/bin/env python3
"""Regression tests for scripts/validate_qwen35_load_metrics.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_qwen35_load_metrics.py"
)
SCRIPTS_DIR = SCRIPT_PATH.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("validate_qwen35_load_metrics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validate_qwen35_load_metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_qwen35_load_metrics
SPEC.loader.exec_module(validate_qwen35_load_metrics)


def _snapshot(
    running: float,
    waiting: float,
    kv_usage: float,
    elapsed_ms: float = 0.0,
):
    return validate_qwen35_load_metrics.MetricSnapshot(
        observed_at_unix_s=1_700_000_000.0,
        elapsed_ms=elapsed_ms,
        running=running,
        waiting=waiting,
        kv_usage=kv_usage,
        raw_lines=[
            f'vllm:num_requests_running{{engine="0",model_name="qwen35-metrics"}} {running}',
            f'vllm:num_requests_waiting{{engine="0",model_name="qwen35-metrics"}} {waiting}',
            f'vllm:kv_cache_usage_perc{{engine="0",model_name="qwen35-metrics"}} {kv_usage}',
        ],
    )


def _request(request_id: str, max_tokens: int):
    return validate_qwen35_load_metrics.RequestResult(
        request_id=request_id,
        ok=True,
        status=200,
        finish_reason="length",
        completion_tokens=max_tokens,
        latency_ms=100.0,
        error=None,
    )


class FakeMetricsHandler(BaseHTTPRequestHandler):
    model = "qwen35-metrics"
    lock = threading.Lock()
    in_flight = 0
    pressure_barrier = threading.Barrier(4)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        if self.path == "/v1/models":
            self._write_json({"data": [{"id": self.model}]})
            return
        if self.path == "/metrics":
            with self.lock:
                in_flight = self.in_flight
            running = 1 if in_flight else 0
            waiting = max(in_flight - 1, 0)
            kv_usage = 0.25 if in_flight else 0.0
            body = (
                f'vllm:num_requests_running{{engine="0",model_name="{self.model}"}} {running}\n'
                f'vllm:num_requests_waiting{{engine="0",model_name="{self.model}"}} {waiting}\n'
                f'vllm:kv_cache_usage_perc{{engine="0",model_name="{self.model}"}} {kv_usage}\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        with self.lock:
            type(self).in_flight += 1
        try:
            if payload.get("request_id", "").startswith("qwen35-load-pressure-"):
                self.pressure_barrier.wait(timeout=5)
                time.sleep(0.15)
            response = {
                "choices": [{"text": "ok", "finish_reason": "length"}],
                "usage": {"completion_tokens": payload["max_tokens"]},
            }
            self._write_json(response)
        finally:
            with self.lock:
                type(self).in_flight -= 1

    def _write_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _: str, *__: object) -> None:
        return


class Qwen35LoadMetricsGateTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeMetricsHandler.in_flight = 0
        FakeMetricsHandler.pressure_barrier = threading.Barrier(4)

    def test_parser_selects_exact_model_and_engine_series(self) -> None:
        metrics = """
# HELP vllm:num_requests_running Number of running requests.
vllm:num_requests_running{engine="0",model_name="other"} 99
vllm:num_requests_running{engine="1",model_name="qwen35-metrics"} 88
vllm:num_requests_running{engine="0",model_name="qwen35-metrics"} 1
vllm:num_requests_waiting{engine="0",model_name="qwen35-metrics"} 3
vllm:kv_cache_usage_perc{engine="0",model_name="qwen35-metrics"} 0.25
"""

        snapshot = validate_qwen35_load_metrics.parse_metric_snapshot(
            metrics,
            model_name="qwen35-metrics",
            engine="0",
            observed_at_unix_s=1_700_000_000.0,
            elapsed_ms=125.0,
        )

        self.assertEqual(snapshot.running, 1.0)
        self.assertEqual(snapshot.waiting, 3.0)
        self.assertEqual(snapshot.kv_usage, 0.25)
        self.assertEqual(len(snapshot.raw_lines), 3)
        self.assertTrue(all('engine="0"' in line for line in snapshot.raw_lines))

    def test_parser_rejects_missing_required_metric(self) -> None:
        metrics = """
vllm:num_requests_running{engine="0",model_name="qwen35-metrics"} 1
vllm:kv_cache_usage_perc{engine="0",model_name="qwen35-metrics"} 0.25
"""

        with self.assertRaisesRegex(ValueError, "vllm:num_requests_waiting"):
            validate_qwen35_load_metrics.parse_metric_snapshot(
                metrics,
                model_name="qwen35-metrics",
                engine="0",
                observed_at_unix_s=1_700_000_000.0,
                elapsed_ms=0.0,
            )

    def test_acceptance_requires_active_waiting_idle_zero_and_recovery(self) -> None:
        baseline = _snapshot(0, 0, 0)
        traffic = [
            _snapshot(1, 0, 0.05, 100),
            _snapshot(1, 3, 0.20, 200),
            _snapshot(1, 2, 0.30, 300),
        ]
        drained = _snapshot(0, 0, 0, 1_000)
        requests = [_request(f"pressure-{index}", 512) for index in range(4)]
        recovery = _request("recovery", 8)
        post_recovery = _snapshot(0, 0, 0, 1_200)

        failures = validate_qwen35_load_metrics.evaluate_acceptance(
            baseline=baseline,
            traffic_samples=traffic,
            drained=drained,
            requests=requests,
            recovery=recovery,
            post_recovery=post_recovery,
            expected_concurrency=4,
            pressure_max_tokens=512,
            recovery_max_tokens=8,
        )

        self.assertEqual(failures, [])

        no_waiting = [
            _snapshot(sample.running, 0, sample.kv_usage, sample.elapsed_ms)
            for sample in traffic
        ]
        failures = validate_qwen35_load_metrics.evaluate_acceptance(
            baseline=baseline,
            traffic_samples=no_waiting,
            drained=drained,
            requests=requests,
            recovery=recovery,
            post_recovery=post_recovery,
            expected_concurrency=4,
            pressure_max_tokens=512,
            recovery_max_tokens=8,
        )

        self.assertIn("waiting requests never became non-zero", failures)

    def test_evidence_selection_keeps_raw_metric_lines(self) -> None:
        baseline = _snapshot(0, 0, 0)
        traffic = [
            _snapshot(1, 0, 0.05, 100),
            _snapshot(1, 3, 0.20, 200),
            _snapshot(1, 1, 0.30, 300),
        ]
        drained = _snapshot(0, 0, 0, 1_000)

        evidence = validate_qwen35_load_metrics.select_evidence_samples(
            baseline, traffic, drained
        )

        self.assertEqual(evidence["active"].elapsed_ms, 100)
        self.assertEqual(evidence["pressure"].waiting, 3)
        self.assertEqual(evidence["drained"].raw_lines, drained.raw_lines)

    def test_markdown_contains_commands_environment_and_metric_output(self) -> None:
        snapshot = _snapshot(0, 0, 0)
        report = {
            "passed": True,
            "commit": "deadbeef1234",
            "model_revision": "0123456789abcdef",
            "server_command": "./target/release/openinfer --max-batch 1",
            "runner_command": "python3 scripts/validate_qwen35_load_metrics.py",
            "hardware_toolchain": {
                "gpu": ["NVIDIA GeForce RTX 5090, 580.76.05, 32607 MiB"],
                "nvcc_version": "Cuda compilation tools, release 12.8",
            },
            "workload": {
                "concurrency": 4,
                "server_max_batch": 1,
                "pressure_max_tokens": 512,
                "sample_interval_ms": 100,
            },
            "summary": {
                "completed_requests": 4,
                "failed_requests": 0,
                "max_running": 1.0,
                "max_waiting": 3.0,
                "max_kv_usage": 0.25,
                "recovery_succeeded": True,
            },
            "evidence": {
                name: validate_qwen35_load_metrics.snapshot_to_json(snapshot)
                for name in ("baseline", "active", "pressure", "drained")
            },
            "failures": [],
        }

        rendered = validate_qwen35_load_metrics.render_community_evidence(report)

        self.assertIn("NVIDIA GeForce RTX 5090", rendered)
        self.assertIn("0123456789abcdef", rendered)
        self.assertIn("--max-batch 1", rendered)
        self.assertIn("vllm:num_requests_waiting", rendered)
        self.assertIn("Completed requests: 4", rendered)

    def test_server_command_max_batch_parser_handles_both_flag_forms(self) -> None:
        self.assertEqual(
            validate_qwen35_load_metrics.command_option(
                "./target/release/openinfer --max-batch 1 --port 18080",
                "--max-batch",
            ),
            "1",
        )
        self.assertEqual(
            validate_qwen35_load_metrics.command_option(
                "./target/release/openinfer --max-batch=2 --port 18080",
                "--max-batch",
            ),
            "2",
        )

    def test_live_runner_covers_http_pressure_metrics_drain_and_recovery(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMetricsHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                model_path = Path(tmp)
                (model_path / "config.json").write_text("{}\n", encoding="utf-8")
                args = SimpleNamespace(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="qwen35-metrics",
                    model_path=model_path,
                    model_revision="0123456789abcdef",
                    server_command="./target/release/openinfer --max-batch 1",
                    engine="0",
                    server_max_batch=1,
                    concurrency=4,
                    pressure_max_tokens=32,
                    recovery_max_tokens=8,
                    sample_interval_ms=10,
                    request_timeout=5.0,
                    scrape_timeout=1.0,
                    idle_timeout=2.0,
                )
                hardware = {
                    "gpu": ["NVIDIA GeForce RTX 5090, 580.76.05, 32607 MiB"],
                    "nvcc_version": "Cuda compilation tools, release 12.8",
                }
                with mock.patch.object(
                    validate_qwen35_load_metrics,
                    "detect_hardware_toolchain",
                    return_value=hardware,
                ):
                    report = validate_qwen35_load_metrics.run_live(
                        args,
                        "python3 scripts/validate_qwen35_load_metrics.py",
                    )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["summary"]["completed_requests"], 4)
        self.assertGreater(report["summary"]["max_running"], 0)
        self.assertGreater(report["summary"]["max_waiting"], 0)
        self.assertGreater(report["summary"]["max_kv_usage"], 0)
        self.assertTrue(report["summary"]["drained_to_zero"])
        self.assertTrue(report["summary"]["recovery_succeeded"])
        self.assertTrue(report["summary"]["post_recovery_zero"])


if __name__ == "__main__":
    unittest.main()
