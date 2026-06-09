#!/usr/bin/env python3
"""Regression tests for scripts/bench_http_serving.py."""

from __future__ import annotations

import importlib.util
import sys
import threading
import unittest
import urllib.parse
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_http_serving.py"
SPEC = importlib.util.spec_from_file_location("bench_http_serving", SCRIPT_PATH)
assert SPEC and SPEC.loader
bench_http_serving = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_http_serving
SPEC.loader.exec_module(bench_http_serving)


class DoneOnlyHandler(BaseHTTPRequestHandler):
    response_body = b"data: [DONE]\n\n"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format: str, *args: object) -> None:
        return


class BenchHttpServingTests(unittest.TestCase):
    def setUp(self) -> None:
        DoneOnlyHandler.response_body = b"data: [DONE]\n\n"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), DoneOnlyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = urllib.parse.urlparse(f"http://{host}:{port}")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_done_only_stream_fails_when_tokens_requested(self) -> None:
        result = bench_http_serving.request_once(
            index=0,
            request_id="req-empty",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("without streamed text chunks", result.error or "")
        self.assertEqual(result.output_chunks, 0)

    def test_finish_reason_error_stream_fails_even_after_text_chunk(self) -> None:
        DoneOnlyHandler.response_body = (
            b'data: {"choices":[{"text":"partial","finish_reason":null}]}\n\n'
            b'data: {"choices":[{"text":"","finish_reason":"error"}]}\n\n'
            b"data: [DONE]\n\n"
        )

        result = bench_http_serving.request_once(
            index=0,
            request_id="req-finish-error",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("finish_reason=error", result.error or "")

    def test_error_payload_stream_fails(self) -> None:
        DoneOnlyHandler.response_body = (
            b'data: {"error":{"message":"generation failed"}}\n\n'
            b"data: [DONE]\n\n"
        )

        result = bench_http_serving.request_once(
            index=0,
            request_id="req-payload-error",
            url=self.url,
            model="fake-model",
            prompt_words=1,
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("SSE error: generation failed", result.error or "")

    def test_server_trace_log_is_attached_by_vllm_completion_id_prefix(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-1",
            prompt_words=16,
            max_tokens=2,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=100.0,
            first_token_s=1.2,
            first_token_wall_s=100.25,
            end_s=1.4,
            end_wall_s=100.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=20.0,
            itl_ms=[20.0],
            output_chunks=2,
            output_chars=4,
            output_hash="abcd",
            text_prefix="text",
        )
        line = (
            'INFO openinfer_http_trace {"request_id":"cmpl-bench-1-generated",'
            '"queued_at_unix_s":100.01,"scheduled_at_unix_s":100.03,'
            '"first_token_emit_unix_s":100.20,"prefill_ms":170.0,'
            '"first_decode_ms":28.0}\\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(line, encoding="utf-8")
            traces = bench_http_serving.load_server_traces(path)
            bench_http_serving.attach_server_traces([result], traces)

        self.assertIsNotNone(result.server_trace)
        assert result.server_trace is not None
        self.assertAlmostEqual(result.server_trace["admission_queue_ms"], 20.0, places=3)
        self.assertAlmostEqual(result.server_trace["stream_flush_ms"], 50.0, places=3)
        self.assertAlmostEqual(result.server_trace["frontend_to_queue_ms"], 10.0, places=3)

    def test_server_stream_error_log_marks_request_failed(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
            prompt_words=16,
            max_tokens=16,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=100.0,
            first_token_s=1.2,
            first_token_wall_s=100.25,
            end_s=1.4,
            end_wall_s=100.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=1,
            output_chars=80,
            output_hash="abcd",
            text_prefix="text",
        )
        line = (
            'ERROR vllm_engine_core_client::client::stream: stream.rs:90 '
            'request failed with an internal error during generation '
            'self.request_id="cmpl-bench-0-generated"\\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(line, encoding="utf-8")
            traces = bench_http_serving.load_server_traces(path)
            bench_http_serving.attach_server_traces([result], traces)

        self.assertFalse(result.ok)
        self.assertIn("server generation error", result.error or "")

    def test_server_trace_zero_completion_tokens_marks_request_failed(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
            prompt_words=16,
            max_tokens=16,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=100.0,
            first_token_s=1.2,
            first_token_wall_s=100.25,
            end_s=1.4,
            end_wall_s=100.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=1,
            output_chars=80,
            output_hash="abcd",
            text_prefix="text",
        )
        line = (
            'INFO openinfer_http_trace {"request_id":"cmpl-bench-0-generated",'
            '"completion_tokens":0}\\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(line, encoding="utf-8")
            traces = bench_http_serving.load_server_traces(path)
            bench_http_serving.attach_server_traces([result], traces)

        self.assertFalse(result.ok)
        self.assertIn("completion_tokens=0", result.error or "")

    def test_mixed_workload_report_records_input_and_output_tokens(self) -> None:
        results = [
            bench_http_serving.RequestResult(
                index=0,
                request_id="bench-0",
                prompt_words=16,
                max_tokens=4,
                ok=True,
                status=200,
                error=None,
                timed_out=False,
                start_s=0.0,
                start_wall_s=0.0,
                first_token_s=0.1,
                first_token_wall_s=0.1,
                end_s=0.2,
                end_wall_s=0.2,
                latency_ms=200.0,
                ttft_ms=100.0,
                tpot_ms=30.0,
                itl_ms=[30.0, 30.0, 30.0],
                output_chunks=4,
                output_chars=8,
                output_hash="aaaa",
                text_prefix="text",
                server_trace={"prompt_tokens": 22, "completion_tokens": 4},
            ),
            bench_http_serving.RequestResult(
                index=1,
                request_id="bench-1",
                prompt_words=128,
                max_tokens=8,
                ok=True,
                status=200,
                error=None,
                timed_out=False,
                start_s=0.0,
                start_wall_s=0.0,
                first_token_s=0.2,
                first_token_wall_s=0.2,
                end_s=0.4,
                end_wall_s=0.4,
                latency_ms=400.0,
                ttft_ms=200.0,
                tpot_ms=25.0,
                itl_ms=[25.0] * 7,
                output_chunks=8,
                output_chars=16,
                output_hash="bbbb",
                text_prefix="more",
                server_trace={"prompt_tokens": 165, "completion_tokens": 8},
            ),
        ]
        args = type(
            "Args",
            (),
            {
                "base_url": "http://127.0.0.1:8000",
                "model": "fake-model",
                "num_requests": 2,
                "concurrency": 2,
                "warmup": 0,
                "prompt_words": [16, 128],
                "max_tokens": [4, 8],
                "temperature": 0.0,
                "ignore_eos": True,
                "timeout": 5.0,
            },
        )()
        report = bench_http_serving.build_report(args, results, wall_s=2.0)

        self.assertEqual(report["summary"]["input_tokens_total"], 187)
        self.assertEqual(report["summary"]["output_tokens_total"], 12)
        self.assertAlmostEqual(report["summary"]["input_tokens_per_s"], 93.5)
        self.assertAlmostEqual(report["summary"]["output_tokens_per_s"], 6.0)
        self.assertEqual(
            report["workload"]["mixed_shapes"],
            {
                "prompt_words=16,max_tokens=4": 1,
                "prompt_words=128,max_tokens=8": 1,
            },
        )

    def test_ignore_eos_output_token_fallback_uses_requested_max_tokens(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-0",
            prompt_words=16,
            max_tokens=16,
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=0.0,
            start_wall_s=0.0,
            first_token_s=0.1,
            first_token_wall_s=0.1,
            end_s=0.2,
            end_wall_s=0.2,
            latency_ms=200.0,
            ttft_ms=100.0,
            tpot_ms=None,
            itl_ms=[],
            output_chunks=1,
            output_chars=80,
            output_hash="aaaa",
            text_prefix="text",
            server_trace=None,
        )
        args = type(
            "Args",
            (),
            {
                "base_url": "http://127.0.0.1:8000",
                "model": "fake-model",
                "num_requests": 1,
                "concurrency": 1,
                "warmup": 0,
                "prompt_words": [16],
                "max_tokens": [16],
                "temperature": 0.0,
                "ignore_eos": True,
                "timeout": 5.0,
            },
        )()
        report = bench_http_serving.build_report(args, [result], wall_s=2.0)

        self.assertEqual(report["summary"]["output_tokens_total"], 16)
        self.assertAlmostEqual(report["summary"]["output_tokens_per_s"], 8.0)


if __name__ == "__main__":
    unittest.main()
