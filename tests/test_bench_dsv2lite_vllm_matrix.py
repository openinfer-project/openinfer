"""Regression tests for scripts/bench_dsv2lite_vllm_matrix.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import ArgumentTypeError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_dsv2lite_vllm_matrix.py"
SPEC = importlib.util.spec_from_file_location("bench_dsv2lite_vllm_matrix", SCRIPT_PATH)
assert SPEC is not None
bench_matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_matrix
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_matrix)


class BenchDsv2LiteMatrixTests(unittest.TestCase):
    def summarize_existing_without_metadata_probe(self, args):
        with mock.patch.object(bench_matrix, "metadata", return_value={"test": True}):
            return bench_matrix.summarize_existing(args)

    def test_display_path_keeps_symlinked_target_repo_relative(self) -> None:
        path = bench_matrix.REPO_ROOT / "target" / "benchmarks" / "deepseek-v2-lite-vllm-tp2-ep2"

        self.assertEqual(
            bench_matrix.display_path(path),
            "target/benchmarks/deepseek-v2-lite-vllm-tp2-ep2",
        )

    def test_error_text_strips_repo_absolute_prefix(self) -> None:
        err = RuntimeError(
            f"failed at {bench_matrix.REPO_ROOT.absolute()}/target/benchmarks/result.json"
        )

        self.assertEqual(
            bench_matrix.error_text(err),
            "failed at target/benchmarks/result.json",
        )

    def test_error_text_redacts_home_absolute_prefix(self) -> None:
        err = RuntimeError(f"failed at {Path.home()}/models/DeepSeek-V2-Lite")

        self.assertEqual(
            bench_matrix.error_text(err),
            "failed at ~/models/DeepSeek-V2-Lite",
        )

    def test_redact_text_strips_repo_and_home_prefixes(self) -> None:
        text = (
            f"{bench_matrix.REPO_ROOT.absolute()}/target/run.log "
            f"{bench_matrix.REPO_ROOT.absolute()} "
            f"{Path.home()}/models/DeepSeek-V2-Lite "
            f"{Path.home()} "
            f"/{'root'}/miniconda3/lib/python3.12/site-packages/vllm "
            "/home/runner/project "
            f"~/{'auto'}{'dl'}-tmp/hf-accuracy-venv/bin/python"
        )

        self.assertEqual(
            bench_matrix.redact_text(text),
            "target/run.log <repo> ~/models/DeepSeek-V2-Lite ~ "
            "~/miniconda3/lib/python3.12/site-packages/vllm ~/project "
            "~/tmp/hf-accuracy-venv/bin/python",
        )

    def test_redact_text_masks_common_sensitive_env_values(self) -> None:
        token_key = "HF" + "_" + "TOKEN"
        hub_key = "HUGGINGFACE" + "_HUB" + "_TOKEN"
        public_key_name = "API" + "_KEY"
        pass_key = "PASS" + "WORD"
        text = (
            f"{token_key}=value_abc {hub_key}='value_def' "
            f"{public_key_name}=live_key {pass_key}=pw"
        )

        redacted = bench_matrix.redact_text(text)

        self.assertIn(f"{token_key}=<redacted>", redacted)
        self.assertIn(f"{hub_key}=<redacted>", redacted)
        self.assertIn(f"{public_key_name}=<redacted>", redacted)
        self.assertIn(f"{pass_key}=<redacted>", redacted)
        self.assertNotIn("value_abc", redacted)
        self.assertNotIn("value_def", redacted)
        self.assertNotIn("live_key", redacted)

    def test_redact_command_applies_text_redaction_per_argument(self) -> None:
        redacted = bench_matrix.redact_command([f"/{'root'}/venv/bin/python", "--version"])

        self.assertEqual(redacted, ["~/venv/bin/python", "--version"])

    def test_redact_payload_recurses_into_lists_and_dicts(self) -> None:
        payload = {
            "command": [f"/{'root'}/venv/bin/python", "--version"],
            "nested": {"path": "/home/runner/project"},
        }

        self.assertEqual(
            bench_matrix.redact_payload(payload),
            {"command": ["~/venv/bin/python", "--version"], "nested": {"path": "~/project"}},
        )

    def test_openinfer_server_command_keeps_default_features(self) -> None:
        args = SimpleNamespace(model_path=Path("models/DeepSeek-V2-Lite"), model_id="DeepSeek-V2-Lite")
        spec = bench_matrix.ENGINES[0]

        cmd = bench_matrix.server_command(args, spec, 8000)

        self.assertNotIn("--no-default-features", cmd)
        self.assertIn("--features", cmd)
        self.assertIn("deepseek-v2-lite", cmd)

    def test_parse_args_allows_dash_prefixed_vllm_extra_args(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "bench",
                "--plan-only",
                "--vllm-serve-extra-args",
                "--max-num-seqs",
                "16",
            ],
        ):
            args = bench_matrix.parse_args()

        self.assertEqual(args.vllm_serve_extra_args, ["--max-num-seqs", "16"])

    def test_parse_args_allows_separator_before_vllm_extra_args(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "bench",
                "--plan-only",
                "--vllm-serve-extra-args",
                "--",
                "--max-num-seqs",
                "16",
            ],
        ):
            args = bench_matrix.parse_args()

        self.assertEqual(args.vllm_serve_extra_args, ["--max-num-seqs", "16"])

    def test_parse_args_defaults_vllm_extra_args_when_omitted(self) -> None:
        with mock.patch.object(sys, "argv", ["bench", "--plan-only"]):
            args = bench_matrix.parse_args()

        self.assertEqual(args.vllm_serve_extra_args, bench_matrix.default_vllm_extra_args())

    def test_vllm_bench_command_leaves_warmups_to_separate_call(self) -> None:
        args = SimpleNamespace(
            vllm_cmd="vllm",
            model_id="DeepSeek-V2-Lite",
            model_path=Path("models/DeepSeek-V2-Lite"),
            input_len=64,
            output_len=64,
            num_warmups=4,
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
        )

        cmd = bench_matrix.vllm_bench_command(
            args,
            port=8000,
            num_prompts=32,
            result_dir=Path("target/results"),
            result_filename="result.json",
            max_concurrency=8,
        )

        self.assertNotIn("--num-warmups", cmd)
        self.assertIn("--save-detailed", cmd)

    def test_metadata_records_custom_hf_python_version_command(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            repeats=3,
            noisy_threshold=0.05,
            hf_python="/tmp/hf-python",
            vllm_cmd="vllm",
        )

        with mock.patch.object(
            bench_matrix,
            "try_command",
            side_effect=lambda cmd: {"command": cmd, "available": False},
        ):
            meta = bench_matrix.metadata(args)

        self.assertEqual(
            meta["versions"]["hf_python"]["command"],
            ["/tmp/hf-python", "--version"],
        )
        self.assertTrue(meta["versions"]["hf_python_explicit"])

    def test_metadata_records_hf_python_default_as_not_explicit(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            repeats=3,
            noisy_threshold=0.05,
            hf_python=None,
            vllm_cmd="vllm",
        )

        with mock.patch.object(
            bench_matrix,
            "try_command",
            side_effect=lambda cmd: {"command": cmd, "available": False},
        ):
            meta = bench_matrix.metadata(args)

        self.assertEqual(meta["versions"]["hf_python"]["command"], [sys.executable, "--version"])
        self.assertFalse(meta["versions"]["hf_python_explicit"])
        self.assertIn("--hf-python", meta["versions"]["hf_python_note"])

    def test_try_command_records_redacted_error_without_raising(self) -> None:
        with mock.patch.object(bench_matrix.shutil, "which", return_value="/bin/tool"):
            with mock.patch.object(
                bench_matrix,
                "run_capture",
                side_effect=RuntimeError(f"timeout in /{'root'}/venv/bin/tool"),
            ):
                result = bench_matrix.try_command(["tool", "--version"])

        self.assertTrue(result["available"])
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("~/venv/bin/tool", result["error"])

    def test_run_correctness_gate_uses_custom_hf_python_only_for_hf_dump(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            hf_python="/opt/hf-venv/bin/python",
            command_timeout_s=30,
            keep_going=True,
        )
        calls: list[list[str]] = []

        def fake_run_capture(cmd, **_kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def fake_load_json(_path):
            return {"classification": "all_token_text_exact", "warnings": []}

        with tempfile.TemporaryDirectory() as tmp:
            correctness = Path(tmp) / "correctness"
            correctness.mkdir()
            (correctness / "comparison.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(bench_matrix, "run_capture", side_effect=fake_run_capture):
                with mock.patch.object(bench_matrix, "load_json", side_effect=fake_load_json):
                    result = bench_matrix.run_correctness_gate(args, Path(tmp))

        self.assertTrue(result["passed"])
        self.assertEqual(calls[0][0], "/opt/hf-venv/bin/python")
        self.assertEqual(calls[3][0], sys.executable)

    def test_plan_records_custom_hf_python_in_correctness_commands(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            out_dir=Path("target/benchmarks/deepseek-v2-lite-vllm-tp2-ep2"),
            port=8000,
            hf_python="/opt/hf-venv/bin/python",
            vllm_cmd="vllm",
            vllm_serve_extra_args=bench_matrix.default_vllm_extra_args(),
            cuda_visible_devices=None,
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            direct_batches=[1, 4, 8],
            repeats=3,
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            noisy_threshold=0.05,
        )

        with mock.patch.object(bench_matrix, "metadata", return_value={"test": True}):
            plan = bench_matrix.plan(args)

        self.assertEqual(
            plan["correctness_commands"][0]["command"][0],
            "/opt/hf-venv/bin/python",
        )
        self.assertEqual(
            plan["correctness_commands"][1]["command"][0],
            bench_matrix.redact_text(sys.executable),
        )

    def test_plan_skips_version_probes(self) -> None:
        args = SimpleNamespace(
            model_path=Path("models/DeepSeek-V2-Lite"),
            model_id="DeepSeek-V2-Lite",
            out_dir=Path("target/benchmarks/deepseek-v2-lite-vllm-tp2-ep2"),
            port=8000,
            hf_python="/opt/hf-venv/bin/python",
            vllm_cmd="vllm",
            vllm_serve_extra_args=bench_matrix.default_vllm_extra_args(),
            cuda_visible_devices=None,
            input_len=64,
            output_len=64,
            num_prompts=32,
            num_warmups=4,
            concurrency=[1, 4, 8],
            direct_batches=[1, 4, 8],
            repeats=3,
            request_rate="inf",
            temperature=0.0,
            ignore_eos=True,
            noisy_threshold=0.05,
        )

        with mock.patch.object(bench_matrix, "try_command") as try_command:
            plan = bench_matrix.plan(args)

        try_command.assert_not_called()
        self.assertTrue(plan["metadata"]["versions"]["probes_skipped"])

    def test_wait_for_server_fails_fast_when_process_exits(self) -> None:
        class FakeServer:
            log_path = Path("server.log")

            def poll(self) -> int:
                return 1

            def log_tail(self) -> str:
                return "boom"

        with self.assertRaisesRegex(RuntimeError, "server exited before readiness"):
            bench_matrix.wait_for_server(
                FakeServer(),
                bench_matrix.ENGINES[0],
                9,
                "DeepSeek-V2-Lite",
                60.0,
            )

    def test_parse_direct_artifact_reports_tpot_and_backend_counters(self) -> None:
        parsed = bench_matrix.parse_direct_artifact(
            {
                "config": {"batch_size": 4},
                "timing": {"per_token_decode_stats": {"mean_us": 2000.0}},
                "accuracy": {
                    "token_sha256": "tok",
                    "text_sha256": "txt",
                    "same_prompt_rows_exact": True,
                },
                "gpu_timing": {"sample_count": 7, "failure_count": 0},
                "ep": {"dispatch_calls": 11, "nccl_exchange_calls": 3},
                "cuda_graph_readiness": {"full_decode_capture_ready": False},
            }
        )

        self.assertEqual(parsed["tpot_ms"], 2.0)
        self.assertEqual(parsed["output_tok_s"], 2000.0)
        self.assertEqual(parsed["token_sha256"], "tok")
        self.assertTrue(parsed["same_prompt_rows_exact"])
        self.assertEqual(parsed["gpu_event_samples"], 7)
        self.assertEqual(parsed["ep"]["nccl_exchange_calls"], 3)
        self.assertEqual(
            parsed["backend_counters"],
            {"host_dispatch_calls": 11, "nccl_exchange_calls": 3},
        )

    def test_parse_vllm_bench_artifact_uses_duration_fallback_for_output_rate(self) -> None:
        parsed = bench_matrix.parse_vllm_bench_artifact(
            {
                "num_completed_requests": 24,
                "num_failed_requests": 0,
                "total_output_tokens": 384,
                "duration": 12.0,
                "mean_tpot_ms": 41.0,
                "mean_ttft_ms": 120.0,
            }
        )

        self.assertEqual(parsed["completed"], 24)
        self.assertEqual(parsed["failed"], 0)
        self.assertEqual(parsed["output_tok_s"], 32.0)
        self.assertEqual(parsed["mean_tpot_ms"], 41.0)
        self.assertEqual(parsed["mean_ttft_ms"], 120.0)
        self.assertTrue(parsed["passed"])

    def test_parse_vllm_bench_artifact_marks_failed_requests_failed(self) -> None:
        parsed = bench_matrix.parse_vllm_bench_artifact(
            {
                "num_completed_requests": 31,
                "num_failed_requests": 1,
                "num_timeouts": 0,
            }
        )

        self.assertFalse(parsed["passed"])

    def test_output_text_hash_handles_openai_and_detail_shapes(self) -> None:
        parsed = bench_matrix.output_text_hash(
            {
                "details": [
                    {"response": {"choices": [{"text": "alpha"}]}},
                    {"generated_text": "beta"},
                ],
            }
        )

        self.assertEqual(parsed["count"], 2)
        self.assertIsNotNone(parsed["sha256"])

    def test_summarize_values_handles_empty_zero_and_noise(self) -> None:
        empty = bench_matrix.summarize_values([], 0.05)
        zero = bench_matrix.summarize_values([0.0, 0.0], 0.05)
        noisy = bench_matrix.summarize_values([10.0, 11.0], 0.05)

        self.assertIsNone(empty["median"])
        self.assertFalse(empty["noisy"])
        self.assertEqual(zero["spread_ratio"], None)
        self.assertFalse(zero["noisy"])
        self.assertEqual(noisy["median"], 10.5)
        self.assertTrue(noisy["noisy"])

    def test_parse_int_list_rejects_empty_and_non_positive_values(self) -> None:
        self.assertEqual(bench_matrix.parse_int_list("1,4,8"), [1, 4, 8])
        with self.assertRaises(ArgumentTypeError):
            bench_matrix.parse_int_list("")
        with self.assertRaises(ArgumentTypeError):
            bench_matrix.parse_int_list("1,0")

    def test_batch_size_from_path_parses_only_positive_batch_files(self) -> None:
        self.assertEqual(bench_matrix.batch_size_from_path(Path("batch8.json")), 8)
        self.assertIsNone(bench_matrix.batch_size_from_path(Path("batch.json")))
        self.assertIsNone(bench_matrix.batch_size_from_path(Path("batch-1.json")))
        self.assertIsNone(bench_matrix.batch_size_from_path(Path("batch0.json")))

    def test_trace_missing_count_prefers_missing_traces_length(self) -> None:
        self.assertEqual(bench_matrix.trace_missing_count({"missing_traces": []}), 0)
        self.assertEqual(
            bench_matrix.trace_missing_count({"missing_traces": ["a", "b"]}),
            2,
        )
        self.assertEqual(bench_matrix.trace_missing_count({"missing_trace_count": 3}), 3)

    def test_correctness_passed_requires_exact_classification_without_warnings(self) -> None:
        self.assertTrue(
            bench_matrix.correctness_passed(
                {"classification": "all_token_text_exact", "warnings": []}
            )
        )
        self.assertFalse(
            bench_matrix.correctness_passed(
                {"classification": "all_token_text_exact", "warnings": ["hash warning"]}
            )
        )
        self.assertFalse(
            bench_matrix.correctness_passed(
                {"classification": "token_mismatch", "warnings": []}
            )
        )
        self.assertFalse(
            bench_matrix.correctness_passed(
                {"classification": "all_token_text_exact"}
            )
        )

    def test_summarize_http_rows_marks_noisy_cells(self) -> None:
        rows = bench_matrix.summarize_http_rows(
            [
                {
                    "engine": "vllm-tp2",
                    "cells": [
                        {"concurrency": 4, "mean_tpot_ms": 40.0, "output_tok_s": 80.0, "completed": 24, "failed": 0},
                        {"concurrency": 4, "mean_tpot_ms": 60.0, "output_tok_s": 100.0, "completed": 24, "failed": 0},
                    ],
                }
            ],
            noisy_threshold=0.05,
        )

        row = rows[0]["summary_by_concurrency"][0]
        self.assertEqual(row["concurrency"], 4)
        self.assertTrue(row["noisy"])
        self.assertEqual(row["mean_tpot_ms"]["median"], 50.0)
        self.assertEqual(row["output_tok_s"]["median"], 90.0)

    def test_summarize_existing_rebuilds_summary_from_raw_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "nccl" / "batch1.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "nccl",
                        "config": {"batch_size": 1},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                        "accuracy": {"token_sha256": "tok", "text_sha256": "txt"},
                    }
                ),
                encoding="utf-8",
            )
            http = root / "http_raw" / "vllm-tp2" / "c8" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    {
                        "num_completed_requests": 24,
                        "num_failed_requests": 0,
                        "total_output_tokens": 384,
                        "duration": 2.0,
                        "mean_tpot_ms": 40.0,
                        "mean_ttft_ms": 110.0,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["kind"], "deepseek_v2_lite_vllm_tp2_ep2_benchmark_matrix")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "nccl")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 1)
        self.assertEqual(summary["http_concurrency_pressure"][0]["engine"], "vllm-tp2")
        self.assertEqual(
            summary["http_concurrency_pressure"][0]["summary_by_concurrency"][0]["output_tok_s"]["median"],
            192.0,
        )

    def test_summarize_existing_marks_warned_correctness_artifact_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correctness = root / "correctness" / "comparison.json"
            correctness.parent.mkdir(parents=True)
            correctness.write_text(
                json.dumps(
                    {
                        "classification": "all_token_text_exact",
                        "warnings": ["case_0: hash warning"],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertEqual(summary["correctness_gate"]["claim_bucket"], "failed_setup")

    def test_summarize_existing_preserves_correctness_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correctness = root / "correctness" / "comparison.json"
            correctness.parent.mkdir(parents=True)
            correctness.write_text(
                json.dumps({"classification": "all_token_text_exact", "warnings": []}),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "correctness_gate": {
                            "artifacts": {
                                "hf": "correctness/hf.json",
                                "host_staged": "correctness/host-staged.json",
                                "nccl": "correctness/nccl.json",
                            },
                            "commands": [{"label": "hf", "command": ["python", "hf_dump.py"]}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        gate = summary["correctness_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["artifacts"]["hf"], "correctness/hf.json")
        self.assertEqual(gate["artifacts"]["host_staged"], "correctness/host-staged.json")
        self.assertEqual(gate["artifacts"]["nccl"], "correctness/nccl.json")
        self.assertEqual(gate["commands"][0]["label"], "hf")

    def test_summarize_existing_falls_back_to_path_for_direct_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch8.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "host-staged")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 8)

    def test_summarize_existing_prefers_direct_json_identity_over_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch8.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "nccl",
                        "config": {"batch_size": 4},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "nccl")
        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 4)
        self.assertTrue(summary["direct_diagnostic_batch"][0]["passed"])

    def test_summarize_existing_marks_invalid_direct_identity_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch8.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "config": {"batch_size": 0},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["direct_diagnostic_batch"][0]["batch_size"], 0)
        self.assertFalse(summary["direct_diagnostic_batch"][0]["passed"])
        self.assertEqual(summary["direct_diagnostic_batch"][0]["claim_bucket"], "failed_setup")

    def test_summarize_existing_keeps_unresolved_failed_setup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "host-staged" / "batch1.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "host-staged",
                        "config": {"batch_size": 1},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "direct_diagnostic_batch": [
                            {
                                "claim_bucket": "failed_setup",
                                "backend": "nccl",
                                "batch_size": 1,
                                "passed": False,
                                "error": "nccl init failed",
                            }
                        ],
                        "http_concurrency_pressure": [
                            {
                                "claim_bucket": "failed_setup",
                                "engine": "vllm-tp2",
                                "passed": False,
                                "error": "server_start_failed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        rows_by_backend = {
            row["backend"]: row for row in summary["direct_diagnostic_batch"]
        }
        self.assertTrue(rows_by_backend["host-staged"]["passed"])
        self.assertFalse(rows_by_backend["nccl"]["passed"])
        self.assertEqual(len(summary["direct_diagnostic_batch"]), 2)
        self.assertEqual(summary["http_concurrency_pressure"][0]["engine"], "vllm-tp2")
        self.assertFalse(summary["http_concurrency_pressure"][0]["passed"])

    def test_summarize_existing_replaces_resolved_failed_setup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct_diagnostic_batch" / "nccl" / "batch1.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "backend": "nccl",
                        "config": {"batch_size": 1},
                        "timing": {"per_token_decode_stats": {"mean_us": 5000}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "direct_diagnostic_batch": [
                            {
                                "claim_bucket": "failed_setup",
                                "backend": "nccl",
                                "batch_size": 1,
                                "passed": False,
                                "error": "nccl init failed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(len(summary["direct_diagnostic_batch"]), 1)
        self.assertEqual(summary["direct_diagnostic_batch"][0]["backend"], "nccl")
        self.assertTrue(summary["direct_diagnostic_batch"][0]["passed"])
        self.assertEqual(summary["direct_diagnostic_batch"][0]["claim_bucket"], "direct_diagnostic_batch")

    def test_summarize_existing_replaces_resolved_http_failed_setup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    {
                        "num_completed_requests": 32,
                        "num_failed_requests": 0,
                        "total_output_tokens": 2048,
                        "duration": 64.0,
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "claim_bucket": "failed_setup",
                                "engine": "vllm-tp2",
                                "passed": False,
                                "error": "old startup failure",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(len(summary["http_concurrency_pressure"]), 1)
        row = summary["http_concurrency_pressure"][0]
        self.assertEqual(row["engine"], "vllm-tp2")
        self.assertTrue(row["passed"])
        self.assertEqual(row["claim_bucket"], "http_pressure")
        self.assertEqual(row["resolved_failed_setup_rows"][0]["error"], "old startup failure")

    def test_summarize_existing_preserves_http_context_and_engine_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for engine in ("vllm-tp2", "openinfer-host-staged"):
                http = root / "http_raw" / engine / "c1" / "r0" / "result.json"
                http.parent.mkdir(parents=True)
                http.write_text(
                    json.dumps(
                        {
                            "num_completed_requests": 32,
                            "num_failed_requests": 0,
                            "total_output_tokens": 2048,
                            "duration": 64.0,
                        }
                    ),
                    encoding="utf-8",
                )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "engine": "vllm-tp2",
                                "label": "vLLM TP2",
                                "family": "vllm",
                                "server_command": ["vllm", "serve"],
                                "cells": [
                                    {
                                        "concurrency": 1,
                                        "repeat": 0,
                                        "artifact": str(
                                            root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
                                        ),
                                        "command": ["vllm", "bench"],
                                    }
                                ],
                            },
                            {
                                "engine": "openinfer-host-staged",
                                "label": "OpenInfer host-staged",
                                "family": "openinfer",
                                "server_command": ["cargo", "run"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        rows = summary["http_concurrency_pressure"]
        self.assertEqual([row["engine"] for row in rows], ["openinfer-host-staged", "vllm-tp2"])
        self.assertEqual(rows[0]["label"], "OpenInfer host-staged")
        self.assertEqual(rows[0]["server_command"], ["cargo", "run"])
        self.assertEqual(rows[1]["label"], "vLLM TP2")
        self.assertEqual(rows[1]["cells"][0]["command"], ["vllm", "bench"])

    def test_summarize_existing_does_not_copy_cell_context_by_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    {
                        "num_completed_requests": 32,
                        "num_failed_requests": 0,
                        "total_output_tokens": 2048,
                        "duration": 64.0,
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "engine": "vllm-tp2",
                                "cells": [
                                    {
                                        "concurrency": 1,
                                        "repeat": 0,
                                        "artifact": "renamed/http_raw/vllm-tp2/c1/r0/result.json",
                                        "command": ["stale", "bench"],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=1,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        cell = summary["http_concurrency_pressure"][0]["cells"][0]
        self.assertNotIn("command", cell)

    def test_summarize_existing_marks_empty_http_engine_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "http_raw" / "vllm-tp2").mkdir(parents=True)
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["http_concurrency_pressure"][0]
        self.assertEqual(row["engine"], "vllm-tp2")
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertIn("no HTTP benchmark result artifacts", row["error"])

    def test_summarize_existing_marks_missing_http_cells_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            http = root / "http_raw" / "vllm-tp2" / "c1" / "r0" / "result.json"
            http.parent.mkdir(parents=True)
            http.write_text(
                json.dumps(
                    {
                        "num_completed_requests": 32,
                        "num_failed_requests": 0,
                        "total_output_tokens": 2048,
                        "duration": 64.0,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=2,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["http_concurrency_pressure"][0]
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertEqual(len(row["missing_result_cells"]), 3)

    def test_summarize_existing_preserves_previous_failed_setup_when_still_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "http_raw" / "vllm-tp2").mkdir(parents=True)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "http_concurrency_pressure": [
                            {
                                "claim_bucket": "failed_setup",
                                "engine": "vllm-tp2",
                                "passed": False,
                                "error": "old startup failure",
                                "server_command": ["vllm", "serve"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["http_concurrency_pressure"][0]
        self.assertFalse(row["passed"])
        self.assertIn("no HTTP benchmark result artifacts", row["error"])
        self.assertEqual(row["previous_failed_setup_rows"][0]["error"], "old startup failure")

    def test_summarize_existing_infers_failed_http_rows_from_server_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_key = "HF" + "_" + "TOKEN"
            log = root / "server_logs" / "vllm-tp2.log"
            log.parent.mkdir(parents=True)
            log.write_text(
                f"{token_key}=value_from_log\n"
                "RuntimeError: Engine core initialization failed\n"
                "ValueError: could not determine the shape of object type "
                "'torch.storage.UntypedStorage'\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        self.assertEqual(summary["http_concurrency_pressure"][0]["engine"], "vllm-tp2")
        self.assertFalse(summary["http_concurrency_pressure"][0]["passed"])
        self.assertIn("UntypedStorage", summary["http_concurrency_pressure"][0]["error"])
        self.assertIn("UntypedStorage", summary["http_concurrency_pressure"][0]["startup_failure"])
        self.assertIn(f"{token_key}=<redacted>", summary["http_concurrency_pressure"][0]["server_log_tail"])
        self.assertNotIn("value_from_log", summary["http_concurrency_pressure"][0]["server_log_tail"])

    def test_summarize_existing_rebuilds_openinfer_trace_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "openinfer_trace" / "openinfer-host-staged" / "c8.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    {
                        "summary": {
                            "completed": 8,
                            "failed": 0,
                            "timeouts": 0,
                            "output_tokens_per_s": 20.0,
                        },
                        "server_trace": {
                            "decode_batch_size_max": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        trace_rows = summary["openinfer_trace_pass"]
        self.assertEqual(trace_rows[0]["engine"], "openinfer-host-staged")
        self.assertTrue(trace_rows[0]["passed"])
        self.assertEqual(trace_rows[0]["cells"][0]["trace"]["decode_batch_size_max"], 5)

    def test_summarize_existing_preserves_unresolved_trace_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "openinfer_trace" / "openinfer-host-staged" / "c1.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    {
                        "summary": {"completed": 8, "failed": 0, "output_tokens_per_s": 20.0},
                        "server_trace": {"decode_batch_size_max": 1, "missing_traces": []},
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "openinfer_trace_pass": [
                            {
                                "engine": "openinfer-nccl",
                                "claim_bucket": "failed_setup",
                                "passed": False,
                                "error": "old trace setup failed",
                                "cells": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        rows_by_engine = {row["engine"]: row for row in summary["openinfer_trace_pass"]}
        self.assertTrue(rows_by_engine["openinfer-host-staged"]["passed"])
        self.assertFalse(rows_by_engine["openinfer-nccl"]["passed"])
        self.assertEqual(rows_by_engine["openinfer-nccl"]["error"], "old trace setup failed")

    def test_summarize_existing_marks_empty_trace_engine_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "openinfer_trace" / "openinfer-host-staged").mkdir(parents=True)
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4, 8],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["openinfer_trace_pass"][0]
        self.assertEqual(row["engine"], "openinfer-host-staged")
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertIn("no OpenInfer trace result artifacts", row["error"])

    def test_summarize_existing_marks_missing_trace_concurrency_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "openinfer_trace" / "openinfer-host-staged" / "c1.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    {
                        "summary": {"completed": 8, "failed": 0, "output_tokens_per_s": 20.0},
                        "server_trace": {"decode_batch_size_max": 1, "missing_traces": []},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                summarize_only=root,
                noisy_threshold=0.05,
                model_path=Path("models/DeepSeek-V2-Lite"),
                model_id="DeepSeek-V2-Lite",
                input_len=64,
                output_len=64,
                num_prompts=32,
                num_warmups=4,
                concurrency=[1, 4],
                request_rate="inf",
                temperature=0.0,
                ignore_eos=True,
                repeats=3,
                hf_python=sys.executable,
                vllm_cmd="vllm",
            )

            summary = self.summarize_existing_without_metadata_probe(args)

        row = summary["openinfer_trace_pass"][0]
        self.assertFalse(row["passed"])
        self.assertEqual(row["claim_bucket"], "failed_setup")
        self.assertEqual(row["missing_trace_concurrency"], [4])

    def test_classify_server_start_failure_prefers_specific_missing_ninja(self) -> None:
        log = (
            "RuntimeError: Engine core initialization failed\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'ninja'\n"
        )

        self.assertEqual(
            bench_matrix.classify_server_start_failure(log),
            "server_start_failed: missing ninja",
        )

    def test_classify_server_start_failure_specific_branches(self) -> None:
        cases = [
            ("group_end failed (ncclUnhandledCudaError)", "server_start_failed: ncclUnhandledCudaError"),
            (
                "ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'",
                "server_start_failed: safetensors UntypedStorage shape inference",
            ),
            (
                "Failed to get device capability: SM 12.x requires CUDA >= 12.9",
                "server_start_failed: FlashInfer SM120 CUDA compatibility",
            ),
            (
                "RuntimeError: FlashInfer requires GPUs with sm75 or higher",
                "server_start_failed: FlashInfer GPU capability detection",
            ),
            (
                "RuntimeError: Engine core initialization failed",
                "server_start_failed: vLLM engine core initialization failed",
            ),
            (
                "Novel launcher failure without traceback",
                "server_start_failed: Novel launcher failure without traceback",
            ),
        ]

        for log, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(bench_matrix.classify_server_start_failure(log), expected)


if __name__ == "__main__":
    unittest.main()
