#!/usr/bin/env python3
"""Regression tests for scripts/bench_http_common.py."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_http_common.py"
SPEC = importlib.util.spec_from_file_location("bench_http_common", SCRIPT_PATH)
assert SPEC is not None
bench_http_common = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_http_common
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_http_common)


class BenchHttpCommonTests(unittest.TestCase):
    def test_shell_command_preserves_argument_boundaries(self) -> None:
        rendered = bench_http_common.shell_command(
            ["bench", "--claim-boundary", "HTTP only; no soak", "plain"]
        )

        self.assertEqual(rendered, "bench --claim-boundary 'HTTP only; no soak' plain")

    def test_artifact_command_makes_repo_paths_relative(self) -> None:
        script = bench_http_common.REPO_ROOT / "scripts" / "bench_http_serving.py"

        rendered = bench_http_common.artifact_command(
            [str(script), "--claim-boundary", "HTTP only; no soak"]
        )

        self.assertEqual(
            rendered,
            "scripts/bench_http_serving.py --claim-boundary 'HTTP only; no soak'",
        )

    def test_model_fingerprint_covers_tokenizer_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")

            fingerprint = bench_http_common.model_fingerprint(str(root))

        self.assertEqual(set(fingerprint), {"config.json", "tokenizer.json"})
        self.assertTrue(all(len(digest) == 64 for digest in fingerprint.values()))

    def test_hardware_metadata_hashes_gpu_identity(self) -> None:
        def command_output(command: list[str], **_: object) -> str | None:
            if "--query-gpu=uuid" in command:
                return "GPU-first\nGPU-second"
            return None

        with mock.patch.object(
            bench_http_common, "run_text", side_effect=command_output
        ):
            metadata = bench_http_common.detect_hardware_toolchain()

        self.assertEqual(
            metadata["gpu_identity_sha256"],
            [
                "7aa60c235ac46afbfbddf4d225387b61ccf89385ab0ca84d6d3213d4a523aaea",
                "ea89ccea8cbe62bc875a1ff5182c5a5c0a84ef6833bd54eacf4debc379a331f7",
            ],
        )

    def test_nvcc_version_falls_back_to_cuda_home(self) -> None:
        with mock.patch.dict("os.environ", {"CUDA_HOME": "/opt/cuda"}, clear=True):
            with mock.patch.object(
                bench_http_common,
                "run_text",
                side_effect=lambda command: "CUDA 12.8"
                if command == ["/opt/cuda/bin/nvcc", "--version"]
                else None,
            ):
                version = bench_http_common.nvcc_version()

        self.assertEqual(version, "CUDA 12.8")

    def test_numeric_summary_ignores_bool_and_non_finite_values(self) -> None:
        summary = bench_http_common.numeric_summary(
            [False, True, 1.0, 3.0, math.nan, math.inf]
        )

        self.assertEqual(summary, {"median": 2.0, "min": 1.0, "max": 3.0, "samples": 2})

    def test_zero_median_with_nonzero_spread_is_noisy(self) -> None:
        marker = bench_http_common.repeat_noise_marker([[0.0, 0.0, 1.0]], 3)

        self.assertEqual(marker, "noisy")


if __name__ == "__main__":
    unittest.main()
