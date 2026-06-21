"""Regression tests for scripts/bench_http_sweep.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_http_sweep.py"
SPEC = importlib.util.spec_from_file_location("bench_http_sweep", SCRIPT_PATH)
assert SPEC is not None
bench_http_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_http_sweep
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_http_sweep)


def args_for(sampling_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        base_url="http://127.0.0.1:8000",
        model="fake-model",
        num_requests=2,
        warmup=0,
        prompt_words=[8],
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        sampling_mode=sampling_mode,
        sample_temperature=0.8,
        sample_top_k=40,
        sample_top_p=0.95,
        no_ignore_eos=False,
        timeout=5.0,
        concurrency=[2],
        max_tokens=[4],
        repeats=2,
    )


def report_with_hashes(
    hashes: list[str],
    *,
    output_chunks: int = 1,
    labels: list[str] | None = None,
) -> dict[str, object]:
    if labels is None:
        labels = ["single"] * len(hashes)
    assert len(labels) == len(hashes)
    return {
        "workload": {
            "prompt_words": 8,
            "concurrency": 2,
            "max_tokens": 4,
        },
        "summary": {
            "failed": 0,
            "timeouts": 0,
            "combined_output_hash": "".join(hashes),
            "qps": 1.0,
            "input_tokens_per_s": 2.0,
            "output_tokens_per_s": 3.0,
        },
        "metrics": {
            "ttft": {"avg_ms": 4.0},
            "tpot": {"avg_ms": 5.0},
            "itl": {"avg_ms": 6.0},
        },
        "server_trace": {"phases_ms": {}},
        "requests": [
            {
                "ok": True,
                "max_tokens": 4,
                "output_chunks": output_chunks,
                "output_hash": value,
                "sampling_label": label,
            }
            for value, label in zip(hashes, labels)
        ],
    }


class BenchHttpSweepTests(unittest.TestCase):
    def test_mixed_sampling_does_not_require_repeat_hash_stability(self) -> None:
        summary = bench_http_sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(["greedy", "sampled-a"], labels=["greedy", "sampled"]),
                report_with_hashes(["greedy", "sampled-b"], labels=["greedy", "sampled"]),
            ],
        )

        self.assertTrue(summary["correctness_gate"]["passed"])
        self.assertFalse(summary["rows"][0]["hash_stability_checked"])
        self.assertTrue(summary["rows"][0]["greedy_hash_stability_checked"])
        self.assertTrue(summary["rows"][0]["output_evidence_present"])
        self.assertFalse(summary["rows"][0]["stable_per_request_hashes"])
        self.assertTrue(summary["rows"][0]["stable_greedy_hashes"])
        self.assertTrue(summary["rows"][0]["sampled_hashes_present"])
        self.assertEqual(summary["workload"]["sampling_profiles"]["sampled"]["top_k"], 40)

    def test_mixed_sampling_requires_sampled_requests(self) -> None:
        summary = bench_http_sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(["greedy-a"], labels=["greedy"]),
                report_with_hashes(["greedy-a"], labels=["greedy"]),
            ],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertTrue(summary["rows"][0]["greedy_hashes_present"])
        self.assertFalse(summary["rows"][0]["sampled_hashes_present"])
        self.assertFalse(summary["rows"][0]["passed"])

    def test_mixed_sampling_requires_greedy_hash_stability(self) -> None:
        summary = bench_http_sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(["greedy-a", "sampled-a"], labels=["greedy", "sampled"]),
                report_with_hashes(["greedy-b", "sampled-b"], labels=["greedy", "sampled"]),
            ],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertTrue(summary["rows"][0]["greedy_hash_stability_checked"])
        self.assertFalse(summary["rows"][0]["stable_greedy_hashes"])
        self.assertFalse(summary["rows"][0]["passed"])

    def test_mixed_sampling_still_requires_output_evidence(self) -> None:
        summary = bench_http_sweep.build_summary(
            args_for("mixed-greedy-sampled"),
            [
                report_with_hashes(["greedy", "sampled"], labels=["greedy", "sampled"]),
                report_with_hashes(["", ""], output_chunks=0, labels=["greedy", "sampled"]),
            ],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertFalse(summary["rows"][0]["output_evidence_present"])
        self.assertFalse(summary["rows"][0]["passed"])

    def test_single_sampling_requires_repeat_hash_stability(self) -> None:
        summary = bench_http_sweep.build_summary(
            args_for("single"),
            [report_with_hashes(["a", "b"]), report_with_hashes(["c", "d"])],
        )

        self.assertFalse(summary["correctness_gate"]["passed"])
        self.assertTrue(summary["rows"][0]["hash_stability_checked"])
        self.assertFalse(summary["rows"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
