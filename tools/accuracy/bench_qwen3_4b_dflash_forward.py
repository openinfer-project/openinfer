#!/usr/bin/env python3
"""Benchmark Qwen3-4B-DFlash forward in Hugging Face and OpenInfer.

The benchmark uses the same synthetic fixed inputs for both engines, so the
result isolates the standalone drafter forward cost. It does not measure the
full speculative decoding loop because the OpenInfer target/controller path is
not implemented yet.

Example:

    .venv/bin/python tools/accuracy/bench_qwen3_4b_dflash_forward.py \
        --draft-model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
        --openinfer-bin target/release/qwen3_dflash_forward_bench \
        --out target/benchmarks/qwen3-dflash/forward.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel

SEED = 0xD4A5_4B16


def stats(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    def pct(q: float) -> float:
        idx = round((len(sorted_values) - 1) * q)
        return float(sorted_values[min(idx, len(sorted_values) - 1)])
    return {
        "mean": float(sum(sorted_values) / len(sorted_values)),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model-path", default="/home/hezhaozhao/models/Qwen3-4B-DFlash-b16")
    parser.add_argument("--fixture-out", default="target/benchmarks/qwen3-dflash/forward-input.safetensors")
    parser.add_argument("--openinfer-bin", type=Path, help="Path to qwen3_dflash_forward_bench")
    parser.add_argument("--openinfer-draft-cache", action="store_true")
    parser.add_argument("--openinfer-context-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", default="target/benchmarks/qwen3-dflash/forward.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ctx-len", type=int, default=2)
    parser.add_argument("--q-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--target-model-path", default="/home/hezhaozhao/models/Qwen3-4B")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DFlash forward benchmark")

    draft = AutoModel.from_pretrained(
        args.draft_model_path,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{args.device}"},
        trust_remote_code=True,
    ).eval()
    device = next(draft.parameters()).device

    gen = torch.Generator(device=device).manual_seed(SEED)
    hidden = draft.config.hidden_size
    target_layer_count = len(draft.target_layer_ids)
    noise_embedding = torch.randn((1, args.q_len, hidden), generator=gen, device=device, dtype=torch.bfloat16)
    target_hidden = torch.randn(
        (1, args.ctx_len, hidden * target_layer_count),
        generator=gen,
        device=device,
        dtype=torch.bfloat16,
    )
    position_ids = torch.arange(args.ctx_len + args.q_len, device=device, dtype=torch.int32).unsqueeze(0)
    fixture_path = Path(args.fixture_out)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "noise_embedding": noise_embedding.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            "target_hidden": target_hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            "position_ids": position_ids.detach().to("cpu", dtype=torch.int32).contiguous(),
        },
        str(fixture_path),
    )

    hf_latencies = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = draft(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                position_ids=position_ids,
                use_cache=False,
                is_causal=False,
            )
        torch.cuda.synchronize(device)
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = draft(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                position_ids=position_ids,
                use_cache=False,
                is_causal=False,
            )
            torch.cuda.synchronize(device)
            hf_latencies.append((time.perf_counter() - start) * 1000.0)

    openinfer_latencies = None
    if args.openinfer_bin is not None:
        cmd = [
            str(args.openinfer_bin),
            "--model-path",
            args.draft_model_path,
            "--fixture",
            str(fixture_path),
            "--device",
            str(args.device),
            "--ctx-len",
            str(args.ctx_len),
            "--q-len",
            str(args.q_len),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        openinfer_draft_cache = args.openinfer_draft_cache or args.openinfer_context_cache
        if openinfer_draft_cache:
            cmd.append("--draft-cache")
        raw = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        payload = json.loads(raw)
        openinfer_latencies = payload["latency_ms"]

    report = {
        "schema": 1,
        "draft_model_path": args.draft_model_path,
        "target_model_path": args.target_model_path,
        "device": args.device,
        "ctx_len": args.ctx_len,
        "q_len": args.q_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "openinfer_draft_cache": args.openinfer_draft_cache or args.openinfer_context_cache,
        "fixture_out": str(fixture_path),
        "hf_remote_code": {
            "engine": "transformers",
            "latency_ms": stats(hf_latencies),
        },
        "openinfer": openinfer_latencies,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
