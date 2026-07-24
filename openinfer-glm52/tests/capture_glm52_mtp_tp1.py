#!/usr/bin/env python3
"""Capture the GLM5.2 MTP layer-78 TP1 reference with official vLLM."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import vllm
from safetensors.torch import load_file, save_file
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.workspace import init_workspace_manager


DEFAULT_VLLM_COMMIT = "dcfebf93f4eccf30f71872283331eee757915daf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--vllm-commit", default=DEFAULT_VLLM_COMMIT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    diff = (actual - expected).abs().flatten()
    return {
        "elements": diff.numel(),
        "exact": int((actual == expected).sum().item()),
        "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
        "p99": float(torch.quantile(diff, 0.99).item()),
        "max": float(diff.max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.vllm_commit[:9] not in vllm.__version__:
        raise RuntimeError(
            f"vLLM {vllm.__version__} does not match commit {args.vllm_commit}"
        )

    model_config = args.model / "config.json"
    weight_index = args.model / "model.safetensors.index.json"
    torch.cuda.set_device(0)
    engine_args = EngineArgs(
        model=str(args.model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=1,
        enforce_eager=True,
        moe_backend="deep_gemm",
        speculative_config={"method": "mtp", "num_speculative_tokens": 5},
    )
    config = engine_args.create_engine_config()
    draft_config = config.speculative_config.draft_model_config

    started = time.monotonic()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        init_workspace_manager(torch.device("cuda:0"))
        model = get_model(vllm_config=config, model_config=draft_config)
    load_seconds = time.monotonic() - started

    fixture = load_file(args.input_fixture, device="cpu")
    residual = fixture["decoder_residual"].cuda()
    expected_mlp = fixture["decoder_hidden"]
    mtp_layer = model.model.layers["78"]
    layer = mtp_layer.mtp_block

    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        normed = layer.post_attention_layernorm(residual)
        router_logits, _ = layer.mlp.gate(normed)
        topk_weights, topk_ids = layer.mlp.experts.router.select_experts(
            hidden_states=normed,
            router_logits=router_logits,
        )
        shared_gate_up, _ = layer.mlp.shared_experts.gate_up_proj(normed)
        shared_silu = layer.mlp.shared_experts.act_fn(shared_gate_up)
        shared_hidden, _ = layer.mlp.shared_experts.down_proj(shared_silu)
        with set_forward_context(None, config, num_tokens=residual.shape[0]):
            mlp = layer.mlp(normed)
        routed_hidden = (mlp.float() - shared_hidden.float()).to(torch.bfloat16)
        raw_hidden = residual + mlp
        recycle_hidden = mtp_layer.shared_head(raw_hidden)
    torch.cuda.synchronize()
    forward_seconds = time.monotonic() - started

    metadata = {
        "reference": "official-vllm",
        "vllm_version": vllm.__version__,
        "vllm_commit": args.vllm_commit,
        "topology": "tp1-ep0",
        "model": "GLM-5.2-FP8",
        "model_config_sha256": sha256(model_config),
        "model_weight_index_sha256": sha256(weight_index),
        "input_fixture_sha256": sha256(args.input_fixture),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "decoder_residual": residual.cpu(),
            "post_attention_norm": normed.cpu(),
            "router_logits": router_logits.cpu(),
            "topk_weights": topk_weights.cpu(),
            "topk_ids": topk_ids.cpu(),
            "shared_gate_up": shared_gate_up.cpu(),
            "shared_silu": shared_silu.cpu(),
            "shared_hidden": shared_hidden.cpu(),
            "routed_hidden": routed_hidden.cpu(),
            "decoder_hidden": mlp.cpu(),
            "raw_hidden": raw_hidden.cpu(),
            "recycle_hidden": recycle_hidden.cpu(),
        },
        args.output,
        metadata=metadata,
    )

    report = {
        **metadata,
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "tp1_vs_tp8_mlp": stats(mlp, expected_mlp),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    main()
