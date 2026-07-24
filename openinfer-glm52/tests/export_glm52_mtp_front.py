#!/usr/bin/env python3
"""Export a deterministic five-row MTP fixture from an official-vLLM trace."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-commit", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-weight-index-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = sorted(args.trace_dir.glob("forward-*.pt"))
    record = None
    selected = None
    selected_index = None
    for index, candidate in enumerate(candidates):
        current = torch.load(candidate, map_location="cpu", weights_only=True)
        if current["positions"].reshape(-1).tolist() == list(range(5)):
            record = current
            selected = candidate
            selected_index = index
            break
    if record is None or selected is None or selected_index is None:
        raise RuntimeError(f"no five-row prompt record in {args.trace_dir}")

    names = (
        "positions",
        "inputs_embeds_raw",
        "previous_hidden_raw",
        "inputs_embeds_norm",
        "previous_hidden_norm",
        "eh_proj",
        "decoder_hidden",
        "decoder_residual",
        "raw_hidden",
        "recycle_hidden",
    )
    tensors = {name: record[name].contiguous() for name in names}
    logits_path = sorted(args.trace_dir.glob("logits-*.pt"))[selected_index]
    logits = torch.load(logits_path, map_location="cpu", weights_only=True)
    matching_rows = [
        row
        for row in range(record["raw_hidden"].shape[0])
        if torch.equal(record["raw_hidden"][row], logits["raw_hidden"][-1])
    ]
    if len(matching_rows) != 1:
        raise RuntimeError(
            f"expected one sampled row for {selected.name}, found {matching_rows}"
        )
    tensors["logits_sampled_row"] = torch.tensor(matching_rows, dtype=torch.int64)
    tensors["logits_topk_ids"] = logits["logits_topk_ids"].contiguous()
    tensors["logits_topk_values"] = logits["logits_topk_values"].contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "reference": "official-vllm",
            "vllm_commit": args.vllm_commit,
            "topology": "tp8-ep0",
            "model": "GLM-5.2-FP8",
            "model_config_sha256": args.model_config_sha256,
            "model_weight_index_sha256": args.model_weight_index_sha256,
            "prompt": "The capital of France is",
            "trace_schema": str(record["trace_schema"]),
            "record": selected.name,
        },
    )


if __name__ == "__main__":
    main()
