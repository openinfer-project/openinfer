#!/usr/bin/env python3
"""Compare Qwen3-4B-DFlash HF drafter vs OpenInfer drafter in one target loop.

This is an end-to-end drafter-substitution probe for the current
`openinfer-qwen3-4b-dflash` boundary. The target model, tokenizer, target KV
cache, target verification, target `lm_head`, and greedy sampler all come from
Transformers. The only variable is the drafter:

  * HF remote-code `DFlashDraftModel.forward`
  * OpenInfer `qwen3_dflash_forward_fixture`

The script intentionally uses a no-draft-cache loop on both sides because the
current OpenInfer crate implements standalone draft forward only, not DFlash's
Python `DynamicCache` path or an OpenInfer target/controller.

Example:

    .venv/bin/python tools/accuracy/compare_qwen3_4b_dflash_drafter_generation.py \
        --target-model-path /home/hezhaozhao/models/Qwen3-4B \
        --draft-model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
        --openinfer-bin target/release/qwen3_dflash_forward_fixture \
        --out target/accuracy/qwen3-dflash/drafter-generation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Qwen is a language model that",
    "1, 1, 2, 3, 5,",
]


def sha256_u32_le(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_diff(left: list[int], right: list[int]) -> dict[str, Any] | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {
                "index": index,
                "hf_drafter": left[index],
                "openinfer_drafter": right[index],
                "reason": "token_mismatch",
            }
    if len(left) != len(right):
        return {
            "index": limit,
            "hf_drafter": left[limit] if len(left) > limit else None,
            "openinfer_drafter": right[limit] if len(right) > limit else None,
            "reason": "length_mismatch",
        }
    return None


def input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def extract_context_feature(hidden_states: tuple[torch.Tensor, ...], layer_ids: list[int]) -> torch.Tensor:
    # HF hidden_states includes the embedding output at index 0.
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def greedy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def tensor_deltas(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    deltas = (got.float() - want.float()).abs().flatten().detach().cpu()
    if deltas.numel() == 0:
        return {"mean": 0.0, "p99": 0.0, "max": 0.0, "n": 0}
    sorted_deltas = torch.sort(deltas).values
    p99_index = min(int(deltas.numel() * 0.99), deltas.numel() - 1)
    return {
        "mean": float(deltas.mean().item()),
        "p99": float(sorted_deltas[p99_index].item()),
        "max": float(sorted_deltas[-1].item()),
        "n": int(deltas.numel()),
    }


def merge_delta_stats(items: list[dict[str, float]]) -> dict[str, float] | None:
    if not items:
        return None
    total_n = sum(int(item["n"]) for item in items)
    if total_n == 0:
        return {"mean": 0.0, "p99": 0.0, "max": 0.0, "n": 0}
    # The exact aggregate p99 needs raw samples. For this report the per-block
    # worst p99 is the conservative summary, and max is exact.
    return {
        "mean": sum(item["mean"] * item["n"] for item in items) / total_n,
        "p99": max(item["p99"] for item in items),
        "max": max(item["max"] for item in items),
        "n": total_n,
    }


@dataclass
class Runtime:
    target: torch.nn.Module
    draft: torch.nn.Module
    tokenizer: Any
    target_layer_ids: list[int]
    block_size: int
    mask_token_id: int
    stop_token_ids: list[int]
    openinfer_bin: Path | None
    draft_model_path: Path
    repo_root: Path
    device_ordinal: int
    collect_hidden_delta: bool


def run_openinfer_draft(
    runtime: Runtime,
    *,
    noise_embedding: torch.Tensor,
    target_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    temp_dir: Path,
    step_index: int,
) -> torch.Tensor:
    fixture = temp_dir / f"dflash-input-{step_index:03d}.safetensors"
    out = temp_dir / f"dflash-output-{step_index:03d}.safetensors"
    save_file(
        {
            "noise_embedding": noise_embedding.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            "target_hidden": target_hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            "position_ids": position_ids.detach().to("cpu", dtype=torch.int32).contiguous(),
        },
        str(fixture),
    )
    if runtime.openinfer_bin is not None:
        cmd = [
            str(runtime.openinfer_bin),
            "--model-path",
            str(runtime.draft_model_path),
            "--fixture",
            str(fixture),
            "--out",
            str(out),
            "--device",
            str(runtime.device_ordinal),
        ]
    else:
        cmd = [
            "cargo",
            "run",
            "--release",
            "-p",
            "openinfer-qwen3-4b-dflash",
            "--bin",
            "qwen3_dflash_forward_fixture",
            "--",
            "--model-path",
            str(runtime.draft_model_path),
            "--fixture",
            str(fixture),
            "--out",
            str(out),
            "--device",
            str(runtime.device_ordinal),
        ]
    subprocess.run(cmd, cwd=runtime.repo_root, check=True)
    tensors = load_file(str(out))
    return tensors["openinfer_output"].to(input_device(runtime.target), dtype=torch.bfloat16)


def draft_hidden(
    runtime: Runtime,
    *,
    kind: str,
    noise_embedding: torch.Tensor,
    target_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    temp_dir: Path,
    step_index: int,
) -> tuple[torch.Tensor, dict[str, float] | None]:
    with torch.inference_mode():
        hf_hidden = runtime.draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            use_cache=False,
            is_causal=False,
        )
    if kind == "hf":
        return hf_hidden, None
    oi_hidden = run_openinfer_draft(
        runtime,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        position_ids=position_ids,
        temp_dir=temp_dir,
        step_index=step_index,
    )
    delta = tensor_deltas(oi_hidden, hf_hidden) if runtime.collect_hidden_delta else None
    return oi_hidden, delta


def generate_with_drafter(
    runtime: Runtime,
    *,
    prompt: str,
    max_new_tokens: int,
    kind: str,
    temp_dir: Path,
) -> dict[str, Any]:
    dev = input_device(runtime.target)
    encoded = runtime.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(dev)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.full(
        (1, max_length + runtime.block_size),
        runtime.mask_token_id,
        dtype=torch.long,
        device=dev,
    )
    all_position_ids = torch.arange(output_ids.shape[1], device=dev).unsqueeze(0)

    target_cache = DynamicCache()
    with torch.inference_mode():
        output = runtime.target(
            input_ids,
            position_ids=all_position_ids[:, :num_input_tokens],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = greedy(output.logits)
    target_hidden = extract_context_feature(output.hidden_states, runtime.target_layer_ids)

    start = num_input_tokens
    accepted_plus_fallback_lengths: list[int] = []
    hidden_deltas: list[dict[str, float]] = []
    step_index = 0
    while start < max_length:
        q_len = runtime.block_size
        block_output_ids = output_ids[:, start : start + q_len].clone()
        block_position_ids = all_position_ids[:, start : start + q_len]
        noise_embedding = runtime.target.model.embed_tokens(block_output_ids)

        ctx_len = target_hidden.shape[1]
        draft_position_ids = all_position_ids[:, start - ctx_len : start + q_len]
        hidden, delta = draft_hidden(
            runtime,
            kind=kind,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=draft_position_ids,
            temp_dir=temp_dir,
            step_index=step_index,
        )
        if delta is not None:
            hidden_deltas.append(delta)
        draft_logits = runtime.target.lm_head(hidden[:, -runtime.block_size + 1 :, :])
        block_output_ids[:, 1:] = greedy(draft_logits)

        with torch.inference_mode():
            output = runtime.target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
        posterior = greedy(output.logits)
        matches = block_output_ids[:, 1:] == posterior[:, :-1]
        acceptance_length = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
        advanced = acceptance_length + 1
        output_ids[:, start : start + advanced] = block_output_ids[:, :advanced]
        output_ids[:, start + advanced] = posterior[:, acceptance_length]
        start += advanced
        target_cache.crop(start)
        target_hidden = extract_context_feature(output.hidden_states, runtime.target_layer_ids)[:, :advanced, :]
        accepted_plus_fallback_lengths.append(advanced)
        step_index += 1

        generated_so_far = output_ids[0, num_input_tokens : min(start + 1, max_length)]
        if runtime.stop_token_ids and torch.isin(
            generated_so_far,
            torch.tensor(runtime.stop_token_ids, device=generated_so_far.device),
        ).any():
            break

    full_ids = output_ids[0, :max_length]
    full_ids = full_ids[full_ids != runtime.mask_token_id]
    if runtime.stop_token_ids:
        generated = full_ids[num_input_tokens:]
        stop_tensor = torch.tensor(runtime.stop_token_ids, device=generated.device)
        stop_positions = torch.isin(generated, stop_tensor).nonzero(as_tuple=True)[0]
        if stop_positions.numel() > 0:
            full_ids = full_ids[: num_input_tokens + int(stop_positions[0].item()) + 1]

    full_token_ids = [int(token) for token in full_ids.detach().cpu().tolist()]
    generated_token_ids = full_token_ids[num_input_tokens:]
    full_text = runtime.tokenizer.decode(full_token_ids, skip_special_tokens=False)
    generated_text = runtime.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    return {
        "prompt_token_ids": [int(token) for token in input_ids[0].detach().cpu().tolist()],
        "full_token_ids": full_token_ids,
        "generated_token_ids": generated_token_ids,
        "full_text": full_text,
        "generated_text": generated_text,
        "token_sha256": sha256_u32_le(generated_token_ids),
        "text_sha256": sha256_text(generated_text),
        "accepted_plus_fallback_lengths": accepted_plus_fallback_lengths,
        "hidden_delta_vs_hf": merge_delta_stats(hidden_deltas),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-path", default="/home/hezhaozhao/models/Qwen3-4B-DFlash-b16")
    parser.add_argument("--out", default="target/accuracy/qwen3-dflash/drafter-generation.json")
    parser.add_argument("--prompt", action="append", help="Prompt to test; can be repeated.")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--openinfer-bin", type=Path, help="Path to a built qwen3_dflash_forward_fixture binary.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-hidden-delta", action="store_true")
    parser.add_argument("--stop-token-id", type=int, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DFlash drafter generation comparison")

    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{args.device}"},
        trust_remote_code=True,
    ).eval()
    draft = AutoModel.from_pretrained(
        args.draft_model_path,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{args.device}"},
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)

    stop_token_ids = list(args.stop_token_id)
    eos = getattr(target.config, "eos_token_id", None)
    if isinstance(eos, int):
        stop_token_ids.append(eos)
    elif isinstance(eos, list):
        stop_token_ids.extend(int(token) for token in eos)
    stop_token_ids = sorted(set(stop_token_ids))

    runtime = Runtime(
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        target_layer_ids=[int(layer) for layer in draft.target_layer_ids],
        block_size=int(draft.block_size),
        mask_token_id=int(getattr(draft, "mask_token_id", None) or draft.config.dflash_config["mask_token_id"]),
        stop_token_ids=stop_token_ids,
        openinfer_bin=args.openinfer_bin,
        draft_model_path=Path(args.draft_model_path),
        repo_root=args.repo_root,
        device_ordinal=args.device,
        collect_hidden_delta=not args.skip_hidden_delta,
    )

    prompts = args.prompt or DEFAULT_PROMPTS
    cases = []
    with tempfile.TemporaryDirectory(prefix="qwen3-dflash-parity-") as tmp:
        temp_dir = Path(tmp)
        for index, prompt in enumerate(prompts):
            hf = generate_with_drafter(
                runtime,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                kind="hf",
                temp_dir=temp_dir,
            )
            openinfer = generate_with_drafter(
                runtime,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                kind="openinfer",
                temp_dir=temp_dir,
            )
            token_diff = first_diff(hf["generated_token_ids"], openinfer["generated_token_ids"])
            text_match = hf["generated_text"] == openinfer["generated_text"]
            token_match = token_diff is None
            classification = "all_token_text_exact" if token_match and text_match else "drafter_generation_mismatch"
            cases.append(
                {
                    "id": f"prompt_{index:03d}",
                    "prompt": prompt,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_token_ids": hf["prompt_token_ids"],
                    "hf_drafter": hf,
                    "openinfer_drafter": openinfer,
                    "token_match": token_match,
                    "text_match": text_match,
                    "classification": classification,
                    "first_diff": token_diff,
                }
            )
            print(
                f"{classification}: {prompt!r}; "
                f"hf_accept={hf['accepted_plus_fallback_lengths']} "
                f"openinfer_accept={openinfer['accepted_plus_fallback_lengths']}"
            )

    result = {
        "schema": 1,
        "comparison": "qwen3_4b_dflash_drafter_generation",
        "mode": "greedy_bs1_no_draft_cache_drafter_substitution",
        "target_model_path": args.target_model_path,
        "draft_model_path": args.draft_model_path,
        "openinfer_bin": str(args.openinfer_bin) if args.openinfer_bin else None,
        "block_size": runtime.block_size,
        "target_layer_ids": runtime.target_layer_ids,
        "mask_token_id": runtime.mask_token_id,
        "stop_token_ids": runtime.stop_token_ids,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "case_count": len(cases),
        "all_token_text_exact": all(case["classification"] == "all_token_text_exact" for case in cases),
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if not result["all_token_text_exact"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
