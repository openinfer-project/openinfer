#!/usr/bin/env python3
"""Dump Hugging Face greedy generation for DeepSeek-V2-Lite EP=2 comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers.dynamic_module_utils as dynamic_module_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_u32_le(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def allow_missing_optional_flash_attn() -> None:
    # This is a standalone oracle-dump script; the process exits after loading the
    # model, so keeping the compatibility patch installed is intentional.
    original_check_imports = dynamic_module_utils.check_imports

    def check_imports(filename: str, *args, **kwargs):
        try:
            return original_check_imports(filename, *args, **kwargs)
        except ImportError as exc:
            message = str(exc)
            if getattr(exc, "name", None) == "flash_attn" or (
                "requires the following packages" in message and "flash_attn" in message
            ):
                return dynamic_module_utils.get_relative_imports(filename)
            raise

    dynamic_module_utils.check_imports = check_imports


def load_model(model_path: str, device_map: str, device: str):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            **kwargs,
        )
    model.eval()
    return model


def first_parameter_device(model, fallback: str) -> str:
    if hasattr(model, "device"):
        return str(model.device)
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback


def generate_with_transformers(
    model,
    tokenizer,
    prompt: str,
    output_len: int,
    device: str,
    ignore_eos: bool,
):
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_token_ids:
        raise RuntimeError("tokenizer returned empty prompt")

    input_device = first_parameter_device(model, device) if device == "cuda" else device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
    generation_kwargs = {
        "max_new_tokens": output_len,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if ignore_eos:
        generation_kwargs["eos_token_id"] = None

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_token_ids = output_ids[0, input_len:].tolist()
    finish_reason = "length" if len(generated_token_ids) >= output_len else "eos"
    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "token_sha256": sha256_u32_le(generated_token_ids),
        "text_sha256": sha256_text(generated_text),
        "finish_reason": finish_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True, help="HF model path or id")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--output-len", type=int, default=16, help="Greedy output length")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="HF device_map value; use 'none' for a single-device local load",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used when device_map=none",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Keep generating even if the model emits eos",
    )
    parser.add_argument("--out", default="-", help="Write JSON to file; '-' prints to stdout")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if model_path.exists() and not model_path.is_dir():
        print(f"error: model path {model_path} is not a directory", file=sys.stderr)
        return 1

    allow_missing_optional_flash_attn()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = load_model(args.model_path, args.device_map, args.device)
    result = generate_with_transformers(
        model,
        tokenizer,
        args.prompt,
        args.output_len,
        args.device,
        args.ignore_eos,
    )

    payload = {
        "model_path": args.model_path,
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "compat_patches": [
            "dynamic_module_utils.check_imports ignores missing optional flash_attn import"
        ],
        "device_map": args.device_map,
        "device": args.device,
        "dtype": "torch.bfloat16",
        "prompt": args.prompt,
        "output_len": args.output_len,
        "prompt_token_ids": result["prompt_token_ids"],
        "generated_token_ids": result["generated_token_ids"],
        "generated_text": result["generated_text"],
        "token_sha256": result["token_sha256"],
        "text_sha256": result["text_sha256"],
        "finish_reason": result["finish_reason"],
        "generation_mode": "transformers_generate_use_cache",
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.output_len,
            "use_cache": True,
            "ignore_eos": args.ignore_eos,
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.out == "-":
        print(text)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
