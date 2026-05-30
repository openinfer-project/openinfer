#!/usr/bin/env python3
"""Generate the HuggingFace bf16 logprob golden for the Qwen3.5-4B gate.

The Rust gate replays these fixed token sequences through pegainfer with
teacher-forced decode and compares top-K logprobs against this stored HF oracle.
For Qwen3.5 the HF oracle follows the same incremental shape: prefill the prompt
with `use_cache=True`, then feed one fixed decode token at a time through
`past_key_values`.

    uv run --no-project python tools/accuracy/dump_qwen35_4b_hf_golden.py \
        --model-path /data/models/Qwen3.5-4B \
        --out test_data/qwen35-4b-hf-golden.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

SEED = 0x_5EED_3535
NUM_SEQS = 12
MIN_PROMPT_LEN = 1
MAX_PROMPT_LEN = 128
DECODE_TOKENS = 8
VOCAB_CEILING = 100_000
TOP_K = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_revision(path: Path) -> str:
    metadata = path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata.exists():
        first = metadata.read_text().splitlines()[0].strip()
        if first:
            return first

    try:
        if (path / ".git").exists():
            out = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out:
                return out
    except Exception:
        pass

    parts = path.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def load_model(model_path: str, dtype: str, device_map: str):
    kwargs = {"trust_remote_code": True, "torch_dtype": DTYPES[dtype]}
    if device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, **kwargs)
    model.eval()
    return model


def input_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="'none' for single-GPU, 'auto' to shard larger models",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    args = parser.parse_args()

    gen = torch.Generator().manual_seed(SEED)
    prompts, decodes = [], []
    for _ in range(NUM_SEQS):
        plen = int(torch.randint(MIN_PROMPT_LEN, MAX_PROMPT_LEN + 1, (1,), generator=gen).item())
        prompts.append(torch.randint(1, VOCAB_CEILING, (plen,), generator=gen).tolist())
        decodes.append(torch.randint(1, VOCAB_CEILING, (DECODE_TOKENS,), generator=gen).tolist())

    model = load_model(args.model_path, args.dtype, args.device_map)
    dev = input_device(model)

    prompt_flat: list[int] = []
    prompt_lens: list[int] = []
    ids_all, lp_all = [], []
    for prompt, decode in zip(prompts, decodes):
        prompt_lens.append(len(prompt))
        prompt_flat.extend(prompt)

        input_ids = torch.tensor([prompt], dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        logits_steps = [out.logits[0, -1].float()]

        for token in decode:
            input_ids = torch.tensor([[token]], dtype=torch.long, device=dev)
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            logits_steps.append(out.logits[0, -1].float())

        logprobs = F.log_softmax(torch.stack(logits_steps), dim=-1)
        ids_seq, lp_seq = [], []
        for pos in range(DECODE_TOKENS + 1):
            vals, idx = torch.topk(logprobs[pos], TOP_K)
            ids_seq.append(idx.tolist())
            lp_seq.append(vals.tolist())
        ids_all.append(ids_seq)
        lp_all.append(lp_seq)

    tensors = {
        "prompt_tokens": torch.tensor(prompt_flat, dtype=torch.int32),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.int32),
        "decode_tokens": torch.tensor(decodes, dtype=torch.int32),
        "topk_ids": torch.tensor(ids_all, dtype=torch.int32),
        "topk_logprobs": torch.tensor(lp_all, dtype=torch.float32),
    }
    model_path = Path(args.model_path)
    model_revision = args.model_revision or infer_revision(model_path)
    tokenizer_revision = args.tokenizer_revision or model_revision
    meta = {
        "fixture_kind": "hf-logits-golden",
        "model": "Qwen3.5-4B",
        "model_path": args.model_path,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "config_sha256": sha256_file(model_path / "config.json"),
        "dtype": args.dtype,
        "backend": "HuggingFace AutoModelForCausalLM eval, use_cache=True",
        "forward_path": "prompt prefill, then one-token teacher-forced decode through past_key_values",
        "seed": str(SEED),
        "top_k": str(TOP_K),
        "decode_tokens": str(DECODE_TOKENS),
        "num_seqs": str(NUM_SEQS),
        "prompt_len_min": str(MIN_PROMPT_LEN),
        "prompt_len_max": str(MAX_PROMPT_LEN),
        "vocab_ceiling": str(VOCAB_CEILING),
        "margin_tol": "0.20",
        "mean_tol": "0.06",
        "p99_tol": "0.20",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(
        f"wrote {out}: {NUM_SEQS} sequences, "
        f"{NUM_SEQS * (DECODE_TOKENS + 1)} positions, top{TOP_K}, {args.dtype}, "
        f"model_revision={model_revision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
