#!/usr/bin/env python3
"""Generate a tiny HuggingFace remote-code golden for Qwen3-4B-DFlash-b16.

The DFlash crate compares its standalone draft forward against this fixture
without importing Python at Rust test time. The input tensors are synthetic but
seed-pinned, so the fixture exercises the exact `DFlashDraftModel.forward`
contract: selected target hidden states, noise embeddings, and absolute
position ids.

    .venv/bin/python tools/accuracy/dump_qwen3_4b_dflash_hf_golden.py \
        --model-path /home/hezhaozhao/models/Qwen3-4B-DFlash-b16 \
        --out test_data/qwen3-4b-dflash-hf-golden.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel

SEED = 0xD4A5_4B16
CTX_LEN = 2
Q_LEN = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/home/hezhaozhao/models/Qwen3-4B-DFlash-b16")
    parser.add_argument("--out", default="test_data/qwen3-4b-dflash-hf-golden.safetensors")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate the DFlash bf16 golden")

    model = AutoModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    gen = torch.Generator(device="cuda").manual_seed(SEED)
    hidden = model.config.hidden_size
    target_layers = len(model.target_layer_ids)
    noise_embedding = torch.randn(
        (1, Q_LEN, hidden),
        generator=gen,
        device="cuda",
        dtype=torch.bfloat16,
    )
    target_hidden = torch.randn(
        (1, CTX_LEN, hidden * target_layers),
        generator=gen,
        device="cuda",
        dtype=torch.bfloat16,
    )
    position_ids = torch.arange(CTX_LEN + Q_LEN, device="cuda", dtype=torch.int64).unsqueeze(0)

    with torch.inference_mode():
        output = model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids,
            use_cache=False,
            is_causal=False,
        )
    torch.cuda.synchronize()

    tensors = {
        "noise_embedding": noise_embedding.cpu(),
        "target_hidden": target_hidden.cpu(),
        "position_ids": position_ids.to(torch.int32).cpu(),
        "output": output.cpu(),
    }
    meta = {
        "model_path": args.model_path,
        "seed": str(SEED),
        "ctx_len": str(CTX_LEN),
        "q_len": str(Q_LEN),
        "hidden_size": str(hidden),
        "target_layer_ids": ",".join(str(layer) for layer in model.target_layer_ids),
        "block_size": str(model.block_size),
        "mask_token_id": str(model.mask_token_id),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out}: ctx_len={CTX_LEN}, q_len={Q_LEN}, hidden={hidden}, seed={SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
