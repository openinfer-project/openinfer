#!/usr/bin/env python
"""Build the GLM5.2 decode bookend oracle: the final RMSNorm + lm_head tail that
turns the last layer's hidden into vocab logits. (The embedding head is validated
in-Rust against the raw checkpoint rows -- a gather is exact, no oracle needed.)

Reference is computed in float32 from the bf16 weights, with the input hidden
rounded to bf16 first to match what the GPU kernels receive (rms_norm + the gemv
read bf16, accumulate f32). Reuses the seed-0 reference activation (`hidden.bin`)
as the stand-in last-layer hidden -- the wiring (norm then lm_head) is what the GPU
test checks.

GLM5.2 facts (config.json): rms_norm_eps=1e-5, lm_head.weight bf16 [154880, 6144],
model.norm.weight bf16 [6144], tie_word_embeddings=false (lm_head is its own
tensor). logits = norm(h) @ lm_head.T.

Run on the H200 build node:

    uv run --no-project --with numpy --with ml_dtypes python -u \
        openinfer-kernels/tools/glm52/bookend_prep.py
"""

import json
import os
import struct

import ml_dtypes
import numpy as np

MODEL = os.environ.get("GLM52_MODEL", "/data/models/GLM-5.2-FP8")
SRC = os.environ.get("GLM52_MOE_PROBE_DIR", "/data/models/glm52_mla_ref/moe_probe")
OUT = os.environ.get("GLM52_BOOKEND_PROBE_DIR", "/data/models/glm52_mla_ref/bookend_probe")
HIDDEN, VOCAB = 6144, 154880
EPS = 1e-5

IDX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def load_raw(name):
    shard = IDX[name]
    with open(f"{MODEL}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        f.seek(8 + n + meta["data_offsets"][0])
        buf = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    return buf, meta["shape"], meta["dtype"]


def bf16(name):
    b, sh, dt = load_raw(name)
    assert dt == "BF16", f"{name} expected BF16 got {dt}"
    return np.frombuffer(b, ml_dtypes.bfloat16).astype(np.float32).reshape(sh)


def to_bf16_f32(x):
    """Round an f32 array through bf16 (what the GPU kernels actually consume)."""
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def main():
    hidden = np.fromfile(f"{SRC}/hidden.bin", np.float32).reshape(-1, HIDDEN)  # [8,6144]
    hidden = to_bf16_f32(hidden)
    norm_w = bf16("model.norm.weight")  # [6144]
    lm_head = bf16("lm_head.weight")  # [154880, 6144]
    assert norm_w.shape == (HIDDEN,), norm_w.shape
    assert lm_head.shape == (VOCAB, HIDDEN), lm_head.shape

    var = np.mean(hidden.astype(np.float32) ** 2, axis=-1, keepdims=True)
    normed = hidden / np.sqrt(var + EPS) * norm_w  # [8,6144]
    normed_bf = to_bf16_f32(normed)
    logits = normed_bf @ lm_head.T  # [8,154880]

    os.makedirs(OUT, exist_ok=True)
    hidden.astype(np.float32).tofile(f"{OUT}/hidden.bin")
    normed.astype(np.float32).tofile(f"{OUT}/final_norm_output.bin")
    logits.astype(np.float32).tofile(f"{OUT}/logits.bin")
    print(f"bookend oracle -> {OUT}")
    print(f"  final_norm_output {normed.shape}  |x| max={np.abs(normed).max():.4f}")
    print(f"  logits            {logits.shape}  argmax(token0)={int(logits[0].argmax())}")


if __name__ == "__main__":
    main()
