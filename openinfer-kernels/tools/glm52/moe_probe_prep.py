#!/usr/bin/env python
"""Convert the GLM5.2 MoE layer-3 oracle npz into flat little-endian .bin probe
files the Rust MoE decode oracle tests read (mirrors flashmla_oracle_prep.py).

The npz itself is built by moe_oracle_prep.py (HF GlmMoeDsaMoE, torch CPU f32).
This step is numpy-only: it just re-emits the keys as raw f32/i32 blobs.

    uv run --no-project --with numpy python \
        openinfer-kernels/tools/glm52/moe_probe_prep.py

Env overrides: GLM52_MOE_ORACLE_NPZ (default /data/models/glm52_mla_ref/moe_layer3.npz),
GLM52_MOE_PROBE_DIR (default <npz dir>/moe_probe).
"""

import os
import pathlib

import numpy as np

NPZ = pathlib.Path(
    os.environ.get("GLM52_MOE_ORACLE_NPZ", "/data/models/glm52_mla_ref/moe_layer3.npz")
)
OUT = pathlib.Path(
    os.environ.get("GLM52_MOE_PROBE_DIR", str(NPZ.parent / "moe_probe"))
)

# key -> (dtype the Rust side reads). f32 for activations/weights, i32 for ids.
F32_KEYS = [
    "hidden",          # (1, 8, 6144) MoE block input
    "router_logits",   # (8, 256) post-sigmoid, pre-bias
    "gate_logits",     # (8, 256) raw gate projection, pre-sigmoid
    "e_score_correction_bias",  # (256,)
    "topk_weights",    # (8, 8) normalized x 2.5
    "shared_output",   # (8, 6144)
    "routed_output",   # (8, 6144)
    "moe_output",      # (8, 6144)
]
I32_KEYS = ["topk_indices"]  # (8, 8) selected expert ids (unsorted)


def main() -> None:
    data = np.load(NPZ)
    OUT.mkdir(parents=True, exist_ok=True)
    for key in F32_KEYS:
        arr = np.ascontiguousarray(data[key], dtype="<f4")
        (OUT / f"{key}.bin").write_bytes(arr.tobytes())
        print(f"{key:28s} {tuple(arr.shape)} f32 -> {key}.bin")
    for key in I32_KEYS:
        arr = np.ascontiguousarray(data[key], dtype="<i4")
        (OUT / f"{key}.bin").write_bytes(arr.tobytes())
        print(f"{key:28s} {tuple(arr.shape)} i32 -> {key}.bin")
    print(f"\nwrote MoE probe bins to {OUT}")


if __name__ == "__main__":
    main()
