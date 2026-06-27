#!/usr/bin/env python
"""Build the GLM5.2 dense-MLP oracle for the first dense layer (layer 0), the
ground truth a Rust GPU test validates `fp8_mlp` against at the dense intermediate
size (12288 -- the MoE shared expert already gates it at 2048).

The dense MLP is a plain SwiGLU: `down(silu(gate(x)) * up(x))` with SEPARATE
gate/up projections (gate/up `[12288, 6144]`, down `[6144, 12288]`, all fp8 e4m3
block-scaled). The reference is computed the textbook way in float32 from the
dequantized weights -- identical to what HF `GlmMoeDsaMLP` computes, but without a
torch/HF dependency (the wiring invariant is what the GPU test checks; the fp8
kernel-vs-f32 gap is the floor).

Input is the seed-0 reference activation reused from the MoE probe (`hidden.bin` =
`torch.manual_seed(0); randn(1,8,6144)*0.1`), so the dense + MoE oracles share one
input. fp8 dequant matches the MLA/MoE convention: 128x128 multiplicative block
scale, ceil block count (layer-0 dense dims 12288/6144 are 128-exact).

Run on the H200 build node:

    uv run --no-project --with numpy --with ml_dtypes python -u \
        openinfer-kernels/tools/glm52/dense_mlp_prep.py
"""

import json
import os
import struct

import ml_dtypes
import numpy as np

MODEL = os.environ.get("GLM52_MODEL", "/data/models/GLM-5.2-FP8")
SRC = os.environ.get("GLM52_MOE_PROBE_DIR", "/data/models/glm52_mla_ref/moe_probe")
OUT = os.environ.get("GLM52_DENSE_PROBE_DIR", "/data/models/glm52_mla_ref/dense_probe")
LAYER = 0  # first dense layer (first_k_dense_replace=3 -> layers 0..2 are dense)
P = f"model.layers.{LAYER}.mlp."
HIDDEN, INTERMEDIATE = 6144, 12288

IDX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def load_raw(name):
    shard = IDX[name]
    with open(f"{MODEL}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        f.seek(8 + n + meta["data_offsets"][0])
        buf = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    return buf, meta["shape"], meta["dtype"]


def deq_fp8(name):
    """fp8 e4m3 weight * 128x128 block scale (multiplicative, ceil block count)."""
    wb, ws, dt = load_raw(name)
    assert dt == "F8_E4M3", f"{name} expected F8_E4M3 got {dt}"
    sb, _, sdt = load_raw(name + "_scale_inv")
    assert sdt == "F32"
    w = np.frombuffer(wb, ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(ws)
    bn, bk = (ws[0] + 127) // 128, (ws[1] + 127) // 128
    s = np.frombuffer(sb, np.float32).reshape(bn, bk)
    sfull = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[: ws[0], : ws[1]]
    return w * sfull


def main():
    hidden = np.fromfile(f"{SRC}/hidden.bin", np.float32).reshape(-1, HIDDEN)  # [8,6144]
    gate = deq_fp8(P + "gate_proj.weight")  # [12288, 6144]
    up = deq_fp8(P + "up_proj.weight")  # [12288, 6144]
    down = deq_fp8(P + "down_proj.weight")  # [6144, 12288]
    assert gate.shape == (INTERMEDIATE, HIDDEN), gate.shape
    assert down.shape == (HIDDEN, INTERMEDIATE), down.shape

    g = hidden @ gate.T  # [8, 12288]
    u = hidden @ up.T  # [8, 12288]
    act = (g / (1.0 + np.exp(-g))) * u  # silu(gate) * up
    out = act @ down.T  # [8, 6144]

    os.makedirs(OUT, exist_ok=True)
    hidden.astype(np.float32).tofile(f"{OUT}/hidden.bin")
    out.astype(np.float32).tofile(f"{OUT}/dense_mlp_output.bin")
    print(f"dense MLP oracle -> {OUT}")
    print(f"  hidden            {hidden.shape}")
    print(f"  dense_mlp_output  {out.shape}")
    print(f"  |out| max={np.abs(out).max():.5f} mean={np.abs(out).mean():.5f}")


if __name__ == "__main__":
    main()
