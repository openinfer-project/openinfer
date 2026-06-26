#!/usr/bin/env python
"""Numpy cross-check of the GLM5.2 MLA decode FRONT-HALF math against the HF oracle,
on the real fp8 checkpoint. Every intermediate matches the oracle exactly (max|d|~0),
so this pins the exact projection/rope/absorption math the Rust forward must
reproduce. Companion to the GPU gates glm52_flashmla_sparse_oracle.rs (query/cache
-> latent) and mla_decode_oracle.rs (latent -> o); this covers hidden -> query/cache.

Run on the build node (no torch needed):
    uv run --no-project --with numpy --with ml_dtypes python \
        openinfer-kernels/tools/glm52/front_half_oracle_check.py

Findings it locks down (see docs/models/glm52 and the project memory):
  - `hidden` (oracle) IS the post-input_layernorm attn-module input: q_a_proj is
    applied directly, no extra layernorm.
  - q path: rms(q_a_proj(hidden), q_a_layernorm, eps=1e-5) -> q_b_proj ->
    reshape[64,256] -> q_pass=[:,:192], q_pe=[:,192:256].
  - kv path: kv_a_proj_with_mqa(hidden)[576] -> compressed_kv=[:512], k_pe=[512:];
    kv_c = rms(compressed_kv, kv_a_layernorm, eps=1e-5). k_pe is the RAW scaled
    kv_a tail — NO extra norm (k_pe == oracle k_rot_raw exactly).
  - absorb: ql_nope = einsum("hp,hpl->hl", q_pass, W_UK), W_UK = kv_b[:,:192,:].
  - rope (interleave): cat([x1*cos[:32]-x2*sin[:32], x2*cos[:32]+x1*sin[:32]]),
    x1=even, x2=odd; applied to q_pe->q_rot and k_pe->k_rot (both exact).
  - PARTIAL-BLOCK fp8 dequant gotcha: kv_a_proj_with_mqa is [576,6144] — the last
    scale-block row group (512..575, the k_pe) is partial. Host dequant MUST
    ceil-divide rows ((N+127)//128); `N//128` silently drops the k_pe scale and
    yields ~651 garbage. (kv_b [28672,512] is 128-exact, so the Rust kv_b dequant
    is unaffected; the trtllm linear handles partial blocks internally.)
"""

import json
import struct

import ml_dtypes
import numpy as np

MODEL = "/data/models/GLM-5.2-FP8"
ORACLE = "/data/models/glm52_mla_ref/layer0.npz"
P = "model.layers.0.self_attn."
T = 7
IDX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def load_raw(name):
    shard = IDX[name]
    with open(f"{MODEL}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        f.seek(8 + n + meta["data_offsets"][0])
        buf = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    return buf, meta["shape"]


def deq_fp8(name):
    wb, ws = load_raw(name)
    sb, _ = load_raw(name + "_scale_inv")
    w = np.frombuffer(wb, ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(ws)
    s = np.frombuffer(sb, np.float32).reshape((ws[0] + 127) // 128, (ws[1] + 127) // 128)
    for bi in range((ws[0] + 127) // 128):  # ceil: cover the partial last block
        for bj in range((ws[1] + 127) // 128):
            w[bi * 128:(bi + 1) * 128, bj * 128:(bj + 1) * 128] *= s[bi, bj]
    return w


def bf16w(name):
    b, sh = load_raw(name)
    return np.frombuffer(b, ml_dtypes.bfloat16).astype(np.float32).reshape(sh)


def rms(x, w, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope_il(x, cos, sin):
    d = x.shape[-1]
    c, s = cos[:d // 2], sin[:d // 2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def cmp(name, a, b):
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    print(f"{name:22s} max|d|={np.abs(a - b).max():.6f} mean={np.abs(a - b).mean():.7f} sig={np.abs(b).max():.5f}")


def main():
    z = np.load(ORACLE)
    hidden = z["hidden"][0, T, :].astype(np.float32)

    q_resid = rms(hidden @ deq_fp8(P + "q_a_proj.weight").T, bf16w(P + "q_a_layernorm.weight"))
    q = (q_resid @ deq_fp8(P + "q_b_proj.weight").T).reshape(64, 256)
    q_pass, q_pe = q[:, :192], q[:, 192:256]

    ckv = hidden @ deq_fp8(P + "kv_a_proj_with_mqa.weight").T
    compressed_kv, k_pe = ckv[:512], ckv[512:]
    kv_c = rms(compressed_kv, bf16w(P + "kv_a_layernorm.weight"))

    kv_b = deq_fp8(P + "kv_b_proj.weight").reshape(64, 448, 512)
    ql_nope = np.einsum("hp,hpl->hl", q_pass, kv_b[:, :192, :])

    cos, sin = z["cos"][0, T, :], z["sin"][0, T, :]
    cmp("q_resid", q_resid, z["q_resid"][0, T, :])
    cmp("kv_c", kv_c, z["kv_c"][0, T, :])
    cmp("q_pass", q_pass, z["q_pass"][0, :, T, :])
    cmp("ql_nope", ql_nope, z["ql_nope"][0, :, T, :])
    cmp("k_pe==k_rot_raw", k_pe, z["k_rot_raw"][0, T, :])
    cmp("rope(q_pe)->q_rot", rope_il(q_pe, cos, sin), z["q_rot"][0, :, T, :])
    cmp("rope(k_pe)->k_rot", rope_il(k_pe[None, :], cos, sin)[0], z["k_rot"][0, 0, T, :])


if __name__ == "__main__":
    main()
