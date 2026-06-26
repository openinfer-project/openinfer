#!/usr/bin/env python
"""Pack the GLM5.2 layer-0 MLA oracle into raw bins for the FlashMLA sparse decode
integration gate (`openinfer-kernels/tests/glm52_flashmla_sparse_oracle.rs`).

Run on the H200 build node (numpy + ml_dtypes only; no torch):

    uv run --no-project --with numpy --with ml_dtypes python \
        openinfer-kernels/tools/glm52/flashmla_oracle_prep.py

Input: an HF GLM5.2 layer-0 forward dump `$IN` (default
`/data/models/glm52_mla_ref/layer0.npz`) with these tensors (B=1, S=8 tokens,
H=64 heads), all float32:

    kv_c          (1, 8, 512)    compressed-KV NoPE per token (the cache ckv)
    k_rot         (1, 1, 8, 64)  post-rope k_pe per token (h_kv=1)
    q_rot         (1, 64, 8, 64) post-rope q_pe per head/token
    ql_nope       (1, 64, 8, 512) absorbed q_nope = q_pass @ W_UK
    latent        (1, 64, 8, 512) FlashMLA latent output per head/token (oracle)
    value_states  (1, 64, 8, 256) v per head/token (for the later v_up stage)
    o             (1, 8, 6144)    final o_proj output (for the later o_proj stage)

Output bins under `$OUT` (default `/data/models/glm52_mla_ref/flashmla_probe`) for
decode token T=7 attending to the 8-token cache, packed with the *verified*
fp8_ds_mla 656-byte convention (512 e4m3 ckv + 4 f32 group scales (amax/448) + 64
bf16 rope-key; see the test's module doc and docs/models/glm52/flashmla-sparse-wrapper.md).
"""

import os

import ml_dtypes
import numpy as np

IN = os.environ.get("GLM52_MLA_ORACLE", "/data/models/glm52_mla_ref/layer0.npz")
OUT = os.environ.get("GLM52_FLASHMLA_PROBE_DIR", "/data/models/glm52_mla_ref/flashmla_probe")
T, NCTX = 7, 8
SM_SCALE = np.float32(0.0625)  # 256**-0.5; FlashMLA applies it internally
FP8_MAX = np.float32(448.0)
FLT_MIN = np.float32(np.finfo(np.float32).tiny)


def pack_token(ckv, kpe):
    """(ckv[512] f32, kpe[64] f32 rope-applied) -> 656 u8 fp8_ds_mla token."""
    ckv = ckv.astype(np.float32)
    kpe = kpe.astype(np.float32)
    fp8 = np.empty(512, np.uint8)
    scales = np.empty(4, np.float32)
    for g in range(4):
        tile = ckv[g * 128:(g + 1) * 128]
        scale = np.maximum(np.max(np.abs(tile)).astype(np.float32) / FP8_MAX, FLT_MIN)
        scales[g] = scale
        fp8[g * 128:(g + 1) * 128] = (tile / scale).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    out = np.empty(656, np.uint8)
    out[0:512] = fp8
    out[512:528] = scales.view(np.uint8)
    out[528:656] = kpe.astype(ml_dtypes.bfloat16).view(np.uint8)
    return out


def dequant_token(tok):
    """656 u8 -> (ckv[512] f32, kpe[64] f32) exactly as the kernel reads it
    (the kernel down-casts the f32 group scale to bf16 before the multiply)."""
    fp8 = tok[0:512].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scales = tok[512:528].view(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    ckv = np.empty(512, np.float32)
    for g in range(4):
        ckv[g * 128:(g + 1) * 128] = fp8[g * 128:(g + 1) * 128] * scales[g]
    kpe = tok[528:656].view(ml_dtypes.bfloat16).astype(np.float32)
    return ckv, kpe


def main():
    z = np.load(IN)
    os.makedirs(OUT, exist_ok=True)

    # cache [1 block, 64 tokens, 656]
    cache = np.zeros((64, 656), np.uint8)
    for t in range(NCTX):
        cache[t] = pack_token(z["kv_c"][0, t, :], z["k_rot"][0, 0, t, :])
    cache.tofile(f"{OUT}/cache.bin")

    # query [64, 576] bf16 = [ql_nope(512) | q_pe(64)] for token T
    ql_nope = z["ql_nope"][0, :, T, :]
    q_rot = z["q_rot"][0, :, T, :]
    query = np.concatenate([ql_nope, q_rot], axis=1).astype(ml_dtypes.bfloat16)
    query.view(np.uint16).tofile(f"{OUT}/query.bin")
    q_f32 = query.astype(np.float32)

    # topk [2048] i32: 0..NCTX-1 then -1 (V3.2 has no dynamic length; -1 is skipped)
    topk = np.full(2048, -1, np.int32)
    topk[:NCTX] = np.arange(NCTX, dtype=np.int32)
    topk.tofile(f"{OUT}/topk.bin")

    # tight reference: numpy attention over the SAME dequantized fp8 cache
    ckv_deq = np.stack([dequant_token(cache[t])[0] for t in range(NCTX)])  # [8,512]
    kpe_deq = np.stack([dequant_token(cache[t])[1] for t in range(NCTX)])  # [8,64]
    key = np.concatenate([ckv_deq, kpe_deq], axis=1)  # [8,576]
    scores = (q_f32 @ key.T) * SM_SCALE  # [64,8]
    scores -= scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=1, keepdims=True)
    latent_fp8ref = w @ ckv_deq  # [64,512]
    latent_fp8ref.astype(np.float32).tofile(f"{OUT}/latent_fp8ref.bin")

    # full-precision oracle + later-stage refs
    lat_oracle = z["latent"][0, :, T, :].astype(np.float32)
    lat_oracle.tofile(f"{OUT}/latent_expected.bin")
    z["value_states"][0, :, T, :].astype(np.float32).tofile(f"{OUT}/value_expected.bin")
    z["o"][0, T, :].astype(np.float32).tofile(f"{OUT}/o_expected.bin")

    d = np.abs(latent_fp8ref - lat_oracle)
    print(f"latent |max|={np.abs(lat_oracle).max():.5f} mean={np.abs(lat_oracle).mean():.5f}")
    print(f"fp8 noise floor (fp8ref vs oracle): max|d|={d.max():.6f} mean|d|={d.mean():.6f}")
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
