"""Adapter for indexer.weights_proj — the indexer stage's two fp8 projections.

Production symbol: `glm52_fp8_weight_only_gemv_batched_cuda`
(csrc/glm52/glm52_moe_gemv.cu), launched twice per forward
(openinfer-glm52/src/indexer.rs:421-441, non-large_m decode path):
  q     = wq_b(q_resid)   [rows,2048] -> [rows,4096]
  k_raw = wk(hidden)      [rows,6144] -> [rows,128]
rows routing: 1 -> m=1 CUDA-core GEMV; 2 -> register tile (bit-parity);
4/8 -> mma_config dispatch between the register tile and single-tile
tensor-core mma (Blackwell batch-8: wq_b {ksplit=16, ntiles=1}, wk {48,1};
batch-4: wk {16,2}, wq_b off-table -> register tile). All mma routes need
caller-owned f32 k-slice scratch [ksplit, rows, n] per launch — one buffer
sized to the max of the two launches is reused (launches are stream-ordered).
rows 16/32/64 are BLOCKED: the multi-subtile mma table lists only q_b
(16384, 2048), so the indexer shapes fail closed with INVALID_VALUE and no
register-tile fallback exists past batch 8 (see manifest notes).
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import indexer

SYMBOL = "glm52_fp8_weight_only_gemv_batched_cuda"
KSPLIT_SYMBOL = "glm52_gemv_mma_ksplit_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k", "wk_n", "wk_k"} at production capacity."""
    torch = require_torch()
    rows = shape["rows"]
    q_resid = data.normal_bf16(
        (rows, indexer.Q_LORA), seed=data.derive_seed(seed, "act:q_resid")
    )
    hidden = data.normal_bf16(
        (rows, indexer.HIDDEN), seed=data.derive_seed(seed, "act:hidden")
    )
    wq_b, wq_b_scales = data.normal_quantized_fp8(
        indexer.WQ_B_N, indexer.WQ_B_K, seed=data.derive_seed(seed, "weight:wq_b")
    )
    wk, wk_scales = data.normal_quantized_fp8(
        indexer.WK_N, indexer.WK_K, seed=data.derive_seed(seed, "weight:wk")
    )
    return {
        "q_resid": q_resid,            # bf16 [rows, 2048]
        "hidden": hidden,              # bf16 [rows, 6144]
        "wq_b": wq_b,                  # e4m3 uint8 [4096, 2048]
        "wq_b_scales": wq_b_scales,    # f32 [32, 16]
        "wk": wk,                      # e4m3 uint8 [128, 6144]
        "wk_scales": wk_scales,        # f32 [1, 48]
        "q_out": torch.empty((rows, indexer.WQ_B_N), dtype=torch.bfloat16, device=q_resid.device),
        "k_out": torch.empty((rows, indexer.WK_N), dtype=torch.bfloat16, device=q_resid.device),
        "out": None,                   # set by reference(): cat(q_out, k_out)
        "scratch": None,               # allocated lazily when an mma route needs it
    }


def _query_ksplit(lib, rows: int, n: int, k: int) -> int:
    fn = resolve(
        lib,
        KSPLIT_SYMBOL,
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
    )
    out = ctypes.c_int(0)
    rc = fn(rows, n, k, ctypes.byref(out))
    if rc != 0:
        raise KernelLaunchError(KSPLIT_SYMBOL, rc)
    return out.value


def _ensure_scratch(lib, tensors: dict, rows: int):
    """f32 partial scratch for the mma routes: per-launch floats are
    ksplit*rows*n (the .cu launchers' bounds check); ksplit == 0 (register
    tile) means that launch takes NULL. One buffer sized to the larger launch
    is shared — the two launches are stream-ordered."""
    floats_wq = _query_ksplit(lib, rows, indexer.WQ_B_N, indexer.WQ_B_K) * rows * indexer.WQ_B_N
    floats_wk = _query_ksplit(lib, rows, indexer.WK_N, indexer.WK_K) * rows * indexer.WK_N
    floats = max(floats_wq, floats_wk)
    if floats == 0:
        return None, 0, 0
    buf = tensors.get("scratch")
    if buf is None or buf.numel() < floats:
        torch = require_torch()
        buf = torch.empty(floats, dtype=torch.float32, device=tensors["q_out"].device)
        tensors["scratch"] = buf
    return buf, floats_wq, floats_wk


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """The two projection launches (wq_b then wk) on `stream`."""
    scratch, floats_wq, floats_wk = _ensure_scratch(lib, tensors, shape["rows"])
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # activation (bf16 as raw bits)
            ctypes.c_void_p,   # weight e4m3
            ctypes.c_void_p,   # weight_scale f32
            ctypes.c_void_p,   # out bf16
            ctypes.c_void_p,   # scratch f32 (NULL on the register-tile route)
            ctypes.c_size_t,   # scratch_floats
            ctypes.c_int,      # batch == rows
            ctypes.c_int,      # n
            ctypes.c_int,      # k
            ctypes.c_void_p,   # stream
        ],
    )
    for act, weight, scales, out, floats, n, k in (
        (tensors["q_resid"], tensors["wq_b"], tensors["wq_b_scales"],
         tensors["q_out"], floats_wq, indexer.WQ_B_N, indexer.WQ_B_K),
        (tensors["hidden"], tensors["wk"], tensors["wk_scales"],
         tensors["k_out"], floats_wk, indexer.WK_N, indexer.WK_K),
    ):
        rc = fn(
            as_dev_ptr(act),
            as_dev_ptr(weight),
            as_dev_ptr(scales),
            as_dev_ptr(out),
            as_dev_ptr(scratch) if floats else None,
            floats,
            shape["rows"],
            n,
            k,
            stream,
        )
        if rc != 0:
            raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    torch = require_torch()
    tensors["out"] = torch.cat(
        [tensors["q_out"].reshape(-1), tensors["k_out"].reshape(-1)]
    )
    return indexer.projections_ref(
        tensors["q_resid"], tensors["hidden"],
        tensors["wq_b"], tensors["wq_b_scales"],
        tensors["wk"], tensors["wk_scales"],
    )
