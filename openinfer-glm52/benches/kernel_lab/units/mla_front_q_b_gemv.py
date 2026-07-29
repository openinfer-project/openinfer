"""Adapter for mla_front.q_b_gemv — the MLA-front q_b fp8 weight-only GEMV.

Production symbol: `glm52_fp8_weight_only_gemv_batched_cuda`
(csrc/glm52/glm52_moe_gemv.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs).
rows routing: 1 -> m=1 CUDA-core GEMV; 2 -> register tile (bit-parity with
m=1); 4/8 -> single-tile tensor-core mma on whitelisted shapes (q_b n=16384
k=2048 is one: Blackwell {ksplit=4, ntiles=1}, Hopper {8,4}) or the register
tile off-table; 16/32/64 (MTP span-mapped verify rows) -> multi-subtile mma
(BTILES = rows/8 column subtiles sharing each weight packet, Blackwell-only
{4,1} initial picks) with NO register-tile fallback — off-table fails closed
with CUDA_ERROR_INVALID_VALUE. All mma routes need caller-owned f32 k-slice
scratch [ksplit, rows, n]. Rows are deterministic per value but NOT
bit-identical across values — the torch-tolerance gate covers the reorder.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import fp8_gemv

SYMBOL = "glm52_fp8_weight_only_gemv_batched_cuda"
KSPLIT_SYMBOL = "glm52_gemv_mma_ksplit_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"} at production capacity (all rows live)."""
    torch = require_torch()
    rows, n, k = shape["rows"], shape["n"], shape["k"]
    act = data.normal_bf16((rows, k), seed=data.derive_seed(seed, "act"))
    weight, scales = data.normal_quantized_fp8(n, k, seed=data.derive_seed(seed, "weight"))
    return {
        "act": act,                      # bf16 [rows, k]
        "weight": weight,                # e4m3 uint8 [n, k]
        "scales": scales,                # f32 [n/128, k/128]
        "out": torch.empty((rows, n), dtype=torch.bfloat16, device=act.device),
        "scratch": None,                 # allocated lazily on the mma route
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


def _ensure_scratch(lib, tensors: dict, shape: dict):
    """f32 partial scratch [ksplit, rows, n] — the .cu mma path's caller-owned
    buffer; sized exactly per the launchers' bounds check
    (`KSPLIT * BATCH * n <= scratch_floats`, BATCH == rows). Register-tile
    routes (ksplit 0) get a NULL pointer, which the dispatch ignores; an
    mma-routed launch with NULL/short scratch fails INVALID_VALUE by design."""
    ksplit = _query_ksplit(lib, shape["rows"], shape["n"], shape["k"])
    if ksplit == 0:
        return None, 0
    torch = require_torch()
    floats = ksplit * shape["rows"] * shape["n"]
    buf = tensors.get("scratch")
    if buf is None or buf.numel() < floats:
        buf = torch.empty(floats, dtype=torch.float32, device=tensors["out"].device)
        tensors["scratch"] = buf
    return buf, floats


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    scratch, scratch_floats = _ensure_scratch(lib, tensors, shape)
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
    rc = fn(
        as_dev_ptr(tensors["act"]),
        as_dev_ptr(tensors["weight"]),
        as_dev_ptr(tensors["scales"]),
        as_dev_ptr(tensors["out"]),
        as_dev_ptr(scratch) if scratch is not None else None,
        scratch_floats,
        shape["rows"],
        shape["n"],
        shape["k"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    return fp8_gemv.fp8_weight_only_gemv_ref(
        tensors["act"], tensors["weight"], tensors["scales"], shape["n"], shape["k"]
    )
