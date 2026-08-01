"""Adapter for quant.fp8_per_token_group_bf16 — shared FP8 per-token-group
activation quant with amax/448 f32 scales.

Production symbol: `glm52_fp8_per_token_group_quant_bf16_cuda`
(csrc/glm52/glm52_moe_quant.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs:286;
ops wrapper openinfer-kernels/src/ops/glm52/moe_quant.rs). Decode-path call
sites quantize the H=6144-wide bf16 GEMM inputs (mla_front q_a/kv_a, indexer
wq_b/wk, dense gate_up/down — all via fp8.rs large-m) and the prefill MoE TP
chunk; the indexer per-head q quant is the same kernel at hidden=128. The
unit pins the dominant decode width hidden=6144 (group=128 -> 48 scales/row).

rows: the kernel is row-agnostic — grid.x = min(rows, 256) with a grid-stride
row loop (the masked EP twin runs the same loop at 2080-row recv capacity),
so the full rows axis {1..64} is served by the existing kernel with ZERO .cu
change, no shape whitelist, and no scratch buffer. Per-row math is
rows-independent and deterministic.

Comparison surface: `out` packs [e4m3 bytes | raw LE f32 scale bytes] into one
flat uint8 buffer (the kernel writes both halves through views); the reference
returns the expected byte values as f32, so the manifest rel_l2 gate is a
bit-exactness gate — the RNE/div.rn argument lives in refs/quant.py.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import quant as quant_ref

SYMBOL = "glm52_fp8_per_token_group_quant_bf16_cuda"
GROUP_SIZE = quant_ref.GROUP_SIZE


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"}; hidden = k (quant is width-preserving)."""
    torch = require_torch()
    rows, hidden = shape["rows"], shape["k"]
    if hidden % GROUP_SIZE:
        raise ValueError(f"{SYMBOL}: hidden {hidden} not divisible by {GROUP_SIZE}")
    act = data.normal_bf16((rows, hidden), seed=data.derive_seed(seed, "act"))
    # Deterministic edge coverage: an all-zero group exercises the amax=0 ->
    # eps-clamped scale branch (kPerTokenGroupQuantEps / 448) on every run.
    act[0, :GROUP_SIZE] = 0.0
    value_bytes = rows * hidden
    packed = torch.empty(
        quant_ref.packed_surface_len(rows, hidden), dtype=torch.uint8, device=act.device
    )
    return {
        "act": act,                         # bf16 [rows, hidden]
        "out": packed,                      # uint8 flat comparison surface
        "out_bytes": packed[:value_bytes].view(rows, hidden),  # e4m3 u8 [rows, hidden]
        "scales": packed[value_bytes:]
        .view(torch.float32)
        .view(rows, hidden // GROUP_SIZE),  # f32 [rows, hidden/128]
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t). No scratch:
    the ABI is (input, output, scales, rows, hidden, group_size, stream)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # input bf16 [rows, hidden]
            ctypes.c_void_p,   # output e4m3 u8 [rows, hidden]
            ctypes.c_void_p,   # scales f32 [rows, hidden/128]
            ctypes.c_int,      # rows
            ctypes.c_int,      # hidden_dim
            ctypes.c_int,      # group_size (must be 128)
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["act"]),
        as_dev_ptr(tensors["out_bytes"]),
        as_dev_ptr(tensors["scales"]),
        shape["rows"],
        shape["k"],
        GROUP_SIZE,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    out, scales = quant_ref.fp8_per_token_group_quant_ref(tensors["act"], ue8m0=False)
    return quant_ref.pack_quant_surface(out, scales)
