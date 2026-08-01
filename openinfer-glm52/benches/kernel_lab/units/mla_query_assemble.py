"""Adapter for mla.query_assemble — the absorb-side FlashMLA query assembly.

Production symbol: `glm52_mla_query_assemble_cuda`
(csrc/glm52/glm52_mla_assembly.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs;
production call `glm52_mla_attend_into`, openinfer-glm52/src/mla_decode.rs:474).

Per (token, head) block: query[t,h,0:512] = ql_nope[t,h,:] (absorb GEMM
output, copied through) and query[t,h,512:576] = interleave-RoPE(q_pe) with
q_pe read IN PLACE from the q_b output at offset 192 / head stride 256 — the
exact production addressing. heads is the full 64 (EP8 FlashMLA path); the
kernel admits any num_q_heads <= 64 (attention-TP shards leave zero pad
slots), not swept here.

rows routing: none — grid is (64, rows) with no upper bound on rows, each
(t, h) block independent, so rows {16,32,64} (MTP span-mapped verify rows)
run the same kernel bit-identically per row. Zero .cu changes.

The output is bit-exact reproducible in torch f32 (see
kernel_lab.refs.mla_attention): the gate tolerance is effectively zero.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import mla_attention as mla

SYMBOL = "glm52_mla_query_assemble_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", ...} at production capacity (heads=64, all rows live).
    ctx is accepted and ignored — the kernel cost/semantics are ctx-free (the
    group sweep relabels the same shape per ctx tier for ledger alignment)."""
    torch = require_torch()
    rows = shape["rows"]
    ql_nope = data.normal_bf16((rows, mla.HEADS, mla.QK_NOPE), seed=data.derive_seed(seed, "ql_nope"))
    q_full = data.normal_bf16((rows, mla.HEADS, mla.Q_HEAD), seed=data.derive_seed(seed, "q_full"))
    cos, sin = mla.rotary_table(rows, data.derive_seed(seed, "rotary"), device=ql_nope.device)
    return {
        "ql_nope": ql_nope,   # bf16 [rows, 64, 512]
        "q_full": q_full,     # bf16 [rows, 64, 256]; q_pe = [.., 192:256]
        "cos": cos,           # bf16 [rows, 32]
        "sin": sin,           # bf16 [rows, 32]
        "out": torch.zeros((rows, mla.HEADS, mla.QUERY_DIM), dtype=torch.bfloat16, device=ql_nope.device),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # ql_nope bf16 [rows,64,512]
            ctypes.c_void_p,   # q_pe_base (q_full bf16 [rows,64,256])
            ctypes.c_int,      # q_pe_offset = 192
            ctypes.c_int,      # q_pe_head_stride = 256
            ctypes.c_int,      # num_q_heads = 64
            ctypes.c_void_p,   # cos bf16 [rows,32]
            ctypes.c_void_p,   # sin bf16 [rows,32]
            ctypes.c_void_p,   # query bf16 [rows,64,576]
            ctypes.c_int,      # tokens = rows
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["ql_nope"]),
        as_dev_ptr(tensors["q_full"]),
        mla.Q_PE_OFFSET,
        mla.Q_HEAD,
        mla.HEADS,
        as_dev_ptr(tensors["cos"]),
        as_dev_ptr(tensors["sin"]),
        as_dev_ptr(tensors["out"]),
        shape["rows"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    return mla.query_assemble_ref(
        tensors["ql_nope"], tensors["q_full"], tensors["cos"], tensors["sin"]
    )
