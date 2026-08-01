"""Adapter for indexer.rope — the indexer half-split RoPE (q per-head + k).

Production symbol: `glm52_indexer_rope_cuda`
(csrc/glm52/glm52_indexer_rope.cu; FFI mirror
openinfer-kernels/src/ffi/glm52/indexer_rope.rs). In-place on q
[rows, 32, 128] and k [rows, 128]; cos/sin carry one [32] row per token.
NON-interleaved (half-split / NeoX-style) pairing — the .cu and
openinfer-glm52/src/layer.rs:105 both pin this for the indexer despite
`indexer_rope_interleave=true` in the config; the torch reference mirrors the
kernel. grid=(n_heads, tokens) with no token bound, so the full rows axis is
measurable; ctx-independent (O(1) per token), no ctx axis.
"""
from __future__ import annotations

import ctypes
import math

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import indexer

SYMBOL = "glm52_indexer_rope_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows"}; q/k rotate in place, so keep pre-rotation clones
    for the reference (memory is trivial: rows*4224 bf16)."""
    torch = require_torch()
    rows = shape["rows"]
    q = data.normal_bf16(
        (rows, indexer.INDEX_HEADS, indexer.HEAD_DIM),
        seed=data.derive_seed(seed, "rope:q"),
    )
    k = data.normal_bf16(
        (rows, indexer.HEAD_DIM), seed=data.derive_seed(seed, "rope:k")
    )
    # Realistic cos/sin rows: angles ~ U(0, 2pi), computed in f32 then bf16
    # (the production table is bf16 too).
    gen = torch.Generator(device="cpu").manual_seed(data.derive_seed(seed, "rope:angle"))
    ang = torch.rand((rows, indexer.ROPE_HALF), generator=gen, dtype=torch.float32) * (2.0 * math.pi)
    cos = ang.cos().to(torch.bfloat16).to(q.device)
    sin = ang.sin().to(torch.bfloat16).to(q.device)
    return {
        "q": q,
        "k": k,
        "cos": cos,
        "sin": sin,
        "q_in": q.clone(),
        "k_in": k.clone(),
        "out": None,  # set by reference(): cat(q, k) post-rotation
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (rotates q and k in place)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # q bf16 [rows, heads, 128] (in-place)
            ctypes.c_void_p,   # k bf16 [rows, 128] (in-place)
            ctypes.c_int,      # n_heads
            ctypes.c_int,      # tokens == rows
            ctypes.c_void_p,   # cos bf16 [rows, 32]
            ctypes.c_void_p,   # sin bf16 [rows, 32]
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["q"]),
        as_dev_ptr(tensors["k"]),
        indexer.INDEX_HEADS,
        shape["rows"],
        as_dev_ptr(tensors["cos"]),
        as_dev_ptr(tensors["sin"]),
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    torch = require_torch()
    tensors["out"] = torch.cat(
        [tensors["q"].reshape(-1), tensors["k"].reshape(-1)]
    )
    q_ref, k_ref = indexer.rope_ref(
        tensors["q_in"], tensors["k_in"], tensors["cos"], tensors["sin"]
    )
    return torch.cat([q_ref.reshape(-1), k_ref.reshape(-1)])
