"""Adapter for bookend.argmax_split — the two-stage device-side vocab argmax.

Production symbol: `argmax_batch_bf16_split_cuda` (csrc/shared/argmax.cu:373;
FFI mirror openinfer-kernels/src/ffi/shared.rs:779 — void return, so the
adapter pins restype=None and launch faults surface at the harness's
synchronize instead of an rc). Same call the decode step makes
(openinfer-glm52/src/model/step_body.rs:214 via argmax_bf16_split_into):
per-4096-tile partials then one finalize block per row; partials carry global
indices, so each row keeps the single-scan total order — greater wins, ties to
the LOWER global index, NaN never wins (verified in argmax.cu:14).

Test data is adversarial by construction (refs/bookends.py argmax_layout):
every row gets an exact bf16 tie (1024.0 at p and at (p+77440)%vocab — the
offsets are 18.9/9.45 tiles, so the maxima always sit in DIFFERENT 4096-wide
tiles and a tile-local tie bug flips the answer) plus one NaN at
(p+38720)%vocab. The gated output `out` is the i32 indices [rows]; the
reference is the explicit lowest-index-of-max reduction (torch.argmax's tie
order is unspecified).
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import as_dev_ptr, require_torch, resolve
from kernel_lab.refs import bookends

SYMBOL = "argmax_batch_bf16_split_cuda"
TILE_ELEMS = 4096  # ARGMAX_BATCH_TILE_ELEMS (argmax.cu) — partials sizing rule


def partials_len(rows: int, vocab: int) -> int:
    """argmax_batch_bf16_split_partials_len: rows * ceil(vocab / 4096)."""
    return rows * ((vocab + TILE_ELEMS - 1) // TILE_ELEMS)


def make_inputs(shape: dict, seed: int) -> dict:
    torch = require_torch()
    rows, vocab = shape["rows"], shape["n"]
    logits = data.normal_bf16((rows, vocab), seed=data.derive_seed(seed, "logits"))
    gen = torch.Generator(device="cpu").manual_seed(data.derive_seed(seed, "argmax_layout"))
    p = torch.randint(0, vocab, (rows,), generator=gen, dtype=torch.int64)
    q = (p + bookends.ARGMAX_TIE_OFFSET) % vocab
    z = (p + bookends.ARGMAX_NAN_OFFSET) % vocab
    r = torch.arange(rows, dtype=torch.int64)
    logits[r, p] = bookends.ARGMAX_TIE_VALUE
    logits[r, q] = bookends.ARGMAX_TIE_VALUE  # exact bf16 tie across tiles
    logits[r, z] = float("nan")               # must never win
    dev = logits.device
    return {
        "logits": logits,  # bf16 [rows, vocab]
        "partial_values": torch.empty(partials_len(rows, vocab), dtype=torch.float32, device=dev),
        "partial_indices": torch.empty(partials_len(rows, vocab), dtype=torch.int32, device=dev),
        "values": torch.empty(rows, dtype=torch.bfloat16, device=dev),
        "out": torch.empty(rows, dtype=torch.int32, device=dev),  # gated indices
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,  # x logits bf16
            ctypes.c_void_p,  # values bf16 [rows]
            ctypes.c_void_p,  # indices i32 [rows]
            ctypes.c_void_p,  # partial_values f32 [rows * tiles]
            ctypes.c_void_p,  # partial_indices i32 [rows * tiles]
            ctypes.c_int,     # rows
            ctypes.c_int,     # n = vocab
            ctypes.c_void_p,  # stream
        ],
    )
    fn.restype = None  # void C ABI — no CUresult to check
    fn(
        as_dev_ptr(tensors["logits"]),
        as_dev_ptr(tensors["values"]),
        as_dev_ptr(tensors["out"]),
        as_dev_ptr(tensors["partial_values"]),
        as_dev_ptr(tensors["partial_indices"]),
        shape["rows"],
        shape["n"],
        stream,
    )


def reference(tensors: dict, shape: dict):
    return bookends.argmax_lowest_index_ref(tensors["logits"])
