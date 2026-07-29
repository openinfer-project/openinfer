"""Adapter for bookend.embed — the decode-step token embedding row gather.

Production symbol: `embedding_batched_cuda` (csrc/shared/elementwise.cu:632;
FFI mirror openinfer-kernels/src/ffi/shared.rs:116), reached in production via
glm52_embed_into (openinfer-glm52/src/bookend.rs:28) -> embedding_rows_into.
The kernel takes the vec4 uint4 path here (hidden 6144 % 8 == 0 and torch's
256B-aligned bases satisfy the 16B rule); ids are NOT bounds-checked, so the
factory draws them ~ U[0, 154880). Any rows value is legal — rows 16-64 are
the MTP span-mapped verify rows and need nothing new.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import bookends

SYMBOL = "embedding_batched_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n"=vocab, "k"=hidden}; the 1.9 GB table is cached
    process-wide (one generation per command, not per rows value)."""
    torch = require_torch()
    rows, vocab = shape["rows"], shape["n"]
    table = bookends.cached_table("embed_table", vocab, shape["k"], seed)
    gen = torch.Generator(device="cpu").manual_seed(data.derive_seed(seed, "token_ids"))
    ids = torch.randint(0, vocab, (rows,), generator=gen, dtype=torch.int64)
    # The kernel reads u32; ids < 2**31 so the i32 bit pattern is identical.
    ids = ids.to(torch.int32).to(table.device)
    return {
        "table": table,  # bf16 [vocab, hidden]
        "token_ids": ids,  # u32-pattern [rows]
        "out": torch.empty((rows, shape["k"]), dtype=torch.bfloat16, device=table.device),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,  # embed table bf16
            ctypes.c_void_p,  # token_ids u32
            ctypes.c_void_p,  # out bf16
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # seq_len == rows
            ctypes.c_void_p,  # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["table"]),
        as_dev_ptr(tensors["token_ids"]),
        as_dev_ptr(tensors["out"]),
        shape["k"],
        shape["rows"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    return bookends.embedding_gather_ref(tensors["table"], tensors["token_ids"])
