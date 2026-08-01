"""Adapter for indexer.local_topk_to_slots — local top-k offsets to KV slots.

Production symbol: `glm52_indexer_local_topk_to_slots_cuda`
(csrc/glm52/glm52_indexer.cu — ported from TokenSpeed's Triton
_local_topk_to_global_slots_kernel; ops wrapper
openinfer-kernels/src/ops/glm52/indexer.rs). slot =
block_table[t, off//64]*64 + off%64 for 0 <= off < seq_len, else -1;
topk_lens counts the valid picks. Pure integer remap -> the check gate is
exact equality (slots and topk_lens).

Stride contract pinned by the P1 regression in glm52_indexer_smoke.rs:
local_topk_stride == topk and block_table_stride == block_table_cols (NOT
topk). Inputs mimic the real producer: per-row ascending offsets sampled
from [0, seq_len) (the SortTopKByIndex output), and per-row block tables
that are seeded permutations of each row's own page range.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import indexer

SYMBOL = "glm52_indexer_local_topk_to_slots_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ctx"?} — ctx defaults to the middle axis stop."""
    torch = require_torch()
    rows = shape["rows"]
    ctx = indexer.ctx_of(shape)
    lens = indexer.seq_lens_for_rows(rows, ctx)
    # Per-row ascending unique offsets in [0, len_r) — the SortTopKByIndex
    # output shape. Sampled on CPU with derived per-row seeds.
    offsets = torch.empty((rows, indexer.TOPK), dtype=torch.int64)
    for r in range(rows):
        gen = torch.Generator(device="cpu").manual_seed(
            data.derive_seed(seed, f"slots:offsets:{r}")
        )
        offsets[r] = torch.randperm(lens[r], generator=gen)[: indexer.TOPK].sort().values
    block_table = indexer.block_table_for(
        rows, ctx, data.derive_seed(seed, "slots:table"), "cuda"
    )
    return {
        "offsets": offsets.to(torch.int32).to("cuda"),
        "seq_lens": torch.tensor(lens, dtype=torch.int32, device="cuda"),
        "block_table": block_table,
        "slots": torch.empty((rows, indexer.TOPK), dtype=torch.int32, device="cuda"),
        "lens": torch.empty(rows, dtype=torch.int32, device="cuda"),
        "out": None,  # set by reference(): the global_slots image
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream`."""
    cols = tensors["block_table"].shape[1]
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # global_slots i32 (mut)
            ctypes.c_void_p,   # topk_lens i32 (mut)
            ctypes.c_void_p,   # local_topk_offsets i32
            ctypes.c_int,      # local_topk_stride (== topk)
            ctypes.c_void_p,   # seq_lens i32
            ctypes.c_void_p,   # block_table i32
            ctypes.c_int,      # block_table_stride (== cols, NOT topk)
            ctypes.c_int,      # block_table_cols
            ctypes.c_int,      # block_size
            ctypes.c_int,      # topk
            ctypes.c_int,      # num_tokens == rows
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["slots"]),
        as_dev_ptr(tensors["lens"]),
        as_dev_ptr(tensors["offsets"]),
        indexer.TOPK,  # local_topk_stride
        as_dev_ptr(tensors["seq_lens"]),
        as_dev_ptr(tensors["block_table"]),
        cols,  # block_table_stride == block_table_cols (P1 regression)
        cols,
        indexer.BLOCK_KV,
        indexer.TOPK,
        shape["rows"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    torch = require_torch()
    want_slots, want_lens = indexer.local_topk_to_slots_ref(
        tensors["offsets"], tensors["seq_lens"], tensors["block_table"]
    )
    # Hard gate: exact integer equality on both outputs.
    if not bool(torch.equal(tensors["slots"], want_slots)):
        bad = (tensors["slots"] != want_slots).nonzero()[:4]
        raise AssertionError(f"local_topk_to_slots: slot mismatches at {bad.tolist()}")
    if not bool(torch.equal(tensors["lens"], want_lens)):
        raise AssertionError(
            f"local_topk_to_slots: topk_lens {tensors['lens'].tolist()} != {want_lens.tolist()}"
        )
    tensors["out"] = tensors["slots"]
    return want_slots
