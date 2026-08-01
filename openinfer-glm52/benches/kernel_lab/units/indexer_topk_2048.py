"""Adapter for indexer.topk_2048 — FlashInfer deterministic top-k K=2048.

Production symbol: `glm52_flashinfer_topk_2048_cuda`
(csrc/glm52/glm52_topk.cu — FilteredTopK(deterministic, TopKTieBreak::Small,
dsa_graph_safe) + LaunchSortTopKByIndex; FFI mirror
openinfer-kernels/src/ffi/glm52/topk.rs). Indices come out sorted ascending
by index; values follow.

Harness input mirrors production: the valid region holds bf16-quantized
values upcast to f32 (DeepGEMM emits bf16 logits, then a cast kernel feeds
this top-k), and columns [lengths[r], max_len) are filled with 1e30 stale
garbage — if the per-row `lengths` filter regressed, the tail would win
(glm52_indexer_smoke.rs's regression pattern). bf16 quantization also makes
exact f32 value ties plentiful, which is what exercises the tie-break rule:
TopKTieBreak::Small picks the smaller index; the torch reference reproduces
it with a stable descending sort, and the hard gate allows at most one
tie-only set divergence per row (oracle-harness discipline), with value
multisets compared exactly.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import indexer

SYMBOL = "glm52_flashinfer_topk_2048_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ctx"?} — ctx defaults to the middle axis stop;
    max_len = ctx + 256 keeps the production stale-tail geometry."""
    torch = require_torch()
    rows = shape["rows"]
    ctx = indexer.ctx_of(shape)
    max_len = indexer.topk_max_len(ctx)
    lens = indexer.seq_lens_for_rows(rows, ctx)
    logits = data.normal_bf16(
        (rows, max_len), seed=data.derive_seed(seed, "topk:logits")
    ).to(torch.float32)
    # Negative control: stale columns (incl. the trailing 256) must be filtered.
    col = torch.arange(max_len, device=logits.device)
    stale = col[None, :] >= torch.tensor(lens, device=logits.device)[:, None]
    logits[stale] = 1.0e30
    return {
        "logits": logits,
        "lengths": torch.tensor(lens, dtype=torch.int32, device=logits.device),
        "indices": torch.empty((rows, indexer.TOPK), dtype=torch.int32, device=logits.device),
        "values": torch.empty((rows, indexer.TOPK), dtype=torch.float32, device=logits.device),
        "max_len": max_len,
        "out": None,  # set by reference(): value-sorted kernel values
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream`."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # logits f32 [rows, max_len]
            ctypes.c_void_p,   # output_indices i32 [rows, top_k]
            ctypes.c_void_p,   # output_values f32 [rows, top_k]
            ctypes.c_void_p,   # lengths i32 [rows]
            ctypes.c_int,      # num_rows
            ctypes.c_int,      # top_k
            ctypes.c_int,      # max_len
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["logits"]),
        as_dev_ptr(tensors["indices"]),
        as_dev_ptr(tensors["values"]),
        as_dev_ptr(tensors["lengths"]),
        shape["rows"],
        indexer.TOPK,
        tensors["max_len"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    ref_i, ref_v = indexer.topk_ref(tensors["logits"], tensors["lengths"], indexer.TOPK)
    # Hard gate: index sets (tie rule) + value multisets, exact.
    indexer.assert_topk_match(
        tensors["indices"], tensors["values"], ref_i, ref_v,
        tensors["logits"], indexer.TOPK,
    )
    # Soft metric on value-sorted values (invariant under tie swaps).
    tensors["out"] = tensors["values"].sort(dim=1).values
    return ref_v.sort(dim=1).values
