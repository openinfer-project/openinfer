"""Adapter for indexer.k_quant_cache — fp8 per-128-group quant + paged scatter.

Production symbol: `glm52_indexer_k_quant_and_cache_cuda`
(csrc/glm52/glm52_indexer.cu; ops wrapper
openinfer-kernels/src/ops/glm52/indexer.rs). Writes each row's bf16 k [128]
into the DeepGEMM block-split paged layout at slot_mapping[row]:
[block_size*128 fp8][block_size*4 f32 scale] per block, stride 8448 B with
the production block_size=64 (INDEX_CACHE_BLOCK).

The quant is deterministic end to end (amax/shuffle-max and one f32 division
are exact, the RNE fp8 cast is deterministic), so the check gate is BYTE
equality of the whole cache image (both sides zero-based — the kernel only
touches mapped slots, so this also catches stray writes). grid=(tokens, 1)
has no rows bound; the ctx axis only changes the cache geometry and slot
sampling, not the per-launch cost.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import indexer

SYMBOL = "glm52_indexer_k_quant_and_cache_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ctx"?} — ctx defaults to the middle axis stop."""
    torch = require_torch()
    rows = shape["rows"]
    ctx = indexer.ctx_of(shape)
    blocks = indexer.block_cols(ctx)
    k = data.normal_bf16(
        (rows, indexer.HEAD_DIM), seed=data.derive_seed(seed, "cache:k")
    )
    # Distinct non-negative global slots inside the cache (rows of one step
    # never share a slot); sampled on CPU for cross-machine determinism.
    gen = torch.Generator(device="cpu").manual_seed(data.derive_seed(seed, "cache:slots"))
    slot_mapping = torch.randperm(blocks * indexer.BLOCK_KV, generator=gen)[:rows].to(torch.int64).to(k.device)
    cache = torch.zeros(indexer.cache_bytes(ctx), dtype=torch.uint8, device=k.device)
    return {
        "k": k,
        "slot_mapping": slot_mapping,
        "cache": cache,
        "ctx": ctx,
        "out": None,  # set by reference(): the cache image itself
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (writes the cache in place)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # k bf16 [rows, 128]
            ctypes.c_void_p,   # indexer_cache u8
            ctypes.c_void_p,   # slot_mapping i64 [rows]
            ctypes.c_int,      # tokens == rows
            ctypes.c_int,      # head_dim == 128
            ctypes.c_int,      # quant_block_size == 128
            ctypes.c_int,      # cache_block_size == 64
            ctypes.c_int64,    # cache_block_stride_bytes == 8448
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["k"]),
        as_dev_ptr(tensors["cache"]),
        as_dev_ptr(tensors["slot_mapping"]),
        shape["rows"],
        indexer.HEAD_DIM,
        indexer.HEAD_DIM,  # quant_block_size == head_dim (one group per token)
        indexer.BLOCK_KV,
        indexer.cache_stride_bytes(),
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    torch = require_torch()
    want = indexer.k_quant_cache_ref(
        tensors["k"], tensors["slot_mapping"], indexer.block_cols(tensors["ctx"])
    )
    # Hard gate: byte-exact over the whole image (unwritten regions included).
    if not bool(torch.equal(tensors["cache"], want)):
        diff = (tensors["cache"] != want).nonzero()
        raise AssertionError(
            f"k_quant_cache: {diff.shape[0]} cache bytes differ; first at byte "
            f"{int(diff[0])} (slot region boundary may indicate layout drift)"
        )
    tensors["out"] = tensors["cache"]
    return want
