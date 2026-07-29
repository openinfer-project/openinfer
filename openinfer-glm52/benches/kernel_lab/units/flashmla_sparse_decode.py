"""Adapter for flashmla_sparse.decode — the FlashMLA sparse decode (sm_100f).

Production symbols (csrc/glm52/glm52_flashmla_sparse.cu; FFI mirror
openinfer-kernels/src/ffi/glm52/flashmla_sparse.rs; production call
`glm52_mla_attend_into`, openinfer-glm52/src/mla_decode.rs:536):
`glm52_flashmla_sparse_decode_launch_cuda` (bench target) +
`glm52_flashmla_sparse_decode_num_sm_parts_cuda` /
`glm52_flashmla_sparse_decode_metadata_cuda` (runtime queries / plan-time
scheduler metadata).

Launch contract (from the .cu launcher): batch = rows (<= 128 capacity), one
paged 656-byte fp8_ds_mla cache (page = 64 tokens, num_blocks = ctx/64), DSA
top-2048 indices per row (all valid at the long-ctx tiers — short ctx is not
benched), sm_scale = 0.0625, h_q = 64 fixed. num_sm_parts is queried from the
.so (SM count, <= 160); the sm-parts-dependent scratch is pre-allocated at
MAX_SM_PARTS = 160 capacity, and the tile-scheduler metadata is launched
lazily on the first run() — plan-time work exactly like production
(Glm52MlaSchedMetadata), so timed launches carry decode only.

rows routing: none — batch is a runtime scheduler parameter (no per-batch
template instantiation), so rows {16,32,64} (MTP span-mapped verify rows)
run the production kernel unchanged. Zero .cu changes.

Reference: f64 naive sparse attention over the same packed cache
(kernel_lab.refs.mla_attention.sparse_attention_ref_f64, ported from
glm52_sparse_mla.cu's reference kernel). The synthetic cache carries UE8M0
pow2 scales, so the sm100 e8m0 truncation is lossless; the 0.02 rel_l2 gate
covers bf16 store rounding + f32 split-KV accumulation reorder.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import mla_attention as mla

SYMBOL = "glm52_flashmla_sparse_decode_launch_cuda"
NUM_SM_PARTS_SYMBOL = "glm52_flashmla_sparse_decode_num_sm_parts_cuda"
METADATA_SYMBOL = "glm52_flashmla_sparse_decode_metadata_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ctx"?, ...} at production capacity (heads=64,
    topk=2048 fully valid — ctx >= 16384 always exceeds the DSA topk)."""
    torch = require_torch()
    rows = shape["rows"]
    ctx = int(shape.get("ctx", mla.DEFAULT_CTX))
    if ctx % mla.PAGE_TOKENS:
        raise ValueError(f"decode needs ctx % {mla.PAGE_TOKENS} == 0, got {ctx}")
    if not 1 <= rows <= mla.BATCH_CAPACITY:
        raise ValueError(f"decode batch {rows} out of 1..={mla.BATCH_CAPACITY}")
    q = data.normal_bf16((rows, mla.HEADS, mla.QUERY_DIM), seed=data.derive_seed(seed, "q"))
    cache = mla.packed_cache(ctx, seed, device=q.device)
    gen = torch.Generator(device="cpu").manual_seed(data.derive_seed(seed, "indices"))
    indices = torch.stack(
        [torch.randperm(ctx, generator=gen)[: mla.TOPK] for _ in range(rows)]
    ).to(torch.int32).to(q.device)
    capacity_splits = rows + mla.MAX_SM_PARTS
    return {
        "q": q,                    # bf16 [rows, 64, 576]
        "cache": cache,            # u8 [ctx * 656] fp8_ds_mla (UE8M0 pow2 scales)
        "indices": indices,        # i32 [rows, 2048] distinct valid slots
        "out": torch.zeros((rows, mla.HEADS, mla.KV_LORA), dtype=torch.bfloat16, device=q.device),
        "tile_meta": torch.zeros(mla.MAX_SM_PARTS * mla.SCHED_META_INTS, dtype=torch.int32, device=q.device),
        "num_splits": torch.zeros(rows + 1, dtype=torch.int32, device=q.device),
        "lse": torch.zeros(rows * mla.HEADS, dtype=torch.float32, device=q.device),
        "lse_accum": torch.zeros(capacity_splits * mla.HEADS, dtype=torch.float32, device=q.device),
        "o_accum": torch.zeros(capacity_splits * mla.HEADS * mla.KV_LORA, dtype=torch.float32, device=q.device),
        "_num_blocks": ctx // mla.PAGE_TOKENS,
        "_num_sm_parts": None,     # queried from the .so on first run
        "_metadata_ready": False,  # plan-time metadata launched on first run
    }


def _query_num_sm_parts(lib) -> int:
    fn = resolve(lib, NUM_SM_PARTS_SYMBOL, [ctypes.POINTER(ctypes.c_int)])
    out = ctypes.c_int(0)
    rc = fn(ctypes.byref(out))
    if rc != 0:
        raise KernelLaunchError(NUM_SM_PARTS_SYMBOL, rc)
    if not 1 <= out.value <= mla.MAX_SM_PARTS:
        raise KernelLaunchError(NUM_SM_PARTS_SYMBOL, -1)
    return out.value


def _ensure_metadata(lib, tensors: dict, rows: int, parts: int, stream) -> None:
    """Plan-time tile-scheduler metadata — production computes it once per
    (batch, topk, num_sm_parts) at model build; the harness launches it on
    the first run() so bench warmup absorbs it and timed launches carry the
    decode kernel only."""
    if tensors["_metadata_ready"]:
        return
    fn = resolve(
        lib,
        METADATA_SYMBOL,
        [
            ctypes.c_void_p,   # tile_scheduler_metadata i32 [num_sm_parts*8]
            ctypes.c_void_p,   # num_splits i32 [rows+1]
            ctypes.c_int,      # batch_size = rows
            ctypes.c_int,      # topk = 2048
            ctypes.c_int,      # num_sm_parts
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["tile_meta"]),
        as_dev_ptr(tensors["num_splits"]),
        rows,
        mla.TOPK,
        parts,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(METADATA_SYMBOL, rc)
    tensors["_metadata_ready"] = True


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production decode launch on `stream` (c_void_p cudaStream_t)."""
    rows = shape["rows"]
    parts = tensors.get("_num_sm_parts")
    if parts is None:
        parts = _query_num_sm_parts(lib)
        tensors["_num_sm_parts"] = parts
    _ensure_metadata(lib, tensors, rows, parts, stream)
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # q bf16 [rows,64,576]
            ctypes.c_void_p,   # packed_kv_cache u8 [num_blocks*64*656]
            ctypes.c_void_p,   # topk_indices i32 [rows,2048]
            ctypes.c_void_p,   # tile_scheduler_metadata i32
            ctypes.c_void_p,   # num_splits i32
            ctypes.c_void_p,   # out_latent bf16 [rows,64,512]
            ctypes.c_void_p,   # lse f32 [rows,64]
            ctypes.c_void_p,   # lse_accum f32 [(rows+parts)*64]
            ctypes.c_void_p,   # o_accum f32 [(rows+parts)*64*512]
            ctypes.c_int,      # batch_size = rows
            ctypes.c_int,      # num_blocks = ctx/64
            ctypes.c_int,      # topk = 2048
            ctypes.c_int,      # num_sm_parts
            ctypes.c_float,    # sm_scale = 0.0625
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["q"]),
        as_dev_ptr(tensors["cache"]),
        as_dev_ptr(tensors["indices"]),
        as_dev_ptr(tensors["tile_meta"]),
        as_dev_ptr(tensors["num_splits"]),
        as_dev_ptr(tensors["out"]),
        as_dev_ptr(tensors["lse"]),
        as_dev_ptr(tensors["lse_accum"]),
        as_dev_ptr(tensors["o_accum"]),
        rows,
        tensors["_num_blocks"],
        mla.TOPK,
        parts,
        mla.SM_SCALE,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    return mla.sparse_attention_ref_f64(
        tensors["q"], tensors["cache"], tensors["indices"], mla.SM_SCALE
    )
