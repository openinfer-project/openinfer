"""Adapter for indexer.mqa_logits — DeepGEMM paged fp8 MQA logits (decode).

Production symbols: `glm52_deepgemm_paged_mqa_metadata_cuda` +
`glm52_deepgemm_paged_mqa_logits_cuda` (csrc/glm52/glm52_deepgemm_mqa.cu,
AOT-instantiated from DeepGEMM device headers — no runtime JIT). One run() is
the production pair (openinfer-glm52/src/indexer.rs:538-560): the schedule
metadata launch, then the logits launch.

AOT instantiation bounds (fail-closed INVALID_VALUE past them):
  batch <= 32 (kAotAlignedBatchSize)  -> rows axis stops at 32; 64 is BLOCKED
  num_sms == 132 (kAotNumSms; production NUM_SMS in model/mod.rs is the same
    pinned constant — passed literally, NOT the device SM count)
  next_n=1, num_heads=32, head_dim=128, block_kv=64, non-varlen, non-2d
Works on sm_90a and sm_100f (build.rs promotes the native target; other archs
compile to NOT_SUPPORTED stubs), so the unit is not Blackwell-only.

Semantics (vllm DeepseekV32Indexer, no Hadamard): q enters as raw fp8 (its
group scale is folded into `weights` upstream by the fold kernel);
logit[j] = sum_h relu(q_h . k_j_deq) * weights[h], k_j_deq = e4m3 * k_scale.
Only columns [0, context_lens[r]) are compared — the kernel's split scheduler
owns the rest of each 256-wide tail split.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import indexer

SYMBOL = "glm52_deepgemm_paged_mqa_logits_cuda"
METADATA_SYMBOL = "glm52_deepgemm_paged_mqa_metadata_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ctx"?} — rows <= 32 (AOT bound), ctx defaults to the
    middle axis stop. Every row owns a distinct context region (block_table
    permutation of its own page range), sizes follow production capacity."""
    torch = require_torch()
    rows = shape["rows"]
    if rows > indexer.MQA_MAX_ROWS:
        raise SystemExit(
            f"indexer.mqa_logits: rows {rows} exceeds the AOT batch bound "
            f"{indexer.MQA_MAX_ROWS} (kAotAlignedBatchSize)"
        )
    ctx = indexer.ctx_of(shape)
    cols = indexer.block_cols(ctx)
    device = "cuda"

    # q: bf16 normals quantized per 128-group (the production q_fp8 recipe);
    # raw fp8 codes are the kernel input (q_scale lives in `weights`).
    q_bf16 = data.normal_bf16(
        (rows * indexer.INDEX_HEADS, indexer.HEAD_DIM),
        seed=data.derive_seed(seed, "mqa:q"),
    )
    q_fp8, _q_scale = indexer.quantize_e4m3_rows(q_bf16)
    q_fp8 = q_fp8.reshape(-1).contiguous()

    gen = torch.Generator(device="cpu").manual_seed(data.derive_seed(seed, "mqa:weights"))
    weights = (torch.randn((rows, indexer.INDEX_HEADS), generator=gen) * 1e-4).to(device)

    cache = indexer.build_paged_cache(rows, ctx, data.derive_seed(seed, "mqa:cache"), device)
    block_table = indexer.block_table_for(
        rows, ctx, data.derive_seed(seed, "mqa:table"), device
    )
    context_lens = torch.tensor(
        indexer.seq_lens_for_rows(rows, ctx), dtype=torch.int32, device=device
    )
    return {
        "q_fp8": q_fp8,                    # u8 [rows*32*128]
        "cache": cache,                    # u8 [rows*cols*8448]
        "weights": weights,                # f32 [rows, 32]
        "context_lens": context_lens,      # i32 [rows]
        "block_table": block_table,        # i32 [rows, cols]
        "logits": torch.empty((rows, ctx), dtype=torch.bfloat16, device=device),
        "schedule_meta": torch.empty(indexer.schedule_meta_len(), dtype=torch.int32, device=device),
        "ctx": ctx,
        "out": None,                       # set by reference(): valid columns only
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """The production metadata + logits launch pair on `stream`."""
    rows = shape["rows"]
    ctx = tensors["ctx"]
    cols = indexer.block_cols(ctx)
    meta = resolve(
        lib,
        METADATA_SYMBOL,
        [
            ctypes.c_void_p,   # context_lens i32 (mut)
            ctypes.c_void_p,   # schedule_metadata i32 (mut)
            ctypes.c_int,      # batch_size
            ctypes.c_int,      # next_n
            ctypes.c_int,      # block_kv
            ctypes.c_int,      # num_sms
            ctypes.c_bool,     # is_context_lens_2d
            ctypes.c_bool,     # is_varlen
            ctypes.c_void_p,   # indices (NULL, non-varlen)
            ctypes.c_void_p,   # stream
        ],
    )
    rc = meta(
        as_dev_ptr(tensors["context_lens"]),
        as_dev_ptr(tensors["schedule_meta"]),
        rows,
        1,  # next_n
        indexer.BLOCK_KV,
        indexer.NUM_SMS,
        False,
        False,
        None,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(METADATA_SYMBOL, rc)

    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # q fp8
            ctypes.c_void_p,   # kv_cache u8
            ctypes.c_int64,    # kv_cache_stride_bytes
            ctypes.c_void_p,   # weights f32
            ctypes.c_void_p,   # context_lens i32
            ctypes.c_void_p,   # logits bf16
            ctypes.c_void_p,   # block_table i32
            ctypes.c_void_p,   # indices (NULL)
            ctypes.c_void_p,   # schedule_meta i32 (mut)
            ctypes.c_int,      # batch_size
            ctypes.c_int,      # next_n
            ctypes.c_int,      # num_heads
            ctypes.c_int,      # head_dim
            ctypes.c_int,      # num_kv_blocks
            ctypes.c_int,      # block_kv
            ctypes.c_bool,     # is_context_lens_2d
            ctypes.c_bool,     # is_varlen
            ctypes.c_int,      # logits_stride
            ctypes.c_int,      # block_table_stride
            ctypes.c_int,      # num_sms
            ctypes.c_int,      # q_elem_size
            ctypes.c_int,      # kv_elem_size
            ctypes.c_int,      # weights_elem_size
            ctypes.c_int,      # kv_scales_elem_size
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["q_fp8"]),
        as_dev_ptr(tensors["cache"]),
        indexer.cache_stride_bytes(),
        as_dev_ptr(tensors["weights"]),
        as_dev_ptr(tensors["context_lens"]),
        as_dev_ptr(tensors["logits"]),
        as_dev_ptr(tensors["block_table"]),
        None,
        as_dev_ptr(tensors["schedule_meta"]),
        rows,
        1,  # next_n
        indexer.INDEX_HEADS,
        indexer.HEAD_DIM,
        rows * cols,  # num_kv_blocks: the whole pool
        indexer.BLOCK_KV,
        False,
        False,
        ctx,   # logits_stride == ctx (256-multiple, exact-fit per the ops ensure)
        cols,  # block_table_stride
        indexer.NUM_SMS,
        1, 1, 4, 4,  # fp8 q/kv, f32 weights/scales
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    torch = require_torch()
    rows = shape["rows"]
    lens = tensors["context_lens"].tolist()
    want_rows = indexer.mqa_logits_ref(
        tensors["q_fp8"], tensors["cache"], tensors["weights"],
        tensors["context_lens"], tensors["block_table"], tensors["ctx"],
    )
    # Compare only each row's valid [0, len) columns — the paged split
    # scheduler owns the 256-wide tail region past context_lens.
    got = torch.cat(
        [tensors["logits"][r, : lens[r]].to(torch.float32) for r in range(rows)]
    )
    tensors["out"] = got
    return torch.cat(want_rows)
