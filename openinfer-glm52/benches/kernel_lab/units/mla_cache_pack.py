"""Adapter for mla.cache_pack — the fp8_ds_mla 656-byte cache token writer.

Production symbol: `glm52_mla_cache_pack_cuda`
(csrc/glm52/glm52_mla_assembly.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs;
production call `glm52_mla_attend_into`, openinfer-glm52/src/mla_decode.rs:501,
fed by `glm52_fp8_per_token_group_quant_bf16_ue8m0_launch`).

Per row t: one paged token at `slot_mapping[t]` =
[512 e4m3 ckv | 4 f32 group scales | 64 bf16 interleave-RoPE(k_pe)]. The
slot is device data (CUDA-graph replayable in production); an out-of-window
slot traps the kernel.

UE8M0 discipline (docs/lessons/flashmla-sm100-ue8m0-kv-scales.md): the sm100
FlashMLA kernel truncates stored f32 scales to e8m0, so the harness generates
scales with 2^ceil(log2(amax/448)) AND asserts the pow2 invariant on every
scale entering the reference path (`assert_ue8m0_scales`) — a non-pow2 scale
is a harness-level failure, not a tolerance event.

rows routing: none — grid is rows blocks x 128 threads with no upper bound,
each block independent, so rows {16,32,64} are bit-identical per row. Zero
.cu changes. ctx sizes the paged window (max_slots = ctx, page = 64 tokens);
the write cost is O(rows x 656B), window-independent. The gate is byte-exact
against a torch-built expected image.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import mla_attention as mla

SYMBOL = "glm52_mla_cache_pack_cuda"


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ctx"?, ...}; ctx defaults to the first axis tier."""
    torch = require_torch()
    rows = shape["rows"]
    ctx = int(shape.get("ctx", mla.DEFAULT_CTX))
    if ctx % mla.PAGE_TOKENS or rows > ctx:
        raise ValueError(f"cache_pack needs ctx % {mla.PAGE_TOKENS} == 0 and rows <= ctx, got {ctx}/{rows}")
    ckv_fp8, scales = mla.normal_quantized_ckv(rows, data.derive_seed(seed, "ckv"), device="cuda")
    k_pe = data.normal_bf16((rows, mla.ROPE_DIM), seed=data.derive_seed(seed, "k_pe"))
    cos, sin = mla.rotary_table(rows, data.derive_seed(seed, "rotary"), device=k_pe.device)
    gen = torch.Generator(device="cpu").manual_seed(data.derive_seed(seed, "slots"))
    slots = torch.randperm(ctx, generator=gen)[:rows].to(torch.int64).to(k_pe.device)
    return {
        "ckv_fp8": ckv_fp8,        # u8 [rows, 512] e4m3 (no NaN patterns)
        "ckv_scales": scales,      # f32 [rows, 4] UE8M0 pow2 (asserted)
        "k_pe": k_pe,              # bf16 [rows, 64] pre-rope
        "cos": cos,                # bf16 [rows, 32]
        "sin": sin,                # bf16 [rows, 32]
        "slot_mapping": slots,     # i64 [rows] distinct in [0, ctx)
        "out": torch.zeros(ctx * mla.CACHE_BYTES, dtype=torch.uint8, device=k_pe.device),
        "_num_slots": ctx,
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # ckv_fp8 u8 [rows,512]
            ctypes.c_void_p,   # ckv_scales f32 [rows,4]
            ctypes.c_void_p,   # k_pe bf16 [rows,64]
            ctypes.c_void_p,   # cos bf16 [rows,32]
            ctypes.c_void_p,   # sin bf16 [rows,32]
            ctypes.c_void_p,   # cache u8 [max_slots,656]
            ctypes.c_void_p,   # slot_mapping i64 [rows]
            ctypes.c_int64,    # max_slots = ctx
            ctypes.c_int,      # tokens = rows
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["ckv_fp8"]),
        as_dev_ptr(tensors["ckv_scales"]),
        as_dev_ptr(tensors["k_pe"]),
        as_dev_ptr(tensors["cos"]),
        as_dev_ptr(tensors["sin"]),
        as_dev_ptr(tensors["out"]),
        as_dev_ptr(tensors["slot_mapping"]),
        tensors["_num_slots"],
        shape["rows"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    return mla.cache_pack_ref(
        tensors["ckv_fp8"],
        tensors["ckv_scales"],
        tensors["k_pe"],
        tensors["cos"],
        tensors["sin"],
        tensors["slot_mapping"],
        tensors["_num_slots"],
    )
