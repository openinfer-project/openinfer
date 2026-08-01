"""Adapter for shared_expert.swiglu — the per-rank-replicated fp8 SwiGLU MLP.

The unit is the three-production-launch chain of `fp8_mlp_into`
(openinfer-glm52/src/fp8.rs:502) at the shared-expert shape
(Glm52MoeSharedExpert, openinfer-glm52/src/moe_decode.rs:79; intermediate 2048
= GLM52_EXPERT_INTERMEDIATE, config.rs:42 — NOT the 12288-wide dense layers):

  1. `glm52_fp8_weight_only_gemv_partials_cuda` — packed gate|up [4096, 6144]
     (SYMBOL; mma routes stop at f32 partials and report ksplit, register
     routes write the bf16 [gate|up] and report 0)
  2. ksplit == 0 ? `glm52_silu_and_mul_bf16_cuda`
                 : `glm52_gemv_reduce_silu_mul_cuda` (fused fixed-order
     reduce+SiLU, bit-identical to the standalone pair by construction)
  3. `glm52_fp8_weight_only_gemv_batched_cuda` — down [6144, 2048]

One f32 scratch buffer serves both GEMVs (the Glm52MlpScratch.gemv_partial
ownership pattern), sized from the runtime queries as
max(ksplit_gu*rows*4096, ksplit_dn*rows*6144); ksplit == 0 on both means the
register routes and a NULL pointer. rows routing follows the shared
mma_config: 1 CUDA-core GEMV, 2 register tile, 4/8 tensor-core mma, 16/32/64
multi-subtile mma via measured GB300 {8,2} table entries (both legs, all
three batches) behind KERNEL_LAB_GEMV_MMA_MULTI — set by default inside the
harness (loader.py); production keeps the #812 register tiles at 16/32/48
pending the A/B, rows=64 has no other route, and knob-unset/non-Blackwell
rows>8 fail closed with CUDA_ERROR_INVALID_VALUE — surfaced here as
KernelLaunchError, never silent.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import shared_expert

SYMBOL = "glm52_fp8_weight_only_gemv_partials_cuda"
SILU_SYMBOL = "glm52_silu_and_mul_bf16_cuda"
REDUCE_SILU_SYMBOL = "glm52_gemv_reduce_silu_mul_cuda"
DOWN_SYMBOL = "glm52_fp8_weight_only_gemv_batched_cuda"
KSPLIT_SYMBOL = "glm52_gemv_mma_ksplit_cuda"


def _query_ksplit(lib, rows: int, n: int, k: int) -> int:
    fn = resolve(
        lib,
        KSPLIT_SYMBOL,
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
    )
    out = ctypes.c_int(0)
    rc = fn(rows, n, k, ctypes.byref(out))
    if rc != 0:
        raise KernelLaunchError(KSPLIT_SYMBOL, rc)
    return out.value


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n"=2*inter=4096, "k"=hidden=6144}; the down
    projection [hidden, inter] derives from the same two numbers."""
    torch = require_torch()
    rows, n_gu, hidden = shape["rows"], shape["n"], shape["k"]
    inter = n_gu // 2
    act = data.normal_bf16((rows, hidden), seed=data.derive_seed(seed, "act"))
    gu_w, gu_s = data.normal_quantized_fp8(n_gu, hidden, seed=data.derive_seed(seed, "gate_up_weight"))
    dn_w, dn_s = data.normal_quantized_fp8(hidden, inter, seed=data.derive_seed(seed, "down_weight"))
    dev = act.device
    return {
        "act": act,                      # bf16 [rows, hidden]
        "gu_weight": gu_w,               # e4m3 uint8 [2*inter, hidden]
        "gu_scales": gu_s,               # f32 [2*inter/128, hidden/128]
        "dn_weight": dn_w,               # e4m3 uint8 [hidden, inter]
        "dn_scales": dn_s,               # f32 [hidden/128, inter/128]
        "gate_up": torch.empty((rows, n_gu), dtype=torch.bfloat16, device=dev),
        "silu_out": torch.empty((rows, inter), dtype=torch.bfloat16, device=dev),
        "out": torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev),
        "scratch": None,                 # allocated lazily on mma routes
    }


def _ensure_scratch(lib, tensors: dict, shape: dict):
    """One f32 buffer for both GEMVs (production's gemv_partial pattern):
    max(ksplit_gu*rows*2*inter, ksplit_dn*rows*hidden) floats — the exact
    bound each launcher's guard checks. Both ksplits 0 (rows 1/2) means the
    register routes: NULL, which those dispatches ignore."""
    rows, n_gu, hidden = shape["rows"], shape["n"], shape["k"]
    inter = n_gu // 2
    ksplit_gu = _query_ksplit(lib, rows, n_gu, hidden)
    ksplit_dn = _query_ksplit(lib, rows, hidden, inter)
    floats = max(ksplit_gu * rows * n_gu, ksplit_dn * rows * hidden)
    if floats == 0:
        return None, 0
    torch = require_torch()
    buf = tensors.get("scratch")
    if buf is None or buf.numel() < floats:
        buf = torch.empty(floats, dtype=torch.float32, device=tensors["out"].device)
        tensors["scratch"] = buf
    return buf, floats


def run(lib, tensors: dict, shape: dict, stream) -> None:
    rows, n_gu, hidden = shape["rows"], shape["n"], shape["k"]
    inter = n_gu // 2
    scratch, scratch_floats = _ensure_scratch(lib, tensors, shape)
    scratch_ptr = as_dev_ptr(scratch) if scratch is not None else None

    # 1) packed gate|up GEMV — mma routes stop at the f32 partials.
    fnp = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
        ],
    )
    ksplit = ctypes.c_int(0)
    rc = fnp(
        as_dev_ptr(tensors["act"]),
        as_dev_ptr(tensors["gu_weight"]),
        as_dev_ptr(tensors["gu_scales"]),
        as_dev_ptr(tensors["gate_up"]),
        scratch_ptr,
        scratch_floats,
        rows,
        n_gu,
        hidden,
        stream,
        ctypes.byref(ksplit),
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)

    # 2) SiLU(gate) * up — standalone on the register routes, fused
    # reduce+SiLU on the mma route (bit-identical by construction).
    if ksplit.value == 0:
        fns = resolve(
            lib,
            SILU_SYMBOL,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p],
        )
        rc = fns(as_dev_ptr(tensors["gate_up"]), as_dev_ptr(tensors["silu_out"]), rows, inter, stream)
        if rc != 0:
            raise KernelLaunchError(SILU_SYMBOL, rc)
    else:
        fnr = resolve(
            lib,
            REDUCE_SILU_SYMBOL,
            [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p,
            ],
        )
        rc = fnr(scratch_ptr, as_dev_ptr(tensors["silu_out"]), rows, inter, ksplit.value, stream)
        if rc != 0:
            raise KernelLaunchError(REDUCE_SILU_SYMBOL, rc)

    # 3) down GEMV.
    fnd = resolve(
        lib,
        DOWN_SYMBOL,
        [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ],
    )
    rc = fnd(
        as_dev_ptr(tensors["silu_out"]),
        as_dev_ptr(tensors["dn_weight"]),
        as_dev_ptr(tensors["dn_scales"]),
        as_dev_ptr(tensors["out"]),
        scratch_ptr,
        scratch_floats,
        rows,
        hidden,
        inter,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(DOWN_SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    return shared_expert.fp8_swiglu_ref(
        tensors["act"],
        tensors["gu_weight"],
        tensors["gu_scales"],
        tensors["dn_weight"],
        tensors["dn_scales"],
        shape["n"] // 2,
    )
