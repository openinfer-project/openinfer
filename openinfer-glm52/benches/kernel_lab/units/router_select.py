"""Adapter for router.select — the standalone GLM5.2 router selection:
sigmoid scoring + noaux_tc correction bias + top-8 + routed_scaling 2.5.

Production symbol: `glm52_router_select_cuda`
(csrc/glm52/glm52_router.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs).
Single kernel router_scores_topk_normalize_kernel: grid = padded_tokens
blocks x 256 threads, smem 2080 B; rank-count selection under the strict
(choice desc, index asc) total order, bit-identical by construction to the
old 8-round masked block reductions (.cu comment). This is the select half
of router.noaux_tc split at the logits boundary — the production caller
(moe_decode.rs) uses it when logits already exist.

Unlike the fused unit, nothing here caps the token count architecturally
(one block per token), so the manifest carries the full rows axis
{1,2,4,8,16,32,64}; production today issues padded_tokens <=
GLM52_MAX_BATCH_PER_RANK = 8, and 16-64 are MTP-verify / future-batch
headroom measurements.

Gates (`kernel_lab check`): reference() consumes the SAME generated f32
logits the kernel is fed (pure function of the inputs — no kernel readback),
hard-asserts topk_idx position-for-position (smoke convention), and returns
the reference weights so the manifest rel_l2 gate lands on tensors["out"]
= topk_weight. Any pick/rank flip moves rel_l2 to O(1e-2), four orders
above the expf-vs-sigmoid ulp floor.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import router as router_ref

SYMBOL = "glm52_router_select_cuda"

TOPK = router_ref.TOPK
ROUTE_SCALE = router_ref.ROUTE_SCALE
# Logit spread for the generated inputs: std 2.0 keeps sigmoid scores in the
# well-separated regime (top-8 order-statistic gaps >> 1 ulp) and far from
# the exact-1.0f saturation plateau. Timing is value-independent.
LOGIT_SCALE = 2.0
BIAS_SCALE = 0.01


def _cpu_normal_f32(shape: tuple[int, ...], seed: int, scale: float):
    """CPU-generated N(0, scale^2) f32 — machine-independent stream, same
    philosophy as data.normal_bf16."""
    torch = require_torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32) * scale


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"}; active == padded == rows. shape["k"] is
    the nominal producer hidden size from the manifest (select has no k
    dimension) and is unused here."""
    torch = require_torch()
    rows, n = shape["rows"], shape["n"]
    logits = _cpu_normal_f32((rows, n), data.derive_seed(seed, "logits"), LOGIT_SCALE)
    bias = _cpu_normal_f32((n,), data.derive_seed(seed, "bias"), BIAS_SCALE)
    return {
        "logits": logits.to("cuda"),  # f32 [rows, 256]
        "bias": bias.to("cuda"),  # f32 [256]
        "out": torch.empty((rows, TOPK), dtype=torch.float32, device="cuda"),  # topk_weight
        "topk_idx": torch.empty((rows, TOPK), dtype=torch.int32, device="cuda"),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # logits f32
            ctypes.c_void_p,   # e_score_correction_bias f32
            ctypes.c_void_p,   # topk_weight f32
            ctypes.c_void_p,   # topk_idx i32
            ctypes.c_int,      # active_tokens == rows
            ctypes.c_int,      # padded_tokens == rows
            ctypes.c_int,      # n_experts (kernel validates == 256)
            ctypes.c_int,      # topk (kernel validates == 8)
            ctypes.c_float,    # route_scale
            ctypes.c_void_p,   # stream
        ],
    )
    rows = shape["rows"]
    rc = fn(
        as_dev_ptr(tensors["logits"]),
        as_dev_ptr(tensors["bias"]),
        as_dev_ptr(tensors["out"]),
        as_dev_ptr(tensors["topk_idx"]),
        rows,
        rows,
        shape["n"],
        TOPK,
        ROUTE_SCALE,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    """Select reference over the same input logits; hard-asserts exact idx
    (smoke convention), returns the reference weights for the rel_l2 gate."""
    ref_idx, ref_weight = router_ref.sigmoid_select_ref(tensors["logits"], tensors["bias"])
    router_ref.assert_select_exact(
        tensors["topk_idx"], tensors["out"], ref_idx, ref_weight,
        tag="router.select",
    )
    return ref_weight
