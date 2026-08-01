"""Adapter for router.noaux_tc — the fused GLM5.2 MoE router:
gate GEMV (6144 -> 256) + sigmoid noaux_tc scoring + top-8 select with
routed_scaling 2.5.

Production symbol: `glm52_router_noaux_tc_cuda`
(csrc/glm52/glm52_router.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs).
The GEMV half is glm52_min_gemv (one PDL kernel, grid=256 blocks x 128
threads, fixed f32 reduction order — it replaced the cublas splitK plan);
the select half is router_scores_topk_normalize_kernel (grid=padded_tokens x
256 threads, smem 2080 B, rank-count selection under the (choice desc, index
asc) total order). EP-agnostic: every rank scores the full 256 experts.

rows>8 is BLOCKED: the min_gemv runtime dispatch instantiates tokens 1..=8
only (kMaxTokens = GLM52_MAX_BATCH_PER_RANK, and production caps per-rank
tokens at 8 as well); above that the launcher fails closed with
CUDA_ERROR_INVALID_VALUE. Instantiating 16/32/64 would put acc[kNumTokens]
per-thread f32 accumulators on the register wall the design doc calls
infeasible for batch 64. See the manifest notes.

Gates (`kernel_lab check`): logits rel_l2 vs the f64 reference (tensors["out"]
= the kernel logits buffer), plus a hard select assertion inside reference()
— the reference consumes the kernel's OWN f32 logits, replicating
glm52_router_smoke.rs's host_select so the exact-order idx check has no
f64-vs-f32 sigmoid near-tie flake.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import router as router_ref

SYMBOL = "glm52_router_noaux_tc_cuda"

TOPK = router_ref.TOPK
ROUTE_SCALE = router_ref.ROUTE_SCALE
# Gate scale mirrors the smoke test's small-weight convention (|w| <= 0.05
# bounded uniform there; normal here). Keeps logits at std ~2.4 so sigmoid
# scores stay clear of the exact-1.0f saturation plateau (timing is
# value-independent: both kernels run fixed trip counts).
GATE_SCALE = 0.03
BIAS_SCALE = 0.01


def _cpu_normal_f32(shape: tuple[int, ...], seed: int, scale: float):
    """CPU-generated N(0, scale^2) f32 — machine-independent stream, same
    philosophy as data.normal_bf16."""
    torch = require_torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32) * scale


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"} at production capacity: active == padded ==
    rows (all rows live). rows <= 8 per the manifest axis (min_gemv cap)."""
    torch = require_torch()
    rows, n, k = shape["rows"], shape["n"], shape["k"]
    hidden = data.normal_bf16((rows, k), seed=data.derive_seed(seed, "act"))
    gate = _cpu_normal_f32((n, k), data.derive_seed(seed, "weight"), GATE_SCALE)
    bias = _cpu_normal_f32((n,), data.derive_seed(seed, "bias"), BIAS_SCALE)
    return {
        "hidden": hidden,  # bf16 [rows, 6144]
        "gate": gate.to(torch.bfloat16).to(hidden.device),  # bf16 [256, 6144]
        "bias": bias.to(hidden.device),  # f32 [256]
        "out": torch.empty((rows, n), dtype=torch.float32, device=hidden.device),  # logits
        "topk_weight": torch.empty((rows, TOPK), dtype=torch.float32, device=hidden.device),
        "topk_idx": torch.empty((rows, TOPK), dtype=torch.int32, device=hidden.device),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # hidden bf16
            ctypes.c_void_p,   # gate_weight bf16
            ctypes.c_void_p,   # e_score_correction_bias f32
            ctypes.c_void_p,   # logits f32 (inter-kernel scratch, gated out)
            ctypes.c_void_p,   # topk_weight f32
            ctypes.c_void_p,   # topk_idx i32
            ctypes.c_int,      # active_tokens == rows
            ctypes.c_int,      # padded_tokens == rows
            ctypes.c_int,      # hidden_dim (kernel validates == 6144)
            ctypes.c_int,      # n_experts (kernel validates == 256)
            ctypes.c_int,      # topk (kernel validates == 8)
            ctypes.c_float,    # route_scale
            ctypes.c_void_p,   # stream
        ],
    )
    rows = shape["rows"]
    rc = fn(
        as_dev_ptr(tensors["hidden"]),
        as_dev_ptr(tensors["gate"]),
        as_dev_ptr(tensors["bias"]),
        as_dev_ptr(tensors["out"]),
        as_dev_ptr(tensors["topk_weight"]),
        as_dev_ptr(tensors["topk_idx"]),
        rows,
        rows,
        shape["k"],
        shape["n"],
        TOPK,
        ROUTE_SCALE,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    """Hard-gate the select half against the kernel's OWN logits (smoke
    convention: idx position-for-position exact, |d weight| < 1e-4), then
    return the f64 logits reference for the manifest rel_l2 gate."""
    ref_idx, ref_weight = router_ref.sigmoid_select_ref(tensors["out"], tensors["bias"])
    router_ref.assert_select_exact(
        tensors["topk_idx"], tensors["topk_weight"], ref_idx, ref_weight,
        tag="router.noaux_tc",
    )
    return router_ref.router_logits_ref(tensors["hidden"], tensors["gate"])
