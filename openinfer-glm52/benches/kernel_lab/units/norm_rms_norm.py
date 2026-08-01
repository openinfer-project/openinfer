"""Adapter for norm.rms_norm — the standalone batched RMSNorm at hidden=6144.

Production symbol: `rms_norm_batched_cuda` (csrc/shared/flashinfer_norm.cu:204
— wraps flashinfer::norm::RMSNorm<DType> with batch_size=rows, one CTA per
row; FFI mirror openinfer-kernels/src/ffi/shared.rs:17). Call site:
openinfer-glm52/src/model/step_body.rs:76 — "Layer 0's input norm is
standalone (the embedding is the residual)"; every later layer's input norm
is fused into the previous closing add (norm.fused_add_rmsnorm_round). The
same kernel + dim also serves the final norm (bookend.rs:47) and the MTP
enorm/hnorm/shared_norm (mtp.rs:195/:204/:263).
ABI: VOID return (no CUresult) — launch failures surface only as sticky CUDA
errors; the adapter mirrors production (ops/norm.rs rms_norm_rows_into) and
checks nothing.
rows: one CTA per row with a self-contained per-row reduction, so each row is
bit-identical to the rows=1 launch for every rows in {1..64} — no .cu change
needed for rows>8.
"""
from __future__ import annotations

from kernel_lab import data
from kernel_lab.loader import require_torch
from kernel_lab.refs import norm

SYMBOL = "rms_norm_batched_cuda"
HIDDEN = norm.GLM52_HIDDEN


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"} with n == k == hidden (all rows live)."""
    torch = require_torch()
    rows, hidden = shape["rows"], shape["n"]
    if hidden != HIDDEN or shape["k"] != HIDDEN:
        raise ValueError(f"{SYMBOL} adapter expects n == k == {HIDDEN}, got {shape}")
    x = data.normal_bf16((rows, hidden), seed=data.derive_seed(seed, "act"))
    return {
        "x": x,                               # bf16 [rows, hidden]
        "weight": norm.norm_weight_bf16((hidden,), seed=data.derive_seed(seed, "norm_weight")),
        "out": torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    norm.launch_rms_norm_batched(
        lib,
        tensors["x"],
        tensors["weight"],
        tensors["out"],
        shape["n"],
        shape["rows"],
        norm.GLM52_RMS_EPS,
        stream,
    )


def reference(tensors: dict, shape: dict):
    """f32 torch reference for the rel_l2 gate (the kernel's f32 reduction
    order is not reproduced; the bf16 store floor dominates the tolerance)."""
    return norm.rms_norm_ref(tensors["x"], tensors["weight"], norm.GLM52_RMS_EPS)
