"""Adapter for norm.fused_add_rmsnorm_round — the layer-closing fused
add+RMSNorm(+bf16 round) kernel.

Production symbol: `fused_add_rms_norm_round_batched_cuda`
(csrc/shared/flashinfer_norm.cu:252 — the custom FusedAddRMSNormRoundKernel,
NOT FlashInfer's FusedAddRMSNorm template; FFI mirror
openinfer-kernels/src/ffi/shared.rs:700). Call sites:
openinfer-glm52/src/layer.rs:332 (post-attention boundary) and :363
(`glm52_layer_finish_fused`, the closing residual add fused with the NEXT
layer's input_layernorm). Semantics: hidden = bf16(hidden + residual) in
place, then out = rms_norm(hidden_rounded, weight) — bit-identical by
construction to the production unfused chain add_cuda +
rms_norm_batched_cuda (layer.rs:322-326); `unfused_byte_compare()` proves
that invariant with both chain kernels loaded from the same .so.
rows: grid.x = batch_size with one CTA per row and a self-contained per-row
reduction, so every rows value in {1..64} runs the same kernel and each row
is bit-identical to the rows=1 launch — no .cu change needed for rows>8.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import norm

SYMBOL = "fused_add_rms_norm_round_batched_cuda"
UNFUSED_CHAIN_SYMBOLS = ("add_cuda", "rms_norm_batched_cuda")
HIDDEN = norm.GLM52_HIDDEN


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"} with n == k == hidden (all rows live)."""
    torch = require_torch()
    rows, hidden = shape["rows"], shape["n"]
    if hidden != HIDDEN or shape["k"] != HIDDEN:
        raise ValueError(f"{SYMBOL} adapter expects n == k == {HIDDEN}, got {shape}")
    hidden_acc = data.normal_bf16((rows, hidden), seed=data.derive_seed(seed, "act"))
    residual = data.normal_bf16((rows, hidden), seed=data.derive_seed(seed, "residual"))
    return {
        "hidden": hidden_acc,                 # bf16 [rows, hidden], mutated in place
        "hidden_orig": hidden_acc.clone(),    # pristine copy for both references
        "residual": residual,                 # bf16 [rows, hidden] (read-only)
        "weight": norm.norm_weight_bf16((hidden,), seed=data.derive_seed(seed, "norm_weight")),
        "out": torch.empty((rows, hidden), dtype=torch.bfloat16, device=hidden_acc.device),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # hidden bf16 (in/out, updated in place)
            ctypes.c_void_p,   # residual bf16
            ctypes.c_void_p,   # weight bf16
            ctypes.c_void_p,   # out bf16
            ctypes.c_int,      # hidden_dim
            ctypes.c_int,      # batch_size == rows
            ctypes.c_float,    # eps
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["hidden"]),
        as_dev_ptr(tensors["residual"]),
        as_dev_ptr(tensors["weight"]),
        as_dev_ptr(tensors["out"]),
        shape["n"],
        shape["rows"],
        norm.GLM52_RMS_EPS,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    """f32 torch reference for the rel_l2 gate (semantic net; the kernel's f32
    reduction order is not reproduced). The bit-exact half of the contract —
    the rounded sum and, via the same reduction, the normed output — is gated
    by unfused_byte_compare()."""
    return norm.fused_add_rmsnorm_round_ref(
        tensors["hidden_orig"], tensors["residual"], tensors["weight"], norm.GLM52_RMS_EPS
    )[1]


def unfused_byte_compare(lib, tensors: dict, shape: dict, stream) -> dict:
    """Second reference layer (decode-op-bench-harness.md item 4): the
    production UNFUSED chain (add_cuda -> rms_norm_batched_cuda, loaded from
    the same .so) must reproduce BOTH fused outputs bit-for-bit. Call AFTER
    run() — it consumes the in-place-updated `hidden` and `out`.
    Not wired into the `check` subcommand (the CLI has no dual-mode hook —
    reported as a harness gap); driven by benches/tests/test_norm.py on GPU
    boxes."""
    sum_ref, out_ref = norm.unfused_add_rmsnorm_chain(
        lib, tensors["hidden_orig"], tensors["residual"], tensors["weight"], stream
    )
    require_torch().cuda.synchronize()
    return {
        "hidden_sum": norm.byte_compare(tensors["hidden"], sum_ref),
        "out": norm.byte_compare(tensors["out"], out_ref),
    }
