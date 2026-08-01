"""Torch references + shared launch helpers for the norm group
(norm.rms_norm / norm.q_a_layernorm / norm.fused_add_rmsnorm_round).

The kernels live in csrc/shared/flashinfer_norm.cu (FFI mirror
openinfer-kernels/src/ffi/shared.rs); model constants mirror
openinfer-glm52/src/config.rs. torch is imported lazily inside functions —
module level stays stdlib + ctypes so CPU boxes can run the registry tests.

Group-level capabilities the shared harness lacks (kept here per the task
rules — kernel_lab/ shared files are off-limits):
- `resolve_void`: loader.resolve() pins restype=c_int (CUresult), but
  `rms_norm_batched_cuda` returns void; reading a garbage register as a
  CUresult would randomly raise KernelLaunchError.
- `unfused_add_rmsnorm_chain` + `byte_compare`: the fused unit's second
  reference layer (production unfused add+rms_norm chain, launched from the
  SAME .so — torch cannot reproduce the f32 association order, design doc
  decode-op-bench-harness.md item 4). The `check` subcommand has no dual-mode
  hook, so this gate is driven by benches/tests/test_norm.py on GPU boxes.
- `norm_weight_bf16`: production-like gamma factory (data.py only has N(0,1)).
"""
from __future__ import annotations

import ctypes

from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve

# Model constants — mirror openinfer-glm52/src/config.rs (GLM52_HIDDEN:9,
# GLM52_Q_LORA_RANK:23, GLM52_RMS_EPS:50). Every RMSNorm in the model shares
# the one checkpoint eps that probe_config_json validates.
GLM52_HIDDEN = 6144
GLM52_Q_LORA = 2048
GLM52_RMS_EPS = 1e-5

RMS_NORM_BATCHED_SYMBOL = "rms_norm_batched_cuda"
FUSED_ADD_RMS_NORM_ROUND_SYMBOL = "fused_add_rms_norm_round_batched_cuda"
ADD_SYMBOL = "add_cuda"


def resolve_void(lib, name: str, argtypes: list):
    """Fetch a void-returning production symbol (no CUresult to check).

    `rms_norm_batched_cuda` / `rms_norm_cuda` return nothing (ffi/shared.rs:8,
    :17) — launch failures surface only as sticky CUDA errors on the stream,
    exactly as production sees them (ops/norm.rs rms_norm_rows_into checks
    nothing either).
    """
    try:
        fn = getattr(lib, name)
    except AttributeError as exc:
        raise SystemExit(f"kernel_lab: symbol {name} not found in {lib._name}") from exc
    fn.restype = None
    fn.argtypes = argtypes
    return fn


def norm_weight_bf16(shape: tuple[int, ...], seed: int, device: str = "cuda"):
    """Layernorm gamma ~ 1 + 0.1*N(0,1) in bf16 (production norm weights
    cluster near 1; a zero-mean N(0,1) gamma would be a stress pattern, not
    the shipped one). CPU-seeded like data.normal_bf16 so the stream is
    identical across machines."""
    torch = require_torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = 1.0 + 0.1 * torch.randn(shape, generator=gen, dtype=torch.float32)
    return w.to(torch.bfloat16).to(device)


def rms_norm_ref(x_bf16, weight_bf16, eps: float):
    """f32 RMSNorm reference [rows, hidden] (semantic-level net only).

    The kernel's f32 reduction order (per-thread vec accumulate + warp
    shfl_xor tree) is NOT reproduced here — the manifest tolerance absorbs the
    reorder; the bf16 store floor dominates (see the unit manifest note).
    """
    torch = require_torch()
    x = x_bf16.to(torch.float32)
    ms = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight_bf16.to(torch.float32)


def fused_add_rmsnorm_round_ref(hidden_bf16, residual_bf16, weight_bf16, eps: float):
    """Reference for `fused_add_rms_norm_round_batched_cuda`.

    Returns (sum_bf16, out_f32):
      sum_bf16 = bf16(hidden + residual) — the `_round` boundary; torch's
        RN cast reproduces the kernel's `static_cast<T>(f32+f32)` bit-exactly,
        so this half IS bit-comparable (done against the production unfused
        chain in `byte_compare`, and sanity-checkable here).
      out_f32  = rms_norm_ref(sum_bf16) — feeds the rel_l2 gate; the rms
        scalar's f32 reduction order is the only irreproducible part.
    """
    torch = require_torch()
    summed = (hidden_bf16.to(torch.float32) + residual_bf16.to(torch.float32)).to(torch.bfloat16)
    return summed, rms_norm_ref(summed, weight_bf16, eps)


def launch_add(lib, a, b, out, n: int, stream) -> None:
    """Production `add_cuda` (csrc/shared/elementwise.cu:382): out = bf16(f32
    a + f32 b), elementwise over n values. Returns CUresult."""
    fn = resolve(
        lib,
        ADD_SYMBOL,
        [
            ctypes.c_void_p,  # a bf16
            ctypes.c_void_p,  # b bf16
            ctypes.c_void_p,  # out bf16
            ctypes.c_int,     # n
            ctypes.c_void_p,  # stream
        ],
    )
    rc = fn(as_dev_ptr(a), as_dev_ptr(b), as_dev_ptr(out), n, stream)
    if rc != 0:
        raise KernelLaunchError(ADD_SYMBOL, rc)


def launch_rms_norm_batched(lib, x, weight, out, hidden_dim: int, rows: int,
                            eps: float, stream) -> None:
    """Production `rms_norm_batched_cuda` (flashinfer_norm.cu:204 — FlashInfer
    norm::RMSNorm, one CTA per row). VOID ABI: nothing to check."""
    fn = resolve_void(
        lib,
        RMS_NORM_BATCHED_SYMBOL,
        [
            ctypes.c_void_p,  # x bf16
            ctypes.c_void_p,  # weight bf16
            ctypes.c_void_p,  # out bf16
            ctypes.c_int,     # hidden_dim
            ctypes.c_int,     # seq_len == rows
            ctypes.c_float,   # eps
            ctypes.c_void_p,  # stream
        ],
    )
    fn(as_dev_ptr(x), as_dev_ptr(weight), as_dev_ptr(out), hidden_dim, rows, eps, stream)


def unfused_add_rmsnorm_chain(lib, hidden_orig, residual, weight, stream):
    """The production UNFUSED reference chain, launched from the same .so:
    sum = add_cuda(hidden_orig, residual); out = rms_norm_batched_cuda(sum).

    This is the exact two-kernel sequence the fused `_round` kernel replaced
    (openinfer-glm52/src/layer.rs:322-326 claims bit-identity). Returns
    (sum_buf, out_ref), both bf16 shaped like hidden_orig.
    """
    torch = require_torch()
    rows, hidden = hidden_orig.shape
    sum_buf = torch.empty_like(hidden_orig)
    out_ref = torch.empty_like(hidden_orig)
    launch_add(lib, hidden_orig, residual, sum_buf, rows * hidden, stream)
    launch_rms_norm_batched(lib, sum_buf, weight, out_ref, hidden, rows, GLM52_RMS_EPS, stream)
    return sum_buf, out_ref


def byte_compare(a, b) -> dict:
    """Bit-level equality over bf16 storage (viewed as int16 bits, so -0.0 vs
    +0.0 and NaN payloads count as mismatches — the strictest gate)."""
    torch = require_torch()
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"equal": False, "mismatches": -1, "first_mismatch": -1,
                "reason": f"shape/dtype mismatch {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"}
    diff = a.view(torch.int16) != b.view(torch.int16)
    count = int(diff.sum().item())
    first = int(diff.flatten().nonzero()[0].item()) if count else -1
    return {"equal": count == 0, "mismatches": count, "first_mismatch": first}
