"""Adapter for quant.fp8_per_token_group_bf16_ue8m0 — per-token-group quant
with power-of-two (UE8M0) scales, the FlashMLA V3.2 fp8_ds_mla cache contract.

Production symbol: `glm52_fp8_per_token_group_quant_bf16_ue8m0_cuda`
(csrc/glm52/glm52_moe_quant.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs:296;
ops wrapper openinfer-kernels/src/ops/glm52/moe_quant.rs). Sole production
shape: the MLA kv_c latent [rows, KV_LORA_RANK=512] before
`glm52_mla_cache_pack` writes the 656-byte fp8 cache rows — decode
(openinfer-glm52/src/mla_decode.rs:490) and prefill
(openinfer-glm52/src/prefill_tp.rs:603) alike, so the unit pins hidden=512
(group=128 -> 4 scales/row).

Why UE8M0: the sm100 FlashMLA decode kernel truncates stored f32 scales to
e8m0 (round-toward-zero) for the tcgen05 block-scaled MMA, so a
non-power-of-two scale reads up to 2x too small on Blackwell while sm90 reads
the f32 scale exactly — H200 masks the bug
(docs/lessons/flashmla-sm100-ue8m0-kv-scales.md). The kernel rounds the
amax/448 scale UP to the next power of two via exact bit manipulation
(`(bits + 0x7FFFFF) & 0x7F800000`).

rows: row-agnostic grid-stride loop identical to the non-ue8m0 twin — full
axis {1..64} with ZERO .cu change and no scratch.

Check path: in addition to the bit-exact packed-surface comparison (see the
base twin), `reference()` hard-asserts every kernel-written scale is a
positive finite power of two — the property the sm100 consumer relies on.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import quant as quant_ref

SYMBOL = "glm52_fp8_per_token_group_quant_bf16_ue8m0_cuda"
GROUP_SIZE = quant_ref.GROUP_SIZE


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"}; hidden = k (quant is width-preserving)."""
    torch = require_torch()
    rows, hidden = shape["rows"], shape["k"]
    if hidden % GROUP_SIZE:
        raise ValueError(f"{SYMBOL}: hidden {hidden} not divisible by {GROUP_SIZE}")
    act = data.normal_bf16((rows, hidden), seed=data.derive_seed(seed, "act"))
    # Deterministic edge coverage: an all-zero group exercises the amax=0 ->
    # eps-clamped scale branch (2^-42 after the ue8m0 bump) on every run.
    act[0, :GROUP_SIZE] = 0.0
    value_bytes = rows * hidden
    packed = torch.empty(
        quant_ref.packed_surface_len(rows, hidden), dtype=torch.uint8, device=act.device
    )
    return {
        "act": act,                         # bf16 [rows, hidden]
        "out": packed,                      # uint8 flat comparison surface
        "out_bytes": packed[:value_bytes].view(rows, hidden),  # e4m3 u8 [rows, hidden]
        "scales": packed[value_bytes:]
        .view(torch.float32)
        .view(rows, hidden // GROUP_SIZE),  # f32 [rows, hidden/128]
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t). No scratch:
    the ABI is (input, output, scales, rows, hidden, group_size, stream)."""
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # input bf16 [rows, hidden]
            ctypes.c_void_p,   # output e4m3 u8 [rows, hidden]
            ctypes.c_void_p,   # scales f32 [rows, hidden/128]
            ctypes.c_int,      # rows
            ctypes.c_int,      # hidden_dim
            ctypes.c_int,      # group_size (must be 128)
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["act"]),
        as_dev_ptr(tensors["out_bytes"]),
        as_dev_ptr(tensors["scales"]),
        shape["rows"],
        shape["k"],
        GROUP_SIZE,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    # UE8M0 property gate: every kernel-written scale must be a power of two
    # (the sm100 e8m0-truncation contract) — hard assertion, runs only in
    # `kernel_lab check`, never in bench/compare.
    quant_ref.assert_scales_power_of_two(tensors["scales"])
    out, scales = quant_ref.fp8_per_token_group_quant_ref(tensors["act"], ue8m0=True)
    return quant_ref.pack_quant_surface(out, scales)
