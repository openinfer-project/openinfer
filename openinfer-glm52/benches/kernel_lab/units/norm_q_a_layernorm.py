"""Adapter for norm.q_a_layernorm — the MLA q_a_layernorm over the q_lora
rank (2048).

Despite the checkpoint name, this op is an **RMSNorm, not a LayerNorm**:
- HF: `self.q_a_layernorm = GlmMoeDsaRMSNorm(config.q_lora_rank)`
  (transformers models/glm_moe_dsa/modeling_glm_moe_dsa.py:339);
- the checkpoint carries only `q_a_layernorm.weight` bf16[2048] — no bias
  tensor exists (openinfer-glm52/src/weights.rs:528);
- production decode applies it via `rms_norm_rows_into(..., Q_LORA, rows, ...)`
  (openinfer-glm52/src/mla_front.rs:480; prefill twin :390).
The LayerNorm-with-bias of docs/models/glm52/indexer-forward.md is the DSA
indexer k_norm (`layer_norm_cuda`, dim=128, eps=1e-6, f32 gamma/beta) — a
different kernel owned by the indexer-chain group, not this unit.

Production symbol: `rms_norm_batched_cuda` (csrc/shared/flashinfer_norm.cu:204;
FFI mirror openinfer-kernels/src/ffi/shared.rs:17), same FlashInfer template
as norm.rms_norm with hidden_dim=2048 and the shared checkpoint eps=1e-5.
ABI: VOID return — see the norm.rms_norm adapter.
rows: one CTA per row, per-row reduction self-contained — bit-identical per
row to the rows=1 launch for every rows in {1..64}; no .cu change needed.
"""
from __future__ import annotations

from kernel_lab import data
from kernel_lab.loader import require_torch
from kernel_lab.refs import norm

SYMBOL = "rms_norm_batched_cuda"
Q_LORA = norm.GLM52_Q_LORA


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n", "k"} with n == k == q_lora_rank (all rows live).
    The input is the q_a projection output (pre-norm), modeled as N(0,1) bf16
    like the other act factories."""
    torch = require_torch()
    rows, dim = shape["rows"], shape["n"]
    if dim != Q_LORA or shape["k"] != Q_LORA:
        raise ValueError(f"{SYMBOL} q_a adapter expects n == k == {Q_LORA}, got {shape}")
    x = data.normal_bf16((rows, dim), seed=data.derive_seed(seed, "act"))
    return {
        "x": x,                               # bf16 [rows, 2048] (q_a proj output)
        "weight": norm.norm_weight_bf16((dim,), seed=data.derive_seed(seed, "norm_weight")),
        "out": torch.empty((rows, dim), dtype=torch.bfloat16, device=x.device),
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
