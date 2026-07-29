"""Torch reference for the shared-expert fp8 SwiGLU chain (semantic-level net).

The chain (production fp8_mlp_into, openinfer-glm52/src/fp8.rs:502) is
gate|up GEMV -> bf16-rounded SiLU -> down GEMV. This reference mirrors BOTH
intermediate bf16 roundings exactly (the fused reduce+SiLU is bit-identical
to the standalone pair by construction, so the rounding points are part of
the semantics, not of the launch route); what it cannot mirror is the f32
accumulation order inside the two GEMVs — the manifest tolerance absorbs
that plus expf/sigmoid ULP differences.
"""
from __future__ import annotations

from kernel_lab.refs import fp8_gemv


def fp8_swiglu_ref(act_bf16, gu_w_u8, gu_s_f32, dn_w_u8, dn_s_f32, inter: int):
    """out[rows, hidden] = down(silu(gate(act)) * up(act)) in f32.

    gu_w is the PACKED gate|up e4m3 weight [2*inter, hidden] (gate rows
    [0, inter), up rows [inter, 2*inter)) with its [2*inter/128, hidden/128]
    block scales; dn_w is [hidden, inter]."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    n_gu, hidden = gu_w_u8.shape
    if n_gu != 2 * inter:
        raise ValueError(f"packed gate|up rows {n_gu} != 2*inter {2 * inter}")
    gu = fp8_gemv.fp8_weight_only_gemv_ref(act_bf16, gu_w_u8, gu_s_f32, n_gu, hidden)
    # Both production routes round g/u to bf16 before the SiLU math (the
    # register tile writes bf16 GEMV output; the fused reduce+SiLU rounds the
    # two f32 sums the same way) — mirror it, then the silu_out bf16 store.
    g = gu[:, :inter].to(torch.bfloat16).to(torch.float32)
    u = gu[:, inter:].to(torch.bfloat16).to(torch.float32)
    s = g * torch.sigmoid(g) * u  # kernel: sg = 1/(1+expf(-g)); out = g*sg*u
    s = s.to(torch.bfloat16)      # silu_out is the down GEMV's bf16 input
    return fp8_gemv.fp8_weight_only_gemv_ref(s, dn_w_u8, dn_s_f32, hidden, inter)
