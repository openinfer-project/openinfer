"""Torch references for the proj-gemv group (o_proj + q_a|kv_a pair).

o_proj reuses refs.fp8_gemv.fp8_weight_only_gemv_ref unchanged (n=6144 is
128-divisible). The pair unit needs two things the shared ref does not cover,
so they live here instead of touching shared files:

(a) kv_a's partial-N width — n=576 is NOT a multiple of 128, yet the
    checkpoint stores ceil(576/128)=5 scale rows and the kernel indexes them
    by `scale[(n0>>7) * (k/128) + (kk>>7)]` (a warp's rows never straddle a
    /128 boundary). `block_scale_gemv_partial_n_ref` models exactly that
    row-block indexing.
(b) the two-projections-one-activation pair surface — `qa_kva_pair_ref` is
    the two independent single-GEMV refs concatenated along n, matching the
    kernel's concatenated grids (each row's dot order is identical to a solo
    launch).

Like every kernel_lab reference these are semantic-level nets: the f32
association order of the production kernel is NOT reproduced; the manifest
tolerance absorbs the reorder.
"""
from __future__ import annotations

from kernel_lab.data import FP8_BLOCK
from kernel_lab.refs import fp8_gemv


def block_scale_gemv_partial_n_ref(act_bf16, weight_u8, scales_f32, n: int, k: int):
    """fp8 weight-only GEMV ref for n not a multiple of 128 (kv_a: 576).

    deq(W)[r, c] = e4m3(W[r, c]) * scale[r >> 7, c >> 7], scale shaped
    [ceil(n/128), k/128] — the kernel's exact scale association. out[bs, n] =
    act[bs, k] @ deq(W)^T accumulated in f32, mirroring
    fp8_gemv.fp8_weight_only_gemv_ref for the divisible case.
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    if k % FP8_BLOCK:
        raise ValueError(f"reference needs {FP8_BLOCK}-divisible k, got {k}")
    s = scales_f32.view(-1, k // FP8_BLOCK)
    if s.shape[0] != -(-n // FP8_BLOCK):
        raise ValueError(f"scale rows {s.shape[0]} != ceil({n}/{FP8_BLOCK})")
    w = fp8_gemv.e4m3_decode_torch(weight_u8).view(n, k)
    row_block = torch.arange(n, device=w.device) // FP8_BLOCK
    deq = w * s[row_block].repeat_interleave(FP8_BLOCK, dim=1)
    return act_bf16.to(torch.float32) @ deq.t()


def qa_kva_pair_ref(
    act_bf16,
    weight_a_u8,
    scales_a,
    weight_b_u8,
    scales_b,
    n_a: int,
    n_b: int,
    k: int,
):
    """out = cat(q_a_ref, kv_a_ref) along n over the shared activation row.

    The pair kernel is two independent GEMVs launched on concatenated grids
    (NOT packed weights), so the reference is the two single-projection refs
    concatenated: q_a takes the 128-divisible shared ref, kv_a the partial-N
    one above. act_bf16 is [1, k]; the result is [1, n_a + n_b].
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    ref_a = fp8_gemv.fp8_weight_only_gemv_ref(act_bf16, weight_a_u8, scales_a, n_a, k)
    ref_b = block_scale_gemv_partial_n_ref(act_bf16, weight_b_u8, scales_b, n_b, k)
    return torch.cat([ref_a, ref_b], dim=-1)
