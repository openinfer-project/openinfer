"""Torch references + pure-stdlib shape math for the GLM5.2 router group.

Units: router.noaux_tc (gate GEMV 6144->256 + fused sigmoid noaux_tc select)
and router.select (standalone select). Conventions mirror the Rust device
gate openinfer-kernels/tests/glm52_router_smoke.rs:

- selection order is the strict total order (choice desc, expert index asc),
  identical to the kernel's rank-count selection (`better_router_choice`);
- select references consume the SAME f32 logits the kernel sees (the
  noaux_tc reference reads the kernel's own logits buffer back), so there is
  no f64-vs-f32 sigmoid near-tie flake;
- the logits reference is an f64 dot product (smoke `host_logits` analogue),
  which also immunizes the gate against torch TF32 matmul defaults.

Model constants (openinfer-kernels/src/ops/glm52/router.rs,
openinfer-glm52/src/config.rs): hidden 6144, routed experts 256, topk 8,
routed_scaling_factor 2.5. The router is EP-agnostic: every rank scores the
full 256 experts.
"""
from __future__ import annotations

HIDDEN = 6144
EXPERTS = 256
TOPK = 8
ROUTE_SCALE = 2.5

# Launch geometry of router_scores_topk_normalize_kernel
# (csrc/glm52/glm52_router.cu): grid = padded_tokens blocks x 256 threads,
# dynamic smem = threads * 2 * f32 + topk * f32.
SELECT_THREADS = 256
SELECT_SMEM_BYTES = SELECT_THREADS * 2 * 4 + TOPK * 4  # 2080

# glm52_min_gemv compile-time dispatch cap (glm52_min_gemv.cuh kMaxTokens,
# = GLM52_MAX_BATCH_PER_RANK): padded_tokens above it fails closed with
# CUDA_ERROR_INVALID_VALUE. Bounds router.noaux_tc; router.select launches
# one block per token and has no such cap.
MIN_GEMV_MAX_TOKENS = 8


def buffer_sizes(rows: int) -> dict:
    """Authoritative per-rows buffer table for the router units (the shared
    manifest ShapeVariant is GEMV-flavored and does not fit the router ABI).
    Pure stdlib so the CPU tests can pin the derivation."""
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    return {
        "hidden_elems": rows * HIDDEN,       # bf16 [rows, 6144] — noaux_tc only
        "gate_elems": EXPERTS * HIDDEN,      # bf16 [256, 6144], rows-independent
        "bias_bytes": EXPERTS * 4,           # f32 [256] as raw bytes (ops ABI: u8 slice)
        "logits_elems": rows * EXPERTS,      # f32 [rows, 256] inter-kernel scratch
        "topk_weight_elems": rows * TOPK,    # f32 [rows, 8]
        "topk_idx_elems": rows * TOPK,       # i32 [rows, 8]
        "select_smem_bytes": SELECT_SMEM_BYTES,
        "select_grid_blocks": rows,          # grid.x = padded_tokens = rows
        "gemv_grid_blocks": EXPERTS,         # one 128-thread block per expert row
        "caller_scratch_bytes": 0,           # no ksplit-style caller scratch
    }


def sigmoid_select_ref(logits_f32, bias_f32, topk: int = TOPK, route_scale: float = ROUTE_SCALE):
    """noaux_tc selection reference, exact-order (smoke `host_select` analogue).

    scores = sigmoid(logits); choice = scores + bias; picks = top-`topk` under
    (choice desc, index asc); weights = scores[picks] * route_scale /
    sum(scores[picks]). The kernel's sequential route-order accumulation of
    the selected-score sum is mirrored add-by-add, so the only remaining
    delta vs the kernel is CUDA expf vs torch sigmoid ulp wobble.

    Returns (ref_idx int32 [rows, topk], ref_weight f32 [rows, topk]).
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    scores = torch.sigmoid(logits_f32.to(torch.float32))
    choice = scores + bias_f32.to(torch.float32)
    # Stable descending sort == (value desc, index asc): equal keys keep
    # ascending expert index, the kernel's tie-break.
    order = torch.sort(choice, dim=-1, descending=True, stable=True).indices
    picks = order[:, :topk]
    sel = scores.gather(1, picks)  # [rows, topk] in route order
    selected_sum = sel[:, 0].clone()
    for r in range(1, topk):
        selected_sum = selected_sum + sel[:, r]
    scale = torch.where(selected_sum > 0, route_scale / selected_sum, torch.zeros_like(selected_sum))
    weights = sel * scale.unsqueeze(1)
    return picks.to(torch.int32), weights


def router_logits_ref(hidden_bf16, gate_bf16):
    """f64 dot-product reference for the gate GEMV (smoke `host_logits`
    analogue; f64 dodges torch TF32 matmul defaults). Returns f32
    [rows, experts]."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    ref = hidden_bf16.to(torch.float64) @ gate_bf16.to(torch.float64).t()
    return ref.to(torch.float32)


def assert_select_exact(idx, weight, ref_idx, ref_weight, weight_abs_tol: float = 1e-4, tag: str = "select"):
    """Hard gate, smoke-test convention: topk_idx position-for-position exact,
    topk_weight to f32 rounding. Raises AssertionError with diagnostics —
    called from the adapters' reference() so `kernel_lab check` fails loudly
    instead of silently gating on the wrong tensor."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    got_idx = idx.to(torch.int32)
    if not torch.equal(got_idx, ref_idx):
        bad = (got_idx != ref_idx).nonzero()[:8]
        raise AssertionError(
            f"{tag}: topk_idx mismatch at {bad.shape[0]}+ positions, first {bad.tolist()}; "
            f"got {got_idx.flatten()[:16].tolist()} want {ref_idx.flatten()[:16].tolist()}"
        )
    err = float((weight.to(torch.float32) - ref_weight).abs().max())
    if err >= weight_abs_tol:
        raise AssertionError(f"{tag}: topk_weight max abs err {err:.3e} >= {weight_abs_tol}")
