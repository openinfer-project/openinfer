"""Adapter for moe_ep_wo.w2_mma — the W2 (down) masked grouped mma with the
fused dispatch route weight.

Production symbol: `glm52_moe_ep_wo_masked_mma_cuda` with the W2 operand
(n=6144, k=2048; csrc/glm52/glm52_moe_ep_wo.cu; FFI mirror
openinfer-kernels/src/ffi/glm52.rs:188; production call
openinfer-glm52/src/moe_ep_wo.rs:265 with row_weights = recv_topk_weight).

Route-weight semantics (read from the .cu, glm52_moe_ep_wo.cu:206-227): the
optional per-aligned-row f32 `row_weights` scales the f32 accumulator BEFORE
the bf16 store — `out = bf16(rw[row] * macc)` — the same association as the
oracle reference's post-down multiply. W13 passes NULL instead. The harness
drives row_weights from the seeded layout (U(0.05, 1.0) on live rows, 0.0 in
the never-read gaps).

Launch contract: identical to the W13 twin — grid dim3(n/128,
state_max_tiles(ep)), buffers at bound_rows capacity, tiles built lazily by
the production tiles kernel on first run(). Input activation rows stand in
for the SiLU output (bf16 [bound_rows, 2048] normal); out bf16
[bound_rows, 6144] pre-filled with the SENTINEL.

Gate: torch f32 per-expert masked GEMM with the kernel's per-128-block
partial x scale association and the post-accumulator route-weight multiply
(refs/moe_ep_wo.masked_mma_ref weighted=True); manifest rel_l2 0.02 plus the
smoke test's per-element 2e-2 hard gate and the gap-sentinel hard assert
inside reference().
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import moe_ep_wo as moe

SYMBOL = moe.MMA_SYMBOL
TILES_SYMBOL = moe.TILES_SYMBOL
N, K = moe.W2_N, moe.W2_K  # 6144 x 2048 (down)


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ep"?, ...} at production capacity: buffers at
    bound_rows(ep, ep*rows); global_tokens = ep * rows."""
    torch = require_torch()
    ep = int(shape.get("ep", moe.DEFAULT_EP))
    pt = moe.shape_point(ep, shape["rows"])
    gt = pt["global_tokens"]
    bound = pt["bound_rows"]
    layout = moe.layout_for(ep, gt, seed)
    act = data.normal_bf16(
        (bound, K), seed=data.derive_seed(seed, "moe_ep_wo.act.w2")
    )
    weight, scales = moe.weight_bank(ep, "w2", seed, device=act.device)
    rw = moe.route_weights(layout, ep, gt, bound, seed)
    tensors = moe.make_layout_tensors(layout, pt["max_tiles"], device=act.device)
    tensors.update(
        {
            "act": act,                    # bf16 [bound, 2048] aligned slots
            "weight": weight,              # e4m3 u8 [groups, 6144, 2048]
            "scales": scales,              # f32 [groups, 48, 16]
            "row_weights": torch.tensor(rw, dtype=torch.float32, device=act.device),
            "out": torch.full((bound, N), moe.SENTINEL, dtype=torch.bfloat16, device=act.device),
            "_layout": layout,
            "_point": pt,
            "_tiles_ready": False,         # tiles launched lazily on first run
        }
    )
    return tensors


def _ensure_tiles(lib, tensors: dict, stream) -> None:
    """psum -> tile work list via the production tiles kernel, once per input
    set (plan-time; the bench warmup absorbs it)."""
    if tensors["_tiles_ready"]:
        return
    pt = tensors["_point"]
    fn = resolve(
        lib,
        TILES_SYMBOL,
        [
            ctypes.c_void_p,   # psum_expert
            ctypes.c_void_p,   # tiles
            ctypes.c_void_p,   # tile_count
            ctypes.c_int,      # groups
            ctypes.c_int,      # m_capacity = bound_rows
            ctypes.c_int,      # masked_cap = global_tokens
            ctypes.c_int,      # max_tiles = state_max_tiles(ep)
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["psum"]),
        as_dev_ptr(tensors["tiles"]),
        as_dev_ptr(tensors["tile_count"]),
        pt["groups"],
        pt["bound_rows"],
        pt["global_tokens"],
        pt["max_tiles"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(TILES_SYMBOL, rc)
    tensors["_tiles_ready"] = True


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production W2 masked-mma launch on `stream` (route weight fused)."""
    _ensure_tiles(lib, tensors, stream)
    pt = tensors["_point"]
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # activation bf16 [bound, k] aligned rows
            ctypes.c_void_p,   # weight e4m3 [groups, n, k]
            ctypes.c_void_p,   # weight_scale f32 [groups, n/128, k/128]
            ctypes.c_void_p,   # tiles
            ctypes.c_void_p,   # tile_count
            ctypes.c_void_p,   # row_weights f32 (dispatch route weights)
            ctypes.c_void_p,   # out bf16 [bound, n]
            ctypes.c_int,      # n = 6144
            ctypes.c_int,      # k = 2048
            ctypes.c_int,      # max_tiles (capacity grid)
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["act"]),
        as_dev_ptr(tensors["weight"]),
        as_dev_ptr(tensors["scales"]),
        as_dev_ptr(tensors["tiles"]),
        as_dev_ptr(tensors["tile_count"]),
        as_dev_ptr(tensors["row_weights"]),
        as_dev_ptr(tensors["out"]),
        N,
        K,
        pt["max_tiles"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    layout = tensors["_layout"]
    want = moe.masked_mma_ref(
        tensors["act"], tensors["weight"], tensors["scales"], layout,
        tensors["out"].shape[0], N, K, weighted=True,
        row_weights_f32=tensors["row_weights"],
    )
    moe.assert_gap_sentinel(tensors["out"], layout, N)
    moe.assert_live_rows_close(tensors["out"], want, layout, rel=2e-2, floor=1.0)
    return want
