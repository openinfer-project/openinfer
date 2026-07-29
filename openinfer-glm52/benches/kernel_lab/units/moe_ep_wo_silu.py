"""Adapter for moe_ep_wo.silu — the tile-list SiLU (silu(gate) * up).

Production symbol: `glm52_moe_ep_wo_silu_cuda`
(csrc/glm52/glm52_moe_ep_wo.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs:202;
production call openinfer-glm52/src/moe_ep_wo.rs:252). Elementwise over the
tile rows: input rows are the W13 gate|up outputs [., 2*inter] (inter=2048),
output the W2 activation rows [., inter], all in the DeepEP aligned receive
layout. No route weight here — it applies to the f32 W2 accumulator in the
mma instead (see the w2_mma unit).

Launch contract: grid dim3(state_max_tiles(ep), TILE_ROWS=8) x 256 threads;
blocks at/past the device tile count (or past the tile's live rows) retire
immediately — the capacity-proportional grid mirrors production. The tiles
list is built by the production tiles kernel lazily on the first run()
(plan-time, absorbed by bench warmup). Buffers at bound_rows capacity; out
pre-filled with the SENTINEL.

Gate: bit-exact (manifest rel_l2 1e-6), MEASURED rel_l2 = 0.0 on GB300
sm_103 tray03 across the full ep x rows axis (2026-07-29). Argument: bf16 ->
f32 widening exact; the kernel's `1.0f / (1.0f + expf(-gate))` is bit-
identical to torch's CUDA f32 sigmoid (probed over every bf16 value in
[-30, 30]: 0/200001 f32 mismatches); the two f32 multiplies share the
kernel's left-to-right order; both stores round to bf16 RNE. The reference
rounds its output to bf16 before comparison — an unrounded f32 reference
re-measures the bf16 store floor (~2.2e-4) as a false FAIL. Plus the
gap-sentinel hard assert inside reference().
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import moe_ep_wo as moe

SYMBOL = moe.SILU_SYMBOL
TILES_SYMBOL = moe.TILES_SYMBOL
INTER = moe.INTER  # 2048 (gate|up input rows are [., 2*INTER])


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ep"?, ...} at production capacity: buffers at
    bound_rows(ep, ep*rows); global_tokens = ep * rows."""
    torch = require_torch()
    ep = int(shape.get("ep", moe.DEFAULT_EP))
    pt = moe.shape_point(ep, shape["rows"])
    bound = pt["bound_rows"]
    layout = moe.layout_for(ep, pt["global_tokens"], seed)
    # N(0,1) gates stress the sigmoid's sensitive region (|gate| <= ~4),
    # where an expf ulp difference would show; gap rows filled too.
    gate_up = data.normal_bf16(
        (bound, 2 * INTER), seed=data.derive_seed(seed, "moe_ep_wo.act.silu")
    )
    tensors = moe.make_layout_tensors(layout, pt["max_tiles"], device=gate_up.device)
    tensors.update(
        {
            "gate_up": gate_up,            # bf16 [bound, 4096] aligned slots
            "out": torch.full((bound, INTER), moe.SENTINEL, dtype=torch.bfloat16, device=gate_up.device),
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
    """One production SiLU launch on `stream` (c_void_p cudaStream_t)."""
    _ensure_tiles(lib, tensors, stream)
    pt = tensors["_point"]
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # input bf16 [bound, 2*inter] aligned rows
            ctypes.c_void_p,   # tiles
            ctypes.c_void_p,   # tile_count
            ctypes.c_void_p,   # output bf16 [bound, inter]
            ctypes.c_int,      # inter = 2048
            ctypes.c_int,      # max_tiles (capacity grid)
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["gate_up"]),
        as_dev_ptr(tensors["tiles"]),
        as_dev_ptr(tensors["tile_count"]),
        as_dev_ptr(tensors["out"]),
        INTER,
        pt["max_tiles"],
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    layout = tensors["_layout"]
    want = moe.silu_ref(tensors["gate_up"], layout, tensors["out"].shape[0], INTER)
    moe.assert_gap_sentinel(tensors["out"], layout, INTER)
    return want
