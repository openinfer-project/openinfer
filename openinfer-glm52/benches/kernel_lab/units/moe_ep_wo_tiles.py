"""Adapter for moe_ep_wo.tiles — the tile-list metadata kernel.

Production symbol: `glm52_moe_ep_wo_tiles_cuda`
(csrc/glm52/glm52_moe_ep_wo.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs:177;
production call openinfer-glm52/src/moe_ep_wo.rs:225). One block scans
psum_expert (i32 aligned running ends from the DeepEP dispatch) into the
compact int2 tile work list {aligned row base, expert | rows<<16} plus a
device tile count; the kernel device-traps on any segment past m_capacity
(bound_rows) or holding more than masked_cap (global_tokens) rows — the
harness's seeded layouts stay valid by construction (refs/moe_ep_wo.py
asserts the invariants on the CPU side).

Launch contract: grid (1) x 32 threads; parameters (groups = 256/ep,
m_capacity = bound_rows, masked_cap = global_tokens, max_tiles = the
production startup tile budget state_max_tiles(ep)) — the same values the
production chain passes per step.

Gate: integer-exact. The comparison surface is the flat i32 tiles buffer
(values < 2^24, exact in f32); the adapter additionally hard-asserts the
device tile count against the layout builder (the CLI's single-tensor
rel_l2 cannot see the count buffer).
"""
from __future__ import annotations

import ctypes

from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import moe_ep_wo as moe

SYMBOL = moe.TILES_SYMBOL


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "ep"?, ...}: rows = per-rank decode bucket,
    ep = EP width (default EP4); global_tokens = ep * rows."""
    torch = require_torch()
    ep = int(shape.get("ep", moe.DEFAULT_EP))
    pt = moe.shape_point(ep, shape["rows"])
    layout = moe.layout_for(ep, pt["global_tokens"], seed)
    tensors = moe.make_layout_tensors(layout, pt["max_tiles"])
    tensors["out"] = tensors["tiles"]  # the CLI comparison surface
    tensors["_layout"] = layout
    tensors["_point"] = pt
    return tensors


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production tiles launch on `stream` (c_void_p cudaStream_t)."""
    pt = tensors["_point"]
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # psum_expert i32 [groups]
            ctypes.c_void_p,   # tiles i32 [2*max_tiles] (int2 entries)
            ctypes.c_void_p,   # tile_count i32 [1]
            ctypes.c_int,      # groups = n_local(ep)
            ctypes.c_int,      # m_capacity = bound_rows
            ctypes.c_int,      # masked_cap = global_tokens (per-expert row cap)
            ctypes.c_int,      # max_tiles = state_max_tiles(ep) (capacity grid)
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
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    """Exact expected tiles surface; hard-asserts the device tile count first
    (runs after the CLI's torch.cuda.synchronize())."""
    torch = require_torch()
    layout = tensors["_layout"]
    pt = tensors["_point"]
    got_count = int(tensors["tile_count"].cpu().item())
    if got_count != len(layout.tiles):
        raise AssertionError(
            f"tile_count {got_count} != expected {len(layout.tiles)} "
            f"(ep={pt['ep']} global_tokens={pt['global_tokens']})"
        )
    return torch.tensor(
        moe.expected_tiles_surface(layout, pt["max_tiles"]),
        dtype=torch.int32,
        device=tensors["tiles"].device,
    )
