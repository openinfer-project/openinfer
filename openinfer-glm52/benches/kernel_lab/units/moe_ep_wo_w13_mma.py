"""Adapter for moe_ep_wo.w13_mma — the W13 (gate|up) masked grouped mma.

Production symbol: `glm52_moe_ep_wo_masked_mma_cuda` with the W13 operand
(n=4096, k=6144; csrc/glm52/glm52_moe_ep_wo.cu; FFI mirror
openinfer-kernels/src/ffi/glm52.rs:188; production call
openinfer-glm52/src/moe_ep_wo.rs:237 with row_weights = None).

Launch contract (capacity-proportional, mirrors production): grid
dim3(n/128, state_max_tiles(ep)) x 128 threads; blocks past the device tile
count retire on one 4-byte read, so the timed launch always carries the
startup tile budget, never the actual tile count. The tiles list is built by
the production tiles kernel lazily on the first run() — plan-time work like
production's per-layer chain order, absorbed by bench warmup; timed runs
launch the mma only. Buffers sit at the per-step bound_rows capacity
(moe_ep_wo.rs:197), not the actual row count.

Inputs: activation bf16 [bound_rows, 6144] over the DeepEP aligned receive
slots (gap rows filled too — the kernel must never address them); weight
bank normal-quantized e4m3 [groups, 4096, 6144] + f32 block scales
[groups, 32, 48] (uniform-random bytes would blow up the accumulators);
out bf16 [bound_rows, 4096] pre-filled with the SENTINEL so alignment-gap
rows prove they were untouched (adapter hard-assert, smoke-test style).

Gate: torch f32 per-expert masked GEMM with the kernel's per-128-block
partial x scale association (refs/moe_ep_wo.masked_mma_ref, ported from the
Rust smoke test's f64 host reference); manifest rel_l2 0.02 (bf16 store
floor + f32 reorder) plus the smoke test's per-element 2e-2 hard gate and
the gap-sentinel hard assert inside reference().
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import moe_ep_wo as moe

SYMBOL = moe.MMA_SYMBOL
TILES_SYMBOL = moe.TILES_SYMBOL
N, K = moe.W13_N, moe.W13_K  # 4096 x 6144 (gate|up)


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
        (bound, K), seed=data.derive_seed(seed, "moe_ep_wo.act.w13")
    )
    weight, scales = moe.weight_bank(ep, "w13", seed, device=act.device)
    tensors = moe.make_layout_tensors(layout, pt["max_tiles"], device=act.device)
    tensors.update(
        {
            "act": act,                    # bf16 [bound, 6144] aligned slots
            "weight": weight,              # e4m3 u8 [groups, 4096, 6144]
            "scales": scales,              # f32 [groups, 32, 48]
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
    """One production W13 masked-mma launch on `stream` (row_weights NULL)."""
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
            ctypes.c_void_p,   # row_weights f32 or NULL (NULL on W13)
            ctypes.c_void_p,   # out bf16 [bound, n]
            ctypes.c_int,      # n = 4096
            ctypes.c_int,      # k = 6144
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
        None,
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
        tensors["out"].shape[0], N, K, weighted=False,
    )
    moe.assert_gap_sentinel(tensors["out"], layout, N)
    moe.assert_live_rows_close(tensors["out"], want, layout, rel=2e-2, floor=1.0)
    return want
