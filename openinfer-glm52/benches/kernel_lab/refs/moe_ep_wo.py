"""MoE EP weight-only expert-chain references + shared factories (phase 2).

Units covered (production chain `glm52_moe_ep_wo_routed_forward`,
openinfer-glm52/src/moe_ep_wo.rs:166; kernels csrc/glm52/glm52_moe_ep_wo.cu;
FFI mirror openinfer-kernels/src/ffi/glm52.rs:174-210):

- `moe_ep_wo.tiles` — `glm52_moe_ep_wo_tiles_cuda`: psum_expert (i32 aligned
  running ends, the DeepEP dispatch metadata) -> compact tile work list
  (int2 entries {aligned row base, expert | rows<<16} + device tile count).
  Integer-exact gate.
- `moe_ep_wo.w13_mma` — `glm52_moe_ep_wo_masked_mma_cuda` W13 operand
  (n=4096, k=6144): masked grouped bf16 x e4m3-block-scale weight mma over
  the tile list, no row weights.
- `moe_ep_wo.silu` — `glm52_moe_ep_wo_silu_cuda`: silu(gate) * up over the
  tile rows, bf16 out (inter=2048; input rows [., 2*inter] gate|up).
- `moe_ep_wo.w2_mma` — `glm52_moe_ep_wo_masked_mma_cuda` W2 operand
  (n=6144, k=2048) with the dispatch route weight fused: `row_weights` scales
  the f32 accumulator BEFORE the bf16 store (glm52_moe_ep_wo.cu:211-227).

Capacity-proportional launch (the bench-honesty contract): production shapes
the grids at the startup tile budget `state.max_tiles` (protocol max global
tokens = ep x GLM52_MAX_BATCH_PER_RANK, openinfer-glm52/src/moe_ep_wo.rs:137)
and passes the per-step row bound `m_capacity = bound_rows` (same file:197);
blocks past the device tile count retire on one 4-byte read. The harness
mirrors both exactly — never re-shapes the grid to the actual tile count,
which would flatter the kernel.

Masked input layout = the DeepEP aligned receive slots, driven exactly like
the Rust smoke test (openinfer-kernels/tests/glm52_moe_ep_wo_smoke.rs): a
synthetic per-expert count vector places each expert's real rows contiguously
at a 64-aligned segment start; alignment-gap rows keep a sentinel in the
output buffers and are hard-asserted untouched by the adapters.

Routing distribution: seeded, reproducible, non-degenerate — per-expert
hotness logits ~ N(0, SKEW_SIGMA) (lognormal skew like the production
router's hot/cold experts; neither all-tokens-one-expert nor perfectly
uniform), per token TOPK=8 DISTINCT experts via Gumbel top-k (PPS without
replacement), folded onto this rank's local window [0, groups). Only
`random.Random.random()` arithmetic is used (Box-Muller / Gumbel are our own
formulas on top), so the stream is stable across Python versions.

ep x rows sweep: the shared CLI iterates only the rows axis (rows = per-rank
decode bucket; global_tokens = ep x rows). Until it grows an --ep selector,
this module provides the group sweep:

    PYTHONPATH=openinfer-glm52/benches \
        python3 -m kernel_lab.refs.moe_ep_wo check moe_ep_wo.w13_mma --ep 4
    PYTHONPATH=openinfer-glm52/benches \
        python3 -m kernel_lab.refs.moe_ep_wo bench moe_ep_wo.w2_mma --ep 16 --rows 8

Module level stays torch-free (CPU pytest / registry import this file).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass

from kernel_lab import data
from kernel_lab.loader import require_torch
from kernel_lab.refs.fp8_gemv import e4m3_decode_torch

GROUP_UNITS = ("moe_ep_wo.tiles", "moe_ep_wo.w13_mma", "moe_ep_wo.silu", "moe_ep_wo.w2_mma")

TILES_SYMBOL = "glm52_moe_ep_wo_tiles_cuda"
MMA_SYMBOL = "glm52_moe_ep_wo_masked_mma_cuda"
SILU_SYMBOL = "glm52_moe_ep_wo_silu_cuda"

# ---- model / protocol constants ----
EXPERTS = 256               # GLM52_ROUTED_EXPERTS (config.rs:43)
TOPK = 8                    # GLM52_TOPK (config.rs:44)
HIDDEN = 6144               # GLM52_HIDDEN
INTER = 2048                # GLM52_EXPERT_INTERMEDIATE
W13_N, W13_K = 2 * INTER, HIDDEN  # 4096 x 6144 (moe_decode.rs:33-34)
W2_N, W2_K = HIDDEN, INTER        # 6144 x 2048 (moe_decode.rs:35-36)
ALIGN = 64                  # GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT
TILE_ROWS = 8               # GLM52_MOE_EP_WO_TILE_ROWS (8 | 64: a tile never
                            # straddles an expert segment)
MAX_BATCH_PER_RANK = 8      # GLM52_MAX_BATCH_PER_RANK (model/mod.rs:87)
SHIM_DECODE_MAX_TOKENS = 128  # kDecodeMaxTokens (deepep_config_glm52*.cuh) —
                            # shim buffer sizing only; the model's own step cap
                            # is MAX_BATCH_PER_RANK
FP8_BLOCK = data.FP8_BLOCK  # 128
EP_AXES = (4, 8, 16)        # phase-2 EP widths (n_local = 256/ep = 64/32/16)
ROWS_AXES = (1, 2, 4, 8)    # per-rank decode buckets (GLM52_DECODE_BUCKETS)
GLOBAL_TOKEN_AXES = (4, 8, 16, 32, 64, 128)  # union of ep x rows
DEFAULT_EP = EP_AXES[0]
SKEW_SIGMA = 1.2            # lognormal hotness spread (~e^2.4 hot:median)
SENTINEL = -0.5             # bf16-exact gap-row fill; small magnitude keeps
                            # the full-surface rel_l2 gate sensitive to live
                            # rows (a -1232 sentinel would dilute the norm)
ROUTE_WEIGHT_LO = 0.05      # route weights ~ U(0.05, 1.0) — production
                            # sigmoid+norm route weights live in (0, 1)


# ---------------------------------------------------------------------------
# Capacity derivations (pure stdlib — the authoritative phase-2 tables; the
# manifest [shape] n/k only keep `kernel_lab list`'s GEMV-flavored derived
# rows meaningful for act/out)
# ---------------------------------------------------------------------------

def align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def n_local(ep: int) -> int:
    """Rank-local experts at an EP width: 256/ep (EP4/8/16 -> 64/32/16)."""
    if EXPERTS % ep:
        raise ValueError(f"ep {ep} does not partition {EXPERTS} experts")
    return EXPERTS // ep


def global_tokens(ep: int, rows: int) -> int:
    """Global tokens in the step: ep ranks x per-rank decode bucket."""
    return ep * rows


def max_global_tokens(ep: int) -> int:
    """The protocol's max global token count at launch (moe_ep_wo.rs:137)."""
    return ep * MAX_BATCH_PER_RANK


def state_max_tiles(ep: int) -> int:
    """Production startup tile budget `state.max_tiles`
    (glm52_moe_ep_wo_max_tiles(n_local, ep*8, TOPK),
    openinfer-kernels/src/ops/glm52/moe_ep_wo.rs:30): every expert can open
    one partial tile, plus one tile per full TILE_ROWS rows of the global
    expanded budget. The mma/silu grids are shaped at THIS value for every
    bucket — the capacity-proportional contract."""
    return n_local(ep) + (max_global_tokens(ep) * TOPK) // TILE_ROWS


def decode_worst_expanded(ep: int) -> int:
    """`decode_worst_expanded_tokens` (deepep_config_derived.cuh:57), the
    production recv-buffer row capacity: align_up(ep * 128 * min(TOPK,
    n_local) + 63 * n_local, 64)."""
    return align_up(
        ep * SHIM_DECODE_MAX_TOKENS * min(TOPK, n_local(ep)) + (ALIGN - 1) * n_local(ep),
        ALIGN,
    )


def bound_rows(ep: int, gt: int) -> int:
    """Per-step `m_capacity` (moe_ep_wo.rs:197):
    min(expanded, gt*TOPK + 63 * min(gt*TOPK, n_local)). Tight against the
    tiles kernel's trap bound: the last segment's aligned end is at most
    gt*TOPK + 63*(active-1) + 63 <= this value, so a valid layout never traps."""
    expanded_rows = gt * TOPK
    return min(
        decode_worst_expanded(ep),
        expanded_rows + (ALIGN - 1) * min(expanded_rows, n_local(ep)),
    )


def shape_point(ep: int, rows: int) -> dict:
    """One (ep, rows) axis point with every launch parameter."""
    if ep not in EP_AXES:
        raise ValueError(f"ep {ep} not in phase-2 axes {EP_AXES}")
    if rows not in ROWS_AXES:
        raise ValueError(f"rows {rows} not in decode buckets {ROWS_AXES}")
    gt = global_tokens(ep, rows)
    return {
        "ep": ep,
        "rows": rows,
        "groups": n_local(ep),
        "global_tokens": gt,
        "bound_rows": bound_rows(ep, gt),
        "max_tiles": state_max_tiles(ep),
        "expanded": decode_worst_expanded(ep),
    }


def iter_shape_points(manifest_shape: dict, ep_axes, rows_axes):
    """ep x rows grid as CLI-style shape dicts (adapters read `ep` via
    shape.get("ep", DEFAULT_EP); the shared CLI never injects it)."""
    for e in ep_axes:
        for r in rows_axes:
            yield {"rows": r, "ep": e, "n": manifest_shape["n"], "k": manifest_shape["k"]}


# ---------------------------------------------------------------------------
# Synthetic aligned receive layout (pure stdlib; ports the Rust smoke test's
# build_layout, openinfer-kernels/tests/glm52_moe_ep_wo_smoke.rs:67)
# ---------------------------------------------------------------------------

@dataclass
class Layout:
    """One synthetic step's DeepEP aligned receive layout."""

    counts: list[int]       # real rows per local expert
    starts: list[int]       # 64-aligned segment starts
    psum: list[int]         # i32 aligned running ends (the dispatch metadata)
    aligned_end: int        # align_up(last psum, 64)
    tiles: list[tuple[int, int, int]]  # (row base, expert, live rows)


def build_layout(counts: list[int]) -> Layout:
    """counts -> starts/psum/tiles. Expert e's rows sit at
    [align_up(psum[e-1], 64), +counts[e]); empty experts pin psum to their
    aligned start (the kernel derives the same count=0 segment). Tiles of
    TILE_ROWS live rows cover [start, start+count) — never into the gap."""
    groups = len(counts)
    starts = [0] * groups
    psum = [0] * groups
    tiles: list[tuple[int, int, int]] = []
    cursor = 0
    for e in range(groups):
        start = 0 if e == 0 else align_up(cursor, ALIGN)
        starts[e] = start
        cursor = start + counts[e]
        psum[e] = cursor
        r = 0
        while r < counts[e]:
            rows = min(TILE_ROWS, counts[e] - r)
            tiles.append((start + r, e, rows))
            r += rows
    return Layout(
        counts=list(counts),
        starts=starts,
        psum=psum,
        aligned_end=align_up(cursor, ALIGN),
        tiles=tiles,
    )


def live_row_mask(layout: Layout, bound: int) -> list[bool]:
    """Per-row liveness over the [0, bound) output surface: real segment rows
    are live; alignment-gap rows (and rows past the last segment) are not."""
    mask = [False] * bound
    for start, count in zip(layout.starts, layout.counts):
        for r in range(start, start + count):
            mask[r] = True
    return mask


def gap_rows(layout: Layout, bound: int) -> list[int]:
    """Gap rows per expert segment — [start+count, next start) — the rows the
    smoke test sentinel-checks; here covering the full surface to `bound`."""
    mask = live_row_mask(layout, bound)
    return [r for r in range(bound) if not mask[r]]


# ---------------------------------------------------------------------------
# Seeded routing distribution (pure stdlib; stable across Python versions —
# only random.Random.random() arithmetic)
# ---------------------------------------------------------------------------

def _standard_normals(rng: random.Random, n: int) -> list[float]:
    """Box-Muller N(0,1) samples from rng.random() pairs."""
    out: list[float] = []
    while len(out) < n:
        u1 = max(rng.random(), 1e-300)
        u2 = rng.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out.append(radius * math.cos(theta))
        out.append(radius * math.sin(theta))
    return out[:n]


def _gumbel(rng: random.Random) -> float:
    u = max(rng.random(), 1e-300)
    return -math.log(-math.log(u))


def _routing_counts_once(ep: int, gt: int, seed: int, purpose: str) -> list[int]:
    """One seeded draw: 256 experts with lognormal hotness logits; per token
    TOPK distinct experts via Gumbel top-k (sampling without replacement
    proportional to exp(logit) — Perturbed-TopK); rows landing on this rank's
    local window [0, groups) form the count vector. Per-token distinctness is
    the production topk contract and bounds every count by `gt` (the tiles
    kernel's row_cap)."""
    rng = random.Random(data.derive_seed(seed, purpose))
    groups = n_local(ep)
    logits = [SKEW_SIGMA * g for g in _standard_normals(rng, EXPERTS)]
    counts = [0] * groups
    for _ in range(gt):
        keyed = sorted(
            ((logits[e] + _gumbel(rng), e) for e in range(EXPERTS)), reverse=True
        )
        for _, e in keyed[:TOPK]:
            if e < groups:
                counts[e] += 1
    return counts


def routing_counts(ep: int, gt: int, seed: int) -> list[int]:
    """The seeded routing distribution: skewed (lognormal hotness, sigma=1.2
    — moderately hotter than the diverse-prompts production point, where
    routing is near-uniform at this batch size,
    docs/lessons/moe-bench-prompt-diversity.md), conditioned on
    non-degenerate by a deterministic retry: a draw with suspiciously little
    local work (near-empty tail of the Binomial-like local hit count, e.g. a
    single active expert) is rejected and redrawn from the next derived
    seed. All draws come from random.Random.random() arithmetic only, so the
    sequence is stable across Python versions and machines."""
    groups = n_local(ep)
    expected = gt * TOPK * groups / EXPERTS
    min_total = max(2, math.ceil(expected / 4))
    for attempt in range(16):
        purpose = f"moe_ep_wo.route.ep{ep}.gt{gt}" + (f".r{attempt}" if attempt else "")
        counts = _routing_counts_once(ep, gt, seed, purpose)
        if sum(counts) >= min_total and sum(1 for c in counts if c) >= min(2, min_total):
            return counts
    raise AssertionError(
        f"routing for ep{ep} gt={gt} stayed degenerate after 16 seeded retries"
    )


def route_weights(layout: Layout, ep: int, gt: int, bound: int, seed: int) -> list[float]:
    """Per aligned row dispatch route weight (W2's row_weights): U(0.05, 1.0)
    on live rows, 0.0 in the gaps (never read — the kernel only indexes
    row_base + col for live tile rows)."""
    rng = random.Random(data.derive_seed(seed, f"moe_ep_wo.rw.ep{ep}.gt{gt}"))
    rw = [0.0] * bound
    for start, count in zip(layout.starts, layout.counts):
        for r in range(start, start + count):
            rw[r] = ROUTE_WEIGHT_LO + (1.0 - ROUTE_WEIGHT_LO) * rng.random()
    return rw


def layout_for(ep: int, gt: int, seed: int) -> Layout:
    """Seeded layout for one (ep, global_tokens) point, with the production
    contract invariants asserted (a violation would device-trap the tiles
    kernel — fail here on the CPU side instead)."""
    counts = routing_counts(ep, gt, seed)
    layout = build_layout(counts)
    if max(counts, default=0) > gt:
        raise AssertionError(f"expert count exceeds row_cap {gt}")
    if len(layout.tiles) > state_max_tiles(ep):
        raise AssertionError(
            f"{len(layout.tiles)} tiles exceed the capacity grid {state_max_tiles(ep)}"
        )
    if layout.aligned_end > bound_rows(ep, gt):
        raise AssertionError(
            f"aligned_end {layout.aligned_end} exceeds bound_rows {bound_rows(ep, gt)}"
        )
    return layout


def expected_tiles_surface(layout: Layout, max_tiles: int) -> list[int]:
    """Flat expected `tiles` buffer: int2 entries {base, expert | rows<<16}
    followed by zeros to 2*max_tiles (untouched slots keep their zero init;
    the kernel writes exactly tile_count entries)."""
    flat: list[int] = []
    for base, expert, rows in layout.tiles:
        flat.append(base)
        flat.append(expert | (rows << 16))
    flat.extend([0] * (2 * max_tiles - len(flat)))
    return flat


# ---------------------------------------------------------------------------
# Torch factories (lazy torch)
# ---------------------------------------------------------------------------

# Weight banks are rows-independent and large (EP4 W13 = 64 x 4096 x 6144
# e4m3 = 1.6 GB); a rows sweep reuses them across every rows value (same
# pattern as the attention group's packed-cache reuse).
_WEIGHT_CACHE: dict = {}


def weight_bank(ep: int, kind: str, seed: int, device: str = "cuda"):
    """(weight u8 [groups, n, k], scales f32 [groups, n/128, k/128]) —
    normal-quantized fp8 per 128x128 block (the checkpoint recipe;
    uniform-random bytes would blow up the mma accumulators). One
    normal_quantized_fp8 call over [groups*n, k]: blocks never straddle an
    expert boundary (n % 128 == 0), so the per-expert block structure is
    identical to per-expert generation."""
    torch = require_torch()
    if kind == "w13":
        n, k = W13_N, W13_K
    elif kind == "w2":
        n, k = W2_N, W2_K
    else:
        raise ValueError(f"kind {kind!r} not in ('w13', 'w2')")
    key = (ep, kind, seed, device)
    hit = _WEIGHT_CACHE.get(key)
    if hit is not None:
        return hit
    groups = n_local(ep)
    weight, scales = data.normal_quantized_fp8(
        groups * n, k, seed=data.derive_seed(seed, f"moe_ep_wo.weight.{kind}"), device=device
    )
    bank = (
        weight.view(groups, n, k),
        scales.view(groups, n // FP8_BLOCK, k // FP8_BLOCK),
    )
    _WEIGHT_CACHE[key] = bank
    return bank


def make_layout_tensors(layout: Layout, max_tiles: int, device: str = "cuda") -> dict:
    """Device views of the dispatch metadata: psum i32 [groups], tiles i32
    zeros [2*max_tiles], tile_count i32 zeros [1] (kernel-written)."""
    torch = require_torch()
    return {
        "psum": torch.tensor(layout.psum, dtype=torch.int32, device=device),
        "tiles": torch.zeros(2 * max_tiles, dtype=torch.int32, device=device),
        "tile_count": torch.zeros(1, dtype=torch.int32, device=device),
    }


# ---------------------------------------------------------------------------
# Torch references (lazy torch)
# ---------------------------------------------------------------------------

def masked_mma_ref(act_bf16, weight_u8, scales_f32, layout: Layout, bound: int,
                   n: int, k: int, weighted: bool, row_weights_f32=None):
    """want f32 [bound, n]: per-expert masked GEMM over the live rows with the
    kernel's association order — per-128-column-block f32 partial, then
    block-scale multiply-accumulate (`macc += scale * cacc`,
    glm52_moe_ep_wo.cu:195-204) — ported from the Rust smoke test's f64 host
    reference (the mma slot order inside a block differs; the 2e-2 gate
    absorbs that and the f32-vs-f64 reorder). W2 (`weighted`) scales the f32
    accumulator by the per-row route weight before the bf16 store — the same
    association as the kernel and the oracle's post-down multiply. Gap rows
    hold SENTINEL (the adapter hard-asserts the kernel left them untouched)."""
    torch = require_torch()
    nb, kb = n // FP8_BLOCK, k // FP8_BLOCK
    groups = len(layout.counts)
    w_bytes = weight_u8.view(groups, n, k)
    s_all = scales_f32.view(groups, nb, kb)
    want = torch.full((bound, n), SENTINEL, dtype=torch.float32, device=act_bf16.device)
    for e, (start, count) in enumerate(zip(layout.starts, layout.counts)):
        if count == 0:
            continue
        x = act_bf16[start : start + count].to(torch.float32)  # [count, k]
        w = e4m3_decode_torch(w_bytes[e])  # [n, k] f32, exact decode
        acc = torch.zeros((count, n), dtype=torch.float32, device=act_bf16.device)
        for b in range(kb):
            lo = b * FP8_BLOCK
            partial = x[:, lo : lo + FP8_BLOCK] @ w[:, lo : lo + FP8_BLOCK].t()
            # scale per (output 128-row block, k-block b), broadcast over the block
            acc += s_all[e, :, b].repeat_interleave(FP8_BLOCK) * partial
        if weighted:
            acc = acc * row_weights_f32[start : start + count].unsqueeze(1)
        want[start : start + count] = acc
    return want


def silu_ref(gate_up_bf16, layout: Layout, bound: int, inter: int):
    """want f32 [bound, inter]: silu(gate) * up per live row, f32 math in the
    kernel's order ((gate * sigmoid(gate)) * up), rounded to bf16 and
    widened back — the bit-exact comparison surface (the CLI compares the
    bf16 kernel store against this; an unrounded f32 reference would
    re-measure the bf16 store floor instead of bit-identity). Gap rows
    SENTINEL.

    Bit-exactness argument (target rel_l2 = 0): bf16 -> f32 widening is exact;
    the kernel's `1.0f / (1.0f + expf(-gate))` is the same expression torch's
    CUDA f32 sigmoid computes (`1/(1+exp(-x))`, precise expf — nvcc runs with
    -O3 and NO fast-math on both sides, so both divisions are div.rn);
    verified bit-identical over every bf16 value in [-30, 30] on GB300
    (2026-07-29 probe: 0/200001 f32 mismatches). The two f32 multiplies share
    the kernel's left-to-right order; both stores round to bf16 RNE. No
    tensor/CPU-scalar division anywhere (the GB300 1-ulp trap from the quant
    group does not apply)."""
    torch = require_torch()
    want = torch.full((bound, inter), SENTINEL, dtype=torch.float32, device=gate_up_bf16.device)
    for start, count in zip(layout.starts, layout.counts):
        if count == 0:
            continue
        rows = gate_up_bf16[start : start + count].to(torch.float32)
        gate, up = rows[:, :inter], rows[:, inter:]
        want[start : start + count] = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).to(torch.float32)
    return want


# ---------------------------------------------------------------------------
# Adapter-side hard gates (the check CLI's single-tensor rel_l2 is the outer
# net; these port the Rust smoke test's exact checks with row-level
# diagnostics — "多输出精确门放 adapter 内硬断言")
# ---------------------------------------------------------------------------

def assert_gap_sentinel(out_bf16, layout: Layout, width: int) -> None:
    """Alignment-gap rows keep the SENTINEL — the kernel must only write live
    tile rows (smoke test: alignment-gap sentinel check). Exact bf16 compare
    on column 0 of every gap row plus the full last-gap-row span would be
    overkill; column-strided exact compare catches stray writes."""
    torch = require_torch()
    gaps = gap_rows(layout, out_bf16.shape[0])
    if not gaps:
        return
    idx = torch.tensor(gaps, dtype=torch.long, device=out_bf16.device)
    got = out_bf16[idx][:, :: max(1, width // 32)]
    want = torch.full_like(got, SENTINEL)
    bad = int((got != want).sum().item())
    if bad:
        first = gaps[int((got != want).any(dim=1).nonzero()[0].item())]
        raise AssertionError(
            f"gap row sentinel violated ({bad} sampled elements, first gap row {first}): "
            "the kernel wrote outside the tile list's live rows"
        )


def assert_live_rows_close(out_bf16, want_f32, layout: Layout, rel: float, floor: float) -> None:
    """Smoke-test per-element gate over the live rows:
    |got - want| / max(|want|, floor) < rel (the f64 host reference used
    rel=2e-2, floor=1.0 for the mma; 1e-2, floor=0.25 for silu)."""
    torch = require_torch()
    rows: list[int] = []
    for start, count in zip(layout.starts, layout.counts):
        rows.extend(range(start, start + count))
    if not rows:
        return
    idx = torch.tensor(rows, dtype=torch.long, device=out_bf16.device)
    got = out_bf16[idx].to(torch.float32)
    want = want_f32[idx]
    errs = (got - want).abs() / want.abs().clamp_min(floor)
    worst = float(errs.max().item())
    if worst >= rel:
        pos = (errs == errs.max()).nonzero()[0]
        raise AssertionError(
            f"live-row gate {worst:.3e} >= {rel} at row {rows[int(pos[0].item())]} "
            f"col {int(pos[1].item())}"
        )


# ---------------------------------------------------------------------------
# ep x rows sweep driver (group-local; the shared CLI iterates rows only)
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m kernel_lab.refs.moe_ep_wo",
        description="moe_ep_wo-group ep x rows sweep (check vs reference / bench)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "bench"):
        p = sub.add_parser(name)
        p.add_argument("unit", choices=GROUP_UNITS)
        p.add_argument("--ep", type=int, action="append", default=None)
        p.add_argument("--rows", type=int, action="append", default=None)
        p.add_argument("--so", default=None)
        p.add_argument("--seed", type=int, default=0x5EED)
        if name == "bench":
            p.add_argument("--warmup", type=int, default=20)
            p.add_argument("--rounds", type=int, default=30)
            p.add_argument("--inner", type=int, default=10)
    args = parser.parse_args(argv)

    from kernel_lab import loader, registry, timing
    from kernel_lab.refs import compute_metrics

    units = registry.discover()
    if args.unit not in units:
        raise SystemExit(f"{args.unit}: not registered; available: {', '.join(units)}")
    u = units[args.unit]
    torch = loader.require_torch()
    if not torch.cuda.is_available():
        raise SystemExit("kernel_lab: no CUDA device visible")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if u.manifest.capability.get("blackwell_only") and major < 10:
        raise SystemExit(f"{u.name}: Blackwell-only unit (fail-closed); device capability major={major}")
    lib = loader.load_library(args.so)
    stream = loader.current_stream_ptr()

    ep_axes = args.ep or list(u.manifest.axes.get("ep", (DEFAULT_EP,)))
    rows_axes = args.rows or list(u.manifest.axes.get("rows", ()))
    ok = True
    for shape in iter_shape_points(u.manifest.shape, ep_axes, rows_axes):
        tensors = u.adapter.make_inputs(shape, args.seed)
        if args.cmd == "check":
            u.adapter.run(lib, tensors, shape, stream)
            torch.cuda.synchronize()
            want = u.adapter.reference(tensors, shape)
            metrics = compute_metrics(tensors["out"], want)
            limit = u.manifest.tolerance.get("rel_l2")
            passed = limit is None or metrics["rel_l2"] <= limit
            ok &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {u.name} ep={shape['ep']} rows={shape['rows']} "
                  f"global_tokens={shape['ep'] * shape['rows']} ({arch})")
            print(f"       rel_l2={metrics['rel_l2']:.4e} (tol {limit})  cosine={metrics['cosine']:.6f}  "
                  f"max_abs={metrics['max_abs']:.4e}  mean_abs={metrics['mean_abs']:.4e}")
        else:
            stats = timing.bench(
                lambda: u.adapter.run(lib, tensors, shape, stream),
                args.warmup, args.rounds, args.inner,
            )
            print(f"bench {u.name} ep={shape['ep']} rows={shape['rows']} "
                  f"global_tokens={shape['ep'] * shape['rows']}: "
                  f"median={stats.median_us:.2f} us  p50={stats.p50_us:.2f}  p99={stats.p99_us:.2f}  "
                  f"mean={stats.mean_us:.2f}  rounds={len(stats.samples_us)} ({arch})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
