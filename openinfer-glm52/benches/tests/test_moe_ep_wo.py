"""CPU-only checks for the moe_ep_wo unit group (phase 2) — MUST NOT import torch.

Covers the four MoE EP weight-only expert-chain units (tiles / W13 mma /
SiLU / W2 mma): manifest parsing and axes, the production capacity
derivations (state.max_tiles / bound_rows / decode_worst_expanded), the
aligned-receive layout builder (ported from the Rust smoke test
openinfer-kernels/tests/glm52_moe_ep_wo_smoke.rs and pinned against its
three boundary-shape cases), the seeded skewed routing distribution
(reproducibility, contract invariants, non-degeneracy), and the exact tiles
surface encoding.
"""
import importlib
import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import manifest, registry  # noqa: E402
from kernel_lab.refs import moe_ep_wo as moe  # noqa: E402

MANIFESTS = BENCHES / "manifests"
SEED = 0x5EED

UNITS = {
    "moe_ep_wo.tiles": (moe.TILES_SYMBOL, "moe_ep_wo_tiles", 2, 128),
    "moe_ep_wo.w13_mma": (moe.MMA_SYMBOL, "moe_ep_wo_w13_mma", 4096, 6144),
    "moe_ep_wo.silu": (moe.SILU_SYMBOL, "moe_ep_wo_silu", 2048, 4096),
    "moe_ep_wo.w2_mma": (moe.MMA_SYMBOL, "moe_ep_wo_w2_mma", 6144, 2048),
}
EXPECTED_ROWS = (1, 2, 4, 8)
EXPECTED_EP = (4, 8, 16)
EXPECTED_GT = (4, 8, 16, 32, 64, 128)


def test_no_torch_leak():
    assert "torch" not in sys.modules, "moe_ep_wo refs/adapters must stay torch-free at import"


# --- manifests -----------------------------------------------------------------


def test_manifests_parse():
    for unit, (symbol, adapter, n, k) in UNITS.items():
        m = manifest.load_manifest(MANIFESTS / f"{unit}.toml")
        assert m.unit == unit and m.phase == 2, f"{unit}: phase-2 unit"
        assert m.symbol == symbol and m.adapter == adapter
        assert m.rows == EXPECTED_ROWS
        assert tuple(m.axes["ep"]) == EXPECTED_EP
        assert tuple(m.axes["global_tokens"]) == EXPECTED_GT
        assert m.shape["n"] == n and m.shape["k"] == k
        assert m.capability.get("blackwell_only") is True
        assert m.reference["mode"] == "torch_tolerance"


def test_global_tokens_axis_is_ep_times_rows():
    # Every (ep, rows) combination derives a declared global_tokens value,
    # and the union covers the declared axis exactly.
    seen = set()
    for ep in EXPECTED_EP:
        for rows in EXPECTED_ROWS:
            seen.add(ep * rows)
    assert seen == set(EXPECTED_GT)


def test_tolerance_shape():
    tol = {"moe_ep_wo.tiles": 1e-6, "moe_ep_wo.w13_mma": 0.02,
           "moe_ep_wo.silu": 1e-6, "moe_ep_wo.w2_mma": 0.02}
    for unit, limit in tol.items():
        m = manifest.load_manifest(MANIFESTS / f"{unit}.toml")
        assert m.tolerance["rel_l2"] == limit
        # tolerances carry the machine/measured-value discipline note
        assert "UNMEASURED" in m.tolerance["note"] or "MEASURED" in m.tolerance["note"]


def test_registry_discovers_group():
    units = registry.discover(MANIFESTS)
    for unit, (symbol, _, _, _) in UNITS.items():
        assert unit in units, f"{unit} not discovered"
        assert units[unit].adapter.SYMBOL == units[unit].manifest.symbol == symbol


def test_adapters_expose_contract():
    for _, (_, adapter, _, _) in UNITS.items():
        mod = importlib.import_module(f"kernel_lab.units.{adapter}")
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(mod, fn, None)), f"{adapter} lacks {fn}()"


def test_phase2_does_not_touch_phase1_set():
    # The phase-1 registry assertion (EXPECTED_PHASE1_UNITS) only collects
    # phase == 1; these four must stay out of it.
    units = registry.discover(MANIFESTS)
    for unit in UNITS:
        assert units[unit].manifest.phase == 2


# --- production capacity derivations --------------------------------------------

# Production constants (pinned against the sources):
# state.max_tiles = glm52_moe_ep_wo_max_tiles(n_local, ep*8, 8)
#   (openinfer-glm52/src/moe_ep_wo.rs:137)
# bound_rows = min(expanded, gt*8 + 63*min(gt*8, n_local))  (same file:197)
# expanded = align_up(ep*128*min(8, n_local) + 63*n_local, 64)
#   (deepep_config_derived.cuh:57, kDecodeMaxTokens=128)
CAPACITY = {
    # ep: (n_local, max_global_tokens, state_max_tiles, decode_worst_expanded)
    4: (64, 32, 96, 8128),
    8: (32, 64, 96, 10240),
    16: (16, 128, 144, 17408),
}
BOUND_ROWS = {
    # (ep, rows): bound_rows
    (4, 1): 2048, (4, 2): 4096, (4, 4): 4160, (4, 8): 4288,
    (8, 1): 2080, (8, 2): 2144, (8, 4): 2272, (8, 8): 2528,
    (16, 1): 1136, (16, 2): 1264, (16, 4): 1520, (16, 8): 2032,
}


def test_capacity_table_matches_production():
    for ep, (groups, max_gt, tiles, expanded) in CAPACITY.items():
        assert moe.n_local(ep) == groups
        assert moe.max_global_tokens(ep) == max_gt
        assert moe.state_max_tiles(ep) == tiles
        assert moe.decode_worst_expanded(ep) == expanded
    for (ep, rows), bound in BOUND_ROWS.items():
        assert moe.bound_rows(ep, ep * rows) == bound, f"ep{ep} rows={rows}"


def test_bound_rows_covers_worst_aligned_end():
    # The tiles kernel traps if align_up(segment_end, 64) > m_capacity. The
    # production formula must cover the worst legal layout: all gt*TOPK rows
    # packed into the trailing experts with maximal alignment padding.
    for ep in EXPECTED_EP:
        groups = moe.n_local(ep)
        for rows in EXPECTED_ROWS:
            gt = ep * rows
            expanded_rows = gt * moe.TOPK
            active = min(expanded_rows, groups)
            worst_aligned_end = expanded_rows + (moe.ALIGN - 1) * active
            assert worst_aligned_end <= moe.bound_rows(ep, gt), (
                f"ep{ep} rows={rows}: {worst_aligned_end} > {moe.bound_rows(ep, gt)}"
            )


def test_shape_point_derives_launch_params():
    pt = moe.shape_point(8, 4)
    assert pt == {
        "ep": 8, "rows": 4, "groups": 32, "global_tokens": 32,
        "bound_rows": 2272, "max_tiles": 96, "expanded": 10240,
    }


# --- layout builder (smoke-test port) --------------------------------------------

def _counts(groups, pairs):
    vec = [0] * groups
    for e, c in pairs:
        vec[e] = c
    return vec


def test_build_layout_smoke_ep4():
    # Mirrors glm52_moe_ep_wo_chain_matches_host_reference_ep4_shape.
    lay = moe.build_layout(_counts(64, [(0, 1), (2, 8), (3, 9), (5, 3), (31, 5), (63, 2)]))
    assert lay.tiles == [
        (0, 0, 1), (64, 2, 8), (128, 3, 8), (136, 3, 1),
        (192, 5, 3), (256, 31, 5), (320, 63, 2),
    ]
    assert lay.psum[:6] == [1, 64, 72, 137, 192, 195]
    assert lay.psum[-1] == 322
    assert lay.aligned_end == 384


def test_build_layout_smoke_ep8():
    lay = moe.build_layout(_counts(32, [(0, 2), (1, 8), (4, 17), (13, 1), (22, 9), (31, 6)]))
    assert lay.tiles == [
        (0, 0, 2), (64, 1, 8), (128, 4, 8), (136, 4, 8), (144, 4, 1),
        (192, 13, 1), (256, 22, 8), (264, 22, 1), (320, 31, 6),
    ]
    assert lay.psum[:6] == [2, 72, 128, 128, 145, 192]
    assert lay.aligned_end == 384


def test_build_layout_smoke_ep64():
    # The local-experts < topk extreme; a full 64-row expert = 8 tiles.
    lay = moe.build_layout(_counts(4, [(0, 1), (1, 9), (2, 64), (3, 3)]))
    assert lay.tiles == [
        (0, 0, 1), (64, 1, 8), (72, 1, 1),
        (128, 2, 8), (136, 2, 8), (144, 2, 8), (152, 2, 8),
        (160, 2, 8), (168, 2, 8), (176, 2, 8), (184, 2, 8),
        (192, 3, 3),
    ]
    assert lay.psum == [1, 73, 192, 195]
    assert lay.aligned_end == 256


def test_tiles_never_straddle_segments():
    for ep in EXPECTED_EP:
        for rows in EXPECTED_ROWS:
            lay = moe.layout_for(ep, ep * rows, SEED)
            for base, e, live in lay.tiles:
                assert lay.starts[e] <= base < lay.starts[e] + lay.counts[e]
                assert base + live <= lay.starts[e] + lay.counts[e]
                assert (base - lay.starts[e]) % moe.TILE_ROWS == 0
                assert 1 <= live <= moe.TILE_ROWS


def test_expected_tiles_surface_encoding():
    lay = moe.build_layout(_counts(64, [(0, 1), (2, 8), (3, 9)]))
    surface = moe.expected_tiles_surface(lay, max_tiles=8)
    assert surface[:8] == [
        0, 0 | (1 << 16),
        64, 2 | (8 << 16),
        128, 3 | (8 << 16),
        136, 3 | (1 << 16),
    ]
    assert surface[8:] == [0] * (16 - 8)  # zero padding to 2*max_tiles
    # every value is exactly representable in f32 (the CLI casts to f32)
    assert all(abs(v) < 2**24 for v in surface)


# --- seeded routing distribution -------------------------------------------------

def test_routing_reproducible():
    for ep in EXPECTED_EP:
        gt = ep * 8
        assert moe.routing_counts(ep, gt, SEED) == moe.routing_counts(ep, gt, SEED)


def test_routing_pins_default_seed():
    # Pins the generator against accidental edits (values captured from the
    # first verified run; regenerate deliberately, never silently).
    expected = {
        # (ep, gt): (total rows, active experts, max count, tiles, aligned_end)
        (4, 4): (11, 11, 1, 11, 704),
        (4, 32): (54, 26, 7, 26, 1664),
        (8, 8): (9, 5, 4, 5, 320),
        (8, 64): (92, 19, 22, 23, 1216),
        (16, 16): (9, 7, 2, 7, 448),
        (16, 128): (68, 11, 17, 15, 704),
    }
    for (ep, gt), (total, active, mx, tiles, end) in expected.items():
        lay = moe.layout_for(ep, gt, SEED)
        assert sum(lay.counts) == total, f"ep{ep} gt={gt} total"
        assert sum(1 for c in lay.counts if c) == active, f"ep{ep} gt={gt} active"
        assert max(lay.counts) == mx, f"ep{ep} gt={gt} max"
        assert len(lay.tiles) == tiles, f"ep{ep} gt={gt} tiles"
        assert lay.aligned_end == end, f"ep{ep} gt={gt} aligned_end"


def test_routing_contract_invariants_all_points():
    for ep in EXPECTED_EP:
        groups = moe.n_local(ep)
        for rows in EXPECTED_ROWS:
            gt = ep * rows
            lay = moe.layout_for(ep, gt, SEED)
            # per-expert row cap (the tiles kernel's masked_cap)
            assert max(lay.counts, default=0) <= gt
            # capacity grid and buffer bounds (production values)
            assert len(lay.tiles) <= moe.state_max_tiles(ep)
            assert lay.aligned_end <= moe.bound_rows(ep, gt)
            # non-degenerate: some real work, more than one active expert
            assert sum(lay.counts) >= 2
            assert sum(1 for c in lay.counts if c) >= 2
            # not perfectly uniform: counts vary across active experts or
            # some experts are empty (both hold at every axis point)
            assert len(set(lay.counts)) > 1
            assert groups == len(lay.counts)


def test_routing_not_degenerate_extremes():
    # Forbidden shapes: all tokens on one expert; perfect uniformity.
    for ep in EXPECTED_EP:
        gt = ep * 8
        lay = moe.layout_for(ep, gt, SEED)
        assert max(lay.counts) < sum(lay.counts)  # not a single hot expert
        assert len(set(lay.counts)) > 1  # not perfectly uniform


def test_routing_edge_coverage_across_axis():
    # The seeded skew covers the smoke test's edge cases somewhere on the
    # axis: empty experts, 1-row experts, and multi-tile experts (>8 rows).
    saw_empty = saw_one = saw_multi = False
    for ep in EXPECTED_EP:
        for rows in EXPECTED_ROWS:
            lay = moe.layout_for(ep, ep * rows, SEED)
            saw_empty |= any(c == 0 for c in lay.counts)
            saw_one |= any(c == 1 for c in lay.counts)
            saw_multi |= any(c > moe.TILE_ROWS for c in lay.counts)
    assert saw_empty and saw_one and saw_multi


def test_route_weights_live_only():
    ep, gt = 8, 64
    bound = moe.bound_rows(ep, gt)
    lay = moe.layout_for(ep, gt, SEED)
    rw = moe.route_weights(lay, ep, gt, bound, SEED)
    assert len(rw) == bound
    mask = moe.live_row_mask(lay, bound)
    for v, live in zip(rw, mask):
        if live:
            assert moe.ROUTE_WEIGHT_LO <= v <= 1.0
        else:
            assert v == 0.0


def test_live_mask_and_gap_rows_partition():
    ep, gt = 4, 32
    bound = moe.bound_rows(ep, gt)
    lay = moe.layout_for(ep, gt, SEED)
    mask = moe.live_row_mask(lay, bound)
    gaps = moe.gap_rows(lay, bound)
    assert sum(mask) + len(gaps) == bound
    assert sum(mask) == sum(lay.counts)
    # gap rows include the alignment padding inside the surface
    assert lay.aligned_end - sum(lay.counts) <= len(gaps)
