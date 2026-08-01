"""CPU-only checks for the router unit group — MUST NOT import torch.

Covers what the generic registry suite cannot: router-specific shape math
(refs/router.py buffer_sizes), the rows>8 BLOCKED contract on
router.noaux_tc (min_gemv kMaxTokens=8 fail-closed) vs the full axis on
router.select, and the GLM5.2 checkpoint constants pinned in the manifests.
"""
import sys
from pathlib import Path

import pytest

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest, registry  # noqa: E402
from kernel_lab.refs import router as router_ref  # noqa: E402

MANIFESTS = BENCHES / "manifests"
NOAUX = MANIFESTS / "router.noaux_tc.toml"
SELECT = MANIFESTS / "router.select.toml"
FULL_ROWS = (1, 2, 4, 8, 16, 32, 64)


def test_router_group_torch_free():
    # Importing the refs module and both adapters (via registry.discover in a
    # later test) must stay torch-free on CPU dev boxes.
    assert "torch" not in sys.modules


def test_manifests_schema_and_constants():
    noaux = manifest.load_manifest(NOAUX)
    select = manifest.load_manifest(SELECT)
    assert noaux.unit == NOAUX.stem == "router.noaux_tc"
    assert select.unit == SELECT.stem == "router.select"
    for m in (noaux, select):
        assert m.phase == 1
        assert m.shape["n"] == router_ref.EXPERTS == 256
        assert m.shape["k"] == router_ref.HIDDEN == 6144
        assert m.shape["k"] % manifest.FP8_BLOCK == 0
        # The router is plain sm_90+ CUDA (PDL only) — no Blackwell gate.
        assert m.capability.get("blackwell_only") is False
        assert m.reference["mode"] == "torch_tolerance"
        assert "UNMEASURED" in m.tolerance["note"]
    assert noaux.symbol == "glm52_router_noaux_tc_cuda"
    assert select.symbol == "glm52_router_select_cuda"
    assert noaux.adapter == "router_noaux_tc"
    assert select.adapter == "router_select"


def test_noaux_rows_blocked_above_8():
    # glm52_min_gemv.cuh: launch_tokens switch instantiates 1..=kMaxTokens=8
    # (= GLM52_MAX_BATCH_PER_RANK) and fails closed INVALID_VALUE above it.
    noaux = manifest.load_manifest(NOAUX)
    assert noaux.rows == (1, 2, 4, 8)
    assert max(noaux.rows) <= router_ref.MIN_GEMV_MAX_TOKENS == 8
    notes = noaux.contract["notes"]
    assert "BLOCKED" in notes and "kMaxTokens" in notes
    # The standalone select has no token cap: full rows axis.
    select = manifest.load_manifest(SELECT)
    assert select.rows == FULL_ROWS
    assert set(select.rows) <= set(manifest.DECODE_ROWS)


def test_tolerance_values():
    # f64-reference reorder gate for the GEMV half; ulp-floor gate for select
    # (any pick flip is O(1e-2), four orders above the select tolerance).
    assert manifest.load_manifest(NOAUX).tolerance["rel_l2"] == 0.005
    assert manifest.load_manifest(SELECT).tolerance["rel_l2"] == 0.0001


def test_buffer_size_derivation():
    for rows in FULL_ROWS:
        b = router_ref.buffer_sizes(rows)
        assert b["hidden_elems"] == rows * 6144
        assert b["gate_elems"] == 256 * 6144 == 1_572_864
        assert b["bias_bytes"] == 256 * 4
        assert b["logits_elems"] == rows * 256
        assert b["topk_weight_elems"] == rows * 8
        assert b["topk_idx_elems"] == rows * 8
        # smem formula mirrors glm52_router.cu: threads*2*f32 + topk*f32.
        assert b["select_smem_bytes"] == 256 * 2 * 4 + 8 * 4 == 2080
        assert b["select_grid_blocks"] == rows
        assert b["gemv_grid_blocks"] == 256
        assert b["caller_scratch_bytes"] == 0
    with pytest.raises(ValueError):
        router_ref.buffer_sizes(0)


def test_registry_discovers_router_units():
    units = registry.discover(MANIFESTS)
    for name, symbol in (
        ("router.noaux_tc", "glm52_router_noaux_tc_cuda"),
        ("router.select", "glm52_router_select_cuda"),
    ):
        assert name in units, f"{name} not discovered"
        unit = units[name]
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(unit.adapter, fn, None)), f"{name} lacks {fn}()"
        assert unit.adapter.SYMBOL == unit.manifest.symbol == symbol
    assert "torch" not in sys.modules, "adapter import must stay torch-free"


def test_router_symbols_resolvable_when_so_present():
    so = loader.default_so_path()
    if not so.is_file():
        pytest.skip("libglm52_kernel_lab.so not built (run `kernel_lab build`)")
    lib = loader.load_library(so)
    for symbol in ("glm52_router_noaux_tc_cuda", "glm52_router_select_cuda"):
        assert getattr(lib, symbol, None) is not None, f"{symbol} not exported by {so}"
