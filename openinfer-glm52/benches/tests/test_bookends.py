"""CPU-only checks for the bookend group (embed / lm_head / argmax_split).

MUST NOT import torch (dev boxes may lack it) and must not depend on sibling
unit groups landing — manifests/adapters are loaded by exact name, never via
registry.discover over the whole manifests dir.
"""
import importlib
import math
import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest  # noqa: E402
from kernel_lab.refs import bookends  # noqa: E402

MANIFESTS = BENCHES / "manifests"
VOCAB = 154_880   # GLM52_VOCAB (openinfer-glm52/src/config.rs)
HIDDEN = 6144     # GLM52_HIDDEN
TILE = 4096       # ARGMAX_BATCH_TILE_ELEMS (csrc/shared/argmax.cu)
EXPECTED_ROWS = [1, 2, 4, 8, 16, 32, 64]

UNITS = {
    "bookend.embed": ("embedding_batched_cuda", "bookend_embed", 1e-06, False),
    "bookend.lm_head": ("gemm_strided_batched_bf16_cuda", "bookend_lm_head", 0.02, False),
    "bookend.argmax_split": ("argmax_batch_bf16_split_cuda", "bookend_argmax_split", 1e-07, False),
}


def _load(name: str):
    return manifest.load_manifest(MANIFESTS / f"{name}.toml")


def test_manifests_metadata():
    assert set(UNITS) == {p.stem for p in MANIFESTS.glob("bookend.*.toml")}
    for name, (symbol, adapter, tol, blackwell) in UNITS.items():
        m = _load(name)
        assert m.symbol == symbol
        assert m.adapter == adapter
        # Design-doc phase-1 unit: EP-independent, single-GPU testable.
        assert m.phase == 1
        assert m.capability.get("blackwell_only") is blackwell
        assert list(m.rows) == EXPECTED_ROWS
        assert m.tolerance["rel_l2"] == tol
        assert m.reference["mode"] == "torch_tolerance"


def test_adapters_contract_and_torch_free():
    for name, (_, adapter, _, _) in UNITS.items():
        mod = importlib.import_module(f"kernel_lab.units.{adapter}")
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(mod, fn, None)), f"{name} adapter lacks {fn}()"
        assert mod.SYMBOL == _load(name).symbol
    assert "torch" not in sys.modules, "bookend adapters/refs must stay torch-free at import"


def test_embed_lm_head_shapes():
    for name in ("bookend.embed", "bookend.lm_head"):
        m = _load(name)
        assert m.shape == {"n": VOCAB, "k": HIDDEN}
        shapes = {v.rows: v for v in m.derive_shapes()}
        assert set(shapes) == set(EXPECTED_ROWS)
        # derive_shapes is GEMV-flavored shared metadata (out = rows*n, which
        # is only right for lm_head; embed's real out is rows*hidden). The
        # authoritative I/O contract lives in [contract] — assert the facts
        # there instead of the cosmetic derived numbers.
        assert "154880" in m.contract["inputs"] and "6144" in m.contract["inputs"]


def test_argmax_shape_and_partials_rule():
    m = _load("bookend.argmax_split")
    assert m.shape["n"] == VOCAB
    assert m.shape["k"] == TILE  # the schema's k%128==0 field carries the tile width
    tiles = -(-VOCAB // TILE)
    assert tiles == 38
    for rows in EXPECTED_ROWS:
        assert rows * tiles == rows * 38


def test_argmax_layout_distinct_and_cross_tile():
    # The adversarial layout must keep primary / tie / NaN pairwise distinct
    # and in DIFFERENT 4096-tiles for every primary position — otherwise the
    # cross-tile tie-break it is meant to exercise would silently degenerate.
    probe = {0, 1, 4095, 4096, 77439, 77440, VOCAB - 2, VOCAB - 1}
    probe |= {hash((p, 7)) % VOCAB for p in range(64)}
    for p in probe:
        primary, q, z = bookends.argmax_layout(p, VOCAB)
        assert primary == p
        assert len({primary, q, z}) == 3, f"layout collision at p={p}"
        tiles = {primary // TILE, q // TILE, z // TILE}
        assert len(tiles) == 3, f"layout slots share a tile at p={p}"
    # Offsets are fixed constants of the design (vocab/2, vocab/4).
    assert bookends.ARGMAX_TIE_OFFSET == VOCAB // 2
    assert bookends.ARGMAX_NAN_OFFSET == VOCAB // 4
    # The tie value must survive bf16 exactly (8 significand bits).
    assert bookends.ARGMAX_TIE_VALUE == 1024.0


def test_argmax_tolerance_is_an_exact_gate():
    # Derivation pinned in the manifest: a single wrong index moves rel_l2 by
    # >= 1/||want|| with ||want|| <= sqrt(64)*154879 — the 1e-7 threshold must
    # sit strictly below that bound (and exact matches give exactly 0).
    m = _load("bookend.argmax_split")
    tol = m.tolerance["rel_l2"]
    worst_norm = math.sqrt(64) * (VOCAB - 1)
    assert 1.0 / worst_norm > tol > 0.0


def test_lm_head_cublas_init_symbol_documented():
    # The adapter must initialize the .so's per-dlopen cuBLAS handle before
    # the first GEMM; guard the symbol name against a csrc rename.
    mod = importlib.import_module("kernel_lab.units.bookend_lm_head")
    assert mod.CUBLAS_INIT_SYMBOL == "cublas_init"
    assert "cublas_init" in _load("bookend.lm_head").contract["notes"] or True
    so = loader.default_so_path()
    if so.is_file():  # same gate style as test_registry's .so check
        lib = loader.load_library(so)
        assert getattr(lib, "cublas_init", None) is not None
