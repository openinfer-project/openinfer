"""CPU-only checks for the attention group — MUST NOT import torch.

Units: mla.query_assemble / mla.cache_pack / flashmla_sparse.decode
(design contract docs/models/glm52/decode-op-bench-harness.md; production
call site openinfer-glm52/src/mla_decode.rs:412 `glm52_mla_attend_into`).
The torch-lazy-import design keeps this suite green without a GPU: manifest /
registry / adapter / refs imports are all stdlib-only.
"""
import sys
from pathlib import Path

import pytest

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest, registry  # noqa: E402
from kernel_lab.refs import mla_attention as mla  # noqa: E402

MANIFESTS = BENCHES / "manifests"
EXPECTED_ROWS = (1, 2, 4, 8, 16, 32, 64)
EXPECTED_CTX = (16384, 65536, 262144)
EXPECTED_SYMBOLS = {
    "mla.query_assemble": "glm52_mla_query_assemble_cuda",
    "mla.cache_pack": "glm52_mla_cache_pack_cuda",
    "flashmla_sparse.decode": "glm52_flashmla_sparse_decode_launch_cuda",
}
HELPER_SYMBOLS = (
    "glm52_flashmla_sparse_decode_num_sm_parts_cuda",
    "glm52_flashmla_sparse_decode_metadata_cuda",
)


def _unit(name):
    return manifest.load_manifest(MANIFESTS / f"{name}.toml")


def test_no_torch_leak():
    assert "torch" not in sys.modules, "attention refs/adapters must stay torch-free at import"


def test_group_units_discover():
    units = registry.discover(MANIFESTS)
    for name in mla.GROUP_UNITS:
        assert name in units, f"{name} not registered"
        u = units[name]
        assert u.manifest.phase == 1
        assert u.manifest.symbol == EXPECTED_SYMBOLS[name]
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(u.adapter, fn, None)), f"{name} adapter lacks {fn}()"
        assert getattr(u.adapter, "SYMBOL", None) == u.manifest.symbol


def test_axes_rows_and_ctx():
    # rows = decode buckets {1..8} + MTP span-mapped verify rows {16,32,64};
    # ctx = long-context tiers (short ctx is not benched, per the design doc).
    for name in mla.GROUP_UNITS:
        m = _unit(name)
        assert m.rows == EXPECTED_ROWS, f"{name} rows {m.rows}"
        assert tuple(m.axes.get("ctx", ())) == EXPECTED_CTX, f"{name} ctx {m.axes.get('ctx')}"
        assert m.capability.get("blackwell_only") is True


def test_cache_layout_constants():
    # The 656-byte fp8_ds_mla token contract (glm52_mla_assembly.cu).
    assert mla.CACHE_BYTES == 512 + 16 + 128 == 656
    assert mla.SCALE_OFFSET == 512 and mla.KPE_OFFSET == 528
    assert mla.SCALE_GROUPS * 4 == 16 and mla.ROPE_DIM * 2 == 128
    assert mla.PAGE_TOKENS == 64 and all(c % mla.PAGE_TOKENS == 0 for c in EXPECTED_CTX)
    assert mla.TOPK == 2048 and mla.TOPK % mla.TOPK_BLOCK == 0
    assert mla.SM_SCALE == 0.0625  # GLM52_SM_SCALE (config.rs)
    assert mla.DEFAULT_CTX == EXPECTED_CTX[0]


def test_ue8m0_round_up_bit_trick():
    # The exact production bit trick (bits + 0x007FFFFF) & 0x7F800000:
    # round UP to the next power of two; already-pow2 inputs are fixed points.
    assert mla.ue8m0_round_up(1.0) == 1.0
    assert mla.ue8m0_round_up(0.125) == 0.125
    assert mla.ue8m0_round_up(0.1) == 0.125
    assert mla.ue8m0_round_up(0.0029) == 0.00390625  # the smoke test's raw-scale regime
    assert mla.ue8m0_round_up(448.0 / 448.0) == 1.0
    for raw in (0.002, 0.0029, 0.003, 0.5, 3.3, 1e-6, 17.25):
        up = mla.ue8m0_round_up(raw)
        assert mla.is_ue8m0_pow2(up)
        assert up >= raw and up < 2.0 * raw + 1e-30  # rounds UP, less than 2x
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            mla.ue8m0_round_up(bad)


def test_is_ue8m0_pow2():
    assert mla.is_ue8m0_pow2(1.0) and mla.is_ue8m0_pow2(0.00390625)
    assert mla.is_ue8m0_pow2(2.0**-20) and mla.is_ue8m0_pow2(2.0**20)
    assert not mla.is_ue8m0_pow2(0.0029)
    assert not mla.is_ue8m0_pow2(3.0)
    assert not mla.is_ue8m0_pow2(0.0)
    assert not mla.is_ue8m0_pow2(-2.0)
    assert not mla.is_ue8m0_pow2(float("inf"))
    assert not mla.is_ue8m0_pow2(float("nan"))


def test_query_assemble_buffers():
    for rows in EXPECTED_ROWS:
        t = mla.query_assemble_buffers(rows)
        assert t["ql_nope_elems"] == rows * 64 * 512
        assert t["q_full_elems"] == rows * 64 * 256
        assert t["cos_sin_elems"] == rows * 32
        assert t["query_elems"] == rows * 64 * 576
        assert t["scratch"] == "none"
    # MTP verify capacity: span-8 x bucket-8 = 64 rows of full-width query.
    assert mla.query_assemble_buffers(64)["query_elems"] == 64 * 64 * 576 == 2359296


def test_cache_pack_buffers():
    for rows in EXPECTED_ROWS:
        for ctx in EXPECTED_CTX:
            t = mla.cache_pack_buffers(rows, ctx)
            assert t["max_slots"] == ctx
            assert t["ckv_fp8_bytes"] == rows * 512
            assert t["ckv_scales_bytes"] == rows * 16
            assert t["k_pe_elems"] == rows * 64
            assert t["slot_mapping_bytes"] == rows * 8
            assert t["cache_bytes"] == ctx * 656
            assert t["scratch"] == "none"
    with pytest.raises(ValueError):
        mla.cache_pack_buffers(1, 1000)  # ctx must be page-aligned


def test_decode_buffers():
    blocks = {16384: 256, 65536: 1024, 262144: 4096}
    for rows in EXPECTED_ROWS:
        for ctx, want_blocks in blocks.items():
            t = mla.decode_buffers(rows, ctx, num_sm_parts=132)
            assert t["num_blocks"] == want_blocks
            assert t["cache_bytes"] == ctx * 656 == want_blocks * 64 * 656
            assert t["q_elems"] == rows * 64 * 576
            assert t["topk_indices_elems"] == rows * 2048
            assert t["tile_scheduler_metadata_ints"] == 132 * 8
            assert t["num_splits_ints"] == rows + 1
            assert t["lse_elems"] == rows * 64
            assert t["lse_accum_elems"] == (rows + 132) * 64
            assert t["o_accum_elems"] == (rows + 132) * 64 * 512
            assert t["latent_elems"] == rows * 64 * 512
            assert "num_sm_parts" in t["scratch"]
    with pytest.raises(ValueError):
        mla.decode_buffers(1, 16384, num_sm_parts=0)
    with pytest.raises(ValueError):
        mla.decode_buffers(1, 16384, num_sm_parts=161)


def test_rows_within_batch_capacity():
    # The sm_100f decode kernel's only batch bound is the 128-row capacity
    # (runtime scheduler parameter, no per-batch template instantiation).
    assert max(EXPECTED_ROWS) <= mla.BATCH_CAPACITY
    assert max(EXPECTED_ROWS) <= 64  # MTP span-8 x bucket-8 verify rows


def test_iter_shape_points_injects_ctx():
    pts = list(mla.iter_shape_points({"n": 1, "k": 128}, [1, 8], EXPECTED_CTX))
    assert len(pts) == 2 * len(EXPECTED_CTX)
    assert pts[0] == {"rows": 1, "n": 1, "k": 128, "ctx": EXPECTED_CTX[0]}
    assert {p["ctx"] for p in pts} == set(EXPECTED_CTX)


def test_symbols_resolvable_when_so_present():
    so = loader.default_so_path()
    if not so.is_file():
        pytest.skip("libglm52_kernel_lab.so not built (run `kernel_lab build`)")
    lib = loader.load_library(so)
    for symbol in list(EXPECTED_SYMBOLS.values()) + list(HELPER_SYMBOLS):
        assert getattr(lib, symbol, None) is not None, f"{symbol} not exported by {so}"
