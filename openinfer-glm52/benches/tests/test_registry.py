"""CPU-only registry checks — MUST NOT import torch (dev boxes may lack it).

The torch-lazy-import design is what keeps this suite green without a GPU:
manifest/registry/adapter imports are all stdlib-only. The optional .so
symbol check runs only when target/release/libglm52_kernel_lab.so exists.
"""
import sys
from pathlib import Path

import pytest

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest, registry  # noqa: E402

MANIFESTS = BENCHES / "manifests"
EXPECTED_PHASE1_UNITS = {
    "mla_front.q_b_gemv",
    # norm group (one unit per line so parallel groups can append theirs).
    "norm.fused_add_rmsnorm_round",
    "norm.q_a_layernorm",
    "norm.rms_norm",
    # quant group.
    "quant.fp8_per_token_group_bf16",
    "quant.fp8_per_token_group_bf16_ue8m0",
    # indexer group.
    "indexer.weights_proj",
    "indexer.rope",
    "indexer.k_quant_cache",
    "indexer.mqa_logits",
    "indexer.topk_2048",
    "indexer.local_topk_to_slots",
    # attention group.
    "mla.query_assemble",
    "mla.cache_pack",
    "flashmla_sparse.decode",
    # bookends+shared group.
    "bookend.embed",
    "bookend.lm_head",
    "bookend.argmax_split",
    "shared_expert.swiglu",
    # proj-gemv + router groups (registered on their behalf — landed without
    # updating this set; verified complete and discovering cleanly).
    "mla_front.o_proj_gemv",
    "mla_front.qa_kva_pair_gemv",
    "router.noaux_tc",
    "router.select",
}
EXPECTED_ROWS = {1, 2, 4, 8, 16, 32, 64}


def test_no_torch_leak():
    assert "torch" not in sys.modules, "registry imports must stay torch-free"


def test_manifests_parse():
    files = sorted(MANIFESTS.glob("*.toml"))
    assert files, "no manifests found"
    for f in files:
        m = manifest.load_manifest(f)
        assert m.unit and m.symbol and m.adapter and m.phase >= 1


def test_unit_names_unique_and_match_filename():
    manifests = [manifest.load_manifest(f) for f in sorted(MANIFESTS.glob("*.toml"))]
    names = [m.unit for m in manifests]
    assert len(names) == len(set(names)), f"duplicate unit names: {names}"
    for m in manifests:
        assert m.unit == m.path.stem


def test_adapter_modules_expose_contract():
    units = registry.discover(MANIFESTS)
    for unit in units.values():
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(unit.adapter, fn, None)), f"{unit.name} adapter lacks {fn}()"
        # Adapter-declared symbol must match the manifest metadata.
        assert getattr(unit.adapter, "SYMBOL", None) == unit.manifest.symbol


def test_shape_derivation():
    m = manifest.load_manifest(MANIFESTS / "mla_front.q_b_gemv.toml")
    shapes = {v.rows: v for v in m.derive_shapes()}
    assert set(shapes) == EXPECTED_ROWS
    for rows, v in shapes.items():
        assert v.weight_bytes == 16384 * 2048
        # scale_len = ceil(n/128) * ceil(k/128) * 4 (moe_gemv.rs convention)
        assert v.scale_len_bytes == (16384 // 128) * (2048 // 128) * 4 == 8192
        assert v.act_elems == rows * 2048
        assert v.out_elems == rows * 16384
        assert "ksplit" in v.scratch_rule


def test_mtp_verify_rows_derivation():
    # rows 16/32/64 = MTP span-mapped verify rows (span-8 x bucket-8 = 64):
    # buffers scale linearly with rows, scale/weight are rows-independent, and
    # the scratch rule stays the runtime-queried ksplit * rows * n.
    m = manifest.load_manifest(MANIFESTS / "mla_front.q_b_gemv.toml")
    shapes = {v.rows: v for v in m.derive_shapes()}
    for rows in (16, 32, 64):
        v = shapes[rows]
        assert v.act_elems == rows * 2048
        assert v.out_elems == rows * 16384
        assert v.weight_bytes == 16384 * 2048
        assert v.scale_len_bytes == 8192
        assert "ksplit * rows * n" in v.scratch_rule
    # span-8 x bucket-8 capacity: 64 rows over the full q_b projection.
    assert shapes[64].out_elems == 64 * 16384 == 1048576


def test_rows_axis_within_kernel_whitelist():
    # Guards the EP4 bucket-label-drift class of accident: the manifest axis
    # must stay within the rows the .cu dispatch admits (decode buckets plus
    # the MTP multi-subtile mma rows).
    for m in (manifest.load_manifest(f) for f in MANIFESTS.glob("*.toml")):
        assert set(m.rows) <= set(manifest.DECODE_ROWS)


def test_phase1_unit_names_stable():
    units = registry.discover(MANIFESTS)
    phase1 = {name for name, u in units.items() if u.manifest.phase == 1}
    assert phase1 == EXPECTED_PHASE1_UNITS
    for name in phase1:
        assert manifest.UNIT_NAME_RE.match(name)


def test_demo_symbol_resolvable_when_so_present():
    so = loader.default_so_path()
    if not so.is_file():
        pytest.skip("libglm52_kernel_lab.so not built (run `kernel_lab build`)")
    lib = loader.load_library(so)
    for unit in registry.discover(MANIFESTS).values():
        assert getattr(lib, unit.manifest.symbol, None) is not None, (
            f"{unit.manifest.symbol} not exported by {so}"
        )
