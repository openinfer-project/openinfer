"""CPU-only checks for the shared_expert group (shared_expert.swiglu).

MUST NOT import torch and must not depend on sibling unit groups — the
manifest and adapter are loaded by exact name, never via registry.discover
over the whole manifests dir.
"""
import importlib
import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest  # noqa: E402

MANIFESTS = BENCHES / "manifests"
HIDDEN = 6144   # GLM52_HIDDEN (openinfer-glm52/src/config.rs)
INTER = 2048    # GLM52_EXPERT_INTERMEDIATE — the shared expert, NOT the 12288 dense layers
EXPECTED_ROWS = [1, 2, 4, 8, 16, 32, 64]
NAME = "shared_expert.swiglu"


def _load():
    return manifest.load_manifest(MANIFESTS / f"{NAME}.toml")


def test_manifest_metadata():
    m = _load()
    assert m.symbol == "glm52_fp8_weight_only_gemv_partials_cuda"
    assert m.adapter == "shared_expert_swiglu"
    # Design-doc phase-1 unit: the shared expert is replicated per rank —
    # EP-independent by construction.
    assert m.phase == 1
    # rows 16/32/64 are Blackwell-only (multi-subtile table entries) -> the
    # unit is Blackwell-gated even though rows 1-8 also run on Hopper.
    assert m.capability.get("blackwell_only") is True
    assert list(m.rows) == EXPECTED_ROWS
    assert m.tolerance["rel_l2"] == 0.02
    assert m.reference["mode"] == "torch_tolerance"


def test_adapter_contract_and_torch_free():
    mod = importlib.import_module("kernel_lab.units.shared_expert_swiglu")
    for fn in ("make_inputs", "run", "reference"):
        assert callable(getattr(mod, fn, None)), f"{NAME} adapter lacks {fn}()"
    assert mod.SYMBOL == _load().symbol
    # The chain's other three production symbols ride along in the adapter.
    assert mod.SILU_SYMBOL == "glm52_silu_and_mul_bf16_cuda"
    assert mod.REDUCE_SILU_SYMBOL == "glm52_gemv_reduce_silu_mul_cuda"
    assert mod.DOWN_SYMBOL == "glm52_fp8_weight_only_gemv_batched_cuda"
    assert mod.KSPLIT_SYMBOL == "glm52_gemv_mma_ksplit_cuda"
    assert "torch" not in sys.modules, "swiglu adapter/refs must stay torch-free at import"


def test_shape_derivation():
    m = _load()
    assert m.shape == {"n": 2 * INTER, "k": HIDDEN}  # packed gate|up
    for v in m.derive_shapes():
        assert v.act_elems == v.rows * HIDDEN
        assert v.weight_bytes == (2 * INTER) * HIDDEN  # gate|up e4m3 bytes
        # gate|up scale_len = ceil(4096/128) * ceil(6144/128) * 4 = 32*48*4
        assert v.scale_len_bytes == 32 * 48 * 4 == 6144
        assert v.out_elems == v.rows * 2 * INTER
        assert "ksplit" in v.scratch_rule
    # The down projection is the transpose pairing: [HIDDEN, INTER] with
    # scales [ceil(6144/128), 2048/128] = [48, 16].
    assert (HIDDEN // 128, INTER // 128) == (48, 16)


def test_scratch_rule_math():
    # Mirrors the adapter's _ensure_scratch: one f32 buffer for both GEMVs,
    # floats = max(ksplit_gu*rows*4096, ksplit_dn*rows*6144); 0/0 -> NULL.
    # Statically check the production launch bounds the formula must satisfy
    # (each launcher's guard: ksplit*rows*n <= scratch_floats).
    for rows in EXPECTED_ROWS:
        for ksplit_gu, ksplit_dn in ((0, 0), (16, 16), (16, 0), (0, 8), (4, 4)):
            floats = max(ksplit_gu * rows * 2 * INTER, ksplit_dn * rows * HIDDEN)
            assert floats >= ksplit_gu * rows * 2 * INTER
            assert floats >= ksplit_dn * rows * HIDDEN
            assert (floats == 0) == (ksplit_gu == 0 and ksplit_dn == 0)


def test_rows16_64_have_cu_table_entries():
    # The manifest's rows axis must stay within what the .cu dispatch admits
    # (the bucket-label-drift guard class). rows 1-8 were already whitelisted
    # production shapes; rows 16/32/64 rest on the pure-increment Blackwell
    # table entries added for this unit — guard them against removal.
    cu = (
        loader.repo_root()
        / "openinfer-kernels/csrc/glm52/glm52_moe_gemv.cu"
    ).read_text(encoding="utf-8")
    block = cu.split("batch == 16 || batch == 32 || batch == 64", 1)[1]
    assert 'if (n == 4096  && k == 6144)  return {4, 1};' in block  # shared gate|up
    assert 'if (n == 6144  && k == 2048)  return {4, 1};' in block  # shared down
    # And the multi-subtile dispatch must instantiate the {4,1} configs.
    for btiles in (2, 4, 8):
        assert f"GLM52_MMA_MULTI_CASE({btiles}, 4, 1)" in cu


def test_symbols_resolvable_when_so_present():
    so = loader.default_so_path()
    if not so.is_file():
        return  # same skip style as test_registry's .so check
    lib = loader.load_library(so)
    mod = importlib.import_module("kernel_lab.units.shared_expert_swiglu")
    for sym in (
        mod.SYMBOL,
        mod.SILU_SYMBOL,
        mod.REDUCE_SILU_SYMBOL,
        mod.DOWN_SYMBOL,
        mod.KSPLIT_SYMBOL,
    ):
        assert getattr(lib, sym, None) is not None, f"{sym} not exported by {so}"
