"""CPU-only checks for the proj-gemv group (o_proj + q_a|kv_a pair).

MUST NOT import torch (dev boxes may lack it): every assertion runs against
the manifests, the adapter modules' stdlib-only surface, and the .cu source
text. Covers: shape-axis derivation (incl. the pair's pack-width scale
identity), the pair's ABI-locked rows=[1] contract, adapter/symbol
consistency, tolerance discipline, and the mma_config entries that route
o_proj rows 16/32/64 to the multi-subtile mma (if those lines are deleted the
rows silently fail closed on GPU — this catches it without one).
"""
import re
import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import manifest, registry  # noqa: E402
from kernel_lab.refs import proj_gemv  # noqa: E402
from kernel_lab.units import mla_front_o_proj_gemv as o_proj_adapter  # noqa: E402
from kernel_lab.units import mla_front_qa_kva_pair_gemv as pair_adapter  # noqa: E402

MANIFESTS = BENCHES / "manifests"
REPO_ROOT = BENCHES.parents[1]  # benches/ -> openinfer-glm52/ -> repo root
MOE_GEMV_CU = REPO_ROOT / "openinfer-kernels" / "csrc" / "glm52" / "glm52_moe_gemv.cu"

EXPECTED_ROWS = (1, 2, 4, 8, 16, 32, 64)


def _load(unit: str):
    return manifest.load_manifest(MANIFESTS / f"{unit}.toml")


def test_no_torch_leak():
    assert "torch" not in sys.modules, "proj-gemv imports must stay torch-free"


def test_o_proj_manifest_parses_and_derives():
    m = _load("mla_front.o_proj_gemv")
    assert m.unit == "mla_front.o_proj_gemv" and m.phase == 1
    assert m.symbol == "glm52_fp8_weight_only_gemv_batched_cuda"
    assert m.adapter == "mla_front_o_proj_gemv"
    assert m.rows == EXPECTED_ROWS
    assert m.shape["n"] == 6144 and m.shape["k"] == 16384
    shapes = {v.rows: v for v in m.derive_shapes()}
    assert set(shapes) == set(EXPECTED_ROWS)
    for rows, v in shapes.items():
        assert v.act_elems == rows * 16384
        assert v.weight_bytes == 6144 * 16384
        # scale_len = ceil(6144/128) * ceil(16384/128) * 4 (moe_gemv.rs convention)
        assert v.scale_len_bytes == 48 * 128 * 4 == 24576
        assert v.out_elems == rows * 6144
        assert "ksplit" in v.scratch_rule
    # MTP span-8 x bucket-8 capacity over the full o_proj width.
    assert shapes[64].out_elems == 64 * 6144 == 393216


def test_pair_manifest_rows_locked_to_bs1():
    m = _load("mla_front.qa_kva_pair_gemv")
    assert m.unit == "mla_front.qa_kva_pair_gemv" and m.phase == 1
    assert m.symbol == "glm52_fp8_weight_only_gemv_pair_cuda"
    assert m.adapter == "mla_front_qa_kva_pair_gemv"
    # Production only launches the pair at bs=1 (and the ABI has no batch
    # parameter), so the axis must stay pinned to [1].
    assert m.rows == (1,)
    assert m.shape["n"] == 2624 and m.shape["k"] == 6144
    assert m.shape["n_a"] == 2048 and m.shape["n_b"] == 576
    (v,) = m.derive_shapes()
    assert v.rows == 1
    assert v.act_elems == 6144
    assert v.out_elems == 2624
    # Pack-width derivation must equal the per-side sums exactly.
    assert v.weight_bytes == 2624 * 6144 == (2048 + 576) * 6144
    assert v.scale_len_bytes == 21 * 48 * 4 == 4032


def test_pair_scale_identity():
    # The manifest derives scale bytes from the pack width n=2624; the Rust
    # side sizes each projection separately. The identity
    # ceil(2048/128) + ceil(576/128) == ceil(2624/128) is what makes the two
    # agree — guard it so a future shape edit can't silently break it.
    ceil = manifest._ceil_div
    assert ceil(2048, 128) + ceil(576, 128) == ceil(2624, 128) == 21
    assert ceil(6144, 128) == 48


def test_adapters_expose_contract_and_symbols():
    for unit_name, adapter in (
        ("mla_front.o_proj_gemv", o_proj_adapter),
        ("mla_front.qa_kva_pair_gemv", pair_adapter),
    ):
        m = _load(unit_name)
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(adapter, fn, None)), f"{unit_name} adapter lacks {fn}()"
        assert adapter.SYMBOL == m.symbol


def test_pair_adapter_abi_constants_match_cu_guard():
    # Mirrors the hard guard in glm52_fp8_weight_only_gemv_pair_cuda
    # (n_a == 2048, n_b == 576, k == 6144 -> else INVALID_VALUE).
    assert (pair_adapter.N_A, pair_adapter.N_B, pair_adapter.K) == (2048, 576, 6144)
    assert pair_adapter.N_A + pair_adapter.N_B == 2624
    # The padded kv_a weight generation must stay a 128 multiple and cover 576.
    assert pair_adapter.N_B_PADDED % 128 == 0
    assert pair_adapter.N_B_PADDED >= pair_adapter.N_B


def test_tolerance_discipline():
    for unit_name in ("mla_front.o_proj_gemv", "mla_front.qa_kva_pair_gemv"):
        m = _load(unit_name)
        assert m.reference["mode"] == "torch_tolerance"
        tol = m.tolerance
        assert isinstance(tol.get("rel_l2"), float) and 0 < tol["rel_l2"] <= 0.02
        # Derived constants go on the record as UNMEASURED until the first
        # GB300 measurement back-fills them (design doc tolerance discipline).
        assert "UNMEASURED" in tol.get("note", "")


def test_o_proj_mma_config_routes():
    # Text-level guard on csrc/glm52/glm52_moe_gemv.cu: o_proj must hold its
    # measured batch-8 entry and the Blackwell-only batch 16/32/64
    # multi-subtile placeholder, or the manifest's rows axis over-promises.
    src = MOE_GEMV_CU.read_text(encoding="utf-8")
    fn = src.split("MmaConfig mma_config(int batch, int n, int k)", 1)[1]
    fn = fn.split("return {0, 0};", 1)[0]
    batch8 = fn.split("if (batch == 8 && arch_is_blackwell())", 1)[1]
    assert re.search(r"n == 6144\s+&& k == 16384\) +return \{16, 2\}", batch8), \
        "o_proj batch-8 Blackwell mma entry {16,2} missing"
    multi = fn.split("batch == 16 || batch == 32 || batch == 64", 1)[1]
    assert re.search(r"n == 6144\s+&& k == 16384\) +return \{4, 1\}", multi), \
        "o_proj batch 16/32/64 multi-subtile placeholder {4,1} missing (UNMEASURED)"
    # The placeholder must reuse the already-instantiated (BTILES, 4, 1)
    # dispatch cases — a new (ksplit, ntiles) pair without a GLM52_MMA_MULTI_CASE
    # would fail closed at launch time.
    for bt in (2, 4, 8):
        assert f"GLM52_MMA_MULTI_CASE({bt}, 4, 1)" in src


def test_pair_cu_guard_matches_manifest():
    src = MOE_GEMV_CU.read_text(encoding="utf-8")
    pair_fn = src.split("CUresult glm52_fp8_weight_only_gemv_pair_cuda", 1)[1]
    guard = pair_fn.split("return CUDA_ERROR_INVALID_VALUE;", 1)[0]
    assert "n_a != 2048" in guard and "n_b != 576" in guard and "k != 6144" in guard


def test_units_discoverable():
    units = registry.discover(MANIFESTS)
    assert "mla_front.o_proj_gemv" in units
    assert "mla_front.qa_kva_pair_gemv" in units
