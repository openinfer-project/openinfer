"""CPU-only checks for the quant unit group — MUST NOT import torch.

Covers the two per-token-group FP8 quant units (base amax/448 + ue8m0 twin):
manifest parsing/derivation, adapter contract, the packed comparison-surface
layout, and the torch-free scalar spec in refs/quant.py (RNE e4m3 encode with
ties-to-even, satfinite clamp, eps-clamped amax/448 scale, ue8m0 bit bump) —
the same semantics the GPU check gates bit-exactly against the kernel.
"""
import importlib
import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import data, manifest, registry  # noqa: E402
from kernel_lab.refs import quant as qref  # noqa: E402

MANIFESTS = BENCHES / "manifests"
BASE = "quant.fp8_per_token_group_bf16"
UE8M0 = "quant.fp8_per_token_group_bf16_ue8m0"
BASE_SYMBOL = "glm52_fp8_per_token_group_quant_bf16_cuda"
UE8M0_SYMBOL = "glm52_fp8_per_token_group_quant_bf16_ue8m0_cuda"
EXPECTED_ROWS = (1, 2, 4, 8, 16, 32, 64)


def test_no_torch_leak():
    assert "torch" not in sys.modules, "quant refs/adapters must stay torch-free at import"


# --- manifests -----------------------------------------------------------------


def test_quant_manifests_parse():
    expected = {
        BASE: (BASE_SYMBOL, "quant_fp8_per_token_group_bf16", 6144, 48),
        UE8M0: (UE8M0_SYMBOL, "quant_fp8_per_token_group_bf16_ue8m0", 512, 4),
    }
    for unit, (symbol, adapter, hidden, groups) in expected.items():
        m = manifest.load_manifest(MANIFESTS / f"{unit}.toml")
        assert m.unit == unit and m.phase == 1
        assert m.symbol == symbol and m.adapter == adapter
        assert m.rows == EXPECTED_ROWS
        # quant is width-preserving: n (out width) == k (in width) == hidden
        assert m.shape["k"] == hidden and m.shape["n"] == hidden
        assert m.shape["group_size"] == 128 and hidden // 128 == groups
        # arch-portable kernel (no sm100 instructions); arch discipline lives
        # in the arch-bucketed ledger, not the capability gate
        assert m.capability.get("blackwell_only") is False
        assert m.reference["mode"] == "torch_tolerance"
        # bit-exactness gate: any single byte/scale-bit diff is >= ~1e-4 rel_l2
        assert 0 < m.tolerance["rel_l2"] <= 1e-6
        assert "UNMEASURED" in m.tolerance["note"]


def test_shape_derivation_matches_quant_buffers():
    for unit, hidden, groups in ((BASE, 6144, 48), (UE8M0, 512, 4)):
        m = manifest.load_manifest(MANIFESTS / f"{unit}.toml")
        shapes = {v.rows: v for v in m.derive_shapes()}
        assert tuple(shapes) == EXPECTED_ROWS
        for rows, v in shapes.items():
            assert v.act_elems == rows * hidden  # bf16 in [rows, hidden]
            assert v.out_elems == rows * hidden  # e4m3 bytes out (width-preserving)
            # real scale buffer is rows * groups * 4 B; the packed comparison
            # surface appends exactly those raw scale bytes to the e4m3 bytes
            assert qref.packed_surface_len(rows, hidden) == v.out_elems + rows * groups * 4


def test_registry_discovers_quant_units():
    units = registry.discover(MANIFESTS)
    for unit, symbol in ((BASE, BASE_SYMBOL), (UE8M0, UE8M0_SYMBOL)):
        assert unit in units, f"{unit} not discovered"
        assert units[unit].adapter.SYMBOL == units[unit].manifest.symbol == symbol


def test_adapters_expose_contract():
    for adapter in ("quant_fp8_per_token_group_bf16", "quant_fp8_per_token_group_bf16_ue8m0"):
        mod = importlib.import_module(f"kernel_lab.units.{adapter}")
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(mod, fn, None)), f"{adapter} lacks {fn}()"
        assert mod.GROUP_SIZE == 128


# --- packed surface layout ------------------------------------------------------


def test_packed_surface_layout():
    for rows in EXPECTED_ROWS:
        for hidden in (512, 6144):
            value_bytes = rows * hidden
            # f32 view precondition: the scale region starts 4-byte aligned
            assert value_bytes % 4 == 0
            total = qref.packed_surface_len(rows, hidden)
            assert total == value_bytes + rows * (hidden // 128) * 4
    # capacity corners
    assert qref.packed_surface_len(64, 6144) == 64 * 6144 + 64 * 48 * 4 == 405504
    assert qref.packed_surface_len(64, 512) == 64 * 512 + 64 * 4 * 4 == 33792


# --- scalar spec: e4m3 RNE encode ----------------------------------------------


def test_e4m3_encode_roundtrip_exhaustive():
    # Every representable e4m3 value must encode to its own byte — pins the
    # whole grid (subnormals, normals, +-448) with no off-by-one anywhere.
    for value, byte in data.e4m3_codebook():
        assert qref.e4m3_encode_rne(value) == byte, hex(byte)


def test_e4m3_encode_ties_go_to_even():
    # Midpoint ties resolve to the even significand LSB (== even byte LSB) —
    # the same rule PTX cvt.rn and torch's float8_e4m3fn cast implement.
    assert qref.e4m3_encode_rne(1.0625) == 0x38  # mid(1.0, 1.125) -> down (even)
    assert qref.e4m3_encode_rne(1.1875) == 0x3A  # mid(1.125, 1.25) -> up (even)
    assert qref.e4m3_encode_rne(-1.0625) == 0xB8  # sign preserved on ties
    # subnormal ties, including round-to-zero at half the min subnormal
    assert qref.e4m3_encode_rne(2.0**-10) == 0x00  # mid(0, 2^-9) -> +0 (even)
    assert qref.e4m3_encode_rne(-(2.0**-10)) == 0x80  # -> -0
    assert qref.e4m3_encode_rne(3 * 2.0**-10) == 0x02  # mid(2^-9, 2^-8) -> up
    # subnormal/normal boundary tie: mid(0x07, 0x08) picks the normal (even)
    assert qref.e4m3_encode_rne(15 * 2.0**-10) == 0x08


def test_e4m3_encode_satfinite_and_signed_zero():
    assert qref.e4m3_encode_rne(448.0) == 0x7E
    assert qref.e4m3_encode_rne(500.0) == 0x7E  # clamp, never NaN (0x7F)
    assert qref.e4m3_encode_rne(-500.0) == 0xFE
    assert qref.e4m3_encode_rne(0.0) == 0x00
    assert qref.e4m3_encode_rne(-0.0) == 0x80  # cvt preserves the sign bit
    assert qref.e4m3_encode_rne(2.0**-9) == 0x01  # min subnormal


# --- scalar spec: amax/448 scale + ue8m0 bump -----------------------------------


def test_ue8m0_ceil_pow2_bits():
    f2b, b2f = qref.f32_to_bits, qref.bits_to_f32
    for pow2 in (1.0, 0.5, 2.0**-42, 2.0**-10, 2.0**40):
        assert b2f(qref.ue8m0_ceil_pow2_bits(f2b(pow2))) == pow2  # unchanged
    assert b2f(qref.ue8m0_ceil_pow2_bits(f2b(1.5))) == 2.0
    assert b2f(qref.ue8m0_ceil_pow2_bits(f2b(448.0))) == 512.0
    # one ulp above a power of two bumps to the next
    assert b2f(qref.ue8m0_ceil_pow2_bits(f2b(2.0**-42) + 1)) == 2.0**-41
    # f32 max -> +inf: kernel bit math verbatim, unreachable from bf16 inputs
    # (amax/448 <= ~7.6e35); pinned here as spec, not as a reachable case
    assert b2f(qref.ue8m0_ceil_pow2_bits(0x7F7FFFFF)) == float("inf")


def test_group_scale_eps_clamp():
    # amax = 0 -> scale = f32(1e-10)/448 (kPerTokenGroupQuantEps branch); the
    # scalar spec emulates the f32 division via f64 -> f32 rounding, so compare
    # against the same emulation — the bit-exact gate runs on GPU.
    eps = qref.bits_to_f32(qref.f32_to_bits(1e-10))
    expected = qref.bits_to_f32(qref.f32_to_bits(eps / 448.0))
    assert qref.group_scale_f32(0.0, ue8m0=False) == expected
    # eps/448 ~= 2.23e-13 sits between 2^-43 and 2^-42 -> ue8m0 bumps to 2^-42
    # (robust to a +-1 ulp emulation difference: the bump only sees the binade)
    assert qref.group_scale_f32(0.0, ue8m0=True) == 2.0**-42


def test_group_scale_exact_quotients():
    # amax values with exactly-representable f32 quotients (no double-rounding
    # caveat): 448 -> 1.0; 896 -> 2.0 (already a power of two, bump is a no-op);
    # 672 -> 1.5 -> ue8m0 -> 2.0.
    assert qref.group_scale_f32(448.0, ue8m0=False) == 1.0
    assert qref.group_scale_f32(896.0, ue8m0=True) == 2.0
    assert qref.group_scale_f32(672.0, ue8m0=False) == 1.5
    assert qref.group_scale_f32(672.0, ue8m0=True) == 2.0


def test_scalar_pipeline_exact_end_to_end():
    # Full per-group math in stdlib with exact quotients only (amax = 448 ->
    # scale 1.0): q = value, and midpoint ties are exercised through the same
    # clamp+encode order as the kernel (fminf/fmaxf, then RNE).
    group = [448.0, -448.0, 1.0625, -1.0625, 2.0**-10, -(2.0**-10), 0.0, -0.0]
    group += [1.0] * (128 - len(group))
    amax = max(abs(v) for v in group)
    scale = qref.group_scale_f32(amax, ue8m0=False)
    assert scale == 1.0
    enc = [qref.e4m3_encode_rne(min(max(v / scale, -448.0), 448.0)) for v in group]
    assert enc[:8] == [0x7E, 0xFE, 0x38, 0xB8, 0x00, 0x80, 0x00, 0x80]


def test_scalar_pipeline_ue8m0_end_to_end():
    # amax = 672 -> scale 1.5 -> ue8m0 -> 2.0; divisions by 2.0 are exact, and
    # 336/2 = 168 lands exactly on the e4m3 midpoint 1.3125*2^7 -> tie to the
    # even code 0x72 (1.25*2^7); 672/2 = 336 ties to 0x7A (1.25*2^8).
    group = [672.0, 336.0] + [0.0] * 126
    amax = max(abs(v) for v in group)
    scale = qref.group_scale_f32(amax, ue8m0=True)
    assert scale == 2.0
    # scale is a positive finite power of two (the property the check asserts)
    bits = qref.f32_to_bits(scale)
    assert 0 < bits < 0x7F80_0000 and bits & 0x007F_FFFF == 0
    enc = [qref.e4m3_encode_rne(min(max(v / scale, -448.0), 448.0)) for v in group]
    assert enc[:2] == [0x7A, 0x72]
