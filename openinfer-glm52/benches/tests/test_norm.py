"""Tests for the norm group units (norm.rms_norm / norm.q_a_layernorm /
norm.fused_add_rmsnorm_round).

CPU-safe: module level and all in-process checks stay torch-free. The GPU
functional gate (torch rel_l2 + the fused unit's production-unfused-chain
byte-compare) runs in a SUBPROCESS so torch never enters this pytest process
— keeps test_registry.py's test_no_torch_leak green on torch-equipped boxes
(this file sorts before test_registry.py).
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import loader, manifest, registry  # noqa: E402

MANIFESTS = BENCHES / "manifests"
EXPECTED_ROWS = (1, 2, 4, 8, 16, 32, 64)
RMS_EPS = 1e-5
# unit -> (production FFI symbol, hidden/q_lora dim) — mirrors
# openinfer-kernels/src/ffi/shared.rs and openinfer-glm52/src/config.rs.
NORM_UNITS = {
    "norm.fused_add_rmsnorm_round": ("fused_add_rms_norm_round_batched_cuda", 6144),
    "norm.rms_norm": ("rms_norm_batched_cuda", 6144),
    "norm.q_a_layernorm": ("rms_norm_batched_cuda", 2048),
}
ADAPTER_NAMES = {unit: unit.replace(".", "_") for unit in NORM_UNITS}


def _load(unit_name: str):
    return manifest.load_manifest(MANIFESTS / f"{unit_name}.toml")


def test_no_torch_leak():
    for mod in ("kernel_lab.refs.norm", *(f"kernel_lab.units.{a}" for a in ADAPTER_NAMES.values())):
        importlib.import_module(mod)
    assert "torch" not in sys.modules, "norm group imports must stay torch-free"


def test_manifests_parse_full_rows_axis():
    for unit, (symbol, dim) in NORM_UNITS.items():
        m = _load(unit)
        assert m.unit == m.path.stem == unit
        assert m.symbol == symbol
        assert m.adapter == ADAPTER_NAMES[unit]
        assert m.phase == 1
        assert m.capability.get("blackwell_only") is True
        # rows 全轴: decode buckets {1,2,4,8} + MTP span-mapped verify {16,32,64}.
        assert m.rows == EXPECTED_ROWS
        assert set(m.rows) <= set(manifest.DECODE_ROWS)
        # shape contract: n == k == the norm's feature dim; eps is the shared
        # checkpoint rms_norm_eps (config.rs GLM52_RMS_EPS).
        assert m.shape["n"] == m.shape["k"] == dim
        assert dim % manifest.FP8_BLOCK == 0
        assert m.shape["eps"] == RMS_EPS


def test_shape_derivation_table():
    # Buffer table per rows (mirrors the .cu indexing, no stride padding):
    #   x/hidden/residual/out: rows*dim bf16 elems each; weight: dim bf16.
    # derive_shapes() speaks the GEMV vocabulary — only act_elems/out_elems
    # carry norm semantics (act = x, out = out, both rows*dim); weight_bytes/
    # scale_len_bytes/scratch_rule are GEMV-only fields with no meaning here
    # (documented in each manifest's contract.notes).
    for unit, (_, dim) in NORM_UNITS.items():
        shapes = {v.rows: v for v in _load(unit).derive_shapes()}
        assert set(shapes) == set(EXPECTED_ROWS)
        for rows, v in shapes.items():
            assert v.act_elems == rows * dim   # x / hidden / residual
            assert v.out_elems == rows * dim   # out
        # rows are byte-scalable across the whole axis: one CTA per row with a
        # self-contained per-row reduction -> per-row bit-identity to rows=1.
        assert shapes[64].out_elems == 64 * dim


def test_adapters_expose_contract_and_symbol():
    for unit, (symbol, _) in NORM_UNITS.items():
        mod = importlib.import_module(f"kernel_lab.units.{ADAPTER_NAMES[unit]}")
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(mod, fn, None)), f"{unit} adapter lacks {fn}()"
        assert mod.SYMBOL == symbol


def test_registry_discovers_norm_units():
    units = registry.discover(MANIFESTS)
    for unit, (symbol, _) in NORM_UNITS.items():
        assert unit in units, f"{unit} not discovered"
        assert units[unit].manifest.symbol == symbol
        assert units[unit].adapter.SYMBOL == symbol


def test_fused_dual_reference_declared():
    # Design doc item 4: the fused unit needs BOTH the torch-tolerance net and
    # the production-unfused-chain byte-compare.
    m = _load("norm.fused_add_rmsnorm_round")
    assert m.reference["mode"] == "torch_tolerance"
    assert m.tolerance["rel_l2"] > 0
    ubc = m.reference.get("unfused_byte_compare")
    assert ubc is not None, "fused unit must declare the unfused byte-compare layer"
    mod = importlib.import_module("kernel_lab.units.norm_fused_add_rmsnorm_round")
    assert tuple(ubc["chain"]) == mod.UNFUSED_CHAIN_SYMBOLS == ("add_cuda", "rms_norm_batched_cuda")
    assert set(ubc["outputs"]) == {"hidden_sum", "out"}
    assert callable(getattr(mod, "unfused_byte_compare", None))


def test_eps_and_dims_match_model_constants():
    # Single source of truth: openinfer-glm52/src/config.rs
    # (GLM52_HIDDEN=6144, GLM52_Q_LORA_RANK=2048, GLM52_RMS_EPS=1e-5 — every
    # RMSNorm in the model shares the one checkpoint eps).
    from kernel_lab.refs import norm

    assert (norm.GLM52_HIDDEN, norm.GLM52_Q_LORA, norm.GLM52_RMS_EPS) == (6144, 2048, 1e-5)


def test_norm_symbols_resolvable_when_so_present():
    so = loader.default_so_path()
    if not so.is_file():
        pytest.skip("libglm52_kernel_lab.so not built (run `kernel_lab build`)")
    lib = loader.load_library(so)
    symbols = {symbol for symbol, _ in NORM_UNITS.values()}
    symbols.add("add_cuda")  # the fused unit's unfused byte-compare chain
    for symbol in sorted(symbols):
        assert getattr(lib, symbol, None) is not None, f"{symbol} not exported by {so}"


# --- GPU functional gate (subprocess-isolated; skips on CPU/torch-less boxes) ---

_GPU_GATE = r"""
import sys
import torch
from kernel_lab import loader, registry
from kernel_lab.refs import compute_metrics

major, _minor = torch.cuda.get_device_capability()
if major < 10:
    print("SKIP: Blackwell-only units (fail-closed), device capability major=%d" % major)
    sys.exit(0)

SEED = 0x5EED
ROWS_SUBSET = (1, 8, 64)  # decode bucket edges + MTP span-mapped verify max
units = registry.discover()
lib = loader.load_library()
stream = loader.current_stream_ptr()
failures = 0
for name in ("norm.rms_norm", "norm.q_a_layernorm", "norm.fused_add_rmsnorm_round"):
    u = units[name]
    for rows in ROWS_SUBSET:
        shape = {"rows": rows, "n": u.manifest.shape["n"], "k": u.manifest.shape["k"]}
        tensors = u.adapter.make_inputs(shape, SEED)
        u.adapter.run(lib, tensors, shape, stream)
        torch.cuda.synchronize()
        want = u.adapter.reference(tensors, shape)
        m = compute_metrics(tensors["out"], want)
        limit = u.manifest.tolerance["rel_l2"]
        ok = m["rel_l2"] <= limit
        failures += 0 if ok else 1
        print("[%s] %s rows=%d rel_l2=%.4e tol=%s" % ("PASS" if ok else "FAIL", name, rows, m["rel_l2"], limit))
        if name == "norm.fused_add_rmsnorm_round":
            # Second reference layer: production unfused chain from the same
            # .so must reproduce both fused outputs bit-for-bit.
            for out_name, res in u.adapter.unfused_byte_compare(lib, tensors, shape, stream).items():
                ok = res["equal"]
                failures += 0 if ok else 1
                print("[%s] %s rows=%d byte_compare[%s] mismatches=%s" % (
                    "PASS" if ok else "FAIL", name, rows, out_name, res["mismatches"]))
sys.exit(1 if failures else 0)
"""


def test_gpu_functional_gate():
    so = loader.default_so_path()
    if not so.is_file():
        pytest.skip("libglm52_kernel_lab.so not built (run `kernel_lab build`)")
    probe = subprocess.run(
        [sys.executable, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("torch with CUDA not available")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BENCHES) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _GPU_GATE],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
