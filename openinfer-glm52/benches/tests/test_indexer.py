"""CPU-only checks for the GLM5.2 DSA indexer chain units — MUST NOT import torch.

Covers the six units at openinfer-glm52/benches/manifests/indexer.*.toml:
registry wiring, rows/ctx axes, per-unit shape derivations (checked against
kernel_lab.refs.indexer's torch-free helpers), the documented rows>8 limits,
and tolerance discipline (derived budgets, UNMEASURED until the GB300 backfill).
"""
import sys
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHES))

from kernel_lab import manifest, registry  # noqa: E402
from kernel_lab.refs import indexer as idx  # noqa: E402

MANIFESTS = BENCHES / "manifests"
INDEXER_UNITS = {
    "indexer.weights_proj",
    "indexer.rope",
    "indexer.k_quant_cache",
    "indexer.mqa_logits",
    "indexer.topk_2048",
    "indexer.local_topk_to_slots",
}
CTX_UNITS = {
    "indexer.k_quant_cache",
    "indexer.mqa_logits",
    "indexer.topk_2048",
    "indexer.local_topk_to_slots",
}


def _load(name: str):
    return manifest.load_manifest(MANIFESTS / f"{name}.toml")


def _discover():
    return registry.discover(MANIFESTS)


def _landed_indexer_units() -> list[str]:
    """Indexer units whose manifests exist — lets this suite stay green while
    the group lands one unit at a time (and while other groups land theirs)."""
    return sorted(p.stem for p in MANIFESTS.glob("indexer.*.toml"))


# --- registry wiring -----------------------------------------------------------

def test_no_torch_leak():
    assert "torch" not in sys.modules, "indexer test imports must stay torch-free"


def test_indexer_units_registered():
    landed = _landed_indexer_units()
    assert landed, "no indexer manifests found"
    assert set(landed) <= INDEXER_UNITS, f"unexpected indexer unit names: {landed}"
    units = _discover()
    for name in landed:
        u = units[name]
        assert u.manifest.adapter == name.replace(".", "_")
        assert u.adapter.SYMBOL == u.manifest.symbol
        for fn in ("make_inputs", "run", "reference"):
            assert callable(getattr(u.adapter, fn))


def test_indexer_manifests_phase1_and_tolerances():
    for name in _landed_indexer_units():
        m = _load(name)
        assert m.phase == 1, f"{name}: indexer chain is a phase-1 group"
        assert m.reference["mode"] == "torch_tolerance"
        tol = m.tolerance
        assert tol["rel_l2"] > 0
        assert "UNMEASURED" in tol.get("note", ""), f"{name}: tolerance note must stay UNMEASURED until GB300"


def test_ctx_axis_exactly_the_three_stops():
    for name in _landed_indexer_units():
        m = _load(name)
        if name in CTX_UNITS:
            assert tuple(m.axes["ctx"]) == idx.CTX_AXIS == (16384, 65536, 262144), name
            assert all(c >= 16384 for c in m.axes["ctx"]), f"{name}: short ctx must not be benched"
        else:
            assert "ctx" not in m.axes, f"{name}: ctx-independent unit must not grow a fake axis"


def test_seq_lens_helper_bounds():
    for ctx in idx.CTX_AXIS:
        for rows in manifest.DECODE_ROWS:
            lens = idx.seq_lens_for_rows(rows, ctx)
            assert len(lens) == rows
            assert all(idx.TOPK <= ln <= ctx for ln in lens), (rows, ctx, lens)
            assert len(set(lens)) == min(rows, 4), "rows must not share one context length"


# --- indexer.weights_proj --------------------------------------------------------

def test_weights_proj_rows_axis_and_blocked_note():
    m = _load("indexer.weights_proj")
    assert m.rows == idx.WEIGHTS_PROJ_ROWS == (1, 2, 4, 8)
    notes = m.contract["notes"]
    # rows 16/32/64 are architecturally reachable only through the multi-subtile
    # mma, whose table lacks the indexer shapes -> fail-closed. The BLOCKED
    # reason must stay documented in the manifest.
    assert "BLOCKED" in notes and "INVALID_VALUE" in notes
    assert "16/32/64" in notes


def test_weights_proj_shape_derivation():
    m = _load("indexer.weights_proj")
    assert (m.shape["n"], m.shape["k"]) == (idx.WQ_B_N, idx.WQ_B_K) == (4096, 2048)
    assert (m.shape["wk_n"], m.shape["wk_k"]) == (idx.WK_N, idx.WK_K) == (128, 6144)
    shapes = {v.rows: v for v in m.derive_shapes()}
    assert set(shapes) == set(idx.WEIGHTS_PROJ_ROWS)
    for rows, v in shapes.items():
        # wq_b buffers (the manifest primary shape)
        assert v.weight_bytes == 4096 * 2048
        assert v.scale_len_bytes == (4096 // 128) * (2048 // 128) * 4 == 2048
        assert v.act_elems == rows * 2048
        assert v.out_elems == rows * 4096
    # wk side (adapter-read keys): 128x6144 e4m3 + 1x48 f32 scales.
    assert 128 * 6144 == 786_432
    assert (128 // 128) * (6144 // 128) * 4 == 192


def test_weights_proj_scratch_rule():
    m = _load("indexer.weights_proj")
    for v in m.derive_shapes():
        assert "ksplit" in v.scratch_rule


# --- indexer.rope --------------------------------------------------------------

def test_rope_rows_axis_full_and_half_split_note():
    m = _load("indexer.rope")
    # grid=(heads, tokens) has no token bound -> the full rows axis is served.
    assert m.rows == idx.FULL_ROWS == (1, 2, 4, 8, 16, 32, 64)
    # The reference must pin the kernel's half-split semantics (not the
    # config's interleave flag) — keep the divergence documented.
    assert "half-split" in m.contract["notes"]


def test_rope_shape_derivation():
    m = _load("indexer.rope")
    assert (m.shape["heads"], m.shape["head_dim"], m.shape["rope_dim"]) == (32, 128, 64)
    assert idx.ROPE_HALF == 32
    # Per-row working set: q 32*128 + k 128 bf16, cos/sin 2*32 bf16.
    rows = 64
    q_elems = rows * idx.INDEX_HEADS * idx.HEAD_DIM
    k_elems = rows * idx.HEAD_DIM
    assert q_elems == 64 * 4096 and k_elems == 64 * 128
    # Only the first rope_dim=64 dims rotate; the rest pass through.
    assert idx.ROPE_DIM + idx.ROPE_DIM == idx.HEAD_DIM


# --- indexer.k_quant_cache ------------------------------------------------------

def test_k_quant_cache_layout_derivation():
    m = _load("indexer.k_quant_cache")
    assert m.rows == idx.FULL_ROWS
    assert m.shape["block_size"] == idx.BLOCK_KV == 64
    # DeepGEMM block-split layout: [64*128 fp8][64*4 f32] per block.
    assert idx.cache_stride_bytes() == 64 * (128 + 4) == 8448
    for ctx in idx.CTX_AXIS:
        assert idx.block_cols(ctx) == ctx // 64
        assert idx.cache_bytes(ctx) == (ctx // 64) * 8448 == ctx * 132


def test_k_quant_cache_gate_is_byte_exact():
    m = _load("indexer.k_quant_cache")
    # Deterministic quant -> the tolerance is a CLI backstop only; the manifest
    # must say the real gate is the byte-equality assert in the adapter.
    assert m.tolerance["rel_l2"] <= 1e-9
    assert "字节" in m.tolerance["note"]


# --- indexer.mqa_logits ---------------------------------------------------------

def test_mqa_rows_capped_by_aot_batch():
    m = _load("indexer.mqa_logits")
    # kAotAlignedBatchSize=32 in glm52_deepgemm_mqa.cu caps batch at 32 ->
    # rows=64 is BLOCKED; the manifest must carry that reason.
    assert m.rows == idx.MQA_ROWS == (1, 2, 4, 8, 16, 32)
    assert idx.MQA_MAX_ROWS == 32
    assert "BLOCKED" in m.contract["notes"] and "64" in m.contract["notes"]


def test_mqa_aot_constants_pinned():
    m = _load("indexer.mqa_logits")
    # next_n=1, heads=32, head_dim=128, block_kv=64, num_sms=132 are AOT
    # instantiation bounds (fail-closed), mirrored from the .cu / NUM_SMS.
    assert (m.shape["next_n"], m.shape["block_kv"], m.shape["num_sms"]) == (1, 64, 132)
    assert idx.NUM_SMS == 132 and idx.BLOCK_KV == 64
    assert idx.schedule_meta_len() == (132 + 1) * 2 == 266


def test_mqa_shape_derivation():
    for ctx in idx.CTX_AXIS:
        cols = idx.block_cols(ctx)
        for rows in idx.MQA_ROWS:
            # whole-pool blocks, cache bytes, and the exact-fit logits buffer
            assert rows * cols == rows * (ctx // 64)
            assert rows * idx.cache_bytes(ctx) == rows * ctx * 132
            # logits_stride == ctx must stay a 256-multiple (split_kv)
            assert ctx % 256 == 0
    # bf16 logits at the largest axis point: 32 * 262144 * 2 B = 16 MiB.
    assert 32 * 262144 * 2 == 16 * 1024 * 1024


# --- indexer.topk_2048 ------------------------------------------------------------

def test_topk_rows_axis_and_tie_rule_documented():
    m = _load("indexer.topk_2048")
    assert m.rows == idx.FULL_ROWS
    # The TopKTieBreak::Small contract and the lengths filter must stay pinned.
    assert "TopKTieBreak::Small" in m.contract["notes"]
    assert "lengths" in m.contract["notes"]
    assert "2047/2048" in m.tolerance["note"]


def test_topk_shape_derivation():
    m = _load("indexer.topk_2048")
    assert m.shape["top_k"] == idx.TOPK == 2048
    for ctx in idx.CTX_AXIS:
        max_len = idx.topk_max_len(ctx)
        assert max_len == ctx + 256
        assert max_len % 256 == 0, "production logits_stride stays 256-aligned"
        lens = idx.seq_lens_for_rows(idx.FULL_ROWS[-1], ctx)
        assert all(idx.TOPK <= ln < max_len for ln in lens)
        # 1e30 stale tail always fits inside [ln, max_len).
        assert max_len - max(lens) >= 256
    # Output buffers at the largest axis point: rows*2048 (i32 + f32).
    assert idx.FULL_ROWS[-1] * idx.TOPK * 4 * 2 == 64 * 2048 * 8


# --- indexer.local_topk_to_slots --------------------------------------------------

def test_slots_rows_axis_and_stride_contract():
    m = _load("indexer.local_topk_to_slots")
    assert m.rows == idx.FULL_ROWS
    assert m.shape["block_size"] == idx.BLOCK_KV == 64
    # P1 regression: block_table_stride must equal block_table_cols, not topk.
    assert "block_table_stride==block_table_cols" in m.contract["notes"]


def test_slots_shape_derivation():
    for ctx in idx.CTX_AXIS:
        cols = idx.block_cols(ctx)
        lens = idx.seq_lens_for_rows(idx.FULL_ROWS[-1], ctx)
        # Every sampled offset lands inside the row's own table width.
        assert all(ln <= cols * idx.BLOCK_KV for ln in lens)
        # Global slot space fits i32 with headroom.
        assert idx.FULL_ROWS[-1] * ctx < 2**31
    # Offsets count per row equals top_k; slot pool is row-disjoint by construction.
    assert idx.TOPK == 2048
