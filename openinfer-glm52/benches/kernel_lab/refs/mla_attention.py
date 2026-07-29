"""Attention-group references + shared factories.

Units covered (production call site `glm52_mla_attend_into`,
openinfer-glm52/src/mla_decode.rs:412):

- `mla.query_assemble` — `glm52_mla_query_assemble_cuda`: per-head
  `query[T,64,576] = [ql_nope(512) | interleave-RoPE(q_pe)(64)]`; q_pe read in
  place from the q_b output at offset 192 / head stride 256.
- `mla.cache_pack` — `glm52_mla_cache_pack_cuda`: writes one 656-byte
  fp8_ds_mla paged token `[512 e4m3 ckv | 4 f32 UE8M0 scales | 64 bf16
  rope(k_pe)]` per row at `slot_mapping[t]` (page = 64 tokens).
- `flashmla_sparse.decode` — `glm52_flashmla_sparse_decode_launch_cuda`
  (sm_100f): FlashMLA sparse decode over the packed cache + DSA top-2048
  indices, latent out `[T,64,512]`.

UE8M0 contract (docs/lessons/flashmla-sm100-ue8m0-kv-scales.md): the sm100
kernel truncates stored f32 group scales to e8m0 with round-toward-zero, so
every scale in the 656B cache MUST be a power of two (rounded UP by the
writer). This module builds the pow2 assertion in: `assert_ue8m0_scales` runs
in every cache factory and in the cache_pack reference path.

References:
- query_assemble / cache_pack are bit-exact by construction: bf16 x bf16
  products are exact in f32 (8-bit significands), the add/sub is a single
  rounding whether or not nvcc contracts to FMA, and both sides round to bf16
  RNE — the torch f32 rope below reproduces `rope_block()` bit-for-bit.
- flashmla_sparse.decode: f64 naive sparse attention ported from
  `glm52_sparse_mla_reference_kernel` (csrc/glm52/glm52_sparse_mla.cu; the
  Rust gate openinfer-kernels/tests/glm52_sparse_mla.rs).

rows x ctx sweep: the shared CLI iterates only the rows axis
(`kernel_lab/__main__._shapes`); until it grows a --ctx selector, this module
provides the group sweep:

    PYTHONPATH=openinfer-glm52/benches \
        python3 -m kernel_lab.refs.mla_attention check flashmla_sparse.decode
    PYTHONPATH=openinfer-glm52/benches \
        python3 -m kernel_lab.refs.mla_attention bench mla.cache_pack --rows 8

Module level stays torch-free (CPU pytest / registry import this file).
"""
from __future__ import annotations

import argparse
import math
import struct
import sys

from kernel_lab import data
from kernel_lab.loader import require_torch
from kernel_lab.refs.fp8_gemv import e4m3_decode_torch

GROUP_UNITS = ("mla.query_assemble", "mla.cache_pack", "flashmla_sparse.decode")

# ---- model constants (openinfer-glm52/src/config.rs, csrc/glm52/glm52_mla_assembly.cu,
#      csrc/glm52/glm52_flashmla_sparse.cu) ----
HEADS = 64            # GLM52_HEADS (full EP8 width; FlashMLA h_q)
Q_HEAD = 256          # GLM52_QK_HEAD_DIM = qk_nope(192) + qk_rope(64) q_b row width
Q_PE_OFFSET = 192     # q_pe inside the q_b row (production in-place layout)
QK_NOPE = 512         # absorbed ql_nope width (== KV_LORA)
KV_LORA = 512         # GLM52_KV_LORA_RANK / ckv width / latent v width
ROPE_DIM = 64         # GLM52_QK_ROPE_HEAD_DIM
ROPE_HALF = 32        # cos/sin rows used by the interleave RoPE
QUERY_DIM = 576       # 512 nope | 64 rope (FlashMLA d_qk)
FP8_BLOCK = data.FP8_BLOCK  # 128
SCALE_GROUPS = KV_LORA // FP8_BLOCK  # 4
CACHE_BYTES = 656     # 512 e4m3 + 16 scale + 128 bf16 kpe
SCALE_OFFSET = 512
KPE_OFFSET = 528
PAGE_TOKENS = 64      # GLM52_FLASHMLA_SPARSE_PAGE_SIZE
TOPK = 2048           # GLM52_FLASHMLA_SPARSE_TOPK (DSA contract)
TOPK_BLOCK = 64
SM_SCALE = 0.0625     # GLM52_SM_SCALE
SCHED_META_INTS = 8   # sizeof(DecodingSchedMeta)/4
MAX_SM_PARTS = 160
BATCH_CAPACITY = 128  # kBatchCapacity in glm52_flashmla_sparse.cu
CTX_AXES = (16384, 65536, 262144)  # long-ctx tiers; short ctx not benched
DEFAULT_CTX = CTX_AXES[0]


# ---------------------------------------------------------------------------
# UE8M0 pow2 helpers (pure stdlib — CPU-testable)
# ---------------------------------------------------------------------------

def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def ue8m0_round_up(value: float) -> float:
    """Round a positive f32 scale UP to the next power of two — the exact bit
    trick of the production UE8M0 quant kernel and the Rust smoke test:
    `(bits + 0x007FFFFF) & 0x7F800000`. A scale already a power of two is
    returned unchanged."""
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"UE8M0 scale must be positive finite, got {value!r}")
    bits = _f32_bits(value)
    return _bits_f32((bits + 0x007F_FFFF) & 0x7F80_0000)


def is_ue8m0_pow2(value: float) -> bool:
    """True iff `value` is a positive finite f32 with a zero mantissa (2^e)."""
    return math.isfinite(value) and value > 0.0 and (_f32_bits(value) & 0x007F_FFFF) == 0


# ---------------------------------------------------------------------------
# Shape derivation tables (pure stdlib — the authoritative per-unit tables;
# the manifest [shape] n/k only keep `kernel_lab list`'s GEMV-flavored derived
# rows meaningful for act/out)
# ---------------------------------------------------------------------------

def query_assemble_buffers(rows: int) -> dict:
    """mla.query_assemble @ rows: element/byte counts (launch grid (64, rows)
    x 192 threads; no scratch)."""
    return {
        "rows": rows,
        "ql_nope_elems": rows * HEADS * QK_NOPE,        # bf16 [rows,64,512]
        "q_full_elems": rows * HEADS * Q_HEAD,          # bf16 [rows,64,256]
        "cos_sin_elems": rows * ROPE_HALF,              # bf16 [rows,32] each
        "query_elems": rows * HEADS * QUERY_DIM,        # bf16 out [rows,64,576]
        "scratch": "none",
        "launch": f"grid ({HEADS}, {rows}) x 192 threads; q_pe offset {Q_PE_OFFSET} stride {Q_HEAD}",
    }


def cache_pack_buffers(rows: int, ctx: int) -> dict:
    """mla.cache_pack @ (rows, ctx): ctx sizes the paged window max_slots
    (page = 64 tokens); the write cost is O(rows x 656B), window-independent."""
    if ctx % PAGE_TOKENS:
        raise ValueError(f"ctx {ctx} must be a multiple of page {PAGE_TOKENS}")
    return {
        "rows": rows,
        "ctx": ctx,
        "max_slots": ctx,
        "ckv_fp8_bytes": rows * KV_LORA,                # u8 [rows,512]
        "ckv_scales_bytes": rows * SCALE_GROUPS * 4,    # f32 [rows,4] UE8M0
        "k_pe_elems": rows * ROPE_DIM,                  # bf16 [rows,64]
        "cos_sin_elems": rows * ROPE_HALF,
        "slot_mapping_bytes": rows * 8,                 # i64 [rows] distinct in [0, ctx)
        "cache_bytes": ctx * CACHE_BYTES,               # u8 [ctx,656] out
        "scratch": "none",
        "launch": f"grid {rows} x 128 threads; trap on slot outside [0, {ctx})",
    }


def decode_buffers(rows: int, ctx: int, num_sm_parts: int) -> dict:
    """flashmla_sparse.decode @ (rows, ctx, num_sm_parts): full buffer table
    (harness pre-allocates the sm_parts-dependent scratch at MAX_SM_PARTS
    capacity; `num_sm_parts` here is the runtime-queried value)."""
    if ctx % PAGE_TOKENS:
        raise ValueError(f"ctx {ctx} must be a multiple of page {PAGE_TOKENS}")
    if not 1 <= num_sm_parts <= MAX_SM_PARTS:
        raise ValueError(f"num_sm_parts {num_sm_parts} out of 1..={MAX_SM_PARTS}")
    num_blocks = ctx // PAGE_TOKENS
    splits = rows + num_sm_parts
    return {
        "rows": rows,
        "ctx": ctx,
        "num_blocks": num_blocks,
        "num_sm_parts": num_sm_parts,
        "q_elems": rows * HEADS * QUERY_DIM,                    # bf16 [rows,64,576]
        "cache_bytes": num_blocks * PAGE_TOKENS * CACHE_BYTES,  # u8 paged
        "topk_indices_elems": rows * TOPK,                      # i32 [rows,2048]
        "tile_scheduler_metadata_ints": num_sm_parts * SCHED_META_INTS,
        "num_splits_ints": rows + 1,
        "lse_elems": rows * HEADS,                              # f32
        "lse_accum_elems": splits * HEADS,                      # f32
        "o_accum_elems": splits * HEADS * KV_LORA,              # f32
        "latent_elems": rows * HEADS * KV_LORA,                 # bf16 out [rows,64,512]
        "scratch": (
            "tile_meta[num_sm_parts*8] i32 + num_splits[rows+1] i32 + "
            "lse_accum[(rows+num_sm_parts)*64] f32 + o_accum[(rows+num_sm_parts)*64*512] f32; "
            "num_sm_parts from glm52_flashmla_sparse_decode_num_sm_parts_cuda "
            f"(capacity {MAX_SM_PARTS})"
        ),
        "launch": (
            f"metadata once (plan-time, like production); decode batch={rows} "
            f"num_blocks={num_blocks} topk={TOPK} sm_scale={SM_SCALE}"
        ),
    }


def iter_shape_points(manifest_shape: dict, rows_axes, ctx_axes):
    """rows x ctx grid as CLI-style shape dicts (adapters read `ctx` via
    shape.get("ctx", DEFAULT_CTX); the shared CLI never injects it)."""
    for r in rows_axes:
        for c in ctx_axes:
            yield {"rows": r, "n": manifest_shape["n"], "k": manifest_shape["k"], "ctx": c}


# ---------------------------------------------------------------------------
# Torch factories (lazy torch)
# ---------------------------------------------------------------------------

def rotary_table(rows: int, seed: int, device="cuda"):
    """(cos, sin) bf16 [rows, 32] on the unit circle — the first half of a
    position's rotary table. CPU-generated for cross-machine determinism."""
    torch = require_torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.rand((rows, ROPE_HALF), generator=gen, dtype=torch.float32)
    theta = theta * (2.0 * math.pi) - math.pi
    return theta.cos().to(torch.bfloat16).to(device), theta.sin().to(torch.bfloat16).to(device)


def rope_block_ref(x, cos, sin):
    """Interleave-in / block-out RoPE: out[r] reads the pair (x[2p], x[2p+1]),
    p = r % 32; r < 32 -> even*c - odd*s, r >= 32 -> odd*c + even*s. f32 math
    + bf16 RNE store — bit-identical to `rope_block()` in
    csrc/glm52/glm52_mla_assembly.cu (products of bf16 values are exact in
    f32, so FMA contraction cannot change the single-rounded result)."""
    torch = require_torch()
    r = torch.arange(ROPE_DIM, device=x.device)
    pair = r % ROPE_HALF
    upper = r >= ROPE_HALF
    even = x[..., 2 * pair].to(torch.float32)
    odd = x[..., 2 * pair + 1].to(torch.float32)
    c = cos[..., pair].to(torch.float32)
    s = sin[..., pair].to(torch.float32)
    v = torch.where(upper, odd * c + even * s, even * c - odd * s)
    return v.to(torch.bfloat16)


def _ue8m0_round_up_tensor(amax):
    """Vectorized `(bits + 0x007FFFFF) & 0x7F800000` on a positive f32 tensor."""
    torch = require_torch()
    bits = amax.to(torch.float32).view(torch.int32).to(torch.int64)
    rounded = (bits + 0x007F_FFFF) & 0x7F80_0000
    return rounded.to(torch.int32).view(torch.float32)


def assert_ue8m0_scales(scales, where: str) -> None:
    """Harness-built-in pow2 assertion: every scale written into / read from
    the 656-byte fp8_ds_mla cache must be a power of two (the Blackwell sm100
    kernel truncates stored scales to e8m0 — a non-pow2 scale is read up to
    2x too small; docs/lessons/flashmla-sm100-ue8m0-kv-scales.md)."""
    torch = require_torch()
    s = scales.detach().reshape(-1).to(torch.float32).cpu()
    bits = s.view(torch.int32).to(torch.int64) & 0xFFFF_FFFF
    ok = (s > 0) & torch.isfinite(s) & ((bits & 0x007F_FFFF) == 0)
    if not bool(ok.all()):
        bad = s[~ok][:8].tolist()
        raise AssertionError(
            f"{where}: non-UE8M0 power-of-two scale(s) {bad} — the fp8_ds_mla "
            "cache contract requires 2^ceil(log2(amax/448)) group scales"
        )


def _e4m3_encode(values):
    """Round-to-nearest e4m3 encode of a CPU f32 tensor (|v| <= 448) via the
    shared codebook — same recipe as data.normal_quantized_fp8."""
    torch = require_torch()
    vals, encodings = zip(*data.e4m3_codebook())
    table_v = torch.tensor(vals, dtype=torch.float32)
    table_b = torch.tensor(encodings, dtype=torch.uint8)
    midpoints = (table_v[:-1] + table_v[1:]) / 2.0
    idx = torch.searchsorted(midpoints, values.reshape(-1))
    return table_b[idx].view(values.shape)


def normal_quantized_ckv(tokens: int, seed: int, device="cuda"):
    """Per-token-group (4 x 128) e4m3 ckv + UE8M0 pow2-up scales — the
    fp8_ds_mla cache fill recipe: N(0,1) values, scale =
    2^ceil(log2(group_amax/448)), nearest-e4m3 codes (never a NaN pattern).
    CPU-generated, then moved to `device`. Returns (ckv_fp8 u8 [T,512],
    scales f32 [T,4]); scales pass assert_ue8m0_scales."""
    torch = require_torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((tokens, KV_LORA), generator=gen, dtype=torch.float32)
    g = x.view(tokens, SCALE_GROUPS, FP8_BLOCK)
    amax = g.abs().amax(dim=-1).clamp_min(1e-12)
    scales = _ue8m0_round_up_tensor(amax / data.E4M3_MAX)  # [T,4], pow2 by construction
    assert_ue8m0_scales(scales, "normal_quantized_ckv")
    q = g / scales.view(tokens, SCALE_GROUPS, 1)  # |q| <= 448 by construction
    return _e4m3_encode(q).to(device), scales.contiguous().to(device)


_PACKED_CACHE_MEMO: dict = {}


def packed_cache(num_tokens: int, seed: int, device="cuda"):
    """Full synthetic fp8_ds_mla paged cache: [T,656] tokens = normal-
    quantized e4m3 ckv + UE8M0 pow2 scales + N(0,1) bf16 rope-keys. The CPU
    master copy is memoized per (num_tokens, seed) — at ctx 262144 one fill
    is 172 MB, and a rows sweep reuses it across every rows value."""
    key = (num_tokens, seed)
    torch = require_torch()
    master = _PACKED_CACHE_MEMO.get(key)
    if master is None:
        fp8, scales = normal_quantized_ckv(num_tokens, data.derive_seed(seed, "cache-ckv"), device="cpu")
        kpe = data.normal_bf16((num_tokens, ROPE_DIM), seed=data.derive_seed(seed, "cache-kpe"), device="cpu")
        master = torch.zeros((num_tokens, CACHE_BYTES), dtype=torch.uint8)
        master[:, :SCALE_OFFSET] = fp8
        master[:, SCALE_OFFSET:KPE_OFFSET] = scales.view(torch.uint8)
        master[:, KPE_OFFSET:CACHE_BYTES] = kpe.view(torch.uint8)
        _PACKED_CACHE_MEMO[key] = master
    return master.reshape(-1).to(device)


# ---------------------------------------------------------------------------
# Torch references (lazy torch)
# ---------------------------------------------------------------------------

def query_assemble_ref(ql_nope, q_full, cos, sin):
    """query[R,64,576] = [ql_nope | rope(q_pe)], q_pe = q_full[..., 192:256]
    (the production in-place q_b layout). Bit-exact vs the kernel — the nope
    half is a copy, the rope half is `rope_block_ref`."""
    torch = require_torch()
    rows = ql_nope.shape[0]
    q_pe = q_full[..., Q_PE_OFFSET:Q_PE_OFFSET + ROPE_DIM]
    out = torch.empty((rows, HEADS, QUERY_DIM), dtype=torch.bfloat16, device=ql_nope.device)
    out[..., :QK_NOPE] = ql_nope
    out[..., QK_NOPE:] = rope_block_ref(q_pe, cos, sin)
    return out.to(torch.float32)


def cache_pack_ref(ckv_fp8, scales, k_pe, cos, sin, slot_mapping, num_slots: int):
    """Expected cache bytes after packing `rows` tokens: byte copy of the
    e4m3 ckv + the (pow2-asserted) f32 scales + `rope_block_ref(k_pe)`.
    Returns the u8 [num_slots*656] tensor — the gate is byte-exact (metrics
    compare in f32 after an exact cast)."""
    torch = require_torch()
    assert_ue8m0_scales(scales, "cache_pack_ref")
    rows = ckv_fp8.shape[0]
    expected = torch.zeros((num_slots, CACHE_BYTES), dtype=torch.uint8, device=ckv_fp8.device)
    slots = slot_mapping.to(torch.long)
    expected[slots, :SCALE_OFFSET] = ckv_fp8
    expected[slots, SCALE_OFFSET:KPE_OFFSET] = scales.contiguous().view(torch.uint8).view(rows, 16)
    expected[slots, KPE_OFFSET:CACHE_BYTES] = rope_block_ref(k_pe, cos, sin).view(torch.uint8)
    return expected.reshape(-1)


def sparse_attention_ref_f64(q, cache_u8, indices, sm_scale: float):
    """f64 naive sparse attention over the packed 656B cache — torch port of
    `glm52_sparse_mla_reference_kernel` (csrc/glm52/glm52_sparse_mla.cu):

        score[j] = sm_scale * (sum_f q[f] * e4m3(kv[f]) * scale[f>>7]
                               + sum_r q[512+r] * kpe[r])     (f64, -inf on idx<0)
        out[d]   = sum_j softmax(score)[j] * kv_j[d]          (f64; 0 when l == 0)

    q bf16 [R,64,576]; cache u8 [num_slots*656]; indices i32 [R, topk]
    (valid slots then -1 padding; an idx >= num_slots is a hard error, same
    as the kernel's __trap). Returns f32 [R,64,512]."""
    torch = require_torch()
    rows, _, _ = q.shape
    topk = indices.shape[1]
    tokens = cache_u8.view(-1, CACHE_BYTES)
    idx = indices.to(torch.long)
    gathered = tokens[idx.clamp_min(0)]                       # [R, topk, 656] u8
    fp8 = gathered[..., :SCALE_OFFSET].contiguous()
    scales = gathered[..., SCALE_OFFSET:KPE_OFFSET].contiguous().view(torch.float32)
    kpe = gathered[..., KPE_OFFSET:CACHE_BYTES].contiguous().view(torch.bfloat16)
    kv = e4m3_decode_torch(fp8).view(rows, topk, SCALE_GROUPS, FP8_BLOCK).to(torch.float64)
    kv = (kv * scales.to(torch.float64)[..., None]).view(rows, topk, KV_LORA)
    k576 = torch.cat([kv, kpe.to(torch.float64)], dim=-1)     # [R, topk, 576]
    scores = torch.einsum("rhd,rtd->rht", q.to(torch.float64), k576) * sm_scale
    invalid = indices < 0                                     # [R, topk]
    scores = scores.masked_fill(invalid[:, None, :], float("-inf"))
    m = scores.amax(dim=-1, keepdim=True)
    w = torch.exp(scores - m)
    w = torch.where(invalid[:, None, :], 0.0, w)              # all-invalid row -> w == 0 (nan-free)
    l = w.sum(dim=-1, keepdim=True)                           # [R, 64, 1]
    out = torch.einsum("rht,rtd->rhd", w, kv)
    out = torch.where(l > 0, out / l, torch.zeros_like(out))
    return out.to(torch.float32)


# ---------------------------------------------------------------------------
# rows x ctx sweep driver (group-local; the shared CLI iterates rows only)
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m kernel_lab.refs.mla_attention",
        description="attention-group rows x ctx sweep (check vs reference / bench)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "bench"):
        p = sub.add_parser(name)
        p.add_argument("unit", choices=GROUP_UNITS)
        p.add_argument("--rows", type=int, action="append", default=None)
        p.add_argument("--ctx", type=int, action="append", default=None)
        p.add_argument("--so", default=None)
        p.add_argument("--seed", type=int, default=0x5EED)
        if name == "bench":
            p.add_argument("--warmup", type=int, default=20)
            p.add_argument("--rounds", type=int, default=30)
            p.add_argument("--inner", type=int, default=10)
    args = parser.parse_args(argv)

    from kernel_lab import loader, registry, timing
    from kernel_lab.refs import compute_metrics

    units = registry.discover()
    if args.unit not in units:
        raise SystemExit(f"{args.unit}: not registered; available: {', '.join(units)}")
    u = units[args.unit]
    torch = loader.require_torch()
    if not torch.cuda.is_available():
        raise SystemExit("kernel_lab: no CUDA device visible")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if u.manifest.capability.get("blackwell_only") and major < 10:
        raise SystemExit(f"{u.name}: Blackwell-only unit (fail-closed); device capability major={major}")
    lib = loader.load_library(args.so)
    stream = loader.current_stream_ptr()

    rows_axes = args.rows or list(u.manifest.axes.get("rows", ()))
    ctx_axes = args.ctx or list(u.manifest.axes.get("ctx", (DEFAULT_CTX,)))
    ok = True
    for shape in iter_shape_points(u.manifest.shape, rows_axes, ctx_axes):
        tensors = u.adapter.make_inputs(shape, args.seed)
        if args.cmd == "check":
            u.adapter.run(lib, tensors, shape, stream)
            torch.cuda.synchronize()
            want = u.adapter.reference(tensors, shape)
            metrics = compute_metrics(tensors["out"], want)
            limit = u.manifest.tolerance.get("rel_l2")
            passed = limit is None or metrics["rel_l2"] <= limit
            ok &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {u.name} rows={shape['rows']} ctx={shape['ctx']} ({arch})")
            print(f"       rel_l2={metrics['rel_l2']:.4e} (tol {limit})  cosine={metrics['cosine']:.6f}  "
                  f"max_abs={metrics['max_abs']:.4e}  mean_abs={metrics['mean_abs']:.4e}")
        else:
            stats = timing.bench(
                lambda: u.adapter.run(lib, tensors, shape, stream),
                args.warmup, args.rounds, args.inner,
            )
            print(f"bench {u.name} rows={shape['rows']} ctx={shape['ctx']}: "
                  f"median={stats.median_us:.2f} us  p50={stats.p50_us:.2f}  p99={stats.p99_us:.2f}  "
                  f"mean={stats.mean_us:.2f}  rounds={len(stats.samples_us)} ({arch})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
