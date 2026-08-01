"""Torch references + shape derivations for the GLM5.2 DSA indexer chain units.

Production chain (openinfer-glm52/src/indexer.rs glm52_indexer_forward_into):
fp8 projections (wq_b / wk) -> k LayerNorm -> half-split RoPE -> q group-quant
-> weights fold -> k quant+cache write -> DeepGEMM paged MQA logits -> bf16->f32
-> FlashInfer top-k 2048 -> local offsets to global slots. This module covers
the six harness units drawn on those kernel boundaries (the norms/quant/fold
glue is not a unit).

rows x ctx sweep: the shared CLI iterates only the rows axis
(`kernel_lab/__main__._shapes`); until it grows a --ctx selector, this module
provides the group sweep (same pattern as refs/mla_attention.py):

    PYTHONPATH=openinfer-glm52/benches \
        python3 -m kernel_lab.refs.indexer check indexer.mqa_logits --rows 8 --ctx 65536
    PYTHONPATH=openinfer-glm52/benches \
        python3 -m kernel_lab.refs.indexer bench indexer.topk_2048 --rows 8

Module level is stdlib-only (CPU-test safe); every reference imports torch
lazily inside the function. Shape helpers stay torch-free so
tests/test_indexer.py can validate derivations without a GPU.
"""
from __future__ import annotations

import argparse
import sys

from kernel_lab.data import derive_seed

GROUP_UNITS = (
    "indexer.weights_proj",
    "indexer.rope",
    "indexer.k_quant_cache",
    "indexer.mqa_logits",
    "indexer.topk_2048",
    "indexer.local_topk_to_slots",
)

# --- model constants (openinfer-glm52/src/config.rs) -------------------------
INDEX_HEADS = 32          # GLM52_INDEX_HEADS
HEAD_DIM = 128            # GLM52_INDEX_HEAD_DIM
ROPE_DIM = 64             # GLM52_QK_ROPE_HEAD_DIM (first 64 dims rotated)
ROPE_HALF = 32            # cos/sin row width (GLM52_ROPE_HALF)
TOPK = 2048               # GLM52_INDEX_TOPK
HIDDEN = 6144             # GLM52_HIDDEN
Q_LORA = 2048             # GLM52_Q_LORA_RANK

# --- cache / kernel layout constants -----------------------------------------
BLOCK_KV = 64             # INDEX_CACHE_BLOCK (model/mod.rs) == DeepGEMM block_kv
SCALE_BYTES_PER_TOKEN = 4
FP8_SCALE_DIVISOR = 448.0
FP8_SCALE_EPS = 1.0e-4
NUM_SMS = 132             # model/mod.rs NUM_SMS — pinned by the DeepGEMM AOT gate
MQA_MAX_ROWS = 32         # kAotAlignedBatchSize in glm52_deepgemm_mqa.cu
TOPK_MAX_LEN_SLACK = 256  # harness: max_len = ctx + slack (stale/garbage tail)

# rows the fp8 projection GEMV serves at the indexer shapes: batches 16/32/64
# run only the multi-subtile mma, whose table lists q_b (16384, 2048) alone —
# (4096, 2048) wq_b and (128, 6144) wk fail closed with INVALID_VALUE there.
WEIGHTS_PROJ_ROWS = (1, 2, 4, 8)
MQA_ROWS = (1, 2, 4, 8, 16, 32)  # AOT instantiation caps batch at 32
FULL_ROWS = (1, 2, 4, 8, 16, 32, 64)
CTX_AXIS = (16384, 65536, 262144)  # decode-op-bench-harness.md: short ctx untested
DEFAULT_CTX = 65536                # middle stop when the CLI passes no ctx

# projection weight shapes (n, k)
WQ_B_N, WQ_B_K = 4096, 2048
WK_N, WK_K = 128, 6144


# --- torch-free shape helpers (shared with tests/test_indexer.py) -------------

def cache_stride_bytes() -> int:
    """Per-block stride: [64*128 fp8][64*4 f32 scale] = 8448 B."""
    return BLOCK_KV * (HEAD_DIM + SCALE_BYTES_PER_TOKEN)


def block_cols(ctx: int) -> int:
    """Blocks covering one row's context (ctx is a multiple of BLOCK_KV)."""
    if ctx % BLOCK_KV:
        raise ValueError(f"ctx {ctx} must be a multiple of {BLOCK_KV}")
    return ctx // BLOCK_KV


def cache_bytes(ctx: int) -> int:
    return block_cols(ctx) * cache_stride_bytes()


def schedule_meta_len() -> int:
    """i32 entries the paged-MQA metadata kernel writes: (num_sms + 1) * 2
    (openinfer-kernels/src/ops/glm52/deepgemm_mqa.rs schedule_metadata_len)."""
    return (NUM_SMS + 1) * 2


def topk_max_len(ctx: int) -> int:
    """Harness logits row width: ctx + a 256-wide stale/garbage tail, mirroring
    production where logits_stride columns past seq_len hold stale data the
    per-row `lengths` filter must exclude."""
    return ctx + TOPK_MAX_LEN_SLACK


def seq_lens_for_rows(rows: int, ctx: int) -> list[int]:
    """Deterministic varied per-row context lengths in [5*ctx/8, ctx] — decode
    rows do not share one length. Every value stays >= TOPK at the ctx axis."""
    return [ctx - (r % 4) * (ctx // 8) for r in range(rows)]


def ctx_of(shape: dict) -> int:
    """ctx axis value for one launch shape (CLI passes only rows/n/k today)."""
    return int(shape.get("ctx", DEFAULT_CTX))


# --- torch-lazy helpers --------------------------------------------------------

def quantize_e4m3_rows(x_bf16):
    """Per-128-group e4m3 quant, bit-exact with indexer_k_quant_and_cache_kernel:
    amax over the group -> scale = max(amax, 1e-4)/448 -> value/scale clamped to
    +-448 -> RNE cast (`torch.float8_e4m3fn` conversion is round-to-nearest-even,
    same as `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)` on finite
    in-range input). Returns (q uint8 [t, 128], scale f32 [t]).

    NOTE: the scale division MUST be tensor/tensor — torch lowers
    `tensor / python_scalar` on CUDA to a reciprocal multiply, which is not the
    IEEE-correctly-rounded f32 division the kernel emits (observed on GB300 as
    a single 1-ulp scale byte over 2M written bytes)."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    xf = x_bf16.to(torch.float32)
    amax = xf.abs().amax(dim=1)
    scale = amax.clamp_min(FP8_SCALE_EPS) / torch.full_like(amax, FP8_SCALE_DIVISOR)
    q = (xf / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), scale


def block_table_for(rows: int, ctx: int, seed: int, device: str = "cuda"):
    """[rows, cols] i32 paged-KV table: row r owns pages [r*cols, (r+1)*cols)
    in a seeded permutation (production tables are arbitrary permutations; the
    pool is row-disjoint so global slots never collide across rows)."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    cols = block_cols(ctx)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    table = torch.empty((rows, cols), dtype=torch.int64)
    for r in range(rows):
        table[r] = r * cols + torch.randperm(cols, generator=gen)
    return table.to(torch.int32).to(device)


def build_paged_cache(rows: int, ctx: int, seed: int, device: str = "cuda"):
    """Physical indexer K cache [rows*cols blocks, 8448 B] filled with
    normal-distributed k quantized per token — the same recipe the production
    cache writer emits (per-row derived seeds keep CPU peak memory at one row).
    Content is keyed by PHYSICAL page id; block_table_for does the logical
    remap, so any permutation reads consistent tokens."""
    from kernel_lab import data
    from kernel_lab.loader import require_torch

    torch = require_torch()
    cols = block_cols(ctx)
    cache = torch.zeros(rows * cache_bytes(ctx), dtype=torch.uint8, device=device)
    stride = cache_stride_bytes()
    for r in range(rows):
        k = data.normal_bf16(
            (cols * BLOCK_KV, HEAD_DIM), seed=derive_seed(seed, f"cache-k:{r}")
        ).to(device)
        q, scale = quantize_e4m3_rows(k)
        region = cache[r * cols * stride : (r + 1) * cols * stride].view(cols, stride)
        region[:, : BLOCK_KV * HEAD_DIM] = q.view(cols, BLOCK_KV * HEAD_DIM)
        region[:, BLOCK_KV * HEAD_DIM :] = scale.contiguous().view(torch.uint8).view(
            cols, BLOCK_KV * SCALE_BYTES_PER_TOKEN
        )
        del k, q, scale, region
    return cache


# --- per-unit references ---------------------------------------------------------

def projections_ref(q_resid, hidden, wq_b, wq_b_scales, wk, wk_scales):
    """indexer.weights_proj: the two fp8 weight-only GEMVs of the indexer
    projection stage, flattened and concatenated [q(rows*4096) | k_raw(rows*128)]
    to match the adapter's out buffer."""
    from kernel_lab.loader import require_torch
    from kernel_lab.refs.fp8_gemv import fp8_weight_only_gemv_ref

    torch = require_torch()
    q = fp8_weight_only_gemv_ref(q_resid, wq_b, wq_b_scales, WQ_B_N, WQ_B_K)
    k = fp8_weight_only_gemv_ref(hidden, wk, wk_scales, WK_N, WK_K)
    return torch.cat([q.reshape(-1), k.reshape(-1)])


def rope_ref(q, k, cos, sin):
    """indexer.rope: NON-interleaved (half-split / NeoX-style) RoPE, mirroring
    glm52_indexer_rope.cu — NOT the interleaved pairing (the landed kernel and
    openinfer-glm52/src/layer.rs:105 both pin half-split for the indexer):
      out[..., j]    = x[j]*cos[j] - x[j+32]*sin[j]
      out[..., j+32] = x[j+32]*cos[j] + x[j]*sin[j]     (j in 0..31)
      out[..., 64:]  = pass-through
    cos/sin carry one [32] row per token. Returns (q_out f32, k_out f32) from
    the PRE-rotation inputs (the kernel rotates in place — the adapter keeps
    input clones for this reference)."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    cf = cos.to(torch.float32)  # [rows, 32]
    sf = sin.to(torch.float32)

    def rot(x, c, s):
        a = x[..., :ROPE_HALF]
        b = x[..., ROPE_HALF:ROPE_DIM]
        lo = a * c - b * s
        hi = b * c + a * s
        return torch.cat([lo, hi, x[..., ROPE_DIM:]], dim=-1)

    q_out = rot(q.to(torch.float32), cf.unsqueeze(-2), sf.unsqueeze(-2))
    k_out = rot(k.to(torch.float32), cf, sf)
    return q_out, k_out


def k_quant_cache_ref(k, slot_mapping, num_blocks: int):
    """indexer.k_quant_cache: full reference cache image (zeros elsewhere) for
    glm52_indexer_k_quant_and_cache_cuda with head_dim=128, quant_block=128,
    block_size=64, stride=8448. The quant is deterministic end to end (amax and
    scale are exact f32 ops, the RNE cast is deterministic), so the adapter
    gates on byte equality. slot_mapping must be non-negative and distinct."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    slots = slot_mapping.to(torch.int64)
    if bool((slots < 0).any()):
        raise ValueError("k_quant_cache_ref: negative slot_mapping")
    if int(slots.unique().numel()) != int(slots.numel()):
        raise ValueError("k_quant_cache_ref: slot_mapping not distinct")
    stride = cache_stride_bytes()
    cache = torch.zeros(num_blocks * stride, dtype=torch.uint8, device=k.device)
    q, scale = quantize_e4m3_rows(k)
    blk = slots // BLOCK_KV
    off = slots % BLOCK_KV
    base = blk * stride
    cols = torch.arange(HEAD_DIM, device=k.device)
    val_off = (base + off * HEAD_DIM)[:, None] + cols[None, :]
    cache[val_off.reshape(-1)] = q.reshape(-1)
    sc_off = (base + BLOCK_KV * HEAD_DIM + off * SCALE_BYTES_PER_TOKEN)[:, None] + torch.arange(
        SCALE_BYTES_PER_TOKEN, device=k.device
    )[None, :]
    cache[sc_off.reshape(-1)] = (
        scale.contiguous().view(torch.uint8).reshape(-1)
    )
    return cache


def mqa_logits_ref(q_fp8, cache, weights, context_lens, block_table, ctx: int):
    """indexer.mqa_logits: per-row f32 reference logits over each row's valid
    context [0, context_lens[r]). Mirrors the DeepGEMM paged kernel semantics
    (vllm DeepseekV32Indexer, no Hadamard):
      logit[j] = sum_h relu(q_h . k_j_deq) * weights[h]
    with q used as raw fp8 (its group scale is folded into `weights` upstream)
    and k_j_deq = e4m3(k_j) * k_scale_j. Returns a list of [len_r] f32 rows —
    columns past context_lens[r] are kernel-scheduler territory and excluded."""
    from kernel_lab.loader import require_torch
    from kernel_lab.refs.fp8_gemv import e4m3_decode_torch

    torch = require_torch()
    rows = int(weights.shape[0])
    stride = cache_stride_bytes()
    k_blocks = cache.view(-1, stride)
    outs = []
    for r in range(rows):
        ln = int(context_lens[r].item())
        ncol = -(-ln // BLOCK_KV)
        pages = block_table[r, :ncol].to(torch.int64)
        blk = k_blocks[pages]  # [ncol, stride] fresh contiguous copy
        k_fp8 = blk[:, : BLOCK_KV * HEAD_DIM].reshape(ncol * BLOCK_KV, HEAD_DIM)[:ln]
        k_scale = (
            blk[:, BLOCK_KV * HEAD_DIM :]
            .contiguous()
            .view(torch.float32)
            .reshape(ncol * BLOCK_KV)[:ln]
        )
        k_deq = e4m3_decode_torch(k_fp8) * k_scale[:, None]
        q_r = e4m3_decode_torch(
            q_fp8[r * INDEX_HEADS * HEAD_DIM : (r + 1) * INDEX_HEADS * HEAD_DIM].view(
                INDEX_HEADS, HEAD_DIM
            )
        )
        dots = q_r @ k_deq.t()  # [heads, ln] f32
        score = dots.clamp_min_(0.0) * weights[r][:, None]
        outs.append(score.sum(dim=0))
        del blk, k_fp8, k_scale, k_deq, dots, score
    return outs


def topk_ref(logits, lengths, top_k: int = TOPK):
    """indexer.topk_2048: reference (indices, values) under the kernel's
    contract — FilteredTopK(deterministic, TopKTieBreak::Small) +
    LaunchSortTopKByIndex (glm52_topk.cu):
      1. columns >= lengths[r] are excluded (stale/garbage tail filter),
      2. value ties resolve to the SMALLER index — a stable descending sort
         keeps equal keys in ascending index order, same winner,
      3. the selected indices are emitted sorted ascending by index.
    Returns (indices [rows, top_k] ascending, values [rows, top_k])."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    cols = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill(cols[None, :] >= lengths[:, None], float("-inf"))
    vals, idxs = masked.sort(dim=1, descending=True, stable=True)
    sel_i, sel_v = idxs[:, :top_k], vals[:, :top_k]
    order = sel_i.argsort(dim=1)
    return sel_i.gather(1, order), sel_v.gather(1, order)


def assert_topk_match(got_i, got_v, ref_i, ref_v, logits, top_k: int = TOPK):
    """Hard gate for indexer.topk_2048, same discipline as the oracle harness
    (docs/models/glm52/indexer-forward.md): FlashInfer TopKTieBreak::Small vs
    the torch stable sort can diverge on exact f32 value ties, so the index
    sets may differ by at most one entry per row and only when the swapped
    logits are exactly equal. Values are compared as multisets (a tie swap
    permutes equal values). Anything else raises."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    rows = logits.shape[0]
    for r in range(rows):
        g = set(got_i[r].tolist())
        w = set(ref_i[r].tolist())
        missing = w - g
        extra = g - w
        if len(missing) > 1 or len(extra) > 1:
            raise AssertionError(
                f"topk row {r}: index sets diverge by {len(missing)} entries "
                f"(allowed: <=1 tie-break swap)"
            )
        if missing:
            ri = next(iter(missing))
            gi = next(iter(extra))
            lv = float(logits[r, ri].item())
            gv = float(logits[r, gi].item())
            if lv != gv:
                raise AssertionError(
                    f"topk row {r}: non-tie divergence — ref picked {ri} (logit "
                    f"{lv}) but kernel picked {gi} (logit {gv})"
                )
    # Kernel self-consistency: emitted values must equal the logits at the
    # emitted indices, exactly.
    gathered = logits.gather(1, got_i.to(torch.int64))
    if not bool(torch.equal(gathered, got_v)):
        raise AssertionError("topk: kernel values != logits at kernel indices")
    # Value multisets must match exactly (tie swaps only permute equal values).
    if not bool(
        torch.equal(got_v.sort(dim=1).values, ref_v.sort(dim=1).values)
    ):
        raise AssertionError("topk: value multisets differ")


def local_topk_to_slots_ref(offsets, seq_lens, block_table, block_size: int = BLOCK_KV):
    """indexer.local_topk_to_slots: integer remap, exact-match semantics.
    slot = block_table[t, off//bs]*bs + off%bs for 0 <= off < seq_len (and the
    block index in range), else -1; topk_lens = valid count. Mirrors
    local_topk_to_global_slots_kernel in glm52_indexer.cu."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    off = offsets.to(torch.int64)
    ln = seq_lens.to(torch.int64)[:, None]
    cols = block_table.shape[1]
    blk = off // block_size
    within = off % block_size
    valid = (off >= 0) & (off < ln) & (blk >= 0) & (blk < cols)
    pages = block_table.to(torch.int64).gather(1, blk.clamp(0, cols - 1))
    slots = torch.where(valid, pages * block_size + within, torch.full_like(pages, -1))
    lens = valid.sum(dim=1).to(torch.int32)
    return slots.to(torch.int32), lens


# ---------------------------------------------------------------------------
# rows x ctx sweep driver (shared CLI has no --ctx selector yet)
# ---------------------------------------------------------------------------

def iter_shape_points(manifest_shape: dict, rows_axes, ctx_axes):
    """rows x ctx grid as CLI-style shape dicts (adapters read `ctx` via
    shape.get("ctx", DEFAULT_CTX); the shared CLI never injects it)."""
    for r in rows_axes:
        for c in ctx_axes:
            yield {"rows": r, "n": manifest_shape["n"], "k": manifest_shape["k"], "ctx": c}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m kernel_lab.refs.indexer",
        description="indexer-group rows x ctx sweep (check vs reference / bench)",
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
                  f"mean={stats.mean_us:.2f}  rounds={len(stats.samples_us)}  ({arch})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
