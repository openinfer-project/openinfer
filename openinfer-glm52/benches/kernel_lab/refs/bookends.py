"""Torch references + shared input helpers for the bookend units (embed /
lm_head / argmax_split). torch is lazy-imported inside functions; the argmax
layout math is pure stdlib so the CPU tests can exercise it.
"""
from __future__ import annotations

from kernel_lab import data

# --- adversarial argmax layout (stdlib-only; shared by adapter and tests) ----
#
# Every logits row gets: an exact bf16 tie at two positions half a vocab apart
# (the kernel must pick the LOWER global index — its partials cross 4096-wide
# tiles, so a tile-local tie bug changes the answer), and one NaN that must
# never win. Offsets are vocab/2 and vocab/4, i.e. 18.90625 and 9.453125
# tiles — never a whole number of 4096-elem tiles, so the three slots always
# land in DIFFERENT tiles for every primary position.
ARGMAX_TIE_OFFSET = 77440  # GLM52_VOCAB / 2
ARGMAX_NAN_OFFSET = 38720  # GLM52_VOCAB / 4
ARGMAX_TIE_VALUE = 1024.0  # exact in bf16; |N(0,1)| background stays < 6


def argmax_layout(p: int, vocab: int) -> tuple[int, int, int]:
    """(primary, tie_partner, nan_slot) for a seeded primary `p` — pairwise
    distinct for every p in [0, vocab) (offsets are nonzero mod vocab and
    distinct from each other)."""
    return p, (p + ARGMAX_TIE_OFFSET) % vocab, (p + ARGMAX_NAN_OFFSET) % vocab


# --- 1.9 GB table cache -------------------------------------------------------

_TABLE_CACHE: dict = {}


def cached_table(name: str, n: int, k: int, seed: int):
    """Process-wide cache for the [154880, 6144] bf16 tables (embed_tokens /
    lm_head — untied, so they get separate name-derived seeds). make_inputs
    runs once per rows value per command; regenerating 951M normals per call
    would dominate the session. The tables are read-only kernel inputs, so
    sharing one tensor across shapes (and across the two make_inputs calls of
    `compare --baseline-so`) is safe."""
    key = (name, n, k, seed)
    t = _TABLE_CACHE.get(key)
    if t is None:
        t = data.normal_bf16((n, k), seed=data.derive_seed(seed, name))
        _TABLE_CACHE[key] = t
    return t


# --- references (f32; semantic-level nets, tolerances in the manifests) -------

def embedding_gather_ref(table_bf16, token_ids):
    """out[r] = table[ids[r]] — a pure row gather, exact in any dtype. Gather
    in bf16 first (cheap), then widen: identical bits to the kernel output."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    rows = table_bf16[token_ids.to(torch.int64)]
    return rows.to(torch.float32)


def lm_head_ref(normed_bf16, weight_bf16):
    """logits[t, v] = sum_h normed[t, h] * weight[v, h], f32 accumulation —
    the cuBLAS CUBLAS_COMPUTE_32F semantics with torch's own association
    order (the tolerance absorbs the reorder)."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    return normed_bf16.to(torch.float32) @ weight_bf16.to(torch.float32).t()


def argmax_lowest_index_ref(logits_bf16):
    """Per-row lowest global index of the max — the kernel's total order
    (argmax_better: greater wins, ties to the lower index, NaN never wins)
    made explicit; torch.argmax's tie order is unspecified and is NOT used.
    NaN is masked to -inf before the max, matching the kernel (a NaN compares
    false against every candidate and can never displace the -inf seed).
    Returns f32 indices for refs.compute_metrics."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    x = logits_bf16.to(torch.float32)
    x = torch.where(torch.isnan(x), torch.full_like(x, float("-inf")), x)
    maxv = x.amax(dim=-1, keepdim=True)
    arange = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)
    idx = torch.where(x == maxv, arange, torch.full_like(arange, x.shape[-1]))
    return idx.amin(dim=-1)
