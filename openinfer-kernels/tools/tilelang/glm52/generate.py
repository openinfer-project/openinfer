#!/usr/bin/env python3
"""Generate the GLM5.2 right-sized sparse MLA decode kernel (TileLang AOT).

The TileLang program below is adapted from DeepSeek-AI's DeepSeek-V3.2
sparse MLA decode kernel. The upstream model repository is MIT licensed.
Upstream notice preserved for the adapted portions:

MIT License
Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Replaces FlashMLA's sparse splitkv kernel on the GLM5.2 attention-TP decode
path (heads <= 16): at the production shape (bucket-8, topk 2048) FlashMLA is
~80-95% fixed overhead (22.8 us where this kernel measures 12.8 us on H200).
Emits one main-kernel instantiation per supported topk plus an extern "C"
launcher; the fixed-order split combine stays in
csrc/glm52/glm52_sparse_mla.cu and consumes this kernel's unnormalized
partials + (m, l) contract.

Cache tokens are the packed 656-byte fp8_ds_mla layout
([512 e4m3 ckv | 4 f32 group scales | 64 bf16 rope]) read in place through
three contiguous aliases of the same pointer: u8 [slots, 656],
f32 [slots, 164] (scales at word 128), bf16 [slots, 328] (rope at half 264).
656 = 4*164 = 2*328, so no strided views and no cache layout change.

Known TileLang traps encoded below (each cost a debugging round):
  - An if/else writing two DIFFERENT smem buffers inside one T.Parallel
    miscompiles: both vectorized stores are emitted unconditionally per
    thread (one from uninitialized registers) with the second buffer
    addressed at base-8192, spraying garbage into the adjacent buffer.
    Dequant is therefore two branch-free loops (one per output tile).
  - Fast-math ex2.approx maps NaN to finite junk, not NaN: garbage bytes
    must never enter smem, so invalid gather slots clamp to slot 0's real
    bytes instead of skipping the copy.
  - T.ptx_cp_async's last argument is an ELEMENT count, not bytes.
"""

import argparse
import re
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH


# TileLang releases change codegen in ways this generator depends on: 0.1.9
# lowers dynamic smem WITHOUT the void*-offset declarations that
# dynamic_smem_bytes() parses (zero regex matches -> loud failure), and the
# kernel signature can drift too. Bumping requires revalidating
# EXPECTED_PARAMS, the smem accounting, AND the H200 parity gate — don't
# just widen this check.
KNOWN_GOOD_TILELANG = "0.1.12"
if tilelang.__version__ != KNOWN_GOOD_TILELANG:
    raise RuntimeError(
        f"tilelang {tilelang.__version__} is not the validated "
        f"{KNOWN_GOOD_TILELANG}; point OPENINFER_TILELANG_PYTHON at an env "
        "with the pinned version, or revalidate the codegen contract and "
        "bump KNOWN_GOOD_TILELANG"
    )

tilelang.set_log_level("WARNING")

# Lower for Hopper explicitly: the kernel is compiled by nvcc later, so
# generation must not depend on the GPU in the build machine ("auto" target
# fails on non-Hopper hosts even when cross-building with
# OPENINFER_CUDA_SM=90). Dict form — 0.1.12 rejects attribute-carrying
# target strings.
SM90A_TARGET = {"kind": "cuda", "arch": "sm_90a"}

# Production only runs topk 2048: the short-context tier (topk 256 while
# every row's context fit in it) was dropped — agent traffic starts well past
# 2048 tokens of context. To build the 256 instantiation again, add 256 here
# and keep `GLM52_SPARSE_MLA_TOPKS` (src/ops/glm52/sparse_mla.rs) and
# `supported_topk` (csrc/glm52/glm52_sparse_mla.cu) in sync; the kernel's
# bound-masking already handles topk < one gather stage.
TOPKS = [2048]
NUM_SPLITS = 16
HEAD_SLOTS_OUT = 16  # partial store width; pad slots past it are never read
SM_SCALE = 0.0625  # GLM52_SM_SCALE, baked; the launcher-side entry validates
DQK, DV, DT = 576, 512, 64

EXPECTED_PARAMS = (
    "(const int* __restrict__ Indices, const uchar* __restrict__ KVBytes, "
    "const bfloat16_t* __restrict__ KVHalves, const float* __restrict__ KVWords, "
    "float* __restrict__ Ml, float* __restrict__ OPart, "
    "const bfloat16_t* __restrict__ Q, int batch, int num_slots)"
)

# dynamic-smem buffer sizes in bytes, used to recover the launch smem size
# from the generated source (TileLang's own launcher is not emitted here)
SMEM_BUFFER_BYTES = {
    "Q_shared_l": 64 * 256 * 2,
    "Q_shared_r": 64 * 256 * 2,
    "Q_tail_shared": 64 * 64 * 2,
    "KV_fp8_0": 32 * 512,
    "KV_fp8_1": 32 * 512,
    "KScale_0": 32 * 4 * 4,
    "KScale_1": 32 * 4 * 4,
    "K_tail_shared_0": 32 * 64 * 2,
    "K_tail_shared_1": 32 * 64 * 2,
    "KV_l_0": 32 * 256 * 2,
    "KV_r_0": 32 * 256 * 2,
    "KV_l_1": 32 * 256 * 2,
    "KV_r_1": 32 * 256 * 2,
    "S_shared": 64 * 32 * 2,
}


@tilelang.jit(
    target=SM90A_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def glm52_sparse_mla_tl(
    Q,
    KVBytes,
    KVWords,
    KVHalves,
    Indices,
    OPart,
    Ml,
    topk,
    sm_scale,
    n_heads_out=HEAD_SLOTS_OUT,
    num_splits=NUM_SPLITS,
    block_I=32,
    threads=384,
):
    sm_scale = sm_scale * 1.44269504  # log2(e)

    batch, num_slots = T.dynamic("batch, num_slots")

    H = 64
    BI = block_I
    TPC = topk // num_splits  # tokens per split CTA
    NI = tilelang.cdiv(TPC, BI)
    OUTER = tilelang.cdiv(NI, 2)
    D = DV
    D_tail = DT
    dtype = T.bfloat16
    fp8 = T.float8_e4m3fn
    accum_dtype = T.float32

    Q: T.Tensor([batch, H, DQK], dtype)  # type: ignore
    KVBytes: T.Tensor([num_slots, 656], T.uint8)  # type: ignore
    KVWords: T.Tensor([num_slots, 164], accum_dtype)  # type: ignore
    KVHalves: T.Tensor([num_slots, 328], dtype)  # type: ignore
    Indices: T.Tensor([batch, topk], T.int32)  # type: ignore
    OPart: T.Tensor([num_splits, batch, n_heads_out, D], accum_dtype)  # type: ignore
    Ml: T.Tensor([num_splits, batch, n_heads_out, 2], accum_dtype)  # type: ignore

    with T.Kernel(num_splits, batch, 1, threads=threads) as (bx, by, bz):
        Q_shared_l = T.alloc_shared([H, D // 2], dtype)
        Q_shared_r = T.alloc_shared([H, D // 2], dtype)
        Q_tail_shared = T.alloc_shared([H, D_tail], dtype)
        KV_fp8_0 = T.alloc_shared([BI, D], fp8)
        KV_fp8_1 = T.alloc_shared([BI, D], fp8)
        KScale_0 = T.alloc_shared([BI, 4], accum_dtype)
        KScale_1 = T.alloc_shared([BI, 4], accum_dtype)
        K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
        K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
        KV_l_0 = T.alloc_shared([BI, D // 2], dtype)
        KV_r_0 = T.alloc_shared([BI, D // 2], dtype)
        KV_l_1 = T.alloc_shared([BI, D // 2], dtype)
        KV_r_1 = T.alloc_shared([BI, D // 2], dtype)
        is_kv_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
        is_kv_valid_1 = T.alloc_shared([BI], "bool", scope="shared")

        acc_o_l = T.alloc_fragment([H, D // 2], accum_dtype)
        acc_o_r = T.alloc_fragment([H, D // 2], accum_dtype)
        acc_s = T.alloc_fragment([H, BI], accum_dtype)
        S_shared = T.alloc_shared([H, BI], dtype)
        sumexp = T.alloc_fragment([H], accum_dtype)
        sumexp_i = T.alloc_fragment([H], accum_dtype)
        alpha_shared = T.alloc_shared([H], accum_dtype, scope="shared")
        alpha_local = T.alloc_fragment([H], accum_dtype)
        m_i = T.alloc_fragment([H], accum_dtype)
        m_i_prev = T.alloc_fragment([H], accum_dtype)
        indices_local = T.alloc_var(T.int32)
        tok_in_split = T.alloc_var(T.int32)

        # raw gather landed (producer cp.async)
        bar_raw_0 = T.alloc_barrier(arrive_count=128)
        bar_raw_1 = T.alloc_barrier(arrive_count=128)
        # both consumer halves dequanted
        bar_deq_0 = T.alloc_barrier(arrive_count=256)
        bar_deq_1 = T.alloc_barrier(arrive_count=256)
        # bf16 tiles + tail + is_valid consumed (both consumer groups arrive,
        # producer waits before reusing the slot)
        bar_free_0 = T.alloc_barrier(arrive_count=256)
        bar_free_1 = T.alloc_barrier(arrive_count=256)
        bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
        bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

        split = bx
        b_i = by

        tx = T.get_thread_binding()

        if tx < 128:
            # consumer 0: QK + softmax + PV-left. Only this group reads Q, so
            # it stages Q itself with plain copies — no TMA descriptor in the
            # kernel ABI (the AOT launcher stays a bare <<<>>> call).
            T.set_max_nreg(224, 1)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))
            T.fill(acc_o_l, 0)
            T.copy(Q[b_i, 0:H, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, 0:H, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, 0:H, D:], Q_tail_shared)

            for i_i in T.serial(OUTER):
                # ---- buffer 0 ----
                T.barrier_wait(bar_raw_0[0], (i_i & 1))
                for bi_i, d_i in T.Parallel(BI, D // 2):
                    KV_l_0[bi_i, d_i] = (
                        KV_fp8_0[bi_i, d_i].astype(accum_dtype)
                        * KScale_0[bi_i, d_i // 128]
                    ).astype(dtype)
                T.barrier_arrive(bar_deq_0[0])
                T.barrier_wait(bar_deq_0[0], (i_i & 1))
                for h_i, bi_i in T.Parallel(H, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        is_kv_valid_0[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                T.wgmma_gemm(Q_shared_l, KV_l_0, acc_s, transpose_B=True)
                T.wgmma_gemm(Q_shared_r, KV_r_0, acc_s, transpose_B=True)
                T.wgmma_gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True)
                T.wait_wgmma(0)

                if i_i != 0:
                    T.barrier_arrive(bar_sScale_and_sS_free)
                    T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H):
                    alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H):
                    sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H, D // 2):
                    acc_o_l[h_i, d_i] *= alpha_local[h_i]
                T.copy(alpha_local, alpha_shared)

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_l_0, acc_o_l)
                T.wait_wgmma(0)

                T.barrier_arrive(bar_sScale_and_sS_ready)
                T.barrier_arrive(bar_free_0[0])

                # ---- buffer 1 ----
                T.barrier_wait(bar_raw_1[0], (i_i & 1))
                for bi_i, d_i in T.Parallel(BI, D // 2):
                    KV_l_1[bi_i, d_i] = (
                        KV_fp8_1[bi_i, d_i].astype(accum_dtype)
                        * KScale_1[bi_i, d_i // 128]
                    ).astype(dtype)
                T.barrier_arrive(bar_deq_1[0])
                T.barrier_wait(bar_deq_1[0], (i_i & 1))
                for h_i, bi_i in T.Parallel(H, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        is_kv_valid_1[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                T.wgmma_gemm(Q_shared_l, KV_l_1, acc_s, transpose_B=True)
                T.wgmma_gemm(Q_shared_r, KV_r_1, acc_s, transpose_B=True)
                T.wgmma_gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True)
                T.wait_wgmma(0)

                T.barrier_arrive(bar_sScale_and_sS_free)
                T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H):
                    alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H):
                    sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H, D // 2):
                    acc_o_l[h_i, d_i] *= alpha_local[h_i]
                T.copy(alpha_local, alpha_shared)

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_l_1, acc_o_l)
                T.wait_wgmma(0)

                T.barrier_arrive(bar_sScale_and_sS_ready)
                T.barrier_arrive(bar_free_1[0])

            # split partial: UNNORMALIZED acc + (m*scale log2-domain, sumexp),
            # the CUDA combine contract; an all-invalid split has l == 0 and
            # combines to zero there. Pad head slots (zero queries) are never
            # read downstream, so skipping their store cuts the f32 partial
            # round-trip 4x.
            for h_i, d_i in T.Parallel(H, D // 2):
                if h_i < n_heads_out:
                    OPart[split, b_i, h_i, d_i] = acc_o_l[h_i, d_i]
            for h_i in T.Parallel(H):
                if h_i < n_heads_out:
                    Ml[split, b_i, h_i, 0] = m_i[h_i] * sm_scale
            for h_i in T.Parallel(H):
                if h_i < n_heads_out:
                    Ml[split, b_i, h_i, 1] = sumexp[h_i]

        elif tx >= 128 and tx < 256:
            # consumer 1: PV-right (dequants its own KV_r half, v1 pipeline —
            # producer-side dequant measured slower AND a same-T.Parallel
            # if/else split store miscompiles; see module docstring)
            T.set_max_nreg(168, 1)
            T.fill(acc_o_r, 0)
            for i_i in T.serial(OUTER):
                T.barrier_wait(bar_raw_0[0], (i_i & 1))
                for bi_i, d_i in T.Parallel(BI, D // 2):
                    KV_r_0[bi_i, d_i] = (
                        KV_fp8_0[bi_i, d_i + D // 2].astype(accum_dtype)
                        * KScale_0[bi_i, (d_i + D // 2) // 128]
                    ).astype(dtype)
                T.barrier_arrive(bar_deq_0[0])

                T.barrier_arrive(bar_sScale_and_sS_ready)
                T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                for h_i, d_i in T.Parallel(H, D // 2):
                    acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                T.gemm(S_shared, KV_r_0, acc_o_r)
                T.wait_wgmma(0)
                T.barrier_arrive(bar_free_0[0])
                T.barrier_arrive(bar_sScale_and_sS_free)

                T.barrier_wait(bar_raw_1[0], (i_i & 1))
                for bi_i, d_i in T.Parallel(BI, D // 2):
                    KV_r_1[bi_i, d_i] = (
                        KV_fp8_1[bi_i, d_i + D // 2].astype(accum_dtype)
                        * KScale_1[bi_i, (d_i + D // 2) // 128]
                    ).astype(dtype)
                T.barrier_arrive(bar_deq_1[0])

                T.barrier_arrive(bar_sScale_and_sS_ready)
                T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                for h_i, d_i in T.Parallel(H, D // 2):
                    acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                T.gemm(S_shared, KV_r_1, acc_o_r)
                T.wait_wgmma(0)
                T.barrier_arrive(bar_free_1[0])
                if i_i != OUTER - 1:
                    T.barrier_arrive(bar_sScale_and_sS_free)

            for h_i, d_i in T.Parallel(H, D // 2):
                if h_i < n_heads_out:
                    OPart[split, b_i, h_i, D // 2 + d_i] = acc_o_r[h_i, d_i]

        elif tx >= 256:
            # producer: cp.async gather of packed 656B tokens through the
            # three aliased views; 16 threads per token, 8 tokens per round.
            # Stage-tail slots (tok >= TPC) and -1 indices flag invalid and
            # clamp to slot 0, gathering slot 0's real bytes. An index
            # >= num_slots is outside the indexer contract: the generated
            # cp.async predicate zero-fills it while is_kv_valid stays TRUE
            # (it attends with zero logits) — unlike the f64 reference,
            # which __trap()s on it.
            T.set_max_nreg(104, 0)
            for i_i in T.serial(OUTER):
                # ---- buffer 0 ----
                if i_i != 0:
                    T.barrier_wait(bar_free_0[0], ((i_i & 1) ^ 1))
                for r in T.serial(4):
                    tok_in_split = (i_i * 2) * BI + r * 8 + (tx - 256) // 16
                    indices_local = T.if_then_else(
                        tok_in_split < TPC,
                        Indices[b_i, split * TPC + T.min(tok_in_split, TPC - 1)],
                        -1,
                    )
                    is_kv_valid_0[r * 8 + (tx - 256) // 16] = indices_local >= 0
                    indices_local = T.max(indices_local, 0)
                    for u in T.serial(2):
                        T.ptx_cp_async(
                            T.access_ptr(
                                KV_fp8_0[
                                    r * 8 + (tx - 256) // 16,
                                    256 * u + (tx - 256) % 16 * 16,
                                ],
                                "w",
                                16,
                            ),
                            T.access_ptr(
                                KVBytes[
                                    indices_local, 256 * u + (tx - 256) % 16 * 16
                                ],
                                "r",
                                16,
                            ),
                            16,
                        )
                    if (tx - 256) % 16 == 0:
                        T.ptx_cp_async(
                            T.access_ptr(
                                KScale_0[r * 8 + (tx - 256) // 16, 0], "w", 4
                            ),
                            T.access_ptr(KVWords[indices_local, 128], "r", 4),
                            4,
                        )
                    if (tx - 256) % 16 < 8:
                        T.ptx_cp_async(
                            T.access_ptr(
                                K_tail_shared_0[
                                    r * 8 + (tx - 256) // 16,
                                    (tx - 256) % 16 * 8,
                                ],
                                "w",
                                8,
                            ),
                            T.access_ptr(
                                KVHalves[indices_local, 264 + (tx - 256) % 16 * 8],
                                "r",
                                8,
                            ),
                            8,
                        )
                T.cp_async_barrier_noinc(bar_raw_0[0])

                # ---- buffer 1 ----
                if i_i != 0:
                    T.barrier_wait(bar_free_1[0], ((i_i & 1) ^ 1))
                for r in T.serial(4):
                    tok_in_split = (i_i * 2 + 1) * BI + r * 8 + (tx - 256) // 16
                    indices_local = T.if_then_else(
                        tok_in_split < TPC,
                        Indices[b_i, split * TPC + T.min(tok_in_split, TPC - 1)],
                        -1,
                    )
                    is_kv_valid_1[r * 8 + (tx - 256) // 16] = indices_local >= 0
                    indices_local = T.max(indices_local, 0)
                    for u in T.serial(2):
                        T.ptx_cp_async(
                            T.access_ptr(
                                KV_fp8_1[
                                    r * 8 + (tx - 256) // 16,
                                    256 * u + (tx - 256) % 16 * 16,
                                ],
                                "w",
                                16,
                            ),
                            T.access_ptr(
                                KVBytes[
                                    indices_local, 256 * u + (tx - 256) % 16 * 16
                                ],
                                "r",
                                16,
                            ),
                            16,
                        )
                    if (tx - 256) % 16 == 0:
                        T.ptx_cp_async(
                            T.access_ptr(
                                KScale_1[r * 8 + (tx - 256) // 16, 0], "w", 4
                            ),
                            T.access_ptr(KVWords[indices_local, 128], "r", 4),
                            4,
                        )
                    if (tx - 256) % 16 < 8:
                        T.ptx_cp_async(
                            T.access_ptr(
                                K_tail_shared_1[
                                    r * 8 + (tx - 256) // 16,
                                    (tx - 256) % 16 * 8,
                                ],
                                "w",
                                8,
                            ),
                            T.access_ptr(
                                KVHalves[indices_local, 264 + (tx - 256) % 16 * 8],
                                "r",
                                8,
                            ),
                            8,
                        )
                T.cp_async_barrier_noinc(bar_raw_1[0])


def kernel_name(topk: int) -> str:
    return f"glm52_tilelang_sparse_mla_topk{topk}_kernel"


def dynamic_smem_bytes(source: str) -> int:
    offsets = re.findall(
        r"void\* (\w+) = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\);", source
    )
    if not offsets:
        raise RuntimeError("no dynamic smem buffers found in kernel source")
    total = 0
    for name, offset in offsets:
        if name not in SMEM_BUFFER_BYTES:
            raise RuntimeError(f"unknown dynamic smem buffer {name}")
        total = max(total, int(offset) + SMEM_BUFFER_BYTES[name])
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "glm52_tilelang_sparse_mla.cu"

    sources = []
    dispatch = []
    smem_bytes = None
    for topk in TOPKS:
        source = glm52_sparse_mla_tl.get_kernel_source(
            topk=topk,
            sm_scale=SM_SCALE,
            n_heads_out=HEAD_SLOTS_OUT,
            num_splits=NUM_SPLITS,
        )
        if EXPECTED_PARAMS not in source:
            raise RuntimeError(
                f"kernel signature drifted for topk={topk}; update the launcher"
            )
        smem = dynamic_smem_bytes(source)
        if smem_bytes is None:
            smem_bytes = smem
        elif smem_bytes != smem:
            raise RuntimeError(f"smem size differs across topk: {smem_bytes} vs {smem}")
        sources.append(
            source.replace("glm52_sparse_mla_tl_kernel", kernel_name(topk))
        )
        dispatch.append(
            f"""  if (topk == {topk}) {{
    cudaError_t err = cudaFuncSetAttribute(
        {kernel_name(topk)},
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes);
    if (err != cudaSuccess) {{
      return static_cast<int>(err);
    }}
    {kernel_name(topk)}<<<grid, kThreads, kSharedBytes, stream>>>(
        indices,
        reinterpret_cast<const uchar*>(cache),
        reinterpret_cast<const bfloat16_t*>(cache),
        reinterpret_cast<const float*>(cache),
        ml,
        o_part,
        reinterpret_cast<const bfloat16_t*>(q),
        batch,
        static_cast<int>(num_slots));
    return static_cast<int>(cudaGetLastError());
  }}"""
        )

    launcher = f"""
// cudaFuncSetAttribute is per-device state; running it on every call keeps
// each TP rank's device opted in (a process-wide once left 7 of 8 ranks
// launching without the opt-in). Launches happen at graph-capture time only,
// so the extra driver call never sits on the decode hot path.
//
// num_splits / head_slots are the caller's compile-time constants, validated
// against the values baked into this instantiation: the split count and
// partial store width live in three languages (this generator, the CUDA
// combine, the Rust scratch sizing) and a silent mismatch is a device-side
// OOB write into the o_part arena.
extern "C" int glm52_tilelang_sparse_mla_decode(
    const void* q,
    const void* cache,
    const int* indices,
    float* o_part,
    float* ml,
    int batch,
    long long num_slots,
    int topk,
    int num_splits,
    int head_slots,
    cudaStream_t stream) {{
  constexpr int kThreads = 384;
  constexpr int kSharedBytes = {smem_bytes};
  if (num_splits != {NUM_SPLITS} || head_slots != {HEAD_SLOTS_OUT}) {{
    return static_cast<int>(cudaErrorInvalidValue);
  }}
  if (num_slots <= 0 || num_slots > 0x7fffffffLL) {{
    return static_cast<int>(cudaErrorInvalidValue);
  }}
  dim3 grid({NUM_SPLITS}, batch, 1);
{chr(10).join(dispatch)}
  return static_cast<int>(cudaErrorInvalidValue);
}}
"""

    out_path.write_text(
        "// Generated by openinfer-kernels/tools/tilelang/glm52/generate.py\n"
        "#include <cuda_runtime.h>\n"
        "\n" + "\n".join(sources) + "\n" + launcher
    )
    print(f"CU_PATH={out_path}")
    print(f"TILELANG_TEMPLATE_PATH={TILELANG_TEMPLATE_PATH}")
    print(f"CUTLASS_INCLUDE_DIR={CUTLASS_INCLUDE_DIR}")


if __name__ == "__main__":
    main()
