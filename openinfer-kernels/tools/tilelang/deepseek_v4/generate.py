#!/usr/bin/env python3
"""Generate DeepSeek V4 TileLang CUDA kernels for openinfer.

The TileLang programs below are adapted from DeepSeek-AI's official
`DeepSeek-V4-Flash/inference/kernel.py` kernels. The upstream model repository
is MIT licensed. Upstream notice preserved for the adapted portions:

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

This generator intentionally emits CUDA kernel source plus small launch
wrappers instead of using TileLang's cython wrapper because the installed
TileLang wrapper cannot marshal `float8_e8m0fnu` scale tensors through its
generic C ABI.
"""

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH


tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

FP8_LINEAR_SHAPES = [
    # name, out_dim (N), in_dim (K)
    ("wkv", 512, 4096),
    ("wq_a", 1024, 4096),
    ("shared_w1_w3", 2048, 4096),
    ("wq_b_wo_b", 4096, 1024),
    ("indexer_wq_b", 1024, 1024),
    ("shared_w2", 4096, 2048),
]

FP4_LINEAR_SHAPES = [
    ("expert_w1_w3", 2048, 4096),
    ("expert_w2", 4096, 2048),
]

# The grouped FP4 kernels only need the TileLang dynamic shared-memory window
# used by their per-CTA tiles. Keeping the old 96 KiB request suppresses
# occupancy on sparse decode MoE shapes without improving correctness.
FP4_GROUPED_GEMM_SHARED_BYTES = 32768

FP4_QUANT_INPLACE_SHAPES = [
    # name, hidden dim (N), block size
    ("n128", 128, 32),
]

SPARSE_ATTN_SHAPES = [
    # name, local_heads, head_dim, softmax_scale
    ("local_h16_d512", 16, 512, (1.0 / 512.0) ** 0.5),
]

HC_SPLIT_SINKHORN_SHAPES = [
    # name, hc, sinkhorn_iters, eps
    ("hc4_i20", 4, 20, 1.0e-6),
]


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(N, block_size=128, in_dtype=BF16, out_dtype=FP8, scale_dtype=FE8M0):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    blk_m = 32
    group_size = block_size
    compute_dtype = FP32

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=0):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FE8M0):
    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            scale_c_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                scale_b = T.Cast(FP32, scales_b[bx * block_N // group_size, k])
                for i in T.Parallel(block_M):
                    scale_c_shared[i] = T.Cast(FP32, scales_a[by * block_M + i, k]) * scale_b

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_c_shared[i]
                T.clear(C_local)
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


@tilelang.jit(pass_configs=pass_configs)
def fp4_quant_kernel(N, block_size=32, in_dtype=BF16, scale_dtype=FE8M0, inplace=True):
    M = T.symbolic("M")
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    blk_m = 32
    group_size = block_size
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else FP4

    @T.prim_func
    def fp4_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(
                                compute_dtype,
                                T.Cast(
                                    FP4,
                                    T.clamp(x_local[i, j] / s_local[i], -fp4_max, fp4_max),
                                ),
                            )
                            * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], -fp4_max, fp4_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return fp4_quant_kernel_


@tilelang.jit(pass_configs=pass_configs)
def fp4_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FE8M0):
    M = T.symbolic("M")
    act_group_size = 128
    weight_group_size = 32
    block_M = 32
    block_N = 128
    block_K = 32
    n_sub = act_group_size // block_K

    @T.prim_func
    def fp4_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP4],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, act_group_size)), scale_dtype],
        scales_b: T.Tensor[(N, T.ceildiv(K, weight_group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_fp4_shared)
                for i, j in T.Parallel(block_N, block_K):
                    B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

                for i in T.Parallel(block_N):
                    scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])

                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp4_gemm_kernel_


@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    threads = 256
    block = 64
    num_blocks = tilelang.cdiv(topk, block)

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            o_shared = T.alloc_shared((h, d), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            T.fill(scores_max, -T.infinity(FP32))
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    idxs[i] = T.if_then_else(
                        t * block + i < topk,
                        topk_idxs[by, bx, t * block + i],
                        -1,
                    )
                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] *= scale
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i in T.Parallel(h):
                sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
            for i, j in T.Parallel(h, d):
                acc_o[i, j] /= sum_exp[i]
            T.copy(acc_o, o_shared)
            T.copy(o_shared, o[by, bx, :, :])

    return sparse_attn_kernel_


@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), FP32],
        hc_base: T.Tensor[(mix_hc,), FP32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = (
                    mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                    + hc_base[j * hc + k + hc * 2]
                )

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_


def renamed_kernel_source(program, old_name: str, new_name: str) -> str:
    source = program.get_kernel_source()
    if old_name not in source:
        raise RuntimeError(f"TileLang kernel source did not contain {old_name}")
    return source.replace(old_name, new_name)


def grouped_fp4_gemm_kernel_source(out_dim: int, in_dim: int, kernel_name: str) -> str:
    source = renamed_kernel_source(
        fp4_gemm_kernel(out_dim, in_dim),
        "fp4_gemm_kernel__kernel",
        kernel_name,
    )
    old_args = (
        "const fp8_e4_t* __restrict__ A, const fp4_e2_t* __restrict__ B, "
        "bfloat16_t* __restrict__ C, const fp8_e8_t* __restrict__ scales_a, "
        "const fp8_e8_t* __restrict__ scales_b, int M"
    )
    new_args = (
        "const fp8_e4_t* __restrict__ A, const fp4_e2_t* const* __restrict__ B_ptrs, "
        "bfloat16_t* __restrict__ C, const fp8_e8_t* __restrict__ scales_a, "
        "const fp8_e8_t* const* __restrict__ scales_b_ptrs, "
        "const int* __restrict__ expert_indptr, int M"
    )
    if old_args not in source:
        raise RuntimeError("TileLang FP4 GEMM signature changed; update grouped transform")
    source = source.replace(old_args, new_args)
    setup = f"""  const int expert = static_cast<int>(blockIdx.z);
  const int expert_start = expert_indptr[expert];
  const int expert_m = expert_indptr[expert + 1] - expert_start;
  if (static_cast<int>(blockIdx.y) * 32 >= expert_m) {{
    return;
  }}
  const fp8_e4_t* __restrict__ A_group = A + static_cast<int64_t>(expert_start) * {in_dim};
  bfloat16_t* __restrict__ C_group = C + static_cast<int64_t>(expert_start) * {out_dim};
  const fp8_e8_t* __restrict__ scales_a_group =
      scales_a + static_cast<int64_t>(expert_start) * {in_dim // 128};
  const fp4_e2_t* __restrict__ B = B_ptrs[expert];
  const fp8_e8_t* __restrict__ scales_b = scales_b_ptrs[expert];
"""
    marker = "  const dim3 blockIdx = tl::rasterization2DRow<10>();\n"
    if marker not in source:
        raise RuntimeError("TileLang FP4 GEMM blockIdx marker changed; update grouped transform")
    source = source.replace(marker, marker + setup)
    source = source.replace("A[", "A_group[")
    source = source.replace("scales_a[", "scales_a_group[")
    source = source.replace("C + (", "C_group + (")
    source = source.replace(" < M", " < expert_m")
    return source


def grouped_fp4_w13_gemm_kernel_source(inter_dim: int, in_dim: int, kernel_name: str) -> str:
    source = renamed_kernel_source(
        fp4_gemm_kernel(inter_dim, in_dim),
        "fp4_gemm_kernel__kernel",
        kernel_name,
    )
    old_args = (
        "const fp8_e4_t* __restrict__ A, const fp4_e2_t* __restrict__ B, "
        "bfloat16_t* __restrict__ C, const fp8_e8_t* __restrict__ scales_a, "
        "const fp8_e8_t* __restrict__ scales_b, int M"
    )
    new_args = (
        "const fp8_e4_t* __restrict__ A, "
        "const fp4_e2_t* const* __restrict__ W1_ptrs, "
        "const fp4_e2_t* const* __restrict__ W3_ptrs, "
        "bfloat16_t* __restrict__ gate_out, "
        "bfloat16_t* __restrict__ up_out, "
        "const fp8_e8_t* __restrict__ scales_a, "
        "const fp8_e8_t* const* __restrict__ W1_scales_ptrs, "
        "const fp8_e8_t* const* __restrict__ W3_scales_ptrs, "
        "const int* __restrict__ expert_indptr, int M"
    )
    if old_args not in source:
        raise RuntimeError("TileLang FP4 GEMM signature changed; update W13 grouped transform")
    source = source.replace(old_args, new_args)
    source = source.replace("((int)blockIdx.x)", "local_block_x")
    setup = f"""  const int expert = static_cast<int>(blockIdx.z);
  const int global_block_x = static_cast<int>(blockIdx.x);
  const bool is_w3 = global_block_x >= {inter_dim // 128};
  const int local_block_x = is_w3 ? global_block_x - {inter_dim // 128} : global_block_x;
  const int expert_start = expert_indptr[expert];
  const int expert_m = expert_indptr[expert + 1] - expert_start;
  if (static_cast<int>(blockIdx.y) * 32 >= expert_m) {{
    return;
  }}
  const fp8_e4_t* __restrict__ A_group = A + static_cast<int64_t>(expert_start) * {in_dim};
  bfloat16_t* __restrict__ C_group =
      (is_w3 ? up_out : gate_out) + static_cast<int64_t>(expert_start) * {inter_dim};
  const fp8_e8_t* __restrict__ scales_a_group =
      scales_a + static_cast<int64_t>(expert_start) * {in_dim // 128};
  const fp4_e2_t* __restrict__ B = is_w3 ? W3_ptrs[expert] : W1_ptrs[expert];
  const fp8_e8_t* __restrict__ scales_b =
      is_w3 ? W3_scales_ptrs[expert] : W1_scales_ptrs[expert];
"""
    marker = "  const dim3 blockIdx = tl::rasterization2DRow<10>();\n"
    if marker not in source:
        raise RuntimeError("TileLang FP4 GEMM blockIdx marker changed; update W13 grouped transform")
    source = source.replace(marker, marker + setup)
    source = source.replace("A[", "A_group[")
    source = source.replace("scales_a[", "scales_a_group[")
    source = source.replace("C + (", "C_group + (")
    source = source.replace(" < M", " < expert_m")
    return source


def fp8_w13_gemm_kernel_source(inter_dim: int, in_dim: int, kernel_name: str) -> str:
    source = renamed_kernel_source(
        fp8_gemm_kernel(inter_dim, in_dim),
        "fp8_gemm_kernel__kernel",
        kernel_name,
    )
    old_args = (
        "const fp8_e4_t* __restrict__ A, const fp8_e4_t* __restrict__ B, "
        "bfloat16_t* __restrict__ C, const fp8_e8_t* __restrict__ scales_a, "
        "const fp8_e8_t* __restrict__ scales_b, int M"
    )
    new_args = (
        "const fp8_e4_t* __restrict__ A, "
        "const fp8_e4_t* __restrict__ W1, "
        "const fp8_e4_t* __restrict__ W3, "
        "bfloat16_t* __restrict__ gate_out, "
        "bfloat16_t* __restrict__ up_out, "
        "const fp8_e8_t* __restrict__ scales_a, "
        "const fp8_e8_t* __restrict__ W1_scales, "
        "const fp8_e8_t* __restrict__ W3_scales, int M"
    )
    if old_args not in source:
        raise RuntimeError("TileLang FP8 GEMM signature changed; update W13 transform")
    source = source.replace(old_args, new_args)
    source = source.replace("((int)blockIdx.x)", "local_block_x")
    setup = f"""  const int global_block_x = static_cast<int>(blockIdx.x);
  const bool is_w3 = global_block_x >= {inter_dim // 128};
  const int local_block_x = is_w3 ? global_block_x - {inter_dim // 128} : global_block_x;
  const fp8_e4_t* __restrict__ B = is_w3 ? W3 : W1;
  bfloat16_t* __restrict__ C = is_w3 ? up_out : gate_out;
  const fp8_e8_t* __restrict__ scales_b = is_w3 ? W3_scales : W1_scales;
"""
    marker = "  const dim3 blockIdx = tl::rasterization2DRow<10>();\n"
    if marker not in source:
        raise RuntimeError("TileLang FP8 GEMM blockIdx marker changed; update W13 transform")
    source = source.replace(marker, marker + setup)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in out_dir.glob("deepseek_v4_tilelang_*.cu"):
        stale_path.unlink()
    out_path = out_dir / "deepseek_v4_tilelang_fp8_linear.cu"

    sources = []
    launchers = []
    generated_act_kernels = set()

    for _, _out_dim, in_dim in FP8_LINEAR_SHAPES:
        if in_dim in generated_act_kernels:
            continue
        generated_act_kernels.add(in_dim)
        act_name = f"deepseek_tilelang_act_quant_k{in_dim}_kernel"
        sources.append(
            renamed_kernel_source(
                act_quant_kernel(in_dim),
                "act_quant_kernel__kernel",
                act_name,
            )
        )
        launchers.append(
            f"""
extern "C" int deepseek_tilelang_act_quant_k{in_dim}(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream) {{
  constexpr int kThreads = 128;
  constexpr int kSharedBytes = 8192;
  dim3 grid((m + 31) / 32, {in_dim // 128}, 1);
  {act_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<fp8_e8_t*>(scales),
      reinterpret_cast<const bfloat16_t*>(x),
      reinterpret_cast<fp8_e4_t*>(y),
      m);
  return static_cast<int>(cudaGetLastError());
}}
"""
        )

    for _shape_name, out_dim, in_dim in FP8_LINEAR_SHAPES:
        gemm_name = f"deepseek_tilelang_fp8_gemm_n{out_dim}_k{in_dim}_kernel"
        sources.append(
            renamed_kernel_source(
                fp8_gemm_kernel(out_dim, in_dim),
                "fp8_gemm_kernel__kernel",
                gemm_name,
            )
        )
        launchers.append(
            f"""
extern "C" int deepseek_tilelang_fp8_gemm_n{out_dim}_k{in_dim}(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream) {{
  constexpr int kThreads = 128;
  constexpr int kSharedBytes = 98304;
  cudaError_t err = cudaFuncSetAttribute(
      {gemm_name},
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue) {{
    return static_cast<int>(err);
  }}
  dim3 grid({out_dim // 128}, (m + 31) / 32, 1);
  {gemm_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<const fp8_e4_t*>(a),
      reinterpret_cast<const fp8_e4_t*>(b),
      reinterpret_cast<bfloat16_t*>(c),
      reinterpret_cast<const fp8_e8_t*>(scales_a),
      reinterpret_cast<const fp8_e8_t*>(scales_b),
      m);
  return static_cast<int>(cudaGetLastError());
}}
"""
        )

        if out_dim == 2048 and in_dim == 4096:
            w13_name = f"deepseek_tilelang_fp8_w13_gemm_n{out_dim}_k{in_dim}_kernel"
            sources.append(fp8_w13_gemm_kernel_source(out_dim, in_dim, w13_name))
            launchers.append(
                f"""
extern "C" int deepseek_tilelang_fp8_w13_gemm_n{out_dim}_k{in_dim}(
    const void* a,
    const void* w1,
    const void* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* scales_w1,
    const void* scales_w3,
    int m,
    cudaStream_t stream) {{
  constexpr int kThreads = 128;
  constexpr int kSharedBytes = 98304;
  cudaError_t err = cudaFuncSetAttribute(
      {w13_name},
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue && err != cudaErrorNotSupported) {{
    return static_cast<int>(err);
  }}
  dim3 grid({(out_dim // 128) * 2}, (m + 31) / 32, 1);
  {w13_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<const fp8_e4_t*>(a),
      reinterpret_cast<const fp8_e4_t*>(w1),
      reinterpret_cast<const fp8_e4_t*>(w3),
      reinterpret_cast<bfloat16_t*>(gate_out),
      reinterpret_cast<bfloat16_t*>(up_out),
      reinterpret_cast<const fp8_e8_t*>(scales_a),
      reinterpret_cast<const fp8_e8_t*>(scales_w1),
      reinterpret_cast<const fp8_e8_t*>(scales_w3),
      m);
  return static_cast<int>(cudaGetLastError());
}}
"""
            )

    for _shape_name, out_dim, in_dim in FP4_LINEAR_SHAPES:
        gemm_name = f"deepseek_tilelang_fp4_gemm_n{out_dim}_k{in_dim}_kernel"
        sources.append(
            renamed_kernel_source(
                fp4_gemm_kernel(out_dim, in_dim),
                "fp4_gemm_kernel__kernel",
                gemm_name,
            )
        )
        launchers.append(
            f"""
extern "C" int deepseek_tilelang_fp4_gemm_n{out_dim}_k{in_dim}(
    const void* a,
    const void* b,
    void* c,
    const void* scales_a,
    const void* scales_b,
    int m,
    cudaStream_t stream) {{
  constexpr int kThreads = 128;
  constexpr int kSharedBytes = 98304;
  cudaError_t err = cudaFuncSetAttribute(
      {gemm_name},
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue) {{
    return static_cast<int>(err);
  }}
  dim3 grid({out_dim // 128}, (m + 31) / 32, 1);
  {gemm_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<const fp8_e4_t*>(a),
      reinterpret_cast<const fp4_e2_t*>(b),
      reinterpret_cast<bfloat16_t*>(c),
      reinterpret_cast<const fp8_e8_t*>(scales_a),
      reinterpret_cast<const fp8_e8_t*>(scales_b),
      m);
  return static_cast<int>(cudaGetLastError());
}}
"""
        )

        grouped_name = f"deepseek_tilelang_fp4_grouped_gemm_n{out_dim}_k{in_dim}_kernel"
        sources.append(grouped_fp4_gemm_kernel_source(out_dim, in_dim, grouped_name))
        launchers.append(
            f"""
extern "C" int deepseek_tilelang_fp4_grouped_gemm_n{out_dim}_k{in_dim}(
    const void* a,
    const void* const* b,
    void* c,
    const void* scales_a,
    const void* const* scales_b,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream) {{
  constexpr int kThreads = 128;
  constexpr int kSharedBytes = {FP4_GROUPED_GEMM_SHARED_BYTES};
  cudaError_t err = cudaFuncSetAttribute(
      {grouped_name},
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue) {{
    return static_cast<int>(err);
  }}
  dim3 grid({out_dim // 128}, (m + 31) / 32, local_experts);
  {grouped_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<const fp8_e4_t*>(a),
      reinterpret_cast<const fp4_e2_t* const*>(b),
      reinterpret_cast<bfloat16_t*>(c),
      reinterpret_cast<const fp8_e8_t*>(scales_a),
      reinterpret_cast<const fp8_e8_t* const*>(scales_b),
      expert_indptr,
      m);
  return static_cast<int>(cudaGetLastError());
}}
"""
        )

        if out_dim == 2048 and in_dim == 4096:
            w13_name = f"deepseek_tilelang_fp4_grouped_w13_gemm_n{out_dim}_k{in_dim}_kernel"
            sources.append(grouped_fp4_w13_gemm_kernel_source(out_dim, in_dim, w13_name))
            launchers.append(
                f"""
extern "C" int deepseek_tilelang_fp4_grouped_w13_gemm_n{out_dim}_k{in_dim}(
    const void* a,
    const void* const* w1,
    const void* const* w3,
    void* gate_out,
    void* up_out,
    const void* scales_a,
    const void* const* scales_w1,
    const void* const* scales_w3,
    const int* expert_indptr,
    int m,
    int local_experts,
    cudaStream_t stream) {{
  constexpr int kThreads = 128;
  constexpr int kSharedBytes = {FP4_GROUPED_GEMM_SHARED_BYTES};
  cudaError_t err = cudaFuncSetAttribute(
      {w13_name},
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue && err != cudaErrorNotSupported) {{
    return static_cast<int>(err);
  }}
  dim3 grid({(out_dim // 128) * 2}, (m + 31) / 32, local_experts);
  {w13_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<const fp8_e4_t*>(a),
      reinterpret_cast<const fp4_e2_t* const*>(w1),
      reinterpret_cast<const fp4_e2_t* const*>(w3),
      reinterpret_cast<bfloat16_t*>(gate_out),
      reinterpret_cast<bfloat16_t*>(up_out),
      reinterpret_cast<const fp8_e8_t*>(scales_a),
      reinterpret_cast<const fp8_e8_t* const*>(scales_w1),
      reinterpret_cast<const fp8_e8_t* const*>(scales_w3),
      expert_indptr,
      m);
  return static_cast<int>(cudaGetLastError());
}}
"""
            )

    for shape_name, hidden_dim, block_size in FP4_QUANT_INPLACE_SHAPES:
        kernel_name = f"deepseek_tilelang_fp4_quant_inplace_{shape_name}_kernel"
        sources.append(
            renamed_kernel_source(
                fp4_quant_kernel(hidden_dim, block_size, inplace=True),
                "fp4_quant_kernel__kernel",
                kernel_name,
            )
        )
        launchers.append(
            f"""
extern "C" int deepseek_tilelang_fp4_quant_inplace_{shape_name}(
    const void* x,
    void* y,
    void* scales,
    int m,
    cudaStream_t stream) {{
  constexpr int kThreads = 128;
  constexpr int kSharedBytes = 8192;
  dim3 grid((m + 31) / 32, {hidden_dim // block_size}, 1);
  {kernel_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<fp8_e8_t*>(scales),
      reinterpret_cast<const bfloat16_t*>(x),
      reinterpret_cast<bfloat16_t*>(y),
      m);
  return static_cast<int>(cudaGetLastError());
}}
"""
        )

    for shape_name, heads, head_dim, softmax_scale in SPARSE_ATTN_SHAPES:
        kernel_name = f"deepseek_tilelang_sparse_attn_{shape_name}_kernel"
        shared_bytes = align_up(
            (heads * head_dim + 64 * head_dim + heads * head_dim + heads * 64) * 2,
            1024,
        )
        sources.append(
            renamed_kernel_source(
                sparse_attn_kernel(heads, head_dim, softmax_scale),
                "sparse_attn_kernel__kernel",
                kernel_name,
            )
        )
        launchers.append(
            f"""
extern "C" int deepseek_tilelang_sparse_attn_{shape_name}(
    const void* q,
    const void* kv,
    const void* attn_sink,
    const int* topk_idxs,
    void* out,
    int m,
    int n,
    int topk,
    cudaStream_t stream) {{
  constexpr int kThreads = 256;
  constexpr int kSharedBytes = {shared_bytes};
  cudaError_t err = cudaFuncSetAttribute(
      {kernel_name},
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  if (err != cudaSuccess && err != cudaErrorInvalidValue) {{
    return static_cast<int>(err);
  }}
  dim3 grid(m, 1, 1);
  {kernel_name}<<<grid, kThreads, kSharedBytes, stream>>>(
      reinterpret_cast<const float*>(attn_sink),
      reinterpret_cast<const bfloat16_t*>(kv),
      reinterpret_cast<bfloat16_t*>(out),
      reinterpret_cast<const bfloat16_t*>(q),
      topk_idxs,
      1,
      m,
      n,
      topk);
  return static_cast<int>(cudaGetLastError());
}}
"""
        )

    for shape_name, hc, sinkhorn_iters, eps in HC_SPLIT_SINKHORN_SHAPES:
        kernel_name = f"deepseek_tilelang_hc_split_sinkhorn_{shape_name}_kernel"
        sources.append(
            renamed_kernel_source(
                hc_split_sinkhorn_kernel(hc, sinkhorn_iters, eps),
                "hc_split_sinkhorn_kernel__kernel",
                kernel_name,
            )
        )
        launchers.append(
            f"""
extern "C" int deepseek_tilelang_hc_split_sinkhorn_{shape_name}(
    const float* mixes,
    const float* hc_scale,
    const float* hc_base,
    float* pre,
    float* post,
    float* comb,
    int n,
    cudaStream_t stream) {{
  constexpr int kThreads = 64;
  constexpr int kSharedBytes = 1024;
  {kernel_name}<<<n, kThreads, kSharedBytes, stream>>>(
      comb,
      hc_base,
      hc_scale,
      mixes,
      post,
      pre,
      n);
  return static_cast<int>(cudaGetLastError());
}}
"""
        )

    out_path.write_text(
        "// Generated by openinfer-kernels/tools/tilelang/deepseek_v4/generate.py\n"
        "#include <cuda_runtime.h>\n"
        "#include <tl_templates/cuda/cuda_fp4.h>\n"
        "\n"
        "#if !defined(__CUDA_ARCH__)\n"
        "struct fp4_e2_t { unsigned char __x; };\n"
        "struct fp4_e2_2_t { unsigned char __x; };\n"
        "struct alignas(2) fp4_e2_4_t { unsigned short __x; };\n"
        "struct alignas(8) fp4_e2_16_t { unsigned long long __x; };\n"
        "static inline float2 __tl_cvt_fp4x2_to_float2(unsigned char) { return make_float2(0.0f, 0.0f); }\n"
        "static inline unsigned char __tl_cvt_float2_to_fp4x2(float2) { return 0; }\n"
        "#endif\n"
        "\n"
        + "\n".join(sources)
        + "\n"
        + "\n".join(launchers)
        + "\n"
    )
    print(f"CU_PATH={out_path}")
    print(f"TILELANG_TEMPLATE_PATH={TILELANG_TEMPLATE_PATH}")
    print(f"CUTLASS_INCLUDE_DIR={CUTLASS_INCLUDE_DIR}")


if __name__ == "__main__":
    main()
