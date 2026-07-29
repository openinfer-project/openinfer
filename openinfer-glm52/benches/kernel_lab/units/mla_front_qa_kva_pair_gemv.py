"""Adapter for mla_front.qa_kva_pair_gemv — the bs=1 MLA q_a + kv_a paired fp8 GEMV.

Production symbol: `glm52_fp8_weight_only_gemv_pair_cuda`
(csrc/glm52/glm52_moe_gemv.cu; FFI mirror openinfer-kernels/src/ffi/glm52.rs).

The ABI is deliberately the one measured MLA fusion, not a generic paired
surface: there is NO batch parameter, and the launcher hard-guards
n_a == 2048 (q_a), n_b == 576 (kv_a), k == 6144 — anything else fails closed
with CUDA_ERROR_INVALID_VALUE. Production calls it only from the t == 1 arm
of `glm52_mla_front_q_into` (openinfer-glm52/src/mla_front.rs); t > 1 takes
the packed qa_kva batched-mma route (fp8_linear_partials_into +
glm52_gemv_split_reduce_launch) instead. The manifest rows axis is therefore
exactly [1].

The kernel concatenates the two block grids over ONE shared activation row —
weights and outputs stay in their checkpoint layouts and each row keeps its
solo-launch dot order — so the torch reference is the two single-GEMV refs
concatenated, with the kv_a side modeled partial-N (576 % 128 != 0) by
kernel_lab.refs.proj_gemv.

The harness `check` scores `tensors["out"]` against the reference, so this
adapter allocates one [n_a + n_b] bf16 buffer and hands the kernel two views
(out_a = out[:n_a], out_b = out[n_a:]; the 2048-elem offset keeps out_b
16B-aligned). No scratch: the pair ABI has no partials surface.
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import proj_gemv

SYMBOL = "glm52_fp8_weight_only_gemv_pair_cuda"

# ABI-locked shapes, mirroring the .cu launcher's hard guard. kv_a's 576 is
# partial-N: data.normal_quantized_fp8 needs a 128-divisible n, so its weight
# is generated at the padded 640 rows and leading-sliced back to 576 (block
# scales are per-128-row, so the kept rows' quantization is untouched).
N_A = 2048
N_B = 576
N_B_PADDED = 640
K = 6144


def _check_shape(shape: dict) -> None:
    if shape["rows"] != 1 or shape["n"] != N_A + N_B or shape["k"] != K:
        raise ValueError(
            f"qa_kva_pair_gemv is ABI-locked to rows=1 n={N_A + N_B} k={K}, got {shape}"
        )


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows": 1, "n": 2624, "k": 6144} — the only legal point."""
    torch = require_torch()
    _check_shape(shape)
    act = data.normal_bf16((1, K), seed=data.derive_seed(seed, "act"))
    weight_a, scales_a = data.normal_quantized_fp8(
        N_A, K, seed=data.derive_seed(seed, "weight_a")
    )
    weight_b_full, scales_b = data.normal_quantized_fp8(
        N_B_PADDED, K, seed=data.derive_seed(seed, "weight_b")
    )
    weight_b = weight_b_full[:N_B].contiguous()
    out = torch.empty(N_A + N_B, dtype=torch.bfloat16, device=act.device)
    return {
        "act": act,                    # bf16 [1, k]
        "weight_a": weight_a,          # e4m3 uint8 [2048, k]
        "scales_a": scales_a,          # f32 [16, k/128]
        "weight_b": weight_b,          # e4m3 uint8 [576, k]
        "scales_b": scales_b,          # f32 [5, k/128]
        "out": out,                    # bf16 [2624] — what check scores
        "out_a": out[:N_A],            # bf16 [2048] view handed to the kernel
        "out_b": out[N_A:],            # bf16 [576] view handed to the kernel
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    """One production launch on `stream` (c_void_p cudaStream_t)."""
    _check_shape(shape)
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_void_p,   # activation (bf16 as raw bits)
            ctypes.c_void_p,   # weight_a e4m3
            ctypes.c_void_p,   # weight_scale_a f32
            ctypes.c_void_p,   # out_a bf16
            ctypes.c_int,      # n_a
            ctypes.c_void_p,   # weight_b e4m3
            ctypes.c_void_p,   # weight_scale_b f32
            ctypes.c_void_p,   # out_b bf16
            ctypes.c_int,      # n_b
            ctypes.c_int,      # k
            ctypes.c_void_p,   # stream
        ],
    )
    rc = fn(
        as_dev_ptr(tensors["act"]),
        as_dev_ptr(tensors["weight_a"]),
        as_dev_ptr(tensors["scales_a"]),
        as_dev_ptr(tensors["out_a"]),
        N_A,
        as_dev_ptr(tensors["weight_b"]),
        as_dev_ptr(tensors["scales_b"]),
        as_dev_ptr(tensors["out_b"]),
        N_B,
        K,
        stream,
    )
    if rc != 0:
        raise KernelLaunchError(SYMBOL, rc)


def reference(tensors: dict, shape: dict):
    _check_shape(shape)
    return proj_gemv.qa_kva_pair_ref(
        tensors["act"],
        tensors["weight_a"],
        tensors["scales_a"],
        tensors["weight_b"],
        tensors["scales_b"],
        N_A,
        N_B,
        K,
    )
