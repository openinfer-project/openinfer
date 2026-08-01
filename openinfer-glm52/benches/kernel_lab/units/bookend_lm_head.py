"""Adapter for bookend.lm_head — the full-vocabulary bf16 logits GEMM.

Production symbol: `gemm_strided_batched_bf16_cuda` (csrc/shared/linear.cu:369;
FFI mirror openinfer-kernels/src/ffi/shared.rs:218) — one
cublasGemmStridedBatchedEx on the workspace-free graph-safe handle, exactly the
call glm52_lm_head_into makes (openinfer-glm52/src/bookend.rs:63):
op_a=T/op_b=N, m=vocab, n=rows, k=hidden, A=lm_head (lda=hidden, stride 0),
B=normed (ldb=hidden, stride 0), C=logits (ldc=vocab, stride 0), batch_count=1.
The col-major C[vocab, rows] is the compact row-major [rows, vocab] layout
argmax_split consumes. This is the weight-BW floor unit: every launch streams
the 1.9 GB matrix once.

The cuBLAS handle lives in the .so's per-dlopen global (`g_cublas_handle`) and
starts null — the adapter runs the exported `cublas_init()` once per loaded
library (compare --baseline-so loads a second handle with its own global).
"""
from __future__ import annotations

import ctypes

from kernel_lab import data
from kernel_lab.loader import KernelLaunchError, as_dev_ptr, require_torch, resolve
from kernel_lab.refs import bookends

SYMBOL = "gemm_strided_batched_bf16_cuda"
CUBLAS_INIT_SYMBOL = "cublas_init"

# id(lib) of every .so whose cuBLAS handles this process already initialized.
_CUBLAS_READY: set[int] = set()


def _ensure_cublas(lib) -> None:
    if id(lib) in _CUBLAS_READY:
        return
    init = resolve(lib, CUBLAS_INIT_SYMBOL, [])
    init.restype = None  # void C return
    init()
    _CUBLAS_READY.add(id(lib))


def make_inputs(shape: dict, seed: int) -> dict:
    """shape = {"rows", "n"=vocab, "k"=hidden}; the 1.9 GB weight is cached
    process-wide (seeded procedural generation once per command)."""
    torch = require_torch()
    rows, vocab, hidden = shape["rows"], shape["n"], shape["k"]
    weight = bookends.cached_table("lm_head_weight", vocab, hidden, seed)
    return {
        "normed": data.normal_bf16((rows, hidden), seed=data.derive_seed(seed, "act")),
        "weight": weight,  # bf16 [vocab, hidden]
        "out": torch.empty((rows, vocab), dtype=torch.bfloat16, device=weight.device),
    }


def run(lib, tensors: dict, shape: dict, stream) -> None:
    _ensure_cublas(lib)
    fn = resolve(
        lib,
        SYMBOL,
        [
            ctypes.c_int,      # op_a (1 == CUBLAS_OP_T)
            ctypes.c_int,      # op_b (0 == CUBLAS_OP_N)
            ctypes.c_int,      # m = vocab
            ctypes.c_int,      # n = rows
            ctypes.c_int,      # k = hidden
            ctypes.c_void_p,   # A = lm_head bf16
            ctypes.c_int,      # lda = hidden
            ctypes.c_int64,    # stride_a = 0
            ctypes.c_void_p,   # B = normed bf16
            ctypes.c_int,      # ldb = hidden
            ctypes.c_int64,    # stride_b = 0
            ctypes.c_void_p,   # C = logits bf16
            ctypes.c_int,      # ldc = vocab
            ctypes.c_int64,    # stride_c = 0
            ctypes.c_int,      # batch_count = 1
            ctypes.c_void_p,   # stream
        ],
    )
    status = fn(
        1,
        0,
        shape["n"],
        shape["rows"],
        shape["k"],
        as_dev_ptr(tensors["weight"]),
        shape["k"],
        0,
        as_dev_ptr(tensors["normed"]),
        shape["k"],
        0,
        as_dev_ptr(tensors["out"]),
        shape["n"],
        0,
        1,
        stream,
    )
    if status != 0:
        raise KernelLaunchError(SYMBOL, status)


def reference(tensors: dict, shape: dict):
    return bookends.lm_head_ref(tensors["normed"], tensors["weight"])
