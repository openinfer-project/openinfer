#!/usr/bin/env python3
"""Generate the SM120 Qwen3.5 GDN prefill artifact.

The checked-in FlashInfer implementation is a CuTe DSL source, not a C ABI
library.  This script is intentionally a build-time tool: it compiles one
shape-general (dynamic token count) Qwen3.5 configuration and exports the
CuTe launcher plus a small C ABI shim.  Python is therefore not required by
the OpenInfer serving process.

The first integration target is the Qwen3.5-4B linear-attention shape after
OpenInfer's prepare stage: 32 value heads, head dimension 128, bf16 I/O and a
single variable-length sequence.  The existing Triton pipeline remains the
fallback for other shapes and architectures.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


HEADS = 32
HEAD_DIM = 128
TENSOR_MAP_BYTES = 256 * 128
PREFIX = "openinfer_qwen35_gdn_sm120"


def _patch_grid_argument_annotation(kernel_cls, cutlass) -> None:
    """Make the upstream ``grid_x`` launch scalar exportable.

    FlashInfer v0.6.14 annotates ``grid_x`` as a Python ``int``.  CuTe's C
    header exporter only accepts DSL numeric types, even though the value is a
    runtime launch scalar.  Treating it as Int32 preserves the ABI and lets
    the generated wrapper vary the grid for future batched callers.
    """

    call = kernel_cls.__call__
    annotations = getattr(call, "__annotations__", None)
    if annotations is None:
        wrapped = getattr(call, "__wrapped__", None)
        annotations = getattr(wrapped, "__annotations__", None)
    if annotations is None:
        raise RuntimeError("cannot find CuTe GDN kernel annotations")
    annotations["grid_x"] = cutlass.Int32


def _fake_tensor(cute, cutlass, dtype, shape, stride):
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=16,
    )


def _write_c_shim(output_dir: Path, header_name: str) -> None:
    """Write the raw-pointer launcher consumed by Rust.

    The generated CuTe header owns the opaque CUDA module and descriptor
    structs.  This shim only fills those descriptors from OpenInfer's
    contiguous token-major buffers and handles one-time module loading.
    """

    header_prefix = PREFIX
    wrapper_name = f"cute_dsl_{PREFIX}_wrapper"
    text = f'''#include "{header_name}"
#include <cuda_runtime.h>
#include <stdint.h>
#include <pthread.h>

static {header_prefix}_Kernel_Module_t g_module;
static pthread_once_t g_module_once = PTHREAD_ONCE_INIT;

static void load_{header_prefix}(void) {{
    {header_prefix}_Kernel_Module_Load(&g_module);
}}

// q/k/v/o are token-major [T, H, D] buffers.  CuTe consumes the equivalent
// TMA views [T, D, H] with the static strides below.  alpha/beta are [T, H].
// state is [1, H, V, K], matching FlashInfer's final-state contract.
CUresult {header_prefix}_cuda(
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* output,
    const float* alpha,
    const float* beta,
    float* state,
    uint8_t* tensormaps,
    const int64_t* cu_seqlens,
    int32_t seq_len,
    cudaStream_t stream) {{
    int once_err = pthread_once(&g_module_once, load_{header_prefix});
    if (once_err != 0) return CUDA_ERROR_UNKNOWN;

    {header_prefix}_Tensor_g_q_t q_desc = {{(void*)q, {{seq_len}}}};
    {header_prefix}_Tensor_g_k_t k_desc = {{(void*)k, {{seq_len}}}};
    {header_prefix}_Tensor_g_v_t v_desc = {{(void*)v, {{seq_len}}}};
    {header_prefix}_Tensor_g_o_t o_desc = {{(void*)output, {{seq_len}}}};
    {header_prefix}_Tensor_g_alpha_t alpha_desc = {{(void*)alpha, {{seq_len}}}};
    {header_prefix}_Tensor_g_beta_t beta_desc = {{(void*)beta, {{seq_len}}}};
    {header_prefix}_Tensor_g_state_t state_desc = {{(void*)state}};
    {header_prefix}_Tensor_g_tensormaps_t tensormaps_desc = {{(void*)tensormaps}};
    {header_prefix}_Tensor_cu_seqlens_t cu_desc = {{(void*)cu_seqlens}};

    {wrapper_name}(
        &g_module,
        &q_desc,
        &k_desc,
        &v_desc,
        &o_desc,
        &alpha_desc,
        &beta_desc,
        &state_desc,
        &tensormaps_desc,
        &cu_desc,
        1.0f / 11.313708498984761f,
        {HEADS},
        {HEADS},
        {HEADS},
        {HEADS},
        1,
        1,
        0,
        {HEADS},
        stream);
    return (CUresult)cudaGetLastError();
}}
'''
    (output_dir / f"{PREFIX}_wrapper.c").write_text(text, encoding="utf-8")


def generate(output_dir: Path) -> None:
    try:
        import cutlass as _cutlass_module
        import cutlass.cute as cute
        import cutlass.cute.runtime
        import cuda.bindings.driver as cuda_driver
    except ImportError as exc:
        raise RuntimeError(
            "FlashInfer SM120 AOT generation requires nvidia-cutlass-dsl and "
            "cuda-python; install flashinfer-python[cu13] build dependencies"
        ) from exc

    cutlass = _cutlass_module
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
        _FullyFusedDeltaRuleSm120,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _patch_grid_argument_annotation(_FullyFusedDeltaRuleSm120, cutlass)

    token_count = cute.sym_int32(divisibility=64, symbol="total_tokens")
    q_stride = (HEADS * HEAD_DIM, 1, HEAD_DIM)
    alpha_stride = (HEADS, 1)
    state_stride = (HEADS * HEAD_DIM * HEAD_DIM, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)

    q = _fake_tensor(cute, cutlass, cutlass.BFloat16, (token_count, HEAD_DIM, HEADS), q_stride)
    k = _fake_tensor(cute, cutlass, cutlass.BFloat16, (token_count, HEAD_DIM, HEADS), q_stride)
    v = _fake_tensor(cute, cutlass, cutlass.BFloat16, (token_count, HEAD_DIM, HEADS), q_stride)
    output = _fake_tensor(
        cute, cutlass, cutlass.BFloat16, (token_count, HEAD_DIM, HEADS), q_stride
    )
    alpha = _fake_tensor(cute, cutlass, cutlass.Float32, (token_count, HEADS), alpha_stride)
    beta = _fake_tensor(cute, cutlass, cutlass.Float32, (token_count, HEADS), alpha_stride)
    state = _fake_tensor(
        cute, cutlass, cutlass.Float32, (1, HEADS, HEAD_DIM, HEAD_DIM), state_stride
    )
    tensormaps = _fake_tensor(
        cute, cutlass, cutlass.Uint8, (TENSOR_MAP_BYTES,), (1,)
    )
    cu_seqlens = _fake_tensor(cute, cutlass, cutlass.Int64, (2,), (1,))
    stream = cuda_driver.CUstream(0)

    kernel = _FullyFusedDeltaRuleSm120(
        needs_alpha=True,
        needs_beta=True,
        needs_init_state=False,
        needs_checkpointing=False,
        dtype=cutlass.BFloat16,
    )
    args = (
        q,
        k,
        v,
        output,
        alpha,
        beta,
        state,
        None,
        None,
        None,
        tensormaps,
        cu_seqlens,
        cutlass.Float32(1.0 / math.sqrt(HEAD_DIM)),
        cutlass.Int32(HEADS),
        cutlass.Int32(HEADS),
        cutlass.Int32(HEADS),
        cutlass.Int32(HEADS),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(0),
        cutlass.Int32(HEADS),
        stream,
    )

    options = (cutlass.GPUArch("sm_120a"),)
    compiled = cute.compile[options](kernel, *args)
    compiled.export_to_c(str(output_dir), PREFIX, function_prefix=PREFIX)
    _write_c_shim(output_dir, f"{PREFIX}.h")

    runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=False)
    (output_dir / "runtime_libs.txt").write_text(
        "\n".join(str(path) for path in runtime_libs) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
