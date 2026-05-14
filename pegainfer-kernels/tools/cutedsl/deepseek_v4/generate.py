#!/usr/bin/env python3
"""Generate DeepSeek V4 CuTe DSL AOT artifacts.

The first production CuTe DSL target is the indexer score dot product:

    dots[(token * local_heads + head), compressed] =
        dot(q[token, head, :], kv[compressed, :])

The original CUDA entry points keep their C ABI and run the small ReLU/weight
epilogue in CUDA. This generator exports the TensorCore GEMM used for the dot
stage and keeps the dot output in FP32 so the following top-k path sees the same
precision class as the retired serial indexer-score kernel.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor

LOCAL_HEADS = 8
HEAD_DIM = 128


def load_sm120_gemm_class(cutlass_root: Path):
    gemm_path = find_sm120_dense_gemm_path(cutlass_root)
    spec = importlib.util.spec_from_file_location("pegainfer_cutedsl_sm120_dense_gemm", gemm_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Sm120GemmKernel


def find_sm120_dense_gemm_path(cutlass_root: Path) -> Path:
    candidates = [
        cutlass_root
        / "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py",
        cutlass_root / "examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py",
    ]
    for gemm_path in candidates:
        if gemm_path.exists():
            return gemm_path
    gemm_path = candidates[0]
    if not gemm_path.exists():
        raise FileNotFoundError(f"SM120 CuTe DSL dense_gemm.py not found: {gemm_path}")


def find_cutlass_root(repo_root: Path, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            repo_root / "../cutlass-upstream",
            repo_root / "cutlass-upstream",
            repo_root / "pegainfer-kernels/third_party/flashinfer/3rdparty/cutlass",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if any(
            path.exists()
            for path in (
                candidate
                / "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py",
                candidate / "examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py",
            )
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find CUTLASS CuTe DSL examples. Set PEGAINFER_CUTEDSL_CUTLASS_ROOT."
    )


def write_wrapper(out_dir: Path) -> Path:
    wrapper = out_dir / "deepseek_v4_cutedsl_wrappers.cu"
    wrapper.write_text(
        r'''
#include "deepseek_indexer_dots_gemm.h"
#include "deepseek_indexer_scores_exact.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <mutex>

namespace {

constexpr int kMaxIndexerDotsWarmupShapes = 64;

struct IndexerDotsWarmupShape {
  int rows = 0;
  int compressed_len = 0;
  bool initialized = false;
};

idx_gemm_Kernel_Module_t g_indexer_dots_module;
idx_scores_Kernel_Module_t g_indexer_scores_module;
std::once_flag g_indexer_dots_once;
std::once_flag g_indexer_scores_once;
std::mutex g_indexer_dots_launch_mutex;
IndexerDotsWarmupShape g_indexer_dots_warmup_shapes[kMaxIndexerDotsWarmupShapes];

void load_indexer_dots_module_once() {
  idx_gemm_Kernel_Module_Load(&g_indexer_dots_module);
}

void load_indexer_scores_module_once() {
  idx_scores_Kernel_Module_Load(&g_indexer_scores_module);
}

bool mark_indexer_dots_shape_for_warmup(int rows, int compressed_len) {
  for (int i = 0; i < kMaxIndexerDotsWarmupShapes; ++i) {
    IndexerDotsWarmupShape &shape = g_indexer_dots_warmup_shapes[i];
    if (shape.initialized && shape.rows == rows && shape.compressed_len == compressed_len) {
      return false;
    }
  }
  for (int i = 0; i < kMaxIndexerDotsWarmupShapes; ++i) {
    IndexerDotsWarmupShape &shape = g_indexer_dots_warmup_shapes[i];
    if (!shape.initialized) {
      shape.rows = rows;
      shape.compressed_len = compressed_len;
      shape.initialized = true;
      return true;
    }
  }
  return false;
}

}  // namespace

extern "C" cudaError_t deepseek_cutedsl_indexer_dots_bf16_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv,
    float *dots,
    int rows,
    int compressed_len,
    cudaStream_t stream) {
  if (q == nullptr || kv == nullptr || dots == nullptr || rows <= 0 || compressed_len <= 0) {
    return cudaErrorInvalidValue;
  }

  std::call_once(g_indexer_dots_once, load_indexer_dots_module_once);

  idx_gemm_Tensor_a_t a{};
  a.data = const_cast<__nv_bfloat16 *>(q);
  a.dynamic_shapes[0] = rows;
  a.dynamic_strides[0] = 128;

  idx_gemm_Tensor_b_t b{};
  b.data = const_cast<__nv_bfloat16 *>(kv);
  b.dynamic_shapes[0] = compressed_len;
  b.dynamic_strides[0] = 128;

  idx_gemm_Tensor_c_t c{};
  c.data = dots;
  c.dynamic_shapes[0] = rows;
  c.dynamic_shapes[1] = compressed_len;
  c.dynamic_strides[0] = compressed_len;
  c.dynamic_strides[1] = 1;

  int32_t ret = 0;
  {
    std::lock_guard<std::mutex> guard(g_indexer_dots_launch_mutex);
    if (mark_indexer_dots_shape_for_warmup(rows, compressed_len)) {
      ret = cute_dsl_idx_gemm_wrapper(&g_indexer_dots_module, &a, &b, &c, stream);
      if (ret != 0) {
        return cudaErrorUnknown;
      }
      cudaError_t warmup_error = cudaStreamSynchronize(stream);
      if (warmup_error != cudaSuccess) {
        return warmup_error;
      }
    }
    ret = cute_dsl_idx_gemm_wrapper(&g_indexer_dots_module, &a, &b, &c, stream);
  }
  if (ret != 0) {
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t deepseek_cutedsl_indexer_scores_exact_bf16_cuda(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *kv,
    const __nv_bfloat16 *weights,
    float *scores,
    int seq_len,
    int compressed_len,
    float score_scale,
    cudaStream_t stream) {
  if (q == nullptr || kv == nullptr || weights == nullptr || scores == nullptr ||
      seq_len <= 0 || compressed_len <= 0) {
    return cudaErrorInvalidValue;
  }

  std::call_once(g_indexer_scores_once, load_indexer_scores_module_once);

  idx_scores_Tensor_q_t q_arg{};
  q_arg.data = const_cast<__nv_bfloat16 *>(q);
  q_arg.dynamic_shapes[0] = seq_len;

  idx_scores_Tensor_kv_t kv_arg{};
  kv_arg.data = const_cast<__nv_bfloat16 *>(kv);
  kv_arg.dynamic_shapes[0] = compressed_len;

  idx_scores_Tensor_weights_t weights_arg{};
  weights_arg.data = const_cast<__nv_bfloat16 *>(weights);
  weights_arg.dynamic_shapes[0] = seq_len;

  idx_scores_Tensor_scores_t scores_arg{};
  scores_arg.data = scores;
  scores_arg.dynamic_shapes[0] = seq_len;
  scores_arg.dynamic_shapes[1] = compressed_len;
  scores_arg.dynamic_strides[0] = compressed_len;

  int32_t ret = cute_dsl_idx_scores_wrapper(
      &g_indexer_scores_module, &q_arg, &kv_arg, &weights_arg, &scores_arg,
      score_scale, stream);
  if (ret != 0) {
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}
'''.lstrip()
    )
    return wrapper


@cute.kernel
def indexer_scores_exact_kernel(
    q: cute.Tensor,
    kv: cute.Tensor,
    weights: cute.Tensor,
    scores: cute.Tensor,
    score_scale: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    compressed_len = scores.shape[1]
    idx = bidx * 128 + tidx
    total = scores.shape[0] * compressed_len
    if idx < total:
        token = idx // compressed_len
        compressed = idx - token * compressed_len
        acc = cutlass.Float32(0.0)
        for head in cutlass.range(LOCAL_HEADS, unroll_full=True):
            dot = cutlass.Float32(0.0)
            for dim in cutlass.range(HEAD_DIM, unroll_full=True):
                dot += cutlass.Float32(q[token, head, dim]) * cutlass.Float32(kv[compressed, dim])
            if dot > cutlass.Float32(0.0):
                acc += dot * cutlass.Float32(weights[token, head])
        scores[token, compressed] = acc * score_scale


@cute.jit
def indexer_scores_exact(
    q: cute.Tensor,
    kv: cute.Tensor,
    weights: cute.Tensor,
    scores: cute.Tensor,
    score_scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    total = scores.shape[0] * scores.shape[1]
    blocks = (total + 127) // 128
    indexer_scores_exact_kernel(q, kv, weights, scores, score_scale).launch(
        grid=(blocks, 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cutlass-root")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(args.repo_root).resolve()
    cutlass_root = find_cutlass_root(repo_root, args.cutlass_root)
    Sm120GemmKernel = load_sm120_gemm_class(cutlass_root)

    rows = cute.SymInt(divisibility=8)
    seq_len = cute.SymInt(divisibility=1)
    compressed_len = cute.SymInt(divisibility=8)
    scores_compressed_len = cute.SymInt(divisibility=1)
    q = make_fake_compact_tensor(
        cutlass.BFloat16,
        (rows, 128, 1),
        stride_order=(1, 0, 2),
    )
    kv = make_fake_compact_tensor(
        cutlass.BFloat16,
        (compressed_len, 128, 1),
        stride_order=(1, 0, 2),
    )
    dots = make_fake_compact_tensor(
        cutlass.Float32,
        (rows, compressed_len, 1),
        stride_order=(1, 0, 2),
    )
    stream = cuda.CUstream(0)

    kernel = Sm120GemmKernel(cutlass.Float32, (128, 128, 64))
    compiled = cute.compile(kernel, q, kv, dots, 1, stream)
    compiled.export_to_c(
        file_path=str(out_dir),
        file_name="deepseek_indexer_dots_gemm",
        function_prefix="idx_gemm",
    )
    q_scores = make_fake_compact_tensor(
        cutlass.BFloat16,
        (seq_len, LOCAL_HEADS, HEAD_DIM),
        stride_order=(2, 1, 0),
    )
    kv_scores = make_fake_compact_tensor(
        cutlass.BFloat16,
        (scores_compressed_len, HEAD_DIM),
        stride_order=(1, 0),
    )
    weights_scores = make_fake_compact_tensor(
        cutlass.BFloat16,
        (seq_len, LOCAL_HEADS),
        stride_order=(1, 0),
    )
    scores = make_fake_compact_tensor(
        cutlass.Float32,
        (seq_len, scores_compressed_len),
        stride_order=(1, 0),
    )
    score_scale = cutlass.Float32(0.125)
    compiled_scores = cute.compile(
        indexer_scores_exact, q_scores, kv_scores, weights_scores, scores, score_scale, stream
    )
    compiled_scores.export_to_c(
        file_path=str(out_dir),
        file_name="deepseek_indexer_scores_exact",
        function_prefix="idx_scores",
    )
    wrapper = write_wrapper(out_dir)

    runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=False)
    runtime_dirs = sorted({str(Path(path).resolve().parent) for path in runtime_libs})
    print(f"OBJ_PATH={out_dir / 'deepseek_indexer_dots_gemm.o'}")
    print(f"OBJ_PATH={out_dir / 'deepseek_indexer_scores_exact.o'}")
    print(f"HEADER_DIR={out_dir}")
    print(f"WRAPPER_PATH={wrapper}")
    for runtime_dir in runtime_dirs:
        print(f"RUNTIME_LIB_DIR={runtime_dir}")
    print(f"CUTLASS_ROOT={cutlass_root}")


if __name__ == "__main__":
    main()
