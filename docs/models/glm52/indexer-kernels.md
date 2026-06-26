# GLM5.2 Indexer Kernels

> **TL;DR:** GLM5.2 indexer cache insert/gather now has source-backed OpenInfer CUDA/Rust wrappers copied from vLLM cache-kernel layout; sparse top-k 2048 has an ABI/workspace contract but the launch intentionally fails closed until the vLLM sampler/persistent/cooperative source slice is fully ported.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes this work under `models/glm52`.
  - `docs/models/glm52/decode-forward-contract.md` - fixed `bs=128`, indexer head dim `128`, top-k `2048`, decode-only scope, and indexer cache/top-k source map.
  - `docs/models/glm52/vllm-kernel-reference.md` - identifies vLLM cache insert/gather and sparse indexer top-k as direct-copy candidates.
  - `../vllm/csrc/libtorch_stable/cache_kernels.cu` - source for `indexer_k_quant_and_cache_kernel` and `cp_gather_indexer_k_quant_cache_kernel`.
  - `../vllm/csrc/libtorch_stable/sampler.cu` - source for `top_k_per_row_decode`.
  - `../vllm/csrc/libtorch_stable/cooperative_topk.cu` and `../vllm/csrc/libtorch_stable/topk.cu` - source for the preferred CUDA top-k paths when `top_k in {512,1024,2048}`.
  - `../vllm/vllm/model_executor/layers/sparse_attn_indexer.py` - decode ordering: cache insert, paged MQA logits, then cooperative/persistent/per-row top-k.
- **Relevant history**:
  - `docs/models/glm52/decode-forward-contract.md` - sparse attention can consume fixture top-k before the full indexer logits/top-k chain exists.
  - `docs/models/glm52/vllm-kernel-reference.md` - handwritten local CUDA needs source justification and measurement before performance claims.
- **Plan**:
  1. Add GLM52 indexer FFI and Rust ops contracts under the allowed `openinfer-kernels/src/{ffi,ops}/glm52/indexer.rs` paths.
  2. Add source-backed CUDA cache insert/gather under `openinfer-kernels/csrc/glm52/glm52_indexer.cu`.
  3. Expose top-k 2048 workspace/launch ABI, but fail closed until the full vLLM top-k source slice is copied.
  4. Run formatting and the narrow compile check that does not require runtime Python.
- **Risks / open questions**:
  - vLLM's FP8 indexer cache block layout is block-major values followed by scale bytes, not token-major `132` byte records.
  - The best top-k path for `rows=128` is vLLM persistent top-k, not cooperative top-k; copying it requires `persistent_topk.cuh` plus its workspace and fallback policy.

## ABI

### K Quant And Cache

Source: `../vllm/csrc/libtorch_stable/cache_kernels.cu::indexer_k_quant_and_cache_kernel`.

Rust wrapper: `glm52_indexer_k_quant_and_cache_launch`.

CUDA symbol:

```c
CUresult glm52_indexer_k_quant_and_cache_cuda(
    const __nv_bfloat16* k,
    unsigned char* indexer_cache,
    const int64_t* slot_mapping,
    int tokens,
    int head_dim,
    int quant_block_size,
    int cache_block_size,
    int64_t cache_block_stride_bytes,
    int use_ue8m0_scale,
    cudaStream_t stream);
```

Contract:

| Buffer | Shape / layout |
| --- | --- |
| `k` | `[T,128]` BF16 |
| `slot_mapping` | `[T]` i64 physical token slots; `-1` rows are skipped |
| `indexer_cache` | block-major bytes: `[block values][block scales]` per block |
| values region | `cache_block_size * 128` FP8 E4M3 bytes |
| scale region | `cache_block_size * 4` bytes, one f32 scale per token because `quant_block_size=128` |
| minimum block stride | `cache_block_size * (128 + 4)` bytes |
| Rust cache extent | `cache_blocks * cache_block_stride_bytes` bytes |

The scale formula follows vLLM CUDA on NVIDIA: `scale = max(amax, 1e-4) / 448`. If `use_ue8m0_scale != 0`, the scale is rounded to the next power of two but is still stored as f32 bytes, matching the source kernel.

### Gather Quant Cache

Source: `../vllm/csrc/libtorch_stable/cache_kernels.cu::cp_gather_indexer_k_quant_cache_kernel`.

Rust wrapper: `glm52_indexer_k_gather_quant_cache_launch`.

CUDA symbol:

```c
CUresult glm52_indexer_k_gather_quant_cache_cuda(
    const unsigned char* indexer_cache,
    unsigned char* dst_k,
    unsigned char* dst_scale,
    const int* block_table,
    const int* cu_seq_lens,
    int batch_size,
    int num_blocks_per_seq,
    int tokens,
    int head_dim,
    int quant_block_size,
    int cache_block_size,
    int64_t cache_block_stride_bytes,
    cudaStream_t stream);
```

Contract:

| Buffer | Shape / layout |
| --- | --- |
| `block_table` | `[batch_size, num_blocks_per_seq]` i32 physical block ids |
| `cu_seq_lens` | `[batch_size + 1]` i32 exact gathered-token ranges |
| `dst_k` | `[tokens,128]` FP8 bytes |
| `dst_scale` | `[tokens,4]` scale bytes, one f32 per token |

This helper is for prefill/chunk and validation paths. The first decode hot path should consume the paged cache directly.

### Top-K 2048

Sources:

- `../vllm/csrc/libtorch_stable/sampler.cu::top_k_per_row_decode`
- `../vllm/csrc/libtorch_stable/cooperative_topk.cu::cooperative_topk`
- `../vllm/csrc/libtorch_stable/topk.cu::persistent_topk`
- `../vllm/vllm/model_executor/layers/sparse_attn_indexer.py` decode dispatch

Rust wrappers:

- `glm52_indexer_topk_2048_workspace_size`
- `glm52_indexer_topk_2048_launch`

CUDA symbols:

```c
CUresult glm52_indexer_topk_2048_contract_cuda(
    int rows,
    int stride,
    int max_seq_len,
    size_t* workspace_bytes);

CUresult glm52_indexer_topk_2048_cuda(
    const float* logits,
    const int* seq_lens,
    int* indices,
    unsigned char* workspace,
    size_t workspace_bytes,
    int rows,
    int stride,
    int max_seq_len,
    int next_n,
    int seq_lens_is_2d,
    cudaStream_t stream);
```

Contract:

| Buffer | Shape / layout |
| --- | --- |
| `logits` | `[rows,stride]` f32 |
| `seq_lens` | `[rows]` or 2-D flattened `[B,next_n]` i32; first GLM52 decode uses `next_n=1` |
| `indices` | `[rows,2048]` i32 |
| `workspace` | `1 MiB` u8, matching vLLM `RADIX_TOPK_WORKSPACE_SIZE` |

Current behavior: contract/workspace query succeeds, launch returns `CUDA_ERROR_NOT_SUPPORTED`. This is deliberate: for fixed bucket `rows=128`, vLLM dispatches persistent top-k, whose source spans `topk.cu`, `persistent_topk.cuh`, and `topk_histogram_4096.cuh`. Returning an error keeps decode-forward from silently consuming fabricated top-k indices.

## Execution Log

### Step 1: Rust ABI

- Added `openinfer-kernels/src/ffi/glm52/indexer.rs`.
- Added `openinfer-kernels/src/ops/glm52/indexer.rs`.
- Wired `openinfer-kernels/src/ffi/glm52.rs` and `openinfer-kernels/src/ops/glm52.rs`.

### Step 2: CUDA Source Port

- Added `openinfer-kernels/csrc/glm52/glm52_indexer.cu`.
- Ported vLLM cache insert and gather page layout with GLM52 constants: `head_dim=128`, `quant_block_size=128`, `scale_bytes=4`.
- Added top-k 2048 contract symbols with fail-closed launch.

### Step 3: Validation

- `rustfmt --check openinfer-kernels/src/ffi/glm52/indexer.rs openinfer-kernels/src/ops/glm52/indexer.rs` passed.
- `/usr/local/cuda-13.3/bin/nvcc -c openinfer-kernels/csrc/glm52/glm52_indexer.cu -o /tmp/glm52_indexer.o -O3 -isystem /usr/local/cuda-13.3/include -I openinfer-kernels/csrc -gencode arch=compute_120,code=sm_120 --compiler-options -fPIC --std=c++17` passed.
- `cargo check -p openinfer-kernels` passed without GLM52 feature.
- `cargo check -p openinfer-kernels --features glm52` is blocked before compiling this slice because `openinfer-kernels/build.rs` requires `OPENINFER_NCCL_ROOT` for the GLM52 DeepEP shim.
- `cargo fmt --check` is blocked by pre-existing GLM52 formatting diffs, including `openinfer-glm52/src/linear.rs`, `runner.rs`, and `weights.rs`, which this task explicitly must not edit.

## Debrief

- **Outcome**: Cache insert/gather now have a concrete, source-backed ABI and wrapper path. Top-k has the decode-forward ABI/workspace boundary but is blocked before first real launch.
- **Pitfalls encountered**:
  - The cache block stride must be explicit because the vLLM layout stores all token values before all token scales inside a page.
  - vLLM's preferred `rows=128` top-k path is persistent top-k, not the simpler cooperative path used for `rows<=32`.
  - The GLM52 feature build currently requires the DeepEP NCCL root even for an indexer-only compile check.
- **Lessons learned**:
  - Treat indexer cache as a page format, like an on-disk database page: callers should pass block size and stride, not infer layout from a pretty tensor shape.
- **Follow-ups**:
  - Port vLLM persistent top-k source and headers, or copy `sampler.cu::top_k_per_row_decode` with an explicit aux workspace policy for long rows.
  - Add an integration smoke that writes a few slots, gathers them by block table, and checks scale/value bytes against a host reference.
