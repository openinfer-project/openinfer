# GLM5.2 FlashMLA Sparse Wrapper

> **TL;DR:** GLM5.2 now has a first FlashMLA sparse decode raw ABI/Rust wrapper boundary for fixture top-k indices plus V3.2 packed FP8 cache into latent `[B,64,512]`. The C/CUDA slice compiles through the build script for SM90a, but full `cargo check -p openinfer-kernels --features glm52` is currently blocked by an existing Rust borrow error in `openinfer-kernels/src/ops/glm52/indexer.rs`.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes GLM5.2 work under `docs/models/glm52`.
  - `docs/models/glm52/decode-forward-contract.md` - says sparse decode should first validate fixture top-k + packed cache before full indexer/runtime integration.
  - `docs/models/glm52/vllm-kernel-reference.md` - identifies FlashMLA/FlashInfer sparse decode as the source-backed attention path and records the `fp8_ds_mla` cache question.
  - `openinfer-kernels/third_party/FlashMLA/csrc/api/sparse_decode.h` - Torch-facing source contract: `q [b,s_q,h_q,d_qk]`, packed fp8 kv, indices, optional top-k lengths, scheduler metadata, split scratch, output/lse.
  - `openinfer-kernels/third_party/FlashMLA/csrc/params.h` - raw `SparseAttnDecodeParams`, `GetDecodeSchedMetaParams`, `CombineParams`, and `DecodingSchedMeta` field layout.
  - `../vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py` - vLLM sparse MLA metadata and packed V3.2 cache layout: `512 fp8 + 16 f32 scales + 128 bf16 RoPE = 656` bytes/token.
  - `../vllm/vllm/v1/attention/ops/flashmla.py` - vLLM imports FlashMLA extension symbols and exposes sparse availability only on Hopper/Blackwell.
  - `openinfer-kernels/src/ops/kimi_k2/mla.rs` and `openinfer-kernels/src/ops/kimi_k2/mla_rt.rs` - existing Rust CUDA wrapper style and fail-early buffer validation.
- **Relevant history**:
  - `docs/models/glm52/vllm-kernel-reference.md` - do not write local attention CUDA before exhausting FlashMLA/FlashInfer.
  - `docs/models/glm52/decode-forward-contract.md` - attention wrappers should output latent `[B,64,512]`; `v_up` is a separate later step before `o_proj`.
- **Plan**:
  1. Add a GLM52-local C/CUDA wrapper that constructs FlashMLA raw params without Torch/Python runtime dependencies.
  2. Add Rust FFI and `ops::glm52` launch wrappers with explicit tensor length contracts.
  3. Wire only the new FlashMLA translation unit into `openinfer-kernels` build flags.
  4. Verify formatting and run the narrow `openinfer-kernels` glm52 compile path far enough to identify real blockers.
- **Risks / open questions**:
  - This first wrapper is SM90/Hopper-only in code. FlashMLA has SM100 sparse decode sources, but this task did not wire Blackwell selection.
  - The local FlashMLA checkout has an empty `csrc/cutlass` directory; build flags therefore also include the repo's existing FlashInfer vendored CUTLASS path.
  - Runtime correctness still depends on proving GLM52 q/kv factor packing against checkpoint layout and seeding packed cache pages correctly.

## ABI Contract

The C ABI lives in `openinfer-kernels/csrc/glm52/glm52_flashmla_sparse.cu`:

| Symbol | Purpose |
| --- | --- |
| `glm52_flashmla_sparse_decode_num_sm_parts_cuda(int*)` | Query Hopper `num_sm_parts` for scheduler/scratch sizing. Returns `CUDA_ERROR_NOT_SUPPORTED` on non-SM90 devices. |
| `glm52_flashmla_sparse_decode_metadata_cuda(...)` | Fill FlashMLA decode scheduler metadata and `num_splits` for sparse top-k decode. |
| `glm52_flashmla_sparse_decode_launch_cuda(...)` | Launch FlashMLA SM90 V3.2 sparse decode and FlashMLA combine into caller-provided latent output/lse buffers. |

Rust FFI lives in `openinfer-kernels/src/ffi/glm52/flashmla_sparse.rs`; safe-ish wrapper checks live in `openinfer-kernels/src/ops/glm52/flashmla_sparse.rs`.

First supported shape:

| Tensor | DType | Shape / length contract |
| --- | --- | --- |
| `q` | bf16 | `[B,1,64,576]`, contiguous; Rust length `B * 64 * 576`. |
| `packed_kv_cache` | u8 / fp8 bytes | `[num_blocks,64,1,656]`, contiguous page blocks; Rust length `num_blocks * 64 * 656`. |
| `topk_indices` | i32 | `[B,1,2048]`; fixture/source-provided indices, no indexer generation in this task. |
| `topk_length` | optional i32 | `[B]`; null means every row uses constant `topk=2048`. |
| `tile_scheduler_metadata` | i32 | `[num_sm_parts,8]`; 8 is `sizeof(DecodingSchedMeta)/sizeof(int)`. |
| `num_splits` | i32 | `[B+1]`. |
| `out_latent` | bf16 | `[B,1,64,512]`; this is latent MLA output, not `o_proj` input. |
| `lse` | f32 | `[B,1,64]`. |
| `lse_accum` | f32 | `[B + num_sm_parts,1,64]`. |
| `o_accum` | f32 | `[B + num_sm_parts,1,64,512]`. |

Constants are fixed to GLM5.2 / V3.2 sparse MLA first cut: `B <= 128`, `s_q=1`, `h_q=64`, `h_kv=1`, `d_qk=576`, `d_v=512`, `page_size=64`, `topk=2048`, packed token bytes `656`.

## Execution Log

### Source Mapping

- Confirmed FlashMLA `sparse_decode.h` accepts `d_qk == 576`, `d_v == 512`, `h_kv == 1`, `h_q == 64 or 128`, and packed V3.2 cache shape `[num_blocks,page_block_size,h_kv,656]`.
- Confirmed vLLM's `flashmla_sparse.py` documents the same 656-byte token format and builds decode metadata separately from the attention launch.
- Confirmed current Kimi wrappers keep raw CUDA ABI thin and validate Rust buffer lengths before launch.

### Implementation

- Added `openinfer-kernels/csrc/glm52/glm52_flashmla_sparse.cu`.
  - It includes FlashMLA raw params and SM90 sparse fp8 decode implementation, not the Torch extension interface.
  - It calls FlashMLA scheduler metadata, SM90 V3.2 sparse decode, then FlashMLA combine.
- Added Rust FFI in `openinfer-kernels/src/ffi/glm52/flashmla_sparse.rs`.
- Added Rust wrapper in `openinfer-kernels/src/ops/glm52/flashmla_sparse.rs`.
  - The wrapper exposes `Glm52FlashMlaSparseDecode`, metadata launch, decode launch, and buffer length helpers.
- Wired module exports in `openinfer-kernels/src/ffi/glm52.rs` and `openinfer-kernels/src/ops/glm52.rs`.
- Updated `openinfer-kernels/build.rs` for `glm52_flashmla_sparse`.
  - Uses `--std=c++20`, relaxed constexpr, extended lambda, FlashMLA includes, `kerutils`, and FlashInfer vendored CUTLASS fallback.
  - Uses the same SM90a promotion path as the existing GLM52 TRTLLM grouped FP8 TU.

### Verification

- `cargo fmt` passed.
- `cargo check -p openinfer-kernels --features glm52` without env stopped before nvcc because `OPENINFER_NCCL_ROOT` was unset.
- `OPENINFER_NCCL_ROOT=/data/code/workspace-rustllm/pegainfer/.venv/lib/python3.13/site-packages/nvidia/nccl OPENINFER_CUDA_SM=90a cargo check -p openinfer-kernels --features glm52` stopped before nvcc because that NCCL is `2.28.9`, below the existing GLM52 DeepEP `>=2.30.4` requirement.
- `OPENINFER_NCCL_ROOT=/data/code/workspace-rustllm/ep-moe-demo/.venv/lib/python3.12/site-packages/nvidia/nccl OPENINFER_CUDA_SM=90a cargo check -p openinfer-kernels --features glm52` got through build script/nvcc and then failed in Rust at `openinfer-kernels/src/ops/glm52/indexer.rs:323`: immutable `workspace.len()` after mutable `workspace.device_ptr_mut(...)`.

## Debrief

- **Outcome**: The focused FlashMLA sparse decode source/wrapper pass is in place for the first compile boundary: fixture top-k + packed V3.2 cache -> latent `[B,64,512]`, without PP wiring and without Python runtime dependency.
- **Pitfalls encountered**:
  - Full glm52 build currently depends on a separate DeepEP NCCL >=2.30.4 setup; the project `.venv` NCCL is too old.
  - The local FlashMLA checkout lacks populated `csrc/cutlass`; using FlashInfer's vendored CUTLASS include path is necessary for this repo snapshot.
  - Full Rust check is blocked by an existing indexer borrow error outside this task's allowed write scope.
- **Lessons learned**:
  - Treat FlashMLA packed cache as a page format: the 656-byte V3.2 token layout should remain a named contract until the GLM52 cache append path proves it.
  - Keep sparse decode metadata allocation outside the launch; graph buckets need stable metadata and scratch addresses.
- **Follow-ups**:
  - Fix the unrelated `indexer.rs` borrow blocker in a separate task, then rerun full `cargo check -p openinfer-kernels --features glm52`.
  - Add a fixture smoke that allocates q/cache/top-k/metadata/scratch and calls metadata + decode on an SM90 GPU.
  - Add the packed cache append/seed wrapper before connecting runtime attention.
