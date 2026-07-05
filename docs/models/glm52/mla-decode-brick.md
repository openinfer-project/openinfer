# GLM5.2 MLA Decode Brick (PR1)

> **TL;DR:** Single-layer GLM5.2 MLA decode forward (`hidden[6144] -> o[6144]`, bs=1, full top-k) — the attention correctness foundation for the DP1/EP8 decode path. Kernel ops + model crate only; runner remains fail-closed. Oracle gate DELIVERED: see `oracle-harness.md` (green on jz38 2026-07-02).
>
> **Last touched:** 2026-06

## What this PR adds

- `fp8.rs` — shared FP8 block-scaled projection primitive (`ProjWeight`, `fp8_linear`, `dequant_kv_b`). Used by MLA now; dense MLP and MoE will reuse it in later PRs.
- `mla_decode.rs` — `Glm52MlaLayerWeights` + `glm52_mla_decode_forward`. Only `from_host` constructor (uploads from raw bytes); `from_device` (from the resident EP8 slab) lands when the executor wires in.
- Kernel ops: `glm52_mla_assembly` (query assemble + cache pack), `glm52_moe_quant` (per-token-group FP8 quant), `glm52_trtllm_fp8_linear` (TRTLLM CUTLASS blockscale GEMM), `gemm_strided_batched_bf16` (cuBLAS strided batched).
- `lib.rs` registers the new modules as `#[allow(dead_code)]` — the runner still rejects all generation requests.

See `dp1-ep8-decode-plan.md` for the full 5-PR roadmap.

## Build

**Requires:** SM90a GPU (H200), CUDA 12.6+ (driver API `cuLibraryLoadData` for DeepGEMM JIT), NCCL 2.30.4+ (DeepEP submodule, pulled in by the `glm52` feature via `moe`).

```bash
export OPENINFER_NCCL_ROOT=/path/to/nvidia/nccl  # include/nccl.h + lib/libnccl.so.2
git submodule update --init --recursive
cargo check --release -p openinfer-glm52
```

## Oracle gate

Delivered in the follow-up `feat/glm52-oracle-harness`: `tools/accuracy/glm52_oracle.py` + `src/oracle/mla.rs`. Design, verification (negative controls), and pitfalls in `oracle-harness.md`.

## Hand-written CUDA kernels

Two files are hand-written (not vendored from FlashInfer/TRTLLM/DeepGEMM/cuBLAS):

| file | lines | what |
|---|---|---|
| `csrc/glm52/glm52_mla_assembly.cu` | 142 | query concat + interleave RoPE + fp8 cache pack. Memory-bound elementwise. RoPE mirrors `openinfer-kimi-k2`'s `rope_out`. |
| `csrc/glm52/glm52_moe_quant.cu` | 180 | per-128-group amax → e4m3 FP8 quant (f32 scale). Standard DeepGEMM/FlashInfer contract. |

Both are correct (validated in the prototype branch against HF oracle) but **not tuned**: single-issue-per-element, no vectorized load/store, no occupancy targeting. They are the first candidates for an ncu profiling pass when decode TPOT is measured. If a fused C-ABI alternative appears in vendored FlashInfer/TRTLLM, replace the hand-written version rather than optimizing it in place.
