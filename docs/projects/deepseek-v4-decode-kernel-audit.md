# DeepSeek V4 Decode Kernel Audit

**Created**: 2026-05-11
**Status**: active

## TL;DR

Decode is being audited at launch granularity instead of wrapper granularity. Final logits now preserves the FP32 SGEMV math path while caching the converted rank-local head weight in per-device scratch, so steady decode no longer converts the full BF16 head weight or creates/destroys cuBLAS resources every token. Exact DeepSeek V4 E2E remains `20/20`; the comparable 1x32 clean bench improved from `98.80ms` to `93.14ms`.

## Preparation

- **Read**:
  - `docs/index.md` - found DeepSeek V4 decode work history and the active MoE AG/RS record.
  - `docs/projects/deepseek-moe-ag-rs.md` - confirmed decode MoE is not the focus for this slice and that HC prenorm fusion currently benchmarks at `98.80ms`.
  - `pegainfer-deepseek-v4/src/direct/worker.rs` - decode token flow is rank workers: token copy, embedding, all-reduce, HC expand, 43 block decodes, final logits, logits all-gather.
  - `pegainfer-deepseek-v4/src/runtime/block.rs` - per-layer decode structure is HC prenorm, attention, FP32 all-reduce, HC post, FFN HC prenorm, MoE AG/RS, HC post.
  - `pegainfer-deepseek-v4/src/runtime/core.rs` - final logits calls `hc_head`, final RMSNorm, and `deepseek_last_token_bf16_logits_cuda`.
  - `pegainfer-kernels/csrc/deepseek_v4/deepseek_hc.cu` - final logits currently allocates temporary F32 buffers, converts the full head weight every call, creates a cuBLAS handle, runs SGEMV, then frees everything.
- **Relevant history**:
  - `docs/projects/deepseek-moe-ag-rs.md` Step 12 rejected a hand-written HC mixes decode kernel because exact E2E passed but clean TPOT regressed to `115.10ms`.
- **Plan**:
  1. Add a launch-granularity decode checklist to this document so future work does not collapse back to local wrapper names.
  2. Change `deepseek_last_token_bf16_logits_cuda` to reuse per-device scratch for the F32 hidden vector, converted F32 head weight, and cuBLAS handle.
  3. Preserve the existing FP32 SGEMV math path: BF16 input/weight are still converted to F32 before dot product.
  4. Run local format/check, sync to 5090, run exact E2E, then run longer decode benches with `output_len >= 160`, `warmup >= 2`, `iters >= 3`.
- **Risks / open questions**:
  - The first token after load still pays the head-weight conversion; steady TPOT should improve more than first decode.
  - 5090 is temporarily returning `no available gateway`; remote E2E and bench will be run once the alias is reachable.

## Decode Launch Checklist

- **Per token, before layers**:
  - H2D token id copy
  - `embedding_batched_vocab_shard_cuda`
  - embedding NCCL all-reduce
  - `deepseek_hc_expand_cuda`
- **Per layer, attention prelude**:
  - `deepseek_hc_mixes_cuda`: BF16->F32 kernel, cuBLAS SGEMV, HC scale kernel
  - `deepseek_hc_pre_norm_from_mixes_cuda`
  - attention projection: FP8 wq_a act+GEMM, q RMSNorm, FP8 wq_b act+GEMM, head RMSNorm, FP8 wkv act+GEMM, kv RMSNorm
  - Q RoPE, KV RoPE, KV nope quant, KV cache row copy
- **ratio 0 attention layers**:
  - window top-k decode
  - indexed cache attention
  - output inverse RoPE, wo_a BF16 linear, wo_b FP8 act+GEMM
  - BF16->F32, NCCL all-reduce, F32->BF16
  - HC post
- **ratio 128 attention layers**:
  - non-overlap compressor decode every token
  - window top-k decode
  - after compressed history exists: compress top-k decode and concat top-k
  - on `(start_pos + 1) % 128 == 0`: compressed KV strided RoPE, nope quant, compressed KV cache copy
  - indexed cache attention, output projection, FP32 all-reduce wrapper, HC post
- **ratio 4 attention layers**:
  - overlap compressor decode every token
  - indexer q FP8 act+GEMM, indexer q RoPE, indexer q Hadamard FP4 quant
  - indexer overlap compressor decode every token
  - after compressed history exists: indexer weights BF16 linear, indexer score kernel, NCCL score all-reduce, indexer top-k, concat top-k
  - on `(start_pos + 1) % 4 == 0`: main compressed KV RoPE/quant/copy plus indexer compressed KV RoPE/Hadamard/copy
  - indexed cache attention, output projection, FP32 all-reduce wrapper, HC post
- **Per layer, FFN/MoE branch**:
  - FFN HC mixes and HC prenorm
  - MoE hidden all-gather; hash layers also gather token ids
  - hash or score gate
  - local mapping, expert-major expand
  - grouped FP4 w1 act+GEMM, grouped FP4 w3 act+GEMM, SwiGLU, grouped FP4 w2 act+GEMM
  - fused output reduce, F32 reduce-scatter
  - shared expert FP8 w1 act+GEMM, FP8 w3 act+GEMM, SwiGLU, FP8 w2 act+GEMM
  - routed/shared add, HC post
- **Per token, final logits**:
  - HC head mixes, head pre, head pre-output
  - final RMSNorm
  - final logits GEMV
  - logits NCCL all-gather
  - rank0 logits D2H

## Execution Log

### Step 1: Cache final logits F32 head weight
- Changed `deepseek_last_token_bf16_logits_cuda` in `pegainfer-kernels/csrc/deepseek_v4/deepseek_hc.cu`:
  - Reuses `DeepseekHcScratch::x_f32` for the last hidden vector conversion.
  - Adds per-device `logits_weight_f32` scratch keyed by source weight pointer and valid element count.
  - Reuses the existing per-device cuBLAS handle instead of creating/destroying one per token.
  - Keeps the original FP32 math path: BF16 hidden and BF16 head weight are converted to F32, then cuBLAS SGEMV computes F32 logits.
- Updated `pegainfer-kernels/KERNELS.md` to record the final logits scratch behavior.
- Local verification:
  - `cargo fmt --check` passed.
  - `git diff --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- 5090 exact verification:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- --model-path /data/DeepSeek-V4-Flash
```

- Result: `All 20 DeepSeek V4 exact cases passed`.
- Comparable short bench:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1
```

- Result: `steady_tpot_ms.avg = 93.14ms`, `p50 = 92.11ms`, `p95 = 99.18ms`, `first_decode_step_ms = 90.98ms`, `decode_tok_s = 10.74`.
- Baseline for the same 1x32 shape after HC prenorm fuse was `steady_tpot_ms.avg = 98.80ms`, so this is a `5.66ms/token` improvement on the short benchmark.

### Step 2: Longer decode reproducibility bench
- New longer bench shape:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3
```

- Reason: `output_len=160` crosses the ratio-128 compressed-attention boundary, so it is a better stability probe than 1x32.
- Round 1 result: `steady_tpot_ms.avg = 106.19ms`, `p50 = 107.07ms`, `p95 = 114.40ms`, `p99 = 116.75ms`, `first_decode_step_ms.avg = 100.54ms`, `decode_tok_s = 9.42`.
- Round 2 result: `steady_tpot_ms.avg = 102.89ms`, `p50 = 98.18ms`, `p95 = 118.52ms`, `p99 = 127.17ms`, `first_decode_step_ms.avg = 93.72ms`, `decode_tok_s = 9.72`.
- Observation: average TPOT differs by about `3.1%` across two clean runs and p99 varies more than the mean. This bench shape is useful for stability tracking, but more runs are needed before treating a single measurement as reproducible.
- Both long bench runs printed a rank-7 `NcclError` panic while aborting the communicator after scheduler exit. The process returned success and metrics were emitted, but shutdown cleanup should be investigated separately.

### Step 3: Fuse decode final HC head plus RMSNorm
- Tried a decode-only final-head fused kernel for `seq_len=1`.
- The experiment kept `deepseek_hc_mixes_cuda` unchanged so final head dot products still used the existing FP32 cuBLAS path, then fused `deepseek_hc_head_pre_cuda`, `deepseek_hc_pre_output_cuda`, and the following RMSNorm into one kernel.
- The fused kernel preserved the old BF16 rounding boundary between HC head output and RMSNorm, and non-`seq_len=1` shapes stayed on the old path.
- Verification:
  - local `cargo fmt --check` passed
  - local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed
  - 5090 full exact E2E passed: `All 20 DeepSeek V4 exact cases passed`
- Bench result:
  - 1x32 run 1: `steady_tpot_ms.avg = 100.69ms`
  - 1x32 run 2: `steady_tpot_ms.avg = 107.65ms`
- Result: reverted the code. Reducing two small launches did not offset replacing FlashInfer RMSNorm with the custom fused reduction kernel.

## Debrief
