# GLM5.2 Decode Forward Contract

> **TL;DR:** This is the shared contract for getting GLM5.2 decode from startup smokes to a real full forward. The current fixed bucket is `bs=128`, hidden is `6144`, decode is prefilled-state only, and GPU work must stay batched. Existing code already owns FP8 linear/quant, router, DeepEP MoE substrate, TRTLLM grouped W13/W2, and persistent full-layer scratch; the missing shared contract is attention/indexer/KV layout plus the exact full-layer kernel sequence. PP8 should reuse this contract later by replacing DeepEP dispatch/combine with local MoE and adding graph-internal hidden handoff.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes this under `models/glm52`.
  - `docs/models/glm52/support.md` - current GLM52 bring-up ledger, decode-only scope, kernel source policy, and fixed-bucket MoE substrate.
  - `docs/models/glm52/pp-decode.md` - PP8 branch constraints and migration map.
  - `openinfer-glm52/src/config.rs` - authoritative model constants.
  - `openinfer-glm52/src/arena.rs` and `openinfer-glm52/src/arena/validation.rs` - current persistent decode scratch contract.
  - `openinfer-kernels/csrc/glm52/glm52_trtllm_grouped_fp8.cu` and `openinfer-kernels/src/ops/glm52/trtllm_linear.rs` - current plain FP8 linear shape ABI.
- **Relevant history**:
  - `docs/models/glm52/support.md` - GLM first runnable path rejects engine-side prompt prefill, hidden per-token prefill fallbacks, and host loops over requests.
  - `docs/models/glm52/vllm-kernel-reference.md` - attention/indexer source candidates must be checked against vLLM/FlashMLA/FlashInfer before local CUDA.
  - `docs/models/glm52/vllm-moe-fp8-kernels.md` - routed FP8 MoE already has a TRTLLM grouped-offset running substrate; further backend choice needs measurement.
- **Plan**:
  1. Keep the decode tensor shapes in one place so attention, indexer, dense/shared, graph, and PP work do not invent incompatible layouts.
  2. Use subagent source mapping to fill the attention/indexer source rows.
  3. Turn the first chosen source path into a narrow implementation slice: wrapper, arena/view glue, startup smoke, then graph integration.
- **Risks / open questions**:
  - `q_b` and `kv_b` output splits are currently recorded by raw tensor shape; do not hard-code semantic splits until verified against vLLM/FlashMLA.
  - Indexer cache layout is the biggest shared dependency between attention and indexer work.
  - The current arena is DP/EP-shaped; PP8 should not copy its DeepEP buffers unchanged.

## Non-Negotiable Runtime Rules

| Rule | Contract |
| --- | --- |
| Decode-only first | GLM runtime receives prefilled KV/page/indexer state; it does not compute prompt prefill. |
| Real batch | `bs > 1` is a tensor dimension. No host loop that launches a bs=1 decode chain per request. |
| Fixed bucket | First graph bucket is `bs=128`; active rows may be fewer, but buffers and graph addresses stay fixed. |
| Source policy | Search FlashInfer, DeepGEMM, FlashMLA, DeepEP, `../vllm`, then cuBLASLt before writing local CUDA. |
| Local CUDA evidence | Handwritten/local CUDA needs source justification and NCU before performance claims. |
| Failure mode | Bad shape, missing cache state, or unsupported topology should error early. |

## Tensor Facts

Authoritative constants live in `openinfer-glm52/src/config.rs`.

| Name | Value |
| --- | ---: |
| layers | `78` |
| dense layers | `0..3` |
| sparse MoE layers | `3..78` |
| hidden | `6144` |
| vocab | `154880` |
| heads / kv heads | `64 / 64` |
| q lora rank | `2048` |
| kv lora rank | `512` |
| qk nope / rope | `192 / 64` |
| v head dim | `256` |
| routed experts / top-k | `256 / 8` |
| dense intermediate | `12288` |
| expert intermediate | `2048` |
| index heads / head dim / top-k | `32 / 128 / 2048` |
| decode bucket cap | `128` rows per rank/stage first cut |

## Per-Layer Decode DAG

This is the target order for one decode token batch. It is intentionally shape-first; source-specific implementation details are filled after the attention/indexer source mapping.

### Attention Block

| Step | Input -> output | Shape contract | Current status |
| --- | --- | --- | --- |
| input RMSNorm | `hidden -> normed` | `[bs,6144] -> [bs,6144]` | pending shared norm glue |
| q_a projection | `normed -> attention_q_a` | `[bs,6144] -> [bs,2048]` | generic FP8 projection smoke passes on node38 |
| q_a RMSNorm | `attention_q_a -> attention_q_a_normed` | `[bs,2048] -> [bs,2048]` | pending |
| q_b projection | `attention_q_a_normed -> attention_q_b` | `[bs,2048] -> [bs,65536]` | generic FP8 projection smoke passes; split unresolved |
| kv_a projection | `normed -> attention_kv_a` | `[bs,6144] -> [bs,576]` | generic FP8 projection smoke passes |
| kv_a split | `attention_kv_a -> kv_lora + k_rope` | `[bs,512] + [bs,64]` | pending split/cache contract |
| kv_a RMSNorm | `kv_lora -> attention_kv_a_normed` | `[bs,512] -> [bs,512]` | pending |
| kv_b projection | `attention_kv_a_normed -> attention_kv_b` | `[bs,512] -> [bs,114688]` | generic FP8 projection smoke passes; split unresolved |
| RoPE/KPE/cache write | q/k/v pieces + position -> cache | paged MLA/DSA cache | pending source choice |
| indexer path | see below | see below | pending source choice |
| MLA/DSA attention decode | cache + sparse indices -> attention_out | `[bs,16384]` before o_proj | pending wrapper |
| o projection | `attention_out -> hidden_delta` | `[bs,16384] -> [bs,6144]` | generic FP8 projection smoke passes |
| residual | `hidden + hidden_delta -> hidden` | `[bs,6144]` | pending glue |

### Attention Source Map

The attention exploration pass recommends FlashMLA sparse decode as the first real attention backend, but with one sequencing rule: validate the sparse decode ABI with externally supplied/fixture top-k indices before implementing the full indexer generation chain.

| Operator | Source path | Reuse level | Needed ABI / glue | GLM52 shape | Blocker | First implementation boundary |
| --- | --- | --- | --- | --- | --- | --- |
| MLA sparse decode / DSA attention | `openinfer-kernels/third_party/FlashMLA/csrc/api/sparse_decode.h`; `openinfer-kernels/third_party/FlashMLA/csrc/params.h`; `../vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py` | High | `glm52_flashmla_sparse_decode_launch_cuda(q, kv_cache, topk_indices, topk_len, sched_meta, num_splits, out_latent, lse, scratch, stream)` plus Rust wrapper under `ops::glm52` | `q [B,1,64,576] bf16`; packed `kv_cache [blocks,64,1,656] u8/fp8`; `indices [B,1,2048] i32`; latent output `[B,1,64,512] bf16` | Must confirm GLM `q_b [B,65536]` and `kv_b [B,114688]` 4-way expansion mapping to vLLM's `q_nope/q_pe`, `W_UK`, and `W_UV`. | First attention backend. Do not write a local attention kernel. Bring up with fixture top-k indices and packed cache. |
| MLA dense decode fallback | `openinfer-kernels/csrc/kimi_k2/kimi_mla.cu`; `openinfer-kernels/src/ops/kimi_k2/mla.rs`; FlashInfer paged decode headers | Medium | `glm52_mla_dense_decode_launch_cuda(q, ckv_cache, kpe_cache, page_indptr, page_indices, last_page_len, out_latent, stream)` | `q [B,64,576]`; separated BF16 caches `ckv [pages,64,512]`, `kpe [pages,64,64]`; latent output `[B,64,512]` | Dense fallback does not express DSA sparse top-k semantics; it is a layout/correctness scaffold, not the final perf path. | Optional verification branch if FlashMLA sparse cache layout blocks. |
| Separated MLA cache append | FlashInfer page helpers; Kimi MLA wrapper; `../vllm` MLA cache update contract | Medium-high | `glm52_mla_append_kv_launch_cuda(kv_c, k_pe, page_indices, page_indptr, last_page_len, cache_ckv, cache_kpe, stream)` | per token `kv_c [B,512]`, `k_pe [B,64]`; separated BF16 cache | Sparse FlashMLA wants packed `656`-byte token layout, while Kimi/FlashInfer dense path uses separated cache. | Use only for dense fallback or layout validation. |
| Packed FlashMLA sparse cache append | FlashMLA sparse decode API; vLLM `flashmla_sparse.py`; vLLM cache kernels | Medium | `glm52_flashmla_pack_append_cache_cuda(kv_c_fp8, kv_scale, k_pe, slot_mapping, packed_cache, stream)` | packed token bytes `512 fp8 ckv + 16 scale + 128 rope = 656`; cache `[blocks,64,1,656]` | vLLM packing path is custom-op/Torch oriented; raw CUDA contract must be extracted. | Required for sparse mainline; first scope is decode append only. P/D input or fixture can seed pre-existing cache. |
| RoPE / KPE | Kimi MLA RoPE patterns; FlashInfer pos-encoding helpers; vLLM MLA rope contract | Medium | `glm52_rope_apply_decode_cuda(q_rope, k_pe, positions, cos_sin_cache, out_q_rope, out_k_pe, interleave=true, stream)` | `q_rope [B,64,64]`; `k_pe [B,64]`; `rope_theta=8000000`; interleaved | Kimi shape is close but theta/interleave/cache layout must be verified for GLM. | Write standalone RoPE/KPE wrapper first; do not fuse with cache append until both dense/sparse paths agree. |
| q/kv projections | current `linear.rs`; `ops::glm52::trtllm_linear`; `glm52_trtllm_grouped_fp8.cu` | High | Generic projection smoke covers q_a/q_b/kv_a/kv_b/o_proj plus indexer wk/wq_b | q_a `[B,2048]`; q_b `[B,65536]`; kv_a `[B,576]`; kv_b `[B,114688]`; o_proj `[B,6144]` | `q_b`/`kv_b` semantic split is still unproven. | Landed as startup/checkpoint smoke; next users should call the helper from eager forward instead of adding projection-specific wrappers. |
| `v_up` latent -> o_proj input | vLLM MLA `_v_up_proj`; Kimi MLA patterns | Medium-high | `glm52_mla_v_up_cuda(latent [B,64,512], w_uv [64,512,256], out [B,64,256], stream)`, likely cuBLASLt strided/batched GEMM | latent output `[B,64,512]`; `attention_out [B,64,256] == [B,16384]` | Need to prove `kv_b_proj [114688,512]` contains `W_UV [64,512,256]` and how to slice it. | Implement as an independent layout-tested op before o_proj integration. |
| o_proj input layout | vLLM MLA attention layout; current arena and plain FP8 linear | High | Quantize `[B,16384]` and call existing plain FP8 linear `o_proj [6144,16384]` | `attention_out [B,16384] -> hidden_delta [B,6144]` | FlashMLA sparse returns latent `[B,64,512]`, not o_proj input. `v_up` is mandatory. | Fixed rule: all attention wrappers output latent 512; only `v_up` output feeds o_proj. |
| Indexer projection shape flow | vLLM sparse indexer; current arena; plain FP8 linear | Medium | Use plain FP8 for `wk [128,6144]`, `wq_b [4096,2048]`; add norm/RoPE/quant wrappers later | `wk_out [B,128]`; `wq_b_out [B,4096] == [B,32,128]`; scores `[B,32]`; top-k `[B,2048]` | `weights_proj [32,6144]`, indexer norm, and FP8 cache wrappers are not wired. | First implement projection dataflow/shape checks; sparse decode can consume fixture top-k first. |

Before writing the FlashMLA sparse wrapper, add a layout probe that validates how checkpoint `q_b_proj [65536,2048]` and `kv_b_proj [114688,512]` map to the vLLM/FlashMLA MLA factors. In database terms, this is the page format: once wrong, every higher layer can look plausible and still decode garbage.

### Indexer Path

Full-indexer layers are `0,1,2,6,10,...,74`; shared-indexer sparse layers reuse the latest full-indexer state. The exact cache layout is still the central open contract.

| Step | Input -> output | Shape contract | Current status |
| --- | --- | --- | --- |
| indexer score projection | `normed -> indexer_scores` | `[bs,6144] -> [bs,32]`, BF16 weight | pending GEMM/source choice |
| indexer wk projection | `normed -> indexer_wk` | `[bs,6144] -> [bs,128]` | generic FP8 projection smoke passes |
| indexer wq_b projection | `attention_q_a_normed -> indexer_wq_b` | `[bs,2048] -> [bs,4096]` | generic FP8 projection smoke passes |
| indexer K norm/cache | `indexer_wk` -> paged indexer cache | `[bs,128]` plus position/page table | pending vLLM cache mapping |
| indexer logits | query + indexer cache -> scores over context | batch decode, no per-request host loop | pending source choice |
| sparse top-k | logits -> `indexer_topk_idx/weight` | `[bs,2048] i32/f32` | pending source choice |

### Indexer Source Map

The first indexer exploration pass says the implementation order should start with decode batch metadata/page-table ownership, because cache insert, logits, top-k, and attention decode all depend on the same page view.

| Item | Source path | Shape / ABI contract | Portability | First implementation boundary |
| --- | --- | --- | --- | --- |
| Decode metadata / page table | `../vllm/vllm/v1/attention/backends/mla/indexer.py` around metadata build | Non-MTP decode is `seq_lens [B,1] i32`; `block_table [B,max_blocks] i32`; `decode_lens [B] i32`; `schedule_metadata [num_sms+1,2] i32`; `slot_mapping [tokens] i64`; GLM indexer is `heads=32`, `head_dim=128`, `topk=2048`. | Contract only: vLLM uses Python builder/Triton metadata helpers, not a runtime dependency. | Add `Glm52DecodeBatchMetadata` with device `seq_lens_2d`, `block_table`, `slot_mapping`, `positions`, `schedule_metadata`, max len, page size, and cache geometry. Scheduler builds it once per batch. |
| Indexer K quant/cache insert | `../vllm/csrc/libtorch_stable/cache_kernels.cu` around `indexer_k_quant_and_cache`; wrapper around the later cache-kernel entrypoint | `k [T,128] bf16 -> kv_cache [num_blocks,block_size,cache_stride] u8`; each token is 128 FP8 bytes plus 4 scale bytes; `slot_mapping [T] i64`, `-1` rows skipped. | CUDA/C++ kernel is a strong copy/adapt candidate. | Add `glm52_indexer_k_quant_and_cache(stream,k_bf16,indexer_cache_u8,slot_mapping,tokens,head_dim=128,block_size,block_stride,scale_fmt)`. Cache storage must be graph-stable KV/indexer state, not temp arena. |
| Indexer K gather | `../vllm/csrc/libtorch_stable/cache_kernels.cu` around `cp_gather_indexer_k_quant_cache` and wrapper | `kv_cache + block_table [B,num_blocks] + cu_seq_lens [B+1] -> dst_k [total_tokens,128]` plus scales. | CUDA/C++ copy/adapt candidate. | Defer for first decode-only hot path unless attention source requires contiguous gathered K; more likely useful for future P worker/chunk path. |
| Indexer logits | `../vllm/vllm/model_executor/layers/sparse_attn_indexer.py`; `../vllm/vllm/utils/deep_gemm.py`; `../vllm/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | Decode uses `q_fp8 [B,next_n,32,128]`, first cut `next_n=1`; `kv_cache [num_blocks,block_size,1,132] u8` view; `weights [B*next_n,32] f32`; `seq_lens [B,1]`; output logits logically `[B,max_model_len] f32`. Q FP8 scale is folded into `weights`. | Not a direct CUDA file in vLLM; it calls a DeepGEMM symbol and uses Python/Triton/CuTeDSL glue for Q RoPE/quant contract. | Add a raw `glm52_indexer_paged_mqa_logits` wrapper only after cache layout is fixed. Decide whether to allocate full `[B,max_model_len]` logits or implement a block/fused top-k path before committing memory. |
| Sparse top-k 2048 | `../vllm/csrc/libtorch_stable/sampler.cu`, `cooperative_topk.cu`, `topk.cu`, dispatched from `sparse_attn_indexer.py` | Input logits `[rows,stride] f32` with row lengths; output `indices [rows,2048] i32`; workspace `u8`. Cooperative path is best for `rows<=32`; persistent/FilteredTopK covers `rows>32`, so it fits fixed bucket 128. | CUDA/C++ copy/adapt candidate. | Add `glm52_indexer_topk_2048(stream,logits,seq_lens,out_idx,workspace,rows,stride,max_seq_len)`. Current `indexer_topk_weight` arena buffer should not get semantics until source path proves weights are needed; vLLM sparse indexer mainly returns indices. |

### Dense MLP Layers `0..3`

| Step | Input -> output | Shape contract | Current status |
| --- | --- | --- | --- |
| post-attn RMSNorm | `hidden -> normed` | `[bs,6144]` | pending |
| dense gate/up | `normed -> dense_gate_up` | two projections `[bs,12288]`, stored as `[bs,24576]` target scratch | FP8 linear ABI supports shape one projection at a time |
| activation | `gate,up -> dense_activated` | `[bs,12288]` | need non-MoE SiLU*up path |
| dense down | `dense_activated -> hidden_delta` | `[bs,12288] -> [bs,6144]` | FP8 linear ABI supports shape |
| residual | `hidden + hidden_delta -> hidden` | `[bs,6144]` | pending glue |

### Sparse MoE Layers `3..78`

Current running substrate is DP/EP-shaped. PP8 will preserve the quant/GEMM pieces but replace DeepEP.

| Step | Input -> output | Shape contract | Current status |
| --- | --- | --- | --- |
| post-attn RMSNorm | `hidden -> normed` | `[bs,6144]` | pending |
| shared gate/up | `normed -> shared_gate_up` | two projections `[bs,2048]`, scratch `[bs,4096]` | FP8 linear ABI supports shape one projection at a time |
| shared activation/down | `shared_gate_up -> shared_delta` | `[bs,2048] -> [bs,6144]` | activation glue pending; down shape supported |
| router | `normed -> topk_idx/topk_weight` | logits `[bs,256]`, top-k `[bs,8]` | startup smoke passes |
| EP dispatch | token rows -> expert-major rows | DP/EP worst expanded rows `10240`, local experts `32` | DeepEP smoke passes |
| W13 quant + grouped FP8 | expert rows -> `[rows,4096]` | TRTLLM grouped W13 smoke passes |
| weighted W2 quant + grouped FP8 | `[rows,4096] -> [rows,6144]` | TRTLLM grouped W2 smoke passes |
| combine | expert rows -> token rows | `[bs,6144]` | DeepEP combine smoke passes |
| residual | shared + routed + hidden | `[bs,6144]` | pending layer glue |

## Kernel / ABI Checklist

| Area | Needed artifact | Existing code | First implementation boundary |
| --- | --- | --- | --- |
| RMSNorm | BF16 norm over `[rows,width]` for `6144`, `2048`, `512` | likely shared kernels; GLM wrapper pending | choose shared wrapper before local CUDA |
| FP8 linear | projection wrapper for all supported shapes | `glm52_trtllm_fp8_linear_launch`; node38 smoke covers q_a/q_b/kv_a/kv_b/o_proj/indexer_wk/indexer_wq_b | reuse generic projection call from eager forward |
| dense activation | non-MoE `silu(gate)*up` BF16 | none GLM-specific | source-check existing kernels before writing |
| indexer score GEMM | BF16 `[bs,6144] x [32,6144]` | none wired | cuBLAS/linear helper likely enough |
| indexer cache | quant/cache/gather | none wired | source-map vLLM cache kernels |
| indexer top-k | top-2048 per row | none wired | source-map vLLM CUDA top-k |
| attention decode | sparse/dense MLA/DSA decode | none wired | source-map FlashMLA/FlashInfer/vLLM |
| logits | final norm + lm_head `[6144 -> 154880]` | final norm/logits arena only | decide BF16 GEMM and sampling path |
| sampling | batched greedy/non-greedy | `openinfer-sample` shared crate | wire last-stage logits rows |
| graph | full fixed-bucket graph | MoE substrate graph only | add nodes after eager correctness smoke |

## Decode Graph Contract

Current graph evidence is a MoE-substrate smoke only. It captures the fixed `128`-row bucket after seeding inputs and synchronizing, then validates by D2H after replay. Those validation operations are startup/test-only and must not enter the forward hot path.

### Current MoE Graph Smoke Sequence

The current captured sequence starts at `openinfer-glm52/src/moe_deepep.rs::decode_graph_smoke_roundtrip` and runs the shared MoE substrate:

| Order | Captured operation |
| ---: | --- |
| 1 | `glm52_router_noaux_tc_launch` |
| 2 | DeepEP `decode_dispatch` |
| 3 | `glm52_deepgemm_grouped_fp8_metadata_launch` |
| 4 | W13 BF16 -> FP8 per-token/group-128 quant |
| 5 | W13 activation-scale layout: DeepGEMM TMA + TRTLLM grouped-offset |
| 6 | TRTLLM grouped FP8 W13 GEMM |
| 7 | weighted W2-input SwiGLU quant |
| 8 | W2 activation-scale layout: DeepGEMM TMA + TRTLLM grouped-offset |
| 9 | TRTLLM grouped FP8 W2 GEMM |
| 10 | DeepEP `decode_combine` |

### Target Full Decode Graph Sequence

First full graph target is greedy decode for a fixed `B=128` bucket. Partial batches are padded to the bucket; only active rows are read after replay.

Graph-external setup per step:

| Buffer | Rule |
| --- | --- |
| token ids | scheduler writes `[B]`, padded rows use a safe token |
| positions | scheduler/P-D handoff writes `[B]` |
| slots/request ids | persistent device buffers, no allocation in graph |
| KV/page table | graph-stable page metadata; no per-request loop in kernels |
| active mask/count | device-visible active rows; empty rows still follow the fixed sequence |
| sampling params | greedy first; non-greedy can enter later after fixed-row contract is proven |

Graph-internal target sequence:

| Order | Operation |
| ---: | --- |
| 1 | token embedding -> `hidden` |
| 2 | for each layer `0..77`: input RMSNorm |
| 3 | q_a quant + FP8 linear + q_a norm |
| 4 | q_b / kv_a / kv_b FP8 linears |
| 5 | RoPE/KPE, indexer cache/logits/top-k |
| 6 | paged KV append + MLA/DSA batched attention decode |
| 7 | o_proj FP8 linear + attention residual |
| 8 | dense layers: gate/up/down FP8 MLP + residual |
| 9 | sparse layers: shared expert + router + DeepEP dispatch + grouped W13/W2 + combine + residual |
| 10 | final RMSNorm + lm_head |
| 11 | batched greedy argmax/top1 |

Non-greedy sampling and logprobs are graph-after or slow-path work until the greedy graph is correct. Kimi's pattern is the reference: graph runs the decode kernels and top1 selection, then host reads only the selected token/value after graph replay.

### Arena / Ownership Changes

The current `Glm52DecodeArena` is useful as a bring-up arena, but full forward should split ownership so long-lived state does not look like scratch:

| Proposed owner | Contents |
| --- | --- |
| `DecodeMetaArena` | token ids, positions, slots, request indices, page indptr/indices/last len, active mask, padding row/page |
| `AttentionArena` | q split views, kv split views, q_nope/q_rope, kv_lora/k_rope, attention latent, o_proj input/output |
| `DenseArena` | dense gate/up, dense activation, dense down output |
| `MoeArena` | current DeepEP/MoE quant/GEMM scratch |
| `SamplingArena` | logits/top1 ids/top1 values, argmax scratch, later sampling scratch |
| `GraphBucketArena` | per-bucket `CudaGraphState`; graph state must not be a local smoke variable |
| rank/stage KV state | MLA/DSA KV pages, indexer cache, full-indexer cadence state; not bucket scratch |

### Graph Hazards

| Hazard | Rule |
| --- | --- |
| D2H validation | Keep `ctx.sync`, `clone_dtoh`, psum snapshots, and smoke validation outside forward. |
| allocation in capture | No device allocation, host `Vec` growth, or first-use workspace init inside capture. |
| TRTLLM runner warmup | Validate contracts and warm all fixed shapes before capture, so runner init/workspace queries do not appear in graph capture. |
| metadata staging | Reuse host staging buffers; Kimi-style per-step `Vec` construction is acceptable for bring-up but not the perf endpoint. |
| DeepEP collectives | All ranks must enter the same sequence, including empty ranks; fail before collective if state is invalid. |
| logprobs | Full-logits D2H is graph-after slow path, not part of low-latency greedy graph. |

## Non-Attention Forward Source Map

The dense/logits exploration pass found that most non-attention work does not need new core GEMM kernels yet. The immediate work is Rust-side typed views, explicit output buffers, and residual/scale boundaries.

| Item | Current buffer/API | Missing glue | Reuse decision | First order | Risk |
| --- | --- | --- | --- | ---: | --- |
| q_a | `arena.normed -> linear_input_fp8/scale -> attention_q_a`; generic smoke launches TRTLLM plain FP8 linear | Add input RMSNorm and q_a norm typed view before eager forward | Reuse plain FP8 linear | 1 | Norm weights are currently raw loaded tensors; add typed/raw-slice wrapper before feeding norm kernels. |
| q_b | `attention_q_a_normed -> attention_q_b`; smoke validates `(65536,2048)` | Keep q layout split out of the generic linear helper | Reuse plain FP8 linear | 1 | Huge output; do not mix semantic split with generic linear glue. |
| kv_a | `attention_kv_a [576]`, `attention_kv_a_normed [512]`, `attention_k_rope [64]`; smoke validates projection shape | Projection then typed split into kv_lora/k_rope; RMSNorm over kv_lora | Reuse plain FP8 linear | 1 | Split/cache write needs a typed boundary, not scattered offsets. |
| kv_b | `attention_kv_b [114688]`; smoke validates `(114688,512)` | Later K/V split after MLA layout probe | Reuse plain FP8 linear | 2 | Largest non-expert output; layout and memory pressure must be checked. |
| o_proj | `attention_out [16384]` as input; smoke validates `(6144,16384)` | Add explicit o_proj output buffer and residual glue | Reuse plain FP8 linear | 3 | Reusing generic hidden/normed scratch can corrupt residual lifetime; add semantic output. |
| dense gate/up/down | weights and arena `dense_gate_up`, `dense_activated` exist | Per-layer dense view; gate/up write strategy; non-MoE SiLU; down output + residual | Reuse plain FP8 linear for separate gate/up/down | 4 | Combined `(24576,6144)` gate/up is unsupported today. Start with two GEMMs, measure later. |
| shared gate/up/down | shared weights and arena `shared_gate_up`, `shared_activated` exist | Per-layer shared view; gate/up write strategy; shared down output; combine with routed output and scale `2.5` | Reuse plain FP8 linear; do not force routed grouped GEMM path | 5 | Shared runs in 75 layers; two gate/up GEMMs may matter, but correctness boundary comes first. |
| final norm | `model.norm.weight`, `hidden`, `normed` exist | Raw BF16 top-weight typed wrapper or raw-slice norm wrapper | Reuse shared RMSNorm kernel if shape-compatible | 6 | eps is `1e-5`; raw `CudaSlice<u8>` cannot be passed blindly. |
| lm_head | BF16 `[154880,6144]` and `logits [B,154880]` exist | BF16 matrix view/wrapper; graph-safe GEMM `normed -> logits` | Reuse cuBLAS/cuBLASLt BF16 GEMM precedent | 7 | Full vocab logits are large; first pass is correctness, perf is measured later. |
| sampling | `logits` exists; `openinfer-sample::select_batch` exists | GLM worker `SampleScratch`, logits tensor wrapper, params/seed/logprobs handling | Reuse `openinfer-sample` | 8 | TP1 full-vocab path is straightforward; logprobs D2H must stay graph-after. |

Recommended first code slice from this lane: introduce a thin GLM forward view/API layer over raw BF16/FP8 weights and arena slices, then replace the q_a-only smoke with a generic `launch_fp8_projection` helper. After that, dense/shared/final/lm_head work can proceed without inventing new CUDA kernels.

## Recommended Implementation Split

This split keeps useful work parallel while making cache/layout a single shared contract.

| Slice | Files/modules | Deliverable | Depends on |
| --- | --- | --- | --- |
| 1. Forward views + generic FP8 projection | `openinfer-glm52/src/linear.rs`, `weights/view.rs`, small forward/view module if needed | Startup smoke now validates q_a/q_b/kv_a/kv_b/o_proj/indexer_wk/indexer_wq_b on node38; eager forward still needs typed call sites | current TRTLLM plain FP8 linear |
| 2. MLA layout probe | GLM doc + startup/IT assertion path | proves `q_b`/`kv_b` splits for `q_nope/q_pe`, `W_UK`, `W_UV`; no performance claim | checkpoint weights, vLLM/FlashMLA contract |
| 3. Decode metadata/cache contract | new GLM metadata structs and arena/KV state | device `seq_lens_2d`, block table, slot mapping, positions, schedule metadata, packed sparse cache owner | shared by attention and indexer |
| 4. FlashMLA sparse decode wrapper | `openinfer-kernels/csrc/glm52/*`, `ops::glm52` wrapper | fixture top-k + packed cache -> latent `[B,64,512]` smoke | slices 2 and 3 |
| 5. `v_up` + o_proj | kernel wrapper or cuBLASLt helper, projection helper | latent -> `[B,16384]` -> o_proj hidden delta | slices 1 and 2 |
| 6. Indexer cache/logits/top-k | vLLM-derived cache kernels, DeepGEMM/indexer logits wrapper, top-k wrapper | real sparse indices replace fixture indices | slice 3 |
| 7. Dense/shared/final/lm_head/sampling | GLM forward glue + shared sampling | full non-attention layer tails and final token selection | slice 1, norm wrapper |
| 8. Full graph stitch | bucket graph arena + runtime glue | greedy fixed `B=128` full decode graph | eager smokes for slices 1-7 |

## PP8 Migration Notes

PP8 should not invent a second decode math contract. It should reuse this per-layer contract and change only placement/runtime:

| Current DP/EP concept | PP8 equivalent |
| --- | --- |
| one rank has all layers and 32 local routed experts | one stage has a layer slice and all 256 routed experts for those layers |
| DeepEP dispatch/combine | local route permute/combine |
| fixed rank arena | fixed stage arena |
| full-rank graph | per-stage graph |
| no inter-layer hidden transfer | graph-internal P2P hidden handoff between stages |

Additional PP8 constraints from the graph/runtime pass:

| PP8 graph item | Contract |
| --- | --- |
| stage0 | owns embedding plus its layer slice |
| middle stages | start with graph-internal P2P hidden wait, then run local layers, then send hidden |
| stage7 | owns final norm, lm_head, and sampling |
| MoE groups | current TRTLLM grouped contract hard-codes `groups=32`, `m_capacity=10240`, offset rows `11232`; PP stage-local all-expert route needs `groups=256` and recomputed capacities/offset rows |
| stage split | prefer splits that do not cross DSA shared-indexer state; otherwise handoff must carry the relevant indexer state/cache contract |

## Debrief

- **Outcome**: Shared decode contract created from config, arena, existing GLM FP8/MoE ABIs, and four parallel source-mapping lanes for attention, indexer, non-attention forward, and graph/runtime. The first projection slice also landed: node38 checkpoint IT validates q_a/q_b/kv_a/kv_b/o_proj/indexer_wk/indexer_wq_b through the shared plain FP8 projection helper on all 8 ranks.
- **Pitfalls encountered**:
  - The current code validates projection shapes, but `q_b`/`kv_b` semantic splits are not yet proven.
  - Attention and indexer cannot be implemented independently until cache layout is settled.
  - FlashMLA sparse decode returns latent `[B,64,512]`; `v_up` is a required boundary before `o_proj`.
- **Follow-ups**:
  - Continue slice 1 by adding typed norm/residual/eager-forward call sites; generic projection smoke is no longer the blocker.
  - Continue slice 2 with the MLA layout probe for q_b/kv_b semantic splits.
  - After the source choice is fixed, add one narrow startup smoke per new operator before full graph integration.
