# GLM5.2 TP4 on GB300

> **TL;DR:** GLM-5.2-FP8 on 4xGB300 now beats matched vLLM pure TP4 at both fully warmed bs=1 shapes: `9.03ms` vs `9.21ms` for 1/256 and `9.56ms` vs `9.60ms` for 1024/256. The path combines FlashInfer sparse MLA, compact decode buckets, SM-selected GEMV/MoE launches, and a vocabulary-sharded greedy tail that reuses the existing TP all-reduce. A controlled vLLM EP4 run is slower at `9.47/9.84ms`, so EP4 is outside this PR.
>
> **Last touched:** 2026-07

## Scope

TP4 is a low-latency topology for the four-GB300 target. It is not an EP8 compatibility mode.

| Topology | Devices | TP / DP / EP | Expert placement | Attention heads/rank |
| --- | ---: | --- | --- | ---: |
| EP8 | 8 | 1 / 8 / 8 | 32 whole routed experts/rank | 64 |
| TP8 | 8 | 1 / 8 / 8 | 1/8 slice of all routed + shared experts | 8 |
| TP4 | 4 | 4 / 1 / 1 | 1/4 slice of all routed + shared experts | 16 |

TP4 launch requires `--tp-size 4 --moe-topo tp4`; omitted DP size resolves to one. Prefix-cache and DSpark behavior follows the existing tensor-replicated path. KV offload remains EP8-only because tensor-replicated ranks mirror the cache.

## Design

### Tensor-parallel runtime

- Every TP rank runs the same eight-row logical bucket with replicated activations and routing.
- Attention weights are head-sharded. MoE gate/up and down weights are sliced along the 2048-wide expert intermediate dimension.
- The shared expert is folded into bank slot 256, so one phase chain handles routed and shared outputs.
- The MoE chain is union, gate/up GEMM, SiLU, down GEMM, LL push, and fixed-order LL receive/reduce.
- Attention `o_proj` partials use a two-shot LL reduce-scatter/broadcast chain. Every rank receives a bit-identical result before redundant downstream routing.
- One device-side epoch tags parity-double-buffered 16-byte LL packets. Spins only wait on packets produced by a previous kernel node.
- VMM buffers use one accessor-specific VA per GPU and reject links without native P2P atomics. Broad peer grants measurably tax the memory-bound GEMMs.

The implementation is shared rather than copied:

- `openinfer-kernels/csrc/glm52/glm52_moe_tp_impl.cuh` contains the MoE kernels and VMM protocol; TP4/TP8 `.cu` files only instantiate rank/slice/grid parameters and ABI names.
- `openinfer-kernels/csrc/glm52/glm52_tp_ar_impl.cuh` and `glm52_tp_ll.cuh` contain the common attention collective and packet primitives.
- `openinfer-kernels/src/ops/glm52/moe_tp.rs` and `tp_ar.rs` own topology-dependent shape validation and FFI dispatch.
- `openinfer-glm52/src/moe_tp.rs` owns one topology-parameterized model runtime state and slice loader.

### MLA backend and cache contract

Kernel selection happens once at model build from the actual device capability and per-rank head shape. There is no `OPENINFER_GLM52_MLA_BACKEND` override.

| Runtime shape | Backend | KV token layout |
| --- | --- | ---: |
| SM100/SM103 and 16 heads/rank | FlashInfer TRTLLM-generation sparse MLA | 576-byte standard E4M3 |
| Other attention-TP shards up to 16 heads/rank | Right-sized sparse MLA | 656-byte `fp8_ds_mla` |
| Full 64-head fallback | FlashMLA sparse decode | 656-byte `fp8_ds_mla` |
| Neither contract supported | Startup error | n/a |

The kernel and cache format are one immutable startup contract. Allocation stride, cache packing, query assembly, offload namespace, schedule state, and attention launch all derive from it. Backend-specific scratch and schedule metadata are enums; invalid `Option` combinations and dummy allocations are not representable.

FlashInfer's header-only runner needs seven checked-in SM100-family cubins for the `{1,2,4,8} x {256,2048}` selector closure. Two are selector seeds and may not appear as final Nsight symbols, but removing them makes the upstream selector reject otherwise-supported shapes. Provenance and hashes are in `openinfer-kernels/cubin/glm52/README.md` and `trtllm_gen/flashInferMetaInfo.h`.

### Blackwell-specific paths

- FlashMLA fallback code is assembled as `sm_100f`; plain `sm_103` cannot encode its CTA-group/tensor-core instructions.
- DeepGEMM MQA has an SM100f instantiation rather than falling into the Hopper-only stub.
- FlashInfer uses standard E4M3 query/KV with static-token sparse indices and a 16MiB persistent workspace per bucket.
- TP4 MoE keeps the same math and fixed reduction order as TP8, but grid sizing is architecture-specific. Blackwell caps GEMM B at 2 CTA/SM and GEMM C at 3 CTA/SM; Hopper retains its occupancy-derived grid.

### Vocabulary-parallel greedy tail

TP4/TP8 decode copies one contiguous `lm_head` vocabulary shard per rank at model build (`38,720` rows for TP4, `19,360` for TP8). The full head remains resident for DSpark and non-greedy sampling.

Each rank computes compact shard logits and a local top-1. The candidate's bf16 value plus three exact bf16 token-id bytes are packed into rank-unique positions of a hidden-width scratch row, gathered through one reserved slot of the existing fixed-order attention TP all-reduce, and selected on-device with the same lowest-global-index tie break. This preserves launch-ahead: every rank has the same global next token before the next graph replay is enqueued. No host merge, new communication protocol, runtime environment variable, or full-logit exchange is added. Sampling steps recompute the full head eagerly outside the graph only when a non-greedy row exists.

## Performance

All serving rows are fully warmed, untraced, bs=1, concurrency 1, fixed input/output lengths, ignore EOS, temperature zero. TTFT is intentionally excluded from the decision.

| Shape | Original TP4 FlashMLA | FlashInfer MLA | + MoE grid | + compact/fused | + vocab shard | vLLM TP4 | Advantage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/256 | `12.30ms` | `10.83ms` | `10.68ms` | `9.28ms` | **`9.03ms`** | `9.21ms` | `0.18ms` (`2.0%`) |
| 1024/256 | `12.45ms` | `11.17ms` | `11.01ms` | `9.78ms` | **`9.56ms`** | `9.60ms` | `0.04ms` (`0.4%`) |

The MLA switch recovered `1.47ms` at 1/256 and `1.28ms` at 1024/256. The MoE grid tune recovered another `0.15-0.16ms/token`. Compact TP4 graph buckets, SM103 one-row dense GEMV launches, the paired `q_a+kv_a` projection, and removal of the compact MoE output bridge recovered the next `1.23-1.40ms/token`. Vocabulary sharding recovered the final `0.22-0.25ms/token`.

### EP4 topology check

The original comparison and OpenInfer implementation both use pure TP4 for MoE. To test whether expert parallelism was the missing route, vLLM was relaunched with the same checkpoint, TP4 attention, standard FP8 KV cache, max model length 4096, and `--enable-expert-parallel`. This places 64 whole experts on each rank and uses the FlashInfer TRTLLM FP8 MoE backend.

| Shape | vLLM pure TP4 | vLLM EP4 | EP4 delta |
| --- | ---: | ---: | ---: |
| 1/256 | `9.21ms` | `9.47ms` | `+2.8%` |
| 1024/256 | `9.60ms` | `9.84ms` | `+2.5%` |

EP4 is not a latency win for the bs=1 target. OpenInfer currently exposes EP8 and tensor-sliced TP4/TP8, not an attention-TP4 plus expert-EP4 hybrid; adding that topology would require 64-expert rank bundles, a four-rank DeepEP path, and composition with the TP4 attention collectives. The controlled vLLM result does not justify that PR expansion.

### Profile attribution

The accepted optimized node trace uses four ranks and exact per-layer instance counts. Totals are all-device kernel duration normalized by tokens and ranks; auxiliary-stream work may overlap.

| Family | Original FlashMLA | FlashInfer FP8 | Delta |
| --- | ---: | ---: | ---: |
| MoE + router + RS | `3.868ms` | `3.840ms` | `-0.028ms` |
| Projection GEMV + reduce | `3.743ms` | `3.765ms` | `+0.022ms` |
| Sparse MLA attention | `2.042ms` | `0.554ms` | **`-1.488ms`** |
| MLA/cache/query/quant glue | `0.734ms` | `0.562ms` | **`-0.172ms`** |
| Norm + fused residual | `0.905ms` | `0.906ms` | `+0.001ms` |
| Attention TP allreduce | `0.693ms` | `0.824ms` | `+0.131ms` |
| MLA absorb W_UK/W_UV | `0.554ms` | `0.545ms` | `-0.009ms` |
| Indexer + lm_head + other | `1.000ms` | `1.001ms` | `+0.002ms` |
| **All-kernel sum** | **`13.540ms`** | **`11.999ms`** | **`-1.541ms`** |

The attention-AR increase is one communication-wait outlier; per-kernel medians did not regress. The traced request measured `11.49ms` versus the untraced `10.83ms`, so the trace is composition evidence rather than the performance result.

The final vocabulary-sharded node trace measures the local LM head at `77.01us` per rank versus about `312us` for the old redundant full-vocabulary head. Pack/unpack cost `1.63/2.46us`; the one extra TP AR triplet averages `8.41us`. The predicted net saving is about `222us/token`, matching serving.

### MoE resource gate and grid tune

The temporary standalone harness directly included the production TP4 `.cu` instantiation and used `UC=9`, one active row, production scratch shapes, and the production kernels. It was used for the measurements below but is intentionally excluded from the production PR.

| NCU metric | GEMM B, grid 456 | GEMM C, grid 608 |
| --- | ---: | ---: |
| Registers/thread | 80 | 56 |
| Theoretical occupancy | 37.50% | 50.00% |
| Achieved occupancy | 30.27% | 35.27% |
| Compute throughput | 33.60% | 20.07% |
| Tensor-pipe active | 9.01% | 6.44% |
| DRAM throughput | 29.62% | 21.31% |
| L1/TEX throughput | 62.83% | 72.89% |

The kernels were not near their compute, Tensor, or DRAM roofs. The occupancy-max grids created excess short-workload scheduling overhead on 152-SM GB300.

Five hot-cache runs used 100 warmups and 1,000 measured launches each:

| Exact kernel | Old grid median | Tuned grid median | Delta |
| --- | ---: | ---: | ---: |
| B | `13.847us` (456) | `13.095us` (304) | `-5.4%` |
| C | `8.912us` (608) | `7.596us` (456) | `-14.8%` |
| **B + C** | **`22.759us`** | **`20.691us`** | **`-2.068us` (`-9.1%`)** |

Across 75 MoE layers, the hot-cache delta predicts `0.155ms/token`, matching the measured serving recovery. Cache-flushed NCU gives an upper bound of `39.33 -> 34.93us` (`-11.2%`).

## Validation

- SM103 release server build passes.
- SM90a and SM103 standalone compilation passes for both TP4/TP8 MoE and attention-AR instantiations.
- FlashInfer sparse MLA numerical smoke passes all eight `batch={1,2,4,8} x topk={256,2048}` combinations.
- Focused GLM5.2 topology/slice tests pass.
- A 64-token greedy HTTP smoke retains the accepted `" Paris. Distance from ..."` prefix.
- A non-greedy temperature/top-p smoke passes through the eager full-head fallback.
- Vocabulary pack/unpack passes a direct SM103 device smoke covering negative logits, cross-rank tie breaking, and a global token id above 65,535; the translation unit also compiles for SM90a.
- Refactoring preserves TP4 B/C register counts (`80/56`) and launch grids. Clean post-refactor and compact/fused serving reruns pass for both target shapes.

## Artifacts

- Serving JSON: `bench_results/glm52-tp4-moe-tune-20260710/moe-grid-in{1,1024}-out256-c1-n20.json`
- NCU reports: `bench_results/glm52-tp4-moe-util-20260710/tp4-gemm-{b,c}.ncu-rep`
- Optimized node trace: `bench_results/glm52-tp4-flashinfer-profile-20260710/openinfer-flashinfer-node.{nsys-rep,sqlite}`
- Trace summary: `bench_results/glm52-tp4-flashinfer-profile-20260710/openinfer-flashinfer-node-summary.md`
- Compact/fused serving JSON: `bench_results/glm52-tp4-fused-front-20260710/fused-front-in{1,1024}-out256-c1-n20.json`
- Compact node trace: `bench_results/glm52-tp4-compact-profile-20260710/openinfer-compact-node.{nsys-rep,sqlite}`
- vLLM EP4 control: `bench_results/glm52-vllm-ep4-20260710/vllm-ep4-in{1,1024}-out256-c1-n20.json`
- Final serving JSON: `bench_results/glm52-tp4-vocab-parallel-20260710/vocab-parallel-in{1,1024}-out256-c1-n20.json`
- Final node trace: `bench_results/glm52-tp4-vocab-profile-20260710/openinfer-vocab-node.{nsys-rep,sqlite}`
- Final trace summaries: `bench_results/glm52-tp4-vocab-profile-20260710/openinfer-vocab-node-{summary,tail}.md`

## Pitfalls

- TP8 historically hid `ranks == tokens == 8`. TP4 scratch and job counts must use the fixed eight-row bucket where appropriate, not the four-rank count.
- TP4 must not initialize the EP8 DeepEP path. It uses the tensor-replicated LL rendezvous.
- Backend labels do not imply cache-layout equality. vLLM's standard FP8 and the original OpenInfer `fp8_ds_mla` cache are different contracts.
- FlashInfer module selection/loading must complete before CUDA Graph capture.
- Node-trace sums include auxiliary-stream overlap and communication waits. Graph-level traces and untraced TPOT remain the wall-time truth.
- H200 results explain the original launch choices but are not a controlled GB300 baseline.
- A host-side merge of vocabulary candidates is invalid under launch-ahead: the global token must be identical on-device before every rank enqueues the speculative next replay.

## PR Cleanup

### Preparation

- **Read**: this model record, the TP4/TP8 CUDA and Rust launch surfaces, the FlashInfer runner/metadata/cubin closure, and the server CLI diff.
- **Plan**: remove runtime backend overrides; dispatch by SM and shape; consolidate TP4/TP8 collectives and runtime state; restore unrelated changes; distill the document; rerun architecture, correctness, kernel, and serving gates.
- **Publish plan**: commit the scoped TP4 optimization/refactor diff, rebase onto the latest `origin/main`, preserve the newer mainline GLM5.2 scheduler/dead-kernel changes while resolving conflicts, rerun the available gates, push with `--force-with-lease`, and move PR #637 from draft to ready for review.
- **Risk**: CUDA templating can change resource generation. Register counts, grids, exact-kernel timing, and serving TPOT are explicit acceptance gates. The latest main contains GLM5.2 scheduler-metrics and dead-kernel cleanup commits, so the rebase may require semantic conflict resolution rather than choosing one side wholesale.

### Execution Log

- Removed `OPENINFER_GLM52_MLA_BACKEND`; model build now selects the MLA/cache contract from actual SM support and heads/rank.
- Replaced backend-specific `Option` scratch and dummy schedule allocations with mutually exclusive enums.
- Consolidated TP4/TP8 LL, MoE, attention AR, Rust FFI wrappers, and model runtime state. Thin `.cu` files retain separate ABI symbols and architecture-specific parameters.
- Audited against the branch merge-base rather than the moving `origin/main`; post-fork Qwen3 changes are not part of this PR and were not cherry-picked during cleanup.
- SM90a/SM103 CUDA instantiations compile. Release Clippy passes with only the repository's existing lint exceptions; GLM5.2 lib tests pass (`59 passed, 14 GPU tests ignored` after the mainline scheduler-metrics merge), server topology tests pass (`4/4`), and the SM103 release server builds.
- The extracted CUDA implementation keeps the exact TP4 resource footprint (`80` registers for GEMM B, `56` for GEMM C). Temporary old/new harness runs under the same externally loaded node were within noise, but those absolute timings are not a clean performance result; the harness source is not part of the production PR.
- A rename-aware PR audit confirms one model TP runtime, one Rust kernel wrapper, and one CUDA implementation per collective. Git still renders the shared-header extraction as deletion plus addition because the original TP8 entry path remains as a thin instantiation.
- Compact TP4 decode now captures buckets 1/2/4/8 while retaining the fixed eight-row MoE ABI behind narrow pad bridges. SM103 selects one-row dense FP8 GEMV launches, `q_a` and `kv_a` share one paired graph node, and the MoE output bridge was removed.
- A same-checkpoint vLLM EP4 control measured `9.47/9.84ms`, slower than pure TP4 at `9.21/9.60ms`; EP4 is therefore excluded from this low-latency PR.
- TP greedy decode now shards the vocabulary head, computes local top-1, and reuses one reserved attention-AR slot for device-side global selection. The full head is retained only for DSpark and sampled rows. Final n20 TPOT is `9.03/9.56ms`; the node trace attributes the serving delta to `312us -> 77us` LM-head work plus about `12.5us` of candidate transport/select overhead.
- Final formatting and `git diff --check` pass. GLM52 Clippy passes with warnings denied after command-line allowances for four pre-existing model lints. GLM52 lib tests pass (`59 passed, 14 GPU tests ignored`), the four server GLM52 config tests pass, and the FlashInfer numerical smoke passes all eight `batch x topk` shapes on SM103.
- Installed GitHub CLI, authenticated through the existing `~/.env` token without persisting or printing it, fetched `origin/main`, and confirmed PR #637 is an open draft targeting `main`. The branch was 16 mainline commits behind before the publish rebase.
- Rebased both PR commits onto `origin/main` at `60ccc5f`. Conflict resolution preserved main's scheduler metrics and dead-kernel/weight-load cleanup: `deepgemm_layout`, `moe_route`, and `trtllm_grouped` were not restored; their stale build/FFI/module registrations and one superseded FlashMLA arch helper were removed. The H200 TP8 right-sized sparse MLA path now coexists with the SM103 TP4 FlashInfer path. Two temporary tuning harnesses were removed from the production diff.
- Addressed the first review pass by deriving scheduler partitions, startup load feeds, and frontend engine count from one topology-owned logical-rank count. TP4/TP8 now both expose one mirrored request partition, while EP8 retains eight independent partitions.

### Debrief

- **Outcome**: TP4 beats the matched latest vLLM TP4 baseline at both requested bs=1 shapes, with exact greedy smoke, sampled fallback, release build/tests, and a final node-level Nsys trace.
- **Pitfalls encountered**: TP8 had hidden the `ranks == bucket rows` assumption; full-vocabulary work was replicated even after the attention/MoE weights were sharded; a host candidate merge would break launch-ahead; EP4 was a plausible but ultimately slower alternative; and the mainline rebase required preserving the right-sized H200 MLA fallback plus both short/full graph pre-captures rather than choosing either conflict side wholesale.
- **Lessons learned**: kernel/backend selection belongs to SM and topology at startup, compact graph shapes matter at bs=1, and replicated bookend work must be audited separately from the layer stack in tensor-parallel profiles.
- **Follow-ups**: add the model-level TP4 golden/logit gate and rerun retained TP8 runtime gates on H200.

## Remaining Work

- Add a model-level TP4 golden/logit gate; HTTP prefix parity and focused kernel gates are narrower evidence.
- Re-run the retained TP8 runtime gates on H200 when that host is available; GB300 can prove compilation but not H200 performance.
