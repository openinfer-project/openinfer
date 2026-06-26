# GLM5.2 PP8 Decode Exploration

> **TL;DR:** PP8 decode is worth exploring as a GLM5.2 low-latency branch, but not because PP magically uses 8-card HBM for one token. The working hypothesis is narrower: a graph-internal NVLink P2P hidden handoff can make stage-boundary cost tiny (`~16us` total for 7 H200/NV18 12KB handoffs from the copied experiment), so PP8 may beat a TP-style path only if TP's per-layer collectives/P2P/runtime edges cost several milliseconds per token. Most existing GLM52 compute kernels can migrate to PP, but MoE layout changes from `EP8 local_experts=32 + DeepEP` to a likely `PP8 TP1 EP1` stage-local plan where each stage owns all 256 experts for its layer slice; that needs new local route permute/combine and graph P2P handoff kernels. Current code has checkpoint load, FP8 weight contracts, q_a plain FP8 linear smoke, and graph-captured MoE substrate; it still lacks real full decode forward, attention/indexer/KV, logits/sampling, and PP handoff/runtime. **PP depth is a memory/replica-density knob, not a bs=1 latency knob** (`W_active/B_gpu` ~7.6ms is pp_size-independent; PP6-vs-PP8 differ ~4.5us): min-fit pp_size = H200→PP8, B200→PP6, B300(288GB)→PP4. Balance resident **sparse-layer count** (~9.4/stage; each sparse layer = all 256 experts ~9GiB), which forces splitting ~3 indexer groups on H200 (cheap: ~2.3us/split). rank0 = layers 0..11 (3 dense + 9 sparse + embed).
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes this as a GLM52 model-line exploration doc.
  - `docs/models/glm52/support.md` - current bring-up target is decode-only `DP8 TP1 EP8` with real batched decode, full decode CUDA Graph, no prefill fallback, no MTP first cut, and no host loops over prompt tokens or requests.
  - `docs/models/glm52/vllm-kernel-reference.md` - GLM52 operator source map: router/top-k, DeepEP/DeepGEMM route contracts, DSA indexer/cache/top-k, and FlashMLA/FlashInfer decode candidates.
  - `docs/models/glm52/vllm-moe-fp8-kernels.md` - FP8 MoE backend map: FlashInfer CUTLASS, DeepGEMM, vLLM CUTLASS, and the current TRTLLM grouped-offset substrate.
  - `docs/playbooks/model-optimization-pipeline.md` - keep roofline/e2e/profiling evidence in the per-model doc, and do not claim wins before A/B data.
  - `docs/models/kimi-k2/dp-design.md` - current DP runtime precedent uses a coordinator plus per-rank engines; PP should preserve the same "GPU-owning worker thread" discipline, but stage dependency replaces DP independence.
  - `/data/code/tilert_play/glm5_tpot_pp_tp_估算.md` - copied the user's PP/TP roofline and H200 P2P handoff measurements into this project doc.
- **Relevant history**:
  - `docs/models/glm52/support.md` - current GLM52 work already rejected hidden prefill paths, `for token in prompt` loops, and `for req in bs1` decode loops; PP must keep the same red lines.
  - `docs/models/kimi-k2/dp-design.md` - Kimi's TP1/DP8 design removed layer-level TP all-reduce by making ranks independent; PP8 removes collectives differently, by serial layer partitioning plus small hidden transfers.
  - `docs/lessons/kimi-bringup-numerics.md` - MoE+TP correctness is sensitive to reduce precision and finalize placement; PP's local MoE path still needs a logits gate before serving.
- **Plan**:
  1. Preserve the user's roofline and H200 P2P experiment data in this doc as the baseline reasoning for PP8.
  2. Add the current OpenInfer GLM52 implementation state and missing kernel list, separated from the PP hypothesis.
  3. Define the first PP8 decode shape: stage-owned layer slices, fixed graph buffers, graph-internal P2P hidden handoff, no engine-side prefill, no MTP initially.
  4. Classify existing GLM52 kernels by whether they migrate unchanged, require shape/layout parameterization, or need new PP-specific kernels.
  5. Record the next measurements needed before committing to PP as a production direction.
- **Risks / open questions**:
  - PP8 low latency can still lose if single-stage HBM reads dominate and TP's graph-fused communication edge cost is already small.
  - The "all 256 experts per stage layer" plan must be checked against H200/B200 memory with real GLM52 FP8 weights and KV budget.
  - Local MoE finalize is a new correctness surface even though W13/W2 GEMM kernels mostly carry over.

## Scope

This doc tracks a low-latency **PP8 decode** branch for GLM5.2. It is not replacing the current `DP8 TP1 EP8` DeepEP bring-up until measurements say it should.

The shared per-layer decode DAG, tensor shapes, source maps, and first implementation split now live in `docs/models/glm52/decode-forward-contract.md`. PP8 should reuse that math contract and change only placement/runtime: stage-local layer slices, all 256 experts per stage layer, local MoE finalize, and graph-internal hidden handoff.

| Item | PP8 first-cut rule |
| --- | --- |
| Objective | Single-request / small-batch decode latency, especially TPOT without DFlash/MTP. |
| Parallelism | `PP8 TP1 EP1` candidate: 8 pipeline stages, one GPU per stage, no NCCL all-to-all, no tensor-parallel all-reduce. |
| Expert placement | Each stage holds all 256 routed experts for the sparse layers assigned to that stage. |
| Prefill | Out of scope inside GLM engine. Decode receives prefilled KV/page/indexer state from future P/D handoff. |
| MTP | Out of scope for the first PP decode branch; GLM5.2 has built-in MTP, but base decode must be measured first. |
| Batch shape | `bs > 1` stays real batched tensor work. No host loop over bs=1 requests. |
| CUDA Graph | Required: per-stage fixed-buffer graph replay plus graph-internal P2P hidden handoff. |
| Frontend | Reuse Qwen3/Kimi request semantics; PP affects model runtime, not OpenAI API behavior. |

## Current OpenInfer GLM52 State

As of this doc, the current implementation is a `DP8 TP1 EP8` decode substrate, not a runnable full forward.

| Area | State |
| --- | --- |
| Model crate/server feature | `openinfer-glm52` exists behind `glm52`; server/bench recognize GLM5.2 and reject request-time forward while bring-up is incomplete. |
| Checkpoint loading | Real `/data/models/GLM-5.2-0614-Provider-FP8` load works on jiuzhang node38; rank plans, raw H2D load, non-expert contracts, routed expert packages, and arena residency are validated. |
| Weight layout | Non-expert FP8 projection contract is validated; routed experts are packed as local expert-major FP8 W13/W2 packages for `local_experts=32` per rank. |
| MoE substrate | Router, DeepEP dispatch/combine, W13/W2 FP8 quant, TRTLLM grouped FP8 W13/W2, and fixed-bucket MoE decode-substrate CUDA Graph smoke pass on node38. |
| Non-expert FP8 linear | Plain TRTLLM FP8 blockscale linear ABI exists; node38 checkpoint IT validates q_a, q_b, kv_a, kv_b, o_proj, indexer_wk, and indexer_wq_b projection smokes at `rows=128`, `workspace=0`, valid activation scales, and nonzero outputs on all ranks. |
| Full decode forward | Pending: no attention/indexer/KV decode, full projection sequencing, dense/shared/residual integration, logits/sampling, scheduler-owned prefilled handoff, or full-forward decode graph. |
| Tests | One node38 checkpoint IT is the useful gate; synthetic probe tests were deleted. |

## Missing Kernel / Runtime Work

This is the current non-PP gap list. PP can reuse much of it, but does not make these operators disappear.

| Area | Current gap | PP8 impact |
| --- | --- | --- |
| RMSNorm/residual sequencing | Need full layer order and persistent graph buffers. | Reuse/parameterize; stage boundaries only see hidden after residual. |
| Non-expert FP8 projections | Attention/indexer q_a/q_b/kv_a/kv_b/o_proj/wk/wq_b smoke exists; dense MLP, shared expert, logits-side projections still need layer integration. | Same kernels migrate; stage-local weight slices reduce per-GPU layer count, not projection shapes. |
| MLA/DSA attention decode | Need GLM KV/cache layout, indexer cache, sparse index top-k, FlashMLA/FlashInfer/vLLM source selection. | Same stage-local attention kernels; KV ownership must follow layer/stage placement and P/D handoff contract. |
| Indexer logits/cache/top-k | vLLM CUDA candidates exist, but OpenInfer wrappers are not wired. | Same compute; PP does not require cross-stage index packets if each stage owns its layers' indexer state. |
| Dense first 3 layers | Need dense SwiGLU path and graph integration. | Stage0 likely owns these layers; no EP/PP change beyond output hidden handoff. |
| Shared expert | Scratch exists, compute/residual not wired. | Same projection kernels; local to the stage. |
| Routed expert compute | DP/EP path has DeepEP expanded layout plus TRTLLM grouped GEMM smoke. | GEMM/quant kernels migrate, but groups likely become `256` experts per stage instead of `32` local experts per rank; metadata and package layout must change. |
| MoE dispatch/finalize | Current path uses DeepEP all-to-all and combine. | Replace with local route permute/pack plus local combine/reduce; no NCCL/DeepEP on the PP-first path. |
| Logits/sampling | Pending shared sampler integration. | Last stage only. Reuse `openinfer-sample`; no model-local sampler. |
| Decode CUDA Graph | MoE substrate graph exists; full forward graph pending. | PP needs one graph per stage plus hidden send/wait nodes/kernels. |
| Scheduler/runtime | Current DP coordinator + rank workers exist. | PP needs a coordinator plus 8 stage worker threads; stage workers are not independent engines, they are one serial pipeline for a request stream. |

## PP8 Decode Shape

### PP is a memory tool, not a latency tool

For bs=1 decode `TPOT_pp ~= W_active / B_gpu + (pp_size - 1) * L_send`. The `W_active / B_gpu` term (~7.6ms observed) is independent of `pp_size` — stages only change *where* the active weights are read, not how many bytes, and the reads are serial either way. `L_send` is ~2.3us/hop on H200 (12KB hidden, half-RTT), so PP6 vs PP8 differ by ~4.5us out of ~7.6ms = 0.09%. **Depth is therefore chosen by HBM capacity and replica density, never by single-request latency.**

Runtime weights (layers 0..77, MTP layer 78 excluded) total ~705 GiB FP8 (full checkpoint on disk is 715 GiB). Per-GPU weight = `705 / pp_size`; KV is also sharded by stage. Min-fit pp_size by target:

| GPU | usable HBM | min-fit pp | per-GPU weight |
| --- | ---: | ---: | ---: |
| H200 | ~131 GiB (141 GB) | **PP8** | ~88 GiB (PP6 = 118 GiB leaves no KV room) |
| B200 | ~170 GiB (192 GB) | PP6 | ~118 GiB |
| B300 | ~255 GiB (288 GB) | PP4 | ~176 GiB (PP3 = 235 GiB too tight at long ctx) |

Deeper PP only buys KV headroom / longer max context; shallower PP buys replica density (GB300 NVL72: PP4 x18 vs PP6 x12 vs PP8 x9). `pp_size` must be a parameter, not a constant — node38/H200 is the PP8 build baseline only.

### PP8 partition (node38 / H200)

Balance the **resident sparse-layer count**, not raw layer count. A sparse layer keeps all 256 experts resident (~9 GiB); a dense layer (0,1,2) is ~0.5 GiB and embed/lm_head are ~1.8 GiB each — nearly free by comparison. Target = 75 sparse layers / 8 ≈ 9.4 per stage.

| Stage (rank) | layers | total | sparse | extra |
| ---: | --- | ---: | ---: | --- |
| 0 | 0..11 | 12 | 9 | + 3 dense (0,1,2) + `embed_tokens` |
| 1 | 12..21 | 10 | 10 | |
| 2 | 22..31 | 10 | 10 | |
| 3 | 32..41 | 10 | 10 | |
| 4 | 42..51 | 10 | 10 | |
| 5 | 52..61 | 10 | 10 | |
| 6 | 62..69 | 8 | 8 | |
| 7 | 70..77 | 8 | 8 | + final norm + `lm_head` + sample |

Stage0 (9 sparse) and stage7 (8 sparse) carry fewer sparse layers precisely to absorb `embed_tokens` / `lm_head`, so every stage lands near the proven ~84 GiB/GPU.

### Indexer groups force quad splits — memory-fit wins

Indexer top-k is recomputed only on **21 full layers** (`0,1,2` then every 4th: `6,10,...,74`); the 3 `shared` layers after each full layer reuse its selected top-2048 index (config `index_topk_freq=4`, `index_skip_topk_offset=3`, `index_share_for_mtp_iteration=true`). The atomic reuse units are `{0}`, `{1}`, `{2,3,4,5}`, `{6,7,8,9}`, ..., `{74,75,76,77}`.

A fully group-aligned split (boundaries only at layer ≡ 2 mod 4) forces stages of 8 or 12 sparse layers; 8 stages of ≤8 sparse can hold at most 64 < 75, so zero-split PP8 needs 12-sparse stages (~107 GiB) that only fit H200 below ~128K context. **So memory-fit forces splitting some groups.** The table above splits 3 groups (after layers 11, 31, 51): the full layer's top-2048 selection (~8 KB at bs=1) ships one extra hop to the next stage, ~2.3us each, ~7us/token total — negligible. 16 of 19 groups stay stage-local, so PP's "index rarely leaves the GPU" edge over TP holds.

Exact boundaries still need tuning by measured stage time (dense head, full-layer indexer cost scales with context); balanced sparse-layer count is the starting point, not the final cut.

### Runtime Shape

| Component | Count | Role |
| --- | ---: | --- |
| Coordinator thread | 1 | Owns request admission, decode epochs, graph replay ordering, token output routing, cancellation, and error handling. |
| Stage worker threads | 8 | One GPU-owning thread per PP stage. Each owns CUDA context, stage weights, stage-local KV/indexer cache, fixed graph buffers, and graph executable(s). |
| Model runtime threads | at least 9 | Same minimum shape as the current DP bring-up, but the dependency graph is a serial pipeline rather than eight independent DP engines. |

The coordinator should not own GPU tensors. It only advances epochs and handles request lifecycle; stage workers own device state.

### Stage Handoff Contract

| Boundary | Contract |
| --- | --- |
| Payload | BF16 hidden `[bs, hidden=6144]`; `bs=1` is 12KB, `bs=4` is 48KB. |
| Buffering | Fixed peer-visible input/output buffers, at least double-buffered by epoch to avoid overwrite. |
| Visibility | System-scope store/fence for payload and flag; receiver waits on expected epoch before reading. |
| Graph | Send/wait kernels or copy nodes live inside the per-stage CUDA Graph; host should not issue per-boundary events in the hot path. |
| Reset | Epoch state must be monotonic or explicitly cleared when graphs are rebuilt. |
| Failure mode | Crash early on stale/mismatched epoch rather than silently consuming the wrong hidden. |

## Kernel Migration Map

| Current GLM52 substrate | PP8 migration | Notes |
| --- | --- | --- |
| FP8 activation quant BF16 -> E4M3 per-token/per-128 | Reuse | Same shapes for projection inputs and routed W13/W2 inputs. |
| Plain TRTLLM FP8 blockscale linear | Reuse | The shared helper already validates q_a/q_b/kv/o plus indexer wk/wq_b; dense/shared projections should call the same path after their typed layer glue lands. |
| TRTLLM grouped FP8 W13/W2 runner | Reuse with parameterization | `G=32` local expert groups becomes likely `G=256` stage-local expert groups; package plan, offsets, scale relayout, and arena sizes must be generalized. |
| Weighted W2-input SwiGLU quant | Reuse with layout checks | Route weights still belong before local combine unless a future logits gate forces a vLLM-style finalize contract. |
| Router noaux top-k | Reuse/benchmark | Existing Kimi-derived router works semantically; vLLM small-batch router GEMM remains a perf candidate. |
| DeepEP dispatch/combine | Do not use in PP-first path | PP stage owns all experts for its layers, so all-to-all is avoidable and would reintroduce latency. Keep DeepEP for DP8/EP8 branch. |
| DeepEP `psum_expert` grouped metadata | Replace or generalize | Local route permute can emit expert offsets directly; no remote recv layout. |
| MoE graph substrate | Partially reuse | Quant + W13 + W2 can stay; dispatch/combine endpoints change. |
| Decode arena | Split into stage arena | Each stage only allocates tensors for its layer slice plus P2P buffers, not all-layer DP scratch. |
| Weight package loader | Needs PP package plan | Current EP packages load 32 experts for all sparse layers. PP package should load 256 experts for only the stage's sparse layers. |
| Scheduler rank workers | Reuse discipline, not topology | Keep GPU-owning worker threads and crash-early invariants; replace independent DP engines with stage pipeline. |

## First Measurements Needed

Do not claim PP is a win until these are measured in OpenInfer.

| Measurement | Why |
| --- | --- |
| Current DP/EP base decode TPOT without DFlash/MTP | Establish the real baseline; the observed 7-8ms is plausible but must be tied to a fixed config and code revision. |
| Minimal PP graph `pp_size=2/4/8`, payload `12KB/48KB`, dummy compute `0/50/100/500us` | Confirms graph replay + P2P flag behavior inside OpenInfer, not only the standalone `/tmp/p2p_lsend` harness. |
| Stage-local weight load memory | Checks whether 256 experts for each stage's layer slice fits with KV/cache headroom on H200/B200. |
| Stage time by layer slice | Finds load imbalance before implementing the full scheduler. |
| Local MoE permute/combine NCU | New PP-specific kernels must be profiled if handwritten or locally adapted. |
| Full PP8 base decode TPOT | Compare against DP/EP and any TP candidate on identical prompt/output/temperature settings. |

## Imported Roofline / Experiment Notes

The rest of this file is copied from `/data/code/tilert_play/glm5_tpot_pp_tp_估算.md` and lightly integrated as the baseline reasoning for the PP8 branch.

## 0. 结论先行

这个问题的核心不是 40B active params 怎么除,而是 **这一 token 的 active weight bytes 是否真的同时使用了 8 张卡的 HBM 带宽**。

如果按 8 卡 B200 聚合峰值算:

```text
42 GB / 64 TB/s ~= 0.66 ms/token ~= 1520 token/s
```

如果按 B200 单卡峰值算:

```text
42 GB / 8 TB/s ~= 5.25 ms/token ~= 190 token/s
```

如果按 7-8 ms TPOT 反推有效带宽:

```text
42 GB / 7.6 ms ~= 5.5 TB/s
```

所以不开 DFlash/MTP 时看到 7-8 ms TPOT,并不一定是数错了。它更像说明实际 critical path 没有吃到 8 卡聚合 HBM,而是在接近"单个 HBM 域的有效带宽"上运行。这里的 `5.5 TB/s` 不是 B200 峰值规格,而是由 7-8 ms 现象反推出来的有效带宽。

这对评估 PP 很关键: **PP 通信轮次少,但 bs=1 自回归 decode 的单 token 延迟通常也只能吃到单 stage/单卡带宽;TP 通信轮次多,但它的价值是让同一层的权重读取并行化,吃聚合 HBM。**

## 1. 估算口径

符号:

| 符号 | 含义 |
|---|---|
| `W_active` | 单 token 实际激活参数字节数,按 GLM5/GLM-5.1 文档取 40-42 GB |
| `B_gpu` | 单卡有效 HBM 带宽 |
| `N` | 参与同一 token 同一层计算的 GPU 数 |
| `B_agg` | 聚合 HBM 带宽,近似 `N * B_gpu` |

最简单的 memory-bound roofline:

```text
TPOT_compute ~= W_active / B_effective
```

其中 `B_effective` 不是机器总带宽,而是 critical path 上真实并发参与读 active weights 的带宽。

| 口径 | 公式 | TPOT | TPS |
|---|---:|---:|---:|
| 8 卡 B200 聚合峰值 | `42 GB / 64 TB/s` | `~0.66 ms` | `~1520 token/s` |
| B200 单卡峰值 | `42 GB / 8.0 TB/s` | `~5.25 ms` | `~190 token/s` |
| 7-8 ms 反推有效带宽 | `42 GB / 5.25-6.0 TB/s` | `~7-8 ms` | `~125-143 token/s` |

因此用户观察的"不开 DFlash 大概 130 TPS / 7-8 ms"和单卡有效带宽口径是自洽的,但低于 B200 单卡峰值 roofline。它不支持"base decode 已接近 8 卡聚合 bandwidth roofline"这个说法。

## 2. PP 与 TP 的本质差别

### TP: 通信多,但吃聚合 HBM

TP 把同一层的权重和计算切到多卡。对 bs=1 单 token decode,它的价值是让同一 token 的同一层在多张卡上并行读权重。

理想情况下:

```text
TPOT_compute_tp ~= W_active / (tp_size * B_gpu)
```

代价是每层都有同步点。按常见 transformer TP:

| 阶段 | 通信 |
|---|---|
| attention `o_proj` 后 | 1 次 allreduce 或等价 reduce/scatter |
| FFN/MoE down 后 | 1 次 allreduce 或等价 reduce/scatter |
| logits/top1/top-p | 末尾还可能有一次跨卡规约或采样通信 |

对 GLM5 DSA 路径还要加:

| 阶段 | 通信 |
|---|---|
| GPU0 sparse index 到 GPU1-7 | 每层 1 次 selected-index P2P packet |
| attention output | 每层 1 次 fused allreduce |
| FFN/MoE down | 每层 1 次 fused allreduce |

所以粗略轮次:

```text
heavy collective rounds ~= 2 * num_layers
selected-index packet rounds ~= num_layers
tail sampling/logits ~= O(1)
```

GLM5 `num_layers = 78` 时:

```text
heavy allreduce ~= 156 次/token
selected-index P2P ~= 78 次/token
```

通信量本身不一定大,因为 hidden 激活很小:

```text
hidden bytes ~= 6144 * 2 = 12 KB/token
```

但轮次很多,每层都是 dependency edge。TileRT 把 allreduce/P2P 融进 graph 内 physical op,主要是在砍这些同步和 launch 边界。

### PP: 通信少,但 bs=1 单 token 吃不到聚合 HBM

PP 把层切到不同 stage。单 token forward 只需要在 stage 之间传 hidden:

```text
pp_comm_rounds ~= pp_size - 1
pp_comm_bytes ~= (pp_size - 1) * hidden_bytes
```

例如 `pp_size=8`:

```text
rounds ~= 7
bytes ~= 7 * 12 KB ~= 84 KB/token
```

这比 TP 每层 allreduce 少很多。这里先不讨论 pipeline 是否填满,因为目标是单请求 bs=1 的极致 TPOT,不是多请求 throughput。对 TPOT 来说,关键是同一个 token 在 PP stage 之间存在严格依赖:

```text
stage i+1 必须等 stage i 输出 hidden
```

因此单 token critical path 是所有 PP stage 串行相加:

```text
stage0 跑 token t 的前几层
stage1 跑 token t 的中间层
...
last stage 采样出 token t+1
stage0 才能开始 token t+1
```

```text
TPOT_compute_pp ~= sum_i W_stage_i / B_gpu
                 ~= W_active / B_gpu
```

这会落回单卡带宽 roofline:按 B200 峰值约 5.25 ms,按用户观察反推的有效带宽约 7-8 ms。

所以 PP 的优点是:

| 优点 | 说明 |
|---|---|
| 通信轮次少 | `pp_size - 1` 个 activation send,不是每层 2 次 allreduce |
| 通信体积小 | hidden 只有十几 KB/token |
| 工程上更少 collective pressure | 不需要 156 个 layer-level allreduce |

PP 的缺点是:

| 缺点 | 说明 |
|---|---|
| 单 token latency 不吃聚合 HBM | stage 串行,critical path 接近单卡读完整 active weights |
| 不能靠 stage overlap 降低同一 token TPOT | stage 间是数据依赖,不是可并行分支 |
| stage 负载必须极准 | 最慢 stage 直接决定局部瓶颈,层数/MoE active bytes 不均会放大尾部 |

## 3. PP 是否还有可能

PP 不是没价值,但目标要分清。

如果目标是 **单条请求 bs=1 极致 TPOT**:

```text
TP 的优势:同一层并行读权重,compute roofline 更低。
PP 的优势:通信轮次少很多,可以砍掉 TP 的 per-layer collective gap。
```

所以 PP 是否可能赢,不是看 pipeline occupancy,而是看这个不等式:

```text
TPOT_pp ~= W_active / B_gpu_eff + (pp_size - 1) * L_send

TPOT_tp ~= W_active / (tp_size * B_gpu_eff)
          + num_layers * (L_attn_comm + L_ffn_comm + L_selected_index)
          + graph/runtime gap

PP win iff TP 的通信/gap > PP 丢掉聚合 HBM 带来的 compute 增量。
```

按 B200 峰值粗算,PP 相对 8 卡聚合 TP 丢掉的 compute roofline 约:

```text
42 GB / 8 TB/s - 42 GB / 64 TB/s
~= 5.25 ms - 0.66 ms
~= 4.6 ms
```

如果按 7-8 ms 单卡有效带宽口径,这个差距会更大。因此 PP 要在单请求 TPOT 上赢,TP 那边每 token 的 156 次 allreduce + 78 次 selected-index P2P + runtime gap 必须吃掉数毫秒级预算。

### `L_send` 在 NVLink 下怎么估

PP stage boundary 传的是 hidden,量级非常小:

```text
hidden_bytes = 6144 * sizeof(bf16) = 12,288 B ~= 12 KB
```

B200/DGX B200 的 NVLink 口径:

```text
DGX B200 aggregate NVLink bandwidth = 14.4 TB/s
per GPU bidirectional NVLink ~= 1.8 TB/s
per GPU one-way 粗略按 ~= 0.9 TB/s
```

所以只看带宽搬运时间:

```text
12 KB / 1.8 TB/s ~= 0.007 us
12 KB / 0.9 TB/s ~= 0.014 us
```

即使 `seq_len=4` 一次传 4 个 hidden:

```text
48 KB / 0.9 TB/s ~= 0.055 us
```

因此 `L_send` 不能按 payload/bandwidth 估成主项。NVLink 上 12KB hidden send 的主项是固定开销:

```text
L_send ~= L_enqueue_or_graph_node
        + L_remote_store_or_copy_setup
        + L_visibility_fence
        + L_receiver_wait
        + payload_bytes / B_nvlink
```

估算时建议用三档:

| 实现方式 | 单次 `L_send` 估计 | 7-stage PP 合计 |
|---|---:|---:|
| graph 内自研 P2P packet/store + flag | `~1-3 us` | `~7-21 us` |
| `cudaMemcpyPeerAsync`/小 kernel copy + event | `~5-10 us` | `~35-70 us` |
| NCCL send/recv 或通用 runtime 路径 | `~10-20+ us` | `~70-140+ us` |

这个量级和 PP/TP 的 compute roofline 差距相比很小。按 B200 峰值,PP 相对 8 卡 TP 丢掉的并行读权重收益约 `4.6 ms`;7 次 PP send 即使用 `20 us` 估也只有 `0.14 ms`。

因此对 bs=1 极致 TPOT,PP 的 stage 间 send 大概率不是瓶颈。真正要比较的是:

```text
PP 丢掉 8 卡并行读权重的 4-7 ms
vs
TP 每 token 156 次 allreduce + 78 次 selected-index P2P + runtime gap
```

倒过来算,如果 PP 要靠少通信赢回 B200 峰值口径的 `~4.6 ms`,TP 侧每个 layer-level 通信/gap edge 的平均成本要达到:

```text
4.6 ms / (156 + 78) ~= 20 us / edge
```

如果 TileRT 的 fused P2P/allreduce 已经把每个 edge 压到个位数微秒,PP 不一定赢。如果现有 TP 实现用通用 NCCL/event/runtime,每个 edge 接近十几到几十微秒,PP 就可能有空间。

### PP 能不能 mega 化

可以。PP mega 化的目标不是填满 pipeline,而是把 stage boundary 从 host/runtime/NCCL 小消息路径里拿掉,变成 graph 内 P2P handoff。

理想形态:

```text
prepare:
  每个 stage 分配固定 input/output hidden buffer
  cudaDeviceEnablePeerAccess
  下游 input buffer 地址写进上游 resource table
  每个 stage capture 一张 CUDA Graph

decode:
  stage0 graph replay
    -> 最后一个 kernel/packet kernel 通过 NVLink P2P 写 stage1 input buffer
    -> system-scope store 写 flag epoch
  stage1 graph 内 receive/wait flag
    -> 读 input hidden
    -> 跑本 stage layers
    -> P2P 写 stage2
  ...
  last stage head/sample 写 TOKEN_OUT
```

这和 TileRT 现在 selected-index 的 P2P packet 思路很像,只是 payload 从 `IDX_SELECTS` 换成 hidden activation。

关键点:

| 组件 | 作用 |
|---|---|
| peer buffer | 每个 stage 暴露下一 stage 的 input hidden 地址 |
| packet/copy kernel | 上游用 remote store 或 P2P copy 写下一卡显存 |
| flag/epoch | 下游 graph 内 spin/wait,保证读到本轮 hidden |
| static buffer | graph replay 时地址不变,只改 epoch/position |
| per-stage graph | host 只触发 replay,stage 间依赖在 device/NVLink 上解决 |

这样可以把 `L_send` 从通用 runtime 路径压到 graph 内 P2P packet 的固定延迟区间。对 12KB hidden,带宽项几乎为零,主项就是 flag 和同步。

但它不能改变这个事实:

```text
stage1 必须等 stage0 hidden
stage2 必须等 stage1 hidden
```

所以 PP mega 化能消掉 stage boundary gap,不能把同一个 token 的多个 stage 变成并行计算。它的收益上限大概是:

```text
省掉 PP stage 间 host/event/NCCL 小消息开销
```

不是:

```text
吃到 8 卡聚合 HBM 读同一层权重
```

### NVLink P2P 是什么

NVLink P2P 指的是 GPU 之间可以直接访问彼此显存。常见层级:

| 路径 | 说明 | 适合度 |
|---|---|---|
| `cudaMemcpyPeerAsync` | runtime 发起 P2P copy,走 NVLink/NVSwitch | 能用,但小消息固定开销偏大 |
| P2P kernel remote load/store | kernel 直接 `LDG/STG` peer GPU 地址 | PP mega 化最适合,可放进 graph |
| packet + flag | payload 和 epoch 一起写,下游轮询 expected flag | 最像 TileRT selected-index |
| NCCL send/recv | 通用通信库路径 | 可靠但对 12KB hidden 可能太重 |

PP hidden handoff 最应该用第三种:

```text
上游: STG.E.STRONG.SYS 写 peer input buffer + flag
下游: LDG.E.STRONG.SYS 轮询 flag,再读 hidden
```

这需要处理几个细节:

| 问题 | 处理 |
|---|---|
| 可见性 | system-scope store/load 或合适 fence |
| buffer 覆盖 | double buffer 或 epoch ring,避免下一轮覆盖上一轮 |
| graph 静态地址 | input/output buffer 地址 prepare 阶段固定 |
| 死等 | expected epoch 必须单调,reset 时清状态 |
| 多 stage | 每条边一套 peer buffer + flag |

所以答案是: **NVLink P2P 正是 PP mega 化的实现工具**。它能把 PP 的 `L_send` 压到很低;但 PP 是否赢,仍要看 TP 的 per-layer allreduce/P2P/gap 是否真的大到超过 PP 丢失聚合 HBM 的 4-7ms。

如果目标是 **多请求吞吐**:

```text
PP 可以用多个 sequence 填 pipeline,通信压力比 TP 小,可能更好。
```

如果目标是 **结合 speculative/MTP**:

```text
PP 可能重新有空间,因为一次 verify 有 seq_len=2/4 或更多 draft token,
可以给 pipeline 更多并发工作。
```

但这已经不是纯 bs=1 单 token PP,而是:

```text
PP + speculative window
PP + 多请求 microbatch
PP + stage 内 TP
```

更现实的混合形态可能是:

| 并行方式 | 作用 |
|---|---|
| stage 内 TP | 保留同一层的聚合 HBM |
| stage 间 PP | 减少全模型范围的 collective,把通信变成少数 activation send |
| speculative window | 给 PP 填 pipeline 的 token 级并发 |

## 4. 当前判断

`42 GB / 64 TB/s ~= 0.66 ms` 这种 8 卡 Blackwell 聚合 roofline 只有在一个前提下成立:

```text
同一 token 的 active weights 被 8 张卡并行读取,且通信/调度 gap 足够小。
```

`7-8 ms` 这个 TPOT 则说明实际更接近:

```text
同一 token critical path 只吃到单卡级有效带宽,而且没有达到 B200 单卡峰值。
```

因此,评估 PP 时不能只看"PP 通信次数更少"。PP 的通信确实少,但它用通信少换掉了 TP 最重要的东西:同一 token 同一层的 HBM 并行读权重。

下一步需要实测确认的不是"PP 通信少不少",而是:

1. base decode 不开 DFlash/MTP 的真实 TPOT。
2. 每层/每阶段 GPU active 时间是否重叠。
3. HBM throughput 是接近单卡还是 8 卡聚合。
4. TP allreduce/P2P 的 round-trip latency 占比。
5. MTP verify step latency 和 accepted length 的乘积收益。

只有这些数据齐了,才能判断 PP、TP、PP+TP、PP+MTP 哪个方向值得写。

## 5. DFlash/MTP 为什么会显得很重要

DFlash/MTP 本质上不是把一次 base replay 变成 1 ms,而是让一次较贵的 verify/replay 产出多个 accepted token。这个结论应该后置到 PP/TP roofline 之后看:如果 base decode 的 critical path 接近单卡有效带宽,那 speculative 接受长度就会成为有效 TPS 的主要放大器。

如果不开 DFlash:

```text
step_tpot ~= 7.5 ms
accepted ~= 1
effective_tpot ~= 7.5 ms
effective_tps ~= 133
```

如果 MTP 平均接受长度 `a = 3.2`:

```text
effective_tpot ~= step_tpot / a
```

例子:

| verify step latency | accepted | effective TPOT | effective TPS |
|---:|---:|---:|---:|
| 8.0 ms | 3.2 | 2.50 ms | 400 |
| 7.0 ms | 3.2 | 2.19 ms | 457 |
| 6.0 ms | 3.2 | 1.88 ms | 533 |

所以如果 base decode 只有 100 多 TPS,最终 500 TPS 级别很可能确实主要来自 speculative/MTP 接受长度,再叠加 verify 图内部的优化。

这里需要实测拆分:

| 项 | 状态 |
|---|---|
| 不开 DFlash/MTP 的 base TPOT | 用户观察约 7-8 ms,需要固定配置复测 |
| MTP verify step latency | 待测 |
| 平均 accepted length | 文档里有约 3.2 的说法,需要同上下文长度复测 |
| effective TPS | 不能直接和 base TPOT 混算,要按 accepted token 归一 |

## 6. 带宽来源

NVIDIA DGX B200 官方规格:8x Blackwell GPUs, GPU memory `1,440 GB total`, `64 TB/s HBM3e bandwidth`。因此本文把 B200 单卡峰值按 `64 / 8 = 8 TB/s` 估算。

历史博客里常见的 `38 TB/s` 不是本文 Blackwell/B300 对比的目标口径。评估当前 B200/B300 机器时,优先使用 Blackwell 8 卡 `64 TB/s` 级别聚合峰值,再用实测 TPOT 反推有效带宽。

## 7. NVL72 相比普通 8x B300 的 NVLink 有没有更快

本文只比较 Blackwell/Blackwell Ultra 内部形态: **NVL72 vs 普通 8x B200/B300 NVSwitch 节点**。结论是:NVL72 的 NVLink domain 更大,但单 GPU 注入带宽没有比普通 8 卡 B200/B300 NVSwitch 节点更快,仍是约 `1.8 TB/s/GPU bidirectional`。

官方规格口径:

| 系统 | GPU 数 | HBM 带宽 | NVLink 带宽 |
|---|---:|---:|---:|
| DGX B200 | 8 | `64 TB/s` total | `14.4 TB/s` aggregate |
| DGX B300 | 8 | B300 单卡仍按 `~8 TB/s` 量级看 | `14.4 TB/s` aggregate |
| GB200 NVL72 | 72 | `576 TB/s` total | `130 TB/s` total,`3.6 TB/s` per Grace Blackwell Superchip |
| GB300 NVL72 | 72 | `576 TB/s` total 量级 | `130 TB/s` total 量级 |

这些数除一下会发现同一代 Blackwell/Blackwell Ultra 的 per-GPU NVLink 注入带宽基本还是:

```text
14.4 TB/s / 8  ~= 1.8 TB/s per GPU
130 TB/s / 72 ~= 1.8 TB/s per GPU
```

所以如果只取 NVL72 里的 4 张或 8 张 GPU 做一个 bs=1 low-latency request,它不会因为在 NVL72 rack 里就获得比普通 8x B300 更高的 per-GPU NVLink 注入带宽。NVL72 的新增价值是:

```text
把 72 张 GPU 放进同一个 NVLink/NVSwitch domain,
避免 8-GPU node 之间掉到 InfiniBand/Ethernet 级别。
```

这对这些场景有价值:

| 场景 | NVL72 价值 |
|---|---|
| 模型/KV/cache 必须跨超过 8 GPU | 很大,因为仍在 NVLink domain 内 |
| 大 batch / 多请求吞吐 | 很大,可以把更多 GPU 当一个 rack-scale pool |
| 训练或大规模 all-to-all | 很大 |
| 单请求 bs=1,只需要 4/8 GPU | 小,甚至可能不如普通 8-GPU 节点划算 |

对本文目标,也就是 **bs=1 极致 TPOT**,NVL72 的 72 卡互联不是直接收益。最优策略一般是:

```text
用尽可能少的 GPU 覆盖模型 active working set,
并保证这些 GPU 在同一个低延迟 NVLink island 内。
```

如果 4/8 张 B300 已经能放下 active weights/KV,并且 PP mega 化只需要 stage 间传 12KB hidden,那么 NVL72 的 72-GPU domain 不会明显降低 `L_send`;`L_send` 已经由 fixed latency 主导,不是带宽主导。NVL72 真正避免的是"超过 8 GPU 后跨节点通信变慢"这个问题。

### Hopper 开发口径

可以先用 Hopper/H100/H200 开发 mega PP 的 P2P handoff。CUDA 接口是同一套:

```text
cudaDeviceCanAccessPeer
cudaDeviceEnablePeerAccess
UVA peer pointer
kernel remote store/load peer allocation
__threadfence_system + flag/epoch
```

需要注意的是性能口径不同:

| GPU 代际 | NVLink per GPU bidirectional | 对 12KB hidden 的影响 |
|---|---:|---|
| Hopper H100/H200 | `~900 GB/s` | `12KB / 900GB/s ~= 0.014 us` |
| Blackwell B200/B300 | `~1.8 TB/s` | `12KB / 1.8TB/s ~= 0.007 us` |

payload 带宽项在两代上都远小于微秒,所以 Hopper 可以很好地验证:

```text
P2P peer access 是否通
remote store + flag 协议是否正确
fixed latency 是 1us / 5us / 10us 哪个量级
多 stage 串起来有没有长尾
```

最终 B200/B300 上仍要复测,因为 NVLink 代际、NVSwitch、GPU clocks、driver 和 system-scope memory ordering 开销都会影响尾延迟。

## 8. 如何测 mega PP 的 `L_send`

NVIDIA/生态里有现成 baseline 工具,但不能直接替代 mega PP microbench。

| 工具 | 测到什么 | 用途 |
|---|---|---|
| CUDA sample `p2pBandwidthLatencyTest` | GPU-GPU P2P bandwidth/latency | 确认 peer access/NVLink 拓扑是否正常 |
| NVIDIA `nvbandwidth` | GPU/CPU/GPU 间 bandwidth 与 latency | 更系统地扫 NVLink/PCIe/内存路径 |
| `nccl-tests` | NCCL allreduce/sendrecv 等 collective/P2P 路径 | 测通用通信库小消息 latency,作为反例或 baseline |
| 自写 packet+flag microbench | remote store + fence + flag + receiver wait | 这才是 mega PP 的 `L_send` |

### 8.1 先测机器 baseline

先确认拓扑:

```bash
nvidia-smi topo -m
```

再跑 NVIDIA/CUDA baseline:

```bash
# CUDA samples
./p2pBandwidthLatencyTest

# NVIDIA nvbandwidth
./nvbandwidth

# NCCL 小消息路径,扫 8B 到 1MB
./sendrecv_perf -b 8 -e 1M -f 2 -g 2
./all_reduce_perf -b 8 -e 1M -f 2 -g 8
```

这些数只回答:

```text
这台机器的 P2P/NVLink/NCCL 有没有坏;
通用 runtime/NCCL 小消息大概多慢。
```

它们不回答:

```text
graph 内 remote store hidden + flag handoff 到底几微秒。
```

### 8.2 测 `cudaMemcpyPeerAsync` 小消息

第二层测 memcpy peer:

```text
for size in {4KB, 12KB, 48KB, 64KB, 256KB}:
  src stream:
    cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, size)
  event timing over many iterations
```

再测 captured graph 版本:

```text
capture:
  cudaMemcpyPeerAsync(...)
instantiate graph
for many iterations:
  cudaGraphLaunch(graph)
sync
```

这个给出:

```text
runtime P2P copy 小消息固定开销
graph replay 是否明显降低 launch/enqueue 开销
```

但它仍然不是最理想的 mega PP,因为 hidden handoff 可以不走 memcpy node,而是直接由上游 kernel remote store 到下游 buffer。

### 8.3 真正要测:kernel remote store + flag ping-pong

mega PP 的 stage boundary 应该这样测:

```text
GPU0: write 12KB hidden to GPU1 peer buffer
GPU0: system-scope fence/store flag epoch
GPU1: wait flag epoch
GPU1: read hidden / optional checksum
GPU1: write ack back to GPU0
GPU0: wait ack
```

用 round-trip ping-pong 的原因:跨 GPU 时钟不一定可靠同步。让 GPU0 自己测:

```text
t0 = GPU0 globaltimer
GPU0 -> GPU1 hidden + flag
GPU1 -> GPU0 ack
t1 = GPU0 globaltimer
RTT = t1 - t0
one_way ~= RTT / 2
```

这个 microbench 最接近实际 `L_send`,因为它包含:

```text
remote store payload
system-scope visibility
flag epoch
receiver polling
ack path
```

建议扫这些变量:

| 变量 | 取值 |
|---|---|
| payload | `4KB, 12KB, 24KB, 48KB, 64KB` |
| buffer | single buffer vs double buffer/ring |
| store width | scalar / 64-bit / 128-bit vectorized store |
| flag | same cacheline vs separate cacheline |
| wait | spin load frequency, backoff/no backoff |
| path | adjacent GPU pairs, all pair matrix |
| mode | standalone kernel loop vs captured graph |

输出至少要有:

```text
p50 / p90 / p99 one-way latency
payload bandwidth
pair matrix
flag wait retries
```

### 8.4 最终端到端测法

最后要测一个最小 PP graph:

```text
GPU0 graph:
  dummy layer kernel, burn X us
  p2p_send_hidden_kernel

GPU1 graph:
  wait_hidden_kernel
  dummy layer kernel, burn Y us
  p2p_send_hidden_kernel

...
```

扫 `pp_size = 2/4/8`,payload = `12KB/48KB`,dummy compute = `0/50/100/500us`。这样能看出:

```text
stage handoff 是否真的只有个位数 us;
多个 stage 串起来是否有长尾;
graph replay + P2P flag 会不会出现抖动;
```

如果这个最小 PP graph 的 7 次 handoff 仍只有几十微秒,那 `L_send` 就不是 PP 的主要风险。下一步就该测真实 layer stage 的 HBM throughput 和 stage balance。

### 8.5 jiuzhang H200 node37 实测 baseline

测试位置:

```text
cluster: jiuzhang
node: host-172-31-13-37 / 172.31.13.37
GPU: 8x H200, GPU-GPU topo = NV18
binary: /tmp/p2p_lsend/p2p_lsend
build: /usr/local/cuda-12.8/bin/nvcc -arch=sm_90
```

关键命令:

```bash
./p2p_lsend --src 0 --dst 1 --scan --touch-payload --iters 50000 --warmup 5000
./p2p_lsend --src 0 --dst 4 --bytes 12288 --touch-payload --iters 50000 --warmup 5000
```

`--touch-payload` 表示接收端 wait flag 后实际读取 payload,更接近下游 stage 读 hidden 的场景。

结果摘要:

| pair | bytes | rtt p50 | rtt p99 | rtt p999 | half-rtt avg | 备注 |
|---|---:|---:|---:|---:|---:|---|
| GPU0→GPU1 | 0 | `2.43 us` | `2.66 us` | `3.30 us` | `1.25 us` | flag+ack baseline |
| GPU0→GPU1 | 12KB | `4.42 us` | `4.61 us` | `5.12 us` | `2.23 us` | GLM5 bf16 hidden |
| GPU0→GPU1 | 48KB | `9.12 us` | `9.28 us` | `10.02 us` | `4.47 us` | seq_len=4 hidden |
| GPU0→GPU4 | 12KB | `4.45 us` | `4.64 us` | `5.22 us` | `2.25 us` | 跨 NUMA 组,仍是 NV18 |

完整 scan(`GPU0→GPU1`, touch payload):

| bytes | rtt avg | rtt p50 | rtt p90 | rtt p99 | rtt p999 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `2.490` | `2.432` | `2.624` | `2.656` | `3.296` | `3.616` |
| 4096 | `3.272` | `3.328` | `3.360` | `3.392` | `4.064` | `4.288` |
| 12288 | `4.461` | `4.416` | `4.576` | `4.608` | `5.120` | `5.504` |
| 24576 | `5.803` | `5.824` | `5.856` | `6.016` | `6.560` | `6.752` |
| 49152 | `8.935` | `9.120` | `9.184` | `9.280` | `10.016` | `10.848` |
| 65536 | `11.004` | `11.104` | `11.136` | `11.264` | `11.968` | `12.768` |
| 262144 | `34.429` | `34.528` | `34.560` | `34.752` | `35.424` | `36.352` |

解释:

```text
12KB hidden handoff 的 one-way proxy ~= half RTT ~= 2.2-2.3 us
48KB seq_len=4 hidden handoff 的 one-way proxy ~= 4.5 us
7-stage PP 的 12KB handoff 合计 ~= 7 * 2.3 us ~= 16 us
```

所以在 H200/NV18 上,mega PP 的 stage-boundary P2P handoff 本身已经是十几微秒总量级,不是毫秒级风险。B200/B300 上需要复测,但由于 12KB payload 仍由 fixed latency 主导,预期不会比这个更差到改变 PP/TP 的毫秒级判断。
