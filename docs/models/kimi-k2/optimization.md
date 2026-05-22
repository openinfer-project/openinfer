# Kimi-K2 Optimization

> **TL;DR:** Kimi-K2.5/2.6 text-only on H20 ×8（当前 TP8 + EP8）。**重点是 decode 性能**：bs4 synthetic output64 steady TPOT avg `14.39ms` / p99 `14.83ms`（≈`278 tok/s`），目标 `decode(bs4) > 300 tok/s`；下一阶段迁到 TP1 + DP8 + EP8（PPLX dispatch/combine）。Decode 路径已经 CUDA Graph 覆盖整段 + Marlin WNA16 routed expert + NCCL RS bridge，真实 vLLM fixture prompt output16 四路 greedy 完全匹配。Prefill 优先级低，短 prompt streaming TTFT `1995.5ms` 待优化但不影响 decode 主线。
>
> **Last touched:** 2026-05

## Goal

PegaInfer Kimi-K2 端到端延迟和吞吐在同 H20 ×8 配置上达到或超过 vLLM 0.19.0 baseline，并保留 greedy token-id parity 作为 keep/revert 硬 gate。**当前重点是 decode 性能**，prefill 与 decode 主线并行改，但不优先。

阶段路线：

1. 当前 TP8 + EP8 形态下把 bs4 decode 推到 `> 300 tok/s`（剩余空间集中在 collective fanout / shared 通信 overlap）。
2. 迁到 **TP1 + DP8 + EP8**（PPLX dispatch/combine），消除 MLA / dense / shared expert 的跨 rank TP all-reduce，throughput 拿 DP 乘数。这是结构改动，需要重新过 greedy parity gate。
3. vLLM `bench serve` H20 ×8 baseline 采集后填齐 dashboard，做横向 keep/revert 判断。

## Status

| 区域 | 当前状态 |
| --- | --- |
| Correctness | ✓ vLLM K2.5 fixture prompt（27 tok）`max_tokens=1/2/8/16` 四并发 greedy token ids 全对；多 prompt vLLM gate 4/4 argmax match，top-20 id overlap 最低 `19/20`。 |
| Decode TPOT (bs4) | ✓ Graph mode synthetic output64 steady avg `14.39ms` / p99 `14.83ms`，真实 fixture output16 steady avg `14.13ms` / p99 `14.26ms`。 |
| Prefill TTFT | ⚠ 优先级低。短 prompt（~30 tok）streaming chat 实测 `1995.5ms`，无稳定 perf gate；per-layer allocation、首个 collective stream drain、host-visible top1 待收敛。不影响 decode 主线。 |
| TP / EP collectives | ⚠ 当前是 NCCL bridge：TP hidden 走 BF16→F32→BF16 桥，MoE routed combine 走 `repeat_f32 + reduce_scatter`。Greedy parity-driven，不是 PPLX EP。下一阶段迁到 TP1 + DP8 + EP8 + PPLX。 |
| CUDA Graph | ✓ 整段 decode 已 capture/replay；prompt prefill 路径还未 graph 化。 |
| Routed expert backend | ✓ vLLM Marlin WNA16（INT4 + BF16 scale），bit-parity 单层 W13 + SwiGLU + W2 + topk sum 对 vLLM reference 0-diff。CUTLASS example69 INT4 probe 已下线。 |

## E2E Dashboard

GPU: 8× NVIDIA H20。Model: Kimi-K2.5 (Kimi-K2.6 同架构权重待发，K2.5 是当前 H20 验证路径)。vLLM: 0.19.0。Concurrency 默认 bs4（参见 [operator-todo.md](operator-todo.md) `KIMI_RUNNER_MAX_BATCH` 硬上限契约）。

| Profile | Metric | pegainfer | vLLM | delta |
| --- | --- | --- | --- | --- |
| short-prompt streaming (~30 tok in, free out) | TTFT | `1995.5ms` | TBD | — |
| short-prompt streaming (~30 tok in, free out) | TPOT | `14.48ms` (≈30.8 tok/s) | TBD | — |
| bs4 synthetic (27 in, 64 out, --cuda-graph) | TPOT avg | `14.39ms` | TBD | — |
| bs4 synthetic (27 in, 64 out, --cuda-graph) | TPOT p99 | `14.83ms` | TBD | — |
| bs4 real fixture (27 in, 16 out, --cuda-graph) | TPOT avg | `14.13ms` | TBD | — |
| bs4 real fixture (27 in, 16 out, --cuda-graph) | TPOT p99 | `14.26ms` | TBD | — |

口径备注：

- 这些 TPOT 数字来自 `target/release/bench_serving request --cuda-graph true ...`，已过四并发 vLLM fixture greedy gate，不会被 prompt prefill 吃掉。
- 短 prompt streaming TTFT 是 OpenAI-compatible `/v1/completions` 端到端窗口（含 first-collective stream drain、scheduler、frontend），不是纯 prefill GPU time；prefill 阶段拆分还没开始（见 Open 章节）。
- vLLM TP8 EP8 H20 `bench serve` 还没采，所以 delta 暂记 TBD。同架构 K2.5 vLLM 单节点 8 卡跑 fixture `max_tokens=16` 的 wall ≈ `4.6s / 4 并发 / 16 token`，跟 PegaInfer steady ≈ 13.86 tok/s 同量级，但口径不一致（HTTP vs bench_serving），不放进对比表。

## Architecture

| 项 | 值 |
| --- | --- |
| Layers | `61`（`1` dense + `60` MoE） |
| `hidden_size` | `7168` |
| `vocab_size` | `163,840` |
| `max_position_embeddings` | `262,144` |
| YARN RoPE | `theta=50_000`, `factor=64`, original `4096`, `beta_fast/slow=32/1` |

MLA attention：

| 项 | 值 |
| --- | --- |
| `num_attention_heads` | `64`（TP8: 8 per rank） |
| `q_lora_rank` | `1536`，q down/up split |
| `kv_lora_rank` | `512`，kv down 共享 `compressed_kv [512] + k_rope [64]` |
| `qk_nope_head_dim / qk_rope_head_dim` | `128 / 64` |
| `v_head_dim` | `128` |
| Decode KV cache shape | latent paged `[ckv=512, kpe=64]` per token，按 page 内存常驻 |

MoE：

| 项 | 值 |
| --- | --- |
| Routed experts | `384`（EP8: 48 per rank） |
| `num_experts_per_tok` (top-k) | `8` |
| Shared experts | `1` |
| Expert intermediate | `2048` |
| Dense layer intermediate (layer 0) | `18,432` |
| Routed expert quant | INT4 + BF16 scale, `group_size=32`, vLLM Marlin WNA16 layout |
| Dense / shared expert dtype | BF16 |
| Router | `noaux_tc` (sigmoid + group top-k)，`routed_scaling_factor=2.827`，`norm_topk_prob=true` |

Sharding：

- TP=8：MLA q_b / kv_b head 切，o_proj / dense gate/up/down 行/列切，LM head vocab 切。
- EP=8：routed expert 按 `384 / 8 = 48` 个本地 expert 切，router 仍 replicate。
- TP hidden all-reduce 当前走 `BF16 → F32 → reduce → BF16` bridge（greedy parity 需要）。
- MoE routed combine 走 `repeat_f32 + NCCL reduce_scatter` bridge（不是 PPLX EP）。

## Per-Layer DAG — Decode

层 0 是 dense，层 1..60 是 MoE。两条路径共享同一个 MLA attention 段。

### MLA + MoE Layer（layer 1..60）

```
RMSNorm input
  → fused_qkv_a GEMM [7168 → q_lora=1536 + kv_lora=512 + k_rope=64]   ← graph-safe BF16 GEMM
  → kimi_mla_split_qkv_a                                              ← 切出 q_a / compressed_kv / k_rope
  → RMSNorm(q_a)                                                      ← q branch norm
  → q_b GEMM [1536 → heads × (qk_nope + qk_rope) = 64 × 192 = 12288]
  → RMSNorm(compressed_kv)
  → kimi_mla_rope_split_decode                                        ← YARN RoPE on q_pe / k_pe，输出 q_nope / q_pe / append_kpe
  → kimi_mla_absorb_q_nope                                            ← q_nope @ W_UK_T，preloaded kv_b 权重
  → kimi_mla_paged_kv_append                                          ← 写 ckv [512] + kpe [64] 进 paged cache
  → kimi_flashinfer_batch_decode_mla                                  ← FlashInfer MLA decode kernel
  → kimi_mla_v_up                                                     ← latent @ W_UV
  → o_proj GEMM [heads × v_head_dim = 64 × 128 = 8192 → 7168]
  → TP all-reduce (BF16-via-F32 bridge)                               ← 3 kernels: cast + NCCL + cast
  → residual add + post-attention RMSNorm
  → kimi_router_noaux_tc                                              ← sigmoid + group top-k, 384 experts → top-8
  ┌───────────────────────┐                              ┌──────────────────────────────┐
  │ shared expert path    │                              │ routed expert path (EP local) │
  │ ─────────────────────  │                              │ ──────────────────────────── │
  │ shared gate/up GEMM   │                              │ kimi_moe_marlin_align_block_size │
  │   (fused, BF16)       │                              │ kimi_marlin_wna16_w13_gemm   │
  │ silu_mul_fused        │                              │ kimi_marlin_w13_swiglu       │
  │ shared down GEMM      │                              │ kimi_marlin_wna16_w2_gemm    │
  │ TP all-reduce (bridge)│                              │ kimi_marlin_sum_topk_rows_f32│
  └───────────┬───────────┘                              └──────────────┬───────────────┘
              │                                                          │
              │                                                          repeat_f32_for_reduce_scatter
              │                                                          NCCL reduce_scatter
              │                                                          │
              └────────────► kimi_scaled_add_f32_bf16_to_bf16 ◄──────────┘
                              (shared_bf16 + routed_f32 * routed_scaling_factor + residual)
```

### Dense Layer 0

跟 MLA + MoE layer 共享 attention 段，MLP 部分换成单一 dense MLP：

```
post-attention RMSNorm
  → dense gate GEMM  [7168 → 18432]
  → dense up   GEMM  [7168 → 18432]
  → silu_mul_batch  [18432]
  → dense down GEMM  [18432 → 7168]
  → TP all-reduce (BF16-via-F32 bridge)
  → residual add
```

Dense layer gate/up 还没 fuse（只 shared expert fuse 了，参见 Optimization Log #5）。

### Decode 路径整体 call count（H20 static trace，bs4 / kv1024）

```
calls 1766
307 gemm_graphsafe
245 rms_norm_batch
123 all_reduce
122 add_batch
120 kimi_marlin_wna16_gemm
61  kimi_mla_split_qkv_a
61  kimi_mla_rope_split_decode
61  kimi_mla_absorb_q_nope
61  kimi_mla_paged_kv_append
61  kimi_flashinfer_batch_decode_mla
61  kimi_mla_v_up
61  silu_mul_batch
60  kimi_router_noaux_tc
60  kimi_moe_marlin_align_block_size
60  kimi_marlin_w13_swiglu
60  kimi_marlin_sum_topk_rows_f32
60  repeat_f32_for_reduce_scatter
60  reduce_scatter
60  kimi_scaled_add_f32_bf16_to_bf16
1   embedding_batch_vocab_shard
1   top1_batch
```

`all_reduce=123 = 61 attention + 60 MoE final + 1 embedding + 1 dense`。`reduce_scatter=60` 是 MoE routed combine bridge。

## Operator Performance

口径说明：当前测量分两种。

1. **Strong-sync profile**（H20 nsys, bs4, output64, --concurrency 4）：每个 collective 边界都强同步，更接近真实 rank skew + tail。
2. **Graph-replay TPOT**（`bench_serving --cuda-graph true`）：整段 decode 走 CUDA Graph，host enqueue collapse 掉，是当前 keep/revert gate。

### Strong-sync profile（bs4 steady step ≈ `35.0ms`）

| 组件 | Time/step | % |
| --- | --- | --- |
| MoE total | `22.8ms` | 65% |
| ├─ shared expert + TP all-reduce | `6.55ms` | — |
| ├─ routed reduce/add + f32 all-reduce | `6.37ms` | — |
| ├─ router (`kimi_router_noaux_tc`) | `3.70ms` | — |
| ├─ Marlin W13 + W2 | `~4.0ms` | — |
| └─ align (`kimi_moe_marlin_align_block_size`) | `1.31ms` | — |
| MLA + 其它 | `~12ms` | 35% |
| **Total** | **`~35ms`** | **100%** |

Tail（nsys p99/max）：BF16 all-reduce `p50=74.7us / p99=780us / max=2.98ms`；F32 all-reduce `p50=64.8us / p99=385us / max=886us`；Marlin WNA16 `p50=14.3us / p99=154us / max=187us`；`cuStreamSynchronize` `p50=28.3us / p99/max=9.87ms`。Tail 由 rank arrival skew + API drain 主导，不是单 kernel 慢。

### Graph-replay snapshot（bs4 synthetic output64, --cuda-graph）

整段 decode 走 graph replay，p50 ≈ p99，整体 `14.39ms / step` 比 strong-sync 的 `35ms` 低一半多：差距主要来自 host enqueue 折叠（`cuGraphLaunch count=504 = 8 ranks * 63 steps`），不是 kernel 本身变快。这是当前 keep/revert gate 口径。

| Profile | TPOT avg | p50 | p95 | p99 | max | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic 27→64 / bs4 | `14.39ms` | `14.53ms` | `14.85ms` | `14.83ms` | — | latest kept gate |
| real fixture 27→16 / bs4 | `14.13ms` | `14.13ms` | `14.31ms` | `14.26ms` | — | greedy parity hold |

Per-call rank0 ledger（runtime model report, bs1/kv2, `measured_schedule_calls=1582`，`missing=183 all_reduce`）：
- `kimi_marlin_wna16_gemm`：120 calls / `118.06ms`（口径是 synthetic all-local route，全部 48 个本地 expert 都参与；不能直接外推 EP8 全局平均）
- `gemm_graphsafe`：367 calls / `5.73ms`
- `kimi_router_noaux_tc`：60 calls / `2.61ms`
- `rms_norm_batch`：245 calls / `2.03ms`
- `kimi_mla_split_qkv_a`：61 calls / `0.44ms`

Marlin 数字是 synthetic all-local route 假设，不是真实 EP8 全局 route 分布；接 full-rank route histogram 之前不能用它当 EP imbalance 结论。

## Optimization Log

### #5 Shared/dense gate-up fusion + routed scaled-add（2026-05-22）

**Bottleneck:** MoE shared expert path 之前是分开的 gate GEMM + up GEMM + silu_mul + down GEMM + TP all-reduce；dense layer 0 同样分开 gate/up。Routed combine 之后还有独立 scale + residual add kernel。

**Approach:**
- Load-time fuse shared expert 的 gate + up 成单个 BF16 GEMM（`DeviceMatrix::vstack`），decode 仅 GEMM 一次 + `silu_mul_fused_batch_into` 拆 SwiGLU。
- 同样把 dense layer 0 的 gate/up 合并。
- 把 routed `scale * routed_f32 + shared_bf16 + residual` 合并成 `kimi_scaled_add_f32_bf16_to_bf16` 单 kernel。
- Marlin output locks clear 在 route metadata 证明所有 consumed row 都写过之前先保留，不可砍。

**Result:** synthetic output64 steady TPOT avg `14.470ms → 14.388ms`，p99 `14.917ms → 14.834ms`。真实 fixture output16 steady avg `14.225ms → 14.126ms`，p99 `14.355ms → 14.258ms`。四路 token 与 vLLM fixture 一致。

### #4 fused qkv_a（2026-05-22）

**Bottleneck:** MLA q/kv down projection 之前是分开的 `q_a` GEMM + `kv_a_proj_with_mqa` GEMM，对应 vLLM `MergedReplicatedLinear`。

**Approach:** load-time 把 `q_a_proj` 和 `kv_a_proj_with_mqa` 在 K 维 vstack 成单一 `fused_qkv_a` 权重，decode 单次 `gemm_graphsafe(fused_qkv_a)` 后用 `kimi_mla_split_qkv_a` 一次切出 `q_a [B,1536] / compressed_kv [B,512] / k_rope [B,64]`。

**Result:** static trace calls `1947 → 1886`，每层减少一次 GEMM。synthetic output64 steady TPOT `16.70ms → 16.43ms`（−1.6%），后续 #5 进一步降到 `14.39ms`。

### #3 整段 decode CUDA Graph capture（2026-05-22）

**Bottleneck:** Strong-sync profile 显示 bs4 step `~35ms`，其中很大比例是 host enqueue + per-collective barrier 引入的 rank skew，不是 kernel compute。

**Approach:** 沿用 Qwen 的 `CudaGraphState` 模板，把 Kimi decode GPU body 拆成 graph-内 launch 区段和 graph-外 top1 D2H 区段。第一次尝试在 `max_tokens=2` 四并发 hang，原因是 Kimi 8 rank worker 独立 begin/end/launch，NCCL graph capture 缺少跨 rank 阶段对齐。新增 `kimi_graph_probe` 验证 local kernel / cuBLAS / NCCL all-reduce / NCCL reduce-scatter 各自都能 capture/replay；`CudaGraphState` 加同步 phase hook 后，rank worker 在 graph begin/enqueue/end/launch 周围插 rank barrier 对齐。

**Result:** `bench_serving --cuda-graph true --concurrency 4 --prompt-len 27 --output-len 64` steady TPOT `16.70ms / p99 17.11ms`，`cuGraphLaunch count = 8 ranks × 63 decode steps`，证明 measured iteration 走 graph replay。HTTP `max_tokens=128` 四并发 warm `20.64ms/token/wave`、`193.8 tok/s`，prefix/tail 一致。`kimi_graph_probe` 完成验证使命后已 retire（参见 [changelog](changelog.md)）。

### #2 Decode 诊断负担清理 + routed RS bridge（2026-05-22）

**Bottleneck:** Marlin atomic 修复后 row-state 收敛，但 decode 主路径仍带着诊断负担：每个 token 都做 row-diff D2H、row-wise F32 collective（per-active-row all-reduce）、collective 前 CPU `Barrier`。

**Approach:**
- 移除 decode 路径的 row-diff D2H。
- decode F32 collective 从 per-row loop 改回单次 contiguous all-reduce。
- decode collective CPU `Barrier` 不再执行（保留 prompt 初次 collective 的 barrier，那是 H20 首次 NCCL call 的独立稳定性问题，不混进 decode steady）。
- MoE routed combine 改成 `local router/Marlin → repeat_f32_for_reduce_scatter → NCCL reduce_scatter`，不做 BF16 all-gather，不把 local expert compute 按 EP world 放大。

**Result:** warm `max_tokens=64` 四并发 `144.247 tok/s`（旧口径 `114 tok/s`，+27%）；`max_tokens=16` wall `4615ms / 13.865 tok/s`，四路 token ids 全对。仍低于 `decode(bs4) > 300 tok/s` 长期目标，下一项 graph capture (#3) 才是真正减少 host enqueue 的地方。

### #1 Marlin atomic split-K row-state fix（2026-05-22）

**Bottleneck:** H20 固定 4 并发 fixture `max_tokens=16` 时 row1 偶发输出 `[1008,2742,924,6454,...]`（应为 `[1008,2742,2531,414,...]`）。Per-phase row first-diff 把切点收缩到 layer1 routed expert path，最早是 `moe_w13_out`。

**Root cause:** PegaInfer Marlin WNA16 wrapper 固定 `use_atomic_add=true` 且没传 `c_tmp`。当 split-K > 1 时，kernel 用 BF16 `atomicAdd` 直接累加进 output C；BF16 atomic 在 H20 上对累加顺序敏感，rank/token 之间的非确定性 ordering 把 row state 弄花。vLLM 自己的 `fused_marlin_moe.py` 对 W13 和 W2 都传 `use_atomic_add=False, use_fp32_reduce=True`，走 global F32 `c_tmp` 累加。

**Approach:** worker / decode arena 预分配 `c_tmp` F32 buffer，Marlin launch 切到 vLLM 的 global-reduce 路径（`use_atomic_add=false`），output / locks 在 step 边界 zero-fill。

**Result:** 4 并发 fixture `max_tokens=16` 四路 token ids 全对；`ROUTER_COUNT / ROUTE_ROW_COUNT / ROW_COUNT` 全部为 0。这之后 row-state 不再是 correctness 风险，可以切回 decode 性能主线。

### #0 Baseline — H20 TP8 EP8 text-only bring-up（2026-05-21）

**E2E:** vLLM K2.5 fixture（27-token prompt）`max_tokens=1` 返回 token id `1008`，`max_tokens=2` 返回 `[1008, 2742]`，4 并发 `max_tokens=8` 四路一致 `[1008,2742,2531,414,19180,6082,1379,387]`。多 prompt vLLM gate（`hello / math_short / self_intro_zh / code_rust`）4/4 greedy argmax match。

**Architecture wiring:**
- Text-only manifest + TP8/EP8 sliced loader，rank-local typed GPU view。
- Routed expert INT4 backend：从 CUTLASS example69 probe 切到 vLLM Marlin WNA16，bit-parity 单层 W13 + SwiGLU + W2 + topk sum 对 vLLM reference 0-diff。
- MLA：full prefill + decode wrapper（FlashInfer MLA decode），paged ckv/kpe cache worker 持有。
- MoE router：`kimi_router_noaux_tc_launch`（sigmoid + group top-k，匹配 DeepSeekV3 noaux_tc 语义）。
- TP collectives：BF16-via-F32 bridge（直接 BF16 NCCL 在 greedy parity 上回退，需要 F32 桥）。
- Scheduler：`KIMI_RUNNER_MAX_BATCH = 4` bs4 wave，prompt prefill 走 slot-local path，第 2 token 起调用真实 bs4 decode body。

**Verdict:** correctness OK。Decode strong-sync 口径 bs4 step `~35ms / 114 tok/s`，主热点 = MoE shared/reduce/router/align + TP/EP collectives；MLA + Marlin 单独都不是 bottleneck，graph fanout 和 collective cadence 是。

## Open

**重点是 decode 性能。Prefill 优先级低。** 按优先级排：

1. **`decode(bs4) > 300 tok/s`（当前主线）**：当前 `~278 tok/s`（`4 / 0.01439s/step`）。剩余空间集中在 collective fanout（123 logical all-reduce + 60 RS bridge）、shared/EP 通信 overlap，以及下一项的 TP1 + DP8 + EP8 重排。
2. **TP1 + DP8 + EP8 迁移（下一阶段）**：当前 TP8 + EP8 的 single-token MLA 每 token 都要做跨 8 rank TP all-reduce（123 次/step），collective cadence 是 graph-replay 下剩余 tail 的主要来源之一。改成 TP1（每 rank 自己跑 MLA / dense / shared expert，不再跨 rank reduce）+ DP8（8 路独立请求并行）+ EP8（MoE token routing 仍跨 rank，但走 PPLX dispatch/combine 而不是 NCCL bridge）后：
   - MLA / dense / shared expert 的 `all_reduce=123` 中绝大部分消失，只剩 MoE 段需要跨 rank。
   - Single-request TPOT 是否变好取决于 PPLX dispatch/combine 的实测；throughput 上能直接拿 8 并发的乘数。
   - 需要 scheduler / weights / KV cache 全部按 DP rank-local 切，PPLX EP backend 替换当前 NCCL RS bridge，并重新过 greedy parity gate。结构改动，独立 milestone。
3. **vLLM baseline 完整采集**：H20 ×8 上跑 `vllm bench serve` 同 profile，把 dashboard 的 TBD 列填上。没有这条做不出 keep/revert 判断。
4. **PPLX EP dispatch/combine**：作为 TP1 DP8 EP8 的前置组件，独立先把 PPLX backend 在 Kimi worker 里接通；当前 NCCL `repeat_f32 + reduce_scatter` bridge 是临时桥，最终被 PPLX 替换。
5. **Prefill 性能优化（优先级低）**：short-prompt streaming TTFT `1995.5ms` 偏高。先量 short-prompt prefill 拆解（embedding / MLA prefill / MoE prefill / sampling），再决定是 scratch allocator 还是首个 collective stream drain 主导。Long prompt（128+ synthetic）已经过 1k tok/s，主要痛点是 short prompt + first-call 路径。不影响 decode 主线。

详细推进点参见 [operator-todo.md](operator-todo.md)。历史实验和 reject 路径参见 [changelog.md](changelog.md)。Decode 路径与 vLLM 算子的逐项对照参见 [vllm-path-comparison.md](vllm-path-comparison.md)。
