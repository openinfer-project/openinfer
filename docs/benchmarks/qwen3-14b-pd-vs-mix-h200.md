# Qwen3-14B P/D 分离 vs mixed vs vLLM（8×H200，多轮长输入负载）

> **TL;DR**: Qwen3-14B bf16 多轮长输入（首轮 8k + 每轮 2k）A/B（2026-07-17）：等卡数下 P/D 全面胜出——2 卡 1P+1D 比 mixed×2 吞吐 +17%（354 vs 303 tok/s）、TPOT p99 -42%（22.8 vs 39.4ms）、**ITL p99 23.5 vs 105ms（-78%）**；对 vLLM 0.25.1×2，vLLM 吞吐/decode 内核更快（453 tok/s、TPOT p50 16.7ms），但 ITL p99 141ms 是 P/D 的 6 倍。重负载（60 会话，KV 1.9× 超 D 的 HBM）下 200GiB hugepage host 池 vs 16GiB 小池：小池丢 decode 纯净性（ITL p99 23.6→84.6ms）+吞吐 -9%。弹性 2P+2D 重负载（openinfer [#705](https://github.com/openinfer-project/openinfer/pull/705) 修复后、每配置冷启动重测）：router round-robin ITL p99 86ms → **前缀亲和 22.8ms + 268 tok/s**（pegaflow [#405](https://github.com/novitalabs/pegaflow/pull/405)），达到 1P+1D 的隔离度和 1.4× 其吞吐；裸吞吐仍低于 2 卡 mixed 的 359——重负载下 P/D 买的是 decode 纯净性（ITL p99 22.8 vs 107.5），不是每卡吞吐。⚠️ 首版 ladder 数据（437/465/495 tok/s）因同栈固定 seed 复跑全量前缀命中而虚高，已作废（见 §6）。

环境：单机 8×NVIDIA H200，每 GPU 1×400G IB NIC（P2P KV 走单边 RDMA READ），Qwen3-14B bf16（40L/40H/8KV/head 128，KV 160 KiB/token；单卡 HBM KV 容量 626k tokens），openinfer main `c116077`（§4 复测用 [#705](https://github.com/openinfer-project/openinfer/pull/705) 分支 `2a1f885`），pegaflow `111ea34`（0.23.4）+ 亲和 router 补丁（#405），vLLM 0.25.1。每实例 `--kv-offload --kv-offload-host-gib <N> --kv-offload-hugepages`（机器预留 1.5 TB 2MiB hugepages，200 GiB 池 NUMA 感知 2×100GiB，分配 ~5s/池）。

## 1. 负载与压测方法

vllm-bench 多轮对话，输入比 [Qwen3-8B 那轮](qwen3-8b-pd-vs-mix-h200.md) 拉长一倍：

```bash
vllm-bench \
  --backend openai-chat --base-url http://<endpoint> \
  --model <model-path> --tokenizer <model-path> \
  --dataset-name random \
  --multi-turn --multi-turn-num-turns 5 \
  --random-input-len 8192 --per-turn-input-len 2048 --random-output-len 128 \
  --num-prompts <N> --multi-turn-concurrency <C> \
  --extra-body '{"min_tokens":1}' --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --result-filename <name>.json --result-dir <dir>
```

- **标准负载**：20 会话、并发 10（会话终态 ~17k tokens ≈ 2.7 GiB KV，总量 ~53 GiB < 单 D HBM）。
- **重负载**：60 会话、并发 12（总 KV ~1.02M tokens ≈ 160 GiB，1.6× 单 D 的 626k HBM 容量——host 池被真实使用）。
- 每组配置测前重启整栈冷起（清 HBM 前缀缓存 + host tier + metaserver 目录）。
- mixed 多实例前面挂会话亲和 LB（首条消息 hash 定实例）；vLLM ×2 同法。
- `max_completion_tokens` 坑（openai-chat 用它不用 `max_tokens`）已由 pegaflow router 修复覆盖，无需 workaround。

## 2. 标准负载（20 会话 × 5 轮，并发 10）

| 配置 | 卡数 | out tok/s | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | ITL p99 (ms) |
|---|---|---|---|---|---|
| mixed ×1 | 1 | 264 | 239 / 5007 | 31.9 / 46.6 | 107.4 |
| mixed ×2 | 2 | 303 | 560 / 3970 | 25.3 / 39.4 | 105.0 |
| **P/D 1P+1D** | 2 | **354** | 493 / 4653 | **19.7 / 22.8** | **23.5** |
| vLLM 0.25.1 ×2 | 2 | 453 | 577 / 3205 | 16.7 / 36.0 | 141.2 |
| mixed ×4 | 4 | 356 | 397 / 2839 | 22.3 / 33.0 | 96.7 |
| **P/D 2P+2D** | 4 | **409** | **312 / 2665** | **19.3 / 22.5** | **22.8** |

分轮（p50 TTFT / p99 ITL，ms）：

| Turn | mixed×2 | P/D 1P+1D | vLLM×2 | mixed×4 | P/D 2P+2D |
|---|---|---|---|---|---|
| 1（冷 8k） | 1001 / 82 | 2734 / 18 | 833 / **387** | 889 / 79 | 840 / 22 |
| 2 | 242 / 88 | 492 / 20 | 431 / 117 | 248 / 87 | 265 / 20 |
| 3 | 351 / 100 | 336 / 27 | 579 / 136 | 271 / 95 | 289 / 21 |
| 4 | 628 / 106 | 302 / 26 | 546 / 141 | 318 / 102 | 312 / 23 |
| 5 | 749 / 109 | 499 / 24 | 488 / 17 | 333 / 107 | 336 / 23 |

读法：

- **输入拉长一倍后，P/D 从"吞吐持平"变成"吞吐也赢"**：8B 4k 输入时 P/D vs mixed×2 吞吐持平；14B 8k 输入下 prefill 干扰变重，mixed 的 unified step 被长 suffix prefill 反复打断，P/D +17%。
- **ITL p99 是分水岭指标**：mixed/vLLM 全轮 80–140ms（decode 被并发 prefill 冻结），P/D 全轮 <27ms。TPOT p99 会平均掉冻结，ITL 不会。
- **vLLM 0.25.1 的 14B decode 内核比我们快**（TPOT p50 16.7 vs 19.7ms，吞吐 453 vs 354）——这是 qwen3 line 在 14B 上的内核差距（历史调优都在 4B/8B/RTX 5090），与 P/D 架构无关；vLLM 的 ITL p99 141ms 同样输给 decode 隔离。
- **P/D 冷 turn1 代价**：1P+1D 2734ms vs mixed×2 1001ms——10 并发 8k prefill 全压在 1 个 P 上排队 + P→D 交接。2P+2D 把它压回 840ms（低于 mixed×4 的 889ms）。M3 layer-wise push 进一步优化交接部分。

## 3. 重负载：hugepage host 池的价值（60 会话，并发 12，1P+1D）

总 KV ~160 GiB，单 D HBM 只装得下 ~95 GiB——host 池装不装得下工作集直接决定 decode 纯净性：

| host 池 | out tok/s | TTFT p50/p99 (ms) | TPOT p99 (ms) | ITL p99 (ms) |
|---|---|---|---|---|
| **200 GiB hugepage ×2** | **194** | 3695 / 11607 | **22.4** | **23.6** |
| 16 GiB ×2（对照） | 177 | 4128 / 11795 | 42.9 | 84.6 |

- 大池：P 的 host tier 保住全部会话前缀，D 每轮只 RDMA 拉新增 suffix，300/300 请求全程 ITL p99 <24ms。
- 小池：P 侧 evict → D metaserver 查询 miss → **D 被迫本地 prefill**（decode 节点被 prefill 污染），turn2/3 ITL p99 飙到 84–97ms，吞吐 -9%。
- 这就是 KV cache 分层与 P/D 一体的意义：**同一个 pegaflow 池既是容量层又是传输基底**——host 容量买到的不只是 prefix cache 命中，还有 decode 节点的纯净性。
- 注意两组 TTFT p50 都是秒级且逐轮爬升（turn5 p50 ~11s）：重负载下 1 个 P 的 prefill 算力饱和，与池大小无关——解法是加 P（见 §4），不是加内存。

## 4. 弹性 xP+yD：router 亲和是必要条件（60 会话，并发 12，2P+2D）

pegaflow-router 原生支持 `--prefill/--decode` 各传多个端点（round-robin）。直接上 2P+2D 重负载暴露了 round-robin 的问题：同一会话的不同 turn 落到不同 P/D，前缀局部性被打碎，退化成跨节点全量 KV 搬运——P 从别的 P（甚至从 D！内容寻址 mesh 是全向的）拉 2.2 GiB 前缀（300 请求 ~80 次），D 每轮全量重拉而非增量。修复 = 前缀亲和选路（pegaflow [#405](https://github.com/novitalabs/pegaflow/pull/405)）。下表为**每配置冷启动**、openinfer [#705](https://github.com/openinfer-project/openinfer/pull/705)（`2a1f885`）复测：

| 选路策略 | out tok/s | TTFT p50/p99 (ms) | TPOT p99 (ms) | ITL p99 (ms) |
|---|---|---|---|---|
| round-robin | 259 | 1267 / 11717 | 37.9 | 86.0 |
| **前缀亲和（P+D）** | **268** | 1911 / 18095 | **22.5** | **22.8** |

- 亲和后 ITL p99 回到 1P+1D 的 23ms 隔离度，吞吐 1.4×（268 vs 194），TTFT p50 从 1P+1D 的 3695ms 降到 1911ms（双 P 分摊 prefill 洪峰）。round-robin 的 TTFT p50 更低（会话摊到两个 P），但代价是决定性的：跨节点搬运把 decode 平滑度打回 86ms。
- 对照 2 卡重负载 mixed×2（359 tok/s、TPOT p99 47ms、ITL p99 107.5ms）：**4 卡 P/D 亲和栈裸吞吐仍不敌 2 卡 mixed**（268 vs 359）——重负载下单 P prefill 饱和是瓶颈，P/D 买到的是 TPOT/ITL 纯净性（22.5/22.8 vs 47/107.5），适合 SLO 约束的服务而非吞吐最大化。
- 任意 D 发现任意 P 已实测（2P+2D 下 D0/D1 均从两个 P 拉过块）；亲和只是把"能跑"变成"跑得好"。
- 剩余的 round-robin ITL 86ms 不再是 bug：restore 后的 suffix prefill 与 decode 共享 unified step，step 被真实计算拉长（nsys：GPU 占空比 ~95% 无空洞）。亲和从源头消掉搬运，故 22.8ms。

### 4.1 restore 冻结 decode 的机制（nsys + A/B 实锤，openinfer [#704](https://github.com/openinfer-project/openinfer/issues/704) → [#705](https://github.com/openinfer-project/openinfer/pull/705) 已修复）

单独探测（8 路 decode + 4 个 16k 冷 restore 共存）复现出全部流**同一毫秒**同步 hiccup。nsys 排除了直觉解释：GPU kernel 占空比全程 ~95%、无 >10ms gap——不是 HBM 带宽也不是 SM 争抢（DMA 上限 64GB/s 仅为 HBM 4.8TB/s 的 ~1.3%）。真实机制是 **scheduler 线程被 CPU 侧簿记卡死**，两层：

1. **主因**：`commit_loaded_blocks` 在 scheduler 线程上逐块注册 restore 的块，~70µs/块（registry radix 插入 + event hook + 频率统计 + store 锁）。1000+ 块的大 restore = ~70ms 内所有流的 token 交付冻结。
2. **次因**：greedy token 回读用 pageable D2H（同步拷贝语义），被并发大批量拷贝排队，从亚毫秒平顶到 23.6ms/步。

修复（#705）：注册改为 `Committing` 阶段每 tick 64 块分期支付（仅在有活跃 decode 行时限速，纯 prefill tick 一次付清）+ token 回读 pinned 化。探测中 >40ms 的 ITL 尖峰从每次 restore 必现降到 **0**；restore 密集的 warm 复跑吞吐 +22%（298 vs 244 tok/s）。试过并被数据否决的方案：off-thread 注册线程（store 锁与 scheduler 争抢，更差）、Kernel 拷贝后端（copy kernel 抢 SM，两波 ~110ms 停顿）。

## 5. 正确性与故障门（14B 复验）

- **逐字节一致**：3 档 prompt（<1 page / 跨页 / ~600 tok），temp=0，router P/D vs 直连 D baseline 输出 IDENTICAL ×3。
- **P2P 实证**：~600 tok prompt D 侧 `RDMA fetch summary: blocks=33/33 bytes_mib=82.5`，连接复用后 rdma_wait 1.80ms（30.4 GiB/s）。
- **故障退化**：杀 metaserver → router 请求 WARN 后本地 prefill 正常完成；杀 P → D 冷请求正常完成。无 crash 无 hang。
- 全部压测 0 失败请求（含 300 请求重负载 ×5 组）。

## 6. 工具坑

- vllm-bench `accept-dist-20260709` 之后的私有构建在 multi-turn synthetic 模式下生成完对话即 **segfault**（"timeout: the monitored command dumped core"，exit 0 且无输出，极易误判为静默无请求）；用 `before-accept-dist-20260709` 构建正常。
- vLLM 0.25.1 起服务需 `ninja` 在 PATH（venv 里 pip 装的 ninja 要把 venv/bin export 进 PATH），且 `--disable-log-requests` 已移除。
- **同栈复跑 = warm 污染**：vllm-bench synthetic 模式 seed 固定，同一栈上按顺序测多个配置时，后面的配置全量命中前面留下的前缀（host 池 + GPU cache），吞吐可虚高 ~70%（本文首版 2P+2D ladder 437/465/495 即此坑，冷启动复测实为 259/268）。协议必须是**每配置重启栈**（清 GPU/host 池 + metaserver），或换 seed。warm 复跑只可用于刻意的 restore 压力测试，并须标注。

## 7. 关联

- P/D 架构与 M2 验收：`../models/qwen3/pd-disaggregation-m2.md`
- Qwen3-8B 首轮 A/B（短输入、吞吐持平结论的出处）：`qwen3-8b-pd-vs-mix-h200.md`
- router 亲和补丁：pegaflow [#405](https://github.com/novitalabs/pegaflow/pull/405)
