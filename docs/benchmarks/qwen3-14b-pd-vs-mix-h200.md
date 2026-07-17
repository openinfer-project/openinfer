# Qwen3-14B P/D 分离 vs mixed vs vLLM（8×H200，多轮长输入负载）

> **TL;DR**: Qwen3-14B bf16 多轮长输入（首轮 8k + 每轮 2k）A/B（2026-07-17）：等卡数下 P/D 全面胜出——2 卡 1P+1D 比 mixed×2 吞吐 +17%（354 vs 303 tok/s）、TPOT p99 -42%（22.8 vs 39.4ms）、**ITL p99 23.5 vs 105ms（-78%）**；对 vLLM 0.25.1×2，vLLM 吞吐/decode 内核更快（453 tok/s、TPOT p50 16.7ms），但 ITL p99 141ms 是 P/D 的 6 倍。重负载（60 会话，KV 1.9× 超 D 的 HBM）下 200GiB hugepage host 池 vs 16GiB 小池：小池丢 decode 纯净性（ITL p99 23.6→84.6ms）+吞吐 -9%。弹性 2P+2D 重负载 ladder：router round-robin ITL p99 85ms → P 亲和 49ms → **P+D 亲和 23.0ms + 495 tok/s**（pegaflow [#405](https://github.com/novitalabs/pegaflow/pull/405)），达到 1P+1D 的隔离度和 2.5× 其吞吐。

环境：单机 8×NVIDIA H200，每 GPU 1×400G IB NIC（P2P KV 走单边 RDMA READ），Qwen3-14B bf16（40L/40H/8KV/head 128，KV 160 KiB/token；单卡 HBM KV 容量 626k tokens），openinfer main `c116077`，pegaflow `111ea34`（0.23.4）+ 亲和 router 补丁（#405），vLLM 0.25.1。每实例 `--kv-offload --kv-offload-host-gib <N> --kv-offload-hugepages`（机器预留 1.5 TB 2MiB hugepages，200 GiB 池 NUMA 感知 2×100GiB，分配 ~5s/池）。

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

pegaflow-router 原生支持 `--prefill/--decode` 各传多个端点（round-robin）。直接上 2P+2D 重负载暴露了 round-robin 的问题：同一会话的不同 turn 落到不同 P/D，前缀局部性被打碎，退化成跨节点全量 KV 搬运——P 从别的 P（甚至从 D！内容寻址 mesh 是全向的）拉 2.2 GiB 前缀，D 每轮全量重拉而非增量。修复 = 前缀亲和选路（pegaflow [#405](https://github.com/novitalabs/pegaflow/pull/405)）：

| 选路策略 | out tok/s | TTFT p50 (ms) | TPOT p99 (ms) | ITL p99 (ms) | P 侧跨节点前缀拉取 |
|---|---|---|---|---|---|
| round-robin | 437 | 429 | 38.9 | 85.2 | 30 次（含 2.2 GiB/次） |
| P 亲和 | 465 | 374 | 30.8 | 49.2 | 1 次 |
| **P+D 亲和** | **495** | **324** | **22.8** | **23.0** | 1 次 |

- P+D 亲和后：ITL p99 回到 1P+1D 的 23ms 隔离度，吞吐 2.5×（495 vs 194），TTFT p50 从 3695ms 降到 324ms（双 P 分摊 prefill 洪峰）。
- 对照等卡重负载 mixed×2（359 tok/s、TPOT p99 47ms、ITL p99 107.5ms）：2 卡 P/D 亲和栈换成 4 卡后每一项都碾压。
- 任意 D 发现任意 P 已实测（2P+2D 下 D0/D1 均从两个 P 拉过块）；亲和只是把"能跑"变成"跑得好"。
- D 侧仍有 ~60 次 evict 后全量重拉（320 MiB@9ms，已不撞 decode）；进一步压掉靠 M3 layer-wise push / D 侧 save 策略。

### 4.1 restore 撞 decode 的机制（nsys 实锤，openinfer [#704](https://github.com/openinfer-project/openinfer/issues/704)）

单独探测（8 路 decode + 1 个 16k 冷 restore 共存）复现出全部流**同一毫秒**同步 hiccup ~110ms。nsys 排除了直觉解释：GPU kernel 占空比全程 ~95%、无 >10ms gap（launch-ahead 在喂队列）——不是 HBM 带宽也不是 SM 争抢（DMA 上限 64GB/s 仅为 HBM 4.8TB/s 的 ~1.3%）。真实机制：restore 注入是 ~1.4 万个平均 63KB 的碎片 H2D，把拷贝路径塞满；executor 线程每步的 greedy token 回读（`openinfer-sample/src/lib.rs:184` 的 `clone_dtoh`，pageable 目的地 → async 退化为同步）从亚毫秒涨到恒定 23.6ms，连续 8 步 ≈190ms——**step loop 被卡，GPU 没停，token 交付停**。修复方向：token 回读 pinned 化 + 独立 stream（碰 sampling 热路径，需重跑精度 gate）；pegaflow 侧合并/限流注入拷贝；M3 layer-wise push 结构性缩小暴露窗口。

## 5. 正确性与故障门（14B 复验）

- **逐字节一致**：3 档 prompt（<1 page / 跨页 / ~600 tok），temp=0，router P/D vs 直连 D baseline 输出 IDENTICAL ×3。
- **P2P 实证**：~600 tok prompt D 侧 `RDMA fetch summary: blocks=33/33 bytes_mib=82.5`，连接复用后 rdma_wait 1.80ms（30.4 GiB/s）。
- **故障退化**：杀 metaserver → router 请求 WARN 后本地 prefill 正常完成；杀 P → D 冷请求正常完成。无 crash 无 hang。
- 全部压测 0 失败请求（含 300 请求重负载 ×5 组）。

## 6. 工具坑

- vllm-bench `accept-dist-20260709` 之后的私有构建在 multi-turn synthetic 模式下生成完对话即 **segfault**（"timeout: the monitored command dumped core"，exit 0 且无输出，极易误判为静默无请求）；用 `before-accept-dist-20260709` 构建正常。
- vLLM 0.25.1 起服务需 `ninja` 在 PATH（venv 里 pip 装的 ninja 要把 venv/bin export 进 PATH），且 `--disable-log-requests` 已移除。

## 7. 关联

- P/D 架构与 M2 验收：`../models/qwen3/pd-disaggregation-m2.md`
- Qwen3-8B 首轮 A/B（短输入、吞吐持平结论的出处）：`qwen3-8b-pd-vs-mix-h200.md`
- router 亲和补丁：pegaflow [#405](https://github.com/novitalabs/pegaflow/pull/405)
