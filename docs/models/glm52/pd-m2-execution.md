# GLM5.2 P/D 分离 M2：vLLM-TP8 prefill + openinfer-EP8 decode 执行记录

> **TL;DR:** GLM P/D 里程碑 2 启动（2026-07-11）。拓扑：P = vLLM TP8（8×H200 节点 A），D = openinfer EP8（8×H200 节点 B），跨节点 pegaflow P2P（首次跨机验证）。**不带 MTP/DSpark**——speculative 状态交接是独立问题（见 `vllm-speculative-pd-audit.md`），且 DSpark × prefix-cache 目前互斥。验收：(1) D 零 prompt 位置计算（硬约束，crash-early invariant + 计数证据）；(2) 多轮长输入压测 vs 同卡数 2× vLLM mixed（会话亲和 LB）符合预期形态（吞吐持平、TPOT p99 更稳、turn2+ TTFT 恒定）。设计决策全部继承 `pd-vllm-prefill.md`（feat/pd-vllm-hash-compat 分支）：vllm-hash-compat provider（`xxhash_cbor` + 钉 `PYTHONHASHSEED`）、D 端有界快轮询、路由变体 A。
>
> **Last touched:** 2026-07

## 与 qwen3 里程碑 1 的差异（GLM 专属工作项)

qwen3 的 vLLM-P + openinfer-D（PR #540，未合）已验证 hash 复刻 + 兼容等式 + RemoteFetch 等待分支。GLM 新增：

1. **双 cache 家族**：78 层 MLA（656 B/token）+ 21 层 index-K（132 B/token）都要 hash 命中 + 字节比对。布局已做过源码级核对（vs vLLM `cdab28319`，见 `pegaflow-offload-pd.md` M2 节），gate 防的是静默漂移。
2. **页大小天然对齐**：GLM 两边都是 page 64（qwen3 要 P 迁就 `--block-size 16`）。vLLM P 直接 `--block-size 64`。
3. **零 prefill 是硬约束**（qwen3 上 D 兜底算 ≤16 token 尾巴可接受，GLM 上尾巴 = token-by-token 骑 decode 内核，最坏 ~3s）：
   - P 端 connector 加法式尾块扩展：`xxh3_128(cbor((last_full_hash, tail_token_ids, None)))` 派生尾块 key，partial page 一并入库；
   - D 端 partial-block restore（kvbm 目前拒收未 seal 块，需要 `(hash, valid_len)` 级契约）+ 从 decode 路径继续填充该页；
   - 路由变体 A：P 的 t1 直接回客户端并追加进转发 D 的上下文，D 首步 = 对 t1 的真 decode（1 token，零 prompt 位置）；
   - D 严格模式：miss/超时 → 429/500（router 重试重走 P），禁止 scratch 兜底；executor 加不变量——P/D 模式下 admitted 请求未算 prompt token > 1 即报错。
4. **跨节点**：M2 qwen3 是单机双卡；本次 P/D 各占一台整机，dma-buf/GID/路由首次实测（两边同构 8×400G IB 1:1 PIX，预期直接成立）。
5. **等待谓词**：D 必须等完整前缀命中才放行（partial-hit 在 GLM 不可接受），复用 #540 的 miss 窗口重查询 + 熔断，但退化路径改为 fail。

## 环境

- 节点 A（P）与节点 B（D）：各 8×H200 + 8×400G IB。B = openinfer GLM 全套 gate 环境（NCCL 2.30.7、DeepGEMM、oracle）；A 新配 vLLM。
- **权重必须同源**：两台机器上原有的两份 GLM-5.2 checkpoint 数值不同（`embed_tokens` md5 不一致，config 只差 `transformers_version`），已统一为 GLM-5.2-FP8（真 5.2，从 B 拷贝到 A 的 /data1，`/data/models/GLM-5.2-FP8` symlink）。错误的 `GLM-5.2-0614-Provider-FP8` 已删。**教训：P/D 两侧权重不同源时 hash 照样命中（key 只看 token），是静默正确性洞——环境搭建时先 spot-hash 权重。**
- P 侧栈模板：里程碑 1 的 `stack.sh`（vLLM `PegaKVConnector` + `--prefix-caching-hash-algo xxhash_cbor` + `PYTHONHASHSEED=0` + pegaflow-server per node + pegaflow-router）。
- 依赖悬挂：openinfer PR #540 与 pegaflow #382 都未合并；GLM 分支基于 #540 cherry-pick，rev pin 协调见 memory。

## 实现落点（2026-07-12）

- **openinfer `feat/glm52-pd`**（基于 main + merge #540）：
  - `Glm52KvOffloadOptions` 增 `p2p`（pegaflow mesh）与 `vllm_compat`；host 加入 P2P mesh 并使用 peer namespace；
  - `scheduler/offload.rs::admit_vllm_pd`：整前缀 restore（vLLM 键）→ 尾块经派生 key 载入**请求私有页**（`schedule_prefill(tail_len)` → pegaflow `load` 整页 H2D → `apply_prefill_chunk`，不进 radix）→ suffix==1 才放行；zero/partial-hit park 在队首按 step 边界重试（全空闲时 5ms 节流）；窗口耗尽 **Reject**（router 重试），3 连窗 miss 开 breaker；`--kv-pd-allow-local-prefill` 为调试逃生门；compat 模式禁自存；
  - 关键洞察：+1 token（变体 A 追加 t1）使 kvbm probe 的 cacheable 窗口恰好覆盖 P 侧全部满页——对齐 prompt 无需特判。部分页的"无效尾行"由后续 decode 自然覆盖，kvbm 零改动。
- **pegaflow `feat/glm52-pd`**：
  - connector `pegaflow.pd_tail_save`：调度完最后一段 prompt 的那一步（KV 已写完、无需等 finish，规避空闲引擎不再 step 的死角）用 **vLLM 自己的 hash 函数**派生尾块 key 并入库；racing 的 t1 行无害（key 只覆盖 prompt 位置）；
  - router `--pd-first-token`（变体 A）：P 带 `return_token_ids` + `min_tokens=1`；t1 拼进客户端响应并追加进发给 D 的 **token-id prompt**（chat 模板只在 P 生效，D 恒走 `/v1/completions`）；D 的 SSE 转换回客户端 API；D 失败整流程重试（`--pd-flow-retries`）。
- vLLM 0.23.0 API 已在 P 机 venv 实证：`return_token_ids`（chat 的 `prompt_token_ids` 在响应**顶层**）、`hash_block_tokens/NONE_HASH`、需补装 `xxhash`+`cbor2`。

## 里程碑与 gate

| # | Gate | 状态 |
| --- | --- | --- |
| 0 | 权重同源（spot-hash 对齐）+ A 机 vLLM TP8 起服 | 权重 ✅（错误的 0614-Provider 已删，真 GLM-5.2-FP8 拷贝并 spot-hash 校验）；vLLM 起服中 |
| 1 | 跨节点 pegaflow P2P RDMA READ 实证 | ⬜（栈已就位：metaserver+pegaserver on A，D on B） |
| 2 | hash + namespace + 双 cache 字节比对（对齐 / 非对齐 prompt 各一档） | ⬜（以 e2e 逐 token 一致形态验证，M1 先例） |
| 3 | 尾块 + 变体 A + 严格模式：D 零 prefill invariant 全绿 | 代码 ✅（60 lib 测试过），运行时 ⬜ |
| 4 | E2E：router P/D vs baseline 逐 token 一致 ×3 档 prompt；多轮 delta 只拉增量；杀 P/杀 metaserver 干净失败 | ⬜ |
| 5 | 验收压测：多轮长输入（首轮 ~8k、每轮 +2k），P/D vs 2× vLLM mixed（会话亲和），吞吐/TTFT/TPOT 分轮报告 + D 零 prefill 证据（严格模式下 admit 即 suffix=1，无 allow_local_prefill） | ⬜ |

## 复跑

两台节点上的 `/root/pd-glm/stack.sh`（A 机：metaserver/pegaserver/vllm/router 四组件，B 机：openinfer-D，从 A 机 vllm 日志自动抓 namespace）+ B 机 `/root/pd-glm/smoke.sh`。D 严格模式下**禁止**直接绕过 router 发请求（键链按"最后一个 token 是 P 生成的"派生，直连请求必然 miss → 全量拒绝）。
