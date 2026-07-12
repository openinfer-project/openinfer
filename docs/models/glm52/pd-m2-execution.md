# GLM5.2 P/D 分离 M2：vLLM-TP8 prefill + openinfer-EP8 decode 执行记录

> **TL;DR:** GLM P/D 里程碑 2：P = vLLM TP8（8×H200 节点 A），D = openinfer EP8（8×H200 节点 B），跨节点 pegaflow P2P，无 MTP/DSpark。**2026-07-12：跨引擎链路全通** —— router 变体 A 下 P/D 输出与 vLLM 基线逐 token 一致（对齐 + 尾块两档），D 全程零 prefill（admit 即 suffix=1）；快压 4k×128 并发 8：TPOT p99 26 ms（≈median）；多轮（8k+2k×5 轮）turn2+ TTFT 390-650 ms vs 直连 858-1380 ms、TPOT p99 25-27 vs ~31-66 ms；gsm8k 0.96/0.97 vs 直连 0.955（同分）；NIAH passkey 4k/8k/16k 全深度 36/36 与直连打平（indexer 风险降级）。硬依赖：vLLM ≥ 0.24.0 + `--kv-cache-dtype fp8_ds_mla`；D 端 compat 注册用 vLLM 层名 + page-first；restore 后 RoPE deinterleave fixup。剩余：故障注入 + 等卡数验收压测；残余风险：indexer rope 约定分歧（NIAH 已过，未完全关闭，见文末）。
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
- **vLLM 版本下限 = 0.24.0**：0.23 给全部 78 层都建 Indexer cache（`_skip_topk` 只是运行时开关）→ 注册 156 层；0.24（#45895）起只在 full-indexer 层建 → 78 MLA + 21 idxk = 99，与 openinfer arena 集一致。首跑 smoke 报 `stored block has 1 slots but instance expects 99` 的另一半原因：pegaflow connector 对 MLA 模型自动 **page-first**（整块所有层拼一页、单 slot，页内偏移按**层名字典序**）。D 端修法：compat 模式下 arena 用 vLLM 层名（`model.layers.N.self_attn.attn` / `...indexer.k_cache`）+ `page_first=true` 注册——同名 ⇒ 同序 ⇒ 同偏移，字节宽度本就同源（fp8_ds_mla 656 B/token、idxk 132 B/token）。

## 里程碑与 gate

| # | Gate | 状态 |
| --- | --- | --- |
| 0 | 权重同源（spot-hash 对齐）+ A 机 vLLM TP8 起服 | ✅（错误的 0614-Provider 已删；vLLM **0.24.0** + `--kv-cache-dtype fp8_ds_mla` 起服） |
| 1 | 跨节点 pegaflow P2P RDMA READ 实证 | ✅（D 侧 RDMA fetch 实测：3 块 9.9 MiB ≈3.3 MiB/块 = 78×41984+21×8448 精确吻合） |
| 2 | hash + namespace + 双 cache 字节比对（对齐 / 非对齐 prompt 各一档） | ✅（e2e 逐 token 一致：250-token 尾块档与 256-token 对齐档均与 vLLM 基线逐 token 相同，重复请求确定性） |
| 3 | 尾块 + 变体 A + 严格模式：D 零 prefill invariant 全绿 | ✅（admit 全部 `suffix=1`，`allow_local_prefill=false`；窗口耗尽→500→router 重试链路实测） |
| 4 | E2E：router P/D vs baseline 逐 token 一致 ×3 档 prompt；多轮 delta 只拉增量；杀 P/杀 metaserver 干净失败 | ✅（逐 token 一致 ×2 档；radix 复用 `gpu_hit=3 pulled=0`；多轮 delta 只拉增量；故障注入见下节——顺带抓出并修复 miss-breaker 死锁） |
| 5 | 验收压测：多轮长输入（首轮 ~8k、每轮 +2k），P/D vs 2× vLLM mixed（会话亲和），吞吐/TTFT/TPOT 分轮报告 + D 零 prefill 证据（严格模式下 admit 即 suffix=1，无 allow_local_prefill） | ⬜ |

### Bring-up 期间抓出的三个正确性 bug（均已修）

1. **层数 156 vs 99**：vLLM 0.23 给全部 78 层建 Indexer cache；0.24（#45895）起只建 21 个 full-indexer 层。**P 侧 vLLM 版本下限 = 0.24.0**。
2. **RoPE 存储排列**：vLLM（`is_neox_style=False`）旋转后 pair 保持 interleaved `(2i,2i+1)`；openinfer 内核是 interleave-in/block-out `[i, i+32]`——同值不同排列。D 端 restore 后对每个新拉取页做一次 deinterleave（MLA 行 528..656 的 64×bf16 + index-K 行前 64×fp8；rank worker 命令，同 stream 先于后续 step）。未修时症状：t1 正确、后续输出全是 prompt 词汇的乱序拼贴（KV 结构性错位），或直接 DeepEP dispatch 断言崩（bf16 布局时代）。
3. **P 侧尾块保存在前缀命中下不触发**：`scheduled >= prompt_len` 没算上本地 prefix-cache 命中的位置——**多轮 turn≥2 必踩**（P 必然命中上轮前缀）。修为 unscheduled（本地命中+pegaflow load）+ scheduled ≥ prompt_len。

### Bring-up 快压：TTFT 增量与 TPOT 稳定性 A/B（2026-07-12，非验收数据）

`vllm bench serve`（random 4096 in / 128 out / temp 0 / ignore-eos，每档独立 seed 防 prefix-cache 污染）。A = vLLM 单机直连（mixed，prefill+decode 同实例），B = router 全 P/D 链路。P = vLLM TP8（8×H200），D = openinfer EP8（8×H200）。

**并发 1（隔离纯交接开销，无排队）：**

| | vLLM 直连 | P/D | 增量 |
| --- | --- | --- | --- |
| TTFT median | 496 ms | 541 ms | **+45 ms（+9%）** |
| TPOT median | 11.9 ms | 19.5 ms | +7.7 ms |

- **P/D 交接的 TTFT 代价 = ~45 ms**，构成与 D admit 日志吻合：park 等待 ~46 ms（P 侧 save 注册可见性，占大头，可优化）+ RDMA fetch ~5 ms（64 块 ~13 MB）+ router 串行往返。相对 4k prefill（~500 ms）是 +9%；prompt 越长占比越小。
- batch=1 的 TPOT 差（19.5 vs 11.9）是 openinfer EP8 lock-step 单请求 decode 的 MoE 固定开销，与 P/D 传输无关（负载下消失，见下）。

**并发 8 × 32 请求（负载下）：**

| | vLLM 直连（mixed） | P/D | Δ |
| --- | --- | --- | --- |
| TTFT median | 1913 ms | 1005 ms | **-47%** |
| TTFT p99 | 6472 ms | 3544 ms | -45% |
| TPOT median | 31.9 ms | 25.0 ms | -22% |
| TPOT p99 | **60.8 ms** | **25.5 ms** | **-58%** |
| 输出吞吐 | 154 tok/s | 215 tok/s | +39% |

- **TPOT 稳定性是核心证据**：mixed 的 p99/median = 1.9（prefill chunk 抢占 decode），P/D 的 p99≈median（25.5/25.0 = 1.02）——decode 完全无抢占。
- 负载下 P/D 的 TTFT 反而更低：P 专注 prefill，无 decode batch 占坑。
- **公平性注意**：本对比 P/D 用 16 卡 vs mixed 8 卡，吞吐/TTFT 优势含卡数加成，不可直接引用；等卡数结论以验收压测（P/D vs 2× mixed 会话亲和）为准。与卡数无关的两个结论：交接开销 ~45 ms、P/D 侧 TPOT p99≈median。
- 零 prefill 全程成立：压测窗口 D 的 admit 全部 `suffix=1`，零拒绝零失败。
- 局限：单轮、恒定长度、无会话复用——**不可引用为验收结论**。

### 多轮 A/B（vllm-bench openai-chat，16 会话 × 5 轮，首轮 8192 +2048/轮，输出 128/轮，并发 8，temp 0 + ignore-eos）

| 轮次 | P/D TTFT med (ms) | 直连 TTFT med (ms) | P/D TPOT med/p99 | 直连 TPOT med/p99 |
| --- | --- | --- | --- | --- |
| 1 (8k) | 4382 | 4611 | 23.8 / 27.3 | 41.8 / **66.6** |
| 2 | **653** | 858 | 26.5 / 27.0 | 27.3 / 31.5 |
| 3 | **521** | 992 | 26.9 / 27.6 | 26.4 / 31.4 |
| 4 | **385** | 999 | 25.4 / 25.5 | 26.4 / 31.5 |
| 5 | **390** | 1380 | 25.4 / 25.7 | 21.8 / 31.3 |

- 160/160 请求零失败（多轮 = P 侧必然前缀命中 = 尾块保存修复的主战场）。
- **turn2+ TTFT：P/D 稳定 ~390-650 ms 且逐轮下降；直连 858→1380 ms 逐轮恶化**（增量 prefill 与在场 decode batch 抢卡）。turn5 差 3.5×。
- TPOT p99：P/D 全程 25-27 ms（≈median）；直连全程 ~31 ms、首轮 66 ms。
- 同样注意 16 卡 vs 8 卡的公平性 caveat；等卡数对比以验收压测为准。turn>2048 上下文的 indexer 风险（见下节）在此工况（峰值 ~17k）未见输出退化迹象（gsm8k 上下文 <2048，不覆盖该风险；多轮 bench 是 random token 无法直接测质量）。

### 故障注入（gate 4 收尾，2026-07-12）

**杀 P（vLLM `kill -9`）**：router 5 ms 内干净 502（`prefill request: error sending request`），无 hang、无 D 侧兜底，D 进程与日志无恙。恢复 = 单独重启 vLLM（~4 min 权重装载），无需动其它组件——namespace 由 seed 确定性派生，P 重启不换名。恢复后链路直接可用（实测 200）。

**杀 metaserver（`kill -9` + 重启）**：故障期间请求 30 s 干净 502（= 2 × 15 s in-flight-fetch 窗口 × router 双尝试），D 把 metaserver 连接拒绝当 miss 持续轮询、严格拒绝，进程无恙。**恢复语义分三层**：

1. **节点会话自愈**：pegaserver 与 D 的 metaserver_client 心跳自动重注册（重启后 ~10 s 内），新保存的块注册/发现全部正常。
2. **存量块元数据丢失（设计事实）**：metaserver 元数据是内存态，重启即清零；pegaserver **不会**重新发布已持有块的目录。叠加 P 侧 connector 对前缀命中的请求只重存尾块、不重存满页——**故障前的老前缀成为黑洞**（数据在 pegaflow 里、注册永久缺失），命中老前缀的请求持续干净 502，直到 P 重启（vLLM prefix cache 清零 → 全量重算重存）或自然逐出。运维口径：metaserver 重启后，若要立即恢复存量会话，连带重启 vLLM。pegaflow 侧正确修法（follow-up）：metaserver_client 重连成功后全量重发布本地块目录（DB 视角 = replica 在 registry failover 后 re-publish catalog）。
3. **miss breaker 死锁（已修，injection 战果）**：修复前 3 连 miss 窗口开断路器后新请求"首查即拒"（µs 级）。而 pegaflow `query()` 是异步模型——首查只**发起**后台 metaserver 解析 + fetch 并立即报 miss/Loading，意味着断路器一旦打开，任何远端块都不可能完成 restore，唯一复位条件（完整 restore 落地）永远无法达成 → metaserver 瞬时抖动即可让 D 进入**永久全量拒绝**（实测：metaserver 早已恢复、query 返回 prefix=8/8，D 仍 µs 级拒绝一切）。修复：breaker open 改为短探针窗口 park（500 ms，同时替换 miss 与 in-flight 两个 deadline，覆盖 ~46 ms save 可见性 + 异步 fetch），完整 restore 照旧复位——P 真挂时 router 仍按探针节奏快速 failover，P 恢复后首个新请求自动闭合断路器。**修复后实测**：故障期失败节奏 30 s → 15.5 s（第 3 次 miss 开断路器）→ 1.07 s/请求（2×500 ms 探针 × router 双尝试）；metaserver 重启 15 s 后首个新请求即 200/0.18 s（park 7 ms），断路器自动闭合，零人工干预。

### lm-eval 精度 gate（gsm8k 5-shot，limit 200，greedy，经由各 endpoint）

| Endpoint | exact_match (strict) |
| --- | --- |
| P/D router 第 1 轮 | 0.960 ± 0.014 |
| P/D router 第 2 轮 | 0.970 ± 0.012 |
| vLLM 直连 | 0.955 ± 0.015 |

- P/D 与 vLLM 直连**同分（噪声内）**——跨引擎 restore 路径无精度损失。
- 两轮 P/D 差 2 个样本：D 的 EP8 批式 decode 非 batch-invariant（batch 组成影响数值），greedy 下仍可能因并发组 batch 不同而在近平局 token 上分叉，属预期。

### 长上下文质量探针：NIAH passkey 检索（已跑，风险降级）

自建 needle-in-a-haystack（passkey 6 位数字，4k/8k/16k × 深度 10%/50%/90% × 4 样本，greedy）——这是对 indexer 风险最尖锐的探针：>2048 时 DSA top-k=2048 真正做选择（16k 时只选 12.5% 的 token），indexer 打分错位会直接漏掉 needle。

| Endpoint | 4k | 8k | 16k | 合计 |
| --- | --- | --- | --- | --- |
| P/D router | 12/12 | 12/12 | 12/12 | **36/36** |
| vLLM 直连 | 12/12 | 12/12 | 12/12 | **36/36** |

- P/D 在 16k（256 块 ≈ 845 MB/请求的跨节点传输）下检索全对，与直连打平。
- 顺带验证了大上下文传输路径（单请求 256 块 RDMA + host 池占用）无异常。

### 残余风险：indexer RoPE 约定分歧（已降级，未完全关闭）

openinfer 的 indexer rope 按**旧版 transformers**（半劈 rotate_half）对齐；新版 transformers 与 vLLM 均为 interleave。上述 NIAH 36/36 说明该分歧在 16k 检索工况下无可见影响（可能 openinfer 权重装载时已做等价变换，或 needle 分数余量足够大）；但 NIAH 是粗探针——若未来长上下文 QA 类任务（longbench 风格）上 P/D 相对直连出现系统性劣化，第一嫌疑仍是这里，修法是把 openinfer indexer rope 切到 interleave 约定（连带 oracle 基线更新）。

## 复跑

两台节点上的 `/root/pd-glm/stack.sh`（A 机：metaserver/pegaserver/vllm/router 四组件，B 机：openinfer-D，从 A 机 vllm 日志自动抓 namespace）+ B 机 `/root/pd-glm/smoke.sh`。D 严格模式下**禁止**直接绕过 router 发请求（键链按"最后一个 token 是 P 生成的"派生，直连请求必然 miss → 全量拒绝）。
