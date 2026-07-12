# GLM5.2 P/D 分离 M2：vLLM-TP8 prefill + openinfer-EP8 decode 执行记录

> **TL;DR:** GLM P/D 里程碑 2：P = vLLM TP8（8×H200 节点 A），D = openinfer EP8（8×H200 节点 B），跨节点 pegaflow P2P，无 MTP/DSpark。**2026-07-12：跨引擎链路全通** —— router 变体 A 下 P/D 输出与 vLLM 基线逐 token 一致（对齐 + 尾块两档），D 全程零 prefill（admit 即 suffix=1）；快压 4k×128 并发 8：32/32 成功、TPOT p99 26 ms（≈median，无 prefill 抢占）。硬依赖：vLLM ≥ 0.24.0 + `--kv-cache-dtype fp8_ds_mla`；D 端 compat 注册用 vLLM 层名 + page-first；restore 后 RoPE deinterleave fixup。剩余：多轮长输入 E2E + 故障注入 + 验收压测；开放风险：indexer rope 约定分歧（>2048 才生效，见文末）。
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
| 4 | E2E：router P/D vs baseline 逐 token 一致 ×3 档 prompt；多轮 delta 只拉增量；杀 P/杀 metaserver 干净失败 | 逐 token 一致 ✅×2 档；同前缀重复请求 `gpu_hit=3 pulled=0`（radix 复用）✅；多轮 delta 长输入档 + 故障注入 ⬜ |
| 5 | 验收压测：多轮长输入（首轮 ~8k、每轮 +2k），P/D vs 2× vLLM mixed（会话亲和），吞吐/TTFT/TPOT 分轮报告 + D 零 prefill 证据（严格模式下 admit 即 suffix=1，无 allow_local_prefill） | ⬜ |

### Bring-up 期间抓出的三个正确性 bug（均已修）

1. **层数 156 vs 99**：vLLM 0.23 给全部 78 层建 Indexer cache；0.24（#45895）起只建 21 个 full-indexer 层。**P 侧 vLLM 版本下限 = 0.24.0**。
2. **RoPE 存储排列**：vLLM（`is_neox_style=False`）旋转后 pair 保持 interleaved `(2i,2i+1)`；openinfer 内核是 interleave-in/block-out `[i, i+32]`——同值不同排列。D 端 restore 后对每个新拉取页做一次 deinterleave（MLA 行 528..656 的 64×bf16 + index-K 行前 64×fp8；rank worker 命令，同 stream 先于后续 step）。未修时症状：t1 正确、后续输出全是 prompt 词汇的乱序拼贴（KV 结构性错位），或直接 DeepEP dispatch 断言崩（bf16 布局时代）。
3. **P 侧尾块保存在前缀命中下不触发**：`scheduled >= prompt_len` 没算上本地 prefix-cache 命中的位置——**多轮 turn≥2 必踩**（P 必然命中上轮前缀）。修为 unscheduled（本地命中+pegaflow load）+ scheduled ≥ prompt_len。

### Bring-up 快压快照（2026-07-12，非验收数据）

`vllm bench serve`（random 4096 in / 128 out / 并发 8 / 32 请求 / temp 0 / ignore-eos）打 router 全 P/D 链路，P = vLLM TP8（8×H200），D = openinfer EP8（8×H200）：

| 指标 | 值 |
| --- | --- |
| 成功率 | 32/32（零失败、零拒绝） |
| TTFT | mean 1226 ms / median 1007 ms / p99 3566 ms |
| TPOT | mean 25.0 ms / median 25.3 ms / **p99 26.1 ms** |
| 输出吞吐 | 213 tok/s（峰值 291） |
| 总吞吐 | 7029 tok/s |

- 零 prefill 全程成立：压测窗口 D 的 admit 全部 `suffix=1`，每请求跨节点拉 64 块（~13 MB RDMA READ，fetch 本体 ~5 ms），park 等待稳定 ~46 ms（等 P 侧 save 注册可见——后续优化点）。
- TPOT p99 与 median 几乎重合 = D 无 prefill 抢占的直接体现；mixed 基线的对比尖刺要等验收压测量化。
- TTFT median ~1 s ≈ P prefill（4k chunked）+ t1 往返 + D admit + router 串行开销；p99 3.5 s 是并发 8 在单 P 实例上的 prefill 排队。
- 局限：单轮、恒定长度、无会话复用、无 mixed 对比——**不可引用为验收结论**。

### 已知开放风险：indexer RoPE 约定分歧（>2048 上下文）

openinfer 的 indexer rope 按**旧版 transformers**（半劈 rotate_half）对齐；新版 transformers 与 vLLM 均为 interleave。seq ≤ 2048 时 top-k 全选、索引分数无关紧要（bring-up 验证不受影响）；**> 2048 时 D 的 index-Q（半劈）与 restore 进来的 index-K（deinterleave 后为 transformers-interleave 布局）配对不一致**，top-k 选择可能偏离。验收压测（首轮 8k）会实测暴露；若输出劣化，修法是把 openinfer indexer rope 切到 interleave 约定（连带其 oracle 基线更新）。

## 复跑

两台节点上的 `/root/pd-glm/stack.sh`（A 机：metaserver/pegaserver/vllm/router 四组件，B 机：openinfer-D，从 A 机 vllm 日志自动抓 namespace）+ B 机 `/root/pd-glm/smoke.sh`。D 严格模式下**禁止**直接绕过 router 发请求（键链按"最后一个 token 是 P 生成的"派生，直连请求必然 miss → 全量拒绝）。
