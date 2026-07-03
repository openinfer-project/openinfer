# GLM5.2 P/D：vLLM prefill + openinfer decode（决策记录）

> **TL;DR:** **里程碑 1（qwen3 smoke）已跑通**：node 34 上 vLLM-P + openinfer-D，3 档 prompt greedy 输出与直连 baseline **逐字节一致**，多轮 delta 复用实证（对齐 prompt 只拉 1 块增量）；TTFT 交接开销 470tok **+14ms** / 1.8k **+51ms** / 7k +147ms（其中 RDMA 992MiB 42ms @23.7GiB/s，达标线 ≤2k）。跑通的兼容等式见 §3.1——核心三条：P `--block-size` 必须等于 D 的 GPU page size（hash 粒度、seal 粒度、1:1 load 映射全部锚在它上）；两侧 per-layer 注册 slot 数相等（vLLM fork 对 qwen3 默认 v2 runner = per-layer 36 slot，**cross-layer 是歧路**）；slot 内 K/V 段布局差异由 pegaflow #382 修复吸收（Contiguous 设备布局加载 split-KV peer 的两段式块）。原有决策不变：GLM prefill 走 vLLM（TP8+EP8）、P→D 就绪 = D 端有界快轮询、D 侧 vllm-hash-compat provider（钉 `xxhash_cbor` + 统一 `PYTHONHASHSEED`）。**未完成**：尾块 connector 扩展 + router 变体 A（当前 D 兜底计算 ≤block_size 的尾巴）、严格 no-prefill 429/500 开关（当前 miss 等 5s 后 scratch 兜底）。GLM 接入为里程碑 2（依赖 PR5）。
>
> **Last touched:** 2026-07

相关：`dp1-ep8-decode-plan.md`（decode 侧 PR5 是 D 节点上线的关键路径）· `../qwen3/pd-disaggregation-m2.md`（M2 全流程，本文大量复用其结论）· pegaflow `docs/pd.md` / `docs/pd-rdma-push.md`。

## 1. 决策一：GLM prefill 走 vLLM

### openinfer 自建 prefill 的账单（审计结论，2026-07）

decode-only 假设焊死在每一层，证据：

| 组件 | decode-only 证据 |
| --- | --- |
| FlashMLA sparse attention | AOT 实例化写死 `kSq = 1`（`csrc/glm52/glm52_flashmla_sparse.cu:16`），只编了 `sm90/decode/sparse_fp8/splitkv_mla.cuh` |
| DeepGEMM indexer logits | 只包了 paged 变体，`next_n ∈ {1,2}`（`ops/glm52/deepgemm_mqa.rs:35`） |
| DeepEP MoE all-to-all | shim 只实例化 elastic **decode** dispatch/combine，128 tokens/rank 上限，延迟优化非吞吐 |
| Executor | bs=1 串行 coordinator，"prefill rides decode token-by-token"（`runner.rs:416`），bring-up 专用 |
| **attention 并行** | **不存在**。DP1：attention/dense/bookend 全在 rank 0，ranks 1–7 只当 MoE expert 工人。prefill 是算力瓶颈，单卡 attention = 扔掉 7/8 FLOPs |

自建需要：attention 分布（TP8-MLA 或 DP-attention，全新）+ FlashMLA sparse prefill 新 AOT/wrapper + DeepGEMM `fp8_mqa_logits`（prefill 变体已 vendored，缺 wrapper）+ DeepEP normal 模式 torch-free shim + chunked-prefill executor/scheduler（PR5 本身未落地）。数周量级，且全程占 8×H200 做 gate。

### vLLM 侧为什么顺

- GLM5.2 在 vLLM 是 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` 一行继承，TP8+EP8 prefill 开箱即用；vLLM 本来就是本 campaign 的 production reference。
- KV 格式两边是**同一个 kernel contract 的两个实现**：MLA cache 656 B/token、page 64（512 fp8 nope + 4×f32 scale + 64×bf16 rope）= vLLM 的 `fp8_ds_mla`；indexer K cache 两边都是 DeepGEMM block-split paged 布局。需字节级验证，但不是格式转换。
- pegaflow→openinfer 的 KV 通路 M2 已打通（openinfer #522 / pegaflow #381）；GLM 扩展为双 cache：78 层 MLA + 21 层 indexer K cache（57 个 shared 层无 indexer）。

**将来若收回 prefill**：更自然的形态是 DP-attention + EP8（MLA 是 MQA、latent cache 每 rank 全量，与现有 DeepEP shim 直接组合），而不是 TP8。由 decode 侧稳定后的真实流量测量触发。

## 2. 决策二：P→D 就绪信号 = D 端有界快轮询

### 问题

pegaflow CPU P2P 链路上，P 结束请求后有一段 offload 尾巴：D2H save（后台线程）→ 写管线 seal → metaserver 注册（**fire-and-forget**，事件驱动攒批，`pegaflow-core/src/metaserver_client.rs:188-222`）。router 是同步流程：等 P 的 HTTP 首 token 响应就转发 D，**不等注册**（`pegaflow-router.rs:206`）。所以 D 可能在注册落地前 query → zero-hit。

M2 里这个竞态不存在：openinfer-P 的 `flush_on_finish` 把 `Finished` 事件扣到注册对 peer 可见后才释放，P 的响应即 ready 信号。vLLM 做 P 后此保证消失——vLLM 的 `wait_for_save` 只入队即返（scheduler 不阻塞，`connector/worker.py:646-660`），`get_finished` 报 sent 也不等注册。**vLLM connector 从未调用 pegaflow-core 里现成的 `flush_saves_and_registrations` 屏障**（`lib.rs:738-748`，为 M2 而造）。

而 openinfer-D 现状：`RemoteFetch` 相位只等 pegaflow 的 `Loading`（拉取在途）；metaserver 无记录时 query 立即空手 → `Scratch` → 本地 prefill（`executor/remote_fetch.rs:78-80`）。

### 三个候选的取舍

| 方案 | 结论 | 理由 |
| --- | --- | --- |
| vLLM/router 异步 `/kv_ready` callback（pd.md 规划过、未实现） | ❌ | router 变有状态（pending 表、callback 丢失超时、P 崩溃清理），改 vLLM connector + router 两处，串行链路延迟与轮询相当，收益只是省几次 query |
| **D 端有界快轮询** | ✅ | 窗口由 P offload 尾巴决定，**有界且短（几十 ms）**，"太细 metaserver 撑不住 / 太粗 TTFT 难看"的两难不成立：5ms 间隔 × 窗口 ≈ 每请求 2–10 次 query，QPS 随请求速率线性而非并发数；改动全在我们自己的代码里 |
| RDMA doorbell（内存代表请求状态） | 推迟到 M3 | 正确形态 pegaflow 已造好——`pd-rdma-push` v2 + WRITE_WITH_IMM（P4 阶段，vLLM↔vLLM 跨机 TP8 已通）。当长 prompt 让 CPU 往返物理成本顶破预算时，答案是整体切 M3，不是更聪明的轮询 |

### 改动清单

1. **openinfer D**（核心，很小）：`remote_fetch.rs` 决策函数加分支——请求带 "expect remote KV" 标记时，zero/partial-hit 不 `Scratch`，进入带独立短 deadline（~500ms）的重查询等待；轮询由现有 per-tick 机制驱动，加 ~5ms 最小间隔节流。该函数是 #532 刚抽出的纯决策逻辑，单测顺手。
2. **"expect remote" 标记**：router 转发 D 时注入字段，或 D 全局配置 "P/D 模式冷请求默认等待"。
3. **GLM 定制等待谓词**：D 必须等**完整前缀**命中才放行——partial-hit 的 suffix 重算在 qwen3 是优雅退化，在 GLM 是 token-by-token 骑 decode 内核，不可接受。同理 deadline 超时的 `Scratch` 回退对 GLM 改为 fail/requeue（router 重派或 503），做成策略开关。
4. **（可选）pegaflow connector 一小刀**：`_process_save_batch` 处理完 finished 请求后调现成的 `flush_saves_and_registrations`，压掉注册尾巴。先测窗口分布再决定，轮询本来就能吸收。

### TTFT 开销预算（<50ms 目标，GLM 8k prompt ≈ 430MB KV）

物理成本：P 端尾部 D2H（chunked prefill 期间流式，尾巴≈最后一 chunk）+ D 轮询期望 ≤5ms + RDMA READ ~20ms（M2 实测 21.7 GiB/s）+ H2D ~18ms。合计 ~40–50ms，预算内但紧；**这是 CPU 中转的结构性成本**——若长 prompt 顶破预算，触发 M3（GPU→GPU push），不在轮询上做文章。

## 3. 前置风险项：跨引擎 hash key（最早能证伪，先做）

M2 的"无 handle"协议靠 P/D 确定性推导相同 key。两边算法**同为 xxhash 族但完全不同**：

| | vLLM | openinfer / kvbm |
| --- | --- | --- |
| 算法 | `xxh3_128`（或 `sha256_cbor`，`--prefix-caching-hash-algo` 可配） | `xxh3_64`（dynamo `compute_hash_v2`） |
| 输入 | CBOR 编码的 `(parent_hash, token_ids_tuple, extra_keys)`（`vllm/v1/core/kv_cache_utils.py:563`） | token 块字节直接链式 |
| 链根 / seed | `NONE_HASH`：`PYTHONHASHSEED` 未设时 = `os.urandom(32)`（**跨实例不可复现**）；设了 = `hash_fn(seed)` | 固定 base seed **1337**（`ROUTER_XXH3_SEED`），LoRA 加盐 |

**决策：openinfer-D 侧复刻 vLLM 的 hash 推导，pegaflow connector 不改。** 备选的"connector 重算引擎中性 key"被否——不想动 connector，且 dynamo 上游有对齐 vLLM hash 的趋势，届时 openinfer 换成上游实现即可白嫖。`PYTHONHASHSEED` 钉死可接受。

复刻的精确语义（vLLM `utils/hashing.py` + `v1/core/kv_cache_utils.py:563`）：

- **必须钉 `--prefix-caching-hash-algo xxhash_cbor`**。裸 `xxhash`/`sha256` 变体走 **pickle** 序列化，Rust 无法复刻；`_cbor` 变体是 `cbor2.dumps(input, canonical=True)`（RFC canonical CBOR），可逐字节复刻。xxh3_128 → 16 字节 key，恰好等宽于 openinfer 现有 content key（`pool.rs:602` 的 `[u8;16]`）。
- 每块 key = `xxh3_128(cbor((parent_hash: bytes, token_ids: tuple[int], extra_keys)))`，text-only 下 `extra_keys = None`。
- 链根 `NONE_HASH = xxh3_128(cbor(PYTHONHASHSEED 字符串))`；**所有 P 节点与 D 的推导配置必须钉同一个 seed**（未设时 vLLM 用 `os.urandom(32)`，跨实例不可复现——部署上 fail-fast 校验）。
- Rust 侧落点：一个 vllm-hash-compat key provider（canonical CBOR 编码 `(bstr, [uint...], null)` + `xxhash-rust` 的 xxh3_128），与 kvbm `SequenceHash` provider 并列，P/D 模式下选用。纯函数，golden vector 单测（从真实 vLLM 进程抓取）。
- **漂移守卫**：vLLM 代码注释明示默认算法在迁移中；P 节点 vLLM 版本升级时必须重跑兼容 gate。extra_keys 语义（cache_salt/mm/LoRA）超出 text-only 范围时 gate 会红。

同时对齐 namespace（pegaflow `derive_namespace(model, TP)`，openinfer 查询时用同一字符串——M2 已有 4B/8B 同几何静默交叉命中的前车之鉴）与 hash 块粒度（GLM page 64 = vLLM `hash_block_size`）。

### 3.1 兼容等式（里程碑 1 实测拍平，qwen3 已验证）

P 存的块能被 D 加载，当且仅当下面全部成立。前三条决定 query 是否命中，第四条是 pegaflow 引擎唯一强制的 load 守卫，第五条引擎不校验、错了**静默数据错位**（pegaflow #382 后 Contiguous←split 这一种组合被吸收，其余组合仍需人工对齐）：

1. **namespace 相等**。`derive_namespace` = sha256(model/dtype/tp/pp/num_kv_heads/head_size/num_hidden_layers/cache_dtype/dcp/pcp/cross_layer_blocks/mla_layer_split)[:8]——**`block_size` 不在 factor 里**；qwen3 栈两侧 = `cd6ed6c5`。openinfer D 侧用显式 override 传入，不自己推导。
2. **块粒度相等 + hash 算法相等**。openinfer 的 hash/seal/load 三条链全部锚在 GPU page size 上（compat hasher 直接读 `budget.block_size`，remote 块→GPU page 是 1:1 映射，无重切分），pegaflow 层面也没有跨粒度重切分。**结论：P 的 `--block-size` 迁就 D 的 page size（qwen3 = 16），openinfer 零改动。** 里程碑 1 首轮 5s 全 miss 的根因就是 P=64 vs D=16——hash 从第 0 块就对不上；query miss 只可能来自 namespace/hash/owner-TTL，拓扑不匹配是 load 期报错，两个失败面不要混。
3. **`total_slots` 相等**（layer-first = `num_layers × tp_size`）。vLLM fork 对 qwen3 默认 v2 model runner，per-layer 注册 36 slot，与 D 天然相等。**cross-layer（1 slot）是歧路**：它只存在于 v1 runner（`VLLM_USE_V2_MODEL_RUNNER=0`），且与 D 的 per-layer 注册必然撞 slot-count 守卫；污染过的 pool 两种布局共存后连 P 自己都加载不了（"namespace is shared by incompatible KV layouts"）——切布局配置必须清池重启。
4. **slot 内 K/V 段布局**。vLLM FA NHD `(2, blocks, ...)` → connector 推断 K/V split 两段；openinfer page-first fused → 单段 `[K|V]` 连续。pegaflow #382 让 Contiguous 设备布局按段拆两笔拷贝吸收这个差异（并加了段长必须恰好铺满设备块的守卫）。GLM/MLA 是单段 latent，天然无此问题。

**第一个 gate（不依赖任何 scheduler/轮询改动）**：vLLM-P 存一段已知 prompt 的 KV → openinfer-D 用 compat provider 推导的 key 查询 → 命中且**逐字节比对** MLA cache（656B layout）与 indexer K cache（DeepGEMM block-split layout）。这个 gate 同时证伪 hash 复刻和字节布局两个风险。prompt 至少两档：**对齐（`% 64 == 0`）与非对齐**，让尾块的 key 推导和字节布局从第一天就在 gate 覆盖内，不留到集成期。

### 尾块问题：vLLM 只 hash 满页，而 D 禁止任何 prefill

vLLM 只对满页（64 token）生成 block hash；`prompt_len % 64` 的尾巴是 partial block——有 block_id、无 hash，connector save_intent 里不存在。P 存进 pegaflow 的最长前缀 = `floor(prompt_len/64)×64`，D 命中后还差 0–63 token 的 KV。GLM 的 D 节点补这个尾巴 = token-by-token 骑 decode 内核（今天 ~50ms/步，worst case ≈3s），且"decode 节点做 prefill"污染调度——**策略红线：D 零 prompt 位置计算**。

**决策：P 端 connector save 路径加一个加法式尾块扩展（P/D 模式 gate 住）。** 请求结束时对 partial block 用同一个 hash 函数派生 key——`xxh3_128(cbor((last_full_hash, tail_token_ids, None)))`，vLLM 自己不算这个但函数良定义，D 侧 compat provider 同样可推导——并把该页一并 D2H。注意与被否的"中性 key"区分：key 体系仍是 vLLM 原生 hash + D 侧复刻，此处只是 save 覆盖面的小扩展；不追 vLLM 内部实现、不会被 dynamo 上游对齐作废（上游对齐的是满块 hash，尾块本来就是 vLLM 不覆盖的空白）。`prompt_len % 64 == 0` 时无尾块，路径退化。

**首 token 归属**（尾块之后 prefill 的最后一丝影子）：

- **变体 A（选定，D 零 prompt 计算）**：router 把 P 响应的 t1 直接发客户端（TTFT = P 响应时刻），并把 t1 追加进转发给 D 的上下文；D 首步 = 对 t1 的真 decode（prompt KV 全部就位，t1 的 KV 由该步写入）。采样一致性天然成立。
- 变体 B（vLLM 式备选）：D 重算最后一个 prompt token 的单步 forward 出 t1（vLLM full-hit 的 `num_computed_tokens -= 1` 语义）。greedy 下与 P 一致，temp>0 时 P 的 t1 作废、P 的一步 decode 纯浪费——最终指向 P 侧 prefill-only 模式（对应 openinfer #526 同类项）。

## 4. 补充拍板（2026-07 讨论）

- **P 端 CPU pool 容量不是问题**：P 节点是 TB 级 CPU memory，evict-before-fetch 按不发生设计；不做保护期机制，出了再说。
- **D 禁 prefill 的失败语义**：miss/timeout 一律对上游回 429/500（可重试），router/客户端重试即重走 P（内容寻址天然幂等）。不做更花的重派机制。
- **多轮 delta 拉取**：pegaflow 内容寻址前缀匹配自动覆盖（P 每轮重 prefill delta 并注册，D 前缀命中旧块、只拉新增），无需新机制——M2 已实证（turn2+ 只拉 64 块 ~15ms）。
- **TP→DP 层映射**：MLA latent 天然 TP 无关（每 rank 全量副本），connector 已有 layer-split 注册，非问题。
- **GLM PR5 与 P/D 的耦合方式**：推迟到 GLM 接入阶段再定（见 §5 路线）。

## 5. 路线：qwen3 smoke test 先行，GLM 后接

**里程碑 1：qwen3 的 vLLM-P + openinfer-D P/D smoke test —— 已跑通（2026-07，node 34）。**

| 步骤 | 状态 |
| --- | --- |
| 1. hash + 字节布局兼容 gate | ✅ 以 e2e 形态验证（§3.1 兼容等式；3 prompt 逐字节一致本身就是布局 gate） |
| 2. vllm-hash-compat key provider | ✅ `openinfer-kv-offload/src/vllm_hash.rs`，golden vector 单测 ×5 |
| 3. 尾块 connector 扩展 + router 变体 A | ⬜ 未做——当前 D 兜底计算尾巴（≤block_size=16 token，qwen3 上可接受；GLM 前必须做） |
| 4. `RemoteFetch` 等待分支 | ✅ miss 窗口内 5ms 重查询；严格 no-prefill 429/500 开关未做（当前 miss 等满 5s 后 scratch 兜底，输出正确） |
| 5. e2e smoke + delta 复验 | ✅ 3 prompt BYTE-IDENTICAL；对齐 prompt 只拉 1 块增量（内容寻址 delta 复用） |

**TTFT A/B**（unique-prefix 冷跑，3 样本取中位，P 直连 vs router P/D）：470tok 34→48ms（**+14ms**）；1.8k tok 55→106ms（**+51ms**）；7k tok 204→351ms（+147ms，其中 RDMA 992MiB 42ms @23.7GiB/s，其余为 P save D2H + 发现轮询 + H2D 回灌——都可流水化，follow-up）。≤2k 在 50ms 预算内。

环境与复跑：node 34 `/data/pd-stack/`（`stack.sh` 组件级启停 = pidfile+setsid 进程组整组杀+清 GPU 显存等待；`smoke.sh` 严格校验 HTTP 200+JSON；`ttft.py` A/B）。openinfer 分支 `feat/pd-vllm-hash-compat`；pegaflow 依赖 [#382](https://github.com/novitalabs/pegaflow/pull/382)（Contiguous←split 加载修复）。**测 P/D 命中前必须重启 D 清 prefix cache，否则量到的是 D 本地缓存**（baseline 请求会把 prompt 灌进 D）。

**里程碑 2：GLM5.2 接入**（另起，依赖 PR5）——GLM 的 paged-KV/scheduler/kvbm 基础设施 + 双 cache（656B MLA + indexer）的注册与字节 gate + GLM 等待谓词（完整前缀 + no-prefill 硬约束）。PR5 scheduler 是否按 P/D-ready 设计届时定。

decode 侧 PR5（scheduler + CUDA graph）照旧并行推进——它是里程碑 2 的前置。
