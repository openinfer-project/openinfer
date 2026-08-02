# pegaflow KV 卸载接入 Spec

> **TL;DR**: 把 `pegaflow-core` 当**进程内 Rust 库**做 KV 卸载的物理后端（HBM→DRAM/SSD/RDMA），补上 kvbm 留着没写的卸载层。connector 大脑（决定 load/save 哪些 block）用 kvbm logical/physical 分层思想自建，pegaflow 退为语义无关的 raw block transfer 后端。**路线已调整为 Qwen3-4B full-attn 首发**（原计划 Kimi 首发）：page-first 单 buffer 经 pegaflow `block_stride_bytes`（PR #331）适配。**端到端已在真实 GPU 上跑通并验证**：async SAVE + 前缀恢复接进 `Qwen3Executor` + scheduler，`tests/kv_offload_cpu_hit.rs` 覆盖纯 CPU-hit 与 GPU+CPU 组合 hit，恢复后 logits 与冷算一致；连接层 `OffloadEngine` + `tests/cpu_roundtrip.rs` 字节级一致。**2026-08 qwen3 已迁到 `openinfer-kv-store` 承载**（写路径 seal/retire、读路径 scheduler 侧 `resolve_prefix`，见 §0/§9；`openinfer-kv-offload` 连接层此后仅 glm52 在用）。默认关（builder flag opt-in）；**server CLI 已接**（#316：`--kv-offload` / `--kv-offload-host-gib` / `--no-prefix-cache`，plain 与 `--enable-lora` 两条启动路径都透传）。纯-L2 基准实测 Qwen3-4B mean TTFT 195→40ms（−79%，evict-before-probe → `gpu_hit=0`，全前缀从 host tier 恢复）。**Qwen3.5 linear/SSM state 明确排除**；**DeepSeek sparse 暂缓**。
>
> Last touched: 2026-08

## 0. 实现状态（2026-08）

已落地并验证：

- **pegaflow `block_stride_bytes`**（PR #331 → novitalabs/pegaflow，`feat/inproc-load` 基于其上）：解耦"块间步长"与"每块拷贝大小"，让 page-first fused buffer 能注册。**已合入 master**。
- **pegaflow 进程内 load API**（PR #333，**已合入**，squash 进 #331 的 `07cac7e`）：`LoadCompletion::{Shm,Channel}` + `batch_load_kv_blocks_multi_layer_inproc` → `oneshot::Receiver`，去掉 in-process 调用方对 shm `LoadState` 的依赖（Rust 进程内不需要），非阻塞 poll。
- **`openinfer-kv-offload::OffloadEngine`**：拥有 `PegaEngine` + 内嵌 tokio runtime；`Registration::from_buffer` 把 fused page-first buffer 映射成 per-layer 注册（**单段 `[K|V]`**：fused layout 里 K/V 本就连续 = `layer_stride` 一段，`block_stride = page_stride`，`segments=1`——不是 K/V split，那条路需要 `kv_stride > bytes_per_block`，此处不成立）。crate 已按 `config`/`host`/`layout`/`handle`/`engine` 分模块（#799/#802 重构）；所有操作统一为 `OffloadHandle<T>` 可轮询句柄：`submit_save`/`submit_query`/`load` 提交到 pegaflow runtime 后立即返回 handle，`save`（fire-and-forget）/`save_blocking`/`query` 是其薄包装（glm52 在用；qwen3 已迁往 `openinfer-kv-store`，见下）。`query` 透传 pegaflow 0.23.5 的 `wait_for_full_prefix`（glm52 native handoff 传 `true`：D 无法重算 miss，partial 命中无用）。
- **`KvBuffer::device_ptr`**（原 kv-cache，物理 KV 层已随 qwen3 首迁搬入 `openinfer-kv-store` 自包含）：注册用的稳定基址。
- **kvbm↔bytes 桥**（`RequestKv`，同在 kv-store）：`assigned_block_hashes` / `prefix_matched_blocks` / `assigned_block_guards`，kvbm `SequenceHash` → 16B content key。
- **`tests/cpu_roundtrip.rs`**：真实 `KvBuffer` 上写已知 pattern → save → query → load 到**另一组** block → 字节级比对 + 零块负向控制。**通过**。
- **live 接线（§9，已落地；2026-08 起由 `openinfer-kv-store` 承载）**：`Qwen3Executor` 持 `Option<Arc<KvStore>>`（`Qwen3OffloadOptions` opt-in，默认关，不变）；写路径在封块边界与请求退休时自动 `store.seal` / `store.retire`（`Cacheable` fire-and-forget，`flush_on_finish` 时 `Handoff`）；读路径为 scheduler 侧 `ResolveHub`——submit 时对非 echo 请求在 store 自己的 runtime 上 spawn `resolve_prefix`，带 `KvPrefix` RAII hold 经 mpsc 回主循环再 admit，hold 护住排队期，首个 prefill chunk 的 `match_and_add_prefix` 重 pin 后即放（executor 侧的 `begin_kv_prefetch` 状态机已整体删除）。`tests/kv_offload_cpu_hit.rs` 单测序跑两幕——纯 host tier 恢复（radix GPU hit == 0；act1 prefill + `drop_request` 自动 seal → `flush_offload_saves` → `evict_cached_blocks`，act2 手动 `store.resolve_prefix` 恢复后 match）与 GPU+host 组合 hit（3+3 块拼成一段连续前缀）——恢复后 first-token logits 与冷算一致的容差不变（argmax regret ≤0.20 / head mean ≤0.06 nat，bf16 floor）。
- **三处正确性加固（toxic-review 后）的现归属**：① query lease 显式释放收编进 kv-store——`resolve_prefix` 在 `reserve_loaded_blocks` 失败（池装不下时不当 TTL-park，释放租约后 sleep 重查）与取消路径上调 `tier.release`（`PegaflowTier::release` → `release_query_lease`；`openinfer-kv-store/src/store.rs`、`tier.rs`），依旧不泄漏到 600s TTL；② admission 拒绝（context/KV budget/未知 LoRA）的清理对应 `PendingRequest.kv_prefix` 的 RAII——hold 随请求析构释放，已 commit 的 block 自然归还，不再有"已 settle prefetch 状态"这条专门泄漏路径；③ async SAVE 的防腐蚀 guard 收编进 `KvStore::seal`——按 `assigned_block_guards` 收集 `KvBlockGuard`（与被存 block 1:1）随 save task 走，`PegaflowTier::save` 在 D2H 落地后才 drop，"请求结束→slot 重分配→在途 D2H 抓到新 KV 写进旧 hash"的静默腐蚀窗口依旧封死，且钉住页计入 `pinned_blocks` 从 admission 预算扣除。

**server CLI 已接（#316，迁移后不变）**：`--kv-offload`（bool）/ `--kv-offload-host-gib`（f64，默认 8.0，pegaflow 启动即整块 `cudaHostAlloc`，RSS 立即反映）/ `--no-prefix-cache`（vLLM 风格；不带 offload = 关前缀匹配，带 offload = 纯-L2 模式，resolve 前的 evict-before-probe 使每个前缀从 host tier 恢复）。plain 与 `--enable-lora` 两条路径都透传 `offload_options` + `no_prefix_cache`；LoRA 下安全，因 resolve/probe 的 scope 以 adapter 名作 salt（qwen3 scheduler 填 `CacheScope.lora`，`compute_salt_hash` 照旧把它揉进 block hash），恢复的 KV（HBM 或 host tier）永不跨 adapter。三处 #316 review 加固的现行形态：echo 请求不进 resolve（其 prefill 跳过 `match_and_add_prefix`，恢复的块用不上）、admission 按 `resolved_prefix_blocks`（`hit_tokens / block_size`）抵扣 resolve hold 已钉住的块、拒收时 hold 随 `PendingRequest` 析构释放。**依赖已从 fork 摘除**：PR #331+#333 均合入上游 master（squash 进 `07cac7e`），`third_party/pegaflow` 已删，`pegaflow-core` 改为 pin 到上游 rev 的 **git 依赖**（机制见 §5.2，pin 的 rev 此后随 #381/#395 等推进），GPU 测试在 git-dep 下行为不变（delta 一致）。

相关：[kv-cache-design.md](kv-cache-design.md)（logical/physical 分层，已把 pegaflow 列为设计调研）· [qwen3-kvbm-integration-spec.md](qwen3-kvbm-integration-spec.md)（kvbm-logical 已接入）· `models/kimi-k2/kv-cache-design.md`（Kimi 已用 `BlockPool`）· `models/qwen3/prefix-cache.md`（HBM 内前缀复用已落地）。

---

## 1. 定位：pegaflow 是 raw 后端，connector 大脑要自建

pegaflow（`third_party/pegaflow`，novita，Apache-2.0）原本是 **vLLM 的 KV connector 服务端**：KV 的编排逻辑（何时 save、query 几个 block、prefix 匹配、与 scheduler 的 admission/preemption 交互）全在 vLLM 的 Python connector 那一侧，`pegaflow-core` 只是底下干 D2H/H2D + 分层存储的**肌肉**。

openinfer 不是 vLLM，那套 Python connector 一行用不上。接入要做的是**用 Rust 自建那颗 connector 大脑**——而 kvbm 的 logical/physical 分层正是它的骨架：

```
per-model scheduler   ← 策略：哪些 block 该 resident（full 前缀 / MLA 全前缀 / 未来稀疏选择）
  ↓ 产出 load/save 意图（一组 block）
connector（kvbm logical/physical 思想）← 机制：block identity、状态机、GPU slot 编排、transfer 调度
  ↓ 语义无关的 raw transfer
pegaflow-core         ← 机制底座：D2H/H2D、DRAM/SSD/RDMA 分层
```

## 2. 战略决策：pegaflow 取代 kvbm 死代码做物理 tier

openinfer 仓里 vendored 的 `kvbm-physical` / `kvbm-engine` 设计目标就是分层卸载，但**至今零接线、是死代码**（无任何非 kvbm crate 依赖）。同时养两套分层卸载违反项目复杂度红线。本 spec 采纳：**`kvbm-logical`（逻辑层 + 前缀匹配）保留，pegaflow-core 顶替它下面缺失的物理卸载层，砍掉 `kvbm-physical`/`kvbm-engine`**。理由：pegaflow 同组维护、已上 PyPI、有 H800 benchmark、库化干净；kvbm 那两层是纯负债。已执行（2026-06）：vendored `kvbm/` 目录只留 `kvbm-logical` fork，`dynamo-tokens`/`dynamo-kv-hashing` 改为 ai-dynamo/dynamo 上游 git 依赖（pin rev），其余 8 个 vendored crate 删除。

## 3. 三模型三 KV 形态 → connector 边界（实据）

| 模型 | KV 形态 | active set | 跨请求复用 | 卸载结论 |
| --- | --- | --- | --- | --- |
| **Qwen3 / Qwen3.5 full-attn** | paged，page-first 单 buffer，`PagePool` | 无（dense 全前缀） | 有（前缀缓存已落地） | **已首发（#316）**：page-first 与 pegaflow `stride==copy-size` ABI 冲突已由 `block_stride`（§5.R1）解掉，端到端跑通 |
| **Kimi-K2 MLA** | paged，per-layer ckv/kpe arena，后端是 `BlockPool`；latent 68.6 KiB/token，无 per-head | 无（dense 全前缀） | 有（HBM 内 prefix cache 已落地） | **下一候选**：layout 直接适配 pegaflow registration（接入面最干净），复用 Qwen3-4B 这套 connector 模式即可 |
| **Qwen3.5 linear（24 层）** | per-request `RecurrentState` [32,128,128] f32 2 MiB/层，非 paged、独立分配 | 无（每步读写整个 matrix） | **零**（this-request 有损摘要，非 content-addressable） | **排除**：offload 无 prefix/dedup 收益；省显存是 per-request swap-out，另一套机制 |

**边界结论**：connector 只收 **block-structured、content-addressable** 的 KV（MLA latent / full-attn paged）。recurrent/SSM state 不进 connector。稀疏的 active-set gather 是独立的、未来的课题。

证据：Kimi `openinfer-kimi-k2/src/runner/{worker.rs:612-619, cache.rs:63-80, mla.rs:38-48}`、`scheduler.rs:16,27,146,180`、`pool.rs:123`；Qwen3.5 linear `openinfer-qwen35/src/...recurrent.rs`、`batch_decode_graph.rs:82-86`。

## 4. 路线

1. **Qwen full-attn 已首发（#316），本轮迁接 kv-store** —— 给 pegaflow 加了 `block_stride_bytes`（R1）解掉 page-first ABI 冲突，async SAVE + 前缀恢复经 `Qwen3Executor` + scheduler 端到端跑通；2026-08 qwen3 从 `openinfer-kv-offload` 迁到 `openinfer-kv-store` 承载（seal/retire 写路径 + scheduler 侧 `resolve_prefix` 读路径），server CLI 不变。
2. **Kimi MLA 下一候选** —— pegaflow 做 `BlockPool` 下的 host/SSD tier；block evict 时 demote 到 host，前缀 query 命中时从 host restore。带宽便宜（latent 小），layout 零阻抗，直接复用 Qwen3-4B 的 connector 模式。
3. **linear 排除、sparse 暂缓**。

## 5. 可行性（对抗验证结论，附证据）

四条承重假设由 10-agent workflow 对抗验证：

1. **✅ 进程内注册裸指针，无 IPC、无第二进程**：`register_context_layer_batch(data_ptrs: &[u64])`（`pegaflow-core/src/lib.rs:242-259`）收裸设备地址，拷贝路径直接喂给 driver API `cuMemcpyDtoHAsync_v2`（`transfer/memcpy.rs:82-89`）；IPC 只在 server/Python 层，core 零 IPC 调用点。cudarc 附设备 **primary context**（与 openinfer 同一），自建 worker stream。
2. **✅ 依赖无致命冲突**：cudarc 单 major（0.19.3↔0.19.7 统一），cuda-12080/12090 共存（build.rs 取高版本），tokio/tonic/prost 兼容。**依赖行**（git rev pin 到上游 master `07cac7e`，含 #331+#333；`default-features=false` 砍掉 pegaflow 自带的 `cuda-12`/`rdma`，靠 workspace cudarc 提供的 `cuda-12090`+`nvrtc` 满足——pegaflow-core 无 `cfg(cuda-12)` gate）：
   ```toml
   pegaflow-core = { git = "https://github.com/novitalabs/pegaflow.git", rev = "07cac7e50e8ae7be15ad1b9311401039c9ee439b", default-features = false }
   ```
   下次再改 pegaflow：临时换回 path dep 共同开发 → 提 PR → 合入后 re-pin rev。
   **为何 `cuda-12` 而非 `cuda-13`**（本机明明是 CUDA 13.3 toolkit / 13.0 driver）：openinfer 有意锁 `cudarc/cuda-12090`（`Cargo.toml:92-93`，issue #263——配 cudarc 0.19.5+ 的 per-symbol lazy loading，压低 binding level 以**不抬高 runtime driver floor**、保宽部署兼容；故意不用 `cuda-version-from-build-system` 自动，否则 driver floor 会跟着构建机 toolkit 走）。cudarc 在 workspace 是**单实例、feature 取并集后选最高版本**：pegaflow 用 `cuda-12` 并集后仍是 12090、不抬 floor；用 `cuda-13`（→ `cudarc/cuda-13000`）会把**整个 workspace 含 openinfer 自己**顶到 13000、driver floor 抬到 CUDA 13，撞翻 #263。整体迁 cu13 是独立决策（须同时改 openinfer 的 cudarc + revisit #263），本期不做。
3. **⚠️ Layout**：block-hash 键直接适配（`u64→Vec<u8>`）；page-first layout **不适配**（见 §5.R1）；Kimi per-layer 布局**天然适配**。
4. **✅ 流同步**：host-side 粗同步可解——save 前 openinfer 必须 `synchronize()` compute stream（pegaflow 私有 stream 只自同步，`gpu_worker.rs:520-528`），restore 前自旋 poll `LoadState`。代价：损 compute/offload 重叠（见 §6.R3）。

## 6. connector 接口（dense-first，稀疏留门不展开）

两层分离，稀疏复杂性全关在 policy 侧：

```rust
// mechanism —— pegaflow backend，永不懂稀疏/前缀
trait KvOffloadBackend {
    fn load(&self, items: &[(BlockHash, GpuSlot)]) -> LoadHandle; // 任意集合，phase 无关
    fn offload(&self, items: &[(GpuSlot, BlockHash, OffloadHint)]);
    fn poll(&self, h: LoadHandle) -> LoadState;
}
enum OffloadHint { ReusableAcrossRequests, TransientDiscard }

// policy —— per-model scheduler，懂自己的拓扑
trait KvResidencyPolicy {
    fn required_blocks(&self, req: &RequestCtx, phase: Phase) -> SmallVec<BlockId>;
    fn save_hint(&self, block: BlockId) -> OffloadHint;
}
```

**现在做对、未来免费受益的三个决策**（即便 dense-first 也按这个写，成本为零）：
- 接口说 **block 集合**不说 prefix-count（full attention 产出的集合恰好连续 = 退化特例）；
- admission 按 **active working set ≤ HBM** 写（dense 下 active=total，退化）；
- `load` **phase-agnostic**（不绑 prefill，未来 decode gather 是"启用"不是"重设计"）。

第一版：`required_blocks` 对 Kimi/Qwen 就是"全前缀"，`OffloadHint` 全 `ReusableAcrossRequests`，只走 prefill-前 + evict 路径。

## 7. 风险

| # | 风险 | 等级 | 处置 |
| --- | --- | --- | --- |
| R1 | Qwen page-first vs pegaflow `stride==copy-size` ABI 不兼容 | major | 给 `KVCacheRegistration` 加 `block_stride_bytes`（改 pegaflow ~几十行，`instance.rs` + `transfer/mod.rs`）；**Kimi 首发绕开此风险** |
| R2 | save 前漏 `synchronize()` → 静默 D2H 半写 KV，pegaflow 不校验 | major | bridge 层把 synchronize 设成不可绕过 + debug 断言 |
| R3 | host-side 粗同步损 compute/offload 重叠 | minor | 第一版接受；后续给 pegaflow 加 device-side event-injection |
| R4 | 依赖误配（裸 default-features=false / 漏 cuda-12） | minor | §5.2 依赖行已定，CI 编译验证 |
| R5 | 稀疏 active-set offload 的 token-vs-block 粒度落差 | 已知开放 | 见下，不在本期 |

**稀疏（已知开放问题，不在本期）**：连 dynamo KVBM 都没解 sparse attention offloading——它的复用是 radix 前缀、offload 是 frequency/LRU、tier 是整请求异步流动，对 SWA 只在 router 透传 `kv_cache_spec_sliding_window` 做 window-aware 前缀，对 topk 零处理。没有现成抽象可继承。openinfer 侧 DeepSeek 的 indexer 已产出显式可拦截的 active-set 信号，但 token/row 粒度 ≠ block 粒度，且 compressor 已控 footprint 当前不需 offload。机制层（内容寻址 + 可插拔 policy + 语义无关 transfer）本就不堵稀疏，真正缺的 decode-loop gather 大脑到时候结合具体模型新写更准。

## 8. 下一步：Kimi MLA 最小 spike

**目标**：进程内跑通一个 page 的 register→save→evict→load，证伪"无先例"风险 + 量带宽。

1. 新 bridge crate，path-dep pegaflow-core（§5.2 依赖行），`cargo build` 验依赖。
2. Kimi：`new_with_config` → `register_context_layer_batch`（per-layer ckv/kpe，segments=2，per-layer 布局天然适配）。
3. 一个 page：`synchronize` → `save` →（手动 evict）→ `query` 命中 → `load` 回 GPU → 比对 bytes 一致。
4. 量 host↔HBM 带宽 + save 前 synchronize 的 host stall（确认 R3 可接受）。
5. 通过后再决定给 pegaflow 加 `block_stride` 上 Qwen page-first（R1）。

**阻塞**：等 §2 战略决策最终拍板（pegaflow 取代 kvbm 卸载层 = 是）。

## 9. live 接线设计（Qwen3-4B，**已落地**）

> 状态：已实现并在真实 GPU 上验证（§0）。2026-08 随 qwen3 → `openinfer-kv-store` 迁移重写过一轮：executor 侧 prefetch 状态机（`begin_kv_prefetch`/`drain_ready_prefetch`/`wait_ready_prefetch` + scheduler `loading` 队列、`executor/remote_fetch.rs`）整体删除；下文记录迁移后的现行接线，迁移删项收进末尾「实现注记」。

连接层已就绪（§0），把它接进 `Qwen3Executor` + `scheduler.rs` 的真实推理路径。`Qwen3Executor` 持 `kv_mgr`（`BlockPool`+`KvBuffer`，物理 KV 层已随迁搬入 kv-store 自包含）与 `request_kvs`；在构造（`from_runtime`/`single`，model 移入 RankWorker 之前，此时 `KvBuffer` + `device_ctx().stream` 都在手）经 `build_kv_store` 建一个 `Option<Arc<KvStore>>`（`KvStoreBuilder::rank_with_offload` 把 fused buffer 的逐层 arena 注册进 store 的 pegaflow host），opt-in（`Qwen3OffloadOptions`，**不加 env**），默认关，保现有路径不动。scheduler 经 `executor.kv_store()` 拿到同一个 store 驱动读路径。

**SAVE（async，best-effort）**：prefill/decode step 封块后（此刻 compute stream 已随 `run_step` 同步 → 满足 §0 的跨 stream ordering 约束），`save_sealed_blocks` 调 `store.seal(KV_STORE_RANK, rkv, cursor, SaveClass::Cacheable)`，按 per-request `SaveCursor`（初值 = `prefix_matched_blocks()`，GPU-hit 前缀已 resident，跳过）只存新封的 `(page_id, hash)`，fire-and-forget 推进 cursor。请求结束的 `drop_request` 是 final seal + release：先 drain 该请求尚未 flush 的 store 注册事件，再 `store.retire(...)`——`Cacheable` 照常 fire-and-forget（guard 钉住源页，见实现注记）；`--kv-p2p-flush-on-finish`（P/D prefill 角色）下为 `SaveClass::Handoff`，KV 停放到 save settle，且该 step 的 `Finished` 事件经 `store.flush_saves_then` 屏障（save 对 tier 与 MetaServer 可见后）才放行——P 的 HTTP 响应即 KV-ready 信号。

**LOAD（scheduler 侧 resolve intake，`ResolveHub`）**：不再有 executor 内的 prefetch 状态机；编排整体在 scheduler 线程 + store runtime 之间：

1. submit 入口对每个非 echo 新请求，spawn `store.resolve_prefix(rank 0, tokens, CacheScope{lora: adapter 名, salt: None}, ResolvePolicy::default(), &req.token_tx)` 到 store 自己的 runtime。echo 请求不 resolve（其 prefill 跳过 `match_and_add_prefix`，恢复的块用不上）。plain `submit_rx` 与 LoRA-control 的 legacy command channel 统一在此——legacy channel 丢不了 `KvPrefix`，这正是 resolve 做在 scheduler 侧的原因。
2. `resolve_prefix` 一条读链路：radix probe 持住 GPU 命中块 → host tier query（`Loading` 与 full-hit 策略下的 `Miss` 在 deadline 内重查，默认冷 miss 即刻收束）→ `Hit` 后 `reserve_loaded_blocks`（池装不下则释放租约、sleep 重查，deadline 兜底）→ load H2D（不可取消区段：spawn 的 task 持有 load future + reservation，超时只弃等待不弃块）→ `commit_loaded_blocks` 恢复即入 radix。终态唯一 `KvPrefix { hit_tokens, hold }`；取消（`TokenSink` 实现 `CancelProbe`，客户端断开即安全取消）返回 `KvPrefix::none`，死在 admission 现有的 `is_closed` 检查处。
3. settled 请求带 `KvPrefix` 经 hub 的 mpsc 折回主循环进 `deferred` 再 admit。hold 是 RAII：钉住已恢复块防逐出，在 chunk1 的 `match_and_add_prefix` 重 pin 进请求自己的状态后（第一个 `PrefillRequestResult`）drop；请求被拒（context/KV budget/未知 LoRA）或断开时随 `PendingRequest` 析构释放。
4. admission 预算同一本账：available = `available_blocks() − store.pinned_blocks()`（in-flight save 未落地的页不进预算），per-request `fresh_needed = footprint − resolved_prefix_blocks`（`hit_tokens / block_size`；hold 钉住的页已离池，不重复计）。

**为何 register→rematch 而非直接注入 sequence**：恢复即入 radix，现成的 `match_and_add_prefix`（GPU+host 在它眼里就是一段连续前缀）自然命中；hold 跨 match 钉住，eviction 窗口为零。最坏（真被 evict）只是少命中、退化为多 prefill，不损正确性。

**scheduler 侧队列**：`ResolveHub` 持 store 句柄 + 回流 channel + in-flight 计数（计入等待请求数）；主循环每轮把 settled resolve `drain` 回 `deferred`，空转时 `park` 在无 in-flight 时是普通 blocking recv、有 in-flight 时按 5ms（`RESOLVE_POLL_INTERVAL`）节奏兼顾 submit 与 resolve 回包两个 channel——scheduler 仍是纯 OS 线程，等的对象从"DMA 完成"变成"resolve 回包"。LoRA-control 变体里排在 pending control 之后的生成请求，等 control 落定后才进 resolve（probe 不抢跑它依赖的 adapter load）。

风险：resolve 恢复的块离池即占预算（hold 存续期 available 变小由 `resolved_prefix_blocks` 抵扣对冲，防双计停摆）；`KvPrefix` 的 RAII 覆盖拒收/断开/关停各出口。

**实现注记（三处 toxic-review 加固的收编 + 迁移删项）**：

- **prefetch 状态机整体删除**：`Qwen3Executor.prefetch: HashMap<RequestId, PrefetchState>`（`PrefixProbe` 持 GPU 命中块、`LoadReservation` DMA 落点、`LoadHandle`）连同 `executor/remote_fetch.rs`、`REMOTE_FETCH_DEADLINE`/`REMOTE_REQUERY_INTERVAL` 全部移除；等价职责收编——GPU 命中块由 store 内 `PrefixProbe` 持住，DMA 落点由 `reserve_loaded_blocks` 的 reservation 撑过 H2D，终态只有 `KvPrefix`。miss-breaker 随之删除：resolve 在 store runtime 上异步跑、无 park 语义，breaker 失去作用对象；若冷 miss 压力再现，以 store policy（`ResolvePolicy`）重议。
- **lease 泄漏修复（收编进 kv-store）**：`resolve_prefix` 在 `reserve_loaded_blocks` 失败 / 取消路径上显式 `tier.release`（`PegaflowTier::release` → `release_query_lease`），不当 TTL-park（`QueryLeaseId` 是 `Copy` 裸 token、无 Drop，丢掉只会挂到 600s TTL）。
- **拒绝清理**：不再需要专门回调——`KvPrefix` hold 随 `PendingRequest` 析构释放，commit 的块自然归还池（admission 的 context/KV budget 拒绝与未知 LoRA 拒绝同此）。
- **SAVE 防 slot 复用腐蚀**：`KvStore::seal` 把被存 block 的 `KvBlockGuard`（与 `block_ids` 1:1）随 save task 持到 pegaflow D2H 落地才 `drop(guards)`（`PegaflowTier::save`）——短请求结束 → slot 回收重分配 → 在途 D2H 抓到新 KV 写进旧 hash 的静默腐蚀窗口依旧封死；guard 在 store runtime 上并发 drop 安全（`BlockPool` 内部同步）。钉住页计入 `pinned_blocks` 供 admission 扣除；`flush` 屏障先 drain 各 save task 再 `flush_saves`（带 P2P 时 `flush_saves_and_registrations`，含 MetaServer 注册），故 evict 前先 flush 仍能把 block 排空，`release_finished_events` 的 `Finished` 延迟释放即 `flush_saves_then`。
- **测试**：`tests/kv_offload_cpu_hit.rs` 合一个顺序 `#[test]`（避免两 executor 撞同一 device + pegaflow instance_id），两幕——纯 host tier 恢复（act1：`execute_prefill` + `drop_request` 自动 seal → `flush_offload_saves` → `evict_cached_blocks`；act2：手动 `store.resolve_prefix(…, &NeverCancelled)` 拿 `KvPrefix`（`hit_tokens ≥` 期望下界），hold 存续过 `execute_prefill` 的 match 后 drop）与 GPU+host 组合 hit（3+3 块接成一段连续前缀）；两幕都对冷/暖 first-token logits（argmax regret ≤0.20 / head mean ≤0.06 nat，bf16 floor）。
