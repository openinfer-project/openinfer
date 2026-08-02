# KV cache 统一设计：openinfer-kv-store

**TL;DR**: 异构 KV（full attn / MLA / SWA / linear state）统一为「组 + checkpoint」模型；`openinfer-kv-store` 收编 qwen3/glm52 各自手写的 offload 编排（resolve/seal/retire 三动词 + 单一 KvPrefix 终态），模型侧只声明 `KvModel`（spec/arenas/repack）。骨架已落地，qwen3 首迁验证接口，glm52 验证 P/D，qwen35 验证 bounded 组。

Last touched: 2026-08

## 北极星

新模型 release 当天，给 agent 一页 KV 契约文档（`KvModel` 的三件套怎么填），它能独立完成该模型的 prefix cache + offloading + P/D 接入。分配、索引、checkpoint、租约、异步管线、可观测性全部是 core 机制，模型作者一行不碰。

先例文档：logical/physical 分层见 `../runtime/kv-cache-design.md`（仍然准确，`BlockPool` 就是它的产物）；pegaflow 集成见 `../runtime/pegaflow-offload-integration.md`。本文在其上定义统一的编排层。

## 语汇与不变量

### 两类存储语义，按「封存（seal）」二分

| | 增长 | 封存方式 | 共享 | 代表 |
|---|---|---|---|---|
| **Paged append-only** | 随 token 线性增长 | 页写满原地即封，封后永不改写（免费） | `ImmutableBlock` Arc 共享、radix 前缀复用 | full attn、MLA latent |
| **Bounded mutable** | 固定预算，每 token 原地改写 | **seal-by-copy**：D2D 拷到 staging，副本即 sealed artifact（付一次拷贝） | 活状态永不可共享，只能拷 | linear state（KDA/GDN）、SWA ring、conv_state、DSpark aux |

SWA 与 linear state 是同一等价类（bounded mutable）：qwen35 的 `LayerRecurrentState` 里 `conv_state` 就是一个 W=k-1 的迷你滑窗，与稠密 state 同槽同生命周期。Gemma4 的 SWA 只是更大的同类。

关键推论：**pin（防 reuse-after-free）对 paged 组足够，对 mutable 组不够**——异步 save 会与下一步的原地改写竞态，所以 mutable 组必须 seal-by-copy 后才可下存。

seal-by-copy 的**频率归策略、对齐归机制**，别混淆：拷贝必须对齐在 step 边界（kernel 原地写 state 时不能快照），但**只在策略触发存储时才拷**，不是每步都拷。bounded 组的封存有拷贝成本，策略上天然比 paged 组稀（每数百 token / turn 边界 / retire / P 侧 handoff）——这与单索引"以最稀疏 checkpointer 为准"自洽：索引条目只存在于 bounded 快照点，paged 组封得更密对命中无贡献。成本量级：qwen35 state 49 MiB/请求，D2D 十几微秒；可走 copy-engine 侧流，用 event 保证"拷完才允许该 slot 下一步改写"，不停 step。

### Checkpoint

> 在 token 边界 t，物化「该请求所有组的 sealed artifacts 集合」，以前缀 hash 索引。

- prefix cache 命中 = checkpoint 恢复；offload = checkpoint 换存储层级；**P/D handoff = P 在请求末尾强制打 checkpoint、D 从它恢复**（store-based P/D，非 transfer-based）。
- **单索引，以最稀疏的 checkpointer 为准**（混合模型即 linear 边界）。显式 trade-off：放弃 full 层独立细粒度命中——恢复必须从 linear 边界重算，forward 一跑所有层 KV 重生成，超出边界的 full 页保存是纯浪费。不要「优化」回去。
- 何时打 checkpoint 是**策略**（每 N 页 / turn 边界前端 hint / P 角色请求末尾）；机制只提供 `seal(at)`。

### 对齐组（aligned groups）

同生命周期、同边界的组共享同一套 block id，各自解释 id→offset，注册 offload 时绑定搬运（分开恢复即静默腐坏）。现役先例：glm52 的 mla(656B/tok) + idxk(132B/tok) 共享 slot_mapping；native MTP L78 KV 1:1 镜像主池 id，radix 命中免费复用。**同生命周期的 paged 组不需要 GCD page size**；只有增长语义不同的组（bounded 类）才有独立分配器（slab）。

### Checkpointable 属性

组内容必须是 token 前缀的确定函数、hit-resume 时可恢复或跳过，才能参与 checkpoint。反例：DSpark/DFlash 的 aux-hidden 捕获（命中时目标模型不重算 hidden，无从恢复）→ `checkpointable: false`，存在该组时强制关 prefix cache（今天的隐式全局开关变成描述符上的显式属性）。

## 架构

### 模型侧：`KvModel` trait（声明为主，随 qwen3 首迁定稿）

```rust
pub trait KvModel {
    fn spec(&self) -> KvSpec;                          // 组结构，build 后不变
    fn arenas(&self, rank: usize) -> Vec<KvArena>;     // 物理登记（glm52 kv_arenas() 的推广）
    fn repack(&self, group, dir, io) -> Result<()> { io.memcpy() }  // 仅 store schema ≠ 执行 layout 时覆写
}

pub struct GroupSpec {
    name: &'static str,
    kind: GroupKind,               // Paged { bytes_per_token } | Bounded { bytes_per_slot }
    sharding: Sharding,            // Replicated | HeadSharded { heads } —— Replicated 推导「rank0 存、恢复广播」（MLA TP 去重、qwen35 linear state 同一属性解决）
    checkpointable: bool,
    aligned_with: Option<GroupId>,
    optional: bool,                // P/D 两侧配置不对称时可缺省（MTP）
}
```

三条模型线的落点：qwen3 = 1 个 Paged 组（head-sharded，page 16）；glm52 = mla+idxk 对齐组 + MTP 对齐组（optional）+ DSpark non-checkpointable slab（全 Replicated，page 64），repack 仅 TP4 生产侧 576B→656B wire（今天 MTP `transfer_cache` 手写的就是它）；qwen35 = Paged 组（8 full 层）+ Bounded 组（49.125 MiB/slot，Replicated）。

### Store 侧：`KvStore` 具体类型（不是 trait）

进程内一个，`Arc` 共享；持有 pegaflow client + I/O runtime + checkpoint 索引 + 每 rank 的 `{ Arc<BlockPool>, host tier }`。五个方法：

```rust
impl KvStore {
    fn register_rank(&self, rank, pool: Arc<BlockPool>, tier: Option<Arc<dyn HostTier>>);
    async fn resolve_prefix(&self, rank, tokens, scope, cancel: &dyn CancelProbe) -> KvPrefix;
    fn seal(&self, rank, kv: &RequestKv, cursor: &mut SaveCursor, class: SaveClass);
    fn retire(&self, rank, kv: RequestKv, cursor: SaveCursor, class: SaveClass);
    fn set_admission_floor(&self, rank, blocks) / fn pinned_blocks(&self, rank) -> usize;
}
```

- **`resolve_prefix` 是整条读链路**：probe radix → host tier query（重查询间隔/deadline 内置）→ 水位门控的 `reserve_loaded_blocks` → H2D → `commit_loaded_blocks`（恢复即入 radix，后续 `match_and_add_prefix` 自然命中——qwen3 已验证的模式）。**终态恰好一种**：`KvPrefix { hit_tokens, hold }`。降级（超时/熔断/池压力）只进 stats，不进类型——不改变调用方控制流的信息不配进返回类型。P/D 侧 `hit_tokens < committed_len` 这个数字即全部信息。
- **`seal`/`retire` 是写链路**：`SaveClass::Cacheable` fire-and-forget 可 shed（丢的只是未来命中）；`SaveClass::Handoff` 必达带租约（P 存的 entry 在 D 取走前不可逐出——eviction 在 store-based P/D 下是正确性问题，8 GiB 打穿 + 15s deadline 事故的教训）。retire 收编三处手写变奏（qwen3 flush barrier、glm52 `save_sealed_on_release` 先存后放、`detach_tail_save` 停放）：seal 尾部 → 有未决 save 则整个 KV 随 save 停放，settle 后 RAII 归还。
- **背压恒等式**：飞行中 save 钉住的页从 admission 预算里扣（`pinned_blocks`，glm52 现状收编）；Cacheable 超预算即 shed（`cacheable_pin_percent`，默认池的 25%，cursor 不前进、压力解除后重试），Handoff 只 backpressure admission，**永不 block decode**。Handoff save 失败：计数 + 大声报错、块照常归还——对端以 hit 短缺观察到缺失并拒绝 handoff；P 侧"扣住 KV-ready 响应直到 save 确认"（glm52 flush-on-finish 语义）的接线随 P/D 迁移落地。

### 请求管线：线性所有权链

```
bridge(收 EngineCoreRequest) → tokio task: store.resolve_prefix(...).await
                             → submit_tx.send((GenerateRequest, KvPrefix))   ← rank 收件箱
                             → scheduler admission: match_and_add_prefix、drop hold
```

- store 不认识 engine/收件箱/`GenerateRequest`——它的词汇只有 token 前缀进、`KvPrefix` 出；路由留在 `EngineHandle`（已有职责）。
- **消灭队头停车**：今天 glm52 的 `HostRestoreState::poll_front` / `NativePdState` Park 是 push_front + break，一个请求等 restore 堵死整 rank FIFO。resolve 前置后 scheduler 收件箱里只有就绪请求。
- **取消 = 已有的共享原子**（`TokenSink::is_closed`，`abort_reason: Arc<AtomicU8>`），不新增机制。resolve task 在 op 之间观察；**已提交的 DMA 是不可取消区段**（guard 陪 op 走到 settle，即 `handle.rs` 现有的 detach 语义）。取消的请求返回空 KvPrefix，死在 admission 现有的 `is_closed` 检查处。
- **池的并发事实**：kvbm `BlockManager` 内部同步（save guard 已跨线程 drop），跨线程分配不是安全问题而是仲裁问题——`set_admission_floor` 水位由 scheduler 维护，resolve 的分配让位于它，不足即降级（fail-soft）。

### Scheduler 的接触面（全同步，共四处）

收件箱 recv `(req, prefix)` / admission 读 `pinned_blocks` + 更新水位 / 边界处 `seal` / 退休时 `retire`。`deferred_releases` 这类 graph-replay 生命周期是 step 语义，**不进** KvStore。

## 可观测性纪律

所有异步任务带 request id + 阶段，**终态（成功/降级/失败/shed）必须汇报 sink，禁止 silent drop**。logging 是打印 sink、metrics 是聚合 sink、tracing 是阶段换 span——皮后补，骨架现在定型。「ttft > 50s 是哪段慢」= resolve 各阶段终态有记录。

## 迁移计划（绞杀式，每步合 main、bench 全绿、删旧码）

1. ✅ 本骨架 PR：文档 + `openinfer-kv-store`（resolve/seal/retire，paged only，CPU 契约测试）+ 分发层 `(GenerateRequest, KvPrefix)`（未迁移模型收 `KvPrefix::none()`，行为零变化）。
2. qwen3 首迁：`resolve_prefix` 替换 `remote_fetch_action` 状态机（executor.rs ~2000 行 prefetch 编排），旧路径留开关至 bench 对齐。
3. glm52 D 侧：收件箱替换 `HostRestoreState`；随后 P/D `Handoff` 租约（对齐 pegaflow `QueryLeaseId` 语义后定稿）。注意与 #812 perf 线的 offload.rs 冲突，约好 rebase 节奏。
4. qwen35：`core::kv_pool` → `BlockPool` 迁移（独立，可并行开工）；然后 Bounded 组 slab + seal-by-copy 首发（唯一的新物理能力，由从零接入的模型验证）。gemma4 照抄。

## 未决

- `Handoff` 租约的超时回收与释放时机（D fetch 完成即释放 vs admission 断言通过才释放）——防腐层里对 pegaflow lease 语义定。
- miss-breaker（连续冷 miss 停止 park）是跨请求策略，qwen3 迁移时决定放 store 还是 scheduler。
- 水位下推进池锁内（彻底闭合 TOCTOU）是 hardening 项，骨架先用 store 侧原子。
- pegaflow fork 与否：防腐层（`HostTier`）稳定运行后再议；`QueryOutcome::Loading` 的轮询套轮询是第一改造对象。

**Next**: qwen3 首迁 PR（迁移计划第 2 步）。
