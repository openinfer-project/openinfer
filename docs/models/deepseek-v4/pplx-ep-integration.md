# pplx-garden EP 后端接入 dsv4-flash

**创建时间**：2026-05-15
**状态**：active（functional baseline 已落地，correctness race 已收敛，进入 perf 结构调整）
**当前 blocker**：

- legacy four-stage PPLX 已靠 CPU placement 拉到接近 NCCL。
- 旧 topology-group 选核策略会让 rank/a2a/TE/UVM 角色跨 rank 撞 CPU。
- 当前代码改为 common NUMA topology helper + per-NUMA contiguous rank slice；H200 复测确认无 CPU 冲突，TPOT 保持 **66ms** 级。

- **Functional baseline**：H200/8-rank decode pplx 路径已经过 `LOC_PROT_ERR` / `CUDA_ERROR_ILLEGAL_ADDRESS` 两轮修复；`recv_tokens_per_expert -> expert_indptr` 已移到 GPU，旧 functional baseline steady TPOT **6.9 s/tok** 降到 **123.91 ms/tok**（`output_len=4` 短测）。相对 NCCL baseline **63.77 ms/tok** 仍慢约 **1.9x**；每层只 D2H 一个 padded-total 标量的 dynamic-rows 试验反而退到 **1544.45 ms/tok**。
- **非 MoE 排查结论**：2026-05-16 full NVTX probe 表明 sampling 不是问题，`logits_dtoh` 的 **62-64 ms** 是最后同步点而不是拷贝本体；ratio128 compressed decode 的 per-token allocator 已 scratch 化清零但 TPOT 未下降。ratio4 first refactor 误打到 single-token helper；清理 single-token fused helper 后，batch fused topk profile 把 `attention_ratio4` p95 从 **15.746 ms** 降到 **1.593 ms**，但 request 侧 `wait_rank_decode` p50 仍是 **143.788 ms**。
- **旧 144ms floor 画像**：CPU-pool separation 修掉 rank worker 与 pplx a2a worker 的 exact CPU overlap 后，H200 `output_len=64` steady tail 从 p95 **216.01 ms** / max **300.01 ms** 降到 p95 **159.96-162.86 ms** / max **164.00-168.01 ms**，p50 仍是 **144.00 ms**。NVTX-only differential profile 显示 PPLX rank0 p50 **143.77 ms**、非 rank0 p50 **74.24-79.96 ms**、`rank.logits_dtoh` p50 **63.79 ms**；NCCL rank0 p50 **63.13 ms**、非 rank0 p50 **36.65 ms**、`rank.logits_dtoh` p50 **26.36 ms**。
- **PPLX worker/protocol 证据**：worker-wait NVTX profile 把每层 `p2p_all_to_all` p50 拆到 **1.609 ms**，乘 43 层解释非 rank0 的 74ms 级；其中 `worker_wait_combine_recv_done` p50 **1.111 ms/layer**，`dispatch` p50 只有 **0.010 ms/layer**。per-token source sync、worker-derived active-source mask、early `tx_ready`、route processing overlap 等局部实验均失败或 wait 迁移。
- **direct routed 只作机制证据**：single-node peer-memory direct routed path 绕过 legacy PPLX 四阶段，H200 `output_len=64` p50 从 **144.00 ms** 降到 **83.94 ms**，rows512 后到 **78.68 / 77.33 ms**；clean profile 为 PPLX **79.08 ms** vs NCCL **63.17 ms**。该路径是绕过 upstream 四阶段语义的 hack，当前代码已回到 legacy four-stage，并清理 `a2a_direct_*` API/kernel、direct worker mode、debug counters 和高侵入 RDMA/fabric probes。
- **当前关键修复**：2026-05-17 `/proc` 采样坐实 CPU0 fabric worker 抢占：旧 CPU0 `tx_engine_domain` 在 **7.0s** decode 窗口只拿到 **3.60s** CPU 且 **2980** 次 nonvoluntary switch。把 rank0 TE worker 从 CPU0 挪到同 topology group 的 CPU10 后，两次 H200 `output_len=64` 复测降到 steady p50 **66.46 / 66.70 ms**、p95 **69.80 / 69.62 ms**，接近 NCCL **63 ms** 级。
- **当前代码状态**：legacy 四阶段 kernel 已恢复 cooperative multi-block launch；保留 done-flag 最后发布 correctness 修复。dsv4 侧临时 NVTX ranges 已清理，pplx-garden 自带 NVTX 保留；CPU placement 迁到 `pegainfer_core::cpu_topology`：读取 CUDA device NUMA、当前 affinity mask、NUMA cpulist，把同一 NUMA 下的 rank 先均分连续 CPU slice，再从 slice 内分配 rank/a2a/TE/UVM。CPU0 保留不用，CPU1 给 scheduler；启动时每个 rank 打一行 `cpu_slice/rank_worker/TE/a2a/UVM`。direct routed hack、临时 profiler API capture 和高侵入诊断均已清理。
- **验证状态**：本地 `cargo test --release -p pegainfer-core cpu_topology -- --nocapture`、`cargo fmt --check -p pegainfer-core -p pegainfer-comm-fabric-lib -p pegainfer-deepseek-v4`、`PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep-bench --bin deepseek_pplx_a2a_bench`、`PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` 均通过。H200 release build 通过；`output_len=2` smoke status 0；`output_len=64` 生成 64/64 token 后 teardown status 139，metrics 已打印：steady p50 **66.65 ms**、p95 **68.15 ms**、max **69.47 ms**。PPLX exact ground truth 19/20；case 13 输出 `2500` 而非 `2500 meters`，NCCL 同 case 同样失败，因此不归因到 PPLX placement/通信改动。
- **Next**：在当前 per-NUMA slice placement 上做低侵入 profile，复核 legacy PPLX 相对 NCCL 剩余 **~3-4 ms** gap 和 rank0 drain 结构。

## TL;DR

**2026-05-17 cleanup note**：direct routed single-node path 的 77-84ms 数据只作为机制证据保留；实现已回到 legacy four-stage PPLX。最新整理把 CPU placement 公共化到 `pegainfer_core::cpu_topology`，按 NUMA cpulist 给 rank 连续切片，CPU0 不用、CPU1 给 scheduler，其它 worker 从本 rank slice 内取。bench 的 pplx bootstrap 入口改成隐藏 wrapper，避免暴露 direct 内部 placement 类型；dsv4 临时 NVTX ranges 已清理，pplx 自带 NVTX 不动。本地 release check 和 `cpu_topology` 单测通过；H200 `output_len=64` 复测 steady p50 **66.65 ms**、p95 **68.15 ms**，退出码 **139** 发生在 metrics 之后的已知 teardown 阶段。

把 `pegainfer-comm`（pplx-garden 派生）的 NVLink + RDMA MoE all-to-all 后端从 skeleton 接成可用实现，给 dsv4-flash **decode MoE** 提供另一条通信路径，运行时通过开关切换到它；走 pplx 时 decode CUDA Graph 全局关闭。范围只覆盖 routed expert 这一段 dispatch/combine，prefill、shared expert、attention、indexer 不动。**不**引入 trait/dyn 抽象——只有一个实现，直接用 concrete 类型。

## 工作场景

- 集群：单节点 NVLink + 多节点 RDMA（H20-3e 8 卡内 NVLink、跨节点 RoCE/IB）。
- 模型：DeepSeek V4-Flash，`n_routed_experts=256`、`num_experts_per_tok=6`，EP8 单节点已跑通。
- 目标：在不破坏 NCCL 路径的前提下，把 pplx EP 路径接入到 `moe_rank_lane_bf16_hidden` / `decode_moe_ag_rs_*` 入口，跑通 1×N decode 一致性，作为后续扩到多节点的基础。

## 现状（读码确认过的事实）

### pegainfer-comm 公共表面（skeleton）

- `EpAllToAll` trait：`dispatch / combine / poll / release` 四个 `&self` 方法，对象安全，`Send + Sync`。
- `EpBackendBuilder::build()`：**两种 feature 模式都返回 Err**——`hw-rdma` off 时 `BackendUnavailable`，`hw-rdma` on 时 `Unimplemented`。
- `EpTopology`：只带 `world_size / rank / num_experts / hidden_dim / max_num_tokens` 五个字段，`#[non_exhaustive]`。
- `DispatchPlan / CombinePlan`：极简，目前只承诺 `num_tokens / num_experts_per_token / accumulate`。
- `SendBuf / RecvBuf`：裸 device pointer + elem_count + elem_size + 可选 scale pointer；调用方持有底层 allocation 的所有权。
- `RdmaBackend`（`src/backend/rdma.rs`）：私有类型，四个 trait 方法全是 `todo!()`，构造函数当前只存了 `EpTopology`，没拿 `AllToAllContext`。

### pplx wrapper（`crates/pegainfer-comm-p2p-all-to-all/`）

- `AllToAllContext::new(...)`：21 个参数，需要外部传入 `TransferEngine`、`rank_handles`、预注册的 send/recv buffer + MR、host pointer arrays（sync/send/recv），构造时启动一个 `"p2p_all_to_all Worker"` 后台线程，固定 CPU 亲和性。
- 调用形态是 **四步**（不是 trait 现在写的两步）：
  - `dispatch_send` —— 提交 token 散发，cuda kernel + worker 协同
  - `dispatch_recv` —— 拉回 token，需要 `out_num_tokens_ptr`、tokens_per_expert 等设备元数据
  - `combine_send` —— expert 输出回程
  - `combine_recv` —— combine 完成，按 indices/weights 在 host 端 dtype 转换写回
- 内部依赖：`a2a_kernels::a2a_dispatch_send` 等 CUDA kernel 接 `fabric-lib` 的 IB Verbs / GDRCopy 路径，**有 host 侧 worker loop 参与每次 op 的进度**。

### dsv4-flash 当前 MoE 通信路径

- decode：`pegainfer-deepseek-v4/src/runtime/moe.rs:1323` 的 `decode_moe_ag_rs_bf16_hidden_with_scratch`
  - NCCL `all_gather_bf16_hidden_into`（拼全局 hidden）
  - 本地路由 + grouped FP4 GEMM（local experts）
  - NCCL `reduce_scatter_f32_hidden_into`（聚合到本地 token）
  - 数据流是 **dense AG/RS**，不是 sparse dispatch/combine。
- prefill：`moe.rs:1289` 的 `moe_rank_lane_bf16_hidden`
  - 路由 → expand → grouped GEMM → reduce → `all_reduce_f32_hidden_in_place`
  - 也是 all-reduce 形态，不是 A2A。
- 通信抽象层：`pegainfer-deepseek-v4/src/runtime/collectives.rs` 包了一组 NCCL `Comm`-based helper，所有 MoE 通信都过它。
- 没有任何 dispatch/combine 形态的接口，**需要新增**而不是替换。

### 不做的事

- **不**改 attention / indexer / compressor 的通信路径。
- **不**拆 shared expert（仍走本地 GEMM）。
- **不**接 prefill。prefill 现在的 `moe_rank_lane_bf16_hidden` 是 NCCL all_reduce，本期保持不动；理由是一次 prefill 几百~几千 token，kernel 时间长，通信占比小，先用 decode 验证一致性更划算。
- **不**追求 CUDA Graph 兼容：pplx 路径有后台 worker + host bookkeeping + 同步 flag，跟 graph capture 不兼容。决策：**走 pplx 时全局关闭 decode CUDA Graph 捕获**；想用 graph 就走 NCCL 那条路。
- **不**引入 trait/dyn 切换。只有 NCCL 和 pplx 两条路径，编译 + 配置就够分发，犯不上做 `dyn MoeBackend`。

## 设计

### 分层

```
dsv4-flash MoE (rank lane)
    │
    ├── 走 NCCL AG/RS（已有）—— CUDA Graph 友好
    │
    └── 走 pegainfer-comm（新增）—— eager only，graph 关闭
            │
            └── EpBackend → AllToAllContext → a2a-kernels + fabric-lib
```

- 切换粒度：**整 process 一致**，由 CLI/Config 决定，启动后不变；同一 layer 不会跨后端。

### pegainfer-comm 表面简化

skeleton 里的 `EpAllToAll` trait + `Box<dyn EpAllToAll>` 删掉。`EpBackend` 改成 concrete 结构，inherent 方法直接暴露四步：

```rust
impl EpBackend {
    pub fn dispatch_send(&self, ...) -> Result<...>;
    pub fn dispatch_recv(&self, ...) -> Result<...>;
    pub fn combine_send(&self, ...) -> Result<...>;
    pub fn combine_recv(&self, ...) -> Result<...>;
}
```

四步分开暴露，而不是合成 `dispatch / combine`——pplx 这个拆分本来就是给 host 一个空隙跑 shared expert / 其它计算用的，合起来就等于浪费。

`SendBuf / RecvBuf / DispatchPlan / CombinePlan / EpTopology` 这些 plain data 容器保留，但去掉 `#[non_exhaustive]` 这种 skeleton 期的过度保险——一个实现，没有"演化兼容"问题。

### dsv4 集成入口

新增 `pegainfer-deepseek-v4/src/runtime/moe_pplx.rs`（flat layout，无 `mod.rs`）：

- `decode_moe_pplx_bf16_hidden_with_scratch(ctx, config, weights, ptr_cache, ep, layer, input, token_ids, scratches)`
  - 顺序大致：`dispatch_send` → 同流跑 shared expert → `dispatch_recv` → grouped FP4 GEMM → `combine_send` → 同流跑后续 layer 准备 → `combine_recv` 写回 hidden。
  - shared expert / grouped GEMM 仍复用现有 helper。

`RankWorker` 的 MoE 调用点直接 `if let Some(ep) = &self.ep { ... } else { decode_moe_ag_rs_bf16_hidden_with_scratch(...) }`。没有枚举，没有 trait。

### scratch / buffer 所有权

pplx 路径的 send/recv buffer 必须**预注册到 fabric-lib 的 MR**，不能复用现有 `MoeAgRsScratch` 的 `CudaSlice`：

- 新增 `MoePplxScratch`，持有 send/recv buffer 的 device pointer + `MemoryRegionHandle`，按 `max_num_tokens × hidden_dim` 上限分配一次，生命周期跟 `RankWorker` 一致。
- AG/RS scratch 在 pplx 路径不需要的字段（`global_hidden / global_token_ids / partial_routed / local_routed`）就别分配，省 VRAM。
- pplx 期望 expert-major packed send buffer，dsv4 现有 `expanded_input` 也是 expert-major，但 `x_stride` 在 pplx 是 **token stride**（不是 row stride），接的时候逐项核对。

### 初始化位置

`pegainfer-deepseek-v4/src/direct.rs` 里 `RankWorker::spawn` 阶段，跟 NCCL `Comm` 同级：

```
RankWorker::spawn
    ├── set_current(device)
    ├── NCCL Comm 初始化（已有）
    ├── 当 cfg!(feature="pplx-ep") && config.moe_backend == Pplx:
    │       TransferEngine 初始化
    │       send/recv buffer 分配 + MR 注册
    │       EpBackend::build()（内部启动 worker thread + pin CPU）
    └── 进入 worker main loop
```

控制平面（rank handles 交换）走现有 NCCL bootstrap 路径同样的 rendezvous 方式即可，**不**新引入 process 级别的初始化阶段。

### 运行时切换

- 编译期：`pegainfer-comm` 的 `hw-rdma` feature 已经存在；dsv4 加一个 `pplx-ep` feature，关掉时 `moe_pplx.rs` 整个 `cfg`-out，不拉 fabric-lib 依赖。
- 运行时：`Config` 加 `moe_backend: MoeBackend { Nccl, Pplx }`，CLI `--moe-backend nccl|pplx`，默认 `nccl`。选 `pplx` 时：
  - `pplx-ep` feature 必须开，否则启动报错。
  - decode CUDA Graph 自动关闭（不需要用户单独传参）。

## 分步实施计划

### Step 0：本文档 review ← **当前位置**

确认 scratch/buffer 形态、初始化位置、CLI 入口。

### Step 1：pegainfer-comm 去 skeleton，砍 trait

- 删 `EpAllToAll` trait 与 `Box<dyn EpAllToAll>`，`EpBackend` 改 concrete + inherent 四步方法。
- 改 `EpBackendBuilder::build()`：`hw-rdma` 分支真正构造 `AllToAllContext`。
- `EpBackendBuilder` 补足 `AllToAllContext::new` 需要的参数（`transfer_engine / rank_handles / send_recv_buffers / imm_base / dp_size / node_size / ...`）。
- 写一个 single-node 2-rank 的集成测试，loopback 验证 dispatch_send→recv 与 combine_send→recv 数据正确。

### Step 2：dsv4 加 pplx 路径

- `pegainfer-deepseek-v4/src/runtime/moe_pplx.rs` 写 `decode_moe_pplx_bf16_hidden_with_scratch`，路由 / grouped GEMM / shared expert 复用现有 helper，只把 AG/RS 替换成 dispatch_send→recv + combine_send→recv。
- 新增 `MoePplxScratch`，跟 `MoeAgRsScratch` 同级，按 `cfg(feature="pplx-ep")` 在 `RankWorker` 里二选一持有。
- `RankWorker` MoE 调用点加 `if let Some(ep) = &self.ep { ... } else { 现有 NCCL 路径 }`。
- `Config` / CLI 加 `--moe-backend`，pplx 时强制关 decode CUDA Graph。

### Step 3：1×N decode 一致性

- 同一 prompt 在 NCCL 与 pplx 两条路径下，logits 差异在已接受的数值阈值内（不要求 bit-level，因为 reduce 顺序不同）。
- dsv4 现有 e2e 20 例 golden 对照。

### Step 4：decode 性能 + overlap profiling

- nsys decode trace，看 `dispatch_send` 后 host 空隙是否被 shared expert 填满，`dispatch_recv` 等待是否把 grouped GEMM 推到了通信之后。
- 决定后续是否值得调 `combine_send/recv` 与下一层 attention 准备的 overlap。

### Step 5（后续，本文档外）

- 跨节点 RDMA 真正跑起来。
- prefill 是否替换。
- KV transfer / PD handoff（见 `docs/projects/deepseek-v4/prefix-paged-kv-pd-handoff.md`）。

## 风险 / Open Questions

1. **fabric-lib 编译依赖**：`hw-rdma` / `pplx-ep` 开启时会拉 `libibverbs-sys` / `gdrapi-sys`，CI 默认 lane 仍要 `cargo check` 通过——skeleton 已经做了 default-off 的保护，开起来后要确认 link OK，以及在没 RDMA 硬件的开发机上能跳过运行时探测。
2. **数据布局对齐**：dsv4 现有 `MoeAgRsScratch::expanded_input` 是 BF16 expert-major packed，跟 pplx 的 send buffer 在 stride 约定上要逐项核对（pplx 的 `x_stride` 是 **token stride**，不是 row stride；`x_scale_stride_*` 也分 elem/token 两个 stride，FP8 路径不开就传 0）。
3. **rank handles 怎么交换**：`AllToAllRankHandle::address` 需要在所有 rank 之间互换才能 setup peer 连接。复用现有 NCCL bootstrap 那套机制即可，但具体怎么挂上去要在 Step 1 落地时确定（最简单是 NCCL all-gather string + rank0 集中分发）。
4. **buffer 容量上限**：`max_num_tokens` 决定 send/recv buffer 一次性预留多少 VRAM，需要按现有 decode 的最大 batch 上界算清楚——单 rank decode 当前 bs=1，buffer 不大；但要为 bs>1 的 continuous batching 预留接口（先按当前上界，留 TODO）。

## 不在范围内

- attention / indexer / compressor 的通信
- shared expert 拆分
- CUDA Graph 兼容（走 pplx 时全局关闭）
- prefill 路径替换（仍 NCCL all_reduce）
- KV transfer / PD handoff（见 `docs/projects/deepseek-v4/prefix-paged-kv-pd-handoff.md`）

## 当前进度（2026-05-16）

**已落地**
- `pegainfer-comm` 去 skeleton + 砍 trait：`EpBackend` 是 concrete struct，四步 `dispatch_send / dispatch_recv / combine_send / combine_recv` 是 inherent method，构造走 `EpBackend::new(EpBackendParams)`，外加 `tokens_per_expert_ptr()` 让下游 grouped GEMM 拿 per-expert 计数。`unsafe impl Send` 让 EpBackend 可以从外部线程移交进 RankWorker。
- **砍掉 LibTorch 依赖**：a2a-kernels 自己定义 cxx `ScalarType` enum（namespace 改 `a2a_kernels::`），a2a-kernels 与 p2p-all-to-all 的 `torch-lib` dep 全部移除。`hw-rdma` feature 现在只需要 CUDA + RDMA Verbs + GDRCopy，不再拉 LibTorch / pyo3。
- dsv4 加 `pplx-ep` feature，optional 依赖 `pegainfer-comm/hw-rdma`。
- `runtime/moe_pplx.rs`：`decode_moe_pplx_bf16_hidden_with_scratch` body 完整——本地 route → dispatch_send → shared expert（overlap） → dispatch_recv → host 端 prefix-sum 出 expert_indptr → grouped FP4 expert → combine_send → combine_recv（`accumulate=true`，把 shared expert 折进 routed 输出）。
- `state.rs` 加 `MoePplxScratch`（MR-recv buffer 还是占位，要在 bootstrap 阶段注册）+ `MoeRunContext` / `MoePplxRunContext` 把两条 MoE 路径统一成一个参数。
- `block_decode_rank_lane_bf16_hidden_with_scratch`（含 batch 变体）签名改成 `moe: &mut MoeRunContext<'_>`，内部 `dispatch_decode_moe_step` 按 `moe.pplx.is_some()` 分发到两条路径。
- `RankWorker` 新增 `RankCommand::EnablePplx { ep_backend }`；`DeepSeekV4DirectGenerator::enable_pplx(Vec<EpBackend>)` 把 per-rank 后端塞进对应 worker。
- `cargo check -p pegainfer-comm` 通过。dsv4 因为 pegainfer-kernels 在本机 CUDA/flashinfer SDK 缺失编译不了（pre-existing），结构性 review 看 diff。
- 修通 H200 decode 全链路的几次硬伤：
  - **per-rank TransferEngine**：每张卡绑自己的 CX-7 NIC，`AllToAllRankHandle` 才能带上 peer 自己的 NIC `main_address`。早期共享 TE 时所有 RankHandle 都指向 worker[0]，触发 RDMA `LOC_PROT_ERR`。
  - **`num_dp_groups = world_size / dp_size`**（纯 EP 下 = world_size）：之前硬编码 1 让 `num_routed[N*num_experts]` 越界写。
  - **`num_routed_host`** 改用 `CudaHostMemory::alloc`（`cudaHostAllocPortable | cudaHostAllocMapped`），满足 Verbs MR + UVA。
  - dispatch / combine 的 `x_stride`、`out_x_stride` 全部按 BF16 **byte** stride 传；`combine_recv` 的 `out_tokens_stride` 按 BF16 **element** stride 传（kernel 内 cast 后再算偏移）。
  - `dispatch_recv` 的 `out_num_tokens_ptr` 是 `(num_local_experts,)`，不是单个 scalar；prefix sum 出 `expert_indptr` 时按 pplx 的 padded layout（`ceil(count / expert_padding) * expert_padding`）写。
  - CUMem sync buffer 在本地映射后 `cudaMemset(0)`，避免 sync flag 读到脏初值。
  - **bootstrap 改成 per-rank thread**：`std::thread::scope` 两段——Phase 1 每个 rank 自己 `cudaSetDevice` 一次后做 TE/CUMem/MR；Phase 2 同样每个 rank 自己映射 peer CUMem handle 并 `EpBackend::new`。彻底去掉了之前主线程 N 次循环里 ambient CUDA context 漂移导致的指针注册错卡问题（`dispatch_recv` `CUDA_ERROR_ILLEGAL_ADDRESS` 的最终根因）。
  - `PplxRankResources.peer_mappings` 接管 peer CUMem `CUMemMapping` 的生命周期，不再 `Box::leak`。

**Functional baseline（commit `0abe8fa`）**
- 命令：`PEGAINFER_DSV4_PPLX=1 NCCL_NVLS_ENABLE=0 ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 request --prompt-len 1 --output-len 4 --warmup 0 --iters 1`
- 结果：prefill 521 ms，first decode step 7331 ms，steady TPOT **6900 ms / tok**（0.14 tok/s）。
- NCCL 对照（同机 H200）：steady TPOT **63.77 ms / tok**（15.69 tok/s）。
- 退出时 a2a_context worker shutdown 路径有 segfault，不影响前向；后续清理。

**Correctness cleanup after functional baseline**
- Per-rank TransferEngine / NIC 绑定修复后，Verbs `LOCAL_PROTECTION_ERROR` 消失，说明 MR/lkey/NIC ownership 的大方向已经对齐。
- `dispatch_recv` / `combine_recv` 的 host-visible done flag 改成最后发布：先 reset kernel/worker 共享状态，再 `fence_release_system()`，最后 `st_mmio_b8(*_done, 1)`。这修的是 worker 看到 done 后推进下一步、而上一轮 kernel 尾部又清 flag 的 timing race。
- `dispatch_recv` 额外修了 single-node 常见的 `num_fabric_tokens == 0` completion path：只允许 block 0 发布完成；有 fabric tokens 时只让 `num_local_tokens > 0` 的 block 参与 `grid_counter`。否则空 block 也可能满足 completion 条件。
- H200 短跑命令：`PEGAINFER_DSV4_PPLX=1 NCCL_NVLS_ENABLE=0 RUST_BACKTRACE=1 ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 request --prompt-len 1 --output-len 2 --warmup 0 --iters 1`
- H200 短跑结果：完成，`prefill_ms=534.96`，`first_decode_step_ms=1487.85`，`e2e_ms=2023.10`，`decode_tok_s=0.67`；日志 `/tmp/pplx_after_flag_order.log`。这是 correctness signal，不作为新的 TPOT baseline。

**GPU expert-indptr update**
- `deepseek_pplx_padded_expert_indptr_cuda` 新增为 1-block helper kernel：读取 `dispatch_recv` 写出的 `recv_tokens_per_expert[local_experts]`，按 pplx `expert_padding` 生成 padded `expert_indptr[local_experts + 1]`。
- `moe_pplx.rs` 删除每层 `moe_stream.synchronize()`、D2H `recv_tokens_per_expert`、CPU prefix sum、H2D `expert_indptr`，改为 `dispatch_recv -> device prefix -> event -> grouped GEMM`。
- 第一版为了不让 host 读动态 padded count，grouped FP4 GEMM 的 host `rows` 使用 `expanded_input.seq_capacity()`；真实 expert 范围仍由 device `expert_indptr` 控制，`combine_send` 仍从 pplx worker 的 device `num_recv_tokens` 读真实 token 数。这个版本优先消掉同步闭环，后续可再把 dynamic rows 也留在 GPU 侧。
- Local validation: `rustfmt --edition 2024 --check pegainfer-kernels/src/ffi.rs pegainfer-deepseek-v4/src/runtime/moe_pplx.rs pegainfer-deepseek-v4/src/runtime/state.rs` passed; `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep` passed.
- H200 validation: `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep` passed; `cargo build --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed.
- H200 `output_len=2`: `prefill_ms=519.38`, `first_decode_ms=191.55`, `e2e_ms=711.36`, `decode_tok_s=5.22`; log `/tmp/pplx_gpu_indptr_olen2.log`.
- H200 `output_len=4`: `prefill_ms=488.55`, `first_decode_ms=199.41`, steady TPOT **123.91 ms/tok**, `e2e_ms=936.22`, `decode_tok_s=6.71`; log `/tmp/pplx_gpu_indptr_olen4.log`.
- Negative experiment: after GPU indptr, reading back only `expert_indptr[local_experts]` per MoE layer to pass exact dynamic `rows` into grouped FP4 regressed H200 `output_len=4` to `first_decode_ms=1451.94`, steady TPOT **1544.45 ms/tok**, `e2e_ms=5040.78`; log `/tmp/pplx_dynamic_rows_olen4.log`. The one-scalar host wait is still far more expensive than running grouped FP4 over scratch capacity, so this change was reverted. Dynamic rows only make sense through a GPU-only launch strategy or a custom kernel wrapper.
- Device wait-counter probe: hard-coded counters in the four a2a kernels recorded wait cycles at teardown. H200 `output_len=4` with instrumentation completed with `first_decode_step_ms=199.39`, steady TPOT **137.91 ms/tok**; log `/tmp/pplx_wait_counters_olen4.log`. All ranks reported `fabric_tokens=0`, confirming this single-node workload is not moving remote payload. The largest per-kernel wait was `combine_send tx_ready_avg_cycles`: rank0 ~16M cycles, ranks1-4 ~25-30M, ranks5-7 ~39-40M. `dispatch_send tx_ready` was only ~2M cycles; `dispatch_recv` flag waits were much smaller. The counter overhead is visible vs the clean 123.91 ms/tok run, but the relative signal is strong.
- Negative fast-path experiments:
  - Changing `a2a_combine_send_kernel` to skip the initial `tx_ready` wait when `num_fabric_tokens == 0` hung after benchmark start; the H200 process was killed and the change was reverted.
  - Skipping dispatch/combine barrier transfers in `WorkerState::step` for `world_size == node_size` also hung; the change was reverted.
  - Publishing `tx_ready` early after `dispatch_recv_done` while still keeping the dispatch barrier completed, but did not improve clean-ish TPOT: counter-enabled run gave **125.91 ms/tok** with `combine_send tx_ready_avg_cycles` reduced to ~0.15M, but disabling counters then measured **157.93 ms/tok**; the early publish was reverted. The wait moved into `combine_recv_flag` and did not create stable wall-clock gain.
  These results show `tx_ready`, barrier transfers, and worker combine-stage publication are coupled. A correct single-node fast path needs a deliberate state-machine split, not a local wait bypass.

## TPOT Breakdown

**已排除 / 低优先级**
- H200 GPU↔NIC topology、per-rank NIC、rank handle peer address 已验证；不是硬件拓扑错误。
- CPU pin 当前不是主因：worker main loop 的 tid 和被 pin 的 tid 一致，8 个 a2a worker 落在 distinct CPU。CPU pin 可以影响 host jitter，但解释不了 7s/tok 级别的 GPU kernel residency。
- Payload 小不是优势。decode bs=1 时每层 payload 很小，固定开销（kernel residency、polling、host worker barrier、GDR flag 往返）占比接近 100%；这类后端是为较大 token batch / 跨节点吞吐设计的，放到单 token decode 会暴露固定成本。
- `log::info!` 会改变 timing 并能遮住 race，但它不是 6.9s TPOT 的根因。race 修复后要删掉 per-layer/per-step logs，只保留必要的一次性 bootstrap 记录。

**当前热路径**
1. `ctx.stream` 上路由，`moe_stream` 等 route event。
2. `dispatch_send` 在 `moe_stream` 启动 pplx dispatch send kernel，host a2a worker 同步推进 route / transfer state。
3. shared expert 在 `ctx.stream` 跑，理论上可与 dispatch send overlap。
4. `dispatch_recv` 在 `moe_stream` 启动 recv kernel，写 `expanded_input` 和 `recv_tokens_per_expert`。
5. `deepseek_pplx_padded_expert_indptr_cuda` 在 `moe_stream` 上把 `recv_tokens_per_expert` 转成 padded `expert_indptr`；`ctx.stream` 等 event 后继续本地 grouped GEMM。
6. grouped FP4 experts 在 `ctx.stream` 跑。
7. `combine_send` / `combine_recv` 再走 pplx kernel + worker barrier，把 routed output accumulate 到 shared expert output。

**证据**
- 旧 nsys kernel summary 中，四个 pplx 通信 kernel 每个 MoE 层合计约 **165 ms**，但这个 profile 包含 per-layer host readback 闭环：
  - `a2a_dispatch_send_kernel` ~52 ms
  - `a2a_dispatch_recv_kernel` ~46 ms
  - `a2a_combine_send_kernel` ~54 ms
  - `a2a_combine_recv_kernel` ~12 ms
- 165 ms × 43 层约等于 7.1 s/token，和实测 6.9 s/token 对齐。也就是说，TPOT 不是隐藏在 grouped GEMM 或 attention 里，主要就是 pplx 四段通信 kernel/worker 协议本身的 wait-inclusive 时间。
- GPU expert-indptr update 后，同口径 `output_len=4` steady TPOT 降到 **123.91 ms/tok**，证明 `moe_pplx.rs` 的 host readback/prefix/H2D 闭环是旧 6.9s/token 的主要放大器之一。
- 2026-05-16 H200 `output_len=8` no-event nsys profile (`/tmp/pplx_profile_nonoverlap_scratch_noevent_olen8.sqlite`) 只完整采到 device 3 kernel，但 CUDA API 计数显示整段仍有大量 host-side activity：`cuMemcpyHtoDAsync_v2` 82,288 calls、`cuMemAllocAsync` 84,359 calls、`cuMemFreeAsync` 1,331 calls。这个 profile 不适合判断 a2a GPU kernel 本体，但适合确认 decode path 仍不是 graph-like 的静态执行。
- 2026-05-16 临时 stage NVTX profile（无 scratch 探针，只给 `block_decode_rank_lane_bf16_hidden_with_scratch` 分段；remote report `/tmp/pplx_profile_stage_nvtx_baseline_attention_olen8.nsys-rep`）把 MoE 前抖动定位到前置 operator：
  - `decode_attention_full`: 2,408 ranges，avg **1.054 ms**，max **33.527 ms**，total **2537.0 ms**。
  - `decode_ffn_pre_norm`: 2,408 ranges，avg **0.408 ms**，max **19.418 ms**，total **982.6 ms**。
  - `decode_attn_hc_pre_norm`: avg **0.031 ms**，max **0.204 ms**，基本不是长尾来源。
  - `decode_moe`: avg **0.840 ms**，max **81.764 ms**；max 主要来自 first decode / layer 0 初始化型成本，steady 的多数长 `p2p_all_to_all` range 是在等待下一层 MoE 进入，而不是 MoE kernel 本体持续执行。
  - layer 聚合看，`decode_attention_full` 长尾集中在若干 compressed attention 层，例如 layer 2 max **33.527 ms**、layer 32 max **29.190 ms**、layer 40 max **18.822 ms**、layer 36 max **18.840 ms**；`decode_ffn_pre_norm` 长尾集中在 layer 28/40/38/32 等，max 在 **17-19 ms** 区间。
  - 结论：当前“MoE 抖”更像 MoE 前置算子/launch/rank 到达方差被 pplx worker range 放大显示，而不是 pplx 四段 kernel 平均时间单独决定。
- 2026-05-16 临时 HC mix bypass 实验（未保留代码）把 `seq_len=1` 且无 raw/rms side output 的 `deepseek_hc_mixes_cuda` 从 BF16->F32 + cuBLAS `Sgemv` + scale kernel 改成已有 `deepseek_hc_mixes_kernel`。H200 `output_len=8` 短测 steady TPOT **149.96 ms/tok**（p50 **127.93 ms**，max **232.07 ms**），nsys event profile steady TPOT **133.31 ms/tok**。profile 证实 cuBLAS GEMV 基本消失，`deepseek_hc_mixes_kernel` 66 次 total **1.77 ms**，但整体仍由 NCCL all-reduce、FP4 grouped GEMM 和 CUDA API/launch 长尾主导；因此该改法不作为主线保留。
- 2026-05-16 加入一次性 NVTX probe（只随 `pplx-ep` feature 编译）：request 主线程标出 `dsv4.request.prefill / step / sample / emit_token / advance_decode`；runtime 标出 `dsv4.runtime.dispatch_rank_decode / wait_rank_decode / rank0_logits`；rank worker 标出 `dsv4.rank.decode / token_upload / embedding / embedding_all_reduce / hc_expand / decode_layer / final_logits / gather_logits / logits_all_gather / logits_dtoh`；layer 内标出 `dsv4.layer.attn_hc_pre_norm / attention / attention_full / attention_ratio4 / attention_compressed / ffn_hc_pre_norm / moe / ffn_hc_post`。现有 pplx worker range（`p2p_all_to_all / dispatch / combine / process_routing_info / barrier`）保留。这样同一条 nsys timeline 能直接判断 steady 60ms gap 是 sampling/logits、rank response wait、MoE 前 operator 到达、还是 pplx worker protocol。
- Validation for the probe: local `cargo fmt --check -p pegainfer-deepseek-v4` passed; local `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep` passed; H200 `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep` passed after syncing the instrumented files and restoring local `deepseek_hc.cu` on the remote tree. The remaining warnings are the pre-existing pplx visibility/unused warnings.
- 2026-05-16 ratio128 compressed decode scratch 化：单 token non-overlap compressed attention 改为复用 `AttentionProjectionScratch / AttentionIndexScratch / AttentionAuxScratch / AttentionOutputScratch`，新增 `compressor_nonoverlap_decode_bf16_hidden_at_scratch` 与 `compress_topk_indices_decode_into`，删除旧 owned-return 单 token入口。Local validation: `cargo fmt --check -p pegainfer-deepseek-v4` passed, `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep` passed, `git diff --check` passed. H200 validation: `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep` passed, `cargo build --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed. H200 `output_len=8` smoke completed with steady TPOT **167.95 ms/tok** (`/tmp/pplx_ratio128_scratch_olen8.log`), and node NVTX profile completed with steady TPOT **147.31 ms/tok** (`/tmp/pplx_ratio128_scratch_nvtx_olen8.{log,nsys-rep,sqlite}`). The new sqlite confirms decode-window `cuMemAllocAsync` and `cuMemFreeAsync` are both **0**; previous probe had `cuMemAllocAsync=11200` all attributed to `dsv4.layer.attention_compressed`. TPOT did not materially improve, so allocator spikes were real but not the current wall-clock root cause.
- 2026-05-16 ratio4 topk refactor correction：第一次实现打到 single-token ratio4 helper，但 H200 profile 和源码路径确认 decode 走的是 `attention_decode_compressed_overlap_rank_local_collective_bf16_hidden_batch_with_scratch`，即 `bs=1` 也走 batch helper。随后删除 single-token fused helper、删除 dead batch `indexer_topk_indices_decode_batch_into` wrapper，在 batch path 中把 `window_topk + indexer_topk + concat` 合为 `deepseek_ratio4_decode_topk_indices_batch_kernel`；`max_compressed_len == 0` 仍走 window-only path。Local validation: `cargo fmt --check -p pegainfer-deepseek-v4 -p pegainfer-kernels` passed, `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep` passed, `git diff --check` passed. H200 validation: `cargo build --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed; bench binary fatbin contains `_Z48deepseek_ratio4_decode_topk_indices_batch_kernel...`, and remote source calls `ratio4_decode_topk_indices_batch_into` from `runtime/block.rs`.
- 2026-05-16 ratio4 batch topk profile：H200 `output_len=16` nsys report `/tmp/pplx_ratio4_batch_topk_nvtx_olen16.{log,nsys-rep,sqlite}` completed with benchmark steady TPOT avg **152.85 ms**, p50 **144.03 ms**, p95 **188.01 ms**, max **196.06 ms**. Compared with `/tmp/pplx_ratio4_refactor_nvtx_olen16.sqlite`, NVTX distributions improved in the specific attention range: `dsv4.layer.attention_ratio4` avg **2.178 -> 0.827 ms**, p95 **15.746 -> 1.593 ms**, max **42.748 -> 34.788 ms**; `dsv4.rank.decode` p50 **114.779 -> 79.695 ms**. Request-level `dsv4.runtime.wait_rank_decode` p50 only moved **175.804 -> 143.788 ms** and still dominates, so this change removes real ratio4 launch fanout but does not complete the TPOT target by itself. Nsight kernel table captured only one device, so kernel-name absence/presence in the sqlite is not a sufficient proof source; source path + fatbin symbol + NVTX movement are the useful evidence.
- 2026-05-17 decode-only driver-contention profile：temporary `cudaProfilerStart/Stop` was hard-coded around decode and nsys was run with `--capture-range=cudaProfilerApi --sample=process-tree --sampling-period=1000000 --cpuctxsw=process-tree --cudabacktrace=all:1000 --cuda-flush-interval=100 --osrt-threshold=1000 --stats=true`; remote report `/tmp/pplx_driver_contention_olen8.{log,nsys-rep,sqlite}`. That profiler API patch was removed during cleanup; this entry is a historical capture record, not the current reusable profiling command. The capture fixed the previous truncated-kernel profile: every device has **17787** kernels and D2H device memcpy time is only **93 us** total across 7 copies. CUDA API summary shows long host-side tails instead:
  - `cudaLaunchCooperativeKernel`: 9632 calls, total **1287.95 ms**, max **168.34 ms**; stack is `cudaLaunchCooperativeKernel -> a2a_dispatch_send -> EpBackend::dispatch_send -> decode_moe_pplx...`.
  - rank-thread OSRT: `pthread_mutex_lock` total **782.16 ms**, max **108.95 ms**; `futex` total **2586.79 ms**, max **104.19 ms**. The long mutex stacks are in `libcuda -> cuModuleGetFunction/cuLibraryGetModule -> cudaLaunchCooperativeKernel -> a2a_dispatch_send`.
  - Non-communication long launches are also real on the host side but not on the device side: `deepseek_hadamard_rotate_bf16_serial_kernel` API launch **18.1-30.8 ms** while GPU kernel is about **0.030 ms**; `deepseek_compressor_norm_serial_kernel` API **16.16 ms** while GPU kernel is **0.008 ms**; `deepseek_hc_scale_mixes_block_kernel` API **16.06 ms** while GPU kernel is **0.0035 ms**; `deepseek_indexer_scores_decode_serial_kernel` API **15.89 ms** while GPU kernel is **0.016 ms**.
  - Those non-communication API tails sit inside `dsv4.layer.attention_ratio4 step=1/3 layer=2` and `dsv4.layer.attn_hc_pre_norm step=3 layer=2`; e.g. `attention_ratio4` 21-37 ms ranges are mostly launch/driver residency, not GPU arithmetic.
  - `cuKernelSetAttribute` has three **20.3-20.7 ms** tails inside `attention_ratio4 step=3 layer=2`, matching the driver-state hypothesis. Since the captured callchains for these rows are absent in Nsight, the current attribution is by NVTX containment and correlation timing rather than Rust stack.
  - CPU sampling marks rank threads as `Running` during samples, so the rank workers are not simply sleeping in application code during the measured decode; the expensive sleeps in OSRT are mostly channel waits outside hot work or libcuda internal mutex/futex behavior during launch.
  This profile changes the interpretation of ratio4/HC spikes: tune kernel arithmetic only after checking API-vs-GPU duration. The immediate optimization axis is launch count / module lookup / attribute setup churn in the decode path, while MoE/NCCL communication remains excluded from the current optimization target.
- 2026-05-17 upstream `pegainfer-comm/benchmarks/bench_all_to_all.py` was run on H200 with dsv4-shaped payloads to get the pplx-side theoretical MoE A2A floor. The script now exposes `--expert-padding` so it can match Rust `PplxBootstrapParams::default().expert_padding = 16`; default upstream padding remains 1. Common command shape:

```bash
cd /root/develop/xingming/pegainfer-pplx-ep/pegainfer-comm
NCCL_NVLS_ENABLE=0 PYTHONPATH=. ../.venv/bin/python benchmarks/bench_all_to_all.py \
  --world-size 8 --dp-size 1 --nets-per-gpu 1 \
  --max-private-tokens 64 \
  --num-experts 256 --hidden-dim 4096 --hidden-dim-scale 0 \
  --num-experts-per-token 6 \
  --in-dtype bfloat16 --out-dtype bfloat16 --scale-dtype float32 \
  --expert-padding 16 --nvlink 8 \
  --num-warmup 20 --num-repeats 100 --no-check
```

The two payload points:

| Payload | Output | Dispatch both p50 | Combine both p50 | Back-to-back dispatch+combine p50 | Split send/recv p50 sum |
| --- | --- | ---: | ---: | ---: | ---: |
| Real bs=1 decode: `max_num_tokens=1`, total EP routes = `8 * 1 * 6 = 48` | `/tmp/dsv4_pplx_a2a_max1_pad16.{log,json}` | **63.87 us** | **21.54 us** | **85.41 us/layer** | **52.64 us/layer** |
| Rust bootstrap capacity: `max_num_tokens=8`, total EP routes = `8 * 8 * 6 = 384` | `/tmp/dsv4_pplx_a2a_max8_pad16.{log,json}` | **63.71 us** | **21.79 us** | **85.50 us/layer** | **53.82 us/layer** |

Interpretation: the upstream multi-process pplx benchmark reports the GPU/protocol A2A floor for our BF16/topk6/EP8 payload as roughly **0.085 ms per MoE layer**, or about **3.7 ms/token** across 43 layers if dispatch+combine are serialized. This is orders below the 140-160 ms request TPOT class, so the current gap is not explained by raw payload movement. This benchmark does not include pegainfer's full model runtime, explicit stream handoffs, model-side operator launch fanout, grouped GEMM/attention/NCCL, or request wait-rank effects; use it as a theoretical lower bound, not as an end-to-end replacement profile.
- 2026-05-17 added `deepseek_pplx_a2a_bench`, a Rust-side microbench that reuses the same dsv4 `build_intra_node_backends_for_devices` wrapper and `EpBackend` methods but excludes all model operators. It allocates BF16 hidden/out buffers on each rank, uses synthetic balanced routes to the next 6 ranks (`topk=6`), and reports both flattened rank×iteration stage times and per-iteration max across the 8 ranks. This isolates the single-process Rust wrapper / bootstrap / stream handoff layer between the upstream Python benchmark and the full dsv4 runtime.

Command shape:

```bash
cd /root/develop/xingming/pegainfer-pplx-ep
PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:$PATH \
  cargo build --release -p pegainfer-deepseek-v4 --features pplx-ep-bench \
    --bin deepseek_pplx_a2a_bench
  ./target/release/deepseek_pplx_a2a_bench \
  --model-path /data/models/DeepSeek-V4-Flash-mp8 \
  --world-size 8 --max-private-tokens 64 \
  --expert-padding 16 --nets-per-gpu 1 \
  --warmup 20 --repeats 100 \
  --max-num-tokens <1-or-8>
```

Results:

| Payload | Log | dispatch_send p50 | dispatch_recv p50 | combine_send p50 | combine_recv p50 | Flattened split-sum p50 | Per-step max-rank split-sum p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `max_num_tokens=1` | `/tmp/dsv4_rust_pplx_a2a_max1.log` | **19.33 us** | **14.05 us** | **16.29 us** | **12.77 us** | **65.98 us** | **77.54 us** |
| `max_num_tokens=8` | `/tmp/dsv4_rust_pplx_a2a_max8.log` | **19.36 us** | **13.82 us** | **16.90 us** | **12.61 us** | **65.06 us** | **76.51 us** |
| `max_num_tokens=1`, per-NUMA slice placement | `/tmp/pplx_ep_bench_slice.log` / `profiles/pplx_ep_bench_slice.log` | **19.46 us** | **13.02 us** | **16.48 us** | **12.61 us** | **64.86 us** | **75.84 us** |

Interpretation: the Rust single-process wrapper still keeps pplx A2A at roughly the same floor as upstream Python; even waiting for the slowest rank per iteration is only **~0.077 ms/layer**, or **~3.3 ms/token** over 43 MoE layers. This kills the hypothesis that `EpBackend` integration alone turns pplx A2A into a millisecond-scale layer cost. The remaining 140-160 ms TPOT gap must come from full-runtime composition: model-side operator launch fanout, explicit stream handoffs, rank arrival/wait structure, grouped GEMM/attention/NCCL, or the way those pieces serialize around MoE. Use full-runtime step correlation for the next attribution pass rather than more isolated A2A tuning.

### Per-NUMA Slice Placement Validation

H200 validation after moving CPU topology helpers to `pegainfer_core::cpu_topology`:

- Local build gates passed:
  - `cargo test --release -p pegainfer-core cpu_topology -- --nocapture`
  - `cargo fmt --check -p pegainfer-core -p pegainfer-comm-fabric-lib -p pegainfer-deepseek-v4`
  - `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep-bench --bin deepseek_pplx_a2a_bench`
  - `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving`
- H200 release builds passed for `bench_serving` and `deepseek_pplx_a2a_bench`.
- Startup placement now reserves CPU0 and CPU1, then slices each NUMA node by rank:
  - NUMA0 ranks 0-3 use even CPU slices: rank0 `2..46`, rank1 `48..94`, rank2 `96..142`, rank3 `144..190`.
  - NUMA1 ranks 4-7 use odd CPU slices: rank4 `3..47`, rank5 `49..95`, rank6 `97..143`, rank7 `145..191`.
  - For each rank, `rank_worker/TE/a2a/UVM` are the first four CPUs in that rank's own slice, so the old cross-rank collision is gone.
- H200 `output_len=2` smoke completed with status **0**, first decode **116.67 ms**, log `profiles/pplx_tpot_slice_olen2.log`.
- H200 full TPOT `output_len=64` completed 64/64 tokens and printed metrics before the known teardown segfault:
  - status file **139**
  - `ttft_ms=441.54`
  - `first_decode_step_ms=116.11`
  - `steady_tpot_ms`: avg **66.88**, p50 **66.65**, p95 **68.15**, p99 **69.18**, max **69.47**, samples **62**
  - `request_tok_s=13.60`, `decode_tok_s=14.78`
  - log `profiles/pplx_tpot_slice_olen64_r2.log`; concise extract `profiles/pplx_slice_summary.log`
- H200 PPLX exact ground truth over all 20 cases ran with `max_new_tokens=64` and failed **1/20**:
  - case 13 expected `2500 meters`, got `2500`.
  - NCCL control run on the same case also got `2500`, so this is not a PPLX-specific correctness regression.
  - logs: `profiles/pplx_e2e_ground_truth.log`, `profiles/nccl_e2e_case13.log`, `profiles/pplx_e2e_case13_rerun.log`.
- The first `output_len=64` attempt after EP bench stayed at zero log output with 8 GPUs holding ~22GB and GPU util 0; it was terminated manually and left `profiles/pplx_tpot_slice_olen64.log` empty. The immediate `output_len=2` smoke and second `output_len=64` run succeeded, so treat that attempt as a transient startup hang unless it reproduces.

### Full-Runtime NVTX-Only Differential Profile

The high-overhead CUDA backtrace profiles are useful for API callchains, but not for wall-clock comparison: under `--cudabacktrace=all:1000`, both PPLX and NCCL runs inflate to ~0.8 s/token, and the PPLX CUPTI kernel table captured only devices 1/2/3/7 while NCCL captured all 8 devices. Do not use those kernel totals as a PPLX-vs-NCCL wall attribution.

Low-intrusion NVTX-only profiles are the current wall-clock source:

```bash
nsys profile --force-overwrite=true --trace=nvtx --sample=none --stats=true \
  -o /tmp/<pplx-or-nccl>_nvtx_only_olen64 \
  ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 \
    request --prompt-len 1 --output-len 64 --warmup 0 --iters 1
```

Results:

| Backend | Log/sqlite | Steady TPOT p50 | p95 | rank0 p50 | non-rank0 p50 | `rank.logits_dtoh` p50 | wait-rank p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPLX | `/tmp/pplx_nvtx_only_olen64.{log,sqlite}` | **144.00 ms** | **156.02 ms** | **143.77 ms** | **74.24-79.96 ms** | **63.79 ms** | **143.78 ms** |
| NCCL | `/tmp/nccl_nvtx_only_olen64.{log,sqlite}` | **63.36 ms** | **64.92 ms** | **63.13 ms** | **36.65 ms** | **26.36 ms** | **63.15 ms** |

Rank spread:

| Backend | max-rank p50 | median-rank p50 | spread p50 | spread p95 |
| --- | ---: | ---: | ---: | ---: |
| PPLX | **143.77 ms** | **74.25 ms** | **69.53 ms** | **77.47 ms** |
| NCCL | **63.13 ms** | **36.65 ms** | **26.37 ms** | **26.73 ms** |

Interpretation:

- Rank0 is the steady tail rank in both backends because it performs final logits gather/D2H for sampling.
- `rank.logits_dtoh` is still a synchronization/drain point, not a raw D2H bandwidth cost: it is **63.79 ms** in PPLX and **26.36 ms** in NCCL, tracking the amount of queued work before the final host copy.
- The p50 gap decomposes into two stacked gaps: PPLX non-rank0 rank-lane p50 is about **37 ms** slower than NCCL, and rank0 then drains a correspondingly longer queue before token selection.
- Attention-local ranges are not the p50 owner in this profile: `attention_ratio4` p50 is **0.140 ms** on PPLX vs **0.260 ms** on NCCL, and compressed/full attention medians are also sub-ms. P95/tails can still matter, but they do not explain the stable 144ms floor.
- Isolated A2A microbench remains microsecond-scale, so the problem is not raw pplx payload movement. It is full-runtime composition around PPLX: per-layer `p2p_all_to_all` ranges, worker/rank synchronization, event bookkeeping, and how those pieces serialize with model work.

### PPLX MoE Stage NVTX Profile

Profile file: `/tmp/pplx_moe_stage_nvtx_olen16.sqlite` on `jzh200-11`.

Command shape:

```bash
PEGAINFER_DSV4_PPLX=1 NCCL_NVLS_ENABLE=0 nsys profile \
  --force-overwrite=true --trace=nvtx --sample=none --stats=false \
  -o /tmp/pplx_moe_stage_nvtx_olen16 \
  ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 \
    request --prompt-len 1 --output-len 16 --warmup 0 --iters 1
```

Run result: generated all 16 tokens, first decode **192.04 ms**, steady avg **145.12 ms**, p50 **144.00 ms**, p95 **148.00 ms**, max **163.99 ms**. The process hit the known teardown segfault after metrics, and the sqlite export completed.

Overall stage facts:

| Range | Count | Avg | p50 | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dsv4.layer.moe` | 10320 | **1.050 ms** | **0.096 ms** | **7.828 ms** | **64.996 ms** |
| `dsv4.pplx_moe.route` | 5160 | **0.100 ms** | **0.009 ms** | **0.962 ms** | **1.098 ms** |
| `dsv4.pplx_moe.dispatch_send` | 5160 | **0.104 ms** | **0.005 ms** | **0.009 ms** | **64.314 ms** |
| `dsv4.pplx_moe.shared_expert` | 5160 | **0.545 ms** | **0.015 ms** | **0.093 ms** | **18.695 ms** |
| `dsv4.pplx_moe.dispatch_recv` | 5160 | **0.006 ms** | **0.005 ms** | **0.007 ms** | **0.049 ms** |
| `dsv4.pplx_moe.grouped_fp4` | 5160 | **0.256 ms** | **0.015 ms** | **0.661 ms** | **18.877 ms** |
| `dsv4.moe.grouped_w1_w3` | 5504 | **0.042 ms** | **0.006 ms** | **0.023 ms** | **18.868 ms** |
| `dsv4.moe.grouped_w2` | 5504 | **0.197 ms** | **0.006 ms** | **0.642 ms** | **18.560 ms** |
| `dsv4.pplx_moe.combine_send` | 5160 | **0.005 ms** | **0.004 ms** | **0.006 ms** | **0.053 ms** |
| `dsv4.pplx_moe.combine_recv` | 5160 | **0.005 ms** | **0.004 ms** | **0.006 ms** | **0.071 ms** |
| `p2p_all_to_all` | 5168 | **6.571 ms** | **1.601 ms** | **16.790 ms** | **2082 ms** |
| `worker_wait_combine_recv_done` | 5160 | **1.101 ms** | **1.106 ms** | **1.174 ms** | **1.206 ms** |
| `worker_wait_combine_send_done` | 5160 | **0.004 ms** | **0.003 ms** | **0.005 ms** | **0.010 ms** |

Layer-local hot spots from the same profile:

| Stage | Layer | Avg | p50 | p95 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4.pplx_moe.shared_expert` | 27 | **6.684 ms** | **10.484 ms** | **18.226 ms** | host/API range, not yet proven GPU arithmetic |
| `dsv4.pplx_moe.shared_expert` | 33 | **5.294 ms** | **6.975 ms** | **18.604 ms** | host/API range, not yet proven GPU arithmetic |
| `dsv4.pplx_moe.shared_expert` | 41 | **5.788 ms** | **0.088 ms** | **18.135 ms** | tail-heavy |
| `dsv4.pplx_moe.grouped_fp4` | 19 | **3.947 ms** | **2.987 ms** | **14.568 ms** | mostly `grouped_w2` |
| `dsv4.moe.grouped_w2` | 19 | **3.692 ms** | **2.941 ms** | **14.546 ms** | only grouped stage with multi-ms p50 |
| `dsv4.moe.grouped_w1_w3` | many | sub-ms | **0.006-0.008 ms** | mostly sub-ms | occasional max tail only |

Interpretation:

- The normal dispatch/combine API ranges are microseconds; the latest layer-local p50 owners are not `dispatch_send`, `dispatch_recv`, `combine_send`, or `combine_recv` API calls.
- `shared_expert` and grouped ranges are host/API NVTX ranges, so their long durations may still be libcuda launch contention or explicit stream handoff rather than GPU arithmetic. A CUDA+NVTX profile with timestamp containment is required before writing another optimization patch.
- The stable worker wait remains `worker_wait_combine_recv_done` p50 **1.106 ms/layer** in the legacy path. The direct-worker-mode experiment proved this wait can move to `combine_send_done` when barriers are skipped, so it should be interpreted as local MoE readiness / stream dependency serialized through the worker, not a standalone fabric tax.

### PPLX MoE CUDA+NVTX Capture Follow-Up

Profile file: `/tmp/pplx_moe_stage_cuda_capture_olen16.sqlite` on `jzh200-11`.

Historical command shape while the temporary profiler API patch was present:

```bash
PEGAINFER_DSV4_PPLX=1 NCCL_NVLS_ENABLE=0 nsys profile \
  --force-overwrite=true --trace=cuda,nvtx --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  --cuda-flush-interval=100 --stats=false \
  -o /tmp/pplx_moe_stage_cuda_capture_olen16 \
  ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 \
    request --prompt-len 1 --output-len 16 --warmup 0 --iters 1
```

The temporary profiler API patch has since been removed. For this historical run, the capture range started and ended inside the application. The wrapper returned status **143** because `timeout` killed shutdown after capture ended, but the sqlite export succeeded. Kernel capture is complete enough for per-device attribution: **38,325 kernels per GPU** across all 8 devices.

Decode-capture stage facts:

| Range | Count | Avg | p50 | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dsv4.rank.decode` | 10680 | **5.299 ms** | **0.622 ms** | **14.920 ms** | **328.335 ms** |
| `dsv4.layer.moe` | 10320 | **0.836 ms** | **0.168 ms** | **1.276 ms** | **163.284 ms** |
| `dsv4.pplx_moe.route` | 10320 | **0.065 ms** | **0.014 ms** | **0.069 ms** | **17.315 ms** |
| `dsv4.pplx_moe.shared_expert` | 5160 | **0.278 ms** | **0.032 ms** | **1.032 ms** | **17.297 ms** |
| `dsv4.pplx_moe.grouped_fp4` | 5160 | **0.114 ms** | **0.025 ms** | **0.830 ms** | **16.096 ms** |
| `dsv4.moe.grouped_w2` | 5160 | **0.014 ms** | **0.011 ms** | **0.023 ms** | **10.134 ms** |
| `p2p_all_to_all` | 5160 | **4.059 ms** | **1.644 ms** | **16.703 ms** | **489.739 ms** |
| `worker_wait_combine_recv_done` | 5160 | **1.068 ms** | **1.115 ms** | **1.186 ms** | **16.032 ms** |

Same-thread CUDA API overlap and correlated kernel durations by stage:

| Stage | Range p50 | Range p95 | API p50 | API p95 | Correlated kernel p50 | Correlated kernel p95 | Dominant API totals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `route` | **0.014 ms** | **0.069 ms** | **0.008 ms** | **0.062 ms** | **0.000 ms** | **0.014 ms** | `cudaLaunchKernel` **561.1 ms**, `cuEventRecord` **36.9 ms** |
| `shared_expert` | **0.032 ms** | **1.032 ms** | **0.024 ms** | **1.024 ms** | **0.069 ms** | **0.086 ms** | `cudaLaunchKernel` **1197.1 ms**, `cuMemcpyDtoDAsync` **199.2 ms** |
| `dispatch_send` | **0.007 ms** | **0.013 ms** | **0.005 ms** | **0.009 ms** | **0.014 ms** | **7.623 ms** | `cudaLaunchCooperativeKernel` **1174.1 ms** |
| `dispatch_recv` | **0.006 ms** | **0.015 ms** | **0.004 ms** | **0.012 ms** | **0.020 ms** | **2.627 ms** | `cudaLaunchCooperativeKernel` **30.9 ms** |
| `grouped_fp4` | **0.025 ms** | **0.830 ms** | **0.016 ms** | **0.813 ms** | **0.921 ms** | **1.087 ms** | `cudaLaunchKernel` **535.8 ms** |
| `grouped_w1_w3` | **0.011 ms** | **0.753 ms** | **0.007 ms** | **0.750 ms** | **0.602 ms** | **0.723 ms** | `cudaLaunchKernel` **479.9 ms** |
| `grouped_w2` | **0.011 ms** | **0.023 ms** | **0.007 ms** | **0.019 ms** | **0.318 ms** | **0.384 ms** | `cudaLaunchKernel` **55.9 ms** |
| `combine_send` | **0.006 ms** | **0.013 ms** | **0.004 ms** | **0.010 ms** | **0.010 ms** | **0.011 ms** | `cudaLaunchCooperativeKernel` **37.2 ms** |
| `combine_recv` | **0.006 ms** | **0.012 ms** | **0.004 ms** | **0.009 ms** | **0.163 ms** | **0.995 ms** | `cudaLaunchCooperativeKernel` **36.4 ms** |

Representative containment examples:

- `dsv4.layer.moe step=3 layer=22 ratio=4` on rank7 lasted **13.908 ms**. Its child range was `grouped_fp4` / `grouped_w1_w3` for **12.827 ms**, but same-thread CUDA attribution shows this was mostly one `cudaLaunchKernel` call lasting **12.806 ms**; the correlated `deepseek_tilelang_act_quant_k4096_kernel` itself was only **0.0166 ms** and did not start until later on the stream. During the same wall window, other devices were running `a2a_combine_send_kernel` for **12.6-13.6 ms**, and device0 was running `a2a_combine_recv_kernel` for **10.166 ms**.
- `dsv4.layer.moe step=7 layer=30 ratio=4` lasted **9.450 ms**. The contained CUDA kernels across devices were `a2a_dispatch_recv_kernel` at about **8.26 ms** on every GPU, then `a2a_combine_recv_kernel` at about **0.94 ms** on the tail devices.
- `dsv4.layer.moe step=2 layer=27 ratio=128` lasted **9.035 ms**. The largest same-thread API was `cudaLaunchKernel` **7.531 ms**, while the overlapping device kernels were again PPLX a2a kernels: `a2a_dispatch_recv_kernel` **7.8 ms** on several devices, plus `a2a_combine_send_kernel` **6.7-7.7 ms** on the lagging devices.

Top decode-capture GPU kernel totals:

| Kernel family | Count | Avg | p50 | p95 | Max | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a2a_dispatch_send_kernel` | 5160 | **0.745 ms** | **0.014 ms** | **7.623 ms** | **15.762 ms** | **3843.8 ms** |
| `a2a_dispatch_recv_kernel` | 5160 | **0.573 ms** | **0.020 ms** | **2.627 ms** | **39.356 ms** | **2959.1 ms** |
| `a2a_combine_recv_kernel` | 5160 | **0.458 ms** | **0.163 ms** | **0.995 ms** | **14.972 ms** | **2364.2 ms** |
| `grouped_w13` | 5160 | **0.373 ms** | **0.585 ms** | **0.706 ms** | **0.739 ms** | **1922.6 ms** |
| `grouped_w2` | 5160 | **0.207 ms** | not queried | not queried | not queried | **1066.5 ms** |
| `a2a_combine_send_kernel` | 5160 | **0.174 ms** | **0.010 ms** | **0.011 ms** | **19.579 ms** | **896.3 ms** |

Interpretation:

- The new profile kills the idea that the multi-ms layer-local MoE ranges are primarily shared-expert or grouped-GEMM arithmetic. The grouped kernels are real work, but the long host ranges are dominated by CUDA launch/API waiting and by PPLX a2a kernels already resident on other devices.
- The stable p50 term is still the full-runtime `p2p_all_to_all` / `worker_wait_combine_recv_done` cadence: `p2p_all_to_all` p50 **1.644 ms/layer**, `worker_wait_combine_recv_done` p50 **1.115 ms/layer**. This matches the earlier worker-wait profile.
- The p95/tail term is PPLX a2a kernel residency plus launch/API serialization, especially `a2a_dispatch_send` / `dispatch_recv`. Isolated A2A microbench stays microsecond-scale, so the cost appears only when the four PPLX kernels are interleaved with full model streams and explicit stream handoffs.
- The next implementation should target the full-runtime PPLX scheduling shape, not an individual expert GEMM. The highest-ceiling direction is reducing per-layer PPLX kernel residency/handshake count or moving to a single-node GPU-resident/persistent progress path; a grouped-GEMM arithmetic rewrite does not have the observed ceiling.

### Decode-Only Driver Profile Raw Facts

Profile file: `/tmp/pplx_driver_contention_olen8.sqlite` on `jzh200-11`.

Historical command shape while the temporary profiler API patch was present:

```bash
PEGAINFER_DSV4_PPLX=1 NCCL_NVLS_ENABLE=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree --sampling-period=1000000 \
  --cpuctxsw=process-tree \
  --cudabacktrace=all:1000 \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --cuda-flush-interval=100 \
  --osrt-threshold=1000 \
  --stats=true \
  -o /tmp/pplx_driver_contention_olen8 \
  ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 \
    request --prompt-len 1 --output-len 8 --warmup 0 --iters 1
```

The temporary profiler API patch has since been removed. Nsight option note: H200 has Nsight Systems 2025.1.3; the accepted CPU sampling spelling is `--sample=process-tree --sampling-period=1000000`, not `--sample=cpu --sampling-frequency=1000`.

Decode-window NVTX request facts from the same report:

| Range | Count | Avg | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dsv4.request.advance_decode` | 8 | 666.38 ms | 729.75 ms | 0.0009 ms | 832.57 ms |
| `dsv4.request.step` | 7 | 668.26 ms | 723.92 ms | 178.08 ms | 828.01 ms |
| `dsv4.request.sample` | 7 | 0.207 ms | 0.207 ms | 0.202 ms | 0.212 ms |
| `dsv4.request.emit_token` | 7 | 0.0047 ms | 0.0043 ms | 0.0024 ms | 0.0071 ms |

Kernel capture completeness:

| Device | Kernel count | Kernel total time |
| ---: | ---: | ---: |
| 0 | 17,787 | 1306.20 ms |
| 1 | 17,787 | 4994.59 ms |
| 2 | 17,787 | 3198.56 ms |
| 3 | 17,787 | 3776.68 ms |
| 4 | 17,787 | 5043.19 ms |
| 5 | 17,787 | 5046.70 ms |
| 6 | 17,787 | 5039.38 ms |
| 7 | 17,787 | 5051.83 ms |

Memcpy facts:

| Copy kind | Count | Bytes | GPU time |
| --- | ---: | ---: | ---: |
| Device-to-Host | 7 | 3.62 MB | 0.093 ms |
| Host-to-Device | 56 | 224 B | 0.0476 ms |
| Device-to-Device | 2408 | 19.73 MB | 2.420 ms |

Selected CUDA Runtime / Driver API totals:

| API | Calls | Total | Avg | Max |
| --- | ---: | ---: | ---: | ---: |
| `cudaLaunchCooperativeKernel` | 9632 | 1287.95 ms | 133.72 us | 168.34 ms |
| `cudaLaunchKernel` | 121184 | 839.06 ms | 6.92 us | 30.82 ms |
| `cuMemcpyDtoHAsync_v2` | 14 | 403.56 ms | 28.83 ms | 34.97 ms |
| `cuEventRecord` | 19264 | 75.87 ms | 3.94 us | 0.617 ms |
| `cuKernelSetAttribute` | 16 | 66.61 ms | 4.16 ms | 20.74 ms |
| `cudaEventRecord` | 14840 | 58.42 ms | 3.94 us | 2.35 ms |
| `cuMemcpyDtoDAsync_v2` | 4816 | 45.58 ms | 9.46 us | 0.100 ms |
| `cuLaunchKernelEx` | 6608 | 39.31 ms | 5.95 us | 2.57 ms |
| `cuStreamWaitEvent` | 16646 | 26.57 ms | 1.60 us | 2.19 ms |
| `cuEventCreate` | 9632 | 19.28 ms | 2.00 us | 2.99 ms |

Selected OSRT totals:

| OSRT call | Calls | Total | Avg | Max |
| --- | ---: | ---: | ---: | ---: |
| `poll` | 805 | 228438.36 ms | 283.77 ms | 1001.05 ms |
| `futex` | 127 | 12416.90 ms | 97.77 ms | 832.39 ms |
| `pthread_mutex_lock` | 298 | 782.16 ms | 2.62 ms | 108.95 ms |
| `pthread_rwlock_wrlock` | 10 | 13.99 ms | 1.40 ms | 13.85 ms |
| `ioctl` | 56 | 8.85 ms | 0.158 ms | 0.906 ms |

Rank-thread OSRT facts only for `dsv4-rank-*`:

| OSRT call | Calls | Total | Avg | Max |
| --- | ---: | ---: | ---: | ---: |
| `futex` | 53 | 2586.79 ms | 48.81 ms | 104.19 ms |
| `pthread_mutex_lock` | 298 | 782.16 ms | 2.62 ms | 108.95 ms |
| `pthread_rwlock_wrlock` | 10 | 13.99 ms | 1.40 ms | 13.85 ms |
| `ioctl` | 40 | 8.45 ms | 0.211 ms | 0.906 ms |

Top non-communication API tails with matched GPU work:

| API time | API | Rank | Correlation | GPU time | GPU kernel / owner |
| ---: | --- | --- | ---: | ---: | --- |
| 30.824 ms | `cudaLaunchKernel` | rank3 | 116338 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 29.799 ms | `cudaLaunchKernel` | rank0 | 116629 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 27.926 ms | `cudaLaunchKernel` | rank2 | 116842 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 20.745 ms | `cuKernelSetAttribute` | rank4 | 302880 | n/a | inside `attention_ratio4 step=3 layer=2` by NVTX containment |
| 20.612 ms | `cuKernelSetAttribute` | rank7 | 302889 | n/a | inside `attention_ratio4 step=3 layer=2` by NVTX containment |
| 20.293 ms | `cuKernelSetAttribute` | rank2 | 302925 | n/a | inside `attention_ratio4 step=3 layer=2` by NVTX containment |
| 19.877 ms | `cudaLaunchKernel` | rank5 | 116010 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 19.434 ms | `cudaLaunchKernel` | rank7 | 116019 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 19.105 ms | `cudaLaunchKernel` | rank4 | 115947 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 18.766 ms | `cudaLaunchKernel` | rank6 | 116054 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 18.138 ms | `cudaLaunchKernel` | rank1 | 116396 | 0.030 ms | `deepseek_hadamard_rotate_bf16_serial_kernel` / `indexer::hadamard_fp4_quant_bf16_hidden_in_place` |
| 16.160 ms | `cudaLaunchKernel` | rank6 | 303116 | 0.008 ms | `deepseek_compressor_norm_serial_kernel` / `compressor_overlap_decode_bf16_hidden_with_dim_scratch` |
| 16.064 ms | `cudaLaunchKernel` | rank3 | 303119 | 0.0035 ms | `deepseek_hc_scale_mixes_block_kernel` / `hc_pre_norm_bf16_hidden_scratch` |
| 15.964 ms | `cudaLaunchKernel` | rank1 | 303167 | 0.0018 ms | `deepseek_compressor_overlap_decode_weighted_kernel` / `compressor_overlap_decode_bf16_hidden_with_dim_scratch` |
| 15.886 ms | `cudaLaunchKernel` | rank5 | 303169 | 0.016 ms | `deepseek_indexer_scores_decode_serial_kernel` / `indexer_scores_decode_bf16_hidden_scratch` |
| 11.895 ms | `cudaLaunchKernel` | rank0 | 303197 | 0.0014 ms | cuBLAS `reduce_1Block_kernel` / `hc_pre_norm_bf16_hidden_scratch` |

Raw stack facts for the largest OSRT mutex tails:

| OSRT | Duration | Rank | Stack head |
| --- | ---: | --- | --- |
| `pthread_mutex_lock` | 108.95 ms | rank5 | `libcuda -> cuModuleGetFunction -> cudaLaunchCooperativeKernel -> a2a_dispatch_send` |
| `pthread_mutex_lock` | 106.22 ms | rank7 | `libcuda -> cuModuleGetFunction -> cudaLaunchCooperativeKernel -> a2a_dispatch_send` |
| `pthread_mutex_lock` | 90.68 ms | rank4 | `libcuda -> cuModuleGetFunction -> cudaLaunchCooperativeKernel -> a2a_dispatch_send` |
| `pthread_mutex_lock` | 84.88 ms | rank2 | `libcuda -> cuLibraryGetModule -> cudaLaunchCooperativeKernel -> a2a_dispatch_send` |
| `pthread_mutex_lock` | 80.52 ms | rank6 | `libcuda -> cuModuleGetFunction -> cudaLaunchCooperativeKernel -> a2a_dispatch_send` |

CPU sampling facts:

| Thread group | Sample state |
| --- | --- |
| `dsv4-rank-*` | 13,266 samples, all `Running` |
| Per rank | rank0 1757, rank1 1716, rank2 1578, rank3 1549, rank4 1682, rank5 1655, rank6 1660, rank7 1669 |

Interpretation guardrail: the tables above are facts from the decode-only profile. The profile itself has Nsight overhead and should not be used as a wall-clock TPOT comparison against non-profile runs; it is useful for attribution of kernel completeness, API tail, OSRT lock, and API-vs-GPU mismatch.

## Experiment Discipline

Every new perf experiment must start by writing these fields in this document before changing code or running H200:

| Field | Required form |
| --- | --- |
| Target metric | Concrete metric + threshold + minimum sample size, for example `steady TPOT p50 <= 110 ms over >= 16 decode steps` or `a2a union p50 reduced by >= 20 ms over >= 5 steady steps`. |
| Falsifiable hypothesis | `If this is true, X metric moves in Y direction by at least Z`. A hypothesis that cannot name X/Y/Z is still diagnosis, not an implementation task. |
| Ceiling estimate | Use current profile attribution to estimate max possible win before coding. Drop the experiment when the upper bound is below the decision threshold. |
| Keep/revert criterion | Written before the run. Decisions cannot be based on a post-hoc metric chosen after seeing the trace. |
| Noise rule | Single-run avg/max can only generate hypotheses. It cannot keep or revert a change unless a hypothesis-specific metric has an unambiguous binary outcome, such as allocator calls becoming zero. |
| Ledger state | Every hypothesis is `alive`, `killed`, or `pending`, with evidence and acceptance criterion in the ledger below. |
| Process check | When the same evidence is discussed twice, stop the technical chase and update the ledger/criteria first. |
| Lesson rule | Every reverted or abandoned experiment leaves one reusable lesson, phrased as process or mechanism, not just as a case description. |

Current working target for the next non-trivial perf change: **H200 EP8 decode steady TPOT p50 <= 110 ms over at least 16 decode steps**, with no correctness regression and no single step above the existing steady max class unless the profile proves a different bottleneck was intentionally exposed. The decision gate is **>= 15 ms p50 improvement** for ratio4/attention-boundary work or **>= 20 ms p50 improvement** for pplx worker/a2a union work; smaller wins are documentation-only unless the code also removes a correctness risk or a future blocker.

Current ratio4 refactor gate:
- **Target metric**: H200 EP8 `attention_ratio4` boundary-token p50 reduced by at least **15 ms** over at least **5 boundary decode steps**; request-level steady TPOT p50 must not regress over at least **16 decode steps**.
- **Hypothesis**: if ratio4 boundary stalls are dominated by tiny-kernel launch fanout, then reducing the decode ratio4 path from `window_topk + indexer_topk + concat` to one fused topk kernel, and then reducing boundary compressor post-processing fanout, will lower boundary-token `attention_ratio4` p50 by at least 15 ms.
- **Ceiling estimate**: current ratio4 boundary layers show 17-33 ms maxima, while the fused topk pass can remove two launches per ratio4 layer when `compressed_len > 0`; compressor boundary fusion has a higher ceiling because each main/indexer compressor boundary expands by roughly six extra tiny launches. The topk fusion is kept only as a structural cleanup unless profile shows a measurable wall win.
- **Keep/revert criterion**: keep fused topk when correctness/build pass and the generated topk count/index semantics match the old path; keep compressor fusion only when boundary-token p50 meets the 15 ms gate or removes a confirmed launch/API stall without TPOT regression.

Current ratio4 bs=1 scratch fast-path experiment:
- **Target metric**: for H200 EP8 bs=1/slot0, `attention_ratio4` should no longer call owned `_at` compressor allocation in the hot loop; `cudaMalloc` inside ratio4 boundary ranges should disappear in the next decode-only profile, and request-level steady TPOT p50 must not regress over at least **16 decode steps**.
- **Hypothesis**: if a meaningful part of ratio4 boundary variance comes from the batch helper still allocating a row copy plus compressor `weighted/out` buffers even when `batch=1`, then routing bs=1/slot0 through the existing scratch compressor will reduce allocator/API churn and lower or stabilize `attention_ratio4` boundary ranges.
- **Ceiling estimate**: this removes three owned allocations and one row-copy launch per ratio4 layer boundary in the current serving path, but it does not fuse compressor kernels or remove hadamard/TileLang launches. Expected wall-clock win is likely smaller than the 15 ms main gate; this change is keepable as hygiene only if allocator churn is eliminated and TPOT does not regress.
- **Keep/revert criterion**: keep when local/H200 builds pass, bs=1 serving smoke passes, no numerical/correctness error appears, and either next profile shows the targeted `cudaMalloc` removal or H200 p50 improves without p95/max regression. Revert if slot0 assumptions leak into non-slot0 batch behavior or p50 regresses without a profile-backed reason.
- **Result**: reverted. Local build passed and H200 build passed, but H200 `output_len=16` serving smoke regressed from the post-graph-removal comparison point p50 **143.99 ms** / p95 **184.00 ms** to p50 **168.01 ms** / p95 **224.03 ms**, with first decode **407.65 ms**; log `/tmp/pplx_ratio4_bs1_scratch_olen16.log`. Mechanism is not fully attributed, but the attempted shortcut changed compressor state/output lifetime enough to hurt the full pipeline. Future ratio4 work should fuse the compressor/update kernels or remove hadamard/FP4 quant launch fanout, not route the batch helper through this narrow bs=1 scratch branch.

Current TileLang dynamic-shmem attribute cache experiment:
- **Target metric**: eliminate repeated `cuKernelSetAttribute` calls from TileLang launch wrappers after the first launch per device/kernel; next decode-only profile should show no steady-step 15-20 ms `cuKernelSetAttribute` tails, and H200 `output_len=16` serving p50 must not regress from the post-graph-removal **143.99 ms** comparison point.
- **Hypothesis**: if part of the ratio4/HC launch variance comes from per-launch dynamic shared-memory attribute setup in generated TileLang wrappers, then replacing unconditional `cudaFuncSetAttribute` with a per-device `std::call_once` cache will remove those API tails without changing GPU kernels or numerical output.
- **Ceiling estimate**: the decode-only driver profile recorded only **16** `cuKernelSetAttribute` calls but three were **20.3-20.7 ms** inside `attention_ratio4 step=3 layer=2`; this is not enough to close the full 80 ms gap, but it is a low-risk driver-state cleanup with a possible 10-20 ms tail reduction on affected steps.
- **Keep/revert criterion**: keep only if generated CUDA builds locally and on H200, serving smoke completes, and p50/p95 do not regress. Treat a p50 improvement below 15 ms as hygiene unless a follow-up profile proves the attribute tails are gone.
- **Result**: reverted. Local generated CUDA build passed, H200 generated CUDA build passed, and H200 serving smoke completed, but `output_len=16` was p50 **144.01 ms** with p95 **280.15 ms** and max **320.01 ms** (`/tmp/pplx_tilelang_attr_once_olen16.log`). p50 matched the post-graph-removal comparison point, but p95/max regressed enough to fail the pre-written gate. Lesson: `cuKernelSetAttribute` tails are real in the profiling trace, but caching the attribute in the generated wrapper did not produce a robust serving win; revisit only with a decode-only profile that proves the tail repeats in steady state and not just as a first-use artifact.

Current Hadamard+FP4 fused experiment:
- **Target metric**: replace `hadamard_fp4_quant_bf16_hidden_in_place`'s rotate kernel + TileLang FP4 quant launch with one fused CUDA launch for `dim=128`, keeping H200 serving p50 non-regressing and ideally reducing ratio4 boundary p95. A retained change must pass build and at least one bs=1 H200 serving smoke; exact E2E should be run before merging if the smoke looks promising.
- **Hypothesis**: if ratio4 boundary API tails are dominated by the two-launch Hadamard+FP4 quant path and its global scratch wrapper, then a fused serial per-vector kernel will remove one launch plus wrapper scratch handling and reduce `attention_ratio4` tail without changing intended quant-dequant semantics.
- **Ceiling estimate**: driver profile shows `deepseek_hadamard_rotate_bf16_serial_kernel` launch API tails up to **30.82 ms**, while the GPU kernel itself is only about **0.030 ms**. The fused kernel cannot remove all driver contention, but it can halve this specific operator's launch count and remove the TileLang wrapper from this path.
- **Keep/revert criterion**: keep only if local CUDA build passes, H200 build passes, serving smoke completes, p50/p95 do not regress, and generated tokens remain sane. Revert immediately on build failure, illegal address, NaN-like behavior, or serving p50/p95 regression.
- **Result**: reverted. Local build passed, H200 build passed, and H200 `output_len=16` serving smoke generated all 16 tokens, but the run measured `first_decode_step_ms` **207.76 ms**, steady TPOT avg **165.41 ms**, p50 **144.03 ms**, p95 **216.01 ms**, max **244.02 ms** (`/tmp/pplx_fused_hadamard_fp4_olen16.log`). The p50 matched the post-graph-removal comparison point, but p95 regressed from **184.00 ms** to **216.01 ms**, so the pre-written p95 gate failed. Lesson: removing one launch from the Hadamard+FP4 pair is not enough when the observed tail is mostly shared libcuda launch-state contention; operator-local fusion must be large enough to reduce whole-range launch pressure, not just replace one wrapper call.

Current pplx worker CPU-pool separation experiment:
- **Target metric**: H200 EP8 `output_len=64` serving p95 should improve by at least **20 ms** without p50 regression over at least **62 steady samples**. Baseline after graph/fused cleanup is p50 **144.01 ms**, p95 **216.01 ms**, max **300.01 ms**, avg **160.19 ms** (`/tmp/pplx_current_olen64.log`).
- **Hypothesis**: if part of the steady tail comes from host progress jitter, then avoiding exact CPU overlap between DeepSeek rank workers and pplx a2a workers will reduce p95/max while leaving p50 roughly unchanged.
- **Ceiling estimate**: the new clean log shows an exact conflict: DeepSeek rank worker 3 is pinned to CPU **6**, and pplx a2a worker for cuda:0 is also pinned to CPU **6**. Earlier profiles showed p95/max dominated by host progress/driver wait, so the plausible benefit is tail reduction rather than a 60-80 ms p50 win.
- **Keep/revert criterion**: keep only if local/H200 build passes, H200 logs show no rank-worker/a2a-worker exact CPU overlap, `output_len=64` smoke generates all 64 tokens, and p95 improves by >=20 ms with p50 <= baseline + 5 ms. Revert if p50 regresses beyond 5 ms, p95/max worsens, or affinity selection fails on a constrained CPU mask.
- **Result**: kept. Local `cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed, H200 `cargo build --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed, and two H200 `output_len=64` serving runs completed all 64 tokens. The new logs show rank workers on CPUs **0/2/4/6/9/11/13/15** and pplx a2a workers on **8/26/50/74/3/27/51/75**, with no exact overlap. Results: `/tmp/pplx_cpu_pool_olen64.log` measured p50 **144.00 ms**, p95 **159.96 ms**, max **164.00 ms**, avg **144.06 ms**; `/tmp/pplx_cpu_pool_olen64_r2.log` measured p50 **144.00 ms**, p95 **162.86 ms**, max **168.01 ms**, avg **145.03 ms**. This passes the tail gate and reduces average TPOT by ~15-16 ms versus `/tmp/pplx_current_olen64.log`; it does not move p50, so the remaining gap is not CPU-overlap tail. Residual risk: the second run printed a teardown-time NCCL abort panic after metrics were emitted; this matches the known shutdown-path instability and is not forward-path evidence.

Current intra-process route exchange experiment:
- **Target metric**: H200 EP8 `output_len=64` serving p50 should improve by at least **10 ms** over the CPU-pool baseline p50 **144.00 ms**, with p95 staying <= **165 ms** and all 64 tokens generated.
- **Hypothesis**: if p50 floor still includes per-layer fabric route all-gather overhead, then in the single-process single-node case replacing `route_write_op + route_counter.wait` with a process-local barrier plus direct reads of peer `num_routed` mapped host pointers will reduce p50. This should not change dispatch/combine payload semantics.
- **Ceiling estimate**: every MoE layer currently performs route exchange before `process_routing_info`, even though all 8 rank workers live in one process and each rank's `num_routed_host` pointer is directly addressable. The ceiling is one worker transfer submission + immediate wait per layer, so a plausible win is **5-15 ms** p50; it will not close the full 80 ms gap alone.
- **Keep/revert criterion**: keep only if local/H200 builds pass, H200 `output_len=64` completes with all tokens, p50 improves by >=10 ms, and p95 stays <=165 ms. Revert on hang, correctness error, p50 regression, or teardown/stop deadlock.
- **Result**: reverted. Local `cargo fmt -p pegainfer-comm -p pegainfer-deepseek-v4` and `cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed, H200 build passed, and the short H200 smoke `/tmp/pplx_direct_route_olen8.log` completed with steady avg **141.96 ms**, p50 **143.98 ms**, p95/max **159.08 ms**. The full gate run `/tmp/pplx_direct_route_olen64.log` completed all 64 tokens with first decode **223.66 ms**, steady avg **142.64 ms**, p50 **144.00 ms**, p95 **155.95 ms**, max **160.04 ms**. p95 stayed good but p50 did not move by the required 10 ms, so the code was removed. Post-revert H200 smoke `/tmp/pplx_cpu_pool_restored_olen8.log` completed all 8 tokens with p50 **151.81 ms** and p95/max **164.01 ms**; this is a short correctness smoke, not a new baseline. Mechanism lesson: removing only the route all-gather submission/wait is too small or already overlapped; the p50 floor is in larger per-layer a2a state-machine/device wait work, not this specific route exchange.

Current bs=1 pplx capacity clamp experiment:
- **Target metric**: H200 EP8 bs=1 `output_len=64` serving p50 should improve by at least **15 ms** over CPU-pool baseline p50 **144.00 ms**, with p95 <= **165 ms** and all 64 tokens generated.
- **Hypothesis**: if a meaningful part of the 144 ms p50 floor is grouped FP4 work over unused pplx scratch capacity, then clamping pplx decode buffers to the actual bs=1 validation envelope will lower `expanded_input.seq_capacity()` and the grouped FP4 `rows` launch bound enough to move p50. This targets the current GPU-only rows issue without reintroducing per-layer host readback.
- **Ceiling estimate**: current default `max_num_tokens=8` plus upstream private-token formula gives `max_recv_tokens=1376` rows for H200 EP8 (`topk=6`, local experts=32, padding=16). For bs=1, setting `max_num_tokens=1` and `max_private_tokens=topk` gives `max_recv_tokens=560`, a **59%** reduction in the W1/W3 and W2 grouped FP4 row bound. If grouped capacity work is a large part of p50, expected win is **15-30 ms**; if p50 is dominated by a2a state-machine waits, p50 will stay near 144 ms.
- **Keep/revert criterion**: keep only if local/H200 builds pass, H200 `output_len=64` generates all 64 tokens, p50 improves by >=15 ms, and p95 stays <=165 ms. Revert on capacity error, illegal address, correctness-looking output failure, p50 regression, or p95 regression. This experiment is explicitly bs=1; it does not claim batch-serving support.
- **Result**: reverted. Local `cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed and H200 build passed. H200 short smoke `/tmp/pplx_bs1_capacity_olen8.log` completed all 8 tokens with first decode **198.61 ms**, steady avg **143.97 ms**, p50 **144.00 ms**, p95/max **156.02 ms**. Full gate `/tmp/pplx_bs1_capacity_olen64.log` completed all 64 tokens with first decode **199.95 ms**, steady avg **143.24 ms**, p50 **144.00 ms**, p95 **155.74 ms**, max **192.25 ms**. The p50 did not move, so the 59% row-bound reduction is not the missing 80 ms. Mechanism lesson: the grouped FP4 capacity overrun may affect small averages, but the p50 floor is dominated by a fixed per-token/per-layer synchronization or worker-state cost outside grouped rows.

Current local CUDA Graph island experiment:
- **Target metric**: H200 EP8 bs=1 pplx decode steady TPOT p50 reduced by at least **10 ms** over at least **16 decode steps**; no correctness regression or CUDA graph capture failure.
- **Hypothesis**: if the remaining non-communication gap is materially caused by eager tiny-kernel launch/API overhead in graph-safe static islands, then graphing embedding+HC expand, per-layer HC pre-norm/post, and final logits will reduce `cudaLaunchKernel`/event API calls on replay and lower steady p50 by at least 10 ms.
- **Ceiling estimate**: driver profile shows `cuEventRecord` 75.87 ms, `cudaEventRecord` 58.42 ms, `cuStreamWaitEvent` 26.57 ms, plus many microsecond kernels inside HC/final-logits paths. The first graph island pass excludes attention `start_pos` and excludes MoE/NCCL, so expected p50 win is **8-15 ms**, with p95/max possibly larger. A 20 ms+ win would justify converting attention host `start_pos` to device metadata and expanding graph coverage.
- **Keep/revert criterion**: keep only if local build passes, H200 build passes, bs=1 correctness smoke completes, and either steady p50 improves by >=10 ms or decode-only profile proves a large API-call reduction without new instability. Revert if graph capture fails, graph replay produces stale-token behavior, or p50 moves less than 5 ms with no clear API reduction.
- **Result**: local build passed, H200 build passed, and bs=1 smoke completed. H200 `output_len=24` non-profile run completed with `prefill_ms=576.78`, `first_decode_step_ms=491.85`, steady TPOT avg **159.81 ms**, p50 **154.78 ms**, p95 **214.96 ms**, max **292.08 ms**, samples **22** (`/tmp/pplx_local_graph_olen24.log`). This is worse than the prior ratio4 batch topk profile class (`output_len=16`, avg **152.85 ms**, p50 **144.03 ms**, p95 **188.01 ms**), so the wall-clock gate failed.
- **Profile result**: decode-only nsys profile `/tmp/pplx_local_graph_profile_olen12.{log,nsys-rep,sqlite}` captured 11 request steps / 88 rank decode ranges. It recorded **1056** `cuGraphInstantiateWithFlags` calls totaling **944.08 ms**, exactly matching 132 graph islands × 8 ranks. Replay added **23232** `cuGraphLaunch` calls totaling **421.25 ms**. Normalized by request step, `cudaLaunchKernel` fell from **17312** calls/step in `/tmp/pplx_driver_contention_olen8.sqlite` to **13827** calls/step, but adding `cuGraphLaunch` gives **15939** launch-class calls/step, only about **8%** below the old profile. `cuEventRecord` stayed **2752** calls/step, and `cuStreamWaitEvent` stayed in the same range (**2378 -> 2418** calls/step). Mechanism lesson: graph islands this small do remove some kernel launches, but they do not remove the explicit stream handoffs and they replace many launches with graph launches; the effective unit must be a larger operator island or a generated static decode block, not per-helper graphlets.
- **Cleanup result**: fine-grained graph island state/wrappers were removed from `state.rs`, `worker.rs`, and `block.rs`; NVTX instrumentation stayed. Local validation passed: `cargo fmt -p pegainfer-deepseek-v4`, `cargo check --release -p pegainfer-deepseek-v4 --features pplx-ep --bin deepseek_pplx_a2a_bench`, and `cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving`. H200 validation passed: `cargo build --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving`, then `PEGAINFER_DSV4_PPLX=1 NCCL_NVLS_ENABLE=0 ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 request --prompt-len 1 --output-len 16 --warmup 0 --iters 1` completed with `prefill_ms=617.40`, `first_decode_ms=239.31`, steady TPOT avg **158.84 ms**, p50 **143.99 ms**, p95 **184.00 ms**, max **255.91 ms**, samples **14**; log `/tmp/pplx_no_graph_islands_olen16.log`.

Current pplx worker-wait decomposition profile:
- **Target metric**: H200 EP8 `output_len=64` NVTX-only profile should explain the PPLX non-rank0 p50 lane (**~74 ms**) by named worker waits, with at least **62 steady samples**. This is diagnostic; it does not decide keep/revert for a perf code path.
- **Hypothesis**: if the 74ms non-rank0 lane is the per-layer PPLX worker state machine rather than model compute or raw payload transfer, then `p2p_all_to_all` p50 should be near `74ms / 43 layers ~= 1.7ms`, and one or two named waits should account for most of that per-layer p50.
- **Ceiling estimate**: eliminating 1ms/layer of worker wait has a direct ceiling of **~43 ms/token** on non-rank0 lanes, and rank0/wait-rank should follow because `logits_dtoh` is the final drain.
- **Result**: confirmed. Instrumented only `WorkerState::step()` NVTX waits in `pegainfer-comm-p2p-all-to-all/src/a2a_worker.rs`; local `cargo fmt -p pegainfer-comm` passed, local `cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed, H200 release build passed. H200 profile `/tmp/pplx_worker_wait_nvtx_olen64.{log,sqlite,nsys-rep}` completed all 64 tokens: first decode **211.51 ms**, steady TPOT p50 **144.00 ms**, p95 **159.92 ms**, max **164.12 ms**, samples **62**.
- **Worker evidence**:
  - `p2p_all_to_all`: count **21680**, p50 **1.609 ms**, p95 **16.720 ms**, avg **3.951 ms**; count matches roughly `63 decode steps * 8 ranks * 43 MoE layers`.
  - `worker_wait_combine_recv_done`: count **21672**, p50 **1.111 ms**, p95 **1.175 ms**, p99 **1.191 ms**. This is the stable per-layer floor.
  - `dispatch`: p50 **0.010 ms**, p95 **0.013 ms**. Dispatch-side worker work is not the p50 owner after correctness fixes.
  - `worker_wait_dispatch_route_done`: p50 **0.345 ms**, p95 **0.574 ms**, p99 **9.357 ms**; it contributes some p50 and the largest first/cold tails.
  - `worker_wait_route_counter`: p50 **0.001 ms**, p95 **3.316 ms**, p99 **15.827 ms**; this is a tail/rank-arrival term, not the p50 floor.
  - `barrier_wait_imm`: p50 **0.008 ms**, p95 **0.018 ms**, p99 **15.106 ms**; barrier immediates drive tail, not the median floor.
- **Rank evidence**: rank0 decode p50 **143.772 ms**; non-rank0 p50s are rank1 **74.283 ms**, rank2 **79.957 ms**, rank3 **74.292 ms**, rank4 **74.287 ms**, rank5 **74.283 ms**, rank6 **74.277 ms**, rank7 **74.274 ms**. This reproduces the previous NVTX-only differential profile.
- **Mechanism lesson**: single-node EP8 p50 is no longer a dispatch-send or payload-copy problem. The steady floor is `combine_recv_done` completion latency repeated 43 times, plus smaller route-done p50; route/barrier counters mainly explain p95/p99. The next implementation should shrink or remove the per-layer combine completion handshake in the single-node path, not keep shaving route submit or grouped-row capacity.

Current single-node combine-recv grid clamp experiment:
- **Target metric**: H200 EP8 `output_len=64` serving completes all 64 tokens; `worker_wait_combine_recv_done` p50 should fall from **1.111 ms/layer** to **<=0.5 ms/layer** in a follow-up NVTX-only profile, and request steady TPOT p50 should improve by at least **20 ms** versus the current p50 **144.00 ms**. p95 must stay <= **165 ms**.
- **Hypothesis**: if the stable 1.111 ms/layer combine floor comes from launching `a2a_combine_recv_kernel` as an SM-count cooperative grid even when `num_tokens=1`, then single-node `combine_recv` can launch only `min(num_tokens, num_sms)` blocks without changing output semantics, reducing the worker's `combine_recv_done` wait and the non-rank0 lane.
- **Ceiling estimate**: `worker_wait_combine_recv_done` p50 **1.111 ms/layer * 43 layers ~= 47.8 ms/token**. Even a 50% reduction is enough to meet the **20 ms** p50 gate.
- **Keep/revert criterion**: keep only if local/H200 builds pass, H200 `output_len=64` generates all tokens, p50 improves by >=20 ms, p95 <=165 ms, and a follow-up worker-wait profile confirms `worker_wait_combine_recv_done` p50 <=0.5 ms/layer. Revert on hang, CUDA illegal address, wrong-looking output failure, p50 regression, or p95 regression.
- **Result**: reverted. Local `cargo fmt -p pegainfer-comm` passed and local `cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed. H200 build passed. H200 gate `/tmp/pplx_combine_recv_grid_olen64.log` generated all 64 tokens but measured first decode **195.83 ms**, steady TPOT avg **146.26 ms**, p50 **144.00 ms**, p95 **164.00 ms**, max **187.52 ms**, samples **62**. The process then hit the known teardown segfault, but metrics were already emitted; the forward gate failed because p50 did not improve. Mechanism lesson: the 1.111 ms/layer `worker_wait_combine_recv_done` floor is not fixed by reducing `a2a_combine_recv_kernel` from SM-count blocks to `num_tokens` blocks. The cost is more likely in the flag/worker completion protocol or combine-send/recv dependency chain than in empty cooperative-grid block count alone.

Current single-node combine-recv host-flag skip experiment:
- **Target metric**: H200 EP8 `output_len=64` serving completes all 64 tokens; steady TPOT p50 improves by at least **20 ms** versus **144.00 ms**, p95 stays <= **165 ms**. Follow-up worker-wait profile should show `worker_wait_combine_recv_done` p50 below **0.5 ms/layer**.
- **Hypothesis**: if the stable `worker_wait_combine_recv_done` p50 comes from `a2a_combine_recv_kernel` polling a host-set GDR flag, then single-node (`world_size == node_size`) can skip `combine_recv_flag` because there are no fabric combine payloads; same-stream ordering plus `sync_ptrs` already protect local NVLink combine copies. This should remove the per-layer host→GPU flag latency without changing cross-node behavior.
- **Ceiling estimate**: CUDA+NVTX profile `/tmp/pplx_cuda_nvtx_olen8.sqlite` captured `a2a_combine_recv_kernel` p50 **179 us** but worker `combine_recv_done` wait p50 **1.113 ms**, leaving roughly **0.9 ms/layer** unexplained by kernel compute. That is **~38 ms/token** across 43 MoE layers.
- **Keep/revert criterion**: keep only if local/H200 builds pass, H200 `output_len=64` generates all tokens, p50 improves by >=20 ms, p95 <=165 ms, and worker-wait profile confirms the `combine_recv_done` floor moved. Revert on hang, CUDA illegal address, wrong-looking output failure, p50 regression, or p95 regression.
- **Result**: reverted. Local build with `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed; H200 release build passed. H200 gate `/tmp/pplx_combine_recv_skip_host_flag_olen64.log` completed with exit status 0 and generated all 64 tokens, but measured first decode **199.75 ms**, steady TPOT avg **144.71 ms**, p50 **144.00 ms**, p95 **160.00 ms**, max **164.00 ms**, samples **62**. The p50 did not move, so skipping the host flag is not enough. Mechanism lesson: the `combine_recv_done` wait range is not explained by either empty cooperative-grid blocks or the `combine_recv_flag` MMIO poll in isolation; the remaining floor is likely the broader same-stream combine-send/recv dependency plus worker state-machine cadence.

Current a2a device wait-counter profile:
- **Target metric**: H200 short run should emit device-side wait counters for all four a2a kernels at shutdown, especially `combine_recv recv_flag_avg_cycles` and `combine_recv nvlink_sum_cycles`. This is diagnostic only.
- **Hypothesis**: if `worker_wait_combine_recv_done` is hiding GPU-side polling/sync, then device counters will show large combine-recv flag or NVLink sync cycles. If counters stay small while worker wait remains large, the floor is outside the measured kernel wait loops and likely in same-stream dependency or host worker cadence.
- **Ceiling estimate**: the previous CUDA+NVTX profile captured only devices 0/1/2/4, so kernel durations are incomplete. Device counters are cheaper and rank-local, giving all 8 ranks at shutdown.
- **Keep/revert criterion**: run with counters enabled, record the emitted logs, then disable the pointer again so steady benchmarks do not carry atomic counter overhead.
- **Result**: confirmed the wait source. H200 `/tmp/pplx_a2a_debug_counters_olen8.log` generated all 8 tokens and then hit the known teardown segfault. Metrics before teardown: steady p50 **143.99 ms**. All ranks reported `fabric_tokens=0`. `combine_recv recv_flag_avg_cycles` was only **543-644 cycles**, but `combine_recv nvlink_sum_cycles` was huge: rank0 **144,368,156,274**, rank4 **150,052,161,007**, other ranks **113-137B**. This points to the `sync_ptrs[local_rank][peer + NODE_SIZE]` all-peer wait inside `a2a_combine_recv_kernel`, not the host-set `combine_recv_flag`.

Current single-node source-specific combine sync experiment:
- **Target metric**: H200 EP8 `output_len=64` serving completes all 64 tokens; steady TPOT p50 improves by at least **20 ms** versus **144.00 ms**, p95 stays <= **165 ms**. A follow-up short debug-counter run should show much lower `combine_recv nvlink_sum_cycles`.
- **Hypothesis**: if the p50 floor comes from `a2a_combine_recv_kernel` waiting for all 8 peer combine-send sync flags even when a token only needs its routed source expert ranks, then in the single-node case waiting only the source expert rank for each token/route will reduce per-layer combine sync time without changing cross-node behavior.
- **Ceiling estimate**: current all-peer `combine_recv nvlink_sum_cycles` is 100B+ cycles over 301 kernels in an `output_len=8` run. Even a partial reduction can exceed the **20 ms/token** gate because this wait repeats 43 layers per token.
- **Keep/revert criterion**: keep only if local/H200 builds pass, H200 `output_len=64` generates all tokens, p50 improves by >=20 ms, and p95 <=165 ms. Revert on hang, illegal address, wrong-looking output failure, p50 regression, or p95 regression.
- **Result**: reverted. Local CUDA build passed after fixing an initial `counter` scope compile error; H200 release build passed. H200 gate `/tmp/pplx_source_specific_sync_olen64.log` timed out with status **124** before emitting request metrics, so the change hung. Mechanism lesson: `combine_recv` cannot safely replace the all-peer sync with per-token source waits using only `expert -> rank` inference. The current sync protocol is coupled to the full combine-send grid completion, not merely the source expert rank for each route.

Current single-node active-source combine mask experiment:
- **Target metric**: H200 EP8 `output_len=64` serving completes all 64 tokens; steady TPOT p50 improves by at least **20 ms** versus **144.00 ms**, p95 stays <= **165 ms**. This was a stricter retry of source-specific sync: compute the active source-rank mask in the worker from `num_routed[self_dp_group, expert]` and pass it to `a2a_combine_recv_kernel`, rather than inferring the wait set per token inside the kernel.
- **Hypothesis**: if the previous source-specific hang came from kernel-side source inference rather than the protocol itself, then a worker-derived exact active-source mask should avoid the hang and reduce the `sync_ptrs[local_rank][peer + NODE_SIZE]` all-peer wait.
- **Ceiling estimate**: same as the previous source-specific attempt: the all-peer `combine_recv nvlink_sum_cycles` counter is large enough that removing non-source peers could plausibly exceed the **20 ms/token** gate if the protocol allowed it.
- **Keep/revert criterion**: keep only if local/H200 builds pass, H200 `output_len=64` generates all tokens, p50 improves by >=20 ms, and p95 <=165 ms. Revert on hang, illegal address, wrong-looking output failure, p50 regression, or p95 regression.
- **Result**: reverted. The implementation used `num_recv_tokens[2]` as a GDR-visible active-source mask, updated the C++/Rust FFI signature, and made warp1 in `a2a_combine_recv_kernel` wait only mask lanes. Local `cargo fmt -p pegainfer-comm` passed and local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed. H200 release build passed, but `/tmp/pplx_active_source_mask_olen64.log` timed out with status **124** before any forward metric. The experiment was removed locally and remotely; restored H200 build passed and `/tmp/pplx_restored_after_mask_olen8.log` completed with 8 tokens, steady p50 **143.96 ms**. Mechanism lesson: even an exact active-source wait set is not a safe local change. The all-peer combine sync is part of a larger bidirectional buffer-reuse/state-machine protocol, not a pure data-dependency wait.

Direct-combine feasibility probe:
- **Target metric**: determine whether a future direct-combine prototype can reuse ordinary `CudaSlice` pointers (`expert_out.data`) through CUDA peer access, or whether `expert_out` must be reallocated as bootstrap-managed CUMem.
- **Probe**: compiled and ran `/tmp/peer_ptr_probe.cu` on H200. It enables peer access for every ordered GPU pair, `cudaMalloc`s one `int` per device, passes all 8 raw device pointers to a kernel on each GPU, and reads every peer pointer.
- **Result**: all ordered pairs reported `cudaDeviceCanAccessPeer=1`; every reader GPU read values `1000..1007` from all 8 allocations with `launch=no error` and `sync=no error`.
- **Mechanism lesson**: direct-combine does not have to force `expert_out` into CUMem purely for addressability on this H200 node, but cudarc async allocations are not equivalent to the raw `cudaMalloc` probe. The first direct kernel hit CUDA 700 until each rank enabled CUDA peer access **and** called `cudaMemPoolSetAccess` on the local default mempool for every peer GPU. Pointer exchange is still viable, but the direct path must own peer/mempool access setup explicitly. The remaining hard parts are (1) a second `EnablePplx`-style handshake to collect those pointers from rank workers, (2) a GPU-side direct padded-index computation from `indices/token_offset/num_routed`, and (3) replacing the worker combine stage with a GPU-only publish/wait path so the a2a worker does not block forever waiting for `combine_send_done`.

Current direct-combine prototype:
- **Target metric**: H200 EP8 `output_len=64` serving completes all 64 tokens; steady TPOT p50 improves by at least **20 ms** versus **144.00 ms**, p95 stays <= **165 ms**. Short `output_len=8` smoke must generate all 8 tokens before the long gate.
- **Hypothesis**: if the 1.111 ms/layer combine-completion floor comes from copying routed expert output through legacy `combine_send -> recv_buffer -> combine_recv`, then a single-node direct-combine kernel can publish local `expert_out` readiness, wait for peer readiness, compute the same padded source index from `indices/token_offset/num_routed`, and reduce directly from peer `expert_out` pointers. This should remove one legacy combine-send payload/copy stage and materially lower the non-rank0 lane.
- **Ceiling estimate**: `worker_wait_combine_recv_done` p50 is **1.111 ms/layer**, or **~47.8 ms/token** across 43 layers. A direct-combine path only needs to recover ~40% of that floor to pass the **20 ms** p50 gate.
- **Implementation result**: local `cargo fmt -p pegainfer-comm -p pegainfer-deepseek-v4` passed. Local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed. H200 release build passed after syncing the direct kernel, cxx FFI, `AllToAllContext::direct_combine_recv`, `EpBackend::direct_combine_recv`, the peer `expert_out` pointer table, and the `moe_pplx.rs` call site.
- **Gate result**: failed and disabled. Initial H200 `/tmp/pplx_direct_combine_olen8.log` timed out with status **124** after benchmark start. Metadata-only direct-combine also timed out until the direct kernel waited for all peer first-half flags before publishing its second-half ready flag; after that protocol fix, `/tmp/pplx_direct_metadata_waitfix_olen8.log` generated all 8 tokens with steady p50 **152.00 ms**. Full direct then localized CUDA 700 to `direct_combine_recv` under `CUDA_LAUNCH_BLOCKING=1` (`/tmp/pplx_direct_full_waitfix_lblock_olen2.log`). Enabling both CUDA peer access and default-mempool peer access fixed the illegal address: `/tmp/pplx_direct_full_mempool_lblock_olen2.log` completed 2 tokens, and `/tmp/pplx_direct_full_mempool_olen8.log` completed 8 tokens with steady avg **141.30 ms**, p50 **143.99 ms**, p95/max **144.04 ms**. The required long gate `/tmp/pplx_direct_full_mempool_olen64.log` did not emit request metrics before the process ended, so the gate failed. The hot path is now hard-disabled with `USE_SINGLE_NODE_DIRECT_COMBINE=false` while the compiled prototype remains dormant. Restored H200 release build passed, and `/tmp/pplx_direct_false_mempool_olen64.log` generated all 64 tokens with first decode **178.04 ms**, steady avg **146.13 ms**, p50 **144.00 ms**, p95 **160.00 ms**, max **164.02 ms**, samples **62**; the process then hit the known teardown-time NCCL abort with status **134** after metrics were printed.
- **Mechanism lesson**: direct peer pointer addressability is solvable, and the first protocol deadlock was specifically an early overwrite of the second-half sync slots before lagging peers consumed the previous value. But replacing only `combine_send + combine_recv` inside the legacy worker step does not prove a p50 win: short full-direct p50 stayed **143.99 ms**, and the long run produced no gate metric. The next version needs a distinct single-node worker mode that removes the legacy combine stage/barrier from the state machine, instead of dropping a GPU-only data path behind the same worker cadence.

Single-node direct worker mode experiment:
- **Target metric**: H200 EP8 `output_len=64` serving completes all 64 tokens; steady TPOT p50 improves by at least **20 ms** versus **144.00 ms**, p95 stays <= **165 ms**. Worker-wait NVTX should show `worker_wait_combine_recv_done` p50 below **0.5 ms/layer** without simply moving the same wait into another range.
- **Change**: added an explicit `single_node_direct_combine_enabled` mode on `WorkerState`, exposed through `AllToAllContext` and `EpBackend`. `moe_pplx.rs` sets the mode before `dispatch_send` when the direct-combine branch is active. In that mode the worker keeps route/dispatch processing but skips the dispatch and combine fabric barriers, waits for the direct kernel's `combine_send_done`/`combine_recv_done`, then releases `tx_ready` directly. This isolates the single-node direct path from the legacy barrier cadence without changing the legacy combine path.
- **Result**: failed and disabled. Local `cargo fmt -p pegainfer-comm -p pegainfer-deepseek-v4` passed and local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed. H200 release build passed. `/tmp/pplx_direct_worker_mode_olen8.log` completed with status 0, steady avg **145.30 ms**, p50 **144.01 ms**, p95/max **151.61 ms**. `/tmp/pplx_direct_worker_mode_olen64.log` generated all 64 tokens, first decode **239.77 ms**, steady avg **147.61 ms**, p50 **144.00 ms**, p95 **164.00 ms**, max **176.21 ms**, samples **62**, then hit known teardown status **134** after metrics. The p50 gate failed, so `USE_SINGLE_NODE_DIRECT_COMBINE` is back to false.
- **Profile result**: `/tmp/pplx_direct_worker_mode_nvtx_olen16.{log,sqlite,nsys-rep}` confirms the new mode really skipped the hot-path barriers: `barrier` ranges dropped from **43344** in `/tmp/pplx_worker_wait_nvtx_olen64.sqlite` to **16** in the direct-worker profile. But the wait moved, not disappeared. Baseline worker p50s were `worker_wait_combine_send_done` **0.003 ms** and `worker_wait_combine_recv_done` **1.111 ms**; direct-worker mode changed them to **0.970 ms** and **0.224 ms** respectively, while `p2p_all_to_all` p50 stayed **1.669 ms** vs baseline **1.609 ms**. This means the worker now waits earlier for grouped-GEMM/direct-kernel readiness rather than later for combine-recv completion.
- **Restore validation**: after disabling direct again, H200 release build passed and `/tmp/pplx_direct_mode_restored_olen64.log` generated all 64 tokens with first decode **199.80 ms**, steady avg **144.84 ms**, p50 **144.00 ms**, p95 **156.01 ms**, max **168.22 ms**, samples **62**. It then hit the known teardown segfault after metrics.
- **Mechanism lesson**: removing legacy barriers around direct-combine is insufficient because the per-layer p50 budget is not only barrier/worker completion overhead. In the direct path, the worker reaches combine stage before local expert output is ready and waits for `combine_send_done`, so the same per-layer lane time remains visible. The next p50 attempt should stop treating `p2p_all_to_all` duration as pure communication overhead and instead correlate `worker_wait_combine_send_done` with grouped FP4/local MoE compute and stream/event handoff. A correct optimization now needs either a cheaper local expert path / fewer grouped rows, or a schedule where the worker is not the serialized lane owner for local expert readiness.

Single-node direct worker early-release experiment:
- **Target metric**: H200 EP8 `output_len=64` serving completes all 64 tokens; steady TPOT p50 improves by at least **20 ms** versus **144.00 ms**, p95 stays <= **165 ms**. Worker-wait NVTX should show `p2p_all_to_all` p50 <= **1.1 ms/layer** and should not move the same wait into another named worker range.
- **Hypothesis**: if direct-worker mode failed because the worker still waited for local expert output readiness (`worker_wait_combine_send_done` p50 **0.970 ms/layer**), then in the direct-combine path the worker can release `tx_ready` immediately after `dispatch_recv_done` and return to the next step. The direct combine kernel remains ordered on `moe_stream` before the next layer's `dispatch_send`, and it owns `sync_counter`/`sync_ptrs` completion without the worker spinning on expert readiness.
- **Ceiling estimate**: direct-worker-mode `p2p_all_to_all` p50 was **1.669 ms/layer** and `worker_wait_combine_send_done` p50 was **0.970 ms/layer**. Removing that worker wait has a theoretical ceiling of **~41.7 ms/token** across 43 MoE layers; even half of that passes the **20 ms** p50 gate.
- **Keep/revert criterion**: keep only if local/H200 builds pass, H200 `output_len=8` smoke generates all tokens, H200 `output_len=64` generates all tokens with p50 <= **124 ms** and p95 <= **165 ms**, and a follow-up worker-wait profile confirms `p2p_all_to_all` p50 <= **1.1 ms/layer**. Revert on hang, CUDA illegal address, stale sync/counter behavior, p50 regression, p95 regression, or teardown/stop deadlock before metrics.
- **Result**: reverted. The change set `USE_SINGLE_NODE_DIRECT_COMBINE=true` and made direct mode release `tx_ready` immediately after `dispatch_recv_done`, before returning from `WorkerState::step()`. Local `cargo fmt -p pegainfer-comm -p pegainfer-deepseek-v4` passed; local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed; H200 release build passed. H200 short smoke `/tmp/pplx_direct_worker_early_release_olen8.log` generated 8 tokens with steady avg **146.73 ms**, p50 **144.00 ms**, p95/max **164.07 ms**, then hit the known teardown segfault after metrics. The full gate `/tmp/pplx_direct_worker_early_release_olen64.log` timed out with status **124** before any forward metric. The experiment was removed and `USE_SINGLE_NODE_DIRECT_COMBINE=false` restored; H200 release build passed and `/tmp/pplx_after_early_release_revert_olen8.log` generated 8 tokens with steady avg **144.30 ms**, p50 **144.03 ms**, p95/max **151.98 ms**.
- **Mechanism lesson**: direct-combine completion cannot be detached from the worker simply by releasing `tx_ready` after dispatch. Short runs can survive, but longer decode eventually wedges, which means the worker still owns part of the per-layer lifetime beyond send-buffer reuse. The next state-machine change needs an explicit completion acknowledgment or a persistent GPU progress design; early host release alone is not a safe scheduling model.

PPLX routed-MoE ceiling experiment:
- **Target metric**: diagnostic only. Hard-code the pplx MoE path to return shared-expert output after local route/shared, bypassing `dispatch_send`, `dispatch_recv`, grouped FP4 local experts, `combine_send`, and `combine_recv`. H200 `output_len=64` should generate metrics; steady p50 tells the lower bound after removing the entire PPLX routed path. The output is intentionally not a correctness signal.
- **Hypothesis**: if the 144 ms p50 is dominated by PPLX routed-MoE composition, this fake shared-only run should drop near the NCCL 60 ms class. If it stays far above 100 ms, the next bottleneck is outside routed MoE and PPLX state-machine work has a lower ceiling.
- **Ceiling estimate**: current PPLX p50 **144 ms** vs NCCL p50 **63 ms** leaves **~81 ms/token**. Removing all routed-MoE PPLX work is the maximum possible PPLX-side win; any real implementation has a lower ceiling.
- **Keep/revert criterion**: never keep the code path. Record the number, then restore the real routed-MoE path and verify local build health.
- **Result**: reverted. The first remote run accidentally synced `moe_pplx.rs` to the repository root and reproduced the baseline p50 **144.00 ms**; the real source was then synced to `pegainfer-deepseek-v4/src/runtime/moe_pplx.rs` and grep verified `PPLX_SHARED_ONLY_CEILING=true`. H200 `/tmp/pplx_shared_only_ceiling_real_olen64.log` generated all 64 tokens, then hit the known teardown segfault after metrics. Metrics: first decode **30.31 ms**, steady TPOT avg **21.69 ms**, p50 **21.84 ms**, p95 **24.27 ms**, max **25.74 ms**, samples **62**. The output is intentionally invalid, but the performance bound is decisive: removing routed MoE/PPLX work drops far below the NCCL p50 target, so the remaining 144 ms p50 is overwhelmingly in routed-MoE/PPLX composition. The code was restored locally and remotely; local release check passed, remote grep confirmed `PPLX_SHARED_ONLY_CEILING` is absent, remote release build passed, and `/tmp/pplx_restored_after_shared_ceiling_olen8.log` returned to the real path with steady p50 **144.03 ms**. The useful optimization direction is a real single-node routed path that avoids the current four PPLX cooperative kernels plus worker state-machine cadence, not attention/sampling/logits work.

Single-node peer-memory routed path groundwork:
- **Target metric**: behavior-preserving setup change only. Existing PPLX `output_len=8` smoke should still generate all tokens and stay in the real-path p50 **144 ms** class. This patch does not claim a TPOT win.
- **Change**: `EnablePplx` now returns a `PplxPeerScratchPtrs` bundle instead of only `expert_out`. Each rank installs peer pointer tables for `expert_out`, `expanded_input`, `recv_tokens_per_expert`, `expert_indptr`, and the EP backend's full `num_routed` table into `MoePplxScratch`. Existing direct-combine keeps consuming `peer_expert_out_ptrs`; the new pointer tables are dormant until a direct-dispatch kernel lands.
- **Why**: a correct single-node direct dispatch should run on sender ranks, read local `input + route_indices`, and write directly into peer `expanded_input` plus peer per-expert counters. That requires persistent peer destination pointers; trying to ask for peer input pointers per layer would fight the rank-worker ownership model.
- **Validation**: local `cargo fmt -p pegainfer-comm -p pegainfer-deepseek-v4`, local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving`, and `git diff --check` passed. H200 release build passed. H200 `/tmp/pplx_peer_ptr_tables_olen8.log` generated all 8 tokens on the real path: first decode **255.71 ms**, steady avg **143.96 ms**, p50 **144.01 ms**, p95/max **144.02 ms**, then hit the known teardown segfault after metrics. After adding the `num_routed` table, H200 `/tmp/pplx_peer_num_routed_tables_olen8.log` generated all 8 tokens with first decode **219.85 ms**, steady avg **142.31 ms**, p50 **144.00 ms**, p95/max **146.10 ms**, then hit the same known teardown segfault after metrics. This validates the peer pointer table expansion as behavior-preserving groundwork.

Single-node peer-memory direct routed experiment:
- **Target metric**: H200 `output_len=64` must generate all 64 tokens and beat the pre-written gate: steady p50 <= **124 ms** and p95 <= **165 ms**. This is a correctness-path experiment, unlike the fake shared-only ceiling run.
- **Change**: added `a2a_direct_dispatch` to `pegainfer-comm-a2a-kernels`, exposed it through `AllToAllContext::direct_dispatch` and `EpBackend::direct_dispatch`, and hard-coded `USE_SINGLE_NODE_DIRECT_ROUTED=true` in `moe_pplx.rs`. The kernel runs on sender ranks, counts local routes, writes each source row into every peer's `num_routed` table, builds the destination rank's `recv_tokens_per_expert` and padded `expert_indptr`, writes routed BF16 activations directly into peer `expanded_input`, and advances the existing `sync_counter/sync_ptrs` protocol so the existing direct-combine kernel can read peer `expert_out` by the same `base + source-prefix + token_offset` formula.
- **Validation**: local `cargo fmt -p pegainfer-comm -p pegainfer-deepseek-v4`, local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving`, and `git diff --check` passed. H200 release build passed. H200 `/tmp/pplx_direct_routed_olen8.log` generated all 8 tokens: first decode **140.19 ms**, steady avg **87.89 ms**, p50 **89.32 ms**, p95/max **92.17 ms**, then hit the known teardown segfault after metrics. H200 `/tmp/pplx_direct_routed_olen64.log` generated all 64 tokens: first decode **152.45 ms**, steady avg **86.05 ms**, p50 **83.94 ms**, p95 **94.12 ms**, p99 **103.54 ms**, max **107.80 ms**, then hit the same teardown segfault after metrics.
- **Result**: keep for the next profiling pass. The p50 gate passed by **~60 ms/token** versus the old **144.00 ms** PPLX p50, and p95 moved from the previous **~160 ms** class to **94.12 ms**. The remaining gap to NCCL p50 **~63 ms** is now about **21 ms/token**; the next evidence should come from a direct-path CUDA+NVTX profile, not the old worker-wait profile.

Direct routed follow-up tightening:
- **One-block direct kernels**: direct-only `a2a_direct_dispatch` and `a2a_direct_combine_recv` no longer need legacy cooperative-kernel fanout. Launching them with one block passed local/H200 build validation and H200 `/tmp/pplx_direct_oneblock_olen64.log` generated all 64 tokens with first decode **135.52 ms**, steady avg **84.14 ms**, p50 **83.99 ms**, p95 **92.21 ms**, p99 **93.05 ms**, max **93.10 ms**. p50 did not move versus **83.94 ms**, but p95/max improved, so block count is not the p50 owner while direct-kernel protocol/wait residency remains relevant for tail.
- **Grouped rows clamp on direct path**: direct routed no longer needs `expanded_input.seq_capacity()` as the grouped FP4 host row bound. Using `local_experts * expert_padding = 512` rows for bs=1 direct routed reduced wasted grouped rows without D2H dynamic-row reads. H200 `/tmp/pplx_direct_rows512_olen64.log` generated all 64 tokens with first decode **131.85 ms**, steady avg **82.00 ms**, p50 **78.68 ms**, p95 **91.60 ms**, p99 **94.08 ms**, max **96.11 ms**. Repeat `/tmp/pplx_direct_rows512_olen64_r2.log` generated all 64 tokens with first decode **133.28 ms**, steady avg **81.71 ms**, p50 **77.33 ms**, p95 **91.49 ms**, p99 **96.45 ms**, max **102.35 ms**. Both runs hit the known teardown segfault after metrics.
- **Interpretation**: rows512 is a real **5-7 ms/token** p50 win over the first direct routed path, but it only closes part of the gap. The new direct path is now p50 **77-79 ms** versus NCCL **~63 ms**; the next profile needs to explain the remaining **14-16 ms/token** before another code change.

Rows512 direct-path CUDA+NVTX profile:
- **Command shape**: full short profile, not delay/NVTX capture: `/usr/local/cuda-12.9/bin/nsys profile --trace=cuda,nvtx --sample=none --cuda-event-trace=false --cuda-flush-interval=100 --force-overwrite=true --stats=false -o /tmp/pplx_direct_rows512_full_olen8 ... --output-len 8`. H200 `/tmp/pplx_direct_rows512_full_olen8.log` generated all 8 tokens, then hit the known teardown segfault after metrics. The profile exported to `/tmp/pplx_direct_rows512_full_olen8.sqlite`.
- **Profile quality note**: CUPTI kernel rows survived for devices **1/2/5/6** only (`kernel_rows=61525`) because of teardown-time process death, so this profile is good for per-kernel distribution and step/range shape, not for 8-GPU total accounting.
- **Kernel distribution**: `a2a_direct_dispatch_kernel` p50 **0.2605 ms**, p95 **0.3276 ms**; `a2a_direct_combine_recv_kernel` p50 **0.2136 ms**, p95 **1.0668 ms**. Across 43 MoE layers, the p50 direct-protocol cost alone is about **20 ms/token**. Grouped FP4 remains substantial but expected: W13 p50 **0.5656 ms**, W2 p50 **0.2834 ms**, or about **36 ms/token** across 43 layers.
- **Step/rank shape**: ignoring first decode cold step, non-rank0 lanes are **40.6-61.6 ms** while rank0 request waits are **76.0-99.3 ms**. Rank0 `logits_dtoh` is **27.7-35.3 ms** and tracks queued work rather than copy bandwidth. This means the remaining PPLX-vs-NCCL gap is still upstream of the final host copy.
- **Code check**: `a2a_direct_dispatch_kernel` still has two all-peer sync phases (`counter+1` for count visibility and `counter+2` for payload visibility), and `a2a_direct_combine_recv_kernel` waits all peers before publishing readiness, publishes readiness to all peers, then waits all peers again before combining. The actual data path later uses per-route `source_rank`, but the synchronization set is still all-peer. This explains why bs=1/topk6 keeps paying fixed per-layer sync even when only a subset of EP ranks owns the selected experts.

Direct active-peer sync attempts:
- **Active combine only**: `a2a_direct_combine_recv_kernel` computed active source/consumer masks and narrowed only combine-side waits. Local/H200 builds passed; H200 `/tmp/pplx_direct_active_combine_olen8.log` generated 8 tokens with p50 **78.90 ms**, p95 **85.32 ms**. H200 `/tmp/pplx_direct_active_combine_olen64.log` generated 64 tokens with p50 **78.22 ms**, p95 **89.66 ms**. This improved tail but did not move p50 versus rows512 enough to pass the **<=70 ms** gate, so it was reverted.
- **Active dispatch+combine using legacy half-slots**: dispatch final payload-ready and combine expert-ready were both narrowed, while still reusing the legacy `sync_ptrs` half-slot protocol. Local/H200 builds passed, but H200 `/tmp/pplx_direct_active_sync_olen8.log` timed out with status **124** before metrics. The wait sets can diverge when direct code overwrites slots that legacy phases still use. The code was reverted.
- **Independent direct flag slots**: sync buffer grew from `2 * world_size` to `5 * world_size`; direct count-ready, payload-ready, and expert-ready used separate slots with monotonic generation checks. Local/H200 builds passed. H200 `/tmp/pplx_direct_independent_flags_olen8.log` generated 8 tokens with p50 **75.06 ms**, p95 **82.67 ms**; `/tmp/pplx_direct_independent_flags_olen64.log` generated 64 tokens with p50 **76.13 ms**, p95 **87.79 ms**. This is a real tail improvement and a small p50 improvement, but it still missed the **<=70 ms** gate, so it was reverted.
- **Interpretation**: active-peer wait narrowing alone is not enough. The direct kernels still do full count broadcast and repeated metadata reconstruction. The next p50 experiment should reduce direct_dispatch/direct_combine metadata work, not just the flag wait set.

GPU-only compact grouped attempt:
- **Change**: added compact scratch buffers and two CUDA wrappers to compact padded `expanded_input` into an unpadded layout, run grouped FP4 with host rows `world_size * num_tokens * topk` (**48** for bs=1/EP8/topk6), then scatter `compact_out` back to padded `expert_out` so direct combine could keep its address formula.
- **Validation**: local `cargo fmt`, local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving`, `git diff --check`, H200 release build, and H200 `/tmp/pplx_compact_grouped_olen8.log` all passed. The smoke generated 8/8 tokens with p50 **86.51 ms**.
- **Gate result**: H200 `/tmp/pplx_compact_grouped_olen64.log` generated all 64 tokens but steady p50 regressed to **84.00 ms**, p95 **97.38 ms**, max **104.21 ms**. This missed the **<=70 ms** p50 / **<=92 ms** p95 gate and was reverted. The result says the extra compact/scatter launches and fixed API/scheduling cost exceed the benefit of reducing grouped rows from 512 to 48 in this shape.

Direct combine on compute stream attempt:
- **Change**: left direct dispatch on `moe_stream` so it could still overlap with shared expert, but moved `direct_combine_recv` to `ctx.stream`, removing the direct-path `expert_handoff` and `combine_handoff` event pair.
- **Validation**: local `cargo fmt`, local `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving`, `git diff --check`, and H200 release build passed.
- **Gate result**: H200 `/tmp/pplx_direct_combine_ctx_stream_olen64.log` generated all 64 tokens with p50 **77.11 ms**, p95 **90.41 ms**, p99 **92.13 ms**, max **103.39 ms**. This is within rows512 noise and missed the prewritten **<=74 ms** p50 gate, so it was reverted. The result says the two event handoffs after grouped FP4 are not a large p50 owner by themselves.

Rows512 direct clean PPLX vs NCCL profile:
- **PPLX command/profile**: `/usr/local/cuda-12.9/bin/nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-event-trace=false --cuda-flush-interval=100 --force-overwrite=true --stats=false -o /tmp/pplx_rows512_narrow_olen16 env PEGAINFER_DSV4_PPLX=1 NCCL_NVLS_ENABLE=0 ./target/release/bench_serving --model-path /data/models/DeepSeek-V4-Flash-mp8 request --prompt-len 1 --output-len 16 --warmup 0 --iters 1`. Artifacts: `/tmp/pplx_rows512_narrow_olen16.log`, `/tmp/pplx_rows512_narrow_olen16.nsys-rep`, `/tmp/pplx_rows512_narrow_olen16.sqlite`. It generated 16/16 tokens with steady avg **80.76 ms**, p50 **79.08 ms**, p95 **86.74 ms**, max **91.02 ms**. Kernel rows survived on all 8 devices: **38,871** per GPU overall and **34,146** per GPU in the steady window.
- **NCCL comparison**: same command without `PEGAINFER_DSV4_PPLX=1`, artifacts `/tmp/nccl_clean_compare_olen16.log`, `/tmp/nccl_clean_compare_olen16.nsys-rep`, `/tmp/nccl_clean_compare_olen16.sqlite`. It generated 16/16 tokens and hit the known teardown segfault after metrics, with steady avg **63.84 ms**, p50 **63.17 ms**, p95 **66.01 ms**, max **69.82 ms**. The NCCL sqlite lost some per-device kernel rows during teardown, so use its runtime/NVTX rank data for comparison rather than full 8-GPU kernel accounting.
- **Rank-lane accounting**: PPLX rank0-like decode p50 **78.334 ms** versus NCCL rank0-like p50 **62.857 ms**, gap **15.477 ms**. On the same rank0-like lane, PPLX launch API p50 **36.307 ms** versus NCCL **27.521 ms** (gap **8.786 ms**), and PPLX final D2H/drain p50 **32.651 ms** versus NCCL **25.964 ms** (gap **6.687 ms**). These two gaps sum to **15.473 ms**, matching the rank0 decode p50 gap. Non-rank0 PPLX decode p50 median is **43.656 ms**; NCCL non-rank0 p50 median is **36.495 ms**.
- **PPLX steady runtime API**: `cudaLaunchKernel_v7000` dominates by total time with **239,568** calls totaling **3913.162 ms**; `cuMemcpyDtoHAsync_v2` has **14** calls totaling **469.559 ms** and is the final queue drain; `cuLaunchKernelEx` totals **76.816 ms** and `cudaLaunchCooperativeKernel_v9000` totals **59.689 ms**. Event waits/records are small at this profile granularity compared with launch and drain.
- **Launch owners**: PPLX launch API time is mostly in HC / GEMV / TileLang / grouped wrappers, not the direct kernels. Top correlated launch totals include `deepseek_hc_bf16_to_f32_kernel` **574.6 ms**, cuBLAS `gemvx::kernel...` **444.3 ms**, `deepseek_hc_scale_mixes_block_kernel` **343.7 ms**, `deepseek_tilelang_fp8_gemm_n4096_k1024_kernel` **272.9 ms**, and `deepseek_hc_pre_norm_from_mixes_kernel` **256.0 ms**. Direct dispatch launch total is **37.0 ms** and direct combine launch total is **22.6 ms** across all rank threads in the steady window.
- **Direct kernel bodies**: `a2a_direct_dispatch_kernel` appears **4816** times in the steady window with p50 **0.2611 ms**, p95 **0.3318 ms**, max **6.9175 ms**. `a2a_direct_combine_recv_kernel` appears **4816** times with p50 **0.2119 ms**, p95 **1.0630 ms**, max **1.0888 ms**. These kernels still contribute queued GPU work, but their launch API and isolated wait-set rewrites no longer explain the whole **14-16 ms/token** gap.
- **Interpretation**: rows512 PPLX vs NCCL p50 gap is accounted for by extra launch/API pressure plus a deeper final logits drain. This points to large decode execution units that reduce launch fanout/queue depth, not another isolated direct flag, stream-handoff, or compact/scatter patch.

Current HC seq_len=1 direct mixes experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, with p95 staying near the retained **~91 ms** class and all tokens generated.
- **Hypothesis**: if a meaningful part of the remaining rows512 gap is launch/API fanout in HC mix helpers, then replacing `deepseek_hc_mixes_cuda`'s `bf16_to_f32 -> cuBLAS Sgemv -> scale_mixes_block` sequence with the existing single-launch `deepseek_hc_mixes_kernel` for `seq_len == 1` and no raw/rms side outputs will reduce `cudaLaunchKernel` and cuBLAS launch pressure enough to move p50. This re-opens the earlier temporary HC mix bypass only because the direct routed path removed the old PPLX worker floor and the clean rows512 profile now names HC/GEMV launch API as a top owner.
- **Ceiling estimate**: clean rows512 PPLX profile shows launch API totals dominated by HC/GEMV classes: `deepseek_hc_bf16_to_f32_kernel` **574.6 ms**, cuBLAS `gemvx::kernel...` **444.3 ms**, and `deepseek_hc_scale_mixes_block_kernel` **343.7 ms** across all rank threads in the steady window. The direct kernel may do more arithmetic work per mix than cuBLAS, so the only keepable outcome is wall-clock p50 movement, not just fewer launch rows.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, H200 `output_len=64` PPLX rows512 generates all 64 tokens, p50 improves by **>=5 ms/token**, and p95 does not regress beyond **~95 ms**. Revert on correctness-looking failure, CUDA error, build failure, p50 movement under the gate, or p95 regression.
- **Result**: reverted. Local release check and H200 release build passed. H200 `/tmp/pplx_hc_direct_mixes_olen64.log` generated 64/64 tokens but measured first decode **149.61 ms**, steady avg **83.83 ms**, p50 **81.35 ms**, p95 **93.58 ms**, p99 **96.96 ms**, max **104.50 ms**; teardown hit the known NCCL abort after metrics. This misses the **>=5 ms/token** p50 improvement gate and regresses versus rows512 p50 **77-79 ms**. Lesson: launch-count reduction is not enough when the replacement kernel gives up cuBLAS efficiency; HC/GEMV launch API is a real profile owner, but the replacement has to preserve or improve the GPU work, not merely collapse launches.

Current direct-MoE single-stream experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, with p95 staying <= **92 ms** and all 64 tokens generated.
- **Hypothesis**: if part of the rows512 gap is explicit stream handoff and queue fragmentation inside the direct routed MoE path, then running the whole direct branch on `ctx.stream` should remove `route_handoff`, `direct_dispatch_handoff`, `expert_handoff`, and `combine_handoff` from that branch. Direct dispatch is small enough that losing its overlap with shared expert should be cheaper than the removed cross-stream boundaries.
- **Ceiling estimate**: the isolated direct-combine-on-`ctx.stream` attempt measured p50 **77.11 ms**, within rows512 noise, so the full single-stream experiment must remove more than the final two handoffs to be keepable. Expected ceiling is modest: **5-8 ms/token** only if the earlier handoffs and queue ordering are a real p50 owner.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, H200 `output_len=64` PPLX rows512 generates all 64 tokens, p50 improves by **>=5 ms/token**, and p95 stays <= **92 ms**. Revert on hang, CUDA error, correctness-looking output failure, p50 movement under gate, or p95 regression.
- **Result**: reverted. Local release check and H200 release build passed, but H200 `/tmp/pplx_direct_single_stream_olen64.log` timed out with status **124** before request metrics. The change moved direct dispatch/combine and their peer flag protocol onto `ctx.stream`, and that invalidated the current direct path's stream/progress assumptions. Lesson: the direct branch's cross-stream events are not just overhead; they are part of the completion ordering contract between route, peer writes, grouped expert output, and direct combine. Re-open only with a redesigned direct completion protocol, not by moving the whole branch to one stream.

Current rank0-only logits gather experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, and NCCL baseline must not regress. Because this changes shared final-logits code, compare both PPLX and NCCL after the patch.
- **Hypothesis**: final logits currently uses NCCL `all_gather`, so every rank receives the full vocab even though only rank0 returns logits for sampling. Replacing this with rank0-only NCCL P2P gather (`rank0 recv` each shard, non-root ranks `send`) should reduce final-logits communication/queue drain and may shrink the rank0-like p50 gap attributed to final D2H/drain.
- **Ceiling estimate**: clean rows512 profile says PPLX final D2H/drain p50 is **32.651 ms** versus NCCL **25.964 ms**, and the overall rank0-like p50 gap contains **6.687 ms** of extra final drain. The absolute win could be larger if all-gather itself is bloating the queue on every rank, but keep/revert is still based on measured p50.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, PPLX `output_len=64` generates all 64 tokens with p50 improved by **>=5 ms/token** and p95 <= **92 ms**, and NCCL `output_len=16` or longer does not regress versus the clean **~63 ms** p50 class. Revert on deadlock/timeout, CUDA/NCCL error, wrong-looking output, or insufficient p50 movement.
- **Result**: reverted. Local release check and H200 release build passed. H200 short smoke `/tmp/pplx_rank0_logits_gather_olen8.log` generated 8/8 tokens but had first decode **412.05 ms** and steady p50 **88.36 ms**. Full gate `/tmp/pplx_rank0_logits_gather_olen64.log` generated 64/64 tokens with first decode **309.34 ms**, steady avg **79.00 ms**, p50 **76.02 ms**, p95 **88.56 ms**, p99 **89.06 ms**, max **97.18 ms**; teardown hit the known segfault after metrics. This is a small p50/tail win but misses the prewritten **>=5 ms/token** p50 gate and heavily regresses first decode, so the code was removed. Lesson: rank0-only P2P gather can reduce steady tail slightly, but NCCL P2P setup/queueing cost makes it unsuitable as a standalone retained change.

Current final-logits local CUDA Graph experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, p95 should stay <= **92 ms**, and all 64 tokens must be generated. Because final logits is shared by PPLX and NCCL paths, NCCL `output_len=16` or longer must remain in the clean **~63 ms** p50 class.
- **Hypothesis**: final logits has stable bs=1 pointers and a fixed kernel/cuBLAS sequence (`hc_head -> rms_norm -> local_logits`) before the existing logits all-gather. Capturing only this local rank subgraph should remove repeat launch/API fanout on replay without touching PPLX peer flags, direct MoE synchronization, or NCCL logits all-gather semantics.
- **Ceiling estimate**: clean rows512 profile attributes the full PPLX-vs-NCCL rank0-like p50 gap to launch API **(+8.786 ms)** plus final drain **(+6.687 ms)**. This graph only covers the local final logits sequence, so a plausible win is single-digit ms/token; it is keepable only if the measured p50 movement clears **>=5 ms/token**.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, graph capture does not fail, PPLX `output_len=64` clears the p50/p95 gate above, and NCCL baseline does not regress. Revert on graph capture failure, CUDA error, correctness-looking output failure, insufficient p50 movement, or p95 regression.
- **Result**: reverted. Local release check and H200 release build passed. H200 smoke `/tmp/pplx_final_logits_graph_olen8.log` generated 8/8 tokens with p50 **76.79 ms**. Full gate `/tmp/pplx_final_logits_graph_olen64.log` generated 64/64 tokens with first decode **133.26 ms**, steady avg **80.20 ms**, p50 **77.21 ms**, p95 **89.77 ms**, p99 **91.32 ms**, max **91.99 ms**; teardown hit the known segfault after metrics. The p95 is healthy, but p50 is still within rows512 baseline noise and misses the **>=5 ms/token** gate. Lesson: a graph covering only local final logits is too narrow; any future graph work has to cover a larger static block or remove queue depth before final logits.

Current direct single-block non-cooperative launch experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, p95 should stay <= **92 ms**, and all 64 tokens must be generated. Because this changes only PPLX direct kernels, NCCL baseline is expected to be unchanged and only needs rebuild-level validation unless shared code moves.
- **Hypothesis**: the retained direct path already launches `a2a_direct_dispatch` and `a2a_direct_combine_recv` with `num_blocks=1`, but both wrappers still use `cudaLaunchCooperativeKernel` and the kernels still call `cooperative_groups::this_grid().sync()`. Replacing the direct-only wrappers with normal CUDA launches and block-local `__syncthreads()` should preserve the one-block protocol while removing cooperative-launch driver overhead and one source of launch-path contention.
- **Ceiling estimate**: clean rows512 rank0 steady profile shows direct dispatch launch API **15.294 ms**, direct combine launch API **2.435 ms**, direct dispatch kernel body **158.331 ms**, and direct combine kernel body **320.282 ms** across the 14 steady rank0 steps. The change does not remove the kernel body wait loops, so expected p50 gain is only keepable if cooperative-launch removal also reduces the cross-kernel launch stalls seen in other ops.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, PPLX `output_len=64` generates all 64 tokens with p50 improved by **>=5 ms/token** and p95 <= **92 ms**. Revert on build failure, CUDA launch error, hang/timeout, wrong-looking output, p50 movement under the gate, or p95 regression.
- **Result**: reverted. Local release check and H200 release build passed after fixing the normal-launch `OutTy*` pointer type. H200 smoke `/tmp/pplx_direct_noncoop_olen8.log` generated 8/8 tokens but measured p50 **78.16 ms** and p95/max **100.79 ms**. Full gate `/tmp/pplx_direct_noncoop_olen64.log` generated 64/64 tokens with first decode **147.78 ms**, steady avg **80.19 ms**, p50 **76.60 ms**, p95 **93.11 ms**, p99 **94.78 ms**, max **100.48 ms**; teardown hit the known segfault after metrics. This misses the **>=5 ms/token** p50 gate and regresses p95 beyond **92 ms**, so the cooperative launch path was restored. Lesson: cooperative launch overhead is visible, but direct-kernel body/wait ordering remains dominant; replacing the launch mechanism alone is not enough.

Current direct route-position reuse experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, p95 should stay <= **92 ms**, and all 64 tokens must be generated. NCCL code path is untouched; rebuild validation is enough unless shared code changes.
- **Hypothesis**: `a2a_direct_dispatch_kernel` already computes the exact `(source_rank, padded position)` for each local token route when writing peer `expanded_input`. `a2a_direct_combine_recv_kernel` recomputes the same base/prefix position from `num_routed` and `token_offset` before reading peer `expert_out`. Persisting dispatch's per-route `position/source_rank` in direct workspace and feeding it to direct combine removes the duplicated metadata pass and shared-memory position staging.
- **Ceiling estimate**: clean rows512 rank0 steady profile shows direct combine kernel body **320.282 ms** and direct dispatch kernel body **158.331 ms** across 14 rank0 steady steps. This change targets direct combine body work only; expected p50 gain is keepable only if the removed position calculation shifts wall-clock by **>=5 ms/token** or materially reduces final drain.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, PPLX `output_len=64` generates all 64 tokens with p50 improved by **>=5 ms/token** and p95 <= **92 ms**. Revert on build failure, CUDA error, hang/timeout, wrong-looking output, p50 movement under the gate, or p95 regression.
- **Result**: reverted. Local `git diff --check` and `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed; H200 release build passed. H200 smoke `/tmp/pplx_routepos_reuse_olen8.log` generated 8/8 tokens with steady p50 **74.41 ms**. Full gates generated all 64 tokens twice: `/tmp/pplx_routepos_reuse_olen64.log` measured first decode **140.74 ms**, steady avg **76.95 ms**, p50 **74.52 ms**, p95 **87.90 ms**, p99 **89.30 ms**, max **98.87 ms**; `/tmp/pplx_routepos_reuse_olen64_r2.log` measured first decode **129.11 ms**, steady avg **78.23 ms**, p50 **74.54 ms**, p95 **89.30 ms**, p99 **89.74 ms**, max **89.82 ms**. Teardown hit the known status 139 segfault after metrics. The result is a repeatable 2.8-4.6 ms p50 improvement and healthier p95, but it misses the prewritten **>=5 ms/token** p50 gate, so the code was removed and H200 was rebuilt after revert. Lesson: duplicated direct-combine position calculation is real but too small alone; the next retained change has to merge a larger direct-side stage or reduce launch/queue depth.

Current reusable PPLX handoff events experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, p95 should stay <= **92 ms**, and all 64 tokens must be generated. NCCL path is untouched; rebuild validation is enough unless shared code moves.
- **Hypothesis**: `moe_pplx.rs` creates a fresh CUDA event for each explicit stream handoff (`route`, `direct_dispatch`, `indptr`, `expert`, `combine`) via `CudaStream::record_event(Some(DISABLE_TIMING))`. The clean rows512 decode window shows `cuEventCreate` **20640** calls / **48.418 ms** and `cuEventDestroy` **20640** calls / **10.168 ms**, while `cudaEventRecord/cuEventRecord/cuStreamWaitEvent` remain separate. Preallocating the handoff events in `MoePplxScratch` and re-recording them each layer should remove event create/destroy fanout without changing stream ordering.
- **Ceiling estimate**: The measured create/destroy total is about **58.6 ms** over the profile's decode window across all rank threads. The keep gate still requires **>=5 ms/token** p50 because API totals across ranks do not directly translate to request p50; this is only worthwhile if event allocation contributes to the launch/queue tail.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, PPLX `output_len=64` generates all 64 tokens with p50 improved by **>=5 ms/token** and p95 <= **92 ms**. Revert on build failure, CUDA event/stream error, hang/timeout, wrong-looking output, insufficient p50 movement, or p95 regression.
- **Result**: reverted. Local `git diff --check` and `PATH=/usr/local/cuda/bin:$PATH cargo check --release -p pegainfer-server --features deepseek-v4,pplx-ep --bin bench_serving` passed; H200 release build passed. H200 smoke `/tmp/pplx_reuse_events_olen8.log` generated 8/8 tokens with steady p50 **77.55 ms**. Full gate `/tmp/pplx_reuse_events_olen64.log` generated 64/64 tokens with first decode **149.52 ms**, steady avg **80.20 ms**, p50 **77.09 ms**, p95 **90.45 ms**, p99 **91.70 ms**, max **91.84 ms**; teardown hit the known status 139 segfault after metrics. The p50 stayed in the retained rows512 baseline band, so removing event create/destroy alone is not enough. Lesson: the event allocation calls show up in API totals, but the request p50 is governed by queued work and waits that remain after event reuse.

Current direct combine ctx-stream plus route-position experiment:
- **Target metric**: H200 rows512 PPLX `output_len=64` steady p50 should improve by at least **5 ms/token** from the retained **77-79 ms** baseline, p95 should stay <= **92 ms**, and all 64 tokens must be generated. NCCL path is untouched; rebuild validation is enough unless shared runtime code moves.
- **Hypothesis**: Two isolated patches each removed too little: direct-combine-on-`ctx.stream` generated p50 **77.11 ms** / p95 **90.41 ms**, while direct route-position reuse generated p50 **74.52 / 74.54 ms** / p95 **87.90 / 89.30 ms**. Combining them removes both the duplicated combine position pass and the direct combine `ctx.stream -> moe_stream -> ctx.stream` handoff pair, while preserving direct dispatch and grouped FP4 ordering. This is a larger direct-combine execution-unit change than either local patch alone.
- **Ceiling estimate**: route-position reuse alone moved p50 by 2.8-4.6 ms, and direct-combine stream handoff alone was within baseline noise. The combination only deserves to stay if the effects compound enough to clear **>=5 ms/token** p50 and keep p95 <= **92 ms**; otherwise the conclusion is that direct combine needs a still larger merge with grouped/dispatch stages.
- **Keep/revert criterion**: keep only if local/H200 release builds pass, PPLX `output_len=64` generates all 64 tokens with p50 improved by **>=5 ms/token** and p95 <= **92 ms**. Revert on build failure, CUDA event/stream error, hang/timeout, wrong-looking output, insufficient p50 movement, or p95 regression.

## Hypothesis Ledger

| Hypothesis | Status | Evidence | Acceptance criterion |
| --- | --- | --- | --- |
| Sampling / token emit explains 120-170 ms TPOT | killed | NVTX `dsv4.request.sample` stays around 0.2 ms and `emit_token` is microsecond-level in both node profiles. | Would need sample/emit to account for at least 20 ms/token; it does not. |
| `logits_dtoh` is a real D2H bandwidth cost | killed | CUPTI memcpy rows show D2H device copy is about 0.1 ms total across 7 decode copies, while `cuMemcpyDtoHAsync` API blocks 58-64 ms. | Would need memcpy activity duration to match API duration; it does not. Treat it as the final synchronization point. |
| ratio128 compressed decode allocator churn is a TPOT root cause | killed as wall-clock root, fixed as hygiene | Before scratch: decode-window `cuMemAllocAsync=11200`, all attributed to `dsv4.layer.attention_compressed`; after scratch: `cuMemAllocAsync=0`, `cuMemFreeAsync=0`, but node-profile steady TPOT remains 147.31 ms/tok. | Keep scratch if alloc/free become zero and correctness holds; expect no large wall win because measured ceiling was only 10 ms-level spikes. Criterion met. |
| ratio4 boundary path drives token-to-token variance | partially killed as wall root, kept as local cleanup | The first ratio4 refactor hit dead single-token helper; real decode uses batch helper. Batch fused topk reduced `attention_ratio4` p95 **15.746 -> 1.593 ms** and `rank.decode` p50 **114.779 -> 79.695 ms**, but `runtime.wait_rank_decode` p50 remains **143.788 ms**. | Ratio4 topk fusion can stay as structural cleanup. Further ratio4 work needs step-correlated evidence that boundary-step request p50/p95, not just layer-local p95, improves by at least 15 ms. |
| Hadamard+FP4 single-pair fusion closes the ratio4 tail | killed | Fused serial kernel built and completed H200 smoke, but serving p95 regressed **184.00 -> 216.01 ms** while p50 stayed **144.03 ms**. | Re-open only for a larger fused ratio4 boundary operator that removes several launches and shows request-level p50/p95 improvement over >=16 decode steps. |
| Fine-grained local CUDA Graph islands can close the non-communication gap | killed as implemented | H200 `output_len=24` graph-island run completed but steady p50 regressed to **154.78 ms**. Decode-only profile shows 1056 graph instantiates totaling **944.08 ms**, **23232** graph launches totaling **421.25 ms**, launch-class calls/step down only about **8%**, and event wait/record counts essentially unchanged. | Re-open only for a larger static island that removes explicit stream handoff boundaries and graph-launch fanout, with a pre-written gate of at least 15 ms p50 improvement over >=16 decode steps. |
| Rank/a2a worker CPU overlap drives p95/max tail | alive, mitigation kept | H200 clean log had exact CPU overlap on CPU **6**; separating rank worker and pplx a2a worker pools cut `output_len=64` p95 **216.01 -> 159.96/162.86 ms** and max **300.01 -> 164.00/168.01 ms** across two runs, with p50 unchanged at **144.00 ms**. | Keep the pool split. Do not claim p50 progress from it; next p50 work needs a different mechanism. |
| Rank0 TE/fabric worker on CPU0 drives the legacy p50 floor | alive, mitigation kept | `/proc` decode-window sampling before the fix showed CPU0 `tx_engine_domain` pinned correctly but only got **3602 ms** runtime over a **7012 ms** sample and took **2980** nonvoluntary switches, while fabric workers on CPU24/48/72 got ~**7008-7012 ms** runtime and **9-17** nonvoluntary switches. Moving rank0 TE off CPU0 produced two H200 `output_len=64` runs with steady p50 **66.46 / 66.70 ms**, p95 **69.80 / 69.62 ms**, max **71.48 / 71.89 ms**. The corrected `/proc` sample saw CPU10 `tx_engine_domain` runtime delta **3452 ms** and only **4** nonvoluntary switches. Logs: `/tmp/pplx_te_repin_olen64.log`, `/tmp/pplx_te_repin_olen64_r2.log`, `/tmp/pplx_te_repin_olen64_r2_proc_summary.txt`. Cleanup first introduced a per-rank placement plan, then a later review found topology-group role selection could collide with rank workers (`rank0 a2a/TE/UVM` on CPUs already used by rank1/2/3 workers). Current code moves the generic pieces to `pegainfer_core::cpu_topology`: read CUDA device NUMA, current affinity, and NUMA cpulist; split each NUMA pool into contiguous rank slices; reserve CPU0 for the system and CPU1 for scheduler; assign rank/a2a/TE/UVM roles from that rank's own slice; log `cpu_slice/rank_worker/TE/a2a/UVM` per rank at startup. H200 per-NUMA slice validation showed no CPU collision and measured `output_len=64` steady p50 **66.65 ms**, p95 **68.15 ms**, max **69.47 ms** before the known teardown segfault. | Keep CPU0/CPU1 reservation and per-NUMA rank slices. Validate future placement changes with startup logs, TPOT, and `/proc/<tid>/sched` deltas; do not rely on topology-group CPU order alone. |
| Route all-gather alone explains the 144 ms p50 floor | killed | Intra-process route exchange skipped `route_write_op + route_counter.wait` through a process-local barrier and direct peer `num_routed` pointer reads. H200 `output_len=64` completed all tokens but measured p50 **144.00 ms**, p95 **155.95 ms**, matching the CPU-pool p50 baseline. | Do not spend more patches on isolated route exchange. Re-open only as part of a full single-node state-machine replacement that reduces per-layer a2a union p50 by at least 20 ms. |
| Grouped FP4 unused capacity explains the 144 ms p50 floor | killed | bs=1 capacity clamp reduced theoretical pplx grouped row bound from **1376** to **560** rows and completed H200 `output_len=64`, but p50 remained **144.00 ms** with p95 **155.74 ms**. | Do not pursue host-side capacity clamps as a p50 fix. Re-open only with a profile proving grouped FP4 kernels, not a2a/driver synchronization, own at least 15 ms p50. |
| libcuda driver lock/contention explains non-communication operator spikes | alive as tail source, not p50 owner | Decode-only `--cudabacktrace=all:1000` + OSRT profile shows rank-thread `pthread_mutex_lock` total **782.16 ms** and max **108.95 ms**. The largest mutex stacks are `libcuda -> cuModuleGetFunction/cuLibraryGetModule -> cudaLaunchCooperativeKernel -> a2a_dispatch_send`, while non-communication `cudaLaunchKernel` tails show 16-30 ms API time for microsecond-scale kernels inside ratio4/HC ranges. NVTX-only `output_len=64` shows attention-local medians are sub-ms and do not explain the **144 ms** p50 floor. | Keep using API-vs-kernel profiles for tails. Do not spend more p50 experiments on single tiny operator launches unless a step-correlated profile shows at least 15 ms p50 ownership. |
| PPLX full-runtime rank lane owns the 144 ms p50 floor | alive, now tied to local MoE readiness / worker serialization | NVTX-only differential profile: PPLX non-rank0 p50 **74.24-79.96 ms** vs NCCL **36.65 ms**; PPLX rank0 p50 **143.77 ms** vs NCCL **63.13 ms**. Worker-wait profile shows per-layer `p2p_all_to_all` p50 **1.609 ms** and stable `worker_wait_combine_recv_done` p50 **1.111 ms/layer**, while `dispatch` p50 is only **0.010 ms/layer**. Direct worker mode skipped barriers (`barrier` ranges **43344 -> 16**) but moved p50 wait to `worker_wait_combine_send_done` **0.970 ms** and left request p50 **144.00 ms**. Stage NVTX shows normal dispatch/combine API ranges are microseconds; layer-local p50 hot spots are `shared_expert` layer 27/33 and grouped W2 layer 19. | A real fix must reduce PPLX non-rank0 p50 by at least **20 ms** on H200 `output_len>=64`, with rank0/wait-rank p50 following. Before coding, prove by CUDA+NVTX containment whether the layer-local hot spots are GPU kernels, explicit stream handoff, or libcuda launch/API waits. Route, capacity, isolated combine flag, and direct-barrier skips are low priority. |
| Direct-combine requires CUMem-owned `expert_out` | killed with mempool-access caveat | `/tmp/peer_ptr_probe.cu` proved raw `cudaMalloc` peer pointers are readable on all H200 GPU pairs. The real cudarc `CudaSlice` path then hit CUDA 700 until the direct setup enabled CUDA peer access and `cudaMemPoolSetAccess` for each peer on the local default mempool; after that `/tmp/pplx_direct_full_mempool_lblock_olen2.log` completed. | Ordinary `CudaSlice` pointer exchange is usable only with explicit peer + default-mempool access setup. Do not require CUMem allocation solely for pointer addressability. |
| Direct-combine can be spliced into the legacy worker step by only replacing the two combine kernels | killed as implemented | First splice timed out; metadata-only passed only after waiting for first-half sync flags before publishing second-half flags; full direct needed mempool access and then `output_len=8` completed with p50 **143.99 ms**, but `output_len=64` emitted no request metrics and failed the gate. | Do not use the legacy worker step as the direct-combine integration point. |
| Single-node direct worker mode removes the 144 ms p50 floor | killed as implemented | The explicit direct mode skipped hot-path barriers (`barrier` ranges **43344 -> 16**) and moved `worker_wait_combine_recv_done` p50 **1.111 -> 0.224 ms**, but `worker_wait_combine_send_done` p50 rose **0.003 -> 0.970 ms** and `output_len=64` p50 stayed **144.00 ms**. | Re-open only after a step-correlated profile proves local expert compute / stream handoff is no longer being counted as worker wait, or after grouped FP4/local MoE work is reduced by at least 20 ms/token. |
| Direct worker early release removes local expert readiness from the worker lane | killed | Early-release direct mode generated 8 tokens but stayed p50 **144.00 ms**; the required `output_len=64` gate timed out with status **124** before metrics. Restored default path generated 8 tokens with p50 **144.03 ms**. | Do not detach direct-combine from the worker with `tx_ready` release alone. Re-open only with an explicit completion protocol or persistent GPU progress path that proves no sync-counter wedge over `output_len>=64`. |
| cudarc automatic `CudaSlice::device_ptr(_mut)` event tracking explains the full-runtime gap | killed by code review | Local and H200 source both have `RankGpuContext::new()` calling `ctx.disable_event_tracking()` immediately after `CudaContext::new()`. In cudarc 0.19.3, `device_ptr` / `device_ptr_mut` only inject automatic `stream.wait(read/write)` guards when `ctx.is_managing_stream_synchronization()` is true, which requires event tracking to be enabled. The `moe_pplx.rs` `device_ptr` calls therefore do not add the suspected automatic guard events in this runtime. | Do not patch raw-pointer bypasses or context-wide event toggles for this hypothesis. Keep analyzing explicit `record_event`/`wait` handoffs, CUDA launch/API residency, and PPLX worker state-machine cadence. |
| Legacy PPLX explicit stream handoffs own the 144 ms p50 floor | killed, cleanup kept | Hard-coded legacy four-stage PPLX to run `dispatch_send -> dispatch_recv -> combine_send -> combine_recv` on `ctx.stream` and skipped the `ctx.stream <-> moe_stream` route/indptr/expert/combine handoff events. H200 `/tmp/pplx_single_stream_diag_olen64.log` generated 64/64 tokens with first decode **219.86 ms**, steady avg **144.13 ms**, p50 **144.00 ms**, p95 **160.00 ms**, max **164.03 ms**, status **0**. This matches the restored two-stream legacy baseline. The follow-up cleanup removed the diagnostic flag, PPLX `moe_stream` plumbing, peer-pointer direct-path scratch tables from dsv4 runtime, and direct peer-pointer install commands; H200 `/tmp/pplx_single_stream_clean_olen64.log` generated 64/64 with first decode **243.84 ms**, steady avg **143.87 ms**, p50 **144.00 ms**, p95 **155.98 ms**, max **164.03 ms**, then hit the known teardown segfault status **139** after metrics. | Keep the simpler fixed-`ctx.stream` legacy path for maintainability and cleaner profiles. Do not claim p50 progress from it; the p50 floor remains in PPLX four-stage worker/cooperative-kernel cadence or queued model work around it, not the explicit cross-stream events alone. |
| Moving shared expert before `dispatch_send` closes the microbench gap | killed, cleanup kept | Shared expert only depends on `input`, so the fixed single-stream legacy path was reordered to `route -> shared_expert -> dispatch_send -> dispatch_recv -> indptr -> grouped_fp4 -> combine_send -> combine_recv(accumulate=true)`. H200 `/tmp/pplx_shared_before_dispatch_olen64.log` generated 64/64 with first decode **227.96 ms**, steady avg **141.80 ms**, p50 **143.97 ms**, p95 **151.82 ms**, max **155.97 ms**, then hit known teardown status **139** after metrics. This is a small tail/avg movement inside the same 144 ms p50 class, not a bridge to microbench. | Keep the order because it separates local shared-expert work from the routed PPLX four-stage block and makes profiles easier to read. Do not treat it as a p50 optimization. |
| Routed MoE/PPLX composition owns the gap to NCCL | alive, mechanism evidence only | A fake shared-only ceiling run kept PPLX enabled but returned after route+shared expert, bypassing dispatch/grouped/combine. It generated 64 tokens with invalid output and steady p50 **21.84 ms**. The real single-node direct routed path then generated 64 tokens with steady p50 **83.94 ms**, down from the old PPLX p50 **144.00 ms**; rows512 tightened it further to p50 **78.68 / 77.33 ms** across two runs. Clean rows512 PPLX profile measured steady p50 **79.08 ms** versus clean NCCL p50 **63.17 ms**. | Treat direct routed as evidence that the old four-stage cadence was the floor, not as retained production code. Current retained path is legacy four-stage plus rank0 TE CPU placement. |
| Single-node direct routed path bypasses the legacy PPLX p50 floor | killed as retained implementation | `a2a_direct_dispatch` plus existing direct combine bypassed `dispatch_send -> dispatch_recv -> combine_send -> combine_recv`. H200 `/tmp/pplx_direct_routed_olen64.log` generated all 64 tokens with steady avg **86.05 ms**, p50 **83.94 ms**, p95 **94.12 ms**, max **107.80 ms**. One-block direct kernels kept p50 **83.99 ms** but cut p95 to **92.21 ms**. Rows512 generated 64 tokens twice with p50 **78.68 / 77.33 ms** and p95 **91.60 / 91.49 ms**; clean profile measured p50 **79.08 ms**. | Removed from code as a hack that bypassed upstream PPLX semantics. Re-open only as a separately designed single-node EP backend with its own protocol, not as patches inside the legacy backend. |
| Direct-path grouped row overrun costs steady p50 | archived with direct path | The legacy bs=1 capacity clamp did not move p50 while worker cadence dominated. After direct routed removed the worker floor, reducing grouped FP4 host rows from `expanded_input.seq_capacity()` to **512** moved H200 `output_len=64` p50 from **83.94 ms** to **78.68 / 77.33 ms**. | Keep the row-cap lesson for a future designed direct backend. It is not active code after direct-path cleanup. |
| GPU-only compact grouped rows closes the remaining gap | killed | Compacting padded receive rows to **48** grouped rows and scattering back generated all tokens, but H200 `/tmp/pplx_compact_grouped_olen64.log` regressed to p50 **84.00 ms**, p95 **97.38 ms** versus rows512 p50 **77-79 ms**, p95 **~91.5 ms**. | Do not add separate compact/scatter kernels for bs=1. Re-open only if the compacting is fused into an existing direct/grouped kernel or grouped FP4 accepts sparse/padded indptr without extra launches. |
| Direct combine stream handoff owns several ms/token | killed as isolated patch | Moving only `direct_combine_recv` from `moe_stream` to `ctx.stream` generated all 64 tokens with p50 **77.11 ms**, p95 **90.41 ms**. This is not enough to beat rows512 p50 **77.33 ms** by a confident margin and missed the **<=74 ms** gate. | Do not keep isolated stream-handoff rewrites. Re-open only as part of a larger single-stream or persistent direct pipeline with a >=5 ms/token p50 gate. |
| Rows512 gap is direct-kernel wait alone | killed / downsized | Clean rows512 PPLX profile shows direct dispatch p50 **0.2611 ms/layer** and direct combine p50 **0.2119 ms/layer**, but direct dispatch launch total is only **37.0 ms** and direct combine launch total **22.6 ms** across all rank threads in the steady window. Active-peer and independent-flag experiments improved tail and at most ~1 ms p50, not the full **14-16 ms/token** gap. | Treat direct kernels as part of the queued GPU work, not a sole owner. Re-open only if a step-correlated wait metric shows **>=5 ms/token** p50 ownership after launch fanout is reduced. |
| Rows512 gap is launch/queue inflation before logits | alive | Clean PPLX vs NCCL profile: rank0-like decode p50 gap **15.477 ms**; launch API p50 gap **8.786 ms** and final D2H/drain p50 gap **6.687 ms** sum to **15.473 ms**. PPLX `cudaLaunchKernel_v7000` steady total is **3913.162 ms** over **239,568** calls; final `cuMemcpyDtoHAsync_v2` totals **469.559 ms** over **14** calls. Top launch owners are HC/GEMV/TileLang/grouped wrappers, while direct kernel launch totals are small. | Next implementation should remove launch fanout or queued work in a larger decode unit and must target **>=5 ms/token** p50 improvement over rows512 PPLX p50 **77-79 ms**. |
| HC seq_len=1 direct mixes closes launch gap | killed | Replacing `deepseek_hc_mixes_cuda`'s `bf16_to_f32 -> cuBLAS Sgemv -> scale_mixes_block` path with existing `deepseek_hc_mixes_kernel` for bs=1/no-side-output built locally and on H200, but `/tmp/pplx_hc_direct_mixes_olen64.log` measured p50 **81.35 ms**, p95 **93.58 ms** versus rows512 p50 **77-79 ms**. | Do not replace cuBLAS GEMV with the simple direct kernel. Re-open HC only with a kernel that preserves math throughput while reducing launch/API work, or with graph/static-block evidence that the launch side is removed without slower GPU work. |
| Direct-MoE full single-stream removes rows512 gap | killed | Moving the direct routed branch from `moe_stream` to `ctx.stream` compiled locally and on H200, but `/tmp/pplx_direct_single_stream_olen64.log` timed out with status **124** before metrics. Earlier isolated direct-combine-on-`ctx.stream` completed but missed p50 gate; the full move wedges the direct completion protocol. | Do not collapse the direct branch onto one stream by local edit. Re-open only with a redesigned direct synchronization protocol or persistent progress path that preserves peer flag ordering. |
| Rank0-only logits gather closes final drain gap | killed as standalone | Replacing logits all-gather with rank0-only NCCL P2P gather completed and improved PPLX rows512 p50 slightly to **76.02 ms** with p95 **88.56 ms**, but missed the **>=5 ms/token** gate and inflated first decode to **309.34 ms**. | Do not keep standalone P2P logits gather. Re-open only if paired with persistent/graph capture that removes P2P first-use cost and proves NCCL baseline does not regress. |
| Final-logits local CUDA Graph closes the launch/drain gap | killed as too narrow | Local final-logits graph captured and replayed correctly, but `/tmp/pplx_final_logits_graph_olen64.log` measured p50 **77.21 ms**, p95 **89.77 ms**, max **91.99 ms** versus rows512 baseline p50 **77-79 ms**. | Do not graph only final logits again. Re-open graph work only for a larger static decode block that can remove at least **>=5 ms/token** p50 and prove NCCL does not regress. |
| Direct single-block non-cooperative launch removes rows512 gap | killed | Replacing direct dispatch/combine cooperative launches with normal one-block launches generated 64/64 tokens, but `/tmp/pplx_direct_noncoop_olen64.log` measured p50 **76.60 ms**, p95 **93.11 ms**, max **100.48 ms** versus rows512 p50 **77-79 ms**, p95 **~91 ms**. | Do not change direct-only launch mechanism without also reducing direct kernel body/wait ordering. |
| Legacy four-stage cooperative launch alone explains 144 ms p50 | killed by experiment | Replacing the four legacy PPLX wrappers (`dispatch_send`, `dispatch_recv`, `combine_send`, `combine_recv`) with ordinary `cudaLaunchKernel`, replacing `grid.sync()` with block-local `__syncthreads()`, and forcing `num_blocks=1` built on H200 after restoring the FlashInfer nested CUTLASS submodule that rsync had overwritten. Smoke `/tmp/pplx_noncoop_1block_olen8.log` generated 8/8 tokens with steady p50 **144.01 ms**. Full gate `/tmp/pplx_noncoop_1block_olen64.log` generated 64/64 tokens with first decode **239.16 ms**, steady avg **142.58 ms**, p50 **143.98 ms**, p95 **148.03 ms**, max **164.03 ms**; teardown hit the known status **139** after metrics. | Cooperative launch is a real profiling hazard and driver-residency amplifier, but removing it with a one-block compatibility experiment does not change the legacy 144 ms p50 class. The floor is the broader four-stage worker/protocol cadence plus queued rank0 drain, not launch API choice alone. |
| Legacy dispatch kernels wait on host/GDR readiness flags | alive | `DeviceWorkspace::get_debug_ptr()` had returned `null_mut()`, so prior CUDA-side wait counters were no-ops. After wiring it to `debug_counters` and adding `pplx.worker.release_tx_ready_after_combine.*` NVTX ranges, H200 `/tmp/pplx_debug_counters_olen2.log` generated 2/2 tokens and printed all 8 ranks. Each rank saw **43** `dispatch_send` calls; `dispatch_send tx_ready_avg_cycles` was **2.934-2.941M cycles** without nsys and **2.114-2.118M cycles** under clean `--trace=cuda,nvtx --cuda-event-trace=false` nsys. `dispatch_recv num_flag_avg_cycles` was **0.702-0.714M cycles** without nsys and **1.468-1.754M cycles** under nsys. Clean NVTX profile `/tmp/pplx_debug_counters_nvtx_olen2.nsys-rep` shows `pplx.worker.release_tx_ready_after_combine.legacy` across 344 ranges: p50 **1.094 ms**, p95 **9.224 ms**, max **16.912 ms**; the top two long windows hit all 8 a2a workers together at **~16.9 ms**. The pulled GUI artifact is `profiles/pplx_debug_counters_nvtx_olen2.nsys-rep`. | Treat the next fix as a worker/protocol release problem, not a launch-type problem. Candidate directions: remove the end-of-layer combine barrier from the single-node pure-EP hot path, publish the next `tx_ready` from a GPU-owned/persistent progress path, or redesign the four-stage cadence so `dispatch_send` no longer begins by waiting on host GDR state from the previous combine. |
| RDMA worker CPU state is visible in Rust logs | alive as diagnostic fact | Added hard-coded `tracing::info!` state logs in `a2a_worker.rs` under `[pplx rdma worker]`, with `rank/device/step/state/elapsed_us/tid/cpu/tx_counter/err_counter` and per-state `wait_us` or `op_us`. H200 `/tmp/pplx_rdma_worker_state_olen2.log` generated 2/2 tokens and printed **5176** worker-state lines; first decode was **219.80 ms**, so this probe is timing-intrusive. Excluding step 0 startup and step 43 teardown, CPU-side aggregation over **336** rank-steps shows `routing_info_ready` p50 **4 us**, p95 **7 us**, max **11 us**; `dispatch_recv_done` p50 **1 us**; `dispatch_send_done` p50 **1 us**; `combine_recv_done` p50 **895 us**, p95 **1011 us**, max **1067 us**; `combine_barrier_done` p50 **35 us**, p95 **9861 us**, max **15990 us**. Long `combine_barrier_done` windows are synchronized across all 8 ranks at steps **8**, **14**, **26**, and **36**; `route_counter_done` also has synchronized events at steps **2** and **21**. A second split-barrier probe `/tmp/pplx_rdma_worker_barrier_state_olen2.log` printed **7928** worker-state lines and shows the combine-barrier tail is almost entirely `barrier_wait_imm_done:combine`: p50 **32 us**, p95 **9361 us**, max **15244 us**; `barrier_submit:combine` and `barrier_tx_ready_set:combine` are **0-1 us**. | `process_routing_info()` is not the CPU p50 owner. The CPU-side long tail is synchronized waiting for barrier immediates, not local metadata work, `tx_ready.set`, or tx-counter drain. The stable p50 floor is still `combine_recv_done` around ~0.8-1.0 ms/layer. Use this logging only for diagnosis because it changes timing; prefer targeted runs and remove or gate before retaining production code. |
| Publishing `tx_ready` immediately after `combine_recv_done` fixes the floor | killed as wait migration | The experiment set `tx_ready=true` right after legacy `combine_recv_done.wait()` and still ran the combine barrier before returning to the next worker step. H200 `/tmp/pplx_early_tx_ready_olen2.log` generated 2/2 tokens and cut `dispatch_send tx_ready_avg_cycles` from **~2.94M** to **~0.6K cycles**, but `dispatch_recv num_flag_avg_cycles` rose from **~0.71M** to **~3.58M cycles**; first decode moved **247.88 -> 219.64 ms**. H200 `/tmp/pplx_early_tx_ready_olen8.log` generated 8/8 tokens but regressed steady p50 to **155.76 ms** and p95/max **188.02 ms**. The patch was reverted locally and on H200; the retained diagnostic build has counters + NVTX only. | Early release proves the `tx_ready` wait is real, but it only moves the stall to the worker-produced recv-count flag. A real fix must remove or overlap the worker's route/count/recv-flag production, not just publish `tx_ready` earlier. |
| Moving route processing before `dispatch_send_done` overlaps the worker floor | killed as insufficient / regressive | Single-node `process_routing_info()` was moved immediately after route-counter completion, before waiting for `dispatch_send_done`, because it only consumes `num_routed` and writes GDR metadata. H200 `/tmp/pplx_overlap_route_process_olen2.log` generated 2/2 tokens and moved first decode **247.88 -> 235.81 ms**; counters changed to `dispatch_send tx_ready_avg_cycles` **~2.02M**, `dispatch_recv num_flag_avg_cycles` **~1.08M**, and `combine_send tx_ready_avg_cycles` **~0.44M**. H200 `/tmp/pplx_overlap_route_process_olen8.log` generated 8/8 tokens but regressed steady p50 to **155.98 ms**, p95/max **179.99 ms**. The patch was reverted locally and on H200. | Host route/count preparation can be shifted, but as long as the four-stage protocol still has separate send/recv/combine waits, the wait reappears at another stage. Do not keep local reordering patches; the fix needs to remove the worker-generated recv metadata path or replace the four-stage cadence. |
| Direct route-position reuse closes rows512 gap | killed as isolated patch | Persisting dispatch-computed route `position/source_rank` and feeding it to direct combine generated 64/64 tokens twice, with p50 **74.52 / 74.54 ms** and p95 **87.90 / 89.30 ms**. This is stable but below the prewritten **>=5 ms/token** p50 gate from the 77-79 ms rows512 baseline. | Do not keep a standalone route-position cache. Re-open only as part of a larger direct-side fusion that also removes another stage, launch, or stream handoff. |
| Reusable PPLX handoff events remove launch/queue gap | killed | Preallocating the five explicit PPLX MoE handoff events removed per-layer event create/destroy conceptually, but `/tmp/pplx_reuse_events_olen64.log` generated 64/64 tokens with p50 **77.09 ms**, p95 **90.45 ms**, max **91.84 ms**. | Do not pursue event allocation reuse as an isolated p50 fix. Event API totals are visible, but request p50 is still owned by remaining queued work and direct/MoE stage boundaries. |
| PPLX MoE layer-local host ranges are mostly API/a2a residency, not grouped arithmetic | alive as scheduling target | `/tmp/pplx_moe_stage_cuda_capture_olen16.sqlite` captured **38,325 kernels per GPU**. Representative long `grouped_fp4` / `grouped_w1_w3` host ranges are dominated by same-thread `cudaLaunchKernel` waits, while the correlated grouped kernels are sub-ms or start later on the stream; the same wall windows show other devices running `a2a_dispatch_recv`, `a2a_combine_send`, or `a2a_combine_recv` for **7-13 ms**. `p2p_all_to_all` p50 is **1.644 ms/layer** and `worker_wait_combine_recv_done` p50 **1.115 ms/layer**. | Next implementation must reduce per-layer PPLX scheduling/handshake residency by at least **20 ms/token** on H200 `output_len>=64`. Do not promote grouped/shared expert arithmetic rewrites unless a new profile shows reproducible GPU-kernel p50 ownership. |
| Single-run TPOT avg is enough for keep/revert | killed process issue | `output_len=8` single runs vary widely: clean/profile runs report 123-170 ms class values with different step distributions. | No keep/revert from one avg/max; require hypothesis-specific metric or repeated samples. |

## Perf Direction

1. **按 rank lane 解释剩余 gap**
   - 旧 PPLX full-runtime request p50 跟 rank0 p50 对齐；rank0 比其它 rank 多出来的 **~69 ms** 基本对应 `rank.logits_dtoh` drain。NCCL 也有相同形状，但非 rank0 lane 只有 **36.65 ms**，rank0 drain 只有 **26.36 ms**。
   - rows512 clean profile 后，PPLX rank0-like p50 **78.334 ms**，NCCL rank0-like p50 **62.857 ms**，gap **15.477 ms**。其中 launch API p50 gap **8.786 ms**，final D2H/drain gap **6.687 ms**，两项合计 **15.473 ms**。优化目标是缩短 logits 前的 launch/queue，而不是 D2H copy 本身。
2. **非通信 operator 不是当前 p50 主因**
   - NVTX-only `output_len=64` 中，PPLX `attention_ratio4` p50 **0.140 ms**，`attention_compressed` p50 **0.070 ms**，`attn_hc_pre_norm` p50 **0.018 ms**。这些 range 有 tail，但中位数不可能解释 80ms p50 gap。
   - Driver-contention profile 里 16-30 ms 的 ratio/HC 尖峰仍应作为 p95/p99 问题保留；但 p50 实验要先看 rank-lane 和 PPLX full-runtime composition。
3. **PPLX scheduling/handshake 的 direct-path 证据只作机制判断**
   - CUDA+NVTX containment 已经把 layer-local 长段拆开：host range 里看到的 multi-ms `grouped_fp4`/`shared_expert` 多数是 CUDA launch/API 等待，真实 grouped/shared kernels 不是 p50 主因。
   - 同一窗口里其它设备经常还在跑 `a2a_dispatch_recv`、`a2a_combine_send`、`a2a_combine_recv`。这解释了为什么 direct-worker-mode 只把 wait 从 `combine_recv_done` 移到 `combine_send_done`：worker 仍然是每层本地 MoE readiness 的序列化观察者。
   - single-node direct routed path 把 p50 从 **144.00 ms** 降到 **83.94 ms**，rows512 再降到 **77-79 ms**。这证明旧 floor 大部分来自 legacy worker/cooperative-kernel cadence，但该 path 绕过 upstream 四阶段语义，已从代码移除。
4. **当前保留实现回到 legacy four-stage**
   - direct routed path 的实验价值是定位机制，不是要把 bypass 留在 `pegainfer-comm`。现在保留的是 legacy `dispatch_send -> dispatch_recv -> combine_send -> combine_recv` 路径，以及 per-NUMA rank-slice placement 修复。
   - CPU placement 修正后，H200 `output_len=64` 两次复测 steady p50 **66.46 / 66.70 ms**、p95 **69.80 / 69.62 ms**，已接近 NCCL **63 ms** 级。剩余方向应先低侵入 profile 新 baseline，而不是继续维护 direct hack。
   - 新 placement 使用 `pegainfer_core::cpu_topology`，把 common CPU list parsing、affinity mask、thread pinning 和 CUDA-device NUMA lookup 从 dsv4 私有 helper 里抽出。本地 gate 覆盖 `cpu_topology` 单测、Rust/CUDA bridge 编译、格式和 diagnostic bench feature 编译；下一次 H200 profile 应基于这个 cleaned legacy path。
5. **把 grouped GEMM 的 host rows 上界收回来，但只能 GPU-only**
   - 旧 GPU indptr 版本用 `expanded_input.seq_capacity()` 作为 grouped GEMM `rows`，会多跑空行的 act-quant / epilogue work。direct routed 后这个浪费重新变得可见，rows512 已实测带来 **5-7 ms/token** p50 收益。
   - 每层 D2H 一个 padded-total 标量已实测会把 TPOT 拉坏到 1.54s/token；单独 compact/scatter 也已实测退化到 p50 **84.00 ms**。后续更进一步只能让 grouped path 原生接受 sparse/padded indptr，或把 compact 融进已有 kernel，不能新增两次 per-layer launch。
6. **重新评估四段 kernel 是否应该 per layer 启动**
   - 现在每层 4 个 pplx kernel，43 层就是 172 个通信 kernel/step；每个 kernel 都带 polling / worker handshake。
   - 更彻底的方向是 persistent/progress kernel 或把 dispatch_recv+combine metadata 准备融合进本地 MoE pipeline；否则 bs=1 decode 永远被固定开销限制。
7. **active-peer 协议已经降级为 tail cleanup**
   - rows512 profile 之后，direct dispatch + direct combine 的 p50 合计约 **0.474 ms/layer**，43 层就是 **~20 ms/token**，正好覆盖当前到 NCCL 的 **14-16 ms/token** gap。
   - 代码上 direct dispatch 仍然为所有 peer 写/等两轮 sync；direct combine 也先等所有 peer、再向所有 peer发布、再等所有 peer。数据读取已经按 route 的 `source_rank` 精确访问，但同步集合没有跟着收窄。
   - 实测结果把这条降级：combine-only active 只改善 tail；legacy half-slot active 会 timeout；独立 direct flag slots 能把 p95 **91.49 -> 87.79 ms**、p50 **77.33 -> 76.13 ms**，但没到 **<=70 ms** gate。同步等待不是唯一 p50 owner，不能继续只调 flag。
8. **compact rows 不是独立 kernel 的方向**
   - rows512 仍按 `local_experts * expert_padding = 512` 行启动 grouped FP4；bs=1/topk6/EP8 的真实 routed rows 上界是 48。
   - 但 GPU-only compact/scatter 实测 p50 **84.00 ms**，说明额外 per-layer kernel/API/stream work 比减少 grouped rows 更贵。rows 的真实上界要通过 fused compact、sparse grouped GEMM，或 direct kernel 原地生成 grouped 可消费布局来拿，不能靠独立搬运 kernel。
9. **CUDA Graph 只能做大粒度实验**
   - 细粒度 graph island 已经失败：instantiate 和 graph launch 本身吃掉收益，launch-class calls/step 只降约 **8%**，事件等待没有消失。
   - rows512 clean profile 反而说明方向更明确：要减少 `cudaLaunchKernel` 级 fanout 和最终 drain，graph 实验必须覆盖更大的静态 decode 单元，并且去掉显式 stream handoff 边界。小 helper 级 graph 不再做。
10. **direct-path stream handoff 单点不够**
   - direct routed 实验当时把 dispatch 放在 `moe_stream`，shared/grouped 放在 `ctx.stream`，direct combine 再回到 `moe_stream`，每层有 route handoff、dispatch handoff、expert handoff、combine handoff。
   - 只把 direct combine 放回 `ctx.stream` 实测 p50 **77.11 ms**，p95 **90.41 ms**，没过 **<=74 ms** gate。单独去掉 grouped 后两段 handoff 不是 p50 大项。
   - 后续要么做更大的 single-stream/persistent direct pipeline，要么把 graph/static block 覆盖到足够大的 decode 子图；只移动一个 kernel 所在 stream 不再作为独立方向。

## Next Action

当前 legacy four-stage 路径在 CPU placement 修正后已经从 **144 ms** 级降到 **66-67 ms**，接近 NCCL **63 ms** 级；direct routed hack、高侵入诊断仪表和 dsv4 临时 NVTX ranges 已清理。CPU 选择已整理成 common NUMA topology helper + per-NUMA rank slice policy。H200 下一步是基于 CUDA API/kernel 数据和 pplx 自带 ranges 复核剩余 **~3-4 ms** gap 和 rank0 drain 结构。
