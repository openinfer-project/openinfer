# Kimi-K2.6 文本算子 TODO

> **TL;DR:** Kimi-K2.6 首阶段只做 text-only。当前 routed expert INT4 package/compute/worker 主链已转向 vLLM Marlin WNA16：signed/unsigned nibble、checkpoint/CUTLASS/Marlin scale layout、fused W13(`gate_then_up`) + W2 runtime package 均已明确；真实 K2.5 rank0 layer1 W13 + SwiGLU + W2 + top-k reduce 对 vLLM fixture 0-diff，H20 全 61 层 prompt forward 多 prompt gate 4/4 greedy argmax match。direct worker/scheduler 的 H20 smoke/candidate/debug 测试入口已清理出主线；decode arena 已从 bs1/bs4 两档改为 `1..=4` 按实际 wave size 选 arena，避免 2/3 并发按 4 行 scratch/router/Marlin/reduce 执行；后续优化禁止假设 `bs==1`。scheduler 已接入 bs4 wave decode：4 并发 fixture prompt `max_tokens=2` 返回 `[1008,2742]`；decode row-state 曾收缩到 layer1 W13 Marlin，根因是 PegaInfer 误走 `use_atomic_add=true` 的 BF16 atomic split-K 路径。现已匹配 vLLM `use_atomic_add=false,use_fp32_reduce=true` 并预分配 `c_tmp`；H20 固定 4 并发 `max_tokens=16` gate 四路 token 全对，`ROUTER/ROUTE_ROW/ROW` diff 全为 0。性能主线已开始撤掉诊断负担：同 token row-diff D2H 主路径硬关，decode collective CPU barrier 不再执行；routed MoE decode combine 已从 dense F32 all-reduce bridge 改为 NCCL reduce-scatter bridge：local router/Marlin 仍按本 rank 实际 batch 行数执行，再用 device repeat 构造 RS 输入，H20 gate 待补。
>
> **Last touched:** 2026-05

## 范围

本清单只覆盖 `/data/models/Kimi-K2.6` 的 `language_model`。多模态相关的 `vision_tower`、`mm_projector`、media placeholder 插入、图像/视频 processor 都不进入首阶段。

目标不是先接完整 server，而是先把 operator surface 和测试夹具准备好。后续 crate、runtime、server 只消费这里列出的算子。

## 当前优先级

1. H20 端到端优先：停止继续新增内部 smoke 作为主线。`pegainfer-server` + OpenAI-compatible `/v1/completions` 已在 H20 跑通 K2.5 `max_tokens=1/2/8`；vLLM fixture 27-token prompt 的 `max_tokens=1` 返回 token id `1008`，`max_tokens=2` 返回 `[1008, 2742]`，4 并发 `max_tokens=8` 四路一致返回 `[1008,2742,2531,414,19180,6082,1379,387]`。
2. Decode(bs4) 生产化：worker-owned MLA KV/cache owner 已落地，prompt prefill 已把每层 compressed KV/KPE 写入 arena，direct crate 的旧 H20 smoke/candidate/perf 测试入口已移除；scheduler 现在按最多 4 个请求组成 wave，第 2 个 token 起调用真实 bs4 decode body。当前 HTTP 端到端 4 并发 output8 曾为 `57.4 tok/s`；强同步 profile 下稳态 decode step 约 `35.0ms`，纯 decode bs4 总吞吐约 `114 tok/s`。W13 atomic row-state bug 修复后，固定 4 并发 output16 gate 的 row diff 已清零；性能主线重新回到 NCCL bridge 下的 MoE collective/route 固定开销、continuous batching、collective/sampling D2H 清理和 graph-ready collectives，PPLX EP 作为后续替换项保留。
3. EP 生产化：decode routed MoE combine 正从 dense all-reduce bridge 迁到 NCCL reduce-scatter bridge。当前实现目标是 `local router/Marlin -> device repeat f32 -> reduce_scatter_f32_hidden`，不做 BF16 all-gather，也不把 local expert compute 按 EP world 放大。这还不是真 PPLX dispatch/combine；PPLX EP 需要后续替换 MoE-side dispatch/combine call sites。
4. Decode batch policy：旧代码把单请求也塞进固定 bs4 scratch，导致 router/Marlin route elems/routed reduce/logits 都按 4 行执行；上一版只拆成 bs1/bs4 两档，2/3 并发仍会落到 bs4。当前改为预分配 `1..=4` 四个 arena，scheduler 用真实 wave size 选择 arena。禁止基于 `bs==1` 做假设优化；所有性能改动必须服务 `bs>1` 和 `decode(bs4)>300 tok/s`。
5. Prefill perf hardening：当前 correctness path 在 128+ synthetic prompt 已过 1k tok/s，但仍有 per-layer allocation、首个 collective stream drain、host-visible final top1；后续要把 scratch/RoPE/cache 预分配，形成稳定 perf gate。
6. vLLM parity hardening：当前 H20 多 prompt gate 4/4 greedy argmax match，top-20 id overlap 最低 `19/20`；后续在 PPLX/perf path 上继续扩 prompt，出现 mismatch 再做 first-diff 定位。
7. 子模块 H20 gate：只证明真实权重能 load/package/route/launch/reduce；数值 gate 只对 Torch/vLLM 外部 fixture，不再用本仓库自写 dequant+cuBLAS 当 correctness reference。

## Decode 精度排查 checklist

当前证据链：

- H20 bs4 decode 第一个 row-state 分叉已经收缩到 layer1 routed expert。
- `moe_router_topk` 没有报告差异，说明同 token / 同 position / 同 layer 的 active rows 选到的 top-k expert id 和 top-k weight bitwise 一致。
- `moe_routed_local` 在 `kimi_marlin_sum_topk_rows_f32` 后、NCCL all-reduce 前已经报告差异，说明主嫌疑是本地 routed expert path，不是 routed NCCL combine。
- vLLM 源码确认 Marlin MoE 默认不用 atomic add；PegaInfer 误用 BF16 atomic split-K，修复为 `c_tmp` + global reduce 后固定 output16 gate 的 row diff 已为 0。
- 因此 row-state bug 当前收敛；后续精度证据仍要升级到外部 vLLM top-k/logit gate，短 output token match 只算 smoke。

| 编号 | 候选点 | 当前状态 | 划掉条件 / 下一刀 |
| --- | --- | --- | --- |
| A | MoE 输入 `scratch.normed` 是否行间 bitwise 相同 | 已在 H20 bs4 fixture 上划掉：`moe_normed_input` 没有任何 row diff。 | 当前不用回到 layer0/layer1 输入侧；继续查 W13 Marlin。 |
| B | Router logits/topk/weights 语义 | vLLM 源码对照完成：`grouped_topk` 返回未乘 `routed_scaling_factor` 的 normalized topk weights；`DeepseekV2MoE.forward` 在 routed expert 总输出后整体乘 scale。PegaInfer 旧实现把 `2.827` 提前乘进 router topk weight，导致 W2 BF16 kernel 内部 rounding boundary 不同。 | 已改为 router 输出 unscaled topk weights，routed F32 sum/all-reduce 后整体乘 `KIMI_K2_ROUTER_SCALE`；H20 还需要短 gate 复核。 |
| C | Route align metadata：`sorted_token_ids` / `expert_ids` / sentinel / local expert filtering / block size | 历史单测对 vLLM contract 通过，但 runtime bs4 layer1 尚未在当前真实 prompt 上对照。 | dump 或 debug 对比 layer1 runtime metadata：同输入 rows 的 `token*topk` 映射必须稳定；对照 vLLM `moe_align_block_size` 语义。 |
| D | W13 Marlin GEMM | 已定位并修复：H20 bs4 fixture 的第一批 `KIMI_DECODE_ROUTE_ROW_DIFF` 出现在 layer1 `moe_w13_out`，根因是 PegaInfer 固定 `use_atomic_add=true`，split-K 时走 BF16 atomicAdd 写 C；vLLM W13/W2 都走 `use_atomic_add=False,use_fp32_reduce=True`。 | 已改为预分配 `c_tmp` 并按 vLLM 关闭 atomic add；H20 固定 bs4 output16 gate 后 `ROUTE_ROW_COUNT=0`。 |
| E | W13 SwiGLU dtype/rounding | row-state gate 已划掉：atomic 修复后 `moe_w13_swiglu` 不再报告行间差异。 | 后续只在外部 top-k/logit parity 发现漂移时重新打开。 |
| F | W2 Marlin GEMM 的 top-k weight 乘法 | vLLM call site 确认为第二次 GEMM `mul_topk_weights=true`，PegaInfer 已匹配；atomic 修复后 `moe_w2_route_output` 不再报告行间差异。 | 后续只在外部 top-k/logit parity 发现漂移时重新打开。 |
| G | `sum_topk_rows_f32` | row-state gate 已划掉：W2 route output 与 routed local sum 在固定 output16 gate 均无行间差异。 | 后续仍需外部 top-k/logit parity 覆盖数值语义。 |
| H | Scratch / locks / c_tmp / output 清零 | vLLM `moe_wna16_marlin_gemm` 在 `use_fp32_reduce && !use_atomic_add` 时为每次调用分配 `c_tmp`，通过 global reduce 合并 split-K；`c_tmp` 不靠清零表达语义。 | PegaInfer decode arena 已改为持久预分配 `c_tmp`，launch 传入非空指针。后续若仍脏，继续查 route metadata 和 kernel 参数，而不是靠清零掩盖。 |

执行顺序：

1. subagent 先按 B/C/D/E/F/G/H 对照 vLLM 源码，把明显不一致项前置。
2. 主线程补最少切点：`moe_normed`、`moe_w13_out`、`moe_w13_swiglu`、`moe_w2_route_output`、`moe_routed_local`。
3. H20 只跑固定 bs4 fixture `max_tokens=16` 一轮；看到第一个脏切点就停止，不跑吞吐、不跑多轮。
4. 等 top-k/logit 数据通路补齐后，精度 gate 从 token ids 升级为 vLLM top20 overlap/logit diff；短 token match 只保留为 smoke。

## Execution Log: vLLM routed scale 对照

- subagent 对照 vLLM Kimi/DeepSeek MoE 路径后发现最高风险不一致：vLLM 的 `grouped_topk` 只返回 normalized topk weights，不在 router 内乘 `routed_scaling_factor`；`DeepseekV2MoE.forward` 在 routed experts 输出合并之后整体乘该 scale。
- PegaInfer 旧代码在 `kimi_router_noaux_tc_launch` 内把 topk weight 直接乘 `2.827`，再传入 W2 Marlin `mul_topk_weights=true`。这会把 scale 提前放进 W2 BF16 output path，和 vLLM 的 rounding boundary 不一致。
- 本轮代码改动：
  - `KimiRouterConfig::kimi_k2().route_scale` 从 `2.827` 改为 `1.0`，router 只输出 normalized topk weights；
  - 新增 `scale_f32_in_place` elementwise wrapper；
  - prompt/decode MoE 都在 routed F32 sum + EP/TP all-reduce 之后整体乘 `KIMI_K2_ROUTER_SCALE`；
  - 文档里的 router output contract 改为 unscaled topk weights。
- H20 短 gate：远端 release build 通过后，只跑 1 轮 4 并发 fixture `max_tokens=16`。输出 token ids 四路全对，wall `4853.571ms`、`13.186 tok/s`；`KIMI_DECODE_ROUTER_DIFF` 仍无输出；`moe_routed_local` 仍有 diff，典型 rank0 `first_abs_diff=0.00000047683716`、`max_abs_diff=0.0000038146973`，rank2 `max_abs_diff=0.000030517578`。结论：scale 放置修正是 vLLM parity 必修项，但不是 row-state root cause。
- 新增下一轮一次性切点：`moe_normed_input`、`moe_w13_out`、`moe_w13_swiglu`、`moe_w2_route_output`、`moe_routed_local`。下一次 H20 只跑固定 bs4 fixture 一轮，按第一条 dirty phase 划掉 checklist A/D/E/F/G，不再每加一个切点重启模型。
- H20 切点 gate：远端编译通过后，只跑 1 轮 4 并发 fixture `max_tokens=16`。输出 token ids 四路全对，wall `4806.792ms`、`13.314 tok/s`。日志计数：`ROUTER_COUNT=0`，`ROUTE_ROW_COUNT=18`，`ROW_COUNT=238`；phase 计数里 `moe_normed_input` 为 0，`moe_w13_out/moe_w13_swiglu/moe_w2_route_output/moe_routed_local` 各 6 条，`moe_routed_reduce` 8 条。第一条 route-row diff 是 layer1 `moe_w13_out`，例如 rank0 `row=1 route=2 dim=3 row0=0.041259766 row1=0.041015625 first_abs_diff=0.00024414063 max_abs_diff=0.0009765625`。结论：A 输入侧和 router 先划掉，当前主嫌疑收缩到 W13 Marlin GEMM 的 route/output/locks/c_tmp 语义。
- vLLM 源码/算子对照：
  - `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` 对 W13 和 W2 的 `ops.moe_wna16_marlin_gemm` 都传 `use_atomic_add=False,use_fp32_reduce=True`。
  - `vllm/csrc/moe/marlin_moe_wna16/ops.cu` 在该模式下分配 FP32 `c_tmp`，让 kernel 走 global reduce；只有 experimental atomic 模式才直接 BF16 atomicAdd 到输出 C。
  - H20 上用真实 K2.5 layer1/rank0 权重、重复 hidden rows、重复 topk pattern 调 vLLM W13 op，route rows bitwise 相同；因此当前 row diff 不是 Marlin MoE 必然行为。
  - PegaInfer 旧 wrapper 固定 `use_atomic_add=true` 且 `c_tmp=null`，与 vLLM path 不一致。当前代码已改为持久 `c_tmp` + `use_atomic_add=false`，下一次 H20 只复核这一处是否消掉 `moe_w13_out` 第一脏点。
- H20 atomic 修复 gate：
  - 本地 `cargo fmt --all --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels --tests`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20 dry-run rsync 后同步 `kimi_marlin_wna16.cu`、`kimi_experts.rs`、本文档；远端 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server --bin pegainfer` 通过。
  - 固定 4 并发 fixture prompt `max_tokens=16`：wall `5109.881ms`，`12.525 tok/s`，四路 token ids 全部匹配 `[1008,2742,2531,414,19180,6082,1379,387,261,5216,63853,13,374,1765,11983,306]`。
  - 日志计数：`ROUTER_COUNT=0`、`ROUTE_ROW_COUNT=0`、`ROW_COUNT=0`。结论：W13/W2 Marlin atomic split-K row-state bug 已修掉，下一步回到 decode(bs4) 性能主线和 vLLM top-k/logit parity。
  - 验证后已停止 `kimi-k2-rowdiff` tmux/port `18080`，`nvidia-smi --query-compute-apps` 无进程。

## Execution Log: decode 主路径诊断负担清理

- atomic 修复后，原来用于定位 row-state 的 `debug_identical_decode_*` 已经不适合留在性能主路径：4 并发同 prompt 会满足同 token / 同 position 条件，导致每层多个切点执行 `sync + D2H + sync`。
- 代码决策：decode worker 里 `debug_same_rows` 硬关为 `false`。诊断 helper 暂留作下一次 first-diff 工具，但默认请求不再触发 D2H。
- 代码决策：`all_reduce_hidden_via_f32_in_place` 继续保留 BF16->F32->BF16 桥，避免 BF16 NCCL row-offset rounding；但 F32 NCCL 从 per-row loop 改回单次 contiguous all-reduce。row-wise F32 collective 是 atomic bug 未修前的诊断桥，bs4 下会把每个 collective 放大成 4 次。
- 代码决策：decode F32 TP/routed collective helper 不再执行 CPU `Barrier`。prompt load 后第一发 vocab-shard embedding TP collective 的 barrier + stream drain 先保留，因为那是 H20 首个 NCCL call 的独立稳定性问题，不混进 decode steady TPOT 的这一刀。
- 本地验证：`cargo fmt --all --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
- H20 验证：
  - dry-run rsync 后同步 `worker.rs` 与 Kimi 文档；
  - `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server --bin pegainfer` 通过；
  - 同一 server 中固定 4 并发 fixture prompt：`max_tokens=16` wall `4615.953ms`、`13.865 tok/s`，四路完整 token ids 匹配 vLLM fixture；
  - warm `max_tokens=64` wall `1774.731ms`、HTTP 端到端输出吞吐 `144.247 tok/s`，四路前 16 token 匹配，64 token 长度一致，tail 一致；
  - `ROUTER_COUNT=0`、`ROUTE_ROW_COUNT=0`、`ROW_COUNT=0`。验证后已停止 tmux/port `18080`，H20 `nvidia-smi --query-compute-apps` 无进程。
- 结论：撤掉 row-diff D2H、row-wise F32 collective 和 decode CPU barrier 后，warm output64 从旧口径约 `114 tok/s` 提升到 `144 tok/s` 量级，但仍低于 `decode(bs4)>300 tok/s`。下一刀评估 decode TP hidden 是否能从 BF16->F32->BF16 bridge 恢复为 BF16 bulk collective，或者直接转向 PPLX/collective cadence。

## Execution Log: routed MoE decode reduce-scatter bridge

- 设计对照：原 routed F32 dense all-reduce bridge 改成 NCCL reduce-scatter bridge，但不引入 BF16 all-gather。local router、Marlin W13/W2、SwiGLU、top-k sum 仍按本 rank 实际 batch 行数执行。
- 代码改动：
  - `KimiWorkerDecodeScratch` 增加 `routed_reduce_scatter_send_f32`，容量为 `batch_size * EP8 * hidden`；
  - 新增 `repeat_f32_for_reduce_scatter_cuda` / Rust wrapper，在 device 上把 local `[B,H]` partial 重复成 reduce-scatter 输入 `[EP8*B,H]`；
  - `forward_moe_layer_decode_into` 在 `kimi_marlin_sum_topk_rows_f32` 后执行 device repeat，再用 `reduce_scatter_f32_hidden_into` 写回本地 `[B,H]`，随后沿用 router scale 与 residual add；
  - `batch_decode_trace.rs` 的每层 MoE trace 从 `routed_allreduce` 改成 `repeat_f32_for_reduce_scatter` + `routed_reduce_scatter`。该路径避免 `B*EP8` expert compute，仍是 NCCL bridge，不是最终 PPLX EP。
- graph 约束：这条 bridge 使用预分配 buffer、同 stream device kernel 和 NCCL reduce-scatter，不需要 D2H、不做 step 内分配。H20 graph probe 已证明 NCCL all-reduce / reduce-scatter 本身可以 capture；整段 decode graph 需要跨 rank begin/enqueue/end/launch 对齐。
- 验证状态：本地 `cargo fmt --all --check` 与 `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins` 已过；H20 短 greedy gate 待同步后补。

## Execution Log: CUDA Graph gate

- baseline：H20 当前稳定 correctness 版本在 4 并发 fixture 上，warm `max_tokens=64` 为 `27.76ms/token`、`144.1 tok/s`，更长 warm `max_tokens=128` 为 `24.92ms/token`、`160.5 tok/s`；四路前 16 token 与 vLLM fixture 一致。
- rejected fusion：尝试把 `allreduce_f32 -> f32_to_bf16 -> add_batch` 改成已有 `add_f32_bf16_to_bf16`，`max_tokens=128` 到 `24.09ms/token`，但 token 从第 3 个开始变成 `[1008,2742,924,6454,...]`。根因是旧语义有 `F32 contribution -> BF16` 的 rounding boundary，新 kernel 变成 `F32 contribution + BF16 residual -> BF16`，数值边界不等价。该语义改动已回退；后续若做 fusion，必须写“先 round contribution 到 BF16，再执行 BF16 residual add”的专用 kernel。
- CUDA Graph gate：按 Qwen 路径把 Kimi decode GPU body 拆成 graph 内 launch 和 graph 外 top1 D2H，server 侧临时把 Kimi `enable_cuda_graph` 打开，H20 `max_tokens=2` 四并发卡在第一轮 decode capture，日志只有 completions request，没有 completion/error；kill server 后客户端断连。结论：当前“整段 decode + NCCL all-reduce/reduce-scatter bridge”不能直接 CUDA Graph capture，表现为 capture-time hang。
- graph root cause 复查：新增 `kimi_graph_probe`，H20 分别验证 local kernel、cuBLAS GEMM、NCCL all-reduce、NCCL reduce-scatter 的 capture/replay 均通过。此前 hang 不是 collective 不能进图，而是 Kimi worker 每个 rank 独立 begin/end/launch，NCCL graph capture 没有跨 rank 阶段对齐。
- 修复：`CudaGraphState` 增加同步 phase hook；Kimi worker 在 graph capture/replay 的 begin、enqueue 后、end、launch 前后使用 rank barrier 对齐。`pegainfer-server` 侧 Kimi 开始尊重 `--cuda-graph true`，用于显式 gate。
- H20 graph gate：`target/release/pegainfer --model-path /data/models/Kimi-K2.5 --port 18080 --cuda-graph true` 启动，4 并发 fixture prompt：
  - `max_tokens=2`：wall `4511.0ms`，四路 token ids `[1008,2742]`，证明原 capture hang 已修；
  - warm `max_tokens=16`：wall `714.4ms`，`89.6 tok/s`，四路 16 token 全对；
  - warm `max_tokens=64`：wall `1523.1ms`，`168.1 tok/s`，`23.80ms/token/wave`，四路 prefix/tail 一致；
  - warm `max_tokens=128`：wall `2641.9ms`，`193.8 tok/s`，`20.64ms/token/wave`，四路 prefix/tail 一致。
- 决策：Kimi graph 主线继续推进；当前 graph 能进整段 decode，但距离 `15ms/token/wave` 仍有约 `5.6ms` 差距。下一步不做 residual fusion，优先看 graph replay 下剩余 kernel/NCCL 时间组成，尤其是 TP hidden bridge、shared expert GEMM+collective、routed RS bridge 和 FlashInfer MLA。

## Rejected: decode TP hidden BF16 bulk collective

- 试验内容：把 decode TP hidden reductions 从 BF16->F32->BF16 bridge 改回 BF16 bulk NCCL all-reduce，覆盖 embedding、attention `o_proj`、dense/shared down-proj 的 active-batch 通用路径；routed expert combine 仍保留 F32。
- H20 结果：远端 `fmt/check/build` 通过；固定 4 并发 fixture 中 `max_tokens=16` wall `4693.312ms`、`13.636 tok/s`，但 row1 输出变成 `[1008,2742,924,6454,2531,...]`；`max_tokens=64` wall `1788.999ms`、`143.097 tok/s`，row2 同样发散。
- 结论：BF16 NCCL row-offset rounding 仍会影响 greedy，不只是诊断日志噪声；这条没有性能收益，且破坏 output16/64 correctness，已回退到 F32 bulk bridge。下一步不再在 BF16 hidden collective 上试探，转向减少 collective 次数/launch 次数和 PPLX EP。

## H20 decode profile 结论

- H20 可用 `/usr/local/cuda/bin/nsys`；本轮同时保留两类数据：
  - 强同步分段 profile：临时硬编码 `ctx.sync()`，只用于定位 logical stage，不作为生产吞吐。
  - nsys sqlite trace：产物在 `/tmp/pegainfer-kimi-nsys/kimi_bs4_decode.{nsys-rep,sqlite}`，tail 汇总在 `/tmp/pegainfer-kimi-nsys/tail-summary.md`。该请求在 nsys 下由于 profiler overhead 只有 `14.26 tok/s`，不能作为吞吐数值；输出 token 仍为 4 路一致的 16-token fixture。
- 4 并发 vLLM fixture prompt 的非 profile server 路径：
  - `max_tokens=8` 四路一致，wall `557.2ms`，32 output tokens，HTTP 端到端输出吞吐 `57.4 tok/s`。
  - 这个口径包含 4 路 prompt prefill、frontend、scheduler wave 和 response 开销，不是纯 decode。
- 强同步 profile 的稳态 decode 口径：
  - steady position `28..33`：`35.0ms/bs4 step`，所以总吞吐是 `4 / 0.035 = 114 tok/s` 左右；单请求 TPOT 等价约 `35ms/token`。
  - 第一 decode step position `27` 明显更慢，主要来自 layer0 MLA/dense/final logits 的冷启动和首步 cache/collective 状态。
- 稳态分段均值：
  - MoE total `22.8ms/step`
  - MLA `6.47ms/step`
  - attention 后 TP all-reduce + residual `5.27ms/step`
  - final logits `0.11ms/step`
  - local top1 + host readback `0.09ms/step`
- MoE 细分均值：
  - shared expert + TP all-reduce `6.55ms/step`
  - routed reduce/add + f32 all-reduce `6.37ms/step`
  - router `3.70ms/step`
  - route align `1.31ms/step`
  - Marlin W13 `2.21ms/step`
  - W13 SwiGLU `0.84ms/step`
  - Marlin W2 `1.81ms/step`
- nsys kernel tail 结论：
  - `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`：`count=1472`，`p50=74.7us`，`std=201us`，`p99=780us`，`max=2.98ms`。p50 看起来很低，但 p99/p50 已到 `10.4x`，max/p50 `39.8x`。
  - `ncclDevKernel_AllReduce_Sum_f32_RING_LL`：`count=718`，`p50=64.8us`，`std=83.6us`，`p99=385us`，`max=886us`。这是 routed reduce/add 侧 tail 信号。
  - `pegainfer_kimi_marlin_moe_wna16::Marlin`：`count=1436`，`p50=14.3us`，`std=40.6us`，`p99=154us`，`max=187us`。这类 p50 极低、p99/max 飞起的 kernel 必须按 route/expert 负载和 rank skew 拆。
  - `flashinfer::BatchDecodeWithPagedKVCacheKernelMLA` 比较稳定：`p50=9.92us`，`p99=11.85us`，`max=12.22us`。当前 attention kernel 本体不是 tail 源头。
- nsys CUDA API tail 结论：
  - `cuMemAllocAsync/cuMemFreeAsync` 各约 `8k` 次，p99 分别约 `132us/134us`，说明请求窗口内仍有大量分配/释放或库侧 workspace churn；下一轮要用 NVTX/cudaProfilerApi 把 prompt 和 steady decode 分开。
  - `cudaLaunchKernel_v7000` / `cuLaunchKernelEx` p50 约 `4us`，但 max 分别到 `14.4ms/15.4ms`，属于 rare outlier，不能只看 launch avg。
  - `cuStreamSynchronize` 只有 `22` 次，但 `p50=28.3us`、`p99/max=9.87ms`。这类 drain 会直接吃掉端到端尾部，后续要按调用点清掉或隔离。
  - `cuMemcpyDtoHAsync_v2` 总量只有 `0.44ms`，D2H 不是当前最大头；host-visible sync/drain 比单次拷贝更值得追。
- 结论：top1 不是当前 decode(bs4) 主瓶颈；Marlin GEMM 平均占比也不是最大头，但它的 p99/max 说明 routed expert 负载和 rank arrival skew 必须进入 profile 口径。下一步应按 DSV4 Flash 的经验先压 MoE/collective cadence：减少每层 shared/routed all-reduce、route/align 固定开销、allocator churn 和 rank phase skew，再考虑更大粒度 graph/static decode block。
- 以后 Kimi 性能 profile 必须同时报告 `count/total/avg/std/p50/p95/p99/max/p99-p50/max-p50`；只给 p50 或 avg 的数据不能支持 keep/revert 决策。

## Qwen3 exporter/report 经验迁移

- 2026-05-21 复盘 Qwen3-4B 的测量链路后，Kimi 性能口径改为先补 model-local exporter/report，再用 HTTP/nsys 做端到端佐证。
- Qwen3 当前不是靠端到端 trace 直接解释单 op，而是三层结构：
  - `batch_decode_trace.rs` 通过 `kernel-call-trace` 导出真实 decode DAG 的 `KernelCall`，包含 op、shape、call-site 和 repeat count；
  - `qwen3_kernel_report.rs` 用 manifest 驱动单 op snapshot，CUDA event/CUPTI 只测目标 op，记录硬件、cache state、iters、CUPTI 指标和变体；
  - `qwen3_model_report.rs` 重新按 runtime trace 的 call count 组合出 model-level decode report。
- Kimi 后续不能再用“HTTP 4 并发 + nsys 整个请求窗口”的 trace 计算纯 decode kernel 时间；该口径混入 prompt prefill、frontend、scheduler、首轮 lazy init 和 response 开销。之前把 `magma_sgemmEx_kernel` 的 `240` 次总耗时按 `16` 个 output token 平摊，是错误口径；`240 = 4 请求 * 60 MoE 层`，主要对应 4 个 prompt prefill 中的 shared expert GEMM，不是 TP8 decode steady shared GEMM。
- Kimi 下一步测量入口：
  1. 增加 `kimi_decode_trace`：导出 bs1/bs4 decode DAG，先覆盖 `embedding`、MLA projections/rope/FlashInfer MLA/o_proj、dense/shared GEMM、router、Marlin W13/W2、SwiGLU、topk reduce、BF16/F32 all-reduce、logits/top1。
  2. 增加 `kimi_kernel_report`：先给 attention-only、shared BF16 GEMM、Marlin WNA16、router/align/reduce、NCCL BF16/F32 bridge 建独立 provider；每个 provider 用 CUDA event/CUPTI，不读 HTTP trace。
  3. 增加 `kimi_model_report decode --batch-size {1,4} --kv-len <n>`：按真实 61 层 schedule 汇总 mean/std/p50/p95/p99/max，并显式区分 prompt/prefill 和 decode steady。
  4. H20 perf keep/revert 只接受：外部 vLLM greedy gate 不回退，加 model report 显示目标 decode stage 下降。端到端 HTTP throughput 作为最终 serving 佐证，不再作为 first-principles kernel 时间来源。
- 本轮已落地第一版 Kimi tooling：
  - `pegainfer-kimi-k2/src/batch_decode_trace.rs` 生成 rank0-local decode `KernelCall` DAG；bs4/kv1024 trace 展开 `1765` 个 call，覆盖 `61` 层 MLA、`60` 层 MoE、`183` 次 all-reduce、final logits/top1。
  - `pegainfer-kimi-k2/src/kernel_report.rs` 提供单 op CUDA event provider；已覆盖 BF16 GEMM、RMSNorm、SiLU、BF16 add、F32 scale、embedding、top1、MLA rope/absorb/v_up/FlashInfer decode、router、route align、W13 SwiGLU、W13/W2 Marlin WNA16 synthetic provider、topk sum、Kimi F32+BF16 add。`kimi_add_f32_bf16_to_bf16` 已提升到 `pegainfer-kernels::ops`，不再用 BF16 add 冒充。
  - `pegainfer-kimi-k2/src/bin/kimi_kernel_report.rs` 支持 `trace` / `run`；`pegainfer-kimi-k2/src/bin/kimi_model_report.rs` 按 trace call count 汇总 `by_op` / `by_call_site` / coverage。
  - 当前缺口必须在 report 里保持显式 missing：`all_reduce` 需要 8-rank H20 harness；`kimi_marlin_wna16_gemm` 已有 synthetic package provider，但缺真实 per-rank route histogram。
- 本轮把 live runtime hook 的基础补上：
  - `pegainfer-core::ops::call_trace` 从纯 thread-local collector 扩展为 TLS + global collector；父线程 `collect_result` 现在可以收集 worker 子线程记录的 `KernelCall`。
  - 新增 CPU-only 单测 `collect_result_captures_calls_from_child_thread` 覆盖跨线程收集，避免 Kimi 的 persistent rank worker trace 永远留在 worker TLS 里。
  - Kimi `kernel-report` feature 现在包含 `kernel-call-trace`。`forward_decode_batch_next_tokens` 的真实 rank worker decode command 在 rank0、collector enabled 时，会按实际 `decode_batch_size` 和 `append_positions` 记录同一份 rank0-local decode DAG。
  - non-HTTP runtime trace CLI 已完成：`kimi_kernel_report trace --source runtime` 和 `kimi_model_report decode --source runtime` 会启动 direct runtime，并用 `call_trace::collect_result` 收集 rank0 worker 发出的 calls；`--source static` 只保留为离线 DAG 对照。
- Rank scope 决策：Kimi 是 8 卡 TP8/EP8。第一版 report 只采 rank0 local compute + collective placeholder，足够解释 dense/shared/attention 的单 rank kernel count；但会丢 MoE EP 真实 route 分布、rank imbalance 和 NCCL tail。后续补 full-rank extension 时要记录每 rank route histogram、local routed rows、collective p99/max，不能用 rank0 代表 EP 全局。
- 已验证命令：

```bash
cargo fmt --all --check
PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins
PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-qwen3-4b --features kernel-report --bins
PEGAINFER_CUDA_SM=90a cargo test --release -p pegainfer-core --features kernel-call-trace collect_result_captures_calls_from_child_thread
PEGAINFER_CUDA_SM=90a cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_kernel_report -- trace --batch-size 4 --kv-len 1024 --out /tmp/kimi_decode_trace_bs4_kv1024.json
```

- trace 摘要：`calls=1765`、`unique_ops=18`，top ops 为 `gemm_graphsafe=489`、`rms_norm_batch=184`、`all_reduce=183`、`add_batch=122`、`kimi_marlin_wna16_gemm=120`。
- H20 验证：
  - 先 probe `h20-100`：仓库路径 `/root/develop/xingming/pegainfer-kimi-k2-main`，模型权重仍在 `/data/models/Kimi-K2.5`；本轮只做 build/trace，没有启动 server。
  - dry-run rsync 后同步本轮精确文件列表；不传模型、日志或 build artifact。
  - 远端首次 `cargo check` 失败于缺少 Triton Python：`Could not find a Python interpreter with Triton installed`。改用现有 `/root/develop/xingming/pegainfer-kimi-k2-main/.venv-kimi/bin/python` 作为 `PEGAINFER_TRITON_PYTHON` 后通过。
  - 远端通过：

```bash
PEGAINFER_CUDA_SM=90a PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer-kimi-k2-main/.venv-kimi/bin/python \
  cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins
PEGAINFER_CUDA_SM=90a PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer-kimi-k2-main/.venv-kimi/bin/python \
  cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_kernel_report -- \
  trace --batch-size 4 --kv-len 1024 --out /tmp/kimi_decode_trace_bs4_kv1024.json
```

  - 远端 trace 摘要同本地：`calls=1765`、`unique_ops=18`、JSON `1921890` bytes。运行结束后没有 Kimi server 进程；当时 H20 上另有无关 `scripts/pd_rdma_e2e.py --cuda-device 2` 占用约 `1520MiB`。
  - 跨线程 trace 补丁同步后，H20 复跑 `cargo test --release -p pegainfer-core --features kernel-call-trace collect_result_captures_calls_from_child_thread`、`cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins` 和同一条 trace 命令均通过；trace 仍为 `1765` calls。
- Runtime trace gate：
  - `kimi_kernel_report trace` 默认 `--source runtime`，通过 `EngineHandle` 直接启动 Kimi direct runtime、提交 `GenerateRequest`，不经过 HTTP/server。`--source static` 只作为离线 DAG 对照。
  - 真实 runtime trace 最小 H20 命令：

```bash
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer-kimi-k2-main/.venv-kimi/bin/python \
cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_kernel_report -- \
  trace --source runtime --batch-size 1 --kv-len 2 --out /tmp/kimi_runtime_trace_bs1_kv2.json
```

  - 第一轮忘记 `LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib`，scheduler 初始化阶段找不到 NCCL；修正环境后 trace 写出 JSON，但退出时 segfault。根因是 Kimi `start_engine` 丢弃 scheduler `JoinHandle`，进程退出时没有等待 CUDA/NCCL worker teardown。
  - 修复：`start_engine` 改为 `EngineHandle::new_with_join_handle(submit_tx, scheduler_handle)`；H20 重跑 runtime trace 退出码 `0`。
  - 成功产物：`/tmp/kimi_runtime_trace_bs1_kv2.json`，`calls=1765`，first `decode.embedding / embedding_batch_vocab_shard`，last `decode.top1 / top1_batch`，top call counts 同静态 DAG。此证据来自真实 direct runtime decode worker，不经过 HTTP。
- Runtime model report gate：
  - H20 最小命令：

```bash
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer-kimi-k2-main/.venv-kimi/bin/python \
cargo run --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_model_report -- \
  decode --source runtime --batch-size 1 --kv-len 2 --iters 1 --format text \
  --out /tmp/kimi_runtime_model_report_bs1_kv2.json
```

  - 结果退出码 `0`，`schedule=1765`，`schedule_source="Kimi direct runtime decode trace via EngineHandle/worker; no HTTP"`，`coverage_missing=7`，`total_measured_us=17094.496`。这是账本结构 gate，不是性能结论；`iters=1` 只证明 model report 能消费真实 runtime call count。
- 当前 provider 边界：`all_reduce` 需要独立 8-rank/NCCL harness；`kimi_marlin_wna16_gemm` 已有 synthetic packed INT4 provider，但还没有真实 per-rank route histogram。CUPTI raw metric 本轮不接入，当前是 CUDA event-only。
- CUPTI 决策更新：先不接 CUPTI，目标收敛到 decode 性能组成解释。`kimi_model_report` schema `2` 已把 total call coverage 拆开：
  - `total_schedule_calls`
  - `measured_schedule_calls`
  - `missing_schedule_calls`
  - `missing_by_op`
- H20 runtime model report schema2 gate：`decode --source runtime --batch-size 1 --kv-len 2 --iters 1` 退出码 `0`。接入 Marlin provider 前输出 `total_schedule_calls=1765`、`measured_schedule_calls=1462`、`missing_schedule_calls=303`；接入 Marlin provider 后输出 `measured_schedule_calls=1582`、`missing_schedule_calls=183`，missing 只剩：
  - `all_reduce`: `183` calls / `5` normalized call-sites，reason 带 `rank participation hint=8`，说明它是 8 rank NCCL collective placeholder，不是单卡 kernel。
- H20 report 产物：
  - runtime bs1/kv2: `/tmp/kimi_runtime_model_report_bs1_kv2_marlin.json`，measured subset `51.796ms`，Marlin WNA16 `34.554ms`，coverage `1582/1765`。
  - static bs4/kv1024: `/tmp/kimi_static_model_report_bs4_kv1024_marlin.json`，measured subset `149.904ms`，Marlin WNA16 `118.476ms`，coverage `1582/1765`。
- 解读规则：`total_measured_us` 只代表 event provider 已覆盖的 rank0-local call subset，不是完整 TPOT；报告里的性能组成要同时读 `by_op` 和 `missing_by_op`。Marlin provider 当前用 synthetic all-local route，缺少真实 EP8 route histogram，所以 bs4 Marlin 占比不能直接当作线上 8 卡全局平均；它用于说明 report 已能把 W13/W2 大块计入账本，并暴露下一步需要 full-rank route histogram。

## H20 active-batch 清理与 bs1 假设禁令

- 2026-05-21 清理固定 bs4 decode scratch：第一轮把 `KimiRankLoadedWeights` 拆成 bs1/bs4 两个 arena，解决单请求按 4 行执行的问题；本轮继续改成 `1..=4` 四个 arena，scheduler 用真实 `reqs.len()` 作为 decode batch size，2/3 并发也不再按 4 行执行。
- 删除 decode dense/MoE 中 `scratch.hidden.seq_len == 4` 的硬断言，改为 `1..=KIMI_DECODE_MAX_BATCH`；bs1/bs2/bs3 不再按 4 行执行 embedding、MLA projection、router、Marlin W13/W2、routed F32 reduce 和 logits。
- H20 验证：
  - 本地 `cargo fmt --all --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20 同步后 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server --bin pegainfer --bin bench_serving` 通过。
  - barrier 版本单请求 fixture prompt：`max_tokens=16` wall `462.832ms`，`max_tokens=32` wall `832.748ms`，`max_tokens=64` wall `1598.763ms`；`16->32` 差分 TPOT `23.12ms`，`32->64` 差分 TPOT `23.94ms`。
  - rejected no-barrier 实验：bs1 跳过 CPU barrier 曾让暖态 `32->64` 差分 TPOT 到 `18.01ms`，但这是 `bs==1` 假设优化，不服务 `decode(bs4)>300 tok/s`，并且掩盖了当时的 rank/stream 状态问题；该路径已回退，当前生产 decode collective 不再使用 CPU barrier。
- 决策：禁止新增 `bs==1` 专用性能分支。后续优化必须按 active batch / bs4 / continuous batching 设计。下一处大头仍是 NCCL bridge 每 token 的 collective cadence：embedding all-reduce、61 个 attention o_proj all-reduce、1 个 dense MLP all-reduce、60 个 shared BF16 all-reduce、60 个 routed F32 all-reduce，合计约 183 次 collective；CUDA Graph capture/replay 的 rank phase 对齐只用于 graph begin/end/launch，不是每个 collective 前的 CPU barrier。

## NCCL barrier 排查：先收紧错误边界

- 用户指出：单 stream 上 NCCL all-reduce 之前还需要 CPU barrier 这件事本身不正常，不能把 barrier 当成最终解释。
- 子 agent review 结论已采纳：
  - decode collective 的调用序列在代码上跨 rank 一致，没有明确某个 rank 跳过 collective；
  - barrier 能影响 correctness，更像 host enqueue phase、rank arrival skew、padding/tail 状态，或更早 CUDA/cuBLAS 错误延迟到 NCCL 才暴露；
  - `gemm_cuda` / `gemm_graphsafe_cuda` 原本丢弃 `cublasSetStream`、`cublasGemmEx` 和 `cudaPeekAtLastError` 状态，bs4 decode 还会因为 `seq_len=4` 走 prefill/workspace cuBLAS handle，这会污染 barrier 判断。
- 本轮代码决策：
  - `pegainfer-kernels/csrc/linear.cu` 的两个 GEMM FFI 改为返回状态；cuBLAS 状态用 `100000 + cublasStatus_t` 编码，CUDA 状态原样返回；
  - `pegainfer-kernels/src/ops/linear.rs` 新增 checked wrapper，旧 infallible wrapper 至少会在 launch 边界 fail fast，不再静默吞错；
  - Kimi decode path 全部改用 `gemm_graphsafe_into_checked`，对 active batch `1..=4` 都走 workspace-free cuBLAS handle，不绑定 `bs==1`；
  - Kimi prompt/prefill path 改用 `gemm_into_checked`，保留默认策略；
  - scheduler 聚合 prompt/decode rank report 前校验 `batch_slot`、`input_token_id`、vocab shard、local/global token 映射、top logit finite、`dense=1/moe=60`、stub=0；协议错位时直接报错，不进入 max-by。
- 当前判断：GEMM/report 改动只排除了 immediate launch/protocol failure，没有缩小到 payload root cause。CPU barrier 仍然保留，但它不是稳定 correctness guard；H20 warm output16 的两路分叉说明后续必须直接抓 device row-state first-diff。
- 第一轮 row-state 证据：H20 4 并发 `max_tokens=16` 复现 1 路坏前缀、3 路正确，旧 single-atomic 日志只抓到 `rank=6 phase=mla_residual layer=Some(0)`，输入 `tokens=[1008,1008,1008,1008]`、`positions=[27,27,27,27]`，`row=1 dim=0 first_abs_diff=0.000015258789 max_abs_diff=0.001953125`。这说明分叉最晚在 layer0 attention residual 后已经出现；由于旧 atomic 会被任一 rank/phase 抢占，它不能排除其他 rank 的 `mla_projected`、`mla_projected_allreduce` 或 `mla_residual_add` 更早先出现 diff。
- 第二轮 per-phase 证据：同一 H20 server 首轮 4 并发 `max_tokens=16` 冷批 wall `2319.265ms`、`27.595 tok/s`，row0 输出 `[1008,2742,924,6454,...]`，rows 1/2/3 对齐 fixture；随后的暖批 wall `786.084ms`、`81.416 tok/s` 四路全对。日志没有 `embedding_allreduce`、`mla_q_a`、`mla_compressed_normed`、`mla_append_ckv`、`mla_latent` 或 `mla_projected` 差异；第一条差异是 `rank=5 phase=mla_projected_allreduce layer=Some(0)`，随后传播到 `mla_residual_add`、`mla_residual`、`dense_projected`、`dense_residual_add` 和 `layer_output`。这把边界推进到 layer0 attention `o_proj` 的 TP BF16 all-reduce 之后。

## Execution Log: layer0 per-phase row-state 诊断

- 2026-05-21 本地把临时 row-state instrumentation 从全局 single atomic 改成 per-phase bitset：同一 phase 最多打印一条 diff，但晚到的 phase 不再屏蔽更早 phase 的后续日志。
- 新增 layer0 切点：
  - `mla_projected`：`o_proj` 本地 GEMM 后、TP all-reduce 前。
  - `mla_projected_allreduce`：TP all-reduce 后、residual add 前。
  - `mla_residual_add`：`hidden + projected` 的 add 输出。
  - `mla_residual`：swap 回主 hidden 后。
- 诊断硬编码只覆盖 `layer_idx <= 0` 和 embedding，这样下一次 H20 只围绕已复现的 layer0 分叉定位，不把每层 D2H/sync 扩散成新的性能噪音。
- 本轮没有启动 H20 server、没有占用 GPU；只做本地编译门：
  - `cargo fmt --all --check`
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests`
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels --tests`
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-server --bin pegainfer`
- H20 第二轮执行后已停止旧 `kimi-k2-rowdiff` server；`nvidia-smi --query-compute-apps` 无 Kimi 进程。最新定位要求下一轮只验证 NCCL 初始化/collective 路径，不再把旧 `81 tok/s` 当性能结论。

## Execution Log: NCCL worker-thread init 诊断

- 2026-05-21 核对 `cudarc::nccl::safe::Comm`：`all_reduce_in_place` 使用 comm 内部保存的 `Arc<CudaStream>` 调 `ncclAllReduce`，并通过 `device_ptr_mut(&self.stream)` 取 buffer 指针。
- 核对 Kimi scheduler：旧代码由 scheduler 线程创建 8 个 `KimiRankGpuContext`，用这些 context 的 stream 调 `Comm::from_devices`，再把 comm 发送给 rank worker。stream 的确是 worker 后续使用的同一条 `Arc<CudaStream>`，所以“comm 绑到另一条 stream”不成立。
- 本轮代码改动：去掉 scheduler 线程 `Comm::from_devices`，改为 scheduler 只创建 NCCL unique id，并发给 8 个 worker；每个 worker 在自己已经 `set_current()` 的持久线程里用 `Comm::from_rank(self.ctx.stream.clone(), rank, world_size, id)` 初始化 communicator。这样 communicator 的创建线程、CUDA context 和后续 enqueue 线程一致。
- 本地编译门：
  - `cargo fmt --all --check`
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests`
- H20 gate：h20-100 release build 通过。前两轮 4 并发 fixture `max_tokens=16` 全对，冷批 wall `4687.630ms`、`13.653 tok/s`，暖批 wall `803.357ms`、`79.666 tok/s`；随后两轮继续复现分叉，round2 row2 与 round3 row1/row2 变为 `[1008,2742,924,6454,2531,...]`。row-diff 仍出现 `mla_projected_allreduce`、`mla_residual_add`、`mla_residual`、dense 和 layer output 差异。因此 worker-thread NCCL init 只能保留为更合理的初始化方式，不能算 bug 修复。
- 诊断修正：旧 `debug_identical_decode_rows_*` 在 `clone_dtoh` 后没有二次 `ctx.sync()`，而 cudarc 的 `clone_dtoh` 只是 enqueue async D2H；同时 phase 全局 bitset 会遮住其他 rank/layer/step。新补丁改为 D2H 后同步，并用全局最多 `256` 条 report budget 替代 phase 去重。下一轮 H20 只用这套修正版 first-diff 判定 root cause。
- 修正版 first-diff：4 并发 fixture 连续 4 轮中第 4 轮复现 row3 分叉。日志显示 `mla_projected_allreduce` 在 rank0..7 都出现同样 row0/row1 差异，且没有 `mla_projected` 本地差异报告；这不像 rank 私有状态或 stream race，更像 BF16 NCCL all-reduce 对不同 contiguous row offset 的归约顺序/舍入差异。诊断桥把 decode BF16 TP reductions 改为 F32 all-reduce：`bf16_hidden_to_f32_into -> all_reduce_f32_in_place -> f32_to_bf16_hidden_into`。
- F32 bridge H20 gate：第 0 轮 output16 四路全对，但第 1 轮 row0/row2/row3 仍变为 `[1008,2742,924,6454,2531,...]`；`ROWDIFF_COUNT=0` 说明 layer0 row-state 已经不再分叉。全层 `layer_output` 诊断显示第一处变脏在 layer1，8 rank 都看到一致的 row1 diff，随后误差逐层放大。layer1 细 phase 显示 first dirty phase 是 `moe_routed_reduce`，即 routed expert F32 combine all-reduce 后出现 row diff。
- Row-wise F32 collective H20 gate：decode hidden F32 bridge 和 routed F32 combine 改成按 active row 分别 all-reduce 后，4 并发 fixture `max_tokens=16` 连续 4 轮 greedy token ids 全部匹配 vLLM fixture。冷批 wall `4711.523ms`、`13.584 tok/s`；暖批 wall `922.795/924.405/923.877ms`、约 `69.2 tok/s`。日志仍有 `ROWDIFF_COUNT=256`，首段仍从 layer1 `moe_routed_reduce` 开始，典型差异 `first_abs_diff=0.000002861023`、`max_abs_diff=0.00003862381`。结论：row-wise collectives 让短 output gate 稳定，但没有消除 row-state 差异，且性能更差；后续不能把它当生产路径。
- Local routed cut H20 gate：新增 `moe_router_topk` 和 `moe_routed_local` 后，只跑 1 轮 4 并发 fixture `max_tokens=16`。输出 token ids 四路全对，wall `4858.721ms`、`13.172 tok/s`；`KIMI_DECODE_ROUTER_DIFF` 无输出，说明 layer1 router topk idx/weight 在同 token/同 position 的 active rows 间一致；第一批 row diff 已经出现在 `moe_routed_local`，即 `kimi_marlin_sum_topk_rows_f32` 之后、NCCL all-reduce 之前，典型 rank2 `first_abs_diff=0.0000038146973`、`max_abs_diff=0.000022888184`。结论：当前 layer1 row-state root cause 不在 NCCL routed combine 本身，而在本地 routed expert path，包括 route align、Marlin W13/W2、sum_topk、locks/output 清零或 scratch 复用。

## H20 decode rank-phase gate

- 去掉 decode per-step stream sync 后，H20 4 并发 fixture 在同一 server 内出现暖批单路发散：第二轮 `max_tokens=8` 有一路从第 3 个输出 token 变成 `[1008,2742,924,6454,...]`，`max_tokens=16` 也能复现同类单路偏移。
- batched top1 改为 deterministic CUDA argmax 后，冷批 `max_tokens=8` 仍四路一致，但暖批和 output16 仍会发散，说明问题不在 FlashInfer top-k row-state。
- `cudarc::nccl::safe::Comm` 使用的正是每个 rank worker 的同一条 CUDA stream；所以“同 stream 缺少 GPU sync”不是合理解释。当前定位转向 rank worker 到达 collective 的相位/尾部状态。
- 临时诊断补丁：decode 路径每次 BF16/F32 NCCL all-reduce 前执行同一个 CPU `Barrier`，只对齐 8 个 rank 的 collective enqueue，不做 `device_ctx.sync()`，不 drain GPU stream。
- H20 验证命令口径：`pegainfer-server` release binary，模型 `/data/models/Kimi-K2.5`，OpenAI `/v1/completions`，4 个并发请求，fixture 27-token prompt，`temperature=0`，`return_token_ids=true`。
- 历史结果：
  - `max_tokens=8` 冷批：wall `2093.9ms`，`15.3 tok/s`，四路一致 `[1008,2742,2531,414,19180,6082,1379,387]`。
  - `max_tokens=8` 暖批：wall `598.1ms`，`53.5 tok/s`，四路一致同上。
  - `max_tokens=16`：wall `787.3ms`，`81.3 tok/s`，四路一致 `[1008,2742,2531,414,19180,6082,1379,387,261,5216,63853,13,374,1765,11983,306]`。
  - 额外两轮 `max_tokens=16`：wall `792.4ms/784.8ms`，`80.8/81.6 tok/s`，均四路一致。
- 最新降级：CPU barrier 不能再被视为足够的端到端 correctness 护栏。2026-05-21 后续 H20 warm bs4 output16 复现两路坏前缀 `[1008,2742,924,6454,...]`，且 GEMM/report 校验没有报错；该坏前缀不再归因于 shared+routed reduce 合并实验独有问题，而是 Kimi decode row-state corruption 的复现签名。
- 当前 H20 状态：row-diff 诊断组跑完后已停止 server，确认无 `kimi-k2` tmux、port `18080` 空闲、`nvidia-smi --query-compute-apps` 无输出。后续 H20 GPU 使用等下一次可用窗口。

## Rejected: shared+routed reduce 合并

- 试验内容：MoE decode 中 shared expert `down_proj` 的 local BF16 输出不先做 BF16 all-reduce，而是累加到 routed expert local F32 buffer 中，与 routed contribution 合并成一次 F32 all-reduce，目标是每个 MoE 层少一次 BF16 collective 和一次 CPU barrier。
- 本地编译门通过，H20 release build 通过；短输出看起来有小幅收益，`max_tokens=16` 从约 `81 tok/s` 到 `85.7/87.1 tok/s`。
- 该改动不能保留：
  - H20 首批 4 并发 `max_tokens=22` 出现单路/双路 `[1008,2742,924,6454,...]` 冷批发散；
  - `max_tokens=64` 总吞吐约 `142.6 tok/s`，但四路不一致；
  - 长输出后同一 server 的后续短请求也出现不稳定，说明这不是可接受的数值噪声。
- 结论：减少 MoE collective cadence 是正确方向，但不能用这个 shared+routed 合并版本作为捷径。后续要么走 PPLX EP dispatch/combine，要么在更强的 vLLM/top-k parity gate 和 page/cache first-diff 工具下重新设计合并点；主线已恢复到 shared BF16 all-reduce + routed F32 all-reduce 的稳定 barrier 版本。

## DSV4 Flash 经验映射到 Kimi

- 不按 NCCL kernel duration 累加判断通信成本；要看一个 bs4 wave 的 logical step 和 rank arrival skew。Kimi 当前 `attention_allreduce_add` 与 `moe_reduce_add` 都可能混入 rank 到达等待。
- 小 helper 级 CUDA Graph、单个 stream handoff、单个 top1 wrapper 的收益很有限；DSV4 已证明 graph 必须覆盖较大的静态 decode block 才可能抵消 launch/API 成本。
- MoE routed path 的独立 compact/scatter kernel 不值得先做；只有融合进现有 routing/Marlin/reduce，或让 grouped/routed kernel 原生消费 sparse/padded metadata，才有机会拿到收益。
- CPU/worker placement 仍要沿用 DSV4 的 per-NUMA rank slice、CPU0/CPU1 保留策略；Kimi 当前已有 placement 骨架，但 PPLX worker 接入后还要用启动日志和 `/proc/<tid>/sched` 复核。
- Kimi 和 DSV4 的差异：Kimi routed expert 已是 vLLM Marlin WNA16 grouped path，GEMM 本体占比低于 route/shared/reduce 固定开销；所以优先级不是再换 CUTLASS/Triton GEMM，而是把 PPLX EP 与 MoE combine/dataflow 做成少 barrier、少 launch、少 host-visible 的路径。

## 端到端优先决策

- 2026-05-21 起，Kimi 下一阶段不再靠继续补内部 smoke 推进主线；direct worker/scheduler 不再保留 H20 smoke/candidate/perf 测试入口，历史 fixture 只作为外部对齐数据保留。
- 新增能力必须优先落到 H20 端到端路径：`pegainfer-server` / `bench_serving` / OpenAI-compatible `/v1/completions`，真实经过 frontend、scheduler、rank workers、权重、tokenizer 和 response stream。
- 端到端 gate 分两档记录：
  - `max_tokens=1`：证明当前 prefill-only 请求路径可以从 HTTP/bench 到真实 K2.5 权重返回 token，并与 vLLM greedy/top-k fixture 对齐。
  - `max_tokens>=2`：证明 prefill KV 能进入 decode state，decode step 使用真实 KV/cache/body 产出后续 token；这才进入 decode(bs4) 性能和 vLLM parity 主线。
- 后续修复顺序按端到端阻断点排：server 请求无法启动、tokenizer/prompt 不一致、scheduler 只返回 1 token、prefill KV 未保存、decode cache position/page metadata 不完整、PPLX EP 未接、sampling/logits 聚合不完整。

## Execution Log: 端到端优先切换

- 2026-05-21 本轮停止把新的内部 full-decode smoke 作为推进主线；本地实验代码不作为 H20 验证入口同步。
- H20 下一跳只跑真实请求路径：先验证 `max_tokens=1` 的 server/bench request 能经过 Kimi scheduler 和 8 rank worker 返回 token，再把 `max_tokens>=2` 的阻断点作为 decode 工作入口。
- 后续文档记录以端到端现象为准：请求能否启动、返回了多少 token、scheduler/worker 卡在哪个真实阶段、与 vLLM fixture 的 greedy/top-k 差异。内部 smoke 只保留为已有回归，不再新增成主线 gate。
- 主请求路径清理：`handle_request` 不再调用 `forward_prompt_smoke`，改走 `forward_prompt_next_token` / `ForwardPromptNextToken`；本轮临时 full-decode smoke 的 report、command、worker/scheduler 方法、H20 ignored gate 和 batch top1 helper 已删除。后续 direct crate 的 `#[cfg(test)]` 只保留 placement 与 page metadata 这类 CPU 小单测。
- 验证：本地 `cargo fmt --all --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过；H20 同步后 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests` 通过。远端 grep 确认 `forward_prompt_smoke` / `ForwardOneTokenSmoke` / full-decode smoke 残留为空。

## Execution Log: H20 server 端到端 gate

- H20 路径：`/root/develop/xingming/pegainfer-kimi-k2-main`，模型 `/data/models/Kimi-K2.5`，端口 `18080`，NCCL symlink 使用 `/tmp/pegainfer-nccl-lib/libnccl.so`。
- server 启动成功：`target/release/pegainfer --model-path /data/models/Kimi-K2.5 --port 18080`，engine load 日志 `elapsed_ms=131754`，OpenAI server 监听 `0.0.0.0:18080`。
- `logprobs=5` 请求当前失败：HTTP 500，错误为 `completion response requested logprobs but generation returned none`。这说明 frontend 已把 logprobs 需求传到 response 层，但 Kimi scheduler 还没有随 token 返回 logprob/top-k payload。
- raw text prompt gate：
  - 请求：`prompt="Hello"`、`max_tokens=1`、`temperature=0`、`return_token_ids=true`。
  - 响应：`prompt_token_ids=[19180]`，`token_ids=[950]`，text 为 ` |`，`completion_tokens=1`。
  - `max_tokens=2` 同样只返回 `token_ids=[950]`、`completion_tokens=1`，证明当前 scheduler 仍是一轮 next-token 结束。
- token-id prompt gate：
  - 独立字段 `prompt_token_ids` 不被当前 frontend schema 接受，错误是缺少 `prompt` 字段。
  - OpenAI completion 的 token-id prompt 形式 `prompt=[19180]` 可用，并返回同样的 `token_ids=[950]`。
- vLLM fixture prompt gate：
  - 输入读取 `/data/fixtures/kimi-k2/k25_hello_vllm/prompt.json` 的 27 个 `input_ids`，用 `prompt=<ids>` 发送到 `/v1/completions`。
  - `max_tokens=1` 返回 `token_ids=[1008]`、text `The`、`prompt_tokens=27`、`completion_tokens=1`，与既有 vLLM greedy token id `1008` 对齐。
  - `max_tokens=2` 仍只返回 `token_ids=[1008]`、`completion_tokens=1`。下一步主线阻断点明确为 scheduler token loop、prefill KV 保存和真实 decode step，而不是继续新增内部 smoke。
- server 验证后已停止，H20 GPU 已释放。

## Execution Log: scheduler token loop bridge

- 2026-05-21 根据 H20 server gate 的真实阻断点，`KimiK2DirectScheduler::handle_request` 改为按 `req.max_tokens` 循环发 token，而不是第一轮 next-token 后立刻 `Finished`。
- 当前循环实现是端到端 bridge：每步把已生成 token append 到上下文，再调用真实 full-prompt `forward_prompt_next_token`。它能让 OpenAI `/v1/completions` 先返回多 token，用于验证 frontend/scheduler/worker/response 链路；它不声称 decode 性能，也不替代 prefill KV + decode state。
- 后续替换点很清楚：第 1 步保留 prefill next-token；第 2 步开始把 full-prompt recompute 替换为 worker-owned decode arena、prefill KV 保存、真实 decode body、batched sampling 和 PPLX EP。
- 本轮验证注意事项：端到端 server 使用 `target/release/pegainfer`，改完 scheduler 后必须在 H20 跑 `cargo build --release -p pegainfer-server`，只跑 `cargo check` 会继续启动旧 binary。
- H20 验证：
  - `cargo fmt --all --check` 通过。
  - `cargo check --release -p pegainfer-server` 通过。
  - `cargo build --release -p pegainfer-server` 通过。
  - 重新启动 server 后，fixture 27-token prompt 的 `max_tokens=2` 返回 `token_ids=[1008,2742]`、text `The user`、`prompt_tokens=27`、`completion_tokens=2`；server 日志记录 `output_tokens=2`。
  - 验证后 server 已停止，H20 GPU 已释放。

## Execution Log: direct smoke cleanup 和 decode page table

- 2026-05-21 本轮把 Kimi direct worker/scheduler 里的旧 H20 smoke/candidate/debug 入口清掉：
  - 删除 worker command/report：`ForwardOneTokenLogitsShard`、`ForwardDecodeLayer0TokensSmoke`、`KimiOneTokenLogitsShard`、`KimiDecodeMlaLayerSmokeReport`。
  - 删除 worker debug dump：rank0 layer0 MLA safetensors、layer1 MoE safetensors、host D2H debug helpers。
  - 删除 scheduler H20 ignored tests：all-rank one-token、layer0 decode、candidate dump、多 prompt candidate、prompt perf。
  - direct crate 现在只保留 CPU 小单测：TP8/EP8 placement 和 decode page metadata。
- `page_size=16` 结论已落实到代码：`16` 是每页 token 数，不是最大上下文长度。`KimiWorkerDecodeArena` 现在按 `batch_size=4`、`pages_per_request=128` 分配 `max_pages=512`，每个 request 可覆盖 `2048` 个 token 位置。
- page table 初始化从旧的“一请求一页且 position=batch_idx”改为按 append position 计算：
  - `append_position=26` 表示 cache 写到第 27 个 token，page table 为 request0 的 page `0,1`，`last_page_len=11`。
  - `append_position=27` 表示下一个 decode token 写入第 28 个位置，仍是 2 pages，`last_page_len=12`。
  - bs4 的每个 request 使用独立 page range：request0 从 page `0` 开始，request1 从 page `128` 开始，避免不同请求 page id 混用。
- RoPE cache 从旧的 `page_size` 长度改为 `page_size * pages_per_request = 2048`，否则 position `26/27` 这种两页上下文虽然 page table 合法，RoPE lookup 仍会越界或读错。
- 验证：
  - 本地 `cargo fmt --all --check` 通过。
  - 本地 `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - `rg` 确认 `pegainfer-kimi-k2/src/direct/{worker,scheduler}.rs` 里 `smoke/candidate/debug/dump/perf` 只剩 `debug_assert`。
  - H20 dry-run 后同步本轮 5 个文件：`pegainfer-kimi-k2/src/direct/{worker,scheduler}.rs`、`docs/index.md`、`docs/models/kimi-k2/{operator-todo,support-analysis}.md`。
  - H20 `/root/develop/xingming/pegainfer-kimi-k2-main` 验证通过：`cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server`。

## Execution Log: prompt prefill 写入 MLA paged KV

- 2026-05-21 本轮把 worker-owned decode arena 从“只分配”推进到“prompt prefill 后持有真实上下文 KV”：
  - `KimiWorkerDecodeArena::configure_single_request_prefill(seq_len)` 在每次 prompt forward 开始时同步 page table、`batch_indices`、`positions`，request0 覆盖完整 prompt，bs4 其他 slot 保持 1-token padding page。
  - `batch_indices_d` / `positions_d` 从 bs4 长度扩成 `batch_size * 2048` append metadata capacity，prefill 可一次 append 最多 2048 token/request。
  - 新增 `kimi_mla_rope_apply_kpe_cuda` / `kimi_mla_rope_apply_kpe`：专门把 prefill 的 raw `k_rope [seq,64]` 按 device positions 与 YARN RoPE cache 转成 `append_kpe [seq,64]`，避免复用 decode split kernel 额外计算无用 q。
  - 每层 MLA prefill 在得到 `compressed_normed [seq,512]` 后，立即调用 `kimi_mla_paged_kv_append` 写入该层 `ckv_cache/kpe_cache`。写入的是 RoPE 后 KPE，不是 raw `k_rope`。
- 该阶段仍保留 full-prompt recompute bridge：第 2 个 token 会重跑完整 prompt，但每次重跑都会刷新 arena 中的完整上下文 KV。后续的 scheduler true decode bridge 已把第 2 步改成读取这份 KV 的 decode body。
- 本地验证：
  - `cargo fmt --all --check` 通过。
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
- H20 验证：
  - 同步代码后，`cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server` 通过。
  - 启动 `target/release/pegainfer --model-path /data/models/Kimi-K2.5 --port 18080`，真实模型 load `131396ms`。
  - vLLM fixture 27-token prompt 仍然对齐：`max_tokens=1` 返回 `token_ids=[1008]`、text `The`；`max_tokens=2` 返回 `token_ids=[1008,2742]`、text `The user`。
  - 本次 HTTP payload 的 `model` 字段必须用 server 暴露的 `"/data/models/Kimi-K2.5"`；误写成 `"kimi-k2.5"` 会被 frontend 以 404 拒绝。
  - 验证后 server 已停止，`nvidia-smi --query-compute-apps` 无 GPU compute 进程。

## Execution Log: scheduler wave bs4 decode 和 scratch 预分配

- 2026-05-21 本轮把 scheduler token loop 从单请求 decode bridge 改成最多 4 请求一组的 wave decode：
  - 每个请求先在自己的 slot 上走 `forward_prompt_next_token_in_slot(slot, prompt_tokens)`，负责 prompt forward、MLA prefill KV 写入和第一个 greedy token；
  - 第 2 个 token 起统一调用 `forward_decode_batch_next_tokens(token_ids, append_positions, slots)`，其中 `append_position = prompt_len + completion_tokens - 1`；
  - scheduler 等当前 wave 完成后再接下一组请求，当前不是 continuous batching。
- 当前代码入口：
  - `KimiRankCommand::ForwardDecodeBatchNextTokens`
  - `KimiRankWorker::forward_decode_batch_next_tokens_async`
  - `KimiRankThreadState::forward_decode_batch_next_tokens`
  - `KimiWorkerDecodeArena::{configure_slot_prefill, configure_batch_decode, upload_batch_tokens, copy_logits_slot}`
- Decode body scratch 改动：
  - dense MLP 的 gate/up/activated、MoE shared expert 中间态、router logits/scores/topk、Marlin route workspace、Marlin WNA16 workspace、W13/W2 中间态、routed f32 reduce 和 top1 scratch 都挪进 worker-owned decode arena；
  - Marlin WNA16 locks、W13 output、W2 output、routed f32 buffer 复用前必须 `memset_zeros`。旧代码每层新建 zero buffer，复用 scratch 后不清零会让非本地 route / padding route 带入 stale 值，H20 表现为 `max_tokens=8` 从第二个 token 起发散。
- 本地验证：
  - `cargo fmt --all --check` 通过。
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
- H20 验证：
  - 同步前 rsync dry-run 覆盖 `pegainfer-kimi-k2/src/direct/worker.rs`，后续文档同步单独 dry-run；
  - `/root/develop/xingming/pegainfer-kimi-k2-main` 下 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server --bin pegainfer --bin bench_serving` 通过。
  - 启动 `target/release/pegainfer --model-path /data/models/Kimi-K2.5 --port 18080`，engine load `131661ms`。
  - 4 并发 vLLM fixture 27-token prompt：
    - `max_tokens=2`：四路均返回 `token_ids=[1008,2742]`、text `The user`，wall `1979.3ms`；
    - `max_tokens=8`：四路均返回 `token_ids=[1008,2742,2531,414,19180,6082,1379,387]`、text `The user said "Hello". This is`，wall `563.5ms`，32 output tokens，HTTP 端到端输出吞吐 `56.8 tok/s`。
  - 验证后 server 已停止，`nvidia-smi --query-compute-apps` 无 GPU compute 进程。
- 当前仍未达到 decode(bs4) 目标：
  - local top1 仍有 host-visible D2H；
  - EP combine 仍是 NCCL-sum correctness bridge；
  - scheduler 是 wave batching，不是 continuous batching；
  - prompt prefill 仍串行进入 wave，HTTP output8 不是纯 decode microbench；
  - TP/EP collectives 还不是 PPLX/graph-ready 路径。

## Decode operator 当前落点

- MLA decode kernel 边界已按 FlashInfer 的真实接口拆开：模型侧先做 `q_nope @ W_UK_T -> q_abs_nope [B,8,512]`，kernel 只消费 `q_abs_nope`、`q_pe [B,8,64]`、paged compressed KV 和 plan arrays，输出 latent `[B,8,512]`；模型侧随后做 `W_UV [8,512,128]` v-up。
- Decode-step q/k 准备也已落到 kernel：`kimi_mla_rope_split_decode_cuda` 从 `q_proj [B,8,192]`、当前 `k_rope [B,64]`、device positions 和常驻 RoPE cache 生成 `q_nope [B,8,128]`、`q_pe [B,8,64]`、`append_kpe [B,64]`，沿用 prefill 已验证的 Kimi split-half RoPE layout。
- `q_abs_nope` absorption 与 `v_up` 已补 graph-safe cuBLAS strided-batched GEMM wrapper，直接复用常驻 `kv_b_proj [8,256,512]`：前 128 行是 `W_UK`，后 128 行是 `W_UV`，每个 local head 一个 cuBLAS batch，不新增权重重排。
- `kimi_mla_paged_kv_append_cuda` / `ops::kimi_mla_paged_kv_append` 写入 `ckv [page,page_size,512]` 与 `kpe [page,page_size,64]`，输入是 device `batch_indices/positions`，不要求 host route readback。Rust layout 显式携带 ckv/kpe page stride 和 token stride，后续可在 separate buffer 与 concat `[512+64]` strided view 间切换。
- `kimi_flashinfer_batch_decode_mla_cuda` / `ops::kimi_flashinfer_batch_decode_mla` 走 non-partition KV 第一版：`request_indices`、`kv_tile_indices`、`kv_chunk_size_ptr` 仍作为 GPU plan arrays 传入，`tmp_v/tmp_s/o_indptr/block_valid_mask` 留给 split-K / CUDA Graph padding 后续接入。
- 本地编译门：`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels --tests` 已通过。H20 gate：`h20_kimi_flashinfer_batch_decode_mla_bs4_smoke` 已在 `host-10-96-191-100` 通过；当前 gate 扩展为 `q_proj/k_rope split+RoPE -> q_abs -> paged append -> MLA decode -> v_up`，只验证 wrapper、bs4 launch 和 finite output，不声称 vLLM parity。
- Worker 侧已开始消费这些 decode ops：`KimiWorkerDecodeArena` 在 rank 权重 load 完成后常驻 bs4 的 MLA paged ckv/kpe cache、device plan arrays、YARN RoPE cache 和 scratch；`forward_mla_decode_layer_into` 复用真实 attention 权重执行 `input_norm -> q_a/q_b -> kv_a/split/norm -> decode split+RoPE -> q_abs -> paged append -> FlashInfer MLA decode -> v_up -> o_proj`。旧 direct H20 layer0 smoke/candidate 入口已经删除；下一步不是恢复内部 smoke，而是把这个 staged decode body 接到真实 server request 的 prefill KV + decode token loop。

## INT4 routed expert 当前结论

- Kimi checkpoint / vLLM pack 语义已经核对：`compressed_tensors` 官方 `pack_to_int32` 接收 signed int4 值，落盘 nibble 是 offset-binary `value + 8`；`unpack_from_int32` 返回 signed 值。view 成 bytes 后，低 nibble 是偶数 `in_col`，高 nibble 是奇数 `in_col`。
- CPU reference 读取 nibble 时必须做 `signed = unsigned - 8`。manual vs official 恒差 `8` 正是 signed/unsigned 解释差异，不是 scale layout 差异。
- CUTLASS package 阶段的 `xor 0x88` 是 offset-binary nibbles 到 `cutlass::int4b_t` signed storage 的转换，前提是只执行一次。当前 focused H20 probe 证明 nibble 路径本身不是首个错误来源。
- vLLM Marlin WNA16 使用 `uint4b8` 表示 signed symmetric INT4 的 bias=8 编码；Marlin weight repack 保留 unsigned nibble，不执行 `xor 0x88`。当前本仓库 Marlin package 从 checkpoint `[expert,out,K/8] int32` 融合 transpose + no-actorder repack 成 `[expert,K/16,N*2] int32`。
- vLLM runtime compute ABI 不吃独立 gate/up 两个 Marlin package；W13 必须在 load/package 阶段融合成 `gate_then_up`。本仓库现在只把独立 gate/up 当临时中间态，最终常驻 package 是 fused W13 + W2。
- Scale layout 已拆开记录：
  - checkpoint / FlashInfer MxInt4 monolithic：`[expert, out, in_group]`，也就是 Kimi safetensors 的 `[out, in/32]` 按 expert stack；
  - CUTLASS example69：物理上吃 group-major `[expert, in_group, out]`，当前 `kimi_cutlass_int4_reorder_scale_sm90a_cuda` 只做 transpose，不做 Marlin permutation；
  - vLLM Marlin WNA16：加载时先形成 group-major `[expert, in_group, out]`，再按 `marlin_moe_permute_scales` 对 flat group-major buffer 做 64-block `scale_perm`；本仓库 `kimi_marlin_int4_reorder_scale_cuda` 从 checkpoint `[expert,out,in_group]` 直接生成单投影 group-major+perm64 buffer，再把 gate/up 沿 out 维融合成 W13 `[expert,in_group,4096]`。
- CUTLASS example69 BF16 scale-only Hopper grouped GEMM 的 `TileShapeK=64`，而 Kimi scale group 是 `32`。它在一个 K tile 内需要两组 BF16 scale，但 example69 的 scale reload 语义不能表达 Kimi `[out, col / 32]`。把 scale 在 checkpoint `[out, group]` 与 CUTLASS group-major `[group, out]` 之间转置都不能修正这个语义。
- `TileShapeK=32` 不是可行补丁：H20/sm90a 本地编译触发 CUTLASS static assertion `K_BLOCK_MAX >= 4`。因此当前路线不是继续调 scale layout，而是换 backend。
- H20 focused probe：`h20_kimi_cutlass_int4_example69_rejects_per32_scale_semantics` 会构造 one-hot input、指定 nibble、非均匀 scale，并断言 example69 与 Kimi per32 scale 语义不匹配。2026-05-21 H20 结果显示 col `0/1/31` 的 signed nibble 与 group0 scale 都匹配；col `32/33` 仍使用 group0 scale，col `64` 使用 group1 scale，证明 example69 实际按 64-wide K tile 换 scale。旧 broad synthetic 只保留为 smoke，不是 correctness gate。

## Execution Log: signed/unsigned 与 Marlin package split

- 复核 vLLM/official 路径：
  - `compressed_tensors` pack 是 signed int4 输入、offset-binary 落盘；manual 与 official 恒差 `8` 来自是否减去 bias，不是 scale transpose。
  - CUTLASS example69 package 继续只在 CUTLASS path 做一次 `xor 0x88`，把 offset-binary nibble 转成 signed `cutlass::int4b_t` storage。
  - vLLM Marlin WNA16 使用 `uint4b8` bias=8 语义，weight repack 必须保留 unsigned nibble。
- 代码改动：
  - `KimiInt4WeightManifest` 现在同时记录 packed-weight checkpoint / CUTLASS signed-reordered / Marlin uint4b8 no-actorder 三种 layout spec；scale 继续保持 checkpoint / CUTLASS / Marlin 三种 layout spec。
  - `KimiExpertMajorProjectionKernelBuffers` 的 runtime package 字段改成显式 `weight_packed_cutlass_example69` / `weight_scale_cutlass_example69`，避免后续 Marlin/WNA16 接入时把 CUTLASS group-major scale buffer 误当成 Marlin group-major+perm64 scale buffer。
  - 新增 `kimi_marlin_int4_reorder_weight_cuda` + Rust wrapper `kimi_marlin_int4_reorder_weight`，把 checkpoint `[expert,out,K/8] int32` repack 成 vLLM no-actorder Marlin `[expert,K/16,N*2] int32`，总字节数不变，不做 signed xor。
  - Marlin scale metadata 改成 `expert_major_group_scale_marlin_group_major_perm64`：shape 仍是 vLLM `[expert,in_group,out]`，64-block `scale_perm` 是 flat group-major buffer 内部重排，不再伪装成 `in_group` 轴本身被重排。
  - `manifest_call()` 的 CUTLASS grouped projection 输入改用 CUTLASS signed-reordered packed spec，避免把 checkpoint raw package 描成 runtime package。
- 验证：
  - 本地：`cargo fmt --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels --tests`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20：rsync 先 dry-run，再同步 8 个代码/文档文件；`cargo fmt --check`、`cargo check --release -p pegainfer-kernels --tests`、`cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20 ignored gate：`h20_kimi_marlin_weight_repack_matches_vllm_noact_layout` 和 `h20_kimi_marlin_scale_reorder_matches_vllm_permute` 均通过。

## Debrief: signed/unsigned 与 Marlin package split

- **Outcome**: signed/unsigned 差异已被限制在 nibble decode/package contract 内；scale layout 不再背锅。CUTLASS signed path 与 Marlin uint4b8 path 现在在 manifest、FFI、wrapper、文档中分开。
- **Pitfalls encountered**:
  - Marlin 和 CUTLASS 对同一 checkpoint nibble 的语义不同：CUTLASS path 需要 signed storage，Marlin path 消费 bias=8 `uint4b8`。把二者共用一个 “reordered weight” 名字会直接制造后续 parity 噪音。
- **Follow-ups**:
  - 接入真正 WNA16/Marlin grouped expert compute backend 后，用外部 Torch/vLLM fixture 做 routed expert parity；当前 package gates 只证明 layout，不声称数值 parity。

## Execution Log: Marlin W13 fused scale package

- 复核 vLLM `CompressedTensorsWNA16` / `fused_marlin_moe` runtime ABI：
  - `moe_wna16_marlin_gemm` 第一次 GEMM 吃 fused `w13_weight`，Kimi shape 是 `[48, 448, 8192]` int32 view，也就是 `[expert,K/16,(2*2048)*2]`；
  - `w13_scale` shape 是 `[48,224,4096]`，layout 是 group-major+perm64；
  - W2 维持 `[48,128,14336]` packed weight 与 `[48,64,7168]` scale。
- 代码改动：
  - 新增 `kimi_marlin_int4_fuse_w13_cuda`，把已经 repack/permute 好的 gate/up 单投影 package 沿最后一维融合成 vLLM runtime W13 package；
  - `KimiMoeLayerExpertMarlinWeights` 常驻态从 `gate+up+down` 改成 `w13+down`，gate/up 只作为 fuse 前的临时 buffer，函数返回前释放；
  - Rust manifest 增加 `KimiMarlinFusedW13Int4Weight`，显式记录 `gate_then_up`、`vllm_w13_group_major_perm64`。
- 约束：
  - 这仍是 package ABI，不是数值 parity；compute wrapper 还缺 `moe_wna16_marlin_gemm` 等价实现。
- 验证：
  - 本地：`cargo fmt --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels --tests`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20：rsync 先 dry-run，再同步本轮代码/文档；`cargo fmt --check`、`cargo check --release -p pegainfer-kernels --tests`、`cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20 ignored gate：`h20_kimi_marlin_weight_repack_matches_vllm_noact_layout`、`h20_kimi_marlin_scale_reorder_matches_vllm_permute`、`h20_kimi_marlin_align_block_size_matches_vllm_contract`、`h20_kimi_k25_rank0_marlin_expert_package_loads`、`h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view` 均通过。

## Execution Log: Marlin route alignment metadata

- 新增 `kimi_moe_marlin_align_block_size_cuda`，输出 vLLM Marlin/WNA16 所需的 `sorted_token_ids`、`expert_ids`、`num_tokens_post_padded`。
- Rust 侧新增 `KimiMarlinRouteWorkspace` / `KimiMarlinRouting` / `kimi_moe_marlin_align_block_size`，capacity 按 vLLM `topk_ids.numel() + local_experts * (block_size - 1)` 规则预分配；decode step 内只复用 workspace，不分配、不 D2H。
- alignment 语义按 H20 Kimi EP 本地 rank：只保留 `[global_start, global_start+48)` 的本地 experts，非本地 experts 被忽略，对齐到 block size `8/16/32/48/64`，padding sentinel 是 `active_tokens * topk`。
- 验证：
  - 本地：`cargo fmt --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels --tests`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20：rsync 先 dry-run，再同步 7 个代码/文档文件；`cargo fmt --check`、`cargo check --release -p pegainfer-kernels --tests`、`cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20 ignored gate：`cargo test --release -p pegainfer-kernels h20_kimi_marlin_align_block_size_matches_vllm_contract -- --ignored --nocapture` 通过，覆盖 bs4/active7、非本地 expert 忽略、per-expert block padding、sentinel、`expert_ids` block mapping。
- 这一步仍是 metadata gate，不声称 routed expert 数值 parity；下一步才是把 Marlin WNA16 compute ABI 接上。

## Execution Log: Marlin WNA16 compute wrapper

- 复核 H20 reference venv：`/root/develop/xingming/vllm_test/.venv` 是 vLLM `0.19.0`，`torch.ops._moe_C.moe_wna16_marlin_gemm` schema 没有 `is_ep`，但有 `a_scales`、`global_scale`、`thread_k/thread_n/blocks_per_sm`。不要再用较新的 `/root/develop/yingshan/vllm` Marlin header 作为当前 fixture ABI。
- vendored csrc 改成 vLLM 0.19.0 ABI：`pegainfer-kernels/csrc/kimi_k2/vllm_marlin/moe/marlin_moe_wna16/{kernel.h,marlin_template.h}` 来自 `/data/code/vllm-int`；`quantization/marlin/{marlin.cuh,marlin_dtypes.cuh,dequant.h,marlin_mma.h}` 保留 standalone 编译，移除 PyTorch/ATen include。
- 新增 Kimi wrapper：`kimi_marlin_wna16_gemm_cuda`、`kimi_marlin_w13_swiglu_cuda`、`kimi_marlin_sum_topk_rows_f32_cuda`；Rust 暴露 `kimi_marlin_wna16_w13_gemm`、`kimi_marlin_wna16_w2_gemm`、`kimi_marlin_w13_swiglu`、`kimi_marlin_sum_topk_rows_f32`。
- vLLM fixture 生成器：`pegainfer-kernels/tools/kimi_k2/vllm_marlin_wna16_reference.py` 用 H20 vLLM op 生成 W13、W2 route output、final reduce 的 BF16 raw fixture。默认生成 synthetic fixture；传 `--model-path /data/models/Kimi-K2.5 --layer-idx 1 --rank 0` 时，直接从真实 checkpoint 读取 rank-local 48 个 experts 的 W13/W2 packed weight 与 scale，按 vLLM `gptq_marlin_moe_repack` / `marlin_moe_permute_scales` 生成 reference。
- `pegainfer-kimi-k2/src/weights.rs` 已按 Qwen3-4B flat module 风格拆成 `weights.rs` + `weights/{context,load,manifest,package,tests}.rs`，旧 CUTLASS raw/kernel package helper 和自写 dequant+cuBLAS self-comparison gate 已移除；当前 weights gate 只保留 Marlin package、真实 vLLM fixture parity、typed view 和 loader contract。
- H20 验证：
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels --tests` 通过。
  - `/root/develop/xingming/vllm_test/.venv/bin/python pegainfer-kernels/tools/kimi_k2/vllm_marlin_wna16_reference.py --out-dir /tmp/kimi_marlin_wna16_reference --tokens 4 --block-size 8` 通过。
  - `cargo test --release -p pegainfer-kernels h20_kimi_marlin_wna16_single_layer_matches_vllm_reference -- --ignored --nocapture` 通过，`w13_out`、`route_output`、`final` 的 max/mean diff 均为 `0`。
  - `/root/develop/xingming/vllm_test/.venv/bin/python pegainfer-kernels/tools/kimi_k2/vllm_marlin_wna16_reference.py --model-path /data/models/Kimi-K2.5 --layer-idx 1 --rank 0 --out-dir /data/fixtures/kimi-k2/k25_rank0_layer1_marlin_vllm --tokens 4 --block-size 8` 通过。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_rank0_layer1_marlin_wna16_matches_vllm_reference -- --ignored --nocapture` 通过，真实 K2.5 rank0 layer1 的 `w13_out`、`route_output`、`final` 全部 `max_diff=0` / `mean_diff=0`。
- 这一步证明真实 K2.5 单层 routed expert 的 Marlin WNA16 package + compute 链路与 vLLM fixture 对齐；full-forward parity 仍以多 prompt vLLM top-k gate 为准，decode(bs4) 仍需要 KV/cache 与 batch decode body。

## Execution Log: Marlin runtime package in loader

- loader 现在保留两条互斥 package 路线，不在同一个 full-rank runtime state 里同时常驻 CUTLASS probe package 和 Marlin/WNA16 package：
  - CUTLASS probe package：`weight_packed_cutlass_example69`、`weight_scale_cutlass_example69`、`weight_shape`；
  - Marlin/WNA16 package：`weight_packed_marlin_uint4b8`、`weight_scale_marlin_permuted`。
- Marlin package 阶段从 checkpoint raw GPU tensors 生成 vLLM WNA16 layout：
  - Marlin weight 复用 `kimi_marlin_int4_reorder_weight`，保留 vLLM `uint4b8` bias=8 nibble；
  - Marlin scale 复用 `kimi_marlin_int4_reorder_scale`，把 checkpoint `[expert,out,in_group]` 转为 vLLM group-major+perm64 `[expert,in_group,out]`。
- 显存统计拆成 `raw_source_bytes` 与 `total_bytes`：前者用于证明 checkpoint raw tensors 被统一移除，后者表示实际 runtime package footprint。Marlin package 不保存 `weight_shape`，所以 `total_bytes < raw_source_bytes`。
- 新增 `as_marlin_weights()` view，后续 WNA16 compute ABI 可以直接消费 runtime-owned Marlin package，不再从 checkpoint raw package 临时转换。
- 2026-05-21 H20 验证：
  - `h20_kimi_marlin_scale_reorder_matches_vllm_permute` 通过，确认 Marlin scale package 与 vLLM `marlin_moe_permute_scales` 的 group-major+perm64 语义一致。
  - `h20_kimi_k25_rank0_marlin_expert_package_loads` 通过，真实 `/data/models/Kimi-K2.5` rank0 60 个 MoE layer 可打成 fused W13 + W2 Marlin-only package，`total_bytes < raw_source_bytes`，且 raw routed expert tensors 被移除。
  - `h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view` 通过，确认 fused W13 改动没有打断真实权重 loader / typed GPU view / CUTLASS probe 路线，且未回退到双 package 常驻 OOM。

## Execution Log: worker backend 切到 Marlin WNA16

- worker 的 MoE layer runtime 已从 expert-major CUTLASS probe path 切到 vLLM Marlin WNA16：
  - router 仍用 `kimi_router_noaux_tc_launch` 产 top-k；
  - route metadata 改用 `KimiMarlinRouteWorkspace` / `kimi_moe_marlin_align_block_size`；
  - W13 使用 fused `gate_then_up` Marlin package，W2 使用 Marlin down package；
  - SwiGLU 与 top-k reduce 使用 `kimi_marlin_w13_swiglu` / `kimi_marlin_sum_topk_rows_f32`。
- 修正 zero-local-route 情况：vendored Marlin kernel 在 `num_tokens_post_padded <= 0` 时直接 return，输出保持预分配 zero buffer 语义；否则 rank 无本地 route 时会把其它 rank 留在 collective。
- NCCL correctness bridge：
  - comm 生命周期改为 load/package 完成后创建，再 attach 到 worker，跟 DSV4/Qwen3 的权重→comm 生命周期一致；
  - 第一轮 vocab-shard embedding all-reduce 前增加 rank barrier 和 stream drain。H20 上无 stream drain 时首个 collective 会报 `ncclUnhandledCudaError`；这是当前 NCCL-sum bridge 的约束，不是最终 PPLX/graph 形态。
- H20 验证：
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_all_rank_one_token_vocab_shard_top1_smoke -- --ignored --nocapture` 通过，真实 `/data/models/Kimi-K2.5`，8 rank，全 61 层，`attention_layers_stubbed=0`、`remaining_layers_stubbed=0`、`moe_layers_executed=60`。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_dump_all_rank_one_token_candidate_logits -- --ignored --nocapture` 通过，写出 `/data/fixtures/kimi-k2/pegainfer_k25_smoke_logits/candidate.safetensors`。
- vLLM top-20 gate：
  - fixture：`/data/fixtures/kimi-k2/k25_hello_vllm/reference.safetensors` 与 HTTP fixture `/data/fixtures/kimi-k2/k25_hello_vllm_http/response.json`。
  - prompt len `27`，vLLM greedy token id `1008`。
  - PegaInfer candidate metadata：argmax id `1008`，argmax logit `24.875`，全 8 个 vocab shard considered。
  - top-20 id overlap `19/20`；PegaInfer top ids 前 8 为 `[1008, 19180, 4052, 18699, 3479, 2512, 16, 40]`，与 vLLM top ids 前 8 一致。
- vLLM 多 prompt gate：
  - fixture：`/data/fixtures/kimi-k2/k25_parity_vllm/{cases.json,hello,math_short,self_intro_zh,code_rust}`。
  - PegaInfer candidate：`/data/fixtures/kimi-k2/pegainfer_k25_parity_candidates/{cases.json,hello,math_short,self_intro_zh,code_rust}`。
  - `compare_vllm_topk_fixture.py --top-k 20 --require-argmax --min-overlap 16` 通过。
  - 结果：`hello` argmax `1008` overlap `19/20`；`math_short` argmax `1008` overlap `20/20`；`self_intro_zh` argmax `4052` overlap `20/20`；`code_rust` argmax `1008` overlap `19/20`；vLLM top-k 上最大 logprob diff `0.749978`。
- Prefill perf smoke：
  - 入口：`h20_kimi_k25_prompt_forward_perf_smoke`，同一次真实 K2.5 runtime load 后跑 `hello_27`、`synthetic_128`、`synthetic_512`、`synthetic_1024`。
  - 输出：`/data/fixtures/kimi-k2/pegainfer_k25_perf_smoke/summary.json`。
  - scope 明确为当前 correctness path：full prompt forward、NCCL-sum bridge、per-layer temporary allocation、host-visible final top1。
  - H20 结果：`hello_27` avg `103.17ms` / `261.70 tok/s`；`synthetic_128` avg `117.19ms` / `1092.20 tok/s`；`synthetic_512` avg `212.75ms` / `2406.63 tok/s`；`synthetic_1024` avg `358.26ms` / `2858.23 tok/s`。
  - 结论：prefill 目标在 128+ synthetic prompt 上已有初步余量，但这还不是 graph-ready/server perf gate；短 prompt 仍被固定开销主导。
  - 这里的历史 summary 已被新的 bs4 wave decode gate 取代；当前已在 H20 server 4 并发 `max_tokens=8` 跑通，最新吞吐记录见上面的 scheduler wave bs4 decode 小节。

## 外部 logits fixture

- HF raw-logits fixture 入口：`pegainfer-kernels/tools/kimi_k2/hf_logits_reference.py`
  - 读取本地模型目录和 prompt payload。
  - 使用模型目录 tokenizer remote code 渲染 chat template。
  - 保存 `reference.safetensors`：`input_ids`、`logits_f32`、`topk_ids`、`topk_logits_f32`、`topk_logprobs_f32`、`argmax_id`。
  - 保存 `metadata.json`：engine、模型路径、prompt、input ids、sha256、依赖版本。
- vLLM serving fixture 入口：`pegainfer-kernels/tools/kimi_k2/vllm_logits_reference.py`
  - 保存 generated token ids 和 top-logprobs；它不作为 raw logits diff 的权威来源。
  - 支持 `--prompt-set-json`，用同一个 vLLM `LLM` load 批量生成多个 prompt case，并写出 root `cases.json`。
  - H20 只用完整 native venv 调它：`/root/develop/xingming/vllm_test/.venv/bin/python`，并把同一 venv 的 `bin` 放入 `PATH`，让 FlashInfer JIT 找到对应的 `ninja`。
  - 不再拼 `PYTHONPATH` overlay；`/root/develop/yingshan/vllm/.venv` 是 vLLM dev 源码环境，但当前 registry 没有 `KimiK25ForConditionalGeneration`，不是 K2.5 fixture 首选。
  - vLLM `0.19.0` 的 SamplingParams 对 sample `logprobs` 上限是 `20`，所以 serving fixture 用 `--top-k 20`。当前第一版 parity gate 使用 top-20 即可；top128/full logits reference 后置到真实 forward 已经跑通以后。
  - 已生成 H20 fixture：`/data/fixtures/kimi-k2/k25_hello_vllm`，模型 `/data/models/Kimi-K2.5`，TP8，thinking=true，prompt `"Hello"`，seq_len `27`，generated token id `1008`，`top_k_returned=20`。
  - 已生成 H20 多 prompt fixture：`/data/fixtures/kimi-k2/k25_parity_vllm`，cases `hello/math_short/self_intro_zh/code_rust`，generated token ids `1008/1008/4052/1008`，`top_k_returned=20`。
  - 已生成 H20 HTTP fixture：`/data/fixtures/kimi-k2/k25_hello_vllm_http`，同一 rendered prompt，OpenAI-compatible `/v1/completions`，`temperature=0`、`max_tokens=1`、`logprobs=20`、`return_token_ids=true`。响应生成 `" The"`，token id `1008`，top logprob `-0.001139111`，prompt token ids 与 Python API fixture 一致。
- PegaInfer 候选 logits 消费入口：`pegainfer-kernels/tools/kimi_k2/compare_logits_fixture.py`
  - 只读取 HF `hf_remote_code` reference。
  - 只比较候选 full-vocab `logits_f32`，报告 argmax、top-k order/overlap、full logits max/mean abs diff。
- PegaInfer vs vLLM top-k fixture 入口：`pegainfer-kernels/tools/kimi_k2/compare_vllm_topk_fixture.py`
  - 支持单 case `--reference-dir/--candidate` 和 batch `--reference-root/--candidate-root`。
  - 只把 vLLM generated token/top-logprob ids 当外部 serving reference；candidate 仍必须是 PegaInfer full-vocab `logits_f32`。
  - 输出 argmax match、top-k overlap/order、candidate logits、candidate logprobs at vLLM top ids；gate 参数为 `--require-argmax --min-overlap 16`。
- PegaInfer 当前 smoke candidate：
  - `h20_kimi_k25_dump_all_rank_one_token_candidate_logits` 会读取 vLLM prompt fixture 的 27 个 input ids，让 8 rank 执行 prompt forward，并把 8 个 `[20480]` vocab shard 拼成 `[163840] logits_f32`。测试名仍保留旧的 `one_token` 字样，语义已经变成 prompt last-token candidate dump，后续可改名。
  - 输出：`/data/fixtures/kimi-k2/pegainfer_k25_smoke_logits/candidate.safetensors` 和 `metadata.json`。
  - 旧 candidate argmax 是 token id `154473`，logit `8.9375`；metadata 明确 `parity_claim=false`，因为当时 attention 仍是 stub 且只执行 layer0 dense + layer1 MoE smoke。MLA/full-forward 版本必须在 H20 重新生成 candidate。
- PegaInfer 多 prompt candidate：
  - `h20_kimi_k25_dump_parity_prompt_candidates` 读取 vLLM root `cases.json`，同一个 PegaInfer runtime load 依次跑每个 prompt，并写出 `/data/fixtures/kimi-k2/pegainfer_k25_parity_candidates`。
  - 每个 case 仍写 full-vocab `candidate.safetensors`，root `cases.json` 记录 reference/candidate path、seq_len 和 argmax。
- 已验证：
  - `uv run --no-project python -m py_compile pegainfer-kernels/tools/kimi_k2/hf_logits_reference.py pegainfer-kernels/tools/kimi_k2/vllm_logits_reference.py pegainfer-kernels/tools/kimi_k2/compare_logits_fixture.py pegainfer-kernels/tools/kimi_k2/compare_vllm_topk_fixture.py` 通过。
  - 2026-05-21 H20 native vLLM fixture 生成通过；日志 `/tmp/kimi_k25_vllm_fixture_top20_20260521_030017.log` 显示 `Resolved architecture: KimiK25ForConditionalGeneration`、`FLASH_ATTN_MLA`、Marlin WNA16 MoE、64 shard 权重加载、KV cache profiling 和一次 greedy generate 成功。
  - 2026-05-21 H20 native vLLM 多 prompt fixture 生成通过；日志 `/tmp/kimi_k25_vllm_parity_20260521_085822.log`，同一次 load 处理 4 个 prompt，输出 `/data/fixtures/kimi-k2/k25_parity_vllm`。
  - 2026-05-21 H20 native vLLM HTTP fixture 生成通过；日志 `/tmp/kimi-k2-vllm-http/server_20260521_031017.log` 显示真实 `/data/models/Kimi-K2.5` TP8 权重加载和 serving readiness，fixture 写入 `request.json`、`response.json`、`metadata.json`、`prompt.json`。HTTP server 生成后已停止，H20-100 GPU 已释放。
  - 2026-05-21 H20 PegaInfer smoke candidate 生成通过，`candidate.safetensors` 大小 `655448` bytes，`logits_f32` shape `[163840]`，top8 ids `[154473, 159083, 160345, 161694, 149515, 159290, 161170, 149762]`。
  - 2026-05-21 H20 PegaInfer 多 prompt candidate gate 通过，`h20_kimi_k25_dump_parity_prompt_candidates` 用时 `133.43s`；compare gate 4/4 argmax match，top-20 overlap 最低 `19/20`。
  - HF raw full-logits fixture 仍未生成；这不阻塞当前 gate。当前只要求 PegaInfer 真实 forward 的 top 候选先进入 vLLM top-20 视野。

## 历史 all-rank prompt forward gate

- 当前生产入口是 `forward_prompt_next_token_async(input_ids)` / `ForwardPromptNextToken`；scheduler/runtime 收到请求后向 8 个 rank 并发发送完整 prompt，每个 rank 计算 last-token 本地 vocab shard top1，并把 `(global token id, local top logit)` 回传给 scheduler。
- scheduler 端用 8 个 shard top1 logit 做一次 host-side merge，返回全 vocab shard 的 greedy top1；这一步修掉了旧路径只看 rank0 vocab shard 的结构错误。
- 已接入主请求路径：
  - vocab-sharded embedding lookup；
  - 每层 MLA prefill：input RMSNorm、q_a/q_b、kv_a split、kv_a norm、kv_b、YARN RoPE、expanded Q/K/V assemble、FlashInfer prefill `<192,128>`、o_proj、TP all-reduce、residual add；
  - layer0 dense MLP local shard：post-attn RMSNorm、gate/up/down cuBLAS GEMM、SiLU-mul、TP all-reduce、residual add；
  - layer1..60 shared expert local shard、router、Marlin route alignment、vLLM Marlin WNA16 fused W13/W2、SwiGLU、f32 top-k reduce、NCCL-sum combine bridge、routed+shared 合并；
  - final RMSNorm、rank-local lm_head、last-token vocab shard top1。
- 显式不声称：
  - Kimi scheduler 当前只返回 greedy token，不返回 logprob/top-k payload。
  - 当前只声称 4 个短 prompt 的 vLLM greedy/top-20 gate，不声称长上下文、tool/preserve-thinking 或 perf path parity；
  - 当前 prompt path 不是 graph-ready hot path，仍有 per-layer temporary allocation 和 host-visible final top1；
  - 当前 EP combine 是 NCCL-sum correctness bridge，不是 PPLX EP 生产路径；首个 TP collective 前仍有 barrier + stream drain。
- H20 历史验证记录：
  - `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_rank0_one_token_forward_smoke -- --ignored --nocapture` 通过，真实权重路径 `/data/models/Kimi-K2.5`，用时约 `23.56s`。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view -- --ignored --nocapture` 通过，旧 rank0 payload/router/expert gate 未回退，用时约 `23.10s`。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_all_rank_one_token_vocab_shard_top1_smoke -- --ignored --nocapture` 通过，真实加载 8 rank K2.5 权重，8 个 rank 都执行 one-token smoke，并验证 `vocab_shards_considered=8`、`selected_from_global_vocab_shards=true`，Marlin worker backend 版本用时 `132.79s`。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_dump_all_rank_one_token_candidate_logits -- --ignored --nocapture` 通过，真实生成 full-vocab smoke candidate safetensors，Marlin worker backend 版本用时 `132.47s`。
  - PegaInfer candidate argmax id `1008`，与 vLLM greedy id `1008` 一致；top-20 id overlap `19/20`。
  - `cargo test --release -p pegainfer-kimi-k2 h20_kimi_k25_dump_parity_prompt_candidates -- --ignored --nocapture` 通过，真实生成 4 个 prompt 的 full-vocab candidate safetensors，Marlin worker backend 版本用时 `133.43s`。
  - `compare_vllm_topk_fixture.py --reference-root /data/fixtures/kimi-k2/k25_parity_vllm --candidate-root /data/fixtures/kimi-k2/pegainfer_k25_parity_candidates --top-k 20 --require-argmax --min-overlap 16` 通过，4/4 argmax match，top-20 overlap 最低 `19/20`。
- 这些 H20 ignored test 入口已从 direct worker/scheduler 删除；后续重新生成候选或性能数据走 server/bench 端到端路径，不再恢复内部 candidate dump 主线。

## Execution Log: worker-owned MLA decode cache

- worker load 阶段新增 `KimiWorkerDecodeArena`，与 `gpu` / `expert_kernels` 同属 `KimiRankLoadedWeights` 生命周期：
  - bs4 固定 arena：`page_size=16`、`pages_per_request=128`、`max_pages=512`，每层一个 separate ckv/kpe cache；
  - plan metadata 常驻 device：`page_indices`、`page_indptr`、`last_page_len`、`batch_indices`、`positions`、`request_indices`、`kv_tile_indices`、`kv_chunk_size`；
  - scratch 常驻 device：hidden/normed、q_a/q_proj、kv_a/compressed_kv/k_rope、q_nope/q_pe、q_abs/latent/attn_out/projected；
  - YARN RoPE cache 在 arena 初始化时 H2D，容量为 `2048` token positions，一步 decode body 内不重建。
- 新增 worker 内部 decode attention body：`forward_mla_decode_layer_into`。
  - 输入 hidden 在生产 decode 中会接 token embedding + TP all-reduce 后的 hidden。
  - cache 写入使用 `compressed_normed` 作为 MLA latent ckv，`append_kpe` 作为 rope cache。
  - attention 输出仍是 rank-local `o_proj` 前后结果，下一步要接 TP all-reduce / residual / MLP / logits。
- 验证：
  - 本地：`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过，只做编译门。
  - H20 历史 layer0 decode smoke 通过，证明过真实 `/data/models/Kimi-K2.5` rank0 和 all-rank layer0 decode wiring；这些 ignored test 入口已从 direct crate 删除。
  - H20 当前编译门：`cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server` 通过。
- H20 NCCL 运行环境记录：
  - `cudarc` 当前动态绑定会查找 `libnccl.so` 且要求 `ncclAlltoAll` 符号；`/root/develop/xingming/vllm_test/.venv/.../libnccl.so.2` 没有该符号。
  - 本轮验证使用临时 symlink：`/tmp/pegainfer-nccl-lib/libnccl.so -> /root/develop/xingming/pegainfer/.venv/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2`，该库是 `NCCL version 2.29.7+cuda12.9` 并包含 `ncclAlltoAll`。这是 H20 测试环境配置，不进入项目代码。

## Decode Graph Boundary

- [x] `decode_graph_ready_contract`
  - 位置：`pegainfer-kimi-k2/src/runtime.rs`
  - 约束对象：rank-local decode compute kernels，包括 RMSNorm、MLA decode、本地 shared/dense MLP、router、expert-major packing、INT4 grouped GEMM、reduce、logits shard 等。
  - contract：metadata device resident、无 D2H、无 host sync、decode step 内无 allocation、Graph replay 指针稳定、scratch 预分配。
  - EP 边界：PPLX dispatch/combine 当前明确在 CUDA Graph capture 外；真实接入后通过 capture harness 验证是否能纳入，未验证前不把 EP 计入 graph-ready 范围。

- [ ] `decode_kernel_graph_audit`
  - 扫描每个 decode CUDA/FFI path：不得出现 `cudaMemcpyDtoH`、pageable allocation、per-step handle 创建、stream synchronize、host-side route/count read、根据 CPU route metadata 改 launch graph 的路径。
  - bs>1 必须显式走 batch/padded token contract，不能靠单 token 特化绕过 metadata。
  - PPLX EP 虽在 graph 外，也必须保持 buffer/metadata 预分配和 device resident，避免把 D2H 或分配带进 decode loop。

## Preparation: CUTLASS INT4 grouped expert launcher

- **Read**:
  - `docs/index.md` — confirmed Kimi model docs and kernels subsystem routing.
  - `docs/models/kimi-k2/operator-todo.md` — confirmed INT4 routed experts must use CUTLASS example 69 style grouped GEMM, with device-resident decode metadata.
  - `docs/models/kimi-k2/support-analysis.md` — confirmed Kimi is text-only for current scope and current runtime EP combine is an explicit NCCL bridge, not PPLX.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` — confirmed per-model kernels live behind the kernels crate boundary.
  - `pegainfer-kernels/third_party/flashinfer/3rdparty/cutlass/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_bf16_grouped_gemm.cu` — source pattern for Hopper INT4/BF16 grouped ptr-array GEMM.
- **Relevant history**:
  - DSV4 MoE docs established that decode routing/expert metadata must stay on GPU; Kimi carries that contract from the first INT4 launcher shape.
- **Plan**:
  1. Add a CUTLASS SM90a grouped projection ABI with explicit params, support probe, workspace size query, prepare, and launch entry points.
  2. Make the workspace contain device-resident problem shapes, ptr arrays, stride/layout arrays, and CUTLASS internal workspace; prepare fills these from `expert_indptr`.
  3. Mirror the ABI in Rust, expose a preallocated workspace type and prepare/launch wrappers from `kimi_experts.rs`.
  4. Keep old non-CUTLASS expert placeholders from becoming a Kimi runtime path.
  5. Run formatting and `cargo check` for the kernels crate.
- **Risks / open questions**:
  - The checkpoint INT4 packed layout still needs a weight-loader side transform into CUTLASS example-69 reordered layout before numerical validation.
  - W1/W3 true fused-N packing requires a fused/reordered weight buffer; the generic projection launcher enables the contract first.

## Execution Log: CUTLASS INT4 grouped expert launcher

### Step 1: CUDA ABI and CUTLASS launcher skeleton

- Updated `pegainfer-kernels/csrc/kimi_k2/kimi_cutlass_int4_sm90a.cu`.
- Added `KimiCutlassInt4GroupedLaunchParams` and `KimiCutlassInt4GroupedWorkspaceSizes`.
- Added support probe, workspace-size query, prepare, and launch externs.
- Workspace is explicitly partitioned into device-resident problem shapes, ptr arrays, stride arrays, reordered-B layout arrays, and CUTLASS internal workspace.
- `prepare` fills per-expert metadata from device `expert_indptr`; `launch` builds CUTLASS example-69 scale-only shuffled grouped GEMM arguments and calls `can_implement`, `initialize`, and `run`.

### Step 2: Rust FFI and ops contract

- Updated `pegainfer-kernels/src/ffi.rs` with repr(C) mirrors and extern declarations.
- Updated `pegainfer-kernels/src/ops/kimi_experts.rs` with:
  - `KimiCutlassSm90aSupport`
  - `KimiCutlassInt4GroupedWorkspace`
  - workspace size query
  - prepare/launch wrappers for a single INT4 grouped projection
  - manifest attributes that now describe prepared device-resident CUTLASS workspace.
- Updated `pegainfer-kernels/src/ops.rs` exports for the new workspace/probe/launcher API.

### Step 3: Verification

- `cargo fmt --check` passed.
- `cargo check --release -p pegainfer-kernels` passed, including NVCC compilation of `kimi_cutlass_int4_sm90a.cu` for detected `sm_120`.
- `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels` passed, including explicit Hopper `sm_90a` NVCC compilation.
- `cargo check --release -p pegainfer-kimi-k2` passed.

### Unexpected

- `make_cute_packed_stride` rejected static `Int<1>` for the dynamic runtime stride shape; using dynamic `1` matches the expected `Shape<int,int,int>`.
- An anonymous namespace in this CUDA TU collided with CuTe anonymous namespace emission in NVCC host stubs; switching the TU to a named namespace fixed the ambiguity.

## Debrief: CUTLASS INT4 grouped expert launcher

- **Outcome**: Kimi now has a real CUTLASS example-69 style grouped INT4 projection launcher skeleton with explicit graph-ready workspace contract and Rust wrappers. It still requires the weight loader to provide CUTLASS-reordered INT4 packed weights before numerical validation.
- **Pitfalls encountered**:
  - CuTe/CUTLASS device-side helper types are sensitive to static-vs-dynamic shape tags.
  - CUDA kernels in anonymous namespaces can produce ambiguous NVCC stub names when heavy CuTe headers are included.
- **Lessons learned**:
  - Keep the launcher ABI generic per projection first; W1/W3 true fused-N should be added when the loader owns a fused/reordered packed buffer.

## 模型事实

| 项 | 值 |
| --- | --- |
| 文本 HF 类 | `DeepseekV3ForCausalLM` |
| `text_config.model_type` | `kimi_k2` |
| dtype | BF16 主干 |
| hidden | `7168` |
| vocab | `163840` |
| layers | `61` |
| dense layers | `1`，仅 layer 0 |
| MoE layers | layer `1..60` |
| context | `262144` |
| attention | MLA |
| heads | `64` |
| `q_lora_rank` | `1536` |
| `kv_lora_rank` | `512` |
| `qk_nope_head_dim` | `128` |
| `qk_rope_head_dim` | `64` |
| `q_head_dim` | `192` |
| `v_head_dim` | `128` |
| routed experts | `384` |
| selected experts | top `8` |
| shared experts | `1` |
| routed expert FFN dim | `2048` |
| dense layer0 FFN dim | `18432` |
| routed expert quant | compressed-tensors native INT4, group size `32` |

## Forward DAG

### Embedding

- [ ] `embedding_lookup`
  - 输入：token ids
  - 输出：BF16 hidden `[tokens, 7168]`
  - 权重：`language_model.model.embed_tokens.weight`
  - 备注：text-only 首版只需要普通 token；media token 不做特殊展开。

### 每层公共结构

- [ ] `rms_norm_hidden`
  - 形状：`[tokens, 7168]`
  - 权重：`input_layernorm.weight` / `post_attention_layernorm.weight`
  - 复用方向：现有 FlashInfer `rms_norm_batched_cuda` / `ops::rms_norm_batch_into` 参数化，Kimi header 用 `RmsNormBackend::FlashInferBatch` 表达。

- [ ] `residual_add`
  - 形状：`[tokens, 7168]`
  - 复用方向：现有 BF16 add / FlashInfer `fused_add_rms_norm_batched_cuda`。

## MLA Attention

Kimi 的 HF 代码先 materialize expanded `Q/K/V`，但生产实现不能长期保存 expanded KV。算子 bring-up 分两层：先做 expanded correctness path，再做 compressed KV production path。

### Projection

- [ ] `q_a_linear`
  - 输入：BF16 `[tokens, 7168]`
  - 输出：BF16 `[tokens, 1536]`
  - 权重：`self_attn.q_a_proj.weight`

- [ ] `q_a_rms_norm`
  - 输入/输出：BF16 `[tokens, 1536]`
  - 权重：`self_attn.q_a_layernorm.weight`

- [ ] `q_b_linear`
  - 输入：BF16 `[tokens, 1536]`
  - 输出：BF16 `[tokens, 12288]`
  - 解释：`64 heads * (128 nope + 64 rope)`
  - 权重：`self_attn.q_b_proj.weight`

- [ ] `split_q_nope_q_rope`
  - 输入：`[tokens, 64, 192]`
  - 输出：`q_nope [tokens, 64, 128]`、`q_rope [tokens, 64, 64]`

- [ ] `kv_a_with_mqa_linear`
  - 输入：BF16 `[tokens, 7168]`
  - 输出：BF16 `[tokens, 576]`
  - 解释：`compressed_kv 512 + k_rope 64`
  - 权重：`self_attn.kv_a_proj_with_mqa.weight`

- [ ] `kv_a_split`
  - 输出：`compressed_kv [tokens, 512]`、`k_rope [tokens, 1, 64]`

- [ ] `kv_a_rms_norm`
  - 输入/输出：BF16 `[tokens, 512]`
  - 权重：`self_attn.kv_a_layernorm.weight`

- [ ] `kv_b_linear`
  - 输入：BF16 `[tokens, 512]`
  - 输出：BF16 `[tokens, 16384]`
  - 解释：`64 heads * (128 k_nope + 128 value)`
  - 权重：`self_attn.kv_b_proj.weight`

- [ ] `split_k_nope_value`
  - 输出：`k_nope [tokens, 64, 128]`、`value [tokens, 64, 128]`

### RoPE

- [ ] `yarn_rope_cache`
  - dim：`64`
  - theta：`50000`
  - factor：`64`
  - original max position：`4096`
  - beta：fast `32`，slow `1`
  - 输出：cos/sin cache，按 HF `DeepseekV3YarnRotaryEmbedding` 对齐。

- [ ] `apply_partial_rope_qk`
  - 输入：`q_rope [tokens, 64, 64]`、`k_rope [tokens, 1, 64]`
  - 输出：rotated q/k rope slice
  - 注意：HF 的 `apply_rotary_pos_emb` 对最后维度做了 view/transpose，不能只按 Qwen full-RoPE 直觉套。

### Attention Core

- [ ] `assemble_q`
  - 拼接：`q_nope[128] + q_rope[64] -> q [tokens, 64, 192]`

- [ ] `assemble_k_expanded`
  - correctness path 拼接：`k_nope[128] + k_rope[64] -> k [tokens, 64, 192]`

- [ ] `attention_prefill_expanded`
  - 输入：`q_dim=192`，`k_dim=192`，`v_dim=128`
  - 输出：BF16 `[tokens, 64, 128]`
  - 用途：短上下文 correctness。

- [ ] `attention_decode_expanded`
  - 输入：单步 q，expanded K/V cache
  - 用途：先跑通 decode parity。

- [ ] `compressed_kv_cache_write`
  - 存储：`compressed_kv[512] + k_rope[64]`
  - 原因：256K context 下 expanded K/V cache 过大。

- [ ] `attention_prefill_mla_compressed`
  - 输入：compressed KV cache
  - kernel 内或临近算子重构 `k_nope/value`
  - 后续生产路径。

- [ ] `attention_decode_mla_compressed`
  - 单 token decode hot path。

- [ ] `o_proj_linear`
  - 输入：BF16 `[tokens, 8192]`
  - 输出：BF16 `[tokens, 7168]`
  - 权重：`self_attn.o_proj.weight`

## Dense MLP 与 Shared Expert

### Layer 0 Dense MLP

- [ ] `dense_gate_linear`
  - 输入：`[tokens, 7168]`
  - 输出：`[tokens, 18432]`
  - 权重：`layers.0.mlp.gate_proj.weight`

- [ ] `dense_up_linear`
  - 输入：`[tokens, 7168]`
  - 输出：`[tokens, 18432]`
  - 权重：`layers.0.mlp.up_proj.weight`

- [ ] `silu_mul_dense`
  - 输入：gate/up `[tokens, 18432]`
  - 输出：`[tokens, 18432]`

- [ ] `dense_down_linear`
  - 输入：`[tokens, 18432]`
  - 输出：`[tokens, 7168]`
  - 权重：`layers.0.mlp.down_proj.weight`

### Shared Expert

- [ ] `shared_gate_linear`
  - 输入：`[tokens, 7168]`
  - 输出：`[tokens, 2048]`
  - 层：MoE layer `1..60`

- [ ] `shared_up_linear`
  - 输入：`[tokens, 7168]`
  - 输出：`[tokens, 2048]`

- [ ] `silu_mul_shared`
  - 输入：gate/up `[tokens, 2048]`
  - 输出：`[tokens, 2048]`

- [ ] `shared_down_linear`
  - 输入：`[tokens, 2048]`
  - 输出：`[tokens, 7168]`

## MoE Router

HF gate 逻辑：`logits = hidden @ gate.weight.T`，`scores = sigmoid(logits)`，选择分数使用 `scores + e_score_correction_bias`，最终权重从未加 bias 的 `scores` gather，normalize 后乘 `2.827`。

- [x] `router_kernel_scaffold`
  - 位置：`pegainfer-kernels/src/ops/kimi_router.rs`
  - 已有：shape validation、bs>1 `active_tokens/padded_tokens` contract、device-resident `topk_weight/topk_idx` 输出、库 GEMM 计算 `hidden @ gate_weight.T`，CUDA body 做 sigmoid / bias / top8 / normalize。
  - H20 gate：2026-05-21 已用真实 K2.5 layer1 router gate/bias typed GPU package 执行 `kimi_router_noaux_tc_launch`，输出直接进入 expert-major route bridge。
  - 后续：接 Kimi runtime 的预分配 scratch。

- [x] `router_score_linear_f32`
  - 输入：BF16 hidden `[tokens, 7168]`
  - 权重：BF16/FP32 gate `[384, 7168]`
  - 输出：FP32 logits `[tokens, 384]`
  - 状态：库 GEMM 路径已在 `kimi_k2_router_noaux_tc_cuda` 中接通，并在 H20 真实 K2.5 layer1 gate 权重上通过。

- [x] `router_sigmoid`
  - 输出：FP32 scores `[tokens, 384]`
  - CUDA body 已有。

- [x] `router_choice_bias_add`
  - 输入：scores、`e_score_correction_bias[384]`
  - 输出：choice scores
  - CUDA body 已有。

- [x] `router_top8`
  - 输入：choice scores `[tokens, 384]`
  - 输出：top8 expert ids
  - 备注：`n_group=1`，没有跨 group 筛选复杂度。
  - CUDA body 已有，形态对齐 DSV4 device-side score gate selection。

- [x] `router_weight_gather_normalize`
  - 从原始 scores gather top8 weights
  - normalize 到 sum 1
  - 乘 `routed_scaling_factor = 2.827`
  - CUDA body 已有。

- [x] `router_output_pack`
  - 输出留在 device：`topk_idx`、`topk_weight`
  - 禁止 D2H route metadata 进入热路径。
  - Rust/CUDA API 已保留 device-resident contract。

## Routed Expert INT4

每个 MoE layer 有 `384` 个 routed experts。TP8/EP8 首版按每 rank `48` 个本地 experts 规划。

### CUTLASS C++ AOT 路线

- [x] `int4_grouped_gemm_library_probe`
  - 结论：FlashInfer 当前没有确认可直接接 Kimi compressed-tensors `signed INT4 + BF16 scale(group=32)` 的 drop-in grouped GEMM；CUTLASS example69 已被 H20 probe 排除为 correctness path。
  - 可复用：FlashInfer grouped GEMM 的 segmented/grouped problem 组织、DSV4 AOT 编译接入形态，以及后续 TRT-LLM/FlashInfer W4A16 路径中已经证明支持 Kimi scale 语义的部分。
  - 不可直接复用：DSV4 FP4/E8M0 grouped kernels、FlashInfer MXFP4/NVFP4 groupwise kernels；它们的数值格式和 Kimi signed INT4/BF16 scale 不一致。

- [x] `cutlass_hopper_mixed_input_grouped_probe` (2026-05-20)
  - **CuTeDSL (nvidia-cutlass-dsl 4.4.2) 的 `mixed_input_helpers` 只覆盖 Blackwell**（依赖 `tcgen05` / TMEM）。`hopper_helpers` 是 dense GEMM (WGMMA) 路径，**没有 Hopper mixed-input helper**。CuTeDSL 不是 Kimi (Hopper H20) 的可行路线。
  - **CUTLASS C++ upstream 有 `examples/69_hopper_mixed_dtype_grouped_gemm/`**，含三个变体：
    - `69_hopper_int4_bf16_grouped_gemm.cu` — 正好就是 BF16 activation × INT4 weight × BF16 group scale 的 Hopper grouped GEMM。
    - `69_hopper_int4_fp8_grouped_gemm.cu` — FP8 activation 变体（暂不需要）。
    - `69_hopper_mixed_dtype_grouped_gemm.cu` — generic 模板。
  - 仓库内 `pegainfer-kernels/third_party/flashinfer/3rdparty/cutlass/` (v4.4.2) 与 upstream v4.5.1 该 example 文件**逐字节相同**，直接复用仓库内拷贝即可，不需要额外 submodule。
  - 2026-05-21 复核结论：example69 的 launch smoke 只能证明当前 launcher/shape metadata 可进入 sm90a grouped GEMM，不能证明 Kimi correctness。focused probe 证明它不能表达 Kimi `group_size=32` 的 BF16 per-row/per-K-group scale 语义；该路径停止作为主线 backend。

- [ ] `kimi_cutlass_hopper_int4_grouped_generator`
  - **历史路线**：基于 `69_hopper_int4_bf16_grouped_gemm.cu` 改造，C++ AOT 编译进 `csrc/kimi_k2/`，feature gate `kimi-k2`，仅编译 sm_90a。
  - **当前结论**：`csrc/kimi_k2/kimi_cutlass_int4_sm90a.cu` 已接入 build/FFI/ops，并能在 H20 launch，但它只保留为 limitation probe 和 smoke scaffold。它不能作为 Kimi routed expert correctness backend，因为 Kimi `group_size=32` scale 语义和 example69 `TileShapeK=64` scale reload 不匹配。
  - **核心改动清单**：
    1. `ElementC = ElementD = cutlass::bfloat16_t` (example 默认 `half_t`)
    2. 删除 CLI option/verify/profiling/main，只留 kernel + `cutlass::reorder_tensor` (offline shuffle) + 一个 `extern "C"` launcher
    3. Swap-and-Transpose 是这一族 kernel 的内置 trick：A↔B 互换，kernel 内部以 `(N, M, K)` 看问题。launcher 需要把 routed activation 和 INT4 weight 按这个约定 wire ptr-array
    4. ptr-array 在 launcher 内即时构造：W/scale 的 per-expert offset 在 weight load 时一次性算好并 cache (device-resident)；A/D 的 per-expert offset 需要从 `expert_indptr` 通过一个 prep kernel 在 device 上生成，**避免 D2H**
    5. `problem_sizes [E]` 同样由 prep kernel 从 `expert_indptr` 构造：`{N, M_e, K}` (注意 swap 后 N 在前)
    6. group-scale `c = 32` 通过 `arguments.mainloop.scale_K` 传入
  - **W1/W3 fused**：example 是单个 GEMM。W1+W3 共享 A，可以拼成 `N = 2 * 2048 = 4096` 的单一 grouped GEMM（weight 沿 N 维度 concat），epilogue 不分裂；上层取 `[:2048]` 当 gate、`[2048:]` 当 up。这样省一次 A 的 TMA。
  - **SwiGLU**：外置 `KimiSwiGluPlan` + `kimi_swiglu_silu_mul`，复用已有 `silu_mul_triton_aot_cuda`；W2 输入是 layer-resident BF16 scratch。
  - **W2**：独立 grouped GEMM，K=2048 N=7168。
  - **第一次 launch 前的 offline weight reorder**：CUTLASS `reorder_tensor` 用 `LayoutAtomQuant` shuffle，目的让同 warp fragment 读连续 INT4 nibble。这一步只在 weight load 阶段做一次，结果存回 `weight_packed` 同一块显存。`xor 0x88` 转 signed nibble 是正确的；scale 语义仍然不满足 Kimi，因此该 package 不能宣称 correctness。
  - **scale layout 现状**：manifest metadata 已区分 checkpoint layout、CUTLASS example69 group-major layout、Marlin group-major+perm64 layout。当前 runtime smoke 仍用 example69 group-major scale；后续 WNA16/Marlin backend 不能复用这个 scale buffer，必须使用 `kimi_marlin_int4_reorder_scale_cuda` 生成 vLLM Marlin 语义的 scale package。
  - **dev (5090) 不能跑**：sm_90a WGMMA 在 sm_120 不存在；cross-arch 只能验编译 + 链接，运行时正确性必须在 H20 上跑，并对齐 `tools/kimi_k2/torch_reference.py` 或 vLLM 产出的外部 fixture。
  - **替代方向**：优先找 TRT-LLM/FlashInfer 已有 W4A16 grouped MoE 路径，要求原生支持 BF16 activation、signed INT4 weight、BF16 `[out, K/32]` scale；没有合适 backend 时写 Kimi-specific AOT kernel。
  - **约束**：禁止把 `weight_shape` 检查放进 GEMM inner loop；shape 在 loader/launch 前验证。

### 权重读取

- [x] `compressed_tensors_int4_header_probe`
  - 确认 `weight_packed` dtype/shape。
  - 确认 `weight_scale` dtype/shape。
  - 确认 `weight_shape` 内容与端序。
  - 确认 INT4 nibble 顺序和 signed/symmetric 解码。
  - 状态：2026-05-20 已在 `pegainfer-kernels::ops` 落地 `KimiInt4WeightManifest` / `KimiInt4Weight` 和 `kimi_int4_metadata_probe`，kernel-facing ABI 覆盖 `weight_packed` u8 bytes `[48,out,in/2]`、`weight_scale` BF16 `[48,out,in/32]`、`weight_shape` I32 `[96]`。
  - **Pack semantics 已通过 compressed-tensors 源码确认 (2026-05-20)**：on-disk per-linear shape `weight_packed [out_dim, in_dim/8] int32`，pack 沿 in 维度 little-endian，element k 占 int32 bits `[4k, 4k+4)`。signed→unsigned via `+8`，dequant: `signed_nibble = unsigned_nibble - 8`。view(uint8) 后每 byte 含两个 in_col：**低 nibble = 偶数 in_col，高 nibble = 奇数 in_col**（`KimiInt4NibbleOrder::LowThenHigh`）。Routed-only：attention / shared experts / dense layer0 MLP / lm_head 不量化（config `ignore` regex 屏蔽）。Per-expert tensor 不预 fuse，W1/W3 各自独立存；EP8 plan 阶段沿 expert 维度 stack 成 `[48,out,in/2]`。
  - Fixture：`tools/kimi_k2/torch_reference.py` 使用 compressed-tensors 官方 `pack_to_int32` 生成 bit-exact 数据，自洽校验 `0-diff`。

- [ ] `kimi_int4_weight_loader`
  - 输入：`weight_packed`、`weight_scale`、`weight_shape`
  - 输出：GPU resident INT4 grouped linear weight。
  - 已有前置：`KimiRankTypedGpuWeights::expert_major_weight_plan()` 基于真实 rank-local typed view 校验每层 48 个本地 expert 的 gate/up/down 三元组：
    - safetensors per-expert packed：`I32 [out, in/8]`
    - CUTLASS-facing packed bytes：`u8 [local_expert, out, in/2]`
    - scale：`BF16 [out, in/32]`
    - shape：`I32 [2]`
  - 已有入口：`pack_expert_major_layer_raw_buffers()` 可把指定 MoE layer 的 gate/up/down 三元组通过 D2D copy 打成连续 raw buffer。
  - 已有入口：`pack_expert_major_layer_kernel_weights()` 产出 `KimiMoeLayerExpertKernelWeights`，内部持有 `CudaSlice<u8>` reordered packed、`CudaSlice<bf16>` scale、`CudaSlice<i32>` shape，并可借用成 `KimiInt4ExpertWeights`。
  - 已有入口：`pack_rank_expert_kernel_weights()` 产出 full-rank `KimiRankExpertKernelWeights`，覆盖 60 个 MoE layer；转换后删除 raw tensor map 里的 `.mlp.experts.` routed expert tensors，worker `LoadSlicedWeights` 直接持有常驻 package。
  - CUTLASS package：`kimi_cutlass_int4_reorder_weight_sm90a_cuda` 在 load/package 阶段调用 CUTLASS `reorder_tensor`，并把 compressed-tensors offset-binary nibble 转成 signed int4b_t 表示。
  - Marlin weight package：`kimi_marlin_int4_reorder_weight_cuda` 已按 vLLM no-actorder `gptq_marlin_moe_repack` 语义落地；输入是 checkpoint offset-binary `[expert,out,K/8] int32`，输出是 Marlin uint4b8 `[expert,K/16,N*2] int32`，总字节数不变，不做 `xor 0x88`。
  - Marlin scale package：`kimi_marlin_int4_reorder_scale_cuda` 已按 vLLM `marlin_moe_permute_scales` 语义落地，将 checkpoint `[expert,out,in_group]` 融合 transpose + 64-block scale permutation 成 `[expert,in_group,out]` group-major+perm64 buffer。它不是 example69 的输入 layout；用于后续 WNA16/Marlin correctness backend。2026-05-21 已在 H20 通过 `h20_kimi_marlin_scale_reorder_matches_vllm_permute`。
  - H20 gate：`/data/models/Kimi-K2.5` rank0 真实 payload 已通过 expert-major package plan、layer1 raw buffer D2D package、CUTLASS sm90a reorder、typed `KimiInt4ExpertWeights` package、full-rank 60 layer package、真实 layer1 router GEMM/top8 输出、expert-major route/expand/reduce、SwiGLU，以及 W1/W3/W2 通用 CUTLASS prepare+launch 零输入不变量校验。
  - 纠偏：上述 gate 只证明 loader/package/launch 可跑；focused H20 probe 已证明 example69 scale 语义不符合 Kimi per32 correctness。下一步是替换 routed expert backend，再回到 full-forward/vLLM gate。

### Expert-major Routing Layout

- [x] `moe_count_local_experts`
  - 输入：top8 ids
  - 输出：本 rank 48 experts 的 token counts
  - 状态：2026-05-21 已在 `kimi_moe_expert_major_route_cuda` 内完成，输入 `topk_idx[active_tokens,8]`，按 `global_expert_start..+48` 过滤本地 experts，全程 device-side。

- [x] `moe_expert_indptr_prefix`
  - 输出：`expert_indptr[49]`
  - 状态：2026-05-21 已输出 `u32 expert_indptr[49]`，直接喂通用 CUTLASS prepare/launch；同时输出 `local_count[1]` 作为 device metadata。

- [x] `moe_expand_to_expert_major`
  - 输入：hidden `[tokens, 7168]`、top8 ids/weights
  - 输出：expert-major packed activations。
  - 状态：2026-05-21 已新增 `KimiExpertMajorRouteWorkspace` / `KimiExpertMajorRouting` / `kimi_moe_expand_to_expert_major`，使用 `pos_to_token` 做 BF16 token-major 到 expert-major copy；无 D2H、无 step 内 allocation。

### INT4 Grouped GEMM

- [ ] `int4_grouped_w1_w3`
  - input dim：`7168`
  - output dim：`2048`
  - local experts：`48`
  - group size：`32`
  - 输出：gate/up 两路 BF16 或 FP32 accumulator buffer。
  - 状态：2026-05-20 已新增 `kimi_int4_grouped_w1_w3` Rust API、manifest `KernelCall`、`kimi_int4_grouped_w1_w3_cuda` 参数校验入口和 `kimi_cutlass_int4_grouped_w1_w3_sm90a_cuda` AOT 接口；输入按 expert-major `[routed_tokens,7168]`，bs>1 通过 `batch_size` / `active_tokens` / `expert_indptr[49]` 显式建模。2026-05-21 H20 gate 已在真实 rank0 reordered package 上分别实跑 W1 gate / W3 up 通用 prepare+launch；focused H20 probe 已证明这个 CUTLASS example69 body 不是 Kimi correctness backend，必须替换。

- [x] `swiglu_silu_mul`
  - 状态：2026-05-20 已新增 `KimiSwiGluPlan` + `kimi_swiglu_silu_mul`，复用 `silu_mul_triton_aot_cuda`；GPU unit test `6/6` 通过。2026-05-21 H20 rank0 gate 已把 SwiGLU 放进 W1/W3 与 W2 之间的真实 package 流程。

- [ ] `int4_grouped_w2`
  - input dim：`2048`
  - output dim：`7168`
  - 输入是 `silu(gate) * up` 的 BF16 scratch。
  - 状态：2026-05-20 已新增 `kimi_int4_grouped_w2_swiglu` Rust API、manifest `KernelCall`、`kimi_int4_grouped_w2_swiglu_cuda` 参数校验入口和 `kimi_cutlass_int4_grouped_w2_sm90a_cuda` AOT 接口。2026-05-21 H20 gate 已在真实 rank0 reordered package 上实跑 SwiGLU scratch + W2 down 通用 prepare/launch；focused H20 probe 已证明这个 CUTLASS example69 body 不是 Kimi correctness backend，必须替换。

- [x] `moe_reduce_expert_outputs`
  - 输入：expert-major output、top8 weights、route map
  - 输出：FP32 routed output `[tokens, 7168]`
  - 状态：2026-05-21 已新增 `kimi_moe_reduce_expert_major_f32`，按 `token_topk_to_pos` gather 本地 expert-major 输出并乘 f32 `topk_weight` 累加到 f32 token-major output；后续接 EP combine / TP reduce 时继续消费 f32 routed output。

## TP8/EP8 Collective

### Attention / Dense TP

- [ ] `tp_linear_shard_policy`
  - 定义 q/k/v/o、dense MLP、shared expert、lm_head 的 shard 方向。

- [ ] `tp_attention_collective`
  - attention heads 可按 TP rank 分片。
  - 输出 o_proj 前后的 all-reduce / reduce-scatter 形态需要定稿。

- [ ] `tp_mlp_collective`
  - dense/shared MLP 的 row/column parallel 组合。

### MoE EP

- [ ] `ep_pplx_dispatch_combine_path`
  - Kimi EP 目标路径是 PPLX dispatch/combine；当前 direct runtime 先保留 NCCL-sum bridge，不做 NCCL AG/RS。
  - 复用 DSV4 PPLX bootstrap、rank worker placement、MR 注册和 scratch 生命周期。
  - buffer shape 改为 hidden `7168`、topk `8`、local experts `48`、expert intermediate `2048`。
  - shared expert 与 dispatch/recv overlap 按 DSV4 PPLX decode 结构设计。
  - route/count/indptr/combine metadata 保持 device resident。
  - 当前作为 CUDA Graph 外阶段处理；Graph 内先只覆盖 rank-local compute kernels。

- [ ] `final_logits_all_gather`
  - lm_head 每 TP rank vocab shard `20480`。
  - 首版 all-gather logits 到 full vocab `163840` 后采样。

## Logits / Sampling

- [ ] `final_rms_norm`
  - 输入/输出：`[tokens, 7168]`
  - batch logits 路径复用 `FlashInferBatch`，单向量路径复用 `FlashInferVec`。

- [ ] `lm_head_sharded_linear`
  - 输入：last token hidden `[7168]`
  - 输出：local logits `[20480]` per TP rank

- [ ] `logits_all_gather`
  - 输出：full logits `[163840]`

- [ ] `greedy_top1`
  - 先只做 greedy。

- [ ] `sampling_top_p_temperature`
  - 后续支持 README 推荐参数：thinking `temperature=1.0`，instant `temperature=0.6`，`top_p=0.95`。

## Tokenizer / Prompt Contract

这不是 GPU 算子，但会决定首个 text-only runner 的输入 token 是否正确。

- [ ] `tiktoken_tokenizer_load`
  - 加载 `tiktoken.model`。
  - 加载 `tokenizer_config.json` special tokens。

- [ ] `chat_template_text_only`
  - 实现 `chat_template.jinja` 的文字路径。
  - 拒绝 image/video content。

- [ ] `thinking_prompt`
  - 默认 generation prompt 以 `<think>` 开始。

- [ ] `instant_prompt`
  - `thinking=false` 时使用 `<think></think>`。

- [ ] `preserve_thinking_prompt`
  - 保留 `reasoning` / `reasoning_content` 的 suffix 规则。

- [ ] `tool_declaration_prompt`
  - 保留 tool declaration token 格式；tool parser 可后置。

## 测试夹具 TODO

- [ ] `hf_config_dump`
  - dump text_config 和 normalized operator shapes。

- [ ] `hf_tokenizer_fixture`
  - 用 README 的 text-only 示例生成 prompt ids。

- [ ] `hf_layer0_fixture`
  - dense layer0：RMSNorm、MLA、dense MLP。

- [ ] `hf_moe_layer_fixture`
  - layer1：router、shared expert、routed expert。

- [ ] `hf_decode_one_token_fixture`
  - 单 token decode：position/RoPE/cache 对齐。

- [x] `int4_single_expert_fixture`
  - `tools/kimi_k2/torch_reference.py` 用 compressed-tensors 官方 pack 路径产生 bit-exact fixture，自洽校验 `0-diff`。

## 建议实现顺序

1. Config/index/header probe。已完成 text-only config、index manifest、TP8/EP8 rank plan、rank-local typed names、shard read plan。
2. Direct scheduler / rank worker 骨架，按 DSV4 Flash 分层：scheduler 管请求/KV，worker 管 rank CUDA/PPLX/runtime。已完成 skeleton、rank plan / typed names / shard plan 移交、CPU binding、decode graph boundary。
3. Tokenizer + chat template text-only fixture。
4. BF16 dense primitives shape 参数化。
5. YARN RoPE + expanded MLA correctness path。
6. Router CUDA body。
7. INT4 dequant format probe。已确认 signed/unsigned 与 nibble 顺序。
8. Expert-major layout + grouped INT4 kernel 接线。已完成 route/expand/reduce 与 launch smoke。
9. 替换 routed expert INT4 backend，不能继续用 CUTLASS example69 作为 correctness path。
10. Rank-local decode kernel graph audit。
11. PPLX EP dispatch/combine path。
12. Full layer fixture。
13. Text-only greedy runner。

## Scheduler / Worker TODO

- [x] `direct_scheduler_worker_skeleton`
  - 位置：`pegainfer-kimi-k2/src/direct/scheduler.rs`、`src/direct/worker.rs`
  - 已有：`EngineHandle` 接入、config/index manifest probe、`device_ordinals=0..7` gate、CUDA Graph 禁用 gate、8 rank worker lifecycle、rank weight plan / typed names / shard plan 移交、请求 `Scheduled` + runtime-not-wired error。
  - 约束：当前已实现路径是 NCCL-sum bridge；EP 生产目标是替换成 PPLX dispatch/combine，不接 NCCL AG/RS。

- [x] `rank_weight_manifest`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`
  - 已有：读取 `model.safetensors.index.json`，生成 text-only manifest；忽略 vision/projector tensors；生成 TP8/EP8 `KimiRankWeightPlan`。
  - 本地 K2.6 index 验证：text tensor `208215`，ignored non-text tensor `335`，shard `64`，每 rank tensor plan `26775`。

- [x] `rank_weight_names_and_shard_plan`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`
  - 已有：`KimiRankWeightNames` typed view、`KimiRankShardPlan` shard grouping。
  - 本地 K2.6 index 验证：rank7 heads `56..64`，vocab `143360..163840`，experts `336..384`，每 rank shard read plan `62` 个 shard。

- [x] `rank_weight_sliced_load_plan`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`
  - 已有：`KimiTensorLoadSlice` / `KimiRankSlicedLoadPlan`，并接入 direct scheduler/worker config。
  - TP8 切片：embedding/lm_head 按 vocab 行切；`q_b/kv_b` 按本地 head 行切；`o_proj` 按本地 head value 列切；dense/shared `gate/up` 行切、`down` 列切。
  - EP8 切片：routed expert tensor 名只包含本 rank 48 个 global experts，tensor 内部全量读取。
  - 单测：rank3 切片计划、sliced header shape/bytes、col slice row-major repack。

- [x] `rank_weight_loader_header_and_gpu_copy`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`
  - 已有：`load_rank_weight_headers` / `load_rank_weights_to_gpu` 保留整 tensor shard plan 路径；`load_rank_sliced_weight_headers` / `load_rank_sliced_weights_to_gpu` 是 TP8/EP8 生产加载入口。
  - 单测：小 safetensors fixture 覆盖多 shard header load、缺失 tensor 报错、sliced local shape/bytes、col slice row-major repack。

- [x] `rank_worker_cpu_binding`
  - 位置：`pegainfer-kimi-k2/src/direct/affinity.rs`
  - 已有：按 DSV4 Flash 策略保留 CPU0、scheduler 优先 pin CPU1、rank worker 根据 CUDA device NUMA node 切连续 CPU slice，并 pin 到各自 slice 的首个 CPU。
  - `role_cpu(offset, role)` 保留给后续 PPLX TE/A2A/UVM worker 的 offset 分配。

- [x] `rank_weight_typed_gpu_view`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`
  - 已有：基于 `KimiRankWeightNames` 将 raw GPU tensor map 包成 top / attention / dense / router / shared / routed expert typed view；routed experts 每 rank 48 个。
  - 已有：header 和 GPU raw map 两条路径共享 rank、tensor count、dtype 校验；router bias 必须 F32，routed expert safetensors dtype 为 `weight_packed/scale/shape = I32/BF16/I32`，kernel-facing packed bytes 仍按 u8 视图使用。

- [x] `rank_expert_major_weight_package_plan`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`
  - 已有：`KimiRankExpertMajorWeightPlan` / `KimiMoeLayerExpertMajorPlan` / `KimiExpertMajorProjectionPlan`。
  - 覆盖：60 个 MoE layer，每层本地 48 experts，gate/up/down 三个 projection 的 dtype、per-expert shape、bytes 与 kernel-facing packed u8 shape。
  - H20 验证：`h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view` 已把真实 K2.5 rank0 payload 加载到 GPU 后通过 package plan 校验。

- [x] `rank_expert_major_raw_buffer_package`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`
  - 已有：`KimiExpertMajorProjectionRawBuffers` / `KimiMoeLayerExpertMajorRawBuffers` / `pack_expert_major_layer_raw_buffers()`。
  - 覆盖：指定 MoE layer 的 gate/up/down `weight_packed`、`weight_scale`、`weight_shape` 从 per-expert raw GPU tensor D2D copy 到 expert-major contiguous raw buffers。
  - 边界：这是 weight-load/package 阶段动作，不进入 decode step；当前输出仍是 checkpoint raw layout，下一阶段接 CUTLASS reorder 后才作为 grouped GEMM kernel package。
  - H20 验证：`/data/models/Kimi-K2.5` rank0 layer1 真实 payload 已通过 raw buffer package bytes/shape 校验。

- [x] `rank_expert_major_kernel_weight_package`
  - 位置：`pegainfer-kimi-k2/src/weights.rs`、`pegainfer-kernels/csrc/kimi_k2/kimi_cutlass_int4_sm90a.cu`、`pegainfer-kernels/src/ops/kimi_experts.rs`。
  - 已有：`KimiExpertMajorProjectionKernelBuffers` / `KimiMoeLayerExpertKernelWeights` / `KimiRankExpertKernelWeights` / `pack_expert_major_layer_kernel_weights()` / `pack_rank_expert_kernel_weights()`。
  - 覆盖：packed 权重 load-time CUTLASS reorder，offset-binary nibble 转 signed int4b_t；scale/shape 进入 typed owning `CudaSlice<bf16>` / `CudaSlice<i32>`；返回对象可直接构造 `KimiInt4ExpertWeights`。
  - 边界：package 阶段允许 allocation、D2D copy 和 CUTLASS reorder；decode step 只借用常驻 package；full-rank package 会先完成全部 60 层转换，再统一释放 raw routed expert tensors，避免 raw + package 双份常驻和中途失败半残状态。
  - worker state：`LoadSlicedWeights` 使用单个 `KimiRankLoadedWeights { gpu, expert_kernels }` loaded state 保存权重，保证 raw non-routed weights 与 expert kernel package 同生共死。
  - 结构 guard：这两条不是实现细节，后续 reset/reload、错误恢复或多 rank worker 接线都必须保持。如果 H20 gate 失败，优先确认 package 失败路径没有留下前 N 层 raw 已删、后续 raw 仍在的半残状态，以及 worker 没有重新拆成两个独立 `Option`。
  - H20 验证：`h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view` 已覆盖 sm90a reorder 实执行、full-rank 60 layer package、统一 raw cleanup、single loaded state、真实 layer1 router GEMM/top8 输出、top8 到 expert-major route/expand/reduce、SwiGLU，以及 W1/W3/W2 通用 CUTLASS prepare+launch 零输入不变量，最近耗时 `23.64s`。

- [ ] `external_expert_fixture_gate`
  - 位置：`tools/kimi_k2/torch_reference.py` 生成 fixture；H20 ignored test 只消费 fixture，不在本仓库内自写第二套 reference。
  - 做法：用 compressed-tensors 官方 pack/dequant 或 vLLM 路径产出 W1/W3/W2、route/reduce 的外部 reference，再比较 PegaInfer kernel 输出。
  - 边界：没有外部 fixture 的子模块测试只能叫 smoke gate，不能叫 parity gate。

- [ ] `pplx_ep_backend_install`
  - 按 DSV4 Flash 的 bootstrap/placement 结构把 PPLX backend 移交给 rank worker。

- [ ] `scheduler_request_state`
  - 管理 request state、KV slot、prefill/decode wave、取消/error cleanup。

- [ ] `worker_decode_commands`
  - 补 prefill/decode/batch decode 命令，worker 持有 CUDA context、weights、KV cache、PPLX scratch。

## 临时 Header 草案

2026-05-20 已在 `/tmp/pegainfer-kimi-k2-headers` 生成一份独立 Rust header/API 草案 crate，用来收敛后续 `pegainfer-kimi-k2` 的模块边界。它只做类型、shape、batch contract 和 unsupported stub，不含 CUDA body。

模块：

| 文件 | 覆盖范围 |
| --- | --- |
| `src/config.rs` | Kimi-K2.6 text-only 常量和 TP8/EP8 derived shapes。 |
| `src/tensor.rs` | 临时 tensor/type vocabulary、stream handle、错误类型。 |
| `src/attention.rs` | MLA projection、YARN RoPE、expanded correctness attention、compressed KV production path、batch decode attention plan；FlashInfer 优先，缺口落到 handwritten CUDA。 |
| `src/dense.rs` | BF16 embedding、RMSNorm、fused add RMSNorm、GEMM、SwiGLU、dense/shared expert、lm_head、greedy top1 header。 |
| `src/router.rs` | Kimi `noaux_tc` router 语义、top8、choice bias、expert-major layout、device-side launch contract。 |
| `src/experts.rs` | compressed-tensors INT4 metadata、packed linear、EP8 grouped expert weights、dequant format probe、fused grouped W1/W3 和 W2+SwiGLU APIs。 |
| `src/collectives.rs` | TP shard policy、当前 NCCL bridge 与后续 PPLX dispatch/combine path、logits all-gather scratch；Kimi EP 不走 NCCL AG/RS。 |
| `src/runtime.rs` | 类 `batch_decode.rs` 的 text-only batch decode orchestration header，支持 bs>1、bucket padding、per-row position/cache metadata。 |
| `src/tokenizer.rs` | text-only tokenizer/chat template contract，thinking/instant/preserve-thinking，多模态显式拒绝。 |

验证：

```bash
cd /tmp/pegainfer-kimi-k2-headers
cargo fmt --check
cargo check
cargo test
```

结果：三项均通过，`cargo test` 为 `5 passed`。
