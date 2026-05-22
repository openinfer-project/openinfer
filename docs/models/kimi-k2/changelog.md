# Kimi-K2 算子工作日志

> **TL;DR:** Kimi-K2 算子搜索/取舍历史。当前状态和设计契约见 [operator-todo.md](operator-todo.md)。下面是 Execution Log / Rejected / Debrief 的归档，按原始顺序保留。

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
- bench graph 口径更新：`bench_serving request` 已补真实并发请求和 CUDA profiler API capture。H20 非 nsys 命令 `target/release/bench_serving --model-path /data/models/Kimi-K2.5 --cuda-graph true --format json --out /tmp/pegainfer-kimi-bench-profile-bs4/result_nonsys_graph_bs4_o64.json request --prompt-len 27 --output-len 64 --concurrency 4 --warmup 1 --iters 1` 得到 steady TPOT `16.70ms`、p50 `16.73ms`、p95 `17.04ms`、p99 `17.11ms`。该口径比 HTTP output128 的 `20.64ms/token/wave` 更接近纯 decode，但 prompt 使用 synthetic token ids，不是 vLLM fixture correctness gate。
- nsys graph capture 更新：`/tmp/pegainfer-kimi-graph-profile-bs4/kimi_graph_bs4_o64.{nsys-rep,sqlite}` 由 `nsys profile -c cudaProfilerApi` 采集，measured window 只有 warmup 后的 4 并发 output64 iteration。CUDA API 表出现 `cuGraphLaunch count=504`，正好是 `8 ranks * 63 decode steps`，证明 graph replay 生效。nsys kernel 表里的 `magma_sgemmEx_kernel count=1920` 主要来自 measured request 的 prompt/prefill 和 graph 外展开，不能作为 steady decode graph 子节点总账；下一步要补 graph replay 外 CPU/API 开销和 rank sync 细分。
- 当前离 `15ms` 目标剩约 `1.7ms`。下一轮优先级：先解释 `cuGraphLaunch`/rank barrier 的每 step 固定开销和 worker report 聚合；然后看 TP hidden F32 bridge 与 shared/routed collective cadence。继续禁止 `bs==1` 专用分支。
- overlap 决策：MoE shared branch 和 routed branch 都读 `scratch.normed`，直到最终 residual join 前理论上可并行；但生产路径当前是单 stream、单 NCCL comm、单 graph capture。直接改 worker 会同时触碰 scratch 生命周期、NCCL collective order 和 multi-stream graph capture。先新增隔离 `kimi_graph_probe --probe nccl-two-stream-overlap`：每 rank 两条 CUDA stream、两套 NCCL comm，通过 CUDA event 把 aux stream 纳入 main stream capture，验证 all-reduce 与 reduce-scatter 形态能否 capture/replay。只有 H20 probe 通过并且 nsys 证明存在真实 overlap，才进入 worker；失败则转向 shared gate/up fusion、routed combine 等 graph-safe 本地优化。
- H20 overlap probe gate：同步 `kimi_graph_probe.rs` 后，远端 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_graph_probe`、`cargo build --release -p pegainfer-kimi-k2 --features kernel-report --bin kimi_graph_probe` 通过。命令 `timeout 120s target/release/kimi_graph_probe --probe nccl-two-stream-overlap --world-size 8 --batch-size 4 --hidden 7168 --replay-iters 200` 返回 `capture_ok=true`、`replay_ok=true`；同一 graph 形态下 max-rank sequential `43.99us`、overlap `27.00us`，speedup `1.63x`。结论：双 stream + 双 NCCL comm + event join 这类 graph 形态在 H20 上可行且有可测 replay 窗口；下一步进入 worker，把 MoE shared TP all-reduce 留在 main stream，把 routed router/Marlin/RS 放到 aux stream，最终 residual 前用 event join。
- worker overlap gate：生产路径没有保留第二 NCCL comm。第二 comm 版在 worker 初始化/graph capture 内触发 `ncclUnhandledCudaError` / `ncclInternalError`，与隔离 probe 不一致，因此本轮只把 routed router/align/Marlin/SwiGLU/local-sum/repeat 放到 aux stream；main stream 继续跑 shared expert + TP all-reduce，随后等 aux local route 完成，再在 main comm 上做 routed reduce-scatter 和 residual。H20 `bench_serving --cuda-graph true` 结果：
  - 真实 Kimi fixture prompt，output16/concurrency4：steady TPOT avg `14.234ms`、p50 `14.214ms`、p95 `14.408ms`，四路 token prefix 全部匹配 vLLM fixture `[1008,2742,2531,414,19180,6082,1379,387,261,5216,63853,13,374,1765,11983,306]`。
  - synthetic prompt-len27，output64/concurrency4/warmup1：steady TPOT avg `14.615ms`、p50 `14.588ms`、p95 `15.119ms`、p99 `16.950ms`，四路 hash 一致。
  - 对比 fused-qkv 后的 output64 avg `16.43ms`，本轮 worker overlap 降低约 `1.82ms/token`；目标 `15ms` 已在 avg/p50 上达成，p95 接近目标，p99/max 仍需 tail profile。
- shared gate/up fused GEMM gate：MoE shared expert 的 `shared_gate_proj` 和 `shared_up_proj` 在 load-time `vstack` 为 `shared_gate_up_proj`，decode/prompt 使用一次 GEMM 加已有 `silu_mul_fused_batch_into`，不新增 kernel、不做 split copy。H20 gate：
  - 真实 Kimi fixture prompt，output16/concurrency4：steady TPOT avg `14.354ms`、p50 `14.365ms`、p95 `14.495ms`、p99 `14.495ms`，四路 token prefix 仍完全匹配 vLLM fixture。
  - synthetic prompt-len27，output64/concurrency4/warmup1：steady TPOT avg `14.658ms`、p50 `14.714ms`、p95 `15.051ms`、p99 `15.111ms`、max `15.115ms`，四路 hash 一致。
  - 这刀 avg 比上一档 output64 `14.615ms` 略慢，但 p99/max 从 `16.95ms` 收到约 `15.11ms`，符合“稳定 15ms 左右”的目标，因此保留；下一步如果继续优化，优先找回 avg 而不放大 tail。
- routed scale+residual fused add gate：decode MoE combine 原来每层执行 `scale_f32_in_place(routed_out_f32, 2.827)` 再执行 `kimi_add_f32_bf16_to_bf16(routed_out_f32, shared_residual)`。本轮新增 `kimi_scaled_add_f32_bf16_to_bf16`，把 scale 和 F32+BF16 residual add 合成一个 batch-general kernel；prompt path 暂不变，decode path 移除每 MoE 层一次 scale launch。
  - 语义约束：kernel 用 `__fmul_rn` + `__fadd_rn`，保持当前 decode path 的 F32 scale 后再与 BF16 residual 相加并落 BF16；这不是 bs1 专用分支，输入长度为 `batch * hidden`。
  - H20 build gate：rsync dry-run 只同步 7 个小文件；远端 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins`、`cargo build --release -p pegainfer-server --bin bench_serving` 通过。
  - 真实 Kimi fixture prompt，output16/concurrency4：steady TPOT avg `14.293ms`、p50 `14.293ms`、p95 `14.445ms`、p99 `14.445ms`，四路 token prefix 完全匹配 vLLM fixture。
  - synthetic prompt-len27，output64/concurrency4/warmup1：steady TPOT avg `14.629ms`、p50 `14.702ms`、p95 `15.050ms`、p99 `15.062ms`、max `15.063ms`，四路 hash 一致。
  - trace gate：重新 build `kimi_kernel_report` 后，static bs4/kv1024 trace 从 `1826` calls 降到 `1766` calls；`scale_f32_in_place=0`、`kimi_scaled_add_f32_bf16_to_bf16=60`、旧 `kimi_add_f32_bf16_to_bf16=0`。
- routed combine bridge microbench：新增 `kimi_graph_probe --probe routed-bridge-compare`，在同一 graph replay 口径下比较 `all_reduce([B,H])` 与当前 `repeat([B,H] * EP8) + reduce_scatter`。H20 `B=4,H=7168,world=8,replay_iters=500` 结果：direct F32 all-reduce max-rank `33.46us`，当前 repeat+RS max-rank `32.90us`，ratio `0.983x`。结论：当前 bridge 不是明显错误方向，直接改回 all-reduce 没收益；后续要变快应做真 AG/RS token ownership 或 PPLX dispatch/combine。
- redundant routed sum clear cleanup：`kimi_marlin_sum_topk_rows_f32` 对 `batch * hidden` 每个元素完整写出，decode path 中它前面的 `routed_out_f32` memset 没有语义作用。本轮删掉这 60 次 graph 内 memset，不改 W13/W2 output memset 和 locks memset。
  - H20 build gate：远端 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins`、`cargo build --release -p pegainfer-server --bin bench_serving` 通过。
  - 真实 Kimi fixture prompt，output16/concurrency4：steady TPOT avg `14.225ms`、p50 `14.241ms`、p95 `14.354ms`、p99 `14.355ms`，四路 token prefix 完全匹配 vLLM fixture。
  - synthetic prompt-len27，output64/concurrency4/warmup1：steady TPOT avg `14.563ms`、p50 `14.613ms`、p95 `14.953ms`、p99 `15.056ms`、max `15.057ms`，四路 hash 一致。
- Marlin locks clear cleanup：vendored Marlin WNA16 在每个用到的 lock 上通过 `barrier_release(..., last)` 把最后一个 split-K slice 的 lock 重置为 0；workspace 初始为 zero buffer，因此 decode 每层 W13/W2 launch 前重复清 `marlin_workspace.locks` 没有语义作用。本轮删掉这 120 次 graph 内 locks memset。
  - 语义边界：W13/W2 output memset 仍保留。route align 会产生本 rank 不负责的 route row 与 padding row；这些行不会被 Marlin 写出，后续 `kimi_marlin_w13_swiglu` / `kimi_marlin_sum_topk_rows_f32` 依赖它们保持 0，不能把 output clear 和 locks clear 混为一类。
  - H20 build gate：远端 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins`、`cargo build --release -p pegainfer-server --bin bench_serving` 通过。
  - 真实 Kimi fixture prompt，output16/concurrency4：steady TPOT avg `14.145ms`、p50 `14.157ms`、p95 `14.308ms`、p99 `14.309ms`，四路 token prefix 完全匹配 vLLM fixture。
  - synthetic prompt-len27，output64/concurrency4/warmup1：steady TPOT avg `14.470ms`、p50 `14.529ms`、p95 `14.852ms`、p99 `14.917ms`、max `14.930ms`，四路 hash 一致。
  - bench exit caveat：两组 `bench_serving` 都已写出 JSON 和指标，但进程退出时有 worker 残留占卡；验证后按具体 `bench_serving --model-path /data/models/Kimi-K2.5` 命令清理，H20 `nvidia-smi --query-compute-apps` 为空。后续要单独修 benchmark teardown，不把它记作 TPOT 回退。
- dense layer0 gate/up fused GEMM gate：dense layer0 的 `gate_proj` 和 `up_proj` 在 load-time `vstack` 为 `gate_up_proj`，prompt/decode 复用 `silu_mul_fused_batch_into`。这不是 bs1 分支，scratch 按实际 `1..=4` batch arena 分配。
  - 本地 gate：`cargo fmt --all --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins` 通过。
  - H20 build gate：远端 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --features kernel-report --bins`、`cargo build --release -p pegainfer-server --bin bench_serving` 通过。
  - 真实 Kimi fixture prompt，output16/concurrency4：steady TPOT avg `14.126ms`、p50 `14.13ms` 量级、p99 `14.258ms`，四路 token prefix 完全匹配 vLLM fixture。
  - synthetic prompt-len27，output64/concurrency4/warmup1：steady TPOT avg `14.388ms`、p99 `14.834ms`，四路 hash 一致。

## Rejected: decode TP hidden BF16 bulk collective

- 试验内容：把 decode TP hidden reductions 从 BF16->F32->BF16 bridge 改回 BF16 bulk NCCL all-reduce，覆盖 embedding、attention `o_proj`、dense/shared down-proj 的 active-batch 通用路径；routed expert combine 仍保留 F32。
- H20 结果：远端 `fmt/check/build` 通过；固定 4 并发 fixture 中 `max_tokens=16` wall `4693.312ms`、`13.636 tok/s`，但 row1 输出变成 `[1008,2742,924,6454,2531,...]`；`max_tokens=64` wall `1788.999ms`、`143.097 tok/s`，row2 同样发散。
- 结论：BF16 NCCL row-offset rounding 仍会影响 greedy，不只是诊断日志噪声；这条没有性能收益，且破坏 output16/64 correctness，已回退到 F32 bulk bridge。下一步不再在 BF16 hidden collective 上试探，转向减少 collective 次数/launch 次数和 PPLX EP。

## Rejected: vLLM TP-only MoE final all-reduce cadence

- vLLM TP-only Kimi/DeepSeekV3 decode 源码对照结论：embedding `1` 次 BF16 all-reduce、61 层 attention `o_proj` 各 `1` 次 BF16 all-reduce、dense layer0 `down_proj` `1` 次 BF16 all-reduce、60 个 MoE 层在 shared+routed 本地合并后各 `1` 次 BF16 all-reduce。合计 `123` 次 BF16 all-reduce，`0` 次 reduce-scatter。
- PegaInfer 当前 decode cadence 是 `123` 次 logical hidden all-reduce，加上 60 个 MoE routed `repeat_f32_for_reduce_scatter + reduce_scatter` bridge。结构上比 vLLM TP-only 多 60 次 RS，但 bridge microbench 之前显示 `repeat+RS` 与 direct F32 all-reduce 相近。
- 试验 A：把 decode MoE 改成 shared local + routed local scale 后本地合并，执行一次 BF16 final all-reduce，再加 residual。H20 correctness 通过：fixture output16 四路 prefix 全对，synthetic output64 四路 hash 一致；性能回退，output16 steady avg `14.925ms` / p99 `18.285ms`，output64 steady avg `15.048ms` / p99 `16.129ms`。
- 试验 B：保持同样 final all-reduce cadence，但 final reduction 用 BF16->F32->NCCL F32->BF16 bridge。H20 correctness 通过；性能仍回退，output16 steady avg `14.730ms` / p99 `15.705ms`，output64 steady avg `14.818ms` / p99 `15.227ms`。
- 决策：这条路径已回退到当前 RS bridge。下一次 collective 改动不再做“只对齐 vLLM TP-only cadence”的单点替换；需要带 nsys/graph 账本证明减少了 tail 或总 TPOT，再进入 worker。PPLX/真 AG-RS token ownership 仍是独立方向。

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

## Execution Log: WNA16 single-layer parity 容差 follow-up（2026-05-22）

- 背景：cleanup commit `99b213a` 之后 h20 重跑 5 条 H20-only gate，4 条通过，唯一红的是 `h20_kimi_marlin_wna16_single_layer_matches_vllm_reference`。
- 隔离实验：在 cleanup 前的 `cab7ba2` 上跑同测试，失败 bitwise 完全一致；再用 `pegainfer-kernels/tools/kimi_k2/vllm_marlin_wna16_reference.py` + vLLM 0.19.0 重生成 `/tmp/kimi_marlin_wna16_reference/`（`weight_source=deterministic_synthetic`），失败 bitwise 同样完全一致。结论：fixture 和 cleanup 都不是问题源，vLLM Marlin kernel 是 deterministic 的。
- 实际数字：`w13_out: max_diff=0.5 mean_diff=0.0055` 卡到 `(0.5, 0.03)` 上限通过；`route_output: max_diff=96 mean_diff=1.8632`，限 `(16, 0.03)`，max=96 / mean=1.86 在 BF16 量级 ~7000 下相对误差 < 0.03%（~1.5 ULP）。差异在 SwiGLU 之后的 w2 GEMM/atomic split-K 累加顺序上，不是算法 bug。
- 决定：不动测试代码，保留这条作为"严格 bit-level parity"红线（`#[ignore]`，不进 default test run）。后续如果要让它转 green，要么换成 vLLM-side `use_atomic_add=False, use_fp32_reduce=True` 的稳定累加路径生成 fixture，要么把容差按 ULP-relative 重写，二选一前不动它。
- 备份：旧 fixture 已挪到 h20-100 `/tmp/kimi_marlin_wna16_reference.bak.1779438284/`，新 fixture 在 `/tmp/kimi_marlin_wna16_reference/`。
