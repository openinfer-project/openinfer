# Kimi-K2.6 文本算子 Bring-up

> **TL;DR:** Kimi-K2.6 非权重文件已拉到 `/data/models/Kimi-K2.6`。当前范围只做文本核心，不支持多模态；H20 验证以 `/data/models/Kimi-K2.5` 作为同架构权重路径。已接入 text-only manifest、TP8/EP8 sliced loader、rank-local typed GPU view、真实 router/top8、SwiGLU、MLA prefill wrapper、8-rank vocab-shard top1 和 vLLM top-20 fixture。routed expert INT4 backend 已从 CUTLASS example69 probe 切到 vLLM Marlin WNA16；真实 K2.5 rank0 layer1 fixture gate 0-diff，真实 K2.5 全 61 层 prompt forward 多 prompt gate 4/4 argmax match。direct worker/scheduler 的 H20 smoke/candidate/debug 测试入口已清理；decode arena 已从 bs1/bs4 两档改为 `1..=4` 按实际 wave size 选 arena，后续优化禁止假设 `bs==1`。scheduler 已接入 bs4 wave decode；Marlin W13/W2 已匹配 vLLM `c_tmp` global-reduce 路径，H20 固定 4 并发 output16 gate 四路 token 全对且 `ROUTER/ROUTE_ROW/ROW` diff 全为 0。CUDA Graph 已覆盖整段 decode；fused qkv_a 已通过 H20 gate，static/runtime trace 为 `1886` calls，synthetic output64 steady TPOT `16.43ms`，真实 fixture output16 四路 token 对齐 vLLM 且 steady TPOT `16.15ms`。
>
> **Last touched:** 2026-05

## 当前测量口径修正

- 2026-05-21 复盘 Qwen3-4B 的 model-local exporter/report 后，Kimi 性能工作先补同类工具链：runtime decode DAG exporter、manifest/snapshot 式单 op report、model-level decode composition report。
- HTTP `/v1/completions` + nsys 全请求窗口只能作为端到端 serving 佐证，不能回答“纯 decode GPU kernel 时间”。它会混入 prompt prefill、frontend、scheduler、lazy init 和 response；之前把 4 并发 output16 trace 里的 `magma_sgemmEx_kernel` 按 output token 平摊，是错误口径。
- 后续 bs1/bs4 TPOT 分析要用 Kimi model report 拆分 prompt/prefill 与 decode steady；H20 keep/revert 仍以 vLLM greedy token gate 不回退为前提。
- 第一版 Kimi report tooling 已接入 crate feature `kernel-report`：`kimi_kernel_report trace` 可导出 bs4/kv1024 的 decode call；trace 修正 MLA 漏项后为 `1947` calls，fused qkv_a 后为 `1886` calls。`kimi_model_report decode` 会按 call count 组合账本，并把未接 provider 的项显式列为 missing。当前采集范围是 rank0 local compute + collective placeholders；MoE EP route imbalance 需要后续 full-rank histogram 扩展，不能用 rank0 账本直接外推 8 卡全局 tail。
- H20 已做非 server 验证：dry-run rsync 后同步 tooling 文件，`PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer-kimi-k2-main/.venv-kimi/bin/python` 下 release check 通过，`kimi_kernel_report trace --batch-size 4 --kv-len 1024` 生成同样 `1765` call 的 JSON。CUDA event provider 已在 H20 跑过 runtime bs1/kv2 和 static bs4/kv1024 model report；CUPTI 本阶段不接入。
- runtime trace 已打通到 direct runtime：`kimi_kernel_report trace --source runtime --batch-size 1 --kv-len 2` 在 H20 通过 `EngineHandle` 提交请求，触发真实 prompt prefill + 一次 decode worker command，导出 `1765` 个 call，退出码 `0`，不经过 HTTP。`kimi_model_report decode --source runtime --iters 1` 也能消费同一 call count 组合 model-level 账本，`schedule_source` 明确为 runtime trace。CUPTI 先不接入，当前目标是解释 decode 性能组成。
- 2026-05-21 event-only report 更新：Marlin WNA16 W13/W2 provider 已接入，复用现有 `kimi_moe_marlin_align_block_size`、`kimi_marlin_wna16_w13_gemm`、`kimi_marlin_wna16_w2_gemm` wrapper；H20 runtime bs1/kv2 账本从 `measured_schedule_calls=1462, missing=303` 改为 `measured_schedule_calls=1582, missing=183`，missing 只剩 `all_reduce=183`。all-reduce missing reason 会带 `rank participation hint=8`，避免把 NCCL collective 当单卡 op 解读。
- Marlin provider 的边界：当前没有真实 per-rank route histogram，synthetic top-k 全部落在本地 48 个 expert 内，所以它测的是 rank-local all-local route 形状，不是 EP8 全局平均 route 分布。H20 static bs4/kv1024 账本显示 measured subset `149.904ms`，其中 synthetic Marlin WNA16 `118.476ms`；这个数只能用于说明“按当前 synthetic route 假设，Marlin 是大项”，不能替代 nsys 中真实 request 的 EP imbalance/tail 结论。

## Preparation

- **Read**:
  - `docs/index.md` — 确认新模型线放在 `docs/models/kimi-k2/`，并定位 DSV4、runtime、scheduler、kernels 相关历史。
  - `docs/models/deepseek-v4/support.md` — 记录 DSV4 初始支持的模型 crate、kernel 构建、E2E 和 HTTP gate 形态。
  - `docs/models/deepseek-v4/pplx-ep-integration.md` — 记录 EP8/PPLX 中 rank 资源、CPU placement、scratch/MR 生命周期和单节点 EP 经验。
  - `docs/models/deepseek-v4/moe-ag-rs.md` — 只借鉴 GPU routing metadata、expert-major compaction 和 grouped expert GEMM 经验；Kimi 不走 NCCL AG/RS。
  - `docs/models/deepseek-v4/kernel-paths.md` — 确认模型专属 CUDA 和 TileLang generator 应进入 `pegainfer-kernels` 的模型子目录。
  - `docs/subsystems/runtime/runtime.md` — 确认共享 runtime 只暴露 generation contract，新模型执行复杂度留在模型 crate。
  - `docs/subsystems/scheduler/scheduler.md` — 确认 scheduler 的 GPU 所有权、paged KV、CUDA Graph 和 bs>1 约束。
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` — 确认 PegaInfer 方向是共享 frontend/runtime/data-plane + per-model engine。
  - `/data/models/Kimi-K2.6/config.json` — 文本核心是 `DeepseekV3ForCausalLM` 风格 `kimi_k2`，外层 `KimiK25ForConditionalGeneration` 的 vision/projector 本阶段忽略。
  - `/data/models/Kimi-K2.6/modeling_deepseek.py` — 确认 MLA attention、YARN RoPE、`noaux_tc` MoE gate、shared expert 和 routed expert 数据流。
  - `/data/models/Kimi-K2.6/model.safetensors.index.json` — 确认 64 shard、约 595GB，总体是一层一个 shard；routed expert 权重是 `weight_packed/weight_scale/weight_shape`。
  - `/data/models/Kimi-K2.6/README.md` 与 `/data/models/Kimi-K2.6/docs/deploy_guidance.md` — 记录官方 API 示例、vLLM/SGLang TP8 serve 命令、thinking/instant/preserve-thinking 参数。
  - `/data/models/Kimi-K2.6/chat_template.jinja` 与 `/data/models/Kimi-K2.6/tokenization_kimi.py` — 确认 text-only prompt 渲染、thinking 开关、tool declaration 和 tool call section 格式。
  - `pegainfer-kernels/KERNELS.md` 与 `pegainfer-deepseek-v4/src/runtime/*` — 对照 DSV4 已有 FP4 grouped expert、routing、PPLX 和 BF16/cuBLAS wrapper。
- **Relevant history**:
  - DSV4 证明 MoE decode 热路径不能把 route/expert metadata 搬回 host；Kimi 也应从第一版算子起保持 GPU routing/compaction。
- DSV4 PPLX 经验说明 EP8 不是单个 kernel 问题，scratch、rank worker、CPU placement、buffer 注册和 CUDA context 都必须作为算子执行环境的一部分设计。
- Kimi EP 的生产目标仍是 PPLX dispatch/combine；当前 direct runtime 明确走临时 NCCL-sum bridge，用于端到端 correctness/perf bring-up，不再沿用旧的 PPLX 命名误导实际路径。
- DSV4 grouped FP4 路线可复用组织方式，但 Kimi routed expert 是 compressed-tensors INT4，不能直接复用 FP4/E8M0 TileLang kernel。
- Kimi decode graph contract 先覆盖 rank-local compute kernels：热路径不 D2H、不 host sync、不 step 内分配，所有 scratch 和 metadata 预分配且 device resident。PPLX EP dispatch/combine 先作为 graph 外阶段处理，等接入真实 PPLX backend 后再用 capture harness 验证能否纳入。
- 2026-05-21 新增 Marlin/WNA16 route alignment metadata：`kimi_moe_marlin_align_block_size_cuda` 生成 `sorted_token_ids`、`expert_ids`、`num_tokens_post_padded`，按本地 EP rank 忽略非本地 experts，保持 device resident。它是接 vLLM Marlin compute ABI 的前置契约，不是数值 parity gate。
- 2026-05-21 rank expert loader 提供互斥的 CUTLASS probe package 与 Marlin/WNA16 package 路线；full-rank runtime state 不能同时常驻两套 routed expert package。Marlin package 最终常驻 fused W13 + W2，包含 uint4b8 no-actorder packed weight 与 group-major+perm64 BF16 scale，后续 compute ABI 不再依赖 checkpoint raw tensors。H20 已用真实 `/data/models/Kimi-K2.5` 验证 rank0 60 层 fused W13 + W2 package load、Marlin weight no-act、scale perm64、route alignment 和 typed GPU view。
- 2026-05-21 Marlin/WNA16 compute wrapper 必须按 H20 vLLM `0.19.0` ABI 接，不按更新的 yingshan vLLM header：当前 fixture op 没有 `is_ep` 参数。H20 单层 synthetic W13 + SwiGLU + W2 + topk reduce 已对 vLLM reference 0-diff；真实 `/data/models/Kimi-K2.5` rank0 layer1 fixture gate 同样在 W13、SwiGLU 后 W2、top-k sum 三段 0-diff。
- 2026-05-21 `pegainfer-kimi-k2/src/weights.rs` 已拆成 flat module：`weights.rs` 保留类型和公开入口，`weights/context.rs`、`weights/load.rs`、`weights/manifest.rs`、`weights/package.rs`、`weights/tests.rs` 分别承载 GPU context、safetensors load、manifest、Marlin package 和 H20 gates。旧 CUTLASS raw/kernel package helper 与自写 dequant+cuBLAS self-comparison gate 已从主代码删除，数值 gate 只对外部 vLLM fixture。
- 2026-05-21 worker routed expert backend 已切到 Marlin/WNA16。H20 真实 `/data/models/Kimi-K2.5` all-rank forward smoke 与 prompt candidate dump 均通过；candidate metadata 显示 `attention_layers_stubbed=0`、`remaining_layers_stubbed=0`、`moe_layers_executed=60`。多 prompt vLLM gate 覆盖 `hello/math_short/self_intro_zh/code_rust`，4/4 greedy argmax match，top-20 id overlap 最低 `19/20`。
- 2026-05-21 worker decode attention 起步已落地：`KimiWorkerDecodeArena` 常驻 bs4 paged ckv/kpe cache、device plan arrays、YARN RoPE cache 和 scratch；历史 H20 smoke 用真实 K2.5 rank0 layer0 权重跑通过 `q/kv projection -> compressed KV append -> FlashInfer MLA decode -> v-up -> o_proj`，all-rank 版本覆盖过 token ids、vocab-shard embedding、TP all-reduce、worker decode arena 和 `o_proj` 后 TP all-reduce。这只证明 worker 层真实权重和 cache wiring，不替代 full decode parity。
- 2026-05-21 决策更新：Kimi 后续不再靠继续新增内部 smoke 推主线；下一阶段直接在 H20 端到端请求路径上推进，优先跑 `pegainfer-server` / `bench_serving` / OpenAI-compatible `/v1/completions`。direct crate 的 H20 smoke/candidate/perf 测试入口已移除，新增修复按真实请求阻断点排序：frontend/server、prompt/tokenizer、scheduler token loop、prefill KV 保存、decode cache/body、PPLX EP、sampling/logits 聚合。
- 2026-05-21 主请求路径清理：server/scheduler 生产路径从 `forward_prompt_smoke` 改成 `forward_prompt_next_token`；临时 full-decode smoke 代码未进入 H20 主入口并已删除。direct worker/scheduler 的 `#[cfg(test)]` 现在只保留 CPU 小单测。
- 2026-05-21 H20 server 端到端 gate：`pegainfer-server` 用真实 `/data/models/Kimi-K2.5` 启动成功，engine load `131754ms`；raw `"Hello"` 请求返回 token id `950`；vLLM fixture 27-token prompt 返回 token id `1008`，与既有 vLLM greedy 对齐。`max_tokens=2` 当前仍只返回 1 token，`logprobs=5` 当前因 Kimi scheduler 不返回 logprob payload 而 500。
- 2026-05-21 scheduler token loop bridge：`handle_request` 已按 `req.max_tokens` 循环发 token；H20 重建 `target/release/pegainfer` 后，vLLM fixture 27-token prompt 的 `max_tokens=2` 返回 `token_ids=[1008,2742]`、text `The user`、`completion_tokens=2`。这仍是 full-prompt recompute bridge，下一步要替换为 prefill KV + decode state。
- 2026-05-21 decode page table 修正：`page_size=16` 只表示每页 token 数，不限制 prompt 长度。decode arena 现在按 bs4、`128 pages/request` 分配 `max_pages=512`，RoPE/cache 覆盖 `2048` token/request；`append_position=26` 时 request0 使用 page `0,1` 且 `last_page_len=11`，所以 27-token prompt 正常由 2 pages 承接。
- 2026-05-21 prompt prefill KV 写入：每层 MLA prefill 在得到 `compressed_normed [seq,512]` 后，先用新增 `kimi_mla_rope_apply_kpe_cuda` 把 raw `k_rope [seq,64]` 转成 RoPE 后 `append_kpe [seq,64]`，再调用 `kimi_mla_paged_kv_append` 写入 worker-owned paged KV。该阶段 H20 server 验证保持 vLLM fixture prompt 的 `max_tokens=1/2` 输出为 `1008` / `[1008,2742]`；后续 true decode bridge 已把第 2 个 token 改成走这份 KV。
- 2026-05-21 scheduler wave bs4 decode：`handle_request_batch` 先为最多 4 个请求分别走 slot-local prompt forward，第 2 个 token 起调用 `forward_decode_batch_next_tokens(token_ids, append_positions, slots)`。worker decode path 使用真实 bs4 arena，执行 61 层 MLA decode + dense/MoE + final logits；本地 release check 通过，H20 server 4 并发 fixture prompt 的 `max_tokens=2` 四路返回 `[1008,2742]`，`max_tokens=8` 四路返回 `[1008,2742,2531,414,19180,6082,1379,387]`，HTTP 端到端输出吞吐 `56.8 tok/s`。
- 2026-05-21 decode scratch 预分配：dense/shared/router/Marlin/top1 scratch 已挪进 worker-owned arena；Marlin WNA16 locks、W13/W2 output 和 routed f32 reduce buffer 复用前必须 device zero-fill，否则非本地 route / padding route 会带入 stale 值，H20 表现为长于 2 token 的 decode 发散。
- 2026-05-21 batched top1 + decode profile：直接用 FlashInfer `TopKDispatch(num_rows=4)` 曾让一路从第 3 token 发散，改成同 stream row-loop wrapper 后恢复 `max_tokens=8` 四路一致，HTTP output8 为 `57.4 tok/s`。该优化只合并 Rust 侧 sync/D2H，profile 证明 top1 只有约 `0.09ms/step`，不是当前主瓶颈。强同步分段 profile 稳态约 `35.0ms/bs4 step`，纯 decode 总吞吐约 `114 tok/s`；MoE `22.8ms/step` 最大，其中 shared expert+TP all-reduce `6.55ms`、routed reduce/add+f32 all-reduce `6.37ms`、router `3.70ms`、align `1.31ms`，Marlin W13/W2 合计约 `4.0ms`。
- 2026-05-21 active-batch 清理与禁令：固定 bs4 decode scratch 已拆成按实际 wave size 选择的 `1..=4` arena，避免 1/2/3 并发按 4 行执行；但禁止新增 `bs==1` 假设优化。bs1 no-barrier 实验曾把暖态 `32->64` 差分 TPOT 降到 `18.01ms`，但已 rejected 并回退；下一步必须解释 CPU barrier 的真实必要性，并围绕 bs>1/bs4 减少每 token 约 183 次 NCCL collective cadence。
- 2026-05-21 nsys tail profile：H20 trace 产物在 `/tmp/pegainfer-kimi-nsys/kimi_bs4_decode.{nsys-rep,sqlite}`，summary 在 `/tmp/pegainfer-kimi-nsys/tail-summary.md`。本轮明确把 `std/p95/p99/max` 纳入性能 gate：BF16 all-reduce `p50=74.7us` 但 `p99=780us`、`max=2.98ms`；F32 all-reduce `p50=64.8us` 但 `p99=385us`、`max=886us`；Marlin WNA16 `p50=14.3us` 但 `p99=154us`、`max=187us`；`cuStreamSynchronize` `p50=28.3us` 但 `p99/max=9.87ms`。这说明 tail/rank skew/API drain 比单看 p50 更关键。
- 2026-05-21 DSV4 Flash 经验迁移：Kimi 后续性能工作不应继续围绕单个 top1/helper 小优化。DSV4 文档显示 logical collective/rank arrival skew、PPLX worker cadence、launch fanout、CPU placement 和大粒度 graph 才是 decode p50 的主要杠杆；Kimi profile 也指向 MoE shared/reduce/router/align 和 TP/EP collectives。下一步优先 PPLX EP 与 MoE combine/dataflow，保留 per-NUMA rank slice/CPU0+CPU1 预留策略，并避免独立 compact/scatter 这类新增 per-layer launch 的路径。
- 2026-05-21 decode rank-phase gate 降级：去掉 decode stream sync 后，deterministic batched argmax 仍会在暖批出现单路 `[1008,2742,924,6454,...]` 发散，排除了 top1 row-state。`cudarc` NCCL comm 已经绑定同一条 rank stream，所以不是同 stream 缺 GPU sync；在每个 decode BF16/F32 all-reduce enqueue 前增加 CPU `Barrier` 后，H20 4 并发 `/v1/completions` 曾连续通过 `max_tokens=8/16`，但后续 warm bs4 output16 又复现两路坏前缀、两路正确前缀。该 barrier 只能算历史上降低复现率的临时 rank-phase 对齐，不是 correctness guard；相关 `81 tok/s` 只能作为 failing run 背景，不作为有效 decode 吞吐。
- 2026-05-21 shared+routed reduce 合并试验已回退：把 shared expert local BF16 输出累加进 routed F32 buffer、每层只做一次 F32 all-reduce，短输出 `max_tokens=16` 可到 `85.7/87.1 tok/s`，但 H20 首批 `max_tokens=22` 复现 `[1008,2742,924,6454,...]` 冷批发散，`max_tokens=64` 四路不一致且会污染后续请求状态。主线恢复为 shared BF16 all-reduce + routed F32 all-reduce；减少 collective cadence 仍要走 PPLX EP 或重新设计带外部 parity gate 的合并点。
- 2026-05-21 NCCL barrier 根因排查先收紧错误边界：sub agent review 指出 decode collective 顺序在代码上跨 rank 一致，CPU barrier 影响 correctness 更像 host enqueue phase/rank skew/tail 状态，或更早 CUDA/cuBLAS 错误延迟到 NCCL 才暴露。本轮 `gemm_cuda` / `gemm_graphsafe_cuda` 改为返回状态，checked Rust wrapper 会在 operator 边界报 cuBLAS/CUDA 错；Kimi decode GEMM 对 active batch `1..=4` 统一走 workspace-free graph-safe handle；scheduler 在 prompt/decode rank report 聚合前校验 slot/token/vocab/layer/stub 协议，避免错误报告被 max-by 选中。
- 2026-05-21 toxic-review 结论采纳：GEMM/report 只排除了 immediate launch/protocol failure，没有校验 payload。warm bs4 output16 的两路分叉说明 row hidden/KV/MoE 状态已经坏得很规整。下一步临时 instrumentation 不再扩 report 字段，而是在 rank worker 内对同 token/同 position 的 active rows 做 first-diff：embedding all-reduce 后、MLA append/decode 后、MoE shared/routed reduce 后、layer output 后。
- 2026-05-21 H20 row first-diff 第一轮：4 并发 `max_tokens=16` 复现 1 路坏前缀、3 路正确，下一轮全正确；server 日志只打印到一条 `KIMI_DECODE_ROW_DIFF rank=6 phase=mla_residual layer=Some(0) active_len=4 tokens=[1008,1008,1008,1008] positions=[27,27,27,27] row=1 dim=0 row0=-0.00017929077 row1=-0.00016403198 first_abs_diff=0.000015258789 max_abs_diff=0.001953125`。旧诊断使用全局 single atomic，不能证明 `mla_projected` 等更早 phase 在所有 rank 上都干净；新一轮本地补丁改为每个 phase 最多打印一条，并新增 `mla_projected_allreduce` / `mla_residual_add` 切点。
- 2026-05-21 row-wise F32 collective 短 gate：H20 4 并发 fixture `max_tokens=16` 连续 4 轮 greedy token ids 全部匹配 vLLM fixture。冷批 `4711.523ms / 13.584 tok/s`，暖批约 `923ms / 69.2 tok/s`。但 `ROWDIFF_COUNT=256`，first dirty 仍从 layer1 `moe_routed_reduce` 开始，差异量级降到约 `1e-6..4e-5`。结论是短输出可稳定，不代表 payload parity；row-wise collectives 只能作为诊断桥，不能作为最终性能路径。H20 GPU 已停止占用，port `18080` 和 `nvidia-smi --query-compute-apps` 均确认空闲。
- 2026-05-21 local routed cut：toxic-review 路径建议先切开 `moe_routed_local` 与 `moe_routed_reduce`，避免继续把短 output16 当 parity。新增切点后 H20 只跑 1 轮 4 并发 fixture `max_tokens=16`，输出全对但 `ROWDIFF_COUNT=256`；`KIMI_DECODE_ROUTER_DIFF` 无输出，第一批 diff 已经在 layer1 `moe_routed_local`，也就是 `kimi_marlin_sum_topk_rows_f32` 后、NCCL 前。结论：本地 routed expert path 是当时主嫌疑。H20 GPU 已停止占用，port `18080` 和 `nvidia-smi --query-compute-apps` 均确认空闲。
- 2026-05-21 Marlin atomic 修复：对照 vLLM 源码发现 `fused_marlin_moe.py` 的 W13/W2 都传 `use_atomic_add=False,use_fp32_reduce=True`，而 PegaInfer wrapper 固定 `use_atomic_add=true` 且不传 `c_tmp`，split-K 时会用 BF16 atomicAdd 直接写 C。修复为 worker/decode arena 预分配 `c_tmp`，launch 走 vLLM global-reduce 路径。H20 固定 4 并发 fixture `max_tokens=16`：wall `5109.881ms`、`12.525 tok/s`，四路 token ids 全对，`ROUTER_COUNT=0`、`ROUTE_ROW_COUNT=0`、`ROW_COUNT=0`。这把 row-state bug 收敛掉；下一步回到 decode(bs4) 性能和 vLLM top-k/logit parity。
- 2026-05-21 decode 诊断负担清理：同 token row-diff D2H 主路径硬关，decode F32 collectives 从 per-row loop 改回单次 contiguous all-reduce，decode collective CPU barrier 不再执行。H20 固定 4 并发 fixture：`max_tokens=16` wall `4615.953ms`、`13.865 tok/s`，四路 token ids 全对；warm `max_tokens=64` wall `1774.731ms`、HTTP 端到端输出吞吐 `144.247 tok/s`，四路 prefix/tail 一致；`ROUTER/ROUTE_ROW/ROW` diff 计数全为 0。H20 GPU 已释放。
- 2026-05-21 routed MoE decode reduce-scatter bridge：上一版 BF16 all-gather 会把 local expert compute 按 EP world 放大，已撤掉。当前代码保持 local router/Marlin 只跑本 rank 实际 batch 行数，在 `kimi_marlin_sum_topk_rows_f32` 后用 `repeat_f32_for_reduce_scatter_cuda` 生成 reduce-scatter send buffer，再 `reduce_scatter` 回本地 `[B,H]`。这仍是 NCCL bridge，不是真 PPLX EP；H20 graph probe 已证明 NCCL RS 本身可 capture。
- 2026-05-21 CUDA Graph gate：当前 correctness baseline 为 warm `max_tokens=128` 四并发 `24.92ms/token`、`160.5 tok/s`，四路前 16 token 与 fixture 一致。第一次整段 decode graph capture 在 H20 `max_tokens=2` 四并发 hang；复查后新增 `kimi_graph_probe`，local kernel、cuBLAS GEMM、NCCL all-reduce、NCCL reduce-scatter 均可 capture/replay，根因是 Kimi rank workers 独立 begin/end/launch，NCCL graph capture 缺少跨 rank 阶段对齐。`CudaGraphState` 增加同步 phase hook 后，Kimi worker 在 graph begin/enqueue/end/launch 周围用 rank barrier 对齐；H20 `--cuda-graph true` gate 通过：`max_tokens=2` 四路 `[1008,2742]`，warm HTTP output64 `168.1 tok/s / 23.80ms/token/wave`，warm HTTP output128 `193.8 tok/s / 20.64ms/token/wave`，prefix/tail 一致。随后把 `bench_serving request` 补成真实并发请求和 CUDA profiler API capture 后，H20 非 nsys `--concurrency 4 --output-len 64 --warmup 1 --iters 1` 得到 steady TPOT `16.70ms`、p95 `17.04ms`、p99 `17.11ms`；这比 HTTP 口径更接近纯 decode。nsys graph capture 产物 `/tmp/pegainfer-kimi-graph-profile-bs4/kimi_graph_bs4_o64.{nsys-rep,sqlite}` 显示 `cuGraphLaunch count=504 = 8 ranks * 63 decode steps`，证明 measured iteration 走 graph replay。下一步瞄准 graph replay 下剩余约 `1.7ms`：graph launch/rank synchronization、TP hidden F32 bridge、shared expert GEMM+collective、routed NCCL bridge。
- **Plan**:
  1. 拉取 `/data/models/Kimi-K2.6` 非权重 snapshot，并确认未下载权重大文件。
  2. 只分析 `language_model` 文本路径，明确排除 `vision_tower`、`mm_projector`、media token 插入和图像预处理。
 3. 从 HF config/code/index 提取文本 operator surface 和 TP8/EP8 形状。
  4. 从 README/deploy guide/chat template 提取首版 text-only 推理入口契约。
  5. 按算子类别写 TODO list：已有可复用、需要参数化、需要新写、需要权重 header 验证。
  6. 初始化 `pegainfer-kimi-k2` crate，把 header/API 草案接进 workspace 和 server model detection。
  7. 按 DSV4 Flash 形态补 Kimi direct scheduler / rank worker 骨架；server 能识别 Kimi，但请求在 runtime decode 未接线前返回明确 error。
  8. 写 text-only safetensors index manifest 和 TP8/EP8 rank ownership。
  9. 后续实现从 operator harness 和 worker runtime body 开始。
- **Risks / open questions**:
  - `weight_scale` dtype、packed INT4 nibble 顺序、scale layout 需要读取 safetensors header 或少量权重后确认。
  - HF 代码的 cache 是 expanded K/V；生产 256K context 不能按 expanded KV 长期存，MLA compressed KV cache 需要单独设计。
  - TP8/EP8 的权重切分策略要和 checkpoint shard/index 对齐；当前 index 只能看到 tensor 名和 shard，不含 tensor shape/dtype header。

## Execution Log

### Step 1: 拉取非权重文件到 `/data/models`

- 命令：

```bash
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path

local_dir = Path('/data/models/Kimi-K2.6')
local_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id='moonshotai/Kimi-K2.6',
    repo_type='model',
    local_dir=str(local_dir),
    ignore_patterns=[
        '*.safetensors', '*.bin', '*.pt', '*.pth', '*.ckpt', '*.gguf',
        '*.onnx', '*.tflite', '*.h5', '*.msgpack',
        'optimizer.*', 'scheduler.*', 'pytorch_model*', 'tf_model*',
        'flax_model*', '*.tar', '*.zip', '*.zst'
    ],
)
PY
```

- 结果：
  - 拉取完成：`/data/models/Kimi-K2.6`
  - `find /data/models/Kimi-K2.6 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.gguf' \)` 无输出。
  - 目录大小约 `26M`。

### Step 2: 文本核心事实

- 范围：忽略多模态，只支持 `language_model`。
- 外层模型：`KimiK25ForConditionalGeneration`，但文本核心配置 `text_config.model_type = kimi_k2`，HF 类是 `DeepseekV3ForCausalLM`。
- 规模：
  - `hidden_size = 7168`
  - `vocab_size = 163840`
  - `num_hidden_layers = 61`
  - `first_k_dense_replace = 1`，第 0 层 dense MLP，第 1-60 层 MoE
  - `max_position_embeddings = 262144`
- MLA attention：
  - `num_attention_heads = 64`
  - `num_key_value_heads = 64`
  - `q_lora_rank = 1536`
  - `kv_lora_rank = 512`
  - `qk_nope_head_dim = 128`
  - `qk_rope_head_dim = 64`
  - `q_head_dim = 192`
  - `v_head_dim = 128`
  - `o_proj` 输入维度是 `64 * 128 = 8192`，输出 `7168`
- RoPE：
  - `rope_theta = 50000`
  - YARN：`factor = 64`，`original_max_position_embeddings = 4096`，`beta_fast = 32`，`beta_slow = 1`
- MoE：
  - `n_routed_experts = 384`
  - `num_experts_per_tok = 8`
  - `n_shared_experts = 1`
  - `moe_intermediate_size = 2048`
  - gate: `sigmoid` scoring，`topk_method = noaux_tc`，`n_group = 1`，`topk_group = 1`
  - `norm_topk_prob = true`
  - `routed_scaling_factor = 2.827`
- Quantization：
  - `compressed-tensors` / `pack-quantized`
  - routed expert `Linear` 权重是 INT4 group quant，`group_size = 32`
  - config ignore 列表排除了 attention、shared experts、dense MLP、lm_head、vision、projector，所以首批 INT4 工作只针对 routed experts。
- 权重 index：
  - `metadata.total_size = 595148192736`
  - 64 个 safetensors shard
  - `language_model.model.layers.0` 是 dense 层，只有 BF16 dense MLP 和 attention 权重。
  - 每个 MoE 层有 `384 experts * 3 projections * (weight_packed, weight_scale, weight_shape) = 3456` 个 routed expert entries。

### Step 3: 仓库自带推理入口

- `README.md` 的使用示例是官方 OpenAI-compatible API 调用，不是本地 `transformers.generate()` 完整脚本。
- `docs/deploy_guidance.md` 给出第三方引擎启动方式：
  - vLLM: `vllm serve $MODEL_PATH -tp 8 --mm-encoder-tp-mode data --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2`
  - SGLang: `sglang serve --model-path $MODEL_PATH --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2`
  - KTransformers 示例用 `--kt-method RAWINT4`，说明其异构路径也把原生 INT4 当核心权重格式。
- README 明确：
  - thinking mode 默认开启。
  - thinking 推荐 `temperature=1.0`，instant 推荐 `temperature=0.6`，`top_p=0.95`。
  - vLLM/SGLang instant mode 通过 `extra_body={'chat_template_kwargs': {'thinking': False}}`。
  - preserve thinking 通过 `chat_template_kwargs: {"thinking": True, "preserve_thinking": True}`。
- `tokenization_kimi.py` 的 `apply_chat_template` 默认参数：
  - `tokenize=False`
  - `add_generation_prompt=True`
  - `thinking=True`
  - `preserve_thinking=False`
  - tools 会先 `deep_sort_dict`，再尝试转 TypeScript-style declaration，注入 `tools_ts_str`。
- `chat_template.jinja` 的 text-only 关键格式：
  - system: `<|im_system|>{role/name}<|im_middle|>{content}<|im_end|>`
  - user: `<|im_user|>{role/name}<|im_middle|>{content}<|im_end|>`
  - assistant generation prompt: `<|im_assistant|>assistant<|im_middle|><think>`，instant mode 则是 `<think></think>`。
  - history assistant 默认会把历史 reasoning 清空为 `<think></think>`，suffix/preserve-thinking 才保留 `reasoning` / `reasoning_content`。
  - tool declaration 前缀是 `<|im_system|>tool_declare<|im_middle|>...<|im_end|>`。
- 结论：首版 text-only runner 应先实现 `chat_template.jinja` 的文字路径和 thinking/instant 开关；tool parser/reasoning parser 是 frontend 输出解析问题，不阻塞底层算子 bring-up。

### Step 4: 初始化 `pegainfer-kimi-k2` crate

- 已从 `/tmp/pegainfer-kimi-k2-headers` 把 header/API 草案迁入工作区 `pegainfer-kimi-k2/`。
- `pegainfer-kimi-k2` 当前承载 text-only Kimi-K2.6 的配置 probe、shape 常量、batch decode header、MLA/router/dense/experts/collectives/tokenizer API 草案，以及 DSV4 Flash 风格的 direct scheduler / rank worker 骨架；runtime 已接真实权重、MLA KV 和 decode body，但 EP combine 仍是 NCCL bridge，不是 PPLX backend。
- workspace 已加入新 crate，server 侧已加入 `ModelType::KimiK2`：
  - 检测 `config.json` 顶层 `model_type = kimi_k25 | kimi_k2`。
  - 检测 `text_config.model_type = kimi_k2`。
  - Kimi 分支优先于通用 `text_config` Qwen3.5 检测，避免多模态外壳被误判。
- `start_engine` 现在会读取并验证 config，要求 `device_ordinals=0..7` 且关闭 CUDA Graph，随后启动 `kimi-k2-scheduler` 和 8 个 `kimi-k2-rank-*` worker skeleton；请求会收到 `Scheduled` 后返回 runtime decode 未接线 error。
- Kimi direct runtime 当前字段名已改为 `use_nccl_ep_bridge=true`：它显式表示 TP sums 与 MoE shared/routed combine 仍走 NCCL all-reduce。PPLX EP 是后续替换项，不再伪装成当前已接入。
- 已验证：

```bash
cargo fmt --check
cargo check --release -p pegainfer-kimi-k2
cargo test -p pegainfer-kimi-k2
cargo check --release -p pegainfer-server
```

### Step 9: CPU binding 与 decode graph boundary

- 新增 `pegainfer-kimi-k2/src/direct/affinity.rs`，按 DSV4 Flash 策略：
  - CPU0 作为系统预留。
  - CPU1 作为 scheduler 目标 CPU；当前 affinity mask 不含 CPU1 时跳过 scheduler pin。
  - 每个 rank 根据 CUDA device NUMA node 拿 CPU pool。
  - 调用 `split_rank_cpu_slices` 按 NUMA/rank 切连续 CPU slice。
  - rank worker pin 到该 rank slice 的第一个 CPU。
  - `KimiRankThreadPlacement` 保留 `role_cpu(offset, role)`，给后续 PPLX TE/A2A/UVM worker 按 DSV4 的 offset 分配。
- scheduler 启动时构建 `KimiRankThreadPlacementPlan`，传入 runtime config；rank worker spawn 时执行 pin。
- 新增 `KimiK2DecodeGraphContract::graph_ready()`，作为 rank-local decode kernel/header 的硬约束：
  - route/count/indptr/combine metadata 必须 device resident。
  - rank-local decode compute hot path 禁止 D2H。
  - rank-local decode compute hot path 禁止 host sync。
  - rank-local decode compute step 内禁止 allocation。
  - CUDA Graph replay 覆盖的 buffer pointer 必须稳定。
  - scratch 必须预分配。
  - PPLX EP dispatch/combine 当前明确在 CUDA Graph capture 外；EP 相关 metadata/buffer 仍按 device-resident、预分配、无 D2H 约束设计。
- `KimiK2BatchDecodePlan::new` 现在先校验 graph contract，再生成 plan。
- 已验证：

```bash
cargo fmt --check
cargo test -p pegainfer-kimi-k2 -- --nocapture
cargo check --release -p pegainfer-kimi-k2
cargo check --release -p pegainfer-server
```

### Step 5: direct scheduler / rank worker 骨架

- 新增 `pegainfer-kimi-k2/src/direct.rs`、`src/direct/scheduler.rs`、`src/direct/worker.rs`。
- `KimiK2DirectRuntimeConfig` 记录 model path、text config、text weight manifest、rank weight plan、rank placement 和 `use_nccl_ep_bridge=true`；这个字段只表示当前临时 NCCL-sum bridge，不表示 PPLX EP。
- `KimiK2RankPlacement` 固定 TP8/EP8 rank 到 8 张卡；当前建立 worker 生命周期，并把对应 `KimiRankWeightPlan` 移交给每个 worker，后续接权重 payload、CUDA context、PPLX backend、KV cache 和 decode 命令。
- scheduler 行为：
  - 通过 `EngineHandle` 接收 `GenerateRequest`。
  - 发送 `TokenEvent::Scheduled`。
  - 在 runtime decode 未接线前发送明确 `TokenEvent::Error`。
- 下一步按 DSV4 Flash 拆：
  - weight manifest/load 到每个 rank worker。
  - worker 当前持有 NCCL comm + decode arena；后续 PPLX EP 接入时再增加 backend 和 scratch ownership。
  - scheduler 管 request state、KV slot、prefill/decode wave、finish/error cleanup。

### Step 6: text-only safetensors index manifest

- 新增 `pegainfer-kimi-k2/src/weights.rs`。
- `KimiK2WeightManifest::from_model_dir` 只读取 `model.safetensors.index.json`，不读取权重 payload。
- manifest 覆盖：
  - `language_model.model.embed_tokens.weight`
  - 61 层 attention / norm
  - layer 0 dense MLP
  - layer 1..60 router、shared expert、384 个 routed expert INT4 三元组
  - final norm 和 lm_head
- 多模态 tensor 只统计为 ignored，不进入 text manifest。
- `rank_plan(rank)` 固定 TP8/EP8：
  - attention heads 每 rank 8 个。
  - vocab 每 rank 20480。
  - routed experts 每 rank 48 个。
  - router gate/bias replicated。
- `rank_weight_names(rank)` 生成 rank-local typed view：top-level、attention、dense layer0、MoE router、shared expert、本地 48 个 routed experts。
  - `rank_shard_plan(rank)` 按 safetensors shard 分组 tensor 名，用于 header/manifest 探测。
  - `rank_sliced_load_plan(rank)` 在 shard 分组之外记录 TP8/EP8 slicing：vocab 行切、attention head 行/列切、dense/shared MLP 行/列切、本地 routed expert 全量读取。
- `start_engine` 启动时已经加载 manifest 并为 8 个 worker 生成 rank plan、typed names、shard plan 和 sliced load plan；rank worker 不再需要猜 tensor 名、expert range、TP slice 或 shard 分组。
- 本地 `/data/models/Kimi-K2.6/model.safetensors.index.json` 验证结果：
  - text tensor：`208215`
  - ignored non-text tensor：`335`
  - shard：`64`
  - 每 rank tensor plan：`26775`
  - 每 rank shard read plan：`62` 个 shard
  - rank7: heads `56..64`，vocab `143360..163840`，experts `336..384`
- 已验证：

```bash
cargo fmt --check
cargo test -p pegainfer-kimi-k2 weights -- --nocapture
cargo test -p pegainfer-kimi-k2 -- --nocapture
cargo check --release -p pegainfer-kimi-k2
cargo check --release -p pegainfer-server
```

### Step 7: rank-local typed weight view

- `KimiRankWeightNames` 将 manifest 收敛成运行时需要的名字结构：
  - `KimiTopWeightNames`
  - `KimiAttentionWeightNames`
  - `KimiDenseMlpWeightNames`
  - `KimiMoeLayerWeightNames`
  - `KimiRoutedExpertWeightNames`
- `KimiRankShardPlan` 将同一 rank 的 `26775` 个 tensor 按 shard 分组；本地 K2.6 index 上每 rank 需要读 `62` 个 shard。
- `KimiRankWorker` 当前持有：
  - `KimiRankWeightPlan`
  - `KimiRankWeightNames`
  - `KimiRankShardPlan`
  - `KimiRankSlicedLoadPlan`
- 下一步 `rank_weight_loader` 在 worker 内应优先调用 slice-aware header/GPU copy 入口，并按 `KimiRankWeightNames` 填充 typed GPU tensor view。
- 已验证：

```bash
cargo fmt --check
cargo test -p pegainfer-kimi-k2 weights -- --nocapture
cargo test -p pegainfer-kimi-k2 -- --nocapture
cargo check --release -p pegainfer-kimi-k2
cargo check --release -p pegainfer-server
```

### Step 8: safetensors rank loader 前置实现

- 新增 `KimiTensorHeader`、`KimiRankWeightHeaders`、`KimiRankGpuContext`、`KimiRankGpuWeights`。
- 新增 `load_rank_weight_headers(model_path, shard_plan)`：
  - 按 `KimiRankShardPlan` 逐 shard mmap。
  - 用 safetensors header 校验计划内 tensor 均存在。
  - 记录 dtype、shape、bytes、shard。
  - 不读取非 text tensor。
- 新增 `load_rank_weights_to_gpu(ctx, model_path, shard_plan)`：
  - 绑定 rank CUDA context。
  - 按 shard plan 逐 tensor copy 到 GPU `CudaSlice<u8>`；这个入口保留给全量/header probe 或后续特殊用途，不作为 TP8 生产加载路径。
  - 输出 rank-local raw GPU tensor map；后续再包成 attention/dense/MoE typed view。
- 新增 `load_rank_sliced_weight_headers(model_path, load_plan)` 与 `load_rank_sliced_weights_to_gpu(ctx, model_path, load_plan)`：
  - `embed_tokens` / `lm_head` 按 `vocab_range` 做行切。
  - `q_b_proj` 按本地 heads 的 `q_head_dim=192` 做行切。
  - `kv_b_proj` 按本地 heads 的 `qk_nope+v=256` 做行切。
  - `o_proj` 按本地 heads 的 `v_head_dim=128` 做列切。
  - layer0 dense `gate/up` 行切、`down` 列切。
  - shared expert `gate/up` 行切、`down` 列切。
  - router gate/bias、norm、q_a/kv_a、q/kv LoRA norm 复制。
  - routed expert INT4 tensor 只枚举本 rank 48 个 local experts，tensor 内部全量读取。
- 单测使用小 safetensors fixture 覆盖：
  - 多 shard header load。
  - 缺失 tensor 报错。
  - sliced header 的本地 shape/bytes。
  - col slice 的 row-major repack。
- 已验证：

```bash
cargo fmt --check
cargo test -p pegainfer-kimi-k2 weights -- --nocapture
cargo test -p pegainfer-kimi-k2 -- --nocapture
cargo check --release -p pegainfer-kimi-k2
cargo check --release -p pegainfer-server
```

### Step 11: TP8/EP8 slice-aware load plan

- 新增 `KimiTensorLoadSlice`、`KimiTensorLoadSpec`、`KimiShardTensorLoadPlan`、`KimiRankSlicedLoadPlan`。
- `KimiRankSlicedLoadPlan` 已接入 `KimiK2DirectRuntimeConfig` 并移交给每个 `KimiRankWorker`。
- 本地 K2.6 index 上 rank3 关键切片：
  - vocab rows `61440..81920`。
  - `q_b_proj` rows `4608..6144`。
  - `kv_b_proj` rows `6144..8192`。
  - `o_proj` cols `3072..4096`。
  - dense layer0 gate/up rows `6912..9216`，down cols `6912..9216`。
  - shared expert down cols `768..1024`。
  - routed expert 只包含 global expert `144..192`。
- 这个修正避免 H20 真权重加载时每个 TP rank 复制全量 BF16 tensor。
- 已验证：

```bash
cargo fmt
cargo test -p pegainfer-kimi-k2 weights -- --nocapture
cargo check --release -p pegainfer-kimi-k2
```

### Step 10: rank-local typed GPU weight view

- 新增 `KimiRankTypedGpuWeights`，把 `KimiRankGpuWeights` 的 raw tensor map 收敛成：
  - top-level embedding / final norm / lm_head。
  - 每层 attention weights。
  - layer0 dense MLP weights。
  - MoE router、shared experts、本 rank 48 个 routed experts 的 INT4 三元组。
- `KimiRankWeightNames::required_tensor_names()` 现在能给出 rank-local typed view 需要的完整 tensor set，并校验无重复、数量等于 `KimiRankWeightPlan::tensor_count`。
- `KimiRankWeightHeaders::validate_typed_names()` 和 `KimiRankGpuWeights::typed_view()` 共享 rank、tensor count、dtype contract：
  - BF16：attention、dense/shared MLP、router gate、embedding、final norm、lm_head。
  - F32：router `e_score_correction_bias`。
  - INT4 routed expert：safetensors 落盘 `weight_packed=I32 [out, in/8]`，GPU raw tensor 保留底层 bytes，CUTLASS-facing manifest 继续按 `u8 [out, in/2]` 解释；`weight_scale=BF16`，`weight_shape=I32`。
- 这个阶段只建立 typed ownership 和 dtype barrier；实际 runtime decode 仍需后续接 typed view、KV cache、PPLX backend 和 expert generated kernels。
- 已验证：

```bash
cargo fmt
cargo test -p pegainfer-kimi-k2 weights -- --nocapture
cargo check --release -p pegainfer-kimi-k2
```

## 在 H20 上验证

每次上下文压缩后继续 Kimi H20 工作前，先读本章节，再读本文件 TL;DR 和 `operator-todo.md`。这里记录的是远端验证约束，不是一次性聊天信息。

### 模型路径

- H20 权重路径：`/data/models/Kimi-K2.5`
- 已知节点提示符记录：`root@host-10-96-191-100:/data/models/Kimi-K2.5#`
- `Kimi-K2.5` 与 `Kimi-K2.6` 架构相同，K2.6 是继续训练后的版本；算子、shape、TP8/EP8 规划可共用。
- 本地非权重参考路径仍是 `/data/models/Kimi-K2.6`。
- 不下载模型权重；远端优先检查 `/data/models`。

### H20 vLLM reference 环境

- K2.5 reference 首选完整 native venv：`/root/develop/xingming/vllm_test/.venv/bin/python`。
- 使用该 venv 时把 `/root/develop/xingming/vllm_test/.venv/bin` 放在 `PATH` 前面；FlashInfer JIT 需要同一环境里的 `ninja`。
- 不拼 `PYTHONPATH` overlay，不补 `_C_stable_libtorch` 符号链接，不混用另一个 vLLM checkout 的 Python 包和 SO。
- 已探测：
  - `/root/develop/xingming/vllm_test/.venv`：`vllm 0.19.0`、`torch 2.10.0+cu128`、registry 含 `KimiK25ForConditionalGeneration`，可直接跑 `/data/models/Kimi-K2.5`。
  - `/root/develop/yingshan/vllm/.venv`：有 vLLM dev 源码环境，但当前 registry 不含 `KimiK25ForConditionalGeneration`，不是 K2.5 fixture 首选。
- Python API reference 命令：

```bash
ssh h20-100 'cd /root/develop/xingming/pegainfer-kimi-k2-main
export PATH=/root/develop/xingming/vllm_test/.venv/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
/root/develop/xingming/vllm_test/.venv/bin/python \
  pegainfer-kernels/tools/kimi_k2/vllm_logits_reference.py \
  --model-path /data/models/Kimi-K2.5 \
  --out-dir /data/fixtures/kimi-k2/k25_parity_vllm \
  --prompt-set-json pegainfer-kernels/tools/kimi_k2/kimi_k25_parity_prompts.json \
  --top-k 20 \
  --tp-size 8 \
  --thinking true \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9'
```

- vLLM `0.19.0` 对 sample `logprobs` 上限是 `20`；当前 gate 用 top-20 serving fixture 判断真实 forward 是否进入同一候选集合。top128/full-vocab raw logits reference 后置，不能再优先于 MLA + 全层执行。
- 2026-05-21 运行通过：
  - 日志：`/tmp/kimi_k25_vllm_fixture_top20_20260521_030017.log`
  - 输出：`/data/fixtures/kimi-k2/k25_hello_vllm/{metadata.json,prompt.json,reference.safetensors}`
  - prompt：`"Hello"`，thinking=true，seq_len `27`
  - generated token id：`1008`
  - `top_k_returned=20`
  - 日志关键信号：`Resolved architecture: KimiK25ForConditionalGeneration`、`FLASH_ATTN_MLA`、`CompressedTensorsWNA16MarlinMoEMethod`、`Marlin backend for WNA16 MoE (group_size=32, num_bits=4)`、权重加载 `274.26s`、GPU KV cache `663,600 tokens`、greedy generate 成功。
- 2026-05-21 多 prompt vLLM fixture 运行通过：
  - 日志：`/tmp/kimi_k25_vllm_parity_20260521_085822.log`
  - 输出：`/data/fixtures/kimi-k2/k25_parity_vllm/{cases.json,hello,math_short,self_intro_zh,code_rust}`
  - cases：`hello` seq_len `27` generated `1008`；`math_short` seq_len `40` generated `1008`；`self_intro_zh` seq_len `34` generated `4052`；`code_rust` seq_len `37` generated `1008`。
  - 同一次 vLLM `LLM` load 处理 4 个 rendered prompt，权重加载 `266.85s`，GPU KV cache `770,416 tokens`。
- Python API 再遇到 vLLM 内部 API 兼容问题时，备用路线是同一 venv 的 `vllm serve --model /data/models/Kimi-K2.5`，通过 OpenAI-compatible HTTP `/v1/completions` 请求 `logprobs=True` 拿 serving top-logprobs；仍不走 overlay。
- 2026-05-21 HTTP fixture 运行通过：
  - 启动环境：`/root/develop/xingming/vllm_test/.venv/bin/vllm serve /data/models/Kimi-K2.5`，TP8，`--served-model-name kimi-k2.5`，`--trust-remote-code`，`--max-model-len 4096`，`--gpu-memory-utilization 0.85`，`--enforce-eager`。
  - 日志：`/tmp/kimi-k2-vllm-http/server_20260521_031017.log`
  - 输出：`/data/fixtures/kimi-k2/k25_hello_vllm_http/{request.json,response.json,metadata.json,prompt.json}`
  - 请求：OpenAI-compatible `/v1/completions`，`temperature=0`，`max_tokens=1`，`logprobs=20`，`return_token_ids=true`
  - 响应：generated text `" The"`，token id `1008`，top token `"The"` logprob `-0.0011391110019758344`，prompt token ids 与 Python API fixture 的 27 个 token 一致。
  - server 生成 fixture 后已停止，H20-100 GPU 已释放。

### 连接与占用检查

- H20 节点开发走 SSH，不通过 Kubernetes 启动开发任务。
- 内网机器需要公司桥接时，以 `mac-office` 为入口；H20 节点按现有规则可直接 SSH 时直接连。
- GPU 工作开始前先检查目标节点空闲情况：

```bash
ssh h20-100 'nvidia-smi'
ssh h20-100 'kubectl get pods -A -o wide --field-selector=status.phase!=Succeeded,status.phase!=Failed | awk '\''NR==1 || ($1 !~ /^(kube-system|monitor|istio-system|calico-system|tigera-operator)$/ && $8 == "host-10-96-191-100")'\'''
```

- 如 `host-10-96-191-100` 上有活跃 GPU 进程或非系统 pod，占用未释放前换节点验证。
- 下载依赖时只在远端 shell/session 临时设置代理；模型权重不走下载。

```bash
export https_proxy="http://127.0.0.1:1083"
export http_proxy="http://127.0.0.1:1083"
export no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
```

### 初始化远端 checkout

远端初始化采用“先从 `origin/main` 得到干净 checkout，再 rsync 当前 patch”的方式，避免把本地工作区整体覆盖远端状态。

1. 在远端工作目录下 clone 仓库，例如 `~/develop/xingming/pegainfer-kimi-k2-main`；如从 `main` 建基线，先 `git fetch origin main && git pull --ff-only origin main`。
2. 初始化 submodule，尤其是 FlashInfer/CUTLASS，不用 rsync third_party 头文件：

```bash
git submodule update --init --recursive pegainfer-kernels/third_party/flashinfer
```

3. 在远端 checkout 内确认分支、状态和路径：

```bash
git status --short
git branch --show-current
pwd
```

4. 从本地同步改动前必须先 dry run，检查文件列表和目标路径；只同步 Kimi 相关 patch，不同步模型、target、third_party 或旧脏目录：

```bash
rsync -avcn --relative \
  Cargo.lock Cargo.toml docs/index.md docs/models/kimi-k2 \
  pegainfer-kimi-k2 \
  pegainfer-kernels/KERNELS.md pegainfer-kernels/build.rs pegainfer-kernels/src/ffi.rs pegainfer-kernels/src/ops.rs \
  pegainfer-kernels/src/ops/kimi_experts.rs pegainfer-kernels/src/ops/kimi_mla.rs pegainfer-kernels/src/ops/kimi_router.rs \
  pegainfer-kernels/csrc/kimi_k2 pegainfer-kernels/tools/kimi_k2 \
  pegainfer-server/Cargo.toml pegainfer-server/src/bin/bench_serving.rs pegainfer-server/src/main.rs pegainfer-server/src/server_engine.rs \
  h20-100:~/develop/xingming/pegainfer-kimi-k2-main/
```

5. dry run 文件列表确认正确后，再执行同形态 rsync 去掉 `-n`。

6. 远端编译与验证在远端 checkout 内执行；H20 root 环境的 uv 在 `/root/.local/bin/uv`，需要把 `/root/.local/bin` 放进当前 session 的 PATH。Python/Triton 用 repo-local `.venv-kimi`，不改全局环境。

首批 H20 验证目标：

- `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kernels`
- `PEGAINFER_CUDA_SM=90a cargo test --release -p pegainfer-kimi-k2 -- --nocapture`
- 后续需要 server 入口时再跑 `PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-server`
- 后续接算子 body 后，再用 `/data/models/Kimi-K2.5` 做 text-only operator fixture 和 TP8/EP8 decode 验证。

### K2.5 rank0 权重加载 gate

- 测试入口：`h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view`
- 模型路径硬编码为 `/data/models/Kimi-K2.5`，不新增环境变量契约。
- 默认 `ignore`，只在 H20 权重节点显式运行。
- 覆盖范围只包含 rank0：从 `model.safetensors.index.json` 生成 rank0 sliced load plan，调用 `load_rank_sliced_weights_to_gpu` 读取真实 payload 并复制到 GPU，然后用 `KimiRankGpuWeights::typed_view` 校验 typed GPU view。
- 不加载全 8 rank，不做 CPU reference，不比较数值；这个 gate 只验证真实 K2.5 payload 与 rank0 slicing / dtype / typed ownership 契约一致。
- 2026-05-20 H20 运行抓到 `weight_packed` 和 `weight_shape` 都是 safetensors `I32`，不是 catalog 里原先假设的 `U8/U32`；K2.5/K2.6 config 对比确认 text 架构一致，此处是 loader dtype contract 问题，不是模型结构差异。

运行命令：

```bash
PEGAINFER_CUDA_SM=90a cargo test --release -p pegainfer-kimi-k2 \
  h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view -- --ignored --nocapture
```

### MLA decode operator gate

- 测试入口：`h20_kimi_flashinfer_batch_decode_mla_bs4_smoke`
- 范围：`pegainfer-kernels` 层的 FlashInfer MLA decode wrapper、paged compressed KV append、decode split+RoPE、`q_nope` absorption 和 `v_up`。该 gate 构造 bs4 synthetic ckv/kpe page table，先从 `q_proj [4,8,192]` 和当前 `k_rope [4,64]` 生成 `q_nope/q_pe/append_kpe`，再用 `kv_b_proj` 前半段做 `q_nope [4,8,128] -> q_abs_nope [4,8,512]`，append `[4,512]` 与 `append_kpe [4,64]`，跑 `BatchDecodeWithPagedKVCacheDispatchedMLA`，最后用 `kv_b_proj` 后半段做 latent `[4,8,512] -> attn_out [4,8,128]`。
- 约束：不比较数值，不代表 vLLM parity；它只证明 C ABI、Rust wrapper、page table、stride layout、bs4 launch 在 H20 上可执行。真实 parity 仍等 worker 侧接入 `W_UK_T` absorption、`W_UV` v-up、全层 decode cache 后，用 vLLM top-k fixture 做。
- cache ABI：Rust `KimiMlaPagedKvLayout` 显式记录 ckv/kpe `page_stride` 与 `token_stride`，默认入口是 separate contiguous buffer；后续也可表达 vLLM 风格 concat `[512+64]` cache 的 strided view。
- 2026-05-21 H20 结果：基础 `paged append + MLA decode` 版本通过；扩展到 `q_abs + paged append + MLA decode + v_up` 后通过；加入 decode split+RoPE 后，同名 gate 再次通过，1 个 ignored gate 运行成功，耗时 `2.87s`。

运行命令：

```bash
PEGAINFER_CUDA_SM=90a cargo test --release -p pegainfer-kernels \
  h20_kimi_flashinfer_batch_decode_mla_bs4_smoke -- --ignored --nocapture
```

### 2026-05-20 H20 sm90a 验证记录

- 节点：`h20-100` / `host-10-96-191-100`
- 基线：`~/develop/xingming/pegainfer-kimi-k2-main`，`origin/main@0cb1355`
- submodule：`flashinfer@779c24d1`，`cutlass@da5e086d`，`spdlog@c3aed4b6`
- 环境：`PEGAINFER_CUDA_SM=90a`，`PEGAINFER_TRITON_PYTHON=$PWD/.venv-kimi/bin/python`，`.venv-kimi` 由 `/root/.local/bin/uv` 创建并安装 `triton==3.7.0`
- GPU 占用：8 张 H20-3e 基本空闲；节点有系统/存储/网关类 pod，无活跃 GPU 进程。
- 命令：

```bash
cargo check --release -p pegainfer-kernels
cargo test --release -p pegainfer-kimi-k2 -- --nocapture
```

- 结果：
  - `pegainfer-kernels` sm90a release check 通过，耗时 `41.98s`；`kimi_cutlass_int4_sm90a.cu` 随 CUDA TUs 一起以 `compute_90a/sm_90a` 编译。
  - `pegainfer-kimi-k2` release tests `19 passed; 0 failed`，耗时 `0.14s`。
  - `h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view` 通过，耗时 `20.75s`；真实 `/data/models/Kimi-K2.5` rank0 sliced payload 已加载到 GPU，并通过 typed GPU view 校验。
  - 这证明 CUTLASS C++ AOT scaffold 在 H20/sm90a 能编译链接，且 K2.5/K2.6 同架构权重的 rank0 slicing / dtype / typed ownership contract 已由真实 payload 验过；真实 grouped GEMM 数值、graph-resident scratch 和多 rank runtime 仍是下一步。

### 2026-05-20/21 H20 rank0 expert-major package gate

- 历史说明：本小节记录 CUTLASS example69 probe 阶段的失败/纠偏路径。当前 runtime 主线已经切到 Marlin WNA16，旧 `KimiRankExpertKernelWeights` / CUTLASS raw/kernel package helper 已从代码删除；保留这段是为了说明为什么不再回到 CUTLASS scale 语义上继续修。
- 变更：`KimiRankTypedGpuWeights::expert_major_weight_plan()` 现在会把 typed view 里的 60 个 MoE layer 收敛成 CUTLASS loader 需要的 expert-major package plan；`pack_expert_major_layer_kernel_weights()` 会把指定 MoE layer 的 48 个本地 expert 三元组做成 `KimiInt4ExpertWeights` 可借用的常驻 typed package。
- 2026-05-21 追加：`pack_rank_expert_kernel_weights()` 现在会把 rank0 全 60 个 MoE layer 全部转换成 `KimiRankExpertKernelWeights`，并在转换完成后从 raw tensor map 删除 routed expert raw tensors；rank worker `LoadSlicedWeights` 会持有这个常驻 package。
- 2026-05-21 结构修复：full-rank package 先完成全部 60 层转换，再统一删除 raw routed tensors；rank worker 用单个 `KimiRankLoadedWeights { gpu, expert_kernels }` loaded state 保存权重，避免 `gpu_weights` / `expert_kernel_weights` 两个 Option 漂移。
- 2026-05-21 结构 guard：这两条必须在后续 H20 gate、reset/reload 和多 rank worker 接线里继续保持。若 package 中途失败，要么全量 pack 成功后才 cleanup，要么显式把 worker 标为不可恢复；不得留下部分 MoE 层 raw 已删、部分仍在的半残权重状态。
- 覆盖：
  - 本 rank local expert 顺序必须等于 `local_expert_range`。
  - gate/up/down per-expert `weight_packed` 必须是 safetensors `I32 [out, in/8]`。
  - kernel-facing packed bytes 规划为 `u8 [48, out, in/2]`。
  - `weight_scale` 必须是 `BF16 [out, in/32]`。
  - `weight_shape` 必须是 `I32 [2]`。
  - layer 1 单层 gate/up/down 真实 payload 会在 GPU 上 D2D 打包为 expert-major contiguous raw buffer。
  - packed 权重会在 load/package 阶段调用 CUTLASS `reorder_tensor`，并把 compressed-tensors offset-binary nibble 转成 signed int4b_t 表示；decode 阶段只借用 reordered package。
  - scale/shape 会成为 typed `CudaSlice<bf16>` / `CudaSlice<i32>`，不在 runtime 里传播 raw byte cast。
  - full-rank package 后 `original_total_bytes == remaining_non_routed_raw_bytes + rank_kernel_package_bytes`，并且 raw tensor map 不再保留 `.mlp.experts.` routed expert tensors。
- H20 命令：

```bash
PEGAINFER_CUDA_SM=90a cargo test --release -p pegainfer-kimi-k2 \
  h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view -- --ignored --nocapture
```

- 结果：
  - 2026-05-20：通过，耗时 `17.26s`。真实 `/data/models/Kimi-K2.5` rank0 payload 已覆盖 sliced load、typed GPU view、expert-major package plan、layer1 raw buffer D2D package、CUTLASS sm90a reorder 和 typed `KimiInt4ExpertWeights` package。
  - 2026-05-21：通过，耗时 `20.62s`。同一 gate 已覆盖全 60 MoE layer `KimiRankExpertKernelWeights` package，并验证 routed expert raw tensors 被释放出 raw map。
  - 2026-05-21：结构修复后通过，耗时 `20.33s`。覆盖先全量 package 后统一删除 raw routed tensors，以及 worker single loaded state。
  - 2026-05-21：CUTLASS launch gate 通过，耗时 `20.71s`。在真实 rank0 package 上构造 bs4 / active_tokens=6 / routed_tokens=48 的 device `expert_indptr`，分别执行 W1 gate、W3 up、W2 down 的通用 `kimi_cutlass_int4_grouped_prepare` + `kimi_cutlass_int4_grouped_launch`；零输入输出保持全零，证明 reordered package 能进入 H20 sm90a grouped GEMM launcher。测试中的 D2H 只用于断言输出，不属于 runtime hot path。
  - 2026-05-21：route bridge gate 通过，耗时 `20.53s`。同一 H20 ignored gate 不再手写 `expert_indptr`，而是用 `topk_idx[48]` 经过 Kimi 专属 CUDA route kernel 生成 `expert_indptr`、`pos_to_token`、`token_topk_to_pos`，随后执行 expert-major hidden expand、W1/W3/W2 CUTLASS launch 和 f32 reduce；零输入输出保持全零。`topk_weight` / maps / indptr 全程 device-resident，D2H 只在测试断言中出现。
  - 2026-05-21：真实 router + SwiGLU gate 通过，耗时 `23.64s`。同一 gate 从真实 K2.5 layer1 router gate/bias raw tensors D2D 打成 typed `DeviceMatrix/CudaSlice<f32>`，执行 `kimi_router_noaux_tc_launch`，再把真实 `topk_idx/topk_weight` 接入 expert-major route/expand/reduce；synthetic local route 分支继续覆盖每个本地 expert 的 W1/W3 CUTLASS launch、`kimi_swiglu_silu_mul` 和 W2 CUTLASS launch。D2H 仍只用于测试断言。
  - 2026-05-21 纠偏：本仓库自写 GPU dequant+cuBLAS 对照 CUTLASS 只能证明内部一致性，不能证明 parity；该 self-comparison 已从 gate 里移除。后续 routed expert 数值只能对 Torch/vLLM 外部 fixture。
  - 2026-05-21 signed/unsigned 复核：compressed-tensors 官方 pack 是 signed int4 输入、offset-binary nibble 落盘；官方 unpack 返回 signed 值。CPU ref 的 `unsigned - 8` 正确；CUTLASS package 的 `xor 0x88` 正确。manual vs official 恒差 `8` 不是 scale layout 问题。Marlin WNA16 使用 `uint4b8` bias=8 语义，weight repack 保留 unsigned nibble，不能复用 CUTLASS 的 signed-xor package。
  - 2026-05-21 Marlin scale layout 复核：vLLM `CompressedTensorsWNA16` Marlin backend 先形成 `[expert,in_group,out]`，再对 flat group-major buffer 做 `marlin_moe_permute_scales` 的 64-block `scale_perm`。本仓库 metadata 已改成 `expert_major_group_scale_marlin_group_major_perm64`；H20 `h20_kimi_marlin_scale_reorder_matches_vllm_permute` 通过。
  - 2026-05-21 Marlin W13 ABI 复核：vLLM runtime 不吃独立 gate/up package；第一次 GEMM 要 fused `w13=[gate,up]`，Kimi packed int32 view 是 `[48,448,8192]`，scale 是 `[48,224,4096]`。本仓库 Marlin package 已改成函数返回时只常驻 fused W13 + W2，gate/up 只作为 load-time 临时 buffer。
  - 2026-05-21 Marlin package split gate：H20 `h20_kimi_k25_rank0_marlin_expert_package_loads` 通过，真实 `/data/models/Kimi-K2.5` rank0 60 层 MoE 可打成 Marlin-only package，未再触发双 package OOM；`h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view` 同步通过，CUTLASS probe package 路线未回退。
  - 2026-05-21 scale 复核：focused H20 probe 证明 CUTLASS example69 `TileShapeK=64` 不能表达 Kimi `group_size=32` 的 BF16 per-row/per-K-group scale。col `0/1/31` 结果匹配，col `32/33` 仍使用 group0 scale，col `64` 使用 group1 scale；这说明实际 scale 粒度是 64-wide K tile。把 scale 当 `[out, group]` 或 `[group, out]` 都不对；`TileShapeK=32` 编译触发 CUTLASS static assertion `K_BLOCK_MAX >= 4`。
- 当前结论：CUTLASS example69 probe 已停止作为 Kimi correctness 路线；后续 routed expert 数值只认 vLLM Marlin WNA16 fixture 和 full-forward vLLM top-k gate。

### 2026-05-21 H20 all-rank vocab-shard top1 gate

- 变更：runtime 不再只调用 rank0 forward；`KimiRankWorker` 暴露 async one-token forward command，scheduler/runtime 先向 8 个 rank 全部发命令，再收集每个 rank 本地 vocab shard 的 top1 token/logit，并在 host 端做 8-way greedy merge。
- 历史边界：这不是 full-vocab logits dump，也不是 vLLM parity；当时只修正 vocab-parallel sampling 的结构错误，attention 仍是 stub，只执行 layer0 dense 和 layer1 MoE smoke。该 gate 的结论不能作为当前 MLA/full-forward parity 依据。
- H20 命令：

```bash
PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests
PEGAINFER_CUDA_SM=90a cargo test --release -p pegainfer-kimi-k2 \
  h20_kimi_k25_all_rank_one_token_vocab_shard_top1_smoke -- --ignored --nocapture
```

- 结果：
  - sm90a test-target check 通过。
  - all-rank gate 通过，耗时 `148.32s`，真实加载 8 rank K2.5 权重，验证 `gpu_weight_ready_rank_count=8`、`vocab_shards_considered=8`、`selected_from_global_vocab_shards=true`、top token global id 落在 `0..163840`。
  - 测试结束后 H20-100 GPU compute app 为空。

### 2026-05-21 H20 full-vocab smoke candidate dump

- 变更：rank worker 增加 test-only logits shard dump command。每个 rank 在当前 one-token smoke 后把本地 vocab shard `[20480]` BF16 logits D2H 成 f32；scheduler test 把 8 个 shard 按 `vocab_start` 拼成 `[163840] logits_f32`，写成 safetensors candidate。
- 历史边界：这是旧 candidate artifact，不是 parity gate。该版本 candidate 仍有 attention stub 和未执行的后续 59 层，metadata 必须保留 `parity_claim=false`。MLA/full-forward 版本需要在 H20 重新生成 candidate。
- H20 命令：

```bash
PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests
PEGAINFER_CUDA_SM=90a cargo test --release -p pegainfer-kimi-k2 \
  h20_kimi_k25_dump_all_rank_one_token_candidate_logits -- --ignored --nocapture
```

- 结果：
  - dump gate 通过，耗时 `148.33s`。
  - 输出：`/data/fixtures/kimi-k2/pegainfer_k25_smoke_logits/{candidate.safetensors,metadata.json}`
  - candidate 文件大小 `655448` bytes，`logits_f32` shape `[163840]`。
  - prompt ids 来自 `/data/fixtures/kimi-k2/k25_hello_vllm/prompt.json`，最后一个 token 是 `163606`。
  - smoke candidate argmax token id `154473`，logit `8.9375`；top8 ids `[154473, 159083, 160345, 161694, 149515, 159290, 161170, 149762]`。
  - 测试结束后 H20-100 GPU compute app 为空。
  - H20-100 已扫现有 venv：`/root/develop/xingming/vllm_test/.venv`、`/root/develop/yingshan/vllm/.venv`、`/root/develop/xingming/pegainfer/.venv`、`/root/develop/xingming/pegainfer-pplx-ep/.venv` 等都没有 `accelerate`，不能直接作为 HF `device_map=auto` raw-logits env；不要用 PYTHONPATH overlay 补环境。本阶段不再继续补 HF env，先做 MLA + 全层 forward。

### 2026-05-21 路线纠偏：先 MLA + 全层 forward

- 结论：HF raw-logits fixture 不是当前瓶颈。PegaInfer 端旧 candidate 仍然有 attention stub，且只执行 layer0 dense + layer1 MoE；在这个状态下对 vLLM top-20 或 full-vocab reference 做 diff 没有诊断价值。
- 当前主线：
  - 不继续搭 HF remote-code / `compressed_tensors` / `accelerate` 环境，不再做 PYTHONPATH overlay。
  - 先让 PegaInfer 端真实执行 Kimi MLA attention 和全部 61 层。
  - 用 `/data/fixtures/kimi-k2/k25_parity_vllm` 多 prompt top-20 fixture 做 logits gate。
- 本地实现状态：
  - 新增 `pegainfer-kernels/csrc/kimi_k2/kimi_mla.cu`，负责 `kv_a` split、Kimi YARN RoPE 后 Q/K/V 组装，以及 FlashInfer `SinglePrefillWithKVCacheDispatched<192,128>` prefill。
  - `KimiRankWorker` 的 prompt forward 已改为遍历 61 层：每层先跑 MLA prefill，再执行 layer0 dense 或 layer1..60 MoE；最终输出 last-token vocab shard logits。
  - 8 rank 之间用 NCCL all-reduce 补齐 embedding、attention o_proj、dense/shared expert 和 routed expert combine 的数学正确性；这是 H20 smoke/parity bridge，不是生产 PPLX EP。
  - 当前 prompt/full-logits dump 仍会做 debug D2H 和逐层 allocation；它是 correctness smoke，不是 decode graph-ready hot path。
- 当前 H20 gate：
  - `cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - `h20_kimi_k25_dump_parity_prompt_candidates` 通过，用同一个 runtime load 依次 dump 4 个 prompt 的 full-vocab candidate，输出 `/data/fixtures/kimi-k2/pegainfer_k25_parity_candidates`。
  - `compare_vllm_topk_fixture.py --require-argmax --min-overlap 16` 通过：4/4 argmax match，top-20 overlap 分别为 `19/20`、`20/20`、`20/20`、`19/20`。
  - 这证明当前 full-forward 进入了 vLLM greedy/top-20 的同一候选集合；下一阶段仍要替换 NCCL-sum bridge 为 PPLX EP，并开始 prefill/decode perf gate。

## Operator TODO

### P0: 权重格式与 manifest

- [x] 写 Kimi text-only config parser，只读取 `text_config`，显式拒绝 vision inputs。
- [x] 写 safetensors index scanner，生成 text-only manifest：embedding、61 层 attention、dense layer0 MLP、60 层 shared expert、60 层 routed expert、final norm、lm_head。
- [x] 读取少量 safetensors header 后确认 `weight_packed` dtype、`weight_scale` dtype/shape、`weight_shape` 表示方式、INT4 nibble 顺序。2026-05-21 复核结论：manual vs official 恒差 `8` 是 signed INT4 与 checkpoint offset-binary nibble 的解释差异，CPU ref 必须 `unsigned - 8`，CUTLASS package 只在 CUTLASS reordered buffer 上做一次 `xor 0x88`。
- [x] 定义 TP8/EP8 权重 ownership：
  - attention/dense/shared/lm_head 走 TP8 shard。
  - routed experts 走 EP8，每 rank 48 个 experts。
  - router gate 权重每 rank 需要完整 384 分数，或者明确做 replicated gate。

### P0: Dense BF16 算子

- [ ] 复用或参数化 BF16 RMSNorm：hidden `7168`，q/kv LoRA norm `1536` 和 `512`。
- [ ] 复用 BF16 residual add / fused add RMSNorm，确认 batch layout 与 Kimi hidden layout 一致。
- [ ] 参数化 BF16 linear/cuBLAS wrapper，覆盖这些形状：
  - embedding lookup：`vocab 163840 -> hidden 7168`
  - q_a: `7168 -> 1536`
  - q_b: `1536 -> 12288`
  - kv_a_with_mqa: `7168 -> 576`
  - kv_b: `512 -> 16384`
  - o_proj: `8192 -> 7168`
  - dense layer0 gate/up: `7168 -> 18432`
  - dense layer0 down: `18432 -> 7168`
  - shared expert gate/up: `7168 -> 2048`
  - shared expert down: `2048 -> 7168`
  - lm_head shard: `7168 -> 20480` per TP rank
- [ ] 写 shape-only operator tests，先不依赖完整权重。

### P0: MLA attention

- [ ] 实现 attention prep：
  - `q = q_b(RMSNorm(q_a(x)))`
  - split `q_nope[128] + q_pe[64]`
  - `kv_a_with_mqa(x)` split `compressed_kv[512] + k_pe[64]`
  - `kv = kv_b(RMSNorm(compressed_kv))` split `k_nope[128] + value[128]`
- [ ] 实现 Kimi YARN RoPE cache，partial dim `64`，theta `50000`，factor `64`。
- [ ] 第一版 correctness path 可 materialize expanded K/V：K head dim `192`，V head dim `128`，用于短上下文 operator parity。
- [ ] 生产路径需要 MLA compressed KV cache：存 `compressed_kv[512] + k_pe[64]`，attention kernel 内或近旁重构 `k_nope/value`，避免 256K context 下 expanded KV 爆内存。
- [ ] 评估 FlashInfer 是否能直接支持 `qk_dim=192, v_dim=128`；不支持时写 Kimi 专用 prefill/decode attention wrapper。
- [ ] 做 decode 单 token attention operator test：HF dump 一层输入/输出，对齐 RoPE、softmax scale 和 V slice。

### P0: MoE router

- [ ] 参数化 DSV4 `score_gate` 或新写 Kimi gate kernel：
  - input BF16 hidden `[tokens, 7168]`
  - gate weight `[384, 7168]`
  - logits 以 FP32 计算
  - `scores = sigmoid(logits)`
  - choice scores 加 `e_score_correction_bias[384]`
  - 因 `n_group=1/topk_group=1`，直接 top8 over 384
  - topk weight 从未加 bias 的 `scores` gather
  - topk weight normalize 后乘 `2.827`
- [ ] 输出保持 device resident：`topk_idx[u32/i32]`、`topk_weight[f32/bf16]`。
- [ ] 写 CUDA router fixtures，覆盖 bs>1、choice bias 只影响选择、top8 weight 从原始 scores gather、normalize 后乘 `2.827`。

### P0: Routed expert INT4 grouped GEMM

- [x] 新增 compressed-tensors INT4 loader：`weight_packed/weight_scale/weight_shape`。runtime package 字段已显式区分 CUTLASS example69 scale/weight layout；Marlin scale/weight repacker 已存在，但完整 WNA16 compute backend 尚未接入。
- [ ] 新增 Kimi INT4 grouped W1/W3 kernel：
  - local experts per EP rank = 48
  - input dim `7168`
  - output dim `2048`
  - group size `32`
  - 两个 projection 分别是 gate/up，输出给 SwiGLU。
- [ ] 新增 Kimi INT4 grouped W2 + SwiGLU kernel：
  - input dim `2048`
  - output dim `7168`
  - 接 W1/W3 输出，做 `silu(gate) * up` 后 down projection。
- [ ] 先做 dequant-to-BF16 format probe，用于权重格式验证和 HF parity，不作为生产 MoE 路线。
- [ ] 再做 fused grouped INT4 kernel，复用 DSV4 expert-major compaction、`expert_indptr`、pointer cache 的组织方式。
- [ ] operator test 要比较 single expert、multi expert、空 expert、top8 duplicate-free route。

### P0: MoE dispatch/combine 与 collective

- [ ] Kimi EP 目标路径是 PPLX dispatch/combine；当前 direct runtime 先保留 NCCL-sum bridge，不接 NCCL AG/RS 路线：
  - 复用 DSV4 PPLX bootstrap、rank worker placement、MR 注册、scratch 生命周期。
  - buffer shape 改为 Kimi hidden `7168`、topk `8`、local experts `48`、expert intermediate `2048`。
  - route/indptr/count/combine metadata 全程保持 GPU resident。
  - shared expert 和 dispatch/recv 的 overlap 按 DSV4 PPLX decode 结构设计。
- [ ] 参数化所有 scratch：hidden `7168`、topk `8`、experts `384`、local experts `48`。
- [ ] 保持 route/indptr/count/reduce metadata 全程在 GPU 上。

### P1: Shared expert 与 dense layer0

- [ ] dense layer0 使用 BF16 SwiGLU MLP：gate/up/down shape `7168 -> 18432 -> 7168`。
- [ ] shared expert 使用 BF16 SwiGLU MLP：`7168 -> 2048 -> 7168`，60 个 MoE 层都有。
- [ ] 判断 shared expert 是否需要 TP8 shard 或 replicated；首版按 TP8 shard 更符合显存与 GEMM 规模。

### P1: Final norm / logits / sampling

- [ ] final RMSNorm hidden `7168`。
- [ ] lm_head untied，TP8 每 rank vocab shard `20480`。
- [ ] all-gather logits 到 vocab `163840` 后 greedy top1。
- [ ] 后续再做分布式 topk，先不要卡住 operator bring-up。

### P1: Text-only 推理入口 / tokenizer

- [ ] 支持 `TikTokenTokenizer` 加载 `tiktoken.model` 和 `tokenizer_config.json` 的 special tokens。
- [ ] 实现或复用 Jinja chat template 渲染，首批只支持 text content；遇到 image/video content 直接拒绝。
- [ ] 支持 thinking 默认开启：generation prompt 以 `<think>` 开始。
- [ ] 支持 instant mode：`chat_template_kwargs.thinking=false` 时 prompt 追加 `<think></think>`。
- [ ] 支持 preserve-thinking 的 prompt 保留规则，用于后续 coding agent benchmark。
- [ ] 先不实现 tool call parser，但保留模板里的 tool declaration / tool call tokens，避免 prompt contract 后面重做。

### P1: Build / test harness

- [ ] 新增 `pegainfer-kimi-k2` crate 或先建 `pegainfer-kernels` Kimi operator tests；首批以 operator tests 为主。
- [ ] 新增 `pegainfer-kernels/csrc/kimi_k2/` 和 `tools/tilelang/kimi_k2/`，不要把 Kimi INT4 kernel 混进 DSV4 文件。
- [ ] 在 `pegainfer-kernels/KERNELS.md` 增加 Kimi-K2 text path routing table。
- [ ] HF probe 只 dump 文本核心一层/单 token fixture，不接 vision。

## Debrief

- **Outcome**: 已把 Kimi-K2.6 非权重文件拉到 `/data/models/Kimi-K2.6`，确认没有权重大文件；文档范围改为 text-only operator bring-up，并拆出 P0/P1 TODO；已把官方推理示例、vLLM/SGLang serve 参数、chat template 契约、`pegainfer-kimi-k2` crate 初始化和 H20 验证规范纳入计划。
- **Pitfalls encountered**:
  - Kimi-K2.6 是多模态外壳，但当前项目目标不支持多模态，所以 `vision_tower` / `mm_projector` / media placeholder 不能进入首批 scope。
  - 现有 DSV4 grouped FP4 kernel 的组织方式能复用，权重格式不能复用；Kimi routed experts 是 compressed-tensors INT4。
- **Lessons learned**:
  - Kimi 的文本核心更接近 DeepSeekV3 MLA + MoE，而不是 DSV4 Flash 的完整执行图；算子层面首先要补 MLA 和 INT4 experts。
  - 只看 `config.json` 不够，`model.safetensors.index.json` 明确说明只有 routed experts 被 pack quantized，attention/shared/lm_head 仍走普通 BF16 权重路径。
  - Kimi INT4 grouped expert 主线固定为 CUTLASS C++ AOT：CuTeDSL Hopper mixed-input helper 不成熟，TileLang 不作为 Kimi 主线；基于仓库内 CUTLASS example 69 改造，W1/W3 合成 N=4096 grouped GEMM，SwiGLU 外置，W2 独立 grouped GEMM。
  - H20 验证必须用 `/data/models/Kimi-K2.5`，K2.5 与 K2.6 架构一致；远端 workspace 先从 `origin/main` 拉到最新，再用 rsync dry-run 补齐小范围源码，随后跑 sm90a 编译和真实 fixture。
