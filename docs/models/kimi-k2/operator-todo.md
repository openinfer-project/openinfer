# Kimi-K2.6 文本算子 TODO

> **TL;DR:** Kimi-K2.6 文本算子主链：MLA + Marlin WNA16 routed expert + NCCL RS bridge，H20 固定 bs4 `max_tokens=16` gate 四路 token 全对，CUDA Graph 覆盖整段 decode，synthetic output64 steady TPOT avg `14.39ms` / p99 `14.83ms`。CUTLASS INT4 后端、kimi_custom_all_reduce 后端、decode row-diff 诊断 helper 已下线，详见 [changelog.md](changelog.md)。下一刀是 H20 端到端 throughput hardening 与 PPLX EP dispatch/combine。
>
> **Last touched:** 2026-05
>
> **历史记录:** [changelog.md](changelog.md) — Execution Log / Rejected / Debrief / 经验迁移 全部归档在那里，按原始顺序保留。

## 范围

本清单只覆盖 `/data/models/Kimi-K2.6` 的 `language_model`。多模态相关的 `vision_tower`、`mm_projector`、media placeholder 插入、图像/视频 processor 都不进入首阶段。

目标不是先接完整 server，而是先把 operator surface 和测试夹具准备好。后续 crate、runtime、server 只消费这里列出的算子。

## 当前优先级

1. H20 端到端优先：停止继续新增内部 smoke 作为主线。`pegainfer-server` + OpenAI-compatible `/v1/completions` 已在 H20 跑通 K2.5 `max_tokens=1/2/8`；vLLM fixture 27-token prompt 的 `max_tokens=1` 返回 token id `1008`，`max_tokens=2` 返回 `[1008, 2742]`，4 并发 `max_tokens=8` 四路一致返回 `[1008,2742,2531,414,19180,6082,1379,387]`。
2. Decode(bs4) 生产化：worker-owned MLA KV/cache owner 已落地，prompt prefill 已把每层 compressed KV/KPE 写入 arena，direct crate 的旧 H20 smoke/candidate/perf 测试入口已移除；scheduler 现在按最多 4 个请求组成 wave，第 2 个 token 起调用真实 bs4 decode body。fused qkv_a 改动把 MLA q/kv down projection 从两个 GEMM 改成一个 `fused_qkv_a` GEMM + 一个 split kernel，静态 trace 从 `1947` calls 降到 `1886`，synthetic output64 steady TPOT 从 `16.70ms` 到 `16.43ms`。
3. EP 生产化：decode routed MoE combine 正从 dense all-reduce bridge 迁到 NCCL reduce-scatter bridge。当前实现目标是 `local router/Marlin -> device repeat f32 -> reduce_scatter_f32_hidden`，不做 BF16 all-gather，也不把 local expert compute 按 EP world 放大。这还不是真 PPLX dispatch/combine；PPLX EP 需要后续替换 MoE-side dispatch/combine call sites。
4. Decode batch policy：旧代码把单请求也塞进固定 bs4 scratch，导致 router/Marlin route elems/routed reduce/logits 都按 4 行执行；上一版只拆成 bs1/bs4 两档，2/3 并发仍会落到 bs4。当前改为预分配 `1..=4` 四个 arena，scheduler 用真实 wave size 选择 arena。禁止基于 `bs==1` 做假设优化；所有性能改动必须服务 `bs>1` 和 `decode(bs4)>300 tok/s`。**硬上限契约**：`pegainfer-kimi-k2/src/runner/scheduler.rs::KIMI_RUNNER_MAX_BATCH = 4` 是单个 wave 的硬上限，和 worker 端 `1..=4` arena 预分配、`KimiK2Runtime` 的 decode batch shape 是同一个数。改它不是改一个 const —— 还要同步 arena 数量、所有按 `decode_batch_size` 走的 scratch/router/Marlin shape、以及 CUDA Graph capture 形态（每个 batch shape 一份 graph）。当前不变，等 wave>4 真有需求时一起做。
5. Prefill perf hardening：当前 correctness path 在 128+ synthetic prompt 已过 1k tok/s，但仍有 per-layer allocation、首个 collective stream drain、host-visible final top1；后续要把 scratch/RoPE/cache 预分配，形成稳定 perf gate。**2026-05-22 streaming chat 实测**：K2.5 自我介绍 prompt（短，~30 tok）TTFT `1995.5ms`、TPOT `14.48ms/token`、`30.8 tok/s`，输出语义正确；TTFT 偏高，说明 prefill 路径还没 ready。下一刀先量 short-prompt prefill 拆解（embedding/MLA prefill/MoE prefill/sampling），再决定是 scratch allocator 还是首个 collective stream drain 主导。
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

## H20 active-batch 清理与 bs1 假设禁令

- 2026-05-21 清理固定 bs4 decode scratch：第一轮把 `KimiRankLoadedWeights` 拆成 bs1/bs4 两个 arena，解决单请求按 4 行执行的问题；本轮继续改成 `1..=4` 四个 arena，scheduler 用真实 `reqs.len()` 作为 decode batch size，2/3 并发也不再按 4 行执行。
- 删除 decode dense/MoE 中 `scratch.hidden.seq_len == 4` 的硬断言，改为 `1..=KIMI_DECODE_MAX_BATCH`；bs1/bs2/bs3 不再按 4 行执行 embedding、MLA projection、router、Marlin W13/W2、routed F32 reduce 和 logits。
- H20 验证：
  - 本地 `cargo fmt --all --check`、`PEGAINFER_CUDA_SM=90a cargo check --release -p pegainfer-kimi-k2 --tests` 通过。
  - H20 同步后 `cargo fmt --all --check`、`cargo check --release -p pegainfer-kimi-k2 --tests`、`cargo build --release -p pegainfer-server --bin pegainfer --bin bench_serving` 通过。
  - barrier 版本单请求 fixture prompt：`max_tokens=16` wall `462.832ms`，`max_tokens=32` wall `832.748ms`，`max_tokens=64` wall `1598.763ms`；`16->32` 差分 TPOT `23.12ms`，`32->64` 差分 TPOT `23.94ms`。
  - rejected no-barrier 实验：bs1 跳过 CPU barrier 曾让暖态 `32->64` 差分 TPOT 到 `18.01ms`，但这是 `bs==1` 假设优化，不服务 `decode(bs4)>300 tok/s`，并且掩盖了当时的 rank/stream 状态问题；该路径已回退，当前生产 decode collective 不再使用 CPU barrier。
- 决策：禁止新增 `bs==1` 专用性能分支。后续优化必须按 active batch / bs4 / continuous batching 设计。下一处大头仍是 NCCL bridge 每 token 的 collective cadence：embedding all-reduce、61 个 attention o_proj all-reduce、1 个 dense MLP all-reduce、60 个 shared BF16 all-reduce、60 个 routed F32 all-reduce，合计约 183 次 collective；CUDA Graph capture/replay 的 rank phase 对齐只用于 graph begin/end/launch，不是每个 collective 前的 CPU barrier。

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
  - 位置：`pegainfer-kimi-k2/src/runner/scheduler.rs`、`src/runner/worker.rs`
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
  - 位置：`pegainfer-kimi-k2/src/runner/affinity.rs`
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
