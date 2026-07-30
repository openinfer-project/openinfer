# GLM5.2 decode 算子 bench harness（agent 调优流水线）

> **TL;DR:** 在 `openinfer-glm52/benches/` 建 Blackwell-only 的 decode 逐内核调优台：**Python harness 层**（torch 参考 + 一键 check/bench/compare）调用 build.rs 用生产 flags 编出的 `libglm52_kernel_lab.so`（内核源码仍是 `csrc/glm52/*.cu` 单一事实源），bench 单位画在**生产融合内核边界**上。分两期：一期只覆盖 EP 无关的 per-rank 内核（bucket {1,2,4,8} 轴；attention/indexer 类另带 ctx 长度轴），单卡可测、无需 NCCL；二期做 MoE 专家链 {EP4, EP8, EP16} 参数化（#668 的 ~3ms 空间在这）+ DeepEP 集合通信。目标是人和 agent 一条命令完成"改内核 → 对拍 → 压测 → 对比基线"，顶门是 `glm52_step_bench` 同会话 A/B。
>
> **Last touched:** 2026-07

## Preparation

- **Read**（经 explore agent 通读并核实代码位置）:
  - `docs/models/glm52/tp4-gb300-bringup.md` — 07-11 node-cut campaign 的融合验收协议（byte-compare harness → 节点数 → byte-identity → same-day TPOT A/B）、被拒融合（recv+norm、push+recv 单内核、缩小 grid 的 gemm_b+SiLU）及其数据
  - `docs/models/glm52/ep4-gb300.md` — EP4 weight-only routed-expert 链六内核变四内核、grid-geometry 融合判据（"A fusion only wins when the consumer's grid can host the producer's parallelism"）、Blackwell 专有数值坑清单
  - `docs/models/glm52/whole-step-decode-graph.md` — decode 一步的内核类别与占比（collective wait ~25%、投影 GEMV ~16%、专家 GEMM ~14%、FlashMLA ~7%、glue ~7%）
  - `docs/models/glm52/cross-node-scaling.md` — EP 宽度 {4,8,16,32,64} 各为一个 constexpr shim 实例化；weight-only 链 ABI-generic，新宽度只是 config header
  - `docs/models/glm52/serving-status.md` — EP4/8/16 现状与 #668（EP4 masked expert kernel 只到 byte-roofline 的 29%，约 3ms/step 空间）
  - `docs/models/glm52/oracle-harness.md` — HF oracle probe 常量生成与容差纪律（不适合单算子 bench，但容差/阴性对照思路可复用）
  - `docs/lessons/flashmla-sm100-ue8m0-kv-scales.md` — Blackwell FlashMLA UE8M0 scale 截断：cache writer 必须输出 2 的幂 group scales，H200 会掩盖此 bug
  - `docs/lessons/moe-bench-prompt-diversity.md` — MoE bench 必须 seeded distinct routing，单 prompt 批次只能标注为 microbench
  - `docs/subsystems/kernels/kernel-op-reports.md` — qwen3 kernel-report 模式（manifest、cold-L2、feature-gated bin）与 `openinfer-bench` 共享层
  - [SGLang RFC #29630](https://github.com/sgl-project/sglang/issues/29630)（已完成迁移）— 统一 `sglang.kernels.ops.<group>` 命名空间 + KernelSpec/CapabilityRequirement 元数据 + tests-first 阶段序；其动机明写 agent-oriented kernel work 需要 stable namespace + consistent tests。可借鉴：bench 单元注册表 = 我们的 "namespace"、契约做成机器可读 manifest、测试先行、bench 不进单元 CI。不借鉴：JIT build infra、280 个散落 kernel 的迁移工程（我们的内核本就在 `csrc/glm52/` 集中）、selector/autotune 机制
  - `docs/subsystems/kernels/tvm-ffi-mvp.md` — 仓库已有 Python↔CUBIN 互操作先例：`tvm-ffi-triton-cubin` 可选桥（packed ABI、raw pointer/stream 传参），证明"Rust 编译产物 + Python 侧加载调用"路线在本仓库已被接受为 optional/test-only
  - `/data/code/workspace-rustllm/TileFoundry`（经 explore agent 通读）— 实为 TVM 风格 Python DSL 编译器（DSL/IR/调度求解不抄），但其**验证/压测维护模式**直接可抄：`check()`/`bench()` 双原语 + Gate（rel_l2 + cosine 取最差 leaf）、容差推导而非手选（bf16 binade-ulp、fp8 block-absmax 量子、"kernel 距 f32 ≤ HF-bf16 距 f32"相对界）、测量常数入档（注明机器/实测值）、输入工厂与 oracle 成对返回 + 分用途命名 seed、`extern "C"` launch shim（grid/block/smem/stream 全走 C ABI）+ device/host TU 分离、JIT 缓存 key=sha256(源文本+target+options)、GPU 测试 fail-not-skip、机器可读策略/任务卡注入。其 bench 层几乎为空（无 baseline ledger）——性能账本用我们 `bench_snapshots/` 的习惯自建
- **Relevant history**:
  - `tp4-gb300-bringup.md` — 每个融合候选都先在 standalone randomized byte-compare harness 验证再进图：**per-kernel bench 与融合本就是同一工作流**
  - `ep4-gb300.md` — 历史事故：`bench_concurrency` 抄 EP8 常数导致 EP4 桶标签全错一位；locked-clock 512-iteration median standalone harness 先例
  - `glm52_step_bench`（`openinfer-server/src/bin/glm52_step_bench.rs`）— 内核迭代的标准 e2e A/B 载体，内建 distinct prompts
- **代码事实**（explore agent 核实）:
  - `openinfer-glm52/benches/` 不存在；crate `autobenches = false`，无 criterion dev-dep、无 `[[bench]]` 段
  - bench 模板：criterion 0.8（workspace 已有），`openinfer-qwen35/benches/qwen35_ops.rs`（`DeviceContext::new` + `iter_sync`）；仓库内**无 cudaEvent 计时先例**，需自加
  - weights-free 自验证骨架：`openinfer-kernels/tests/glm52_moe_ep_wo_smoke.rs`（EP4/EP8/EP64 shape 已参数化、host f32 参考、Lcg RNG）、`glm52_sparse_mla.rs`（f64 naive 参考）、`e4m3_to_f32` helper
  - decode 路径：`run_step_body`（`openinfer-glm52/src/model/step_body.rs:47`）；EP 链选择 `Glm52MoeEpState`（`moe_ep_wo.rs:300`）；拓扑枚举 `Glm52MoeTopo`（`lib.rs:202`，EP8→32 experts/rank、EP4→64、EP16→16）
  - Blackwell 上 EP 任意宽度只有 `glm52_moe_ep_wo` 链可 bench（DeepGEMM masked 是 sm_90a-only）；注意力后端 = FlashMLA sparse（sm_100f）
  - DeepEP dispatch/combine 是跨 rank 集合通信且 wait-bound，**单卡单算子 bench 不可测**

- **Plan**:
  1. **build.rs 出双产物（flags 单一事实源）**：`OPENINFER_KERNEL_LAB=1` 时除静态库外额外产出 `libglm52_kernel_lab.so`——同一批 `csrc/glm52/*.cu`、同一 nvcc flags 与 arch 检测（sm_100f、`OPENINFER_CUDA_SM` 等），每个注册单元配一个 `extern "C"` launch shim（grid/block/smem/stream 全走 C ABI 参数，TileFoundry 同款设计，对 FFI 友好）。Python 绝不自己编内核，保证调优对象 = 生产 SASS
  2. **Python 包 `openinfer-glm52/benches/kernel_lab/`**：pyproject + editable 装进 repo `.venv`（torch 复用 oracle 已有的 pinned 环境）；ctypes 加载 .so、torch tensor `data_ptr()` 直传；子命令：`list`（注册表）/ `check <unit> --ep N --bucket B`（对拍）/ `bench`（torch.cuda.Event 计时、锁频检查、capacity-shaped grid、512-iter median）/ `compare`（同会话交替测量出 delta）。一轮迭代 = 改 .cu → `kernel_lab build` → `check` → `bench` → `compare`
  3. **manifest（TOML/单元）**：稳定命名（如 `mla_front.q_b_gemv.rows64`、`flashmla_sparse.decode.rows8.ctx65536`）+ shape 轴声明（一期 rows 轴 = {1,2,4,8,16,32,64}——1–8 是 decode bucket，16–64 覆盖 MTP span-mapped verify 行数（bucket×span）与未来更大 per-rank batch；attention/indexer 类加 ctx 轴，**默认扫 {16384, 65536, 262144} 三档，短 ctx 不覆盖**；EP 宽度字段预留二期）+ I/O 契约（layout/stride/累加序/grid 约束）+ capability（Blackwell-only fail-closed）+ 参考模式声明 + 容差常数入档（注明机器/实测值）；CPU-only pytest 校验注册表（shape 轴推导、命名唯一；二期加 EP 宽度推导，防 EP4 桶标签错位事故重演）
  4. **对拍两层**：(a) torch 参考 + 推导容差（bf16 binade-ulp、fp8 block-absmax 量子、相对界"kernel 距 f32 ≤ HF-bf16 距 f32"）覆盖语义级；(b) 融合单元必须同时过**生产未融合链 byte-compare**（参考链同样从 .so 加载——torch 无法复现 f32 结合序）。输入工厂与参考成对返回、seed 分用途命名、fp8 用 normal-quantized 填充
  5. **一期单元**（EP 无关，单卡可测；按 decode 占比排序）：fp8 weight-only GEMV 投影（q_a\|kv_a twin、q_b、o_proj；bs=1/2 CUDA-core GEMV、bs=4/8 tensor-core mma 两档）→ FlashMLA sparse decode + query assemble/cache pack（bucket × ctx{16k,64k,256k}）→ fused add+RMSNorm round → quant（per-token-group fp8、UE8M0）→ indexer 链（21 个 full 层；bucket × ctx{16k,64k,256k}）→ router（bs 轴，每 rank 全量 256 expert 打分，EP 无关）→ bookends（lm_head / argmax / embed）→ shared-expert dense SwiGLU（bs 轴）。**二期**（EP 参数化）：MoE weight-only 专家链四内核（tiles / W13 mma / SiLU / W2 mma，EP4/8/16 = 64/32/16 experts/rank，#668 优先）+ DeepEP dispatch/combine（wait-bound，需多卡协同测量，单独设计）
  6. **基线账本 + 融合任务卡**：每单元 JSON 基线（shape/grid/时钟/median/p50/p99，arch 分桶，H200 结果不迁移 Blackwell）；agent 产出 = 候选 + check 绿 + compare delta；融合任务卡 = 相邻单元 + 组合参考，验收同协议，grid-geometry 判据显式记录
  7. **构建验证**：本机 `OPENINFER_KERNEL_LAB=1 cargo build --release -p openinfer-kernels --features glm52` 出 .so + pytest 注册表校验过；有 Blackwell GPU 则跑一个单元 check+bench 冒烟

- **Risks / open questions**:
  - 本机是否有 Blackwell GPU 可跑冒烟未知；没有则只验证编译与 CPU 测试
  - glm52 feature 构建依赖 CUDA + DeepGEMM/FlashMLA submodule，编译时间长；build.rs 双产物必须保持默认构建（不设 env 时）零变化
  - torch 无法复现生产链 f32 结合序——融合单元的 bit-exact 只能靠生产未融合链 byte-compare，torch 容差对拍只是语义级网
  - `.venv` 的 torch 与 oracle pinned `transformers==5.12.1` 已共存（oracle 在用）；kernel_lab 运行时依赖保持只有 torch + stdlib（ctypes）
  - collective 类内核（dispatch/combine/LL AR，~25% 时长）不可单卡 bench，显式排除，避免 agent 拿到假任务
  - H200 调优结果不迁移 Blackwell（`arch_is_blackwell()` 分桶），harness 基线必须按 arch 隔离
  - attention/indexer 单元的真实耗时依赖 ctx 长度分布；ctx 轴已定 {16k, 64k, 256k}，短 ctx 形状不测不调（生产负载为长上下文）

## Execution Log

- 2026-07-29 二期第一块落地：MoE 专家链单元组 `moe_ep_wo.{tiles,w13_mma,silu,w2_mma}`:
  - **shape 轴**：`rows`（per-rank bucket {1,2,4,8}，CLI 原生可扫）× `ep`（{4,8,16} →
    n_local 64/32/16，新轴，组 driver `python3 -m kernel_lab.refs.moe_ep_wo check|bench`
    仿 attention 组 --ctx 模式）；global_tokens = ep×rows ∈ {4..128}（并集入 [axes]）。
    **capacity-proportional 内建**：grid 恒为生产启动预算 state.max_tiles（96/96/144，
    moe_ep_wo.rs:137），缓冲按 bound_rows（同 file:197 公式，CPU 测试钉死推导表）；
    mma/silu 的 tiles 列表由生产 tiles 内核惰性首跑（plan-time，warmup 吸收）。
  - **路由分布**：seeded 可复现非退化——lognormal 热度 σ=1.2 + Gumbel top-8 无放回
    （仅用 random() 算术，跨 Python 版本稳定）；非退化 retry 守卫（近空 draw 确定性
    重抽）修掉了 EP8×gt8 sum=1 的 Binomial 尾部（~1%）。CPU 测试钉死 6 个代表点的
    总数/活跃数/max/tiles/aligned_end。skew 标定依据 = moe-bench-prompt-diversity
    （diverse 生产点近均匀）。
  - **GB300 check 48/48 一次通过**（sm_103 tray03）：tiles rel_l2=0.0（整数全等 +
    tile_count adapter 硬断言）；**silu bit-exact 实测成立 rel_l2=0.0**（探针证实
    torch sigmoid 与内核 1/(1+expf(-gate)) 对 [-30,30] 全部 bf16 输入逐位一致）；
    w13/w2 mma rel_l2 ≤ 1.67e-3 / 1.66e-3（正中 bf16 floor，headroom ~12×，容差
    0.02 未动；adapter 内 smoke 同款 per-element 2e-2 硬门 + gap sentinel 全等硬断言）。
    bench 冒烟：EP8×gt64 W13 103µs、W2 62µs、tiles 7µs、silu 5µs。
  - **参考端坑（新教训）**：bit-exact 门的参考必须先舍到目标存储精度再比——silu 首测
    参考返回未舍入 f32，CLI 拿 bf16 内核输出对 f32 参考，把 bf16 store floor 当成
    "内核误差"测出 rel_l2≈2.2e-4 假 FAIL；参考 `.to(bf16)` 回舍后归零（quant 组的
    packed-byte surface 是同构解法）。另一侧：torch sigmoid 无 tensor÷CPU标量 陷阱。
  - gap sentinel 取值论证：-0.5（bf16 精确、小量级）——大 sentinel（如 smoke 的
    -1234.5）会稀释全面 rel_l2 范数让单点坏内核漏网（EP4×rows1 实测推导）；sentinel
    全等性由 adapter 硬断言兜底。
  - 容差已回填 MEASURED（机器 GB300 sm_103 tray03 2026-07-29）。**未做**：基线账本
    （bench --save 未跑全轴）；EP4 最大点 hottest expert 仅 7 行（近均匀路由的真实
    瘦瓦片负载，多瓦片覆盖在 EP8/EP16 大档自然出现）。

- 2026-07-29 一期骨架落地（本地 x86_64 + RTX 5070 Ti，无 torch 环境）:
  - build.rs 双产物：`OPENINFER_KERNEL_LAB=1` 时在 `ar` 之后用同一 nvcc 把同一批 obj 链成
    `libglm52_kernel_lab.so` 并复制到 `target/release/`；env 未设置零新命令。
    **PIC 判定**：build.rs 全部 nvcc task 本就带 `--compiler-options -fPIC`（三处），
    用 2026-07-28 构建的 46 个 obj 手工 `nvcc -shared` 链接零报错，无需改编译参数。
    **dlopen 判定**：.so 内 DeepEP shim 引用 NCCL 符号（链接时不解析），ctypes 必须
    `RTLD_LAZY`（`RTLD_NOW` 报 `ncclCommQueryProperties` 未定义）；一期单元不调 DeepEP，安全。
  - Python 包 `openinfer-glm52/benches/kernel_lab/`：build/list/check/bench/compare 五子命令，
    torch 全链路惰性导入（list/pytest 在无 torch CPU 机器绿）；示范单元 `mla_front.q_b_gemv`
    （bucket=rows ∈ {1,2,4,8}，n=16384 k=2048）。
    **scratch 规则**：mma 路由（ksplit>0，经 `glm52_gemv_mma_ksplit_cuda` 运行时查询）分配
    `ksplit*batch*n` f32；ksplit=0（register tile）传 NULL。Blackwell q_b bucket8 = {4,1}，
    bucket4 不在 batch-4 表 → register tile。
    **数据**：normal-quantized（per-128²-block absmax/448 scale + 码本 searchsorted 最近舍入），
    非 uniform 随机字节；seed 按用途 sha256 派生（act/weight）。
    **账本**：`baselines/<unit>.json` entries 按 (shape, arch) 键；compare 默认对账本，
    `--baseline-so` 走同会话交替测量。
  - 验收：compileall 绿；pytest 8/8 绿（含 .so 符号解析，用上述手工链接的 .so 放到
    target/release/ 实测）；`kernel_lab list` 输出正确；build.rs rustfmt 解析零错。
    **未做**：真实 GPU check/bench（本地无 torch；待 tray03 GB300 首测后回填 manifest
    tolerance 的 measured 值与机器名）。

- 2026-07-29 rows 轴扩到 64（MTP verify：span-8 × bucket-8 = 64 行/step）:
  - **路由判定**（读 .cu 结论）：单 tile mma 硬件上限 batch≤8（`m16n8k16` 的 N=8 就是 batch 维，
    直接实例化 BATCH=16 会只算 0–7 行静默错）；register tile 的 `acc[ROWS][BATCH]` +
    `xv[BATCH][2]` 寄存器随 batch 线性膨胀（batch 64 ≈ 512+ regs）不可行。故 rows=16/32/64
    走**新 multi-subtile mma 内核** `glm52_gemv_batched_mma_multi_kernel<BTILES,KSPLIT,NTILES>`
    （BTILES=batch/8 列子瓦片共享同一 weight packet，weight HBM 流量与 batch-8 相同，
    tensor-core 工作随 batch 线性；逐行数值结构与 batch-8 mma 相同——同 ksplit 时同一行在
    batch-8 与 batch-64 发射下 bit-identical）。
  - **纯增量、不动旧路径**：只加新内核/新 launcher/新表项/新 case（batch 16/32/64），
    case 1/2/4/8 文本零改动；`glm52_gemv_mma_ksplit_cuda` 仅在查询条件里并集新 batch 值
    （batch≤8 求值不变）→ rows∈{1,2,4,8} 同输入输出 bit-identical 于改动前（推理保证）。
    16/32/64 无 register-tile fallback，off-table fail-closed（INVALID_VALUE）。
  - **表项**：`mma_config` 新增 Blackwell-only 块 `batch∈{16,32,64} q_b → {4,1}`，
    **未实测占位**（镜像 batch-8 胜者），首次 GB300 sweep 负责替换；Hopper 不发表项。
  - **scratch**：规则不变 `ksplit*rows*n` f32（新 launcher 同一条 bounds check）；
    ksplit 查询对 16/32/64 在 Blackwell q_b 返回 4，off-table/非 Blackwell 返回 0 → NULL。
  - **验证**：`nvcc -c -arch=sm_120 -Xptxas -v` 单 TU 过，新内核 BTILES=2/4/8 分别
    56/80/127 regs、0 spill；manifest/适配器/CLI 的 `bucket` 轴整体改名 `rows`
    （`--rows` 可重复）；pytest 9/9 绿（新增 16/32/64 推导断言）。
  - **未做**：GB300 上 rows 16/32/64 的 check+bench 首测与 (ksplit,ntiles) 实测替换。

- 2026-07-29 tray03 环境（容器镜像 = `openinfer-susundev:glm52-kv-layout`，用户指定）:
  - 运行中的 `susundev` 容器是另一会话的（只挂 repo、无 cargo、torch cuda False），不动它；
    新建 `kernel-lab` 容器：同镜像 + `/mnt/shared/home/susun→/work`（cargo 1.97、.venv NCCL
    2.30.7、torch 2.11.0+cu130 4×GB300 可见）+ `--gpus all --ipc=host --network=host`
    + `git config --global safe.directory "*"`。
  - 不碰共享仓另一会话的 `feat/glm52-pd-mtp-arenas` checkout：独立 worktree
    `/mnt/shared/home/susun/openinfer-kernel-lab`（track origin/feat/glm52-kernel-lab）。
  - **坑**：worktree 里 build.rs 自动 submodule init 以容器 root 身份跑失败（ceph 上 .git
    属 susun + 网络 flake）；改在 host 以 susun 跑 `git submodule update --init --recursive`。
  - **坑**：本地 pre-commit `clippy-kernels-kimi`（`^openinfer-kernels/` 触发）需要
    `OPENINFER_NCCL_ROOT` ≥2.30.4——本机 `.venv` 装 `nvidia-nccl-cu13==2.30.7` 解决；
    该钩子每次全量 nvcc 编译，kernels 改动的提交需 15-25 min。

- 2026-07-29 swarm 7 组 → 23 个一期单元（92 pytest 绿 / 1 skipped）:
  - 7 个 coder agent 并行交付：proj-gemv（o_proj 全轴 + qa_kva_pair rows=1）、attention
    （query_assemble/cache_pack/flashmla_sparse.decode，rows×ctx{16k,64k,256k}）、norm 三单元
    （fused 带生产未融合链 byte-compare）、quant 两单元（bit-exact 门 + ue8m0 pow2 硬断言；
    torch RNE 侧已在 CPU 对全部 65282 个 bf16 位型交叉验证 0 差异）、indexer 六单元、
    router 两单元、bookends 三单元 + shared_expert.swiglu。
  - 合并验证：92 passed、`kernel_lab list` 23 单元、合并后 .cu nvcc 单 TU 编译绿。
    3 个 agent 并发改 `glm52_moe_gemv.cu` 的 mma_config Blackwell 16/32/64 块（o_proj、
    shared gate|up、shared down 各补 {4,1} UNMEASURED 占位），纯增量合并干净。
  - **协调约定（agent 自发）**：`test_registry.py::EXPECTED_PHASE1_UNITS` 全量断言与并行
    落地冲突 → "一行一单元追加"约定，编排方统一收口。
  - **BLOCKED（manifest notes 已记）**：`router.noaux_tc` rows>8（min_gemv kMaxTokens=8
    编译期上限 + acc 寄存器墙）；`indexer.weights_proj` rows>8（multi-mma 表无该 shape，
    不写无消费者表项）；`indexer.mqa_logits` rows=64（AOT kAotAlignedBatchSize=32 硬顶）。
  - **事实修正**：`norm.q_a_layernorm` 实为 RMSNorm(2048)（HF `GlmMoeDsaRMSNorm`，无 bias）；
    LayerNorm-with-bias 是 indexer k_norm（indexer 组已覆盖）。**doc 分歧**：
    `indexer-forward.md` 写 rope interleave，落地 `glm52_indexer_rope.cu` 是 half-split
    （NeoX）——torch 参考按 .cu 实际语义，该 doc 建议后续修正。
  - **共享层缺口（agent 提请，暂不阻塞）**：CLI 无 `--ctx` 轴（attention/indexer 组在 refs
    里自写 sweep driver）；`derive_shapes` GEMV 口径（非 GEMV 单元 list 输出有误导数字）；
    `loader.resolve` 固定 c_int restype 不适配 void ABI（norm 组自写 resolve_void）；
    check 门是单 tensor rel_l2（多输出精确门放 adapter 内硬断言）。
  - GB300 构建在 `kernel-lab` 容器后台运行（日志 `/work/kernel-lab-build.log`）。

- 2026-07-29 GB300 首测：20/23 一次通过 → 修复后 23/23 全绿（sm_103 tray03）:
  - 构建 47s（144 nvcc jobs，单 target sm_103；FlashMLA 走 sm_100f，DeepGEMM grouped /
    TileLang 按设计 stub）。示范单元 `mla_front.q_b_gemv` rel_l2=1.66e-3，正中 bf16 floor 推导。
  - 首测 4 个参考端 bug + 1 个容差 FAIL，根因全部查明、kernel 无一有问题：
    1. `flashmla.decode`/`mqa_logits`/`cache_pack`：torch 参考 shape bug（fp8 编码输出
       `[T,4,128]` 未 reshape `[T,512]`；cos/sin 广播把 rows 错位到 head 轴——rows=1
       恰好合法漏网）。两组独立犯同款 expand/reshape 错误——盲写参考（本地无 torch）
       的必然成本，GB300 迭代一轮即修完。
    2. `quant.fp8_per_token_group_bf16` FAIL（rel_l2=4.19e-4，max_abs=1 code step）：
       **torch 把 `tensor ÷ CPU标量` 降级为倒数乘法**（BinaryDivTrueKernel is_cpu_scalar
       快路径），`amax / 448.0` 变 `amax * rn(1/448)`（448 含 1/7 因子，倒数不精确），
       参考 scale 在 57% group 偏 +1 ulp（295 字节差 = 220 scale + 75 连带 value）。
       除数改 device 张量后与 kernel `div.rn` 逐位一致，全 rows rel_l2=0.0。
       **教训（广播全组）**：torch 参考里任何 `tensor / python标量` 都中此陷阱；
       除数是 2 的幂不受影响（mla_attention 的 `/2.0` 安全）。ue8m0 变体因 pow2 bump
       吸收 sub-ulp 差异而天然免疫。
  - 修复后复测：23/23 PASS；实测值已回填 manifest（flashmla ≤2.04e-3、mqa ≤1.67e-3、
    query_assemble/cache_pack/quant = 0.0，均 GB300 sm_103 tray03 2026-07-29）。
  - 迭代通道：本地改文件 → `rsync -e 'ssh -J gb300-login'` 单文件到 tray03 worktree →
    `docker exec -i kernel-lab` 重跑，绕过 git 往返，修复周期 <10 min/组。

- 2026-07-29 EP4 实测 + nsys 归因 + 二期 MoE 单元 + 首轮调优（目标：bs1 vs bs8 差异归因）:
  - **引擎实测**（`glm52_step_bench` EP4 单 tray，同会话）：bucket 1/2/4/8 p50 =
    22.71/27.70/33.74/40.30 ms——bs1→8 差 +17.6ms(+77%)，与 ep4-gb300.md 历史值一致。
  - **nsys node-trace 双 trace diff**（2025.6.3 + `--cuda-graph-trace=node --cuda-flush-interval 1000`，
    配方见 gb300.md）：bs1→8 delta 的 **57% 来自 `moe_ep_wo_masked_mma_kernel`**（b8 GPU 时间
    占比 47%）、DeepEP combine+dispatch ~17%、flashmla ~3%；投影 GEMV 的表观 delta 主要是
    m=1→mma 路径切换 + node-trace 膨胀，solo 账本校准后真实 delta ~1ms——**nsys 原始数必须
    用 kernel_lab solo 账本交叉校准，不能直接当 delta 用**。
  - **二期 MoE 单元落地**：`moe_ep_wo.{tiles,w13_mma,silu,w2_mma}`（EP4/8/16 参数化，
    capacity-proportional shape + seeded 偏斜路由；48/48 check PASS）。**首测教训**：参考返回
    未舍入 f32 会把 bf16 store floor 测成"内核误差"——参考必须回舍到输出 dtype。
    基线（EP4）：w13 rows=1 78.4µs（= #668 的 29% roofline 形状，实测 roofline ~44%）、
    rows=8 118.5µs；w2 rows=8 69.4µs。
  - **调优卡 A（moe masked mma）**：k-loop 双模板 `kPipe=2/4` 兄弟内核 + device guard
    （live blocks ≤448 走 deep pipe）——w13 rows=1 78.4→67.4µs（**-14%**，roofline 44→51%），
    rows≥2 因死兄弟启动 +1.5-2µs；bucket-1 净收益 ≈ -0.7ms/step（-9.5µs×75 层）。
    负结果（都重要）：全局 depth-4 反噬 rows≥2 +13~32%（DRAM row locality + 寄存器占用墙）、
    单内核运行时分支被最深路径连坐压占用。**真正的天花板是 DRAM activate-bound**
    （fragment 布局把事务钉在 64B/6KB stride），下一步 = smem 宽连续加载解耦。
  - **调优卡 B（rows=64 投影占位替换）**：q_b/o_proj/shared gate|up/down 的 {4,1} UNMEASURED
    占位 → per-batch 实测表（+6 个模板实例）：rows=64 q_b 47.1→31.4µs（-32%）、
    o_proj 133.6→78.8（-41%）、swiglu 59.2→44.8（-24%）。NTILES=activation L2 流量杠杆，
    但 BTILES×NTILES×8 f32/thread 寄存器墙限制 BT=8 只能 NT=2。rows≤8 零字节改动
    （diff 核对 + check canary）。
  - **agent 并行调优基建**：tray03 每 agent 独立 worktree（kl-a/kl-b，避免 cargo target 锁与
    .cu 半成品互染）+ 独立 GPU（CUDA_VISIBLE_DEVICES 钉卡）；openinfer-kernel-lab worktree
    永保干净做基线。
  - **引擎 A/B（同会话背靠背）**：bucket-1 22.73→21.75ms（**-4.3%，预测 -0.7ms 实测
    -0.98ms**）；bucket 2/4/8 +0.5~0.7%（死兄弟启动开销，预测 +0.24ms 一致）——内核级
    测量精确预测引擎级结果，harness 可信度闭环。净收益为正，入库；死启动开销的消除
    方案（device 条件启动）与 smem 宽连续加载（DRAM activate-bound，rows=1 预计再 -30%）
    列为下一轮调优卡；EP8/16 的 kDeepPipeBlocks=448 分类阈值需单独实测。

- 2026-07-30 rows=64 专项（用户指令"只针对 bs=64"）：两张卡均**负结果**，根因定量比赢值钱:
  - **卡 C（multi-subtile mma 结构级，o_proj/q_b/swiglu rows=64）**：两个结构尝试全亏并回滚
    （cp.async 256B smem 暂存 +8~13%、depth-4 深流水 +3~10%）。**证伪两条假设**：DRAM
    模式不是主约束（rows=8 同布局跑到 4.4TB/s）；加寄存器深流水被占用墙抵消。
    真正未排除项 = **activation L2 读取的串行依赖**（act uint2 load 直接喂 mma，~500cyc
    L2 延迟暴露）。下一步：先 ncu（`--set full` 抓 o_proj rows=64 单 launch，看 stall 分解
    与 L1/L2 hit rate）再动手；终极答案可能是 tcgen05.mma（tmem 累加绕开寄存器墙，立项级）。
  - **卡 D（flashmla rows=64）**：config 级无可挖——num_sm_parts=152（SM 数）已是谷底
    （扫 64/76/104/128/152/160 证实；parts>152 双波次 69µs）。拟合模型：**P≈11-12µs
    固定 prologue/launch**（152 CTA 的 tmem alloc/mbarrier/TMA 描述符预取）+ **B≈1.75µs/
    topk-block/CTA**（流水线延迟主导，非带宽——ctx 16k→256k 池 10.7MB→172MB 超 L2，
    median 仅 43.5→44.6µs）。rows=64 = 16 blocks/CTA 线性 B 项。下一步：prologue 审计
    （全 rows 档通吃的最大单项）+ B 项内核级 pass（TMA 深度/softmax warp 特化）。
  - **卡 D 副产物（超范围但免费）**：批感知 split `parts=min(32×rows,152)` 给 rows 1-4
    带来 -15~20%（rows=1 23.6→18.1µs）——落地只需 Rust ops 层一行 + step_bench A/B，
    待用户决策。harness 新增两个 env 旋钮（`KERNEL_LAB_FLASHMLA_NUM_SM_PARTS` /
    `KERNEL_LAB_FLASHMLA_GRAPH`，默认=生产行为）已入库。
  - **集成教训**：调优卡改 .cu 表项后必须全量跑 pytest——两个组的 CPU 测试硬断言了旧
    {4,1} 占位文本，红在集成时才发现；已改成"解析实际表项 ↔ 实例覆盖"的持久不变量
    （bare return 只覆盖无 per-batch 项的 batch，须先剥离 per-batch 行再匹配）。

- 2026-07-29 GB300 bench：23/23 单元首批基线落账（`baselines/<unit>.json`）:
  - 全 rows 轴（各单元支持范围内）× 默认 ctx；30 rounds × 10 inner，clocks.sm=2070 MHz，
    git_rev=da68d4f9。示例：q_b_gemv rows=64 median 46.07µs、o_proj rows=64 134.36µs、
    swiglu rows=64 58.92µs、quant/norm/router 4.7–9.0µs 量级。
  - 账本已回拉本地入库（git 追踪，键 (shape, arch) 分桶——H200 数字永不混入）。

## Debrief

- **Outcome**: 一期 harness 全量落地并验收——build.rs 双产物（默认构建零变化由两次无 env
  的 pre-commit clippy 编译实证）、kernel_lab Python 包（list/check/bench/compare）、
  23 个一期单元（7 组）配齐 manifest + torch 参考；CPU pytest 92 绿；GB300 sm_103 上
  check 23/23 PASS（融合单元带生产未融合链 byte-compare：norm.fused_add_rmsnorm_round）、
  bench 23/23 基线入 `baselines/`；Execution Log 全程记录。分支 feat/glm52-kernel-lab
  已推送，tray03 worktree 同步。
- **Pitfalls encountered**:
  - torch `tensor ÷ CPU标量` 倒数乘法降级：除数含 1/7 类因子时参考系统性偏 1 ulp——
    除数改 device 张量。写 torch 参考的通用陷阱，已记入 quant manifest note。
  - 盲写 torch 参考（本地无 torch）必有 shape bug：首测 4 个失败全是参考端，kernel
    零问题；修复通道 = rsync 单文件 + 容器重跑（<10 min/组）。
  - rows>8 不是配置问题：BATCH=16 直接实例化会静默算错（mma N=8 维是 batch 维），
    需要 multi-subtile mma 新内核（纯增量，rows 1–8 bit-identical）。
  - worktree 里 build.rs 以容器 root 跑 submodule init 失败（ceph 权限）→ host 以属主跑。
  - pre-commit clippy-kernels-kimi 钩子需要 OPENINFER_NCCL_ROOT + 全量 nvcc（15-25 min/提交）。
- **Lessons learned**:
  - swarm 铺单元组的前提 = 先有一个端到端验证过的示范单元（模式锚点）；并行落地 vs
    全量断言（EXPECTED_PHASE1_UNITS）冲突用"一行一单元追加"约定解决。
  - 容差推导（bf16 binade / fp8 量子 / bit-exact 论证）首测全部成立：6 个 bit-exact 门
    实测 0.0，其余在 bf16 floor 量级，无一个需要放宽。
- **Follow-ups**:
  - 共享层缺口（agent 提请）：CLI `--ctx` 轴、per-group derive_shapes 钩子、loader
    void-restype、多输出 gate 钩子——二期前值得补。
  - BLOCKED 项：router.noaux_tc rows>8（kMaxTokens=8）、mqa_logits rows=64（AOT 32）——
    需要生产侧决策（是否扩 min_gemv 实例 / AOT 重建）。
  - {4,1} mma 表项是 UNMEASURED 占位：首轮调优任务就是扫描替换（q_b/o_proj/shared
    rows 16/32/64）。
  - 二期：MoE EP weight-only 链 × {EP4,8,16}（#668 优先）+ DeepEP 多卡测量设计。
  - `indexer-forward.md` 的 rope interleave 描述与 .cu 实际（half-split）不符，建议修正。
