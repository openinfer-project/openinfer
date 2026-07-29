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

## Debrief

（待执行后填写）
