# Kimi-K2 TP1 Decode 路径逐算子性能剖析

> **TL;DR:** 一篇**理论**博客：按真实执行顺序走一遍 Kimi-K2 TP1+DP8+EP8（PPLX EP backend）decode 路径上的每个 GPU 算子，从计算特性的角度分析——算子在算什么、shape 如何随 batch size `B` 变化、FLOPs/字节/算术强度（AI）以及它在 roofline 上的位置。结论一律用变量表达（如 GEMM 的 `AI=B`、H20 ridge `=30.83`），不绑定某个具体工作点。数据源是 `tp1-dp8-ep8-decode-optimization-master.md` 主表和各 `*_report.md`。进度见下方全景表。
>
> **Last touched:** 2026-06

---

> ### ✍️ 写作者提示（风格 + 反例，动笔前先读）
>
> 本文是**理论博客**，目标是讲清每个算子的计算特性，不是开发记录、不是优化日志。
>
> **风格要求：**
> 1. **只讲理论结论，用变量表达。** 例：GEMM 的 `AI ≈ B`、`memory-bound ⟺ B < ridge`。不要把某个工作点（如 profiling 锚点 `B=8`）当成普遍前提——**decode 的 batch size 是自由变量，可以很大**（如 WideEP 用宽专家并行摊薄权重时，单 rank 的 `B` 可越过 ridge，使同一个 GEMM 从 memory-bound 翻到 compute-bound）。给具体数时标明"在 `B=8` 锚点处"。
> 2. **不写实现细节。** 不出现 CTA / grid / block / waves/SM / launch / kernel 符号名 / 文件行号。这些是代码考古，读者要的是特性与 shape。
> 3. **不下开发结论。** 不写"是不是优化目标""该不该融合""值不值得调"。是否优化是工程决策，不属于理论分析。
> 4. **禁止主观引导词。** 结论靠不等式与数字，不靠形容词。禁用：牢牢、死死、闲置、浪费、可惜、轻松、显然、当然、不出所料、令人意外 等。
>
> **反例（bad case）：**
> - ❌ "decode 的 `B=8 ≪ 31`，所以 GEMM **牢牢** memory-bound，剩下 74% 算力**闲置**。" —— 把 `B=8` 当普遍前提 + 主观词 + 价值判断。
> - ✅ "GEMM 的 `AI = B`。当 `B < ridge(30.83)` 时 memory-bound、瓶颈是读权重；`B > ridge` 时 compute-bound。可达 FLOP 利用率 `≈ min(B/ridge, 1)`，在 `B=8` 锚点处约 `26%`。"

---

## 一、先说清楚"TP1 path"到底是什么

这里的 "TP1" 不是"单卡张量并行=1、其它默认"。在 Kimi-K2 的部署语境里，**TP1 = TP1 + DP8 + EP8，走 PPLX EP backend，跑在 H20 ×8 上**：

| 维度 | 取值 | 含义 |
|---|---|---|
| Tensor Parallel | 1 | 每个 rank 持有完整的一份模型权重，attention/dense 段**没有跨卡 all-reduce** |
| Data Parallel | 8 | 8 个独立的 decode engine，各自跑自己的请求 batch |
| Expert Parallel | 8 | 384 个 routed expert 切到 8 个 rank，MoE 段用 PPLX all-to-all 做 dispatch/combine |
| Backend | PPLX | routed expert 计算走 Marlin WNA16（INT4 权重 + BF16 激活），不是 NCCL all-reduce 那条 MoE 路径 |
| 硬件 | H20 | BF16 峰值 `148 TFLOP/s`、HBM 峰值 `4.8 TB/s`、ridge point `30.83 flop/byte` |

**为什么要强调这个**：TP1 让 attention 和 dense 段彻底没有通信，但代价是 MoE 段从"卡内 all-reduce"变成了"跨卡 all-to-all"。所以这条路径的性能故事天然分成两半——**attention/dense 是纯本地 skinny-GEMM + 一堆 tiny-grid 控制 kernel**，**MoE 是 PPLX 路由 + Marlin 量化 GEMM + EP 通信**。

### 基准负载（master 表的锚点）

所有延迟数字都钉在同一个负载点上，不然没法比：

| 项 | 值 | 说明 |
|---|---|---|
| 单 DP-rank batch | `active_rows = 8` | 主表的锚定 shape |
| 全局 batch | `bs ≈ 64` | DP8 × 每 rank 8 行 |
| 上下文长度 | `ctx = 1` | decode 单步，KV 只 append 1 个 token |
| 层数 | 61 | layer 0 是 dense FFN，layer 1–60 是 MoE |

> ⚠️ `ctx=1` 是这张表的默认锚点。attention 里唯一对上下文长度敏感的算子是 `flashinfer_mla_decode`：它读 KV cache 的字节量 ∝ ctx，`ctx=1` 时极小，长上下文下成本随 ctx 线性增长，写到那一节时单独展开。

### Roofline 约定与定性词汇

后面每个算子都按 roofline 定性，先约定几个词：

- **memory-bound**：`AI < ridge`，理论瓶颈是 HBM 带宽。decode 的 skinny GEMM（M=B 很小）几乎都落这里，因为主成本是固定的权重读取。
- **compute-bound**：`AI > ridge`，理论瓶颈是 BF16 算力。decode 单步几乎没有算子能到这一侧（B 太小）。
- **低 AI / 激活体量算子**：embedding、norm、激活、残差加这类，`AI ≤ 1`，只搬激活（∝ `B·H`），没有权重矩阵，绝对数据量极小。
- **roofline 左端延迟区**：roofline 假设能打满对应瓶颈；但当一个算子搬的数据少到连 HBM 都喂不饱时，它落在带宽饱和之前的延迟区，实测吞吐远低于带宽线——这是模型对小负载的失真处，decode 小 B 下大量小算子都在这里。
- **通信（EP comm）**：PPLX 的 dispatch/combine all-to-all 传输，属互连带宽/延迟而非本地算力，单列。

---

## 二、一个 decode step 的算子全景（按执行顺序）

下面是一个 decode step 的真实算子执行顺序：embedding 调 1 次 → 61 层循环（每层 attention + FFN）→ 收尾 norm/lm_head/argmax 调 1 次。

类型一栏是 roofline 预判（逐节验证）。"次/step" 是该算子一步里被调用的次数——它和单次成本一起决定 step 占比。

**Pre-loop（每 step 1 次）**

| 执行序 | 算子 | 次/step | 特性（roofline 预判） | § | 状态 |
|---:|---|---:|---|---|---|
| 1 | `embedding` | 1 | gather，AI=0，激活体量 | §1 | ✅ |

**每层 attention 块（layer 0–60 都有 → ×61）**

| 执行序 | 算子 | 次/step | 特性（roofline 预判） | § | 状态 |
|---:|---|---:|---|---|---|
| 2 | `attention.input_norm` | 61 | 低 AI（norm），memory 侧 | §2 | ✅ |
| 3 | `attention.qkv_a` | 61 | memory-bound（权重读取主导，AI≈B） | §3 | ✅ |
| 4 | `attention.qkv_a_split_norm` | 61 | 低 AI（split + norm） | §4 | ✅ |
| 5 | `attention.q_b` | 61 | memory-bound（权重读取主导） | §5 | ✅ |
| 6 | `attention.rope_split` | 61 | 低 AI（elementwise） | §6 | ✅ |
| 7 | `attention.absorb_q_nope` | 61 | memory-bound（per-head batched GEMM） | §7 | ✅ |
| 8 | `attention.paged_kv_append` | 61 | 低 AI（写 KV cache） | §8 | ✅ |
| 9 | `attention.flashinfer_mla_decode` | 61 | ctx 敏感（ctx=1 低 AI；长 ctx → 读 KV memory-bound） | §9 | ✅ |
| 10 | `attention.v_up` | 61 | memory-bound（per-head batched GEMM） | §10 | ✅ |
| 11 | `attention.o_proj` | 61 | memory-bound（权重读取主导） | §11 | ✅ |
| 12 | `attention.post_attn_add_norm` | 61 | 低 AI（add + norm） | §12 | ✅ |

> TP all-reduce 在 attention 与 o_proj 后各有一处，**TP1 下无 TP 通信、全部跳过**，不计入。

**每层 FFN —— layer 0 走 dense（×1）**

| 执行序 | 算子 | 次/step | 特性（roofline 预判） | § | 状态 |
|---:|---|---:|---|---|---|
| 13 | `dense.gate_up` | 1 | memory-bound（权重读取主导） | §13 | ✅ |
| 14 | `dense.swiglu` | 1 | 低 AI（elementwise/SFU） | §14 | ✅ |
| 15 | `dense.down` | 1 | memory-bound（权重读取主导） | §15 | ✅ |
| 16 | `dense.residual_add` | 1 | 低 AI（elementwise） | §16 | ✅ |

**每层 FFN —— layer 1–60 走 MoE / PPLX（×60）**

| 执行序 | 算子 | 次/step | 特性（roofline 预判） | § | 状态 |
|---:|---|---:|---|---|---|
| 17 | `moe.router` | 60 | 低 AI（小 GEMM + topk 选择） | §17 | ✅ |
| 18 | `moe.shared_gate_up` | 60 | memory-bound（权重读取主导） | §18 | ✅ |
| 19 | `moe.shared_swiglu` | 60 | 低 AI（elementwise/SFU） | §19 | ✅ |
| 20 | `moe.shared_down` | 60 | memory-bound（权重读取主导） | §20 | ✅ |
| — | `pplx.dispatch_send/recv` | 60 | EP 通信（all-to-all 传输） | §comm | ✅ |
| 21 | `moe.pplx_build_marlin_routing` | 60 | 低 AI（routing 元数据） | §21 | ✅ |
| 22 | `moe.pplx_marlin_w13` | 60 | memory-bound（INT4 权重读取） | §22 | ✅ |
| 23 | `moe.pplx_swiglu` | 60 | compute 侧（SFU 密集） | §23 | ✅ |
| 24 | `moe.pplx_marlin_w2` | 60 | memory-bound（INT4 权重读取） | §24 | ✅ |
| — | `pplx.combine_send/recv` | 60 | EP 通信（all-to-all 传输） | §comm | ✅ |
| 25 | `moe.residual_add_scaled` | 60 | 低 AI（elementwise） | §25 | ✅ |

> MoE/PPLX 段内 shared-expert 计算（18–20）与 routed dispatch/combine 通信存在 overlap，§17–25 的精确顺序写 MoE 章节时再核对，本表先按 stage 分组列出。

**Post-loop（每 step 1 次）**

| 执行序 | 算子 | 次/step | 特性（roofline 预判） | § | 状态 |
|---:|---|---:|---|---|---|
| 26 | `final.norm` | 1 | 低 AI（norm） | §26 | ✅ |
| 27 | `final.lm_head` | 1 | memory-bound（全 vocab 权重读取，近 HBM 峰值） | §27 | ✅ |
| 28 | `final.argmax` | 1 | memory-bound（读 logits） | §28 | ✅ |

**为什么 "次/step" 重要**：embedding / final.\* 一步只 1 次；attention 段每个算子 ×61；dense 只在 layer 0 ×1；MoE 段 ×60。一个算子在 step 中的总成本 = 单次成本 × 次数，所以单次成本低的算子在 ×60/×61 后仍可能占据可观份额。

---

## 三、逐算子拆解

### §1 — `decode.embedding`

**本质**：把每个待解码序列的 token id 查嵌入表，写进残差流 `hidden`。一 thread 一个输出元素的纯 gather——`out[b, :] = embed[token_id[b], :]`，零乘加，AI=0。

**Shapes**（记 batch size = `B`；decode 下每序列只解 1 个 token，所以行数 = `B`）：

| tensor | shape | dtype | 随 B |
|---|---|---|---|
| `embed`（嵌入表） | `[V=163840, H=7168]` | BF16 | 固定 |
| `token_ids` | `[B]` | u32 | ∝ B |
| `out`（`hidden`） | `[B, H=7168]` | BF16 | 行数 = B |

- **固定（const）**：词表 `V=163840`、hidden `H=7168`。TP1 不分片，整张表常驻、`vocab_start=0`，所以这两维对这条路径就是常量。
- **随 B 变**：只有行数。搬动字节 = 读 B 行 + 写 B 行 = `2·B·H·2` B（`B=8` 时 ≈ 224 KB），**线性于 B**。

**理论分析**：纯 gather，**AI = 0**（零乘加），没有 compute 维，理论天花板只有 HBM 带宽线。它只搬 `B` 行激活（∝ `B·H`，B=8 ≈ 224 KB），不读整张表——成本随 B 线性增长但绝对量极小。这么小的数据量连 HBM 都喂不饱，落在 roofline 左端"带宽饱和之前"的延迟区：这是 roofline 模型对小负载失真的地方，实测吞吐（<1% 峰值）反映的是延迟下限而非带宽。

---

### §2 — `attention.input_norm`

每个 transformer 层进 attention 前对残差流做的 RMSNorm，一步里出现 61 次。

**本质**：逐行归一化 —— `out[b,:] = hidden[b,:] / sqrt(mean(hidden[b,:]²) + ε) · γ`。各行相互独立，沿 H 维做一次均方归约再逐元素缩放。

**Shapes**：

| tensor | shape | dtype | 随 B |
|---|---|---|---|
| `hidden`（输入） | `[B, H=7168]` | BF16 | 行数 = B |
| `γ`（缩放向量） | `[H=7168]` | BF16 | 固定 |
| `out`（输出） | `[B, H=7168]` | BF16 | 行数 = B |

**理论分析**：
- FLOPs ≈ `4·B·H`（每元素约 2 次乘加算均方 + 2 次缩放）。
- 字节（BF16）≈ 读 `hidden` + 写 `out` = `4·B·H`，外加一次性的 `γ` 读 `2·H`。
- **AI = FLOPs / bytes ≈ 1 flop/byte，且与 B 无关**——分子分母都 ∝ `B·H`。H20 ridge 是 `30.83`，`AI=1 < ridge`，属 memory-bound，且因 AI 不随 B 变，这个结论对任意 B 都成立。

**关键点：它只搬激活、不读权重矩阵**（`γ` 仅是个 `[H]` 向量）。所以成本 ∝ `B·H`，绝对量很小（B=8 ≈ 238 KB），**随 B 线性增长而 AI 恒定**。这与下一节的 GEMM 形成对照：GEMM 在 decode 下的主成本是固定的权重读取、几乎与 B 无关。和 §1 embedding 同属"激活体量"算子——便宜、线性于 B，且数据量太小时一样落在 roofline 左端的延迟区。

---

### §3 — `attention.qkv_a`：第一个 GEMM，也是理解"decode 为何 memory-bound"的关键

MLA 进 attention 的第一个投影：把 `H=7168` 的残差流压成一个 `2112` 维的低秩表示。这个 `2112` 是三段拼起来的——query 的低秩 `1536` + KV 压缩潜变量 `512` + key 的 RoPE 分量 `64`。MLA 之所以 KV cache 极小，就靠这步把 KV 压到 `512` 维。

**本质**：标准矩阵乘 `Y = W · X`。

**Shapes**：

| tensor | shape | dtype | 随 B |
|---|---|---|---|
| `W`（权重） | `[N=2112, K=7168]` | BF16 | 固定 |
| `X`（输入） | `[K=7168, B]` | BF16 | 列数 = B |
| `Y`（输出） | `[N=2112, B]` | BF16 | 列数 = B |

**理论分析（这套推导对后面每个 decode GEMM 都成立）**：

- FLOPs = `2·N·K·B`（每个输出元素一次 K 长的点积）。
- 字节 = 读 `W`（`2·N·K`）+ 读 `X`（`2·K·B`）+ 写 `Y`（`2·N·B`）。decode 下 `B` 很小，**`W` 的读取压倒一切**：`W` 30.3 MB，而 `X+Y` 才 ~0.15 MB。
- 于是算术强度
  $$\text{AI} = \frac{2NKB}{2NK + 2KB + 2NB} \approx \frac{2NKB}{2NK} = B$$
  即 **权重读取主导时，GEMM 的 `AI ≈ B`（batch size）**。代入 `N=2112, K=7168`，在 `B=8` 锚点处精确值为 `7.96 flop/byte`，接近 `B`。

- H20 ridge point `= 148 TFLOP/s ÷ 4.8 TB/s = 30.83 flop/byte`。`AI ≈ B` 给出一个干净的判据：roofline 上的位置只取决于 `B` 与 ridge 的关系。

  | 条件 | 结论 | 可达 FLOP 利用率 |
  |---|---|---|
  | `B < 30.83` | AI < ridge → memory-bound，瓶颈是读 `W` | `≈ B / 30.83` |
  | `B > 30.83` | AI > ridge → compute-bound，瓶颈是 BF16 算力 | `≈ 1` |

  `B` 是自由变量：profiling 锚点取 `B=8`，此时 `AI≈8 < 30.83`，落在 memory-bound 一侧，可达算力 `≈ 8/30.83 ≈ 26%`；但当 `B` 增大越过 `ridge≈31`（例如 WideEP 用宽专家并行把更多请求摊到单 rank 时），同一个 GEMM 就翻到 compute-bound 一侧。**结论是 `B` 的函数，而不是某个固定工作点的属性。**

**两条由 `AI ≈ B` 直接推出的结论，构成 decode 性能的骨架：**

1. **memory-bound 区（`B < ridge`）内，GEMM 的时间下限 = 读一遍权重的时间 = `2NK / 带宽`，与 `B` 无关。** qkv_a 为 `30.3 MB / 4.8 TB/s ≈ 6.3 µs`。因此在此区间内增大 `B`，墙钟时间近似不变而完成的行数 ∝ `B`，吞吐 ∝ `B`；这一线性关系在 `B = ridge` 处饱和。
2. 与 §1/§2 的"激活体量"算子对照：norm/embedding 成本 ∝ `B`、AI 恒定且 `≤1`；GEMM 在 memory-bound 区成本 ≈ 常数（权重体量）、`AI = B`。一步的时间 = 各 GEMM 的权重读取项 + 各激活算子的 `∝B` 搬运项之和。

---

> **约定（后文复用）**：对 GEMM `Y = W·X`、`W:[N,K]`、`X:[K,B]`，FLOPs `= 2NKB`。BF16 权重读取主导时 `AI ≈ B`（见 §3）；在 memory-bound 区（`B < ridge`）时间下限 `= 2·N·K·s_w / 带宽`（`s_w` = 每权重字节，BF16 为 2、INT4 为 0.5）。"激活体量算子" = 只搬激活、不读权重矩阵的算子（norm / 激活 / 残差加 / RoPE / 拆分 / 写 cache），其 `AI ≤ 1` 且字节、FLOPs 都 ∝ `B`。下文对这两类只给出与基准的差异。

### §4 — `attention.qkv_a_split_norm`

**本质**：把 §3 的 `2112` 维输出按语义切成三段——q 的低秩 `[B,1536]`、KV 压缩潜变量 `[B,512]`、key 的 RoPE 分量 `[B,64]`——并对前两段各做一次 RMSNorm。

**Shapes**：输入 `[B,2112]` → 三个输出 `[B,1536] / [B,512] / [B,64]`，全部行数 = B。两个 norm 权重 `γ[1536]`、`γ[512]` 固定。

**理论**：激活体量算子。字节 ≈ 读 `2112` + 写 `2112`（每行），FLOPs ∝ 两段的归约/缩放，`AI ≈ 1`，与 §2 同档：memory 侧、∝ B、绝对量极小。

### §5 — `attention.q_b`：query 上投影

**本质**：把 q 低秩 `[B,1536]` 升回每头的 query —— `Y = W·X`，`W:[N=12288, K=1536]`。`12288 = 64 头 × 192`（每头 `128` nope + `64` rope）。

**理论**：标准 GEMM，`AI = B`。权重 `12288×1536×2 ≈ 37.7 MB`，memory-bound 区时间下限 `≈ 37.7MB / 4.8TB/s ≈ 7.9 µs`，与 B 无关。roofline 位置同 §3：`B<ridge` memory-bound、`B>ridge` compute-bound。

### §6 — `attention.rope_split`

**本质**：对 query / key 的 RoPE 分量施加旋转位置编码（按元素的 sin/cos 重组），并把 nope / rope 分量摆放到各头的布局里。

**Shapes**：作用在 `[B, 64头, 192]` 量级的激活上，行数 = B。

**理论**：激活体量算子，逐元素三角运算，无权重矩阵、无跨 token 归约。`AI` 由每元素几次乘加决定（`≪ ridge`），字节、FLOPs 均 ∝ B。

### §7 — `attention.absorb_q_nope`：把 W_UK 吸收进 query（MLA 的核心技巧）

**本质**：MLA 不在每头里展开 K，而是把 key 上投影 `W_UK` **吸收**进 query：将每头的 q_nope `[B,128]` 投到压缩潜变量空间 `[B,512]`，之后 attention 直接在 `512` 维潜空间里做。这是一个对 64 个头的 strided-batched GEMM——每头 `[B,128]×[128,512]→[B,512]`。

**Shapes**：输入 `[B, 64×128]`、输出 `[B, 64×512=32768]`，吸收权重 `[64,128,512]` 固定。

**理论**：批量 GEMM，对 H=64 头、batch=B：FLOPs `= 2·H·B·K·N`，权重字节 `= 2·H·K·N`，故 **`AI = B`** 同 §3。其特点是每头的 `K=128、N=512` 较小，属许多小 GEMM 的批量，绝对算术强度低但 roofline 判据仍是 `B vs ridge`。

### §8 — `attention.paged_kv_append`

**本质**：把本 step 新算出的压缩 KV（潜变量 `512` + k_rope `64` = `576`/token）写进分页 KV cache 的对应槽位。

**Shapes**：每行写 `576` 个 BF16，行数 = B。

**理论**：纯写入，`AI ≈ 0`，激活体量。**注意它写入的字节 ∝ B 而非 ∝ ctx**——这正是 MLA 省显存的来源：每 token 的 KV 只有 `576` 维，而非标准 MHA 的 `头数×头维×2`。这一压缩比直接决定下一节的 KV 流量。

### §9 — `attention.flashinfer_mla_decode`：唯一对 ctx 敏感的算子

**本质**：MLA 解码注意力。每个 query（B 行 × 64 头）在压缩潜空间里与 KV cache 的全部 `ctx` 个条目做注意力。因为 §7 已把 `W_UK` 吸收进 query，所有 64 个头共享同一份压缩 KV `[ctx, 576]`。

**理论**：这是全表唯一**字节、FLOPs 同时 ∝ ctx** 的算子：
- 读 KV `≈ ctx · 576 · 2` 字节；FLOPs `≈ 2 · B · 64 · ctx · 576`。
- 理想情形（KV 只读一次、被 `64·B` 个 query-头复用）算术强度 `≈ 64·B`，远高于投影 GEMM——这是 MLA 把 KV 压缩 + 头间共享带来的高复用。
- 但两个量都 ∝ ctx：`ctx=1` 时绝对成本极小（与其它激活算子同档）；ctx 增大时成本线性增长，并成为 attention 内随上下文唯一增长的项。其它所有 decode 算子的成本都与 ctx 无关。

换句话说，**短上下文下 attention 的成本在投影 GEMM（§3/§5/§7/§10/§11）上，长上下文下逐渐转移到这一步的 KV 流量上**。

### §10 — `attention.v_up`：从潜空间还原 value

**本质**：注意力在 `512` 维潜空间得到结果后，用吸收的 `W_UV` 把每头还原到 value 头维 `128`。对 64 头的 strided-batched GEMM——每头 `[B,512]×[512,128]→[B,128]`。

**Shapes**：输入 `[B, 64×512]`、输出 `[B, 64×128=8192]`，权重 `[64,512,128]` 固定。

**理论**：同 §7 的批量 GEMM，`AI = B`；与 §7 互为 MLA 压缩/解压的两端。

### §11 — `attention.o_proj`：输出投影

**本质**：把 64 头拼接的 `[B,8192]` 投回残差流维度 —— `Y = W·X`，`W:[N=7168, K=8192]`。

**理论**：标准 GEMM，`AI = B`。权重 `7168×8192×2 ≈ 117 MB`——是单层 attention 里最大的一次权重读取，时间下限 `≈ 117MB / 4.8TB/s ≈ 24 µs`（memory-bound 区，与 B 无关）。

### §12 — `attention.post_attn_add_norm`

**本质**：把 attention 输出加回残差流，并对结果做 RMSNorm（一次融合的 add + norm）。

**理论**：激活体量算子，`AI ≈ 1`，∝ B，与 §2 同档。它是 attention 块与 FFN 块之间的衔接。

---

#### FFN —— layer 0：dense MLP（×1）

layer 0 是唯一的 dense 层，结构是标准 SwiGLU MLP。

### §13 — `dense.gate_up`

`Y = W·X`，`W:[N=36864, K=7168]`（`36864 = 18432×2`，gate 与 up 拼在一起）。标准 GEMM，`AI = B`。权重 `≈ 528 MB`——是全模型单次最大的稠密权重读取之一，时间下限 `≈ 110 µs`（memory-bound 区）。

### §14 — `dense.swiglu`

`out = SiLU(gate) ⊙ up`，把 `[B,18432]` 的 gate/up 逐元素合成 `[B,18432]`。激活体量算子，含 SiLU 的 SFU 运算；字节、FLOPs ∝ B，`AI` 取决于每元素的 SiLU+乘，`≪ ridge`。

### §15 — `dense.down`

`Y = W·X`，`W:[N=7168, K=18432]`。标准 GEMM，`AI = B`。权重 `≈ 264 MB`，时间下限 `≈ 55 µs`。

### §16 — `dense.residual_add`

把 MLP 输出加回残差流。激活体量算子，`AI ≈ 0.5`（一次加法/元素），∝ B。

---

#### FFN —— layer 1–60：MoE（PPLX + INT4 Marlin，×60）

每个 MoE 层有一个**共享专家**（所有 token 都过，§18–20）和 `384` 个**路由专家**（每 token 选 `8` 个，§21–25），路由专家用 EP all-to-all 分散到 8 个 rank、权重为 INT4。

### §17 — `moe.router`

**本质**：算每个 token 对 `384` 个专家的亲和度并选 top-8（Kimi 的 noaux_tc：分组 + sigmoid 打分 + 选择）。

**Shapes**：打分是 GEMM `W:[N=384, K=7168]` → `[B,384]`，随后是 `384→8` 的 topk/分组选择。

**理论**：打分 GEMM 的 `AI = B`，但 `N=384` 很小，权重仅 `≈ 5.5 MB`；选择部分是控制密集、低 AI 的归约。整体属低 AI、∝ B 的小算子。

### §18 — `moe.shared_gate_up`

`Y = W·X`，`W:[N=4096, K=7168]`（共享专家 gate+up，`4096 = 2048×2`）。标准 GEMM，`AI = B`，权重 `≈ 58.7 MB`。**注意它 ×60 层**：单次便宜，但乘以 60 后是 step 内的可观项。

### §19 — `moe.shared_swiglu`

共享专家的 SwiGLU，`[B,4096]→[B,2048]`。激活体量算子，∝ B，×60。

### §20 — `moe.shared_down`

`Y = W·X`，`W:[N=7168, K=2048]`。标准 GEMM，`AI = B`，权重 `≈ 29.4 MB`，×60。

### §comm-a — `pplx.dispatch_send / dispatch_recv`：EP all-to-all（分发）

**本质**：把每个 token 的隐藏向量 `[7168] BF16`（≈14 KB）发往它选中的 8 个专家所在的 rank，并接收路由到本地专家的 token。

**理论**：这是**通信**而非本地算力，瓶颈是 GPU 间互连带宽与延迟，不在 BF16/HBM roofline 上。传输字节 ∝ `B · topk · H`（发送）与本 rank 接收到的 token 数 × H。它与本地计算可重叠，单列、不计入本地 roofline。

### §21 — `moe.pplx_build_marlin_routing`

**本质**：根据 all-to-all 收到的 token 数，为 Marlin 分组 GEMM 构造路由/分块元数据（每个本地专家的行区间、padding 到块大小等）。

**理论**：纯控制/元数据，`AI ≈ 0`，规模 ∝ 接收到的 token 数；为下两步的分组 GEMM 做准备。

### §22 — `moe.pplx_marlin_w13`：路由专家 gate/up（INT4）

**本质**：对本 rank 收到的、路由到各本地专家的 token 做 gate/up 投影。这是一个**分组 GEMM**——每个活跃本地专家用自己的 INT4 权重 `[4096, 7168]` 处理分配给它的若干行。

**理论（INT4 把 roofline 判据整体左移）**：
- FLOPs `= 2·N·K·M`（`M` = 该层全部路由工作行）；权重字节 `= N·K·s_w`，**INT4 `s_w = 0.5`**（外加少量分组 scale）。
- 单专家 `AI = 2NKM_e / (0.5·NK) = 4·M_e`（`M_e` = 路由到该专家的行数）。**INT4 让 AI 是等价 BF16 GEMM 的 4 倍**，于是 memory/compute 的分界点从 `M_e = ridge` 左移到 `M_e = ridge/4 ≈ 7.7`。
- 成本结构：要读**所有活跃本地专家**的 INT4 权重（∝ 活跃专家数），而每个专家只摊到少量行（decode 小 B 时 `M_e` 小）。所以瓶颈是"为少量行读多份专家权重"——这正是 MoE 解码的典型 memory-bound 形态。增大全局 batch 或用 WideEP 提高每专家行数 `M_e`，会抬高 `AI = 4·M_e`、把它推向 compute-bound（呼应开头写作提示里的 `B` 自由变量）。

### §23 — `moe.pplx_swiglu`

路由专家的 SwiGLU，作用在路由行 `[M,4096]→[M,2048]`。激活体量/SFU 密集算子，规模 ∝ 路由行数 `M`。

### §24 — `moe.pplx_marlin_w2`：路由专家 down（INT4）

同 §22 的分组 INT4 GEMM，权重 `[N=7168, K=2048]`。`AI = 4·M_e`，roofline 判据同 §22 左移 4 倍。

### §comm-b — `pplx.combine_send / combine_recv`：EP all-to-all（合并）

**本质**：把各专家算完的 token 输出沿来路送回原 rank、按 top-k 权重合并。与 §comm-a 对称的通信，传输字节同量级，互连带宽/延迟主导，不计入本地 roofline。

### §25 — `moe.residual_add_scaled`

**本质**：把（共享专家 + 路由专家加权和）的结果按缩放系数加回残差流。

**理论**：激活体量算子，`AI ≈ 0.5`，∝ B，×60。

---

#### Post-loop（×1）

### §26 — `final.norm`

最后一层 RMSNorm，shape 与 §2 完全相同（`[B,7168]`，`γ[7168]`）。激活体量算子，`AI ≈ 1`，∝ B。区别只是一步调一次。

### §27 — `final.lm_head`：全词表投影，单次最大的权重读取

**本质**：把残差流投到词表 logits —— `Y = W·X`，`W:[N=V=163840, K=7168]`。

**理论**：标准 GEMM，`AI = B`。权重 `163840×7168×2 ≈ 2.35 GB`——**全模型单次最大的权重读取**，比任何单层 GEMM 大一到两个数量级。memory-bound 区时间下限 `≈ 2.35GB / 4.8TB/s ≈ 490 µs`，与 B 无关。它和 lm 输入 embedding（§1）共享同一张表，但一个是 `AI=0` 的查表（只取 B 行）、一个是 `AI=B` 的全表 GEMM（读全部 V 行）——同一份权重，两种极端的访问模式。

### §28 — `final.argmax`

**本质**：对每行 logits `[V=163840]` 取最大值的下标（greedy 采样）。

**Shapes**：读 `[B, V]`、写 `[B]`，行数 = B。

**理论**：归约算子。读 `B·V·2` 字节（`B=8` 时 `≈ 2.6 MB`）、几乎零 FLOPs（只比较），`AI ≈ 0`，memory 侧。字节 ∝ B（每行独立扫一遍 V）。

---

## 四、把 28 个算子拼回一个 step：理论结构

抛开绝对数字，decode 一步的时间在理论上是两类项之和：

- **权重读取项（各 GEMM）**：在 memory-bound 区（`B < ridge`）与 B 无关，等于"读一遍该权重 / 带宽"，再乘以该 GEMM 在一步里的调用次数。
- **激活搬运项（norm/激活/残差/RoPE/拆分/写 cache/argmax）**：∝ B，`AI ≤ 1`，绝对字节小。

按"权重字节 × 次数"排序，理论上的大项是：

| 来源 | 单次权重 | 次/step | 量级特征 |
|---|---|---:|---|
| 路由专家 INT4（§22+§24） | 活跃本地专家权重之和（INT4，`s_w=0.5`） | 60 | 为少量行读多份专家权重；`AI=4·M_e`，随每专家行数左右横跨 ridge |
| `final.lm_head`（§27） | `≈2.35 GB` | 1 | 单次最大权重读取 |
| attention 投影（§3/5/7/10/11） | 单层合计 `≈200 MB`（o_proj 为最大项） | 61 | `AI=B`，BF16 |
| 共享专家（§18/20） | 单层 `≈88 MB` | 60 | `AI=B`，BF16 |
| dense MLP（§13/15） | `≈792 MB` | 1 | `AI=B`，仅 layer 0 |

三条贯穿全文的理论主线：

1. **`AI = B` 让 batch size 成为 roofline 上的唯一旋钮**：BF16 GEMM 在 `B<ridge(30.83)` 时 memory-bound、`B>ridge` 时 compute-bound；`B` 是自由变量（WideEP 可推高它）。
2. **量化把判据左移**：INT4 路由专家 `AI = 4·M_e`，分界点降到 `ridge/4`，所以 MoE 段对"每专家行数"比稠密段更敏感。
3. **MLA 把 KV 成本压到最小且仅它随 ctx 增长**：KV 每 token `576` 维、头间共享，短上下文下 attention 成本在投影 GEMM 上，长上下文下转移到 §9 的 KV 流量上；其余算子全部与 ctx 无关。

---

> 写作约定：数字一律以 `tp1-dp8-ep8-decode-optimization-master.md` 主表和对应 `*_report.md` 为准；主表更新了就回来同步本文。本文是叙事层，主表是数据层。
