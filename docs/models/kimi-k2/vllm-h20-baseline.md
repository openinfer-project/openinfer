# Kimi-K2 vLLM H20 Baseline (decode-heavy)

> **TL;DR:** vLLM `0.19.0` + Kimi-K2.5 + 8× H20，TP1+DP8+EP8（NCCL allgather/reducescatter all2all）跑 `bench serve` decode-heavy profile（input=1, output=128, ignore-eos）。bs=1..256 扫描。同口径下 pegainfer TP8+EP8 bs=4 TPOT med `19.13ms` 比 vLLM `24.97ms` 低 23%，但 pegainfer HTTP 口径比同硬件 in-process（`14.39ms`）高 33%，frontend/streaming overhead 偏大值得专门排查。
>
> **Last touched:** 2026-05

## Setup

| 项 | 值 |
| --- | --- |
| GPU | 8× NVIDIA H20（143 GB） |
| Model | Kimi-K2.5（local `/data/models/Kimi-K2.5`，INT4 + BF16 scale Marlin WNA16） |
| vLLM | `0.19.0`（venv `/root/develop/xingming/vllm_test/.venv`） |
| Sharding | **vLLM**: TP=1, DP=8, EP=8，all2all backend `allgather_reducescatter`（NCCL，默认） |
| Sharding | **pegainfer**: TP=8, EP=8，NCCL F32 hidden bridge + RS routed bridge |
| Profile | input_len=1, output_len=128, `--ignore-eos`, `--random-range-ratio 0` |
| Bench | `vllm bench serve --backend openai --endpoint /v1/completions`（同一 client，两边对齐） |
| 数据 | `/root/develop/xingming/vllm_test/kimi_dp8_baseline/result_*.json` |

## vLLM bs sweep

`vllm bench serve` 同 client 打 vLLM 0.19.0 TP1+DP8+EP8 server。`num_prompts = max(bs*4, 32)`，`request_rate=inf`（asap）。

| bs | reqs | duration (s) | out tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | ITL med (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1   |   32 |  74.93 |   54.7 |   70.8 |  104.0 |  17.94 |  18.12 |  17.93 |
| 2   |   32 |  43.24 |   94.7 |   60.3 |  121.0 |  20.78 |  22.77 |  19.40 |
| 4   |   32 |  25.94 |  157.9 |   69.6 |  135.4 |  24.97 |  29.46 |  23.02 |
| 8   |   32 |  13.27 |  308.6 |   71.8 |  121.8 |  26.44 |  26.57 |  26.40 |
| 16  |   64 |  17.12 |  478.4 |   79.5 |  109.9 |  33.01 |  33.20 |  33.19 |
| 32  |  128 |  28.96 |  565.7 |  128.7 |  189.3 |  55.96 |  56.25 |  56.55 |
| 64  |  256 |  56.12 |  583.9 |  175.2 |  276.2 | 109.00 | 109.76 | 110.52 |
| 128 |  512 |  85.10 |  770.1 |  287.4 |  446.0 | 165.11 | 165.57 | 166.91 |
| 256 | 1024 | 115.86 | 1131.3 |  451.2 |  669.8 | 223.96 | 224.38 | 225.60 |

### 形态

- **TPOT 在 bs≤8 几乎线性**（17.9 → 26.4 ms，单 token 慢 47%；吞吐 5.7×）。bs=8 是 vLLM TP1+DP8+EP8 "免费午餐" 区——8 路独立请求各自落到一个 DP rank，跨 rank 不需要 reduce。
- **bs=8→16→32**：TPOT 26→33→56 ms。一个 DP rank 上要排 2 / 4 个 decode，rank-local MLA / MoE compute 直接 ×2、×4。
- **bs=64..256**：TPOT 109 / 165 / 224 ms 线性增长；fed-batch 已经塞满，aggregate tok/s 还有但单请求体验恶化。
- **TTFT**：bs≤16 维持在 ~70-130 ms（input 只有 1 token，prefill = 一发 graph-safe forward），bs=256 升到 451 ms，被排队拖到。

### 关键 inflection

| 指标 | bs=8 | bs=256 |
| --- | --- | --- |
| Aggregate output tok/s | `308.6` | `1131.3`（峰值，`3.7×` of bs=8） |
| Per-request TPOT med | `26.4ms`（≈38 tok/s/req） | `224.0ms`（≈4.5 tok/s/req） |
| TTFT med | `71.8ms` | `451.2ms` |

**bs=8 ≈ 拐点**：从这一点开始多塞 batch 单请求体验快速恶化，aggregate throughput 增益逐渐被 8 倍 batch / 8 倍 latency 抵消。Decode 性能口径下 vLLM TP1+DP8+EP8 的 "sweet spot" 是 bs=8（8 路 DP 各自 bs=1）。

## pegainfer bs=4 对照点

pegainfer 当前 `KIMI_RUNNER_MAX_BATCH=4` 硬上限，没扫 bs。同 client / 同 profile（`vllm bench serve`，input=1, output=128, ignore-eos, max_concurrency=4）打 pegainfer OpenAI-compatible server：

| 指标 | pegainfer TP8+EP8（HTTP, vllm bench） | vLLM TP1+DP8+EP8 bs=4 | pegainfer in-process bench, bs4 |
| --- | ---: | ---: | ---: |
| TPOT median | `19.13ms` | `24.97ms` | `14.39ms` |
| TPOT p99    | `23.63ms` | `29.46ms` | `14.83ms` |
| ITL median  | `17.42ms` | `23.02ms` | — |
| TTFT median | `313.10ms` | `69.60ms` | — |
| TTFT p99    | `4239.97ms` | `135.40ms` | — |
| Output tok/s | `159.99` | `157.94` | `≈278` |

数据来源：
- pegainfer HTTP：`result_pegainfer_bs4.json`，server 是 `target/release/pegainfer --model-path /data/models/Kimi-K2.5 --port 8124 --cuda-graph true`，client 同 vLLM 那条 bench。
- vLLM bs=4：上面 sweep 表的 bs=4 行。
- pegainfer in-process：`bench_serving request --cuda-graph true --concurrency 4`，见 optimization.md。

### 结论

1. **同硬件、同 client、同 profile，pegainfer TPOT 比 vLLM 低 23%**（`19.13 vs 24.97`）。预期内：pegainfer 走 TP=8 把单 token MLA / dense / shared expert 的 GEMM 切到 8 rank，每发 token 跨 rank reduce 一次；vLLM TP=1 时单 rank 自己跑完整 GEMM，靠 DP=8 拿 throughput 但单请求慢。Decode latency 主线上 TP8 仍然赢。

2. **TTFT 这边 vLLM 完胜**：median `69.60ms` vs pegainfer `313.10ms`。pegainfer p99 飙到 `4239.97ms`——基本是 first-request 冷启动（first NCCL collective stream drain + scheduler warmup）。decode 优先的方案在 prefill 路径上欠的债集中爆在 p99。

3. **HTTP overhead 异常高**：pegainfer 同 bs=4，HTTP 口径 TPOT med `19.13ms`，in-process bench 是 `14.39ms`——4.74ms / token，~33% overhead。streaming JSON + frontend bridge 不该这么多。**这条单独提出来作为后续要查的 finding**，优先级介于 decode kernel 和 prefill 之间。

4. **Aggregate throughput 不公平比较**：pegainfer 卡在 `KIMI_RUNNER_MAX_BATCH=4` 不能扫 bs，vLLM TP1+DP8 在 bs=256 拉到 `1131 tok/s`。Kimi optimization.md 的下一阶段（TP1+DP8+EP8 + PPLX）正好是去拿这块 throughput 的，这次数据直接给那条 milestone 提供了上限：H20 ×8 上，相同 client 口径，**TP1+DP8+EP8 baseline 单请求 TPOT `17.94ms`（bs=1）/ aggregate `1131 tok/s`（bs=256）**，pegainfer 重写后要把单请求 TPOT 卡到 ≤ 这个数同时拿到 ≥ 它的 throughput 才算 TP1+DP8+EP8 这步赚到。

## 复现命令

vLLM server（h20-100）：

```bash
source /root/develop/xingming/vllm_test/.venv/bin/activate
vllm serve /data/models/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --served-model-name kimi-k2.5 \
  --port 8123 --max-num-seqs 256 --max-model-len 4096
```

pegainfer server（h20-100）：

```bash
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:$LD_LIBRARY_PATH \
  /root/develop/xingming/pegainfer-kimi-k2-main/target/release/pegainfer \
  --model-path /data/models/Kimi-K2.5 --port 8124 --cuda-graph true
```

bench（client 端，对哪个 server 改 `--base-url` 即可）：

```bash
vllm bench serve \
  --backend openai \
  --model /data/models/Kimi-K2.5 --tokenizer /data/models/Kimi-K2.5 --trust-remote-code \
  --base-url http://127.0.0.1:8124 --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 --random-output-len 128 --random-range-ratio 0 \
  --num-prompts 32 --max-concurrency 4 --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99
```

完整 sweep 脚本：`/root/develop/xingming/vllm_test/kimi_dp8_baseline/kimi_dp8_sweep.sh`。
