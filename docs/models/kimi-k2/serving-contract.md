# Kimi-K2 serving contract

> **TL;DR:** Kimi-K2 经 `/v1/completions` 的 OpenAI 参数面审计(issue #237):greedy(`temperature=0`)+ EOS 停止 + `ignore_eos` 是 honored 集合;`temperature>0` 在 scheduler 准入处显式拒绝(decode 路径只有 split-vocab argmax,没有采样 kernel);其余参数按表格归属,没有 silently-wrong 状态。
>
> **Last touched:** 2026-06

## 参数表

行为分三类:**honored**(语义正确生效)、**rejected**(请求返回明确错误)、**ignored (documented)**(不生效,本表即声明)。

| 参数 | 行为 | 生效层 / 依据 |
| --- | --- | --- |
| `temperature=0` | honored — greedy argmax decode | engine(`runner/worker/forward.rs` split-vocab top1) |
| `temperature>0` | **rejected** — HTTP 500,无生成文本;拒绝原因见 server 日志(详见下文) | scheduler 准入(`runner/scheduler/lifecycle.rs::finish_unschedulable`),两条 shape 路径共用 |
| `top_k` / `top_p` | `temperature=0` 时无语义;`temperature>0` 时随请求整体被拒 | — |
| `max_tokens` | honored | engine,两条路径(#238 起 EOS 优先于 length) |
| `ignore_eos` | honored — 跳过 EOS 检测,生成满 `max_tokens` | engine(#238;frontend 推导修复见 `pegainfer-vllm-frontend::convert_sampling`) |
| `stop`(字符串) | honored — detokenize 后匹配,token 流在 frontend 截断 | vllm-server frontend(`text/output/decoded.rs`),engine 不参与 |
| `stop_token_ids`(自定义) | ignored (documented) — engine 只认模型级 stop 集合(`generation_config.json`);给了自定义 stop token 时 frontend 会保持 EOS 检测开启,但不会在自定义 token 上停 | 所有 pegainfer 引擎一致,Kimi 无特例 |
| `seed` | ignored (documented) — greedy 下无语义;wire 有字段,`convert_sampling` 不读 | pegainfer-vllm-frontend |
| `presence_penalty` / `frequency_penalty` / `repetition_penalty` | ignored (documented) — wire 有字段,`convert_sampling` 不读。注意:penalty 在 greedy 下本应改变 argmax,这是 pegainfer 全引擎层缺口,不是 Kimi 特有 | pegainfer-vllm-frontend |
| `min_p` / `min_tokens` / `logit_bias` / `bad_words` | ignored (documented) — 同上,frontend 层丢弃 | pegainfer-vllm-frontend |
| `logprobs` | ignored (documented) — Kimi-K2 始终回 `logprob: None`(issue #236 跟踪) | engine |
| `echo` | ignored (documented) — `PromptTokens` 事件未实现(issue #236 跟踪) | engine |

## 拒绝行为

非 greedy 请求在 prefill 之前被拒,不占用 GPU step,引擎继续服务后续请求。客户端看到的是 HTTP 500(无任何生成文本):vllm-server 的 HTTP 层把 `EngineCoreFinishReason::Error` 统一折叠成 generic `Internal server error`,引擎侧的错误文本(`StopReason::Text`)在 wire 上被丢弃——这是 pinned vllm-server crate 的限制,不是 Kimi 特有。具体拒绝原因(`Kimi-K2 decodes greedy only; ... Send temperature=0`)由 `pegainfer-vllm-frontend` 在 server 日志里打 warn,运维可见。

后续如果要真正支持采样:decode 路径需要先聚合全 logits(目前 TP8 是 per-rank vocab 分片 top1 直接合并),再接 shared FlashInfer sampling ops;那是功能项,不在 #237 范围。
