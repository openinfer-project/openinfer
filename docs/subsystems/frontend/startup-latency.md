# Frontend startup latency

TL;DR: Qwen3-4B default serving now overlaps vLLM frontend metadata loading with engine startup; `/v1/models` readiness on the local RTX 5090 box moved from `3087 ms` external poll / `2897 ms` server bind to `1855-2082 ms` external poll, with completions smoke passing.

Last touched: 2026-06

## 2026-06-11 Qwen3-4B startup pass

The original startup path loaded the GPU engine first, then entered `vllm_server::serve_with_router_extension`, which loaded Hugging Face tokenizer/text/chat backend metadata before binding HTTP. The measured internal timeline was:

- `0.00s`: `openinfer` main starts.
- `0.23s`: 3 safetensor shards mmaped, 8045 MB.
- `1.30s`: GPU weights loaded.
- `1.75s`: engine/KV/scheduler loaded.
- `2.90s`: vLLM text/chat backend loaded and HTTP bound.
- `3.09s`: external `/v1/models` poll observed ready.

The frontend metadata load does not need an `EngineHandle`. The default non-LoRA path now starts the vLLM frontend task before synchronous engine load, and the local bridge waits on a one-shot engine handle before completing the bootstrapped engine handshake. This keeps normal request handling unchanged: `/v1/completions` and `/v1/chat/completions` still connect through the same vLLM server, IPC bridge, and scheduler after the real engine handle is available.

Verified command:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-} \
  ./target/release/openinfer --model-path /data/models/Qwen3-4B --port 18080
```

Three `/v1/models` runs with 100 ms polling landed at `2082 ms`, `1952 ms`, and `1855 ms` after rebasing onto main's sampling guard changes. In server logs, tokenizer loading starts immediately after runtime options, frontend metadata finishes around `1.14-1.22s`, engine load finishes around `1.69-1.90s`, and HTTP binds immediately after the bridge handshake. A `max_tokens=1` `/v1/completions` request returned 200 after ready.

LoRA startup intentionally remains sequential because startup adapter loading needs the real engine control handle before serving.
