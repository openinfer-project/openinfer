# GLM5.2 MLA Layout — Resolved

> **TL;DR:** GLM5.2 MLA is **bog-standard 64-head DeepSeek-DSA**, no exotic packing. Per layer: `q_a_proj [2048,6144]` → RMSNorm → `q_b_proj [16384,2048]` (= 64 heads × `qk_head_dim` 256); `kv_a_proj_with_mqa [576,6144]` → split(512 ckv, 64 k_pe), RMSNorm(ckv) → `kv_b_proj [28672,512]` (= 64 × (`qk_nope` 192 + `v_head_dim` 256)); `o_proj [6144,16384]` (= 64 × 256). Decode reuses **FlashMLA sparse FP8** (DeepSeek-V3.2 cache): query `[B,1,64,576]` = `[q_nope@W_UK_T, rope(q_pe)]`, 656-byte cache token (512 fp8 ckv + 16 f32 scale + 64 bf16 k_pe), top-k indices from the indexer → latent `[B,1,64,512]` → `v_up` via `W_UV` → `o_proj`. **Do not hand-write MLA** — crib the call from vLLM `flashmla_sparse.py` and the vendored `third_party/FlashMLA`; kimi-k2's `kimi_mla.cu` already uses the same `kv_b_proj→W_UK/W_UV` split.
>
> **Last touched:** 2026-06

## The "factor-4" was a corrupt checkpoint, not an architecture

An earlier version of this doc spent its length theorizing a "256→64 head-fold" because the on-disk weights looked 4× too big. **That was a bad checkpoint, full stop.**

`/data/models/GLM-5.2-0614-Provider-FP8` (a vendor FP8 repack) had:

| tensor | corrupt Provider | OFFICIAL `zai-org/GLM-5.2-FP8` | implies |
|---|---|---|---|
| `q_b_proj`  | `[65536,2048]` ✗ | `[16384,2048]` ✓ | 64 heads × 256 |
| `kv_b_proj` | `[114688,512]` ✗ | `[28672,512]` ✓  | 64 heads × 448 |
| `o_proj`    | `[6144,16384]` ✓ | `[6144,16384]` ✓ | 64 heads × 256 |

The Provider checkpoint's `q_b`/`kv_b` were 4× while `o_proj` was right — internally inconsistent (q/kv imply 256 heads, o_proj implies 64), which is the signature of a mis-repack, not a novel design. The official HF config (`num_attention_heads=64`, `qk_head_dim=256`, `qk_nope=192`, `qk_rope=64`, `v_head_dim=256`, `q_lora=2048`, `kv_lora=512`) and the official weights are fully self-consistent at 64 heads, identical in layout to GLM-5.1-FP8. Verified by range-reading the official safetensors header over the proxy (no full download). The corrupt copy was deleted; the official is downloading to `/data/models/GLM-5.2-FP8`. Lesson captured in memory `lesson-verify-checkpoint-vs-official`.

## Config constants (official)

| Symbol | Value | | Symbol | Value |
| --- | ---: | --- | --- | ---: |
| hidden | `6144` | | heads | `64` |
| q lora | `2048` | | kv lora | `512` |
| qk nope | `192` | | qk rope | `64` |
| qk head | `256` | | v head | `256` |
| layers | `78` (0-2 dense, 3-77 MoE) | | routed experts | `256` top-8 |
| index_topk | `2048` | | index heads × dim | `32 × 128` |

## Decode-forward contract (standard MLA, per row)

```text
q_a   = q_a_proj(hidden)                         [B,2048]
q     = q_b_proj(rmsnorm(q_a))                   [B,16384] -> view [B,64,256]
q_nope, q_pe = split(q, [192,64])                [B,64,192], [B,64,64]
ql_nope = q_nope @ W_UK_T                         [B,64,512]   (W_UK = kv_b[:, :192, :])
kv_a  = kv_a_proj_with_mqa(hidden)               [B,576]
kv_c, k_pe = split(kv_a, [512,64]); kv_c = rmsnorm(kv_c)
append cache token = fp8(kv_c) + scales + bf16(rope(k_pe))     (656 bytes)
q_flashmla = concat(ql_nope, rope(q_pe))         [B,64,576]
topk = indexer(hidden, q_resid, ...)             [B,2048]
latent = flashmla_sparse(q_flashmla, cache, topk)[B,64,512]
attn   = latent @ W_UV                            [B,64,256]   (W_UV = kv_b[:, 192:, :])
hidden_delta = o_proj(attn.reshape[B,16384])      [B,6144]
```

`kv_b_proj` is row-major `[64, (192+256), 512]`: first 192 rows per head = `W_UK` (query absorption), last 256 = `W_UV` (value materialize after attention). Same interpretation kimi-k2 already ships.

## Reuse map (don't hand-write)

- **FlashMLA sparse FP8 decode:** vendored `openinfer-kernels/third_party/FlashMLA/csrc/api/sparse_decode.h` (`DISPATCH_NUM_HEADS`, `get_meta(h_q,s_q)`); call shape mirrored in vLLM `vllm/v1/attention/backends/mla/flashmla_sparse.py` (`num_heads_q=padded_heads` — pads 64→64, `num_heads_k=1`, `topk=2048`, query `[*,576]`, latent out 512).
- **Indexer (DSA top-k):** `GlmMoeDsaIndexer` in transformers `modeling_glm_moe_dsa.py` (`wq_b`, `wk`, `k_norm`, `index_head_dim=128`, `index_n_heads=32`); vLLM `Indexer` / `DeepseekV32IndexerBackend` in `deepseek_v2.py`. Hardest slice (DeepGEMM-JIT) — Slice 4.
- **MLA weight split / v_up:** `openinfer-kernels/csrc/kimi_k2/kimi_mla.cu`.
- **HF reference forward (numeric oracle):** `/data/code/workspace-rustllm/transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` `GlmMoeDsaAttention.forward`.

## Next

1. After the official download lands, re-assert the standard shapes (rewrite/replace `tests/mla_layout_probe.rs` against `/data/models/GLM-5.2-FP8`).
2. Build per-stage MLA decode by adapting FlashMLA sparse + kimi MLA split; validate intermediate shapes/values against the HF reference on one layer before wiring the PP hot path.
