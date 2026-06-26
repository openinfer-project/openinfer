# GLM5.2 MLA Layout Probe

> **TL;DR:** GLM5.2 checkpoint shapes are source-compatible with vLLM MLA only if the extra factor is kept explicit: `q_b_proj [65536,2048] = [64,4,256,2048] = [256,256,2048]` and `kv_b_proj [114688,512] = [64,4,448,512] = [256,448,512]`. Inside each 256/448 row group, vLLM's rule is still `[W_UQ 192; W_QR 64]` and `[W_UK 192; W_UV 256]`. FlashMLA sparse decode wants an absorbed local query `q=[B,1,H_local,576] = [q_nope@W_UK, q_pe]`, packed FP8 MLA cache token bytes `512 fp8 ckv + 16 f32 scale + 128 bf16 k_pe`, sparse indices, and returns latent `[B,1,H_local,512]`, which must then run `v_up` through `W_UV` before `o_proj`. What is not yet proven is whether GLM's factor `4` is a TP-like packing dimension, a head expansion dimension, or a slice/collapse rule hidden in vLLM/transformers glue.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routes GLM5.2 work under `docs/models/glm52`.
  - `docs/models/glm52/decode-forward-contract.md` - records `q_b`/`kv_b` split as the explicit unresolved attention boundary.
  - `docs/models/glm52/pp-decode.md` - confirms this task should advance the shared decode-forward contract, not PP placement.
  - `docs/models/glm52/vllm-kernel-reference.md` - source map says to check vLLM/FlashMLA before local attention kernels.
  - `openinfer-glm52/src/config.rs` - GLM constants: heads `64`, q lora `2048`, kv lora `512`, qk nope `192`, rope `64`, v dim `256`, q_b out `65536`, kv_b out `114688`.
  - `openinfer-glm52/src/weights/view.rs` - typed weight views currently validate tensor shape/dtype but do not encode MLA semantic slices.
  - `../vllm/vllm/model_executor/models/deepseek_v2.py` - GLM5.2 uses the DeepSeek MLA/DSA path; eager attention splits `q_b` and `kv_b` after viewing by head.
  - `../vllm/vllm/model_executor/layers/attention/mla_attention.py` - explains MLA symbols and implements `W_UK_T` absorption plus `_v_up_proj`.
  - `../vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py` - sparse FP8 MLA cache format and FlashMLA call shape.
  - `../vllm/csrc/libtorch_stable/cache_kernels.cu` - `fp8_ds_mla` cache append checks `kv_lora_rank=512`, `pe_dim=64`, cache token `656` bytes; `concat_mla_q` requires `ql_nope` multiple of 512 plus `q_pe=64`.
  - `openinfer-kernels/third_party/FlashMLA/csrc/api/sparse_decode.h` and `csrc/params.h` - sparse decode ABI accepts `q [b,s_q,h_q,d_qk]`, packed KV, top-k indices, and `d_v=512`.
  - `openinfer-kernels/csrc/kimi_k2/kimi_mla.cu` - OpenInfer's Kimi MLA kernels already use `kv_b_proj` as row-major `[head, k_nope + v, kv_lora_rank]`, first slice for Q absorption and second slice for `v_up`.
- **Relevant history**:
  - `docs/models/glm52/decode-forward-contract.md` - first attention backend should validate FlashMLA sparse with externally supplied top-k before wiring the full indexer.
- **Plan**:
  1. Record the exact vLLM/FlashMLA/Kimi source evidence for q/kv MLA factors.
  2. Derive GLM5.2 tensor shapes from checkpoint dimensions and config constants.
  3. Add a small ignored integration test that checks real checkpoint metadata shape/dtype and the derived layout arithmetic.
  4. List what remains unproved before decode-forward can consume this contract.
- **Risks / open questions**:
  - The local `/data/models` directory in this session does not contain the GLM5.2 checkpoint, so the ignored IT was not run against real metadata here.
  - GLM5.2 uses FP8 checkpoint weights; this probe proves layout and header contracts, not numeric equivalence after dequantization.

## Source Evidence

vLLM names the MLA matrices in `mla_attention.py`: `q_b_proj` is `[W_UQ; W_QR]` concatenated per head, and `kv_b_proj` is `[W_UK; W_UV]` concatenated per head. The same file's decode path does:

1. split `q` into `q_nope` and `q_pe`;
2. compute `ql_nope = q_nope @ W_UK_T`;
3. concatenate `ql_nope` and `q_pe` into the FlashMLA query;
4. run sparse/dense MLA to get latent output with width `kv_lora_rank`;
5. run `_v_up_proj`, i.e. latent `@ W_UV`, to produce `[B, heads, v_head_dim]` before `o_proj`.

`DeepseekV2MLAAttention` constructs `q_b_proj` with output `num_heads * qk_head_dim` and `kv_b_proj` with output `num_heads * (qk_nope_head_dim + v_head_dim)`. The eager fallback views and splits:

```text
q_b output -> [-1, heads, qk_nope_head_dim + qk_rope_head_dim]
kv_b output -> [-1, heads, qk_nope_head_dim + v_head_dim]
```

That fallback is the strongest shape oracle, but sparse FlashMLA does not consume full `k_nope/v`; it consumes compressed KV cache plus absorbed query.

FlashMLA sparse FP8 decode uses the DeepSeek V3.2 cache format when `kv_lora_rank=512` and `rope_dim=64`:

```text
token bytes = 512 fp8 ckv + 16 bytes f32 scales + 64 bf16 k_pe = 656 bytes
query       = [B, S_q, H, 576] bf16
output      = [B, S_q, H, 512] bf16 latent
```

Kimi's local CUDA code confirms the OpenInfer side of the `kv_b_proj` memory interpretation: it treats `kv_b_proj` as row-major `[local_heads, k_nope + v, kv_lora_rank]`; the first `k_nope` rows are used as `W_UK_T` for Q absorption, and the slice starting at `kNopeDim * kKvLoraRank` is used as `W_UV` for `v_up`.

## GLM5.2 Shape Derivation

Config constants:

| Symbol | Value |
| --- | ---: |
| heads | `64` |
| q lora rank | `2048` |
| kv lora rank | `512` |
| qk nope head dim | `192` |
| qk rope head dim | `64` |
| v head dim | `256` |
| FlashMLA q latent dim | `512 + 64 = 576` |
| o_proj input dim | `64 * 256 = 16384` in current OpenInfer contract |

The raw checkpoint dimensions are not `64 * (qk_nope + qk_rope)` and `64 * (qk_nope + v)`. Both have the same extra factor:

```text
65536  / (64 * (192 + 64))  = 4
114688 / (64 * (192 + 256)) = 4
```

So there are two shape views that must be kept distinct:

```text
config-head view:      [64, 4, q_or_kv_dim, rank]
projection-head view:  [256, q_or_kv_dim, rank]
```

The projection-head view is exactly what vLLM's generic MLA code expects after tensor-parallel slicing: each local projection head has `[qk_nope, qk_rope]` or `[qk_nope, v]`. The config-head view is how current OpenInfer constants describe the checkpoint: 64 model heads with a 4-way expansion. This probe does not prove which view the first PP8/TP1 decode path should execute.

### `kv_b_proj [114688,512]`

Raw rows in the two views:

```text
114688 / 64  = 1792 = 4 * (192 + 256)
114688 / 256 = 448  = 192 + 256
```

The source-backed per-projection-head split is:

```text
kv_b_weight_fp8: [256, 448, 512] or [64, 4, 448, 512]
W_UK:            [256, 192, 512] or [64, 4, 192, 512]
W_UV:            [256, 256, 512] or [64, 4, 256, 512]
```

For one local head partition, Kimi/vLLM-style decode uses `W_UK` for query absorption and `W_UV` after attention:

```text
q_nope [B,H_local,192] @ W_UK_T [H_local,192,512] -> ql_nope [B,H_local,512]
latent [B,H_local,512] @ W_UV [H_local,512,256] -> attn_out [B,H_local,256]
```

The unresolved part is `H_local`: with 4-way packing it could be 64 after a TP-like slice, or 256 if all projection heads are local. The latter cannot feed the current `o_proj [6144,16384]` contract without an additional slice/reduction, because `256 * 256 = 65536`, not `16384`.

### `q_b_proj [65536,2048]`

Raw rows in the two views:

```text
65536 / 64  = 1024 = 4 * (192 + 64)
65536 / 256 = 256  = 192 + 64
```

vLLM says `q_b_proj` is `[W_UQ; W_QR]` per projection head:

```text
q_b_weight_fp8: [256, 256, 2048] or [64, 4, 256, 2048]
W_UQ:           [256, 192, 2048] or [64, 4, 192, 2048]
W_QR:           [256,  64, 2048] or [64, 4,  64, 2048]
```

The sparse FlashMLA query for one local head partition is:

```text
q_flashmla = concat(q_nope @ W_UK_T, q_pe)
shape      = [B, H_local, 512 + 64] = [B,H_local,576]
```

Do not collapse `[64,4,...]` to `[64,...]` by picking a group, summing groups, or assuming group 0 until vLLM's GLM5.2 load/runtime path or an intermediate numeric probe proves that rule.

## Decode-Forward Contract

For a GLM5.2 sparse MLA decode row:

```text
q_a = q_a_proj(hidden)                         [B,2048]
q_b = q_b_proj(rmsnorm(q_a))                   [B,65536]
kv_a_raw = kv_a_proj_with_mqa(hidden)          [B,576]
kv_c, k_pe_raw = split(kv_a_raw, [512,64])
kv_c_normed = rmsnorm(kv_c)                    [B,512]

q_nope, q_pe = split_q_b(q_b)                  [B,H_local,192], [B,H_local,64]
ql_nope = q_nope @ W_UK_T                      [B,H_local,512]
k_pe = rope(k_pe_raw)                          [B,64]

append cache token = fp8(kv_c_normed) + scales + bf16(k_pe)
q_flashmla = concat(ql_nope, rope(q_pe))       [B,H_local,576]
latent = flashmla_sparse(q_flashmla, cache, topk_indices) [B,H_local,512]
attn_out = latent @ W_UV                       [B,H_local,256]
hidden_delta = o_proj(attn_out)                [B,6144]
```

Use the database analogy: `kv_b_proj` is both an index projection (`W_UK`) and a value materializer (`W_UV`). FlashMLA's cache stores the compressed row (`kv_c`) as the page payload; `W_UK` is used on the query side to build the lookup key, while `W_UV` is used after lookup to reconstruct the value payload for `o_proj`.

The current OpenInfer `o_proj` input contract is `[B,16384] == [B,64,256]`. Therefore a decode-forward implementation must first prove how the checkpoint's factor `4` maps to `H_local=64` before wiring `v_up -> o_proj`. Running all `256` projection heads locally would produce `[B,65536]`, which does not match `o_proj`.

## Probe / IT

Added `openinfer-glm52/tests/mla_layout_probe.rs` as an ignored integration test. It is intentionally metadata-only:

- reads hard-coded `/data/models/GLM-5.2-0614-Provider-FP8`;
- checks `config.json` constants that define the layout;
- finds layer-0 `q_b_proj` and `kv_b_proj` shards through `model.safetensors.index.json`;
- reads safetensor headers and asserts FP8 weight / F32 scale dtype and shapes;
- asserts the 4-way projection expansion arithmetic, vLLM per-projection-head split arithmetic, and FlashMLA's `656` byte FP8 cache token.

The test does not load GPU weights, dequantize FP8, or run attention. It should stay ignored until the checkpoint is present on the target node.

## Still Not Proven

- Whether the 4-way projection expansion is TP-like packing, an expanded-head dimension, or a GLM-specific slice/collapse rule.
- Which `H_local` FlashMLA should see in PP8/TP1. FlashMLA sparse headers support 64/128 heads, while the raw projection-head view is 256.
- Which slice/group of `q_b/kv_b` feeds current `o_proj [6144,16384]`.
- Numeric equivalence between OpenInfer's future split/absorb/v_up implementation and vLLM after FP8 dequantization.
- Sparse indexer top-k and cache layout correctness. This probe only covers MLA projection/cache factor shapes.

## Next Minimal Probe

Run vLLM on one GLM5.2 layer with a tiny synthetic hidden batch and dump the intermediate tensors immediately before FlashMLA sparse decode:

```text
q_b raw shape and any reshape/slice used for q_nope/q_pe
kv_b dequantized per-head split used for W_UK/W_UV
the runtime value of H_local passed to FlashMLA
q_flashmla shape and first few finite checks
latent output shape before v_up
o_proj input shape after v_up
```

Then add an OpenInfer ignored IT that uses the same checkpoint layer and compares only these intermediate shape/value relationships before integrating the decode hot path.

## Execution Log

### Step 1: Source inspection

- Read the GLM52 docs and config/view files requested in the task.
- Inspected vLLM `deepseek_v2.py`, `mla_attention.py`, `flashmla_sparse.py`, cache kernels, FlashMLA sparse decode headers, and Kimi MLA CUDA.
- Result: source-backed contract for `kv_b_proj -> W_UK/W_UV`, FlashMLA query/cache/output, and `v_up`; GLM-specific 4-way projection packing remains open.

### Step 2: Metadata probe

- Added `openinfer-glm52/tests/mla_layout_probe.rs`.
- Result: ignored IT captures checkpoint metadata assertions without touching runtime code.

## Debrief

- **Outcome**: The document now records the strongest source evidence and a concrete shape contract for continuing GLM5.2 decode-forward attention work.
- **Pitfalls encountered**:
  - The local `/data/models` tree did not include GLM5.2, so the new ignored IT could not be run against the real checkpoint here.
  - The first-order vLLM MLA documentation describes normal per-head dims; GLM5.2's `q_b=65536` and `kv_b=114688` add a 4-way packing factor that still needs a vLLM intermediate dump before coding hot-path splits.
- **Lessons learned**:
  - Treat `kv_b_proj` as layout metadata, not a normal forward-only projection: decode consumes its first slice during query absorption and its second slice after FlashMLA returns latent output.
  - FlashMLA sparse FP8 cache stores compressed KV pages, so `v_up` is mandatory before `o_proj`.
- **Follow-ups**:
  - Run the ignored metadata IT on the GLM checkpoint host.
  - Add a vLLM intermediate dump probe for the unresolved 4-way packing before writing OpenInfer split kernels.
