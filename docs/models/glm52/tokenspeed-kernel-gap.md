# GLM5.2 TokenSpeed Kernel Gap

> **TL;DR:** TokenSpeed 的 GLM5.2 路径除了已经在 OpenInfer 暴露的 DeepGEMM scale layout、grouped FP8 合约/metadata、FlashMLA sparse decode 外，还依赖 DeepGEMM DSA indexer logits、Blackwell TRTLLM sparse MLA、deterministic top-k、FP8 token-group quant、DSA index-K cache layout、Hadamard 以及 MoE/router/expert 路径；Hadamard 上游已确认是 Dao-AILab 的公开 CUDA 库，其余外部入口按“公开源码 / vendored 源码 / wheel-only torch.ops”分开标注。
>
> **Last touched:** 2026-06

## OpenInfer 当前公开面

`openinfer-kernels` 里 GLM5.2 现在只有三类接口：

### `glm52.deepgemm.scale_layout`

把 row-major f32 scales 转成 DeepGEMM 需要的 MN-major/TMA-aligned layout。这是 load/package helper，不是 GEMM compute。

### `glm52.deepgemm.grouped_fp8_contract`

校验 GLM5.2 W13/W2 grouped FP8 contract，并生成 grouped metadata。compute entry 仍返回 `CUDA_ERROR_NOT_SUPPORTED`，还没有接真实 DeepGEMM runner。

### `glm52.flashmla.sparse_decode`

SM90 FlashMLA V32 sparse decode wrapper。固定 `batch<=128`、`heads=64`、`qk_dim=576`、`v_dim=512`、`page_size=64`、packed KV row 656 bytes、`topk=2048`；没有 sparse prefill，也没有动态 `topk_lens` ABI。

所以现在的 OpenInfer GLM5.2 surface 更像“第三方 substrate 的窄门面”，还不是可直接跑 TokenSpeed GLM5.2 的完整 kernel set。

## TokenSpeed GLM5.2 Kernel DAG

TokenSpeed 的 GLM5.2 关键路径可以按数据流看：

1. Dense/linear 阶段
   `GlmDsaAttention` 里 fused q/k layernorm、q_b/kv_b projection 等仍依赖 dense GEMM。FP8 权重路径会让 `process_weights` 在 scale layout 转换后选择 DeepGEMM；TokenSpeed 的 `deep_gemm_mm_fp8_blockscale` 对应 DeepGEMM `fp8_gemm_nt`。

2. DSA indexer 阶段
   `GlmDsaIndexer` 生成 index query/key，包含 RMSNorm、RoPE 和 Hadamard rotate。Hadamard 通过 `tokenspeed_kernel.thirdparty.fast_hadamard_transform.hadamard_transform` 调用。

3. index-K cache 写入
   `DSATokenToKVPool.set_index_k_buffer` 把 index key 做 FP8 token-group quant，然后按 DeepGEMM `fp8_paged_mqa_logits` 需要的 page layout 写入 cache：每页先放 `page_size * head_dim` 的 FP8 bytes，再放 `page_size * num_groups` 的 f32 scale。

4. Prefill top-k
   `_compute_prefill_topk_indices_deepgemm` 对 q 做同样的 FP8 token-group quant，调用 `deep_gemm.fp8_mqa_logits` 计算每个 token 到历史 index-K 的 logits，再用 `torch.ops.trtllm.indexer_topk_prefill` 选本地 top-k，最后把 workspace offset 转成全局 KV slot。

5. Decode top-k
   `_compute_decode_topk_indices_deepgemm` 调用 `deep_gemm.get_paged_mqa_logits_metadata` 和 `deep_gemm.fp8_paged_mqa_logits` 计算 paged indexer logits，再用 FlashInfer deterministic top-k 选本地 offset，随后用 `local_topk_to_global_slots` 转成 KV slot。

6. Sparse attention
   `DSABackend.forward_sparse_prefill` 和 decode 都调用 `trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=...)`。prefill 会先把 workspace indices 映射成 KV slots；decode 直接把 top-k slots reshape 成 TRTLLM sparse MLA 需要的 block table。TokenSpeed 在这里显式传 `backend="trtllm-gen"`，也就是 Blackwell 路径不是我们当前 SM90 FlashMLA V32 wrapper。

7. MoE
   `GlmMoeDsaDecoderLayer` 复用 `DeepseekV3MoE`：gate/router 先出 router logits 和 grouped top-k，再走 `tokenspeed_kernel.moe_plan` 选 expert 后端。FP8 MoE 后端会做 activation token-group quant，并调用 FlashInfer/TRTLLM FP8 block-scale MoE。

## 搬运入口

下面逐个 kernel 写清楚：TokenSpeed 从哪里 import / 调用，真正实现在哪里，OpenInfer 应该落成什么接口。每个小节都按同一组字段写，方便后面按项开工。

## 外部源码核对

这一节只记录“后续要抄代码时应该去哪里看”。TokenSpeed 的 Python wrapper 只是调用图，不等于 CUDA 源码。

### Fast Hadamard Transform

- **上游仓库**: `https://github.com/Dao-AILab/fast-hadamard-transform.git`。
- **本地核对路径**: 已 clone 到 `/tmp/fast-hadamard-transform`，当前 HEAD 是 `e7706fa`。
- **源码入口**:
  - `/tmp/fast-hadamard-transform/fast_hadamard_transform/fast_hadamard_transform_interface.py` 暴露 `hadamard_transform(x, scale=1.0)`。
  - `/tmp/fast-hadamard-transform/csrc/fast_hadamard_transform.cpp` 做 PyTorch binding、reshape/contiguous/pad 和 dtype/device check。
  - `/tmp/fast-hadamard-transform/csrc/fast_hadamard_transform_cuda.cu` 是 CUDA kernel launcher。
- **包名关系**: TokenSpeed 的依赖名是 `tokenspeed-fast-hadamard-transform`，公网 PyPI metadata 的 Homepage 仍指向 Dao-AILab 的 `fast-hadamard-transform`；TokenSpeed 包只发布 wheel，真正可读源码看上游 GitHub。
- **License**: `/tmp/fast-hadamard-transform/LICENSE` 是 BSD-3-Clause。搬源码前要保留 license。
- **结论**: 这个不再是“没有源码”。可以按上游 CUDA 实现移植一个窄版 `glm52.indexer.hadamard_bf16`，不要只抄 TokenSpeed 的 `__init__.py` wrapper。

### DeepGEMM

- **TokenSpeed 包**: `tokenspeed-deepgemm`，公网 PyPI metadata 没有 homepage/project_urls，当前发布物只有 wheel。
- **OpenInfer vendored 源码**: `openinfer-kernels/third_party/DeepGEMM`，origin 是 `https://github.com/deepseek-ai/DeepGEMM`，当前 HEAD 是 `54e2261`。
- **结论**: 后续抄 GLM5.2 的 MQA logits / dense FP8 GEMM 时，不要从 TokenSpeed wheel 找源码，直接从 OpenInfer vendored DeepGEMM 接窄 C/Rust ABI。

### FlashInfer

- **上游仓库**: `https://github.com/flashinfer-ai/flashinfer`。
- **OpenInfer vendored 源码**: `openinfer-kernels/third_party/flashinfer`，当前 HEAD 是 `d768c14`。
- **覆盖的 GLM5.2 入口**: deterministic top-k、TRTLLM-gen MLA wrapper、TRTLLM FP8 MoE wrapper 都应优先看 vendored FlashInfer，而不是 TokenSpeed 的 Python registry。
- **结论**: 这是可读源码路径。需要注意 FlashInfer 的 Python API 可能还有 JIT/TVM artifact loader，OpenInfer wrapper 要把这层构建和加载边界收窄。

### FlashMLA

- **TokenSpeed 包**: `tokenspeed-flashmla`，公网 PyPI metadata 没有 homepage/project_urls，当前发布物只有 wheel。
- **OpenInfer vendored 源码**: `openinfer-kernels/third_party/FlashMLA`，origin 是 `https://github.com/deepseek-ai/FlashMLA`，当前 HEAD 是 `9241ae3`。
- **结论**: 这只对应 SM90 FlashMLA packed-row 路径。TokenSpeed GLM5.2 的 Blackwell sparse MLA 主路径不是这里，而是 FlashInfer 的 TRTLLM-gen MLA wrapper。

### TRTLLM custom ops

- **TokenSpeed 包**: `tokenspeed-trtllm-kernel`，PyPI summary 是 `Standalone TensorRT-LLM CUDA kernels as PyTorch custom ops`，license 标成 Apache-2.0，但没有 homepage/project_urls，当前发布物只有 wheel。
- **相关 op**: `torch.ops.trtllm.fp8_quantize_1x128`、`torch.ops.trtllm.indexer_topk_prefill`、`torch.ops.trtllm.indexer_topk_decode`、`torch.ops.trtllm.dsv3_fused_a_gemm_op`。
- **公开上游参考**: `https://github.com/NVIDIA/TensorRT-LLM.git` 是 TensorRT-LLM 官方仓库，但还没有确认它包含 TokenSpeed wheel 中这些 exact custom ops 的同源实现。
- **结论**: 这里仍按 black box 处理。文档里的 TRTLLM op 只能抄调用合同；要么找到 wheel 对应源码，要么写本地 CUDA replacement。

## DSA Attention 搬运顺序

1. FP8 token-group quant：先 quant q 和 index-K。
2. index-K cache set/gather：把 index-K 写成 DeepGEMM paged MQA logits 需要的 block-split layout。
3. DeepGEMM MQA logits：prefill 用 contiguous MQA logits，decode 用 paged MQA logits。
4. Top-k select：decode 用 deterministic top-k，prefill 用 prefill top-k selector。
5. Offset to KV slot：把本地 top-k offset 转成全局 KV slot，并写 `topk_lens`。
6. Blackwell sparse MLA：用 TRTLLM-gen MLA 消费普通 MLA KV cache、top-k slots 和 `topk_lens`。

### DSA Hadamard rotate

- **Source tag**: public GitHub C++/CUDA package via TokenSpeed wheel wrapper。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/models/glm5.py` 的 `_glm_dsa_hadamard_rotate` 和 `_glm_dsa_hadamard_rotate_pair`。
- **TokenSpeed wrapper/import**: caller 从 TokenSpeed wrapper import：

```python
from tokenspeed_kernel.thirdparty.fast_hadamard_transform import (
    hadamard_transform,
)
```

wrapper 文件是 `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/fast_hadamard_transform/__init__.py`，里面真正的 import 是：

```python
from fast_hadamard_transform import hadamard_transform
```

- **真实实现来源**: TokenSpeed 仓库没有 CUDA 实现源码；外部包名在 `/data/code/tokenspeed/python/tokenspeed/env.py`，依赖项叫 `tokenspeed-fast-hadamard-transform`。已确认公开上游是 `https://github.com/Dao-AILab/fast-hadamard-transform.git`，本地核对 clone 在 `/tmp/fast-hadamard-transform`，当前 HEAD 是 `e7706fa`。核心源码看 `csrc/fast_hadamard_transform.cpp` 和 `csrc/fast_hadamard_transform_cuda.cu`。
- **OpenInfer landing**: `openinfer-kernels/src/ops/glm52` 下新增 `glm52.indexer.hadamard_bf16`。
- **Shape/layout contract**: 输入转 BF16，reshape 成 `[-1, head_dim]` contiguous，调用 `hadamard_transform(..., scale=head_dim ** -0.5)`，再 reshape 回原 shape；pair 版本会先 concat query/key rows，再切回来。
- **是否可直接搬**: 可以从 Dao-AILab 上游移植，license 是 BSD-3-Clause；不要从 TokenSpeed wrapper 误判实现，wrapper 只有转发 import。

### DSA decode indexer logits

- **Source tag**: DeepGEMM vendored C++/CUDA。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/models/glm5.py` 的 `_compute_decode_topk_indices_deepgemm`。
- **TokenSpeed wrapper/import**: 通过 `tokenspeed_kernel.thirdparty.deep_gemm` 调用 `deep_gemm.get_paged_mqa_logits_metadata(...)`、`deep_gemm.get_num_sms()`、`deep_gemm.fp8_paged_mqa_logits(...)`。
- **真实实现来源**: `openinfer-kernels/third_party/DeepGEMM/csrc/apis/attention.hpp` 有 `get_paged_mqa_logits_metadata` 和 `fp8_paged_mqa_logits` C++ binding；kernel 在 `openinfer-kernels/third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp`，底层 include 是 `openinfer-kernels/third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`。
- **OpenInfer landing**: `glm52.deepgemm.paged_mqa_logits`。
- **Shape/layout contract**: GLM5.2 decode indexer 固定 page size 64、index head dim 128、top-k 512/1024/2048，输入是 paged FP8 index-K cache、f32 scale、block table、context lens、schedule metadata。
- **是否可直接搬**: 不抄 TokenSpeed Python wrapper；直接接 DeepGEMM C++/CUDA 入口。

### DSA prefill indexer logits

- **Source tag**: DeepGEMM vendored C++/CUDA。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/models/glm5.py` 的 `_compute_prefill_topk_indices_deepgemm`。
- **TokenSpeed wrapper/import**: 通过 `tokenspeed_kernel.thirdparty.deep_gemm` 调用 `deep_gemm.fp8_mqa_logits(...)`。
- **真实实现来源**: `openinfer-kernels/third_party/DeepGEMM/csrc/apis/attention.hpp` 的 `fp8_mqa_logits` binding；kernel 在 `openinfer-kernels/third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp` 和对应 include。
- **OpenInfer landing**: `glm52.deepgemm.mqa_logits`。
- **Shape/layout contract**: prefill 输入是 contiguous q/k FP8、f32 scale、row start/end；不要和 decode 的 paged layout 混成一个 ABI。
- **是否可直接搬**: 直接接 DeepGEMM C++/CUDA 入口。

### DSA FP8 token-group quant

- **Source tag**: TokenSpeed Triton/CUDA path plus TRTLLM `torch.ops` path。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/models/glm5.py` quant q；`/data/code/tokenspeed/python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py` quant index-K cache。
- **TokenSpeed wrapper/import**: TokenSpeed 通用入口是 `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/fp8_utils.py` 的 `per_token_group_quant_fp8`。Blackwell TRTLLM 路径走 `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/trtllm/__init__.py` 的 `per_token_group_quant_8bit`，里面调用：

```python
torch.ops.trtllm.fp8_quantize_1x128(x, use_ue8m0)
```

- **真实实现来源**: `fp8_utils.py` 里有 TokenSpeed 本地 Triton/CUDA 实现；Blackwell 快路径只暴露为 `torch.ops.trtllm.fp8_quantize_1x128`。这个 op 来自 `tokenspeed-trtllm-kernel` wheel，当前 PyPI metadata 没有源码 URL。
- **OpenInfer landing**: `glm52.fp8.token_group_quant_128_f32`。
- **Shape/layout contract**: group size 128，float32 scale；先覆盖 q 和 index-K cache 两个 caller。
- **是否可直接搬**: 可以抄 TokenSpeed 本地 quant 语义；Blackwell TRTLLM quant 只能从 `torch.ops` 进，接入前要决定是用 FlashInfer/TRTLLM 入口还是写本地 CUDA kernel。

### DSA index-K cache set/gather

- **Source tag**: TokenSpeed model/runtime layout logic。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py`。
- **TokenSpeed wrapper/import**: `set_index_k_buffer` quant index-K 并写入 paged cache；`gather_index_k` 给 prefill `fp8_mqa_logits` 取 contiguous k_fp8/k_scale。
- **真实实现来源**: 主要是 TokenSpeed Python layout 逻辑，quant kernel 来自 `per_token_group_quant_fp8`。
- **OpenInfer landing**: GLM5.2 model crate 的 DSA index-K cache type，外加 `glm52.dsa.index_k_pack` / `glm52.dsa.index_k_gather`。
- **Shape/layout contract**: DeepGEMM `fp8_paged_mqa_logits` 需要每个 page 先放 `page_size * head_dim` 的 FP8 bytes，再放 `page_size * num_groups` 的 f32 scales，不是 per-token interleave。
- **是否可直接搬**: 值得抄 layout，不建议只做裸 kernel；cache type 要编码 page/block-split 不变量。

### Decode deterministic top-k

- **Source tag**: FlashInfer Python API plus existing OpenInfer FlashInfer wrapper pattern。
- **TokenSpeed caller**: `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/flashinfer/dsa_topk.py` 的 `deterministic_decode_topk`。
- **TokenSpeed wrapper/import**: 它 import FlashInfer：

```python
from flashinfer import TopKTieBreak, top_k
```

然后调用：

```python
top_k(
    logits.contiguous(),
    int(topk),
    deterministic=True,
    tie_break=TopKTieBreak.SMALL,
    dsa_graph_safe=True,
)
```

- **真实实现来源**: FlashInfer `top_k`。OpenInfer 近邻代码是 `openinfer-kernels/csrc/shared/flashinfer_top1.cu`，已经用 FlashInfer `TopKDispatch` 做 top-1。
- **OpenInfer landing**: `glm52.indexer.deterministic_topk`。
- **Shape/layout contract**: 输入是 pre-masked logits，输出 int32 local offsets；支持 K=512/1024/2048；必须保留 `TopKTieBreak.SMALL` 和 graph-safe 语义。
- **是否可直接搬**: 可基于现有 OpenInfer FlashInfer TopKDispatch wrapper 扩展。不要用 TRTLLM decode top-k，因为 TokenSpeed 注释指出它 tie-break 不稳定。

### Prefill top-k select

- **Source tag**: TRTLLM `torch.ops` only, with TokenSpeed CUDA fallback reference。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/models/glm5.py`，prefill logits 出来后调用：

```python
torch.ops.trtllm.indexer_topk_prefill(
    logits,
    local_starts,
    causal_lens,
    workspace_indices,
    topk,
)
```

- **TokenSpeed wrapper/import**: TRTLLM wrapper 路径是 `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/trtllm/__init__.py`。
- **真实实现来源**: `torch.ops.trtllm.indexer_topk_prefill` 的源码不在 TokenSpeed 仓库；它来自 `tokenspeed-trtllm-kernel` wheel，当前 PyPI metadata 没有源码 URL。可参考 `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/deepseek_v4_topk.cu` 的 `deepseek_v4_indexer_topk_prefill` 做本地 CUDA replacement。
- **OpenInfer landing**: `glm52.indexer.prefill_topk`。
- **Shape/layout contract**: 输入 logits、row starts/local starts、causal lens、workspace output、top-k；输出 workspace-local top-k offsets。
- **是否可直接搬**: 优先写本地 CUDA replacement；只有本地实现不合适，再考虑 TRTLLM 窄 wrapper。

### top-k offset to KV slot

- **Source tag**: TokenSpeed Triton local kernel。
- **TokenSpeed caller**: `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/dsa_sparse_layout.py`。
- **TokenSpeed wrapper/import**: 函数是 `local_topk_to_global_slots` 和 `full_context_topk_to_global_slots`。
- **真实实现来源**: TokenSpeed 本地 Triton kernel。
- **OpenInfer landing**: `glm52.dsa.local_topk_to_slots` 和 `glm52.dsa.full_context_to_slots`。
- **Shape/layout contract**: 输入 block table、page size、本地 top-k offsets，输出全局 KV slot 和 `topk_lens`，buffer dtype 为 int32。
- **是否可直接搬**: 可以按语义搬成 CUDA。

### sparse KV pack

- **Source tag**: TokenSpeed Triton local kernel, SM90 FlashMLA-only path。
- **TokenSpeed caller**: `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/dsa_sparse_layout.py`。
- **TokenSpeed wrapper/import**: 函数是 `pack_sparse_decode_kv`。
- **真实实现来源**: TokenSpeed 本地 Triton kernel。
- **OpenInfer landing**: 只有保留 SM90 FlashMLA sparse path 时，才需要 `glm52.flashmla.pack_sparse_decode_kv`。
- **Shape/layout contract**: 把 BF16 NoPE/RoPE cache 打成 FlashMLA sparse packed row；当前 FlashMLA V32 row 是 656 bytes。
- **是否可直接搬**: 不能和 Blackwell TRTLLM sparse MLA 混用。Blackwell TRTLLM sparse MLA 用普通 MLA KV cache，不等价于当前 FlashMLA packed KV row。

### Blackwell sparse MLA

- **Source tag**: FlashInfer vendored Python API plus FlashInfer C++ launcher using TRTLLM-gen artifacts。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/dsa.py` 的 `_forward_sparse_prefill_trtllm` 和 `_forward_sparse_decode_trtllm`。
- **TokenSpeed wrapper/import**: 两者都调用 `trtllm_batch_decode_with_kv_cache_mla(..., backend="trtllm-gen", sparse_mla_top_k=self.index_topk)`。
- **真实实现来源**: FlashInfer Python API 在 `openinfer-kernels/third_party/flashinfer/flashinfer/mla/_core.py` 的 `trtllm_batch_decode_with_kv_cache_mla`；C++ launcher 在 `openinfer-kernels/third_party/flashinfer/csrc/trtllm_fmha_kernel_launcher.cu`，`trtllm_paged_attention_decode` 支持 `sparse_mla_top_k`，另外有 DSV4 sparse helper。vendored FlashInfer origin 是 `https://github.com/flashinfer-ai/flashinfer`。
- **OpenInfer landing**: `glm52.trtllm_mla.sparse_decode`，同一 ABI 覆盖 prefill-token rows 和 decode-token rows。
- **Shape/layout contract**: query `[tokens, 1, heads, 576]` 或等价 flatten；普通 MLA KV cache，不是 FlashMLA packed row；block table 在 sparse 模式下是 `[tokens, 1, topk]`；`seq_lens` 对 sparse table 来说应来自 `topk_lens`；输出 latent `[tokens, heads * v_head_dim]`。
- **是否可直接搬**: 值得基于 vendored FlashInfer 接窄 ABI；不是抄 TokenSpeed wrapper，也不是复用当前 `glm52.flashmla.sparse_decode`。接这个需要处理 FlashInfer TVM FFI/JIT artifact loader build glue。

### Dense FP8 block-scale GEMM

- **Source tag**: DeepGEMM vendored C++/CUDA。
- **TokenSpeed caller**: `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/deep_gemm.py` 的 `deep_gemm_mm_fp8_blockscale`。
- **TokenSpeed wrapper/import**: 底层调用 DeepGEMM `fp8_gemm_nt`。
- **真实实现来源**: `openinfer-kernels/third_party/DeepGEMM/csrc/apis/gemm.hpp`。vendored DeepGEMM origin 是 `https://github.com/deepseek-ai/DeepGEMM`；TokenSpeed 的 `tokenspeed-deepgemm` wheel metadata 没有源码 URL。
- **OpenInfer landing**: `glm52.deepgemm.fp8_gemm_nt`。
- **Shape/layout contract**: dense FP8 block-scale GEMM，先服务 GLM5.2 dense Linear / fused qkv-a。
- **是否可直接搬**: 可直接接 DeepGEMM；不要和 grouped MoE runner 放进同一个接口。

### FP8 MoE expert

- **Source tag**: FlashInfer fused MoE plus TRTLLM-gen artifacts。
- **TokenSpeed caller**: `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer/trtllm_fp8.py` 的 `flashinfer_trtllm_fp8_moe_apply`。
- **TokenSpeed wrapper/import**: 它先做 activation token-group FP8 quant，再调用 FlashInfer：

```python
from flashinfer.fused_moe import RoutingMethodType, trtllm_fp8_block_scale_moe
```

- **真实实现来源**: vendored FlashInfer 的相关 C++ 在 `openinfer-kernels/third_party/flashinfer/csrc/trtllm_fused_moe_*`，origin 是 `https://github.com/flashinfer-ai/flashinfer`。
- **OpenInfer landing**: `glm52.moe.fp8_expert`。
- **Shape/layout contract**: 需要 GLM5.2 MoE shape、EP/DeepEP 路径和 router contract 先明确；activation quant 是 token-group FP8。
- **是否可直接搬**: 不要搬整个 `moe_plan` registry；先接一个 GLM5.2 FP8 expert 主路径。

### MoE router/top-k

- **Source tag**: TokenSpeed model/router code plus scattered CUDA/Triton thirdparty kernels。
- **TokenSpeed caller**: `/data/code/tokenspeed/python/tokenspeed/runtime/layers/moe/topk.py` 和 `/data/code/tokenspeed/python/tokenspeed/runtime/models/deepseek_v3.py`。
- **TokenSpeed wrapper/import**: 相关入口包括 `grouped_topk_gpu`、`minimax_biased_grouped_topk`、`cuda_routing_flash` 和 `dsv3_router_gemm`。
- **真实实现来源**: 后端分散在 TokenSpeed thirdparty cuda/triton 里，必须按 GLM5.2 config 再确认。
- **OpenInfer landing**: `glm52.moe.router_topk`。
- **Shape/layout contract**: 需要确认 `n_group`、`topk_group`、`top_k`、`correction_bias`、zero expert 语义。
- **是否可直接搬**: 先等 GLM5.2 config 和 MoE backend 主路径定下来，不要提前搬所有 routing variant。

## Blackwell/TRTLLM 标注

TokenSpeed 在 Blackwell 上有几类算子切到了 TRTLLM/FlashInfer TRTLLM-gen：

- DSA sparse MLA：`DSABackend` 走 `trtllm_batch_decode_with_kv_cache_mla(..., backend="trtllm-gen", sparse_mla_top_k=index_topk)`。FlashInfer 的 Python API 在 SM100/SM103 auto 选择 `trtllm-gen`，SM120 可能走 `xqa`，但 TokenSpeed 这里显式指定了 `trtllm-gen`。
- FP8 token-group quant：TRTLLM wrapper 调 `torch.ops.trtllm.fp8_quantize_1x128`，只支持 group size 128。
- MoE FP8 expert：TokenSpeed 的 `flashinfer_trtllm_fp8_moe_apply` 调 FlashInfer `trtllm_fp8_block_scale_moe`，内部也属于 TRTLLM-gen fused MoE 路线。
- Prefill top-k：`torch.ops.trtllm.indexer_topk_prefill` 是 TRTLLM op；decode top-k 没用 TRTLLM，因为 TokenSpeed 注释里说明 TRTLLM decode top-k tie-break 不稳定，改用 FlashInfer deterministic top-k。

这意味着后续搬运时不要把 `TRTLLM` 当成一个大依赖一次性接入。优先按稳定 caller 接四个窄边界：sparse MLA、FP8 token-group quant、prefill top-k、FP8 MoE expert。每个边界都要确认是否能从 vendored FlashInfer/DeepGEMM 直接落到 C ABI；如果只能从 Python `torch.ops` 进，就先标成不可直接搬。

## 优先级索引

P0 是 DSA attention 正确性路径，按运行时依赖顺序搬：

1. `glm52.fp8.token_group_quant_128_f32`，见上面的 `DSA FP8 token-group quant`。
2. `glm52.dsa.index_k_pack/gather`，见 `DSA index-K cache set/gather`。
3. `glm52.deepgemm.mqa_logits` 和 `glm52.deepgemm.paged_mqa_logits`，见两个 DeepGEMM indexer logits 小节。
4. `glm52.indexer.deterministic_topk` 和 `glm52.indexer.prefill_topk`，见 top-k 两个小节。
5. `glm52.dsa.local_topk_to_slots/full_context_to_slots`，见 `top-k offset to KV slot`。
6. `glm52.trtllm_mla.sparse_decode`，见 `Blackwell sparse MLA`。

P1 是让 GLM5.2 跑得像 TokenSpeed 主路径的性能和 MoE 缺口：

1. `glm52.deepgemm.fp8_gemm_nt`，见 `Dense FP8 block-scale GEMM`。
2. `glm52.moe.fp8_expert`，见 `FP8 MoE expert`。
3. `glm52.moe.router_topk`，见 `MoE router/top-k`。
4. `glm52.indexer.hadamard_bf16`，见 `DSA Hadamard rotate`。
5. `glm52.flashmla.pack_sparse_decode_kv`，仅当保留 SM90 FlashMLA packed-row 路径时需要，见 `sparse KV pack`。

P2 是扩展路径：

1. speculative verify / NextN decode：支持 `q_len_per_req` 1..6 和多 token per request top-k。
2. full-context top-k fast path：上下文长度小于 top-k 时不跑 DeepGEMM logits，可和 slot conversion 一起做。

## 不建议现在直接搬

- TokenSpeed 的完整 `tokenspeed_kernel` registry。它解决的是 Python runtime 的动态选型；OpenInfer 更适合让 GLM5.2 model crate 先给稳定 shape，再暴露小而明确的 Rust wrapper。
- 整套 TRTLLM thirdparty ops。当前真正卡 GLM5.2 DSA 的是 sparse MLA、prefill top-k、FP8 quant/MoE 中的少数入口；这些应按 caller 收窄。Blackwell sparse MLA 优先看 vendored FlashInfer 的 TRTLLM-gen launcher，不从 TokenSpeed 的 `torch.ops` wrapper 开始。
- 多个 MoE backend 同时接入。GLM5.2 FP8 主路径先选一个 expert 后端，bench 以后再决定是否保留替代后端。

## 还缺的后续文档

- `docs/models/glm52/model-crate.md`：GLM5.2 模型 crate 的 config、weight layout、scheduler/executor 和 CUDA Graph 边界。
- `docs/models/glm52/dsa-attention.md`：DSA attention 的完整 prefill/decode DAG，包括 indexer top-k、index-K cache、sparse MLA 输入输出合同。
- `docs/models/glm52/moe-backend.md`：GLM5.2 MoE router/top-k、EP/DeepEP、expert backend 的选型和测量口径。
- `docs/subsystems/kernels/openinfer-kernels-boundary.md`：每次新增稳定 GLM5.2 kernel wrapper 后，把当前窄 surface 的说明同步更新。
- `openinfer-kernels/KERNELS.md`：新增任何 GLM5.2 op 时同步登记 op id、Rust wrapper、FFI symbol、source 和 shape/layout。

## Preparation

- **Read**:
  - `docs/index.md` - 没有 GLM5.2 模型线入口，需要新增。
  - `docs/subsystems/kernels/openinfer-kernels-boundary.md` - 说明 GLM5.2 当前只保留窄接口，router/indexer/TRTLLM/local route 等要等模型 crate 证明稳定调用者。
  - `openinfer-kernels/KERNELS.md` - 当前 GLM5.2 公开面只有 scale layout、grouped FP8 contract/metadata、FlashMLA sparse decode。
  - `openinfer-kernels/src/ops/glm52/*.rs` - grouped FP8 compute 仍是 fail-closed，FlashMLA sparse decode 固定 V32/topk=2048/SM90。
  - `/data/code/tokenspeed/python/tokenspeed/runtime/models/glm5.py` - GLM5.2 的 DSA indexer、DeepGEMM top-k logits、prefill/decode sparse attention 和 index-K cache 写入都在这里串起来。
  - `/data/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/dsa.py` - sparse prefill/decode 走 FlashInfer 的 TRTLLM MLA wrapper。
  - `/data/code/tokenspeed/python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py` - index-K FP8 cache 采用 DeepGEMM paged MQA logits 需要的 block-split layout。
  - `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/dsa_sparse_layout.py` - sparse KV pack 和 top-k offset 到 KV slot 的 Triton kernels。
  - `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/flashinfer/dsa_topk.py` - deterministic decode top-k wrapper。
  - `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/quantization/` - GLM5.2 路径使用 FP8 token-group quant，group size 128，scale 为 float32。
  - `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/fast_hadamard_transform/__init__.py` 与 `/data/code/tokenspeed/python/tokenspeed/env.py` - Hadamard 在 TokenSpeed 中只是外部包 wrapper，依赖名是 `tokenspeed-fast-hadamard-transform`。
  - `openinfer-kernels/third_party/DeepGEMM/csrc/apis/attention.hpp`、`openinfer-kernels/third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp` - DeepGEMM MQA logits 的真实入口。
  - `openinfer-kernels/third_party/flashinfer/flashinfer/mla/_core.py`、`openinfer-kernels/third_party/flashinfer/csrc/trtllm_fmha_kernel_launcher.cu` - Blackwell TRTLLM-gen sparse MLA 的 FlashInfer 入口和 C++ launcher。
  - `/data/code/tokenspeed/python/tokenspeed/runtime/layers/moe/` 与 `/data/code/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/` - GLM5.2 MoE 复用 DeepSeekV3MoE/TokenSpeed MoE plan，FP8 场景会落到 FlashInfer/TRTLLM FP8 MoE 或同类后端。
- **Relevant history**:
  - `openinfer-kernels/KERNELS.md` 已经声明当前 GLM5.2 surface 只收稳定 substrate，不提前搬整套 TokenSpeed/TRTLLM fallback。
  - `docs/subsystems/kernels/openinfer-kernels-boundary.md` 记录过相同边界：先把 DeepEP/DeepGEMM/FlashMLA 纳入 `moe` feature，再按 GLM5.2 真实 caller 暴露小接口。
- **Plan**:
  1. 对照 TokenSpeed 的 GLM5.2 DSA attention、indexer、MoE 调用链。
  2. 列出现有 OpenInfer 已暴露接口。
  3. 把尚未暴露的 kernel 按 P0/P1/P2 分类，并给出建议边界。
  4. 更新 `docs/index.md`，让 GLM5.2 文档能被路由到。

## Execution Log

### Step 1: 对照 TokenSpeed GLM5.2 调用链

- 读取了 TokenSpeed 的 `glm5.py`、DSA backend、DSA KV cache、sparse layout、deterministic top-k、FP8 quant、MoE layer 和 FP8 MoE backend。
- 结论：TokenSpeed 的 GLM5.2 不是只需要 DeepGEMM grouped MoE；DSA indexer logits、top-k、slot conversion、sparse MLA 和 index-K cache layout 是 attention 正确性路径上的第一批缺口。

### Step 2: 对照 OpenInfer 当前 GLM5.2 surface

- 读取了 `openinfer-kernels/KERNELS.md` 和 `openinfer-kernels/src/ops/glm52/*.rs`。
- 结论：OpenInfer 当前只暴露了 substrate 级的小面，不能覆盖 TokenSpeed GLM5.2 的完整 kernel DAG。

### Step 3: 写入文档和索引

- 新增本文件。
- 在 `docs/index.md` 新增 `models / glm52` 路由入口。

### Step 4: 补充搬运入口

- 增加逐 kernel 搬运入口，区分 TokenSpeed Python 调用点、真实实现来源、OpenInfer 推荐落点和是否能直接搬。
- 标注 `fast_hadamard_transform` 在 TokenSpeed 中只有 wrapper，真实实现来自外部 `tokenspeed-fast-hadamard-transform` 包。
- 标注 Blackwell sparse MLA 应优先参考 vendored FlashInfer 的 TRTLLM-gen 路径，而不是当前 OpenInfer 的 SM90 FlashMLA sparse decode。

### Step 5: 改成逐 kernel 讲解

- 移除文档本体里的表格，把 OpenInfer 当前公开面、搬运入口和缺口清单都改成逐项小节。
- Hadamard 项明确写出两级 import：GLM5.2 从 `tokenspeed_kernel.thirdparty.fast_hadamard_transform` import，TokenSpeed wrapper 再从外部 `fast_hadamard_transform` import。

### Step 6: 搬运者视角评估

- 起 sub-agent 扮演“准备搬 TokenSpeed GLM5.2 kernel 的工程师”，只评估文档友好程度，不做实际搬运。
- 评估结论：文档能逐 kernel 定位，但需要更像施工单；主要建议是固定字段、后移项目记录、把缺口清单改为优先级索引、补清 Blackwell sparse MLA 输入差异。
- 已按反馈调整：搬运入口统一成 `Source tag` / `TokenSpeed caller` / `TokenSpeed wrapper/import` / `真实实现来源` / `OpenInfer landing` / `Shape/layout contract` / `是否可直接搬`。

### Step 7: 外部源码核对

- clone `https://github.com/Dao-AILab/fast-hadamard-transform.git` 到 `/tmp/fast-hadamard-transform`，当前 HEAD 是 `e7706fa`。
- 确认 `fast_hadamard_transform` 的 Python API、C++ binding 和 CUDA launcher 分别在 `fast_hadamard_transform/fast_hadamard_transform_interface.py`、`csrc/fast_hadamard_transform.cpp`、`csrc/fast_hadamard_transform_cuda.cu`。
- 查 PyPI metadata：`tokenspeed-fast-hadamard-transform` 的 Homepage 指向 Dao-AILab 上游；`tokenspeed-deepgemm`、`tokenspeed-flashmla` 没有 homepage/project_urls；`tokenspeed-trtllm-kernel` 只说明是 TensorRT-LLM custom ops，license 是 Apache-2.0，但没有源码 URL。
- 查 OpenInfer vendored remotes：DeepGEMM 来自 `https://github.com/deepseek-ai/DeepGEMM`，FlashInfer 来自 `https://github.com/flashinfer-ai/flashinfer`，FlashMLA 来自 `https://github.com/deepseek-ai/FlashMLA`。
- 已补 `外部源码核对` 小节，并同步更新 Hadamard、TRTLLM quant/top-k、DeepGEMM、FlashInfer/FP8 MoE 的真实实现来源。

## Debrief

- **Outcome**: 已形成 TokenSpeed GLM5.2 kernel gap 清单，并按逐 kernel 施工单标出 TokenSpeed 调用点、真实实现来源、OpenInfer 落点、shape/layout contract 和搬运判断；外部来源已区分为公开 GitHub、OpenInfer vendored 源码和 wheel-only `torch.ops`。
- **Pitfalls encountered**:
  - 当前 OpenInfer 的 `glm52.flashmla.sparse_decode` 名字容易让人误以为覆盖 TokenSpeed DSA sparse attention；实际 TokenSpeed 用的是 TRTLLM sparse MLA wrapper，且包含 sparse prefill 与 `topk_lens` 语义。
  - DeepGEMM “需要”不是一个接口：GLM5.2 至少有 dense FP8 GEMM、prefill MQA logits、decode paged MQA logits、grouped MoE runner 四类调用，不能用一个 grouped FP8 wrapper 代表全部。
  - TokenSpeed 的 `tokenspeed_kernel.thirdparty.fast_hadamard_transform` 不是实现源码，只是外部包转发；真实源码在 Dao-AILab 上游。
  - `tokenspeed-trtllm-kernel`、`tokenspeed-deepgemm`、`tokenspeed-flashmla` 的公网 PyPI metadata 不给源码 URL，不能把 wheel 名当成源码路径。
- **Lessons learned**:
  - 下一步如果要让 GLM5.2 attention 先跑起来，第一批接口应围绕 DSA top-k 和 sparse MLA，而不是先扩满 MoE backend registry。
  - 标注第三方 kernel 时要同时写清“TokenSpeed import 名”和“可读源码路径”；两者经常不是同一件事。
- **Follow-ups**:
  - 实现前先补 `docs/models/glm52/dsa-attention.md`，把 q/index-K/logits/top-k/slots/sparse MLA 的 tensor shape 写清楚。
  - 后续每加一个 wrapper，都要同步 `openinfer-kernels/KERNELS.md`，避免接口已经存在但模型侧不知道怎么用。
