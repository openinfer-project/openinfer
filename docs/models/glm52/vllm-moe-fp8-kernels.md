# GLM5.2 vLLM MoE FP8 Kernel Map

> **TL;DR:** vLLM has three GLM5.2-relevant FP8 MoE routes: FlashInfer CUTLASS fused MoE, DeepGEMM grouped FP8 MoE, and vLLM's own CUTLASS 3.x grouped GEMM. vLLM's automatic Hopper+EP policy prefers FlashInfer CUTLASS for block-FP8, but that path is a Python/TVM-FFI/JIT package API. DeepGEMM gives the best shape/layout oracle for current GLM expanded DeepEP work. vLLM CUTLASS is the closest raw CUDA/C++ fallback source, but its public wrapper is `torch::stable::Tensor` based and allocates pointer arrays/workspace per call, so OpenInfer needs a fixed-arena raw-pointer port before it can enter decode CUDA Graph. The FlashInfer/TRTLLM `GroupedWithOffset` scale-row blocker is substrate-covered in GLM (`11232` padded scale rows), and the vendored TRTLLM raw runner now has a GLM fixed-shape C ABI/workspace gate (`workspace_bytes=0` for W13/W2), real W13/W2 launch smoke from arena buffers, and fixed-bucket decode CUDA Graph capture/replay evidence. H200 NCU/perf evidence and full-layer integration are still pending.
>
> **Last touched:** 2026-06

## Source Snapshot

| Source | Revision | Files read for this pass |
| --- | --- | --- |
| vLLM sibling | `../vllm` at `4d3b4b9b01efbca77872e3d4a568b273c7a245a7` | `fused_moe/oracle/fp8.py`, `experts/{cutlass_moe.py,deep_gemm_moe.py,batched_deep_gemm_moe.py,flashinfer_cutlass_moe.py,trtllm_fp8_moe.py,triton_cutlass_moe.py}`, `deep_gemm_utils.py`, `utils/deep_gemm.py`, `csrc/libtorch_stable/quantization/w8a8/cutlass/*` |
| FlashInfer submodule | `openinfer-kernels/third_party/flashinfer` at `d768c14e7cf5dd5df45a8a1de78ae815879f108a` | `flashinfer/fused_moe/core.py`, `flashinfer/jit/fused_moe.py`, `csrc/fused_moe/cutlass_backend/*`, `csrc/trtllm_fused_moe_*` |
| DeepGEMM submodule | `openinfer-kernels/third_party/DeepGEMM` at `54e22612409371d6364144b69086735beb54e98b` | public grouped FP8 APIs and SM90 grouped GEMM runtime/JIT boundary from the prior GLM52 pass |

GLM52 first runnable target remains decode-only `DP8 TP1 EP8` on H200, with expanded DeepEP `psum_expert` layout, real `bs > 1`, fixed-bucket decode graph, no prefill work inside the GLM engine, and no NCCL all-to-all backend.

## vLLM Backend Policy

For FP8 W8A8 MoE, vLLM's backend oracle starts with this priority list:

| Order | Backend enum | Kernel class |
| ---: | --- | --- |
| 1 | `AITER` | ROCm only for our purposes |
| 2 | `FLASHINFER_TRTLLM` | `TrtLlmFp8Experts*`; SM100-family for FP8 block path |
| 3 | `FLASHINFER_CUTLASS` | `FlashInferExperts` |
| 4 | `DEEPGEMM` | `TritonOrDeepGemmExperts` |
| 5 | `VLLM_CUTLASS` | `TritonOrCutlassExperts` |
| 6 | `TRITON` | `TritonExperts` |
| later | `MARLIN`, batched variants, XPU, CPU | not first GLM H200 route |

Important policy details:

| vLLM rule | GLM52 implication |
| --- | --- |
| On Hopper (`SM90`) with block-FP8 and EP, vLLM moves `FLASHINFER_CUTLASS` to the front. | FlashInfer is not absent. If we do not use it, the reason must be ABI/layout/graph integration, not lack of a kernel. |
| On Hopper block-FP8 without EP, vLLM moves `TRITON` to the front. | Not our first topology; GLM is `DP8 TP1 EP8`. |
| `VLLM_CUTLASS` is removed unless `allow_vllm_cutlass=true` or explicitly requested and allowed. | vLLM's own CUTLASS path is source material, not vLLM's default production choice for our shape. |
| Explicit `--moe-backend deep_gemm` maps to DeepGEMM; batched activation format maps to `BATCHED_DEEPGEMM`. | The backend name alone is not enough; the prepare/finalize activation format decides contiguous vs batched expert layout. |

## Backend Map

| Route | Main files | What it does | Portability to OpenInfer |
| --- | --- | --- | --- |
| FlashInfer CUTLASS fused MoE | vLLM `flashinfer_cutlass_moe.py`; FlashInfer `flashinfer/fused_moe/core.py`; FlashInfer `csrc/fused_moe/cutlass_backend/*` | One fused expert API over routing ids/weights, W13/W2, optional FP8 block scales, TP/EP rank metadata, and output. On SM90 block-FP8 it sets `use_deepseek_fp8_block_scale=true`. | Strong candidate but not drop-in: the exposed path is Python -> FlashInfer JIT -> TVM FFI module -> TensorView runner. Need a raw C/C++ wrapper, fixed workspace ownership, and exact GLM weight/scale layout proof. |
| FlashInfer TRTLLM FP8 | vLLM `trtllm_fp8_moe.py`; FlashInfer `trtllm_fp8_block_scale_*` | TRTLLM-Gen routed MoE. Supports DeepSeek FP8 block scale and MXFP8 variants. | Deferred for H200 GLM: vLLM marks current FP8 modular support as Blackwell-family. Useful for API semantics and future SM100. |
| DeepGEMM standard grouped FP8 | vLLM `deep_gemm_moe.py`, `deep_gemm_utils.py`, `utils/deep_gemm.py`; DeepGEMM `m_grouped_fp8_gemm_nt_contiguous` | Permute/pack activations into a DeepGEMM contiguous layout, run W13 and W2 grouped FP8 GEMMs under `mk_alignment_scope(align_used)`, then gather/reduce with route weights. | Best layout oracle for current GLM package plan. Public vLLM route is Python + DeepGEMM Python extension/JIT; raw runtime wrapper still needed. Route-weight placement differs from current GLM W2-input weighting. |
| DeepGEMM batched grouped FP8 | vLLM `batched_deep_gemm_moe.py`; DeepGEMM `fp8_m_grouped_gemm_nt_masked` | Expert-major `[E,T,H]` format with `expert_num_tokens`, masked grouped GEMMs, and persistent masked SiLU quant. | Possibly closer to DeepEP batched P/F layouts. Current GLM arena is flat expanded `psum_expert`, so adopting this would be a layout change or a new view over the arena. |
| vLLM CUTLASS 3.x grouped GEMM | vLLM `cutlass_moe.py`; `grouped_mm_c3x_sm90.cu`; `grouped_mm_c3x.cuh`; `moe_data.cu`; `get_group_starts.cuh` | Two explicit grouped GEMM calls with activation quant between them. Standard path uses `moe_permute`/`moe_unpermute`; batched path uses `expert_num_tokens` and padded expert rows. | Closest CUDA/C++ source to adapt into OpenInfer, but wrapper is Torch Stable ABI and dynamically allocates pointer arrays/workspace. Need raw pointer arrays in arena and direct CUTLASS launch. |
| vLLM Triton MoE fallback | vLLM `triton_moe.py`, `fused_moe.py` | Triton fused MoE kernels and fallback logic. | Reference only. If we rewrite a Triton-only gap in CUDA, it is local CUDA and needs NCU. |

## vLLM CUTLASS Details

The vLLM CUTLASS path is worth keeping as the C++ adaptation reference because it is not a Python-only kernel.

| Piece | Source | Contract |
| --- | --- | --- |
| SM90 tile selection | `grouped_mm_c3x_sm90.cu` | FP8 E4M3 grouped GEMM uses CUTLASS 3.x `GemmUniversal` with SM90 schedules. It switches configs by total `M`: `M<=4`, `M<=64`, `N>=8192`, `K>=8192`, otherwise default. |
| Small-M `swap_ab` | `grouped_mm_c3x_sm90.cu`, `moe_data.cu` | For `M<=64`, vLLM swaps logical A/B to reduce padding. The problem-size generator must match the dispatch threshold. |
| Problem sizes from offsets | `moe_data.cu::get_cutlass_moe_mm_problem_sizes_from_expert_offsets_caller` | Given `expert_first_token_offset`, it emits W13 problem sizes `[M,2N,K]` and W2 `[M,K,N]`, with swapped variants when `swap_ab=true`. |
| Batched expert data | `moe_data.cu::get_cutlass_batched_moe_mm_data_caller` | Given `expert_num_tokens` and `padded_m`, emits `expert_offsets[e]=e*padded_m` plus W13/W2 problem sizes. |
| Pointer arrays | `get_group_starts.cuh` | Builds per-expert A/B/D/scale pointers from `expert_offsets`, base tensors, and per-token/per-output scale flags. |
| Torch wrapper | `grouped_mm_c3x.cuh::cutlass_group_gemm_caller` | Allocates `a_ptrs`, `b_ptrs`, `out_ptrs`, scale pointer arrays, and CUTLASS workspace via `torch::stable::empty` every call. |
| Support gate | `scaled_mm_entry.cu::cutlass_group_gemm_supported` | SM90 needs CUDA >= 12.3 and the SM90 CUTLASS MoE objects compiled in; SM100 needs CUDA >= 12.8 and its compile flag. |

Raw-port checklist for OpenInfer:

| Requirement | Why |
| --- | --- |
| Replace `torch::stable::Tensor` with raw pointers, dims, strides, dtype enums, and stream. | `openinfer-kernels` must stay Rust/C ABI friendly and not link libtorch. |
| Preallocate pointer arrays and CUTLASS workspace in `Glm52DecodeArena`. | Decode CUDA Graph requires stable addresses and no per-launch allocator. |
| Pick either flat contiguous or batched expert format explicitly. | vLLM has both; mixing problem-size generation from one with buffer layout from another will silently corrupt expert rows. |
| Encode W13/W2 shapes in types: W13 `[32,4096,6144]`, W2 `[32,6144,2048]`, block scales `[32,32,48]` and `[32,48,16]`. | Avoid runtime string/shape guesses in the kernel path. |
| Decide route-weight placement before porting unpermute. | vLLM CUTLASS standard path weights in `moe_unpermute`; current GLM expanded layout weights during W2-input quant so DeepEP combine reduces already weighted rows. |
| Run H200 microbench/NCU before calling it better than DeepGEMM/FlashInfer. | vLLM's source choice is a hint, not proof in this repo. |

## DeepGEMM Details

DeepGEMM is the current GLM layout oracle, but the public path is not a small C ABI call.

| Piece | Source | Contract |
| --- | --- | --- |
| Standard FP8 path | `deep_gemm_moe.py::DeepGemmExperts.apply` | `deepgemm_moe_permute` -> W13 `m_grouped_fp8_gemm_nt_contiguous` -> SiLU/quant -> W2 `m_grouped_fp8_gemm_nt_contiguous` -> `deepgemm_unpermute_and_reduce`. |
| Alignment guard | `utils/deep_gemm.py::mk_alignment_scope` | Caps DeepGEMM's grouped-contiguous `BLOCK_M` heuristic to the workspace alignment; vLLM says mismatch can pick wrong expert id under CUDA Graph replay. |
| Standard route weights | `deep_gemm_utils.py::deepgemm_unpermute_and_reduce` | Applies route weights while gathering/reducing W2 outputs back to token rows. This is not the current GLM expanded-DeepEP contract. |
| Batched path | `batched_deep_gemm_moe.py::BatchedDeepGemmExperts.apply` | Uses `[E,T,H]`, `expert_num_tokens`, `fp8_m_grouped_gemm_nt_masked`, and persistent masked SiLU quant. |
| Python API boundary | `utils/deep_gemm.py` | Dynamically imports `deep_gemm`, sets JIT cache paths, resolves Python-callable symbols, and wraps scale-format decisions. |

OpenInfer decision for the next MoE GEMM step:

| Decision | Rationale |
| --- | --- |
| Keep DeepGEMM grouped FP8 as the first GLM H200 shape target unless FlashInfer raw wrapper lands first. | GLM already has W13/W2 expert-major packages, `psum_expert` slice validation, activation quant, and F32 scale layout matching DeepGEMM. |
| Do not link DeepGEMM's PyTorch extension into `openinfer-kernels`. | That would pull the runtime shape away from pure Rust + CUDA and break the current kernel boundary. |
| If porting DeepGEMM raw JIT is too large, compare against vLLM CUTLASS raw port. | vLLM CUTLASS has more self-contained CUTLASS source, though still Torch-wrapped today. |

## FlashInfer Details

FlashInfer is not missing. There are two relevant families in the vendored submodule and in vLLM's wrappers.

| Route | Source | Fit |
| --- | --- | --- |
| `cutlass_fused_moe` | FlashInfer `flashinfer/fused_moe/core.py`, `csrc/fused_moe/cutlass_backend/*`; vLLM `flashinfer_cutlass_moe.py` | vLLM's preferred Hopper+EP block-FP8 route. It supports `use_deepseek_fp8_block_scale` on `SM90` with CUDA >= 12.8 and handles route weights inside the fused MoE API. |
| `trtllm_fp8_block_scale_routed_moe` | FlashInfer `flashinfer/fused_moe/core.py`, `csrc/trtllm_fused_moe_*`; vLLM `trtllm_fp8_moe.py` | More relevant to Blackwell in vLLM's current modular FP8 path; keep as future reference. |
| FlashInfer Cutlass JIT | FlashInfer `flashinfer/jit/fused_moe.py` | Builds a TVM FFI module from TensorRT-LLM/CUTLASS sources plus generated kernels. This is usable source, but not a ready raw C ABI. |
| FlashInfer/TRTLLM `GroupedWithOffset` | FlashInfer `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/*`, `deep_gemm/{scheduler.cuh,fp8_gemm.cuh}` | Raw pointer + `cudaStream_t` candidate for W13/W2 grouped FP8. `problem_m_offsets` fit the existing `expert_offsets`; activation scales are indexed by `compute_padded_offset(offset, problem_idx)` and need `11232` rows for GLM's fixed `10240`-row bucket. GLM now has a persistent scale relayout substrate for that padded space and a fixed-shape C ABI over `CutlassFp8BlockScaleGemmRunner<fp8,fp8,bf16>`; startup validates W13/W2 runner workspace initialization at `0` bytes, launches real W13/W2 from arena buffers, validates nonzero W13/W2/combined outputs, and captures/replays the same MoE substrate in a fixed-bucket decode CUDA Graph. |

Why GLM52 is not immediately using FlashInfer fused MoE:

| Blocker | Evidence |
| --- | --- |
| API boundary | vLLM calls a Python FlashInfer function; FlashInfer internally JIT-builds a TVM FFI module and exposes TensorView-based runners. |
| Workspace ownership | FlashInfer runner owns tuning/workspace behavior; OpenInfer decode graph needs persistent arena ownership and predictable address stability. |
| Layout proof | GLM currently has DeepGEMM-style expert-major packages and expanded DeepEP `psum_expert`; FlashInfer fused MoE expects its own weight transforms and route/finalize contract. |
| GroupedWithOffset scale layout | The raw TRTLLM grouped-offset path indexes activation scales through an extra 32-row padded offset formula. GLM now owns separate persistent `11232`-row W13/W2 activation-scale buffers and a CUDA relayout kernel validated in startup and decode graph replay; the raw-runner ABI/workspace gate also lands with `workspace_bytes=0` for FP8+FP8 W13/W2. Actual runner launch inside the MoE decode substrate and fixed-bucket graph capture is proven; full layer integration and performance profiling remain open. |
| Measurement | No H200 OpenInfer A/B has been run. vLLM policy alone is not performance evidence here. |

## Current GLM52 Direction

For the immediate routed expert GEMM implementation, use this order:

1. Keep the current TRTLLM `GroupedWithOffset` path as the first running W13/W2 substrate while collecting correctness/perf evidence.
2. Compare it against a DeepGEMM grouped FP8 raw wrapper only after the same GLM package plan, weighted W2 input, F32 scale layout, and expanded `psum_expert` layout can be consumed without Python/PyTorch runtime dependencies.
3. If DeepGEMM raw boundary is too large, adapt vLLM CUTLASS 3.x grouped GEMM into an arena-backed raw C ABI and benchmark it on the same GLM W13/W2 shapes.
4. Keep FlashInfer CUTLASS fused MoE as a serious candidate, but only after mapping its weight transforms, TVM FFI/raw wrapper boundary, and graph workspace ownership.
5. Treat Triton-only vLLM pieces as source contracts; any CUDA rewrite needs NCU evidence and a note explaining why FlashInfer/DeepGEMM/vLLM CUTLASS could not be used.

Do not implement routed expert compute as host loops over experts, tokens, or requests. Empty ranks and empty experts are legal, but all ranks must keep a graph-stable kernel sequence.

## Open Questions

| Question | Next evidence |
| --- | --- |
| DeepGEMM contiguous vs masked/batched layout for current `psum_expert` arena? | Prototype the smallest raw wrapper or harness and compare pointer/layout requirements before integrating with the rank worker. |
| FlashInfer CUTLASS weight transform for GLM FP8 block scales? | Trace `prepare_fp8_moe_layer_for_fi` against GLM W13/W2 package order and FlashInfer `use_deepseek_fp8_block_scale` expectations. |
| FlashInfer/TRTLLM GroupedWithOffset fallback? | Scale relayout landed and is profiled in `profile/glm52_trtllm_offset_scale_20260626/`: W13-scale shape is `22.46-23.04us` on H200, not DRAM-bound. The raw TRTLLM runner ABI/workspace gate passes node38 startup for W13/W2 with `workspace_bytes=0`; actual W13 then W2 launch from arena buffers and fixed-bucket decode graph capture now pass. Next evidence is H200 NCU/perf data and full-layer correctness. |
| vLLM CUTLASS raw-port workspace size and graph capture behavior? | Build a micro-harness with preallocated pointer arrays/workspace, then run H200 NCU and graph capture smoke. |
| Route-weight placement after real W2 GEMM? | Keep current weighted W2-input quant unless logits gate proves the contract wrong; changing to vLLM gather/reduce or FlashInfer fused weighting is a layout change. |
