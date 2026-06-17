//! Centralized description of the kernel choices used by the Qwen3.5 crate.
//!
//! Mirrors the qwen3-4b `kernel_plan` module: a static [`KernelPlan`] data
//! structure that records which Rust function / FFI call / backend serves
//! each op, plus a [`kernel_plan()`] accessor that exposes it for dump / debug.
//!
//! **This module is purely descriptive. It does not change kernel selection —
//! every entry documents a call site that already exists in the crate. The
//! refactor goal is to make the active plan visible without reading call
//! sites in `batch_decode.rs`, `prefill.rs`, `recurrent.rs`, and friends.**
//!
//! **Backend tags follow the qwen3-4b convention:** a single-word or
//! compound label (CUDA, cuBLAS, FlashInfer, Triton AOT, etc.). Concrete
//! csrc file paths are not included — they belong in the kernel ledger
//! (see roadmap). The `rust` field names the exact function, which is
//! enough to grep the codebase for the actual implementation.
//!
//! Use it like:
//!
//! ```ignore
//! use openinfer_qwen35_4b::kernel_plan;
//! for phase in kernel_plan().phases {
//!     for op in phase.ops {
//!         println!("[{}] {} -> {}", phase.name, op.id, op.backend);
//!     }
//! }
//! ```

pub struct KernelPlan {
    pub model: &'static str,
    pub phases: &'static [KernelPhase],
}

pub struct KernelPhase {
    pub name: &'static str,
    pub ops: &'static [KernelOp],
}

pub struct KernelOp {
    pub id: &'static str,
    pub rust: &'static str,
    pub backend: &'static str,
    pub notes: &'static str,
}

pub static KERNEL_PLAN: KernelPlan = KernelPlan {
    model: "qwen35-4b",
    phases: &[
        // ── Prefill ─────────────────────────────────────────────────────────
        //
        // Full-attention layers: 8. Linear-attention layers: 24. Each layer
        // runs either full-attn or linear-attn ops (not both).
        //
        KernelPhase {
            name: "prefill",
            ops: &[
                // shared prefill prologue
                KernelOp {
                    id: "embedding_prefill",
                    rust: "prefill::prefill_last_hidden -> ops::embedding_batch",
                    backend: "CUDA",
                    notes: "prompt tokens to hidden states",
                },
                // full-attention prefill (8 layers)
                KernelOp {
                    id: "qkv_gemm_prefill_full",
                    rust: "prefill::prefill_full_attention -> ops::gemm (q/k/v_proj)",
                    backend: "cuBLAS",
                    notes: "3 cuBLAS GEMMs — Q, K, V projection",
                },
                KernelOp {
                    id: "qk_norm_partial_rope_prefill",
                    rust: "prefill::prefill_full_attention -> ffi::prefill_attention_hd256_prep_cuda",
                    backend: "CUDA",
                    notes: "Q/K RMSNorm + partial RoPE; head_dim=256",
                },
                KernelOp {
                    id: "paged_kv_scatter_prefill",
                    rust: "prefill::prefill_full_attention -> ffi::paged_kv_scatter_cuda",
                    backend: "CUDA",
                    notes: "scatter processed K/V from HND staging buffer into paged pool",
                },
                KernelOp {
                    id: "paged_prefill_attention",
                    rust: "prefill::prefill_full_attention -> ffi::batch_prefill_paged_cuda_hd256",
                    backend: "CUDA",
                    notes: "custom paged prefill attention, head_dim=256 — NOT FlashInfer",
                },
                KernelOp {
                    id: "attention_gate_prefill",
                    rust: "prefill::prefill_full_attention -> ffi::attention_gate_batch_hd256_cuda",
                    backend: "CUDA",
                    notes: "Q-gated attention output scaling",
                },
                KernelOp {
                    id: "o_proj_prefill_full",
                    rust: "prefill::prefill_full_attention -> ops::gemm (o_proj)",
                    backend: "cuBLAS",
                    notes: "attention output projection",
                },
                // linear-attention prefill (24 layers)
                KernelOp {
                    id: "linear_in_proj_prefill",
                    rust: "prefill::prefill_linear_attention -> ops::gemm (in_proj_qkv/z/b/a)",
                    backend: "cuBLAS",
                    notes: "4 cuBLAS GEMMs: in_proj_qkv (fuses q+k+v), z, b, a",
                },
                KernelOp {
                    id: "conv1d_prefill",
                    rust: "prefill::prefill_linear_attention -> ops::conv1d_prefill_batch_into",
                    backend: "CUDA",
                    notes: "causal depthwise conv1d over prefill sequence",
                },
                KernelOp {
                    id: "gated_delta_rule_prefill_chunkwise",
                    rust: "prefill::prefill_linear_attention -> ops::gated_delta_rule_prefill_chunkwise_into",
                    backend: "Triton AOT",
                    notes: "GDR chunkwise: prepare + cumsum + A + solve + recompute + state + O (7 Triton-AOT kernels, generated at build time)",
                },
                KernelOp {
                    id: "rms_norm_gated_prefill",
                    rust: "prefill::prefill_linear_attention -> ops::rms_norm_gated_batch_into",
                    backend: "CUDA",
                    notes: "z-gated custom RMSNorm on GDR output (in csrc/shared/norm.cu, NOT FlashInfer)",
                },
                KernelOp {
                    id: "out_proj_prefill_linear",
                    rust: "prefill::prefill_linear_attention -> ops::gemm (out_proj)",
                    backend: "cuBLAS",
                    notes: "linear-attention output projection",
                },
                // shared prefill epilogue (every layer)
                KernelOp {
                    id: "rms_norm_offset_prefill",
                    rust: "prefill::prefill_layer -> ops::rms_norm_batch_offset_into",
                    backend: "FlashInfer",
                    notes: "(1+w) GemmaRMSNorm — input + post-attention (via flashinfer::norm::GemmaRMSNorm)",
                },
                KernelOp {
                    id: "mlp_prefill",
                    rust: "prefill::prefill_layer -> ops::gemm (gate/up/down) + silu_mul_batch",
                    backend: "CUDA + cuBLAS",
                    notes: "SwiGLU MLP — 3 cuBLAS GEMMs (gate, up, down) + 1 CUDA silu_mul",
                },
                KernelOp {
                    id: "residual_add_prefill",
                    rust: "prefill::prefill_layer -> ops::add_batch",
                    backend: "CUDA",
                    notes: "residual connections (post-attn, post-mlp)",
                },
                // final norm + lm head (once per prefill step, not per request row)
                KernelOp {
                    id: "final_norm_prefill",
                    rust: "prefill::batch_last_hidden_logits -> ops::rms_norm_batch_offset_into",
                    backend: "FlashInfer",
                    notes: "batched final RMSNorm over gathered last hidden states (via flashinfer::norm::GemmaRMSNorm)",
                },
                KernelOp {
                    id: "lm_head_prefill",
                    rust: "prefill::batch_last_hidden_logits -> ops::gemm (tied embed_tokens)",
                    backend: "cuBLAS",
                    notes: "LM head using tied embeddings (one cuBLAS GEMM over request rows)",
                },
            ],
        },
        // ── Decode ──────────────────────────────────────────────────────────
        //
        // CUDA-Graph captured. 1 token per request, bucket-padded to the
        // nearest BATCH_BUCKET size. Recurrent state managed per slot.
        //
        KernelPhase {
            name: "decode",
            ops: &[
                // shared decode prologue
                KernelOp {
                    id: "embedding_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::embedding_batch",
                    backend: "CUDA",
                    notes: "one token per request; bucket-padded for CUDA Graph",
                },
                KernelOp {
                    id: "rms_norm_offset_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::rms_norm_batch_offset_into",
                    backend: "FlashInfer",
                    notes: "(1+w) GemmaRMSNorm per layer (via flashinfer::norm::GemmaRMSNorm)",
                },
                // full-attention decode (8 layers)
                KernelOp {
                    id: "qkv_gemm_decode_full",
                    rust: "batch_decode::batch_decode_full_attention -> ops::gemm_into (q/k/v_proj)",
                    backend: "cuBLAS",
                    notes: "3 cuBLAS GEMMs — Q, K, V projection over bucket-padded batch",
                },
                KernelOp {
                    id: "qk_norm_partial_rope_decode",
                    rust: "batch_decode::batch_decode_full_attention -> ops::qk_norm_partial_rope_batched_decode_hd256_into",
                    backend: "CUDA",
                    notes: "Q/K RMSNorm + partial RoPE, head_dim=256",
                },
                KernelOp {
                    id: "paged_decode_attention",
                    rust: "batch_decode::batch_decode_full_attention -> ops::paged_attention_batch_decode_hd256_into",
                    backend: "CUDA",
                    notes: "wraps paged_kv_scatter + paged_attention_decode — custom CUDA, NOT FlashInfer",
                },
                KernelOp {
                    id: "attention_gate_decode",
                    rust: "batch_decode::batch_decode_full_attention -> ffi::attention_gate_batch_hd256_cuda",
                    backend: "CUDA",
                    notes: "Q-gated attention output scaling",
                },
                KernelOp {
                    id: "o_proj_decode_full",
                    rust: "batch_decode::batch_decode_full_attention -> ops::gemm_into (o_proj)",
                    backend: "cuBLAS",
                    notes: "attention output projection",
                },
                // linear-attention decode (24 layers)
                KernelOp {
                    id: "linear_in_proj_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::gemm_into (in_proj_qkv/z/b/a)",
                    backend: "cuBLAS",
                    notes: "4 cuBLAS GEMMs: in_proj_qkv (fuses q+k+v), z, b, a",
                },
                KernelOp {
                    id: "conv1d_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::conv1d_decode_into",
                    backend: "CUDA",
                    notes: "depthwise conv1d on per-slot recurrent conv state",
                },
                KernelOp {
                    id: "gated_delta_rule_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::gated_delta_rule_decode_vec_into",
                    backend: "CUDA",
                    notes: "GDR per-slot decode — fixed-budget recurrent state update",
                },
                KernelOp {
                    id: "rms_norm_gated_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::rms_norm_gated_batch_into",
                    backend: "CUDA",
                    notes: "z-gated custom RMSNorm on GDR output (in csrc/shared/norm.cu, NOT FlashInfer)",
                },
                KernelOp {
                    id: "out_proj_decode_linear",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::gemm_into (out_proj)",
                    backend: "cuBLAS",
                    notes: "linear-attention output projection",
                },
                // shared decode epilogue (every layer)
                KernelOp {
                    id: "residual_add_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::add_batch_into",
                    backend: "CUDA",
                    notes: "residual connections (post-attn, post-mlp)",
                },
                KernelOp {
                    id: "mlp_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::gemm_into (gate/up/down) + silu_mul_batch_into",
                    backend: "CUDA + cuBLAS",
                    notes: "SwiGLU MLP — 3 cuBLAS GEMMs (gate, up, down) + 1 CUDA silu_mul",
                },
                // final norm + sampling (once per decode step, not per layer)
                KernelOp {
                    id: "final_norm_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::rms_norm_batch_offset_into",
                    backend: "FlashInfer",
                    notes: "final RMSNorm on logits hidden state (via flashinfer::norm::GemmaRMSNorm)",
                },
                KernelOp {
                    id: "lm_head_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::gemm_into (tied embed_tokens)",
                    backend: "cuBLAS",
                    notes: "LM head using tied embeddings (one cuBLAS GEMM over the bucket-padded batch)",
                },
                // batched sampling
                KernelOp {
                    id: "sampling_decode",
                    rust: "batch_decode::select_tokens_batch_varied -> openinfer_sample::select_batch",
                    backend: "FlashInfer + CUDA",
                    notes: "greedy rows use indexed batched argmax; non-greedy rows use one batched FlashInfer sampling pass per step",
                },
            ],
        },
        // ── Unified ─────────────────────────────────────────────────────────
        //
        // Mixed prefill + decode step called by the scheduler.
        //
        KernelPhase {
            name: "unified",
            ops: &[
                KernelOp {
                    id: "mixed_prefill_decode",
                    rust: "unified_forward::unified_step",
                    backend: "CUDA + cuBLAS + FlashInfer + Triton AOT",
                    notes: "scheduler step combining new prefill requests and active CUDA-Graph decode requests",
                },
                KernelOp {
                    id: "decode_sampling",
                    rust: "scheduler::process_decode_logits -> batch_decode::select_tokens_batch_varied -> openinfer_sample::select_batch",
                    backend: "CUDA + FlashInfer",
                    notes: "logprobs snapshot only extracts requested rows; token selection samples directly from batched logits",
                },
            ],
        },
    ],
};

pub fn kernel_plan() -> &'static KernelPlan {
    &KERNEL_PLAN
}
