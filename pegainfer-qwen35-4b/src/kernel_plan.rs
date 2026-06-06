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
//! Use it like:
//!
//! ```ignore
//! use pegainfer_qwen35_4b::kernel_plan;
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
        KernelPhase {
            name: "prefill",
            ops: &[
                // ── shared prefill prologue ───────────────────────────────────
                KernelOp {
                    id: "embedding_prefill",
                    rust: "prefill::prefill_forward -> ops::embedding_batch",
                    backend: "CUDA",
                    notes: "prompt tokens to hidden states",
                },
                // ── full-attention prefill (8 layers) ────────────────────────
                KernelOp {
                    id: "qkv_gemm_prefill_full",
                    rust: "prefill::prefill_full_attention -> ops::gemm (q/k/v_proj)",
                    backend: "cuBLAS",
                    notes: "fused Q/K/V projection",
                },
                KernelOp {
                    id: "qk_norm_partial_rope_prefill",
                    rust: "prefill::prefill_full_attention -> ffi::prefill_attention_hd256_prep_cuda",
                    backend: "CUDA (csrc/qwen35/prefill_attention_hd256.cu)",
                    notes: "Q/K RMSNorm + partial RoPE; head_dim=256",
                },
                KernelOp {
                    id: "paged_kv_scatter_prefill",
                    rust: "prefill::prefill_full_attention -> ffi::paged_kv_scatter_cuda",
                    backend: "CUDA (csrc/shared/paged_attention.cu)",
                    notes: "scatter processed K/V from HND staging buffer into paged pool",
                },
                KernelOp {
                    id: "paged_prefill_attention",
                    rust: "prefill::prefill_full_attention -> ffi::batch_prefill_paged_cuda_hd256",
                    backend: "CUDA (csrc/shared/paged_attention.cu)",
                    notes: "custom paged prefill attention, head_dim=256 (NOT FlashInfer)",
                },
                KernelOp {
                    id: "attention_gate_prefill",
                    rust: "prefill::prefill_full_attention -> ffi::attention_gate_batch_hd256_cuda",
                    backend: "CUDA (csrc/qwen35/prefill_attention_hd256.cu)",
                    notes: "Q-gated attention output scaling",
                },
                KernelOp {
                    id: "o_proj_prefill_full",
                    rust: "prefill::prefill_full_attention -> ops::gemm (o_proj)",
                    backend: "cuBLAS",
                    notes: "attention output projection",
                },
                // ── linear-attention prefill (24 layers) ─────────────────────
                KernelOp {
                    id: "in_proj_qkvzab_prefill",
                    rust: "prefill::prefill_linear_attention -> ops::gemm (in_proj_qkv/z/b/a)",
                    backend: "cuBLAS",
                    notes: "fused 5-way linear-attention input projection",
                },
                KernelOp {
                    id: "conv1d_prefill",
                    rust: "prefill::prefill_linear_attention -> ops::conv1d_prefill_batch_into",
                    backend: "CUDA (csrc/conv1d.cu)",
                    notes: "causal depthwise conv1d over prefill sequence",
                },
                KernelOp {
                    id: "gated_delta_rule_prefill_chunkwise",
                    rust: "prefill::prefill_linear_attention -> ops::gated_delta_rule_prefill_chunkwise_into",
                    backend: "Triton AOT (tools/triton/gated_delta_rule_chunkwise_kernels.py)",
                    notes: "GDR chunkwise: prepare + cumsum + A + solve + recompute + state + O",
                },
                KernelOp {
                    id: "rms_norm_gated_prefill",
                    rust: "prefill::prefill_linear_attention -> ops::rms_norm_gated_batch_into",
                    backend: "CUDA",
                    notes: "z-gated RMSNorm on GDR output",
                },
                KernelOp {
                    id: "out_proj_prefill_linear",
                    rust: "prefill::prefill_linear_attention -> ops::gemm (out_proj)",
                    backend: "cuBLAS",
                    notes: "linear-attention output projection",
                },
                // ── shared prefill epilogue ──────────────────────────────────
                KernelOp {
                    id: "rms_norm_offset_prefill",
                    rust: "prefill::prefill_layer -> ops::rms_norm_batch_offset_into",
                    backend: "CUDA",
                    notes: "(1+w) RMSNorm — input + post-attention",
                },
                KernelOp {
                    id: "mlp_prefill",
                    rust: "prefill::prefill_layer -> ops::gemm (gate/up/down) + silu_mul_batch",
                    backend: "CUDA + cuBLAS",
                    notes: "SwiGLU MLP — gate, up, silu*mul, down",
                },
                KernelOp {
                    id: "residual_add_prefill",
                    rust: "prefill::prefill_layer -> ops::add_batch",
                    backend: "CUDA",
                    notes: "residual connections (post-attn, post-mlp)",
                },
                KernelOp {
                    id: "final_norm_prefill",
                    rust: "prefill::prefill_forward -> ops::rms_norm_offset_into",
                    backend: "CUDA",
                    notes: "final RMSNorm on last hidden state",
                },
                KernelOp {
                    id: "lm_head_prefill",
                    rust: "prefill::prefill_forward -> ops::linear (tied embed_tokens)",
                    backend: "cuBLAS",
                    notes: "LM head using tied embeddings",
                },
            ],
        },
        KernelPhase {
            name: "decode",
            ops: &[
                // ── shared decode prologue (CUDA Graph captured) ─────────────
                KernelOp {
                    id: "embedding_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::embedding_batch",
                    backend: "CUDA",
                    notes: "one token per request; bucket-padded for CUDA Graph",
                },
                KernelOp {
                    id: "rms_norm_offset_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::rms_norm_batch_offset_into",
                    backend: "CUDA",
                    notes: "(1+w) RMSNorm on hidden state per layer",
                },
                // ── full-attention decode (8 layers) ─────────────────────────
                KernelOp {
                    id: "qkv_gemm_decode_full",
                    rust: "batch_decode::batch_decode_full_attention -> ops::gemm_into (q/k/v_proj)",
                    backend: "cuBLAS",
                    notes: "fused Q/K/V projection over bucket-padded decode batch",
                },
                KernelOp {
                    id: "qk_norm_partial_rope_decode",
                    rust: "batch_decode::batch_decode_full_attention -> ops::qk_norm_partial_rope_batched_decode_hd256_into",
                    backend: "CUDA (csrc/qwen35/prefill_attention_hd256.cu)",
                    notes: "Q/K RMSNorm + partial RoPE, head_dim=256",
                },
                KernelOp {
                    id: "paged_decode_attention",
                    rust: "batch_decode::batch_decode_full_attention -> ops::paged_attention_batch_decode_hd256_into",
                    backend: "CUDA (csrc/shared/paged_attention.cu)",
                    notes: "wraps paged_kv_scatter + paged_attention_decode (both in shared/paged_attention.cu) — NOT FlashInfer; custom kernels only",
                },
                KernelOp {
                    id: "attention_gate_decode",
                    rust: "batch_decode::batch_decode_full_attention -> ffi::attention_gate_batch_hd256_cuda",
                    backend: "CUDA (csrc/qwen35/prefill_attention_hd256.cu)",
                    notes: "Q-gated attention output scaling",
                },
                KernelOp {
                    id: "o_proj_decode_full",
                    rust: "batch_decode::batch_decode_full_attention -> ops::gemm_into (o_proj)",
                    backend: "cuBLAS",
                    notes: "attention output projection",
                },
                // ── linear-attention decode (24 layers) ──────────────────────
                KernelOp {
                    id: "in_proj_qkvzab_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::gemm_into (in_proj_qkv/z/b/a)",
                    backend: "cuBLAS",
                    notes: "5-way input projection over the bucket-padded decode batch",
                },
                KernelOp {
                    id: "conv1d_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::conv1d_decode_into",
                    backend: "CUDA (csrc/conv1d.cu)",
                    notes: "depthwise conv1d on per-slot recurrent conv state",
                },
                KernelOp {
                    id: "gated_delta_rule_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::gated_delta_rule_decode_vec_into",
                    backend: "CUDA (csrc/gated_delta_rule_decode.cu)",
                    notes: "GDR per-slot decode — fixed-budget recurrent state update",
                },
                KernelOp {
                    id: "rms_norm_gated_decode",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::rms_norm_gated_batch_into",
                    backend: "CUDA",
                    notes: "z-gated RMSNorm on GDR output",
                },
                KernelOp {
                    id: "out_proj_decode_linear",
                    rust: "batch_decode::batch_decode_linear_attention_slots -> ops::gemm_into (out_proj)",
                    backend: "cuBLAS",
                    notes: "linear-attention output projection",
                },
                // ── shared decode epilogue ──────────────────────────────────
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
                    notes: "SwiGLU MLP — gate, up, silu*mul, down",
                },
                KernelOp {
                    id: "final_norm_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::rms_norm_batch_offset_into",
                    backend: "CUDA",
                    notes: "final RMSNorm on logits hidden state",
                },
                KernelOp {
                    id: "lm_head_decode",
                    rust: "batch_decode::batch_decode_kernels_graph -> ops::gemm_into (tied embed_tokens)",
                    backend: "cuBLAS",
                    notes: "LM head using tied embeddings",
                },
                // ── per-request sampling ─────────────────────────────────────
                KernelOp {
                    id: "sampling_decode",
                    rust: "batch_decode::select_tokens_batch_varied -> ops::gpu_sample_into",
                    backend: "FlashInfer/CUDA",
                    notes: "greedy / top-k / top-p token selection (one per request)",
                },
            ],
        },
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
                    id: "extract_logits",
                    rust: "unified_forward::unified_step / executor::execute_decode -> ops::extract_vec",
                    backend: "CUDA",
                    notes: "extract per-request logits from the batched logits buffer",
                },
            ],
        },
    ],
};

pub fn kernel_plan() -> &'static KernelPlan {
    &KERNEL_PLAN
}
