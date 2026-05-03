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
    model: "qwen3-4b",
    phases: &[
        KernelPhase {
            name: "prefill",
            ops: &[
                KernelOp {
                    id: "embedding_batch",
                    rust: "prefill::get_embeddings_batch -> ops::embedding_batch",
                    backend: "CUDA",
                    notes: "prompt tokens to hidden states",
                },
                KernelOp {
                    id: "qkv_gemm",
                    rust: "prefill::process_layer_batch -> ops::gemm_into",
                    backend: "cuBLAS",
                    notes: "fused q/k/v projection",
                },
                KernelOp {
                    id: "paged_prefill_attention",
                    rust: "prefill::process_layer_batch -> ops::prefill_paged_attention",
                    backend: "FlashInfer",
                    notes: "full attention with paged KV write",
                },
                KernelOp {
                    id: "mlp",
                    rust: "prefill::process_layer_batch -> ops::silu_mul_batch + gemm_into",
                    backend: "CUDA + cuBLAS",
                    notes: "gate/up activation and down projection",
                },
            ],
        },
        KernelPhase {
            name: "decode",
            ops: &[
                KernelOp {
                    id: "embedding_decode",
                    rust: "batch_decode::batch_decode_kernels -> ops::embedding_batch",
                    backend: "CUDA",
                    notes: "one token per request, bucket padded for CUDA graph",
                },
                KernelOp {
                    id: "paged_decode_attention",
                    rust: "batch_decode::batch_decode_layer -> ops::batch_decode_with_paged_kv",
                    backend: "FlashInfer",
                    notes: "paged KV read with per-request CSR metadata",
                },
                KernelOp {
                    id: "sampling",
                    rust: "executor::LocalQwen3Lane::sample_from_logits -> ops::gpu_sample_into",
                    backend: "FlashInfer/CUDA",
                    notes: "greedy and sampling token selection",
                },
            ],
        },
        KernelPhase {
            name: "unified",
            ops: &[KernelOp {
                id: "mixed_prefill_decode",
                rust: "unified_forward::unified_step",
                backend: "CUDA + cuBLAS + FlashInfer",
                notes: "scheduler step combining new prefill requests and active decode requests",
            }],
        },
    ],
};

pub fn kernel_plan() -> &'static KernelPlan {
    &KERNEL_PLAN
}
