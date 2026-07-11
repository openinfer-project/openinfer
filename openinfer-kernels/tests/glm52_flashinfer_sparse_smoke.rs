//! Focused numerical gate for the GLM5.2 TP4 FlashInfer sparse MLA wrapper.
//!
//! The query is exactly zero and every selected FP8 value element is exactly
//! one, so softmax is uniform and every BF16 output element must be one.

#![cfg(feature = "glm52")]

use openinfer_kernels::ops::{
    GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES, Glm52FlashInferSparseDecode,
    glm52_flashinfer_sparse_mla_fp8_launch, glm52_flashinfer_sparse_mla_supported,
};
use openinfer_kernels::tensor::DeviceContext;

#[test]
fn glm52_flashinfer_sparse_fp8_uniform_value() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    if !glm52_flashinfer_sparse_mla_supported(16).expect("query FlashInfer support") {
        eprintln!("skip: GLM5.2 FlashInfer sparse MLA requires SM100/SM103");
        return;
    }

    for topk_size in [256usize, 2048] {
        for batch_size in [1usize, 2, 4, 8] {
            let contract = Glm52FlashInferSparseDecode {
                batch_size,
                heads: 16,
                num_blocks: topk_size / 64,
                topk: topk_size,
                sm_scale: 0.0625,
            };
            // E4M3 encodings: zero = 0x00, one = 0x38.
            let query = ctx
                .stream
                .clone_htod(&vec![0x00u8; contract.query_len()])
                .expect("query H2D");
            let cache = ctx
                .stream
                .clone_htod(&vec![0x38u8; contract.cache_len()])
                .expect("cache H2D");
            let topk_host: Vec<i32> = (0..batch_size).flat_map(|_| 0..topk_size as i32).collect();
            let topk = ctx.stream.clone_htod(&topk_host).expect("topk H2D");
            let seq_lens = ctx
                .stream
                .clone_htod(&vec![topk_size as i32; batch_size])
                .expect("seq_lens H2D");
            let mut out = ctx
                .stream
                .alloc_zeros(contract.output_len())
                .expect("output alloc");
            let mut workspace = ctx
                .stream
                .alloc_zeros::<u8>(GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES)
                .expect("workspace alloc");

            glm52_flashinfer_sparse_mla_fp8_launch(
                &ctx,
                contract,
                &query,
                &cache,
                &topk,
                &seq_lens,
                &mut out,
                &mut workspace,
            )
            .expect("FlashInfer sparse MLA launch");
            let host = ctx.stream.clone_dtoh(&out).expect("output D2H");
            ctx.sync().expect("CUDA sync");
            let max_error = host
                .iter()
                .map(|value| (value.to_f32() - 1.0).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 0.01,
                "batch {batch_size} topk {topk_size} max error {max_error}"
            );
        }
    }
}
