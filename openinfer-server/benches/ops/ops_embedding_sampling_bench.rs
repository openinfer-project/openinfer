use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer::ops;
use openinfer::sampler::SamplingParams;
use openinfer::tensor::{DeviceContext, DeviceVec, HiddenStates};

use super::common::{
    BATCH_SEQ_LEN, VECTOR_DIM, VOCAB_SIZE, configure_group, decode_token_id, embedding_matrix,
    iter_sync, token_ids,
};

pub(crate) fn bench_embedding_sampling_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_embedding_sampling");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("embedding_decode_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let embed = embedding_matrix(&ctx, VOCAB_SIZE, VECTOR_DIM)
            .expect("failed to allocate embedding matrix");
        let token_id = 17_u32;
        let mut embed_out =
            DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate embedding out");
        let decode_token = decode_token_id(&ctx, token_id).expect("failed to allocate token id");
        iter_sync(b, &ctx, || {
            ops::embedding_decode_into(&ctx, &embed, &decode_token, &mut embed_out)
                .expect("embedding_decode_into failed");
        });
    });

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("embedding_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let embed = embedding_matrix(&ctx, VOCAB_SIZE, VECTOR_DIM)
            .expect("failed to allocate embedding matrix");
        let token_ids_gpu =
            token_ids(&ctx, BATCH_SEQ_LEN, VOCAB_SIZE).expect("failed to allocate token ids");
        let mut embed_batch_out = HiddenStates::zeros(&ctx, VECTOR_DIM, BATCH_SEQ_LEN)
            .expect("failed to allocate batched embedding out");
        iter_sync(b, &ctx, || {
            ops::embedding_batch(&ctx, &embed, &token_ids_gpu, &mut embed_batch_out)
                .expect("embedding_batch failed");
        });
    });

    let sampling_shapes = [
        (
            "top_k_top_p",
            128_256usize,
            SamplingParams {
                temperature: 0.8,
                top_k: 50,
                top_p: 0.95,
                ..Default::default()
            },
        ),
        (
            "top_p_only",
            128_256usize,
            SamplingParams {
                temperature: 0.8,
                top_k: -1,
                top_p: 0.9,
                ..Default::default()
            },
        ),
        (
            "top_k_top_p",
            248_320usize,
            SamplingParams {
                temperature: 0.8,
                top_k: 50,
                top_p: 0.95,
                ..Default::default()
            },
        ),
        (
            "top_p_only",
            248_320usize,
            SamplingParams {
                temperature: 0.8,
                top_k: -1,
                top_p: 0.9,
                ..Default::default()
            },
        ),
    ];
    for (label, vocab_size, params) in sampling_shapes {
        group.throughput(Throughput::Elements(vocab_size as u64));
        group.bench_function(
            BenchmarkId::new(format!("gpu_sample_into/{label}"), vocab_size),
            |b| {
                let ctx = DeviceContext::new().expect("failed to create CUDA context");
                let host_logits: Vec<bf16> = (0..vocab_size)
                    .map(|idx| {
                        let bucket = (idx % 1024) as f32 / 1024.0;
                        bf16::from_f32((bucket * 6.0) - 3.0)
                    })
                    .collect();
                let logits =
                    DeviceVec::from_host(&ctx, &host_logits).expect("failed to allocate logits");
                let mut probs: CudaSlice<f32> = ctx
                    .stream
                    .alloc_zeros(vocab_size)
                    .expect("failed to allocate probs scratch");
                let mut top1_value: CudaSlice<bf16> = ctx
                    .stream
                    .alloc_zeros(1)
                    .expect("failed to allocate top1 value scratch");
                let mut row_states: CudaSlice<u8> = ctx
                    .stream
                    .alloc_zeros(openinfer::ops::flashinfer_topk_row_states_bytes())
                    .expect("failed to allocate row state scratch");
                let mut valid: CudaSlice<u8> = ctx
                    .stream
                    .alloc_zeros(1)
                    .expect("failed to allocate valid scratch");
                let mut out_gpu: CudaSlice<i32> = ctx
                    .stream
                    .alloc_zeros(1)
                    .expect("failed to allocate output token");
                iter_sync(b, &ctx, || {
                    let token = ops::gpu_sample_into(
                        &ctx,
                        &logits,
                        &mut probs,
                        &mut top1_value,
                        &mut row_states,
                        &mut valid,
                        &mut out_gpu,
                        &params,
                        0.37,
                    )
                    .expect("gpu_sample_into failed");
                    black_box(token);
                });
            },
        );
    }

    let greedy_shapes = [128_256usize, 248_320usize];
    for vocab_size in greedy_shapes {
        group.throughput(Throughput::Elements(vocab_size as u64));
        group.bench_function(BenchmarkId::new("argmax", vocab_size), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let host_logits: Vec<bf16> = (0..vocab_size)
                .map(|idx| {
                    let bucket = (idx % 1024) as f32 / 1024.0;
                    bf16::from_f32((bucket * 6.0) - 3.0)
                })
                .collect();
            let logits =
                DeviceVec::from_host(&ctx, &host_logits).expect("failed to allocate logits");
            iter_sync(b, &ctx, || {
                let token = ops::argmax(&ctx, &logits).expect("argmax failed");
                black_box(token);
            });
        });

        group.bench_function(
            BenchmarkId::new("flashinfer_greedy_top_k_1", vocab_size),
            |b| {
                let ctx = DeviceContext::new().expect("failed to create CUDA context");
                let host_logits: Vec<bf16> = (0..vocab_size)
                    .map(|idx| {
                        let bucket = (idx % 1024) as f32 / 1024.0;
                        bf16::from_f32((bucket * 6.0) - 3.0)
                    })
                    .collect();
                let logits =
                    DeviceVec::from_host(&ctx, &host_logits).expect("failed to allocate logits");
                let mut probs: CudaSlice<f32> = ctx
                    .stream
                    .alloc_zeros(vocab_size)
                    .expect("failed to allocate probs scratch");
                let mut top1_value: CudaSlice<bf16> = ctx
                    .stream
                    .alloc_zeros(1)
                    .expect("failed to allocate top1 value scratch");
                let mut row_states: CudaSlice<u8> = ctx
                    .stream
                    .alloc_zeros(openinfer::ops::flashinfer_topk_row_states_bytes())
                    .expect("failed to allocate row state scratch");
                let mut valid: CudaSlice<u8> = ctx
                    .stream
                    .alloc_zeros(1)
                    .expect("failed to allocate valid scratch");
                let mut out_gpu: CudaSlice<i32> = ctx
                    .stream
                    .alloc_zeros(1)
                    .expect("failed to allocate output token");
                let params = SamplingParams {
                    temperature: 1.0,
                    top_k: 1,
                    top_p: 1.0,
                    ..Default::default()
                };
                iter_sync(b, &ctx, || {
                    let token = ops::gpu_sample_into(
                        &ctx,
                        &logits,
                        &mut probs,
                        &mut top1_value,
                        &mut row_states,
                        &mut valid,
                        &mut out_gpu,
                        &params,
                        0.37,
                    )
                    .expect("flashinfer greedy top_k=1 failed");
                    black_box(token);
                });
            },
        );
    }

    group.finish();
}
