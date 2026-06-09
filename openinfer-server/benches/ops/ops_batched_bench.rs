use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use openinfer::ops;
use openinfer::tensor::DeviceContext;

use super::common::{
    BATCH_SEQ_LEN, OUT_DIM, VECTOR_DIM, configure_group, device_matrix, hidden_states, iter_sync,
};

pub(crate) fn bench_batched_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_batched");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(
        (OUT_DIM * VECTOR_DIM * BATCH_SEQ_LEN) as u64,
    ));
    group.bench_function(BenchmarkId::new("gemm", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let gemm_weight =
            device_matrix(&ctx, OUT_DIM, VECTOR_DIM).expect("failed to allocate gemm weight");
        let gemm_x =
            hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate gemm input");
        iter_sync(b, &ctx, || {
            let out = ops::gemm(&ctx, &gemm_weight, &gemm_x).expect("gemm failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("add_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let add_a =
            hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate add lhs");
        let add_b =
            hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate add rhs");
        iter_sync(b, &ctx, || {
            let out = ops::add_batch(&ctx, &add_a, &add_b).expect("add_batch failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("silu_mul_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let gate = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate gate");
        let up = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate up");
        iter_sync(b, &ctx, || {
            let out = ops::silu_mul_batch(&ctx, &gate, &up).expect("silu_mul_batch failed");
            black_box(out);
        });
    });

    group.finish();
}
