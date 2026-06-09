use criterion::{BenchmarkId, Criterion, Throughput};
use openinfer::ops;
use openinfer::tensor::{DeviceContext, DeviceVec};

use super::common::{
    EPS, OUT_DIM, VECTOR_DIM, configure_group, device_matrix, device_vec, iter_sync,
    positive_device_vec,
};

pub(crate) fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_elementwise");
    configure_group(&mut group);

    let gemv_shapes = [
        ("legacy_1024x1024", OUT_DIM, VECTOR_DIM),
        ("wide_8192x2560", 8192, 2560),
        ("proj_4096x2560", 4096, 2560),
        ("small_32x2560", 32, 2560),
    ];

    for (label, rows, cols) in gemv_shapes {
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("gemv", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let matrix = device_matrix(&ctx, rows, cols).expect("failed to allocate matrix");
            let x = device_vec(&ctx, cols).expect("failed to allocate x");
            let mut gemv_out = DeviceVec::zeros(&ctx, rows).expect("failed to allocate gemv out");
            iter_sync(b, &ctx, || {
                ops::gemv(&ctx, &matrix, &x, &mut gemv_out).expect("gemv failed");
            });
        });
    }
    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("rms_norm_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let rms_x = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate rms x");
        let rms_weight =
            positive_device_vec(&ctx, VECTOR_DIM).expect("failed to allocate rms weight");
        let mut rms_out = DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate rms out");
        iter_sync(b, &ctx, || {
            ops::rms_norm_into(&ctx, &rms_x, &rms_weight, EPS, &mut rms_out)
                .expect("rms_norm_into failed");
        });
    });

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(
        BenchmarkId::new("fused_add_rms_norm_into", VECTOR_DIM),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let mut hidden = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate hidden");
            let residual = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate residual");
            let fused_weight =
                positive_device_vec(&ctx, VECTOR_DIM).expect("failed to allocate fused weight");
            let mut fused_out =
                DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate fused out");
            iter_sync(b, &ctx, || {
                ops::fused_add_rms_norm_into(
                    &ctx,
                    &mut hidden,
                    &residual,
                    &fused_weight,
                    EPS,
                    &mut fused_out,
                )
                .expect("fused_add_rms_norm_into failed");
            });
        },
    );

    group.finish();
}
