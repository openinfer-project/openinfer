//! NCCL all-reduce microbenchmark for the openinfer TP=2 bring-up path.
//!
//! Reference data below was measured on a dual-GPU PCIe 4.0 setup
//! (`4090-3`, 2x RTX 4090, NCCL 2.28.3, BF16, in-place all-reduce):
//!
//! - 4KB: 21.65-21.72 us, ~180 MiB/s
//! - 16KB: 22.35-22.43 us, ~698 MiB/s
//! - 64KB: 29.33-29.48 us, ~2.08 GiB/s
//! - 256KB: 54.00-54.50 us, ~4.52 GiB/s
//! - 1MB: 101.17-101.38 us, ~9.65 GiB/s
//! - 4MB: 313.45-313.77 us, ~12.46 GiB/s
//! - 16MB: 1.139-1.144 ms, ~13.7 GiB/s
//! - 64MB: 4.480-4.492 ms, ~13.95 GiB/s
//!
//! These numbers are intended as a rough PCIe 4.0 reference point for future
//! TP communication budgeting, not as a universal NCCL baseline.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cudarc::nccl::{
    ReduceOp,
    safe::{Comm, group_end, group_start},
};
use half::bf16;
use openinfer::tensor::DeviceContext;

const PAYLOAD_BYTES: &[usize] = &[
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
];

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);
}

fn bench_nccl_all_reduce(c: &mut Criterion) {
    let ctx0 = DeviceContext::new_with_device(0).expect("failed to create CUDA context on GPU0");
    let ctx1 = DeviceContext::new_with_device(1).expect("failed to create CUDA context on GPU1");
    let comms = Comm::from_devices(vec![ctx0.stream.clone(), ctx1.stream.clone()])
        .expect("failed to create NCCL communicators for two GPUs");
    assert_eq!(comms.len(), 2, "expected one communicator per GPU");

    let mut group = c.benchmark_group("nccl_all_reduce_in_place_bf16_tp2");
    configure_group(&mut group);

    for &payload_bytes in PAYLOAD_BYTES {
        let element_count = payload_bytes / std::mem::size_of::<bf16>();
        let zeros = vec![bf16::from_f32(0.0); element_count];
        let mut buf0 = ctx0
            .stream
            .clone_htod(&zeros)
            .expect("failed to allocate GPU0 benchmark buffer");
        let mut buf1 = ctx1
            .stream
            .clone_htod(&zeros)
            .expect("failed to allocate GPU1 benchmark buffer");

        // Warm communicators and streams outside Criterion measurement.
        all_reduce_pair(&comms, &mut buf0, &mut buf1).expect("NCCL warmup failed");
        ctx0.sync().expect("GPU0 warmup sync failed");
        ctx1.sync().expect("GPU1 warmup sync failed");

        group.throughput(Throughput::Bytes(payload_bytes as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format_bytes(payload_bytes)),
            |b| {
                b.iter(|| {
                    all_reduce_pair(&comms, &mut buf0, &mut buf1).expect("NCCL all-reduce failed");
                    ctx0.sync().expect("GPU0 sync failed");
                    ctx1.sync().expect("GPU1 sync failed");
                    black_box((&buf0, &buf1));
                });
            },
        );
    }

    group.finish();
}

fn all_reduce_pair(
    comms: &[Comm],
    buf0: &mut cudarc::driver::CudaSlice<bf16>,
    buf1: &mut cudarc::driver::CudaSlice<bf16>,
) -> Result<(), cudarc::nccl::result::NcclError> {
    group_start()?;
    comms[0].all_reduce_in_place(buf0, &ReduceOp::Sum)?;
    comms[1].all_reduce_in_place(buf1, &ReduceOp::Sum)?;
    group_end()?;
    Ok(())
}

fn format_bytes(bytes: usize) -> String {
    match bytes {
        b if b >= 1024 * 1024 => format!("{}MB", b / 1024 / 1024),
        b if b >= 1024 => format!("{}KB", b / 1024),
        b => format!("{b}B"),
    }
}

criterion_group!(benches, bench_nccl_all_reduce);
criterion_main!(benches);
