mod ops;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    ops::ops_elementwise_bench::bench_elementwise_ops,
    ops::ops_embedding_sampling_bench::bench_embedding_sampling_ops,
    ops::ops_batched_bench::bench_batched_ops
);
criterion_main!(benches);
