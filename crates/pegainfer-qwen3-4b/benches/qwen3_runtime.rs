use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pegainfer_core::sampler::SamplingParams;
use pegainfer_qwen3_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen3Executor, RequestId,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen3-4B");
const PREFILL_LENGTHS: &[usize] = &[128, 512, 1024, 2048, 4096, 10_000];
const DECODE_CONTEXT_LEN: usize = 1024;
const DECODE_BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32];

fn model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn synthetic_prompt(seq_len: usize) -> Vec<u32> {
    (0..seq_len).map(|i| ((i % 1000) + 100) as u32).collect()
}

fn greedy_ignore_eos() -> SamplingParams {
    SamplingParams {
        ignore_eos: true,
        ..Default::default()
    }
}

fn next_request_id(next_id: &mut u64) -> RequestId {
    let request_id = RequestId::new(*next_id);
    *next_id += 1;
    request_id
}

fn prefill_one(
    executor: &mut Qwen3Executor,
    request_id: RequestId,
    prompt: &[u32],
    params: SamplingParams,
    rng: &mut StdRng,
) -> u32 {
    let requests = [PrefillStepItem::new(
        request_id,
        prompt.to_vec(),
        params,
        0,
        false,
        rng.random(),
    )];
    let result = executor
        .execute_prefill(PrefillPlan {
            requests: &requests,
            echo: false,
        })
        .expect("prefill failed");
    result.requests[0].first_token
}

fn decode_one_step(
    executor: &mut Qwen3Executor,
    request_ids: &[RequestId],
    tokens: &mut [u32],
    params: SamplingParams,
    rng: &mut StdRng,
) {
    let requests: Vec<_> = request_ids
        .iter()
        .zip(tokens.iter())
        .map(|(&request_id, &token_id)| {
            DecodeStepItem::new(request_id, token_id, params, 0, rng.random())
        })
        .collect();
    let result = executor
        .execute_decode(DecodePlan {
            requests: &requests,
        })
        .expect("decode failed");
    for (slot, request) in tokens.iter_mut().zip(result.requests) {
        *slot = request.token;
    }
}

fn bench_prefill_ttft(c: &mut Criterion) {
    let path = model_path();
    let mut executor =
        Qwen3Executor::from_runtime(&path, true, &[0]).expect("failed to load Qwen3 executor");
    let params = greedy_ignore_eos();
    let mut rng = StdRng::seed_from_u64(42);
    let mut next_id = 0u64;
    let prompts: Vec<_> = PREFILL_LENGTHS
        .iter()
        .map(|&seq_len| (seq_len, synthetic_prompt(seq_len)))
        .collect();

    let mut group = c.benchmark_group("qwen3_executor_prefill_ttft");
    for (seq_len, prompt) in &prompts {
        group.throughput(Throughput::Elements(*seq_len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), prompt, |b, prompt| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let request_id = next_request_id(&mut next_id);
                    let start = Instant::now();
                    let token = prefill_one(&mut executor, request_id, prompt, params, &mut rng);
                    total += start.elapsed();
                    black_box(token);
                    executor
                        .drop_request(request_id)
                        .expect("drop request failed");
                }
                total
            });
        });
    }
    group.finish();
}

fn bench_decode_tpot(c: &mut Criterion) {
    let path = model_path();
    let mut executor =
        Qwen3Executor::from_runtime(&path, true, &[0]).expect("failed to load Qwen3 executor");
    let params = greedy_ignore_eos();
    let mut rng = StdRng::seed_from_u64(42);
    let mut next_id = 0u64;
    let prompt = synthetic_prompt(DECODE_CONTEXT_LEN);

    let mut group = c.benchmark_group("qwen3_executor_decode_tpot");
    for &batch_size in DECODE_BATCH_SIZES {
        group.throughput(Throughput::Elements(batch_size as u64));
        let request_ids: Vec<_> = (0..batch_size)
            .map(|_| next_request_id(&mut next_id))
            .collect();
        let mut tokens: Vec<_> = request_ids
            .iter()
            .map(|&request_id| prefill_one(&mut executor, request_id, &prompt, params, &mut rng))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        decode_one_step(&mut executor, &request_ids, &mut tokens, params, &mut rng);
                        total += start.elapsed();
                        black_box(&tokens);
                    }
                    total
                });
            },
        );

        for request_id in request_ids {
            executor
                .drop_request(request_id)
                .expect("drop request failed");
        }
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10))
}

criterion_group! {
    name = qwen3_runtime;
    config = criterion_config();
    targets = bench_prefill_ttft, bench_decode_tpot
}
criterion_main!(qwen3_runtime);
