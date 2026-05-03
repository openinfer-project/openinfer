//! E2E test for paged attention decode path.
//!
//! Prefills a BOS token to create KV state, then exercises the production
//! executor decode phase backed by paged KV cache + FlashInfer decode kernels.

use pegainfer_core::sampler::SamplingParams;
use pegainfer_qwen3_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen3Executor, RequestId,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen3-4B");

fn get_model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn greedy_ignore_eos() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        ignore_eos: true,
    }
}

fn prefill_one(
    executor: &mut Qwen3Executor,
    request_id: RequestId,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    rng: &mut StdRng,
) -> u32 {
    let requests = [PrefillStepItem::new(
        request_id,
        prompt_tokens,
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

fn decode_one(
    executor: &mut Qwen3Executor,
    request_id: RequestId,
    token_id: u32,
    params: SamplingParams,
    rng: &mut StdRng,
) -> u32 {
    let requests = [DecodeStepItem::new(
        request_id,
        token_id,
        params,
        0,
        rng.random(),
    )];
    let result = executor
        .execute_decode(DecodePlan {
            requests: &requests,
        })
        .expect("decode failed");
    result.requests[0].token
}

fn generate_from_bos(executor: &mut Qwen3Executor, request_id: RequestId, seed: u64) -> Vec<u32> {
    let params = greedy_ignore_eos();
    let mut rng = StdRng::seed_from_u64(seed);
    let bos_token = 151_643_u32;
    let mut tokens = vec![prefill_one(
        executor,
        request_id,
        vec![bos_token],
        params,
        &mut rng,
    )];

    for _ in 0..9 {
        let next = decode_one(
            executor,
            request_id,
            *tokens.last().unwrap(),
            params,
            &mut rng,
        );
        tokens.push(next);
    }
    tokens
}

#[test]
fn paged_attention_single_token_decode() {
    let model_path = get_model_path();
    let mut executor =
        Qwen3Executor::from_runtime(&model_path, true, &[0]).expect("Failed to load executor");
    let request_id = RequestId::new(0);
    let tokens = generate_from_bos(&mut executor, request_id, 42);
    eprintln!("Generated 10 tokens from BOS: {:?}", tokens);

    // Basic sanity: should produce non-zero, valid tokens
    for &t in &tokens {
        assert!(t < 151_936, "Token {t} out of vocab range");
    }

    // Verify determinism: reset and regenerate with same seed
    executor.drop_request(request_id).expect("drop failed");
    let request_id_again = RequestId::new(1);
    let tokens_again = generate_from_bos(&mut executor, request_id_again, 42);
    assert_eq!(
        tokens, tokens_again,
        "Paged attention decode is not deterministic"
    );
    executor
        .drop_request(request_id_again)
        .expect("drop failed");

    eprintln!("Determinism check passed");
}
