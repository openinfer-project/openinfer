use std::path::Path;

use openinfer_qwen35_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen35Executor, RequestId,
    VerifiedToken, VerifyPlan, VerifyStepItem,
};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const LOGPROBS: usize = 1;

#[derive(Clone)]
struct CaseSpec {
    request_id: RequestId,
    prompt_tokens: Vec<u32>,
    draft_len: usize,
    reject_at: Option<usize>,
}

#[derive(Clone)]
struct CaseExpectation {
    request_id: RequestId,
    first_token: u32,
    draft_tokens: Vec<u32>,
    expected_matched: usize,
    expected_accepted: Vec<VerifiedToken>,
    followup_token: u32,
}

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 speculative_verify: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn build_executor(model_path: &str, capacity: usize) -> Qwen35Executor {
    let capacity = [1usize, 2, 4, 8, 16, 32, 64]
        .into_iter()
        .find(|bucket| *bucket >= capacity)
        .expect("test batch exceeds Qwen3.5 decode bucket capacity");
    Qwen35Executor::from_runtime_with_capacity(model_path, false, &[0], capacity)
        .expect("load Qwen3.5 executor")
}

fn prefill(exec: &mut Qwen35Executor, cases: &[CaseSpec]) -> Vec<u32> {
    let reqs: Vec<_> = cases
        .iter()
        .map(|case| PrefillStepItem::new(case.request_id, case.prompt_tokens.clone(), LOGPROBS))
        .collect();
    exec.execute_prefill(PrefillPlan { requests: &reqs })
        .expect("prefill")
        .requests
        .into_iter()
        .map(|result| result.first_token)
        .collect()
}

fn decode_once(exec: &mut Qwen35Executor, tokens: &[u32], cases: &[CaseSpec]) -> Vec<u32> {
    let reqs: Vec<_> = cases
        .iter()
        .zip(tokens.iter())
        .map(|(case, &token)| DecodeStepItem::new(case.request_id, token, LOGPROBS))
        .collect();
    exec.execute_decode(DecodePlan { requests: &reqs })
        .expect("decode")
        .requests
        .into_iter()
        .map(|result| result.token)
        .collect()
}

fn build_expectations(model_path: &str, cases: &[CaseSpec]) -> Vec<CaseExpectation> {
    let mut exec = build_executor(model_path, cases.len());
    let first_tokens = prefill(&mut exec, cases);
    let max_len = cases
        .iter()
        .map(|case| case.draft_len + 3)
        .max()
        .expect("at least one case");
    let mut generated: Vec<Vec<u32>> = first_tokens.into_iter().map(|token| vec![token]).collect();
    while generated.iter().any(|tokens| tokens.len() < max_len) {
        let fed: Vec<u32> = generated
            .iter()
            .map(|tokens| *tokens.last().expect("prefill token"))
            .collect();
        for (tokens, next) in generated
            .iter_mut()
            .zip(decode_once(&mut exec, &fed, cases).into_iter())
        {
            tokens.push(next);
        }
    }

    cases
        .iter()
        .zip(generated.iter())
        .map(|(case, generated)| {
            let first = generated[0];

            let mut draft_tokens = generated[1..=case.draft_len].to_vec();
            let mut expected_matched = case.draft_len;
            if let Some(reject_at) = case.reject_at {
                draft_tokens[reject_at] = draft_tokens[reject_at].wrapping_add(17);
                expected_matched = reject_at;
            }
            let mut accepted_ids = draft_tokens[..expected_matched].to_vec();
            accepted_ids.push(generated[expected_matched + 1]);

            let mut expected_accepted = Vec::with_capacity(accepted_ids.len());
            for expected in &accepted_ids {
                expected_accepted.push(VerifiedToken {
                    token: *expected,
                    logprob: None,
                });
            }
            let followup_token = generated[expected_matched + 2];

            CaseExpectation {
                request_id: case.request_id,
                first_token: first,
                draft_tokens,
                expected_matched,
                expected_accepted,
                followup_token,
            }
        })
        .collect()
}

fn run_speculative_case(model_path: &str, cases: Vec<CaseSpec>) {
    let expectations = build_expectations(model_path, &cases);
    let mut exec = build_executor(model_path, cases.len());
    let first_tokens = prefill(&mut exec, &cases);
    assert_eq!(
        first_tokens,
        expectations
            .iter()
            .map(|expect| expect.first_token)
            .collect::<Vec<_>>()
    );
    let before_state = exec.debug_state_summary();

    let verify_items: Vec<_> = expectations
        .iter()
        .map(|expect| {
            let mut token_ids = Vec::with_capacity(expect.draft_tokens.len() + 1);
            token_ids.push(expect.first_token);
            token_ids.extend_from_slice(&expect.draft_tokens);
            VerifyStepItem::new(expect.request_id, token_ids, LOGPROBS)
        })
        .collect();
    let result = exec
        .execute_speculative_verify(VerifyPlan {
            requests: &verify_items,
        })
        .expect("speculative verify");
    let after_state = exec.debug_state_summary();

    assert_eq!(result.requests.len(), expectations.len());
    for (row, expect) in result.requests.iter().zip(expectations.iter()) {
        assert_eq!(row.request_id, expect.request_id);
        assert_eq!(row.matched_draft_tokens, expect.expected_matched);
        assert_eq!(
            row.accepted_tokens
                .iter()
                .map(|token| token.token)
                .collect::<Vec<_>>(),
            expect
                .expected_accepted
                .iter()
                .map(|token| token.token)
                .collect::<Vec<_>>()
        );
    }
    for ((before, after), expect) in before_state
        .iter()
        .zip(after_state.iter())
        .zip(expectations.iter())
    {
        assert_eq!(before.request_id, after.request_id);
        assert_eq!(
            before.kv_seq_len + expect.expected_accepted.len(),
            after.kv_seq_len
        );
        assert_eq!(
            before.recurrent_seq_len + expect.expected_accepted.len(),
            after.recurrent_seq_len
        );
    }

    let last_tokens: Vec<u32> = result
        .requests
        .iter()
        .map(|row| row.accepted_tokens.last().expect("accepted token").token)
        .collect();
    let followup_reqs: Vec<_> = cases
        .iter()
        .zip(last_tokens.iter())
        .map(|(case, &token)| DecodeStepItem::new(case.request_id, token, LOGPROBS))
        .collect();
    let followup = exec
        .execute_decode(DecodePlan {
            requests: &followup_reqs,
        })
        .expect("post-spec followup")
        .requests;
    for (actual, expect) in followup.iter().zip(expectations.iter()) {
        assert_eq!(actual.request_id, expect.request_id);
        assert_eq!(actual.token, expect.followup_token);
    }
}

#[test]
fn qwen35_speculative_accept_all_state_matches_plain_decode() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    run_speculative_case(
        &model_path,
        vec![CaseSpec {
            request_id: RequestId::new(1),
            prompt_tokens: vec![9707],
            draft_len: 3,
            reject_at: None,
        }],
    );
}

#[test]
fn qwen35_speculative_accept_prefix_and_reject_first() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    run_speculative_case(
        &model_path,
        vec![
            CaseSpec {
                request_id: RequestId::new(1),
                prompt_tokens: vec![3838, 374, 220, 17, 10, 17],
                draft_len: 4,
                reject_at: Some(2),
            },
            CaseSpec {
                request_id: RequestId::new(2),
                prompt_tokens: vec![9707],
                draft_len: 4,
                reject_at: Some(0),
            },
        ],
    );
}

#[test]
fn qwen35_speculative_mixed_batch_state_matches_plain_decode() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    run_speculative_case(
        &model_path,
        vec![
            CaseSpec {
                request_id: RequestId::new(1),
                prompt_tokens: vec![9707],
                draft_len: 2,
                reject_at: None,
            },
            CaseSpec {
                request_id: RequestId::new(2),
                prompt_tokens: vec![3838, 374, 220, 17, 10, 17],
                draft_len: 4,
                reject_at: Some(1),
            },
            CaseSpec {
                request_id: RequestId::new(3),
                prompt_tokens: vec![785, 9282, 374, 3565],
                draft_len: 3,
                reject_at: Some(0),
            },
        ],
    );
}
