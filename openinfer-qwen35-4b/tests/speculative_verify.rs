use std::path::Path;

use openinfer_qwen35_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen35Executor, RequestId,
    VerifyPlan, VerifyStepItem,
};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const LOGPROBS: usize = 1;
const DIAG_LOGPROBS: usize = 20;
const MARGIN_TOL: f32 = 0.20;

#[derive(Clone, Debug)]
struct TokenDiag {
    token: u32,
    top_logprobs: Vec<(u32, f32)>,
}

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
}

fn model_path() -> String {
    let path =
        std::env::var("OPENINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string());
    assert!(
        Path::new(&path).join("config.json").exists(),
        "Qwen3.5 model is missing at {path}; set OPENINFER_TEST_MODEL_PATH"
    );
    path
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

fn prefill_with_logprobs(
    exec: &mut Qwen35Executor,
    cases: &[CaseSpec],
    logprobs: usize,
) -> Vec<TokenDiag> {
    let reqs: Vec<_> = cases
        .iter()
        .map(|case| PrefillStepItem::new(case.request_id, case.prompt_tokens.clone(), logprobs))
        .collect();
    exec.execute_prefill(PrefillPlan { requests: &reqs })
        .expect("prefill")
        .requests
        .into_iter()
        .map(|result| TokenDiag {
            token: result.first_token,
            top_logprobs: result
                .first_token_logprob
                .map(|lp| lp.top_logprobs)
                .unwrap_or_default(),
        })
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

fn decode_once_with_logprobs(
    exec: &mut Qwen35Executor,
    tokens: &[u32],
    cases: &[CaseSpec],
    logprobs: usize,
) -> Vec<TokenDiag> {
    let reqs: Vec<_> = cases
        .iter()
        .zip(tokens.iter())
        .map(|(case, &token)| DecodeStepItem::new(case.request_id, token, logprobs))
        .collect();
    exec.execute_decode(DecodePlan { requests: &reqs })
        .expect("decode")
        .requests
        .into_iter()
        .map(|result| TokenDiag {
            token: result.token,
            top_logprobs: result.logprob.map(|lp| lp.top_logprobs).unwrap_or_default(),
        })
        .collect()
}

fn regret(top_logprobs: &[(u32, f32)], token: u32) -> Option<f32> {
    top_logprobs.first().and_then(|(_, top_lp)| {
        top_logprobs
            .iter()
            .find(|(id, _)| *id == token)
            .map(|(_, lp)| top_lp - lp)
    })
}

fn top_ids(top_logprobs: &[(u32, f32)]) -> Vec<u32> {
    top_logprobs.iter().take(8).map(|(id, _)| *id).collect()
}

fn deterministic_long_prompt(len: usize, request_idx: usize) -> Vec<u32> {
    (0..len)
        .map(|i| 100 + ((i * 7919 + request_idx * 104_729) % 99_000) as u32)
        .collect()
}

fn stable_text_like_prompt(len: usize, request_idx: usize) -> Vec<u32> {
    let segment = [
        2387, 220, 16, 321, 9707, 374, 3565, 3838, 374, 220, 17, 10, 17, 785, 9282, 374, 3565, 198,
        15123, 839, 13, 220, 1024, 11, 256, 11, 4096, 13, 220, 2301, 374, 690, 1012, 13, 220,
    ];
    let mut tokens = Vec::with_capacity(len);
    while tokens.len() < len {
        tokens.extend(segment.iter().map(|token| token + request_idx as u32));
    }
    tokens.truncate(len);
    tokens
}

fn prefill_oracle_tokens(
    model_path: &str,
    prompts: &[Vec<u32>],
    token_count: usize,
) -> Vec<Vec<TokenDiag>> {
    assert!(token_count > 0, "oracle must request at least one token");
    let batch = prompts.len();
    let mut exec = build_executor(model_path, batch);
    let mut generated: Vec<Vec<TokenDiag>> = Vec::with_capacity(batch);

    let cases: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(idx, prompt_tokens)| CaseSpec {
            request_id: RequestId::new((idx + 1) as u64),
            prompt_tokens: prompt_tokens.clone(),
            draft_len: 0,
            reject_at: None,
        })
        .collect();
    let first = prefill_with_logprobs(&mut exec, &cases, DIAG_LOGPROBS);
    for diag in first {
        generated.push(vec![diag]);
    }
    for case in &cases {
        exec.drop_request(case.request_id)
            .expect("drop oracle prefill");
    }

    while generated[0].len() < token_count {
        let step_idx = generated[0].len();
        let reqs: Vec<_> = prompts
            .iter()
            .zip(generated.iter())
            .enumerate()
            .map(|(idx, (prompt, tokens))| {
                let mut context = prompt.clone();
                context.extend(tokens.iter().map(|diag| diag.token));
                PrefillStepItem::new(
                    RequestId::new((step_idx * batch + idx + 1) as u64),
                    context,
                    DIAG_LOGPROBS,
                )
            })
            .collect();
        let result = exec
            .execute_prefill(PrefillPlan { requests: &reqs })
            .expect("oracle continuation prefill");
        for (row, token) in generated.iter_mut().zip(result.requests.iter()) {
            row.push(TokenDiag {
                token: token.first_token,
                top_logprobs: token
                    .first_token_logprob
                    .as_ref()
                    .map(|lp| lp.top_logprobs.clone())
                    .unwrap_or_default(),
            });
        }
        for idx in 0..reqs.len() {
            let request_id = RequestId::new((step_idx * batch + idx + 1) as u64);
            exec.drop_request(request_id)
                .expect("drop oracle continuation");
        }
    }
    generated
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
            .zip(decode_once(&mut exec, &fed, cases))
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
            if let Some(reject_at) = case.reject_at {
                draft_tokens[reject_at] = draft_tokens[reject_at].wrapping_add(17);
            }

            CaseExpectation {
                request_id: case.request_id,
                first_token: first,
                draft_tokens,
            }
        })
        .collect()
}

fn run_speculative_case(model_path: &str, cases: &[CaseSpec]) {
    let expectations = build_expectations(model_path, cases);
    let mut exec = build_executor(model_path, cases.len());
    let first_tokens = prefill(&mut exec, cases);
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
    for ((row, expect), case) in result
        .requests
        .iter()
        .zip(expectations.iter())
        .zip(cases.iter())
    {
        assert_eq!(row.request_id, expect.request_id);
        let accepted_ids = row
            .accepted_tokens
            .iter()
            .map(|token| token.token)
            .collect::<Vec<_>>();
        assert!(
            !accepted_ids.is_empty(),
            "speculative verify must commit at least one token for request {:?}",
            expect.request_id
        );
        let expected_matched = case.reject_at.unwrap_or(expect.draft_tokens.len());
        assert_eq!(
            row.matched_draft_tokens, expected_matched,
            "speculative fixture did not execute its intended acceptance branch for request {:?}: expected matched={}, actual matched={}, drafts={:?}",
            expect.request_id, expected_matched, row.matched_draft_tokens, expect.draft_tokens
        );
        assert_eq!(
            accepted_ids.len(),
            row.matched_draft_tokens + 1,
            "accepted token count must equal matched drafts plus bonus for request {:?}: first={}, drafts={:?}, actual_accepted={:?}",
            expect.request_id,
            expect.first_token,
            expect.draft_tokens,
            accepted_ids
        );
        assert_eq!(
            accepted_ids
                .iter()
                .copied()
                .take(row.matched_draft_tokens)
                .collect::<Vec<_>>(),
            expect.draft_tokens[..row.matched_draft_tokens],
            "spec verify accepted prefix mismatch for request {:?}: first={}, drafts={:?}",
            expect.request_id,
            expect.first_token,
            expect.draft_tokens
        );
    }
    for ((before, after), row) in before_state
        .iter()
        .zip(after_state.iter())
        .zip(result.requests.iter())
    {
        assert_eq!(before.request_id, after.request_id);
        assert_eq!(
            before.kv_seq_len + row.accepted_tokens.len(),
            after.kv_seq_len
        );
        assert_eq!(
            before.recurrent_seq_len + row.accepted_tokens.len(),
            after.recurrent_seq_len
        );
    }

    let accepted_rows: Vec<Vec<u32>> = result
        .requests
        .iter()
        .map(|row| {
            row.accepted_tokens
                .iter()
                .map(|token| token.token)
                .collect()
        })
        .collect();
    let last_tokens: Vec<u32> = accepted_rows
        .iter()
        .map(|row| *row.last().expect("accepted token"))
        .collect();
    let followup_reqs: Vec<_> = cases
        .iter()
        .zip(last_tokens.iter())
        .map(|(case, &token)| DecodeStepItem::new(case.request_id, token, DIAG_LOGPROBS))
        .collect();
    let followup = exec
        .execute_decode(DecodePlan {
            requests: &followup_reqs,
        })
        .expect("post-spec followup")
        .requests;
    let actual_followup: Vec<_> = followup
        .iter()
        .zip(expectations.iter())
        .map(|(actual, expect)| {
            assert_eq!(actual.request_id, expect.request_id);
            TokenDiag {
                token: actual.token,
                top_logprobs: actual
                    .logprob
                    .as_ref()
                    .map(|logprob| logprob.top_logprobs.clone())
                    .unwrap_or_default(),
            }
        })
        .collect();
    assert_eq!(actual_followup.len(), cases.len());
    drop(exec);

    let oracle_cases: Vec<_> = cases
        .iter()
        .zip(expectations.iter())
        .zip(accepted_rows.iter())
        .map(|((case, expect), accepted)| {
            let mut prompt_tokens = case.prompt_tokens.clone();
            prompt_tokens.push(expect.first_token);
            prompt_tokens.extend_from_slice(accepted);
            CaseSpec {
                request_id: case.request_id,
                prompt_tokens,
                draft_len: 0,
                reject_at: None,
            }
        })
        .collect();
    let mut oracle_exec = build_executor(model_path, cases.len());
    let oracle_followup = prefill_with_logprobs(&mut oracle_exec, &oracle_cases, DIAG_LOGPROBS);

    let mut hard_mismatches = Vec::new();
    let mut logprob_deltas = Vec::new();
    for (idx, (actual, oracle)) in actual_followup
        .iter()
        .zip(oracle_followup.iter())
        .enumerate()
    {
        if actual.token != oracle.token {
            let actual_regret_for_oracle = regret(&actual.top_logprobs, oracle.token);
            let oracle_regret_for_actual = regret(&oracle.top_logprobs, actual.token);
            let within_actual = actual_regret_for_oracle.is_some_and(|regret| regret <= MARGIN_TOL);
            let within_oracle = oracle_regret_for_actual.is_some_and(|regret| regret <= MARGIN_TOL);
            if !within_actual && !within_oracle {
                hard_mismatches.push(format!(
                    "idx={idx} actual={} oracle={} actual_regret_for_oracle={actual_regret_for_oracle:?} oracle_regret_for_actual={oracle_regret_for_actual:?}",
                    actual.token, oracle.token
                ));
            }
        }
        for (token, oracle_logprob) in &oracle.top_logprobs {
            if let Some((_, actual_logprob)) = actual
                .top_logprobs
                .iter()
                .find(|(actual_token, _)| actual_token == token)
            {
                logprob_deltas.push((actual_logprob - oracle_logprob).abs());
            }
        }
    }
    assert!(
        hard_mismatches.is_empty(),
        "Qwen3.5 speculative commit state diverged from a full-context replay:\n{}",
        hard_mismatches.join("\n")
    );
    assert!(
        !logprob_deltas.is_empty(),
        "Qwen3.5 speculative commit oracle found no shared top-logprob entries"
    );
    logprob_deltas.sort_by(f32::total_cmp);
    let mean_delta = logprob_deltas.iter().sum::<f32>() / logprob_deltas.len() as f32;
    let p99_delta = logprob_deltas[(logprob_deltas.len() - 1) * 99 / 100];
    assert!(
        mean_delta <= 0.06 && p99_delta <= MARGIN_TOL,
        "Qwen3.5 speculative commit logits exceed the bf16 envelope: mean={mean_delta:.6}, p99={p99_delta:.6}, samples={}",
        logprob_deltas.len()
    );
}

#[test]
#[ignore = "requires Qwen3.5 weights on a CUDA GPU"]
fn qwen35_speculative_verify_first_token_matches_decode_long_batch() {
    let model_path = model_path();
    let batch = 8usize;
    let cases: Vec<_> = (0..batch)
        .map(|idx| CaseSpec {
            request_id: RequestId::new((idx + 1) as u64),
            prompt_tokens: deterministic_long_prompt(4096, idx),
            draft_len: 1,
            reject_at: Some(0),
        })
        .collect();

    let mut decode_exec = build_executor(&model_path, batch);
    let first_diag = prefill_with_logprobs(&mut decode_exec, &cases, DIAG_LOGPROBS);
    let first_tokens: Vec<u32> = first_diag.iter().map(|diag| diag.token).collect();
    let decode_next =
        decode_once_with_logprobs(&mut decode_exec, &first_tokens, &cases, DIAG_LOGPROBS);
    drop(decode_exec);

    let mut verify_exec = build_executor(&model_path, batch);
    let verify_first_tokens = prefill(&mut verify_exec, &cases);
    assert_eq!(verify_first_tokens, first_tokens);
    let verify_items: Vec<_> = cases
        .iter()
        .zip(first_tokens.iter())
        .map(|(case, &first)| {
            VerifyStepItem::new(
                case.request_id,
                vec![first, first.wrapping_add(17)],
                DIAG_LOGPROBS,
            )
        })
        .collect();
    let verify = verify_exec
        .execute_speculative_verify(VerifyPlan {
            requests: &verify_items,
        })
        .expect("speculative verify");
    let verify_next: Vec<TokenDiag> = verify
        .requests
        .iter()
        .map(|row| TokenDiag {
            token: row.accepted_tokens[0].token,
            top_logprobs: row.accepted_tokens[0]
                .logprob
                .as_ref()
                .map(|lp| lp.top_logprobs.clone())
                .unwrap_or_default(),
        })
        .collect();

    let hard_mismatches: Vec<String> = decode_next
        .iter()
        .zip(verify_next.iter())
        .enumerate()
        .filter_map(|(idx, (decode, verify))| {
            if decode.token == verify.token {
                return None;
            }
            let decode_regret_for_verify = regret(&decode.top_logprobs, verify.token);
            let verify_regret_for_decode = regret(&verify.top_logprobs, decode.token);
            let within_decode = decode_regret_for_verify.is_some_and(|r| r <= MARGIN_TOL);
            let within_verify = verify_regret_for_decode.is_some_and(|r| r <= MARGIN_TOL);
            (!within_decode && !within_verify).then(|| {
                format!(
                    "idx={idx} first={} decode={} verify={} decode_regret_for_verify={:?} verify_regret_for_decode={:?} decode_top={:?} verify_top={:?}",
                    first_tokens[idx],
                    decode.token,
                    verify.token,
                    decode_regret_for_verify,
                    verify_regret_for_decode,
                    top_ids(&decode.top_logprobs),
                    top_ids(&verify.top_logprobs),
                )
            })
        })
        .collect();
    assert!(
        hard_mismatches.is_empty(),
        "Qwen3.5 speculative verifier first posterior has non-tie divergence from decode:\n{}",
        hard_mismatches.join("\n")
    );
}

#[test]
#[ignore = "requires Qwen3.5 weights on a CUDA GPU"]
fn qwen35_speculative_verify_first_token_matches_decode_benchmark_c16() {
    let model_path = model_path();
    let batch = 16usize;
    let cases: Vec<_> = (0..batch)
        .map(|idx| CaseSpec {
            request_id: RequestId::new((idx + 1) as u64),
            prompt_tokens: stable_text_like_prompt(1024, idx),
            draft_len: 1,
            reject_at: Some(0),
        })
        .collect();

    let mut decode_exec = build_executor(&model_path, batch);
    let first_diag = prefill_with_logprobs(&mut decode_exec, &cases, DIAG_LOGPROBS);
    let first_tokens: Vec<u32> = first_diag.iter().map(|diag| diag.token).collect();
    let decode_next =
        decode_once_with_logprobs(&mut decode_exec, &first_tokens, &cases, DIAG_LOGPROBS);
    drop(decode_exec);

    let mut verify_exec = build_executor(&model_path, batch);
    let verify_first_tokens = prefill(&mut verify_exec, &cases);
    assert_eq!(verify_first_tokens, first_tokens);
    let verify_items: Vec<_> = cases
        .iter()
        .zip(first_tokens.iter())
        .map(|(case, &first)| {
            VerifyStepItem::new(
                case.request_id,
                vec![first, first.wrapping_add(17)],
                DIAG_LOGPROBS,
            )
        })
        .collect();
    let verify = verify_exec
        .execute_speculative_verify(VerifyPlan {
            requests: &verify_items,
        })
        .expect("speculative verify");
    let verify_next: Vec<TokenDiag> = verify
        .requests
        .iter()
        .map(|row| TokenDiag {
            token: row.accepted_tokens[0].token,
            top_logprobs: row.accepted_tokens[0]
                .logprob
                .as_ref()
                .map(|lp| lp.top_logprobs.clone())
                .unwrap_or_default(),
        })
        .collect();

    let hard_mismatches: Vec<String> = decode_next
        .iter()
        .zip(verify_next.iter())
        .enumerate()
        .filter_map(|(idx, (decode, verify))| {
            if decode.token == verify.token {
                return None;
            }
            let decode_regret_for_verify = regret(&decode.top_logprobs, verify.token);
            let verify_regret_for_decode = regret(&verify.top_logprobs, decode.token);
            let within_decode = decode_regret_for_verify.is_some_and(|r| r <= MARGIN_TOL);
            let within_verify = verify_regret_for_decode.is_some_and(|r| r <= MARGIN_TOL);
            (!within_decode && !within_verify).then(|| {
                format!(
                    "idx={idx} first={} decode={} verify={} decode_regret_for_verify={:?} verify_regret_for_decode={:?} decode_top={:?} verify_top={:?}",
                    first_tokens[idx],
                    decode.token,
                    verify.token,
                    decode_regret_for_verify,
                    verify_regret_for_decode,
                    top_ids(&decode.top_logprobs),
                    top_ids(&verify.top_logprobs),
                )
            })
        })
        .collect();
    assert!(
        hard_mismatches.is_empty(),
        "Qwen3.5 speculative verifier benchmark c16 posterior has non-tie divergence from decode:\n{}",
        hard_mismatches.join("\n")
    );
}

#[test]
#[ignore = "requires Qwen3.5 weights on a CUDA GPU"]
fn qwen35_speculative_verify_multitoken_span_matches_prefill_benchmark_c16() {
    let model_path = model_path();
    let batch = 16usize;
    let verify_span = 5usize;
    let prompts: Vec<_> = (0..batch)
        .map(|idx| stable_text_like_prompt(1024, idx))
        .collect();
    let oracle = prefill_oracle_tokens(&model_path, &prompts, verify_span + 1);

    let cases: Vec<_> = prompts
        .into_iter()
        .enumerate()
        .map(|(idx, prompt_tokens)| CaseSpec {
            request_id: RequestId::new((idx + 1) as u64),
            prompt_tokens,
            draft_len: verify_span - 1,
            reject_at: None,
        })
        .collect();
    let mut verify_exec = build_executor(&model_path, batch);
    let first_tokens = prefill(&mut verify_exec, &cases);
    let expected_first: Vec<u32> = oracle.iter().map(|row| row[0].token).collect();
    assert_eq!(first_tokens, expected_first);

    let verify_items: Vec<_> = cases
        .iter()
        .zip(oracle.iter())
        .map(|(case, row)| {
            let token_ids = row[..verify_span]
                .iter()
                .map(|diag| diag.token)
                .collect::<Vec<_>>();
            VerifyStepItem::new(case.request_id, token_ids, DIAG_LOGPROBS)
        })
        .collect();
    let verify = verify_exec
        .execute_speculative_verify(VerifyPlan {
            requests: &verify_items,
        })
        .expect("speculative verify");

    let mut hard_mismatches = Vec::new();
    for (idx, (result, oracle_row)) in verify.requests.iter().zip(oracle.iter()).enumerate() {
        let expected = &oracle_row[1..=verify_span];
        let actual = result
            .accepted_tokens
            .iter()
            .map(|token| TokenDiag {
                token: token.token,
                top_logprobs: token
                    .logprob
                    .as_ref()
                    .map(|lp| lp.top_logprobs.clone())
                    .unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        let mut saw_tie_divergence = false;
        for (row_idx, (actual_diag, expected_diag)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            if actual_diag.token == expected_diag.token {
                continue;
            }
            let oracle_regret = regret(&expected_diag.top_logprobs, actual_diag.token);
            let verify_regret = regret(&actual_diag.top_logprobs, expected_diag.token);
            let within_oracle = oracle_regret.is_some_and(|regret| regret <= MARGIN_TOL);
            let within_verify = verify_regret.is_some_and(|regret| regret <= MARGIN_TOL);
            if within_oracle || within_verify {
                saw_tie_divergence = true;
            } else {
                hard_mismatches.push(format!(
                    "idx={idx} row={row_idx} accepted_len={} matched_drafts={} expected={} actual={} oracle_regret={oracle_regret:?} verify_regret={verify_regret:?}",
                    actual.len(),
                    result.matched_draft_tokens,
                    expected_diag.token,
                    actual_diag.token,
                ));
            }
        }
        if actual.len() != expected.len() && !saw_tie_divergence {
            hard_mismatches.push(format!(
                "idx={idx} accepted_len={} expected_len={} matched_drafts={} without a bf16-tie divergence",
                actual.len(),
                expected.len(),
                result.matched_draft_tokens,
            ));
        }
    }

    assert!(
        hard_mismatches.is_empty(),
        "Qwen3.5 speculative verifier benchmark c16 multitoken posterior diverged from prefill oracle:\n{}",
        hard_mismatches.join("\n")
    );
}

#[test]
#[ignore = "requires Qwen3.5 weights on a CUDA GPU"]
fn qwen35_speculative_single_request_commits_transaction_state() {
    let model_path = model_path();
    run_speculative_case(
        &model_path,
        &[CaseSpec {
            request_id: RequestId::new(1),
            prompt_tokens: vec![9707],
            draft_len: 3,
            reject_at: None,
        }],
    );
}

#[test]
#[ignore = "requires Qwen3.5 weights on a CUDA GPU"]
fn qwen35_speculative_accept_prefix_and_reject_first_transaction_state() {
    let model_path = model_path();
    run_speculative_case(
        &model_path,
        &[
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
#[ignore = "requires Qwen3.5 weights on a CUDA GPU"]
fn qwen35_speculative_mixed_batch_commits_transaction_state() {
    let model_path = model_path();
    run_speculative_case(
        &model_path,
        &[
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
