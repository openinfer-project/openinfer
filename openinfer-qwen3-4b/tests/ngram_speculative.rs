//! GPU validation for n-gram speculative decoding: greedy speculation is
//! lossless, so the token sequence produced with the proposer must match plain
//! greedy decode token-for-token.
//!
//! Methodology note: the verify forward runs the prefill attention kernel while
//! the reference decode runs the decode attention kernel. The math is identical
//! but bf16 reduction order can differ, so a near-tie position could in
//! principle diverge; the repetitive prompt keeps the greedy argmax
//! well-separated. Prefix caching is disabled so both runs are cold and
//! identically shaped.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when absent
//! (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use openinfer_core::sampler::SamplingParams;
use openinfer_qwen3_4b::ngram::{NgramConfig, NgramProposer};
use openinfer_qwen3_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen3Executor, RequestId,
    SpeculativeStepItem,
};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const N_GENERATE: usize = 24;
const MAX_OUTPUT: usize = N_GENERATE + 8;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 ngram_speculative: {MODEL_PATH}/config.json missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Repetitive prompt so the n-gram proposer fires (its suffix recurs earlier).
fn repetitive_prompt() -> Vec<u32> {
    let unit: [u32; 8] = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007];
    let mut p = Vec::new();
    for _ in 0..8 {
        p.extend_from_slice(&unit);
    }
    p
}

fn prefill_item(id: u64, prompt: &[u32]) -> PrefillStepItem {
    PrefillStepItem::new(
        RequestId::new(id),
        prompt.to_vec(),
        MAX_OUTPUT,
        SamplingParams::default(),
        0,
        false,
        0.0,
    )
}

/// Plain greedy decode (spec-off): one model token per step.
fn greedy_decode(ex: &mut Qwen3Executor, id: u64, prompt: &[u32]) -> Vec<u32> {
    let pr = ex
        .execute_prefill(PrefillPlan {
            requests: &[prefill_item(id, prompt)],
            echo: false,
        })
        .expect("prefill");
    let mut last = pr.requests[0].first_token;
    let mut out = vec![last];
    while out.len() < N_GENERATE {
        let dr = ex
            .execute_decode(DecodePlan {
                requests: &[DecodeStepItem::new(
                    RequestId::new(id),
                    last,
                    SamplingParams::default(),
                    0,
                    0.0,
                )],
            })
            .expect("decode");
        last = dr.requests[0].token;
        out.push(last);
    }
    out
}

/// Greedy n-gram speculative decode (spec-on).
fn speculative_decode(ex: &mut Qwen3Executor, id: u64, prompt: &[u32]) -> Vec<u32> {
    let proposer = NgramProposer::new(NgramConfig {
        max_ngram: 3,
        min_ngram: 1,
        num_speculative: 4,
    });
    let pr = ex
        .execute_prefill(PrefillPlan {
            requests: &[prefill_item(id, prompt)],
            echo: false,
        })
        .expect("prefill");
    let first = pr.requests[0].first_token;
    let mut history = prompt.to_vec();
    history.push(first);
    let mut out = vec![first];

    while out.len() < N_GENERATE {
        let last = *out.last().unwrap();
        let drafts = proposer.propose(&history);
        if drafts.is_empty() {
            let dr = ex
                .execute_decode(DecodePlan {
                    requests: &[DecodeStepItem::new(
                        RequestId::new(id),
                        last,
                        SamplingParams::default(),
                        0,
                        0.0,
                    )],
                })
                .expect("decode");
            let t = dr.requests[0].token;
            history.push(t);
            out.push(t);
        } else {
            let committed = ex
                .execute_speculative(&SpeculativeStepItem::new(RequestId::new(id), last, drafts))
                .expect("speculative");
            for t in committed {
                history.push(t);
                out.push(t);
            }
        }
    }
    out.truncate(N_GENERATE);
    out
}

#[test]
fn ngram_speculative_is_lossless_vs_greedy() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let mut ex =
        Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("build eager executor");
    // Cold, identically-shaped runs so the comparison is exact.
    ex.set_prefix_cache_enabled(false);

    let prompt = repetitive_prompt();
    let reference = greedy_decode(&mut ex, 1, &prompt);
    let speculative = speculative_decode(&mut ex, 2, &prompt);

    assert_eq!(
        reference, speculative,
        "greedy speculative decode must be token-identical to plain greedy decode"
    );
}
