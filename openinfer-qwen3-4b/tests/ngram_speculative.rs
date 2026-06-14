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
    prefill_item_max(id, prompt, MAX_OUTPUT)
}

fn prefill_item_max(id: u64, prompt: &[u32], max_output: usize) -> PrefillStepItem {
    PrefillStepItem::new(
        RequestId::new(id),
        prompt.to_vec(),
        max_output,
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

/// Counters for the speculative run: how many forward passes (verify + fallback
/// single decodes) it took and how many tokens those passes committed.
struct SpecStats {
    tokens: usize,
    forwards: usize,
    drafted_forwards: usize,
    drafted_tokens: usize,
}

/// Greedy speculative decode that also tallies forward passes / acceptance.
fn speculative_decode_timed(
    ex: &mut Qwen3Executor,
    id: u64,
    prompt: &[u32],
    n_generate: usize,
) -> (Vec<u32>, SpecStats) {
    let proposer = NgramProposer::new(NgramConfig {
        max_ngram: 3,
        min_ngram: 1,
        num_speculative: 4,
    });
    let pr = ex
        .execute_prefill(PrefillPlan {
            requests: &[prefill_item_max(id, prompt, n_generate + 8)],
            echo: false,
        })
        .expect("prefill");
    let first = pr.requests[0].first_token;
    let mut history = prompt.to_vec();
    history.push(first);
    let mut out = vec![first];
    let mut stats = SpecStats {
        tokens: 0,
        forwards: 0,
        drafted_forwards: 0,
        drafted_tokens: 0,
    };

    while out.len() < n_generate {
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
            stats.forwards += 1;
        } else {
            let committed = ex
                .execute_speculative(&SpeculativeStepItem::new(RequestId::new(id), last, drafts))
                .expect("speculative");
            stats.forwards += 1;
            stats.drafted_forwards += 1;
            stats.drafted_tokens += committed.len();
            for t in committed {
                history.push(t);
                out.push(t);
            }
        }
    }
    out.truncate(n_generate);
    stats.tokens = n_generate;
    (out, stats)
}

/// Wall-clock + acceptance benchmark; ignored by default (needs a GPU + weights
/// and is timing-sensitive). Run with:
/// `cargo test -p openinfer-qwen3-4b --release --test ngram_speculative \
///   ngram_speculative_speedup -- --ignored --nocapture`
#[test]
#[ignore]
fn ngram_speculative_speedup() {
    use std::time::Instant;

    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let n_generate = std::env::var("OPENINFER_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(192usize);

    let mut ex =
        Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("build eager executor");
    ex.set_prefix_cache_enabled(false);
    let prompt = repetitive_prompt();

    // Warm up CUDA/cuBLAS so the first run's lazy init doesn't skew timings.
    let _ = greedy_decode(&mut ex, 100, &prompt);

    let t0 = Instant::now();
    let baseline = {
        // Inline greedy decode for `n_generate` tokens (greedy_decode is fixed at
        // N_GENERATE), counting decode forwards.
        let pr = ex
            .execute_prefill(PrefillPlan {
                requests: &[prefill_item_max(1, &prompt, n_generate + 8)],
                echo: false,
            })
            .expect("prefill");
        let mut last = pr.requests[0].first_token;
        let mut out = vec![last];
        while out.len() < n_generate {
            let dr = ex
                .execute_decode(DecodePlan {
                    requests: &[DecodeStepItem::new(
                        RequestId::new(1),
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
    };
    let baseline_dt = t0.elapsed();
    let baseline_forwards = baseline.len() - 1; // one decode forward per token after prefill

    let t1 = Instant::now();
    let (spec_out, stats) = speculative_decode_timed(&mut ex, 2, &prompt, n_generate);
    let spec_dt = t1.elapsed();

    assert_eq!(
        baseline, spec_out,
        "speculative output must stay token-identical to greedy"
    );

    let baseline_tpot = baseline_dt.as_secs_f64() / baseline_forwards as f64 * 1e3;
    let spec_tpot = spec_dt.as_secs_f64() / (stats.tokens - 1) as f64 * 1e3;
    let accept_per_verify = stats.drafted_tokens as f64 / stats.drafted_forwards.max(1) as f64;

    eprintln!("=== n-gram speculative decode benchmark ===");
    eprintln!("tokens generated   : {}", stats.tokens);
    eprintln!(
        "baseline (greedy)  : {:>7.1} ms, {} forwards, {:.3} ms/token",
        baseline_dt.as_secs_f64() * 1e3,
        baseline_forwards,
        baseline_tpot
    );
    eprintln!(
        "speculative        : {:>7.1} ms, {} forwards ({} verify), {:.3} ms/token",
        spec_dt.as_secs_f64() * 1e3,
        stats.forwards,
        stats.drafted_forwards,
        spec_tpot
    );
    eprintln!(
        "accepted/verify    : {:.2} tokens (K=4, so up to 5 per verify)",
        accept_per_verify
    );
    eprintln!(
        "forward-pass saving: {:.1}% fewer model calls",
        (1.0 - stats.forwards as f64 / baseline_forwards as f64) * 100.0
    );
    eprintln!(
        "wall-clock speedup : {:.2}x",
        baseline_dt.as_secs_f64() / spec_dt.as_secs_f64()
    );
}
