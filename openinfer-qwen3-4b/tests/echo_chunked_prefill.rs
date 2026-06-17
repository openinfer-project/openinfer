//! Chunked-echo self-consistency gate for echo + logprobs prefill.
//!
//! Echo with `logprobs > 0` returns a logprob for every prompt token: the
//! logprob of prompt token `k` is read from the model's distribution at
//! position `k - 1`. The naive implementation materializes all-position logits
//! (`vocab × prompt_len`) in one forward, which OOMs on long prompts (#358).
//! The fix chunks the echo prefill like any other prompt: each chunk computes
//! all-position logits only for its own slice and the executor stitches the
//! per-chunk prompt logprobs back together (see `merge_echo_prompt_logprobs`).
//!
//! The numerically interesting part is the cross-chunk seam: the logprob of the
//! token at a chunk boundary comes from the *previous* chunk's last position,
//! and an off-by-one there would silently corrupt or drop one logprob per
//! boundary. There is no HF golden for within-prompt distributions, so this gate
//! uses a stronger, hardware-independent invariant instead: **the chunked echo
//! prompt logprobs must match the single-pass result** — same model, same GPU,
//! the only difference being how many tokens each forward processed.
//!
//! Crossing a seam moves later positions onto the `kv_len > q_len` attention
//! path, which drifts a few bf16 ULPs (exactly like the prefix-cache replay in
//! `hf_golden_gate`), so the match is asserted within a tight bf16 tolerance
//! rather than bit-exact. The comparison is over each *actual* prompt token's
//! logprob — the same token id in both runs — so a benign bf16 argmax tie
//! cannot trip it, while a seam off-by-one (reading the logprob from the wrong
//! position) moves it by far more than ULP noise. A realistic prompt (a golden
//! prompt plus its own teacher-forced continuation) keeps the per-position
//! distributions peaked, so that noise floor stays low.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the HF golden token file; skips
//! cleanly when the model is absent (point `OPENINFER_TEST_MODEL_PATH` at the
//! weights to run it).

use std::path::Path;

use openinfer_core::engine::TokenLogprob;
use openinfer_core::sampler::SamplingParams;
use openinfer_qwen3_4b::runtime::{PrefillPlan, PrefillStepItem, Qwen3Executor, RequestId};
use safetensors::{Dtype, SafeTensors};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/qwen3-4b-hf-golden.safetensors"
);

/// Top-K logprobs requested per position — wide enough that the chunked and
/// single-pass top-K sets overlap for the seam comparison.
const LOGPROBS: usize = 16;
const MAX_OUTPUT_TOKENS: usize = 8;
/// Small budget so even a short realistic sequence is split into several
/// chunks, crossing the seam multiple times (and ending on a partial chunk).
const CHUNK_BUDGET: usize = 4;

/// Engine-vs-engine on a peaked sequence: the only difference is the forward
/// *shape* (a 4-row chunk vs a full-width pass picks different bf16 GEMM
/// reduction orders, and seams move later positions onto the `kv_len > q_len`
/// attention path). `MEAN_TOL` stays under the HF gate's own 0.06 floor — both
/// runs sit much closer to each other than either does to HF. `MAX_TOL` allows
/// the irreducible bf16 tail on a single position (observed ~0.26 over 160
/// positions) while staying far below a real seam off-by-one, which reads a
/// peaked token's logprob from the wrong position and moves it by nats.
const MEAN_TOL: f32 = 0.04;
const MAX_TOL: f32 = 0.50;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping echo_chunked_prefill: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn as_i32(st: &SafeTensors, name: &str) -> (Vec<i32>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("golden missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::I32, "{name} must be i32");
    let v = t
        .data()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (v, t.shape().to_vec())
}

/// A realistic, peaked prompt: golden sequence 0's prompt followed by its own
/// teacher-forced decode tokens. Concatenating the model's own continuation
/// keeps every next-token distribution sharp, so the actual-token logprobs are
/// stable across the two forward shapes and the tolerance can stay tight.
fn realistic_prompt() -> Vec<u32> {
    let bytes = std::fs::read(GOLDEN).unwrap_or_else(|e| panic!("read {GOLDEN}: {e}"));
    let st = SafeTensors::deserialize(&bytes).expect("parse golden safetensors");
    let (prompt_tokens, _) = as_i32(&st, "prompt_tokens");
    let (prompt_lens, _) = as_i32(&st, "prompt_lens");
    let (decode_tokens, dshape) = as_i32(&st, "decode_tokens");
    let decode_len = dshape[1];

    let p0_len = prompt_lens[0] as usize;
    let mut seq: Vec<u32> = prompt_tokens[..p0_len].iter().map(|&t| t as u32).collect();
    seq.extend((0..decode_len).map(|step| decode_tokens[step] as u32));
    seq
}

fn echo_item(id: RequestId, prompt: Vec<u32>) -> PrefillStepItem {
    PrefillStepItem::new(
        id,
        prompt,
        MAX_OUTPUT_TOKENS,
        SamplingParams::default(),
        LOGPROBS,
        true, // echo
        0.0,
    )
}

/// Echo-prefill `prompt` in a single forward (budget ≥ prompt_len) and return
/// the prompt logprobs.
fn single_pass(ex: &mut Qwen3Executor, prompt: &[u32]) -> Vec<Option<TokenLogprob>> {
    let id = RequestId::new(1);
    let result = ex
        .execute_prefill(PrefillPlan {
            requests: &[echo_item(id, prompt.to_vec())],
            echo: true,
        })
        .expect("single-pass echo prefill");
    let req = &result.requests[0];
    assert!(
        req.completed,
        "single pass must finish the prompt in one step"
    );
    let lps = req
        .prompt_logprobs
        .clone()
        .expect("echo prefill must return prompt logprobs");
    ex.drop_request(id).expect("drop single-pass request");
    lps
}

/// Echo-prefill `prompt` in a single chunk via the chunked plumbing
/// (`with_chunk_budget` ≥ prompt_len). Exercises the per-chunk partial build
/// and the accumulator merge for the contiguous case, where they must be the
/// identity transform — so the result has to equal [`single_pass`] bit-for-bit.
fn single_chunk_via_budget(ex: &mut Qwen3Executor, prompt: &[u32]) -> Vec<Option<TokenLogprob>> {
    let id = RequestId::new(3);
    let result = ex
        .execute_prefill(PrefillPlan {
            requests: &[echo_item(id, prompt.to_vec()).with_chunk_budget(prompt.len())],
            echo: true,
        })
        .expect("single-chunk echo prefill");
    let req = &result.requests[0];
    assert!(
        req.completed,
        "budget == prompt_len must finish in one chunk"
    );
    let lps = req.prompt_logprobs.clone().expect("prompt logprobs");
    ex.drop_request(id).expect("drop single-chunk request");
    lps
}

/// Echo-prefill `prompt` one `budget`-token chunk per `execute_prefill` call,
/// mirroring how the scheduler drives a long prompt across steps, and return
/// the stitched prompt logprobs from the final chunk. Asserts that only the
/// final chunk surfaces prompt logprobs.
fn chunked(ex: &mut Qwen3Executor, prompt: &[u32], budget: usize) -> Vec<Option<TokenLogprob>> {
    let id = RequestId::new(2);
    let mut steps = 0;
    loop {
        let result = ex
            .execute_prefill(PrefillPlan {
                requests: &[echo_item(id, prompt.to_vec()).with_chunk_budget(budget)],
                echo: true,
            })
            .expect("chunked echo prefill");
        let req = &result.requests[0];
        steps += 1;
        if req.completed {
            let lps = req
                .prompt_logprobs
                .clone()
                .expect("final chunk must return the stitched prompt logprobs");
            assert!(
                steps > 1,
                "budget {budget} < prompt {} must take more than one chunk",
                prompt.len()
            );
            ex.drop_request(id).expect("drop chunked request");
            return lps;
        }
        assert!(
            req.prompt_logprobs.is_none(),
            "non-final chunk must not surface prompt logprobs (step {steps})"
        );
    }
}

#[test]
fn chunked_echo_prompt_logprobs_match_single_pass() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    if !Path::new(GOLDEN).exists() {
        eprintln!("skipping echo_chunked_prefill: {GOLDEN} is missing");
        return;
    }
    let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0])
        .unwrap_or_else(|e| panic!("build executor: {e:#}"));
    // Echo bypasses the prefix cache anyway, but disable it so neither run can
    // reuse the other's blocks and shrink a forward.
    ex.set_prefix_cache_enabled(false);

    let prompt = realistic_prompt();
    assert!(
        prompt.len() > CHUNK_BUDGET,
        "need a prompt longer than one chunk to cross a seam"
    );
    let reference = single_pass(&mut ex, &prompt);

    // Control: routing the whole prompt through the chunked plumbing as a single
    // chunk must reproduce the single-pass result *exactly*. This isolates a
    // bug in the partial-build / accumulator merge (which would show here, with
    // no numerical drift to hide behind) from the unavoidable bf16 path drift
    // that crossing real seams introduces below.
    let one_chunk = single_chunk_via_budget(&mut ex, &prompt);
    assert_eq!(
        one_chunk.len(),
        reference.len(),
        "single-chunk plumbing changed the prompt logprobs length"
    );
    for (k, (r, o)) in reference.iter().zip(&one_chunk).enumerate() {
        match (r, o) {
            (None, None) => {}
            (Some(r), Some(o)) => {
                assert_eq!(
                    r.logprob.to_bits(),
                    o.logprob.to_bits(),
                    "single-chunk plumbing perturbed prompt index {k}'s logprob with no forward-shape change"
                );
                assert_eq!(
                    r.top_logprobs, o.top_logprobs,
                    "single-chunk plumbing perturbed prompt index {k}'s top-K"
                );
            }
            _ => panic!("single-chunk plumbing changed Some/None at prompt index {k}"),
        }
    }

    let got = chunked(&mut ex, &prompt, CHUNK_BUDGET);

    assert_eq!(
        reference.len(),
        prompt.len(),
        "prompt logprobs has one slot per prompt token"
    );
    assert_eq!(got.len(), reference.len(), "chunked length must match");
    assert!(
        reference[0].is_none() && got[0].is_none(),
        "the first prompt token has no predecessor, so its logprob is None in both runs"
    );

    // Compare the logprob of each *actual* prompt token — the same token id in
    // both runs, so this is apples-to-apples regardless of bf16 argmax ties
    // (two near-equal tokens can swap the top-1 slot between a 4-row and a
    // full-width forward without either being wrong; that is why
    // `hf_golden_gate` also refuses to assert exact argmax). A seam off-by-one,
    // by contrast, reads the token's logprob from the wrong position and moves
    // it by far more than bf16 noise — which the per-position delta below
    // catches, with the worst offender pinpointed by index.
    let mut deltas: Vec<(usize, f32)> = Vec::new();
    for (k, (r, g)) in reference.iter().zip(&got).enumerate().skip(1) {
        let r = r
            .as_ref()
            .unwrap_or_else(|| panic!("single pass missing logprob at prompt index {k}"));
        let g = g
            .as_ref()
            .unwrap_or_else(|| panic!("chunked run missing logprob at prompt index {k}"));
        deltas.push((k, (r.logprob - g.logprob).abs()));
    }

    let count = deltas.len() as f32;
    let mean = deltas.iter().map(|&(_, d)| d).sum::<f32>() / count;
    let (worst_k, max) = deltas
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    eprintln!(
        "echo_chunked_prefill: prompt_len={}, budget={CHUNK_BUDGET}, {} positions — \
         mean Δlogprob {mean:.5}, max {max:.5} @ index {worst_k}",
        prompt.len(),
        deltas.len()
    );
    assert!(
        mean <= MEAN_TOL,
        "mean |Δlogprob| {mean:.5} > {MEAN_TOL} — chunked echo drifted from the single pass beyond bf16 noise"
    );
    assert!(
        max <= MAX_TOL,
        "max |Δlogprob| {max:.5} @ index {worst_k} > {MAX_TOL} — that prompt position is materially wrong (likely a seam off-by-one)"
    );
}
