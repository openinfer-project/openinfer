//! N-gram (prompt-lookup) speculative-decoding losslessness gate.
//!
//! Greedy speculative decoding must be *lossless*: every draft is verified by a
//! target forward and only the matching-argmax prefix (plus one bonus) is
//! committed, so the accepted tokens are the target model's own greedy
//! continuation. The catch is pure numerics — the verify path runs the
//! *prefill* attention kernel over the `K + 1` span while a plain decode runs
//! the *decode* kernel, and the two differ by ~1 bf16 ULP. On a near-tie that
//! flips one argmax, and from there two greedy runs fan out completely.
//!
//! So an exact `spec == baseline` token match is the wrong gate: it false-fails
//! on a benign tie flip. We use the same *regret* test as the DFlash gate (and
//! `hf_golden_gate`): at the first position the two sequences disagree (where
//! they still share an identical context, so the comparison is valid) we ask how
//! far below the argmax the speculative pick sits — measured *in the prefill
//! kernel's own distribution*, because that is the kernel the verify path runs.
//! A re-prefill of the shared context (`prefill_next`) gives that reference.
//! Within `MARGIN_TOL` of the prefill argmax ⇒ a benign numerical tie. Clearly
//! worse (or outside the prefill top-K) ⇒ the verify/accept logic chose a token
//! the forward never favored — a real bug.
//!
//! Unlike the DFlash gate this needs no draft model: n-gram drafts come from
//! scanning each request's own token context, so the only knob is
//! `--ngram-speculative`. Prompts are repetitive/structured so the proposer
//! actually fires (its suffix recurs earlier) and the verify path is exercised;
//! losslessness holds regardless, but a no-op proposer would make the gate
//! vacuous.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when absent (set
//! `OPENINFER_TEST_MODEL_PATH` to the weights to run it).

use std::path::Path;
use std::time::Duration;

use openinfer_core::engine::{EngineHandle, GenerateRequest, TokenEvent, TokenSink};
use openinfer_core::sampler::SamplingParams;
use openinfer_qwen3_4b::{
    DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES, DEFAULT_MAX_PREFILL_TOKENS, DecodeOverlap,
    Qwen3LaunchOptions, Qwen3MemoryOptions, Qwen3OffloadOptions,
};
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const GENERATED_TOKENS: usize = 64;
/// Top-K logprobs requested from the baseline; wide enough that the speculative
/// pick is in the set on any real tie (a pick outside the top-K is itself a red
/// flag the gate should catch).
const LOGPROBS: usize = 20;
/// Max acceptable regret: how far below the baseline's argmax (in the baseline's
/// own logprobs) the speculative pick may sit at the divergence point. ~3 bf16
/// ULP at typical logit magnitudes — mirrors `hf_golden_gate`'s `MARGIN_TOL`.
const MARGIN_TOL: f32 = 0.20;

/// Both tests launch a Qwen3-4B engine, and two at once overflow a 16 GB card.
/// Cargo runs tests in one binary concurrently, so serialize the engine-holding
/// bodies — only one engine is ever resident on the GPU.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn target_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => Some(MODEL_PATH.to_string()),
        Err(_) => {
            eprintln!(
                "skipping ngram gate: {MODEL_PATH}/config.json missing; set OPENINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

/// `ngram = true` switches on host-side n-gram speculation (no draft model). The
/// baseline (`false`) still forces the prefix cache off so both runs take the
/// same cold prefill path.
fn launch_options(ngram: bool) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: true,
        max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(0.85, DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES)
            .validate()
            .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: None,
        ngram_speculative: ngram,
        enable_kv_events: false,
    }
}

/// One decoded position: the chosen token and (when requested) the top-K
/// `(token, logprob)` distribution that produced it.
struct Step {
    id: u32,
    top_logprobs: Vec<(u32, f32)>,
}

/// Submit one greedy request and collect the decoded steps until `Finished`.
fn generate(
    handle: &EngineHandle,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    max_tokens: usize,
) -> Vec<Step> {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs,
            echo: false,
        })
        .expect("submit failed");

    let mut steps = Vec::new();
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, logprob }) => steps.push(Step {
                id,
                top_logprobs: logprob.map(|lp| lp.top_logprobs).unwrap_or_default(),
            }),
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => return steps,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

/// Submit several greedy requests at once, then collect each one's steps. They
/// run concurrently in the one engine — the scheduler batches them — so with
/// heterogeneous `max_tokens` the verify spans differ across a batch, exercising
/// the real bs>1 verify path. Each tuple is `(prompt_tokens, max_tokens)`;
/// logprobs are off so the speculative path stays active. Returns one step list
/// per request, in submission order.
fn generate_concurrent(handle: &EngineHandle, requests: Vec<(Vec<u32>, usize)>) -> Vec<Vec<Step>> {
    let receivers: Vec<_> = requests
        .into_iter()
        .map(|(prompt_tokens, max_tokens)| {
            let (token_tx, rx) = TokenSink::standalone();
            handle
                .submit(GenerateRequest {
                    request_id: None,
                    queued_at_unix_s: None,
                    prompt_tokens,
                    params: SamplingParams::default(),
                    max_tokens,
                    lora_adapter: None,
                    token_tx,
                    logprobs: 0,
                    echo: false,
                })
                .expect("submit failed");
            rx
        })
        .collect();

    receivers
        .into_iter()
        .map(|mut rx| {
            let mut steps = Vec::new();
            loop {
                match rx.blocking_recv().map(|(_, event)| event) {
                    Some(TokenEvent::Token { id, logprob }) => steps.push(Step {
                        id,
                        top_logprobs: logprob.map(|lp| lp.top_logprobs).unwrap_or_default(),
                    }),
                    Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
                    Some(TokenEvent::Finished { .. }) => break,
                    Some(TokenEvent::Error { message, .. }) => {
                        panic!("generation failed: {message}")
                    }
                    Some(TokenEvent::Rejected { message, .. }) => {
                        panic!("generation rejected: {message}")
                    }
                    None => panic!("scheduler channel closed without Finished"),
                }
            }
            steps
        })
        .collect()
}

/// Prefill `context` (echo) and return the next-token distribution the *prefill*
/// kernel produces — the kernel the speculative verify path also uses. This is
/// the reference the spec pick should match (vs the plain-decode baseline, whose
/// kernel resolves bifurcation ties to the other side).
fn prefill_next(handle: &EngineHandle, context: Vec<u32>, logprobs: usize) -> Step {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens: context,
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs,
            echo: true,
        })
        .expect("submit failed");

    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, logprob }) => {
                return Step {
                    id,
                    top_logprobs: logprob.map(|lp| lp.top_logprobs).unwrap_or_default(),
                };
            }
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => panic!("echo prefill finished without a token"),
            Some(TokenEvent::Error { message, .. }) => panic!("prefill failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("prefill rejected: {message}"),
            None => panic!("scheduler channel closed without a token"),
        }
    }
}

/// Compare one prompt's speculative `spec` steps against its plain-greedy `base`,
/// tolerating only the benign prefill-vs-decode kernel-gap tie flip (the spec
/// pick sits within `MARGIN_TOL` of the prefill kernel's own argmax, measured in
/// the prefill distribution the verify path actually runs). `Ok(())` ⇒ lossless
/// or a benign tie; `Err(diagnostic)` ⇒ a real spec bug. `handle` must be the
/// live speculative engine — at a divergence it re-prefills the shared context
/// (`prompt_tokens` + the matched prefix) to read that prefill-kernel reference.
fn check_lossless(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    i: usize,
    prompt: &str,
    prompt_tokens: &[u32],
    base: &[Step],
    spec: &[Step],
) -> Result<(), String> {
    let matched = base
        .iter()
        .zip(spec)
        .take_while(|(b, s)| b.id == s.id)
        .count();

    if matched == base.len().min(spec.len()) {
        eprintln!(
            "prompt {i} ({prompt:?}): {matched}/{} tokens identical (100% lossless)",
            base.len()
        );
        return Ok(());
    }

    let spec_id = spec[matched].id;
    let decode_argmax = base[matched].top_logprobs[0].0;

    {
        let lo = matched.saturating_sub(2);
        let hi = (matched + 3).min(base.len()).min(spec.len());
        let base_ids: Vec<u32> = base[..hi].iter().map(|s| s.id).collect();
        eprintln!("  [diag] prompt {i} matched={matched}");
        eprintln!(
            "  [diag] context+gen base ids {:?} = {:?}",
            &base_ids,
            tokenizer.decode(&base_ids, false).unwrap_or_default()
        );
        eprintln!(
            "  [diag] base[{lo}..{hi}] = {:?}",
            base[lo..hi]
                .iter()
                .map(|s| (s.id, tokenizer.decode(&[s.id], false).unwrap_or_default()))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "  [diag] spec[{lo}..{hi}] = {:?}",
            spec[lo..hi]
                .iter()
                .map(|s| (s.id, tokenizer.decode(&[s.id], false).unwrap_or_default()))
                .collect::<Vec<_>>()
        );
    }

    let mut context = prompt_tokens.to_vec();
    context.extend(base[..matched].iter().map(|s| s.id));
    let prefill_ref = prefill_next(handle, context, LOGPROBS);

    if prefill_ref.id == spec_id {
        let decode_lp = base[matched]
            .top_logprobs
            .iter()
            .find(|(t, _)| *t == spec_id)
            .map(|(_, lp)| base[matched].top_logprobs[0].1 - lp);
        eprintln!(
            "prompt {i} ({prompt:?}): kernel-gap flip at token {matched} — verify(prefill)→{spec_id}, \
             decode→{decode_argmax}; spec matches prefill greedy (decode-margin {:?}). Not a spec bug.",
            decode_lp
        );
        return Ok(());
    }

    let prefill_regret = prefill_ref
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == spec_id)
        .map(|(_, lp)| prefill_ref.top_logprobs[0].1 - lp);

    if let Some(regret) = prefill_regret {
        if regret <= MARGIN_TOL {
            eprintln!(
                "prompt {i} ({prompt:?}): tie flip at token {matched} — \
                 verify(prefill)→{}, spec→{spec_id}, decode→{decode_argmax}; \
                 spec pick is #2 in the prefill distribution (regret {regret:.3} ≤ {MARGIN_TOL}). \
                 Not a spec bug.",
                prefill_ref.id,
            );
            return Ok(());
        }
    }

    let decode_regret = base[matched]
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == spec_id)
        .map(|(_, lp)| base[matched].top_logprobs[0].1 - lp);
    Err(format!(
        "prompt {i}: at token {matched} spec chose {spec_id} but prefill greedy says {} and \
         decode greedy says {decode_argmax} (spec regret in prefill dist: {prefill_regret:?} > \
         {MARGIN_TOL}; in decode dist: {decode_regret:?}) — real spec bug",
        prefill_ref.id,
    ))
}

/// bs=1 losslessness: each prompt's greedy n-gram speculative decode must match
/// its plain-greedy baseline token-for-token (modulo the benign bf16 tie flip).
#[test]
fn ngram_speculative_greedy_matches_plain_greedy() {
    let Some(model_path) = target_path_or_skip() else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());

    // Repetitive / structured prompts so the n-gram proposer fires often (its
    // suffix recurs earlier in the context) and the verify path is exercised.
    let prompts = [
        "1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(",
        "The quick brown fox. The quick brown fox. The quick brown fox.",
        "Q: What is 17 multiplied by 23? A: Let's think step by step.",
        "{\"a\": 1, \"b\": 2, \"a\": 1, \"b\": 2, \"a\": 1, \"b\": 2,",
    ];

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    // 1. Baseline: plain greedy decode (speculative off), with logprobs so the
    //    regret check has the reference distribution at the divergence point.
    let baseline: Vec<Vec<Step>> = {
        let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(false))
            .expect("failed to start baseline engine");
        let out = encoded
            .iter()
            .map(|t| generate(&handle, t.clone(), LOGPROBS, GENERATED_TOKENS))
            .collect();
        drop(handle);
        // Free the target before the speculative engine loads the same weights.
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    // 2. Speculative: n-gram propose + verify (logprobs off ⇒ spec path active).
    //    Keep the engine alive through analysis: at a divergence it re-prefills
    //    the shared context to read the prefill-kernel reference.
    let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(true))
        .expect("failed to start speculative engine");

    let mut failures = Vec::new();
    for (i, &prompt) in prompts.iter().enumerate() {
        let spec = generate(&handle, encoded[i].clone(), 0, GENERATED_TOKENS);
        if let Err(failure) =
            check_lossless(&handle, &tokenizer, i, prompt, &encoded[i], &baseline[i], &spec)
        {
            failures.push(failure);
        }
    }

    drop(handle);

    assert!(
        failures.is_empty(),
        "n-gram speculative greedy decode is not lossless:\n{}",
        failures.join("\n")
    );
}

/// Concurrent, heterogeneous-`max_tokens` losslessness for the bs>1 verify path:
/// several greedy requests at staggered budgets run in one engine so the in-flight
/// batch mixes full and near-budget (truncated) verify spans. Each must stay
/// lossless vs its own plain-greedy baseline. A per-request context mix-up (the
/// proposer scanning the wrong request's history) or a bs>1 verify-span indexing
/// bug would surface here as a real (non-tie) divergence.
#[test]
fn ngram_concurrent_heterogeneous_is_lossless() {
    let Some(model_path) = target_path_or_skip() else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());

    let cases: [(&str, usize); 4] = [
        ("def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(", 64),
        ("1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4", 24),
        ("Q: What is 17 multiplied by 23? A: Let's think step by step.", 48),
        ("The quick brown fox. The quick brown fox. The quick brown fox.", 40),
    ];

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = cases
        .iter()
        .map(|(p, _)| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    // 1. Baselines: each prompt's plain-greedy decode (spec off) at ITS budget,
    //    with logprobs for the regret reference. Sequential, one engine.
    let baselines: Vec<Vec<Step>> = {
        let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(false))
            .expect("failed to start baseline engine");
        let out = encoded
            .iter()
            .zip(&cases)
            .map(|(t, (_, max_tokens))| generate(&handle, t.clone(), LOGPROBS, *max_tokens))
            .collect();
        drop(handle);
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    // 2. Speculative engine: submit all four at once so they form real batches.
    let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(true))
        .expect("failed to start speculative engine");
    let specs = generate_concurrent(
        &handle,
        encoded
            .iter()
            .zip(&cases)
            .map(|(t, (_, max_tokens))| (t.clone(), *max_tokens))
            .collect(),
    );

    let mut failures = Vec::new();
    for (i, (prompt, _)) in cases.iter().enumerate() {
        if let Err(failure) =
            check_lossless(&handle, &tokenizer, i, prompt, &encoded[i], &baselines[i], &specs[i])
        {
            failures.push(failure);
        }
    }
    drop(handle);

    assert!(
        failures.is_empty(),
        "concurrent heterogeneous n-gram speculative decode is not lossless:\n{}",
        failures.join("\n")
    );
}
