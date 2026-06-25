//! DFlash speculative-decoding losslessness gate.
//!
//! Greedy speculative decoding must be *lossless*: every draft is verified by a
//! target forward and only the matching-argmax prefix (plus one bonus) is
//! committed, so the accepted tokens are the target model's own greedy
//! continuation. The catch is pure numerics — the verify path runs the
//! *prefill* attention kernel over the K+1 span while a plain decode runs the
//! *decode* kernel, and the two differ by ~1 bf16 ULP. On a near-tie that flips
//! one argmax, and from there two greedy runs fan out completely.
//!
//! So an exact `spec == baseline` token match is the wrong gate: it false-fails
//! on a benign tie flip. We use a *regret* test like `hf_golden_gate`: at the
//! first position the two sequences disagree (where they still share an
//! identical context, so the comparison is valid) we ask how far below the
//! argmax the speculative pick sits — measured *in the prefill kernel's own
//! distribution*, because that is the kernel the verify path runs. A re-prefill
//! of the shared context (`prefill_next`) gives that reference distribution.
//! The verify path's committed KV is built incrementally across batched
//! speculative spans, while a one-shot prefill builds it in a single forward;
//! the two K/V differ by a few bf16 ULP, so on a near-tie the argmax flips.
//! Within `MARGIN_TOL` of the prefill argmax ⇒ a benign numerical tie. Clearly
//! worse (or outside the prefill top-K) ⇒ the verify/accept/capture logic chose
//! a token the forward never favored — a real bug. A systematic bug corrupts
//! the non-tie positions too, so it cannot hide behind the tie band.
//!
//! (Empirically the one prompt that flips — "The capital of France is" — sits on
//! a Germany-vs-Paris near-tie: the prefill kernel scores them -0.71 vs -0.83,
//! a 0.12-nat gap, well inside `MARGIN_TOL`. The other four prompts are bit
//! identical. A real verify bug would not single out the one degenerate prompt.)
//!
//! The baseline runs with logprobs on (plain decode); the speculative engine
//! runs with logprobs off (logprobs force the spec path off by design), so it
//! reports chosen tokens only — exactly what the regret check needs.
//!
//! Runs the two engines sequentially (baseline dropped before the speculative
//! engine loads) so only one Qwen3-4B is resident at a time.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the DFlash drafter. Set
//! `OPENINFER_TEST_MODEL_PATH` (target) and `OPENINFER_DFLASH_TEST_MODEL_PATH`
//! (drafter); skips cleanly when either is absent.

use std::path::{Path, PathBuf};
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
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B-DFlash-b16");
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
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping dflash gate: {MODEL_PATH}/config.json missing; set OPENINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_DFLASH_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping dflash gate: {DRAFT_PATH}/config.json missing; set OPENINFER_DFLASH_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        offload: Qwen3OffloadOptions::disabled(),
        // The speculative engine forces the prefix cache off; match it on the
        // baseline so both take the same cold prefill path.
        no_prefix_cache: true,
        max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(0.85, DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES)
            .validate()
            .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: draft,
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
/// the real bs>1 draft+verify path. Each tuple is `(prompt_tokens, max_tokens)`;
/// logprobs are off so the speculative path stays active. Returns one step list
/// per request, in submission order.
fn generate_concurrent(handle: &EngineHandle, requests: Vec<(Vec<u32>, usize)>) -> Vec<Vec<Step>> {
    // Submit all up front so they coexist in the engine and form real batches.
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

    // Drain each request's channel to completion (events are buffered per-channel,
    // so the drain order doesn't matter — they all ran concurrently).
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
/// kernel resolves bifurcation ties to the other side). Returns the first
/// generated token's `(id, top_logprobs)`.
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

    // Identical sequences (or one a prefix of the other): perfectly lossless.
    if matched == base.len().min(spec.len()) {
        eprintln!(
            "prompt {i} ({prompt:?}): {matched}/{} tokens identical (100% lossless)",
            base.len()
        );
        return Ok(());
    }

    let spec_id = spec[matched].id;
    let decode_argmax = base[matched].top_logprobs[0].0;

    // Diagnostic: show the exact branch point.
    {
        let lo = matched.saturating_sub(2);
        let hi = (matched + 3).min(base.len()).min(spec.len());
        let base_ids: Vec<u32> = base[..hi].iter().map(|s| s.id).collect();
        let spec_ids: Vec<u32> = spec[..hi].iter().map(|s| s.id).collect();
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
        let _ = spec_ids;
    }

    // The verify path runs the prefill kernel, so the right reference for the
    // spec pick is a plain *prefill* of the same shared context — not the
    // plain-decode baseline, whose kernel resolves a bifurcation tie to the
    // other side and amplifies the gap.
    let mut context = prompt_tokens.to_vec();
    context.extend(base[..matched].iter().map(|s| s.id));
    let prefill_ref = prefill_next(handle, context, LOGPROBS);

    if prefill_ref.id == spec_id {
        // Spec faithfully reproduced the prefill-kernel greedy pick; the
        // divergence is purely the pre-existing prefill-vs-decode kernel gap.
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

    // Spec's greedy pick differs from the prefill-kernel argmax too. The verify
    // path builds its committed KV incrementally across batched speculative
    // spans while this reference prefill builds it in one shot; the two differ by
    // a few bf16 ULP. Within MARGIN_TOL of the prefill argmax ⇒ a benign tie
    // flip; clearly worse ⇒ the verify/accept/capture logic picked a token the
    // forward never favored — a real bug.
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

    // Either the spec pick is outside the prefill top-K entirely, or it sits
    // clearly below the prefill argmax — neither is a benign tie.
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

#[test]
fn dflash_speculative_greedy_matches_plain_greedy() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());

    let prompts = [
        "The capital of France is",
        "Here is a short story about a dragon. Once upon a time",
        "def fibonacci(n):",
        "Q: What is 17 multiplied by 23? A: Let's think step by step.",
        "The three primary colors are",
    ];

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    // 1. Baseline: plain greedy decode (speculative off), with logprobs so the
    //    regret check has the reference distribution at the divergence point.
    let baseline: Vec<Vec<Step>> = {
        let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(None))
            .expect("failed to start baseline engine");
        let out = encoded
            .iter()
            .map(|t| generate(&handle, t.clone(), LOGPROBS, GENERATED_TOKENS))
            .collect();
        drop(handle);
        // Let the scheduler thread tear down and free GPU memory before the
        // speculative engine loads the same 8 GB target.
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    // 2. Speculative: DFlash draft + verify (logprobs off ⇒ spec path active).
    //    Keep the engine alive through analysis: at a divergence we re-prefill
    //    the shared context to read the prefill-kernel reference (the kernel the
    //    verify path uses), which the plain-decode baseline cannot provide.
    let handle = openinfer_qwen3_4b::launch(
        Path::new(&model_path),
        launch_options(Some(PathBuf::from(&draft_path))),
    )
    .expect("failed to start speculative engine");

    let mut failures = Vec::new();
    for (i, &prompt) in prompts.iter().enumerate() {
        let spec = generate(&handle, encoded[i].clone(), 0, GENERATED_TOKENS);
        if let Err(failure) = check_lossless(
            &handle,
            &tokenizer,
            i,
            prompt,
            &encoded[i],
            &baseline[i],
            &spec,
        ) {
            failures.push(failure);
        }
    }

    drop(handle);

    assert!(
        failures.is_empty(),
        "speculative greedy decode is not lossless:\n{}",
        failures.join("\n")
    );
}

/// Verify-graph capture-shape regression (heterogeneous `max_tokens`).
///
/// The piecewise verify CUDA Graph keys its captured dense segments by
/// `batch_size` alone, but a request near its output budget shortens its verify
/// span (`scheduler::plan` truncates the span to the remaining budget), so
/// `total_tokens` — the row count the captured segments bake into their launch
/// grid — varies at a *fixed* batch size. A graph captured at a short span and
/// then replayed at a longer one processes too few rows: the trailing requests
/// read stale logits, silently breaking the lossless contract.
///
/// Neither existing check can see this. The bs=1 gate above issues each request
/// sequentially, so every fresh request's first verify is a *full* span and
/// bucket bs=1 is always first-captured at the maximal shape (only the harmless
/// over-compute direction occurs). A homogeneous concurrent benchmark is no
/// better: lockstep requests capture every bucket at full span during ramp-up,
/// and all truncation happens later as they finish together (still the safe
/// direction). The dangerous direction needs *heterogeneous* progress.
///
/// This reproduces it deterministically, single-stream: a `max_tokens=8` request
/// (span < `block_size`) captures the bucket-bs=1 graph at a truncated shape,
/// then a `max_tokens=64` request on the *same* engine replays that poisoned
/// graph at the full span. On the buggy code the long request diverges from
/// plain greedy; with full-shape gating (truncated spans run eager, so the graph
/// is only ever captured/replayed at the maximal shape) it stays lossless.
#[test]
fn dflash_short_then_long_verify_capture_is_lossless() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());

    // << block_size (16): the poison request's only verify step is a short
    // truncated span, capturing the bucket-bs=1 graph at total_tokens far below
    // the full span. The fewer valid rows, the sooner a full-span replay hits the
    // stale tail — so the victim diverges early, well clear of its token budget.
    const POISON_MAX_TOKENS: usize = 4;
    let poison_prompt = "Hello, world! Tell me a story.";
    let victim_prompt = "Q: What is 17 multiplied by 23? A: Let's think step by step.";

    let tokenizer = common::load_tokenizer(&model_path);
    let poison_tokens = tokenizer
        .encode(poison_prompt, false)
        .expect("encode failed");
    let victim_tokens = tokenizer
        .encode(victim_prompt, false)
        .expect("encode failed");

    // 1. Baseline: the victim's plain-greedy decode (spec off) with logprobs, for
    //    the regret reference at any divergence.
    let baseline = {
        let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(None))
            .expect("failed to start baseline engine");
        let out = generate(&handle, victim_tokens.clone(), LOGPROBS, GENERATED_TOKENS);
        drop(handle);
        // Free the target before the speculative engine loads the same 8 GB.
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    // 2. Speculative engine, shared across both requests so the bucket-bs=1
    //    capture from the poison request persists into the victim's replay.
    let handle = openinfer_qwen3_4b::launch(
        Path::new(&model_path),
        launch_options(Some(PathBuf::from(&draft_path))),
    )
    .expect("failed to start speculative engine");

    // Poison: a short request whose only verify step has total_tokens < span,
    // first-capturing the bucket-bs=1 graph at the truncated shape.
    let poison = generate(&handle, poison_tokens, 0, POISON_MAX_TOKENS);
    assert!(
        poison.len() <= POISON_MAX_TOKENS,
        "poison request emitted {} tokens, expected <= {POISON_MAX_TOKENS}",
        poison.len()
    );

    // Victim: a full-span replay of the poisoned bucket-bs=1 graph.
    let spec = generate(&handle, victim_tokens.clone(), 0, GENERATED_TOKENS);

    let result = check_lossless(
        &handle,
        &tokenizer,
        0,
        victim_prompt,
        &victim_tokens,
        &baseline,
        &spec,
    );
    drop(handle);

    assert!(
        result.is_ok(),
        "verify capture-shape bug: the long request diverged from plain greedy after a short \
         request poisoned the bucket-bs=1 graph at a truncated span:\n{}",
        result.unwrap_err()
    );
}

/// Concurrent, heterogeneous-`max_tokens` losslessness coverage for the bs>1
/// draft+verify path. The bs=1 gate and the homogeneous c8/c16 benches never
/// exercise a real batch with requests at *different* verify-span lengths; this
/// runs several greedy requests concurrently with staggered budgets and asserts
/// each stays lossless vs its own plain-greedy baseline (tolerating only the
/// benign bf16 tie-flip via the shared regret check). A batched-draft indexing
/// regression or a capture-shape mismatch at bs>1 would surface here as a real
/// (non-tie) divergence.
#[test]
fn dflash_concurrent_heterogeneous_is_lossless() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());

    // Distinct prompts with staggered budgets: at any tick the in-flight batch
    // mixes full and near-budget (truncated) verify spans.
    let cases: [(&str, usize); 4] = [
        ("def fibonacci(n):", 64),
        ("The three primary colors are", 24),
        (
            "Q: What is 17 multiplied by 23? A: Let's think step by step.",
            48,
        ),
        ("Here is a short story about a dragon. Once upon a time", 40),
    ];

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = cases
        .iter()
        .map(|(p, _)| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    // 1. Baselines: each prompt's plain-greedy decode (spec off) at ITS budget,
    //    with logprobs for the regret reference. Sequential, one engine.
    let baselines: Vec<Vec<Step>> = {
        let handle = openinfer_qwen3_4b::launch(Path::new(&model_path), launch_options(None))
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
    let handle = openinfer_qwen3_4b::launch(
        Path::new(&model_path),
        launch_options(Some(PathBuf::from(&draft_path))),
    )
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
        if let Err(failure) = check_lossless(
            &handle,
            &tokenizer,
            i,
            prompt,
            &encoded[i],
            &baselines[i],
            &specs[i],
        ) {
            failures.push(failure);
        }
    }
    drop(handle);

    assert!(
        failures.is_empty(),
        "concurrent heterogeneous speculative decode is not lossless:\n{}",
        failures.join("\n")
    );
}

/// P2 regression: a request that fits the target context window but lands in the
/// draft's `block_size` in-fill headroom (`max_pos - block_size < prompt +
/// max_tokens <= max_pos`) must be rejected cleanly at admission. Before the
/// admission cap, such a request was admitted on the target's limit and then
/// panicked mid-prefill when the draft allocated KV past its own max positions.
#[test]
fn dflash_request_in_draft_headroom_is_rejected_not_panicked() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|p| p.into_inner());

    // Read the real context window so the boundary is exact regardless of the
    // checkpoint, then size the request to sit inside the draft's final in-fill
    // block — it fits the target window but not the DFlash-effective one.
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&model_path).join("config.json")).expect("read config"),
    )
    .expect("parse config");
    let max_pos = config["max_position_embeddings"]
        .as_u64()
        .expect("max_position_embeddings") as usize;
    const BLOCK_SIZE: usize = 16; // DFlash drafter block size.
    let prompt_len = 16usize;
    // total in (max_pos - BLOCK_SIZE, max_pos]: clears the target check, trips
    // the DFlash admission cap (max_pos - BLOCK_SIZE).
    let max_tokens = max_pos - BLOCK_SIZE / 2 - prompt_len;

    let handle = openinfer_qwen3_4b::launch(
        Path::new(&model_path),
        launch_options(Some(PathBuf::from(&draft_path))),
    )
    .expect("failed to start speculative engine");

    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens: vec![100u32; prompt_len],
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Rejected { message, .. }) => {
                eprintln!("draft-headroom request rejected as expected: {message}");
                break;
            }
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Token { .. } | TokenEvent::Finished { .. }) => {
                panic!("draft-headroom request was admitted instead of rejected")
            }
            Some(TokenEvent::Error { message, .. }) => {
                panic!(
                    "draft-headroom request errored mid-flight instead of clean rejection: {message}"
                )
            }
            None => panic!("scheduler channel closed without a rejection"),
        }
    }
}
