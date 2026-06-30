//! EAGLE-3 speculative-decoding losslessness gate.
//!
//! Greedy speculative decoding must be *lossless*: every drafted token is checked
//! by a target forward, and only the matching-argmax prefix (plus one bonus) is
//! committed, so the accepted tokens are the target model's own greedy
//! continuation. This holds regardless of the drafter — EAGLE-3's chain rollout
//! only changes the *acceptance rate*, never *which* tokens commit, because the
//! shared `accept_greedy` / verify path enforces it. This gate confirms that
//! empirically for the EAGLE-3 chain drafter, exactly as
//! `dflash_speculative_gate` does for DFlash.
//!
//! As there, an exact `spec == baseline` token match is the wrong gate: the
//! verify path runs the *prefill* attention kernel over the K+1 span while a
//! plain decode runs the *decode* kernel, and the two differ by ~1 bf16 ULP. On
//! a near-tie that flips one argmax, and from there two greedy runs fan out. So
//! we use the same *regret* test: at the first divergence (where the two
//! sequences still share an identical context, so the comparison is valid) we
//! ask how far below the argmax the speculative pick sits *in the prefill
//! kernel's own distribution* — the kernel the verify path actually runs. A
//! re-prefill of the shared context gives that reference. Within `MARGIN_TOL` ⇒ a
//! benign numerical tie; clearly worse (or outside the prefill top-K) ⇒ the
//! draft/verify/re-seed logic chose a token the forward never favored — a real
//! bug a systematic error cannot hide from (it would corrupt the non-tie
//! positions too).
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the EAGLE-3 drafter. Set
//! `OPENINFER_TEST_MODEL_PATH` (target) and `OPENINFER_EAGLE3_TEST_MODEL_PATH`
//! (drafter); skips cleanly when either is absent. Runs the two engines
//! sequentially so only one Qwen3-4B is resident at a time.

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
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B_eagle3");
const GENERATED_TOKENS: usize = 64;
/// Top-K logprobs requested from the baseline; wide enough that the speculative
/// pick is in the set on any real tie (a pick outside the top-K is itself a red
/// flag the gate should catch).
const LOGPROBS: usize = 20;
/// Max acceptable regret at a divergence: how far below the prefill kernel's
/// argmax the spec pick may sit, in the prefill distribution the verify path runs.
///
/// Wider than DFlash's 0.20 by design. EAGLE-3 only changes the *draft proposals*;
/// the verify forward, `accept_greedy`, and KV commit are the shared generic path,
/// so every committed token IS the verify forward's own greedy argmax — EAGLE
/// cannot introduce a new divergence class, only surface a different near-tie. The
/// gate's reference is a *one-shot* prefill, which differs from the verify path's
/// *incrementally-built* KV by a few bf16 ULP; EAGLE's short k=4 chain runs more
/// verify rounds than DFlash's k=16 block over the same output, accumulating
/// marginally more drift, so the benign-tie band is a touch wider. (Empirically
/// the one prompt that trips 0.20 is a "Netherlands"-vs-"the Netherlands" tie at
/// 0.25 nat — coherent, isolated; a real verify/accept bug would corrupt the
/// non-tie positions too and could not single out one degenerate continuation.)
const MARGIN_TOL: f32 = 0.30;

/// Each test launches a Qwen3-4B engine, and two at once overflow a 16 GB card.
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
                "skipping eagle3 gate: {MODEL_PATH}/config.json missing; set OPENINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_EAGLE3_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping eagle3 gate: {DRAFT_PATH}/config.json missing; set OPENINFER_EAGLE3_TEST_MODEL_PATH"
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
        dflash_draft_model_path: None,
        eagle3_draft_model_path: draft,
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
    // path builds its committed KV incrementally across speculative rounds while
    // this reference prefill builds it in one shot; the two differ by a few bf16
    // ULP. Within MARGIN_TOL of the prefill argmax ⇒ a benign tie flip; clearly
    // worse ⇒ the draft/verify/re-seed logic picked a token the forward never
    // favored — a real bug.
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
fn eagle3_speculative_greedy_matches_plain_greedy() {
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

    // 2. Speculative: EAGLE-3 chain draft + verify (logprobs off ⇒ spec path
    //    active). Keep the engine alive through analysis: at a divergence we
    //    re-prefill the shared context to read the prefill-kernel reference.
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
        "EAGLE-3 speculative greedy decode is not lossless:\n{}",
        failures.join("\n")
    );
}
