//! vLLM-golden accuracy gate for Kimi-K2 — the first gate for this model line
//! that a fresh clone can re-run from committed code and fixtures alone (#223).
//!
//! vLLM is the reference, not HuggingFace: Kimi-K2.6 is INT4
//! (compressed-tensors), and vLLM executes the same quantized model through
//! marlin kernels — the closest equal-precision reference available. The HF
//! route decompresses to bf16 (a different numerical regime) behind a fragile
//! trust_remote_code load. The fixture
//! (`test_data/kimi-k2.6-vllm-golden.safetensors`, produced once on the
//! serving hardware by `tools/accuracy/dump_kimi_k2_vllm_golden.py`) pins the
//! prompt token ids, vLLM's greedy tail, and vLLM's top-K logprobs per
//! position.
//!
//! The kimi engine exposes no logprobs yet (#236), so unlike the Qwen gates
//! this one cannot bound a |Δlogprob| distribution. It asserts what the public
//! engine surface can express, in two passes through the *real serving path*
//! (EngineHandle → DP coordinator → PPLX EP → MLA kernels, TP1/DP8/EP8):
//!
//!   * teacher-forced argmax sweep — for every position i, prefill
//!     `prompt + tail[..i]` with max_tokens=1. pegainfer's pick must satisfy
//!     the flatness-scaled regret rule (see `REGRET_BASE`): how far it may
//!     sit below vLLM's own argmax *in vLLM's logprobs* grows with vLLM's
//!     own uncertainty at that position — near-exact agreement where vLLM is
//!     confident, room for cross-engine noise where the distribution is flat
//!     and there is no single correct token. A pick vLLM ranks clearly worse
//!     — or not at all — is a real wrong-token bug, and an aggregate
//!     exact-match floor (`EXACT_FLOOR`) catches "many small flips" drift
//!     that passes position-by-position. Teacher-forcing means one flip
//!     cannot cascade: every position is independently comparable. This pass
//!     covers prefill numerics position-by-position.
//!   * free-greedy decode parity — generate the tail end-to-end and compare
//!     token-by-token. This is the only public way to exercise the *decode*
//!     kernels (MLA decode, batched PPLX MoE), at the cost that comparison
//!     stops at the first benign divergence: an in-bound mismatch ends that
//!     sequence's comparison (the engines walked into different contexts); a
//!     mismatch beyond the regret bound fails the gate. An
//!     aggregate coverage floor keeps mass early divergence from passing
//!     silently. Runs sequentially (bs=1), concurrently (DP8 routing +
//!     batched decode), and twice on one sequence (determinism: identical
//!     inputs must reproduce identical tokens).
//!
//! Requires 8 GPUs and Kimi-K2.6 weights. `PEGAINFER_TEST_MODEL_PATH` must
//! point at the weights and the fixture must exist — both fail loudly when
//! missing. No silent skip: a gate that can quietly report "ok 0.00s" guards
//! nothing (the qwen35 gate's env-gated skip taught us that). Building the
//! target at all requires the `kimi-k2` feature (`required-features` in
//! Cargo.toml), so feature-less workspace test runs never see it.

use std::path::Path;
use std::time::{Duration, Instant};

use pegainfer_core::engine::{EngineHandle, EngineLoadOptions, EpBackend, TokenEvent};
use pegainfer_core::parallel::ParallelConfig;
use pegainfer_core::sampler::SamplingParams;
use safetensors::{Dtype, SafeTensors};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/kimi-k2.6-vllm-golden.safetensors"
);

/// Per-position regret rule: pegainfer's pick must satisfy
///
///   regret ≤ REGRET_BASE + REGRET_FLATNESS_SLOPE × (−vllm_top1_lp)
///
/// where *regret* is how far the pick sits below vLLM's own argmax in
/// vLLM's logprobs. The allowance grows with vLLM's own uncertainty: at a
/// confident position (top-1 ≈ 90%) the bound is ≈ 0.34 nat — near-exact
/// agreement — while at a flat, multi-modal position (top-1 ≈ 11%) it
/// reaches ≈ 1.07, because there is no single correct token for
/// cross-engine noise to deviate from. The bound depends only on the
/// committed vLLM fixture, so pegainfer cannot influence its own
/// tolerance.
///
/// Calibration (three 8×H200 runs, 2026-06-05/06): cross-engine INT4
/// disagreements appeared exclusively at low-confidence positions and
/// scaled with flatness — regret 0.375 @ lp −1.50, 0.50 @ −0.85,
/// 0.625 @ −1.42, 1.00 @ −2.20 (each pick a top-4 vLLM token; the worst
/// is "invent the next fictional project name" where vLLM's top-8 bunch
/// within 1.8 nat). A fixed threshold either fails on these or, widened
/// to cover them, goes slack at confident positions; the linear-in-
/// flatness rule keeps ≤ ~2 grid notches (vLLM logprobs are 1/16-
/// quantized) of headroom over every observed point. The slope's tight
/// fit is the `json` deep flip: (1.00 − 0.30)/2.20 = 0.318, rounded up.
const REGRET_BASE: f32 = 0.30;
const REGRET_FLATNESS_SLOPE: f32 = 0.35;

/// Aggregate guard: per pass, the fraction of positions where pegainfer's
/// pick equals vLLM's argmax exactly. A systematic numerical bug shows up
/// as *many* small in-bound flips long before any single pick violates the
/// per-position rule — this floor catches that. Measured 97.7–98.4% across
/// all runs and passes; 0.95 leaves ~2.5 pp of headroom, so a doubling of
/// the flip rate fails the gate.
const EXACT_FLOOR: f64 = 0.95;

/// Free-greedy parity must compare at least this fraction of all tail
/// positions before benign divergences cut the sequences short. Guards
/// against "every sequence tie-flips at position 0" passing as vacuously
/// green. Measured 72–81% across runs and modes.
const COVERAGE_FLOOR: f64 = 0.70;

/// Per-request wait budget. Decode of a 32-token tail takes ~1 s at bs=64
/// TPOT; teacher-forced waves prefill up to 32 requests. A request that
/// produces nothing for this long is a hung engine — fail, don't hang the CI
/// job.
const RECV_TIMEOUT: Duration = Duration::from_mins(10);

struct Fixture {
    meta: Meta,
    seqs: Vec<Seq>,
}

struct Meta {
    vllm_version: String,
    model: String,
    decode_tokens: usize,
    top_k: usize,
}

struct Seq {
    name: String,
    prompt_token_ids: Vec<u32>,
    tail_token_ids: Vec<u32>,
    topk_ids: Vec<Vec<u32>>,
    topk_logprobs: Vec<Vec<f32>>,
}

impl Seq {
    /// vLLM's logprob for `token` at tail position `pos`, if ranked.
    fn vllm_logprob(&self, pos: usize, token: u32) -> Option<f32> {
        let idx = self.topk_ids[pos].iter().position(|&t| t == token)?;
        Some(self.topk_logprobs[pos][idx])
    }

    fn vllm_top1(&self, pos: usize) -> (u32, f32) {
        (self.topk_ids[pos][0], self.topk_logprobs[pos][0])
    }
}

fn as_i32(st: &SafeTensors, name: &str) -> (Vec<i32>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("fixture missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::I32, "{name} must be i32");
    let v = t
        .data()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (v, t.shape().to_vec())
}

fn as_f32(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("fixture missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::F32, "{name} must be f32");
    let v = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (v, t.shape().to_vec())
}

fn load_fixture() -> Fixture {
    let bytes = std::fs::read(FIXTURE).unwrap_or_else(|e| {
        panic!(
            "kimi vllm_golden_gate: fixture {FIXTURE} is missing ({e}); \
             generate it with tools/accuracy/dump_kimi_k2_vllm_golden.py"
        )
    });
    let (_, header) = SafeTensors::read_metadata(&bytes).expect("parse fixture header");
    let kv = header
        .metadata()
        .clone()
        .expect("fixture has no __metadata__");
    assert_eq!(kv["reference"], "vllm", "fixture is not vLLM-golden");
    let meta = Meta {
        vllm_version: kv["vllm_version"].clone(),
        model: kv["model"].clone(),
        decode_tokens: kv["decode_tokens"].parse().expect("decode_tokens"),
        top_k: kv["top_k"].parse().expect("top_k"),
    };
    let names: Vec<&str> = kv["seq_names"].split(',').collect();

    let st = SafeTensors::deserialize(&bytes).expect("parse fixture safetensors");
    let (prompt_tokens, _) = as_i32(&st, "prompt_tokens");
    let (prompt_lens, _) = as_i32(&st, "prompt_lens");
    let (tails, tail_shape) = as_i32(&st, "tail_tokens");
    let (ids, ids_shape) = as_i32(&st, "topk_ids");
    let (lps, lp_shape) = as_f32(&st, "topk_logprobs");

    let (s, d, k) = (names.len(), meta.decode_tokens, meta.top_k);
    assert_eq!(prompt_lens.len(), s);
    assert_eq!(tail_shape, [s, d]);
    assert_eq!(ids_shape, [s, d, k]);
    assert_eq!(lp_shape, [s, d, k]);

    let mut seqs = Vec::with_capacity(s);
    let mut off = 0usize;
    for (i, name) in names.into_iter().enumerate() {
        let plen = prompt_lens[i] as usize;
        let prompt_token_ids: Vec<u32> = prompt_tokens[off..off + plen]
            .iter()
            .map(|&t| t as u32)
            .collect();
        off += plen;
        let tail_token_ids: Vec<u32> = tails[i * d..(i + 1) * d]
            .iter()
            .map(|&t| t as u32)
            .collect();
        let topk_ids: Vec<Vec<u32>> = (0..d)
            .map(|p| {
                let base = (i * d + p) * k;
                ids[base..base + k].iter().map(|&t| t as u32).collect()
            })
            .collect();
        let topk_logprobs: Vec<Vec<f32>> = (0..d)
            .map(|p| {
                let base = (i * d + p) * k;
                lps[base..base + k].to_vec()
            })
            .collect();
        // Greedy reference: the tail token must be vLLM's own argmax.
        for (pos, &tok) in tail_token_ids.iter().enumerate() {
            assert_eq!(tok, topk_ids[pos][0], "{name} pos {pos}");
        }
        seqs.push(Seq {
            name: name.to_string(),
            prompt_token_ids,
            tail_token_ids,
            topk_ids,
            topk_logprobs,
        });
    }
    assert_eq!(off, prompt_tokens.len(), "ragged prompt_tokens mismatch");
    Fixture { meta, seqs }
}

fn model_path() -> String {
    let path = std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| {
        panic!(
            "kimi vllm_golden_gate: PEGAINFER_TEST_MODEL_PATH is not set. \
             This gate needs 8 GPUs and Kimi-K2.6 weights; it fails rather \
             than silently skipping."
        )
    });
    assert!(
        Path::new(&path).join("config.json").exists(),
        "kimi vllm_golden_gate: {path}/config.json does not exist"
    );
    path
}

fn start_engine(path: &str) -> EngineHandle {
    pegainfer_kimi_k2::start_engine(
        Path::new(path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: (0..8).collect(),
            parallel_config: Some(ParallelConfig::new(1, 8)),
            ep_backend: EpBackend::Pplx,
            seed: 42,
        },
    )
    .expect("start kimi-k2 TP1/DP8/EP8 PPLX engine")
}

struct PendingRequest {
    label: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<TokenEvent>,
}

fn submit(
    engine: &EngineHandle,
    label: String,
    prompt: &[u32],
    max_tokens: usize,
) -> PendingRequest {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    engine
        .submit(pegainfer_core::engine::GenerateRequest {
            request_id: Some(label.clone()),
            queued_at_unix_s: None,
            prompt_tokens: prompt.to_vec(),
            params: SamplingParams {
                temperature: 0.0,
                top_k: -1,
                top_p: 1.0,
                ignore_eos: true,
            },
            max_tokens,
            lora_adapter: None,
            token_tx: tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit to kimi engine");
    PendingRequest { label, rx }
}

impl PendingRequest {
    /// Drain the event stream to completion, returning the generated tokens.
    /// Any engine-side error or a stall beyond `RECV_TIMEOUT` is a loud fail.
    fn collect(mut self) -> Vec<u32> {
        let mut tokens = Vec::new();
        let deadline = Instant::now() + RECV_TIMEOUT;
        loop {
            match self.rx.try_recv() {
                Ok(TokenEvent::Token { id, .. }) => tokens.push(id),
                Ok(TokenEvent::Finished { .. }) => return tokens,
                Ok(TokenEvent::Error { message, .. }) => {
                    panic!("[{}] engine error: {message}", self.label)
                }
                Ok(TokenEvent::Rejected { message, .. }) => {
                    panic!("[{}] request rejected: {message}", self.label)
                }
                Ok(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                    assert!(
                        Instant::now() < deadline,
                        "[{}] no token event within {RECV_TIMEOUT:?} — engine hung",
                        self.label
                    );
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    panic!(
                        "[{}] engine dropped the request channel without Finished",
                        self.label
                    )
                }
            }
        }
    }
}

#[derive(Default)]
struct RegretStats {
    positions: usize,
    exact_matches: usize,
    flips: Vec<f32>, // regret of in-bound disagreements
    violations: Vec<String>,
}

/// Fold pegainfer's pick at one position into the stats, applying the
/// flatness-scaled regret rule.
fn check_pick(stats: &mut RegretStats, seq: &Seq, pos: usize, pick: u32) {
    stats.positions += 1;
    let (vllm_top1, vllm_top1_lp) = seq.vllm_top1(pos);
    if pick == vllm_top1 {
        stats.exact_matches += 1;
        return;
    }
    match seq.vllm_logprob(pos, pick) {
        None => stats.violations.push(format!(
            "{} pos {pos}: pegainfer picked {pick}, absent from vLLM's top-{} \
             (vLLM argmax {vllm_top1}) — confidently wrong on a token vLLM does not rank",
            seq.name,
            seq.topk_ids[pos].len(),
        )),
        Some(lp) => {
            let regret = vllm_top1_lp - lp;
            let bound = REGRET_BASE + REGRET_FLATNESS_SLOPE * (-vllm_top1_lp);
            if regret <= bound {
                stats.flips.push(regret);
            } else {
                stats.violations.push(format!(
                    "{} pos {pos}: pegainfer picked {pick}, which vLLM scores \
                     {regret:.4} nat below its argmax {vllm_top1} (top-1 lp \
                     {vllm_top1_lp:.2}, bound {bound:.4})",
                    seq.name,
                ));
            }
        }
    }
}

/// Print a pass summary and fold its failures into `failures`. The test
/// asserts once at the end, so every pass still contributes calibration
/// data when an earlier pass has violations.
fn report(label: &str, stats: &RegretStats, failures: &mut Vec<String>) {
    let mut flips = stats.flips.clone();
    flips.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let exact_rate = stats.exact_matches as f64 / stats.positions as f64;
    eprintln!(
        "vllm_golden_gate [{label}]: {} positions — {} exact ({:.1}%), \
         {} in-bound flips (max regret {:.4}), {} violations",
        stats.positions,
        stats.exact_matches,
        exact_rate * 100.0,
        flips.len(),
        flips.last().copied().unwrap_or(0.0),
        stats.violations.len(),
    );
    for v in &stats.violations {
        failures.push(format!("[{label}] {v}"));
    }
    if exact_rate < EXACT_FLOOR {
        failures.push(format!(
            "[{label}] exact-match rate {exact_rate:.3} is below the \
             {EXACT_FLOOR} floor — many small flips is systematic drift \
             even when every one is in-bound",
        ));
    }
}

/// Pass A: teacher-forced argmax sweep. One wave per sequence: D concurrent
/// single-token requests at prompt lengths `len(prompt) .. len(prompt)+D-1`.
/// Mixed-length concurrent prefill is exactly the DP-coordinator admission
/// shape the PPLX path must handle (uneven per-rank rows, empty ranks).
fn teacher_forced_sweep(engine: &EngineHandle, fixture: &Fixture) -> RegretStats {
    let mut stats = RegretStats::default();
    for seq in &fixture.seqs {
        let pending: Vec<PendingRequest> = (0..fixture.meta.decode_tokens)
            .map(|i| {
                let mut prompt = seq.prompt_token_ids.clone();
                prompt.extend_from_slice(&seq.tail_token_ids[..i]);
                submit(engine, format!("tf:{}:{i}", seq.name), &prompt, 1)
            })
            .collect();
        for (i, req) in pending.into_iter().enumerate() {
            let tokens = req.collect();
            assert_eq!(
                tokens.len(),
                1,
                "tf:{}:{i} returned {} tokens, expected 1",
                seq.name,
                tokens.len()
            );
            check_pick(&mut stats, seq, i, tokens[0]);
        }
    }
    stats
}

/// Pass B: free-greedy decode parity over one set of sequences. Returns the
/// per-sequence compared-position counts alongside the regret stats.
fn greedy_parity(
    engine: &EngineHandle,
    fixture: &Fixture,
    concurrent: bool,
) -> (RegretStats, usize) {
    let mut stats = RegretStats::default();
    let mut compared = 0usize;

    let outputs: Vec<(usize, Vec<u32>)> = if concurrent {
        let pending: Vec<PendingRequest> = fixture
            .seqs
            .iter()
            .map(|seq| {
                submit(
                    engine,
                    format!("greedy:{}", seq.name),
                    &seq.prompt_token_ids,
                    fixture.meta.decode_tokens,
                )
            })
            .collect();
        pending
            .into_iter()
            .enumerate()
            .map(|(i, req)| (i, req.collect()))
            .collect()
    } else {
        fixture
            .seqs
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let req = submit(
                    engine,
                    format!("greedy:{}", seq.name),
                    &seq.prompt_token_ids,
                    fixture.meta.decode_tokens,
                );
                (i, req.collect())
            })
            .collect()
    };

    for (i, tokens) in outputs {
        let seq = &fixture.seqs[i];
        assert_eq!(
            tokens.len(),
            fixture.meta.decode_tokens,
            "greedy:{}: got {} tokens, expected {} (ignore_eos was set)",
            seq.name,
            tokens.len(),
            fixture.meta.decode_tokens
        );
        for (pos, &tok) in tokens.iter().enumerate() {
            check_pick(&mut stats, seq, pos, tok);
            compared += 1;
            if tok != seq.tail_token_ids[pos] {
                // The engines now sit in different contexts; later positions
                // are incomparable. check_pick already classified this token
                // as benign tie-flip or violation.
                eprintln!(
                    "vllm_golden_gate: greedy:{} diverged at pos {pos}/{} \
                     (pegainfer {tok}, vLLM {})",
                    seq.name, fixture.meta.decode_tokens, seq.tail_token_ids[pos],
                );
                break;
            }
        }
    }
    (stats, compared)
}

#[test]
fn kimi_greedy_matches_vllm_golden() {
    let fixture = load_fixture();
    let path = model_path();
    eprintln!(
        "vllm_golden_gate: {} seqs x {} positions, reference vLLM {} on {}",
        fixture.seqs.len(),
        fixture.meta.decode_tokens,
        fixture.meta.vllm_version,
        fixture.meta.model,
    );
    let engine = start_engine(&path);
    let total = fixture.seqs.len() * fixture.meta.decode_tokens;
    let mut failures = Vec::new();

    // Pass A — prefill numerics, position-independent.
    let stats = teacher_forced_sweep(&engine, &fixture);
    assert_eq!(stats.positions, total);
    report("teacher-forced sweep", &stats, &mut failures);

    // Pass B — decode-path parity, sequential then concurrent.
    let (seq_stats, seq_compared) = greedy_parity(&engine, &fixture, false);
    report(
        &format!("greedy parity bs=1 ({seq_compared}/{total} positions compared)"),
        &seq_stats,
        &mut failures,
    );
    let (con_stats, con_compared) = greedy_parity(&engine, &fixture, true);
    report(
        &format!("greedy parity concurrent ({con_compared}/{total} positions compared)"),
        &con_stats,
        &mut failures,
    );
    for (label, compared) in [("bs=1", seq_compared), ("concurrent", con_compared)] {
        let coverage = compared as f64 / total as f64;
        if coverage < COVERAGE_FLOOR {
            failures.push(format!(
                "greedy parity {label}: only {compared}/{total} positions \
                 ({coverage:.2}) compared before divergence — below the \
                 {COVERAGE_FLOOR} floor, the parity signal is too thin"
            ));
        }
    }

    // Determinism: identical input must reproduce identical tokens. Catches
    // nondeterministic kernels and uninitialised scratch independently of the
    // vLLM reference.
    let seq = &fixture.seqs[0];
    let a = submit(
        &engine,
        format!("det:{}:a", seq.name),
        &seq.prompt_token_ids,
        fixture.meta.decode_tokens,
    )
    .collect();
    let b = submit(
        &engine,
        format!("det:{}:b", seq.name),
        &seq.prompt_token_ids,
        fixture.meta.decode_tokens,
    )
    .collect();
    if a != b {
        failures.push(format!(
            "det:{}: identical inputs produced different token streams",
            seq.name
        ));
    }

    assert!(
        failures.is_empty(),
        "vllm_golden_gate failed:\n  {}",
        failures.join("\n  ")
    );
}
