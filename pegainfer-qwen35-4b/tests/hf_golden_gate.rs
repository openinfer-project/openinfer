//! HuggingFace-golden logits gate for Qwen3.5-4B.
//!
//! This is the Qwen3.5 instance of the reusable logits-golden method in
//! `docs/subsystems/correctness/logits-golden-gate.md`: store HF bf16 top-K
//! logprobs for fixed teacher-forced token sequences, replay those sequences
//! through pegainfer, and compare bounded logprob drift instead of exact text.
//! The Qwen3.5 fixture is produced through HF's incremental `past_key_values`
//! path so the oracle matches pegainfer's prefill + decode shape.
//!
//! Qwen3.5 currently has no eager batched decode path; decode goes through the
//! CUDA-graph bucketed path. This gate therefore covers sequential bs=1 and two
//! bucket-straddling batched graph passes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pegainfer_core::engine::TokenLogprob;
use pegainfer_qwen35_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen35Executor, RequestId,
};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/qwen35-4b-hf-golden.safetensors"
);

const LOGPROBS: usize = 64;
const MAX_EXECUTOR_BATCH: usize = 8;

// Initial Qwen3.5 gate scope is deliberately small: the goal is a portable HF
// oracle slice, not broad regression coverage. Regenerate + recalibrate through
// `tools/accuracy/dump_qwen35_4b_hf_golden.py` when widening the fixture.
const MARGIN_TOL: f32 = 0.20;
const MEAN_TOL: f32 = 0.06;
const P99_TOL: f32 = 0.20;
const HEAD_K: usize = 8;

const BUCKET_STRADDLES: [usize; 2] = [5, 3];

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 hf_golden_gate: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn sha256_file(path: impl AsRef<Path>) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    Some(
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    )
}

fn safetensors_metadata(bytes: &[u8]) -> HashMap<String, String> {
    let header_len_bytes: [u8; 8] = bytes[..8]
        .try_into()
        .expect("safetensors file missing 8-byte header length");
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let header = &bytes[8..8 + header_len];
    let value: serde_json::Value =
        serde_json::from_slice(header).expect("parse safetensors JSON header");
    value
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .map(|metadata| {
            metadata
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn model_revision(model_path: &str) -> Option<String> {
    if let Ok(value) = std::env::var("PEGAINFER_TEST_MODEL_REVISION") {
        return Some(value);
    }
    let path = Path::new(model_path);
    let metadata_path = path
        .join(".cache")
        .join("huggingface")
        .join("download")
        .join("config.json.metadata");
    if let Ok(content) = std::fs::read_to_string(metadata_path) {
        if let Some(first) = content.lines().next().map(str::trim) {
            if !first.is_empty() {
                return Some(first.to_string());
            }
        }
    }
    if path.join(".git").exists() {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(path)
            .arg("rev-parse")
            .arg("HEAD")
            .output()
            .ok()?;
        if output.status.success() {
            return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
        }
    }
    let parts: Vec<_> = path.components().collect();
    for window in parts.windows(2) {
        if window[0].as_os_str() == "snapshots" {
            return Some(window[1].as_os_str().to_string_lossy().to_string());
        }
    }
    None
}

fn require_metadata<'a>(metadata: &'a HashMap<String, String>, key: &str) -> &'a str {
    metadata
        .get(key)
        .unwrap_or_else(|| panic!("qwen35 hf_golden_gate fixture missing metadata key {key}"))
}

fn assert_metadata_eq(metadata: &HashMap<String, String>, key: &str, expected: &str) {
    let actual = require_metadata(metadata, key);
    assert_eq!(
        actual, expected,
        "qwen35 hf_golden_gate fixture metadata {key} mismatch; regenerate the fixture"
    );
}

fn check_fixture_metadata(model_path: &str, metadata: &HashMap<String, String>) -> bool {
    assert_metadata_eq(metadata, "fixture_kind", "hf-logits-golden");
    assert_metadata_eq(metadata, "model", "Qwen3.5-4B");
    assert_metadata_eq(metadata, "dtype", "bfloat16");
    assert_metadata_eq(
        metadata,
        "backend",
        "HuggingFace AutoModelForCausalLM eval, use_cache=True",
    );
    assert_metadata_eq(
        metadata,
        "forward_path",
        "prompt prefill, then one-token teacher-forced decode through past_key_values",
    );
    assert_metadata_eq(metadata, "top_k", &LOGPROBS.to_string());
    assert_metadata_eq(metadata, "margin_tol", &format!("{MARGIN_TOL:.2}"));
    assert_metadata_eq(metadata, "mean_tol", &format!("{MEAN_TOL:.2}"));
    assert_metadata_eq(metadata, "p99_tol", &format!("{P99_TOL:.2}"));

    let config = PathBuf::from(model_path).join("config.json");
    let actual_config_sha256 = sha256_file(&config).unwrap_or_else(|| {
        panic!(
            "qwen35 hf_golden_gate cannot read local config for metadata check: {}",
            config.display()
        )
    });
    assert_eq!(
        actual_config_sha256,
        require_metadata(metadata, "config_sha256"),
        "qwen35 hf_golden_gate config.json hash mismatch; regenerate the fixture for this model/config revision"
    );

    let expected_revision = require_metadata(metadata, "model_revision");
    assert_ne!(
        expected_revision, "unknown",
        "qwen35 hf_golden_gate fixture must record a pinned model_revision"
    );
    let Some(actual_revision) = model_revision(model_path) else {
        eprintln!(
            "skipping qwen35 hf_golden_gate: fixture requires model_revision={expected_revision}, but local model revision is unknown"
        );
        return false;
    };
    assert_eq!(
        actual_revision, expected_revision,
        "qwen35 hf_golden_gate model revision mismatch; set PEGAINFER_TEST_MODEL_REVISION or use the fixture's model snapshot"
    );

    if let Some(expected_tokenizer_revision) = metadata.get("tokenizer_revision") {
        assert_ne!(
            expected_tokenizer_revision, "unknown",
            "qwen35 hf_golden_gate fixture must record a pinned tokenizer_revision"
        );
    }
    true
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

fn as_f32(st: &SafeTensors, name: &str) -> Vec<f32> {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("golden missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::F32, "{name} must be f32");
    t.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn top_logprobs(lp: Option<&TokenLogprob>) -> Vec<(u32, f32)> {
    lp.expect("logprobs requested but none returned")
        .top_logprobs
        .clone()
}

#[derive(Default)]
struct Stats {
    positions: usize,
    argmax_violations: Vec<String>,
    head_deltas: Vec<f32>,
    worst: Option<(f32, usize, usize, u32, f32, f32)>,
}

fn check_position(
    stats: &mut Stats,
    seq: usize,
    pos: usize,
    pega: &[(u32, f32)],
    hf: &[(u32, f32)],
) {
    stats.positions += 1;
    let hf_top = hf[0].1;
    let pega_argmax = pega[0].0;
    let hf_map: HashMap<u32, f32> = hf.iter().copied().collect();

    match hf_map.get(&pega_argmax) {
        None => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: pegainfer argmax {pega_argmax} absent from HF top-{}",
            hf.len()
        )),
        Some(&hlp) if hf_top - hlp > MARGIN_TOL => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: pegainfer chose {pega_argmax}, HF scores it {:.4} nat below its argmax",
            hf_top - hlp
        )),
        Some(_) => {}
    }

    for &(token, plp) in pega.iter().take(HEAD_K) {
        if let Some(&hlp) = hf_map.get(&token) {
            let delta = (plp - hlp).abs();
            stats.head_deltas.push(delta);
            if stats.worst.is_none_or(|(w, ..)| delta > w) {
                stats.worst = Some((delta, seq, pos, token, plp, hlp));
            }
        }
    }
}

struct Golden {
    prompt_tokens: Vec<i32>,
    prompt_lens: Vec<i32>,
    decode_tokens: Vec<i32>,
    topk_ids: Vec<i32>,
    topk_lp: Vec<f32>,
    metadata: HashMap<String, String>,
    num_seqs: usize,
    decode_len: usize,
    positions: usize,
    k: usize,
}

impl Golden {
    fn load() -> Golden {
        let bytes = std::fs::read(GOLDEN).unwrap_or_else(|e| panic!("read {GOLDEN}: {e}"));
        let metadata = safetensors_metadata(&bytes);
        let st = SafeTensors::deserialize(&bytes).expect("parse golden safetensors");
        let (prompt_tokens, _) = as_i32(&st, "prompt_tokens");
        let (prompt_lens, _) = as_i32(&st, "prompt_lens");
        let (decode_tokens, dshape) = as_i32(&st, "decode_tokens");
        let (topk_ids, ishape) = as_i32(&st, "topk_ids");
        let topk_lp = as_f32(&st, "topk_logprobs");
        let num_seqs = prompt_lens.len();
        let decode_len = dshape[1];
        let positions = ishape[1];
        let k = ishape[2];
        assert_eq!(positions, decode_len + 1);
        Golden {
            prompt_tokens,
            prompt_lens,
            decode_tokens,
            topk_ids,
            topk_lp,
            metadata,
            num_seqs,
            decode_len,
            positions,
            k,
        }
    }

    fn prompt(&self, seq: usize) -> Vec<u32> {
        let off: usize = self.prompt_lens[..seq].iter().map(|&l| l as usize).sum();
        let len = self.prompt_lens[seq] as usize;
        self.prompt_tokens[off..off + len]
            .iter()
            .map(|&t| t as u32)
            .collect()
    }

    fn decode(&self, seq: usize, step: usize) -> u32 {
        self.decode_tokens[seq * self.decode_len + step] as u32
    }

    fn topk(&self, seq: usize, pos: usize) -> Vec<(u32, f32)> {
        let base = (seq * self.positions + pos) * self.k;
        (0..self.k)
            .map(|j| (self.topk_ids[base + j] as u32, self.topk_lp[base + j]))
            .collect()
    }
}

fn prefill_item(id: RequestId, prompt: Vec<u32>) -> PrefillStepItem {
    PrefillStepItem::new(id, prompt, LOGPROBS)
}

fn decode_item(id: RequestId, fed: u32) -> DecodeStepItem {
    DecodeStepItem::new(id, fed, LOGPROBS)
}

fn run(g: &Golden, ex: &mut Qwen35Executor, seqs: &[usize], batched: bool) -> (Stats, Vec<f32>) {
    let mut stats = Stats::default();
    let mut fingerprint = Vec::new();
    let mut fold = |stats: &mut Stats, seq, pos, pega: &[(u32, f32)]| {
        fingerprint.push(pega[0].1);
        check_position(stats, seq, pos, pega, &g.topk(seq, pos));
    };

    if batched {
        let ids: Vec<RequestId> = seqs
            .iter()
            .map(|&seq| RequestId::new(1000 + seq as u64))
            .collect();
        let items: Vec<PrefillStepItem> = seqs
            .iter()
            .zip(&ids)
            .map(|(&seq, &id)| prefill_item(id, g.prompt(seq)))
            .collect();
        let pr = ex
            .execute_prefill(PrefillPlan { requests: &items })
            .expect("prefill");
        for (i, &seq) in seqs.iter().enumerate() {
            fold(
                &mut stats,
                seq,
                0,
                &top_logprobs(pr.requests[i].first_token_logprob.as_ref()),
            );
        }

        for step in 0..g.decode_len {
            let items: Vec<DecodeStepItem> = seqs
                .iter()
                .zip(&ids)
                .map(|(&seq, &id)| decode_item(id, g.decode(seq, step)))
                .collect();
            let dr = ex
                .execute_decode(DecodePlan { requests: &items })
                .expect("decode");
            for (i, &seq) in seqs.iter().enumerate() {
                fold(
                    &mut stats,
                    seq,
                    step + 1,
                    &top_logprobs(dr.requests[i].logprob.as_ref()),
                );
            }
        }

        for &id in &ids {
            ex.drop_request(id).expect("drop request");
        }
    } else {
        for &seq in seqs {
            let id = RequestId::new(seq as u64);
            let pr = ex
                .execute_prefill(PrefillPlan {
                    requests: &[prefill_item(id, g.prompt(seq))],
                })
                .expect("prefill");
            fold(
                &mut stats,
                seq,
                0,
                &top_logprobs(pr.requests[0].first_token_logprob.as_ref()),
            );
            for step in 0..g.decode_len {
                let dr = ex
                    .execute_decode(DecodePlan {
                        requests: &[decode_item(id, g.decode(seq, step))],
                    })
                    .expect("decode");
                fold(
                    &mut stats,
                    seq,
                    step + 1,
                    &top_logprobs(dr.requests[0].logprob.as_ref()),
                );
            }
            ex.drop_request(id).expect("drop request");
        }
    }
    (stats, fingerprint)
}

fn dist(deltas: &[f32]) -> (f32, f32, f32, f32) {
    let mut s = deltas.to_vec();
    s.sort_by(f32::total_cmp);
    let pct = |q: f64| s[((s.len() as f64 * q) as usize).min(s.len() - 1)];
    (
        s.iter().sum::<f32>() / s.len() as f32,
        pct(0.50),
        pct(0.99),
        *s.last().unwrap(),
    )
}

fn report_and_assert(label: &str, stats: &Stats) {
    assert!(
        stats.head_deltas.len() >= stats.positions,
        "[{label}] only {} head deltas over {} positions; top-K overlap collapsed",
        stats.head_deltas.len(),
        stats.positions
    );
    let (mean, p50, p99, max) = dist(&stats.head_deltas);
    eprintln!(
        "qwen35 hf_golden_gate [{label}]: {} positions, {} head deltas — mean {mean:.4} p50 {p50:.4} p99 {p99:.4} max {max:.4}",
        stats.positions,
        stats.head_deltas.len(),
    );
    if let Some((d, s, p, tok, plp, hlp)) = stats.worst {
        eprintln!(
            "qwen35 hf_golden_gate [{label}]: worst head delta {d:.4} @ seq {s} pos {p} token {tok} (pega {plp:.4}, HF {hlp:.4})"
        );
    }

    assert!(
        stats.argmax_violations.is_empty(),
        "[{label}] pegainfer picked a token HF does not rank near its best:\n  {}",
        stats.argmax_violations.join("\n  ")
    );
    assert!(
        mean <= MEAN_TOL,
        "[{label}] mean head logprob delta {mean:.4} > {MEAN_TOL}"
    );
    assert!(
        p99 <= P99_TOL,
        "[{label}] p99 head logprob delta {p99:.4} > {P99_TOL}"
    );
    let _ = max;
}

#[test]
fn pega_logprobs_match_hf_golden_within_qwen35_tolerance() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let golden = Golden::load();
    if !check_fixture_metadata(&model_path, &golden.metadata) {
        return;
    }
    let all: Vec<usize> = (0..golden.num_seqs).collect();

    let mut ex =
        Qwen35Executor::from_runtime_with_capacity(&model_path, true, &[0], MAX_EXECUTOR_BATCH)
            .expect("build Qwen3.5 logits executor");

    let (stats, fp1) = run(&golden, &mut ex, &all, false);
    report_and_assert("sequential bs=1 graph", &stats);
    let (_, fp2) = run(&golden, &mut ex, &all, false);
    assert_eq!(
        fp1, fp2,
        "sequential Qwen3.5 replay must reproduce identical logprobs"
    );

    for n in BUCKET_STRADDLES {
        let (batched, _) = run(&golden, &mut ex, &all[..n], true);
        report_and_assert(&format!("batched graph ({n} padded)"), &batched);
    }
}
