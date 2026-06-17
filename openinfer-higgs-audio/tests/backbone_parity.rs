//! Higgs backbone parity gate — validates that our backbone forward produces
//! logits within the bf16 noise floor of a pre-computed golden (from the
//! actual Higgs checkpoint via chat-template forward).
//!
//! The golden (`test_data/higgs/backbone_golden.safetensors`) was produced by
//! converting `backbone_golden.pt` to safetensors:
//! ```python
//! import torch
//! from safetensors.torch import save_file
//! g = torch.load("test_data/higgs/backbone_golden.pt", map_location="cpu", weights_only=False)
//! save_file({"input_ids": g["input_ids"].to(torch.int64).contiguous(),
//!            "logits": g["logits"].to(torch.float32).contiguous(),
//!            "hidden": g["hidden_states"].to(torch.float32).contiguous()},
//!           "test_data/higgs/backbone_golden.safetensors")
//! ```
//!
//! Assertions:
//!   * top-64 logprobs comparison: regret (reference has clear winner, we pick
//!     wrong), mean delta, p99 delta.
//!   * mean ≤ ~0.06 nat, p99 ≤ ~0.20 nat — tolerances calibrated from the
//!     bf16 noise floor (same pattern as qwen3 golden gate).
//!   * absolute max delta is printed but NOT asserted (coverage-unstable).
//!
//! Requires: CUDA GPU, Higgs checkpoint weights, and backbone_golden.safetensors.
//! Skipped cleanly when the model or golden is absent.

use half::bf16;
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::ops::PrefillPagedPlan;
use openinfer_higgs_audio::backbone::HiggsBackbone;
use safetensors::SafeTensors;
use std::path::Path;

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/higgs/backbone_golden.safetensors"
);

/// Number of top logprobs to compare (matching qwen3's LOGPROBS).
const LOGPROBS: usize = 64;
/// Max acceptable regret: how far below golden's argmax our pick may sit.
const MARGIN_TOL: f32 = 0.20;
/// Mean delta bound (systematic drift trips this).
const MEAN_TOL: f32 = 0.06;
/// P99 delta bound (spread inflation trips this).
const P99_TOL: f32 = 0.20;
/// Head depth for the logprob tolerance check.
const HEAD_K: usize = 8;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) => {
            eprintln!(
                "skipping backbone parity — set OPENINFER_TEST_MODEL_PATH \
                 to Higgs checkpoint directory"
            );
            None
        }
    }
}

fn load_golden() -> Option<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    let path = Path::new(GOLDEN_PATH);
    if !path.exists() {
        eprintln!(
            "skipping backbone parity — golden file not found at {}. \
             Convert backbone_golden.pt to safetensors first.",
            GOLDEN_PATH
        );
        return None;
    }
    let data = std::fs::read(path).expect("failed to read golden safetensors");
    let tensors = SafeTensors::deserialize(&data).expect("failed to deserialize golden");

    let ids = tensors.tensor("input_ids").expect("missing input_ids");
    let logits_view = tensors.tensor("logits").expect("missing logits");
    let hidden_view = tensors.tensor("hidden").expect("missing hidden");

    let input_ids: Vec<u32> = ids
        .data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as u32)
        .collect();

    let logits: Vec<f32> = logits_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let hidden: Vec<f32> = hidden_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    Some((input_ids, logits, hidden))
}

/// Top-64 logprobs regret / mean / p99 suite (same pattern as qwen3 golden gate).
fn check_logprobs(our_logprobs: &[f32], ref_logprobs: &[f32]) {
    assert_eq!(
        our_logprobs.len(),
        ref_logprobs.len(),
        "vocab size mismatch"
    );
    let vocab = our_logprobs.len();

    // Find top-64 indices in our sorted order for consistent coverage
    let mut our_sorted: Vec<(f32, usize)> = our_logprobs
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    our_sorted.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top_indices: Vec<usize> = our_sorted.iter().take(LOGPROBS).map(|(_, i)| *i).collect();

    // Regret: for each top token, if golden has a clear winner (margin > MARGIN_TOL),
    // ensure we pick the same token.
    let golden_best_idx = ref_logprobs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let golden_best = ref_logprobs[golden_best_idx];
    let golden_second = ref_logprobs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != golden_best_idx)
        .map(|(_, v)| *v)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f32::NEG_INFINITY);
    let margin = golden_best - golden_second;

    if margin > MARGIN_TOL {
        // Golden has a clear winner — we must pick the same token
        let our_best_idx = our_logprobs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if our_best_idx != golden_best_idx {
            // Compute regret: difference between golden's best and what golden
            // assigns to our choice
            let regret = golden_best - ref_logprobs[our_best_idx];
            assert!(
                regret <= MARGIN_TOL,
                "regret {:.4} > {:.4}: golden best token={golden_best_idx} (logprob={golden_best:.4}), \
                 our best token={our_best_idx} (golden assigns it logprob={:.4})",
                regret,
                MARGIN_TOL,
                ref_logprobs[our_best_idx]
            );
        }
    }

    // Delta stats on head tokens
    let mut deltas: Vec<f32> = top_indices
        .iter()
        .take(HEAD_K)
        .map(|&i| (our_logprobs[i] - ref_logprobs[i]).abs())
        .collect();
    deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = deltas.len();
    let mean_delta: f32 = deltas.iter().sum::<f32>() / n as f32;
    let p99_idx = ((n - 1) as f64 * 0.99).ceil() as usize;
    let p99_delta = deltas[p99_idx.min(n - 1)];
    let max_delta = deltas[n - 1];

    eprintln!(
        "delta stats (top-{HEAD_K} of top-{LOGPROBS}): \
         mean={mean_delta:.6}, p99={p99_delta:.6}, max={max_delta:.6}"
    );

    assert!(
        mean_delta <= MEAN_TOL,
        "mean delta {mean_delta:.6} > {MEAN_TOL} (systematic drift)"
    );
    assert!(
        p99_delta <= P99_TOL,
        "p99 delta {p99_delta:.6} > {P99_TOL} (spread inflation)"
    );
    // max_delta is reported but NOT asserted (coverage-unstable)
}

#[test]
fn backbone_parity_against_golden() {
    let model_path = match model_path_or_skip() {
        Some(p) => p,
        None => return,
    };
    let (input_ids, golden_logits, _golden_hidden) = match load_golden() {
        Some(g) => g,
        None => return,
    };

    let seq_len = input_ids.len();
    assert!(seq_len > 0, "golden input_ids is empty");
    eprintln!("golden input_ids length: {seq_len}");

    // Load the Higgs backbone
    let backbone =
        HiggsBackbone::from_safetensors(&model_path, 0).expect("failed to load Higgs backbone");

    let config = backbone.config();
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_hidden_layers, 36);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);

    // Set up KV buffer: one page big enough for the entire sequence
    let page_size = seq_len.next_power_of_two().max(16);
    let layout = KvLayout::new(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        page_size,
    );
    eprintln!(
        "KV layout: page_size={page_size}, page_stride={} elements ({:.1} MB)",
        layout.page_stride,
        layout.page_stride as f64 * 2.0 / 1e6
    );

    let ctx = backbone.device_ctx();
    let kv_buffer: cudarc::driver::CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride)
        .expect("failed to allocate KV buffer");

    // Create prefill plan for single sequence
    let plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
        ctx,
        &[vec![0i32]],
        &[seq_len],
        &[0usize],
        &[seq_len],
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        0, // use default CTA tile size
    )
    .expect("failed to create prefill plan");

    eprintln!("running backbone forward...");
    let hidden = backbone
        .forward(&input_ids, &kv_buffer, &layout, &plan)
        .expect("backbone forward failed");

    assert_eq!(hidden.hidden_dim, config.hidden_size);
    assert_eq!(hidden.seq_len, seq_len);

    // Compute last-token logits
    let logits = backbone
        .last_token_logits(&hidden)
        .expect("last_token_logits failed");

    assert_eq!(logits.hidden_dim, config.vocab_size);
    assert_eq!(logits.seq_len, 1);

    // Copy logits to host and convert bf16 → f32
    let n_logits = logits.data.len();
    let mut logits_bf16 = vec![bf16::ZERO; n_logits];
    ctx.stream
        .memcpy_dtoh(&logits.data, &mut logits_bf16)
        .expect("failed to copy logits to host");
    ctx.stream.synchronize().expect("sync failed");
    let logits_host: Vec<f32> = logits_bf16.iter().map(|&x| f32::from(x)).collect();

    // Compute log_softmax for both our logits and golden logits
    let our_logprobs = log_softmax(&logits_host);
    let golden_logprobs = log_softmax(&golden_logits);

    assert_eq!(
        our_logprobs.len(),
        golden_logprobs.len(),
        "vocab size mismatch: ours={}, golden={}",
        our_logprobs.len(),
        golden_logprobs.len()
    );

    eprintln!("comparing logprobs (vocab={})...", our_logprobs.len());
    check_logprobs(&our_logprobs, &golden_logprobs);
}

/// Compute log_softmax in f32.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum = sum_exp.ln();
    logits.iter().map(|&x| x - max_val - log_sum).collect()
}
