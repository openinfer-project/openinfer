//! Decode-GEMM batch-invariance under CUDA-graph, per-policy isolated.
//!
//! The decode graph is captured once per (bucket, attention_path) and replayed; the cache key
//! excludes numeric_policy and replay skips the kernel closure, so each policy needs its own fresh
//! executor with the policy set before build. A (row 0) fixed; filler count varies A's bucket → its
//! decode-GEMM N (Tuned: a plan per bucket = drift; Pin: one algo = invariant).
//!
//!   OPENINFER_TEST_MODEL_PATH=<Qwen3-4B-base> cargo test --release \
//!     -p openinfer-qwen3 --test batch_invariance_decode_gemm_graph -- --nocapture

use openinfer_core::sampler::SamplingParams;
use openinfer_kernels::ops::NumericPolicy;
use openinfer_kernels::ops::pin_served;
use openinfer_kernels::ops::reset_numeric_policy_counters;
use openinfer_kernels::ops::set_numeric_policy;
use openinfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use openinfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use openinfer_qwen3::Qwen3LoraOptions;
use openinfer_qwen3::Qwen3MemoryOptions;
use openinfer_qwen3::Qwen3OffloadOptions;
use openinfer_qwen3::runtime::DecodePlan;
use openinfer_qwen3::runtime::DecodeStepItem;
use openinfer_qwen3::runtime::PrefillPlan;
use openinfer_qwen3::runtime::PrefillStepItem;
use openinfer_qwen3::runtime::Qwen3Executor;
use openinfer_qwen3::runtime::RequestId;

const LOGPROBS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 4;
const SHORT_LEN: usize = 8; // short decode context; path now depends on bucket, not length (<=32 SplitKv, else NonPartition)

// (label, real_small, real_large). bucket_for: 1->1, 3->4, 7->8, 15->16.
// A (row 0) identical in all; only the bucket A pads into changes (decode GEMM N).
const PAIRS: [(&str, usize, usize); 6] = [
    ("ident_b8v8  ", 7, 7), // sanity: identical batch -> must be equal under every policy
    ("b1v8(1,7)   ", 1, 7), // bucket 1 vs 8
    ("b4v8(3,7)   ", 3, 7), // bucket 4 vs 8
    ("b4v16(3,15) ", 3, 15), // bucket 4 vs 16
    // Large buckets: the pin keys on {M,K} only, so A's algo must stay identical across two LARGE
    // decode buckets — where a capture/replay layout-lifetime bug would surface.
    ("b40v72     ", 40, 71),   // buckets 40 vs 72
    ("b200v256   ", 200, 255), // top buckets 200 vs 256
];

fn model_path_or_skip() -> Option<String> {
    let Ok(p) = std::env::var("OPENINFER_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping qwen3 batch_invariance_decode_gemm_graph: set OPENINFER_TEST_MODEL_PATH to Qwen3-4B-base"
        );
        return None;
    };
    Some(p)
}

fn pitem(id: RequestId, prompt: Vec<u32>) -> PrefillStepItem {
    PrefillStepItem::new(
        id,
        prompt,
        MAX_OUTPUT_TOKENS,
        SamplingParams::default(),
        LOGPROBS,
        false,
    )
}

fn short_prompt(seed: u32) -> Vec<u32> {
    (0..SHORT_LEN as u32)
        .map(|i| 1000 + (seed * 131 + i * 7) % 50000)
        .collect()
}

/// A (request 0)'s `(prefill first_token, one-step decode top-K)` when co-batched with
/// `n_requests - 1` fillers. first_token lets the caller confirm prefill is unchanged.
fn a_first_and_decode(ex: &mut Qwen3Executor, n_requests: usize) -> (u32, Vec<(u32, f32)>) {
    let batch: Vec<(RequestId, Vec<u32>)> = (0..n_requests)
        .map(|i| (RequestId::new(1 + i as u64), short_prompt(i as u32)))
        .collect();
    let pitems: Vec<PrefillStepItem> = batch.iter().map(|(id, p)| pitem(*id, p.clone())).collect();
    let pr = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &pitems,
            echo: false,
        })
        .expect("prefill");
    let a_first = pr.requests[0].first_token;
    let ditems: Vec<DecodeStepItem> = batch
        .iter()
        .zip(&pr.requests)
        .map(|((id, _), req)| {
            DecodeStepItem::new(*id, req.first_token, SamplingParams::default(), LOGPROBS)
        })
        .collect();
    let dr = ex
        .execute_decode(DecodePlan {
            sample_seed: 0,
            requests: &ditems,
        })
        .expect("decode");
    let topk = dr.requests[0]
        .logprob
        .as_ref()
        .expect("logprobs requested but none returned")
        .top_logprobs
        .clone();
    for (id, _) in &batch {
        ex.drop_request(*id).expect("drop request");
    }
    (a_first, topk)
}

/// One fresh executor with `policy` active before construction — decode graphs are
/// pre-captured at startup under it. Returns the pin-GEMM count from that sweep plus
/// per-pair `(first_token_eq, topk_eq)` results. Replay skips the kernel closure, so no
/// later decode step can add to the pin count (Tuned/PerToken are structurally 0 = N/A).
fn run_policy(policy: NumericPolicy, model_path: &str) -> (u64, Vec<(bool, bool)>) {
    set_numeric_policy(policy);
    reset_numeric_policy_counters();
    // PerToken graphs bake one N=1 GemmEx node per (row, projection, layer), so graph
    // exec memory grows with the bucket (~1.8GB across all 36 startup-captured buckets;
    // Tuned/Pin stay flat ~2MB/bucket). Shrink the KV pool so the three policy executors
    // fit on consumer GPUs — the assertions are numerics-only, pool size is irrelevant.
    let mut ex = Qwen3Executor::from_runtime_with_lora_options(
        model_path,
        true,
        &[0],
        Qwen3LoraOptions::default(),
        Qwen3OffloadOptions::disabled(),
        DEFAULT_MAX_PREFILL_TOKENS,
        None,
        Qwen3MemoryOptions::new(
            0.75,
            DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
            DEFAULT_KV_PAGE_SIZE,
        ),
        false,
    )
    .expect("build executor");
    let capture_served = pin_served();
    ex.set_prefix_cache_enabled(false);
    let mut out = Vec::new();
    for (_, small, large) in PAIRS {
        let (ft_s, tk_s) = a_first_and_decode(&mut ex, small);
        let (ft_l, tk_l) = a_first_and_decode(&mut ex, large);
        out.push((ft_s == ft_l, tk_s == tk_l));
    }
    (capture_served, out)
}

#[test]
fn batch_invariance_decode_gemm_graph() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let (_, baseline) = run_policy(NumericPolicy::Tuned, &model_path);
    let (pin_served_at_capture, pin) = run_policy(NumericPolicy::Pin, &model_path);
    let (_, pertoken) = run_policy(NumericPolicy::PerToken, &model_path);

    // Prefill must be identical for A in every batch/policy, else the decode comparison is
    // prefill-contaminated.
    for (name, rows) in [
        ("baseline", &baseline),
        ("pin", &pin),
        ("per_token", &pertoken),
    ] {
        for (i, (ft_eq, _)) in rows.iter().enumerate() {
            assert!(
                *ft_eq,
                "{name}: A's prefill first_token differs across batch on pair {} — decode \
                 comparison is prefill-contaminated, not a decode result",
                PAIRS[i].0.trim()
            );
        }
    }

    // Control: baseline must drift on at least one pair (else not reproduced here).
    let baseline_drifted = baseline.iter().any(|(_, tk_eq)| !tk_eq);
    assert!(
        baseline_drifted,
        "baseline did NOT drift on any pair — decode-GEMM coupling not reproduced; \
         the pin proof would be vacuous"
    );

    // Pin: every pair invariant, and pin actually ran during the startup capture sweep.
    assert!(
        pin_served_at_capture > 0,
        "PIN: pin did not run during the startup pre-capture sweep (served=0) — vacuous"
    );
    for (i, (_, tk_eq)) in pin.iter().enumerate() {
        assert!(
            *tk_eq,
            "PIN: A's graph decode top-K changed on pair {} — pin did NOT make graph decode \
             batch-invariant (graphs captured under Pin)",
            PAIRS[i].0.trim()
        );
    }
    for (i, (_, tk_eq)) in pertoken.iter().enumerate() {
        assert!(
            *tk_eq,
            "PER_TOKEN: A's graph decode top-K changed on pair {} — harness bug",
            PAIRS[i].0.trim()
        );
    }
}
