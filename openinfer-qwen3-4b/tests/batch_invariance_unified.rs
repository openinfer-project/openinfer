//! Unified within-path GEMM-N invariance: a decode row co-batched with prefill chunks of different
//! lengths (so the projection GEMM runs at different N) must give a bit-identical top-K under Pin.
//! The pure-Decode-vs-Unified cross-path case drifts even under Pin and is the `#[ignore]` below.
use openinfer_core::sampler::SamplingParams;
use openinfer_kernels::ops::{NumericPolicy, pin_served, reset_pin_counters, set_numeric_policy};
use openinfer_qwen3_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen3Executor, RequestId, UnifiedPlan,
};
use std::sync::Mutex;

// Serialize the two #[test]s — they share the process-global numeric policy.
static POLICY_LOCK: Mutex<()> = Mutex::new(());

fn model_path_or_skip() -> Option<String> {
    if let Ok(p) = std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Some(p)
    } else {
        eprintln!("skip batch_invariance_unified: set OPENINFER_TEST_MODEL_PATH to Qwen3-4B-base");
        None
    }
}

fn prefill_first(ex: &mut Qwen3Executor, id: RequestId, prompt: &[u32]) -> u32 {
    ex.execute_prefill(PrefillPlan {
        requests: &[PrefillStepItem::new(
            id,
            prompt.to_vec(),
            64,
            SamplingParams::default(),
            0,
            false,
        )],
        echo: false,
        sample_seed: 0,
    })
    .expect("prefill")
    .requests[0]
        .first_token
}

fn unified_decode_row(
    ex: &mut Qwen3Executor,
    p: &[u32],
    cobatch: usize,
    id_dec: u64,
    id_pf: u64,
) -> (Vec<(u32, f32)>, u64) {
    let id_a = RequestId::new(id_dec);
    let t0 = prefill_first(ex, id_a, p);
    let id_b = RequestId::new(id_pf);
    let chunk: Vec<u32> = (0..cobatch as u32).map(|i| (i % 1000) + 10).collect();
    reset_pin_counters();
    let ur = ex
        .execute_unified(UnifiedPlan {
            prefill_requests: &[PrefillStepItem::new(
                id_b,
                chunk,
                64,
                SamplingParams::default(),
                0,
                false,
            )],
            decode_requests: &[DecodeStepItem::new(id_a, t0, SamplingParams::default(), 64)],
            sample_seed: 0,
        })
        .expect("unified");
    let topk = ur.decode_requests[0]
        .logprob
        .clone()
        .map(|l| l.top_logprobs)
        .unwrap_or_default();
    let served = pin_served();
    let _ = ex.drop_request(id_b);
    let _ = ex.drop_request(id_a);
    (topk, served)
}

fn pure_decode_row(ex: &mut Qwen3Executor, p: &[u32], id_dec: u64) -> Vec<(u32, f32)> {
    let id = RequestId::new(id_dec);
    let t0 = prefill_first(ex, id, p);
    reset_pin_counters();
    let dr = ex
        .execute_decode(DecodePlan {
            requests: &[DecodeStepItem::new(id, t0, SamplingParams::default(), 64)],
            sample_seed: 0,
        })
        .expect("decode");
    let topk = dr.requests[0]
        .logprob
        .clone()
        .map(|l| l.top_logprobs)
        .unwrap_or_default();
    let _ = ex.drop_request(id);
    topk
}

fn maxd(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let n = a.len().min(b.len());
    (0..n).fold(0.0f32, |m, i| m.max((a[i].1 - b[i].1).abs()))
}

const PROMPT: [u32; 8] = [9707, 785, 11, 1879, 13, 358, 1079, 264];
const COBATCH: [usize; 4] = [100, 200, 512, 1023];

#[test]
fn unified_within_path_gemm_n_invariant_under_pin() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let _g = POLICY_LOCK.lock().unwrap();
    let p = PROMPT.to_vec();

    set_numeric_policy(NumericPolicy::Pin);
    let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
    ex.set_prefix_cache_enabled(false);
    let mut base: Option<Vec<(u32, f32)>> = None;
    for (i, &c) in COBATCH.iter().enumerate() {
        let n = c + 1;
        let (topk, served) =
            unified_decode_row(&mut ex, &p, c, 100 + i as u64 * 2, 101 + i as u64 * 2);
        assert!(served > 0, "Pin N={n}: served=0 — pin never ran (vacuous)");
        match &base {
            None => base = Some(topk),
            Some(b) => assert_eq!(
                *b, topk,
                "Pin: Unified decode row drifted at N={n} vs N=101 — GEMM-N not invariant within Unified"
            ),
        }
        eprintln!("[unified-gate] Pin N={n}: served={served} bit-eq-vs-N101=ok");
    }
    drop(ex);

    set_numeric_policy(NumericPolicy::Tuned);
    let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
    ex.set_prefix_cache_enabled(false);
    let mut tbase: Option<Vec<(u32, f32)>> = None;
    let mut drifted = false;
    for (i, &c) in COBATCH.iter().enumerate() {
        let (topk, _) = unified_decode_row(&mut ex, &p, c, 200 + i as u64 * 2, 201 + i as u64 * 2);
        match &tbase {
            None => tbase = Some(topk),
            Some(b) => drifted |= *b != topk,
        }
    }
    eprintln!(
        "[unified-gate] Tuned within-Unified baseline drift: {}",
        if drifted {
            "drifts (reproduced)"
        } else {
            "STABLE"
        }
    );
    assert!(
        drifted,
        "Tuned within-Unified did not drift across N — batch-dependence not reproduced here, Pin pass vacuous"
    );
}

#[test]
#[ignore = "decode/unified cross-path drift survives the pin; tracked follow-up"]
fn cross_path_decode_vs_unified_drifts_under_pin() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let _g = POLICY_LOCK.lock().unwrap();
    let p = PROMPT.to_vec();
    set_numeric_policy(NumericPolicy::Pin);
    let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
    ex.set_prefix_cache_enabled(false);
    let dec = pure_decode_row(&mut ex, &p, 900);
    let (uni, _) = unified_decode_row(&mut ex, &p, 100, 901, 902);
    let d = maxd(&dec, &uni);
    eprintln!("[cross-path] pure-Decode vs Unified under Pin: maxΔ={d:e}");
    assert!(
        d > 0.0,
        "cross-path drift vanished — the cross-path residual may now be closed; revisit the follow-up"
    );
}
