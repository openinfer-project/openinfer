//! Production-envelope fallback guard: under Pin the {M,K} algo is pinned and reused for every N;
//! this asserts zero per-token fallback across the swept Qwen3-4B envelope — Unified at
//! N=101/201/513/1024/1279 and pure-Decode at bs=256 (exactly these points, not all N).
use openinfer_core::sampler::SamplingParams;
use openinfer_kernels::ops::{
    NumericPolicy, pin_counters, pin_fallback_shapes, reset_pin_counters, set_numeric_policy,
};
use openinfer_qwen3_4b::runtime::{
    DecodePlan, DecodeStepItem, PrefillPlan, PrefillStepItem, Qwen3Executor, RequestId, UnifiedPlan,
};

fn model_path_or_skip() -> Option<String> {
    if let Ok(p) = std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Some(p)
    } else {
        eprintln!("skip batch_invariance_envelope: set OPENINFER_TEST_MODEL_PATH to Qwen3-4B-base");
        None
    }
}

fn synth(ctx: usize, seed: u64) -> Vec<u32> {
    (0..ctx as u32)
        .map(|i| (i + seed as u32 * 13) % 2000 + 10)
        .collect()
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

fn assert_no_fallback(label: &str, n: usize) {
    let (served, fb) = pin_counters();
    let shapes = pin_fallback_shapes();
    assert!(
        served > 0,
        "{label} N={n}: served=0 — pin never ran (vacuous)"
    );
    assert!(
        shapes.is_empty(),
        "{label} N={n}: {fb} per-token fallback(s), shapes {shapes:?} — pin cannot serve this N"
    );
    eprintln!("[envelope] {label} N={n}: served={served} fallback={fb} ok");
}

#[test]
fn pin_serves_production_envelope_without_fallback() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    set_numeric_policy(NumericPolicy::Pin);
    let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
    ex.set_prefix_cache_enabled(false);

    for (i, &pf) in [100usize, 200, 512, 1023].iter().enumerate() {
        let id_d = RequestId::new(50000 + i as u64);
        let t = prefill_first(&mut ex, id_d, &synth(8, i as u64));
        let id_p = RequestId::new(51000 + i as u64);
        reset_pin_counters();
        let _ = ex
            .execute_unified(UnifiedPlan {
                prefill_requests: &[PrefillStepItem::new(
                    id_p,
                    synth(pf, 100 + i as u64),
                    64,
                    SamplingParams::default(),
                    0,
                    false,
                )],
                decode_requests: &[DecodeStepItem::new(id_d, t, SamplingParams::default(), 64)],
                sample_seed: 0,
            })
            .expect("unified envelope");
        assert_no_fallback("Unified", pf + 1);
        let _ = ex.drop_request(id_p);
        let _ = ex.drop_request(id_d);
    }

    let dec: Vec<(RequestId, u32)> = (0..256u64)
        .map(|i| {
            let id = RequestId::new(20000 + i);
            let t = prefill_first(&mut ex, id, &synth(16, i));
            (id, t)
        })
        .collect();
    let items: Vec<DecodeStepItem> = dec
        .iter()
        .map(|&(id, tok)| DecodeStepItem::new(id, tok, SamplingParams::default(), 0))
        .collect();
    reset_pin_counters();
    let _ = ex
        .execute_decode(DecodePlan {
            requests: &items,
            sample_seed: 0,
        })
        .expect("decode bs=256");
    assert_no_fallback("pure-Decode", 256);
    for (id, _) in &dec {
        let _ = ex.drop_request(*id);
    }

    let decoders: Vec<(RequestId, u32)> = (0..255u64)
        .map(|i| {
            let id = RequestId::new(30000 + i);
            let t = prefill_first(&mut ex, id, &synth(16, 1000 + i));
            (id, t)
        })
        .collect();
    let id_big = RequestId::new(40000);
    let decode_items: Vec<DecodeStepItem> = decoders
        .iter()
        .map(|&(id, tok)| DecodeStepItem::new(id, tok, SamplingParams::default(), 0))
        .collect();
    reset_pin_counters();
    let _ = ex
        .execute_unified(UnifiedPlan {
            prefill_requests: &[PrefillStepItem::new(
                id_big,
                synth(1024, 7),
                64,
                SamplingParams::default(),
                0,
                false,
            )],
            decode_requests: &decode_items,
            sample_seed: 0,
        })
        .expect("unified N=1279");
    assert_no_fallback("Unified", 1279);
    let _ = ex.drop_request(id_big);
    for (id, _) in &decoders {
        let _ = ex.drop_request(*id);
    }
}
