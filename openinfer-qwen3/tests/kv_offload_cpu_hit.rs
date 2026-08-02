//! Live GPU+CPU prefix-hit gate for the kv-store (pegaflow host tier).
//!
//! Drives a real Qwen3-4B [`Qwen3Executor`] with offload enabled to prove the
//! end-to-end wiring on actual model weights:
//!   * a cold prefill seals its KV blocks to the store's host tier (write path:
//!     per-step `save_sealed_blocks`, plus the final seal when the request
//!     retires at `drop_request`);
//!   * after `flush_offload_saves` (D2H + query-visibility barrier) and
//!     `evict_cached_blocks` (L1 drain), the prefix survives only on the host
//!     tier;
//!   * a scheduler-style `KvStore::resolve_prefix` then finds it there and
//!     restores it into HBM (query → reserve → load → radix commit), so the
//!     following prefill's `match_and_add_prefix` reuses the restored blocks
//!     instead of recomputing them;
//!   * the restored KV reproduces the original first-token logits.
//!
//! This is the one test that exercises save → host-tier persistence → resolve →
//! load → radix commit → prefill-rematch through the live executor, not a unit
//! harness. `openinfer-kv-store/tests/` covers the raw store byte path; this
//! covers the executor wiring. If a load landed in the wrong
//! layer/segment/block the warm logits would be whole nats off.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when absent
//! (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::collections::HashMap;
use std::path::Path;

use openinfer_core::sampler::SamplingParams;
use openinfer_kv_store::CacheScope;
use openinfer_kv_store::KvPrefix;
use openinfer_kv_store::KvStore;
use openinfer_kv_store::NeverCancelled;
use openinfer_kv_store::ResolvePolicy;
use openinfer_qwen3::Qwen3LoraOptions;
use openinfer_qwen3::Qwen3OffloadOptions;
use openinfer_qwen3::runtime::PrefillPlan;
use openinfer_qwen3::runtime::PrefillStepItem;
use openinfer_qwen3::runtime::Qwen3Executor;
use openinfer_qwen3::runtime::RequestId;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const BLOCK: usize = 16;
const LOGPROBS: usize = 16;
const MAX_OUTPUT: usize = 8;
/// 512 MiB host tier — comfortably more than the handful of dense Qwen3-4B
/// blocks this test offloads (~2.25 MiB/block).
const HOST_TIER_BYTES: usize = 512 << 20;
/// The single-GPU executor registers exactly one store rank: 0.
const STORE_RANK: usize = 0;

/// Warm-vs-cold bounds, following the prefix-cache methodology: the restored
/// KV is byte-identical to the original GPU compute, so the only legitimate
/// drift is the prefill GEMM shrinking to the uncached tail (bf16 reduction
/// order). The warm argmax must sit within `REGRET_TOL` of cold; the mean head
/// delta must stay at the bf16 floor.
const REGRET_TOL: f32 = 0.20;
const MEAN_TOL: f32 = 0.06;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 kv_offload_cpu_hit: {MODEL_PATH}/config.json is missing; \
                 set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Deterministic synthetic prompt; different seeds share no prefix.
fn prompt(seed: usize, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((seed * 100_003 + i * 17) % 50_000 + 1_000) as u32)
        .collect()
}

fn prefill_item(id: u64, prompt: &[u32]) -> PrefillStepItem {
    PrefillStepItem::new(
        RequestId::new(id),
        prompt.to_vec(),
        MAX_OUTPUT,
        SamplingParams::default(),
        LOGPROBS,
        false,
    )
}

fn first_token_top(pr: &openinfer_qwen3::runtime::PrefillResult) -> Vec<(u32, f32)> {
    pr.requests[0]
        .first_token_logprob
        .as_ref()
        .expect("logprobs requested but none returned")
        .top_logprobs
        .clone()
}

/// The warm (CPU-restored) first-token logits must agree with the cold compute
/// up to bf16 reduction noise: warm argmax within `REGRET_TOL` of cold, mean
/// head-token delta under `MEAN_TOL`.
fn assert_close(cold: &[(u32, f32)], warm: &[(u32, f32)]) {
    let cold_map: HashMap<u32, f32> = cold.iter().copied().collect();
    let cold_top = cold[0].1;
    match cold_map.get(&warm[0].0) {
        None => panic!(
            "warm argmax {} absent from cold top-{}",
            warm[0].0,
            cold.len()
        ),
        Some(&clp) => assert!(
            cold_top - clp <= REGRET_TOL,
            "warm argmax {} sits {:.4} nat below cold argmax",
            warm[0].0,
            cold_top - clp
        ),
    }
    let deltas: Vec<f32> = warm
        .iter()
        .take(8)
        .filter_map(|&(token, wlp)| cold_map.get(&token).map(|&clp| (wlp - clp).abs()))
        .collect();
    assert!(!deltas.is_empty(), "no head-token overlap");
    let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
    let max = deltas.iter().copied().fold(0.0f32, f32::max);
    eprintln!(
        "kv_offload_cpu_hit: {} head deltas — mean {mean:.4} max {max:.4}",
        deltas.len()
    );
    assert!(
        mean <= MEAN_TOL,
        "mean head logprob delta {mean:.4} > {MEAN_TOL} — restored KV drifted past bf16 noise"
    );
}

/// Play the scheduler's read-path part by hand (the test drives the executor
/// directly, so there is no scheduler to do it): run the store's async
/// `resolve_prefix` on the store's own runtime — the same pattern as
/// `KvStore::flush_saves_blocking` — and return the hold. Keep it alive across
/// the following `execute_prefill`: it pin-protects the resolved blocks until
/// the prefill's `match_and_add_prefix` has consumed them (the production
/// scheduler drops it at the first `PrefillRequestResult`).
///
/// `CacheScope::default()` is qwen3's serving scope: no cache salt, no LoRA.
fn resolve(store: &KvStore, req_id: &str, tokens: &[u32]) -> KvPrefix {
    store.runtime().block_on(store.resolve_prefix(
        STORE_RANK,
        req_id,
        tokens,
        CacheScope::default(),
        ResolvePolicy::default(),
        &NeverCancelled,
    ))
}

/// One executor, two scenarios, run sequentially. cargo runs `#[test]`
/// functions on parallel threads; two Qwen3-4B executors sharing device 0 and
/// the same pegaflow instance id ("qwen3-dev0") would collide on the host
/// tier. Production wires exactly one executor per model, so the realistic
/// shape is one executor servicing both prefixes. The two scenarios use
/// disjoint prompt seeds, so they share no prefix and cannot cross-contaminate.
#[test]
fn live_gpu_and_cpu_prefix_hits() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let mut ex = Qwen3Executor::from_runtime_with_lora_options(
        &model_path,
        false,
        &[0],
        Qwen3LoraOptions::default(),
        Qwen3OffloadOptions::enabled(HOST_TIER_BYTES),
        openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
        None,
        openinfer_qwen3::Qwen3MemoryOptions::default(),
        false,
    )
    .expect("build offload executor");
    assert!(ex.offload_enabled(), "offload must be active");

    cpu_tier_restores_evicted_prefix(&mut ex);
    gpu_and_cpu_combined_hit(&mut ex);
}

/// A prefix that is evicted from HBM and restored entirely from the host tier
/// (GPU radix hit == 0): the baseline CPU round-trip through the live executor.
fn cpu_tier_restores_evicted_prefix(ex: &mut Qwen3Executor) {
    let p = prompt(7, 50); // 3 full blocks (48 tok) + 2-token tail

    // ── Cold: first sight of P. Computes all of P on GPU; the per-step save
    // plus the retire-time final seal put the 3 sealed blocks on the host
    // tier. ──
    let cold = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(1, &p)],
            echo: false,
        })
        .expect("cold prefill");
    assert_eq!(
        cold.requests[0].cached_tokens, 0,
        "first sight of P is cold"
    );
    let cold_first = first_token_top(&cold);
    ex.drop_request(RequestId::new(1)).expect("drop req1");

    // ── Persist the saves (D2H landed + query-visible), then evict P from
    // HBM so it lives only on the host tier. ──
    ex.flush_offload_saves();
    ex.evict_cached_blocks();

    // ── A GPU miss now: resolve must find P on the host tier and restore all
    // 3 blocks into HBM (radix commit included). The cacheable cap leaves the
    // 2-token tail out, so the hit is exactly 3 blocks = 48 tokens. ──
    let store = ex.kv_store().expect("offload store wired");
    let prefix = resolve(&store, "kv-cpu-hit-req2", &p);
    assert!(
        prefix.hit_tokens() >= 3 * BLOCK,
        "host tier must hold P after GPU eviction; resolve hit {} tokens, expected {}",
        prefix.hit_tokens(),
        3 * BLOCK
    );

    // ── Warm: the restored blocks are matched (the hold above pins them
    // across the match), only the 2-token tail recomputes — the full-block
    // cap keeps the 3rd block's last token off the match the same way the GPU
    // prefix cache does. ──
    let warm = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(2, &p)],
            echo: false,
        })
        .expect("warm prefill");
    // The match consumed the resolved blocks; release the hold the way the
    // scheduler does at the first PrefillRequestResult.
    drop(prefix);
    assert_eq!(
        warm.requests[0].cached_tokens,
        3 * BLOCK,
        "CPU-restored prefix: 3 blocks matched, tail recomputed"
    );
    let warm_first = first_token_top(&warm);
    ex.drop_request(RequestId::new(2)).expect("drop req2");

    // ── The restored KV must reproduce the original GPU first-token logits. ──
    assert_close(&cold_first, &warm_first);
}

/// A single prefix that is part GPU-resident, part host-only: the resolve must
/// stack the host continuation onto the GPU hit and the re-match must see one
/// contiguous prefix. This is the case that catches an off-by-`gpu_hit` bug in
/// the query/commit offset math — the pure-host test (GPU hit == 0) cannot.
fn gpu_and_cpu_combined_hit(ex: &mut Qwen3Executor) {
    let full = prompt(9, 100); // 6 full blocks (96 tok) + 4-token tail
    let short = full[..50].to_vec(); // a 3-block prefix of `full`

    // ── Cold-compute `full`, saving all 6 blocks to the host tier. ──
    let cold = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(1, &full)],
            echo: false,
        })
        .expect("cold full prefill");
    assert_eq!(
        cold.requests[0].cached_tokens, 0,
        "first sight of full is cold"
    );
    let cold_first = first_token_top(&cold);
    ex.drop_request(RequestId::new(1)).expect("drop req1");
    ex.flush_offload_saves();

    // ── Drop the whole prefix from HBM (host keeps all 6 blocks), then
    // re-establish ONLY the first 3 blocks in HBM by cold-prefilling `short`.
    // GPU radix now holds blocks 0..3; the host tier holds blocks 0..6. ──
    ex.evict_cached_blocks();
    let s = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(2, &short)],
            echo: false,
        })
        .expect("short prefill");
    assert_eq!(
        s.requests[0].cached_tokens, 0,
        "short re-warms blocks 0..3 cold"
    );
    ex.drop_request(RequestId::new(2)).expect("drop req2");

    // ── Resolve `full`: the GPU probe hits blocks 0..3, the host tier must
    // supply the continuation 3..6, and the resolve reports the combined
    // prefix (6 blocks = 96 tokens; the 4-token tail stays uncacheable).
    // Without the host continuation this would stop at 3. ──
    let store = ex.kv_store().expect("offload store wired");
    let prefix = resolve(&store, "kv-combined-hit-req3", &full);
    assert!(
        prefix.hit_tokens() >= 6 * BLOCK,
        "resolve must stack host blocks 3..6 onto the GPU hit; got {} tokens, expected {}",
        prefix.hit_tokens(),
        6 * BLOCK
    );

    // ── Warm prefill `full`: all 6 blocks match (3 GPU + 3 host). ──
    let warm = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(3, &full)],
            echo: false,
        })
        .expect("warm full prefill");
    drop(prefix);
    assert_eq!(
        warm.requests[0].cached_tokens,
        6 * BLOCK,
        "combined hit: 3 GPU-resident + 3 host-restored blocks match as one prefix"
    );
    let warm_first = first_token_top(&warm);
    ex.drop_request(RequestId::new(3)).expect("drop req3");

    assert_close(&cold_first, &warm_first);
}
