//! GLM5.2 PP8 P2P spine measurement (Slice 0).
//!
//! Node38-gated: needs an NVLink-connected multi-GPU box (8xH200). On a single
//! GPU it skips. Run on node38 with:
//!
//! ```bash
//! OPENINFER_PP_SPINE=1 cargo test -p openinfer-glm52 --release \
//!     --test pp_p2p_spine -- --nocapture
//! ```
//!
//! Emits one CSV row per hop. The pass-bar (per docs/models/glm52/pp-decode.md):
//! the pp=8 / 12KB / no-burn cell must take ~16us over 7 hops with p99 within a
//! few % of p50 and ZERO samples >100us across >=50k iters. The absolute
//! microseconds are hardware-dependent (H200 NVLink vs B300), so the test prints
//! them and asserts only the tail-latency invariant.

use openinfer_glm52::{Glm52PpSpineConfig, run_pp_p2p_spine};

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
fn pp_p2p_spine_sweep() {
    if std::env::var("OPENINFER_PP_SPINE").is_err() {
        eprintln!(
            "skip pp_p2p_spine: set OPENINFER_PP_SPINE=1 on an 8x NVLink node (node38) to run"
        );
        return;
    }

    let iters = env_u64("OPENINFER_PP_ITERS", 50_000);
    let warmup = env_u64("OPENINFER_PP_WARMUP", 5_000);
    let deadline_ns = 2_000_000_000; // 2s per spin before crash-early trap

    let pp_sizes = [2usize, 4, 8];
    let words_cells = [6144usize, 24576]; // 12KB and 48KB bf16 hidden
    let rings = [2usize];
    let burns_ns = [0u64, 100_000, 500_000]; // 0, 100us, 500us modelled stage compute

    println!(
        "pp_size,words,bytes,ring,burn_ns,hop,rtt_p50_us,rtt_p90_us,rtt_p99_us,rtt_p999_us,\
         rtt_max_us,gt10us,gt100us,chain_rtt_p50_us,wall_per_iter_us"
    );

    for &pp in &pp_sizes {
        for &words in &words_cells {
            for &ring in &rings {
                for &burn_ns in &burns_ns {
                    let config = Glm52PpSpineConfig {
                        device_ordinals: (0..pp).collect(),
                        words,
                        ring,
                        burn_ns,
                        warmup,
                        iters,
                        deadline_ns,
                    };
                    let report = run_pp_p2p_spine(config).unwrap_or_else(|err| {
                        panic!("pp={pp} words={words} ring={ring} burn_ns={burn_ns}: {err:#}")
                    });
                    for hop in &report.hops {
                        println!(
                            "{pp},{words},{},{ring},{burn_ns},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{:.3}",
                            words * 2,
                            hop.hop,
                            hop.rtt_p50_us,
                            hop.rtt_p90_us,
                            hop.rtt_p99_us,
                            hop.rtt_p999_us,
                            hop.rtt_max_us,
                            hop.gt10us,
                            hop.gt100us,
                            report.chain_rtt_p50_us,
                            report.wall_per_iter_us,
                        );
                    }

                    // Tail-latency invariant on the primary cell: a healthy spine
                    // never spikes >100us. A nonzero count means scheduling /
                    // fence pathology worth chasing before adding real layers.
                    if pp == 8 && words == 6144 && burn_ns == 0 {
                        let spikes: usize = report.hops.iter().map(|h| h.gt100us).sum();
                        assert_eq!(
                            spikes, 0,
                            "pp=8 12KB spine: {spikes} hop-samples exceeded 100us"
                        );
                    }
                }
            }
        }
    }
}
