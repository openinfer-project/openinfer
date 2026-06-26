//! PP8 runtime spine (Slice 0): a thread-per-GPU coordinator that captures one
//! stage graph per GPU, serializes them by device-memory flags, and measures the
//! per-hop `L_send` latency. Intentionally bs=1 -- a single token walks the open
//! chain 0..n-1 each replay; the `R>=2` ring exists for later microbatch/MTP but
//! is driven serially here.
//!
//! Orchestration (all stages in parallel, two barriers):
//!   1. each stage grants its neighbours access to its memory pool, allocs its
//!      rings, and publishes their peer VAs;
//!   2. [barrier] each stage resolves its peer edge from the neighbours' VAs,
//!      zeroes its control words, and captures `wait/burn/send`;
//!   3. [barrier] each stage enqueues `warmup+iters` async graph launches and
//!      syncs -- the device flags serialize the chain across GPUs;
//!   4. host gathers the producer `deltas` and reduces per-hop percentiles.

use std::sync::{Barrier, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use openinfer_kernels::ops::{
    Glm52PpSendParams, glm52_pp_dummy_burn_launch, glm52_pp_send_hidden_launch,
    glm52_pp_source_inject_launch, glm52_pp_wait_hidden_launch,
};

use super::peer::{Glm52PeerEdge, Glm52StageBuffers, Glm52StageVas, grant_pool_peer_access};
use super::stage_graph::Glm52StageGraph;
use crate::weights::Glm52RankGpuContext;

/// Spine measurement configuration. One stage per `device_ordinals` entry.
#[derive(Clone, Debug)]
pub struct Glm52PpSpineConfig {
    /// GPU ordinal for each stage, in chain order (`>= 2` stages).
    pub device_ordinals: Vec<usize>,
    /// bf16 elements per hidden payload (12KB hidden = 6144, multiple of 8).
    pub words: usize,
    /// Ring depth `R` (`slot = epoch % ring`); 2 double-buffers.
    pub ring: usize,
    /// Per-stage modelled compute time in ns (`dummy_burn`).
    pub burn_ns: u64,
    /// Replays to discard before recording (pipeline fill / clock warmup).
    pub warmup: u64,
    /// Measured replays.
    pub iters: u64,
    /// Per-spin timeout in ns; exceeding it traps the stage (crash-early).
    pub deadline_ns: u64,
}

/// Per-hop forward-RTT distribution (the producer's own globaltimer deltas).
#[derive(Clone, Debug)]
pub struct Glm52PpHopStats {
    pub hop: usize,
    pub rtt_p50_us: f64,
    pub rtt_p90_us: f64,
    pub rtt_p99_us: f64,
    pub rtt_p999_us: f64,
    pub rtt_max_us: f64,
    pub gt10us: usize,
    pub gt100us: usize,
}

/// Spine measurement result.
#[derive(Clone, Debug)]
pub struct Glm52PpSpineReport {
    pub pp_size: usize,
    pub words: usize,
    pub ring: usize,
    pub burn_ns: u64,
    pub hops: Vec<Glm52PpHopStats>,
    /// Sum of per-hop p50 RTTs -- a chain-traversal proxy.
    pub chain_rtt_p50_us: f64,
    /// Slowest stage's wall-clock per replay, averaged over all launches
    /// (warmup + measured). A coarse cross-check against the deltas, not a metric.
    pub wall_per_iter_us: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StageRole {
    Source,
    Middle,
    Sink,
}

struct StageOutput {
    wall: Duration,
    deltas: Option<Vec<u64>>,
}

pub fn run_pp_p2p_spine(config: Glm52PpSpineConfig) -> Result<Glm52PpSpineReport> {
    let n = config.device_ordinals.len();
    ensure!(n >= 2, "PP spine needs >= 2 stages, got {n}");
    ensure!(
        config.iters > 0,
        "PP spine needs a positive measured iteration count"
    );
    ensure!(
        config.words > 0 && config.words % 8 == 0,
        "PP words must be a positive multiple of 8, got {}",
        config.words
    );
    {
        let mut seen = config.device_ordinals.clone();
        seen.sort_unstable();
        seen.dedup();
        ensure!(
            seen.len() == n,
            "PP stages must use distinct GPUs, got {:?}",
            config.device_ordinals
        );
    }

    let handshakes: Vec<Mutex<Option<Glm52StageVas>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let barrier = Barrier::new(n);
    let cfg = &config;
    let hs = &handshakes;
    let bar = &barrier;

    std::thread::scope(|scope| -> Result<Glm52PpSpineReport> {
        let handles: Vec<_> = (0..n)
            .map(|i| scope.spawn(move || run_stage(i, n, cfg, hs, bar)))
            .collect();

        let mut outputs = Vec::with_capacity(n);
        for handle in handles {
            outputs.push(
                handle
                    .join()
                    .map_err(|_| anyhow!("PP stage thread panicked"))??,
            );
        }
        Ok(build_report(cfg, &outputs))
    })
}

fn run_stage(
    stage: usize,
    n: usize,
    cfg: &Glm52PpSpineConfig,
    handshakes: &[Mutex<Option<Glm52StageVas>>],
    barrier: &Barrier,
) -> Result<StageOutput> {
    let role = if stage == 0 {
        StageRole::Source
    } else if stage == n - 1 {
        StageRole::Sink
    } else {
        StageRole::Middle
    };
    let ordinal = cfg.device_ordinals[stage];
    let total_iters = cfg.warmup + cfg.iters;

    let gctx = Glm52RankGpuContext::new(ordinal)?;
    gctx.set_current()?;
    let dctx = gctx.as_device_context();

    // Grant both neighbours access to this stage's memory pool BEFORE allocating
    // from it -- the upstream remote-writes our hidden/flag rings, the downstream
    // our ack ring. Neighbour ordinals come straight from the config; only the
    // peer VAs need the post-barrier handshake.
    let mut neighbors = Vec::new();
    if stage > 0 {
        neighbors.push(cfg.device_ordinals[stage - 1]);
    }
    if stage + 1 < n {
        neighbors.push(cfg.device_ordinals[stage + 1]);
    }
    grant_pool_peer_access(ordinal, &neighbors)?;

    // `buffers` is declared before `graph` so the graph's handles are destroyed
    // before the rings they reference are freed.
    let mut buffers =
        Glm52StageBuffers::new(gctx.stream(), cfg.ring, cfg.words, cfg.iters as usize)?;

    // Phase 1: publish this stage's peer target VAs.
    *handshakes[stage].lock().unwrap() = Some(buffers.peer_targets(gctx.stream()));
    barrier.wait();

    // Phase 2: resolve the peer edge from the neighbours' published VAs.
    let down = (stage + 1 < n).then(|| read_handshake(handshakes, stage + 1));
    let up = (stage > 0).then(|| read_handshake(handshakes, stage - 1));
    let edge = Glm52PeerEdge {
        down_hidden: down.map(|v| v.hidden_in_ring).unwrap_or(0),
        down_flag: down.map(|v| v.flag_ring).unwrap_or(0),
        up_ack: up.map(|v| v.ack_ring).unwrap_or(0),
    };

    buffers.reset_control(gctx.stream())?;
    gctx.sync()?;

    let send_params = Glm52PpSendParams {
        words: cfg.words as i32,
        ring: cfg.ring as i32,
        warmup: cfg.warmup,
        n_samples: cfg.iters,
        deadline_ns: cfg.deadline_ns,
    };
    let graph = Glm52StageGraph::capture(&dctx, || {
        if role != StageRole::Source {
            glm52_pp_wait_hidden_launch(
                &dctx,
                &buffers.flag_ring,
                &mut buffers.epoch_counter,
                edge.up_ack,
                &mut buffers.err_code,
                cfg.deadline_ns,
                cfg.ring as i32,
            )?;
        } else {
            glm52_pp_source_inject_launch(&dctx, &mut buffers.epoch_counter)?;
        }
        glm52_pp_dummy_burn_launch(&dctx, cfg.burn_ns)?;
        if role != StageRole::Sink {
            glm52_pp_send_hidden_launch(
                &dctx,
                &buffers.src_hidden,
                edge.down_hidden,
                edge.down_flag,
                &buffers.epoch_counter,
                &buffers.ack_ring,
                Some(&mut buffers.deltas),
                &mut buffers.err_code,
                send_params,
            )?;
        }
        Ok(())
    })?;

    // Phase 3: drive the chain. All stages cross this barrier together, then each
    // enqueues every replay; the device flags do the cross-GPU serialization.
    barrier.wait();
    gctx.set_current()?;
    let start = Instant::now();
    for _ in 0..total_iters {
        graph.launch(&dctx)?;
    }
    gctx.sync()?;
    let wall = start.elapsed();

    // No host-side err_code read: a desync `__trap()`s in-kernel and poisons the
    // context, so `sync()` above already returned the sticky error and aborted
    // the run. The per-gate err_code latch (1=wait deadline, 2=ring lap, 3=WAR
    // deadline, 4=RTT deadline) survives in device memory for cuda-gdb only --
    // the poisoned context rejects the D2H copy that would read it back here.
    let deltas = match role {
        StageRole::Sink => None,
        _ => Some(gctx.stream().clone_dtoh(&buffers.deltas)?),
    };
    Ok(StageOutput { wall, deltas })
}

fn read_handshake(handshakes: &[Mutex<Option<Glm52StageVas>>], idx: usize) -> Glm52StageVas {
    handshakes[idx]
        .lock()
        .unwrap()
        .expect("neighbour published its VAs before the barrier")
}

fn build_report(cfg: &Glm52PpSpineConfig, outputs: &[StageOutput]) -> Glm52PpSpineReport {
    // The wall clock spans warmup + measured launches, so amortize over both.
    let total_iters = (cfg.warmup + cfg.iters) as f64;
    let mut hops = Vec::new();
    for (stage, out) in outputs.iter().enumerate() {
        let Some(raw) = &out.deltas else { continue };
        let mut us: Vec<f64> = raw.iter().map(|&ns| ns as f64 / 1000.0).collect();
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        hops.push(Glm52PpHopStats {
            hop: stage,
            rtt_p50_us: percentile(&us, 0.50),
            rtt_p90_us: percentile(&us, 0.90),
            rtt_p99_us: percentile(&us, 0.99),
            rtt_p999_us: percentile(&us, 0.999),
            rtt_max_us: us.last().copied().unwrap_or(0.0),
            gt10us: us.iter().filter(|&&v| v > 10.0).count(),
            gt100us: us.iter().filter(|&&v| v > 100.0).count(),
        });
    }
    let chain_rtt_p50_us = hops.iter().map(|h| h.rtt_p50_us).sum();
    let wall_per_iter_us = outputs
        .iter()
        .map(|o| o.wall.as_secs_f64() * 1e6 / total_iters)
        .fold(0.0_f64, f64::max);
    Glm52PpSpineReport {
        pp_size: cfg.device_ordinals.len(),
        words: cfg.words,
        ring: cfg.ring,
        burn_ns: cfg.burn_ns,
        hops,
        chain_rtt_p50_us,
        wall_per_iter_us,
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}
