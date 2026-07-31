//! Free-running per-rank engines. Every logical DP rank is an autonomous
//! engine: its own request queue, slots, KV pool, offload/P-D state, and
//! load feed, driven by its own thread. There is no coordinator — the only
//! coupling between engines is the fixed-cadence DeepEP collective chain
//! itself (75 MoE layers per step plus the fixed MTP round): every engine
//! steps unconditionally, idle ranks enter with padding rows, and the
//! collective's back-pressure is the synchronization. The invariants that
//! make that safe are structural, not negotiated: a fixed chain (no
//! conditional collectives), conservative protocol-max shapes, and
//! deterministic padding rows (`docs/models/glm52/free-running-dp.md` §4).
//!
//! Each engine admits up to `GLM52_MAX_BATCH_PER_RANK` requests from its own
//! queue (the `EngineHandle` routes frontend requests to rank queues by
//! `data_parallel_rank`). KV pages come from the rank's [`BlockPool`]
//! (64-token pages, content-hashed blocks): admission reserves a request's
//! full-lifetime page count up front (honor-or-reject — a request that can
//! never fit is rejected, one that can't fit *now* stays queued), so decode
//! can never run out of pages mid-request, and released requests' sealed
//! blocks stay matchable as the prefix cache.
//!
//! Every step the engine forwards its OWN batch bucket — each active slot's
//! *span* of next tokens (mid-prefill slots batch up to a bucket of
//! consecutive prompt positions through one step; decode slots feed one
//! row), idle slots feed a padding row whose output is discarded. The
//! bucket is the smallest member of `GLM52_DECODE_BUCKETS` covering the
//! rank's own row demand. The TP4 replicated topology is the N=1
//! special case: ONE logical rank's engine drives every mirrored worker
//! with the identical step, and the join asserts bit-identical results (the
//! replicated-activations contract); as the sole issuer of its collectives
//! it may block while fully idle instead of free-running.
//!
//! The per-request decisions (what to feed next, what a step's output means)
//! live in [`Glm52SlotState`] as pure data transitions, and the
//! admission/step-shape decisions in [`admission`] / [`plan`] as pure
//! functions over the occupancy and feed wants.

mod admission;
#[cfg(test)]
mod contract_tests;
mod graph;
mod load;
mod mtp;
mod offload;
mod plan;
mod slot;
#[cfg(test)]
mod testkit;

use std::collections::VecDeque;

use admission::admit_from_queue;
use anyhow::Context as _;
use graph::GraphDumpRequest;
use graph::dump_rank0_decode_graph;
use graph::precapture_step_graphs;
use load::publish_load;
use mtp::run_mtp_round;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::LoadSnapshot;
use openinfer_core::engine::TokenEvent;
use openinfer_kv_cache::BlockPool;
use openinfer_kv_cache::RequestKv;
use openinfer_kv_offload::OffloadEngine;
use openinfer_sample::mix_seed;
use plan::collect_sampling_rows;
use plan::feed_wants;
use plan::lease_flags;
use plan::plan_prefill_spans;
use plan::plan_step_shape;
use plan::takes_argmax;
use sha2::Digest as _;
use sha2::Sha256;
use slot::GLM52_PADDING_STEP;
use slot::Glm52SlotState;
use slot::Glm52StepOutcome;
#[cfg(test)]
pub(crate) use slot::MTP_PRODUCTION_GATE_REQUEST_ID;
#[cfg(test)]
pub(crate) use slot::MTP_SLOT_REUSE_GATE_REQUEST_ID;
#[cfg(test)]
pub(crate) use slot::mtp_production_stats;
#[cfg(test)]
pub(crate) use slot::reset_mtp_production_stats;
use tokio::sync::mpsc;
use tokio::sync::watch;

use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::GLM52_MODEL_LEN_ALIGN;
use crate::model::Glm52StepKv;
use crate::model::Glm52StepShape;
use crate::model::glm52_pool_blocks;
use crate::model::glm52_table_width;
use crate::runner::Glm52MtpAppend;
use crate::runner::Glm52PrefillBatch;
use crate::runner::Glm52StepFlags;
use crate::runner::Glm52Worker;

/// The KV page size (== the FlashMLA page / index-K block / model-len
/// alignment — one 64 everywhere).
const PAGE: usize = GLM52_MODEL_LEN_ALIGN;

/// Engine-level philox seed for unseeded non-greedy rows (the Kimi
/// convention: unseeded requests need no replay guarantee, so a fixed engine
/// seed suffices; per-request `seed` params replay through `mix_seed`).
const GLM52_SAMPLE_SEED: u64 = 42;

fn prefix_cache_enabled(drafter: &crate::Glm52Drafter, no_prefix_cache: bool) -> bool {
    !drafter.enabled() && !no_prefix_cache
}

#[cfg(test)]
mod prefix_cache_policy_tests {
    use super::*;

    #[test]
    fn native_mtp_never_matches_target_only_prefix_state() {
        assert!(!prefix_cache_enabled(
            &crate::Glm52Drafter::NativeMtp,
            false
        ));
        assert!(prefix_cache_enabled(&crate::Glm52Drafter::None, false));
    }
}

struct ActiveRequest {
    req: GenerateRequest,
    state: Glm52SlotState,
    /// Prompt length from the client request. Native P/D v2 appends P's
    /// anchor internally, but OpenAI usage must still report the original.
    client_prompt_tokens: usize,
    /// The request's page assignments in the rank's pool. Block RAII: blocks
    /// return to the pool (registered ones as matchable prefix-cache entries)
    /// when this drops or `release()`s.
    kv: RequestKv,
}

/// Per-rank slot occupancy: `slots[slot]`.
type RankSlots = [Option<ActiveRequest>; GLM52_MAX_BATCH_PER_RANK];

/// What one slot's span asked kvbm for this step — decides which `apply_*`
/// commits the outputs (schedule and apply must pair exactly).
#[derive(Clone, Copy, Debug)]
enum SpanKind {
    /// Prompt span that does NOT finish the prompt: KV advances, no token.
    PrefillChunk,
    /// Prompt span whose last row feeds the final prompt token: its output
    /// is the first generated token.
    PrefillBoundary,
    /// Single decode row (the zero-draft case).
    Decode,
    /// Verify span: anchor + fed drafts, committing the accepted prefix.
    Speculative,
}

/// Everything one engine needs at spawn: the per-rank pieces (queue,
/// workers, load feed) plus the shared launch configuration. Construction
/// happens inside the engine thread ([`Glm52Engine::spawn`]) so a startup
/// failure tears this rank's workers down concurrently with its siblings'.
pub(crate) struct Glm52EngineSpec {
    pub(crate) rank: usize,
    pub(crate) submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    /// This rank's executors: exactly one under EP, every mirrored worker
    /// under the tensor-replicated topologies.
    pub(crate) workers: Vec<Glm52Worker>,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) drafter: crate::Glm52Drafter,
    pub(crate) prefill_chunk_size: Option<usize>,
    pub(crate) max_model_len: usize,
    pub(crate) no_prefix_cache: bool,
    /// This rank's offload engines (several only under a mirrored topology,
    /// which uses the first — the historical layout); they hold the shared
    /// pegaflow host, which must outlive every in-flight save.
    pub(crate) offload: Option<Vec<OffloadEngine>>,
    /// Fleet-wide logical rank count — the P/D states are sized by it so
    /// their indexing (and their log lines) keep the global rank numbers.
    pub(crate) logical_ranks: usize,
    pub(crate) moe_topo: crate::Glm52MoeTopo,
    pub(crate) load_tx: watch::Sender<LoadSnapshot>,
    pub(crate) graph_dump_request: Option<GraphDumpRequest>,
    /// Bootstrap report: the engine sends once after graph pre-capture (and
    /// the rank-0 graph dump), or once with the failure that killed it.
    pub(crate) startup_tx: crossbeam_channel::Sender<anyhow::Result<()>>,
}

/// One logical DP rank's autonomous engine. Owns its workers — and its
/// offload engines: they hold the shared pegaflow host, which must outlive
/// every in-flight save and dies with the engine.
pub(crate) struct Glm52Engine {
    rank: usize,
    submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52Worker>,
    eos_token_ids: Vec<u32>,
    drafter: crate::Glm52Drafter,
    prefill_chunk_size: Option<usize>,
    max_model_len: usize,
    prefix_cache: bool,
    offload: Option<Vec<offload::RankOffload>>,
    native_pd: Option<offload::NativePdState>,
    /// Plain host-tier restore in flight for this rank's queue front (the
    /// non-P/D admission leg) — polled at step boundaries, never blocking.
    host_restore: Option<offload::HostRestoreState>,
    moe_topo: crate::Glm52MoeTopo,
    load_tx: watch::Sender<LoadSnapshot>,
    graph_dump_request: Option<GraphDumpRequest>,
    startup_tx: crossbeam_channel::Sender<anyhow::Result<()>>,
    /// Tensor-replicated topology: this rank drives mirrored executors with
    /// identical steps (bit-identical outputs asserted at the join).
    mirrored: bool,
    /// Verify-span draft budget: EP feeds 3 (the measured bucket-4 optimum);
    /// TP4 mirrored topology feeds the drafter's full proposal.
    span_drafts: usize,
    pool: BlockPool,
    table_width: usize,
    /// Pool pages available to requests (total minus the padding page) —
    /// constant for the engine's lifetime.
    usable_blocks: usize,
    slots: RankSlots,
    pending: VecDeque<GenerateRequest>,
    /// Slot draft states to clear on the next draft round (request left the
    /// slot, or a new one was admitted into it). Flushed with each step's
    /// Draft commands; the handler is idempotent, so duplicates are harmless.
    pending_resets: Vec<usize>,
    /// The shape this engine leased the NEXT step as: the device already
    /// holds that step's speculative replay, so the next step is pinned to
    /// this shape (see [`plan::lease_flags`]).
    leased_shape: Option<Glm52StepShape>,
    /// Slots whose requests finished while a lease was outstanding: their
    /// rows ride the leased replay (outputs discarded), so their physical
    /// release waits for the consume step — freeing the pages earlier would
    /// let admission hand them to another request while the replay still
    /// writes them.
    deferred_releases: Vec<usize>,
    /// Rank-local step counter driving the non-greedy rows' philox seeds (a
    /// fresh well-mixed seed per (step, rank); ranks never compare seeds).
    sample_step: u64,
    channel_open: bool,
}

impl Glm52Engine {
    /// Spawn the engine's thread. The KV pool is built inside the thread:
    /// its sizing arithmetic is the one fallible construction step, and a
    /// failure must still report on `startup_tx` and let the spec's drop
    /// shut this rank's workers down (each worker's own Drop sends Shutdown
    /// and joins — the destroy barrier pairs once the launcher tears the
    /// fleet down on the failed report).
    pub(crate) fn spawn(spec: Glm52EngineSpec) -> std::io::Result<std::thread::JoinHandle<()>> {
        let rank = spec.rank;
        std::thread::Builder::new()
            .name(format!("glm52-engine-{rank}"))
            .spawn(move || {
                let startup_tx = spec.startup_tx.clone();
                match Glm52Engine::new(spec) {
                    Ok(engine) => engine.run(),
                    Err(err) => {
                        let _ = startup_tx.send(Err(
                            err.context(format!("GLM5.2 rank {rank} KV pool construction"))
                        ));
                    }
                }
            })
    }

    fn new(spec: Glm52EngineSpec) -> anyhow::Result<Self> {
        let prefill_only = spec.prefill_chunk_size.is_some();
        let mirrored = spec.moe_topo.uses_tensor_replicated_moe();
        debug_assert_eq!(
            spec.workers.len(),
            if mirrored {
                spec.moe_topo.device_count()
            } else {
                1
            },
            "one executor per EP rank; every mirrored worker under TP"
        );
        let offload: Option<Vec<offload::RankOffload>> = spec
            .offload
            .map(|engines| engines.into_iter().map(offload::RankOffload::new).collect());
        let native_pd = (spec.drafter.is_mtp() && offload.is_some() && !prefill_only)
            .then(|| offload::NativePdState::new(spec.logical_ranks));
        // One KV page pool for this rank: pool block ids index the rank's
        // per-layer MLA and index-K arenas directly (the arenas were built
        // for `glm52_pool_blocks` blocks). Block 0-equivalent is the reserved
        // padding page. Under mirrored TP the single pool drives every executor — the
        // mirrored steps write the identical block ids on all 8 arenas.
        let pool = BlockPool::new(
            PAGE,
            glm52_pool_blocks(
                spec.max_model_len,
                if prefill_only {
                    1
                } else {
                    GLM52_MAX_BATCH_PER_RANK
                },
            ),
        )?;
        let table_width = glm52_table_width(spec.max_model_len);
        // A cache-hit prefix skips state required by either speculative lane:
        // DSpark loses the aux-hidden captures it consumes. Native MTP loses
        // continuity in its separate KV cache; TP4 P additionally restores only
        // the 656-byte wire cache, not its 576-byte local proposal cache. Prefix
        // matching therefore stays off while any drafter is active.
        let prefix_cache = prefix_cache_enabled(&spec.drafter, spec.no_prefix_cache);
        if spec.drafter.enabled() && !prefix_cache && !spec.no_prefix_cache {
            log::info!("GLM5.2 prefix cache disabled: speculative decoding is on");
        }
        let host_restore = (offload.is_some() && prefix_cache).then(offload::HostRestoreState::new);
        Ok(Self {
            rank: spec.rank,
            submit_rx: spec.submit_rx,
            workers: spec.workers,
            eos_token_ids: spec.eos_token_ids,
            drafter: spec.drafter,
            prefill_chunk_size: spec.prefill_chunk_size,
            max_model_len: spec.max_model_len,
            prefix_cache,
            offload,
            native_pd,
            host_restore,
            moe_topo: spec.moe_topo,
            load_tx: spec.load_tx,
            graph_dump_request: spec.graph_dump_request,
            startup_tx: spec.startup_tx,
            mirrored,
            span_drafts: if mirrored {
                crate::dspark::GLM52_DSPARK_DRAFTS
            } else {
                slot::GLM52_DSPARK_EP8_SPAN_DRAFTS
            },
            usable_blocks: pool.total_blocks() - 1,
            table_width,
            pool,
            slots: std::array::from_fn(|_| None),
            pending: VecDeque::new(),
            pending_resets: Vec::new(),
            leased_shape: None,
            deferred_releases: Vec::new(),
            sample_step: 0,
            channel_open: true,
        })
    }

    pub(crate) fn run(mut self) {
        if let Err(err) = self.bootstrap() {
            let _ = self.startup_tx.send(Err(err));
            self.shutdown_workers();
            return;
        }
        let _ = self.startup_tx.send(Ok(()));
        self.serve_loop();
        self.teardown();
    }

    /// Pre-capture every bucket graph, then (rank 0 only) service the graph
    /// dump. Every engine captures the same fixed bucket sequence, so the
    /// collectives inside capture pair across the fleet without any
    /// coordinator — the pre-capture IS the bootstrap rendezvous.
    fn bootstrap(&mut self) -> anyhow::Result<()> {
        if self.prefill_chunk_size.is_none() {
            precapture_step_graphs(
                &self.workers,
                std::slice::from_ref(&self.pool),
                self.table_width,
                self.mirrored,
            )?;
        }
        if let Some((png_path, response)) = self.graph_dump_request.take() {
            match dump_rank0_decode_graph(&self.workers, self.moe_topo, png_path) {
                Ok(summary) => {
                    let _ = response.send(Ok(summary));
                }
                Err(err) => {
                    log::error!("GLM5.2 CUDA Graph export failed: {err:#}");
                    let _ = response.send(Err(anyhow::anyhow!("{err:#}")));
                    return Err(err.context("GLM5.2 CUDA Graph export"));
                }
            }
        }
        Ok(())
    }

    fn serve_loop(&mut self) {
        'serve: loop {
            // Intake: EP engines never block — an idle EP rank runs padding
            // steps at full speed so its peers never wait on it in a
            // collective (the free-running contract). A mirrored engine is
            // the sole issuer of its collectives, so it may block while
            // fully idle instead of burning the machine on padding.
            if self.channel_open && self.all_idle() && self.pending.is_empty() {
                self.publish();
                if self.mirrored {
                    match self.submit_rx.blocking_recv() {
                        Some(req) => self.intake(req),
                        None => self.channel_open = false,
                    }
                }
            }
            while self.channel_open {
                match self.submit_rx.try_recv() {
                    Ok(req) => self.intake(req),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => self.channel_open = false,
                }
            }
            if !self.channel_open && self.all_idle() && self.pending.is_empty() {
                break;
            }

            // Admission freezes while a speculation is outstanding: the
            // lease pins the next step's shape, so newcomers wait one step.
            if self.leased_shape.is_none()
                && let Err(err) = self.admit()
            {
                self.fatal(&err);
            }
            self.publish();

            // A mirrored engine with nothing to run has nothing to pace: its
            // collectives are all intra-process, so skipping the step changes
            // nothing observable (and an empty prefill batch is invalid). EP
            // engines step unconditionally — their peers may be busy.
            if self.mirrored && self.all_idle() {
                continue 'serve;
            }

            if let Some(max_rows) = self.prefill_chunk_size {
                if self
                    .slots
                    .iter()
                    .flatten()
                    .any(|active| !active.state.mid_prefill())
                {
                    self.fatal(&anyhow::anyhow!(
                        "GLM5.2 prefill-only invariant failed: a request reached decode"
                    ));
                }
                self.sample_step += 1;
                if let Err(err) = self.prefill_step(max_rows) {
                    self.fatal(&err);
                }
                continue 'serve;
            }

            // One step: this rank's own bucket — each active slot's span of
            // consecutive next tokens, padding rows on the free slots. The
            // shape comes from the lease if one is outstanding (the device
            // already holds that step's speculative replay — the shape is
            // pinned, which is why admission froze), else from the rank's
            // own feed wants.
            let consume = self.leased_shape.is_some();
            let shape = match self.leased_shape.take() {
                Some(leased) => leased,
                None => plan_step_shape(&feed_wants(&self.slots)),
            };
            let flags = lease_flags(
                consume,
                self.pending.is_empty(),
                self.drafter.enabled(),
                self.offload.is_some(),
                !self.deferred_releases.is_empty(),
                &self.slots,
                self.max_model_len,
            );
            self.leased_shape = flags.lease.then_some(shape);
            self.sample_step += 1;
            let (outputs, span_kinds, step_inputs) = match self.submit_and_join_step(&shape, flags)
            {
                Ok(step) => step,
                Err(err) => self.fatal(&err),
            };
            let (rank_appends, mtp_appends, mut rank_proposals) =
                match self.apply_step_outputs(&outputs, &shape, span_kinds, &step_inputs) {
                    Ok(walked) => walked,
                    Err(err) => self.fatal(&err),
                };
            // Deferred releases complete ONLY at the end of the consume
            // step: the speculation they wait on was enqueued by the lease
            // step and replays during this one — freeing the pages any
            // earlier would hand them to admission while the replay still
            // writes them. (A lease step may ADD deferrals; it never
            // completes them.)
            if consume {
                self.release_deferred();
            }

            // Mirrored-TP speculative policy: draft only when the rank is
            // solo — a concurrent batch's bucket rows go to liveness first.
            // Suppress the proposals (appends and resets still flow, so the
            // drafter's shadow KV stays fresh and proposals resume the round
            // after the batch drains back to solo). Drafts already installed
            // on the solo slot are deliberately left to drain.
            if self.mirrored && self.slots.iter().flatten().count() != 1 {
                rank_proposals.clear();
            }

            let draft_result = if self.drafter.is_dspark() {
                self.run_draft_round(shape.bucket, rank_appends, rank_proposals)
            } else if self.drafter.is_mtp() {
                run_mtp_round(
                    self.rank,
                    &self.workers[0],
                    &mut self.slots,
                    shape.bucket,
                    &mut self.pending_resets,
                    mtp_appends,
                    rank_proposals,
                )
            } else {
                Ok(())
            };
            if let Err(err) = draft_result {
                self.fatal(&err);
            }
        }
    }

    fn all_idle(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    fn intake(&mut self, req: GenerateRequest) {
        if let Err(message) = admission::validate_request(
            &req,
            self.max_model_len,
            self.prefill_chunk_size.is_some(),
            self.prefill_chunk_size.is_some() && self.drafter.is_mtp(),
        ) {
            admission::reject(&req, message);
            return;
        }
        debug_assert!(
            req.data_parallel_rank.is_none_or(|rank| rank == self.rank),
            "GLM5.2 rank {} received a request bound for rank {:?}",
            self.rank,
            req.data_parallel_rank
        );
        self.pending.push_back(req);
    }

    fn admit(&mut self) -> anyhow::Result<()> {
        admit_from_queue(
            self.rank,
            &mut self.pending,
            &mut self.slots,
            &self.pool,
            self.usable_blocks,
            self.offload.as_deref().and_then(<[_]>::first),
            &mut self.native_pd,
            &mut self.host_restore,
            self.prefix_cache,
            self.drafter.enabled(),
            self.prefill_chunk_size.is_some() && self.drafter.is_mtp(),
            &mut self.pending_resets,
        )
    }

    fn publish(&self) {
        publish_load(&self.load_tx, &self.pool, &self.slots, &self.pending);
    }

    /// One step: submit — schedule each active span's KV (full-lifetime
    /// reservation makes every schedule succeed; a failure is an accounting
    /// bug and is engine-fatal), build the row inputs, page rows and write
    /// slots, collect the step's sampling rows, and fire — then join ALL
    /// executors before failing: the executor recv'd first often reports the
    /// ~100 s DeepEP device-timeout trap, not the root cause. Returns the
    /// rank's outputs plus what the submit phase scheduled per slot
    /// (`span_kinds[slot]`), which the output walk pairs exactly.
    #[allow(clippy::type_complexity)]
    fn submit_and_join_step(
        &mut self,
        shape: &Glm52StepShape,
        flags: Glm52StepFlags,
    ) -> anyhow::Result<(
        [u32; GLM52_MAX_BATCH_PER_RANK],
        [Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK],
        [(u32, usize); GLM52_MAX_BATCH_PER_RANK],
    )> {
        let pool = &self.pool;
        let padding_page = pool.padding_block_id();
        let sampling = collect_sampling_rows(shape, &self.slots);
        let seed = mix_seed(
            mix_seed(GLM52_SAMPLE_SEED, self.sample_step),
            self.rank as u64,
        );
        let mut span_kinds = [None; GLM52_MAX_BATCH_PER_RANK];
        let mut inputs =
            [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position); GLM52_MAX_BATCH_PER_RANK];
        // A consumed speculation replays with device-advanced inputs and
        // never reads the step KV — skip building the page rows (the whole
        // point of launch-ahead is keeping this host path off the hot step
        // boundary). KV *scheduling* still runs: kvbm's bookkeeping must
        // advance every step.
        let mut pages = if flags.consume {
            Vec::new()
        } else {
            vec![padding_page; shape.bucket * self.table_width]
        };
        let mut slot_mapping = [padding_page as i64 * PAGE as i64; GLM52_MAX_BATCH_PER_RANK];
        // Walk the shape's contiguous per-slot runs.
        let mut row = 0usize;
        while row < shape.bucket {
            let slot_id = shape.slots[row] as usize;
            let mut end = row + 1;
            while end < shape.bucket && shape.slots[end] as usize == slot_id {
                end += 1;
            }
            let span = end - row;
            if self.deferred_releases.contains(&slot_id) {
                // A dead slot's rows ride the replay with the padding
                // defaults; its KV bookkeeping stopped at the finish.
                row = end;
                continue;
            }
            let Some(active) = self.slots[slot_id].as_mut() else {
                // Padding rows keep the padding-page defaults.
                row = end;
                continue;
            };
            for (offset, r) in (row..end).enumerate() {
                let step = active.state.next_input_at(offset);
                inputs[r] = (step.token, step.position);
            }
            // The span must extend kvbm's view exactly: its first row's
            // position is the next KV slot to write. Drift between the
            // slot state's position math and the pool's bookkeeping
            // writes KV into the wrong page — fail the step instead.
            if inputs[row].1 != active.kv.kv_position() {
                return Err(anyhow::anyhow!(
                    "GLM5.2 rank {} slot {slot_id} span starts at position {} but the \
                     KV pool is at {}",
                    self.rank,
                    inputs[row].1,
                    active.kv.kv_position()
                ));
            }
            let mid_prefill = active.state.mid_prefill();
            let (kind, scheduled) = if mid_prefill {
                let kind = if active.state.remaining_prompt() == span {
                    SpanKind::PrefillBoundary
                } else {
                    SpanKind::PrefillChunk
                };
                (kind, active.kv.schedule_prefill(span, pool))
            } else if span == 1 {
                (SpanKind::Decode, active.kv.schedule_decode(pool))
            } else {
                (
                    SpanKind::Speculative,
                    active.kv.schedule_speculative(span, pool),
                )
            };
            if let Err(err) = scheduled {
                return Err(anyhow::anyhow!(
                    "GLM5.2 rank {} slot {slot_id} violated its full-lifetime KV \
                     reservation ({kind:?}, span {span}): {err}",
                    self.rank
                ));
            }
            span_kinds[slot_id] = Some(kind);
            if !flags.consume {
                let row_pages = active.kv.step_page_indices(span);
                for r in row..end {
                    pages[r * self.table_width..r * self.table_width + row_pages.len()]
                        .copy_from_slice(&row_pages);
                    let position = inputs[r].1;
                    slot_mapping[r] =
                        row_pages[position / PAGE] as i64 * PAGE as i64 + (position % PAGE) as i64;
                }
            }
            row = end;
        }
        let kv = Glm52StepKv {
            pages: pages.into_boxed_slice(),
            slot_mapping,
        };
        // Logical-to-executor mapping: 1:1 under EP, or this rank's step
        // mirrored onto every worker under the replicated topology
        // (identical inputs/KV/seed, bit-identical outputs asserted at the
        // join).
        let mut responses = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            responses.push(worker.step_async(
                inputs,
                *shape,
                kv.clone(),
                flags,
                sampling.clone(),
                seed,
            )?);
        }
        let mut outputs = Vec::with_capacity(responses.len());
        let mut step_err: Option<anyhow::Error> = None;
        for (executor, resp) in responses.into_iter().enumerate() {
            let result = resp.recv().map_err(|_| {
                anyhow::anyhow!(
                    "GLM5.2 rank {} executor {executor} dropped its step response",
                    self.rank
                )
            });
            match result {
                Ok(Ok(step_tokens)) => outputs.push(step_tokens),
                Ok(Err(err)) | Err(err) => {
                    let err = err.context(format!(
                        "GLM5.2 rank {} executor {executor} step",
                        self.rank
                    ));
                    log::error!(
                        "GLM5.2 rank {} executor {executor} step failed: {err:#}",
                        self.rank
                    );
                    step_err.get_or_insert(err);
                    outputs.push([0; GLM52_MAX_BATCH_PER_RANK]);
                }
            }
        }
        if let Some(err) = step_err {
            return Err(err);
        }
        if self.mirrored {
            // The replicated contract: every executor computed the identical
            // step, so any divergence means the redundant compute desynced —
            // serving on it would emit rank-dependent garbage. Crash early.
            for (executor, out) in outputs.iter().enumerate().skip(1) {
                anyhow::ensure!(
                    out == &outputs[0],
                    "GLM5.2 mirrored executor {executor} step outputs diverge from executor 0 \
                     (the replicated bit-identity contract broke)"
                );
            }
            outputs.truncate(1);
        }
        Ok((outputs[0], span_kinds, inputs))
    }

    /// Fold this rank's span of outputs into its slot states, commit the
    /// span's KV bookkeeping under the exact kind the submit phase scheduled
    /// (a mispairing is an engine bug and is fatal), emit tokens and
    /// finish/disconnect releases, and collect the draft lane's context
    /// appends and next-round proposals.
    #[allow(clippy::type_complexity)]
    fn apply_step_outputs(
        &mut self,
        outputs: &[u32; GLM52_MAX_BATCH_PER_RANK],
        shape: &Glm52StepShape,
        span_kinds: [Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK],
        step_inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
    ) -> anyhow::Result<(
        Vec<(usize, usize)>,
        Vec<Glm52MtpAppend>,
        Vec<(usize, u32, usize)>,
    )> {
        let offload = self.offload.as_deref().and_then(<[_]>::first);
        let mut rank_appends = Vec::new();
        let mut mtp_appends = Vec::new();
        let mut rank_proposals = Vec::new();
        // Walk the shape's contiguous per-slot runs; each active slot folds
        // its whole span of row outputs in at once.
        let mut row = 0usize;
        while row < shape.bucket {
            let slot_id = shape.slots[row] as usize;
            let mut end = row + 1;
            while end < shape.bucket && shape.slots[end] as usize == slot_id {
                end += 1;
            }
            let span_rows = row..end;
            let span_outputs = &outputs[span_rows.clone()];
            row = end;
            if self.deferred_releases.contains(&slot_id) {
                // The replay row's output is discarded; the release was
                // handled at the finish and completes in `release_deferred`.
                continue;
            }
            let slot = &mut self.slots[slot_id];
            let Some(active) = slot.as_mut() else {
                continue;
            };
            let prompt_tokens = active.client_prompt_tokens;
            let outcome = active.state.advance_span(span_outputs, &self.eos_token_ids);
            // Commit the span's KV bookkeeping under the exact kind the
            // submit phase scheduled — a mispairing is an engine bug and
            // is fatal.
            let pool = &self.pool;
            let applied = match (&outcome, span_kinds[slot_id]) {
                (Glm52StepOutcome::Prefilling, Some(SpanKind::PrefillChunk)) => {
                    active.kv.apply_prefill_chunk(pool)
                }
                (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::PrefillBoundary)) => {
                    active.kv.apply_prefill(committed[0], pool)
                }
                (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::Decode)) => {
                    active.kv.apply_decode(committed[0], pool).map(|_| ())
                }
                (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::Speculative)) => {
                    active.kv.apply_speculative(committed, pool).map(|_| ())
                }
                (outcome, kind) => Err(anyhow::anyhow!(
                    "GLM5.2 rank {} slot {slot_id} outcome {outcome:?} does not pair \
                     with scheduled span kind {kind:?}",
                    self.rank
                )),
            };
            if let Err(err) = applied {
                return Err(
                    err.context(format!("GLM5.2 rank {} slot {slot_id} KV apply", self.rank))
                );
            }
            let (freed, context_rows) = match outcome {
                Glm52StepOutcome::Prefilling => {
                    // Prefill never sends, so a disconnect is only
                    // visible through the sink probe — without it a
                    // long prompt zombies the slot until prefill
                    // completes. Every prompt row is committed
                    // context.
                    (active.req.token_tx.is_closed(), span_outputs.len())
                }
                Glm52StepOutcome::Commit {
                    committed,
                    emit,
                    finish,
                    context_rows,
                } => {
                    // A dropped receiver (client disconnect) frees the
                    // slot; its pool pages release with the request
                    // (sealed blocks stay matchable as prefix cache).
                    let mut freed = false;
                    for &token in &committed[..emit] {
                        if active
                            .req
                            .token_tx
                            .send(TokenEvent::Token {
                                id: token,
                                logprob: None,
                            })
                            .is_err()
                        {
                            freed = true;
                            break;
                        }
                    }
                    if let Some(finish_reason) = finish
                        && !freed
                    {
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        freed = true;
                    }
                    (freed, context_rows)
                }
            };
            if freed {
                #[cfg(test)]
                active
                    .state
                    .record_mtp_production_gate(active.req.request_id.as_deref());
                active.state.log_spec_stats(self.rank, slot_id);
                // Offload the freshly-sealed blocks BEFORE release: the
                // hashes and guards come off the still-assigned request
                // state, and the guards keep the pages pinned through the
                // async D2H copy.
                if let Some(offload) = offload {
                    offload.save_sealed_on_release(&active.kv);
                }
                if self.leased_shape.is_some() {
                    // A speculation for the next step is already on the
                    // device: the slot's row rides the replay (its output
                    // is discarded), so the physical release waits for the
                    // consume step — freeing the pages now would let
                    // admission hand them to another request while the
                    // replay still writes them.
                    self.deferred_releases.push(slot_id);
                } else {
                    if let Err(err) = active.kv.release() {
                        // Blocks still return via assignment RAII when the
                        // slot drops — the explicit release only failed to
                        // run from a clean Idle state.
                        log::warn!(
                            "GLM5.2 rank {} slot {slot_id} KV release failed \
                             (blocks return via RAII): {err:#}",
                            self.rank
                        );
                    }
                    if self.drafter.enabled() {
                        self.pending_resets.push(slot_id);
                    }
                    *slot = None;
                }
            } else if self.drafter.enabled() {
                if self.drafter.is_dspark() {
                    rank_appends.extend(span_rows.clone().take(context_rows).map(|r| (r, slot_id)));
                } else {
                    for (offset, target_row) in span_rows.clone().take(context_rows).enumerate() {
                        let input_token = if offset + 1 < context_rows {
                            step_inputs[target_row + 1].0
                        } else {
                            active.state.next_input_at(0).token
                        };
                        mtp_appends.push(Glm52MtpAppend {
                            target_row,
                            slot: slot_id,
                            input_token,
                            position: step_inputs[target_row].1,
                            pages: active.kv.current_page_indices(),
                        });
                    }
                }
                let wants_drafts = if self.drafter.is_mtp() {
                    active.state.wants_full_draft(crate::mtp::GLM52_MTP_DRAFTS)
                } else {
                    active.state.wants_drafts()
                };
                if wants_drafts && let Some((anchor, anchor_pos)) = active.state.decode_anchor() {
                    rank_proposals.push((slot_id, anchor, anchor_pos));
                }
            }
        }
        Ok((rank_appends, mtp_appends, rank_proposals))
    }

    /// Physically release the slots whose finishes were deferred by the
    /// lease this step consumed. Their replay rows were skipped by both the
    /// submit and the apply walk; their pages stayed mapped through the
    /// replay, and the client events and offload saves already happened at
    /// the finish.
    fn release_deferred(&mut self) {
        for slot_id in self.deferred_releases.drain(..) {
            let Some(mut active) = self.slots[slot_id].take() else {
                continue;
            };
            if let Err(err) = active.kv.release() {
                log::warn!(
                    "GLM5.2 rank {} slot {slot_id} deferred KV release failed \
                     (blocks return via RAII): {err:#}",
                    self.rank
                );
            }
        }
    }

    /// DSpark draft round (rank-local, no collectives): resets, context
    /// appends from THIS step's capture buffer, and new proposals for the
    /// next verify span. FIFO per-worker channels order it before the next
    /// step; the blocking join keeps the round cadence (draft sits between
    /// verify steps, ~2 ms against a 22-46 ms step).
    fn run_draft_round(
        &mut self,
        bucket: usize,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
    ) -> anyhow::Result<()> {
        let resets = std::mem::take(&mut self.pending_resets);
        if resets.is_empty() && appends.is_empty() && proposals.is_empty() {
            return Ok(());
        }
        let proposal_slots: Vec<usize> = proposals.iter().map(|&(slot, _, _)| slot).collect();
        // Same logical-to-executor mapping as the step submit: under the
        // mirrored topology every worker drafts from its own (identical)
        // capture buffer and must propose the identical spans.
        let (last_worker, fanned) = self.workers.split_last().expect("one executor per rank");
        let mut rxs = Vec::with_capacity(self.workers.len());
        for worker in fanned {
            rxs.push(worker.draft_async(
                bucket,
                resets.clone(),
                appends.clone(),
                proposals.clone(),
            )?);
        }
        // The payloads are fanned out to every executor but the last; the
        // last takes ownership instead of another clone.
        rxs.push(last_worker.draft_async(bucket, resets, appends, proposals)?);
        let mut all_spans = Vec::with_capacity(rxs.len());
        for (executor, rx) in rxs.into_iter().enumerate() {
            let result = rx
                .recv()
                .map_err(|_| {
                    anyhow::anyhow!(
                        "GLM5.2 rank {} executor {executor} dropped its draft response",
                        self.rank
                    )
                })
                .and_then(|r| r);
            match result {
                Ok(spans) => all_spans.push(spans),
                // A draft failure is rank-local, but it means the drafter's
                // invariants broke — crash early rather than silently degrade
                // to plain decode.
                Err(err) => {
                    return Err(err.context(format!(
                        "GLM5.2 rank {} executor {executor} draft",
                        self.rank
                    )));
                }
            }
        }
        for (executor, spans) in all_spans.iter().enumerate().skip(1) {
            anyhow::ensure!(
                spans == &all_spans[0],
                "GLM5.2 mirrored executor {executor} draft spans diverge from executor 0 \
                 (the replicated bit-identity contract broke)"
            );
        }
        let spans = all_spans.swap_remove(0);
        anyhow::ensure!(
            spans.len() == proposal_slots.len(),
            "GLM5.2 rank {} draft returned {} spans for {} proposals",
            self.rank,
            spans.len(),
            proposal_slots.len()
        );
        for (slot_id, span) in proposal_slots.into_iter().zip(spans) {
            if let Some(active) = self.slots[slot_id].as_mut() {
                active.state.set_drafts(span.to_vec(), self.span_drafts);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn prefill_step(&mut self, max_rows: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.workers.len() > 1,
            "GLM5.2 native prefill requires one logical rank mirrored across local TP workers"
        );
        let wants = feed_wants(&self.slots);
        let spans = plan_prefill_spans(&wants, max_rows);
        let pool = &self.pool;
        let mut batch = Glm52PrefillBatch {
            token_ids: Vec::new(),
            positions: Vec::new(),
            request_indptr: vec![0],
            block_indptr: vec![0],
            block_ids: Vec::new(),
            request_slots: Vec::new(),
            padding_block: pool.padding_block_id(),
            slot_mapping: Vec::new(),
            mtp_next_tokens: Vec::new(),
            output_rows: Vec::new(),
            sampling: Vec::new(),
            seed: mix_seed(GLM52_SAMPLE_SEED, self.sample_step),
        };
        let mut scheduled = Vec::new();
        for (slot_id, &span) in spans.iter().enumerate() {
            if span == 0 {
                continue;
            }
            let active = self.slots[slot_id]
                .as_mut()
                .expect("prefill planner assigns only active slots");
            anyhow::ensure!(
                active.state.mid_prefill() && span <= active.state.remaining_prompt(),
                "GLM5.2 prefill planner produced an invalid span"
            );
            anyhow::ensure!(
                active.state.next_input_at(0).position == active.kv.kv_position(),
                "GLM5.2 prefill slot {slot_id} position drift"
            );
            active
                .kv
                .schedule_prefill(span, pool)
                .map_err(|err| anyhow::anyhow!("GLM5.2 prefill slot {slot_id} schedule: {err}"))?;
            let view = active.kv.prefill_view(span);
            for offset in 0..span {
                let input = active.state.next_input_at(offset);
                batch.token_ids.push(input.token);
                batch.positions.push(input.position as u32);
                let page = view.page_indices()[input.position / PAGE];
                batch
                    .slot_mapping
                    .push(page as i64 * PAGE as i64 + (input.position % PAGE) as i64);
            }
            batch.block_ids.extend_from_slice(view.page_indices());
            batch.request_slots.push(slot_id);
            batch.request_indptr.push(batch.token_ids.len() as u32);
            batch.block_indptr.push(batch.block_ids.len() as u32);
            let boundary = span == active.state.remaining_prompt();
            batch
                .mtp_next_tokens
                .push((!boundary).then(|| active.state.next_input_at(span).token));
            if boundary {
                batch.output_rows.push((batch.token_ids.len() - 1) as u32);
                if !takes_argmax(&active.req.params) {
                    batch.sampling.push(crate::runner::Glm52RowSample {
                        row: batch.output_rows.len() - 1,
                        params: active.req.params,
                        step: active.state.completion_tokens() as u64,
                    });
                }
            }
            scheduled.push((slot_id, span, boundary));
        }
        batch.validate()?;

        let responses = self
            .workers
            .iter()
            .map(|worker| worker.prefill_chunk_async(batch.clone()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut outputs = Vec::with_capacity(responses.len());
        for (executor, response) in responses.into_iter().enumerate() {
            outputs.push(
                response
                    .recv()
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "GLM5.2 rank {} executor {executor} dropped prefill response",
                            self.rank
                        )
                    })?
                    .with_context(|| {
                        format!("GLM5.2 rank {} executor {executor} prefill", self.rank)
                    })?,
            );
        }
        for (executor, output) in outputs.iter().enumerate().skip(1) {
            anyhow::ensure!(
                output == &outputs[0],
                "GLM5.2 TP prefill output diverged on executor {executor}: \
                 executor0={:?}, executor{executor}={output:?}",
                outputs[0],
            );
        }
        anyhow::ensure!(
            outputs[0].target_tokens.len() == batch.output_rows.len(),
            "GLM5.2 prefill returned {} boundary outputs, expected {}",
            outputs[0].target_tokens.len(),
            batch.output_rows.len()
        );
        anyhow::ensure!(
            outputs[0].mtp_draft1.is_empty()
                || outputs[0].mtp_draft1.len() == batch.output_rows.len(),
            "GLM5.2 prefill returned {} MTP draft-1 tokens, expected zero or {}",
            outputs[0].mtp_draft1.len(),
            batch.output_rows.len()
        );
        anyhow::ensure!(
            outputs[0].mtp_drafts.is_empty()
                || outputs[0].mtp_drafts.len() == batch.output_rows.len(),
            "GLM5.2 prefill returned {} complete MTP proposals, expected zero or {}",
            outputs[0].mtp_drafts.len(),
            batch.output_rows.len()
        );

        let offload = self.offload.as_deref().and_then(<[_]>::first);
        let mut boundary_output = outputs[0].target_tokens.iter();
        let mut boundary_drafts = outputs[0].mtp_drafts.iter();
        for (slot_id, span, boundary) in scheduled {
            // Proposals are positional batch outputs: consume one for every
            // boundary even if that request's client disconnected. Whether the
            // proposal is published must not shift later requests' mapping.
            let drafts = take_boundary_drafts(boundary, &mut boundary_drafts);
            let slot = &mut self.slots[slot_id];
            let active = slot
                .as_mut()
                .expect("scheduled prefill slot remains active");
            let mut span_outputs = vec![0; span];
            if boundary {
                span_outputs[span - 1] = *boundary_output.next().expect("validated output count");
            }
            let prompt_tokens = active.client_prompt_tokens;
            let outcome = active
                .state
                .advance_span(&span_outputs, &self.eos_token_ids);
            let freed = match outcome {
                Glm52StepOutcome::Prefilling => {
                    active.kv.apply_prefill_chunk(pool)?;
                    active.req.token_tx.is_closed()
                }
                Glm52StepOutcome::Commit {
                    committed,
                    emit,
                    finish,
                    ..
                } => {
                    active.kv.apply_prefill(committed[0], pool)?;
                    let mut freed = false;
                    for &token in &committed[..emit] {
                        if active
                            .req
                            .token_tx
                            .send(TokenEvent::Token {
                                id: token,
                                logprob: None,
                            })
                            .is_err()
                        {
                            freed = true;
                            break;
                        }
                    }
                    if !freed && let Some(drafts) = drafts {
                        let committed_len = active.kv.kv_position();
                        let tail_len = offload::native_pd_tail_len(committed_len);
                        let tail_key = if tail_len > 0 {
                            let key = native_mtp_tail_key(
                                &active.req.prompt_tokens[..committed_len],
                                committed[0],
                            );
                            if let Some(offload) = offload {
                                if let Err(err) = offload.save_native_tail(&active.kv, key) {
                                    let message =
                                        format!("GLM5.2 native P/D tail save failed: {err:#}");
                                    log::warn!("{message}");
                                    let _ = active.req.token_tx.send(TokenEvent::Error {
                                        message,
                                        prompt_tokens,
                                        completion_tokens: active.state.completion_tokens(),
                                    });
                                    freed = true;
                                }
                            }
                            Some(hex::encode(key))
                        } else {
                            None
                        };
                        if !freed {
                            let _ = active.req.token_tx.send(TokenEvent::KvTransfer {
                                params: serde_json::json!({
                                    "openinfer_pd": {
                                        "version": 2,
                                        "native_mtp": {
                                            "draft_tokens": drafts,
                                            "committed_len": committed_len,
                                            "arena_count": 101,
                                            "tail_len": tail_len,
                                            "tail_key": tail_key,
                                            "anchor_token_id": committed[0],
                                            "anchor_emitted": emit == 1
                                        }
                                    }
                                }),
                            });
                        }
                    }
                    if let Some(finish_reason) = finish
                        && !freed
                    {
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        freed = true;
                    }
                    freed
                }
            };
            if freed {
                if let Some(offload) = offload {
                    offload.save_sealed_on_release(&active.kv);
                }
                if let Err(err) = active.kv.release() {
                    log::warn!(
                        "GLM5.2 rank {} prefill slot {slot_id} KV release failed: {err:#}",
                        self.rank
                    );
                }
                *slot = None;
            }
        }
        Ok(())
    }

    /// A failed step leaves the ranks permanently out of lockstep: whichever
    /// collective the survivors are spinning in would pair with the NEXT
    /// step's first dispatch and every layer after it would run against the
    /// wrong expert bank — byte-deterministic garbage, no crash. The fleet
    /// cannot be re-synced; fail this rank's requests and exit the process.
    /// The peers fail-stop on their own collective errors/timeouts, and the
    /// router pulls the traffic (`docs/models/glm52/free-running-dp.md` §6).
    fn fatal(&mut self, err: &anyhow::Error) -> ! {
        log::error!(
            "GLM5.2 rank {} fatal; the engine process exits \
             (the EP collective group cannot recover): {err:#}",
            self.rank
        );
        for slot in &mut self.slots {
            let Some(active) = slot.take() else {
                continue;
            };
            let _ = active.req.token_tx.send(TokenEvent::Error {
                message: format!("{err:#}"),
                prompt_tokens: active.client_prompt_tokens,
                completion_tokens: active.state.completion_tokens(),
            });
        }
        for req in self.pending.drain(..) {
            let _ = req.token_tx.send(TokenEvent::Error {
                message: format!("{err:#}"),
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
        std::process::exit(1);
    }

    /// Graceful teardown (the submit channel closed and every request
    /// drained): fail whatever never got a slot, flush and drop the offload
    /// engines BEFORE the workers drop the models, then shut the workers
    /// down. Every engine reaches here together (its channel closed with
    /// the others), so the collective DeepEP destroy barrier pairs across
    /// the fleet.
    fn teardown(mut self) {
        for req in self.pending.drain(..) {
            let _ = req.token_tx.send(TokenEvent::Error {
                message: "GLM5.2 engine shut down before the request was scheduled".to_owned(),
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
        // Drain in-flight release saves and drop the offload engines BEFORE
        // the workers drop the models: the registered arenas' device memory
        // must outlive every D2H copy (the `with_arenas_on` contract), and
        // pegaflow's save worker cannot cancel a copy already handed to it.
        // `flush_saves` is deadline-bounded, so a stuck host tier cannot hang
        // teardown. Admission loads first: an abandoned restore's H2D can
        // still be writing arena memory (both barriers are deadline-bounded).
        if let Some(state) = self.host_restore.as_mut() {
            state.drain_loads();
        }
        if let Some(state) = self.native_pd.as_mut() {
            state.drain_loads();
        }
        if let Some(offload) = self.offload.take() {
            for rank in &offload {
                rank.engine.flush_saves();
            }
            drop(offload);
        }
        self.shutdown_workers();
    }

    /// The DeepEP context drop is collective: broadcast Shutdown to this
    /// rank's workers BEFORE their Drop joins them — a sequential
    /// shutdown-then-join would leave a worker spinning in the destroy
    /// barrier for ranks that never got the command (until the ~100 s
    /// device timeout).
    fn shutdown_workers(self) {
        for worker in &self.workers {
            let _ = worker.request_shutdown();
        }
        drop(self.workers);
    }
}

fn take_boundary_drafts<'a>(
    boundary: bool,
    drafts: &mut std::slice::Iter<'a, [u32; crate::mtp::GLM52_MTP_DRAFTS]>,
) -> Option<&'a [u32; crate::mtp::GLM52_MTP_DRAFTS]> {
    if boundary { drafts.next() } else { None }
}

/// Scope every lineage-hashed page in one native-MTP P/D request by its full
/// committed prompt. Layer 78 consumes shifted tokens, so the last row of a
/// page depends on the first token of the following page; token-only per-page
/// hashes would otherwise alias MTP bytes across diverging continuations.
fn native_mtp_cache_salt(committed_prompt: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"openinfer-glm52-native-mtp-pages-v1");
    hasher.update((committed_prompt.len() as u64).to_le_bytes());
    for token in committed_prompt {
        hasher.update(token.to_le_bytes());
    }
    let digest = hasher.finalize();
    hex::encode(&digest[..16])
}

fn native_mtp_tail_key(committed_prompt: &[u32], anchor_token: u32) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"openinfer-glm52-native-mtp-tail-v1");
    hasher.update((committed_prompt.len() as u64).to_le_bytes());
    for token in committed_prompt {
        hasher.update(token.to_le_bytes());
    }
    // The final MTP context row is shifted by P's sampled anchor, so the
    // stored tail is not identified by the prompt alone.
    hasher.update(anchor_token.to_le_bytes());
    let digest = hasher.finalize();
    let mut key = [0_u8; 16];
    key.copy_from_slice(&digest[..16]);
    key
}

#[cfg(test)]
mod tp_prefill_output_tests {
    use super::PAGE;
    use super::native_mtp_cache_salt;
    use super::native_mtp_tail_key;
    use super::take_boundary_drafts;

    #[test]
    fn disconnected_boundary_does_not_shift_the_next_requests_drafts() {
        let proposals = [
            [11; crate::mtp::GLM52_MTP_DRAFTS],
            [22; crate::mtp::GLM52_MTP_DRAFTS],
        ];
        let mut drafts = proposals.iter();

        let _discarded_after_disconnect = take_boundary_drafts(true, &mut drafts);
        assert_eq!(take_boundary_drafts(false, &mut drafts), None);
        assert_eq!(take_boundary_drafts(true, &mut drafts), Some(&proposals[1]));
    }

    #[test]
    fn sampled_anchor_is_part_of_the_native_mtp_tail_identity() {
        let prompt = [1, 2, 3];
        assert_ne!(
            native_mtp_tail_key(&prompt, 10),
            native_mtp_tail_key(&prompt, 11)
        );
        assert_eq!(
            native_mtp_tail_key(&prompt, 10),
            native_mtp_tail_key(&prompt, 10)
        );
    }

    #[test]
    fn token_after_a_full_page_is_part_of_the_native_mtp_page_identity() {
        let mut first = vec![7; PAGE + 1];
        let mut second = first.clone();
        first[PAGE] = 10;
        second[PAGE] = 11;

        assert_ne!(
            native_mtp_cache_salt(&first),
            native_mtp_cache_salt(&second),
            "the last MTP row in page 0 consumes token PAGE through shifted input"
        );
        assert_eq!(
            native_mtp_cache_salt(&first),
            native_mtp_cache_salt(&first),
            "P and D must derive the same cache scope from the committed prompt"
        );
    }
}
