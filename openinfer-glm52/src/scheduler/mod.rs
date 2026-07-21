//! Lock-step continuous-batching scheduler. EP8 runs DP8: up to
//! `GLM52_MAX_BATCH_PER_RANK` requests per rank, each owning one slot of the
//! rank's decode batch. The TP8 replicated topology collapses to ONE logical
//! rank driving 8 mirrored executors: admission, planning, and output
//! application all happen on the single logical rank, every worker receives
//! the identical step, and the joins assert bit-identical results (the
//! replicated-activations contract). KV pages come from a per-logical-rank [`BlockPool`]
//! (64-token pages, content-hashed blocks): admission reserves a request's
//! full-lifetime page count up front (honor-or-reject — a request that can
//! never fit is rejected, one that can't fit *now* stays queued), so decode
//! can never run out of pages mid-request, and released requests' sealed
//! blocks stay matchable as the prefix cache.
//!
//! Every global step ALL ranks run the full-model forward simultaneously with
//! the SAME batch bucket — ranks feed each active slot's *span* of next
//! tokens (mid-prefill slots batch up to a bucket of consecutive prompt
//! positions through one step; decode slots feed one row), idle slots feed a
//! padding row whose output is discarded. This satisfies the DeepEP contract
//! that every rank enters every MoE layer's dispatch/combine collective with
//! the agreed global row count. The bucket is the smallest member of
//! `GLM52_DECODE_BUCKETS` covering the hungriest rank's row demand, so the
//! fleet pays for prefill only while someone is prefilling and returns to the
//! cheap 1-row bucket for pure decode. Requests join and leave slots at step
//! boundaries (continuous batching). The vLLM frontend assigns HTTP requests
//! least-load-first to rank-owned queues, so decode-only fleets leave the
//! 1-row bucket only past `GLM52_EP_RANKS` concurrent requests.
//!
//! The per-request decisions (what to feed next, what a step's output means)
//! live in [`Glm52SlotState`] as pure data transitions, and the
//! admission/step-shape decisions in [`intake`] /
//! [`plan_step_shapes`] as pure functions over the occupancy and feed wants;
//! the coordinator is a thin shell that moves tokens between channels and the
//! rank workers.

mod admission;
#[cfg(test)]
mod contract_tests;
mod graph;
mod load;
mod offload;
mod plan;
mod slot;
#[cfg(test)]
mod testkit;

use std::collections::VecDeque;

use admission::admit_from_queue;
use admission::intake;
use graph::GraphDumpRequest;
use graph::dump_rank0_decode_graph;
use graph::precapture_step_graphs;
use load::pending_is_empty;
use load::publish_load;
use load::running_counts;
pub(crate) use offload::REMOTE_FETCH_DEADLINE;
use offload::VllmPdState;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::LoadSnapshot;
use openinfer_core::engine::TokenEvent;
use openinfer_kv_cache::BlockPool;
use openinfer_kv_cache::RequestKv;
use openinfer_kv_offload::OffloadEngine;
use openinfer_sample::mix_seed;
use plan::collect_sampling_rows;
use plan::feed_wants;
use plan::launch_ahead_flags;
use plan::plan_step_shapes;
use slot::GLM52_PADDING_STEP;
use slot::Glm52SlotState;
use slot::Glm52StepOutcome;
use tokio::sync::mpsc;
use tokio::sync::watch;

use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::GLM52_MODEL_LEN_ALIGN;
use crate::model::Glm52StepKv;
use crate::model::Glm52StepShape;
use crate::model::glm52_pool_blocks;
use crate::model::glm52_table_width;
use crate::runner::Glm52StepFlags;
use crate::runner::Glm52Worker;

/// The KV page size (== the FlashMLA page / index-K block / model-len
/// alignment — one 64 everywhere).
const PAGE: usize = GLM52_MODEL_LEN_ALIGN;

/// Engine-level philox seed for unseeded non-greedy rows (the Kimi
/// convention: unseeded requests need no replay guarantee, so a fixed engine
/// seed suffices; per-request `seed` params replay through `mix_seed`).
const GLM52_SAMPLE_SEED: u64 = 42;

struct ActiveRequest {
    req: GenerateRequest,
    state: Glm52SlotState,
    /// The request's page assignments in the rank's pool. Block RAII: blocks
    /// return to the pool (registered ones as matchable prefix-cache entries)
    /// when this drops or `release()`s.
    kv: RequestKv,
}

/// Per-rank slot occupancy: `slots[rank][slot]`.
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

/// DP8 coordinator: admits up to `GLM52_MAX_BATCH_PER_RANK` requests per rank
/// (least-loaded rank with pool budget first) and drives all ranks in
/// lock-step. Consumes the workers — and the offload engines: they hold the
/// shared pegaflow host, which must outlive every in-flight save and dies
/// with the coordinator. Returns when the submit channel closes or a step
/// fails (the EP8 collective group cannot recover from a failed step — see
/// the teardown comment below).
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dp8_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52Worker>,
    eos_token_ids: &[u32],
    dspark_enabled: bool,
    max_model_len: usize,
    no_prefix_cache: bool,
    offload: Option<Vec<OffloadEngine>>,
    vllm_compat: Option<crate::Glm52VllmCompatOptions>,
    moe_topo: crate::Glm52MoeTopo,
    load_txs: Vec<watch::Sender<LoadSnapshot>>,
    graph_dump_request: Option<GraphDumpRequest>,
) {
    // Tensor-replicated topology: ONE logical rank drives mirrored executors.
    // Every worker receives the identical step (inputs, shape, KV, seed) and
    // must return bit-identical outputs — the scheduler admits, plans, and
    // applies on the single logical rank; the fan-out lives in the submit
    // and draft joins.
    let mirrored = moe_topo.uses_tensor_replicated_moe();
    // TP8's phase kernels retain their original single bucket-8 contract.
    // TP4 pads only the MoE phase chain internally, so attention, projections,
    // norms, and sampling can use the smallest regular decode bucket.
    let full_bucket = matches!(moe_topo, crate::Glm52MoeTopo::Tp8);
    let logical_ranks = moe_topo.logical_rank_count();
    debug_assert_eq!(logical_ranks, if mirrored { 1 } else { workers.len() });
    assert_eq!(
        load_txs.len(),
        logical_ranks,
        "one GLM5.2 load feed is required per logical rank"
    );
    // Verify-span draft budget: EP8 feeds 3 (the measured bucket-4 optimum);
    // the tp8 full-bucket shape always computes 8 rows, so it feeds the
    // drafter's full proposal.
    let span_drafts = if mirrored {
        crate::dspark::GLM52_DSPARK_DRAFTS
    } else {
        slot::GLM52_DSPARK_EP8_SPAN_DRAFTS
    };
    // vLLM-compat P/D disables self-saves: the content domain carries the
    // peer's key scheme, and the peer re-registers the full history each turn.
    let save_enabled = vllm_compat.is_none();
    let offload: Option<Vec<offload::RankOffload>> = offload.map(|engines| {
        engines
            .into_iter()
            .map(|engine| offload::RankOffload::new(engine, save_enabled))
            .collect()
    });
    let mut vllm_pd =
        vllm_compat.map(|opts| VllmPdState::new(&opts, moe_topo.logical_rank_count()));
    // One KV page pool per LOGICAL rank: pool block ids index the rank's
    // per-layer MLA and index-K arenas directly (the arenas were built for
    // `glm52_pool_blocks` blocks). Block 0-equivalent is the reserved
    // padding page. Under tp8 the single pool drives every executor — the
    // mirrored steps write the identical block ids on all 8 arenas.
    let pools: Vec<BlockPool> = match (0..logical_ranks)
        .map(|_| BlockPool::new(PAGE, glm52_pool_blocks(max_model_len)))
        .collect::<anyhow::Result<Vec<_>>>()
    {
        Ok(pools) => pools,
        Err(err) => {
            log::error!("GLM5.2 KV pool construction failed: {err:#}");
            for worker in &workers {
                let _ = worker.request_shutdown();
            }
            return;
        }
    };
    let table_width = glm52_table_width(max_model_len);
    // Pool pages available to requests per rank (total minus the padding
    // page) — constant for the engine's lifetime.
    let usable_blocks: Vec<usize> = pools.iter().map(|pool| pool.total_blocks() - 1).collect();
    // The DSpark draft lane asserts every anchor position equals its
    // committed + pending context rows — a skipped (cache-hit) prefix never
    // produces the aux-hidden captures the draft consumes, so prefix
    // matching is off while the drafter is on. Speculative decoding and
    // prefix caching are mutually exclusive for now (the qwen3 offload path
    // draws the same line). `--no-prefix-cache` is the explicit kill switch.
    let prefix_cache_enabled = !dspark_enabled && !no_prefix_cache;
    if dspark_enabled && !no_prefix_cache {
        log::info!("GLM5.2 prefix cache disabled: the DSpark drafter is on");
    }
    let mut slots: Vec<RankSlots> = (0..logical_ranks)
        .map(|_| std::array::from_fn(|_| None))
        .collect();
    // Slot draft states to clear on the next draft round (request left the
    // slot, or a new one was admitted into it). Flushed with each step's
    // Draft commands; the handler is idempotent, so duplicates are harmless.
    let mut pending_resets: Vec<Vec<usize>> = (0..logical_ranks).map(|_| Vec::new()).collect();
    let mut pending: Vec<VecDeque<GenerateRequest>> =
        (0..logical_ranks).map(|_| VecDeque::new()).collect();
    let mut channel_open = true;
    let all_idle = |slots: &[RankSlots]| {
        slots
            .iter()
            .all(|rank_slots| rank_slots.iter().all(Option::is_none))
    };

    // Pre-capture before serving (see [`precapture_step_graphs`]). The DeepEP
    // contexts already exist: on failure broadcast Shutdown before the
    // workers' sequential Drop joins them (the same collective-teardown
    // contract as the exit path).
    if let Err(err) = precapture_step_graphs(&workers, &pools, table_width, mirrored, full_bucket) {
        if let Some((_, response)) = graph_dump_request {
            let _ = response.send(Err(anyhow::anyhow!("{err:#}")));
        }
        log::error!("GLM5.2 graph pre-capture failed: {err:#}");
        for worker in &workers {
            let _ = worker.request_shutdown();
        }
        return;
    }
    if let Some((png_path, response)) = graph_dump_request {
        match dump_rank0_decode_graph(&workers, moe_topo, full_bucket, png_path) {
            Ok(summary) => {
                let _ = response.send(Ok(summary));
            }
            Err(err) => {
                log::error!("GLM5.2 CUDA Graph export failed: {err:#}");
                let _ = response.send(Err(err));
                for worker in &workers {
                    let _ = worker.request_shutdown();
                }
                return;
            }
        }
    }

    // The step shapes that carried the last launch-ahead lease, and whether
    // any slot changed hands since it was granted — together they decide
    // whether this step consumes the speculation (must be all-ranks-or-none,
    // so the decision lives here, on global data, not on the ranks).
    let mut leased_shapes: Option<Vec<Glm52StepShape>> = None;
    let mut slots_changed = false;
    // Global step counter driving the non-greedy rows' philox seeds: a fresh
    // well-mixed seed per (step, rank), so no two ranks — whose row indices
    // collide — ever share a philox stream.
    let mut sample_step: u64 = 0;

    'serve: loop {
        // Intake: block when fully idle, otherwise drain what's queued.
        if channel_open && all_idle(&slots) && pending_is_empty(&pending) {
            publish_load(&load_txs, &pools, &slots, &pending);
            match submit_rx.blocking_recv() {
                Some(req) => intake(req, &mut pending, &running_counts(&slots), max_model_len),
                None => channel_open = false,
            }
        }
        while channel_open {
            match submit_rx.try_recv() {
                Ok(req) => intake(req, &mut pending, &running_counts(&slots), max_model_len),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => channel_open = false,
            }
        }
        if !channel_open && all_idle(&slots) && pending_is_empty(&pending) {
            break;
        }

        // Admission (see [`admit_from_queue`]); an admission failure is a kvbm
        // state invariant break — crash early, don't serve on a pool whose
        // bookkeeping already lied once.
        if let Err(err) = admit_from_queue(
            &mut pending,
            &mut slots,
            &pools,
            &usable_blocks,
            offload.as_deref(),
            &mut vllm_pd,
            &workers,
            mirrored,
            prefix_cache_enabled,
            dspark_enabled,
            &mut pending_resets,
            &mut slots_changed,
        ) {
            fail_step(&mut slots, &err);
            break 'serve;
        }
        publish_load(&load_txs, &pools, &slots, &pending);
        if all_idle(&slots) {
            // A parked P/D front re-queries at admission; with no running
            // slots there is no step cadence to pace the retries — throttle
            // instead of spinning on the MetaServer.
            if vllm_pd.as_ref().is_some_and(VllmPdState::any_parked) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            continue;
        }

        // One lock-step step: every rank forwards the SAME bucket — each
        // active slot's span of consecutive next tokens, padding rows on the
        // free slots — and all responses are joined before any output is
        // interpreted.
        let shapes = plan_step_shapes(&feed_wants(&slots), full_bucket);
        let flags = launch_ahead_flags(
            &shapes,
            leased_shapes.as_deref(),
            slots_changed,
            pending_is_empty(&pending),
            dspark_enabled,
            offload.is_some(),
            &slots,
            max_model_len,
        );
        leased_shapes = flags.lease.then(|| shapes.clone());
        slots_changed = false;
        sample_step += 1;
        // One lock-step step (see [`submit_and_join_step`]).
        let (outputs, span_kinds) = match submit_and_join_step(
            &workers,
            &pools,
            &mut slots,
            &shapes,
            flags,
            table_width,
            sample_step,
        ) {
            Ok(step) => step,
            Err(err) => {
                fail_step(&mut slots, &err);
                break 'serve;
            }
        };

        let (rank_appends, mut rank_proposals) = match apply_step_outputs(
            &mut slots,
            outputs,
            &shapes,
            &span_kinds,
            &pools,
            offload.as_deref(),
            eos_token_ids,
            dspark_enabled,
            &mut pending_resets,
            &mut slots_changed,
        ) {
            Ok(walked) => walked,
            Err(err) => {
                fail_step(&mut slots, &err);
                break 'serve;
            }
        };
        // TP8 speculative policy: draft only when the fleet is solo — a
        // concurrent fleet's bucket rows go to liveness first, and feeding
        // partial verify spans there is unmeasured territory (a follow-up
        // lever: the full bucket COULD verify several requests' spans at
        // once). Suppress the proposals (appends and resets still flow, so
        // the drafter's shadow KV stays fresh and proposals resume the
        // round after the fleet drains back to solo). Drafts already
        // installed on the solo slot are deliberately left to drain: the one
        // transition step verifies a prefix of them (spare rows split
        // round-robin with the newcomer's prefill, every slot keeps its
        // liveness row) and `advance_span` discards the rest — clearing them
        // here would throw away paid-for speculation to speed one step of
        // one prefill.
        if mirrored
            && slots
                .iter()
                .flat_map(|rank_slots| rank_slots.iter().flatten())
                .count()
                != 1
        {
            for proposals in &mut rank_proposals {
                proposals.clear();
            }
        }

        if dspark_enabled
            && let Err(err) = run_draft_round(
                &workers,
                &mut slots,
                &shapes,
                &mut pending_resets,
                rank_appends,
                rank_proposals,
                span_drafts,
            )
        {
            fail_step(&mut slots, &err);
            break 'serve;
        }
    }

    // Also fail whatever never got a slot.
    for req in pending.into_iter().flatten() {
        let _ = req.token_tx.send(TokenEvent::Error {
            message: "GLM5.2 engine shut down before the request was scheduled".to_owned(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
    }

    // Drain in-flight release saves and drop the offload engines BEFORE the
    // workers drop the models: the registered arenas' device memory must
    // outlive every D2H copy (the `with_arenas` contract), and pegaflow's
    // save worker cannot cancel a copy already handed to it. `flush_saves`
    // is deadline-bounded, so a stuck host tier cannot hang teardown.
    if let Some(offload) = offload {
        for rank in &offload {
            rank.engine.flush_saves();
        }
        drop(offload);
    }
    // The DeepEP context drop is collective: broadcast Shutdown to every rank
    // BEFORE the workers' Drop joins them one by one — a sequential
    // shutdown-then-join would leave a rank spinning in the destroy barrier
    // for ranks that never got the command (until the ~100 s device timeout).
    for worker in &workers {
        let _ = worker.request_shutdown();
    }
    drop(workers);
}

/// One lock-step step: per-rank submit — schedule each active span's KV
/// (full-lifetime reservation makes every schedule succeed; a failure is an
/// accounting bug and fails the step), build the row inputs, page rows and
/// write slots, collect the step's sampling rows, and fire — then join ALL
/// ranks before failing: the rank the coordinator happens to recv first
/// often reports the ~100 s DeepEP device-timeout trap, not the root cause.
/// Returns every rank's outputs plus what the submit phase scheduled per
/// slot (`span_kinds[rank][slot]`), which the output walk pairs exactly.
#[allow(clippy::type_complexity)]
fn submit_and_join_step(
    workers: &[Glm52Worker],
    pools: &[BlockPool],
    slots: &mut [RankSlots],
    shapes: &[Glm52StepShape],
    flags: Glm52StepFlags,
    table_width: usize,
    sample_step: u64,
) -> anyhow::Result<(
    Vec<[u32; GLM52_MAX_BATCH_PER_RANK]>,
    Vec<[Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK]>,
)> {
    // Logical-to-executor mapping: 1:1 under EP8, or the single logical
    // rank's step mirrored onto every worker under the replicated tp8
    // topology (identical inputs/KV/seed, bit-identical outputs asserted at
    // the join).
    let mirrored = slots.len() == 1 && workers.len() > 1;
    let mut span_kinds: Vec<[Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK]> = slots
        .iter()
        .map(|_| [None; GLM52_MAX_BATCH_PER_RANK])
        .collect();
    let mut responses = Vec::with_capacity(workers.len());
    let mut submit_err: Option<anyhow::Error> = None;
    'submit: for (rank, (rank_slots, shape)) in slots.iter_mut().zip(shapes).enumerate() {
        let pool = &pools[rank];
        let padding_page = pool.padding_block_id();
        let sampling = collect_sampling_rows(shape, rank_slots);
        let seed = mix_seed(mix_seed(GLM52_SAMPLE_SEED, sample_step), rank as u64);
        let mut inputs =
            [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position); GLM52_MAX_BATCH_PER_RANK];
        // A consumed speculation replays with device-advanced inputs and
        // never reads the step KV — skip building the page rows (the
        // whole point of launch-ahead is keeping this host path off the
        // hot step boundary). KV *scheduling* still runs: kvbm's
        // bookkeeping must advance every step.
        let mut pages = if flags.consume {
            Vec::new()
        } else {
            vec![padding_page; shape.bucket * table_width]
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
            let Some(active) = rank_slots[slot_id].as_mut() else {
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
                submit_err = Some(anyhow::anyhow!(
                    "GLM5.2 rank {rank} slot {slot_id} span starts at position {} but the \
                     KV pool is at {}",
                    inputs[row].1,
                    active.kv.kv_position()
                ));
                break 'submit;
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
                submit_err = Some(anyhow::anyhow!(
                    "GLM5.2 rank {rank} slot {slot_id} violated its full-lifetime KV \
                     reservation ({kind:?}, span {span}): {err}"
                ));
                break 'submit;
            }
            span_kinds[rank][slot_id] = Some(kind);
            if !flags.consume {
                let row_pages = active.kv.step_page_indices(span);
                for r in row..end {
                    pages[r * table_width..r * table_width + row_pages.len()]
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
        let executors: &[Glm52Worker] = if mirrored {
            workers
        } else {
            std::slice::from_ref(&workers[rank])
        };
        for worker in executors {
            match worker.step_async(inputs, *shape, kv.clone(), flags, sampling.clone(), seed) {
                Ok(rx) => responses.push(rx),
                Err(err) => {
                    submit_err = Some(err);
                    break 'submit;
                }
            }
        }
    }
    if let Some(err) = submit_err {
        return Err(err);
    }
    // Join ALL ranks before failing: the rank the coordinator happens to
    // recv first often reports the ~100 s DeepEP device-timeout trap, not
    // the root cause — the rank that actually failed answers later. Log
    // every error so the root cause has a landing spot, then tear down on
    // the first one in rank order.
    let mut outputs = Vec::with_capacity(responses.len());
    let mut step_err: Option<anyhow::Error> = None;
    for (rank, resp) in responses.into_iter().enumerate() {
        let result = resp
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its step response"));
        match result {
            Ok(Ok(step_tokens)) => outputs.push(step_tokens),
            Ok(Err(err)) | Err(err) => {
                let err = err.context(format!("GLM5.2 rank {rank} step"));
                log::error!("GLM5.2 rank {rank} step failed: {err:#}");
                step_err.get_or_insert(err);
                outputs.push([0; GLM52_MAX_BATCH_PER_RANK]);
            }
        }
    }
    if let Some(err) = step_err {
        return Err(err);
    }
    if mirrored {
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
    Ok((outputs, span_kinds))
}

/// Fold every rank's span of outputs into its slot state, commit the span's
/// KV bookkeeping under the exact kind the submit phase scheduled (a
/// mispairing is a coordinator bug and fails the step), emit tokens and
/// finish/disconnect releases, and collect the draft lane's context appends
/// and next-round proposals.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn apply_step_outputs(
    slots: &mut [RankSlots],
    outputs: Vec<[u32; GLM52_MAX_BATCH_PER_RANK]>,
    shapes: &[Glm52StepShape],
    span_kinds: &[[Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK]],
    pools: &[BlockPool],
    offload: Option<&[offload::RankOffload]>,
    eos_token_ids: &[u32],
    dspark_enabled: bool,
    pending_resets: &mut [Vec<usize>],
    slots_changed: &mut bool,
) -> anyhow::Result<(Vec<Vec<(usize, usize)>>, Vec<Vec<(usize, u32, usize)>>)> {
    let mut rank_appends: Vec<Vec<(usize, usize)>> = slots.iter().map(|_| Vec::new()).collect();
    let mut rank_proposals: Vec<Vec<(usize, u32, usize)>> =
        slots.iter().map(|_| Vec::new()).collect();
    for (rank, ((rank_slots, rank_outputs), shape)) in
        slots.iter_mut().zip(outputs).zip(shapes).enumerate()
    {
        // Walk the shape's contiguous per-slot runs; each active slot
        // folds its whole span of row outputs in at once.
        let mut row = 0usize;
        while row < shape.bucket {
            let slot_id = shape.slots[row] as usize;
            let mut end = row + 1;
            while end < shape.bucket && shape.slots[end] as usize == slot_id {
                end += 1;
            }
            let span_rows = row..end;
            let span_outputs = &rank_outputs[span_rows.clone()];
            row = end;
            let slot = &mut rank_slots[slot_id];
            let Some(active) = slot.as_mut() else {
                continue;
            };
            let prompt_tokens = active.req.prompt_tokens.len();
            let outcome = active.state.advance_span(span_outputs, eos_token_ids);
            // Commit the span's KV bookkeeping under the exact kind the
            // submit phase scheduled — a mispairing is a coordinator bug
            // and fails the step.
            let pool = &pools[rank];
            let applied = match (&outcome, span_kinds[rank][slot_id]) {
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
                    "GLM5.2 rank {rank} slot {slot_id} outcome {outcome:?} does not pair \
                     with scheduled span kind {kind:?}"
                )),
            };
            if let Err(err) = applied {
                return Err(err.context(format!("GLM5.2 rank {rank} slot {slot_id} KV apply")));
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
                active.state.log_spec_stats(rank, slot_id);
                // Offload the freshly-sealed blocks BEFORE release: the
                // hashes and guards come off the still-assigned request
                // state, and the guards keep the pages pinned through the
                // async D2H copy.
                if let Some(offload) = offload {
                    offload[rank].save_sealed_on_release(&active.kv);
                }
                if let Err(err) = active.kv.release() {
                    // Blocks still return via assignment RAII when the
                    // slot drops — the explicit release only failed to
                    // run from a clean Idle state.
                    log::warn!(
                        "GLM5.2 rank {rank} slot {slot_id} KV release failed \
                         (blocks return via RAII): {err:#}"
                    );
                }
                if dspark_enabled {
                    pending_resets[rank].push(slot_id);
                }
                *slot = None;
                *slots_changed = true;
            } else if dspark_enabled {
                // Committed rows' captured hidden feeds the draft
                // context; then re-propose from the new anchor.
                rank_appends[rank].extend(span_rows.take(context_rows).map(|r| (r, slot_id)));
                if active.state.wants_drafts()
                    && let Some((anchor, anchor_pos)) = active.state.decode_anchor()
                {
                    rank_proposals[rank].push((slot_id, anchor, anchor_pos));
                }
            }
        }
    }
    Ok((rank_appends, rank_proposals))
}

/// Draft round (rank-local, no collectives): resets, context appends from
/// THIS step's capture buffer, and new proposals for the next verify span.
/// FIFO per-rank channels order it before the next step; the blocking join
/// keeps the round cadence (draft sits between verify steps, ~2 ms against a
/// 22-46 ms step).
fn run_draft_round(
    workers: &[Glm52Worker],
    slots: &mut [RankSlots],
    shapes: &[Glm52StepShape],
    pending_resets: &mut [Vec<usize>],
    rank_appends: Vec<Vec<(usize, usize)>>,
    rank_proposals: Vec<Vec<(usize, u32, usize)>>,
    span_drafts: usize,
) -> anyhow::Result<()> {
    // Same logical-to-executor mapping as the step submit: under the
    // mirrored tp8 topology every worker drafts from its own (identical)
    // capture buffer and must propose the identical spans.
    let mirrored = slots.len() == 1 && workers.len() > 1;
    let mut draft_joins = Vec::new();
    for (rank, (appends, proposals)) in rank_appends.into_iter().zip(rank_proposals).enumerate() {
        let resets = std::mem::take(&mut pending_resets[rank]);
        if resets.is_empty() && appends.is_empty() && proposals.is_empty() {
            continue;
        }
        let proposal_slots: Vec<usize> = proposals.iter().map(|&(slot, _, _)| slot).collect();
        let executors: &[Glm52Worker] = if mirrored {
            workers
        } else {
            std::slice::from_ref(&workers[rank])
        };
        let rxs = executors
            .iter()
            .map(|worker| {
                worker.draft_async(
                    shapes[rank].bucket,
                    resets.clone(),
                    appends.clone(),
                    proposals.clone(),
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        draft_joins.push((rank, proposal_slots, rxs));
    }
    for (rank, proposal_slots, rxs) in draft_joins {
        let mut all_spans = Vec::with_capacity(rxs.len());
        for (executor, rx) in rxs.into_iter().enumerate() {
            let result = rx
                .recv()
                .map_err(|_| {
                    anyhow::anyhow!("GLM5.2 executor {executor} dropped its draft response")
                })
                .and_then(|r| r);
            match result {
                Ok(spans) => all_spans.push(spans),
                // A draft failure is rank-local, but it means the drafter's
                // invariants broke — crash early rather than silently degrade
                // to plain decode.
                Err(err) => return Err(err.context(format!("GLM5.2 executor {executor} draft"))),
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
        if spans.len() != proposal_slots.len() {
            return Err(anyhow::anyhow!(
                "GLM5.2 rank {rank} draft returned {} spans for {} proposals",
                spans.len(),
                proposal_slots.len()
            ));
        }
        for (slot_id, span) in proposal_slots.into_iter().zip(spans) {
            if let Some(active) = slots[rank][slot_id].as_mut() {
                active.state.set_drafts(span.to_vec(), span_drafts);
            }
        }
    }
    Ok(())
}

/// A failed step leaves the ranks permanently out of lockstep: whichever
/// collective the survivors are spinning in would pair with the NEXT step's
/// first dispatch and every layer after it would run against the wrong
/// expert bank — byte-deterministic garbage, no crash. The group cannot be
/// re-synced; fail every active request and tear the engine down.
fn fail_step(slots: &mut [RankSlots], err: &anyhow::Error) {
    log::error!(
        "GLM5.2 step failed; shutting the engine down (the EP8 collective group cannot recover): {err:#}"
    );
    for slot in slots.iter_mut().flatten() {
        let Some(active) = slot.take() else {
            continue;
        };
        let _ = active.req.token_tx.send(TokenEvent::Error {
            message: format!("{err:#}"),
            prompt_tokens: active.req.prompt_tokens.len(),
            completion_tokens: active.state.completion_tokens(),
        });
    }
}
