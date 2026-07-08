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
//! boundaries (continuous batching) — admission is least-loaded rank first,
//! so decode-only fleets leave the 1-row bucket only past `GLM52_EP_RANKS`
//! concurrent requests.
//!
//! The per-request decisions (what to feed next, what a step's output means)
//! live in [`Glm52SlotState`] as pure data transitions, and the
//! admission/step-shape decisions in [`admission_target`] /
//! [`plan_step_shapes`] as pure functions over the occupancy and feed wants;
//! the coordinator is a thin shell that moves tokens between channels and the
//! rank workers.

mod admission;
#[cfg(test)]
mod contract_tests;
mod offload;
mod plan;
mod slot;
#[cfg(test)]
mod testkit;

use anyhow::Context as _;

use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};
use openinfer_kv_cache::{BlockPool, RequestKv};
use openinfer_kv_offload::OffloadEngine;
use openinfer_sample::mix_seed;
use tokio::sync::mpsc;

use crate::model::{
    GLM52_DECODE_BUCKETS, GLM52_MAX_BATCH_PER_RANK, GLM52_MLA_TOPK_SHORT, GLM52_MODEL_LEN_ALIGN,
    Glm52StepKv, Glm52StepShape, glm52_pool_blocks, glm52_table_width,
};
use crate::runner::{Glm52RankWorker, Glm52StepFlags};

use admission::{admission_target, intake, lifetime_blocks};
use plan::{collect_sampling_rows, feed_wants, launch_ahead_flags, occupancy, plan_step_shapes};
use slot::{GLM52_PADDING_STEP, Glm52SlotState, Glm52StepOutcome};

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

/// The all-padding-rows step KV: every page-table entry is the pool's
/// padding page, every write slot points into it (positions are `< PAGE` by
/// construction for padding rows — pre-capture probes position
/// [`GLM52_MLA_TOPK_SHORT`], a PAGE multiple, and serving pads sit at 0).
fn padding_step_kv(
    bucket: usize,
    table_width: usize,
    padding_page: i32,
    inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
) -> Glm52StepKv {
    let pages = vec![padding_page; bucket * table_width].into_boxed_slice();
    let mut slot_mapping = [0i64; GLM52_MAX_BATCH_PER_RANK];
    for (row, slot) in slot_mapping.iter_mut().enumerate().take(bucket) {
        *slot = padding_page as i64 * PAGE as i64 + (inputs[row].1 % PAGE) as i64;
    }
    Glm52StepKv {
        pages,
        slot_mapping,
    }
}

/// DP8 coordinator: admits up to `GLM52_MAX_BATCH_PER_RANK` requests per rank
/// (least-loaded rank with pool budget first) and drives all ranks in
/// lock-step. Consumes the workers — and the offload engines: they hold the
/// shared pegaflow host, which must outlive every in-flight save and dies
/// with the coordinator. Returns when the submit channel closes or a step
/// fails (the EP8 collective group cannot recover from a failed step — see
/// the teardown comment below).
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run_dp8_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52RankWorker>,
    eos_token_ids: &[u32],
    dspark_enabled: bool,
    max_model_len: usize,
    no_prefix_cache: bool,
    offload: Option<Vec<OffloadEngine>>,
    moe_topo: crate::Glm52MoeTopo,
) {
    // TP8 replicated topology: ONE logical rank drives 8 mirrored executors.
    // Every worker receives the identical step (inputs, shape, KV, seed) and
    // must return bit-identical outputs — the scheduler admits, plans, and
    // applies on the single logical rank; the fan-out lives in the submit
    // and draft joins.
    let mirrored = moe_topo == crate::Glm52MoeTopo::Tp8;
    let logical_ranks = if mirrored { 1 } else { workers.len() };
    // Verify-span draft budget: EP8 feeds 3 (the measured bucket-4 optimum);
    // the tp8 full-bucket shape always computes 8 rows, so it feeds the
    // drafter's full proposal.
    let span_drafts = if mirrored {
        crate::dspark::GLM52_DSPARK_DRAFTS
    } else {
        slot::GLM52_DSPARK_EP8_SPAN_DRAFTS
    };
    let offload: Option<Vec<offload::RankOffload>> =
        offload.map(|engines| engines.into_iter().map(offload::RankOffload::new).collect());
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
    let mut pending = std::collections::VecDeque::<GenerateRequest>::new();
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
    if let Err(err) = precapture_step_graphs(&workers, &pools, table_width, mirrored) {
        log::error!("GLM5.2 graph pre-capture failed: {err:#}");
        for worker in &workers {
            let _ = worker.request_shutdown();
        }
        return;
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
        if channel_open && all_idle(&slots) && pending.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => intake(req, &mut pending, max_model_len),
                None => channel_open = false,
            }
        }
        while channel_open {
            match submit_rx.try_recv() {
                Ok(req) => intake(req, &mut pending, max_model_len),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => channel_open = false,
            }
        }
        if !channel_open && all_idle(&slots) && pending.is_empty() {
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
            prefix_cache_enabled,
            dspark_enabled,
            &mut pending_resets,
            &mut slots_changed,
        ) {
            fail_step(&mut slots, &err);
            break 'serve;
        }
        if all_idle(&slots) {
            continue;
        }

        // One lock-step step: every rank forwards the SAME bucket — each
        // active slot's span of consecutive next tokens, padding rows on the
        // free slots — and all responses are joined before any output is
        // interpreted.
        let shapes = plan_step_shapes(&feed_wants(&slots), mirrored);
        let flags = launch_ahead_flags(
            &shapes,
            leased_shapes.as_deref(),
            slots_changed,
            pending.is_empty(),
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
    for req in pending {
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

/// Pre-capture every whole-step graph (bucket × attention tier) while the
/// ranks are idle and trivially in lock-step. Launch-ahead speculation
/// requires captured-ness to be UNIFORM across ranks — a lazily capturing
/// rank would skip the speculative replay the others enqueued and desync the
/// collectives — and pre-capturing also removes the old mid-serving capture
/// stall. Row 0 at position GLM52_MLA_TOPK_SHORT lifts the step into the
/// full tier; every row is a padding write into the pool's padding page.
fn precapture_step_graphs(
    workers: &[Glm52RankWorker],
    pools: &[BlockPool],
    table_width: usize,
    mirrored: bool,
) -> anyhow::Result<()> {
    // tp8 serves exactly one shape (the full bucket, every worker mirrored);
    // EP8 captures every bucket.
    let capture_bucket = |bucket: usize| !mirrored || bucket == GLM52_MAX_BATCH_PER_RANK;
    for &bucket in GLM52_DECODE_BUCKETS
        .iter()
        .filter(|&&bucket| capture_bucket(bucket))
    {
        for full_tier in [false, true] {
            let mut shape = Glm52StepShape {
                bucket,
                slots: [0; GLM52_MAX_BATCH_PER_RANK],
                active_rows: 0,
            };
            for (slot, dst) in shape.slots.iter_mut().enumerate().take(bucket) {
                *dst = slot as u8;
            }
            let mut inputs =
                [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position); GLM52_MAX_BATCH_PER_RANK];
            if full_tier {
                inputs[0] = (GLM52_PADDING_STEP.token, GLM52_MLA_TOPK_SHORT);
            }
            let flags = Glm52StepFlags::plain();
            let responses = workers
                .iter()
                .enumerate()
                .map(|(rank, worker)| {
                    let pool = &pools[if mirrored { 0 } else { rank }];
                    let kv = padding_step_kv(bucket, table_width, pool.padding_block_id(), &inputs);
                    worker.step_async(inputs, shape, kv, flags, Vec::new(), 0)
                })
                .collect::<anyhow::Result<Vec<_>>>()
                .context("GLM5.2 graph pre-capture submit")?;
            for (rank, resp) in responses.into_iter().enumerate() {
                resp.recv()
                    .map_err(|_| anyhow::anyhow!("rank dropped its pre-capture response"))
                    .and_then(|r| r)
                    .with_context(|| {
                        format!(
                            "GLM5.2 graph pre-capture (bucket {bucket}, full_tier \
                             {full_tier}) on rank {rank}"
                        )
                    })?;
            }
        }
    }
    log::info!(
        "GLM5.2 whole-step graphs pre-captured: {} buckets x 2 tiers",
        GLM52_DECODE_BUCKETS
            .iter()
            .filter(|&&bucket| capture_bucket(bucket))
            .count()
    );
    Ok(())
}

/// Admission: fill free slots from the queue, least-loaded rank (with pool
/// budget for the request's full lifetime) first. New requests join the
/// lock-step at the next step boundary (their prefill rides decode alongside
/// everyone else's rows). An `Err` is a kvbm invariant break — the caller
/// fails the step (the affected request was already answered here).
#[allow(clippy::too_many_arguments)]
fn admit_from_queue(
    pending: &mut std::collections::VecDeque<GenerateRequest>,
    slots: &mut [RankSlots],
    pools: &[BlockPool],
    usable_blocks: &[usize],
    offload: Option<&[offload::RankOffload]>,
    prefix_cache_enabled: bool,
    dspark_enabled: bool,
    pending_resets: &mut [Vec<usize>],
    slots_changed: &mut bool,
) -> anyhow::Result<()> {
    while let Some(front) = pending.front() {
        let need_blocks = lifetime_blocks(front.prompt_tokens.len(), front.max_tokens);
        let committed: Vec<usize> = slots
            .iter()
            .map(|rank_slots| {
                rank_slots
                    .iter()
                    .flatten()
                    .map(|active| {
                        lifetime_blocks(active.req.prompt_tokens.len(), active.req.max_tokens)
                    })
                    .sum()
            })
            .collect();
        // Pages pinned by in-flight release saves are physically
        // unallocatable until their D2H lands — hide them from the
        // full-lifetime budget so admission defers instead of promising
        // pages a later `schedule_prefill` cannot get (which would
        // fail_step the whole engine).
        let usable: Vec<usize> = match offload {
            Some(offload) => usable_blocks
                .iter()
                .zip(offload)
                .map(|(&usable, rank)| usable.saturating_sub(rank.pinned_blocks()))
                .collect(),
            None => usable_blocks.to_vec(),
        };
        let Some((rank, slot)) =
            admission_target(&occupancy(slots), &committed, &usable, need_blocks)
        else {
            break;
        };
        let req = pending.pop_front().expect("checked non-empty");
        // The client left while the request sat in the queue — admitting
        // it would burn a slot (and whole global steps) on a dead sink.
        if req.token_tx.is_closed() {
            continue;
        }
        let mut kv = pools[rank].new_request(req.prompt_tokens.clone(), req.max_tokens, None);
        // Host-tier restore first, so the single GPU prefix match below sees
        // the union of HBM-resident and freshly-restored blocks. The probe
        // stays alive across the match: it holds the committed blocks, and
        // dropping it earlier would open an eviction window between commit
        // and re-match.
        let _restored_hold = offload.filter(|_| prefix_cache_enabled).map(|offload| {
            offload::restore_host_prefix(&offload[rank].engine, &pools[rank], &req.prompt_tokens)
        });
        let cached_tokens = if prefix_cache_enabled {
            match kv.match_and_add_prefix(&pools[rank]) {
                Ok(cached) => cached,
                Err(err) => {
                    // A fresh request failing to match is a kvbm state
                    // invariant break — crash early, don't serve on a
                    // pool whose bookkeeping already lied once. This
                    // request is already out of `pending` and never
                    // reaches a slot, so fail it explicitly (fail_step
                    // and the shutdown sweep can't see it).
                    let err = err.context("GLM5.2 prefix match at admission");
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: format!("{err:#}"),
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                    return Err(err);
                }
            }
        } else {
            0
        };
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens: req.prompt_tokens.len(),
            cached_tokens,
        });
        let state = Glm52SlotState::new(
            req.prompt_tokens.clone(),
            req.max_tokens,
            req.params.ignore_eos,
            cached_tokens,
        );
        if dspark_enabled {
            pending_resets[rank].push(slot);
        }
        slots[rank][slot] = Some(ActiveRequest { req, state, kv });
        *slots_changed = true;
    }
    Ok(())
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
    workers: &[Glm52RankWorker],
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
        let executors: &[Glm52RankWorker] = if mirrored {
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
    workers: &[Glm52RankWorker],
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
        let executors: &[Glm52RankWorker] = if mirrored {
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
