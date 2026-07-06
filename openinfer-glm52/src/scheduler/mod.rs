//! DP8 lock-step continuous-batching scheduler: up to
//! `GLM52_MAX_BATCH_PER_RANK` requests per rank, each owning one slot of the
//! rank's decode batch. KV pages come from a per-rank [`BlockPool`]
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
mod plan;
mod slot;
#[cfg(test)]
mod testkit;

use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};
use openinfer_kv_cache::{BlockPool, RequestKv};
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
/// lock-step. Consumes the workers; returns when the submit channel closes
/// or a step fails (the EP8 collective group cannot recover from a failed
/// step — see the teardown comment below).
pub(crate) fn run_dp8_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52RankWorker>,
    eos_token_ids: &[u32],
    dspark_enabled: bool,
    max_model_len: usize,
    no_prefix_cache: bool,
) {
    // One KV page pool per rank: pool block ids index the rank's per-layer
    // MLA and index-K arenas directly (the arenas were built for
    // `glm52_pool_blocks` blocks). Block 0-equivalent is the reserved
    // padding page.
    let pools: Vec<BlockPool> = match workers
        .iter()
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
    let mut slots: Vec<RankSlots> = workers
        .iter()
        .map(|_| std::array::from_fn(|_| None))
        .collect();
    // Slot draft states to clear on the next draft round (request left the
    // slot, or a new one was admitted into it). Flushed with each step's
    // Draft commands; the handler is idempotent, so duplicates are harmless.
    let mut pending_resets: Vec<Vec<usize>> = workers.iter().map(|_| Vec::new()).collect();
    let mut pending = std::collections::VecDeque::<GenerateRequest>::new();
    let mut channel_open = true;
    let all_idle = |slots: &[RankSlots]| {
        slots
            .iter()
            .all(|rank_slots| rank_slots.iter().all(Option::is_none))
    };

    // Pre-capture every whole-step graph (bucket × attention tier) while the
    // ranks are idle and trivially in lock-step. Launch-ahead speculation
    // requires captured-ness to be UNIFORM across ranks — a lazily capturing
    // rank would skip the speculative replay the others enqueued and desync
    // the collectives — and pre-capturing also removes the old mid-serving
    // capture stall. Row 0 at position GLM52_MLA_TOPK_SHORT lifts the step
    // into the full tier; every row is a padding write into the pool's
    // padding page.
    for &bucket in &GLM52_DECODE_BUCKETS {
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
            let responses = match workers
                .iter()
                .zip(&pools)
                .map(|(worker, pool)| {
                    let kv = padding_step_kv(bucket, table_width, pool.padding_block_id(), &inputs);
                    worker.step_async(inputs, shape, kv, Glm52StepFlags::plain(), Vec::new(), 0)
                })
                .collect::<anyhow::Result<Vec<_>>>()
            {
                Ok(responses) => responses,
                Err(err) => {
                    log::error!("GLM5.2 graph pre-capture failed to submit: {err:#}");
                    // The DeepEP contexts already exist: broadcast Shutdown
                    // before the workers' sequential Drop joins them (the
                    // same collective-teardown contract as the exit path).
                    for worker in &workers {
                        let _ = worker.request_shutdown();
                    }
                    return;
                }
            };
            for (rank, resp) in responses.into_iter().enumerate() {
                let result = resp
                    .recv()
                    .map_err(|_| anyhow::anyhow!("rank dropped its pre-capture response"))
                    .and_then(|r| r);
                if let Err(err) = result {
                    log::error!(
                        "GLM5.2 graph pre-capture (bucket {bucket}, full_tier {full_tier}) \
                         failed on rank {rank}: {err:#}"
                    );
                    for worker in &workers {
                        let _ = worker.request_shutdown();
                    }
                    return;
                }
            }
        }
    }
    log::info!(
        "GLM5.2 whole-step graphs pre-captured: {} buckets x 2 tiers",
        GLM52_DECODE_BUCKETS.len()
    );

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

        // Admission: fill free slots from the queue, least-loaded rank (with
        // pool budget for the request's full lifetime) first. New requests
        // join the lock-step at the next step boundary (their prefill rides
        // decode alongside everyone else's rows).
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
            let Some((rank, slot)) =
                admission_target(&occupancy(&slots), &committed, &usable_blocks, need_blocks)
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
                        fail_step(&mut slots, &err);
                        break 'serve;
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
            slots_changed = true;
        }
        if all_idle(&slots) {
            continue;
        }

        // One lock-step step: every rank forwards the SAME bucket — each
        // active slot's span of consecutive next tokens, padding rows on the
        // free slots — and all responses are joined before any output is
        // interpreted.
        let shapes = plan_step_shapes(&feed_wants(&slots));
        let flags = launch_ahead_flags(
            &shapes,
            leased_shapes.as_deref(),
            slots_changed,
            pending.is_empty(),
            dspark_enabled,
            &slots,
            max_model_len,
        );
        leased_shapes = flags.lease.then(|| shapes.clone());
        slots_changed = false;
        sample_step += 1;
        // Per-rank submit: schedule each active span's KV (full-lifetime
        // reservation makes every schedule succeed — a failure is an
        // accounting bug and fails the step), build the row inputs, page
        // rows and write slots, collect the step's sampling rows, and fire
        // the step. `span_kinds[rank][slot]` records what was scheduled so
        // the output walk applies the exact pairing.
        let mut span_kinds: Vec<[Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK]> = workers
            .iter()
            .map(|_| [None; GLM52_MAX_BATCH_PER_RANK])
            .collect();
        let mut responses = Vec::with_capacity(workers.len());
        let mut submit_err: Option<anyhow::Error> = None;
        'submit: for (rank, ((rank_slots, worker), shape)) in
            slots.iter_mut().zip(&workers).zip(&shapes).enumerate()
        {
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
                        slot_mapping[r] = row_pages[position / PAGE] as i64 * PAGE as i64
                            + (position % PAGE) as i64;
                    }
                }
                row = end;
            }
            let kv = Glm52StepKv {
                pages: pages.into_boxed_slice(),
                slot_mapping,
            };
            match worker.step_async(inputs, *shape, kv, flags, sampling, seed) {
                Ok(rx) => responses.push(rx),
                Err(err) => {
                    submit_err = Some(err);
                    break 'submit;
                }
            }
        }
        if let Some(err) = submit_err {
            fail_step(&mut slots, &err);
            break 'serve;
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
            fail_step(&mut slots, &err);
            break 'serve;
        }

        let mut rank_appends: Vec<Vec<(usize, usize)>> =
            workers.iter().map(|_| Vec::new()).collect();
        let mut rank_proposals: Vec<Vec<(usize, u32, usize)>> =
            workers.iter().map(|_| Vec::new()).collect();
        let mut walk_err: Option<anyhow::Error> = None;
        'walk: for (rank, ((rank_slots, rank_outputs), shape)) in
            slots.iter_mut().zip(outputs).zip(&shapes).enumerate()
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
                    (
                        Glm52StepOutcome::Commit { committed, .. },
                        Some(SpanKind::PrefillBoundary),
                    ) => active.kv.apply_prefill(committed[0], pool),
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
                    walk_err =
                        Some(err.context(format!("GLM5.2 rank {rank} slot {slot_id} KV apply")));
                    break 'walk;
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
                    slots_changed = true;
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
        if let Some(err) = walk_err {
            fail_step(&mut slots, &err);
            break 'serve;
        }

        // Draft round (rank-local, no collectives): resets, context appends
        // from THIS step's capture buffer, and new proposals for the next
        // verify span. FIFO per-rank channels order it before the next step;
        // the blocking join keeps the round cadence (draft sits between
        // verify steps, ~2 ms against a 22-46 ms step).
        if dspark_enabled {
            let mut draft_joins = Vec::new();
            for (rank, worker) in workers.iter().enumerate() {
                let resets = std::mem::take(&mut pending_resets[rank]);
                let appends = std::mem::take(&mut rank_appends[rank]);
                let proposals = std::mem::take(&mut rank_proposals[rank]);
                if resets.is_empty() && appends.is_empty() && proposals.is_empty() {
                    continue;
                }
                let proposal_slots: Vec<usize> =
                    proposals.iter().map(|&(slot, _, _)| slot).collect();
                match worker.draft_async(shapes[rank].bucket, resets, appends, proposals) {
                    Ok(rx) => draft_joins.push((rank, proposal_slots, rx)),
                    Err(err) => {
                        fail_step(&mut slots, &err);
                        break 'serve;
                    }
                }
            }
            for (rank, proposal_slots, rx) in draft_joins {
                let result = rx
                    .recv()
                    .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its draft response"))
                    .and_then(|r| r);
                let spans = match result {
                    Ok(spans) => spans,
                    Err(err) => {
                        // A draft failure is rank-local, but it means the
                        // drafter's invariants broke — crash early rather
                        // than silently degrade to plain decode.
                        fail_step(
                            &mut slots,
                            &err.context(format!("GLM5.2 rank {rank} draft")),
                        );
                        break 'serve;
                    }
                };
                if spans.len() != proposal_slots.len() {
                    let err = anyhow::anyhow!(
                        "GLM5.2 rank {rank} draft returned {} spans for {} proposals",
                        spans.len(),
                        proposal_slots.len()
                    );
                    fail_step(&mut slots, &err);
                    break 'serve;
                }
                for (slot_id, span) in proposal_slots.into_iter().zip(spans) {
                    if let Some(active) = slots[rank][slot_id].as_mut() {
                        active.state.set_drafts(span.to_vec());
                    }
                }
            }
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

    // The DeepEP context drop is collective: broadcast Shutdown to every rank
    // BEFORE the workers' Drop joins them one by one — a sequential
    // shutdown-then-join would leave a rank spinning in the destroy barrier
    // for ranks that never got the command (until the ~100 s device timeout).
    for worker in &workers {
        let _ = worker.request_shutdown();
    }
    drop(workers);
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
