//! DP8 lock-step continuous-batching scheduler: up to
//! `GLM52_MAX_BATCH_PER_RANK` requests per rank, each owning one slot of the
//! rank's decode batch (and that slot's disjoint region of the paged caches).
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

use openinfer_core::engine::{FinishReason, GenerateRequest, TokenEvent, unix_now_s};
use tokio::sync::mpsc;

use crate::dspark::GLM52_DSPARK_DRAFTS;
use crate::model::{
    GLM52_DECODE_BUCKETS, GLM52_MAX_BATCH_PER_RANK, GLM52_MAX_MODEL_LEN, Glm52StepShape,
};
use crate::runner::Glm52RankWorker;

/// What a rank forwards this step. Idle ranks feed the padding input; its
/// KV/index-cache writes land in the idle rank's own dead cache slots and are
/// overwritten when a request is admitted there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StepInput {
    pub(crate) token: u32,
    pub(crate) position: usize,
}

pub(crate) const GLM52_PADDING_STEP: Glm52StepInput = Glm52StepInput {
    token: 0,
    position: 0,
};

/// The consequence of a step's output token for one request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Glm52StepOutcome {
    /// Mid-prefill: the model's output is discarded, keep feeding the prompt.
    Prefilling,
    /// Emit this token and keep decoding.
    Emit(u32),
    /// Emit this token, then the request is finished (length cap).
    EmitAndFinish(u32, FinishReason),
    /// The request is finished without emitting (EOS is suppressed but counts
    /// toward the completion length — the engine-wide contract).
    Finish(FinishReason),
}

/// One rank's active request as a pure state machine: `feed_want` /
/// `next_input_at` decide what the rank forwards (a span of consecutive
/// prompt positions mid-prefill, one decode row after), `advance_span` folds
/// the step's span of outputs in.
#[derive(Debug)]
pub(crate) struct Glm52SlotState {
    prompt: Vec<u32>,
    max_tokens: usize,
    ignore_eos: bool,
    /// Prompt tokens already fed to the model.
    fed: usize,
    /// Generated tokens (a suppressed EOS counts).
    completion: usize,
    /// The model's latest output; the next decode input once the prompt is
    /// fully fed.
    last_token: u32,
}

impl Glm52SlotState {
    pub(crate) fn new(prompt: Vec<u32>, max_tokens: usize, ignore_eos: bool) -> Self {
        Self {
            prompt,
            max_tokens,
            ignore_eos,
            fed: 0,
            completion: 0,
            last_token: 0,
        }
    }

    pub(crate) fn completion_tokens(&self) -> usize {
        self.completion
    }

    /// Rows this request can usefully fill in one step: the whole remaining
    /// prompt while mid-prefill (the planner caps it to the bucket), one
    /// decode row afterwards.
    pub(crate) fn feed_want(&self) -> usize {
        if self.fed < self.prompt.len() {
            self.prompt.len() - self.fed
        } else {
            1
        }
    }

    /// The `offset`-th row of this step's span: consecutive prompt positions
    /// while mid-prefill, the single decode row afterwards.
    pub(crate) fn next_input_at(&self, offset: usize) -> Glm52StepInput {
        if self.fed < self.prompt.len() {
            debug_assert!(self.fed + offset < self.prompt.len());
            Glm52StepInput {
                token: self.prompt[self.fed + offset],
                position: self.fed + offset,
            }
        } else {
            debug_assert_eq!(offset, 0);
            Glm52StepInput {
                token: self.last_token,
                position: self.prompt.len() + self.completion - 1,
            }
        }
    }

    /// The next spec round's anchor once the request is decoding: the latest
    /// emitted token and the position it will be fed at. `None` mid-prefill
    /// (no token to extend yet).
    pub(crate) fn decode_anchor(&self) -> Option<(u32, usize)> {
        (self.fed >= self.prompt.len() && self.completion > 0)
            .then(|| (self.last_token, self.prompt.len() + self.completion - 1))
    }

    /// Fold one step's span of outputs in. Mid-prompt rows' outputs are
    /// discarded; the row that fed the LAST prompt token yields the first
    /// generated token, and decode spans are a single row — so only
    /// `outputs.last()` ever carries a real token.
    pub(crate) fn advance_span(
        &mut self,
        outputs: &[u32],
        eos_token_ids: &[u32],
    ) -> Glm52StepOutcome {
        debug_assert!(!outputs.is_empty());
        if self.fed < self.prompt.len() {
            debug_assert!(self.fed + outputs.len() <= self.prompt.len());
            self.fed += outputs.len();
            if self.fed < self.prompt.len() {
                return Glm52StepOutcome::Prefilling;
            }
            // The last prompt token's row yielded the first generated token
            // — fall through to the decode accounting.
        } else {
            debug_assert_eq!(outputs.len(), 1);
        }
        let output = *outputs.last().expect("span outputs are non-empty");
        self.completion += 1;
        if !self.ignore_eos && eos_token_ids.contains(&output) {
            return Glm52StepOutcome::Finish(FinishReason::Stop);
        }
        if self.completion >= self.max_tokens {
            return Glm52StepOutcome::EmitAndFinish(output, FinishReason::Length);
        }
        self.last_token = output;
        Glm52StepOutcome::Emit(output)
    }
}

pub(crate) fn validate_request(req: &GenerateRequest) -> Result<(), String> {
    if req.prompt_tokens.is_empty() {
        return Err("GLM5.2 requires a non-empty prompt".to_owned());
    }
    if req.max_tokens == 0 {
        return Err("GLM5.2 requires max_tokens > 0".to_owned());
    }
    // Highest position any forward step can touch: the (max_tokens-1)-th
    // generated token is fed at position prompt+max_tokens-2, so requiring
    // prompt+max_tokens-1 <= cap keeps every step strictly below the cap.
    let last_position = req.prompt_tokens.len() + req.max_tokens - 1;
    if last_position > GLM52_MAX_MODEL_LEN {
        return Err(format!(
            "GLM5.2 bring-up context cap: prompt {} + max_tokens {} exceeds {GLM52_MAX_MODEL_LEN}",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    if !req.params.is_greedy() {
        return Err("GLM5.2 bring-up supports greedy sampling only (temperature 0)".to_owned());
    }
    if req.logprobs > 0 || req.echo {
        return Err("GLM5.2 bring-up does not support logprobs/echo".to_owned());
    }
    if req.lora_adapter.is_some() {
        return Err("GLM5.2 does not support LoRA adapters".to_owned());
    }
    Ok(())
}

struct ActiveRequest {
    req: GenerateRequest,
    state: Glm52SlotState,
    /// DSpark shadow scoring (M2): `Some` when the drafter is loaded. Each
    /// round proposes 7 drafts from the current anchor and scores them
    /// against the tokens the engine actually emits — the real accept rate
    /// the M3 verify loop will get, with zero decode-path change.
    shadow: Option<ShadowState>,
}

/// One request's shadow-round bookkeeping: at most one outstanding proposal
/// (matching the M3 round cadence — a new proposal only after the previous
/// round is fully scored), plus the accept histogram.
#[derive(Default)]
struct ShadowState {
    outstanding: Vec<u32>,
    matched: usize,
    rounds: u64,
    accepted_sum: u64,
    hist: [u64; GLM52_DSPARK_DRAFTS + 1],
}

impl ShadowState {
    /// Score one emitted token against the outstanding proposal. A mismatch
    /// or a fully-matched proposal closes the round (the M3 loop would
    /// re-anchor there).
    fn score_token(&mut self, token: u32) {
        if self.outstanding.is_empty() {
            return;
        }
        if self.outstanding[self.matched] == token {
            self.matched += 1;
            if self.matched == self.outstanding.len() {
                self.finish_round();
            }
        } else {
            self.finish_round();
        }
    }

    fn finish_round(&mut self) {
        self.rounds += 1;
        self.accepted_sum += self.matched as u64;
        self.hist[self.matched] += 1;
        self.outstanding.clear();
        self.matched = 0;
    }

    fn log_on_release(&self, rank: usize, slot: usize) {
        if self.rounds == 0 {
            return;
        }
        let mean_accepted = self.accepted_sum as f64 / self.rounds as f64;
        log::info!(
            "GLM5.2 dspark shadow: rank={rank} slot={slot} rounds={} mean_accepted_drafts={mean_accepted:.3} \
             mean_accepted_incl_bonus={:.3} hist={:?}",
            self.rounds,
            mean_accepted + 1.0,
            self.hist,
        );
    }
}

/// Per-rank slot occupancy: `slots[rank][slot]`.
type RankSlots = [Option<ActiveRequest>; GLM52_MAX_BATCH_PER_RANK];

/// Where the next queued request goes: the least-loaded rank (ties → lowest
/// rank id), its lowest free slot. `None` when every slot in the fleet is
/// taken. Least-loaded-first keeps occupancy balanced, which keeps the fleet
/// in the cheap 1-row bucket until concurrency exceeds the rank count.
fn admission_target(occupied: &[[bool; GLM52_MAX_BATCH_PER_RANK]]) -> Option<(usize, usize)> {
    let (rank, row) = occupied
        .iter()
        .enumerate()
        .min_by_key(|(rank, row)| (row.iter().filter(|&&o| o).count(), *rank))?;
    let slot = row.iter().position(|&o| !o)?;
    Some((rank, slot))
}

/// Every rank's forward shape for one step, decided together from the same
/// feed-want snapshot (`wants[rank][slot]` = rows that slot can usefully
/// fill: 0 free, 1 decode, remaining-prompt while mid-prefill).
///
/// The bucket is the smallest [`GLM52_DECODE_BUCKETS`] member covering the
/// hungriest rank's row demand (each rank's demand = Σ wants, capped at the
/// max bucket; never smaller than its active count — a smaller bucket would
/// silently drop rows). Per rank, every active slot first gets one row
/// (liveness), then the leftover bucket capacity extends mid-prefill slots
/// into *spans* (consecutive prompt positions batched through one step),
/// round-robin across the hungry slots so co-resident prefills drain in
/// parallel; padding rows ride the free slots. Span rows are emitted as one
/// contiguous run per slot — the [`Glm52StepShape`] contract.
/// Deriving the bucket and every rank's row list from the same data in one
/// place is what keeps them consistent.
fn plan_step_shapes(wants: &[[usize; GLM52_MAX_BATCH_PER_RANK]]) -> Vec<Glm52StepShape> {
    let hungriest = wants
        .iter()
        .map(|row| row.iter().sum::<usize>().min(GLM52_MAX_BATCH_PER_RANK))
        .max()
        .unwrap_or(0);
    let bucket = *GLM52_DECODE_BUCKETS
        .iter()
        .find(|&&rows| rows >= hungriest.max(1))
        .expect("the largest bucket covers every demand by construction");
    wants
        .iter()
        .map(|row| {
            // Every active slot gets one row, then leftover capacity extends
            // spans one row per slot per round (round-robin), so two
            // mid-prefill slots on one rank drain in parallel instead of the
            // lowest slot starving the later one down to a liveness row for
            // its whole prefill.
            let mut spans = [0usize; GLM52_MAX_BATCH_PER_RANK];
            let mut used = 0usize;
            for (slot, &want) in row.iter().enumerate() {
                if want > 0 {
                    // bucket >= this rank's capped demand >= its active count
                    // by construction; a dropped active would stall forever.
                    assert!(used < bucket, "bucket {bucket} smaller than active count");
                    spans[slot] = 1;
                    used += 1;
                }
            }
            loop {
                let mut gave = false;
                for (slot, &want) in row.iter().enumerate() {
                    if used < bucket && spans[slot] > 0 && spans[slot] < want {
                        spans[slot] += 1;
                        used += 1;
                        gave = true;
                    }
                }
                if !gave || used == bucket {
                    break;
                }
            }
            let mut slots: [u8; GLM52_MAX_BATCH_PER_RANK] = std::array::from_fn(|slot| slot as u8);
            let mut dst = 0usize;
            for (slot, &span) in spans.iter().enumerate() {
                for _ in 0..span {
                    slots[dst] = slot as u8;
                    dst += 1;
                }
            }
            // Padding rows on free slots: there are always enough, because
            // used >= actives and bucket <= MAX, so bucket - used <= frees.
            let mut frees = (0..GLM52_MAX_BATCH_PER_RANK).filter(|&slot| row[slot] == 0);
            while dst < bucket {
                slots[dst] = frees.next().expect("bucket - used <= free slots") as u8;
                dst += 1;
            }
            Glm52StepShape { bucket, slots }
        })
        .collect()
}

fn feed_wants(slots: &[RankSlots]) -> Vec<[usize; GLM52_MAX_BATCH_PER_RANK]> {
    slots
        .iter()
        .map(|rank_slots| {
            std::array::from_fn(|slot| {
                rank_slots[slot]
                    .as_ref()
                    .map_or(0, |active| active.state.feed_want())
            })
        })
        .collect()
}

fn occupancy(slots: &[RankSlots]) -> Vec<[bool; GLM52_MAX_BATCH_PER_RANK]> {
    slots
        .iter()
        .map(|rank_slots| std::array::from_fn(|slot| rank_slots[slot].is_some()))
        .collect()
}

/// DP8 coordinator: admits up to `GLM52_MAX_BATCH_PER_RANK` requests per rank
/// (least-loaded rank first) and drives all ranks in lock-step. Consumes the
/// workers; returns when the submit channel closes or a step fails (the EP8
/// collective group cannot recover from a failed step — see the teardown
/// comment below).
pub(crate) fn run_dp8_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52RankWorker>,
    eos_token_ids: &[u32],
    dspark_enabled: bool,
) {
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

    'serve: loop {
        // Intake: block when fully idle, otherwise drain what's queued.
        if channel_open && all_idle(&slots) && pending.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => intake(req, &mut pending),
                None => channel_open = false,
            }
        }
        while channel_open {
            match submit_rx.try_recv() {
                Ok(req) => intake(req, &mut pending),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => channel_open = false,
            }
        }
        if !channel_open && all_idle(&slots) && pending.is_empty() {
            break;
        }

        // Admission: fill free slots from the queue, least-loaded rank first.
        // New requests join the lock-step at the next step boundary (their
        // prefill rides decode alongside everyone else's rows).
        while !pending.is_empty() {
            let Some((rank, slot)) = admission_target(&occupancy(&slots)) else {
                break;
            };
            let req = pending.pop_front().expect("checked non-empty");
            // The client left while the request sat in the queue — admitting
            // it would burn a slot (and whole global steps) on a dead sink.
            if req.token_tx.is_closed() {
                continue;
            }
            let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
            let _ = req.token_tx.send(TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s: unix_now_s(),
                prompt_tokens: req.prompt_tokens.len(),
                cached_tokens: 0,
            });
            let state = Glm52SlotState::new(
                req.prompt_tokens.clone(),
                req.max_tokens,
                req.params.ignore_eos,
            );
            if dspark_enabled {
                pending_resets[rank].push(slot);
            }
            slots[rank][slot] = Some(ActiveRequest {
                req,
                state,
                shadow: dspark_enabled.then(ShadowState::default),
            });
        }
        if all_idle(&slots) {
            continue;
        }

        // One lock-step step: every rank forwards the SAME bucket — each
        // active slot's span of consecutive next tokens, padding rows on the
        // free slots — and all responses are joined before any output is
        // interpreted.
        let shapes = plan_step_shapes(&feed_wants(&slots));
        let responses = slots
            .iter()
            .zip(&workers)
            .zip(&shapes)
            .map(|((rank_slots, worker), shape)| {
                let mut inputs = [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position);
                    GLM52_MAX_BATCH_PER_RANK];
                // Row r is offset `span_offset[slot]` into its slot's span —
                // spans are contiguous runs, so a per-slot counter walks them.
                let mut span_offset = [0usize; GLM52_MAX_BATCH_PER_RANK];
                for (row, input) in inputs.iter_mut().enumerate().take(shape.bucket) {
                    let slot = shape.slots[row] as usize;
                    if let Some(active) = &rank_slots[slot] {
                        let step = active.state.next_input_at(span_offset[slot]);
                        span_offset[slot] += 1;
                        *input = (step.token, step.position);
                    }
                }
                worker.step_async(inputs, *shape)
            })
            .collect::<anyhow::Result<Vec<_>>>();
        let responses = match responses {
            Ok(responses) => responses,
            Err(err) => {
                fail_step(&mut slots, &err);
                break 'serve;
            }
        };
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
        for (rank, ((rank_slots, rank_outputs), shape)) in
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
                let freed = match active.state.advance_span(span_outputs, eos_token_ids) {
                    Glm52StepOutcome::Prefilling => {
                        // Prefill never sends, so a disconnect is only
                        // visible through the sink probe — without it a long
                        // prompt zombies the slot until prefill completes.
                        active.req.token_tx.is_closed()
                    }
                    Glm52StepOutcome::Emit(token) => {
                        if let Some(shadow) = active.shadow.as_mut() {
                            shadow.score_token(token);
                        }
                        // A dropped receiver (client disconnect) frees the
                        // slot; its KV lives in the slot's own cache region
                        // and dies with the slot.
                        active
                            .req
                            .token_tx
                            .send(TokenEvent::Token {
                                id: token,
                                logprob: None,
                            })
                            .is_err()
                    }
                    Glm52StepOutcome::EmitAndFinish(token, finish_reason) => {
                        if let Some(shadow) = active.shadow.as_mut() {
                            shadow.score_token(token);
                        }
                        let _ = active.req.token_tx.send(TokenEvent::Token {
                            id: token,
                            logprob: None,
                        });
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        true
                    }
                    Glm52StepOutcome::Finish(finish_reason) => {
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        true
                    }
                };
                if freed {
                    if let Some(shadow) = &active.shadow {
                        shadow.log_on_release(rank, slot_id);
                        pending_resets[rank].push(slot_id);
                    }
                    *slot = None;
                } else if let Some(shadow) = &active.shadow {
                    // Every step row of a live slot feeds the draft context;
                    // a new proposal starts once the previous round is fully
                    // scored (the M3 verify cadence).
                    rank_appends[rank].extend(span_rows.map(|r| (r, slot_id)));
                    if shadow.outstanding.is_empty()
                        && let Some((anchor, anchor_pos)) = active.state.decode_anchor()
                    {
                        rank_proposals[rank].push((slot_id, anchor, anchor_pos));
                    }
                }
            }
        }

        // Draft round (rank-local, no collectives): resets, context appends
        // from THIS step's capture buffer, and new proposals. FIFO per-rank
        // channels order it before the next step; the blocking join mirrors
        // the M3 round cadence (draft sits between verify steps).
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
                    if let Some(active) = slots[rank][slot_id].as_mut()
                        && let Some(shadow) = active.shadow.as_mut()
                    {
                        shadow.outstanding = span.to_vec();
                        shadow.matched = 0;
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

/// Fast-reject invalid requests at intake (Scheduled → Rejected, the same
/// event order the bs=1 coordinator emitted); valid ones queue for a rank.
fn intake(req: GenerateRequest, pending: &mut std::collections::VecDeque<GenerateRequest>) {
    if let Err(message) = validate_request(&req) {
        let prompt_tokens = req.prompt_tokens.len();
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens: 0,
        });
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens: 0,
        });
        return;
    }
    pending.push_back(req);
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

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: &[u32] = &[7];

    #[test]
    fn prefill_rides_decode_then_emits() {
        let mut state = Glm52SlotState::new(vec![10, 11, 12], 4, false);

        assert_eq!(state.feed_want(), 3);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 10,
                position: 0
            }
        );
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 11,
                position: 1
            }
        );
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);

        // The last prompt token's step yields the first generated token.
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 12,
                position: 2
            }
        );
        assert_eq!(state.advance_span(&[42], EOS), Glm52StepOutcome::Emit(42));
        assert_eq!(state.completion_tokens(), 1);

        // Decode continues from the emitted token at the next position.
        assert_eq!(state.feed_want(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 3
            }
        );
    }

    #[test]
    fn prompt_span_feeds_consecutive_positions_and_keeps_only_the_last_output() {
        let mut state = Glm52SlotState::new(vec![10, 11, 12, 13], 4, false);

        // One span covers three prompt tokens; mid-prompt outputs discarded.
        assert_eq!(state.feed_want(), 4);
        assert_eq!(
            (0..3).map(|i| state.next_input_at(i)).collect::<Vec<_>>(),
            vec![
                Glm52StepInput {
                    token: 10,
                    position: 0
                },
                Glm52StepInput {
                    token: 11,
                    position: 1
                },
                Glm52StepInput {
                    token: 12,
                    position: 2
                },
            ]
        );
        assert_eq!(
            state.advance_span(&[99, 98, 97], EOS),
            Glm52StepOutcome::Prefilling
        );

        // The next span finishes the prompt; its last output is the first
        // generated token.
        assert_eq!(state.feed_want(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 13,
                position: 3
            }
        );
        assert_eq!(state.advance_span(&[42], EOS), Glm52StepOutcome::Emit(42));
        assert_eq!(state.completion_tokens(), 1);
    }

    #[test]
    fn whole_prompt_in_one_span_emits_from_the_boundary_row() {
        let mut state = Glm52SlotState::new(vec![10, 11, 12], 4, false);
        assert_eq!(
            state.advance_span(&[99, 98, 42], EOS),
            Glm52StepOutcome::Emit(42)
        );
        assert_eq!(state.completion_tokens(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 3
            }
        );
    }

    #[test]
    fn eos_is_suppressed_and_counts_toward_completion() {
        let mut state = Glm52SlotState::new(vec![10], 4, false);
        assert_eq!(
            state.advance_span(&[7], EOS),
            Glm52StepOutcome::Finish(FinishReason::Stop)
        );
        assert_eq!(state.completion_tokens(), 1);
    }

    #[test]
    fn ignore_eos_decodes_through_the_stop_token() {
        let mut state = Glm52SlotState::new(vec![10], 4, true);
        assert_eq!(state.advance_span(&[7], EOS), Glm52StepOutcome::Emit(7));
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 7,
                position: 1
            }
        );
    }

    #[test]
    fn length_cap_emits_the_final_token() {
        let mut state = Glm52SlotState::new(vec![10], 2, false);
        assert_eq!(state.advance_span(&[42], EOS), Glm52StepOutcome::Emit(42));
        assert_eq!(
            state.advance_span(&[43], EOS),
            Glm52StepOutcome::EmitAndFinish(43, FinishReason::Length)
        );
        assert_eq!(state.completion_tokens(), 2);
    }

    #[test]
    fn eos_outranks_the_length_cap() {
        let mut state = Glm52SlotState::new(vec![10], 1, false);
        assert_eq!(
            state.advance_span(&[7], EOS),
            Glm52StepOutcome::Finish(FinishReason::Stop)
        );
    }

    #[test]
    fn max_tokens_one_emits_then_finishes() {
        let mut state = Glm52SlotState::new(vec![10, 11], 1, false);
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.advance_span(&[42], EOS),
            Glm52StepOutcome::EmitAndFinish(42, FinishReason::Length)
        );
    }

    #[test]
    fn shadow_closes_rounds_at_mismatch_and_at_full_match() {
        let mut shadow = ShadowState::default();
        // No outstanding proposal: tokens are ignored.
        shadow.score_token(9);
        assert_eq!(shadow.rounds, 0);

        shadow.outstanding = vec![1, 2, 3, 4, 5, 6, 7];
        shadow.score_token(1);
        shadow.score_token(2);
        shadow.score_token(9); // mismatch closes the round at 2 accepted
        assert_eq!((shadow.rounds, shadow.accepted_sum), (1, 2));
        assert!(shadow.outstanding.is_empty());

        shadow.outstanding = vec![1, 2, 3, 4, 5, 6, 7];
        for token in [1, 2, 3, 4, 5, 6, 7] {
            shadow.score_token(token);
        }
        assert_eq!((shadow.rounds, shadow.accepted_sum), (2, 9));
        assert_eq!(shadow.hist[2], 1);
        assert_eq!(shadow.hist[7], 1);
        assert!(shadow.outstanding.is_empty());
    }

    #[test]
    fn decode_anchor_is_the_latest_token_at_its_feed_position() {
        let mut state = Glm52SlotState::new(vec![10, 11], 4, false);
        assert_eq!(state.decode_anchor(), None);
        assert_eq!(
            state.advance_span(&[99, 42], EOS),
            Glm52StepOutcome::Emit(42)
        );
        // The anchor is what the next decode row would feed.
        assert_eq!(state.decode_anchor(), Some((42, 2)));
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 2
            }
        );
        assert_eq!(state.advance_span(&[43], EOS), Glm52StepOutcome::Emit(43));
        assert_eq!(state.decode_anchor(), Some((43, 3)));
    }

    fn occ(counts: &[usize]) -> Vec<[bool; GLM52_MAX_BATCH_PER_RANK]> {
        counts
            .iter()
            .map(|&c| std::array::from_fn(|slot| slot < c))
            .collect()
    }

    /// `counts` decode-phase requests per rank (each wants one row).
    fn decode_wants(counts: &[usize]) -> Vec<[usize; GLM52_MAX_BATCH_PER_RANK]> {
        counts
            .iter()
            .map(|&c| std::array::from_fn(|slot| usize::from(slot < c)))
            .collect()
    }

    #[test]
    fn admission_prefers_least_loaded_rank_then_lowest_slot() {
        // Empty fleet: rank 0, slot 0.
        assert_eq!(admission_target(&occ(&[0, 0, 0])), Some((0, 0)));
        // Rank 1 is the least loaded.
        assert_eq!(admission_target(&occ(&[2, 1, 2])), Some((1, 1)));
        // Tie between ranks 0 and 2 → lowest rank id.
        assert_eq!(admission_target(&occ(&[1, 2, 1])), Some((0, 1)));
        // A hole in the middle of a rank's slots is reused first.
        let mut holey = occ(&[3, 3]);
        holey[1][1] = false;
        assert_eq!(admission_target(&holey), Some((1, 1)));
        // Full fleet: no target.
        assert_eq!(admission_target(&occ(&[GLM52_MAX_BATCH_PER_RANK; 2])), None);
    }

    /// The observable part of a shape: the bucket and the forwarded rows'
    /// slots (trailing entries beyond the bucket are never read).
    fn forwarded(shapes: &[Glm52StepShape]) -> Vec<(usize, Vec<u8>)> {
        shapes
            .iter()
            .map(|shape| (shape.bucket, shape.slots[..shape.bucket].to_vec()))
            .collect()
    }

    #[test]
    fn bucket_is_the_smallest_covering_the_hungriest_rank() {
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[0, 0]))),
            vec![(1, vec![0]), (1, vec![0])]
        );
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[1; 8]))),
            vec![(1, vec![0]); 8]
        );
        // One rank at two requests lifts EVERY rank to the 2-row bucket —
        // idle ranks pad with free slots.
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[2, 1]))),
            vec![(2, vec![0, 1]), (2, vec![0, 1])]
        );
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[3, 1])))[0],
            (4, vec![0, 1, 2, 3])
        );
        // Past the 4-row bucket the full batch takes over.
        assert_eq!(forwarded(&plan_step_shapes(&decode_wants(&[5, 1])))[0].0, 8);
    }

    #[test]
    fn partial_buckets_pack_actives_first() {
        // A rank holding slots {1, 5} forwards them in rows 0..2; the padding
        // rows (bucket 4) ride on the lowest free slots.
        let mut holey = decode_wants(&[0, 3]);
        holey[0][1] = 1;
        holey[0][5] = 1;
        assert_eq!(
            forwarded(&plan_step_shapes(&holey)),
            vec![(4, vec![1, 5, 0, 2]), (4, vec![0, 1, 2, 3])]
        );
        let mut deep = decode_wants(&[5, 0]);
        deep[0][0] = 0;
        deep[0][7] = 1;
        assert_eq!(
            forwarded(&plan_step_shapes(&deep))[0],
            (8, vec![1, 2, 3, 4, 7, 0, 5, 6])
        );
    }

    #[test]
    fn prefill_want_extends_one_slot_into_a_span() {
        // A lone mid-prefill request with plenty of prompt left fills the
        // whole max bucket with its span; idle ranks pad.
        let mut wants = decode_wants(&[0, 0]);
        wants[0][2] = 3000;
        let shapes = plan_step_shapes(&wants);
        assert_eq!(
            forwarded(&shapes)[0],
            (8, vec![2, 2, 2, 2, 2, 2, 2, 2]),
            "one hungry slot owns every row of the max bucket"
        );
        assert_eq!(
            forwarded(&shapes)[1],
            (8, (0..8).map(|s| s as u8).collect())
        );

        // A short prompt remainder only lifts the bucket as far as needed.
        let mut wants = decode_wants(&[0, 0]);
        wants[0][0] = 3;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants))[0],
            (4, vec![0, 0, 0, 1])
        );
    }

    #[test]
    fn spans_share_the_bucket_with_decode_slots_actives_first() {
        // Slot 0 decodes (1 row), slot 1 is mid-prefill: liveness rows first,
        // then the leftover capacity extends the prefill span — one
        // contiguous run per slot.
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 1;
        wants[0][1] = 100;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants))[0],
            (8, vec![0, 1, 1, 1, 1, 1, 1, 1])
        );

        // Two mid-prefill slots with small wants: both met, remaining rows
        // pad on free slots.
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 3;
        wants[0][1] = 2;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants))[0],
            (8, vec![0, 0, 0, 1, 1, 2, 3, 4]),
            "wants met, remaining rows pad on free slots"
        );
    }

    #[test]
    fn two_long_prefills_split_the_leftover_round_robin() {
        // Two co-resident long prefills split the bucket evenly — neither
        // starves at a single liveness row while the other eats the leftover.
        let mut wants = decode_wants(&[0]);
        wants[0][2] = 3000;
        wants[0][5] = 3000;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants))[0],
            (8, vec![2, 2, 2, 2, 5, 5, 5, 5])
        );

        // A decode slot in the mix keeps its single row; the prefills split
        // what remains (7 rows -> 4 + 3 by round-robin order).
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 1;
        wants[0][3] = 3000;
        wants[0][6] = 3000;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants))[0],
            (8, vec![0, 3, 3, 3, 3, 6, 6, 6])
        );
    }
}
