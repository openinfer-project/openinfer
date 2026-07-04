//! DP8 lock-step continuous-batching scheduler: up to
//! `GLM52_MAX_BATCH_PER_RANK` requests per rank, each owning one slot of the
//! rank's decode batch (and that slot's disjoint region of the paged caches).
//!
//! Every global step ALL ranks run the full-model forward simultaneously with
//! the SAME batch bucket — ranks feed each active slot's next token (prompt
//! tokens ride the decode path one position at a time), idle slots feed a
//! padding row whose output is discarded. This satisfies the DeepEP contract
//! that every rank enters every MoE layer's dispatch/combine collective with
//! the agreed global row count. The bucket is {1, 8}: while no rank holds
//! more than one request, every rank steps a single row (the cheap 1-row
//! graphs); as soon as any rank holds two, every rank steps the full 8-row
//! batch. Requests join and leave slots at step boundaries (continuous
//! batching) — admission is least-loaded rank first, so the fleet reaches
//! bucket 8 only past `GLM52_EP_RANKS` concurrent requests.
//!
//! The per-request decisions (what to feed next, what a step's output means)
//! live in [`Glm52SlotState`] as pure data transitions, and the
//! admission/step-shape decisions in [`admission_target`] /
//! [`plan_step_shapes`] as pure functions over the occupancy; the coordinator
//! is a thin shell that moves tokens between channels and the rank workers.

use openinfer_core::engine::{FinishReason, GenerateRequest, TokenEvent, unix_now_s};
use tokio::sync::mpsc;

use crate::model::{GLM52_MAX_BATCH_PER_RANK, GLM52_MAX_MODEL_LEN, Glm52StepShape};
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

/// One rank's active request as a pure state machine: `next_input` decides
/// what the rank forwards, `advance` folds the step's output token in.
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

    pub(crate) fn next_input(&self) -> Glm52StepInput {
        if self.fed < self.prompt.len() {
            Glm52StepInput {
                token: self.prompt[self.fed],
                position: self.fed,
            }
        } else {
            Glm52StepInput {
                token: self.last_token,
                position: self.prompt.len() + self.completion - 1,
            }
        }
    }

    pub(crate) fn advance(&mut self, output: u32, eos_token_ids: &[u32]) -> Glm52StepOutcome {
        if self.fed < self.prompt.len() {
            self.fed += 1;
            if self.fed < self.prompt.len() {
                return Glm52StepOutcome::Prefilling;
            }
            // The last prompt token's step yielded the first generated token
            // — fall through to the decode accounting.
        }
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
/// occupancy snapshot: the 1-row bucket only when EVERY rank holds at most
/// one request (each rank forwards its single occupied slot — or slot 0 as
/// padding when idle), the full batch everywhere otherwise. Deriving the
/// bucket and the per-rank `One` slot from the same row data in one place is
/// what keeps them consistent — the MoE collectives require all ranks to
/// agree on the step's global row count, and a rank with two active slots
/// must never be handed a 1-row shape (its second row would be silently
/// dropped).
fn plan_step_shapes(occupied: &[[bool; GLM52_MAX_BATCH_PER_RANK]]) -> Vec<Glm52StepShape> {
    let one_row = occupied
        .iter()
        .all(|row| row.iter().filter(|&&o| o).count() <= 1);
    occupied
        .iter()
        .map(|row| {
            if one_row {
                Glm52StepShape::One {
                    slot: row.iter().position(|&o| o).unwrap_or(0),
                }
            } else {
                Glm52StepShape::Full
            }
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
) {
    let mut slots: Vec<RankSlots> = workers
        .iter()
        .map(|_| std::array::from_fn(|_| None))
        .collect();
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
            slots[rank][slot] = Some(ActiveRequest { req, state });
        }
        if all_idle(&slots) {
            continue;
        }

        // One lock-step step: every rank forwards the SAME bucket — each
        // active slot's next token, padding rows elsewhere — and all
        // responses are joined before any output is interpreted.
        let shapes = plan_step_shapes(&occupancy(&slots));
        let responses = slots
            .iter()
            .zip(&workers)
            .zip(shapes)
            .map(|((rank_slots, worker), shape)| {
                let mut inputs = [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position);
                    GLM52_MAX_BATCH_PER_RANK];
                for (slot, active) in rank_slots.iter().enumerate() {
                    if let Some(active) = active {
                        let input = active.state.next_input();
                        inputs[slot] = (input.token, input.position);
                    }
                }
                worker.step_async(inputs, shape)
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

        for (rank_slots, rank_outputs) in slots.iter_mut().zip(outputs) {
            for (slot, output) in rank_slots.iter_mut().zip(rank_outputs) {
                let Some(active) = slot.as_mut() else {
                    continue;
                };
                let prompt_tokens = active.req.prompt_tokens.len();
                match active.state.advance(output, eos_token_ids) {
                    Glm52StepOutcome::Prefilling => {
                        // Prefill never sends, so a disconnect is only
                        // visible through the sink probe — without it a long
                        // prompt zombies the slot until prefill completes.
                        if active.req.token_tx.is_closed() {
                            *slot = None;
                        }
                    }
                    Glm52StepOutcome::Emit(token) => {
                        // A dropped receiver (client disconnect) frees the
                        // slot; its KV lives in the slot's own cache region
                        // and dies with the slot.
                        if active
                            .req
                            .token_tx
                            .send(TokenEvent::Token {
                                id: token,
                                logprob: None,
                            })
                            .is_err()
                        {
                            *slot = None;
                        }
                    }
                    Glm52StepOutcome::EmitAndFinish(token, finish_reason) => {
                        let _ = active.req.token_tx.send(TokenEvent::Token {
                            id: token,
                            logprob: None,
                        });
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        *slot = None;
                    }
                    Glm52StepOutcome::Finish(finish_reason) => {
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        *slot = None;
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

        assert_eq!(
            state.next_input(),
            Glm52StepInput {
                token: 10,
                position: 0
            }
        );
        assert_eq!(state.advance(99, EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.next_input(),
            Glm52StepInput {
                token: 11,
                position: 1
            }
        );
        assert_eq!(state.advance(99, EOS), Glm52StepOutcome::Prefilling);

        // The last prompt token's step yields the first generated token.
        assert_eq!(
            state.next_input(),
            Glm52StepInput {
                token: 12,
                position: 2
            }
        );
        assert_eq!(state.advance(42, EOS), Glm52StepOutcome::Emit(42));
        assert_eq!(state.completion_tokens(), 1);

        // Decode continues from the emitted token at the next position.
        assert_eq!(
            state.next_input(),
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
            state.advance(7, EOS),
            Glm52StepOutcome::Finish(FinishReason::Stop)
        );
        assert_eq!(state.completion_tokens(), 1);
    }

    #[test]
    fn ignore_eos_decodes_through_the_stop_token() {
        let mut state = Glm52SlotState::new(vec![10], 4, true);
        assert_eq!(state.advance(7, EOS), Glm52StepOutcome::Emit(7));
        assert_eq!(
            state.next_input(),
            Glm52StepInput {
                token: 7,
                position: 1
            }
        );
    }

    #[test]
    fn length_cap_emits_the_final_token() {
        let mut state = Glm52SlotState::new(vec![10], 2, false);
        assert_eq!(state.advance(42, EOS), Glm52StepOutcome::Emit(42));
        assert_eq!(
            state.advance(43, EOS),
            Glm52StepOutcome::EmitAndFinish(43, FinishReason::Length)
        );
        assert_eq!(state.completion_tokens(), 2);
    }

    #[test]
    fn eos_outranks_the_length_cap() {
        let mut state = Glm52SlotState::new(vec![10], 1, false);
        assert_eq!(
            state.advance(7, EOS),
            Glm52StepOutcome::Finish(FinishReason::Stop)
        );
    }

    #[test]
    fn max_tokens_one_emits_then_finishes() {
        let mut state = Glm52SlotState::new(vec![10, 11], 1, false);
        assert_eq!(state.advance(99, EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.advance(42, EOS),
            Glm52StepOutcome::EmitAndFinish(42, FinishReason::Length)
        );
    }

    fn occ(counts: &[usize]) -> Vec<[bool; GLM52_MAX_BATCH_PER_RANK]> {
        counts
            .iter()
            .map(|&c| std::array::from_fn(|slot| slot < c))
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

    #[test]
    fn shapes_are_one_row_until_a_rank_holds_two() {
        use Glm52StepShape::{Full, One};
        assert_eq!(
            plan_step_shapes(&occ(&[0, 0])),
            vec![One { slot: 0 }, One { slot: 0 }]
        );
        assert_eq!(
            plan_step_shapes(&occ(&[1, 1, 1, 1, 1, 1, 1, 1])),
            vec![One { slot: 0 }; 8]
        );
        // One rank at two requests flips EVERY rank to the full batch.
        assert_eq!(plan_step_shapes(&occ(&[2, 1])), vec![Full, Full]);
        // The single active slot need not be slot 0 for the 1-row bucket,
        // and each rank's One slot is its own occupied slot.
        let mut mid_slot = occ(&[1, 0]);
        mid_slot[1][5] = true;
        assert_eq!(
            plan_step_shapes(&mid_slot),
            vec![One { slot: 0 }, One { slot: 5 }]
        );
        assert_eq!(admission_target(&mid_slot), Some((0, 1)));
    }
}
