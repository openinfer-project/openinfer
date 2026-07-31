//! Request validation and slot placement for one rank: [`validate_request`]
//! fast-rejects at the door (past it, a bad value only surfaces inside a
//! collective and tears the engine down), then [`admit_from_queue`] fills the
//! rank's free slots from its own FIFO queue at step boundaries under the
//! full-lifetime KV budget. Requests arrive pre-bound to this rank — the
//! `EngineHandle` routes by `data_parallel_rank` (the vLLM frontend's DP
//! choice) and least-load-places unbound ones — so admission never moves a
//! request and metrics/KV ownership agree with the frontend's engine index.

use std::collections::VecDeque;

use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::unix_now_s;
use openinfer_kv_cache::BlockPool;

use super::ActiveRequest;
use super::PAGE;
use super::RankSlots;
use super::offload::NativeAdmitOutcome;
use super::offload::{self};
use super::slot::Glm52SlotState;

pub(super) fn validate_request(
    req: &GenerateRequest,
    max_model_len: usize,
    prefill_only: bool,
    native_mtp_prefill: bool,
) -> Result<(), String> {
    if req.prompt_tokens.is_empty() {
        return Err("GLM5.2 requires a non-empty prompt".to_owned());
    }
    if req.max_tokens == 0 {
        return Err("GLM5.2 requires max_tokens > 0".to_owned());
    }
    if prefill_only && req.max_tokens != 1 {
        return Err(format!(
            "GLM5.2 prefill-only mode requires max_tokens=1, got {}",
            req.max_tokens
        ));
    }
    // Highest position any forward step can touch: the (max_tokens-1)-th
    // generated token is fed at position prompt+max_tokens-2, so requiring
    // prompt+max_tokens-1 <= cap keeps every step strictly below the cap.
    let last_position = req.prompt_tokens.len() + req.max_tokens - 1;
    if last_position > max_model_len {
        return Err(format!(
            "GLM5.2 context cap: prompt {} + max_tokens {} exceeds max_model_len {max_model_len}",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    if native_mtp_prefill
        && req.prompt_tokens.len() + crate::mtp::GLM52_MTP_DRAFTS - 1 > max_model_len
    {
        return Err(format!(
            "GLM5.2 native-MTP prefill requires {} positions of proposal headroom: \
             prompt {} exceeds max_model_len {max_model_len}",
            crate::mtp::GLM52_MTP_DRAFTS - 1,
            req.prompt_tokens.len()
        ));
    }
    // Mirror the sampler kernel's parameter ensures HERE: past intake a bad
    // value only surfaces as a failed step, and a failed step is fatal to the
    // engine — user input must be rejected at the door, never inside a
    // collective.
    if !req.params.is_greedy() {
        let p = &req.params;
        if !p.temperature.is_finite() {
            return Err(format!(
                "GLM5.2 sampling requires a finite temperature, got {}",
                p.temperature
            ));
        }
        if !(p.top_p > 0.0 && p.top_p <= 1.0) {
            return Err(format!(
                "GLM5.2 sampling requires top_p in (0, 1], got {}",
                p.top_p
            ));
        }
        if !(p.min_p.is_finite() && (0.0..1.0).contains(&p.min_p)) {
            return Err(format!(
                "GLM5.2 sampling requires min_p in [0, 1), got {}",
                p.min_p
            ));
        }
    }
    if req.logprobs > 0 || req.echo {
        return Err("GLM5.2 bring-up does not support logprobs/echo".to_owned());
    }
    if req.lora_adapter.is_some() {
        return Err("GLM5.2 does not support LoRA adapters".to_owned());
    }
    Ok(())
}

/// Pool pages a request draws over its whole lifetime, reserved at
/// admission. One more token than the last KV-written position: kvbm appends
/// the final generated token to the sequence and provisions its page even
/// though its KV is never written (the dangling-token contract — the same
/// off-by-one Kimi's admission had to learn empirically).
pub(super) fn lifetime_blocks(prompt_tokens: usize, max_tokens: usize) -> usize {
    (prompt_tokens + max_tokens).div_ceil(PAGE)
}

fn admission_lifetime_blocks(
    req: &GenerateRequest,
    native_anchor: Option<offload::NativeAnchorPlan>,
) -> anyhow::Result<usize> {
    let (input_tokens, max_output_tokens) = match native_anchor {
        Some(anchor) => {
            let shape = offload::native_kv_shape(req, anchor)?;
            (shape.input_tokens, shape.max_output_tokens)
        }
        None => (req.prompt_tokens.len(), req.max_tokens),
    };
    Ok(lifetime_blocks(input_tokens, max_output_tokens))
}

pub(super) fn reject(req: &GenerateRequest, message: String) {
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
}

fn reject_native_pd_error(
    state: &mut offload::NativePdState,
    rank: usize,
    req: &GenerateRequest,
    err: &anyhow::Error,
) {
    state.clear(rank);
    reject(
        req,
        format!("GLM5.2 native-MTP P/D restore failed ({err:#}); retry via P"),
    );
}

/// Admission: fill this rank's free slots from its own FIFO queue while its
/// full-lifetime KV budget permits. New requests join the step cadence at
/// the next step boundary. An `Err` is a kvbm invariant break — the caller
/// treats it as engine-fatal (the affected request was already answered
/// here).
#[allow(clippy::too_many_arguments)]
pub(super) fn admit_from_queue(
    rank: usize,
    pending: &mut VecDeque<GenerateRequest>,
    slots: &mut RankSlots,
    pool: &BlockPool,
    usable_blocks: usize,
    offload: Option<&offload::RankOffload>,
    native_pd: &mut Option<offload::NativePdState>,
    host_restore: &mut Option<offload::HostRestoreState>,
    prefix_cache_enabled: bool,
    drafter_enabled: bool,
    native_mtp_prefill: bool,
    pending_resets: &mut Vec<usize>,
) -> anyhow::Result<()> {
    // Release the page holds of abandoned loads whose DMA settled.
    if let Some(state) = host_restore.as_mut() {
        state.reap();
    }
    if let Some(state) = native_pd.as_mut() {
        state.reap();
    }
    let mut committed: usize = slots
        .iter()
        .flatten()
        .map(|active| active.kv.lifetime_blocks())
        .sum();
    // Pages pinned by in-flight release saves are physically unallocatable
    // until their D2H lands. Hide them from the rank's full-lifetime budget
    // so admission defers instead of promising pages a later schedule cannot
    // get (which would fail the whole engine).
    let usable =
        usable_blocks.saturating_sub(offload.map_or(0, offload::RankOffload::pinned_blocks));

    while let Some(slot) = slots.iter().position(Option::is_none) {
        let Some(front) = pending.front() else {
            break;
        };
        // Drop a disconnected FIFO front before it can block valid work
        // behind an admission budget it will never consume.
        if front.token_tx.is_closed() {
            pending.pop_front();
            if let Some(pd) = native_pd.as_mut() {
                pd.clear(rank);
            }
            if let Some(state) = host_restore.as_mut() {
                state.abandon_front();
            }
            continue;
        }
        // Parse the native contract before budgeting: it changes the
        // logical input/output capacity of the RequestKv created below.
        // Invalid metadata must be rejected, not left at the FIFO head.
        let native_handoff = match offload::native_mtp_handoff(front) {
            Ok(handoff) => handoff,
            Err(err) => {
                let req = pending.pop_front().expect("checked non-empty");
                reject(&req, format!("{err:#}"));
                continue;
            }
        };
        let native_anchor = match native_handoff
            .as_ref()
            .map(|handoff| offload::native_anchor_plan(front, handoff))
            .transpose()
        {
            Ok(plan) => plan,
            Err(err) => {
                let req = pending.pop_front().expect("checked non-empty");
                reject(&req, format!("{err:#}"));
                continue;
            }
        };
        let need_blocks = match admission_lifetime_blocks(front, native_anchor) {
            Ok(blocks) => blocks,
            Err(err) => {
                let req = pending.pop_front().expect("checked non-empty");
                reject(&req, format!("{err:#}"));
                continue;
            }
        };
        // `usable` accounts for the block classes the scheduler knows
        // about. The allocator is the final authority: duplicate
        // primaries, restore probes, or another guard lifetime can make
        // fewer pages physically allocatable than that bookkeeping
        // predicts. Add back only pages held by active requests (already
        // represented in `committed`) and defer the FIFO front if the
        // resulting physical lifetime budget is smaller.
        let active_resident: usize = slots
            .iter()
            .flatten()
            .map(|active| active.kv.resident_blocks())
            .sum();
        // The parked front's own restore holds are already inside
        // `need_blocks`; credit them back or the front wedges forever.
        let front_restore_held = host_restore
            .as_ref()
            .map_or(0, offload::HostRestoreState::front_held_blocks)
            + native_pd
                .as_ref()
                .map_or(0, |pd| pd.front_held_blocks(rank));
        let physical_usable = pool
            .available_blocks()
            .saturating_add(active_resident)
            .saturating_add(front_restore_held);
        if committed + need_blocks > usable.min(physical_usable) {
            break;
        }

        let mut req = pending.pop_front().expect("checked non-empty");
        let client_prompt_tokens = req.prompt_tokens.len();
        let native_admitted = if let Some(handoff) = native_handoff.as_ref() {
            let Some(state) = native_pd.as_mut() else {
                reject(
                    &req,
                    "native-MTP P/D metadata reached a decode engine without native P/D offload"
                        .to_string(),
                );
                continue;
            };
            let offload = offload.expect("native P/D state requires offload");
            let outcome =
                match offload::admit_native_mtp_pd(state, rank, offload, pool, &req, handoff) {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        reject_native_pd_error(state, rank, &req, &err);
                        continue;
                    }
                };
            match outcome {
                NativeAdmitOutcome::Admit { kv, cached_tokens } => Some((*kv, cached_tokens)),
                NativeAdmitOutcome::Park => {
                    pending.push_front(req);
                    break; // head-of-line wait: retry next step boundary
                }
                NativeAdmitOutcome::Reject { message } => {
                    reject(&req, message);
                    continue;
                }
            }
        } else {
            None
        };
        let (mut kv, cached_tokens) = if let Some(admitted) = native_admitted {
            admitted
        } else {
            // Host-tier restore before the prefix match; the H2D is polled
            // at step boundaries, never awaited (#799). The probe must
            // outlive the match (eviction window).
            let _restored_hold = match host_restore.as_mut() {
                Some(state) if prefix_cache_enabled => {
                    match state.poll_front(offload.map(|o| &o.engine), pool, &req) {
                        offload::HostRestoreOutcome::Ready(probe) => probe,
                        offload::HostRestoreOutcome::Park => {
                            pending.push_front(req);
                            break; // head-of-line wait: retry next step boundary
                        }
                    }
                }
                _ => None,
            };
            let mut kv = if native_mtp_prefill {
                let cache_salt = super::native_mtp_cache_salt(&req.prompt_tokens);
                pool.new_request_with_cache_salt(
                    req.prompt_tokens.clone(),
                    req.max_tokens,
                    Some(&cache_salt),
                    None,
                )
            } else {
                pool.new_request(req.prompt_tokens.clone(), req.max_tokens, None)
            };
            let cached_tokens = if prefix_cache_enabled {
                match kv.match_and_add_prefix(pool) {
                    Ok(cached) => cached,
                    Err(err) => {
                        // The request is already out of `pending` and never
                        // reaches a slot, so fail it explicitly before the
                        // engine-fatal invariant error propagates.
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
            (kv, cached_tokens)
        };
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
        if let Some(plan) = native_anchor.filter(|plan| plan.replay_to_client) {
            req.prompt_tokens.push(plan.token);
        }
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens: client_prompt_tokens,
            cached_tokens,
        });
        if let Some(plan) = native_anchor {
            let replay_failed = plan.replay_to_client
                && plan.emitted_by_prefill
                && req
                    .token_tx
                    .send(TokenEvent::Token {
                        id: plan.token,
                        logprob: None,
                    })
                    .is_err();
            let finish_reason = native_anchor_finish_reason(plan, req.max_tokens);
            if replay_failed || finish_reason.is_some() {
                if let Some(finish_reason) = finish_reason
                    && !req.token_tx.is_closed()
                {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason,
                        prompt_tokens: client_prompt_tokens,
                        completion_tokens: 1,
                    });
                }
                kv.release()?;
                continue;
            }
        }
        let mut state = Glm52SlotState::new(
            req.prompt_tokens.clone(),
            req.max_tokens,
            req.params.ignore_eos,
            cached_tokens,
        );
        if let Some(handoff) = native_handoff {
            anyhow::ensure!(
                cached_tokens == handoff.committed_len,
                "native-MTP P/D admitted {} cached tokens, expected {}",
                cached_tokens,
                handoff.committed_len
            );
            if native_anchor.is_some_and(|plan| plan.replay_to_client) {
                state.seed_native_pd_replayed_anchor();
            } else {
                state.seed_native_pd_anchor();
            }
            state.set_drafts(handoff.draft_tokens.to_vec(), crate::mtp::GLM52_MTP_DRAFTS);
            log::info!(
                "GLM5.2 native P/D admitted: rank={rank} slot={slot} \
                 committed_len={} drafts={} first_step=verify",
                handoff.committed_len,
                handoff.draft_tokens.len()
            );
        }
        if drafter_enabled {
            pending_resets.push(slot);
        }
        anyhow::ensure!(
            kv.lifetime_blocks() == need_blocks,
            "GLM5.2 admission budget drift: planned {need_blocks} blocks, RequestKv owns \
             lifetime capacity for {}",
            kv.lifetime_blocks()
        );
        slots[slot] = Some(ActiveRequest {
            req,
            state,
            client_prompt_tokens,
            kv,
        });
        committed += need_blocks;
    }
    Ok(())
}

fn native_anchor_finish_reason(
    plan: offload::NativeAnchorPlan,
    max_tokens: usize,
) -> Option<openinfer_core::engine::FinishReason> {
    if !plan.emitted_by_prefill {
        // P consumed EOS without exposing it as a token. This is terminal for
        // both router replay and manual handoffs that already carry the anchor.
        Some(openinfer_core::engine::FinishReason::Stop)
    } else if plan.replay_to_client && max_tokens == 1 {
        Some(openinfer_core::engine::FinishReason::Length)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use openinfer_core::engine::FinishReason;
    use openinfer_sample::SamplingParams;

    use super::*;
    use crate::scheduler::testkit::EOS;
    use crate::scheduler::testkit::request;
    use crate::scheduler::testkit::sampled;

    #[test]
    fn malformed_sampling_params_die_at_intake() {
        // Values the sampler kernel would reject with an `ensure!` — which
        // past intake means a failed step and a fatal engine exit.
        let cases = [
            openinfer_sample::SamplingParams {
                top_p: 0.0,
                ..sampled(0.8)
            },
            openinfer_sample::SamplingParams {
                top_p: 1.5,
                ..sampled(0.8)
            },
            openinfer_sample::SamplingParams {
                top_p: f32::NAN,
                ..sampled(0.8)
            },
            sampled(f32::INFINITY),
            sampled(f32::NAN),
            openinfer_sample::SamplingParams {
                min_p: 1.0,
                ..sampled(0.8)
            },
            openinfer_sample::SamplingParams {
                min_p: -0.1,
                ..sampled(0.8)
            },
        ];
        for params in cases {
            let req = request(vec![10], params, 4);
            assert!(
                validate_request(&req, 4096, false, false).is_err(),
                "params must be rejected at intake: {params:?}"
            );
        }
        // The greedy path never reaches the sampler: out-of-range values that
        // ride a greedy request stay accepted (temperature 0 ignores top_p).
        let req = request(
            vec![10],
            openinfer_sample::SamplingParams {
                top_p: 0.0,
                ..Default::default()
            },
            4,
        );
        assert!(validate_request(&req, 4096, false, false).is_ok());
    }

    #[test]
    fn lifetime_blocks_counts_the_dangling_token() {
        // 64 prompt + 1 max_tokens: the generated token is appended to the
        // sequence (dangling) and provisions page 2 even though its KV is
        // never written.
        assert_eq!(lifetime_blocks(64, 1), 2);
        assert_eq!(lifetime_blocks(63, 1), 1);
        assert_eq!(lifetime_blocks(64, 64), 2);
        assert_eq!(lifetime_blocks(64, 65), 3);
    }

    #[test]
    fn native_pd_admission_counts_the_internal_anchor_position() {
        let manual = request(vec![10; PAGE], SamplingParams::default(), PAGE);
        let manual_anchor = offload::NativeAnchorPlan {
            token: 10,
            replay_to_client: false,
            emitted_by_prefill: true,
        };
        assert_eq!(
            admission_lifetime_blocks(&manual, Some(manual_anchor)).unwrap(),
            3,
            "manual v2 needs one internal output position beyond the client budget"
        );

        let router = request(vec![10; PAGE], SamplingParams::default(), PAGE);
        let router_anchor = offload::NativeAnchorPlan {
            token: 11,
            replay_to_client: true,
            emitted_by_prefill: true,
        };
        assert_eq!(
            admission_lifetime_blocks(&router, Some(router_anchor)).unwrap(),
            3,
            "router replay appends the anchor to the KV input shape"
        );

        assert_eq!(
            admission_lifetime_blocks(&manual, None).unwrap(),
            2,
            "ordinary requests retain their existing lifetime geometry"
        );
    }

    #[test]
    fn prefill_only_accepts_exactly_one_output_token() {
        let one = request(vec![10, 11], SamplingParams::default(), 1);
        assert!(validate_request(&one, 4096, true, false).is_ok());

        let many = request(vec![10, 11], SamplingParams::default(), 2);
        let error =
            validate_request(&many, 4096, true, false).expect_err("decode must be rejected");
        assert!(error.contains("requires max_tokens=1"), "{error}");
    }

    #[test]
    fn native_mtp_prefill_reserves_the_fixed_proposal_positions() {
        let fits = request(vec![10; 4092], SamplingParams::default(), 1);
        assert!(validate_request(&fits, 4096, true, true).is_ok());

        let overflows = request(vec![10; 4093], SamplingParams::default(), 1);
        let error = validate_request(&overflows, 4096, true, true)
            .expect_err("fixed MTP proposal must fit inside the context cap");
        assert!(
            error.contains("4 positions of proposal headroom"),
            "{error}"
        );

        assert!(
            validate_request(&overflows, 4096, true, false).is_ok(),
            "plain TP4 prefill does not execute the native-MTP proposal loop"
        );
    }

    #[test]
    fn native_pd_restore_error_rejects_only_the_request() {
        let mut req = request(vec![10], SamplingParams::default(), 1);
        let (token_tx, mut token_rx) = openinfer_core::engine::TokenSink::standalone();
        req.token_tx = token_tx;
        let mut state = offload::NativePdState::new(1);

        reject_native_pd_error(&mut state, 0, &req, &anyhow::anyhow!("invalid handoff"));

        assert!(matches!(
            token_rx.try_recv().map(|(_, event)| event),
            Ok(TokenEvent::Scheduled { .. })
        ));
        assert!(matches!(
            token_rx.try_recv().map(|(_, event)| event),
            Ok(TokenEvent::Rejected { message, .. })
                if message.contains("invalid handoff")
        ));
    }

    #[test]
    fn suppressed_eos_stops_router_and_manual_native_handoffs() {
        let manual_eos = offload::NativeAnchorPlan {
            token: EOS[0],
            replay_to_client: false,
            emitted_by_prefill: false,
        };
        assert_eq!(
            native_anchor_finish_reason(manual_eos, 8),
            Some(FinishReason::Stop)
        );

        let router_eos = offload::NativeAnchorPlan {
            replay_to_client: true,
            ..manual_eos
        };
        assert_eq!(
            native_anchor_finish_reason(router_eos, 8),
            Some(FinishReason::Stop)
        );

        let visible_manual = offload::NativeAnchorPlan {
            emitted_by_prefill: true,
            ..manual_eos
        };
        assert_eq!(native_anchor_finish_reason(visible_manual, 1), None);

        let visible_router = offload::NativeAnchorPlan {
            replay_to_client: true,
            ..visible_manual
        };
        assert_eq!(
            native_anchor_finish_reason(visible_router, 1),
            Some(FinishReason::Length)
        );
        assert_eq!(native_anchor_finish_reason(visible_router, 8), None);
    }
}
