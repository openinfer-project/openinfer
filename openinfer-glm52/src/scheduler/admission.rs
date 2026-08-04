//! Request validation and slot placement for one rank: [`validate_request`]
//! fast-rejects at the door (past it, a bad value only surfaces inside a
//! collective and tears the engine down), then [`admit_from_queue`] fills the
//! rank's free slots from its own FIFO queue at step boundaries under the
//! full-lifetime KV budget. Requests arrive pre-bound to this rank — the
//! `EngineHandle` routes by `data_parallel_rank` (the vLLM frontend's DP
//! choice) and least-load-places unbound ones — so admission never moves a
//! request and metrics/KV ownership agree with the frontend's engine index.

use std::collections::VecDeque;
use std::sync::Arc;

use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::unix_now_s;
use openinfer_kv_store::BlockPool;
use openinfer_kv_store::KvPrefix;
use openinfer_kv_store::KvStore;
use openinfer_kv_store::LoadReservation;
use openinfer_kv_store::RequestKv;
use openinfer_kv_store::SaveCursor;

use super::ActiveRequest;
use super::PAGE;
use super::RankSlots;
use super::offload::Resolved;
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
        && req.prompt_tokens.len() + crate::mtp::glm52_mtp_draft_len() - 1 > max_model_len
    {
        return Err(format!(
            "GLM5.2 native-MTP prefill requires {} positions of proposal headroom: \
             prompt {} exceeds max_model_len {max_model_len}",
            crate::mtp::glm52_mtp_draft_len() - 1,
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

pub(super) fn admit_from_queue(
    rank: usize,
    pending: &mut VecDeque<Resolved>,
    slots: &mut RankSlots,
    pool: &Arc<BlockPool>,
    usable_blocks: usize,
    store: &KvStore,
    prefix_cache_enabled: bool,
    drafter_enabled: bool,
    native_mtp_prefill: bool,
    pending_resets: &mut Vec<usize>,
) -> anyhow::Result<()> {
    let mut committed: usize = slots
        .iter()
        .flatten()
        .map(|active| active.kv.lifetime_blocks())
        .sum();
    // Pages pinned by in-flight release saves are physically unallocatable
    // until their D2H lands. Hide them from the rank's full-lifetime budget
    // so admission defers instead of promising pages a later schedule cannot
    // get (which would fail the whole engine).
    let usable = usable_blocks.saturating_sub(store.pinned_blocks(rank));

    // Admission fills only the configured slot count; the fixed array's tail
    // beyond `glm52_decode_slots()` stays permanently empty. The queue holds
    // only RESOLVED intakes — restore waiting happened off-thread, so a
    // front here is never parked on storage.
    while let Some(slot) = slots[..crate::model::glm52_decode_slots()]
        .iter()
        .position(Option::is_none)
    {
        let Some(front) = pending.front() else {
            break;
        };
        // Drop a disconnected FIFO front before it can block valid work
        // behind an admission budget it will never consume (any resolved
        // state — a built KV, a prefix hold — releases via RAII).
        let front_req = match front {
            Resolved::Plain { req, .. }
            | Resolved::Native { req, .. }
            | Resolved::Failed { req, .. } => req,
        };
        if front_req.token_tx.is_closed() {
            drop(pending.pop_front());
            continue;
        }
        if matches!(front, Resolved::Failed { .. }) {
            let Some(Resolved::Failed { req, message }) = pending.pop_front() else {
                unreachable!("front matched Failed");
            };
            reject(&req, message);
            continue;
        }

        // Full-lifetime budget, honor-or-reject. `usable` accounts for the
        // block classes the scheduler knows about; the allocator is the
        // final authority, so add back pages held by active requests and by
        // the front's own resolution (its restored blocks / built KV — and,
        // for a Native, its lifetime headroom claim — are already out of
        // the free pool) and defer if the physical lifetime budget is
        // smaller.
        let (need_blocks, front_held) = match front {
            Resolved::Plain { req, prefix } => match admission_lifetime_blocks(req, None) {
                Ok(blocks) => (blocks, prefix.hit_tokens() / PAGE),
                Err(err) => {
                    let Some(resolved) = pending.pop_front() else {
                        unreachable!("front exists");
                    };
                    reject(&resolved.into_request(), format!("{err:#}"));
                    continue;
                }
            },
            Resolved::Native {
                kv, reservation, ..
            } => (
                kv.lifetime_blocks(),
                kv.resident_blocks() + reservation.as_ref().map_or(0, LoadReservation::len),
            ),
            Resolved::Failed { .. } => unreachable!("handled above"),
        };
        let active_resident: usize = slots
            .iter()
            .flatten()
            .map(|active| active.kv.resident_blocks())
            .sum();
        let physical_usable = pool
            .available_blocks()
            .saturating_add(active_resident)
            .saturating_add(front_held);
        if committed + need_blocks > usable.min(physical_usable) {
            // The budget credits only the FRONT's hold; holds pinned by
            // requests queued BEHIND it shrink the free pool without any
            // release path of their own (they release at their admission,
            // which the stuck front blocks) — with enough concurrent
            // large-hit resolutions that's a permanent stall. A hold is an
            // anti-eviction pin, not correctness: shed the rearmost queued
            // one and retry — its blocks fall to the inactive (evictable,
            // still matchable) pool, which `available_blocks` counts.
            // Bounded: each shed permanently clears one hold. Built Native
            // KVs cannot shed (the pages are the request's own assignment).
            let mut shed = false;
            for entry in pending.iter_mut().skip(1).rev() {
                if let Resolved::Plain { prefix, .. } = entry {
                    if prefix.hit_tokens() > 0 {
                        *prefix = KvPrefix::none();
                        shed = true;
                        break;
                    }
                }
            }
            if shed {
                continue;
            }
            // Nothing left to shed. A resident Native queued behind the
            // stalled front holds pages that are its own KV assignment — it
            // cannot shed them and cannot re-resolve — and with no active
            // slots there is no retirement to free them either. Bounded
            // FIFO exception: a resident restore whose own admission fits
            // the current budget may bypass a budget-stalled front, because
            // admitting it is the only transition that ever frees its pages.
            let bypass = pending.iter().enumerate().skip(1).find_map(|(idx, entry)| {
                let Resolved::Native {
                    req,
                    kv,
                    reservation,
                    ..
                } = entry
                else {
                    return None;
                };
                if req.token_tx.is_closed() {
                    return None;
                }
                let physical = pool
                    .available_blocks()
                    .saturating_add(active_resident)
                    .saturating_add(kv.resident_blocks())
                    .saturating_add(reservation.as_ref().map_or(0, LoadReservation::len));
                (committed + kv.lifetime_blocks() <= usable.min(physical)).then_some(idx)
            });
            let Some(idx) = bypass else {
                // No queued Native fits the logical budget either — a
                // transient defer, never a deadlock: every queued Native
                // holds its FULL lifetime (resident restore + headroom
                // claim), so restores can never jointly exhaust the pool
                // while each still needs output headroom. A restore whose
                // lifetime could not be claimed failed at resolution
                // instead of reaching this queue.
                break;
            };
            let Some(Resolved::Native {
                req,
                kv,
                cached_tokens,
                handoff,
                plan,
                reservation,
            }) = pending.remove(idx)
            else {
                unreachable!("bypass index selected a Native entry");
            };
            let need_blocks = kv.lifetime_blocks();
            admit_native(
                rank,
                slot,
                req,
                kv,
                reservation,
                cached_tokens,
                handoff,
                plan,
                need_blocks,
                drafter_enabled,
                pool,
                slots,
                pending_resets,
                &mut committed,
            )?;
            continue;
        }

        match pending.pop_front().expect("checked non-empty") {
            Resolved::Failed { .. } => unreachable!("handled above"),
            Resolved::Plain { mut req, prefix } => {
                let client_prompt_tokens = req.prompt_tokens.len();
                let mut kv = if native_mtp_prefill {
                    pool.new_request_with_cache_salt(
                        req.prompt_tokens.clone(),
                        req.max_tokens,
                        Some(super::native_mtp_cache_salt()),
                        None,
                    )
                } else {
                    pool.new_request(req.prompt_tokens.clone(), req.max_tokens, None)
                };
                let cached_tokens = if prefix_cache_enabled {
                    match kv.match_and_add_prefix(pool) {
                        Ok(cached) => cached,
                        Err(err) => {
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
                // The resolve's anti-eviction hold has served its purpose:
                // the match above re-pinned whatever it restored.
                drop(prefix);
                let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
                let _ = req.token_tx.send(TokenEvent::Scheduled {
                    queued_at_unix_s,
                    scheduled_at_unix_s: unix_now_s(),
                    prompt_tokens: client_prompt_tokens,
                    cached_tokens,
                });
                let state = Glm52SlotState::new(
                    req.prompt_tokens.clone(),
                    req.max_tokens,
                    req.params.ignore_eos,
                    cached_tokens,
                );
                if drafter_enabled {
                    pending_resets.push(slot);
                }
                anyhow::ensure!(
                    kv.lifetime_blocks() == need_blocks,
                    "GLM5.2 admission budget drift: planned {need_blocks} blocks, RequestKv \
                     owns lifetime capacity for {}",
                    kv.lifetime_blocks()
                );
                // Slot handoff: the request's unallocated lifetime remainder
                // moves under the pool's active-headroom debt, so resolver
                // allocations can never strand the pages this admission just
                // promised (a stranded page fails the request's next
                // schedule, which is engine-fatal).
                pool.assume_active_headroom(&mut kv, None);
                let _ = &mut req;
                slots[slot] = Some(ActiveRequest {
                    req,
                    state,
                    client_prompt_tokens,
                    kv,
                    save_cursor: SaveCursor::new(),
                });
                committed += need_blocks;
            }
            Resolved::Native {
                req,
                kv,
                cached_tokens,
                handoff,
                plan,
                reservation,
            } => {
                admit_native(
                    rank,
                    slot,
                    req,
                    kv,
                    reservation,
                    cached_tokens,
                    handoff,
                    plan,
                    need_blocks,
                    drafter_enabled,
                    pool,
                    slots,
                    pending_resets,
                    &mut committed,
                )?;
            }
        }
    }
    Ok(())
}

/// Slot a resolver-built native P/D intake — anchor replay, anchored-finish
/// short circuit, state seeding, and the budget-drift check — shared by
/// front admission and the budget-bypass path. An anchored finish leaves the
/// slot empty and `committed` untouched.
#[allow(clippy::too_many_arguments)]
fn admit_native(
    rank: usize,
    slot: usize,
    mut req: GenerateRequest,
    mut kv: Box<RequestKv>,
    reservation: Option<LoadReservation>,
    cached_tokens: usize,
    handoff: offload::NativeMtpHandoff,
    plan: offload::NativeAnchorPlan,
    need_blocks: usize,
    drafter_enabled: bool,
    pool: &Arc<BlockPool>,
    slots: &mut RankSlots,
    pending_resets: &mut Vec<usize>,
    committed: &mut usize,
) -> anyhow::Result<()> {
    let client_prompt_tokens = req.prompt_tokens.len();
    anyhow::ensure!(
        cached_tokens == handoff.committed_len,
        "native-MTP P/D admitted {} cached tokens, expected {}",
        cached_tokens,
        handoff.committed_len
    );
    if plan.replay_to_client {
        req.prompt_tokens.push(plan.token);
    }
    let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s,
        scheduled_at_unix_s: unix_now_s(),
        prompt_tokens: client_prompt_tokens,
        cached_tokens,
    });
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
        // No slot forms: the claim (still held) and the KV both release
        // plainly — there is no future page to owe the pool for.
        if let Err(err) = kv.release() {
            log::warn!("GLM5.2 native P/D anchored-finish release: {err:#}");
        }
        return Ok(());
    }
    let mut state = Glm52SlotState::new(
        req.prompt_tokens.clone(),
        req.max_tokens,
        req.params.ignore_eos,
        cached_tokens,
    );
    if plan.replay_to_client {
        state.seed_native_pd_replayed_anchor();
    } else {
        state.seed_native_pd_anchor();
    }
    state.set_drafts(
        handoff.draft_tokens.to_vec(),
        crate::mtp::glm52_mtp_draft_len(),
    );
    log::info!(
        "GLM5.2 native P/D admitted: rank={rank} slot={slot} \
         committed_len={} drafts={} first_step=verify",
        handoff.committed_len,
        handoff.draft_tokens.len()
    );
    if drafter_enabled {
        pending_resets.push(slot);
    }
    anyhow::ensure!(
        kv.lifetime_blocks() == need_blocks,
        "GLM5.2 native admission budget drift: planned {need_blocks}, KV owns {}",
        kv.lifetime_blocks()
    );
    // Slot handoff: the request's unallocated lifetime remainder moves under
    // the pool's active-headroom debt, and the resolve-time claim — whose
    // pages ARE that remainder — dissolves in the same atomic section, so
    // no resolver can take them in between. The rank's committed-lifetime
    // bookkeeping stays advisory; the pool's ledger is what a resolver
    // allocation actually observes.
    pool.assume_active_headroom(&mut kv, reservation);
    slots[slot] = Some(ActiveRequest {
        req,
        state,
        client_prompt_tokens,
        kv: *kv,
        save_cursor: SaveCursor::new(),
    });
    *committed += need_blocks;
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
        // Headroom tracks the configured draft span, not the compile ceiling.
        let headroom = crate::mtp::glm52_mtp_draft_len() - 1;
        let fits = request(vec![10; 4096 - headroom], SamplingParams::default(), 1);
        assert!(validate_request(&fits, 4096, true, true).is_ok());

        let overflows = request(vec![10; 4096 - headroom + 1], SamplingParams::default(), 1);
        let error = validate_request(&overflows, 4096, true, true)
            .expect_err("fixed MTP proposal must fit inside the context cap");
        assert!(
            error.contains(&format!("{headroom} positions of proposal headroom")),
            "{error}"
        );

        assert!(
            validate_request(&overflows, 4096, true, false).is_ok(),
            "plain TP4 prefill does not execute the native-MTP proposal loop"
        );
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
