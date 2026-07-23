//! Request intake and slot placement: [`validate_request`] fast-rejects at
//! the door (past it, a bad value only surfaces inside a collective and tears
//! the engine down), then binds every request to one rank queue. HTTP requests
//! arrive with the vLLM frontend's DP choice; direct requests are placed once
//! using the same waiting-weighted least-load policy. [`admit_from_queue`]
//! then fills each rank's free slots at step boundaries under the
//! full-lifetime KV budget.

use std::collections::VecDeque;

use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::unix_now_s;
use openinfer_kv_cache::BlockPool;

use super::ActiveRequest;
use super::PAGE;
use super::RankSlots;
use super::offload::VllmAdmitOutcome;
use super::offload::VllmPdState;
use super::offload::{self};
use super::slot::Glm52SlotState;
use crate::runner::Glm52Worker;

fn validate_request(req: &GenerateRequest, max_model_len: usize) -> Result<(), String> {
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
    if last_position > max_model_len {
        return Err(format!(
            "GLM5.2 context cap: prompt {} + max_tokens {} exceeds max_model_len {max_model_len}",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    // Mirror the sampler kernel's parameter ensures HERE: past intake a bad
    // value only surfaces as a failed step, and a failed step tears the whole
    // EP8 engine down (`fail_step`) — user input must be rejected at the
    // door, never inside a collective.
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

/// Pick a rank for a direct, unbound request. Waiting carries the same 4x
/// weight as vLLM's DP load balancer; ties go to the lowest rank. Frontend
/// requests bypass this function because their selected engine index is the
/// rank assignment.
fn least_loaded_rank(running: &[usize], pending: &[VecDeque<GenerateRequest>]) -> usize {
    assert_eq!(running.len(), pending.len());
    running
        .iter()
        .enumerate()
        .min_by_key(|&(rank, &running)| (running + pending[rank].len() * 4, rank))
        .map(|(rank, _)| rank)
        .expect("GLM5.2 must expose at least one logical rank")
}

fn reject(req: &GenerateRequest, message: String) {
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

/// Fast-reject invalid requests at intake (Scheduled → Rejected), otherwise
/// bind the request to exactly one rank queue. The binding is permanent so
/// frontend `engine_index`, metrics labels, and actual KV ownership agree.
pub(super) fn intake(
    req: GenerateRequest,
    pending: &mut [VecDeque<GenerateRequest>],
    running: &[usize],
    max_model_len: usize,
) {
    if let Err(message) = validate_request(&req, max_model_len) {
        reject(&req, message);
        return;
    }
    let rank = match req.data_parallel_rank {
        Some(rank) if rank < pending.len() => rank,
        Some(rank) => {
            reject(
                &req,
                format!(
                    "GLM5.2 data_parallel_rank {rank} is outside 0..{}",
                    pending.len()
                ),
            );
            return;
        }
        None => least_loaded_rank(running, pending),
    };
    pending[rank].push_back(req);
}

/// Admission: fill each rank's free slots from its own FIFO queue while its
/// full-lifetime KV budget permits. The frontend already selected the rank;
/// admission must never move the request or its metrics/KV ownership diverge.
/// New requests join the lock-step at the next step boundary. An `Err` is a
/// kvbm invariant break — the caller fails the step (the affected request was
/// already answered here).
#[allow(clippy::too_many_arguments)]
pub(super) fn admit_from_queue(
    pending: &mut [VecDeque<GenerateRequest>],
    slots: &mut [RankSlots],
    pools: &[BlockPool],
    usable_blocks: &[usize],
    offload: Option<&[offload::RankOffload]>,
    vllm_pd: &mut Option<VllmPdState>,
    workers: &[Glm52Worker],
    mirrored: bool,
    prefix_cache_enabled: bool,
    dspark_enabled: bool,
    pending_resets: &mut [Vec<usize>],
    slots_changed: &mut bool,
) -> anyhow::Result<()> {
    assert_eq!(pending.len(), slots.len());
    let mut committed: Vec<usize> = slots
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
    // Pages pinned by in-flight release saves are physically unallocatable
    // until their D2H lands. Hide them from each rank's full-lifetime budget
    // so admission defers instead of promising pages a later schedule cannot
    // get (which would fail the whole engine).
    let usable: Vec<usize> = match offload {
        Some(offload) => usable_blocks
            .iter()
            .zip(offload)
            .map(|(&usable, rank)| usable.saturating_sub(rank.pinned_blocks()))
            .collect(),
        None => usable_blocks.to_vec(),
    };

    for rank in 0..slots.len() {
        while let Some(slot) = slots[rank].iter().position(Option::is_none) {
            let Some(front) = pending[rank].front() else {
                break;
            };
            let need_blocks = lifetime_blocks(front.prompt_tokens.len(), front.max_tokens);
            if committed[rank] + need_blocks > usable[rank] {
                break;
            }

            let req = pending[rank].pop_front().expect("checked non-empty");
            // The client left while the request sat in the queue — admitting
            // it would burn a slot (and whole global steps) on a dead sink.
            if req.token_tx.is_closed() {
                if let Some(pd) = vllm_pd.as_mut() {
                    pd.clear_parked(rank);
                }
                continue;
            }
            // vLLM-compat P/D admission: the full peer-prefilled prefix must
            // restore (this node never computes prompt positions), a racing
            // registration parks the request at the queue front for the next
            // step boundary, and an exhausted wait window rejects it for the
            // router to retry through the prefill peer.
            let pd_admitted = match vllm_pd.as_mut() {
                Some(pd) => {
                    let offload = offload.expect("vLLM-compat P/D requires --kv-offload");
                    // Launch validation pins vllm-compat to the EP topology
                    // (kv-offload ⇒ EP8): each rank's executor owns the only
                    // replica of its arenas, so it alone runs the fixup. A
                    // mirrored topology would need every worker here.
                    assert!(
                        !mirrored,
                        "vLLM-compat P/D admission assumes the EP topology"
                    );
                    match offload::admit_vllm_pd(
                        pd,
                        rank,
                        &offload[rank],
                        &pools[rank],
                        &req,
                        &workers[rank],
                    ) {
                        Ok(VllmAdmitOutcome::Admit { kv, cached_tokens }) => {
                            Some((*kv, cached_tokens))
                        }
                        Ok(VllmAdmitOutcome::Park) => {
                            pending[rank].push_front(req);
                            break; // head-of-line wait: retry next step boundary
                        }
                        Ok(VllmAdmitOutcome::Reject { message }) => {
                            reject(&req, message);
                            continue;
                        }
                        Ok(VllmAdmitOutcome::LocalFallback) => None,
                        Err(err) => {
                            let err = err.context("GLM5.2 P/D admission");
                            let _ = req.token_tx.send(TokenEvent::Error {
                                message: format!("{err:#}"),
                                prompt_tokens: req.prompt_tokens.len(),
                                completion_tokens: 0,
                            });
                            return Err(err);
                        }
                    }
                }
                None => None,
            };
            let (kv, cached_tokens) = if let Some(admitted) = pd_admitted {
                admitted
            } else {
                let mut kv =
                    pools[rank].new_request(req.prompt_tokens.clone(), req.max_tokens, None);
                // Host-tier restore first, so the GPU prefix match sees the union
                // of HBM-resident and freshly-restored blocks. The probe stays
                // alive across the match to close the eviction window.
                let _restored_hold = offload
                    .filter(|_| prefix_cache_enabled && vllm_pd.is_none())
                    .map(|offload| {
                        offload::restore_host_prefix(
                            &offload[rank].engine,
                            &pools[rank],
                            &req.prompt_tokens,
                        )
                    });
                let cached_tokens = if prefix_cache_enabled {
                    match kv.match_and_add_prefix(&pools[rank]) {
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
            committed[rank] += need_blocks;
            *slots_changed = true;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use openinfer_sample::SamplingParams;

    use super::*;
    use crate::scheduler::testkit::request;
    use crate::scheduler::testkit::sampled;

    #[test]
    fn malformed_sampling_params_die_at_intake() {
        // Values the sampler kernel would reject with an `ensure!` — which
        // past intake means a failed step and a whole-engine teardown.
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
                validate_request(&req, 4096).is_err(),
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
        assert!(validate_request(&req, 4096).is_ok());
    }

    #[test]
    fn intake_keeps_frontend_binding_and_load_balances_direct_requests() {
        let mut pending: Vec<VecDeque<GenerateRequest>> = (0..3).map(|_| VecDeque::new()).collect();

        let mut bound = request(vec![10], SamplingParams::default(), 4);
        bound.data_parallel_rank = Some(2);
        intake(bound, &mut pending, &[0, 0, 0], 4096);
        assert_eq!(
            pending.iter().map(VecDeque::len).collect::<Vec<_>>(),
            [0, 0, 1]
        );

        intake(
            request(vec![11], SamplingParams::default(), 4),
            &mut pending,
            &[2, 1, 2],
            4096,
        );
        assert_eq!(
            pending.iter().map(VecDeque::len).collect::<Vec<_>>(),
            [0, 1, 1]
        );

        // Rank 1's queued request adds a 4x waiting penalty, so the next
        // direct request goes to the lower-rank member of the 2/2 tie.
        intake(
            request(vec![12], SamplingParams::default(), 4),
            &mut pending,
            &[2, 1, 2],
            4096,
        );
        assert_eq!(
            pending.iter().map(VecDeque::len).collect::<Vec<_>>(),
            [1, 1, 1]
        );
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
}
