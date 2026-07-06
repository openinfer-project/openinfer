//! Request intake and slot placement: [`validate_request`] fast-rejects at
//! the door (past it, a bad value only surfaces inside a collective and tears
//! the engine down), [`admission_target`] picks the least-loaded rank with
//! pool budget for the request's full [`lifetime_blocks`] reservation.

use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};

use crate::model::GLM52_MAX_BATCH_PER_RANK;

use super::PAGE;

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

/// Where the next queued request goes: among the ranks with a free slot AND
/// enough unreserved pool pages for the request's full lifetime, the
/// least-loaded one (ties → lowest rank id), its lowest free slot. `None`
/// when no rank can take it — the queue holds (a request that fits the pool
/// geometry always fits an EMPTY rank, so FCFS deferral never livelocks).
/// Least-loaded-first keeps occupancy balanced, which keeps the fleet in the
/// cheap 1-row bucket until concurrency exceeds the rank count.
///
/// `committed[rank]` = Σ active requests' [`lifetime_blocks`];
/// `usable[rank]` = pool blocks minus the reserved padding page. The
/// reservation is conservative: prefix-cache hits share pages between
/// requests, but each holder reserves them in full — over-reserving can only
/// defer admission, never strand a decode.
pub(super) fn admission_target(
    occupied: &[[bool; GLM52_MAX_BATCH_PER_RANK]],
    committed: &[usize],
    usable: &[usize],
    need_blocks: usize,
) -> Option<(usize, usize)> {
    let (rank, row) = occupied
        .iter()
        .enumerate()
        .filter(|(rank, row)| {
            committed[*rank] + need_blocks <= usable[*rank] && row.iter().any(|&o| !o)
        })
        .min_by_key(|(rank, row)| (row.iter().filter(|&&o| o).count(), *rank))?;
    let slot = row.iter().position(|&o| !o)?;
    Some((rank, slot))
}

/// Fast-reject invalid requests at intake (Scheduled → Rejected, the same
/// event order the bs=1 coordinator emitted); valid ones queue for a rank.
pub(super) fn intake(
    req: GenerateRequest,
    pending: &mut std::collections::VecDeque<GenerateRequest>,
    max_model_len: usize,
) {
    if let Err(message) = validate_request(&req, max_model_len) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::testkit::{request, sampled};

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

    fn occ(counts: &[usize]) -> Vec<[bool; GLM52_MAX_BATCH_PER_RANK]> {
        counts
            .iter()
            .map(|&c| std::array::from_fn(|slot| slot < c))
            .collect()
    }

    /// `admission_target` with an unconstrained pool budget — the pure
    /// occupancy-placement behavior.
    fn target(occupied: &[[bool; GLM52_MAX_BATCH_PER_RANK]]) -> Option<(usize, usize)> {
        let committed = vec![0usize; occupied.len()];
        let usable = vec![usize::MAX; occupied.len()];
        admission_target(occupied, &committed, &usable, 1)
    }

    #[test]
    fn admission_prefers_least_loaded_rank_then_lowest_slot() {
        // Empty fleet: rank 0, slot 0.
        assert_eq!(target(&occ(&[0, 0, 0])), Some((0, 0)));
        // Rank 1 is the least loaded.
        assert_eq!(target(&occ(&[2, 1, 2])), Some((1, 1)));
        // Tie between ranks 0 and 2 → lowest rank id.
        assert_eq!(target(&occ(&[1, 2, 1])), Some((0, 1)));
        // A hole in the middle of a rank's slots is reused first.
        let mut holey = occ(&[3, 3]);
        holey[1][1] = false;
        assert_eq!(target(&holey), Some((1, 1)));
        // Full fleet: no target.
        assert_eq!(target(&occ(&[GLM52_MAX_BATCH_PER_RANK; 2])), None);
    }

    #[test]
    fn admission_respects_the_pool_budget() {
        // Rank 0 has free slots but its pool is fully reserved; rank 1 (more
        // loaded but with budget) takes the request. No rank fits → defer.
        let occupied = occ(&[1, 2]);
        assert_eq!(
            admission_target(&occupied, &[90, 40], &[100, 100], 20),
            Some((1, 2))
        );
        assert_eq!(
            admission_target(&occupied, &[90, 90], &[100, 100], 20),
            None
        );
        // Exact fit admits.
        assert_eq!(
            admission_target(&occupied, &[80, 90], &[100, 100], 20),
            Some((0, 1))
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
