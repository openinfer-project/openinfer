//! Request intake and slot placement: [`validate_request`] fast-rejects at
//! the door (past it, a bad value only surfaces inside a collective and tears
//! the engine down), then binds every request to one rank queue. HTTP requests
//! arrive with the vLLM frontend's DP choice; direct requests are placed once
//! using the same waiting-weighted least-load policy.

use std::collections::VecDeque;

use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::testkit::{request, sampled};
    use openinfer_sample::SamplingParams;

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
