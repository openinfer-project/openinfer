//! Method-agnostic core of **greedy** speculative decoding for Qwen3.
//!
//! Speculative decoding is an optimistic-concurrency transaction over the
//! decode loop: *propose* a span of `K` cheap draft tokens, *verify* them with a
//! single target forward over the `K + 1` span positions, *accept* the longest
//! prefix the target agrees with (plus one bonus token), then *commit* the
//! accepted KV and roll back the rejected draft KV. Only the **propose** step
//! varies between methods (n-gram lookup, DFlash draft model, EAGLE, …); verify,
//! accept, and the KV transaction are shared.
//!
//! This module owns the shared half. The draft/verify boundary is a **pure
//! token span** — a model proposer's hidden states never cross it; they stay
//! inside the proposer (see [`crate::dflash`]). DFlash is the only proposer
//! today and is kept concrete; a proposer trait is deferred until a second
//! implementation (n-gram / EAGLE) validates the shape.
//!
//! What is *not* generic yet: [`accept_greedy`] returns argmax-based acceptance,
//! which is the *greedy* rule. Sampling-correct acceptance would need the target
//! and draft distributions, touching both the verify path and the proposer — so
//! it is left until a sampling method actually lands.

use anyhow::Result;

use crate::executor::RequestId;
use openinfer_core::sampler::SamplingParams;

/// One request's verify span: the current dangling token followed by the draft
/// candidates (`token_ids[0]` is the confirmed last token, `token_ids[1..]` are
/// the `K` drafts). Token-only by construction — the proposer that produced the
/// drafts keeps any hidden state to itself.
#[derive(Clone)]
pub(crate) struct VerifyStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) params: SamplingParams,
}

impl VerifyStepItem {
    pub(crate) fn new(request_id: RequestId, token_ids: Vec<u32>, params: SamplingParams) -> Self {
        Self {
            request_id,
            token_ids,
            params,
        }
    }

    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.token_ids
    }
}

#[derive(Clone, Copy)]
pub(crate) struct VerifyPlan<'a> {
    pub requests: &'a [VerifyStepItem],
}

#[derive(Clone, Debug)]
pub(crate) struct VerifyRequestResult {
    pub request_id: RequestId,
    /// Number of draft candidates accepted before the posterior bonus.
    pub matched_draft_tokens: usize,
    /// Tokens to commit: the accepted draft prefix followed by the target's
    /// posterior token at the first mismatch (or the block-end continuation
    /// when every draft is accepted). Always `1..=K + 1` tokens, so a verify
    /// step always makes at least one token of progress. The scheduler still
    /// owns stop-token suppression before client emission.
    pub accepted_tokens: Vec<u32>,
}

pub(crate) struct VerifyResult {
    pub requests: Vec<VerifyRequestResult>,
}

/// One request's draft request: the proposer continues from `current_token`.
#[derive(Clone)]
pub(crate) struct DraftStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) current_token: u32,
    pub(crate) params: SamplingParams,
}

impl DraftStepItem {
    pub(crate) fn new(request_id: RequestId, current_token: u32, params: SamplingParams) -> Self {
        Self {
            request_id,
            current_token,
            params,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DraftPlan<'a> {
    pub requests: &'a [DraftStepItem],
}

#[derive(Clone, Debug)]
pub(crate) struct DraftRequestResult {
    pub request_id: RequestId,
    /// Verify-span tokens: current dangling token first, then draft candidates.
    pub token_ids: Vec<u32>,
}

pub(crate) struct DraftResult {
    pub requests: Vec<DraftRequestResult>,
}

/// Greedy speculative acceptance — the shared seam every method reuses.
///
/// * `proposed` — the `K` candidate tokens from the proposer.
/// * `target_argmax` — the target model's greedy token at each of the `K + 1`
///   verify positions. `target_argmax[i]` is the model's prediction *after*
///   consuming verify input `i`; `target_argmax[0]` follows the last confirmed
///   token and `target_argmax[K]` is the model's own continuation after the
///   whole candidate run.
///
/// Returns the longest accepted prefix of `proposed` followed by exactly one
/// model token (the correction at the first divergence, or the bonus
/// continuation when every candidate is accepted) — always `1..=K + 1` tokens.
///
/// # Panics
/// Panics (debug builds) if `target_argmax.len() != proposed.len() + 1`.
#[must_use]
pub(crate) fn accept_greedy(proposed: &[u32], target_argmax: &[u32]) -> Vec<u32> {
    debug_assert_eq!(
        target_argmax.len(),
        proposed.len() + 1,
        "verify must produce one greedy token per candidate plus a bonus"
    );
    let n = num_accepted(proposed, target_argmax);
    let mut committed = Vec::with_capacity(n + 1);
    committed.extend_from_slice(&proposed[..n]);
    // The model's own token at the first divergence (or the bonus continuation
    // when the whole run was accepted). `n <= proposed.len() < target_argmax.len()`
    // so this index is always valid.
    committed.push(target_argmax[n]);
    committed
}

/// Length of the accepted prefix: leading drafts whose token matches the
/// target's argmax.
fn num_accepted(proposed: &[u32], target_argmax: &[u32]) -> usize {
    let mut i = 0;
    while i < proposed.len() && proposed[i] == target_argmax[i] {
        i += 1;
    }
    i
}

/// Batched greedy acceptance over a verify forward's flattened per-position
/// argmax. `target_tokens` is the concatenation of each request's `K + 1`
/// posterior columns, in `requests` order. Each request applies the shared
/// [`accept_greedy`] over its own span.
pub(crate) fn build_verify_results(
    requests: &[VerifyStepItem],
    target_tokens: &[u32],
) -> Result<Vec<VerifyRequestResult>> {
    let mut outputs = Vec::with_capacity(requests.len());
    let mut offset = 0usize;
    for req in requests {
        let span_len = req.token_ids.len();
        anyhow::ensure!(
            span_len > 0,
            "speculative verify request {:?} has an empty verify span",
            req.request_id
        );
        let end = offset + span_len;
        anyhow::ensure!(
            end <= target_tokens.len(),
            "speculative target-token result is shorter than the verify span"
        );
        let posterior = &target_tokens[offset..end];
        // proposed = the K drafts (span minus the leading confirmed token);
        // posterior = the K + 1 argmax columns. accept_greedy ties them together.
        let accepted_tokens = accept_greedy(&req.token_ids[1..], posterior);
        outputs.push(VerifyRequestResult {
            request_id: req.request_id,
            matched_draft_tokens: accepted_tokens.len() - 1,
            accepted_tokens,
        });
        offset = end;
    }
    anyhow::ensure!(
        offset == target_tokens.len(),
        "unused speculative target-token result columns: used {offset}, total {}",
        target_tokens.len()
    );
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_full_run_plus_bonus() {
        let proposed = [10u32, 11, 12];
        let argmax = [10u32, 11, 12, 13];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![10, 11, 12, 13]);
        assert_eq!(num_accepted(&proposed, &argmax), 3);
    }

    #[test]
    fn accepts_prefix_then_correction() {
        let proposed = [10u32, 11, 99];
        let argmax = [10u32, 11, 22, 33];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![10, 11, 22]);
        assert_eq!(num_accepted(&proposed, &argmax), 2);
    }

    #[test]
    fn rejects_first_candidate_commits_one() {
        let proposed = [10u32, 11, 12];
        let argmax = [7u32, 8, 9, 10];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![7]);
        assert_eq!(num_accepted(&proposed, &argmax), 0);
    }

    #[test]
    fn empty_proposal_commits_model_token() {
        let proposed: [u32; 0] = [];
        let argmax = [42u32];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![42]);
        assert_eq!(num_accepted(&proposed, &argmax), 0);
    }

    #[test]
    fn always_commits_at_least_one_token() {
        let proposed = [1u32, 2];
        let argmax = [9u32, 9, 9];
        assert!(!accept_greedy(&proposed, &argmax).is_empty());
    }

    #[test]
    fn batched_accepts_matching_prefix_plus_posterior_bonus() {
        let req = VerifyStepItem::new(
            RequestId(7),
            vec![10, 11, 12, 13],
            SamplingParams::default(),
        );
        let results = build_verify_results(&[req], &[11, 12, 99, 100]).expect("verify results");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, RequestId(7));
        assert_eq!(results[0].matched_draft_tokens, 2);
        assert_eq!(results[0].accepted_tokens, vec![11, 12, 99]);
    }

    #[test]
    fn batched_all_match_still_adds_block_end_posterior() {
        let req = VerifyStepItem::new(RequestId(8), vec![20, 21, 22], SamplingParams::default());
        let results = build_verify_results(&[req], &[21, 22, 23]).expect("verify results");
        assert_eq!(results[0].matched_draft_tokens, 2);
        assert_eq!(results[0].accepted_tokens, vec![21, 22, 23]);
    }

    #[test]
    fn batched_multi_request_splits_columns_by_span() {
        let a = VerifyStepItem::new(RequestId(1), vec![5, 6], SamplingParams::default());
        let b = VerifyStepItem::new(RequestId(2), vec![7, 8, 9], SamplingParams::default());
        // a: posterior [6, 100] -> accept draft 6, bonus 100. b: posterior [8, 77, 0]
        // -> accept draft 8, correction 77.
        let results = build_verify_results(&[a, b], &[6, 100, 8, 77, 0]).expect("verify results");
        assert_eq!(results[0].accepted_tokens, vec![6, 100]);
        assert_eq!(results[1].accepted_tokens, vec![8, 77]);
    }
}
