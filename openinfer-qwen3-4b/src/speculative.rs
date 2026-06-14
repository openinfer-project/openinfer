use anyhow::Result;

use crate::executor::RequestId;
use openinfer_core::sampler::SamplingParams;

pub struct VerifyPlan<'a> {
    pub requests: &'a [VerifyStepItem],
}

pub struct DraftPlan<'a> {
    pub requests: &'a [DraftStepItem],
}

#[derive(Clone)]
pub struct VerifyStepItem {
    pub(crate) request_id: RequestId,
    /// Verify-span token ids: current dangling token first, then draft
    /// candidates. DFlash uses a fixed 16-token span.
    pub(crate) token_ids: Vec<u32>,
    pub(crate) params: SamplingParams,
    pub(crate) lora_adapter: Option<String>,
}

impl VerifyStepItem {
    pub fn new(request_id: RequestId, token_ids: Vec<u32>, params: SamplingParams) -> Self {
        Self {
            request_id,
            token_ids,
            params,
            lora_adapter: None,
        }
    }

    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.token_ids
    }
}

#[derive(Clone, Debug)]
pub struct VerifyRequestResult {
    pub request_id: RequestId,
    /// Number of draft candidates accepted before the posterior bonus.
    pub matched_draft_tokens: usize,
    /// Tokens committed to request state: accepted draft prefix followed by
    /// the target posterior token at the first mismatch (or at the block end).
    /// The scheduler still owns stop-token suppression before client emission.
    pub accepted_tokens: Vec<u32>,
    /// Target-selected posterior token for each position in the verify span.
    pub target_tokens: Vec<u32>,
}

pub struct VerifyResult {
    pub requests: Vec<VerifyRequestResult>,
}

#[derive(Clone)]
pub struct DraftStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) current_token: u32,
    pub(crate) params: SamplingParams,
}

impl DraftStepItem {
    pub fn new(request_id: RequestId, current_token: u32, params: SamplingParams) -> Self {
        Self {
            request_id,
            current_token,
            params,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DraftRequestResult {
    pub request_id: RequestId,
    /// Verify-span tokens: current dangling token first, then draft candidates.
    pub token_ids: Vec<u32>,
}

pub struct DraftResult {
    pub requests: Vec<DraftRequestResult>,
}

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

        let mut matched = 0usize;
        while matched + 1 < span_len && req.token_ids[matched + 1] == posterior[matched] {
            matched += 1;
        }

        let mut accepted_tokens = req.token_ids[1..1 + matched].to_vec();
        accepted_tokens.push(posterior[matched]);
        outputs.push(VerifyRequestResult {
            request_id: req.request_id,
            matched_draft_tokens: matched,
            accepted_tokens,
            target_tokens: posterior.to_vec(),
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
    use super::{VerifyStepItem, build_verify_results};
    use crate::executor::RequestId;
    use openinfer_core::sampler::SamplingParams;

    #[test]
    fn speculative_verify_accepts_matching_prefix_plus_posterior_bonus() {
        let req = VerifyStepItem::new(
            RequestId::new(7),
            vec![10, 11, 12, 13],
            SamplingParams::default(),
        );

        let results = build_verify_results(&[req], &[11, 12, 99, 100]).expect("verify results");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, RequestId::new(7));
        assert_eq!(results[0].matched_draft_tokens, 2);
        assert_eq!(results[0].accepted_tokens, vec![11, 12, 99]);
        assert_eq!(results[0].target_tokens, vec![11, 12, 99, 100]);
    }

    #[test]
    fn speculative_verify_all_match_still_adds_block_end_posterior() {
        let req = VerifyStepItem::new(
            RequestId::new(8),
            vec![20, 21, 22],
            SamplingParams::default(),
        );

        let results = build_verify_results(&[req], &[21, 22, 23]).expect("verify results");

        assert_eq!(results[0].matched_draft_tokens, 2);
        assert_eq!(results[0].accepted_tokens, vec![21, 22, 23]);
    }
}
