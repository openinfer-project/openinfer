use anyhow::{Context, Result};
use rand::rngs::StdRng;

use crate::executor::{
    DecodePlan, DecodeResult, DecodeStepItem, ModelExecutor, PrefillPlan, PrefillResult,
    PrefillStepItem, UnifiedPlan, UnifiedResult,
};

use super::{ActiveRequestState, PendingRequest};

pub(super) enum ExecutionPlan {
    Prefill { pending: Vec<PendingRequest> },
    Decode,
    Unified { pending: Vec<PendingRequest> },
}

pub(super) enum ExecutionArtifacts {
    Prefill {
        pending: Vec<PendingRequest>,
        result: PrefillResult,
    },
    Decode {
        result: DecodeResult,
    },
    Unified {
        pending: Vec<PendingRequest>,
        result: UnifiedResult,
    },
}

fn len_summary(values: impl Iterator<Item = usize>) -> (usize, usize, usize) {
    let mut count = 0;
    let mut total = 0;
    let mut min = usize::MAX;
    let mut max = 0;
    for value in values {
        count += 1;
        total += value;
        min = min.min(value);
        max = max.max(value);
    }
    if count == 0 {
        (0, 0, 0)
    } else {
        (total, min, max)
    }
}

pub(super) fn build_next_plan(
    have_active: bool,
    pending: Vec<PendingRequest>,
) -> Option<ExecutionPlan> {
    if !pending.is_empty() && have_active {
        Some(ExecutionPlan::Unified { pending })
    } else if !pending.is_empty() {
        Some(ExecutionPlan::Prefill { pending })
    } else if have_active {
        Some(ExecutionPlan::Decode)
    } else {
        None
    }
}

pub(super) fn execute_plan(
    executor: &mut impl ModelExecutor,
    active: &mut [ActiveRequestState],
    plan: ExecutionPlan,
    rng: &mut StdRng,
) -> Result<ExecutionArtifacts> {
    match plan {
        ExecutionPlan::Prefill { pending } => {
            let requests: Vec<PrefillStepItem> = pending
                .iter()
                .map(|r| PrefillStepItem {
                    request_id: r.request_id,
                    prompt_tokens: r.prompt_tokens.clone(),
                    params: r.params,
                    logprobs: r.logprobs,
                    echo: r.echo,
                    random_val: rand::RngExt::random(rng),
                })
                .collect();
            let any_echo = pending.iter().any(|r| r.echo);
            let (prompt_total, prompt_min, prompt_max) =
                len_summary(requests.iter().map(|r| r.prompt_tokens.len()));
            let result = executor
                .execute_prefill(PrefillPlan {
                    requests: &requests,
                    echo: any_echo,
                })
                .with_context(|| {
                    format!(
                        "prefill plan failed: requests={}, prompt_tokens_total={}, prompt_tokens_min={}, prompt_tokens_max={}, echo={any_echo}",
                        requests.len(),
                        prompt_total,
                        prompt_min,
                        prompt_max
                    )
                })?;
            Ok(ExecutionArtifacts::Prefill { pending, result })
        }
        ExecutionPlan::Decode => {
            let requests: Vec<DecodeStepItem> = active
                .iter()
                .map(|r| DecodeStepItem {
                    request_id: r.request_id,
                    token_id: r.last_token,
                    params: r.params,
                    logprobs: r.logprobs,
                    random_val: rand::RngExt::random(rng),
                })
                .collect();
            let result = executor
                .execute_decode(DecodePlan {
                    requests: &requests,
                })
                .with_context(|| format!("decode plan failed: requests={}", requests.len()))?;
            Ok(ExecutionArtifacts::Decode { result })
        }
        ExecutionPlan::Unified { pending } => {
            let prefill_requests: Vec<PrefillStepItem> = pending
                .iter()
                .map(|r| PrefillStepItem {
                    request_id: r.request_id,
                    prompt_tokens: r.prompt_tokens.clone(),
                    params: r.params,
                    logprobs: r.logprobs,
                    echo: r.echo,
                    random_val: rand::RngExt::random(rng),
                })
                .collect();
            let decode_requests: Vec<DecodeStepItem> = active
                .iter()
                .map(|r| DecodeStepItem {
                    request_id: r.request_id,
                    token_id: r.last_token,
                    params: r.params,
                    logprobs: r.logprobs,
                    random_val: rand::RngExt::random(rng),
                })
                .collect();
            let (prompt_total, prompt_min, prompt_max) =
                len_summary(prefill_requests.iter().map(|r| r.prompt_tokens.len()));
            let result = executor
                .execute_unified(UnifiedPlan {
                    prefill_requests: &prefill_requests,
                    decode_requests: &decode_requests,
                })
                .with_context(|| {
                    format!(
                        "unified plan failed: prefill_requests={}, decode_requests={}, prefill_tokens_total={}, prefill_tokens_min={}, prefill_tokens_max={}",
                        prefill_requests.len(),
                        decode_requests.len(),
                        prompt_total,
                        prompt_min,
                        prompt_max
                    )
                })?;
            Ok(ExecutionArtifacts::Unified { pending, result })
        }
    }
}
