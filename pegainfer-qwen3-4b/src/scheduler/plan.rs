use std::collections::BTreeMap;

use anyhow::Result;
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

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum LoraGroupKey {
    Base,
    Adapter(String),
}

impl LoraGroupKey {
    fn from_option(adapter: &Option<String>) -> Self {
        match adapter {
            Some(adapter) => Self::Adapter(adapter.clone()),
            None => Self::Base,
        }
    }

    fn as_deref(&self) -> Option<&str> {
        match self {
            Self::Base => None,
            Self::Adapter(adapter) => Some(adapter.as_str()),
        }
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
            let mut result = PrefillResult {
                requests: Vec::with_capacity(pending.len()),
            };
            for (key, group) in group_pending_indices(&pending) {
                executor.activate_lora_adapter(key.as_deref())?;
                let requests = build_prefill_items(&pending, &group, rng);
                let any_echo = group.iter().any(|&index| pending[index].echo);
                let group_result = executor.execute_prefill(PrefillPlan {
                    requests: &requests,
                    echo: any_echo,
                })?;
                result.requests.extend(group_result.requests);
            }
            sort_prefill_results(&mut result.requests);
            Ok(ExecutionArtifacts::Prefill { pending, result })
        }
        ExecutionPlan::Decode => {
            let mut result = DecodeResult {
                requests: Vec::with_capacity(active.len()),
            };
            for (key, group) in group_active_indices(active) {
                executor.activate_lora_adapter(key.as_deref())?;
                let requests = build_decode_items(active, &group, rng);
                let group_result = executor.execute_decode(DecodePlan {
                    requests: &requests,
                })?;
                result.requests.extend(group_result.requests);
            }
            sort_decode_results(&mut result.requests);
            Ok(ExecutionArtifacts::Decode { result })
        }
        ExecutionPlan::Unified { pending } => {
            let mut result = UnifiedResult {
                prefill_requests: Vec::with_capacity(pending.len()),
                decode_requests: Vec::with_capacity(active.len()),
            };
            let pending_groups = group_pending_indices(&pending);
            let active_groups = group_active_indices(active);
            let keys = union_group_keys(&pending_groups, &active_groups);

            for key in keys {
                executor.activate_lora_adapter(key.as_deref())?;
                let pending_group = pending_groups.get(&key).cloned().unwrap_or_default();
                let active_group = active_groups.get(&key).cloned().unwrap_or_default();
                match (pending_group.is_empty(), active_group.is_empty()) {
                    (false, false) => {
                        let prefill_requests = build_prefill_items(&pending, &pending_group, rng);
                        let decode_requests = build_decode_items(active, &active_group, rng);
                        let group_result = executor.execute_unified(UnifiedPlan {
                            prefill_requests: &prefill_requests,
                            decode_requests: &decode_requests,
                        })?;
                        result
                            .prefill_requests
                            .extend(group_result.prefill_requests);
                        result.decode_requests.extend(group_result.decode_requests);
                    }
                    (false, true) => {
                        let prefill_requests = build_prefill_items(&pending, &pending_group, rng);
                        let any_echo = pending_group.iter().any(|&index| pending[index].echo);
                        let group_result = executor.execute_prefill(PrefillPlan {
                            requests: &prefill_requests,
                            echo: any_echo,
                        })?;
                        result.prefill_requests.extend(group_result.requests);
                    }
                    (true, false) => {
                        let decode_requests = build_decode_items(active, &active_group, rng);
                        let group_result = executor.execute_decode(DecodePlan {
                            requests: &decode_requests,
                        })?;
                        result.decode_requests.extend(group_result.requests);
                    }
                    (true, true) => {}
                }
            }
            sort_prefill_results(&mut result.prefill_requests);
            sort_decode_results(&mut result.decode_requests);
            Ok(ExecutionArtifacts::Unified { pending, result })
        }
    }
}

fn group_pending_indices(pending: &[PendingRequest]) -> BTreeMap<LoraGroupKey, Vec<usize>> {
    let mut groups = BTreeMap::new();
    for (index, req) in pending.iter().enumerate() {
        groups
            .entry(LoraGroupKey::from_option(&req.lora_adapter))
            .or_insert_with(Vec::new)
            .push(index);
    }
    groups
}

fn group_active_indices(active: &[ActiveRequestState]) -> BTreeMap<LoraGroupKey, Vec<usize>> {
    let mut groups = BTreeMap::new();
    for (index, req) in active.iter().enumerate() {
        groups
            .entry(LoraGroupKey::from_option(&req.lora_adapter))
            .or_insert_with(Vec::new)
            .push(index);
    }
    groups
}

fn union_group_keys(
    pending: &BTreeMap<LoraGroupKey, Vec<usize>>,
    active: &BTreeMap<LoraGroupKey, Vec<usize>>,
) -> Vec<LoraGroupKey> {
    let mut keys: Vec<LoraGroupKey> = pending.keys().chain(active.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    keys
}

fn build_prefill_items(
    pending: &[PendingRequest],
    indices: &[usize],
    rng: &mut StdRng,
) -> Vec<PrefillStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &pending[index];
            PrefillStepItem {
                request_id: r.request_id,
                prompt_tokens: r.prompt_tokens.clone(),
                max_output_tokens: r.max_tokens,
                params: r.params,
                logprobs: r.logprobs,
                echo: r.echo,
                random_val: rand::RngExt::random(rng),
                cached_tokens: 0,
            }
        })
        .collect()
}

fn build_decode_items(
    active: &[ActiveRequestState],
    indices: &[usize],
    rng: &mut StdRng,
) -> Vec<DecodeStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &active[index];
            DecodeStepItem {
                request_id: r.request_id,
                token_id: r.last_token,
                params: r.params,
                logprobs: r.logprobs,
                random_val: rand::RngExt::random(rng),
            }
        })
        .collect()
}

fn sort_prefill_results(results: &mut [crate::executor::PrefillRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

fn sort_decode_results(results: &mut [crate::executor::DecodeRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::RequestId;
    use pegainfer_core::sampler::SamplingParams;

    fn pending() -> PendingRequest {
        let (token_tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        PendingRequest {
            request_id: RequestId::new(0),
            lora_adapter: None,
            prompt_tokens: vec![1, 2, 3],
            params: SamplingParams::default(),
            max_tokens: 8,
            token_tx,
            logprobs: 0,
            echo: false,
        }
    }

    // The plan selector is the whole batch-formation policy: what the scheduler
    // does each tick is fully determined by (have_active, has_pending). Pin the
    // 2×2 truth table so a policy regression can't slip through silently.
    #[test]
    fn plan_selection_follows_active_and_pending_state() {
        assert!(
            build_next_plan(false, vec![]).is_none(),
            "idle scheduler (no active, no pending) produces no plan"
        );
        assert!(
            matches!(build_next_plan(true, vec![]), Some(ExecutionPlan::Decode)),
            "active-only ticks decode the running batch"
        );
        assert!(
            matches!(
                build_next_plan(false, vec![pending()]),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "pending-only prefills the new arrivals"
        );
        assert!(
            matches!(
                build_next_plan(true, vec![pending()]),
                Some(ExecutionPlan::Unified { pending }) if pending.len() == 1
            ),
            "active + pending fuses prefill and decode into one unified step"
        );
    }
}
