use anyhow::{Result, bail};
use vllm_engine_core_client::protocol::logprobs::{
    PositionLogprobs, TokenLogprob as WireTokenLogprob,
};
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use openinfer_engine::engine::{FinishReason, TokenLogprob};
use openinfer_engine::sampler::SamplingParams;

pub(crate) const LORA_ADAPTER_XARG: &str = "openinfer_lora_adapter";

pub(crate) fn to_wire_position_logprobs(
    token_id: u32,
    logprob: Option<TokenLogprob>,
) -> Option<PositionLogprobs> {
    let lp = logprob?;
    let mut entries = Vec::with_capacity(1 + lp.top_logprobs.len());
    // openinfer-core does not currently expose the sampled token's vocab rank.
    // rank: 1 is correct for greedy sampling, where the sampled token is top-1,
    // and is a lossy placeholder for non-greedy sampling.
    // See discussion on PR #96.
    entries.push(WireTokenLogprob {
        token_id,
        logprob: lp.logprob,
        rank: 1,
    });
    // The msgpack ndarray is rectangular (vLLM's LogprobsTensors is
    // [positions, max_num_logprobs + 1]), so every position must carry the
    // full top-k even when the sampled token already appears in it — the
    // duplicate collapses back out when clients build per-position dicts.
    for (index, (alt_id, alt_logprob)) in lp.top_logprobs.into_iter().enumerate() {
        entries.push(WireTokenLogprob {
            token_id: alt_id,
            logprob: alt_logprob,
            rank: (index + 1) as u32,
        });
    }
    Some(PositionLogprobs { entries })
}

pub(crate) fn convert_sampling(params: &EngineCoreSamplingParams) -> SamplingParams {
    // The vLLM frontend lowers a client `ignore_eos=true` to `_eos_token_id:
    // None`, but `_all_stop_token_ids` always carries the model EOS set (it
    // exists for min_tokens masking, not stop detection). Deriving ignore_eos
    // from all_stop_token_ids would therefore void every ignore_eos request on
    // models with a real EOS. Only `_eos_token_id` and the client's explicit
    // `stop_token_ids` express a stop intent.
    let ignore_eos = params.eos_token_id.is_none() && params.stop_token_ids.is_empty();
    if params.temperature <= 0.0 {
        return SamplingParams {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
            ignore_eos,
        };
    }

    SamplingParams {
        temperature: params.temperature,
        top_k: if params.top_k == 0 {
            -1
        } else {
            i32::try_from(params.top_k).unwrap_or(i32::MAX)
        },
        top_p: params.top_p,
        min_p: params.min_p,
        // Per-request seeds need the scheduler to feed request-local step
        // counts into select_batch; until that lands, seeded requests are
        // rejected in the bridge instead of silently ignored. See the
        // sampling-parity tracking issue.
        seed: None,
        ignore_eos,
    }
}

/// Reject sampling parameters the engine would otherwise silently ignore.
/// Returns the offending description; `None` means the request is servable.
///
/// The float comparisons are exact on purpose: they detect "the client sent
/// anything other than the wire default", not numeric closeness — a request
/// carrying 1.0000001 wants a penalty and must be rejected, not rounded away.
#[allow(clippy::float_cmp)]
pub(crate) fn unsupported_sampling(params: &EngineCoreSamplingParams) -> Option<String> {
    if !(0.0..1.0).contains(&params.min_p) || !params.min_p.is_finite() {
        return Some(format!("min_p {} outside [0, 1)", params.min_p));
    }
    if params.temperature > 0.0 && params.seed.is_some() {
        return Some("per-request seed is not supported yet".to_string());
    }
    if params.frequency_penalty != 0.0 {
        return Some(format!(
            "frequency_penalty {} is not supported yet",
            params.frequency_penalty
        ));
    }
    if params.presence_penalty != 0.0 {
        return Some(format!(
            "presence_penalty {} is not supported yet",
            params.presence_penalty
        ));
    }
    if params.repetition_penalty != 1.0 {
        return Some(format!(
            "repetition_penalty {} is not supported yet",
            params.repetition_penalty
        ));
    }
    for (field, value) in [
        ("logprobs", params.logprobs),
        ("prompt_logprobs", params.prompt_logprobs),
    ] {
        match value {
            // `-1` means the full vocabulary in the vLLM contract; fail loud
            // instead of silently degrading it to "disabled".
            Some(-1) => {
                return Some(format!(
                    "{field}=-1 (full-vocabulary logprobs) is not supported yet; \
                     request a finite top-k count instead"
                ));
            }
            Some(value) if value < -1 => {
                return Some(format!("{field}={value} is invalid (must be >= -1)"));
            }
            _ => {}
        }
    }
    None
}

/// Map the pinned contract's `Option<i32>` logprob counts onto the engine's
/// `Option<usize>`: `None` stays disabled, `Some(0)` stays "scored token
/// only", and `Some(k)` stays top-`k`. Negative values are rejected upstream
/// by [`unsupported_sampling`], so they are unreachable here.
fn logprob_count(value: Option<i32>) -> Option<usize> {
    value.map(|value| usize::try_from(value).expect("negative logprobs rejected upstream"))
}

pub(crate) fn requested_logprobs(params: &EngineCoreSamplingParams) -> Option<usize> {
    logprob_count(params.logprobs)
}

pub(crate) fn requested_prompt_logprobs(params: &EngineCoreSamplingParams) -> Option<usize> {
    logprob_count(params.prompt_logprobs)
}

pub(crate) fn lora_adapter_from_sampling_params(
    params: &EngineCoreSamplingParams,
) -> Result<Option<String>> {
    let Some(extra_args) = params.extra_args.as_ref() else {
        return Ok(None);
    };
    let Some(value) = extra_args.get(LORA_ADAPTER_XARG) else {
        return Ok(None);
    };
    match value.as_str() {
        Some(name) if !name.is_empty() => Ok(Some(name.to_string())),
        Some(_) => bail!("{LORA_ADAPTER_XARG} must not be empty"),
        None => bail!("{LORA_ADAPTER_XARG} must be a string"),
    }
}

pub(crate) fn convert_finish_reason(reason: FinishReason) -> EngineCoreFinishReason {
    match reason {
        FinishReason::Length => EngineCoreFinishReason::Length,
        FinishReason::Stop => EngineCoreFinishReason::Stop,
        FinishReason::Error => EngineCoreFinishReason::Error,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use vllm_engine_core_client::protocol::logprobs::{Logprobs, MaybeWireLogprobs};

    use super::*;

    fn to_wire_logprobs(token_id: u32, logprob: Option<TokenLogprob>) -> Option<MaybeWireLogprobs> {
        let position = to_wire_position_logprobs(token_id, logprob)?;
        Some(MaybeWireLogprobs::Direct(Logprobs {
            positions: vec![position],
        }))
    }

    #[test]
    fn convert_sampling_honors_ignore_eos_lowering() {
        // ignore_eos=true lowering: _eos_token_id=None while
        // _all_stop_token_ids still carries the model EOS set.
        let mut params = EngineCoreSamplingParams::for_test();
        params.all_stop_token_ids = BTreeSet::from([163_586]);
        assert!(convert_sampling(&params).ignore_eos);

        // Normal request: _eos_token_id present.
        params.eos_token_id = Some(163_586);
        assert!(!convert_sampling(&params).ignore_eos);

        // Explicit client stop tokens keep EOS detection on even when the
        // frontend dropped _eos_token_id.
        params.eos_token_id = None;
        params.stop_token_ids = vec![42];
        assert!(!convert_sampling(&params).ignore_eos);
    }

    #[test]
    fn convert_sampling_passes_min_p_and_never_seed() {
        let mut params = EngineCoreSamplingParams::for_test();
        params.eos_token_id = Some(1);
        params.temperature = 0.8;
        params.min_p = 0.15;
        params.seed = Some(42);
        let converted = convert_sampling(&params);
        assert!((converted.min_p - 0.15).abs() < f32::EPSILON);
        // Seeds are rejected upstream until scheduler step wiring lands;
        // convert must never smuggle one through.
        assert_eq!(converted.seed, None);

        // Greedy lowering zeroes min_p along with the rest.
        params.temperature = 0.0;
        assert_eq!(convert_sampling(&params).min_p.to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn unsupported_sampling_rejects_what_the_engine_would_ignore() {
        let mut params = EngineCoreSamplingParams::for_test();
        params.repetition_penalty = 1.0;
        assert_eq!(unsupported_sampling(&params), None);

        params.min_p = 0.2;
        assert_eq!(unsupported_sampling(&params), None);
        params.min_p = 1.5;
        assert!(unsupported_sampling(&params).is_some());
        params.min_p = 0.0;

        params.temperature = 0.8;
        params.seed = Some(7);
        assert!(unsupported_sampling(&params).is_some());
        // A greedy request's seed is a no-op, not a lie — allowed.
        params.temperature = 0.0;
        assert_eq!(unsupported_sampling(&params), None);
        params.seed = None;

        params.frequency_penalty = 0.5;
        assert!(unsupported_sampling(&params).is_some());
        params.frequency_penalty = 0.0;
        params.presence_penalty = -0.5;
        assert!(unsupported_sampling(&params).is_some());
        params.presence_penalty = 0.0;
        params.repetition_penalty = 1.2;
        assert!(unsupported_sampling(&params).is_some());
        params.repetition_penalty = 1.0;
    }

    #[test]
    fn unsupported_sampling_rejects_full_vocabulary_and_invalid_logprobs() {
        let mut params = EngineCoreSamplingParams::for_test();
        assert_eq!(unsupported_sampling(&params), None);

        params.logprobs = Some(-1);
        let message = unsupported_sampling(&params).expect("-1 must be rejected");
        assert!(message.contains("full-vocabulary"), "{message}");

        params.logprobs = Some(-2);
        let message = unsupported_sampling(&params).expect("<-1 must be rejected");
        assert!(message.contains("invalid"), "{message}");

        params.logprobs = Some(0);
        assert_eq!(unsupported_sampling(&params), None);

        params.prompt_logprobs = Some(-1);
        let message = unsupported_sampling(&params).expect("prompt -1 must be rejected");
        assert!(message.contains("prompt_logprobs"), "{message}");
        params.prompt_logprobs = Some(3);
        assert_eq!(unsupported_sampling(&params), None);
    }

    #[test]
    fn requested_logprobs_preserves_the_option_contract() {
        let mut params = EngineCoreSamplingParams::for_test();
        assert_eq!(requested_logprobs(&params), None);
        assert_eq!(requested_prompt_logprobs(&params), None);

        // Some(0) requests the scored token's logprob with no top entries —
        // distinct from the disabled value, not an alias for it.
        params.logprobs = Some(0);
        params.prompt_logprobs = Some(0);
        assert_eq!(requested_logprobs(&params), Some(0));
        assert_eq!(requested_prompt_logprobs(&params), Some(0));

        params.logprobs = Some(5);
        params.prompt_logprobs = Some(2);
        assert_eq!(requested_logprobs(&params), Some(5));
        assert_eq!(requested_prompt_logprobs(&params), Some(2));
    }

    #[test]
    fn lora_adapter_from_sampling_params_reads_proxy_xarg() {
        let mut params = EngineCoreSamplingParams::for_test();
        params.extra_args = Some(HashMap::from([(
            LORA_ADAPTER_XARG.to_string(),
            serde_json::Value::String("adapter-a".to_string()),
        )]));

        assert_eq!(
            lora_adapter_from_sampling_params(&params)
                .expect("extract adapter")
                .as_deref(),
            Some("adapter-a")
        );
    }

    fn assert_logprob_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "logprob mismatch: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn to_wire_logprobs_emits_sampled_then_alternatives() {
        let lp = TokenLogprob {
            logprob: -0.5,
            top_logprobs: vec![(7, -0.5), (42, -1.5)],
        };
        let wire = to_wire_logprobs(7, Some(lp)).expect("logprob payload");
        let direct = match wire {
            MaybeWireLogprobs::Direct(d) => d,
            MaybeWireLogprobs::Wire(_) => panic!("expected Direct logprobs"),
        };
        assert_eq!(direct.positions.len(), 1);
        // Rectangular wire shape: the sampled token keeps its top-k slot too,
        // so every position stays max_num_logprobs + 1 wide.
        let entries = &direct.positions[0].entries;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].token_id, 7);
        assert_logprob_eq(entries[0].logprob, -0.5);
        assert_eq!(entries[0].rank, 1);
        assert_eq!(entries[1].token_id, 7);
        assert_logprob_eq(entries[1].logprob, -0.5);
        assert_eq!(entries[1].rank, 1);
        assert_eq!(entries[2].token_id, 42);
        assert_logprob_eq(entries[2].logprob, -1.5);
        assert_eq!(entries[2].rank, 2);
    }

    #[test]
    fn to_wire_logprobs_positions_have_uniform_width() {
        // Regression test for the engine-core msgpack encode crash: a prompt
        // logprobs batch whose sampled tokens fall inside the top-k for some
        // positions and outside it for others must stay rectangular.
        let sampled_in_topk = to_wire_position_logprobs(
            7,
            Some(TokenLogprob {
                logprob: -0.5,
                top_logprobs: vec![(7, -0.5), (42, -1.5)],
            }),
        )
        .expect("position");
        let sampled_outside_topk = to_wire_position_logprobs(
            1,
            Some(TokenLogprob {
                logprob: -14.0,
                top_logprobs: vec![(7, -0.5), (42, -1.5)],
            }),
        )
        .expect("position");
        assert_eq!(
            sampled_in_topk.entries.len(),
            sampled_outside_topk.entries.len()
        );
        assert_eq!(sampled_in_topk.entries.len(), 3);
    }

    #[test]
    fn to_wire_logprobs_keeps_distinct_top_k_alternatives() {
        let lp = TokenLogprob {
            logprob: -0.5,
            top_logprobs: vec![(8, -1.0), (9, -1.5)],
        };
        let wire = to_wire_logprobs(7, Some(lp)).expect("logprob payload");
        let direct = match wire {
            MaybeWireLogprobs::Direct(d) => d,
            MaybeWireLogprobs::Wire(_) => panic!("expected Direct logprobs"),
        };
        assert_eq!(direct.positions.len(), 1);
        let entries = &direct.positions[0].entries;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].token_id, 7);
        assert_logprob_eq(entries[0].logprob, -0.5);
        assert_eq!(entries[0].rank, 1);
        assert_eq!(entries[1].token_id, 8);
        assert_logprob_eq(entries[1].logprob, -1.0);
        assert_eq!(entries[1].rank, 1);
        assert_eq!(entries[2].token_id, 9);
        assert_logprob_eq(entries[2].logprob, -1.5);
        assert_eq!(entries[2].rank, 2);
    }
}
