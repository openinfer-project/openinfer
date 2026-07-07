//! Pure translation between Dynamo's request/response protocol and openinfer's
//! engine contract.
//!
//! Functional core: every function here is a total function over plain data —
//! no I/O, no async, no GPU. The branchy mapping logic that turns openinfer
//! `TokenEvent`s into Dynamo stream items lives here so it is unit-testable
//! without a model loaded, and the `generate` loop in [`crate::engine`] stays a
//! thin imperative shell over these functions.

use dynamo_backend_common::{
    BackendError, CompletionUsage, ComponentSnapshot, DynamoError, ErrorType, LLMEngineOutput,
    LLMEngineOutputExt, PreprocessedRequest, chunk, usage,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use openinfer_engine::engine::{
    FinishReason as EngineFinishReason, KvBlockEvent, LoadSnapshot, TokenEvent,
};
use openinfer_engine::sampler::SamplingParams;

/// Fallback token cap when the client leaves `stop_conditions.max_tokens`
/// unset. The Dynamo frontend almost always fills it from the request or the
/// model card; this only guards the genuinely-unset path so the engine never
/// runs unbounded.
pub const DEFAULT_MAX_TOKENS: usize = 16_384;

/// Map Dynamo `SamplingOptions` onto openinfer's `SamplingParams`.
///
/// openinfer's sampler is deliberately small — temperature / top-k / top-p /
/// min-p / ignore-eos. Penalties, beam search and guided decoding have no
/// engine-side knob yet, so they are dropped here rather than silently faked;
/// per-request seeds wait on the scheduler step wiring (sampling-parity
/// tracking issue) and are dropped for the same reason. `ignore_eos` lives on
/// `StopConditions` in Dynamo but on `SamplingParams` in openinfer.
pub fn to_sampling_params(request: &PreprocessedRequest) -> SamplingParams {
    let s = &request.sampling_options;
    SamplingParams {
        temperature: s.temperature.unwrap_or(0.0),
        top_k: s.top_k.unwrap_or(-1),
        top_p: s.top_p.unwrap_or(1.0),
        min_p: s.min_p.unwrap_or(0.0),
        seed: None,
        ignore_eos: request.stop_conditions.ignore_eos.unwrap_or(false),
    }
}

/// Generation cap, falling back to [`DEFAULT_MAX_TOKENS`] when unset.
pub fn resolve_max_tokens(request: &PreprocessedRequest) -> usize {
    request
        .stop_conditions
        .max_tokens
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// Outcome of mapping a single openinfer `TokenEvent` into the Dynamo stream.
pub enum Mapped {
    /// Prefix-cache hit count (`Scheduled.cached_tokens`): not a client-visible
    /// item, but the caller must remember it to stamp the terminal usage. Keep
    /// draining.
    Cached(u32),
    /// Non-terminal chunk to yield; keep draining.
    Chunk(LLMEngineOutput),
    /// Terminal chunk to yield, then stop (the single `finish_reason`-bearing
    /// item the contract requires last).
    Terminal(LLMEngineOutput),
    /// Typed terminal failure — yielded as `Err` so the frontend preserves the
    /// `BackendError` category (e.g. a rejected over-long prompt surfaces as a
    /// 4xx, not a 200 with an error finish_reason).
    Fail(DynamoError),
    /// No client-visible output (scheduler bookkeeping); keep draining.
    Ignore,
}

/// Translate one openinfer `TokenEvent` into at most one Dynamo stream item.
///
/// Total over the event alone: terminal events carry their own authoritative
/// prompt/completion counts, so no running accumulator is threaded in. The
/// cancelled terminal has no corresponding `TokenEvent` — the shell builds it.
pub fn map_token_event(event: TokenEvent) -> Mapped {
    match event {
        TokenEvent::Token { id, .. } => Mapped::Chunk(chunk::token(id)),
        TokenEvent::Finished {
            finish_reason,
            prompt_tokens,
            completion_tokens,
        } => {
            let u = usage(prompt_tokens as u32, completion_tokens as u32);
            match finish_reason {
                EngineFinishReason::Length => {
                    Mapped::Terminal(LLMEngineOutput::length().with_usage(u))
                }
                EngineFinishReason::Stop => Mapped::Terminal(LLMEngineOutput::stop().with_usage(u)),
                // openinfer signals real errors via TokenEvent::Error; a
                // Finished carrying Error is defensive — surface it as a typed
                // failure rather than a successful Stop.
                EngineFinishReason::Error => {
                    Mapped::Fail(backend_error("engine finished with FinishReason::Error"))
                }
            }
        }
        TokenEvent::Error { message, .. } => Mapped::Fail(backend_error(message)),
        TokenEvent::Rejected { message, .. } => Mapped::Fail(invalid_argument(message)),
        // The schedule event carries the matched prefix length — the only place
        // openinfer reports a cache hit. Carry it out so the terminal usage can
        // surface it as OpenAI `prompt_tokens_details.cached_tokens`.
        TokenEvent::Scheduled { cached_tokens, .. } => Mapped::Cached(cached_tokens as u32),
        // Echo / prompt-logprobs are not surfaced in M1.
        TokenEvent::PromptTokens { .. } => Mapped::Ignore,
    }
}

/// Stamp the prefix-cache hit onto an existing usage as OpenAI
/// `prompt_tokens_details.cached_tokens`. A zero hit is left absent (the field
/// stays `None`) so "no cache" and "cache reported zero" read alike downstream.
/// This is the only signal openinfer gives the Dynamo frontend's KV-hit metric.
pub fn apply_cached_tokens(usage: &mut CompletionUsage, cached_tokens: u32) {
    if cached_tokens == 0 {
        return;
    }
    let mut details = usage.prompt_tokens_details.take().unwrap_or_default();
    details.cached_tokens = Some(cached_tokens);
    usage.prompt_tokens_details = Some(details);
}

/// Build the Dynamo router/Prometheus snapshot from openinfer's live KV load.
///
/// `gpu_cache_usage` is the fraction of the pool in use; a zero-capacity pool
/// (degenerate, should not happen post-load) maps to 0.0 rather than the NaN
/// `0/0` would feed the gauge. `kv_cache_hit_rate` is `None`: M2 does not yet
/// surface a prefix-cache hit rate, and `None` (tri-state "no data") is the
/// honest value — `Some(0.0)` would read as a measured 0% hit rate.
pub fn load_to_component_snapshot(load: LoadSnapshot, dp_rank: u32) -> ComponentSnapshot {
    let gpu_cache_usage = if load.kv_total_blocks == 0 {
        0.0
    } else {
        load.kv_used_blocks as f32 / load.kv_total_blocks as f32
    };
    ComponentSnapshot {
        kv_used_blocks: load.kv_used_blocks,
        kv_total_blocks: load.kv_total_blocks,
        gpu_cache_usage,
        kv_cache_hit_rate: None,
        dp_rank,
    }
}

/// Translate one neutral engine [`KvBlockEvent`] into the router's wire event.
///
/// The engine already speaks the router's u64 hash space (block sequence-hash
/// and tokens-hash come straight off the token blocks), so this is a pure
/// field rename into Dynamo's newtypes — no hashing, no projection. `event_id`
/// is the publisher's monotonic counter; `dp_rank` is 0 because each openinfer
/// worker process owns exactly one rank. Multimodal extra-info and absolute
/// `start_position` have no openinfer analogue and are left `None`.
pub fn kv_block_event_to_dynamo(event: KvBlockEvent, event_id: u64) -> KvCacheEvent {
    let data = match event {
        KvBlockEvent::Stored {
            parent_hash,
            blocks,
        } => KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: parent_hash.map(ExternalSequenceBlockHash),
            start_position: None,
            blocks: blocks
                .into_iter()
                .map(|b| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(b.sequence_hash),
                    tokens_hash: LocalBlockHash(b.tokens_hash),
                    mm_extra_info: None,
                })
                .collect(),
        }),
        KvBlockEvent::Removed { sequence_hash } => KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(sequence_hash)],
        }),
    };
    KvCacheEvent {
        event_id,
        data,
        dp_rank: 0,
    }
}

pub fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(message)
        .build()
}

pub fn backend_error(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::Unknown))
        .message(message)
        .build()
}

pub fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
        .message(message)
        .build()
}

/// The token channel closed before a terminal event arrived — the engine
/// dropped the request mid-stream (crash or forced teardown that bypassed the
/// cancel path).
pub fn stream_incomplete() -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::StreamIncomplete))
        .message("openinfer engine closed the token channel before finishing")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::{FinishReason, SamplingOptions, StopConditions};

    fn request(
        stop: StopConditions,
        sampling: SamplingOptions,
        prompt: Vec<u32>,
    ) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("qwen3".to_string())
            .token_ids(prompt)
            .stop_conditions(stop)
            .sampling_options(sampling)
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    #[test]
    fn sampling_defaults_to_greedy_when_unset() {
        let p = to_sampling_params(&request(
            StopConditions::default(),
            SamplingOptions::default(),
            vec![1, 2],
        ));
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_k, -1);
        assert_eq!(p.top_p, 1.0);
        assert!(!p.ignore_eos);
        assert!(p.is_greedy());
    }

    #[test]
    fn sampling_passes_through_provided_values() {
        let p = to_sampling_params(&request(
            StopConditions {
                ignore_eos: Some(true),
                ..Default::default()
            },
            SamplingOptions {
                temperature: Some(0.7),
                top_k: Some(40),
                top_p: Some(0.95),
                ..Default::default()
            },
            vec![1],
        ));
        assert_eq!(p.temperature, 0.7);
        assert_eq!(p.top_k, 40);
        assert_eq!(p.top_p, 0.95);
        assert!(p.ignore_eos);
        assert!(!p.is_greedy());
    }

    #[test]
    fn max_tokens_resolves_with_fallback() {
        let set = request(
            StopConditions {
                max_tokens: Some(128),
                ..Default::default()
            },
            SamplingOptions::default(),
            vec![1],
        );
        assert_eq!(resolve_max_tokens(&set), 128);

        let unset = request(
            StopConditions::default(),
            SamplingOptions::default(),
            vec![1],
        );
        assert_eq!(resolve_max_tokens(&unset), DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn token_event_becomes_nonterminal_chunk() {
        match map_token_event(TokenEvent::Token {
            id: 42,
            logprob: None,
        }) {
            Mapped::Chunk(c) => {
                assert_eq!(c.token_ids, vec![42]);
                assert!(c.finish_reason.is_none());
            }
            _ => panic!("Token must map to a non-terminal chunk"),
        }
    }

    #[test]
    fn finished_length_carries_reason_and_usage() {
        match map_token_event(TokenEvent::Finished {
            finish_reason: EngineFinishReason::Length,
            prompt_tokens: 10,
            completion_tokens: 5,
        }) {
            Mapped::Terminal(t) => {
                assert!(matches!(t.finish_reason, Some(FinishReason::Length)));
                let u = t.completion_usage.expect("terminal carries usage");
                assert_eq!(u.prompt_tokens, 10);
                assert_eq!(u.completion_tokens, 5);
                assert_eq!(u.total_tokens, 15);
            }
            _ => panic!("Finished{{Length}} must map to a terminal"),
        }
    }

    #[test]
    fn finished_stop_maps_to_stop_terminal() {
        match map_token_event(TokenEvent::Finished {
            finish_reason: EngineFinishReason::Stop,
            prompt_tokens: 3,
            completion_tokens: 1,
        }) {
            Mapped::Terminal(t) => assert!(matches!(t.finish_reason, Some(FinishReason::Stop))),
            _ => panic!("Finished{{Stop}} must map to a terminal"),
        }
    }

    #[test]
    fn rejected_becomes_typed_invalid_argument() {
        match map_token_event(TokenEvent::Rejected {
            message: "prompt too long".to_string(),
            prompt_tokens: 9000,
            completion_tokens: 0,
        }) {
            Mapped::Fail(e) => assert_eq!(
                e.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            ),
            _ => panic!("Rejected must map to a typed Fail so the frontend returns 4xx"),
        }
    }

    #[test]
    fn engine_error_becomes_backend_fail() {
        match map_token_event(TokenEvent::Error {
            message: "kernel launch failed".to_string(),
            prompt_tokens: 4,
            completion_tokens: 2,
        }) {
            Mapped::Fail(e) => {
                assert_eq!(e.error_type(), ErrorType::Backend(BackendError::Unknown))
            }
            _ => panic!("Error must map to a Fail"),
        }
    }

    #[test]
    fn load_snapshot_maps_to_component_snapshot() {
        let snap = load_to_component_snapshot(
            LoadSnapshot {
                kv_used_blocks: 25,
                kv_total_blocks: 100,
                ..LoadSnapshot::default()
            },
            0,
        );
        assert_eq!(snap.kv_used_blocks, 25);
        assert_eq!(snap.kv_total_blocks, 100);
        assert!((snap.gpu_cache_usage - 0.25).abs() < 1e-6);
        // Tri-state: M2 has no hit-rate counter, so "no data" not measured-0%.
        assert_eq!(snap.kv_cache_hit_rate, None);
        assert_eq!(snap.dp_rank, 0);
    }

    #[test]
    fn zero_capacity_maps_usage_to_zero_not_nan() {
        let snap = load_to_component_snapshot(LoadSnapshot::default(), 0);
        assert_eq!(snap.gpu_cache_usage, 0.0);
        assert!(snap.gpu_cache_usage.is_finite());
    }

    #[test]
    fn stored_event_renames_hashes_into_dynamo_newtypes() {
        use openinfer_engine::engine::KvStoredBlock;
        let ev = kv_block_event_to_dynamo(
            KvBlockEvent::Stored {
                parent_hash: Some(0xAAAA),
                blocks: vec![
                    KvStoredBlock {
                        sequence_hash: 0x1111,
                        tokens_hash: 0x2222,
                    },
                    KvStoredBlock {
                        sequence_hash: 0x3333,
                        tokens_hash: 0x4444,
                    },
                ],
            },
            7,
        );
        assert_eq!(ev.event_id, 7);
        assert_eq!(ev.dp_rank, 0);
        match ev.data {
            KvCacheEventData::Stored(store) => {
                assert_eq!(store.parent_hash, Some(ExternalSequenceBlockHash(0xAAAA)));
                assert_eq!(store.start_position, None);
                assert_eq!(store.blocks.len(), 2);
                // u64 carried verbatim — block_hash is the router's
                // ExternalSequenceBlockHash, tokens_hash its LocalBlockHash.
                assert_eq!(
                    store.blocks[0].block_hash,
                    ExternalSequenceBlockHash(0x1111)
                );
                assert_eq!(store.blocks[0].tokens_hash, LocalBlockHash(0x2222));
                assert_eq!(
                    store.blocks[1].block_hash,
                    ExternalSequenceBlockHash(0x3333)
                );
                assert_eq!(store.blocks[1].tokens_hash, LocalBlockHash(0x4444));
                assert!(store.blocks[0].mm_extra_info.is_none());
            }
            _ => panic!("Stored must map to KvCacheEventData::Stored"),
        }
    }

    #[test]
    fn stored_root_block_has_no_parent() {
        let ev = kv_block_event_to_dynamo(
            KvBlockEvent::Stored {
                parent_hash: None,
                blocks: vec![],
            },
            0,
        );
        match ev.data {
            KvCacheEventData::Stored(store) => assert_eq!(store.parent_hash, None),
            _ => panic!("Stored must map to KvCacheEventData::Stored"),
        }
    }

    #[test]
    fn removed_event_carries_single_block_hash() {
        let ev = kv_block_event_to_dynamo(
            KvBlockEvent::Removed {
                sequence_hash: 0x9999,
            },
            3,
        );
        assert_eq!(ev.event_id, 3);
        match ev.data {
            KvCacheEventData::Removed(rm) => {
                assert_eq!(rm.block_hashes, vec![ExternalSequenceBlockHash(0x9999)]);
            }
            _ => panic!("Removed must map to KvCacheEventData::Removed"),
        }
    }

    #[test]
    fn scheduled_event_carries_prefix_cache_hit() {
        assert!(matches!(
            map_token_event(TokenEvent::Scheduled {
                queued_at_unix_s: 0.0,
                scheduled_at_unix_s: 0.0,
                prompt_tokens: 100,
                cached_tokens: 48,
            }),
            Mapped::Cached(48)
        ));
    }

    #[test]
    fn prompt_tokens_event_is_ignored() {
        assert!(matches!(
            map_token_event(TokenEvent::PromptTokens {
                ids: vec![1, 2],
                logprobs: vec![None, None],
            }),
            Mapped::Ignore
        ));
    }

    #[test]
    fn apply_cached_tokens_stamps_only_positive_hits() {
        // Zero hit leaves the field absent — "no cache" and "reported zero" alike.
        let mut u = usage(100, 5);
        apply_cached_tokens(&mut u, 0);
        assert!(u.prompt_tokens_details.is_none());

        apply_cached_tokens(&mut u, 48);
        assert_eq!(
            u.prompt_tokens_details.expect("details set").cached_tokens,
            Some(48)
        );
    }
}
