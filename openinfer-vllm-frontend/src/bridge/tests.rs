//! Focused tests over the real `dispatch_burst` / `reduce_request` demux — fed
//! actual `TokenEvent`s, asserting on actual `EngineCoreOutputs`, no mocks. They
//! pin what the HTTP layer structurally can't observe: cross-request bucketing,
//! first-token metadata riding the first output exactly once, prefill-cache
//! accounting (`UsageInfo` is built from raw counts and never reads
//! `prefill_stats`, so cached/computed surface only here and in Prometheus), and
//! abort dropping late tokens. The full HTTP→ZMQ→bridge happy path is covered
//! end to end by `openinfer-sim`'s `frontend_e2e` integration test (the CI gate).

use std::sync::atomic::Ordering;

use openinfer_engine::engine::FinishReason;
use openinfer_engine::engine::RequestAbortReason;
use openinfer_engine::engine::TokenLogprob;

use super::*;

/// Test harness that exercises the bridge's demux path directly: register
/// requests, emit tagged events onto the shared channel, drain one ready
/// burst at a time, and inspect the coalesced outputs — the same flow the
/// `run` loop drives, minus the sockets.
struct Demux {
    event_tx: mpsc::UnboundedSender<(RequestTag, TokenEvent)>,
    event_rx: TokenStreamReceiver,
    streams: HashMap<RequestTag, RequestStreamState>,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
    output_rx: mpsc::UnboundedReceiver<EngineCoreOutputs>,
}

impl Demux {
    fn new() -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        Self {
            event_tx,
            event_rx,
            streams: HashMap::new(),
            output_tx,
            output_rx,
        }
    }

    /// Register a request as `start_request` does and return its abort reason.
    fn add(&mut self, id: &str) -> Arc<AtomicU8> {
        let tag: RequestTag = Arc::from(id);
        let abort_reason = Arc::new(AtomicU8::new(RequestAbortReason::None as u8));
        self.streams.insert(
            Arc::clone(&tag),
            RequestStreamState::new(Arc::clone(&abort_reason)),
        );
        abort_reason
    }

    fn emit(&self, id: &str, event: TokenEvent) {
        self.event_tx
            .send((Arc::from(id), event))
            .expect("emit token event");
    }

    /// Process one ready burst. Returns false if nothing was queued.
    fn drain(&mut self) -> bool {
        match self.event_rx.try_recv() {
            Ok(first) => {
                dispatch_burst(
                    0,
                    first,
                    &mut self.event_rx,
                    &mut self.streams,
                    &self.output_tx,
                )
                .expect("dispatch burst");
                true
            }
            Err(_) => false,
        }
    }

    fn next_output(&mut self) -> Option<RequestBatchOutputs> {
        let outputs = self.output_rx.try_recv().ok()?;
        match outputs {
            EngineCoreOutputs::RequestBatch(batch) => Some(batch),
            other => panic!("expected a request batch, got {other:?}"),
        }
    }
}

/// Token(s) and the terminal arriving in one burst (EmitAndFinish) coalesce
/// into a single output carrying both the tokens and the finish reason —
/// the canonical vLLM shape, one wire message instead of two.
#[test]
fn token_and_finish_in_one_burst_coalesce() {
    let mut d = Demux::new();
    d.add("req-1");
    d.emit(
        "req-1",
        TokenEvent::Scheduled {
            queued_at_unix_s: 1.0,
            scheduled_at_unix_s: 2.0,
            prompt_tokens: 16,
            cached_tokens: 0,
        },
    );
    d.emit(
        "req-1",
        TokenEvent::Token {
            id: 11,
            logprob: Some(TokenLogprob {
                logprob: -0.1,
                top_logprobs: vec![(11, -0.1), (12, -0.5)],
            }),
        },
    );
    d.emit(
        "req-1",
        TokenEvent::Token {
            id: 21,
            logprob: Some(TokenLogprob {
                logprob: -0.2,
                top_logprobs: vec![(21, -0.2), (22, -0.6)],
            }),
        },
    );
    d.emit(
        "req-1",
        TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: 16,
            completion_tokens: 2,
        },
    );
    assert!(d.drain());

    let outputs = d.next_output().expect("coalesced output");
    assert_eq!(outputs.outputs.len(), 1);
    let output = &outputs.outputs[0];
    assert_eq!(output.request_id, "req-1");
    assert_eq!(output.new_token_ids, vec![11, 21]);
    assert_eq!(output.finish_reason, Some(EngineCoreFinishReason::Length));
    assert!(output.events.is_some());
    assert!(output.prefill_stats.is_some());
    assert!(
        outputs
            .finished_requests
            .as_ref()
            .is_some_and(|requests| requests.contains("req-1"))
    );

    let direct = match output.new_logprobs.as_ref().expect("batched logprobs") {
        MaybeWireLogprobs::Direct(direct) => direct,
        MaybeWireLogprobs::Wire(_) => panic!("expected direct batched logprobs"),
    };
    assert_eq!(direct.positions.len(), 2);
    assert_eq!(direct.positions[0].entries[0].token_id, 11);
    assert_eq!(direct.positions[1].entries[0].token_id, 21);

    assert!(d.next_output().is_none());
}

/// A lone `Scheduled` (no token yet) emits nothing; its metadata waits in the
/// stream state across bursts and flushes onto the first real output. This is
/// the reason `RequestStreamState` holds `first_token_*` between bursts.
#[test]
fn lone_scheduled_defers_until_first_token() {
    let mut d = Demux::new();
    d.add("req-defer");
    d.emit(
        "req-defer",
        TokenEvent::Scheduled {
            queued_at_unix_s: 1.0,
            scheduled_at_unix_s: 2.0,
            prompt_tokens: 4,
            cached_tokens: 0,
        },
    );
    assert!(d.drain());
    assert!(d.next_output().is_none(), "scheduled alone emits nothing");
    assert!(d.streams.contains_key("req-defer"), "stream is retained");

    d.emit(
        "req-defer",
        TokenEvent::Token {
            id: 7,
            logprob: None,
        },
    );
    assert!(d.drain());
    let output = d.next_output().expect("first token output");
    assert_eq!(output.outputs[0].new_token_ids, vec![7]);
    assert!(
        output.outputs[0].events.is_some(),
        "deferred scheduled events flush onto the first token"
    );
}

/// First-token metadata (queued/scheduled events + prefill stats) rides the
/// first output exactly once, and `num_computed_tokens` is prompt minus the
/// prefix-cache hit — not the full prompt.
#[test]
fn first_token_metadata_is_only_sent_with_first_output() {
    let mut d = Demux::new();
    d.add("req-2");
    d.emit(
        "req-2",
        TokenEvent::Scheduled {
            queued_at_unix_s: 1.0,
            scheduled_at_unix_s: 2.0,
            prompt_tokens: 8,
            cached_tokens: 5,
        },
    );
    d.emit(
        "req-2",
        TokenEvent::Token {
            id: 1,
            logprob: None,
        },
    );
    assert!(d.drain());

    d.emit(
        "req-2",
        TokenEvent::PromptTokens {
            ids: vec![9],
            logprobs: vec![None],
        },
    );
    d.emit(
        "req-2",
        TokenEvent::Token {
            id: 2,
            logprob: None,
        },
    );
    assert!(d.drain());

    let first_batch = d.next_output().expect("first batch");
    let second_batch = d.next_output().expect("second batch");
    assert_eq!(first_batch.outputs[0].new_token_ids, vec![1]);
    assert_eq!(second_batch.outputs[0].new_token_ids, vec![2]);
    assert!(first_batch.outputs[0].events.is_some());
    let stats = first_batch.outputs[0]
        .prefill_stats
        .as_ref()
        .expect("first batch carries prefill stats");
    assert_eq!(stats.num_prompt_tokens, 8);
    assert_eq!(stats.num_cached_tokens, 5);
    assert_eq!(stats.num_local_cached_tokens, 5);
    assert_eq!(
        stats.num_computed_tokens, 3,
        "computed must be prompt minus cached, not the full prompt"
    );
    assert!(second_batch.outputs[0].events.is_none());
    assert!(second_batch.outputs[0].prefill_stats.is_none());
    assert!(d.next_output().is_none());
}

/// A request that stops on its first sampled token never emits `Token` — the
/// terminal output must still deliver the scheduled events and prefill stats or
/// cached_tokens silently vanishes from usage.
#[test]
fn stop_on_prefill_terminal_output_carries_prefill_stats() {
    let mut d = Demux::new();
    d.add("req-stop");
    d.emit(
        "req-stop",
        TokenEvent::Scheduled {
            queued_at_unix_s: 1.0,
            scheduled_at_unix_s: 2.0,
            prompt_tokens: 16,
            cached_tokens: 4,
        },
    );
    d.emit(
        "req-stop",
        TokenEvent::Finished {
            finish_reason: FinishReason::Stop,
            prompt_tokens: 16,
            completion_tokens: 0,
        },
    );
    assert!(d.drain());

    let terminal = d.next_output().expect("terminal output");
    let output = &terminal.outputs[0];
    assert_eq!(output.finish_reason, Some(EngineCoreFinishReason::Stop));
    assert!(
        output.events.is_some(),
        "queued/scheduled events must flush"
    );
    let stats = output
        .prefill_stats
        .as_ref()
        .expect("terminal output must flush prefill stats");
    assert_eq!(stats.num_cached_tokens, 4);
    assert_eq!(stats.num_computed_tokens, 12);
    assert!(d.next_output().is_none());
}

/// Many requests' tokens in one burst coalesce into a single `EngineCoreOutputs`
/// with one `EngineCoreOutput` each, each request's tokens kept in its own
/// bucket — the N→1 collapse must never cross-contaminate requests.
#[test]
fn burst_batches_multiple_requests_into_one_message() {
    let mut d = Demux::new();
    d.add("req-a");
    d.add("req-b");
    d.emit(
        "req-a",
        TokenEvent::Token {
            id: 1,
            logprob: None,
        },
    );
    d.emit(
        "req-b",
        TokenEvent::Token {
            id: 2,
            logprob: None,
        },
    );
    assert!(d.drain());

    let outputs = d.next_output().expect("one coalesced message");
    assert_eq!(outputs.outputs.len(), 2);
    let a = outputs
        .outputs
        .iter()
        .find(|o| o.request_id == "req-a")
        .expect("req-a output");
    let b = outputs
        .outputs
        .iter()
        .find(|o| o.request_id == "req-b")
        .expect("req-b output");
    assert_eq!(a.new_token_ids, vec![1]);
    assert_eq!(b.new_token_ids, vec![2]);
    assert!(d.next_output().is_none(), "exactly one wire message");
}

/// After an abort removes the stream entry, a token already in flight for that
/// request is dropped instead of producing a stray output.
#[test]
fn aborted_request_drops_late_tokens() {
    let mut d = Demux::new();
    let abort_reason = d.add("req-abort");

    // Replicate the Abort handler for a request that already emitted output:
    // drop the stream and mark it as cancelled.
    RequestAbortReason::Cancelled.store(&abort_reason);
    d.streams.remove("req-abort");

    d.emit(
        "req-abort",
        TokenEvent::Token {
            id: 99,
            logprob: None,
        },
    );
    assert!(d.drain(), "the late event is consumed");
    assert!(
        d.next_output().is_none(),
        "no output is produced for an aborted request"
    );
}

#[test]
fn abort_reason_tracks_first_output_boundary() {
    let mut d = Demux::new();
    let disconnected = d.add("req-before-output");
    let cancelled = d.add("req-after-output");

    d.emit(
        "req-after-output",
        TokenEvent::Token {
            id: 7,
            logprob: None,
        },
    );
    assert!(d.drain());
    assert_eq!(d.next_output().expect("first output").outputs.len(), 1);
    assert!(
        d.streams
            .get("req-after-output")
            .is_some_and(|state| state.has_emitted_tokens)
    );

    let state = d
        .streams
        .remove("req-before-output")
        .expect("disconnect stream");
    let reason = if state.has_emitted_tokens {
        RequestAbortReason::Cancelled
    } else {
        RequestAbortReason::Disconnected
    };
    state.abort(reason);

    let state = d.streams.remove("req-after-output").expect("cancel stream");
    let reason = if state.has_emitted_tokens {
        RequestAbortReason::Cancelled
    } else {
        RequestAbortReason::Disconnected
    };
    state.abort(reason);

    assert_eq!(
        RequestAbortReason::from_raw(disconnected.load(Ordering::Acquire)),
        RequestAbortReason::Disconnected
    );
    assert_eq!(
        RequestAbortReason::from_raw(cancelled.load(Ordering::Acquire)),
        RequestAbortReason::Cancelled
    );
}

/// A rejected request (could not be admitted, e.g. too large for the KV cache)
/// surfaces to the client as an error with the rejection message, and its stream
/// is retired.
#[test]
fn rejected_request_is_reported_as_error() {
    let mut d = Demux::new();
    d.add("req-1");
    d.emit(
        "req-1",
        TokenEvent::Rejected {
            message: "request is too large for KV cache".to_string(),
            prompt_tokens: 16,
            completion_tokens: 0,
        },
    );
    assert!(d.drain());

    let outputs = d.next_output().expect("terminal output");
    assert!(
        outputs
            .finished_requests
            .as_ref()
            .is_some_and(|requests| requests.contains("req-1"))
    );
    assert_eq!(outputs.outputs.len(), 1);
    let output = &outputs.outputs[0];
    assert_eq!(output.request_id, "req-1");
    assert_eq!(output.finish_reason, Some(EngineCoreFinishReason::Error));
    assert_eq!(
        output.stop_reason,
        Some(StopReason::Text(
            "request is too large for KV cache".to_string()
        ))
    );
    assert!(d.next_output().is_none());
    assert!(
        !d.streams.contains_key("req-1"),
        "finished stream is removed"
    );
}

/// The scheduler-stats task turns each load-watch snapshot into a stats-only
/// batch (no request outputs, no finished set) with the queue gauges and the
/// fractional KV usage the frontend records into Prometheus, sends the current
/// snapshot up front, and follows every watch update with exactly one message.
#[tokio::test]
async fn load_snapshots_become_stats_only_batches() {
    let (load_tx, load_rx) = tokio::sync::watch::channel(LoadSnapshot {
        kv_used_blocks: 25,
        kv_total_blocks: 100,
        num_running_reqs: 2,
        num_waiting_reqs: 1,
    });
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let shutdown = CancellationToken::new();
    let task = tokio::spawn(publish_scheduler_stats(
        7,
        load_rx,
        output_tx,
        shutdown.clone(),
    ));

    let batch = match output_rx.recv().await.expect("initial stats batch") {
        EngineCoreOutputs::RequestBatch(batch) => batch,
        other => panic!("expected a stats batch, got {other:?}"),
    };
    assert!(batch.outputs.is_empty());
    assert!(batch.finished_requests.is_none());
    assert_eq!(batch.engine_index, 7);
    let stats = batch.scheduler_stats.expect("scheduler stats");
    assert_eq!(stats.num_running_reqs, 2);
    assert_eq!(stats.num_waiting_reqs, 1);
    assert!((stats.kv_cache_usage - 0.25).abs() < 1e-9);

    load_tx.send_replace(LoadSnapshot::default());
    let batch = match output_rx.recv().await.expect("drained stats batch") {
        EngineCoreOutputs::RequestBatch(batch) => batch,
        other => panic!("expected a stats batch, got {other:?}"),
    };
    let stats = batch.scheduler_stats.expect("scheduler stats");
    assert_eq!(stats.num_running_reqs, 0);
    assert_eq!(stats.kv_cache_usage.to_bits(), 0.0_f64.to_bits());

    shutdown.cancel();
    task.await
        .expect("stats task exits on shutdown")
        .expect("stats publisher shuts down cleanly");
}
