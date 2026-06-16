use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use log::{info, warn};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::logprobs::{Logprobs, MaybeWireLogprobs, PositionLogprobs};
use vllm_engine_core_client::protocol::utility::{
    UtilityCallId, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType, ModelDtype, StopReason,
    encode_msgpack, stats::PrefillStats,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use openinfer_engine::engine::{
    EngineHandle, GenerateRequest, RequestTag, TokenEvent, TokenSink, TokenStreamReceiver,
};

use crate::wire::{
    convert_finish_reason, convert_sampling, lora_adapter_from_sampling_params, requested_logprobs,
    to_wire_position_logprobs,
};

const ENGINE_INDEX: u32 = 0;

pub(crate) struct LocalEngineBridge {
    pub(crate) input_address: String,
    pub(crate) output_address: String,
    pub(crate) handle: EngineHandle,
    pub(crate) max_model_len: u32,
}

impl LocalEngineBridge {
    pub(crate) async fn run(self, shutdown: CancellationToken) -> Result<()> {
        wait_for_ipc_endpoint(&self.input_address, &shutdown).await?;
        wait_for_ipc_endpoint(&self.output_address, &shutdown).await?;

        let engine_id = EngineId::from_engine_index(ENGINE_INDEX);
        let mut socket_options = SocketOptions::default();
        socket_options.peer_identity(PeerIdentity::try_from(engine_id)?);

        let mut input = DealerSocket::with_options(socket_options);
        input.connect(&self.input_address).await.with_context(|| {
            format!(
                "failed to connect local engine input {}",
                self.input_address
            )
        })?;

        let ready = EngineCoreReadyResponse {
            max_model_len: self.max_model_len as u64,
            num_gpu_blocks: 0,
            // TODO(#401): report the real paged-KV block size and capacity from the
            // openinfer scheduler once the vLLM frontend consumes ready_response KV fields.
            block_size: 16,
            dp_stats_address: None,
            dtype: ModelDtype::BFloat16,
            vllm_version: "openinfer-local-bridge".to_string(),
            kv_cache_size_tokens: None,
            kv_cache_max_concurrency: None,
        };
        input
            .send(ZmqMessage::from(encode_msgpack(&ready)?))
            .await
            .context("failed to send local engine ready response")?;

        let mut output = PushSocket::new();
        output
            .connect(&self.output_address)
            .await
            .with_context(|| {
                format!(
                    "failed to connect local engine output {}",
                    self.output_address
                )
            })?;

        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let output_task = tokio::spawn(output_loop(output, output_rx));

        // One shared channel carries every request's token events, tagged by
        // request id; this loop is the sole consumer. Per-request state lives
        // in `streams`, keyed by the same tag, and holds the cancel flag the
        // scheduler observes (via `TokenSink`) when an abort flips it.
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let mut streams: HashMap<RequestTag, RequestStreamState> = HashMap::new();

        info!(
            "local vLLM engine bridge connected: input={}, output={}, max_model_len={}",
            self.input_address, self.output_address, self.max_model_len
        );

        loop {
            tokio::select! {
                () = shutdown.cancelled() => break,
                Some(first) = event_rx.recv() => {
                    if let Err(error) =
                        dispatch_burst(first, &mut event_rx, &mut streams, &output_tx)
                    {
                        warn!("local engine bridge output failed: {error:#}");
                    }
                }
                recv = input.recv() => {
                    let message = recv.context("failed to receive local engine request")?;
                    if let Err(error) = self.handle_message(
                        message,
                        &event_tx,
                        &output_tx,
                        &mut streams,
                    ) {
                        warn!("local engine bridge request failed: {error:#}");
                    }
                }
            }
        }

        // Cancel every in-flight request so the scheduler retires them on its
        // next emit instead of pushing into a channel no one drains.
        for state in streams.values() {
            state.cancelled.store(true, Ordering::Release);
        }
        drop(output_tx);
        output_task.abort();

        Ok(())
    }

    fn handle_message(
        &self,
        message: ZmqMessage,
        event_tx: &mpsc::UnboundedSender<(RequestTag, TokenEvent)>,
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        streams: &mut HashMap<RequestTag, RequestStreamState>,
    ) -> Result<()> {
        let frames = message.into_vec();
        if frames.len() != 2 {
            bail!(
                "expected 2 local engine request frames, got {}",
                frames.len()
            );
        }

        match frames[0].as_ref() {
            ty if ty == EngineCoreRequestType::Add.to_frame().as_ref() => {
                let request: EngineCoreRequest =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                self.start_request(request, event_tx, output_tx, streams)
            }
            ty if ty == EngineCoreRequestType::Abort.to_frame().as_ref() => {
                let request_ids: Vec<String> =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                for request_id in request_ids {
                    // Drop our state first, then flip the cancel flag (so the
                    // scheduler's next emit fails and retires the request). The
                    // `Release` store orders the `streams.remove` before the
                    // flag the scheduler reads with `Acquire`; any token already
                    // in flight for this id is discarded by the demux when it
                    // finds no stream entry.
                    if let Some(state) = streams.remove(request_id.as_str()) {
                        state.cancelled.store(true, Ordering::Release);
                    }
                }
                Ok(())
            }
            ty if ty == EngineCoreRequestType::Utility.to_frame().as_ref() => {
                let (_client_index, call_id, method_name, _args): (
                    u32,
                    UtilityCallId,
                    String,
                    rmpv::Value,
                ) = rmp_serde::from_slice(&frames[1])?;
                send_utility_response(output_tx, call_id, &method_name)
            }
            other => bail!("unsupported local engine request type frame: {other:?}"),
        }
    }

    fn start_request(
        &self,
        request: EngineCoreRequest,
        event_tx: &mpsc::UnboundedSender<(RequestTag, TokenEvent)>,
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        streams: &mut HashMap<RequestTag, RequestStreamState>,
    ) -> Result<()> {
        let EngineCoreRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            ..
        } = request;
        let Some(prompt_tokens) = prompt_token_ids else {
            warn!("request {request_id} dropped: missing prompt_token_ids");
            send_terminal_output(
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            )?;
            return Ok(());
        };
        let Some(sampling_params) = sampling_params else {
            warn!("request {request_id} dropped: missing sampling_params");
            send_terminal_output(
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            )?;
            return Ok(());
        };

        let tag: RequestTag = Arc::from(request_id.as_str());
        let cancelled = Arc::new(AtomicBool::new(false));
        let token_tx = TokenSink::new(tag.clone(), event_tx.clone(), Arc::clone(&cancelled));
        self.handle
            .submit(GenerateRequest {
                request_id: Some(request_id),
                queued_at_unix_s: Some(request.arrival_time),
                prompt_tokens,
                params: convert_sampling(&sampling_params),
                max_tokens: sampling_params.max_tokens as usize,
                lora_adapter: lora_adapter_from_sampling_params(&sampling_params)?,
                token_tx,
                logprobs: requested_logprobs(&sampling_params),
                echo: false,
            })
            .context("failed to submit request to scheduler")?;

        streams.insert(tag, RequestStreamState::new(cancelled));
        Ok(())
    }
}

/// Per-request demux state held by the bridge loop, keyed by [`RequestTag`].
/// Replaces the former per-request task's locals; `first_token_*` flush onto
/// the request's first output, `cancelled` is the flag the scheduler's
/// [`TokenSink`] checks so an abort retires the request without closing the
/// shared channel.
struct RequestStreamState {
    first_token_events: Option<Vec<EngineCoreEvent>>,
    first_token_prefill_stats: Option<PrefillStats>,
    cancelled: Arc<AtomicBool>,
}

impl RequestStreamState {
    fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self {
            first_token_events: None,
            first_token_prefill_stats: None,
            cancelled,
        }
    }
}

/// Drain the ready burst from the shared token channel (the `first` event plus
/// everything already queued), bucket it per request preserving event order,
/// fold each request's events into at most one `EngineCoreOutput`, and ship the
/// whole burst as a single `EngineCoreOutputs` — collapsing what used to be N
/// per-request ZMQ messages per step into one.
fn dispatch_burst(
    first: (RequestTag, TokenEvent),
    event_rx: &mut TokenStreamReceiver,
    streams: &mut HashMap<RequestTag, RequestStreamState>,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
) -> Result<()> {
    // Bucket the burst by request, keeping first-seen order so outputs are
    // deterministic and each request's events stay in arrival order.
    let mut order: Vec<RequestTag> = Vec::new();
    let mut buckets: HashMap<RequestTag, Vec<TokenEvent>> = HashMap::new();
    let mut bucket = |tag: RequestTag, event: TokenEvent| {
        if let Some(events) = buckets.get_mut(&tag) {
            events.push(event);
        } else {
            order.push(Arc::clone(&tag));
            buckets.insert(tag, vec![event]);
        }
    };
    bucket(first.0, first.1);
    while let Ok((tag, event)) = event_rx.try_recv() {
        bucket(tag, event);
    }

    let mut outputs: Vec<EngineCoreOutput> = Vec::with_capacity(order.len());
    let mut finished_requests: BTreeSet<String> = BTreeSet::new();
    for tag in order {
        let events = buckets.remove(&tag).expect("bucket for ordered tag");
        // No stream entry means the request was aborted or already finished;
        // its late events are dropped.
        let Some(state) = streams.get_mut(&tag) else {
            continue;
        };
        let (output, terminated) = reduce_request(&tag, state, events);
        if let Some(output) = output {
            outputs.push(output);
        }
        if terminated {
            streams.remove(&tag);
            finished_requests.insert(tag.to_string());
        }
    }

    if outputs.is_empty() {
        return Ok(());
    }
    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            outputs,
            finished_requests: (!finished_requests.is_empty()).then_some(finished_requests),
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
}

/// Fold one request's events from a single burst into at most one output.
/// Tokens coalesce, and a trailing terminal rides the same output carrying its
/// finish reason; `first_token_events`/`prefill_stats` flush onto whichever
/// output goes first. A lone `Scheduled` (no token, no terminal) yields no
/// output — its metadata waits in `state` for the first real output. Returns
/// `(output, terminated)`.
fn reduce_request(
    request_id: &str,
    state: &mut RequestStreamState,
    events: Vec<TokenEvent>,
) -> (Option<EngineCoreOutput>, bool) {
    let mut token_ids: Vec<u32> = Vec::new();
    let mut positions: Vec<PositionLogprobs> = Vec::new();
    let mut has_logprobs = false;
    let mut finish_reason: Option<EngineCoreFinishReason> = None;
    let mut stop_reason: Option<StopReason> = None;
    let mut terminated = false;

    for event in events {
        match event {
            TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s,
                prompt_tokens,
                cached_tokens,
            } => {
                state.first_token_events = Some(vec![
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Queued,
                        timestamp: queued_at_unix_s,
                    },
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Scheduled,
                        timestamp: scheduled_at_unix_s,
                    },
                ]);
                // Upstream invariant: computed (actual prefill work) +
                // cached (prefix-cache hit) == prompt; double-counting skews
                // the per-source prompt token metrics.
                state.first_token_prefill_stats = Some(PrefillStats {
                    num_prompt_tokens: prompt_tokens as u32,
                    num_computed_tokens: prompt_tokens.saturating_sub(cached_tokens) as u32,
                    num_cached_tokens: cached_tokens as u32,
                    num_local_cached_tokens: cached_tokens as u32,
                    num_external_cached_tokens: 0,
                });
            }
            TokenEvent::Token { id, logprob } => {
                token_ids.push(id);
                if let Some(position) = to_wire_position_logprobs(id, logprob) {
                    has_logprobs = true;
                    positions.push(position);
                } else {
                    positions.push(PositionLogprobs {
                        entries: Vec::new(),
                    });
                }
            }
            TokenEvent::PromptTokens { .. } => {
                // Prompt logprobs are intentionally deferred for this bridge.
            }
            TokenEvent::Finished { finish_reason: fr, .. } => {
                finish_reason = Some(convert_finish_reason(fr));
                terminated = true;
            }
            TokenEvent::Error { message, .. } => {
                warn!("request {request_id} failed: {message}");
                finish_reason = Some(EngineCoreFinishReason::Error);
                stop_reason = Some(StopReason::Text(message));
                terminated = true;
            }
            TokenEvent::Rejected { message, .. } => {
                // Rejected means the request could not be admitted, not that it
                // completed cleanly.
                warn!("request {request_id} rejected: {message}");
                finish_reason = Some(EngineCoreFinishReason::Error);
                stop_reason = Some(StopReason::Text(message));
                terminated = true;
            }
        }
    }

    if token_ids.is_empty() && !terminated {
        return (None, false);
    }

    let logprobs = has_logprobs.then_some(MaybeWireLogprobs::Direct(Logprobs { positions }));
    let output = engine_output(
        request_id.to_string(),
        token_ids,
        logprobs,
        finish_reason,
        stop_reason,
        state.first_token_events.take(),
        state.first_token_prefill_stats.take(),
    );
    (Some(output), terminated)
}

async fn output_loop(
    mut output: PushSocket,
    mut output_rx: mpsc::UnboundedReceiver<EngineCoreOutputs>,
) -> Result<()> {
    while let Some(outputs) = output_rx.recv().await {
        output
            .send(ZmqMessage::from(encode_msgpack(&outputs)?))
            .await
            .context("failed to send local engine output")?;
    }
    Ok(())
}

fn send_terminal_output(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
    stop_reason: Option<StopReason>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> Result<()> {
    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            outputs: vec![engine_output(
                request_id.clone(),
                Vec::new(),
                None,
                Some(finish_reason),
                stop_reason,
                events,
                prefill_stats,
            )],
            finished_requests: Some(BTreeSet::from([request_id])),
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
}

fn send_utility_response(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    call_id: UtilityCallId,
    method_name: &str,
) -> Result<()> {
    let result = match method_name {
        "is_sleeping" | "is_paused" | "reset_prefix_cache" => rmpv::ext::to_value(false)?,
        "sleep" | "wake_up" | "reset_mm_cache" | "reset_encoder_cache" | "collective_rpc" => {
            rmpv::Value::Nil
        }
        _ => rmpv::Value::Nil,
    };

    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            utility_output: Some(UtilityOutput {
                call_id,
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(result)),
            }),
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
}

fn send_outputs(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    outputs: EngineCoreOutputs,
) -> Result<()> {
    output_tx
        .send(outputs)
        .map_err(|_| anyhow::anyhow!("local engine output channel closed"))
}

fn engine_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    new_logprobs: Option<MaybeWireLogprobs>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        new_logprobs,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason,
        events,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

pub(crate) fn local_ipc_namespace() -> Result<PathBuf> {
    let base_dir =
        std::env::var_os("OPENINFER_IPC_DIR").map_or_else(|| PathBuf::from("/tmp"), PathBuf::from);
    let uuid = uuid::Uuid::new_v4().to_string();
    let path = base_dir.join(format!("pgi-{}-{}", std::process::id(), &uuid[..8]));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("failed to create IPC namespace {}", path.display()))?;
    Ok(path)
}

pub(crate) fn ipc_endpoint(namespace: &Path, name: &str) -> String {
    format!("ipc://{}", namespace.join(name).to_string_lossy())
}

async fn wait_for_ipc_endpoint(address: &str, shutdown: &CancellationToken) -> Result<()> {
    let Some(path) = address.strip_prefix("ipc://") else {
        return Ok(());
    };
    let path = Path::new(path);
    loop {
        if path.exists() {
            return Ok(());
        }
        tokio::select! {
            () = shutdown.cancelled() => bail!("shutdown before IPC endpoint appeared"),
            () = tokio::time::sleep(Duration::from_millis(20)) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use openinfer_engine::engine::{FinishReason, TokenLogprob};

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

        /// Register a request as `start_request` does and return its cancel flag.
        fn add(&mut self, id: &str) -> Arc<AtomicBool> {
            let tag: RequestTag = Arc::from(id);
            let cancelled = Arc::new(AtomicBool::new(false));
            self.streams
                .insert(Arc::clone(&tag), RequestStreamState::new(Arc::clone(&cancelled)));
            cancelled
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
                    dispatch_burst(first, &mut self.event_rx, &mut self.streams, &self.output_tx)
                        .expect("dispatch burst");
                    true
                }
                Err(_) => false,
            }
        }

        fn next_output(&mut self) -> Option<EngineCoreOutputs> {
            self.output_rx.try_recv().ok()
        }
    }

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
        assert!(!d.streams.contains_key("req-1"), "finished stream is removed");
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

    /// Tokens that stream over several steps and a finish that lands later
    /// arrive in separate bursts: a token output first, a terminal output once
    /// the finish shows up.
    #[test]
    fn tokens_then_finish_in_separate_bursts() {
        let mut d = Demux::new();
        d.add("req-1");
        d.emit(
            "req-1",
            TokenEvent::Token {
                id: 11,
                logprob: None,
            },
        );
        d.emit(
            "req-1",
            TokenEvent::Token {
                id: 21,
                logprob: None,
            },
        );
        assert!(d.drain());
        let token_out = d.next_output().expect("token output");
        assert_eq!(token_out.outputs[0].new_token_ids, vec![11, 21]);
        assert!(token_out.outputs[0].finish_reason.is_none());
        assert!(token_out.finished_requests.is_none());

        d.emit(
            "req-1",
            TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: 16,
                completion_tokens: 2,
            },
        );
        assert!(d.drain());
        let terminal = d.next_output().expect("terminal output");
        assert!(terminal.outputs[0].new_token_ids.is_empty());
        assert_eq!(
            terminal.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert!(
            terminal
                .finished_requests
                .as_ref()
                .is_some_and(|requests| requests.contains("req-1"))
        );
        assert!(d.next_output().is_none());
    }

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

    /// A lone `Scheduled` (no token yet) emits nothing; its metadata waits in
    /// the stream state and flushes onto the first real output.
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

    /// A request that stops on its first sampled token never emits `Token`
    /// — the terminal output must still deliver the scheduled events and
    /// prefill stats or cached_tokens silently vanishes from usage.
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

    #[test]
    fn mixed_logprob_batch_keeps_token_alignment() {
        let mut d = Demux::new();
        d.add("req-3");
        d.emit(
            "req-3",
            TokenEvent::Token {
                id: 31,
                logprob: None,
            },
        );
        d.emit(
            "req-3",
            TokenEvent::Token {
                id: 32,
                logprob: Some(TokenLogprob {
                    logprob: -0.3,
                    top_logprobs: vec![(32, -0.3), (33, -0.7)],
                }),
            },
        );
        assert!(d.drain());

        let batch = d.next_output().expect("batched output");
        let direct = match batch.outputs[0]
            .new_logprobs
            .as_ref()
            .expect("batched logprobs")
        {
            MaybeWireLogprobs::Direct(direct) => direct,
            MaybeWireLogprobs::Wire(_) => panic!("expected direct batched logprobs"),
        };

        assert_eq!(batch.outputs[0].new_token_ids, vec![31, 32]);
        assert_eq!(direct.positions.len(), 2);
        assert!(direct.positions[0].entries.is_empty());
        assert_eq!(direct.positions[1].entries[0].token_id, 32);
        assert!(d.next_output().is_none());
    }

    /// Many requests' tokens in one burst coalesce into a single
    /// `EngineCoreOutputs` with one `EngineCoreOutput` each — the N→1 wire
    /// collapse that motivated the demux.
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

    /// After an abort removes the stream entry, a token already in flight for
    /// that request is dropped instead of producing a stray output.
    #[test]
    fn aborted_request_drops_late_tokens() {
        let mut d = Demux::new();
        let cancelled = d.add("req-abort");

        // Replicate the Abort handler: flip the cancel flag, drop the stream.
        cancelled.store(true, Ordering::Relaxed);
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
    fn local_ipc_namespace_uses_short_path() {
        let namespace = local_ipc_namespace().expect("create namespace");
        let input = ipc_endpoint(&namespace, "input.sock");
        let output = ipc_endpoint(&namespace, "output.sock");
        assert!(input.len() < 100, "input IPC endpoint is too long: {input}");
        assert!(
            output.len() < 100,
            "output IPC endpoint is too long: {output}"
        );
        let _ = std::fs::remove_dir_all(namespace);
    }
}
