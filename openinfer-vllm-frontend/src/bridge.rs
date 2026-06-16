use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use log::{info, warn};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::task::JoinHandle;
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

use openinfer_engine::engine::{EngineHandle, GenerateRequest, TokenEvent, TokenLogprob};

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

        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<String>();
        let mut active: HashMap<String, JoinHandle<()>> = HashMap::new();

        info!(
            "local vLLM engine bridge connected: input={}, output={}, max_model_len={}",
            self.input_address, self.output_address, self.max_model_len
        );

        loop {
            tokio::select! {
                () = shutdown.cancelled() => break,
                Some(request_id) = done_rx.recv() => {
                    active.remove(&request_id);
                }
                recv = input.recv() => {
                    let message = recv.context("failed to receive local engine request")?;
                    if let Err(error) = self.handle_message(
                        message,
                        &output_tx,
                        &done_tx,
                        &mut active,
                    ) {
                        warn!("local engine bridge request failed: {error:#}");
                    }
                }
            }
        }

        for (_, task) in active {
            task.abort();
        }
        drop(output_tx);
        output_task.abort();

        Ok(())
    }

    fn handle_message(
        &self,
        message: ZmqMessage,
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        done_tx: &mpsc::UnboundedSender<String>,
        active: &mut HashMap<String, JoinHandle<()>>,
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
                self.start_request(request, output_tx, done_tx, active)
            }
            ty if ty == EngineCoreRequestType::Abort.to_frame().as_ref() => {
                let request_ids: Vec<String> =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                for request_id in request_ids {
                    if let Some(task) = active.remove(&request_id) {
                        task.abort();
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
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        done_tx: &mpsc::UnboundedSender<String>,
        active: &mut HashMap<String, JoinHandle<()>>,
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

        let (token_tx, token_rx) = mpsc::unbounded_channel();
        self.handle
            .submit(GenerateRequest {
                request_id: Some(request_id.clone()),
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

        let output_tx = output_tx.clone();
        let done_tx = done_tx.clone();
        let task_request_id = request_id.clone();
        let task = tokio::spawn(async move {
            run_request_stream(task_request_id.clone(), token_rx, output_tx).await;
            let _ = done_tx.send(task_request_id);
        });
        active.insert(request_id, task);

        Ok(())
    }
}

async fn run_request_stream(
    request_id: String,
    mut token_rx: mpsc::UnboundedReceiver<TokenEvent>,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
) {
    let mut first_token_events = None;
    let mut first_token_prefill_stats = None;
    let mut has_sent_token_output = false;
    let mut pending_event = None;
    loop {
        let event = match pending_event.take() {
            Some(event) => event,
            None => match token_rx.recv().await {
                Some(event) => event,
                None => return,
            },
        };
        match event {
            TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s,
                prompt_tokens,
                cached_tokens,
            } => {
                first_token_events = Some(vec![
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
                first_token_prefill_stats = Some(PrefillStats {
                    num_prompt_tokens: prompt_tokens as u32,
                    num_computed_tokens: prompt_tokens.saturating_sub(cached_tokens) as u32,
                    num_cached_tokens: cached_tokens as u32,
                    num_local_cached_tokens: cached_tokens as u32,
                    num_external_cached_tokens: 0,
                });
            }
            TokenEvent::Token { id, logprob } => {
                // Keep the first streamed token on the direct path so TTFT
                // does not pay an extra scheduler turn. Later decode bursts
                // still benefit from one-turn coalescing before draining the
                // ready queue into one bridge output.
                if has_sent_token_output {
                    tokio::task::yield_now().await;
                }
                let (token_ids, batched_logprobs, next_event) =
                    collect_ready_token_batch(id, logprob, &mut token_rx);
                pending_event = next_event;
                if send_token_output(
                    &output_tx,
                    &request_id,
                    token_ids,
                    batched_logprobs,
                    first_token_events.take(),
                    first_token_prefill_stats.take(),
                )
                .is_err()
                {
                    return;
                }
                has_sent_token_output = true;
            }
            TokenEvent::PromptTokens { .. } => {
                // Prompt logprobs are intentionally deferred for this bridge.
            }
            TokenEvent::Finished { finish_reason, .. } => {
                // A request can finish without emitting a token (EOS sampled
                // on prefill) — flush the pending scheduled events and prefill
                // stats with the terminal output or they are lost.
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    convert_finish_reason(finish_reason),
                    None,
                    first_token_events.take(),
                    first_token_prefill_stats.take(),
                );
                return;
            }
            TokenEvent::Error { message, .. } => {
                warn!("request {request_id} failed: {message}");
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    EngineCoreFinishReason::Error,
                    Some(StopReason::Text(message)),
                    None,
                    None,
                );
                return;
            }
            TokenEvent::Rejected { message, .. } => {
                // Rejected means the request could not be admitted, not that it completed cleanly.
                warn!("request {request_id} rejected: {message}");
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    EngineCoreFinishReason::Error,
                    Some(StopReason::Text(message)),
                    None,
                    None,
                );
                return;
            }
        }
    }
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

fn send_token_output(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    request_id: &str,
    token_ids: Vec<u32>,
    logprobs: Option<MaybeWireLogprobs>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> Result<()> {
    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            outputs: vec![engine_output(
                request_id.to_string(),
                token_ids,
                logprobs,
                None,
                None,
                events,
                prefill_stats,
            )],
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
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

fn collect_ready_token_batch(
    first_id: u32,
    first_logprob: Option<TokenLogprob>,
    token_rx: &mut mpsc::UnboundedReceiver<TokenEvent>,
) -> (Vec<u32>, Option<MaybeWireLogprobs>, Option<TokenEvent>) {
    let mut token_ids = Vec::with_capacity(4);
    let mut positions = Vec::with_capacity(4);
    let mut has_logprobs = false;

    let mut push_token = |token_id: u32, logprob: Option<TokenLogprob>| {
        token_ids.push(token_id);
        if let Some(position) = to_wire_position_logprobs(token_id, logprob) {
            has_logprobs = true;
            positions.push(position);
        } else {
            positions.push(PositionLogprobs {
                entries: Vec::new(),
            });
        }
    };
    push_token(first_id, first_logprob);

    loop {
        match token_rx.try_recv() {
            Ok(TokenEvent::Token { id, logprob }) => push_token(id, logprob),
            Ok(other) => {
                return (
                    token_ids,
                    has_logprobs.then_some(MaybeWireLogprobs::Direct(Logprobs { positions })),
                    Some(other),
                );
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                return (
                    token_ids,
                    has_logprobs.then_some(MaybeWireLogprobs::Direct(Logprobs { positions })),
                    None,
                );
            }
        }
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
    use openinfer_engine::engine::FinishReason;

    use super::*;

    #[tokio::test]
    async fn rejected_request_is_reported_as_error() {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();

        token_tx
            .send(TokenEvent::Rejected {
                message: "request is too large for KV cache".to_string(),
                prompt_tokens: 16,
                completion_tokens: 0,
            })
            .expect("send rejected event");
        drop(token_tx);

        run_request_stream("req-1".to_string(), token_rx, output_tx).await;

        let outputs = output_rx.recv().await.expect("terminal output");
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
    }

    #[tokio::test]
    async fn consecutive_tokens_are_batched_into_one_output() {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();

        token_tx
            .send(TokenEvent::Scheduled {
                queued_at_unix_s: 1.0,
                scheduled_at_unix_s: 2.0,
                prompt_tokens: 16,
                cached_tokens: 0,
            })
            .expect("send scheduled");
        token_tx
            .send(TokenEvent::Token {
                id: 11,
                logprob: Some(TokenLogprob {
                    logprob: -0.1,
                    top_logprobs: vec![(11, -0.1), (12, -0.5)],
                }),
            })
            .expect("send token 1");
        token_tx
            .send(TokenEvent::Token {
                id: 21,
                logprob: Some(TokenLogprob {
                    logprob: -0.2,
                    top_logprobs: vec![(21, -0.2), (22, -0.6)],
                }),
            })
            .expect("send token 2");
        token_tx
            .send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: 16,
                completion_tokens: 2,
            })
            .expect("send finished");
        drop(token_tx);

        run_request_stream("req-1".to_string(), token_rx, output_tx).await;

        let token_outputs = output_rx.recv().await.expect("token output");
        assert_eq!(token_outputs.outputs.len(), 1);
        assert_eq!(token_outputs.outputs[0].request_id, "req-1");
        assert_eq!(token_outputs.outputs[0].new_token_ids, vec![11, 21]);
        assert!(token_outputs.outputs[0].finish_reason.is_none());
        assert!(token_outputs.outputs[0].events.is_some());
        assert!(token_outputs.outputs[0].prefill_stats.is_some());

        let direct = match token_outputs.outputs[0]
            .new_logprobs
            .as_ref()
            .expect("batched logprobs")
        {
            MaybeWireLogprobs::Direct(direct) => direct,
            MaybeWireLogprobs::Wire(_) => panic!("expected direct batched logprobs"),
        };
        assert_eq!(direct.positions.len(), 2);
        assert_eq!(direct.positions[0].entries[0].token_id, 11);
        assert_eq!(direct.positions[1].entries[0].token_id, 21);

        let terminal = output_rx.recv().await.expect("terminal output");
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
        assert!(output_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn first_token_metadata_is_only_sent_with_first_batch() {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();

        token_tx
            .send(TokenEvent::Scheduled {
                queued_at_unix_s: 1.0,
                scheduled_at_unix_s: 2.0,
                prompt_tokens: 8,
                cached_tokens: 5,
            })
            .expect("send scheduled");
        token_tx
            .send(TokenEvent::Token {
                id: 1,
                logprob: None,
            })
            .expect("send first token");
        token_tx
            .send(TokenEvent::PromptTokens {
                ids: vec![9],
                logprobs: vec![None],
            })
            .expect("send prompt token metadata");
        token_tx
            .send(TokenEvent::Token {
                id: 2,
                logprob: None,
            })
            .expect("send second token");
        drop(token_tx);

        run_request_stream("req-2".to_string(), token_rx, output_tx).await;

        let first_batch = output_rx.recv().await.expect("first batch");
        let second_batch = output_rx.recv().await.expect("second batch");
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
        assert!(output_rx.recv().await.is_none());
    }

    /// A request that stops on its first sampled token never emits `Token`
    /// — the terminal output must still deliver the scheduled events and
    /// prefill stats or cached_tokens silently vanishes from usage.
    #[tokio::test]
    async fn stop_on_prefill_terminal_output_carries_prefill_stats() {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();

        token_tx
            .send(TokenEvent::Scheduled {
                queued_at_unix_s: 1.0,
                scheduled_at_unix_s: 2.0,
                prompt_tokens: 16,
                cached_tokens: 4,
            })
            .expect("send scheduled");
        token_tx
            .send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 16,
                completion_tokens: 0,
            })
            .expect("send finished");
        drop(token_tx);

        run_request_stream("req-stop".to_string(), token_rx, output_tx).await;

        let terminal = output_rx.recv().await.expect("terminal output");
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
        assert!(output_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn mixed_logprob_batch_keeps_token_alignment() {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();

        token_tx
            .send(TokenEvent::Token {
                id: 31,
                logprob: None,
            })
            .expect("send token without logprob");
        token_tx
            .send(TokenEvent::Token {
                id: 32,
                logprob: Some(TokenLogprob {
                    logprob: -0.3,
                    top_logprobs: vec![(32, -0.3), (33, -0.7)],
                }),
            })
            .expect("send token with logprob");
        drop(token_tx);

        run_request_stream("req-3".to_string(), token_rx, output_tx).await;

        let batch = output_rx.recv().await.expect("batched output");
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
        assert!(output_rx.recv().await.is_none());
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
