use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use log::{info, warn};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType, EngineCoreSamplingParams,
    Logprobs, MaybeWireLogprobs, PositionLogprobs, StopReason, TokenLogprob as WireTokenLogprob,
    UtilityOutput, UtilityResultEnvelope, encode_msgpack, stats::PrefillStats,
};
use vllm_engine_core_client::{EngineId, TransportMode};
use vllm_server::{
    ChatTemplateContentFormatOption, Config, CoordinatorMode, HttpListenerMode, ParserSelection,
    RendererSelection,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use pegainfer_engine::engine::{
    EngineHandle, FinishReason, GenerateRequest, TokenEvent, TokenLogprob,
};
use pegainfer_engine::sampler::SamplingParams;

const ENGINE_INDEX: u32 = 0;

#[derive(Debug, Deserialize)]
struct ModelLenConfig {
    max_position_embeddings: Option<u32>,
    text_config: Option<Box<ModelLenConfig>>,
}

impl ModelLenConfig {
    fn max_model_len(&self) -> Option<u32> {
        self.max_position_embeddings
            .or_else(|| self.text_config.as_ref()?.max_model_len())
    }
}

struct LocalEngineBridge {
    input_address: String,
    output_address: String,
    handle: EngineHandle,
    max_model_len: u32,
}

impl LocalEngineBridge {
    async fn run(self, shutdown: CancellationToken) -> Result<()> {
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
            dp_stats_address: None,
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
                let (_client_index, call_id, method_name, _args): (u32, i64, String, rmpv::Value) =
                    rmp_serde::from_slice(&frames[1])?;
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
            send_terminal_output(output_tx, request_id, EngineCoreFinishReason::Error, None)?;
            return Ok(());
        };
        let Some(sampling_params) = sampling_params else {
            send_terminal_output(output_tx, request_id, EngineCoreFinishReason::Error, None)?;
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

pub async fn serve(
    handle: EngineHandle,
    model_path: &Path,
    port: u16,
    shutdown: CancellationToken,
) -> Result<()> {
    let max_model_len = load_max_model_len(model_path).unwrap_or(4096);
    serve_model(
        handle,
        model_path.to_string_lossy().into_owned(),
        port,
        max_model_len,
        shutdown,
    )
    .await
}

pub async fn serve_model(
    handle: EngineHandle,
    model_id: impl Into<String>,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
) -> Result<()> {
    let namespace = local_ipc_namespace()?;
    let input_address = ipc_endpoint(&namespace, "input.sock");
    let output_address = ipc_endpoint(&namespace, "output.sock");
    let model_id = model_id.into();

    let bridge = LocalEngineBridge {
        input_address: input_address.clone(),
        output_address: output_address.clone(),
        handle,
        max_model_len,
    };
    let bridge_shutdown = shutdown.child_token();
    let bridge_task = tokio::spawn(async move {
        if let Err(error) = bridge.run(bridge_shutdown).await {
            warn!("local vLLM engine bridge exited: {error:#}");
        }
    });

    let config = Config {
        transport_mode: TransportMode::Bootstrapped {
            input_address,
            output_address,
            engine_count: 1,
            ready_timeout: Duration::from_secs(30),
        },
        coordinator_mode: CoordinatorMode::None,
        model: model_id,
        listener_mode: HttpListenerMode::BindTcp {
            host: "0.0.0.0".to_string(),
            port,
        },
        tool_call_parser: ParserSelection::default(),
        reasoning_parser: ParserSelection::default(),
        renderer: RendererSelection::default(),
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: ChatTemplateContentFormatOption::default(),
        enable_log_requests: true,
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(10),
    };

    let result = vllm_server::serve(config, shutdown).await;
    bridge_task.abort();
    let _ = std::fs::remove_dir_all(namespace);
    result
}

async fn run_request_stream(
    request_id: String,
    mut token_rx: mpsc::UnboundedReceiver<TokenEvent>,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
) {
    let mut first_token_events = None;
    let mut first_token_prefill_stats = None;
    while let Some(event) = token_rx.recv().await {
        match event {
            TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s,
                prompt_tokens,
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
                first_token_prefill_stats = Some(PrefillStats {
                    num_prompt_tokens: prompt_tokens as u32,
                    num_computed_tokens: prompt_tokens as u32,
                    num_cached_tokens: 0,
                    num_local_cached_tokens: 0,
                    num_external_cached_tokens: 0,
                });
            }
            TokenEvent::Token { id, logprob } => {
                if send_token_output(
                    &output_tx,
                    &request_id,
                    id,
                    logprob,
                    first_token_events.take(),
                    first_token_prefill_stats.take(),
                )
                .is_err()
                {
                    return;
                }
            }
            TokenEvent::PromptTokens { .. } => {
                // Prompt logprobs are intentionally deferred for this bridge.
            }
            TokenEvent::Finished { finish_reason, .. } => {
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    convert_finish_reason(finish_reason),
                    None,
                );
                return;
            }
            TokenEvent::Error { message, .. } => {
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    EngineCoreFinishReason::Error,
                    Some(StopReason::Text(message)),
                );
                return;
            }
            TokenEvent::Rejected { message, .. } => {
                // Rejected means the request could not be admitted, not that it completed cleanly.
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    EngineCoreFinishReason::Error,
                    Some(StopReason::Text(message)),
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
    token_id: u32,
    logprob: Option<TokenLogprob>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> Result<()> {
    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            outputs: vec![engine_output(
                request_id.to_string(),
                vec![token_id],
                to_wire_logprobs(token_id, logprob),
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
                None,
                None,
            )],
            finished_requests: Some(BTreeSet::from([request_id])),
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
}

fn send_utility_response(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    call_id: i64,
    method_name: &str,
) -> Result<()> {
    let result = match method_name {
        "is_sleeping" | "reset_prefix_cache" => rmpv::ext::to_value(false)?,
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

fn to_wire_logprobs(token_id: u32, logprob: Option<TokenLogprob>) -> Option<MaybeWireLogprobs> {
    let lp = logprob?;
    let mut entries = Vec::with_capacity(1 + lp.top_logprobs.len());
    // pegainfer-core does not currently expose the sampled token's vocab rank.
    // rank: 1 is correct for greedy sampling, where the sampled token is top-1,
    // and is a lossy placeholder for non-greedy sampling.
    // See discussion on PR #96.
    entries.push(WireTokenLogprob {
        token_id,
        logprob: lp.logprob,
        rank: 1,
    });
    for (index, (alt_id, alt_logprob)) in lp.top_logprobs.into_iter().enumerate() {
        if alt_id == token_id {
            continue;
        }
        entries.push(WireTokenLogprob {
            token_id: alt_id,
            logprob: alt_logprob,
            rank: (index + 1) as u32,
        });
    }
    Some(MaybeWireLogprobs::Direct(Logprobs {
        positions: vec![PositionLogprobs { entries }],
    }))
}

fn convert_sampling(params: &EngineCoreSamplingParams) -> SamplingParams {
    if params.temperature <= 0.0 {
        return SamplingParams {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: params.eos_token_id.is_none() && params.all_stop_token_ids.is_empty(),
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
        ignore_eos: params.eos_token_id.is_none() && params.all_stop_token_ids.is_empty(),
    }
}

fn requested_logprobs(params: &EngineCoreSamplingParams) -> usize {
    params
        .logprobs
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0)
}

fn convert_finish_reason(reason: FinishReason) -> EngineCoreFinishReason {
    match reason {
        FinishReason::Length => EngineCoreFinishReason::Length,
        FinishReason::Stop => EngineCoreFinishReason::Stop,
        FinishReason::Error => EngineCoreFinishReason::Error,
    }
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

fn local_ipc_namespace() -> Result<PathBuf> {
    let base_dir =
        std::env::var_os("PEGAINFER_IPC_DIR").map_or_else(|| PathBuf::from("/tmp"), PathBuf::from);
    let uuid = uuid::Uuid::new_v4().to_string();
    let path = base_dir.join(format!("pgi-{}-{}", std::process::id(), &uuid[..8]));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("failed to create IPC namespace {}", path.display()))?;
    Ok(path)
}

fn ipc_endpoint(namespace: &Path, name: &str) -> String {
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

fn load_max_model_len(model_path: &Path) -> Option<u32> {
    let content = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    serde_json::from_str::<ModelLenConfig>(&content)
        .ok()?
        .max_model_len()
}

pub fn shutdown_token_from_ctrl_c() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();
    tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!("failed to install CTRL+C handler: {error}");
        }
        shutdown.cancel();
    });
    token
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn to_wire_logprobs_returns_none_when_input_is_none() {
        assert!(to_wire_logprobs(7, None).is_none());
    }

    fn assert_logprob_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "logprob mismatch: actual={actual}, expected={expected}"
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
        let entries = &direct.positions[0].entries;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].token_id, 7);
        assert_logprob_eq(entries[0].logprob, -0.5);
        assert_eq!(entries[0].rank, 1);
        assert_eq!(entries[1].token_id, 42);
        assert_logprob_eq(entries[1].logprob, -1.5);
        assert_eq!(entries[1].rank, 2);
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
