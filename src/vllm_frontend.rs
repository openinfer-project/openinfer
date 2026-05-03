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
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
    EngineCoreRequestType, EngineCoreSamplingParams, StopReason, UtilityOutput,
    UtilityResultEnvelope, encode_msgpack,
};
use vllm_engine_core_client::{EngineId, TransportMode};
use vllm_server::{
    ChatTemplateContentFormatOption, Config, CoordinatorMode, HttpListenerMode, ParserSelection,
    RendererSelection,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use crate::sampler::SamplingParams;
use pegainfer_core::engine::{EngineHandle, FinishReason, GenerateRequest, TokenEvent};

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
    let namespace = local_ipc_namespace()?;
    let input_address = ipc_endpoint(&namespace, "input.sock");
    let output_address = ipc_endpoint(&namespace, "output.sock");
    let max_model_len = load_max_model_len(model_path).unwrap_or(4096);

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
        model: model_path.to_string_lossy().into_owned(),
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
    result
}

async fn run_request_stream(
    request_id: String,
    mut token_rx: mpsc::UnboundedReceiver<TokenEvent>,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
) {
    while let Some(event) = token_rx.recv().await {
        match event {
            TokenEvent::Token { id, .. } => {
                if send_token_output(&output_tx, &request_id, id).is_err() {
                    return;
                }
            }
            TokenEvent::PromptTokens { .. } => {}
            TokenEvent::Finished { finish_reason, .. } => {
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    convert_finish_reason(finish_reason),
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
    token_id: u32,
) -> Result<()> {
    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            outputs: vec![engine_output(
                request_id.to_string(),
                vec![token_id],
                None,
                None,
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
                Some(finish_reason),
                stop_reason,
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
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason,
        events: None,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
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
    }
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

fn local_ipc_namespace() -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "pegainfer-vllm-engine-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
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
