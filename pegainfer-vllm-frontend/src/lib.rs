use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header::CONTENT_LENGTH};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::logprobs::{
    Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob as WireTokenLogprob,
};
use vllm_engine_core_client::protocol::utility::{
    UtilityCallId, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType, EngineCoreSamplingParams,
    ModelDtype, StopReason, encode_msgpack, stats::PrefillStats,
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
    EngineControlError, EngineHandle, FinishReason, GenerateRequest, LoadLoraAdapterRequest,
    TokenEvent, TokenLogprob, UnloadLoraAdapterRequest,
};
use pegainfer_engine::sampler::SamplingParams;

const ENGINE_INDEX: u32 = 0;
const LORA_ROUTE_BODY_LIMIT: usize = 128 * 1024 * 1024;
const COMPLETION_ROUTE_BODY_LIMIT: usize = 2 * 1024 * 1024;
const LORA_ADAPTER_XARG: &str = "pegainfer_lora_adapter";

#[derive(Clone)]
struct LoraRouteState {
    handle: EngineHandle,
    adapter_names: Arc<RwLock<HashSet<String>>>,
}

#[derive(Clone)]
struct LoraOpenAiState {
    vllm_router: Router,
    base_model_name: String,
    served_model_names: Vec<String>,
    adapter_names: Arc<RwLock<HashSet<String>>>,
}

#[derive(Debug, Deserialize)]
struct LoadLoraAdapterHttpRequest {
    lora_name: String,
    lora_path: PathBuf,
    #[serde(default)]
    load_inplace: bool,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

#[derive(Debug, Deserialize)]
struct UnloadLoraAdapterHttpRequest {
    lora_name: String,
    #[serde(default)]
    lora_int_id: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoraModule {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug, Serialize)]
struct ModelListBody {
    object: &'static str,
    data: Vec<ModelCardBody>,
}

#[derive(Debug, Serialize)]
struct ModelCardBody {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: &'static str,
}

pub fn lora_routes(handle: EngineHandle, adapter_names: Arc<RwLock<HashSet<String>>>) -> Router {
    Router::new()
        .route("/v1/load_lora_adapter", post(load_lora_adapter))
        .route("/v1/unload_lora_adapter", post(unload_lora_adapter))
        .with_state(LoraRouteState {
            handle,
            adapter_names,
        })
}

fn lora_openai_routes(
    vllm_router: Router,
    base_model_name: String,
    served_model_names: Vec<String>,
    adapter_names: Arc<RwLock<HashSet<String>>>,
) -> Router {
    Router::new()
        .route("/v1/models", get(lora_models))
        .route("/v1/completions", post(forward_lora_openai_request))
        .route("/v1/chat/completions", post(forward_lora_openai_request))
        .with_state(LoraOpenAiState {
            vllm_router,
            base_model_name,
            served_model_names,
            adapter_names,
        })
}

async fn load_lora_adapter(
    axum::extract::State(state): axum::extract::State<LoraRouteState>,
    Json(request): Json<LoadLoraAdapterHttpRequest>,
) -> Response {
    if request.lora_name.is_empty() {
        return bad_request("lora_name must not be empty");
    }
    if request.lora_path.as_os_str().is_empty() {
        return bad_request("lora_path must not be empty");
    }
    if request.is_3d_lora_weight {
        return bad_request("is_3d_lora_weight=true is not supported by Qwen3 LoRA PR1");
    }

    let lora_name = request.lora_name.clone();
    match state
        .handle
        .load_lora_adapter(LoadLoraAdapterRequest {
            lora_name: request.lora_name,
            lora_path: request.lora_path,
            load_inplace: request.load_inplace,
        })
        .await
    {
        Ok(()) => {
            state.adapter_names.write().await.insert(lora_name.clone());
            (
                StatusCode::OK,
                format!("Success: LoRA adapter '{lora_name}' added successfully."),
            )
                .into_response()
        }
        Err(EngineControlError::Unsupported(message)) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: message.to_string(),
            }),
        )
            .into_response(),
        Err(EngineControlError::ChannelClosed) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: EngineControlError::ChannelClosed.to_string(),
            }),
        )
            .into_response(),
        Err(EngineControlError::OperationFailed(message)) => {
            (StatusCode::BAD_REQUEST, Json(ErrorBody { error: message })).into_response()
        }
    }
}

async fn unload_lora_adapter(
    axum::extract::State(state): axum::extract::State<LoraRouteState>,
    Json(request): Json<UnloadLoraAdapterHttpRequest>,
) -> Response {
    if request.lora_name.is_empty() {
        return bad_request("lora_name must not be empty");
    }

    let lora_name = request.lora_name.clone();
    match state
        .handle
        .unload_lora_adapter(UnloadLoraAdapterRequest {
            lora_name: request.lora_name,
            lora_int_id: request.lora_int_id,
        })
        .await
    {
        Ok(()) => {
            state.adapter_names.write().await.remove(&lora_name);
            (
                StatusCode::OK,
                format!("Success: LoRA adapter '{lora_name}' removed successfully."),
            )
                .into_response()
        }
        Err(EngineControlError::Unsupported(message)) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: message.to_string(),
            }),
        )
            .into_response(),
        Err(EngineControlError::ChannelClosed) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: EngineControlError::ChannelClosed.to_string(),
            }),
        )
            .into_response(),
        Err(EngineControlError::OperationFailed(message)) => {
            (StatusCode::BAD_REQUEST, Json(ErrorBody { error: message })).into_response()
        }
    }
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorBody {
            error: message.into(),
        }),
    )
        .into_response()
}

/// Rejects `max_tokens` over the cap. A too-long prompt may still 500.
async fn enforce_capacity(State(servable): State<u32>, request: Request, next: Next) -> Response {
    #[derive(Deserialize)]
    struct Probe {
        max_tokens: Option<u64>,
    }
    let path = request.uri().path();
    if path != "/v1/completions" && path != "/v1/chat/completions" {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let Ok(bytes) = to_bytes(body, COMPLETION_ROUTE_BODY_LIMIT).await else {
        return bad_request("failed to read request body");
    };
    let over = serde_json::from_slice::<Probe>(&bytes)
        .ok()
        .and_then(|probe| probe.max_tokens)
        .filter(|&max_tokens| max_tokens > u64::from(servable));
    if let Some(max_tokens) = over {
        warn!("rejecting request: max_tokens {max_tokens} exceeds servable cap {servable}");
        return bad_request(format!(
            "max_tokens ({max_tokens}) exceeds the model's maximum context length of {servable} tokens"
        ));
    }
    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

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
            dtype: ModelDtype::BFloat16,
            vllm_version: "pegainfer-local-bridge".to_string(),
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
            send_terminal_output(output_tx, request_id, EngineCoreFinishReason::Error, None)?;
            return Ok(());
        };
        let Some(sampling_params) = sampling_params else {
            warn!("request {request_id} dropped: missing sampling_params");
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

pub async fn serve(
    handle: EngineHandle,
    model_path: &Path,
    served_model_name: Option<&str>,
    port: u16,
    shutdown: CancellationToken,
) -> Result<()> {
    let max_model_len = load_max_model_len(model_path).unwrap_or_else(|| {
        const FALLBACK_MAX_MODEL_LEN: u32 = 4096;
        warn!(
            "max_position_embeddings not found in {}/config.json; capping max_model_len at {FALLBACK_MAX_MODEL_LEN}. \
             Requests are limited to this length — set max_position_embeddings in the model config if it supports more.",
            model_path.display()
        );
        FALLBACK_MAX_MODEL_LEN
    });
    serve_model(
        handle,
        model_path.to_string_lossy().into_owned(),
        served_model_name
            .into_iter()
            .map(|name| name.to_string())
            .collect(),
        port,
        max_model_len,
        shutdown,
    )
    .await
}

pub async fn serve_model(
    handle: EngineHandle,
    model_id: impl Into<String>,
    served_model_name: Vec<String>,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
) -> Result<()> {
    let model_id = model_id.into();
    serve_model_on_host(
        handle,
        model_id,
        served_model_name,
        "0.0.0.0".to_string(),
        port,
        max_model_len,
        shutdown,
    )
    .await
}

pub async fn serve_model_with_lora_routes(
    handle: EngineHandle,
    model_id: impl Into<String>,
    served_model_name: Vec<String>,
    lora_modules: Vec<LoraModule>,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
) -> Result<()> {
    let model_id = model_id.into();
    let adapter_names = Arc::new(RwLock::new(HashSet::new()));
    load_startup_lora_modules(&handle, &adapter_names, &lora_modules).await?;
    let base_model_name = served_model_name
        .first()
        .cloned()
        .unwrap_or_else(|| model_id.clone());
    serve_model_on_host_with_router_extension(
        handle.clone(),
        model_id,
        served_model_name.clone(),
        "0.0.0.0".to_string(),
        port,
        max_model_len,
        shutdown,
        move |router| {
            let lora_router = lora_routes(handle.clone(), Arc::clone(&adapter_names));
            let openai_router = lora_openai_routes(
                router.clone(),
                base_model_name,
                served_model_name,
                Arc::clone(&adapter_names),
            );
            openai_router.merge(lora_router).fallback_service(router)
        },
    )
    .await
}

async fn load_startup_lora_modules(
    handle: &EngineHandle,
    adapter_names: &Arc<RwLock<HashSet<String>>>,
    lora_modules: &[LoraModule],
) -> Result<()> {
    for module in lora_modules {
        handle
            .load_lora_adapter(LoadLoraAdapterRequest {
                lora_name: module.name.clone(),
                lora_path: module.path.clone(),
                load_inplace: false,
            })
            .await
            .with_context(|| {
                format!(
                    "failed to load startup LoRA module {} from {}",
                    module.name,
                    module.path.display()
                )
            })?;
        adapter_names.write().await.insert(module.name.clone());
    }
    Ok(())
}

async fn serve_model_on_host(
    handle: EngineHandle,
    model_id: String,
    served_model_name: Vec<String>,
    host: String,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_model_on_host_with_router_extension(
        handle,
        model_id,
        served_model_name,
        host,
        port,
        max_model_len,
        shutdown,
        |router| router,
    )
    .await
}

async fn serve_model_on_host_with_router_extension<F>(
    handle: EngineHandle,
    model_id: String,
    served_model_name: Vec<String>,
    host: String,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
    extend_router: F,
) -> Result<()>
where
    F: FnOnce(Router) -> Router,
{
    let namespace = local_ipc_namespace()?;
    let input_address = ipc_endpoint(&namespace, "input.sock");
    let output_address = ipc_endpoint(&namespace, "output.sock");

    let servable_limit = handle.servable_len().map(|cap| max_model_len.min(cap));
    let max_model_len = servable_limit.unwrap_or(max_model_len);
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
        served_model_name,
        listener_mode: HttpListenerMode::BindTcp { host, port },
        tool_call_parser: ParserSelection::default(),
        reasoning_parser: ParserSelection::default(),
        renderer: RendererSelection::default(),
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: ChatTemplateContentFormatOption::default(),
        enable_log_requests: true,
        enable_request_id_headers: false,
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(10),
    };

    let extend_router = move |router: Router| {
        let router = extend_router(router);
        match servable_limit {
            Some(limit) => router.layer(from_fn_with_state(limit, enforce_capacity)),
            None => router,
        }
    };
    let result = vllm_server::serve_with_router_extension(config, shutdown, extend_router).await;
    bridge_task.abort();
    let _ = std::fs::remove_dir_all(namespace);
    result
}

async fn lora_models(State(state): State<LoraOpenAiState>) -> Response {
    lora_models_response(
        &state.served_model_names,
        &state.base_model_name,
        &state.adapter_names,
    )
    .await
}

async fn forward_lora_openai_request(
    State(state): State<LoraOpenAiState>,
    request: Request,
) -> Response {
    match forward_lora_openai_request_inner(state, request).await {
        Ok(response) => response,
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: format!("failed to prepare LoRA request: {error:#}"),
            }),
        )
            .into_response(),
    }
}

async fn forward_lora_openai_request_inner(
    state: LoraOpenAiState,
    request: Request,
) -> Result<Response> {
    let (mut parts, body) = request.into_parts();
    let mut body = to_bytes(body, LORA_ROUTE_BODY_LIMIT)
        .await
        .context("failed to read OpenAI request body")?;
    let _ =
        rewrite_lora_request_body(&mut body, &state.base_model_name, &state.adapter_names).await?;
    parts.headers.insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&body.len().to_string())
            .context("rewritten body length is invalid")?,
    );

    state
        .vllm_router
        .oneshot(Request::from_parts(parts, Body::from(body)))
        .await
        .context("vLLM router failed to handle LoRA request")
}

async fn rewrite_lora_request_body(
    body: &mut Bytes,
    base_model_name: &str,
    adapter_names: &Arc<RwLock<HashSet<String>>>,
) -> Result<Option<String>> {
    let mut value: serde_json::Value =
        serde_json::from_slice(body).context("failed to parse OpenAI request JSON")?;
    let Some(model) = value.get("model").and_then(serde_json::Value::as_str) else {
        return Ok(None);
    };
    if model == base_model_name {
        return Ok(None);
    }
    if !adapter_names.read().await.contains(model) {
        return Ok(None);
    }
    let adapter = model.to_string();
    value["model"] = serde_json::Value::String(base_model_name.to_string());
    let Some(map) = value.as_object_mut() else {
        return Ok(None);
    };
    let xargs = map
        .entry("vllm_xargs")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !xargs.is_object() {
        *xargs = serde_json::Value::Object(serde_json::Map::new());
    }
    xargs
        .as_object_mut()
        .expect("vllm_xargs must be object")
        .insert(
            LORA_ADAPTER_XARG.to_string(),
            serde_json::Value::String(adapter.clone()),
        );
    *body = Bytes::from(serde_json::to_vec(&value)?);
    Ok(Some(adapter))
}

async fn lora_models_response(
    served_model_names: &[String],
    base_model_name: &str,
    adapter_names: &Arc<RwLock<HashSet<String>>>,
) -> Response {
    let mut ids: Vec<String> = if served_model_names.is_empty() {
        vec![base_model_name.to_string()]
    } else {
        served_model_names.to_vec()
    };
    ids.extend(adapter_names.read().await.iter().cloned());
    ids.sort();
    ids.dedup();
    Json(ModelListBody {
        object: "list",
        data: ids
            .into_iter()
            .map(|id| ModelCardBody {
                id,
                object: "model",
                created: 0,
                owned_by: "vllm-frontend-rs",
            })
            .collect(),
    })
    .into_response()
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
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    convert_finish_reason(finish_reason),
                    None,
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
    call_id: UtilityCallId,
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
                    has_logprobs.then(|| MaybeWireLogprobs::Direct(Logprobs { positions })),
                    Some(other),
                );
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                return (
                    token_ids,
                    has_logprobs.then(|| MaybeWireLogprobs::Direct(Logprobs { positions })),
                    None,
                );
            }
        }
    }
}

#[cfg(test)]
fn to_wire_logprobs(token_id: u32, logprob: Option<TokenLogprob>) -> Option<MaybeWireLogprobs> {
    let position = to_wire_position_logprobs(token_id, logprob)?;
    Some(MaybeWireLogprobs::Direct(Logprobs {
        positions: vec![position],
    }))
}

fn to_wire_position_logprobs(
    token_id: u32,
    logprob: Option<TokenLogprob>,
) -> Option<PositionLogprobs> {
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
    Some(PositionLogprobs { entries })
}

fn convert_sampling(params: &EngineCoreSamplingParams) -> SamplingParams {
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
        ignore_eos,
    }
}

fn requested_logprobs(params: &EngineCoreSamplingParams) -> usize {
    params
        .logprobs
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0)
}

fn lora_adapter_from_sampling_params(params: &EngineCoreSamplingParams) -> Result<Option<String>> {
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

pub fn load_max_model_len(model_path: &Path) -> Option<u32> {
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

    fn route_state(handle: EngineHandle) -> LoraRouteState {
        LoraRouteState {
            handle,
            adapter_names: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    #[tokio::test]
    async fn load_lora_adapter_route_reports_unsupported_engine() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
        let state = route_state(EngineHandle::new(submit_tx));
        let response = load_lora_adapter(
            axum::extract::State(state),
            Json(LoadLoraAdapterHttpRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: PathBuf::from("/tmp/adapter-a"),
                load_inplace: false,
                is_3d_lora_weight: false,
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn load_lora_adapter_route_rejects_pr1_unsupported_fields() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
        let state = route_state(EngineHandle::new(submit_tx));
        let response = load_lora_adapter(
            axum::extract::State(state),
            Json(LoadLoraAdapterHttpRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: PathBuf::from("/tmp/adapter-a"),
                load_inplace: false,
                is_3d_lora_weight: true,
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unload_lora_adapter_route_reports_unsupported_engine() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
        let state = route_state(EngineHandle::new(submit_tx));
        let response = unload_lora_adapter(
            axum::extract::State(state),
            Json(UnloadLoraAdapterHttpRequest {
                lora_name: "adapter-a".to_string(),
                lora_int_id: None,
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rewrite_lora_request_body_maps_adapter_model_to_base_and_xarg() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from(["adapter-a".to_string()])));
        let mut body = Bytes::from(
            serde_json::json!({
                "model": "adapter-a",
                "prompt": "hello"
            })
            .to_string(),
        );

        let selected = rewrite_lora_request_body(&mut body, "base-model", &adapter_names)
            .await
            .expect("rewrite request");

        assert_eq!(selected.as_deref(), Some("adapter-a"));
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(value["model"], "base-model");
        assert_eq!(value["prompt"], "hello");
        assert_eq!(value["vllm_xargs"][LORA_ADAPTER_XARG], "adapter-a");
    }

    #[tokio::test]
    async fn rewrite_lora_request_body_leaves_base_and_unknown_models_untouched() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from(["adapter-a".to_string()])));
        let mut base_body = Bytes::from(r#"{"model":"base-model","prompt":"hello"}"#);
        let selected = rewrite_lora_request_body(&mut base_body, "base-model", &adapter_names)
            .await
            .expect("base request");
        assert_eq!(selected, None);
        assert_eq!(
            &base_body[..],
            br#"{"model":"base-model","prompt":"hello"}"#
        );

        let mut unknown_body = Bytes::from(r#"{"model":"missing-adapter","prompt":"hello"}"#);
        let selected = rewrite_lora_request_body(&mut unknown_body, "base-model", &adapter_names)
            .await
            .expect("unknown adapter request");
        assert_eq!(selected, None);
        assert_eq!(
            &unknown_body[..],
            br#"{"model":"missing-adapter","prompt":"hello"}"#
        );
    }

    #[tokio::test]
    async fn lora_openai_forwarder_rewrites_then_calls_vllm_router() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from(["adapter-a".to_string()])));
        let vllm_router = Router::new().route(
            "/v1/completions",
            post(|headers: axum::http::HeaderMap, body: Bytes| async move {
                let content_length = headers
                    .get(CONTENT_LENGTH)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<usize>().ok())
                    .expect("content-length header");
                assert_eq!(content_length, body.len());
                Json(serde_json::from_slice::<serde_json::Value>(&body).expect("json body"))
            }),
        );
        let state = LoraOpenAiState {
            vllm_router,
            base_model_name: "base-model".to_string(),
            served_model_names: vec!["base-model".to_string()],
            adapter_names,
        };
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .body(Body::from(
                serde_json::json!({
                    "model": "adapter-a",
                    "prompt": "hello"
                })
                .to_string(),
            ))
            .expect("request");

        let response = forward_lora_openai_request_inner(state, request)
            .await
            .expect("forward request");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), LORA_ROUTE_BODY_LIMIT)
            .await
            .expect("read body");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json response");
        assert_eq!(value["model"], "base-model");
        assert_eq!(value["prompt"], "hello");
        assert_eq!(value["vllm_xargs"][LORA_ADAPTER_XARG], "adapter-a");
    }

    #[tokio::test]
    async fn lora_models_response_includes_base_and_loaded_adapters() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from([
            "adapter-b".to_string(),
            "adapter-a".to_string(),
        ])));

        let response =
            lora_models_response(&["served-base".to_string()], "model-path", &adapter_names).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), LORA_ROUTE_BODY_LIMIT)
            .await
            .expect("read body");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("models JSON");
        let ids = value["data"]
            .as_array()
            .expect("data array")
            .iter()
            .map(|entry| entry["id"].as_str().expect("id string"))
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["adapter-a", "adapter-b", "served-base"]);
    }

    #[test]
    fn convert_sampling_honors_ignore_eos_lowering() {
        // ignore_eos=true lowering: _eos_token_id=None while
        // _all_stop_token_ids still carries the model EOS set.
        let mut params = EngineCoreSamplingParams::for_test();
        params.all_stop_token_ids = BTreeSet::from([163586]);
        assert!(convert_sampling(&params).ignore_eos);

        // Normal request: _eos_token_id present.
        params.eos_token_id = Some(163586);
        assert!(!convert_sampling(&params).ignore_eos);

        // Explicit client stop tokens keep EOS detection on even when the
        // frontend dropped _eos_token_id.
        params.eos_token_id = None;
        params.stop_token_ids = vec![42];
        assert!(!convert_sampling(&params).ignore_eos);
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
        assert!(first_batch.outputs[0].prefill_stats.is_some());
        assert!(second_batch.outputs[0].events.is_none());
        assert!(second_batch.outputs[0].prefill_stats.is_none());
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
