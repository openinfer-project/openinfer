use std::collections::HashSet;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::Router;
use axum::middleware::from_fn_with_state;
use log::warn;
use serde::Deserialize;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::TransportMode;
use vllm_server::{
    ApiServerOptions, ChatTemplateContentFormatOption, Config, CoordinatorMode, CorsConfig,
    HttpListenerMode, ParserSelection, RendererSelection,
};

use openinfer_engine::engine::EngineHandle;

mod bridge;
mod lora;
mod sampling_guard;
mod wire;

use bridge::{LocalEngineBridge, ipc_endpoint, local_ipc_namespace};
use lora::{bad_request, load_startup_lora_modules, lora_openai_routes, lora_routes};
use sampling_guard::{ServableCap, guard_generation_request};

pub use lora::LoraModule;

const COMPLETION_ROUTE_BODY_LIMIT: usize = 2 * 1024 * 1024;

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

/// Serve while the engine is still loading: the HTTP frontend (tokenizer,
/// chat templates) starts immediately and the engine bridge attaches once
/// `engine` resolves. HTTP binds only after the bridge registers, so a
/// reachable port still means the engine is ready.
///
/// Pass `max_model_len: None` to read `max_position_embeddings` from
/// `model_path/config.json`; pass `Some(n)` when the path has no config
/// (e.g. a HuggingFace model id for the sim frontend).
pub async fn serve(
    engine: impl Future<Output = Result<EngineHandle>> + Send + 'static,
    model_path: &Path,
    served_model_name: Vec<String>,
    port: u16,
    max_model_len: Option<u32>,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_model_on_host(
        engine,
        model_path.to_string_lossy().into_owned(),
        served_model_name,
        "0.0.0.0".to_string(),
        port,
        resolve_max_model_len(model_path, max_model_len),
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
        std::future::ready(Ok(handle.clone())),
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

async fn serve_model_on_host(
    engine: impl Future<Output = Result<EngineHandle>> + Send + 'static,
    model_id: String,
    served_model_name: Vec<String>,
    host: String,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_model_on_host_with_router_extension(
        engine,
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
    engine: impl Future<Output = Result<EngineHandle>> + Send + 'static,
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

    // The HTTP server runs concurrently with the engine load: vllm-server
    // spends ~1s loading the tokenizer and chat templates before it waits for
    // an engine to register, so neither waits on the other. This task attaches
    // the bridge once the engine resolves and runs it to completion; on engine
    // failure it cancels the server so the error surfaces instead of hanging
    // in the registration wait.
    let servable_cap = ServableCap::default();
    let server_shutdown = shutdown.child_token();
    let bridge_shutdown = shutdown.child_token();
    let engine_task = tokio::spawn({
        let servable_cap = servable_cap.clone();
        let server_shutdown = server_shutdown.clone();
        let bridge_shutdown = bridge_shutdown.clone();
        let input_address = input_address.clone();
        let output_address = output_address.clone();
        async move {
            let handle = match engine.await {
                Ok(handle) => handle,
                Err(error) => {
                    server_shutdown.cancel();
                    return Err(error);
                }
            };
            let servable_limit = handle.servable_len().map(|cap| max_model_len.min(cap));
            servable_cap.set(servable_limit);
            let bridge = LocalEngineBridge {
                input_address,
                output_address,
                handle,
                max_model_len: servable_limit.unwrap_or(max_model_len),
            };
            if let Err(error) = bridge.run(bridge_shutdown).await {
                warn!("local vLLM engine bridge exited: {error:#}");
            }
            Ok(())
        }
    });

    let config = Config {
        transport_mode: TransportMode::Bootstrapped {
            input_address,
            output_address,
            engine_start_index: 0,
            engine_count: 1,
            // The in-process bridge registers once the engine future resolves,
            // so this bounds the whole engine load (multi-GPU MoE models take
            // minutes, cold starts longer). Load *failure* already cancels the
            // server via the engine task; this only catches a truly hung load.
            ready_timeout: Duration::from_mins(30),
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
        max_logprobs: None,
        language_model_only: true,
        cors: CorsConfig::default(),
        api_keys: Vec::new(),
        api_server_options: ApiServerOptions {
            enable_log_requests: true,
            enable_prompt_tokens_details: true,
            enable_request_id_headers: false,
        },
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(10),
    };

    let extend_router = move |router: Router| {
        extend_router(router).layer(from_fn_with_state(servable_cap, guard_generation_request))
    };
    let result =
        vllm_server::serve_with_router_extension(config, server_shutdown, extend_router).await;
    // Stop the bridge (no-op if the caller's shutdown already cancelled it),
    // then collect the engine task. If the server failed while the engine is
    // still loading, the uncancellable blocking load must finish first.
    bridge_shutdown.cancel();
    if result.is_err() && !engine_task.is_finished() {
        warn!("HTTP server failed; waiting for the in-flight engine load to finish before exit");
    }
    let result = match engine_task.await {
        Ok(Ok(())) => result,
        // Engine failed: the server saw a cancel and returned Ok — the engine
        // error is the one worth reporting.
        Ok(Err(engine_error)) => result.and(Err(engine_error)),
        Err(join_error) => result.and(Err(anyhow::anyhow!(
            "engine startup task panicked: {join_error}"
        ))),
    };
    let _ = std::fs::remove_dir_all(namespace);
    result
}

pub fn load_max_model_len(model_path: &Path) -> Option<u32> {
    let content = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    serde_json::from_str::<ModelLenConfig>(&content)
        .ok()?
        .max_model_len()
}

fn resolve_max_model_len(model_path: &Path, max_model_len: Option<u32>) -> u32 {
    max_model_len.unwrap_or_else(|| {
        load_max_model_len(model_path).unwrap_or_else(|| {
            const FALLBACK_MAX_MODEL_LEN: u32 = 4096;
            warn!(
                "max_position_embeddings not found in {}/config.json; capping max_model_len at {FALLBACK_MAX_MODEL_LEN}. \
                 Requests are limited to this length — set max_position_embeddings in the model config if it supports more.",
                model_path.display()
            );
            FALLBACK_MAX_MODEL_LEN
        })
    })
}

/// Cancel `token` on the first CTRL+C. Installing the handler replaces the
/// default SIGINT kill behavior — only call this once whatever the token
/// guards can actually wind down (e.g. after an uncancellable blocking engine
/// load has finished), otherwise CTRL+C turns into a no-op wait.
pub fn cancel_token_on_ctrl_c(token: &CancellationToken) {
    let shutdown = token.clone();
    tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!("failed to install CTRL+C handler: {error}");
        }
        shutdown.cancel();
    });
}

pub fn shutdown_token_from_ctrl_c() -> CancellationToken {
    let token = CancellationToken::new();
    cancel_token_on_ctrl_c(&token);
    token
}
