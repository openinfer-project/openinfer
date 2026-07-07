use std::fs;
use std::net::TcpListener;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use openinfer_sim::{SimulatedEngineConfig, start_engine};
use reqwest::Client;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
const MODEL_NAME: &str = "openinfer-sim-e2e";
const SERVER_START_ATTEMPTS: usize = 5;

struct SimServer {
    base_url: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
    _model_dir: TempDir,
}

impl SimServer {
    async fn spawn() -> Result<Self> {
        Self::spawn_with_model_dir(model_dir_with_minimal_metadata()?).await
    }

    async fn spawn_with_lora_routes() -> Result<Self> {
        Self::spawn_with_model_dir_and_lora_routes(model_dir_with_minimal_metadata()?, true).await
    }

    async fn spawn_with_model_dir(model_dir: TempDir) -> Result<Self> {
        Self::spawn_with_model_dir_and_lora_routes(model_dir, false).await
    }

    async fn spawn_with_model_dir_and_lora_routes(
        model_dir: TempDir,
        enable_lora_routes: bool,
    ) -> Result<Self> {
        let mut last_error = None;
        for attempt in 1..=SERVER_START_ATTEMPTS {
            match Self::spawn_once(&model_dir, enable_lora_routes).await {
                Ok(started) => {
                    return Ok(Self {
                        base_url: started.base_url,
                        shutdown: started.shutdown,
                        task: started.task,
                        _model_dir: model_dir,
                    });
                }
                Err(error) => {
                    last_error = Some(error);
                    if attempt < SERVER_START_ATTEMPTS {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("sim frontend startup was not attempted")))
            .with_context(|| {
                format!("failed to start sim frontend after {SERVER_START_ATTEMPTS} attempts")
            })
    }

    async fn spawn_once(model_dir: &TempDir, enable_lora_routes: bool) -> Result<StartedSimServer> {
        let port = reserve_loopback_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let shutdown = CancellationToken::new();
        let engine = start_engine(SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?);
        let server_shutdown = shutdown.clone();
        let model_path = model_dir.path().to_string_lossy().into_owned();
        let model_path_buf = model_dir.path().to_path_buf();
        let mut task = tokio::spawn(async move {
            if enable_lora_routes {
                openinfer_vllm_frontend::serve_model_with_lora_routes(
                    engine,
                    model_path,
                    vec![MODEL_NAME.to_string()],
                    Vec::new(),
                    port,
                    128,
                    server_shutdown,
                )
                .await
            } else {
                openinfer_vllm_frontend::serve(
                    std::future::ready(Ok(engine)),
                    &model_path_buf,
                    vec![MODEL_NAME.to_string()],
                    port,
                    Some(128),
                    server_shutdown,
                )
                .await
            }
        });

        let client = test_client()?;
        let health_result = tokio::select! {
            result = wait_for_health(&client, &base_url) => result,
            result = &mut task => {
                return match result {
                    Ok(Ok(())) => Err(anyhow!("sim frontend exited before becoming healthy")),
                    Ok(Err(error)) => Err(error).context("sim frontend exited before becoming healthy"),
                    Err(error) => Err(error).context("sim frontend task panicked"),
                };
            }
        };

        if let Err(error) = health_result {
            shutdown.cancel();
            let _ = tokio::time::timeout(Duration::from_secs(10), task).await;
            return Err(error);
        }

        Ok(StartedSimServer {
            base_url,
            shutdown,
            task,
        })
    }

    async fn shutdown(self) -> Result<()> {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), self.task)
            .await
            .context("timed out waiting for sim frontend shutdown")?
            .context("sim frontend task panicked")?
    }
}

struct StartedSimServer {
    base_url: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
}

fn empty_model_dir() -> Result<TempDir> {
    tempfile::tempdir().context("failed to create temp model dir")
}

fn model_dir_with_minimal_metadata() -> Result<TempDir> {
    let dir = empty_model_dir()?;

    // The simulated frontend still builds the normal vLLM text/chat stack.
    // Token-id prompts avoid tokenizer encode work, but generated ids still
    // need a tokenizer for detokenization and a tiny config for metadata.
    fs::write(dir.path().join("tokenizer.json"), TINY_TOKENIZER_JSON)
        .context("failed to write tiny tokenizer.json")?;
    fs::write(
        dir.path().join("tokenizer_config.json"),
        TINY_TOKENIZER_CONFIG_JSON,
    )
    .context("failed to write tiny tokenizer_config.json")?;
    fs::write(dir.path().join("config.json"), TINY_CONFIG_JSON)
        .context("failed to write tiny config.json")?;

    Ok(dir)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_engine_serves_openai_completions_over_http() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_models_endpoint(&client, &server.base_url).await?;
    assert_non_streaming_completion_has_output(&client, &server.base_url).await?;
    assert_streaming_completion_emits_done(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_lora_routes_are_mounted_on_openai_frontend() -> Result<()> {
    let server = SimServer::spawn_with_lora_routes().await?;
    let client = test_client()?;

    let response = client
        .post(format!("{}/v1/load_lora_adapter", server.base_url))
        .json(&json!({
            "lora_name": "adapter-a",
            "lora_path": "/tmp/adapter-a"
        }))
        .send()
        .await?;

    let status = response.status();
    let body: Value = response.json().await?;
    if status != reqwest::StatusCode::NOT_FOUND {
        bail!("expected mounted LoRA route to report unsupported engine, got {status}: {body}");
    }
    let error = body["error"]
        .as_str()
        .ok_or_else(|| anyhow!("LoRA route response has no error string: {body}"))?;
    if !error.contains("dynamic LoRA adapter loading") {
        bail!("LoRA route returned unexpected error: {body}");
    }

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_streaming_completion_returns_nonempty_output_for_positive_max_tokens() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_non_streaming_completion_has_output(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_completion_emits_terminal_done() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_streaming_completion_emits_done(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_frontend_metadata_contract_is_executable() -> Result<()> {
    let model_dir = model_dir_with_minimal_metadata()?;
    for file in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
        if !model_dir.path().join(file).is_file() {
            bail!("minimal simulated frontend metadata fixture is missing {file}");
        }
    }

    let server = SimServer::spawn_with_model_dir(model_dir).await?;
    server.shutdown().await?;

    let error = match SimServer::spawn_with_model_dir(empty_model_dir()?).await {
        Ok(server) => {
            server.shutdown().await?;
            bail!("empty local model metadata directory should fail frontend startup");
        }
        Err(error) => error,
    };
    let message = format!("{error:#}");
    if !message.contains("supported tokenizer file") || !message.contains("tokenizer.json") {
        bail!("empty metadata dir failed with unexpected error: {message}");
    }

    Ok(())
}

async fn assert_models_endpoint(client: &Client, base_url: &str) -> Result<()> {
    let models: Value = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let advertised = models["data"]
        .as_array()
        .ok_or_else(|| anyhow!("/v1/models response has no data array"))?;
    if !advertised.iter().any(|model| model["id"] == MODEL_NAME) {
        bail!("/v1/models did not advertise {MODEL_NAME}: {models}");
    }

    Ok(())
}

async fn assert_non_streaming_completion_has_output(client: &Client, base_url: &str) -> Result<()> {
    let completion: Value = post_completion(client, base_url, false).await?;
    let text = completion["choices"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("non-streaming completion has no text: {completion}"))?;
    if text.is_empty() {
        bail!("non-streaming completion returned empty text for max_tokens > 0");
    }

    Ok(())
}

async fn assert_streaming_completion_emits_done(client: &Client, base_url: &str) -> Result<()> {
    let stream = post_completion_stream(client, base_url).await?;
    if !stream.lines().any(|line| line.trim() == "data: [DONE]") {
        bail!("streaming completion did not emit terminal data: [DONE]: {stream}");
    }

    Ok(())
}

async fn post_completion(client: &Client, base_url: &str, stream: bool) -> Result<Value> {
    let response = client
        .post(format!("{base_url}/v1/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(completion_body(stream).to_string())
        .send()
        .await?
        .error_for_status()?;
    response
        .json()
        .await
        .context("failed to parse non-streaming completion response")
}

async fn post_completion_stream(client: &Client, base_url: &str) -> Result<String> {
    client
        .post(format!("{base_url}/v1/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(completion_body(true).to_string())
        .send()
        .await?
        .error_for_status()?
        .text()
        .await
        .context("failed to read streaming completion response")
}

fn test_client() -> Result<Client> {
    Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .context("failed to build HTTP test client")
}

fn completion_body(stream: bool) -> Value {
    json!({
        "model": MODEL_NAME,
        "prompt": [1, 2],
        "max_tokens": 3,
        "temperature": 0.0,
        "ignore_eos": true,
        "stream": stream
    })
}

async fn wait_for_health(client: &Client, base_url: &str) -> Result<()> {
    let health_url = format!("{base_url}/health");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out waiting for sim frontend health at {health_url}");
        }

        match client
            .get(&health_url)
            .timeout(Duration::from_secs(1))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(_) | Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

fn reserve_loopback_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .context("failed to reserve loopback port for sim e2e test")?;
    Ok(listener.local_addr()?.port())
}

const TINY_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "<unk>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "alpha": 1,
      "beta": 2
    },
    "unk_token": "<unk>"
  }
}"#;

const TINY_TOKENIZER_CONFIG_JSON: &str = r#"{
  "unk_token": "<unk>",
  "tokenizer_class": "PreTrainedTokenizerFast"
}"#;

const TINY_CONFIG_JSON: &str = r#"{
  "model_type": "openinfer_sim",
  "max_position_embeddings": 128,
  "vocab_size": 3
}"#;
