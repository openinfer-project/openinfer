use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use pegainfer_sim::{SimulatedEngineConfig, start_engine};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const MODEL_NAME: &str = "pegainfer-sim-e2e";
static TEMP_MODEL_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

struct SimServer {
    base_url: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
    _model_dir: TempModelDir,
}

impl SimServer {
    async fn spawn() -> Result<Self> {
        Self::spawn_with_model_dir(TempModelDir::with_minimal_metadata()?).await
    }

    async fn spawn_with_model_dir(model_dir: TempModelDir) -> Result<Self> {
        let port = reserve_loopback_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let shutdown = CancellationToken::new();
        let engine = start_engine(SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?);
        let server_shutdown = shutdown.clone();
        let model_path = model_dir.path.to_string_lossy().into_owned();
        let mut task = tokio::spawn(async move {
            pegainfer_vllm_frontend::serve_model(
                engine,
                model_path,
                vec![MODEL_NAME.to_string()],
                port,
                128,
                server_shutdown,
            )
            .await
        });

        let client = Client::new();
        tokio::select! {
            result = wait_for_health(&client, &base_url) => result?,
            result = &mut task => {
                return match result {
                    Ok(Ok(())) => Err(anyhow!("sim frontend exited before becoming healthy")),
                    Ok(Err(error)) => Err(error).context("sim frontend exited before becoming healthy"),
                    Err(error) => Err(error).context("sim frontend task panicked"),
                };
            }
        }

        Ok(Self {
            base_url,
            shutdown,
            task,
            _model_dir: model_dir,
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

struct TempModelDir {
    path: PathBuf,
}

impl TempModelDir {
    fn empty() -> Result<Self> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before Unix epoch")?
            .as_nanos();
        let sequence = TEMP_MODEL_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "pegainfer-sim-e2e-{}-{now}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path)
            .with_context(|| format!("failed to create temp model dir {}", path.display()))?;

        Ok(Self { path })
    }

    fn with_minimal_metadata() -> Result<Self> {
        let dir = Self::empty()?;

        // The simulated frontend still builds the normal vLLM text/chat stack.
        // Token-id prompts avoid tokenizer encode work, but generated ids still
        // need a tokenizer for detokenization and a tiny config for metadata.
        fs::write(dir.path.join("tokenizer.json"), TINY_TOKENIZER_JSON)
            .context("failed to write tiny tokenizer.json")?;
        fs::write(
            dir.path.join("tokenizer_config.json"),
            TINY_TOKENIZER_CONFIG_JSON,
        )
        .context("failed to write tiny tokenizer_config.json")?;
        fs::write(dir.path.join("config.json"), TINY_CONFIG_JSON)
            .context("failed to write tiny config.json")?;

        Ok(dir)
    }
}

impl Drop for TempModelDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_engine_serves_openai_completions_over_http() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = Client::new();

    assert_models_endpoint(&client, &server.base_url).await?;
    assert_non_streaming_completion_has_output(&client, &server.base_url).await?;
    assert_streaming_completion_emits_done(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_streaming_completion_returns_nonempty_output_for_positive_max_tokens() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = Client::new();

    assert_non_streaming_completion_has_output(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_completion_emits_terminal_done() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = Client::new();

    assert_streaming_completion_emits_done(&client, &server.base_url).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_frontend_metadata_contract_is_executable() -> Result<()> {
    let model_dir = TempModelDir::with_minimal_metadata()?;
    for file in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
        if !model_dir.path.join(file).is_file() {
            bail!("minimal simulated frontend metadata fixture is missing {file}");
        }
    }

    let server = SimServer::spawn_with_model_dir(model_dir).await?;
    server.shutdown().await?;

    let error = match SimServer::spawn_with_model_dir(TempModelDir::empty()?).await {
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
  "model_type": "pegainfer_sim",
  "max_position_embeddings": 128
}"#;
