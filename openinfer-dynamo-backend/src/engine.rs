//! `OpeninferBackend` — adapts openinfer's `EngineHandle` to Dynamo's
//! `LLMEngine`, so a pure-Rust openinfer worker plugs into a Dynamo frontend +
//! KV router. Run one process per GPU (`--device-ordinal 0..N`); the router
//! fans requests across the replicas.
//!
//! M1 scope: `start` (load Qwen3) / `generate` (stream tokens) / `cleanup`.
//! The engine advertises its real KV block size + capacity, so KV-aware
//! routing is well-defined — with no KV events yet the router's radix tree is
//! empty, every prefix misses, and routing falls back to load / round-robin.
//! The load signal (`setup_metrics`) and KV-event publishing
//! (`kv_event_sources`) are later milestones and use the trait defaults here.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, CommonArgs, DynamoError, EngineConfig, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, PreprocessedRequest, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use openinfer_engine::engine::{EngineHandle, GenerateRequest, TokenSink, TokenStreamReceiver};
use openinfer_qwen3_4b::{
    DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
    DEFAULT_MAX_PREFILL_TOKENS, DecodeOverlap, Qwen3LaunchOptions, Qwen3MemoryOptions,
    Qwen3OffloadOptions,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::convert::{self, Mapped};

/// Single-rank worker: one Qwen3 instance per process, always dp_rank 0.
const DP_RANK: u32 = 0;

#[derive(clap::Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "openinfer (Qwen3) backend worker for a Dynamo frontend + KV router."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Local path to the Qwen3 model directory (weights + tokenizer + chat
    /// template). The Dynamo frontend reads the tokenizer/template from here.
    #[arg(long)]
    model_path: PathBuf,

    /// Public model name advertised to clients. Defaults to the model dir name.
    #[arg(long)]
    served_model_name: Option<String>,

    /// CUDA device ordinal this worker loads on (run one process per GPU).
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,

    /// Disable CUDA Graph capture for decode (capture is on by default).
    #[arg(long, default_value_t = false)]
    no_cuda_graph: bool,

    /// Fraction of GPU memory the engine may use (weights + KV cache).
    #[arg(long, default_value_t = DEFAULT_GPU_MEMORY_UTILIZATION)]
    gpu_memory_utilization: f64,
}

pub struct OpeninferBackend {
    model_path: PathBuf,
    served_model_name: String,
    launch: Qwen3LaunchOptions,
    /// Set by `start`, cleared by `cleanup`. Interior mutability because every
    /// `LLMEngine` method takes `&self`. `EngineHandle` is itself an `Arc`
    /// clone, so `generate` clones cheaply; the stored copy is the last clone,
    /// and dropping it closes the submit channel that signals the (detached)
    /// scheduler thread to finish.
    handle: Mutex<Option<EngineHandle>>,
    /// Fired by `cleanup`; every in-flight `generate` stream selects on it so
    /// shutdown yields a clean `Cancelled` terminal instead of racing the
    /// channel-close path into a spurious "stream incomplete" error.
    cancel: CancellationToken,
}

impl OpeninferBackend {
    /// Parse process argv into the backend + its Dynamo `WorkerConfig`.
    pub fn from_args() -> Result<(Self, WorkerConfig), DynamoError> {
        let args = <Args as clap::Parser>::try_parse()
            .map_err(|e| convert::invalid_argument(e.to_string()))?;
        Self::from_parsed(args)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        let served_model_name = args.served_model_name.clone().unwrap_or_else(|| {
            args.model_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "openinfer".to_string())
        });

        let memory = Qwen3MemoryOptions::new(
            args.gpu_memory_utilization,
            DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
        )
        .validate()
        .map_err(|e| convert::invalid_argument(format!("invalid memory options: {e:#}")))?;

        let launch = Qwen3LaunchOptions {
            device_ordinal: args.device_ordinal,
            tp_size: 1,
            cuda_graph: !args.no_cuda_graph,
            offload: Qwen3OffloadOptions::disabled(),
            // Keep the prefix cache on: it is both a single-worker win and the
            // source of the KV events a later milestone will publish to the
            // router for cache-aware routing.
            no_prefix_cache: false,
            max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
            memory,
            lora: None,
            decode_overlap: DecodeOverlap::Off,
            batch_invariant: false,
        };

        let backend = OpeninferBackend {
            model_path: args.model_path.clone(),
            served_model_name: served_model_name.clone(),
            launch,
            handle: Mutex::new(None),
            cancel: CancellationToken::new(),
        };

        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            disaggregation_mode: args.common.disaggregation_mode,
            model_name: args.model_path.to_string_lossy().into_owned(),
            served_model_name: Some(served_model_name),
            ..Default::default()
        };

        Ok((backend, config))
    }

    fn handle(&self) -> std::sync::MutexGuard<'_, Option<EngineHandle>> {
        self.handle.lock().expect("engine handle mutex poisoned")
    }
}

#[async_trait]
impl LLMEngine for OpeninferBackend {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.handle().is_some() {
            return Err(convert::engine_shutdown(
                "openinfer backend already started",
            ));
        }

        tracing::info!(
            model_path = %self.model_path.display(),
            device_ordinal = self.launch.device_ordinal,
            cuda_graph = self.launch.cuda_graph,
            "loading Qwen3 engine (weights -> GPU); this can take a while"
        );

        // The model load is blocking (weights -> GPU, kernel warmup, optional
        // graph capture) and must not stall the runtime's reactor.
        let model_path = self.model_path.clone();
        let launch = self.launch;
        let handle =
            tokio::task::spawn_blocking(move || openinfer_qwen3_4b::launch(&model_path, launch))
                .await
                .map_err(|e| convert::backend_error(format!("engine loader thread panicked: {e}")))?
                .map_err(|e| convert::backend_error(format!("Qwen3 engine load failed: {e:#}")))?;

        let kv = handle.kv_capacity();
        let context_length = handle.servable_len();
        tracing::info!(
            context_length = ?context_length,
            kv_block_size = ?kv.map(|k| k.block_size),
            total_kv_blocks = ?kv.map(|k| k.total_blocks),
            "Qwen3 engine loaded; ready to serve"
        );

        *self.handle() = Some(handle);

        Ok(EngineConfig {
            model: self.served_model_name.clone(),
            served_model_name: Some(self.served_model_name.clone()),
            runtime_data: Default::default(),
            llm: Some(LlmRegistration {
                context_length,
                kv_cache_block_size: kv.map(|k| k.block_size as u32),
                total_kv_blocks: kv.map(|k| k.total_blocks as u64),
                max_num_seqs: None,
                max_num_batched_tokens: Some(self.launch.max_prefill_tokens as u64),
                data_parallel_size: Some(1),
                data_parallel_start_rank: Some(DP_RANK),
                // openinfer uses no Dynamo-handshake KV transport.
                bootstrap_host: None,
                bootstrap_port: None,
            }),
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let handle = self
            .handle()
            .clone()
            .ok_or_else(|| convert::engine_shutdown("generate called before start"))?;

        let prompt_tokens = request.token_ids.len() as u32;
        let params = convert::to_sampling_params(&request);
        let max_tokens = convert::resolve_max_tokens(&request);

        // Per-request private channel + cancel flag. openinfer's scheduler
        // learns to retire this request by observing the flag (its next emit
        // sees the sink closed) — the reactive abort the engine is built
        // around. `TokenSink::standalone()` hard-codes a never-tripped flag,
        // so we build the sink by hand to own the flag. The channel is
        // unbounded, but each request emits at most `max_tokens` items before
        // its terminal, so growth is bounded by the token cap, not by consumer
        // backpressure.
        let (tx, rx): (_, TokenStreamReceiver) = mpsc::unbounded_channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let tag: Arc<str> = Arc::from(ctx.id());
        let sink = TokenSink::new(tag, tx, cancelled.clone());

        let req = GenerateRequest {
            request_id: Some(ctx.id().to_string()),
            queued_at_unix_s: request.request_timestamp_ms.map(|ms| ms / 1000.0),
            prompt_tokens: request.token_ids,
            params,
            max_tokens,
            lora_adapter: None,
            token_tx: sink,
            // M1 does not surface per-token logprobs (the Dynamo `log_probs`
            // slot stays None), so pin 0 rather than make openinfer pay the
            // full-vocab O(V) logprob pass for a value we would then drop.
            logprobs: 0,
            echo: false,
        };

        if handle.submit(req).is_err() {
            return Err(convert::engine_shutdown(
                "openinfer engine is not accepting requests",
            ));
        }

        Ok(Box::pin(token_stream(
            rx,
            cancelled,
            ctx.inner_arc(),
            self.cancel.clone(),
            prompt_tokens,
        )))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        // Cancel first so in-flight `generate` streams take their
        // `cancel.cancelled()` arm and yield a clean Cancelled terminal. Then
        // drop the stored handle: that closes the submit channel, which is how
        // the scheduler thread learns to finish. qwen3's `EngineHandle` carries
        // no join handle — the scheduler thread is spawned detached in
        // `scheduler::start_with_executor` — so the drop is non-blocking and
        // there is no synchronous engine teardown to await; the detached thread
        // drains its current step and exits once the channel is closed.
        // Idempotent: a second call sees an already-cancelled token and a
        // `None` handle.
        self.cancel.cancel();
        let _ = self.handle().take();
        tracing::info!("openinfer backend: cleanup complete");
        Ok(())
    }
}

/// The `generate` response stream: drain the engine's per-request channel,
/// mapping each `TokenEvent` to a Dynamo chunk, and close on the first terminal
/// — or on cancellation, whichever comes first.
fn token_stream(
    mut rx: TokenStreamReceiver,
    cancelled: Arc<AtomicBool>,
    ctx: Arc<dyn AsyncEngineContext>,
    cancel: CancellationToken,
    prompt_tokens: u32,
) -> impl futures::Stream<Item = Result<LLMEngineOutput, DynamoError>> {
    async_stream::stream! {
        let mut completion_tokens: u32 = 0;
        loop {
            // `biased` is load-bearing: when both a cancel and a pending token
            // are ready we must prefer cancellation (yield Cancelled, not one
            // more token), and on shutdown we must beat the `rx.recv() -> None`
            // close so it never reads as a "stream incomplete" error. This also
            // relabels Cancelled an already-buffered natural terminal that lands
            // in the same poll as a cancel — only reachable at cleanup/shutdown,
            // where Cancelled is the right answer anyway.
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    cancelled.store(true, Ordering::Release);
                    yield Ok(LLMEngineOutput::cancelled()
                        .with_usage(usage(prompt_tokens, completion_tokens)));
                    break;
                }
                _ = ctx.stopped() => {
                    cancelled.store(true, Ordering::Release);
                    yield Ok(LLMEngineOutput::cancelled()
                        .with_usage(usage(prompt_tokens, completion_tokens)));
                    break;
                }
                recv = rx.recv() => {
                    let Some((_tag, event)) = recv else {
                        yield Err(convert::stream_incomplete());
                        break;
                    };
                    match convert::map_token_event(event) {
                        Mapped::Chunk(c) => { completion_tokens += 1; yield Ok(c); }
                        Mapped::Terminal(t) => { yield Ok(t); break; }
                        Mapped::Fail(e) => { yield Err(e); break; }
                        Mapped::Ignore => {}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    #[test]
    fn served_model_name_defaults_to_dir_name() {
        let (backend, config) = OpeninferBackend::from_parsed(
            Args::try_parse_from(["bin", "--model-path", "/data/models/Qwen3-4B"]).unwrap(),
        )
        .unwrap();
        assert_eq!(backend.served_model_name, "Qwen3-4B");
        assert_eq!(config.served_model_name.as_deref(), Some("Qwen3-4B"));
        assert_eq!(config.model_name, "/data/models/Qwen3-4B");
        // Worker starts unloaded; start() populates the handle.
        assert!(backend.handle().is_none());
    }

    #[test]
    fn explicit_served_model_name_and_common_args_flow_through() {
        let (_backend, config) = OpeninferBackend::from_parsed(
            Args::try_parse_from([
                "bin",
                "--model-path",
                "/m",
                "--served-model-name",
                "qwen3-4b",
                "--namespace",
                "prod",
                "--component",
                "worker",
            ])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(config.served_model_name.as_deref(), Some("qwen3-4b"));
        assert_eq!(config.namespace, "prod");
        assert_eq!(config.component, "worker");
    }

    #[test]
    fn no_cuda_graph_flag_disables_capture() {
        let (backend, _) = OpeninferBackend::from_parsed(
            Args::try_parse_from(["bin", "--model-path", "/m", "--no-cuda-graph"]).unwrap(),
        )
        .unwrap();
        assert!(!backend.launch.cuda_graph);
        assert_eq!(backend.launch.tp_size, 1);
    }

    // ---- GPU integration tests (require a real Qwen3 load) ----
    // Both gate on the model being present so a GPU-less `cargo test` stays
    // green: set `OPENINFER_TEST_MODEL_PATH=/path/to/Qwen3-4B`.

    use dynamo_backend_common::testing::mock_context;
    use dynamo_backend_common::{FinishReason, SamplingOptions, StopConditions};
    use futures::StreamExt as _;
    use std::time::Duration;

    fn test_model_path() -> Option<String> {
        let p = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "models/Qwen3-4B".to_string());
        if std::path::Path::new(&p).exists() {
            Some(p)
        } else {
            eprintln!("skipping GPU test: no model at {p} (set OPENINFER_TEST_MODEL_PATH)");
            None
        }
    }

    fn test_backend(model_path: &str) -> OpeninferBackend {
        let args =
            Args::try_parse_from(["bin", "--model-path", model_path]).expect("parse test args");
        OpeninferBackend::from_parsed(args)
            .expect("build test backend")
            .0
    }

    fn gen_request(max_tokens: u32) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("qwen3".to_string())
            .token_ids(vec![9707, 11, 1879])
            .stop_conditions(StopConditions {
                max_tokens: Some(max_tokens),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(Default::default())
            .build()
            .expect("build request")
    }

    /// Fast GPU smoke e2e against a real Qwen3 load: one bounded `generate`
    /// (well-formed stream — exactly one terminal, last, with usage matching the
    /// streamed token count), a cancellation (prompt `FinishReason::Cancelled`),
    /// and idempotent `cleanup`. The small `max_tokens` keeps it to seconds; the
    /// exhaustive official contract is the `#[ignore]`d test below.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gpu_smoke_generate_cancel_cleanup() {
        let Some(model_path) = test_model_path() else {
            return;
        };
        let backend = test_backend(&model_path);
        backend.start(0).await.expect("start");
        eprintln!("[smoke] engine loaded");

        // 1. A bounded generate yields a well-formed stream.
        let stream = backend
            .generate(gen_request(32), GenerateContext::new(mock_context(), None))
            .await
            .expect("generate");
        let chunks: Vec<_> = stream.map(|r| r.expect("stream item Ok")).collect().await;
        let terminal = chunks.last().expect("at least one chunk");
        assert!(
            terminal.finish_reason.is_some(),
            "the last chunk must be terminal"
        );
        assert!(
            chunks[..chunks.len() - 1]
                .iter()
                .all(|c| c.finish_reason.is_none()),
            "only the last chunk may carry a finish_reason"
        );
        let streamed: usize = chunks.iter().map(|c| c.token_ids.len()).sum();
        if let Some(u) = terminal.completion_usage.as_ref() {
            assert_eq!(
                streamed, u.completion_tokens as usize,
                "reported completion_tokens must equal the streamed token count"
            );
        }
        eprintln!(
            "[smoke] generate ok: {streamed} tokens, finish={:?}",
            terminal.finish_reason
        );

        // 2. Cancellation yields a Cancelled terminal within a deadline.
        let ctx = mock_context();
        let stream = backend
            .generate(gen_request(10_000), GenerateContext::new(ctx.clone(), None))
            .await
            .expect("generate for cancel");
        ctx.stop_generating();
        let last = tokio::time::timeout(Duration::from_secs(5), async {
            let mut last = None;
            let mut s = stream;
            while let Some(item) = s.next().await {
                last = Some(item.expect("stream item Ok"));
            }
            last
        })
        .await
        .expect("stream must terminate within the cancel deadline")
        .expect("cancelled stream still yields a terminal");
        assert!(
            matches!(last.finish_reason, Some(FinishReason::Cancelled)),
            "a cancelled stream must end with FinishReason::Cancelled, got {:?}",
            last.finish_reason
        );
        eprintln!("[smoke] cancellation ok");

        // 3. cleanup is idempotent.
        backend.cleanup().await.expect("cleanup");
        backend
            .cleanup()
            .await
            .expect("second cleanup must be idempotent");
        eprintln!("[smoke] cleanup idempotent ok");
    }

    /// Exhaustive official Dynamo `LLMEngine` conformance: start →
    /// kv_event_sources / setup_metrics → well-formed generate → 8 concurrent
    /// generates → cancellation → idempotent cleanup → cleanup-without-start.
    /// No mocks. `#[ignore]`d because the kit leaves `max_tokens` unset, so each
    /// generate runs to the 16k fallback cap — minutes of GPU time. Run on
    /// demand: `OPENINFER_TEST_MODEL_PATH=… cargo test --release --bins -- --ignored`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "full official conformance generates to the 16k max-token cap (minutes); run with --ignored"]
    async fn satisfies_dynamo_llmengine_contract() {
        let Some(model_path) = test_model_path() else {
            return;
        };
        let factory = || test_backend(&model_path);
        dynamo_backend_common::testing::run_conformance(factory)
            .await
            .expect("openinfer backend satisfies the Dynamo LLMEngine contract");
    }
}
