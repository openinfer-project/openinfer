//! `OpeninferBackend` — adapts openinfer's `EngineHandle` to Dynamo's
//! `LLMEngine`, so a pure-Rust openinfer worker plugs into a Dynamo frontend +
//! KV router. Run one process per GPU (`--device-ordinal 0..N`); the router
//! fans requests across the replicas.
//!
//! `start` (load Qwen3) / `generate` (stream tokens) / `cleanup` cover serving;
//! `setup_metrics` (M2) publishes the live KV-load signal the router scores
//! against, so an idle/busy replica is visible and load-balancing works.
//! `kv_event_sources` (M3) streams block store/remove events to the router's
//! radix tree, so a warm prefix on this replica steers matching requests here.
//! The engine advertises its real KV block size + capacity too, so the router's
//! KV-aware cost function is well-defined. Both publishers run only under
//! `enable_kv_routing`; a routing-off worker stays event-free and zero-cost.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, CommonArgs, DynamoError, EngineConfig, GenerateContext, KvEventPublisher,
    KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration,
    MetricsBindings, MetricsCtx, OnPublisherReady, OnSnapshotPublisherReady, PreprocessedRequest,
    SnapshotPublisher, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use openinfer_engine::engine::{
    EngineHandle, GenerateRequest, KvBlockEvent, LoadSnapshot, TokenSink, TokenStreamReceiver,
};
use openinfer_qwen3::{
    DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES, DEFAULT_KV_PAGE_SIZE,
    DEFAULT_MAX_PREFILL_TOKENS, DecodeOverlap, Qwen3LaunchOptions, Qwen3MemoryOptions,
    Qwen3OffloadOptions,
};
use tokio::sync::{mpsc, watch};
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
    /// KV cache page (block) size in tokens. FlashInfer only accepts 16 or 64.
    #[arg(long, default_value_t = DEFAULT_KV_PAGE_SIZE)]
    kv_page_size: usize,
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
    /// Fired by `cleanup`; every in-flight `generate` stream and the metrics
    /// publish task select on it so shutdown yields a clean `Cancelled` terminal
    /// instead of racing the channel-close path into a spurious "stream
    /// incomplete" error. Behind a `Mutex` and reset at the top of `start`: a
    /// `cleanup` → `start` cycle on the same instance must get a fresh token,
    /// else the next run's `generate`/metrics tasks observe the already-tripped
    /// token and silently yield `Cancelled` / go dark.
    cancel: Mutex<CancellationToken>,
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
            args.kv_page_size,
        )
        .validate()
        .map_err(|e| convert::invalid_argument(format!("invalid memory options: {e:#}")))?;

        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            disaggregation_mode: args.common.disaggregation_mode,
            model_name: args.model_path.to_string_lossy().into_owned(),
            served_model_name: Some(served_model_name.clone()),
            ..Default::default()
        };

        // KV block events are consumed only when the worker enables KV routing:
        // the framework calls `kv_event_sources()` (which takes the engine's
        // event receiver) solely under `enable_kv_routing`. Tying the engine's
        // event production to the same flag keeps the invariant that we never
        // feed a channel no one drains — a routing-off worker stays event-free
        // and zero-cost, exactly like plain single-machine openinfer.
        let enable_kv_events = config.enable_kv_routing;

        let launch = Qwen3LaunchOptions {
            device_ordinal: args.device_ordinal,
            tp_size: 1,
            cuda_graph: !args.no_cuda_graph,
            offload: Qwen3OffloadOptions::disabled(),
            // Keep the prefix cache on: it is both a single-worker win and the
            // source of the KV store/remove events published to the router for
            // cache-aware routing.
            no_prefix_cache: false,
            max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
            memory,
            lora: None,
            decode_overlap: DecodeOverlap::Off,
            batch_invariant: false,
            // Speculative decoding is a standalone-server knob; the Dynamo
            // worker never drafts.
            dflash_draft_model_path: None,
            enable_kv_events,
        };

        let backend = OpeninferBackend {
            model_path: args.model_path.clone(),
            served_model_name,
            launch,
            handle: Mutex::new(None),
            cancel: Mutex::new(CancellationToken::new()),
        };

        Ok((backend, config))
    }

    fn handle(&self) -> std::sync::MutexGuard<'_, Option<EngineHandle>> {
        self.handle.lock().expect("engine handle mutex poisoned")
    }

    /// The current cancellation token (cloned). Reset by `start`, tripped by
    /// `cleanup`.
    fn cancel_token(&self) -> CancellationToken {
        self.cancel.lock().expect("cancel mutex poisoned").clone()
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
        // Fresh lifecycle: a prior `cleanup` left the token tripped, which would
        // make this run's generate/metrics tasks born-cancelled. Reset it.
        *self.cancel.lock().expect("cancel mutex poisoned") = CancellationToken::new();

        tracing::info!(
            model_path = %self.model_path.display(),
            device_ordinal = self.launch.device_ordinal,
            cuda_graph = self.launch.cuda_graph,
            "loading Qwen3 engine (weights -> GPU); this can take a while"
        );

        // The model load is blocking (weights -> GPU, kernel warmup, optional
        // graph capture) and must not stall the runtime's reactor.
        let model_path = self.model_path.clone();
        // `Qwen3LaunchOptions` is no longer `Copy` (it now carries an optional
        // draft-model path); clone it into the blocking task and keep `self`
        // intact for the `enable_kv_events` check below.
        let launch = self.launch.clone();
        let handle =
            tokio::task::spawn_blocking(move || openinfer_qwen3::launch(&model_path, launch))
                .await
                .map_err(|e| convert::backend_error(format!("engine loader thread panicked: {e}")))?
                .map_err(|e| convert::backend_error(format!("Qwen3 engine load failed: {e:#}")))?;

        let kv = handle.kv_capacity();
        // KV events require a block-structured KV cache: the framework advertises
        // (and drains) the Push source only when `kv_cache_block_size` is set, so
        // events without a paged cache would leave the scheduler producing into a
        // channel no one reads. Fail loudly at load rather than leak silently.
        if self.launch.enable_kv_events && kv.is_none() {
            return Err(convert::backend_error(
                "enable_kv_events requires a paged KV cache, but the engine reported none",
            ));
        }
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
            self.cancel_token(),
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
        self.cancel.lock().expect("cancel mutex poisoned").cancel();
        let _ = self.handle().take();
        tracing::info!("openinfer backend: cleanup complete");
        Ok(())
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        // The advertised source set must be stable across calls (the
        // conformance kit calls this twice and rejects a changed dp_rank set),
        // so gate it on the immutable launch flag — NOT on the take-once event
        // receiver, whose presence flips after the first take. Events off
        // (routing disabled at load) ⇒ opt out; the router leans on the load
        // signal.
        if !self.launch.enable_kv_events {
            return Ok(Vec::new());
        }
        let handle = self
            .handle()
            .clone()
            .ok_or_else(|| convert::engine_shutdown("kv_event_sources called before start"))?;
        let cancel = self.cancel_token();

        // Framework builds the KvEventPublisher and invokes this once after a
        // successful start; we take the engine's neutral event receiver here
        // (lazily, so the twice-called check above never consumes it) and own
        // the pump thereafter. Cancelled in `cleanup` via `self.cancel`, so a
        // start→cleanup→start cycle never leaks a task.
        let on_ready: OnPublisherReady = Box::new(move |publisher| {
            let events = handle.take_kv_events().ok_or_else(|| {
                convert::engine_shutdown("kv event receiver already taken or unavailable")
            })?;
            tokio::spawn(publish_kv_events_loop(publisher, events, cancel));
            Ok(())
        });

        Ok(vec![KvEventSource::Push {
            on_ready,
            dp_rank: DP_RANK,
        }])
    }

    async fn setup_metrics(&self, _ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        // The scheduler thread republishes live KV occupancy to this watch feed
        // after every step (and a heartbeat when idle). `setup_metrics` runs
        // only after a successful `start`, so the handle and its feed exist.
        let load_watch = self
            .handle()
            .as_ref()
            .and_then(EngineHandle::load_watch)
            .ok_or_else(|| convert::engine_shutdown("setup_metrics called before start"))?;
        let cancel = self.cancel_token();

        // Framework constructs the SnapshotPublisher and hands it back here; we
        // own the publish loop thereafter. It is cancelled in `cleanup` via
        // `self.cancel`, so a start→cleanup→start cycle never leaks a task.
        let on_publisher_ready: OnSnapshotPublisherReady = Box::new(move |publisher| {
            tokio::spawn(publish_load_loop(publisher, load_watch, cancel));
            Ok(())
        });

        Ok(MetricsBindings {
            dp_ranks: vec![DP_RANK],
            on_publisher_ready: Some(on_publisher_ready),
        })
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
        // Prefix-cache hit, carried out of the schedule event and stamped onto
        // whichever terminal we emit (natural, cancelled, or stopped).
        let mut cached_tokens: u32 = 0;
        // A cancelled/stopped terminal has no engine usage to inherit, so build
        // one from the counts we have and stamp the cache hit the same way the
        // natural terminal carries it.
        let cancel_usage = |completion_tokens, cached_tokens| {
            let mut u = usage(prompt_tokens, completion_tokens);
            convert::apply_cached_tokens(&mut u, cached_tokens);
            u
        };
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
                        .with_usage(cancel_usage(completion_tokens, cached_tokens)));
                    break;
                }
                _ = ctx.stopped() => {
                    cancelled.store(true, Ordering::Release);
                    yield Ok(LLMEngineOutput::cancelled()
                        .with_usage(cancel_usage(completion_tokens, cached_tokens)));
                    break;
                }
                recv = rx.recv() => {
                    let Some((_tag, event)) = recv else {
                        yield Err(convert::stream_incomplete());
                        break;
                    };
                    match convert::map_token_event(event) {
                        Mapped::Cached(c) => { cached_tokens = c; }
                        Mapped::Chunk(c) => { completion_tokens += 1; yield Ok(c); }
                        Mapped::Terminal(mut t) => {
                            // Every terminal is built `.with_usage(...)` in
                            // `map_token_event`, so usage is an invariant, not an
                            // option — crash rather than silently drop the hit.
                            let u = t
                                .completion_usage
                                .as_mut()
                                .expect("terminal token event carries completion usage");
                            convert::apply_cached_tokens(u, cached_tokens);
                            yield Ok(t);
                            break;
                        }
                        Mapped::Fail(e) => { yield Err(e); break; }
                        Mapped::Ignore => {}
                    }
                }
            }
        }
    }
}

/// Idle heartbeat for the load signal. An idle worker stops stepping, so the
/// watch goes quiet; without a floor the router could read "no recent update"
/// as "worker gone" and stop routing to a healthy idle replica. Far coarser
/// than the per-step change wakeups that carry the real signal under load.
const LOAD_HEARTBEAT: Duration = Duration::from_millis(100);

/// Forward openinfer's live KV load to the Dynamo KV router.
///
/// openinfer's scheduler is a plain OS thread that writes a lock-free `watch`
/// each step; this task bridges that to the async `SnapshotPublisher` without
/// coupling the engine to tokio and without ever letting a metrics read stall
/// the GPU step. It republishes on every change (≈ once per decode step under
/// load) and at least every [`LOAD_HEARTBEAT`] when idle — until `cleanup`
/// cancels it or the engine drops the watch sender.
async fn publish_load_loop(
    publisher: Arc<SnapshotPublisher>,
    mut load_watch: watch::Receiver<LoadSnapshot>,
    cancel: CancellationToken,
) {
    loop {
        let load = *load_watch.borrow_and_update();
        publisher.publish(DP_RANK, convert::load_to_component_snapshot(load, DP_RANK));

        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            changed = load_watch.changed() => {
                // Err means the engine dropped the sender (scheduler gone) —
                // nothing left to report.
                if changed.is_err() {
                    break;
                }
            }
            _ = tokio::time::sleep(LOAD_HEARTBEAT) => {}
        }
    }
}

/// Forward openinfer's neutral KV block events to the Dynamo KV router.
///
/// Each event the engine emits — a block sealed into the prefix cache, or one
/// evicted — becomes a single `KvCacheEvent` on the router's radix tree,
/// stamped with the publisher's monotonic id. The engine already speaks the
/// router's u64 hash space, so the translation is a pure field rename
/// ([`convert::kv_block_event_to_dynamo`]). Spawned from the Push `on_ready`
/// handoff; ends when `cleanup` cancels it or the engine drops the event sender
/// (scheduler gone) or the router-side receiver closes.
async fn publish_kv_events_loop(
    publisher: Arc<KvEventPublisher>,
    mut events: mpsc::UnboundedReceiver<KvBlockEvent>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            recv = events.recv() => {
                let Some(event) = recv else { break };
                let dynamo_event =
                    convert::kv_block_event_to_dynamo(event, publisher.next_event_id());
                if publisher.publish(dynamo_event).is_err() {
                    break;
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
                "qwen3",
                "--namespace",
                "prod",
                "--component",
                "worker",
            ])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(config.served_model_name.as_deref(), Some("qwen3"));
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

    /// Each GPU test loads a full Qwen3 onto the device; cargo runs tests
    /// concurrently by default, which races two model loads into one GPU and
    /// OOMs the smaller cards. Serialize the model-loading tests behind one
    /// async lock so the suite is green without `--test-threads=1`.
    static GPU_TEST_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

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
        let _gpu = GPU_TEST_LOCK.lock().await;
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

    /// The router-facing load signal actually tracks KV occupancy: zero while
    /// idle, rises above zero while a request decodes, and falls back to zero
    /// once it finishes — bounded by the advertised capacity throughout. Reads
    /// the same `EngineHandle::load_watch` feed `setup_metrics` publishes from,
    /// so it covers the openinfer→watch wiring without needing a live router.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn load_signal_tracks_kv_usage() {
        let Some(model_path) = test_model_path() else {
            return;
        };
        let _gpu = GPU_TEST_LOCK.lock().await;
        let backend = test_backend(&model_path);
        backend.start(0).await.expect("start");

        let handle = backend.handle().as_ref().expect("started").clone();
        let kv_total = handle.kv_capacity().expect("kv capacity").total_blocks as u64;
        let mut load_rx = handle.load_watch().expect("load feed wired");

        // Idle: nothing in flight, so the whole pool reads as free.
        let idle = *load_rx.borrow_and_update();
        assert_eq!(idle.kv_used_blocks, 0, "idle pool must report 0 used");
        assert_eq!(idle.kv_total_blocks, kv_total);

        // Drive a bounded generate and watch usage climb above zero.
        let stream = backend
            .generate(gen_request(64), GenerateContext::new(mock_context(), None))
            .await
            .expect("generate");

        let rose = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if load_rx.borrow_and_update().kv_used_blocks > 0 {
                    return true;
                }
                if load_rx.changed().await.is_err() {
                    return false; // engine dropped the feed
                }
            }
        })
        .await
        .expect("kv_used_blocks must rise within the deadline");
        assert!(rose, "engine dropped the load feed before usage rose");
        let peak = load_rx.borrow().kv_used_blocks;
        assert!(
            peak <= kv_total,
            "used {peak} cannot exceed total {kv_total}"
        );

        // Drain to completion; usage falls back to zero as the request's blocks
        // free.
        let _: Vec<_> = stream.map(|r| r.expect("stream item Ok")).collect().await;
        let settled = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if load_rx.borrow_and_update().kv_used_blocks == 0 {
                    return;
                }
                if load_rx.changed().await.is_err() {
                    return;
                }
            }
        })
        .await;
        assert!(
            settled.is_ok(),
            "kv_used_blocks must return to 0 after the request finishes"
        );

        backend.cleanup().await.expect("cleanup");
        eprintln!("[load] idle=0 → peak={peak}/{kv_total} → settled=0 ok");
    }

    /// M3 on-GPU proof: the KV events a real generate emits round-trip through
    /// the exact dynamo primitive the router runs. Replaying them into a
    /// `RadixTree` must succeed for every event — `apply_stored` rejects a
    /// broken parent chain with `ParentBlockNotFound`, so a swapped
    /// parent/block hash (the M3 crux) would fail here — and querying the tree
    /// by the blocks' content hashes must return a full-length prefix match for
    /// this worker. That closes the loop that source review opened: openinfer's
    /// u64 block hashes *are* the router's hash space, end to end on real
    /// weights.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn kv_events_roundtrip_through_dynamo_radix_tree() {
        use dynamo_kv_router::indexer::RadixTree;
        use dynamo_kv_router::protocols::{LocalBlockHash, RouterEvent, WorkerWithDpRank};
        use openinfer_engine::engine::KvBlockEvent;

        let Some(model_path) = test_model_path() else {
            return;
        };
        let _gpu = GPU_TEST_LOCK.lock().await;
        let backend = test_backend(&model_path);
        backend.start(0).await.expect("start");

        let handle = backend.handle().as_ref().expect("started").clone();
        let block_size = handle.kv_capacity().expect("kv capacity").block_size;
        // The default backend has KV routing (hence events) on; take the neutral
        // receiver the framework would otherwise consume in kv_event_sources.
        let mut events = handle
            .take_kv_events()
            .expect("kv events are enabled for the dynamo backend");

        // A prompt spanning several full blocks plus a partial tail, so the
        // sealed prefix has real multi-block chain structure to validate.
        let prompt_len = 3 * block_size + block_size / 2;
        let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + (i % 200)).collect();
        let request = PreprocessedRequest::builder()
            .model("qwen3".to_string())
            .token_ids(prompt)
            .stop_conditions(StopConditions {
                max_tokens: Some(16),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(Default::default())
            .build()
            .expect("build request");

        let stream = backend
            .generate(request, GenerateContext::new(mock_context(), None))
            .await
            .expect("generate");
        let _: Vec<_> = stream.map(|r| r.expect("stream item Ok")).collect().await;

        // The producer emits at the top of the scheduler loop, one iteration
        // behind registration, and the request's final blocks are stashed at
        // drop and emitted on a later tick — so poll until the feed goes quiet
        // rather than reading once.
        let mut stored: Vec<KvBlockEvent> = Vec::new();
        let mut removed = 0usize;
        loop {
            match tokio::time::timeout(Duration::from_millis(500), events.recv()).await {
                Ok(Some(ev)) => match ev {
                    KvBlockEvent::Stored { .. } => stored.push(ev),
                    KvBlockEvent::Removed { .. } => removed += 1,
                },
                Ok(None) => break, // engine dropped the feed
                Err(_) => break,   // quiet for the timeout => drained
            }
        }
        assert!(
            !stored.is_empty(),
            "a multi-block generate must emit at least one KV store event"
        );

        // Replay into the router's own tree, exactly as the publisher pump
        // would. A single small request against a large pool should not evict,
        // so removes are not expected; tolerate but report them.
        const WORKER_ID: u64 = 0;
        let mut tree = RadixTree::new();
        let mut query: Vec<LocalBlockHash> = Vec::new();
        for (event_id, ev) in stored.iter().enumerate() {
            if let KvBlockEvent::Stored { blocks, .. } = ev {
                for b in blocks {
                    query.push(LocalBlockHash(b.tokens_hash));
                }
            }
            let dynamo_event = convert::kv_block_event_to_dynamo(ev.clone(), event_id as u64);
            tree.apply_event(RouterEvent::new(WORKER_ID, dynamo_event))
                .expect("every emitted store applies — parent chain matches the router's");
        }
        assert!(!query.is_empty(), "store events must carry blocks");

        let scores = tree.find_matches(query.clone(), false);
        let matched = scores
            .scores
            .get(&WorkerWithDpRank::new(WORKER_ID, 0))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            matched as usize,
            query.len(),
            "the full stored prefix must match for this worker (got {matched}/{})",
            query.len()
        );

        backend.cleanup().await.expect("cleanup");
        eprintln!(
            "[kv-events] {} store events, {} blocks, {removed} removes, full prefix match ok",
            stored.len(),
            query.len()
        );
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
