use std::any::Any;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::thread::{self};

use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;

use crate::parallel::ParallelConfig;
use crate::sampler::SamplingParams;

#[derive(Clone, Debug)]
pub struct EngineLoadOptions {
    pub enable_cuda_graph: bool,
    pub device_ordinals: Vec<usize>,
    pub parallel_config: Option<ParallelConfig>,
    pub ep_backend: EpBackend,
    pub seed: u64,
}

impl Default for EngineLoadOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EpBackend {
    #[default]
    Nccl,
    DeepEp,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: &'static str,
    pub display_name: String,
    pub model_path: PathBuf,
    pub max_model_len: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
    Error,
}

pub struct GenerateRequest {
    pub request_id: Option<String>,
    pub queued_at_unix_s: Option<f64>,
    /// Trace context of the caller's request span, when tracing is on. The
    /// model scheduler opens its queue/prefill/decode spans as children of this
    /// so the host-side phase breakdown attaches to the same trace the frontend
    /// started. `None` when tracing is disabled — the scheduler then skips span
    /// work entirely. `SpanContext` is `Copy`, so it rides through the
    /// scheduler's `Clone` request state without holding a live (non-`Clone`)
    /// `Span`.
    pub trace_parent: Option<fastrace::collector::SpanContext>,
    /// Logical data-parallel rank selected by the frontend. `None` leaves
    /// placement to the model scheduler (the direct `EngineHandle` path).
    pub data_parallel_rank: Option<usize>,
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub lora_adapter: Option<String>,
    /// Where the scheduler emits this request's `TokenEvent`s. All requests on
    /// one engine share a single tagged output channel behind this sink (see
    /// [`TokenSink`]); the frontend demuxes by tag.
    pub token_tx: TokenSink,
    pub logprobs: usize,
    pub echo: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadLoraAdapterRequest {
    pub lora_name: String,
    pub lora_path: PathBuf,
    pub load_inplace: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnloadLoraAdapterRequest {
    pub lora_name: String,
    pub lora_int_id: Option<i64>,
}

pub enum EngineControlRequest {
    LoadLoraAdapter {
        request: LoadLoraAdapterRequest,
        response_tx: oneshot::Sender<std::result::Result<(), String>>,
    },
    UnloadLoraAdapter {
        request: UnloadLoraAdapterRequest,
        response_tx: oneshot::Sender<std::result::Result<(), String>>,
    },
    ListLoraAdapters {
        response_tx: oneshot::Sender<std::result::Result<Vec<String>, String>>,
    },
}

pub enum EngineCommand {
    Generate(GenerateRequest),
    Control(EngineControlRequest),
}

#[derive(Debug, Eq, PartialEq)]
pub enum EngineControlError {
    Unsupported(&'static str),
    ChannelClosed,
    OperationFailed(String),
}

impl fmt::Display for EngineControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => f.write_str(message),
            Self::ChannelClosed => f.write_str("engine control channel closed"),
            Self::OperationFailed(message) => {
                write!(f, "engine control operation failed: {message}")
            }
        }
    }
}

impl Error for EngineControlError {}

pub type EngineControlResult<T> = std::result::Result<T, EngineControlError>;

#[derive(Debug)]
pub enum TokenEvent {
    Scheduled {
        queued_at_unix_s: f64,
        scheduled_at_unix_s: f64,
        prompt_tokens: usize,
        /// Prompt tokens served from the prefix cache (0 when the engine has
        /// no prefix cache or the value is not known at emit time).
        cached_tokens: usize,
    },
    Token {
        id: u32,
        logprob: Option<TokenLogprob>,
    },
    PromptTokens {
        ids: Vec<u32>,
        logprobs: Vec<Option<TokenLogprob>>,
    },
    Finished {
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error {
        message: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Rejected {
        message: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
}

/// The tag that routes a [`TokenEvent`] back to its request on the shared
/// output channel — the external request id (vLLM's `request_id`). `Arc<str>`
/// keeps per-event tagging to a refcount bump instead of a string copy.
pub type RequestTag = Arc<str>;

/// The single output channel an engine dispatches *all* requests' token events
/// into, each tagged with its [`RequestTag`]. One receiver (the frontend demux
/// loop) drains it, replacing the former per-request fan-out of N channels and
/// N consumer tasks — N distinct sleeping consumers cost N wakeups per step,
/// one shared consumer costs ~1.
pub type TokenStreamSender = mpsc::UnboundedSender<(RequestTag, TokenEvent)>;
pub type TokenStreamReceiver = mpsc::UnboundedReceiver<(RequestTag, TokenEvent)>;

/// Per-request handle the scheduler holds to emit [`TokenEvent`]s.
///
/// Drop-in for the former `UnboundedSender<TokenEvent>`: it keeps the same
/// `send` / `is_closed` / `Clone` surface, so scheduler call sites are
/// unchanged. Internally each event is tagged with the request's
/// [`RequestTag`] and pushed onto one shared [`TokenStreamSender`].
///
/// Cancellation moved from "drop the per-request receiver" to a shared abort
/// reason: the frontend aborts a *single* request by setting its reason without
/// closing the channel the other requests still use. `send` and `is_closed`
/// then report that request as gone, so the scheduler retires it on its next
/// emit — the same *reactive* retirement the old consumer-drop gave, reached
/// through the reason rather than channel closure. `tx.is_closed()` is the
/// engine-wide signal (the whole demux is gone); the per-request signal is the
/// abort reason. The reason is set with `Release` and read with `Acquire` so
/// the abort is ordered against the frontend dropping the request's stream
/// state.
#[derive(Clone)]
pub struct TokenSink {
    tag: RequestTag,
    tx: TokenStreamSender,
    abort_reason: Arc<AtomicU8>,
}

impl TokenSink {
    pub fn new(tag: RequestTag, tx: TokenStreamSender, abort_reason: Arc<AtomicU8>) -> Self {
        Self {
            tag,
            tx,
            abort_reason,
        }
    }

    /// Emit one event for this request. Returns `Err` (handing the event back)
    /// when the request was aborted or the shared receiver is gone — both of
    /// which the scheduler reads as "consumer dropped, retire the request",
    /// the same contract as the old per-request channel.
    #[allow(clippy::result_large_err)]
    pub fn send(&self, event: TokenEvent) -> Result<(), mpsc::error::SendError<TokenEvent>> {
        if self.abort_reason() != RequestAbortReason::None {
            return Err(mpsc::error::SendError(event));
        }
        self.tx.send((self.tag.clone(), event)).map_err(|err| {
            let (_, event) = err.0;
            mpsc::error::SendError(event)
        })
    }

    /// `true` once the request is aborted or the shared receiver is gone.
    pub fn is_closed(&self) -> bool {
        self.abort_reason() != RequestAbortReason::None || self.tx.is_closed()
    }

    /// `true` once the frontend explicitly cancelled this request after the
    /// stream had already started.
    pub fn is_cancelled(&self) -> bool {
        self.abort_reason() == RequestAbortReason::Cancelled
    }

    /// `true` once the frontend observed a client disconnect before the first
    /// response chunk for this request reached the client.
    pub fn is_disconnected(&self) -> bool {
        self.abort_reason() == RequestAbortReason::Disconnected
    }

    /// Current per-request abort reason.
    pub fn abort_reason(&self) -> RequestAbortReason {
        RequestAbortReason::from_raw(self.abort_reason.load(Ordering::Acquire))
    }

    /// The request id this sink tags its events with.
    pub fn tag(&self) -> &RequestTag {
        &self.tag
    }

    /// A sink backed by its own private channel, for direct drivers
    /// (benchmarks, integration tests, the simulator) that consume one
    /// request's events without the shared frontend demux. The returned
    /// receiver yields the tagged events; the cancel flag is never tripped.
    pub fn standalone() -> (Self, TokenStreamReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        let sink = Self::new(
            Arc::from("local"),
            tx,
            Arc::new(AtomicU8::new(RequestAbortReason::None as u8)),
        );
        (sink, rx)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RequestAbortReason {
    None = 0,
    Cancelled = 1,
    Disconnected = 2,
}

impl RequestAbortReason {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::Cancelled,
            2 => Self::Disconnected,
            _ => Self::None,
        }
    }

    pub fn store(self, abort_reason: &AtomicU8) {
        abort_reason.store(self as u8, Ordering::Release);
    }
}

/// Seconds since `UNIX_EPOCH` as `f64` — the clock base for `TokenEvent`
/// timestamps.
pub fn unix_now_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before UNIX_EPOCH")
        .as_secs_f64()
}

/// KV pool capacity as the scheduler actually allocates it: whole blocks of
/// `block_size` tokens. A request of `L` tokens occupies `⌈L / block_size⌉`
/// blocks no matter how `L` divides, so a fit check must round per request —
/// summing raw token counts under-counts and can admit a batch that the
/// scheduler then has to defer. Lets a caller (e.g. the prefill/decode bench)
/// decide up front whether a batch fits without computing per-token KV by hand.
#[derive(Clone, Copy, Debug)]
pub struct KvCapacity {
    /// Blocks available for requests when the pool is empty.
    pub total_blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
}

impl KvCapacity {
    /// Total tokens the pool can hold (`total_blocks × block_size`).
    #[must_use]
    pub fn total_tokens(self) -> usize {
        self.total_blocks.saturating_mul(self.block_size)
    }

    /// Blocks a single request of `tokens` tokens occupies — whole-block
    /// allocation rounds up.
    #[must_use]
    pub fn blocks_for(self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size.max(1))
    }
}

/// Live KV-cache occupancy the scheduler republishes after every step.
///
/// `kv_used_blocks` is the load signal an out-of-band consumer (e.g. a Dynamo
/// KV router) scores against; `kv_total_blocks` is the engine's whole-pool
/// capacity (the same number advertised as the servable ceiling), so the
/// consumer can derive fractional usage without a second query. Carried over a
/// [`watch`] channel: the scheduler is the sole writer and never blocks on a
/// reader, and a reader only ever sees the latest snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadSnapshot {
    pub kv_used_blocks: u64,
    pub kv_total_blocks: u64,
    /// Requests currently occupying a decode/prefill slot.
    pub num_running_reqs: u64,
    /// Requests admitted but not yet running (KV pressure, prefetch wait).
    pub num_waiting_reqs: u64,
}

/// One full KV block that just became reusable from this engine's prefix cache.
///
/// The hashes are the *u64* sequence-aware / per-block token hashes a Dynamo KV
/// router indexes by (`dynamo_tokens::TokenBlock::{sequence_hash, block_hash}`),
/// kept as plain integers so this contract type stays free of any kvbm/dynamo
/// dependency. They are NOT the engine's internal 128-bit lineage hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvStoredBlock {
    /// Chained, sequence-aware block id (dynamo `ExternalSequenceBlockHash`).
    pub sequence_hash: u64,
    /// Un-chained per-block token hash (dynamo `LocalBlockHash`); the field a
    /// prefix-routing radix tree keys its children by.
    pub tokens_hash: u64,
}

/// A KV-cache block lifecycle event for an out-of-band cache-aware router.
///
/// Emitted only when the engine was built with a KV-event feed wired (off by
/// default); see [`EngineHandle::take_kv_events`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvBlockEvent {
    /// A contiguous run of newly-registered blocks became cacheable. `parent_hash`
    /// is the sequence hash of the block preceding `blocks[0]` (`None` if the run
    /// starts the sequence); each later block chains off the previous one.
    Stored {
        parent_hash: Option<u64>,
        blocks: Vec<KvStoredBlock>,
    },
    /// A previously-stored block was evicted from this engine's cache.
    Removed { sequence_hash: u64 },
}

#[derive(Clone)]
pub struct EngineHandle {
    inner: Arc<EngineInner>,
    servable_len: Option<u32>,
    /// KV pool capacity in blocks + block size, or `None` if the engine did not
    /// report it. See [`KvCapacity`].
    kv_capacity: Option<KvCapacity>,
    /// One optional live-load feed per scheduler partition. A normal engine
    /// has one partition; a data-parallel engine exposes one per logical DP
    /// rank. Each handle clone owns cloned receivers and `watch` fans out.
    load_watches: Vec<Option<watch::Receiver<LoadSnapshot>>>,
    /// Block store/remove feed for a cache-aware router, or `None` if not wired.
    /// `mpsc` (every event matters — unlike the coalescing load feed), so the
    /// single receiver is handed out exactly once via [`Self::take_kv_events`];
    /// the shared cell lets all handle clones agree on who took it.
    kv_events: Option<Arc<Mutex<Option<mpsc::UnboundedReceiver<KvBlockEvent>>>>>,
}

struct EngineInner {
    submit_tx: Option<mpsc::UnboundedSender<GenerateRequest>>,
    command_tx: Option<mpsc::UnboundedSender<EngineCommand>>,
    join_handle: Option<JoinHandle<()>>,
}

impl EngineHandle {
    pub fn new(submit_tx: mpsc::UnboundedSender<GenerateRequest>) -> Self {
        Self::from_parts(Some(submit_tx), None, None)
    }

    pub fn new_with_command_channel(command_tx: mpsc::UnboundedSender<EngineCommand>) -> Self {
        Self::from_parts(None, Some(command_tx), None)
    }

    pub fn new_with_command_channel_and_join_handle(
        command_tx: mpsc::UnboundedSender<EngineCommand>,
        join_handle: JoinHandle<()>,
    ) -> Self {
        Self::from_parts(None, Some(command_tx), Some(join_handle))
    }

    /// Construct a handle that owns the engine thread shutdown.
    ///
    /// Dropping the last handle clone closes the submit channel and then waits
    /// for the thread to return. That final drop may block until in-flight
    /// generation and backend teardown finish.
    pub fn new_with_join_handle(
        submit_tx: mpsc::UnboundedSender<GenerateRequest>,
        join_handle: JoinHandle<()>,
    ) -> Self {
        Self::from_parts(Some(submit_tx), None, Some(join_handle))
    }

    fn from_parts(
        submit_tx: Option<mpsc::UnboundedSender<GenerateRequest>>,
        command_tx: Option<mpsc::UnboundedSender<EngineCommand>>,
        join_handle: Option<JoinHandle<()>>,
    ) -> Self {
        Self {
            inner: Arc::new(EngineInner {
                submit_tx,
                command_tx,
                join_handle,
            }),
            servable_len: None,
            kv_capacity: None,
            load_watches: vec![None],
            kv_events: None,
        }
    }

    #[must_use]
    pub fn with_servable_len(mut self, servable_len: u32) -> Self {
        self.servable_len = Some(servable_len);
        self
    }

    pub fn servable_len(&self) -> Option<u32> {
        self.servable_len
    }

    #[must_use]
    pub fn with_kv_capacity(mut self, kv_capacity: KvCapacity) -> Self {
        self.kv_capacity = Some(kv_capacity);
        self
    }

    /// KV pool capacity, if the engine reported it. A batch whose per-request
    /// block footprint exceeds [`KvCapacity::total_blocks`] cannot be resident
    /// at once.
    pub fn kv_capacity(&self) -> Option<KvCapacity> {
        self.kv_capacity
    }

    #[must_use]
    pub fn with_load_watch(mut self, load_watch: watch::Receiver<LoadSnapshot>) -> Self {
        self.load_watches = vec![Some(load_watch)];
        self
    }

    /// Attach one load feed per logical scheduler partition.
    ///
    /// The vector is also the engine's frontend-visible DP topology, so an
    /// empty vector is an invalid engine rather than a single partition with
    /// missing metrics.
    #[must_use]
    pub fn with_load_watches(mut self, load_watches: Vec<watch::Receiver<LoadSnapshot>>) -> Self {
        assert!(
            !load_watches.is_empty(),
            "an engine must expose at least one scheduler partition"
        );
        self.load_watches = load_watches.into_iter().map(Some).collect();
        self
    }

    /// Number of logical scheduler partitions exposed to the frontend.
    pub fn scheduler_partition_count(&self) -> usize {
        self.load_watches.len()
    }

    /// A receiver for one partition's live load, if it wired one. Awaiting
    /// [`watch::Receiver::changed`] wakes once per scheduler step under load and
    /// stays quiet when idle, so a consumer republishes on real change rather
    /// than polling. `None` if the partition does not report a load feed or the
    /// index is outside the engine topology.
    pub fn load_watch_for(&self, partition: usize) -> Option<watch::Receiver<LoadSnapshot>> {
        self.load_watches.get(partition)?.clone()
    }

    /// Single-partition compatibility accessor.
    pub fn load_watch(&self) -> Option<watch::Receiver<LoadSnapshot>> {
        self.load_watch_for(0)
    }

    #[must_use]
    pub fn with_kv_events(mut self, rx: mpsc::UnboundedReceiver<KvBlockEvent>) -> Self {
        self.kv_events = Some(Arc::new(Mutex::new(Some(rx))));
        self
    }

    /// Take the engine's KV block-event receiver. Returns the receiver on the
    /// first call and `None` thereafter (there is one stream and one consumer —
    /// the cache-aware router pump). `None` also if the engine wired no feed.
    pub fn take_kv_events(&self) -> Option<mpsc::UnboundedReceiver<KvBlockEvent>> {
        self.kv_events
            .as_ref()?
            .lock()
            .expect("kv-events cell poisoned")
            .take()
    }

    #[allow(clippy::result_large_err)]
    pub fn submit(
        &self,
        req: GenerateRequest,
    ) -> std::result::Result<(), mpsc::error::SendError<GenerateRequest>> {
        match self.inner.submit_tx.as_ref() {
            Some(submit_tx) => submit_tx.send(req),
            None => match self.inner.command_tx.as_ref() {
                Some(command_tx) => command_tx
                    .send(EngineCommand::Generate(req))
                    .map_err(|err| match err.0 {
                        EngineCommand::Generate(req) => mpsc::error::SendError(req),
                        EngineCommand::Control(_) => unreachable!("submitted generate command"),
                    }),
                None => Err(mpsc::error::SendError(req)),
            },
        }
    }

    pub fn supports_lora_control(&self) -> bool {
        self.inner.command_tx.is_some()
    }

    pub async fn load_lora_adapter(
        &self,
        request: LoadLoraAdapterRequest,
    ) -> EngineControlResult<()> {
        match self.inner.command_tx.as_ref() {
            Some(command_tx) => {
                let (response_tx, response_rx) = oneshot::channel();
                command_tx
                    .send(EngineCommand::Control(
                        EngineControlRequest::LoadLoraAdapter {
                            request,
                            response_tx,
                        },
                    ))
                    .map_err(|_| EngineControlError::ChannelClosed)?;

                response_rx
                    .await
                    .map_err(|_| EngineControlError::ChannelClosed)?
                    .map_err(EngineControlError::OperationFailed)
            }
            None => Err(EngineControlError::Unsupported(
                "engine does not support dynamic LoRA adapter loading",
            )),
        }
    }

    pub async fn list_lora_adapters(&self) -> EngineControlResult<Vec<String>> {
        match self.inner.command_tx.as_ref() {
            Some(command_tx) => {
                let (response_tx, response_rx) = oneshot::channel();
                command_tx
                    .send(EngineCommand::Control(
                        EngineControlRequest::ListLoraAdapters { response_tx },
                    ))
                    .map_err(|_| EngineControlError::ChannelClosed)?;

                response_rx
                    .await
                    .map_err(|_| EngineControlError::ChannelClosed)?
                    .map_err(EngineControlError::OperationFailed)
            }
            None => Err(EngineControlError::Unsupported(
                "engine does not support dynamic LoRA adapter loading",
            )),
        }
    }

    pub async fn unload_lora_adapter(
        &self,
        request: UnloadLoraAdapterRequest,
    ) -> EngineControlResult<()> {
        match self.inner.command_tx.as_ref() {
            Some(command_tx) => {
                let (response_tx, response_rx) = oneshot::channel();
                command_tx
                    .send(EngineCommand::Control(
                        EngineControlRequest::UnloadLoraAdapter {
                            request,
                            response_tx,
                        },
                    ))
                    .map_err(|_| EngineControlError::ChannelClosed)?;

                response_rx
                    .await
                    .map_err(|_| EngineControlError::ChannelClosed)?
                    .map_err(EngineControlError::OperationFailed)
            }
            None => Err(EngineControlError::Unsupported(
                "engine does not support dynamic LoRA adapter loading",
            )),
        }
    }
}

pub fn panic_message(payload: &dyn Any) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        let _ = self.submit_tx.take();
        let _ = self.command_tx.take();
        if let Some(join_handle) = self.join_handle.take() {
            if join_handle.thread().id() != thread::current().id() {
                if let Err(panic) = join_handle.join() {
                    log::warn!(
                        "engine thread panicked during shutdown: {}",
                        panic_message(panic.as_ref())
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn joins_owned_thread_after_last_handle_drop() {
        let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
        let exited = Arc::new(AtomicBool::new(false));
        let thread_exited = Arc::clone(&exited);
        let join_handle = thread::spawn(move || {
            while submit_rx.blocking_recv().is_some() {}
            thread_exited.store(true, Ordering::SeqCst);
        });
        let handle = EngineHandle::new_with_join_handle(submit_tx, join_handle);
        let clone = handle.clone();

        drop(handle);
        assert!(!exited.load(Ordering::SeqCst));

        drop(clone);
        assert!(exited.load(Ordering::SeqCst));
    }

    #[test]
    fn lora_control_support_is_opt_in() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
        let handle = EngineHandle::new(submit_tx);
        assert!(!handle.supports_lora_control());

        let (command_tx, _command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);
        assert!(handle.supports_lora_control());
    }

    #[test]
    fn load_watches_define_scheduler_partitions() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
        let (_load_tx0, load_rx0) = watch::channel(LoadSnapshot {
            num_running_reqs: 1,
            ..LoadSnapshot::default()
        });
        let (_load_tx1, load_rx1) = watch::channel(LoadSnapshot {
            num_waiting_reqs: 2,
            ..LoadSnapshot::default()
        });
        let handle = EngineHandle::new(submit_tx).with_load_watches(vec![load_rx0, load_rx1]);

        assert_eq!(handle.scheduler_partition_count(), 2);
        assert_eq!(
            handle
                .load_watch_for(0)
                .expect("rank 0 watch")
                .borrow()
                .num_running_reqs,
            1
        );
        assert_eq!(
            handle
                .load_watch_for(1)
                .expect("rank 1 watch")
                .borrow()
                .num_waiting_reqs,
            2
        );
        assert!(handle.load_watch_for(2).is_none());
    }

    #[test]
    fn token_sink_distinguishes_cancelled_from_closed_receiver() {
        let abort_reason = Arc::new(AtomicU8::new(RequestAbortReason::None as u8));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = TokenSink::new(Arc::from("request-a"), tx, Arc::clone(&abort_reason));

        assert!(!sink.is_cancelled());
        assert!(!sink.is_disconnected());
        assert!(!sink.is_closed());
        sink.send(TokenEvent::Token {
            id: 7,
            logprob: None,
        })
        .expect("uncancelled sink should send");
        assert_eq!(rx.try_recv().expect("tagged event").0.as_ref(), "request-a");

        RequestAbortReason::Cancelled.store(&abort_reason);
        assert!(sink.is_cancelled());
        assert!(!sink.is_disconnected());
        assert!(sink.is_closed());
        assert!(
            sink.send(TokenEvent::Token {
                id: 8,
                logprob: None,
            })
            .is_err()
        );
    }

    #[test]
    fn token_sink_closed_receiver_is_not_explicit_cancel() {
        let (sink, rx) = TokenSink::standalone();

        drop(rx);

        assert!(!sink.is_cancelled());
        assert!(!sink.is_disconnected());
        assert!(sink.is_closed());
        assert!(
            sink.send(TokenEvent::Token {
                id: 7,
                logprob: None,
            })
            .is_err()
        );
    }

    #[test]
    fn token_sink_distinguishes_disconnected_from_cancelled() {
        let abort_reason = Arc::new(AtomicU8::new(RequestAbortReason::None as u8));
        let (tx, _rx) = mpsc::unbounded_channel();
        let sink = TokenSink::new(Arc::from("request-a"), tx, Arc::clone(&abort_reason));

        RequestAbortReason::Disconnected.store(&abort_reason);

        assert!(!sink.is_cancelled());
        assert!(sink.is_disconnected());
        assert!(sink.is_closed());
        assert!(
            sink.send(TokenEvent::Token {
                id: 7,
                logprob: None,
            })
            .is_err()
        );
    }

    #[tokio::test]
    async fn load_lora_adapter_sends_control_command() {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);

        let request = LoadLoraAdapterRequest {
            lora_name: "adapter-a".to_string(),
            lora_path: PathBuf::from("/tmp/adapter-a"),
            load_inplace: false,
        };
        let load = tokio::spawn({
            let handle = handle.clone();
            let request = request.clone();
            async move { handle.load_lora_adapter(request).await }
        });

        let command = command_rx.recv().await.expect("control command");
        match command {
            EngineCommand::Control(EngineControlRequest::LoadLoraAdapter {
                request: actual,
                response_tx,
            }) => {
                assert_eq!(actual, request);
                response_tx.send(Ok(())).expect("send load result");
            }
            EngineCommand::Control(
                EngineControlRequest::UnloadLoraAdapter { .. }
                | EngineControlRequest::ListLoraAdapters { .. },
            ) => {
                panic!("expected LoRA load command")
            }
            EngineCommand::Generate(_) => panic!("expected LoRA control command"),
        }

        load.await.expect("join load task").expect("load succeeded");
    }

    #[tokio::test]
    async fn list_lora_adapters_sends_control_command() {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);

        let list = tokio::spawn({
            let handle = handle.clone();
            async move { handle.list_lora_adapters().await }
        });

        let command = command_rx.recv().await.expect("control command");
        match command {
            EngineCommand::Control(EngineControlRequest::ListLoraAdapters { response_tx }) => {
                response_tx
                    .send(Ok(vec!["adapter-a".to_string()]))
                    .expect("send list result");
            }
            EngineCommand::Control(
                EngineControlRequest::LoadLoraAdapter { .. }
                | EngineControlRequest::UnloadLoraAdapter { .. },
            ) => {
                panic!("expected LoRA list command")
            }
            EngineCommand::Generate(_) => panic!("expected LoRA control command"),
        }

        assert_eq!(
            list.await.expect("join list task").expect("list succeeded"),
            vec!["adapter-a"]
        );
    }

    #[tokio::test]
    async fn load_lora_adapter_reports_unsupported_without_control() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
        let handle = EngineHandle::new(submit_tx);
        let error = handle
            .load_lora_adapter(LoadLoraAdapterRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: PathBuf::from("/tmp/adapter-a"),
                load_inplace: false,
            })
            .await
            .expect_err("control should be unsupported");
        assert_eq!(
            error,
            EngineControlError::Unsupported("engine does not support dynamic LoRA adapter loading")
        );
    }

    #[tokio::test]
    async fn unload_lora_adapter_sends_control_command() {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);

        let request = UnloadLoraAdapterRequest {
            lora_name: "adapter-a".to_string(),
            lora_int_id: None,
        };
        let unload = tokio::spawn({
            let handle = handle.clone();
            let request = request.clone();
            async move { handle.unload_lora_adapter(request).await }
        });

        let command = command_rx.recv().await.expect("control command");
        match command {
            EngineCommand::Control(EngineControlRequest::UnloadLoraAdapter {
                request: actual,
                response_tx,
            }) => {
                assert_eq!(actual, request);
                response_tx.send(Ok(())).expect("send unload result");
            }
            EngineCommand::Control(
                EngineControlRequest::LoadLoraAdapter { .. }
                | EngineControlRequest::ListLoraAdapters { .. },
            ) => {
                panic!("expected LoRA unload command")
            }
            EngineCommand::Generate(_) => panic!("expected LoRA control command"),
        }

        unload
            .await
            .expect("join unload task")
            .expect("unload succeeded");
    }
}
