use std::{
    error::Error,
    fmt,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

use tokio::sync::{mpsc, oneshot};

use crate::parallel::ParallelConfig;
use crate::sampler::SamplingParams;

#[derive(Clone, Debug)]
pub struct EngineLoadOptions {
    pub enable_cuda_graph: bool,
    pub enable_prefill_profile: bool,
    pub device_ordinals: Vec<usize>,
    pub parallel_config: Option<ParallelConfig>,
    pub ep_backend: EpBackend,
    pub seed: u64,
}

impl Default for EngineLoadOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            enable_prefill_profile: false,
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

// Request tracing level policy:
// - DEBUG span/event: successful request lifecycle, useful while debugging.
// - WARN terminal event: rejected requests.
// - ERROR terminal event: execution failures.
// High-volume success counts and token totals belong to `metrics`; the shared
// serving path intentionally does not emit per-token tracing events.
macro_rules! request_span {
    ($request_id:expr) => {
        tracing::debug_span!(
            "openinfer.request",
            request_id = $request_id,
            queued_at_unix_s = tracing::field::Empty,
            scheduled_at_unix_s = tracing::field::Empty,
            terminal_at_unix_s = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
            prompt_tokens = tracing::field::Empty,
            cached_prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
        )
    };
}

macro_rules! request_scheduled_event {
    ($span:expr, $queued_at_unix_s:expr, $scheduled_at_unix_s:expr, $prompt_tokens:expr, $cached_tokens:expr) => {
        tracing::debug!(
            parent: $span,
            queued_at_unix_s = $queued_at_unix_s,
            scheduled_at_unix_s = $scheduled_at_unix_s,
            prompt_tokens = $prompt_tokens as u64,
            cached_prompt_tokens = $cached_tokens as u64,
            "openinfer_request_scheduled"
        )
    };
}

macro_rules! request_terminal_event {
    ($level:ident, $span:expr, $terminal_at_unix_s:expr, $finish_reason:expr, $prompt_tokens:expr, $completion_tokens:expr) => {
        tracing::$level!(
            parent: $span,
            terminal_at_unix_s = $terminal_at_unix_s,
            finish_reason = $finish_reason,
            prompt_tokens = $prompt_tokens as u64,
            completion_tokens = $completion_tokens as u64,
            "openinfer_request_finished"
        )
    };
    ($level:ident, $span:expr, $terminal_at_unix_s:expr, $finish_reason:expr, $prompt_tokens:expr, $completion_tokens:expr, $message:expr) => {
        tracing::$level!(
            parent: $span,
            terminal_at_unix_s = $terminal_at_unix_s,
            finish_reason = $finish_reason,
            prompt_tokens = $prompt_tokens as u64,
            completion_tokens = $completion_tokens as u64,
            message = $message.as_str(),
            "openinfer_request_finished"
        )
    };
}

/// Per-request handle the scheduler holds to emit [`TokenEvent`]s.
///
/// Drop-in for the former `UnboundedSender<TokenEvent>`: it keeps the same
/// `send` / `is_closed` / `Clone` surface, so scheduler call sites are
/// unchanged. Internally each event is tagged with the request's
/// [`RequestTag`] and pushed onto one shared [`TokenStreamSender`].
///
/// Cancellation moved from "drop the per-request receiver" to a shared
/// `cancelled` flag: the frontend aborts a *single* request by flipping its
/// flag without closing the channel the other requests still use. `send` and
/// `is_closed` then report that request as gone, so the scheduler retires it on
/// its next emit — the same *reactive* retirement the old consumer-drop gave,
/// reached through the flag rather than channel closure. `tx.is_closed()` is
/// the engine-wide signal (the whole demux is gone); the per-request signal is
/// the flag. The flag is set with `Release` and read with `Acquire` so the
/// abort is ordered against the frontend dropping the request's stream state.
#[derive(Clone)]
pub struct TokenSink {
    tag: RequestTag,
    tx: TokenStreamSender,
    cancelled: Arc<AtomicBool>,
    span: tracing::Span,
}

impl TokenSink {
    pub fn new(tag: RequestTag, tx: TokenStreamSender, cancelled: Arc<AtomicBool>) -> Self {
        let span = request_span!(tag.as_ref());
        Self {
            tag,
            tx,
            cancelled,
            span,
        }
    }

    /// Emit one event for this request. Returns `Err` (handing the event back)
    /// when the request was cancelled or the shared receiver is gone — both of
    /// which the scheduler reads as "consumer dropped, retire the request",
    /// the same contract as the old per-request channel.
    #[allow(clippy::result_large_err)]
    pub fn send(&self, event: TokenEvent) -> Result<(), mpsc::error::SendError<TokenEvent>> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(mpsc::error::SendError(event));
        }
        self.observe_event(&event);
        self.tx.send((self.tag.clone(), event)).map_err(|err| {
            let (_, event) = err.0;
            mpsc::error::SendError(event)
        })
    }

    /// `true` once the request is cancelled or the shared receiver is gone.
    pub fn is_closed(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || self.tx.is_closed()
    }

    /// The request id this sink tags its events with.
    pub fn tag(&self) -> &RequestTag {
        &self.tag
    }

    fn observe_event(&self, event: &TokenEvent) {
        match event {
            TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s,
                prompt_tokens,
                cached_tokens,
            } => {
                metrics::counter!("openinfer_engine_requests_scheduled_total").increment(1);
                metrics::counter!("openinfer_engine_cached_prompt_tokens_total")
                    .increment(*cached_tokens as u64);
                metrics::histogram!("openinfer_engine_queue_wait_ms")
                    .record((scheduled_at_unix_s - queued_at_unix_s).max(0.0) * 1_000.0);
                self.span.record("queued_at_unix_s", *queued_at_unix_s);
                self.span
                    .record("scheduled_at_unix_s", *scheduled_at_unix_s);
                self.span.record("prompt_tokens", *prompt_tokens as u64);
                self.span
                    .record("cached_prompt_tokens", *cached_tokens as u64);
                request_scheduled_event!(
                    &self.span,
                    *queued_at_unix_s,
                    *scheduled_at_unix_s,
                    *prompt_tokens,
                    *cached_tokens
                );
            }
            TokenEvent::Token { .. } | TokenEvent::PromptTokens { .. } => {}
            TokenEvent::Finished {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                let terminal_at_unix_s = unix_now_s();
                let finish_reason = finish_reason_label(*finish_reason);
                metrics::counter!(
                    "openinfer_engine_requests_finished_total",
                    "outcome" => finish_reason
                )
                .increment(1);
                metrics::counter!("openinfer_engine_prompt_tokens_total")
                    .increment(*prompt_tokens as u64);
                metrics::counter!("openinfer_engine_completion_tokens_total")
                    .increment(*completion_tokens as u64);
                self.record_terminal_fields(
                    terminal_at_unix_s,
                    finish_reason,
                    *prompt_tokens,
                    *completion_tokens,
                );
                if finish_reason == "error" {
                    request_terminal_event!(
                        error,
                        &self.span,
                        terminal_at_unix_s,
                        finish_reason,
                        *prompt_tokens,
                        *completion_tokens
                    );
                } else {
                    request_terminal_event!(
                        debug,
                        &self.span,
                        terminal_at_unix_s,
                        finish_reason,
                        *prompt_tokens,
                        *completion_tokens
                    );
                }
            }
            TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens,
            } => {
                let terminal_at_unix_s = unix_now_s();
                metrics::counter!(
                    "openinfer_engine_requests_finished_total",
                    "outcome" => "error"
                )
                .increment(1);
                metrics::counter!("openinfer_engine_prompt_tokens_total")
                    .increment(*prompt_tokens as u64);
                metrics::counter!("openinfer_engine_completion_tokens_total")
                    .increment(*completion_tokens as u64);
                self.record_terminal_fields(
                    terminal_at_unix_s,
                    "error",
                    *prompt_tokens,
                    *completion_tokens,
                );
                request_terminal_event!(
                    error,
                    &self.span,
                    terminal_at_unix_s,
                    "error",
                    *prompt_tokens,
                    *completion_tokens,
                    message
                );
            }
            TokenEvent::Rejected {
                message,
                prompt_tokens,
                completion_tokens,
            } => {
                let terminal_at_unix_s = unix_now_s();
                metrics::counter!(
                    "openinfer_engine_requests_finished_total",
                    "outcome" => "rejected"
                )
                .increment(1);
                metrics::counter!("openinfer_engine_prompt_tokens_total")
                    .increment(*prompt_tokens as u64);
                metrics::counter!("openinfer_engine_completion_tokens_total")
                    .increment(*completion_tokens as u64);
                self.record_terminal_fields(
                    terminal_at_unix_s,
                    "rejected",
                    *prompt_tokens,
                    *completion_tokens,
                );
                request_terminal_event!(
                    warn,
                    &self.span,
                    terminal_at_unix_s,
                    "rejected",
                    *prompt_tokens,
                    *completion_tokens,
                    message
                );
            }
        }
    }

    fn record_terminal_fields(
        &self,
        terminal_at_unix_s: f64,
        finish_reason: &'static str,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) {
        self.span.record("terminal_at_unix_s", terminal_at_unix_s);
        self.span.record("finish_reason", finish_reason);
        self.span.record("prompt_tokens", prompt_tokens as u64);
        self.span
            .record("completion_tokens", completion_tokens as u64);
    }

    /// A sink backed by its own private channel, for direct drivers
    /// (benchmarks, integration tests, the simulator) that consume one
    /// request's events without the shared frontend demux. The returned
    /// receiver yields the tagged events; the cancel flag is never tripped.
    pub fn standalone() -> (Self, TokenStreamReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        let sink = Self::new(Arc::from("local"), tx, Arc::new(AtomicBool::new(false)));
        (sink, rx)
    }
}

fn finish_reason_label(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Error => "error",
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

#[derive(Clone)]
pub struct EngineHandle {
    inner: Arc<EngineInner>,
    servable_len: Option<u32>,
    /// KV pool capacity in blocks + block size, or `None` if the engine did not
    /// report it. See [`KvCapacity`].
    kv_capacity: Option<KvCapacity>,
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

    #[allow(clippy::result_large_err)]
    pub fn submit(
        &self,
        req: GenerateRequest,
    ) -> std::result::Result<(), mpsc::error::SendError<GenerateRequest>> {
        let result = match self.inner.submit_tx.as_ref() {
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
        };
        if result.is_ok() {
            metrics::counter!("openinfer_engine_requests_submitted_total").increment(1);
        }
        result
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

impl Drop for EngineInner {
    fn drop(&mut self) {
        let _ = self.submit_tx.take();
        let _ = self.command_tx.take();
        if let Some(join_handle) = self.join_handle.take() {
            if join_handle.thread().id() != thread::current().id() {
                let _ = join_handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

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
