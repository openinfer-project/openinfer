use std::{
    error::Error,
    fmt,
    path::PathBuf,
    sync::Arc,
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
    Pplx,
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
    pub token_tx: mpsc::UnboundedSender<TokenEvent>,
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

#[derive(Clone)]
pub struct EngineHandle {
    inner: Arc<EngineInner>,
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
        }
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
            EngineCommand::Control(EngineControlRequest::UnloadLoraAdapter { .. }) => {
                panic!("expected LoRA load command")
            }
            EngineCommand::Control(EngineControlRequest::ListLoraAdapters { .. }) => {
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
            EngineCommand::Control(EngineControlRequest::LoadLoraAdapter { .. }) => {
                panic!("expected LoRA list command")
            }
            EngineCommand::Control(EngineControlRequest::UnloadLoraAdapter { .. }) => {
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
            EngineCommand::Control(EngineControlRequest::LoadLoraAdapter { .. }) => {
                panic!("expected LoRA unload command")
            }
            EngineCommand::Control(EngineControlRequest::ListLoraAdapters { .. }) => {
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
