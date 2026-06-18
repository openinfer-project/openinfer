#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionRecovery {
    Recoverable,
    DomainFatal,
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum ExecutionError {
    #[error("{message}")]
    StepFailed { message: String },
    #[error("worker command channel closed during {op}")]
    WorkerCommandChannelClosed { op: String },
    #[error("{worker} worker dropped {op} response")]
    WorkerResponseDropped { worker: String, op: String },
    #[error("worker {worker} panicked: {message}")]
    WorkerPanic { worker: String, message: String },
    #[error("unexpected worker response during {op}: {got}")]
    UnexpectedWorkerResponse { op: String, got: String },
}

impl ExecutionError {
    pub fn step_failed(message: impl Into<String>) -> Self {
        Self::StepFailed {
            message: message.into(),
        }
    }

    pub fn worker_command_channel_closed(op: impl Into<String>) -> Self {
        Self::WorkerCommandChannelClosed { op: op.into() }
    }

    pub fn worker_response_dropped(worker: impl Into<String>, op: impl Into<String>) -> Self {
        Self::WorkerResponseDropped {
            worker: worker.into(),
            op: op.into(),
        }
    }

    pub fn worker_panic(worker: impl Into<String>, message: impl Into<String>) -> Self {
        Self::WorkerPanic {
            worker: worker.into(),
            message: message.into(),
        }
    }

    pub fn unexpected_worker_response(op: impl Into<String>, got: impl Into<String>) -> Self {
        Self::UnexpectedWorkerResponse {
            op: op.into(),
            got: got.into(),
        }
    }

    pub fn recovery(&self) -> ExecutionRecovery {
        match self {
            Self::StepFailed { .. } => ExecutionRecovery::Recoverable,
            Self::WorkerCommandChannelClosed { .. }
            | Self::WorkerResponseDropped { .. }
            | Self::WorkerPanic { .. }
            | Self::UnexpectedWorkerResponse { .. } => ExecutionRecovery::DomainFatal,
        }
    }

    pub fn is_domain_fatal(&self) -> bool {
        self.recovery() == ExecutionRecovery::DomainFatal
    }
}

pub type ExecutionResult<T> = std::result::Result<T, ExecutionError>;
