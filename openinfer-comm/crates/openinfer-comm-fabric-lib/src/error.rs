use cuda_lib::{driver::CudaDriverError, rt::CudartError};
use syscalls::Errno;

pub type Result<T> = std::result::Result<T, FabricLibError>;

#[derive(Clone, Debug, thiserror::Error)]
pub enum FabricLibError {
    #[error("DomainError: {0}")]
    Domain(String),
    #[error("{0}")]
    Verbs(#[from] VerbsError),
    #[error("VerbsCompletionError: {0}")]
    VerbsCompletionError(String),
    #[error("CompletionError: {0}")]
    CompletionError(String),
    #[error("{0}")]
    CudaDriver(#[from] CudaDriverError),
    #[error("{0}")]
    Cudart(#[from] CudartError),
    #[error("{0}")]
    Errno(#[from] Errno),
    #[error("FabricLibError: {0}")]
    Custom(&'static str),
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("VerbsError: code {code}, context: {context}")]
pub struct VerbsError {
    pub code: Errno,
    pub context: &'static str,
}

impl VerbsError {
    pub fn with_last_os_error(context: &'static str) -> Self {
        Self {
            code: Errno::new(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ),
            context,
        }
    }

    pub fn with_code(code: i32, context: &'static str) -> Self {
        Self { code: Errno::new(code), context }
    }
}
