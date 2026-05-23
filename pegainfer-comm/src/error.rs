//! Public error type. Backend errors from the upstream wrapper crates are
//! erased through [`Error::Backend`] so the public surface stays narrow.

use std::error::Error as StdError;

use thiserror::Error;

/// Result alias for `pegainfer-comm` public API.
pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for `pegainfer-comm`.
#[derive(Debug, Error)]
pub enum Error {
    /// A parameter passed to a backend method was malformed or inconsistent
    /// with the construction-time topology.
    #[error("invalid parameter: {0}")]
    InvalidParam(&'static str),

    /// Backend-internal failure (CUDA error, fabric transfer error, kernel
    /// launch failure, ...). The underlying error is type-erased so the
    /// public surface does not depend on wrapper-crate types.
    #[error("backend error: {source}")]
    Backend {
        /// Erased backend error.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl Error {
    /// Wrap an `anyhow::Error` (the wrapper crates' error type) as
    /// [`Error::Backend`].
    pub(crate) fn from_anyhow(err: anyhow::Error) -> Self {
        Self::Backend { source: err.into() }
    }
}
