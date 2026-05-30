//! Text-only Kimi-K2.6 model crate.
//!
//! The current crate stage owns the compile-checked operator API surface and
//! text-only config probing. CUDA/runtime bodies land behind these headers.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::path::Path;

use anyhow::Result;
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions};

#[cfg(feature = "kimi-k2")]
pub mod batch_decode_trace;
pub mod config;
#[cfg(feature = "kernel-report")]
pub mod kernel_report;
#[cfg(feature = "kimi-k2")]
mod runner;
#[cfg(feature = "kimi-k2")]
mod typed_scratch;
#[cfg(feature = "kimi-k2")]
mod weights;

pub use config::probe_config_json;

#[cfg(feature = "kimi-k2")]
pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    runner::start_engine(model_path, options)
}

#[cfg(not(feature = "kimi-k2"))]
pub fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!("Kimi-K2 runtime is feature-gated; rebuild with --features kimi-k2")
}
