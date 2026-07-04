//! Text-only Kimi-K2.6 model crate.

#![allow(incomplete_features)]
// `use super::*` is the flat-module-layout idiom for these tightly-coupled
// submodules (weights.rs + weights/*, runner/worker.rs + worker/*), and the
// safetensors-mirroring field names (gate_proj, weight_packed, …) read clearer
// with their shared affix than without. Both are pedantic-only lints.
#![allow(clippy::wildcard_imports, clippy::struct_field_names)]
#![feature(generic_const_exprs)]

use std::path::Path;

use anyhow::Result;
use openinfer_core::engine::EpBackend;
use openinfer_core::engine::{EngineHandle, EngineLoadOptions};

pub mod batch_decode_trace;
pub(crate) mod config;
#[cfg(feature = "kernel-report")]
pub mod kernel_report;
mod runner;
mod typed_scratch;
mod weights;

pub use config::{KIMI_K2_LAYERS, probe_config_json};

#[allow(clippy::needless_pass_by_value)]
pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    runner::start_engine(model_path, &options)
}

/// Server-facing launch knobs for Kimi-K2. The binary passes raw CLI values;
/// [`launch`] owns the EP topology policy — validating TP/DP, deriving the EP
/// world and its device ordinals — so the server never hardcodes Kimi's
/// parallel layout.
#[derive(Clone, Copy, Debug)]
pub struct KimiLaunchOptions {
    pub tp_size: usize,
    pub dp_size: usize,
    pub ep_backend: EpBackend,
    pub cuda_graph: bool,
}

/// Start the Kimi-K2 engine from server-facing [`KimiLaunchOptions`].
pub fn launch(model_path: &Path, options: KimiLaunchOptions) -> Result<EngineHandle> {
    use log::info;
    use openinfer_core::parallel::ParallelConfig;

    anyhow::ensure!(
        options.tp_size > 0 && options.dp_size > 0,
        "Kimi-K2 --tp-size and --dp-size must be positive"
    );
    let parallel = ParallelConfig::new(options.tp_size, options.dp_size);
    info!(
        "Kimi-K2 EP options: tp_size={}, dp_size={}, ep_world={}, ep_backend={:?}",
        options.tp_size,
        options.dp_size,
        parallel.ep_world(),
        options.ep_backend
    );
    runner::start_engine(
        model_path,
        &EngineLoadOptions {
            enable_cuda_graph: options.cuda_graph,
            enable_prefill_profile: false,
            device_ordinals: (0..parallel.ep_world()).collect(),
            parallel_config: Some(parallel),
            ep_backend: options.ep_backend,
            seed: 42,
        },
    )
}
