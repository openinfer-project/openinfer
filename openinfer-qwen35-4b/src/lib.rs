// The whole crate is gated: Qwen3.5 needs the Triton AOT kernels from
// `openinfer-kernels/qwen35-4b`, which need Python + Triton at build time.
// Without the feature this compiles to an empty crate so plain workspace
// builds stay Python-free.
#![cfg(feature = "qwen35-4b")]

pub mod kernel_plan;

mod batch_decode;
pub(crate) mod batch_decode_graph;
pub(crate) mod config;
mod decode_buffers;
mod dflash;
mod executor;
mod ffi;
mod logprobs;
mod ops;
mod prefill;
pub mod prefill_buffers;
pub(crate) mod recurrent;
pub(crate) mod recurrent_state;
mod scheduler;
pub mod speculative;
mod unified_forward;
mod verify_buffers;
mod weights;

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use openinfer_core::engine::{EngineHandle, EngineLoadOptions, EpBackend};

pub use kernel_plan::kernel_plan;
pub use scheduler::DEFAULT_MAX_PREFILL_TOKENS;

/// Low-level Qwen3.5 execution interface.
///
/// This is for model-local tests, debugging, and benchmarks. The root server
/// should use `start_engine` instead.
pub mod runtime {
    pub use crate::batch_decode_graph::MAX_BATCH;
    pub use crate::executor::{
        DecodePlan, DecodeRequestResult, DecodeResult, DecodeStepItem, ExecutorStateSummary,
        PrefillPlan, PrefillRequestResult, PrefillResult, PrefillStepItem, Qwen35Executor,
        RequestId,
    };
    pub use crate::scheduler::start_with_capacity;
    pub use crate::speculative::{
        VerifiedToken, VerifyPlan, VerifyRequestResult, VerifyResult, VerifyStepItem,
    };
    pub use crate::weights::Qwen35Model;
}

/// Public operator surface used by Qwen3.5-local benches.
pub mod runtime_ops {
    pub use crate::ops::{
        gated_delta_rule_prefill_chunkwise_into, rms_norm_batch_offset_into, rms_norm_offset_into,
    };
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    start_engine_with_capacity(
        model_path,
        options,
        batch_decode_graph::MAX_BATCH,
        DEFAULT_MAX_PREFILL_TOKENS,
    )
}

/// Start the Qwen3.5 engine for the server. Qwen3.5 is single-GPU, so the
/// knobs are the device ordinal, whether to capture a decode CUDA Graph, and
/// the per-step chunked-prefill budget (from `--max-prefill-tokens`).
pub fn launch(
    model_path: &Path,
    device_ordinal: usize,
    cuda_graph: bool,
    max_prefill_tokens: usize,
    dflash_draft_model_path: Option<PathBuf>,
) -> Result<EngineHandle> {
    start_engine_with_capacity_and_dflash(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: cuda_graph,
            enable_prefill_profile: false,
            device_ordinals: vec![device_ordinal],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
        batch_decode_graph::MAX_BATCH,
        max_prefill_tokens,
        dflash_draft_model_path,
    )
}

pub fn start_engine_with_capacity(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    start_engine_with_capacity_and_dflash(model_path, options, max_batch, max_prefill_tokens, None)
}

pub fn start_engine_with_capacity_and_dflash(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
    dflash_draft_model_path: Option<PathBuf>,
) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let device_ordinal = match device_ordinals.as_slice() {
        [] => 0,
        [device_ordinal] => *device_ordinal,
        ordinals => {
            return Err(anyhow!(
                "Qwen3.5 engine supports exactly one CUDA device, got {}",
                ordinals.len()
            ));
        }
    };
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow!("model path must be valid UTF-8"))?;
    let dflash_path = dflash_draft_model_path
        .as_ref()
        .map(|path| {
            path.to_str()
                .ok_or_else(|| anyhow!("DFlash draft model path must be valid UTF-8"))
        })
        .transpose()?;
    let dflash_reservation = dflash_path
        .map(|path| dflash::DFlashMemoryReservation::from_path(path, max_batch))
        .transpose()?;
    let model = weights::Qwen35Model::from_safetensors_with_device_options_and_reservation(
        model_path,
        enable_cuda_graph,
        device_ordinal,
        dflash_reservation.as_ref(),
    )?;
    scheduler::start_with_capacity_and_dflash(
        model,
        seed,
        max_batch,
        max_prefill_tokens,
        dflash_draft_model_path,
    )
}
