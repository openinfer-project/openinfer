pub mod kernel_plan;

mod batch_decode;
pub(crate) mod batch_decode_graph;
pub(crate) mod config;
mod decode_buffers;
mod executor;
mod ffi;
mod ops;
mod prefill;
pub mod prefill_buffers;
pub(crate) mod recurrent;
pub(crate) mod recurrent_state;
mod scheduler;
mod unified_forward;
mod weights;

use std::path::Path;

use anyhow::{Result, anyhow};
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions};

pub use kernel_plan::kernel_plan;

/// Low-level Qwen3.5 execution interface.
///
/// This is for model-local tests, debugging, and benchmarks. The root server
/// should use `start_engine` instead.
pub mod runtime {
    pub use crate::batch_decode_graph::MAX_BATCH;
    pub use crate::executor::{
        DecodePlan, DecodeRequestResult, DecodeResult, DecodeStepItem, PrefillPlan,
        PrefillRequestResult, PrefillResult, PrefillStepItem, Qwen35Executor, RequestId,
    };
    pub use crate::scheduler::start_with_capacity;
    pub use crate::weights::Qwen35Model;
}

/// Public operator surface used by Qwen3.5-local benches.
pub mod runtime_ops {
    pub use crate::ops::{
        gated_delta_rule_prefill_chunkwise_into, rms_norm_batch_offset_into, rms_norm_offset_into,
    };
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    start_engine_with_capacity(model_path, options, batch_decode_graph::MAX_BATCH)
}

pub fn start_engine_with_capacity(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
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
    let model = weights::Qwen35Model::from_safetensors_with_device_options(
        model_path,
        enable_cuda_graph,
        device_ordinal,
    )?;
    scheduler::start_with_capacity(model, seed, max_batch)
}
