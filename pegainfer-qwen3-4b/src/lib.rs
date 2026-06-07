pub mod kernel_plan;

mod batch_decode;
mod batch_decode_buffers;
mod batch_decode_dag;
pub mod batch_decode_trace;
mod config;
mod executor;
pub mod kernel_bench;
mod lora;
mod prefill;
mod scheduler;
mod unified_forward;
mod weights;

use std::path::Path;

use anyhow::Result;
use pegainfer_core::engine::{EngineHandle, EngineLoadOptions};

pub use kernel_plan::kernel_plan;

/// Low-level Qwen3 execution interface.
///
/// This is the production phase boundary used by the Qwen3 scheduler and by
/// model-local benchmarks. The root server should use `start_engine` instead.
pub mod runtime {
    pub use crate::executor::{
        DecodePlan, DecodeRequestResult, DecodeResult, DecodeStepItem, PrefillPlan,
        PrefillRequestResult, PrefillResult, PrefillStepItem, Qwen3Executor, RequestId,
        UnifiedPlan, UnifiedResult,
    };
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    scheduler::start_qwen3(model_path, enable_cuda_graph, &device_ordinals, seed)
}

pub fn start_engine_with_lora_control(
    model_path: &Path,
    options: EngineLoadOptions,
) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    scheduler::start_qwen3_with_lora_control(model_path, enable_cuda_graph, &device_ordinals, seed)
}
