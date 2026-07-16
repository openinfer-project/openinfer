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
mod executor;
mod ffi;
mod logprobs;
mod ops;
mod prefill;
pub mod prefill_buffers;
pub(crate) mod recurrent;
pub(crate) mod recurrent_state;
mod scheduler;
mod tp_executor;
mod unified_forward;
mod weights;

use std::path::Path;

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
        DecodePlan, DecodeRequestResult, DecodeResult, DecodeStepItem, PrefillPlan,
        PrefillRequestResult, PrefillResult, PrefillStepItem, Qwen35Executor, RequestId,
    };
    pub use crate::scheduler::start_with_capacity;
    pub use crate::tp_executor::Qwen35TpExecutor;
    pub use crate::weights::Qwen35Model;
}

/// Public operator surface used by Qwen3.5-local benches.
pub mod runtime_ops {
    pub use crate::ops::{
        gated_delta_rule_prefill_chunkwise_into, rms_norm_batch_offset_into, rms_norm_offset_into,
    };
}

pub fn start_engine(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    start_engine_with_capacity(model_path, options, max_batch, max_prefill_tokens)
}

#[derive(Clone, Debug)]
pub struct Qwen35LaunchOptions {
    /// CUDA device for single-GPU loads (ignored when `tp_size > 1`).
    pub device_ordinal: usize,
    /// Tensor-parallel world size; `> 1` uses devices `0..tp_size`.
    pub tp_size: usize,
    /// TP Phase 1 supports eager-only multi-GPU execution.
    pub cuda_graph: bool,
    pub max_batch: usize,
    pub max_prefill_tokens: usize,
}

impl Qwen35LaunchOptions {
    fn device_ordinals(&self) -> Result<Vec<usize>> {
        anyhow::ensure!(self.tp_size >= 1, "Qwen3.5 tp_size must be >= 1");
        Ok(if self.tp_size == 1 {
            vec![self.device_ordinal]
        } else {
            (0..self.tp_size).collect()
        })
    }
}

/// Start the Qwen3.5 engine for the server. TP Phase 1 supports eager-only
/// multi-GPU execution; single-GPU keeps the existing CUDA Graph-capable path.
pub fn launch(
    model_path: &Path,
    device_ordinal: usize,
    cuda_graph: bool,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    launch_with_options(
        model_path,
        Qwen35LaunchOptions {
            device_ordinal,
            tp_size: 1,
            cuda_graph,
            max_batch: batch_decode_graph::MAX_BATCH,
            max_prefill_tokens,
        },
    )
}

pub fn launch_with_options(
    model_path: &Path,
    options: Qwen35LaunchOptions,
) -> Result<EngineHandle> {
    let device_ordinals = options.device_ordinals()?;
    start_engine_with_capacity(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: options.cuda_graph,
            device_ordinals,
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
        options.max_batch,
        options.max_prefill_tokens,
    )
}

pub fn start_engine_with_capacity(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    if device_ordinals.len() > 1 {
        if enable_cuda_graph {
            return Err(anyhow!(
                "Qwen3.5 TP Phase 1 supports eager execution only; disable CUDA Graph"
            ));
        }
        let model_path = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path must be valid UTF-8"))?;
        return scheduler::start_tp_with_capacity(
            model_path,
            seed,
            &device_ordinals,
            max_batch,
            max_prefill_tokens,
        );
    }

    anyhow::ensure!(
        enable_cuda_graph,
        "Qwen3.5 decode always captures CUDA Graphs; --cuda-graph=false is not supported"
    );
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
    let model = weights::Qwen35Model::from_safetensors(model_path, device_ordinal, max_batch)?;
    scheduler::start(model, seed, max_prefill_tokens)
}

#[cfg(test)]
mod tests {
    use super::Qwen35LaunchOptions;

    #[test]
    fn launch_options_reject_zero_tp_size() {
        let options = Qwen35LaunchOptions {
            device_ordinal: 3,
            tp_size: 0,
            cuda_graph: false,
            max_batch: 1,
            max_prefill_tokens: 1,
        };

        let err = options.device_ordinals().unwrap_err().to_string();
        assert!(err.contains("tp_size must be >= 1"));
    }
}
