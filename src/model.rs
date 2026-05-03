//! Model implementations: Qwen3 and Qwen3.5.

pub(crate) mod cuda_graph {
    pub(crate) use pegainfer_core::cuda_graph::*;
}
pub(crate) mod kv_cache {
    pub(crate) use pegainfer_core::kv_cache::*;
}

pub mod qwen3;
pub mod qwen35;

pub use pegainfer_core::model::{GenerationState, ModelForward};
pub use qwen3::{ModelRuntimeConfig, Qwen3Model, Qwen3State, TensorParallelConfig};
pub use qwen35::Qwen35Model;
