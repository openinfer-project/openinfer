//! Root-local model implementations.

pub(crate) mod cuda_graph {
    pub(crate) use pegainfer_core::cuda_graph::*;
}
pub(crate) mod kv_cache {
    pub(crate) use pegainfer_core::kv_cache::*;
}

pub mod qwen35;

pub use qwen35::Qwen35Model;
