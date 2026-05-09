use std::ptr;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::{
    ReduceOp,
    safe::{Comm, group_end, group_start},
};
use half::bf16;
use pegainfer_kernels::ffi;

use crate::{
    config::Config,
    model::{
        AttentionWeights, CompressorWeights, ExpertWeights, FfnWeights, IndexerWeights,
        QuantLinearRef, RankWeightView, TensorRef,
    },
    weights::RankGpuContext,
};

mod attention;
mod attention_base;
mod block;
mod collectives;
mod compressor;
mod core;
mod indexer;
mod moe;
mod state;

pub use self::attention::*;
pub use self::attention_base::*;
pub use self::block::*;
pub use self::collectives::*;
pub use self::compressor::*;
pub use self::core::*;
pub use self::indexer::*;
pub use self::moe::*;
pub use self::state::*;
