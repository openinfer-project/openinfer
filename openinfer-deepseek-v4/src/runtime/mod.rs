use std::ptr;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::{ReduceOp, safe::Comm};
use half::bf16;
use openinfer_kernels::ffi;

use crate::{
    config::Config,
    model::{
        AttentionWeights, CompressorWeights, FfnWeights, IndexerWeights, QuantLinearRef,
        RankWeightView, TensorRef,
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
#[cfg(feature = "pplx-ep")]
mod moe_pplx;
mod state;

pub use self::attention::*;
pub use self::attention_base::*;
pub use self::block::*;
pub use self::collectives::*;
pub use self::compressor::*;
pub use self::core::*;
pub use self::indexer::*;
pub use self::moe::*;
#[cfg(feature = "pplx-ep")]
pub(crate) use self::state::MoePplxRunContext;
pub(crate) use self::state::MoeRunContext;
pub use self::state::*;
