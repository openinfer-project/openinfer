use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec};

pub use pegainfer_kernels::ops::{
    argmax, argmax_batch_bf16_into, flashinfer_topk_row_states_bytes,
};

/// GPU sampling: temperature -> softmax -> top-k -> top-p -> multinomial.
///
/// Root owns request sampling policy; the kernels crate only sees primitive
/// launch parameters.
pub fn gpu_sample(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    params: &SamplingParams,
    random_val: f32,
) -> Result<u32> {
    pegainfer_kernels::ops::gpu_sample(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        params.temperature,
        params.top_k,
        params.top_p,
        random_val,
    )
}

/// GPU sampling into pre-allocated buffers.
pub fn gpu_sample_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
    params: &SamplingParams,
    random_val: f32,
) -> Result<u32> {
    pegainfer_kernels::ops::gpu_sample_into(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        valid_scratch,
        out,
        params.temperature,
        params.top_k,
        params.top_p,
        random_val,
    )
}
