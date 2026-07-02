use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_DEEPGEMM_MQA_HEAD_DIM: usize = 128;
pub const GLM52_DEEPGEMM_MQA_BLOCK_KV: usize = 128;
pub const GLM52_DEEPGEMM_MQA_SPLIT_KV: usize = 256;
pub const GLM52_DEEPGEMM_MQA_FP8_ELEM_SIZE: usize = 1;
pub const GLM52_DEEPGEMM_MQA_BF16_ELEM_SIZE: usize = 2;
pub const GLM52_DEEPGEMM_MQA_F32_ELEM_SIZE: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52DeepGemmMqaLogitsShape {
    pub batch_size: usize,
    pub next_n: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_blocks: usize,
    pub block_kv: usize,
    pub is_context_lens_2d: bool,
    pub is_varlen: bool,
    pub logits_stride: usize,
    pub block_table_stride: usize,
    pub num_sms: usize,
}

impl Glm52DeepGemmMqaLogitsShape {
    pub fn validate(self) -> Result<()> {
        ensure!(self.batch_size > 0, "batch_size must be positive");
        ensure!(
            self.next_n == 1 || self.next_n == 2,
            "next_n must be 1 or 2"
        );
        ensure!(self.num_heads > 0, "num_heads must be positive");
        ensure!(
            self.head_dim == GLM52_DEEPGEMM_MQA_HEAD_DIM,
            "head_dim must be {}",
            GLM52_DEEPGEMM_MQA_HEAD_DIM
        );
        ensure!(
            128 % self.num_heads == 0,
            "128 must be divisible by num_heads"
        );
        ensure!(self.block_kv > 0, "block_kv must be positive");
        ensure!(
            GLM52_DEEPGEMM_MQA_SPLIT_KV.is_multiple_of(self.block_kv),
            "split_kv must be divisible by block_kv"
        );
        ensure!(
            self.logits_stride
                .is_multiple_of(GLM52_DEEPGEMM_MQA_SPLIT_KV),
            "logits_stride must be divisible by split_kv"
        );
        ensure!(self.num_sms > 0, "num_sms must be positive");
        Ok(())
    }

    pub fn schedule_metadata_len(self) -> usize {
        let aligned_bs = self.batch_size.div_ceil(32) * 32;
        if self.is_varlen {
            3 * aligned_bs + 1
        } else {
            aligned_bs
        }
    }
}

pub fn glm52_deepgemm_paged_mqa_metadata_launch(
    ctx: &DeviceContext,
    shape: Glm52DeepGemmMqaLogitsShape,
    context_lens: &mut CudaSlice<i32>,
    schedule_metadata: &mut CudaSlice<i32>,
    indices: Option<&CudaSlice<i32>>,
) -> Result<()> {
    shape.validate()?;
    let need = shape.schedule_metadata_len();
    ensure!(
        schedule_metadata.len() >= need,
        "GLM5.2 DeepGEMM MQA schedule_metadata too small: have {}, need {need}",
        schedule_metadata.len()
    );
    ensure!(
        context_lens.len() >= shape.batch_size,
        "GLM5.2 DeepGEMM MQA context_lens too small: have {}, need {}",
        context_lens.len(),
        shape.batch_size
    );

    let (cl_ptr, _cl_guard) = context_lens.device_ptr_mut(&ctx.stream);
    let (sm_ptr, _sm_guard) = schedule_metadata.device_ptr_mut(&ctx.stream);
    let indices_ptr = match indices {
        Some(buf) => {
            ensure!(shape.is_varlen, "indices provided but is_varlen=false");
            ensure!(buf.len() >= shape.batch_size, "indices too small");
            let (ptr, _guard) = buf.device_ptr(&ctx.stream);
            ptr as *const i32
        }
        None => {
            ensure!(!shape.is_varlen, "is_varlen=true but no indices provided");
            std::ptr::null()
        }
    };

    let result = unsafe {
        ffi::glm52_deepgemm_paged_mqa_metadata_cuda(
            cl_ptr as *mut i32,
            sm_ptr as *mut i32,
            shape.batch_size as i32,
            shape.next_n as i32,
            shape.block_kv as i32,
            shape.num_sms as i32,
            shape.is_context_lens_2d,
            shape.is_varlen,
            indices_ptr,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM MQA metadata launch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_deepgemm_paged_mqa_logits_launch(
    ctx: &DeviceContext,
    shape: Glm52DeepGemmMqaLogitsShape,
    q: &CudaSlice<u8>,
    kv_cache: &CudaSlice<u8>,
    kv_cache_scales: &CudaSlice<f32>,
    weights: &CudaSlice<u8>,
    context_lens: &CudaSlice<i32>,
    logits: &mut CudaSlice<u8>,
    block_table: &CudaSlice<i32>,
    indices: Option<&CudaSlice<i32>>,
    schedule_meta: &mut CudaSlice<i32>,
) -> Result<()> {
    shape.validate()?;

    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (kv_ptr, _kv_guard) = kv_cache.device_ptr(&ctx.stream);
    let (kvs_ptr, _kvs_guard) = kv_cache_scales.device_ptr(&ctx.stream);
    let (w_ptr, _w_guard) = weights.device_ptr(&ctx.stream);
    let (cl_ptr, _cl_guard) = context_lens.device_ptr(&ctx.stream);
    let (logits_ptr, _logits_guard) = logits.device_ptr_mut(&ctx.stream);
    let (bt_ptr, _bt_guard) = block_table.device_ptr(&ctx.stream);
    let (sm_ptr, _sm_guard) = schedule_meta.device_ptr_mut(&ctx.stream);
    let indices_ptr = match indices {
        Some(buf) => {
            ensure!(shape.is_varlen, "indices provided but is_varlen=false");
            let (ptr, _guard) = buf.device_ptr(&ctx.stream);
            ptr as *const i32
        }
        None => {
            ensure!(!shape.is_varlen, "is_varlen=true but no indices provided");
            std::ptr::null()
        }
    };

    let result = unsafe {
        ffi::glm52_deepgemm_paged_mqa_logits_cuda(
            q_ptr as *const std::ffi::c_void,
            kv_ptr as *const std::ffi::c_void,
            kvs_ptr as *const f32,
            w_ptr as *const std::ffi::c_void,
            cl_ptr as *const i32,
            logits_ptr as *mut std::ffi::c_void,
            bt_ptr as *const i32,
            indices_ptr,
            sm_ptr as *mut i32,
            shape.batch_size as i32,
            shape.next_n as i32,
            shape.num_heads as i32,
            shape.head_dim as i32,
            shape.num_kv_blocks as i32,
            shape.block_kv as i32,
            shape.is_context_lens_2d,
            shape.is_varlen,
            shape.logits_stride as i32,
            shape.block_table_stride as i32,
            shape.num_sms as i32,
            GLM52_DEEPGEMM_MQA_FP8_ELEM_SIZE as i32,
            GLM52_DEEPGEMM_MQA_FP8_ELEM_SIZE as i32,
            GLM52_DEEPGEMM_MQA_FP8_ELEM_SIZE as i32,
            GLM52_DEEPGEMM_MQA_F32_ELEM_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM MQA logits launch failed: {err}"))
}
