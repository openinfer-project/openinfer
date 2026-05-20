#[cfg(feature = "kernel-call-trace")]
use anyhow::Result;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_trace;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_kernels::tensor::KernelCall;

#[cfg(feature = "kernel-call-trace")]
use crate::batch_decode_buffers::BatchDecodeBuffers;
#[cfg(feature = "kernel-call-trace")]
use crate::weights::{ModelRuntimeConfig, Qwen3Model};

pub const MODEL: &str = "qwen3-4b";
pub const PHASE_DECODE: &str = "decode";
pub const HIDDEN_SIZE: usize = 2560;
pub const INTERMEDIATE_SIZE: usize = 9728;
pub const NUM_LAYERS: usize = 36;
pub const NUM_Q_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM_VALUE: usize = 128;
pub const KV_DIM_VALUE: usize = NUM_KV_HEADS * HEAD_DIM_VALUE;
pub const RMS_NORM_EPS: f32 = 1.0e-6;

#[cfg(feature = "kernel-call-trace")]
pub fn trace_decode_kernel_calls(
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
) -> Result<Vec<KernelCall>> {
    anyhow::ensure!(batch_size > 0, "batch_size must be greater than zero");
    anyhow::ensure!(kv_len > 0, "kv_len must be greater than zero");

    let model = Qwen3Model::from_safetensors_with_runtime(
        model_path,
        ModelRuntimeConfig {
            enable_cuda_graph: false,
            tensor_parallel: None,
            device_ordinal: 0,
        },
    )?;
    let mut kv_states = (0..batch_size)
        .map(|_| {
            let mut kv = model.alloc_kv();
            if kv_len > 1 {
                kv.ensure_capacity(kv_len - 1)?;
                kv.advance(kv_len - 1);
            }
            Ok(kv)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut bufs = BatchDecodeBuffers::new(
        model.device_ctx(),
        model.config().hidden_size,
        model.local_q_dim(),
        model.local_kv_dim(),
        model.local_intermediate_size(),
        model.config().vocab_size,
        batch_size,
        model.kv_pool.capacity_pages(),
        model.kv_pool.padding_page_id(),
        model.local_num_attention_heads(),
    )?;
    let token_ids = vec![0_u32; batch_size];
    let ((), calls) = call_trace::collect_result(|| {
        let mut kv_refs = kv_states.iter_mut().collect::<Vec<_>>();
        model.batch_decode(&token_ids, &mut kv_refs, &mut bufs)
    })?;
    Ok(calls)
}

pub fn normalize_call_site(label: &str) -> String {
    let Some(rest) = label.strip_prefix('L') else {
        return label.to_string();
    };
    let digit_count = rest
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 || rest.as_bytes().get(digit_count) != Some(&b'.') {
        return label.to_string();
    }
    format!("layer.*{}", &rest[digit_count..])
}
