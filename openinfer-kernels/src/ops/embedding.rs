use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

/// Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
pub fn embedding_decode_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: &CudaSlice<u32>,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(embed.cols, out.len);

    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (token_ptr, _gt) = token_id.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_decode_cuda(
            embed_ptr as *const ffi::Half,
            token_ptr as *const u32,
            out_ptr as *mut ffi::Half,
            embed.cols as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// Batched embedding lookup
pub fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<u32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let (e_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (t_ptr, _gt) = token_ids_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_batched_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const u32,
            o_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// Vocab-sharded batched embedding lookup for tensor-parallel models.
///
/// Tokens outside `[vocab_start, vocab_start + part_vocab_size)` write zeros;
/// callers should all-reduce `out` across ranks to recover the full embedding.
pub fn embedding_batch_vocab_shard(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<u32>,
    out: &mut HiddenStates,
    vocab_start: u32,
    part_vocab_size: u32,
) -> Result<()> {
    let (e_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (t_ptr, _gt) = token_ids_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_batched_vocab_shard_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const u32,
            o_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            vocab_start,
            part_vocab_size,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    #[test]
    fn embedding_batch_vocab_shard_masks_nonlocal_tokens() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let hidden_size = 3;
        let seq_len = 4;
        let embed_host = vec![
            bf16::from_f32(10.0),
            bf16::from_f32(11.0),
            bf16::from_f32(12.0),
            bf16::from_f32(20.0),
            bf16::from_f32(21.0),
            bf16::from_f32(22.0),
        ];
        let embed = DeviceMatrix::from_host(&ctx, &embed_host, 2, hidden_size).expect("embed");
        let token_ids = ctx.stream.clone_htod(&[4_u32, 5, 6, 4]).expect("token ids");
        let mut out = HiddenStates::zeros(&ctx, hidden_size, seq_len).expect("out");

        embedding_batch_vocab_shard(&ctx, &embed, &token_ids, &mut out, 4, 2)
            .expect("embedding shard");
        let got = ctx.stream.clone_dtoh(&out.data).expect("dtoh");
        ctx.sync().expect("sync");
        let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();

        assert_eq!(
            got,
            vec![
                10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0
            ]
        );
    }
}
