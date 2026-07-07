use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::DeviceContext;

/// Advance the decode-step device inputs in place for a self-fed next step:
/// `token_ids[row] = argmax_indices[row]`, and `positions`/`slot_mapping`/
/// `seq_lens` each move one position forward. Enqueued between two whole-step
/// graph replays; a host prologue that follows instead simply overwrites all
/// four buffers.
pub fn glm52_decode_feed_launch(
    ctx: &DeviceContext,
    argmax_indices: &CudaSlice<i32>,
    token_ids: &mut CudaSlice<u32>,
    positions: &mut CudaSlice<u32>,
    slot_mapping: &mut CudaSlice<i64>,
    seq_lens: &mut CudaSlice<i32>,
    rows: usize,
) -> Result<()> {
    ensure!(rows > 0, "GLM5.2 decode feed needs at least one row");
    for (len, name) in [
        (argmax_indices.len(), "argmax_indices"),
        (token_ids.len(), "token_ids"),
        (positions.len(), "positions"),
        (slot_mapping.len(), "slot_mapping"),
        (seq_lens.len(), "seq_lens"),
    ] {
        ensure!(
            len >= rows,
            "GLM5.2 decode feed buffer {name} holds {len} rows, step needs {rows}"
        );
    }

    let (argmax_ptr, _argmax_guard) = argmax_indices.device_ptr(&ctx.stream);
    let (token_ptr, _token_guard) = token_ids.device_ptr_mut(&ctx.stream);
    let (pos_ptr, _pos_guard) = positions.device_ptr_mut(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_mapping.device_ptr_mut(&ctx.stream);
    let (seq_ptr, _seq_guard) = seq_lens.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::glm52_decode_feed_launch_cuda(
            argmax_ptr as *const i32,
            token_ptr as *mut u32,
            pos_ptr as *mut u32,
            slot_ptr as *mut i64,
            seq_ptr as *mut i32,
            rows as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 decode feed CUDA launch failed: {err}"))
}
