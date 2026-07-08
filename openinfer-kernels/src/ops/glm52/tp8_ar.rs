//! GLM5.2 TP8 attention allreduce: the o_proj epilogue
//! collective for the attention-TP topology. Every rank contributes a partial
//! projection output for ALL bucket rows (its heads' share, full hidden
//! width); the kernel pair sums the 8 partials in a fixed order so every rank
//! ends with the bit-identical full result — the replicated-activation
//! topology relies on that identity for redundant routing/sampling. See
//! `csrc/glm52/glm52_tp8_ar.cu`.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

use super::moe_tp8::{GLM52_TP8_HIDDEN, GLM52_TP8_RANKS, GLM52_TP8_TOKENS};

/// LL packets per (row, src) hidden chunk: H/8 bf16 packed 6 (12 B) per
/// packet. This is also the pre-offset unit for peer pointers.
pub const GLM52_TP8_AR_CHUNK_PACKETS: usize = (GLM52_TP8_HIDDEN / GLM52_TP8_RANKS) * 2 / 12;

/// One layer slot's AR region in packets: parity(2) x stage(2: reduce-scatter
/// + broadcast) x rows x src x chunk packets. Multiply by layer-slot count
/// and 16 B for the buffer size.
pub const GLM52_TP8_AR_SLOT_PACKETS: usize =
    2 * 2 * GLM52_TP8_TOKENS * GLM52_TP8_RANKS * GLM52_TP8_AR_CHUNK_PACKETS;

/// AR LL buffer bytes for `layer_slots` layers (allocate via
/// [`super::moe_tp8::Glm52Tp8LlBuffer::alloc`]).
pub const fn glm52_tp8_ar_buffer_bytes(layer_slots: usize) -> usize {
    layer_slots * GLM52_TP8_AR_SLOT_PACKETS * 16
}

/// Launch the two-shot AR chain (push, fused reduce+broadcast, recv) for one
/// layer. `partial` is this rank's partial output for `rows` bucket rows
/// (`[rows][HIDDEN]` bf16); `out` receives the reduced result, bit-identical
/// on all ranks. `peer_ar[p]` is rank p's AR buffer VA pre-offset by
/// `myrank * GLM52_TP8_AR_CHUNK_PACKETS` packets (this rank's src slot).
/// Shares the step epoch with the MoE chain:
/// [`super::moe_tp8::glm52_moe_tp8_epoch_advance`] once per step.
/// `active_rows` is the want-mask (device leading-active row count, staged
/// identically on every rank): pad rows skip the wire and get zero-filled
/// `out`. `None` = all `rows` active.
#[allow(clippy::too_many_arguments)]
pub fn glm52_tp8_ar_launch(
    ctx: &DeviceContext,
    layer_slot: usize,
    rows: usize,
    partial: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    ar_local: u64,
    peer_ar: [u64; GLM52_TP8_RANKS],
    epoch_dev: &CudaSlice<u64>,
    active_rows: Option<&CudaSlice<i32>>,
    myrank: usize,
) -> Result<()> {
    ensure!(myrank < GLM52_TP8_RANKS, "AR myrank {myrank} out of range");
    ensure!(
        rows >= 1 && rows <= GLM52_TP8_TOKENS,
        "AR rows {rows} out of 1..={GLM52_TP8_TOKENS}"
    );
    ensure!(
        partial.len() >= rows * GLM52_TP8_HIDDEN && out.len() >= rows * GLM52_TP8_HIDDEN,
        "AR hidden buffers too small: partial {}, out {}",
        partial.len(),
        out.len()
    );
    ensure!(!epoch_dev.is_empty(), "AR epoch_dev is empty");
    ensure!(
        ar_local != 0 && peer_ar.iter().all(|&p| p != 0),
        "AR LL pointers not wired"
    );

    let (partial_ptr, _g0) = partial.device_ptr(&ctx.stream);
    let (out_ptr, _g1) = out.device_ptr_mut(&ctx.stream);
    let (epoch_ptr, _g2) = epoch_dev.device_ptr(&ctx.stream);
    let active_ptr = match active_rows {
        Some(active) => active.device_ptr(&ctx.stream).0 as *const i32,
        None => std::ptr::null(),
    };
    let peer_ar: [*const std::ffi::c_void; GLM52_TP8_RANKS] =
        peer_ar.map(|p| p as *const std::ffi::c_void);
    unsafe {
        ffi::glm52_tp8_ar_launch_cuda(
            partial_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            ar_local as *mut std::ffi::c_void,
            peer_ar.as_ptr(),
            epoch_ptr as *const u64,
            active_ptr,
            layer_slot as i32,
            rows as i32,
            GLM52_TP8_RANKS as i32,
            myrank as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 TP8 attention allreduce launch failed: {err}"))
}
