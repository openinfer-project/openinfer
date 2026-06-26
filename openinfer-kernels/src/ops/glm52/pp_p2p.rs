//! PP8 stage-boundary P2P spine op wrappers (Slice 0).
//!
//! Unlike the dense glm52 ops (`router.rs` et al.) where every tensor is a local
//! [`CudaSlice`], the send/wait ops straddle two GPUs: the *local* buffers
//! (`src_hidden`, `epoch`, `down_ack`, `deltas`, `err_code`, `my_flag`) resolve
//! through `device_ptr` as usual, while the *peer* targets (`peer_hidden`,
//! `peer_flag`, `up_ack`) arrive as raw `u64` virtual addresses already read off
//! the neighbour's allocation by `glm52::pp::peer`. Peer VAs are allocation
//! stable, so baking them into a captured graph node is sound.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Fixed parameters of one stage-boundary send, baked into the captured graph.
#[derive(Clone, Copy, Debug)]
pub struct Glm52PpSendParams {
    /// bf16 elements moved this step; must be a multiple of 8 (one `int4`).
    pub words: i32,
    /// Double-buffer depth `R` (`slot = epoch % ring`).
    pub ring: i32,
    /// Epochs to skip before recording into `deltas`.
    pub warmup: u64,
    /// `deltas` capacity; 0 when `deltas` is `None`.
    pub n_samples: u64,
    /// Per-spin timeout in ns; exceeding it traps the stage.
    pub deadline_ns: u64,
}

/// Chain head: advance the local epoch with no inbound wait.
pub fn glm52_pp_source_inject_launch(
    ctx: &DeviceContext,
    epoch: &mut CudaSlice<u64>,
) -> Result<()> {
    let (epoch_ptr, _g) = epoch.device_ptr_mut(&ctx.stream);
    unsafe { ffi::glm52_pp_source_inject(epoch_ptr as *mut u64, ctx.stream.cu_stream()) }
        .result()
        .map_err(|err| anyhow!("GLM5.2 PP source_inject launch failed: {err}"))
}

/// Acquire side: gate on the inbound flag, fence, ack the upstream stage.
pub fn glm52_pp_wait_hidden_launch(
    ctx: &DeviceContext,
    my_flag: &CudaSlice<u64>,
    epoch: &mut CudaSlice<u64>,
    up_ack_peer_va: u64,
    err_code: &mut CudaSlice<u32>,
    deadline_ns: u64,
    ring: i32,
) -> Result<()> {
    let (flag_ptr, _gf) = my_flag.device_ptr(&ctx.stream);
    let (epoch_ptr, _ge) = epoch.device_ptr_mut(&ctx.stream);
    let (err_ptr, _gc) = err_code.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_pp_wait_hidden(
            flag_ptr as *const u64,
            epoch_ptr as *mut u64,
            up_ack_peer_va as *mut u64,
            err_ptr as *mut u32,
            deadline_ns,
            ring,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 PP wait_hidden launch failed: {err}"))
}

/// Release side: remote-store the payload into the peer ring slot, fence,
/// release the flag, and (with `deltas`) record the forward half-RTT.
#[allow(clippy::too_many_arguments)]
pub fn glm52_pp_send_hidden_launch(
    ctx: &DeviceContext,
    src_hidden: &CudaSlice<bf16>,
    peer_hidden_va: u64,
    peer_flag_va: u64,
    epoch: &CudaSlice<u64>,
    down_ack: &CudaSlice<u64>,
    deltas: Option<&mut CudaSlice<u64>>,
    err_code: &mut CudaSlice<u32>,
    params: Glm52PpSendParams,
) -> Result<()> {
    let (src_ptr, _gs) = src_hidden.device_ptr(&ctx.stream);
    let (epoch_ptr, _ge) = epoch.device_ptr(&ctx.stream);
    let (ack_ptr, _ga) = down_ack.device_ptr(&ctx.stream);
    let (err_ptr, _gc) = err_code.device_ptr_mut(&ctx.stream);
    let (deltas_ptr, _gd) = match deltas {
        Some(d) => {
            let (p, g) = d.device_ptr_mut(&ctx.stream);
            (p as *mut u64, Some(g))
        }
        None => (std::ptr::null_mut::<u64>(), None),
    };
    unsafe {
        ffi::glm52_pp_send_hidden(
            src_ptr as *const core::ffi::c_void,
            peer_hidden_va as *mut core::ffi::c_void,
            peer_flag_va as *mut u64,
            epoch_ptr as *const u64,
            ack_ptr as *const u64,
            deltas_ptr,
            err_ptr as *mut u32,
            params.words,
            params.ring,
            params.warmup,
            params.n_samples,
            params.deadline_ns,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 PP send_hidden launch failed: {err}"))
}

/// Models a stage's per-token compute latency (one warp spinning globaltimer).
pub fn glm52_pp_dummy_burn_launch(ctx: &DeviceContext, burn_ns: u64) -> Result<()> {
    unsafe { ffi::glm52_pp_dummy_burn(burn_ns, ctx.stream.cu_stream()) }
        .result()
        .map_err(|err| anyhow!("GLM5.2 PP dummy_burn launch failed: {err}"))
}
