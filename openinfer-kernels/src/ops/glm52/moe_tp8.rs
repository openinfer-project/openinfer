//! GLM5.2 bucket-1 TP8 MoE: one cooperative kernel per layer (gate/up + SiLU +
//! down + LL reduce-scatter) over 1/8-intermediate slices of all 257 experts
//! (shared expert folded in at bank index 256). Routing stays on the production
//! router — the kernel consumes its (idx, prob) output, so expert selection is
//! byte-identical to the EP8 path. See `csrc/glm52/glm52_moe_tp8.cu` and
//! `docs/models/glm52/moe-tp8-low-latency.md`.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_TP8_RANKS: usize = 8;
pub const GLM52_TP8_HIDDEN: usize = 6144;
pub const GLM52_TP8_TOPK: usize = 8;
/// Routed experts + the shared expert at bank index 256.
pub const GLM52_TP8_BANK_EXPERTS: usize = 257;
/// gate|up rows per expert per rank (2 x 2048/8).
pub const GLM52_TP8_SLICE_ROWS: usize = 512;
/// Intermediate slice per rank (2048/8).
pub const GLM52_TP8_SLICE_I: usize = 256;
/// Worst-case active-expert union: 8 tokens x (topk + shared).
pub const GLM52_TP8_UNION_MAX: usize = GLM52_TP8_RANKS * (GLM52_TP8_TOPK + 1);
/// gate/up k-split factor (kernel-side kKsplitB).
pub const GLM52_TP8_KSPLIT_B: usize = 16;
/// LL allgather packets per rank: H/4 data + 2 idx + 4 prob (16 B each).
pub const GLM52_TP8_AG_PACKETS: usize = GLM52_TP8_HIDDEN / 4 + 6;

/// LL allgather buffer length in u128 packets (parity-double-buffered).
pub const GLM52_TP8_AG_BUF_PACKETS: usize = 2 * GLM52_TP8_RANKS * GLM52_TP8_AG_PACKETS;
/// LL reduce-scatter buffer length in u128 packets (parity-double-buffered).
pub const GLM52_TP8_RS_BUF_PACKETS: usize = 2 * GLM52_TP8_RANKS * GLM52_TP8_HIDDEN;

/// f32 scratch length for gate|up k-split partials.
pub const GLM52_TP8_BPART_LEN: usize =
    GLM52_TP8_KSPLIT_B * GLM52_TP8_UNION_MAX * GLM52_TP8_RANKS * GLM52_TP8_SLICE_ROWS;
/// f32 scratch length for per-expert down partials.
pub const GLM52_TP8_CPART_LEN: usize = GLM52_TP8_UNION_MAX * GLM52_TP8_RANKS * GLM52_TP8_HIDDEN;
/// bf16 scratch length for the SiLU-combined intermediate.
pub const GLM52_TP8_UG_LEN: usize = GLM52_TP8_UNION_MAX * GLM52_TP8_RANKS * GLM52_TP8_SLICE_I;

/// A zeroed, peer-accessible LL packet buffer on the CURRENT device, from a
/// dedicated per-device `cudaMemPool` whose access is granted to the peer
/// devices (`cudaMemPoolSetAccess`). Deliberately NOT
/// `cudaDeviceEnablePeerAccess`: that is device-wide — it maps the whole
/// 105 GiB expert slab into every peer's address space and the page-table
/// pressure taxes the memory-bound expert GEMMs on all layers (~0.8 ms/step
/// flat on 8xH200). Freed on drop; the address is stable for the buffer's
/// lifetime — safe to embed in captured graphs and to hand to peer ranks.
pub struct Glm52Tp8LlBuffer {
    ptr: u64,
    #[allow(dead_code)]
    bytes: usize,
}

// The buffer is device memory touched only by kernels; the owning rank thread
// coordinates lifetime with peers (destroy barrier before teardown).
unsafe impl Send for Glm52Tp8LlBuffer {}
unsafe impl Sync for Glm52Tp8LlBuffer {}

impl Glm52Tp8LlBuffer {
    /// `device_ordinals` is the full DP fleet (own ordinal included); peer
    /// access is granted to every other member on first allocation.
    pub fn alloc(bytes: usize, device_ordinals: &[usize]) -> Result<Self> {
        ensure!(bytes > 0, "TP8 LL buffer needs positive size");
        ensure!(
            !device_ordinals.is_empty() && device_ordinals.len() <= 64,
            "TP8 LL buffer needs 1..=64 fleet device ordinals"
        );
        let ordinals: Vec<i32> = device_ordinals.iter().map(|&d| d as i32).collect();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            ffi::glm52_moe_tp8_alloc_ll_cuda(
                bytes,
                ordinals.as_ptr(),
                ordinals.len() as i32,
                &mut ptr,
            )
        }
        .result()
        .map_err(|err| anyhow!("TP8 LL buffer alloc ({bytes} B) failed: {err}"))?;
        Ok(Self {
            ptr: ptr as u64,
            bytes,
        })
    }

    pub fn addr(&self) -> u64 {
        self.ptr
    }
}

impl Drop for Glm52Tp8LlBuffer {
    fn drop(&mut self) {
        let _ = unsafe { ffi::glm52_moe_tp8_free_ll_cuda(self.ptr as *mut std::ffi::c_void) };
    }
}

/// Co-resident grid size for the cooperative launch on the current device.
pub fn glm52_moe_tp8_max_blocks() -> Result<usize> {
    let mut blocks: i32 = 0;
    unsafe { ffi::glm52_moe_tp8_max_blocks_cuda(&mut blocks) }
        .result()
        .map_err(|err| anyhow!("glm52_moe_tp8_max_blocks_cuda failed: {err}"))?;
    ensure!(blocks > 0, "TP8 MoE kernel has zero co-resident blocks");
    Ok(blocks as usize)
}

/// Per-rank state for one TP8 MoE layer launch. All buffers must be
/// pointer-stable across CUDA-graph capture/replay (resident arena, never
/// reallocated); `peer_ag`/`peer_rs` are the 8 ranks' LL buffer device
/// addresses, each pre-offset to THIS rank's slot.
pub struct Glm52MoeTp8Buffers<'a> {
    pub xg: &'a mut CudaSlice<bf16>,
    pub topk_all_idx: &'a mut CudaSlice<i32>,
    pub topk_all_prob: &'a mut CudaSlice<f32>,
    pub guidx: &'a mut CudaSlice<i32>,
    pub guprob: &'a mut CudaSlice<f32>,
    pub gucnt: &'a mut CudaSlice<i32>,
    pub gused: &'a mut CudaSlice<i32>,
    pub bpart: &'a mut CudaSlice<f32>,
    pub ug: &'a mut CudaSlice<bf16>,
    pub cpart: &'a mut CudaSlice<f32>,
    /// Own LL buffers (`Glm52Tp8LlBuffer` addresses; AG sized
    /// `GLM52_TP8_AG_BUF_PACKETS`, RS `GLM52_TP8_RS_BUF_PACKETS` x 16 B).
    pub ag_local: u64,
    pub rs_local: u64,
    pub peer_ag: [u64; GLM52_TP8_RANKS],
    pub peer_rs: [u64; GLM52_TP8_RANKS],
    pub epoch_dev: &'a mut CudaSlice<u64>,
}

/// Launch the whole-layer TP8 MoE cooperative kernel for this rank's token.
/// `topk_idx`/`topk_prob` are the production router's output for the local
/// token; `mlp_out` receives routed + shared (no residual), like the EP8
/// arm's combined-plus-shared sum.
#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_tp8_layer_launch(
    ctx: &DeviceContext,
    normed2: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    topk_prob: &CudaSlice<f32>,
    w13: &CudaSlice<u8>,
    w13_scale: &CudaSlice<f32>,
    w2: &CudaSlice<u8>,
    w2_scale: &CudaSlice<f32>,
    mlp_out: &mut CudaSlice<bf16>,
    bufs: &mut Glm52MoeTp8Buffers<'_>,
    myrank: usize,
    grid_blocks: usize,
) -> Result<()> {
    const H: usize = GLM52_TP8_HIDDEN;
    const E: usize = GLM52_TP8_BANK_EXPERTS;
    ensure!(myrank < GLM52_TP8_RANKS, "TP8 myrank {myrank} out of range");
    ensure!(
        normed2.len() >= H && mlp_out.len() >= H,
        "TP8 hidden buffers too small: normed2 {}, mlp_out {}",
        normed2.len(),
        mlp_out.len()
    );
    ensure!(
        topk_idx.len() >= GLM52_TP8_TOPK && topk_prob.len() >= GLM52_TP8_TOPK,
        "TP8 topk buffers too small: idx {}, prob {}",
        topk_idx.len(),
        topk_prob.len()
    );
    ensure!(
        w13.len() == E * GLM52_TP8_SLICE_ROWS * H
            && w13_scale.len() == E * (GLM52_TP8_SLICE_ROWS / 128) * (H / 128)
            && w2.len() == E * H * GLM52_TP8_SLICE_I
            && w2_scale.len() == E * (H / 128) * (GLM52_TP8_SLICE_I / 128),
        "TP8 weight slice shape mismatch: w13 {} scale {} w2 {} scale {}",
        w13.len(),
        w13_scale.len(),
        w2.len(),
        w2_scale.len()
    );
    ensure!(
        bufs.xg.len() >= GLM52_TP8_RANKS * H
            && bufs.topk_all_idx.len() >= GLM52_TP8_RANKS * GLM52_TP8_TOPK
            && bufs.topk_all_prob.len() >= GLM52_TP8_RANKS * GLM52_TP8_TOPK
            && bufs.guidx.len() >= GLM52_TP8_UNION_MAX
            && bufs.guprob.len() >= GLM52_TP8_UNION_MAX * GLM52_TP8_RANKS
            && !bufs.gucnt.is_empty()
            && bufs.gused.len() >= 256
            && bufs.bpart.len() >= GLM52_TP8_BPART_LEN
            && bufs.ug.len() >= GLM52_TP8_UG_LEN
            && bufs.cpart.len() >= GLM52_TP8_CPART_LEN
            && !bufs.epoch_dev.is_empty(),
        "TP8 scratch arena too small"
    );
    ensure!(grid_blocks > 0, "TP8 grid_blocks must be positive");
    ensure!(
        bufs.ag_local != 0
            && bufs.rs_local != 0
            && bufs.peer_ag.iter().all(|&p| p != 0)
            && bufs.peer_rs.iter().all(|&p| p != 0),
        "TP8 LL pointers not wired"
    );

    let (normed2_ptr, _g0) = normed2.device_ptr(&ctx.stream);
    let (idx_ptr, _g1) = topk_idx.device_ptr(&ctx.stream);
    let (prob_ptr, _g2) = topk_prob.device_ptr(&ctx.stream);
    let (w13_ptr, _g3) = w13.device_ptr(&ctx.stream);
    let (w13s_ptr, _g4) = w13_scale.device_ptr(&ctx.stream);
    let (w2_ptr, _g5) = w2.device_ptr(&ctx.stream);
    let (w2s_ptr, _g6) = w2_scale.device_ptr(&ctx.stream);
    let (out_ptr, _g7) = mlp_out.device_ptr_mut(&ctx.stream);
    let (xg_ptr, _g8) = bufs.xg.device_ptr_mut(&ctx.stream);
    let (tai_ptr, _g9) = bufs.topk_all_idx.device_ptr_mut(&ctx.stream);
    let (tap_ptr, _g10) = bufs.topk_all_prob.device_ptr_mut(&ctx.stream);
    let (guidx_ptr, _g11) = bufs.guidx.device_ptr_mut(&ctx.stream);
    let (guprob_ptr, _g12) = bufs.guprob.device_ptr_mut(&ctx.stream);
    let (gucnt_ptr, _g13) = bufs.gucnt.device_ptr_mut(&ctx.stream);
    let (gused_ptr, _g14) = bufs.gused.device_ptr_mut(&ctx.stream);
    let (bpart_ptr, _g15) = bufs.bpart.device_ptr_mut(&ctx.stream);
    let (ug_ptr, _g16) = bufs.ug.device_ptr_mut(&ctx.stream);
    let (cpart_ptr, _g17) = bufs.cpart.device_ptr_mut(&ctx.stream);
    let (epoch_ptr, _g20) = bufs.epoch_dev.device_ptr_mut(&ctx.stream);
    let peer_ag: [*const std::ffi::c_void; GLM52_TP8_RANKS] =
        bufs.peer_ag.map(|p| p as *const std::ffi::c_void);
    let peer_rs: [*const std::ffi::c_void; GLM52_TP8_RANKS] =
        bufs.peer_rs.map(|p| p as *const std::ffi::c_void);
    unsafe {
        ffi::glm52_moe_tp8_layer_launch_cuda(
            normed2_ptr as *const ffi::Half,
            idx_ptr as *const i32,
            prob_ptr as *const f32,
            w13_ptr as *const u8,
            w13s_ptr as *const f32,
            w2_ptr as *const u8,
            w2s_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            xg_ptr as *mut ffi::Half,
            tai_ptr as *mut i32,
            tap_ptr as *mut f32,
            guidx_ptr as *mut i32,
            guprob_ptr as *mut f32,
            gucnt_ptr as *mut i32,
            gused_ptr as *mut i32,
            bpart_ptr as *mut f32,
            ug_ptr as *mut ffi::Half,
            cpart_ptr as *mut f32,
            bufs.ag_local as *mut std::ffi::c_void,
            bufs.rs_local as *mut std::ffi::c_void,
            peer_ag.as_ptr(),
            peer_rs.as_ptr(),
            epoch_ptr as *mut u64,
            GLM52_TP8_RANKS as i32,
            myrank as i32,
            grid_blocks as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 TP8 MoE layer launch failed: {err}"))
}
