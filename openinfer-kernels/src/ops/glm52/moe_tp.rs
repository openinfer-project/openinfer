//! GLM5.2 tensor-parallel MoE launch surface shared by TP4 and TP8.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_TP_MAX_RANKS: usize = 8;
pub const GLM52_TP_HIDDEN: usize = 6144;
const GLM52_TP_TOPK: usize = 8;
pub const GLM52_TP_BANK_EXPERTS: usize = 257;
pub const GLM52_TP_TOKENS: usize = 8;
pub const GLM52_TP_UNION_MAX: usize = GLM52_TP_TOKENS * (GLM52_TP_TOPK + 1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52TpTopology {
    Tp4,
    Tp8,
}

impl Glm52TpTopology {
    pub const fn ranks(self) -> usize {
        match self {
            Self::Tp4 => 4,
            Self::Tp8 => 8,
        }
    }

    pub const fn slice_i(self) -> usize {
        2048 / self.ranks()
    }

    pub const fn slice_rows(self) -> usize {
        2 * self.slice_i()
    }

    pub const fn rs_slot_packets(self) -> usize {
        2 * GLM52_TP_TOKENS * self.ranks() * GLM52_TP_HIDDEN
    }

    pub const fn guprob_len(self) -> usize {
        GLM52_TP_UNION_MAX * GLM52_TP_TOKENS
    }

    pub const fn ug_len(self) -> usize {
        GLM52_TP_UNION_MAX * GLM52_TP_TOKENS * self.slice_i()
    }

    pub const fn cpart_len(self) -> usize {
        GLM52_TP_UNION_MAX * GLM52_TP_TOKENS * GLM52_TP_HIDDEN
    }
}

/// Zeroed LL packet memory with one accessor-specific VA per fleet device.
pub struct Glm52TpLlBuffer {
    topology: Glm52TpTopology,
    vas: Vec<u64>,
}

unsafe impl Send for Glm52TpLlBuffer {}
unsafe impl Sync for Glm52TpLlBuffer {}

impl Glm52TpLlBuffer {
    pub fn alloc(
        topology: Glm52TpTopology,
        bytes: usize,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        ensure!(bytes > 0, "TP LL buffer needs positive size");
        ensure!(
            device_ordinals.len() == topology.ranks(),
            "{topology:?} LL buffer needs {} device ordinals, got {}",
            topology.ranks(),
            device_ordinals.len()
        );
        let ordinals: Vec<i32> = device_ordinals.iter().map(|&d| d as i32).collect();
        let mut vas = vec![0u64; ordinals.len()];
        let result = unsafe {
            match topology {
                Glm52TpTopology::Tp4 => ffi::glm52_moe_tp4_alloc_ll_cuda(
                    bytes,
                    ordinals.as_ptr(),
                    ordinals.len() as i32,
                    vas.as_mut_ptr(),
                ),
                Glm52TpTopology::Tp8 => ffi::glm52_moe_tp8_alloc_ll_cuda(
                    bytes,
                    ordinals.as_ptr(),
                    ordinals.len() as i32,
                    vas.as_mut_ptr(),
                ),
            }
        };
        result
            .result()
            .map_err(|err| anyhow!("{topology:?} LL buffer alloc ({bytes} B) failed: {err}"))?;
        Ok(Self { topology, vas })
    }

    pub fn addr_for(&self, idx: usize) -> u64 {
        self.vas[idx]
    }
}

impl Drop for Glm52TpLlBuffer {
    fn drop(&mut self) {
        let ptr = self.vas[0] as *mut std::ffi::c_void;
        let _ = unsafe {
            match self.topology {
                Glm52TpTopology::Tp4 => ffi::glm52_moe_tp4_free_ll_cuda(ptr),
                Glm52TpTopology::Tp8 => ffi::glm52_moe_tp8_free_ll_cuda(ptr),
            }
        };
    }
}

pub fn glm52_moe_tp_epoch_advance(
    ctx: &DeviceContext,
    topology: Glm52TpTopology,
    epoch_dev: &mut CudaSlice<u64>,
) -> Result<()> {
    let (epoch_ptr, _guard) = epoch_dev.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        match topology {
            Glm52TpTopology::Tp4 => ffi::glm52_moe_tp4_epoch_advance_cuda(
                epoch_ptr as *mut std::ffi::c_void,
                ctx.stream.cu_stream(),
            ),
            Glm52TpTopology::Tp8 => ffi::glm52_moe_tp8_epoch_advance_cuda(
                epoch_ptr as *mut std::ffi::c_void,
                ctx.stream.cu_stream(),
            ),
        }
    };
    result
        .result()
        .map_err(|err| anyhow!("{topology:?} epoch advance failed: {err}"))
}

pub fn glm52_moe_tp_max_blocks(topology: Glm52TpTopology) -> Result<usize> {
    let mut blocks = 0i32;
    let result = unsafe {
        match topology {
            Glm52TpTopology::Tp4 => ffi::glm52_moe_tp4_max_blocks_cuda(&raw mut blocks),
            Glm52TpTopology::Tp8 => ffi::glm52_moe_tp8_max_blocks_cuda(&raw mut blocks),
        }
    };
    result
        .result()
        .map_err(|err| anyhow!("{topology:?} grid query failed: {err}"))?;
    ensure!(blocks > 0, "{topology:?} MoE kernel has zero blocks");
    Ok(blocks as usize)
}

pub struct Glm52MoeTpBuffers<'a> {
    pub guidx: &'a mut CudaSlice<i32>,
    pub guprob: &'a mut CudaSlice<f32>,
    pub gucnt: &'a mut CudaSlice<i32>,
    pub gused: &'a mut CudaSlice<i32>,
    pub ug: &'a mut CudaSlice<bf16>,
    pub cpart: &'a mut CudaSlice<f32>,
    pub rs_local: u64,
    pub peer_rs: [u64; GLM52_TP_MAX_RANKS],
    pub epoch_dev: &'a mut CudaSlice<u64>,
    pub active_rows: Option<&'a CudaSlice<i32>>,
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_tp_layer_launch(
    ctx: &DeviceContext,
    topology: Glm52TpTopology,
    layer_slot: usize,
    normed2: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    topk_prob: &CudaSlice<f32>,
    w13: &CudaSlice<u8>,
    w13_scale: &CudaSlice<f32>,
    w2: &CudaSlice<u8>,
    w2_scale: &CudaSlice<f32>,
    mlp_out: &mut CudaSlice<bf16>,
    bufs: &mut Glm52MoeTpBuffers<'_>,
    myrank: usize,
    grid_blocks: usize,
) -> Result<()> {
    let ranks = topology.ranks();
    ensure!(myrank < ranks, "{topology:?} rank {myrank} out of range");
    ensure!(
        normed2.len() >= GLM52_TP_TOKENS * GLM52_TP_HIDDEN
            && mlp_out.len() >= GLM52_TP_TOKENS * GLM52_TP_HIDDEN,
        "{topology:?} hidden buffers too small"
    );
    ensure!(
        topk_idx.len() >= GLM52_TP_TOKENS * GLM52_TP_TOPK
            && topk_prob.len() >= GLM52_TP_TOKENS * GLM52_TP_TOPK,
        "{topology:?} topk buffers too small"
    );
    ensure!(
        w13.len() == GLM52_TP_BANK_EXPERTS * topology.slice_rows() * GLM52_TP_HIDDEN
            && w13_scale.len()
                == GLM52_TP_BANK_EXPERTS * (topology.slice_rows() / 128) * (GLM52_TP_HIDDEN / 128)
            && w2.len() == GLM52_TP_BANK_EXPERTS * GLM52_TP_HIDDEN * topology.slice_i()
            && w2_scale.len()
                == GLM52_TP_BANK_EXPERTS * (GLM52_TP_HIDDEN / 128) * (topology.slice_i() / 128),
        "{topology:?} weight slice shape mismatch"
    );
    ensure!(
        bufs.guidx.len() >= GLM52_TP_UNION_MAX
            && bufs.guprob.len() >= topology.guprob_len()
            && !bufs.gucnt.is_empty()
            && bufs.gused.len() >= 256
            && bufs.ug.len() >= topology.ug_len()
            && bufs.cpart.len() >= topology.cpart_len()
            && !bufs.epoch_dev.is_empty(),
        "{topology:?} scratch arena too small"
    );
    ensure!(grid_blocks > 0, "{topology:?} grid must be positive");
    ensure!(
        bufs.rs_local != 0 && bufs.peer_rs[..ranks].iter().all(|&ptr| ptr != 0),
        "{topology:?} LL pointers not wired"
    );

    let (normed2_ptr, _g0) = normed2.device_ptr(&ctx.stream);
    let (idx_ptr, _g1) = topk_idx.device_ptr(&ctx.stream);
    let (prob_ptr, _g2) = topk_prob.device_ptr(&ctx.stream);
    let (w13_ptr, _g3) = w13.device_ptr(&ctx.stream);
    let (w13s_ptr, _g4) = w13_scale.device_ptr(&ctx.stream);
    let (w2_ptr, _g5) = w2.device_ptr(&ctx.stream);
    let (w2s_ptr, _g6) = w2_scale.device_ptr(&ctx.stream);
    let (out_ptr, _g7) = mlp_out.device_ptr_mut(&ctx.stream);
    let (guidx_ptr, _g8) = bufs.guidx.device_ptr_mut(&ctx.stream);
    let (guprob_ptr, _g9) = bufs.guprob.device_ptr_mut(&ctx.stream);
    let (gucnt_ptr, _g10) = bufs.gucnt.device_ptr_mut(&ctx.stream);
    let (gused_ptr, _g11) = bufs.gused.device_ptr_mut(&ctx.stream);
    let (ug_ptr, _g13) = bufs.ug.device_ptr_mut(&ctx.stream);
    let (cpart_ptr, _g14) = bufs.cpart.device_ptr_mut(&ctx.stream);
    let (epoch_ptr, _g15) = bufs.epoch_dev.device_ptr_mut(&ctx.stream);
    let active_ptr = bufs.active_rows.map_or(std::ptr::null(), |active| {
        active.device_ptr(&ctx.stream).0 as *const i32
    });
    let peer_rs = bufs.peer_rs.map(|ptr| ptr as *const std::ffi::c_void);

    macro_rules! launch {
        ($ffi:ident) => {
            ffi::$ffi(
                normed2_ptr as *const ffi::Half,
                idx_ptr as *const i32,
                prob_ptr as *const f32,
                w13_ptr as *const u8,
                w13s_ptr as *const f32,
                w2_ptr as *const u8,
                w2s_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                guidx_ptr as *mut i32,
                guprob_ptr as *mut f32,
                gucnt_ptr as *mut i32,
                gused_ptr as *mut i32,
                ug_ptr as *mut ffi::Half,
                cpart_ptr as *mut f32,
                bufs.rs_local as *mut std::ffi::c_void,
                peer_rs.as_ptr(),
                epoch_ptr as *mut u64,
                active_ptr,
                layer_slot as i32,
                ranks as i32,
                myrank as i32,
                grid_blocks as i32,
                ctx.stream.cu_stream(),
            )
        };
    }
    let result = unsafe {
        match topology {
            Glm52TpTopology::Tp4 => launch!(glm52_moe_tp4_layer_launch_cuda),
            Glm52TpTopology::Tp8 => launch!(glm52_moe_tp8_layer_launch_cuda),
        }
    };
    result
        .result()
        .map_err(|err| anyhow!("{topology:?} MoE launch failed: {err}"))
}
