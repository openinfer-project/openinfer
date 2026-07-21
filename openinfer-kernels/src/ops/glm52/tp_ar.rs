//! GLM5.2 tensor-parallel attention allreduce shared by TP4 and TP8.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use super::moe_tp::GLM52_TP_HIDDEN;
use super::moe_tp::GLM52_TP_MAX_RANKS;
use super::moe_tp::GLM52_TP_TOKENS;
use super::moe_tp::Glm52TpTopology;
use crate::ffi;
use crate::tensor::DeviceContext;

pub const fn glm52_tp_ar_chunk_packets(topology: Glm52TpTopology) -> usize {
    (GLM52_TP_HIDDEN / topology.ranks()) * 2 / 12
}

pub const fn glm52_tp_ar_buffer_bytes(topology: Glm52TpTopology, layer_slots: usize) -> usize {
    let slot_packets =
        2 * 2 * GLM52_TP_TOKENS * topology.ranks() * glm52_tp_ar_chunk_packets(topology);
    layer_slots * slot_packets * 16
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_tp_ar_launch(
    ctx: &DeviceContext,
    topology: Glm52TpTopology,
    layer_slot: usize,
    rows: usize,
    partial: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    ar_local: u64,
    peer_ar: [u64; GLM52_TP_MAX_RANKS],
    epoch_dev: &CudaSlice<u64>,
    active_rows: Option<&CudaSlice<i32>>,
    myrank: usize,
) -> Result<()> {
    let ranks = topology.ranks();
    ensure!(myrank < ranks, "{topology:?} AR rank {myrank} out of range");
    ensure!(
        (1..=GLM52_TP_TOKENS).contains(&rows),
        "{topology:?} AR rows {rows} out of range"
    );
    ensure!(
        partial.len() >= rows * GLM52_TP_HIDDEN && out.len() >= rows * GLM52_TP_HIDDEN,
        "{topology:?} AR hidden buffers too small"
    );
    ensure!(!epoch_dev.is_empty(), "{topology:?} AR epoch is empty");
    ensure!(
        ar_local != 0 && peer_ar[..ranks].iter().all(|&ptr| ptr != 0),
        "{topology:?} AR pointers not wired"
    );

    let (partial_ptr, _g0) = partial.device_ptr(&ctx.stream);
    let (out_ptr, _g1) = out.device_ptr_mut(&ctx.stream);
    let (epoch_ptr, _g2) = epoch_dev.device_ptr(&ctx.stream);
    let active_ptr = active_rows.map_or(std::ptr::null(), |active| {
        active.device_ptr(&ctx.stream).0 as *const i32
    });
    let peer_ar = peer_ar.map(|ptr| ptr as *const std::ffi::c_void);

    macro_rules! launch {
        ($ffi:ident) => {
            ffi::$ffi(
                partial_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                ar_local as *mut std::ffi::c_void,
                peer_ar.as_ptr(),
                epoch_ptr as *const u64,
                active_ptr,
                layer_slot as i32,
                rows as i32,
                ranks as i32,
                myrank as i32,
                ctx.stream.cu_stream(),
            )
        };
    }
    let result = unsafe {
        match topology {
            Glm52TpTopology::Tp4 => launch!(glm52_tp4_ar_launch_cuda),
            Glm52TpTopology::Tp8 => launch!(glm52_tp8_ar_launch_cuda),
        }
    };
    result
        .result()
        .map_err(|err| anyhow!("{topology:?} attention allreduce failed: {err}"))
}
