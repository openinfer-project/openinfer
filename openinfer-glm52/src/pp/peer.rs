//! PP8 spine: per-stage P2P ring buffers, NVLink peer-access enable, and the
//! resolved peer virtual-address table that each stage bakes into its captured
//! graph.
//!
//! This is the first peer-access user in the repo. The driver-API protocol
//! (`cuDeviceCanAccessPeer` + `cuCtxEnablePeerAccess`, peer pointers remote
//! stored from a kernel) is lifted from `tilert_play/benchmarks/p2p_lsend`.

use std::sync::Arc;

use anyhow::{Result, bail, ensure};
use cudarc::driver::sys::{self, CUresult};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;

/// Per-stage persistent device buffers. Allocated once on the stage's own
/// context and never freed: the captured graph bakes their addresses, and the
/// peer-facing ones (`hidden_in_ring`, `flag_ring`, `ack_ring`) are remote
/// written by neighbours over NVLink, so their VAs must stay stable for the run.
pub(crate) struct Glm52StageBuffers {
    /// Inbound payload ring, peer-written by the upstream send. `[ring * words]`.
    pub(crate) hidden_in_ring: CudaSlice<bf16>,
    /// Inbound epoch flags, peer-written by the upstream send. `[ring]`.
    pub(crate) flag_ring: CudaSlice<u64>,
    /// Local replay counter; wait/source `atomicAdd` it, send reads it. `[1]`.
    pub(crate) epoch_counter: CudaSlice<u64>,
    /// Reverse ack ring, peer-written by the downstream wait. `[ring]`.
    pub(crate) ack_ring: CudaSlice<u64>,
    /// In-kernel fault latch (0=ok; see `glm52_pp_p2p.cu`). `[1]`.
    pub(crate) err_code: CudaSlice<u32>,
    /// Local payload this stage sends downstream. `[words]`.
    pub(crate) src_hidden: CudaSlice<bf16>,
    /// Forward half-RTT samples (globaltimer deltas). `[max(n_samples, 1)]`.
    pub(crate) deltas: CudaSlice<u64>,
}

impl Glm52StageBuffers {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        ring: usize,
        words: usize,
        n_samples: usize,
    ) -> Result<Self> {
        ensure!(ring >= 1, "PP ring depth must be >= 1, got {ring}");
        ensure!(
            words > 0 && words % 8 == 0,
            "PP words must be a positive multiple of 8, got {words}"
        );
        Ok(Self {
            hidden_in_ring: stream.alloc_zeros::<bf16>(ring * words)?,
            flag_ring: stream.alloc_zeros::<u64>(ring)?,
            epoch_counter: stream.alloc_zeros::<u64>(1)?,
            ack_ring: stream.alloc_zeros::<u64>(ring)?,
            err_code: stream.alloc_zeros::<u32>(1)?,
            src_hidden: stream.alloc_zeros::<bf16>(words)?,
            deltas: stream.alloc_zeros::<u64>(n_samples.max(1))?,
        })
    }

    /// Base virtual addresses a neighbour needs to remote-write into this stage.
    pub(crate) fn peer_targets(&self, stream: &Arc<CudaStream>) -> Glm52StageVas {
        let (hidden, _gh) = self.hidden_in_ring.device_ptr(stream);
        let (flag, _gf) = self.flag_ring.device_ptr(stream);
        let (ack, _ga) = self.ack_ring.device_ptr(stream);
        Glm52StageVas {
            hidden_in_ring: hidden,
            flag_ring: flag,
            ack_ring: ack,
        }
    }

    /// Zero the flags/epoch/ack/err so every stage restarts at epoch 0 in
    /// lockstep. Run before each (re)capture and before each measured pass; the
    /// captured graph stays epoch-agnostic because the epoch is device state.
    pub(crate) fn reset_control(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        stream.memset_zeros(&mut self.flag_ring)?;
        stream.memset_zeros(&mut self.epoch_counter)?;
        stream.memset_zeros(&mut self.ack_ring)?;
        stream.memset_zeros(&mut self.err_code)?;
        Ok(())
    }
}

/// Buffer base VAs one stage exports for its neighbours.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Glm52StageVas {
    pub(crate) hidden_in_ring: u64,
    pub(crate) flag_ring: u64,
    pub(crate) ack_ring: u64,
}

/// The peer VAs one stage bakes into its captured graph: where its send writes
/// (downstream `hidden`/`flag`) and where its wait acks (upstream `ack`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Glm52PeerEdge {
    pub(crate) down_hidden: u64,
    pub(crate) down_flag: u64,
    pub(crate) up_ack: u64,
}

/// Enable NVLink P2P from the *current* context into the peer context. The
/// caller must have its own context bound to the calling thread; `peer_ctx` is
/// the neighbour's raw `CUcontext` (a handle, safe to pass across threads).
/// Asserts the link exists and both ends use unified addressing, else bails: a
/// silent fallback to staged copies would wreck the very latency budget this
/// spine exists to prove.
pub(crate) fn enable_peer_access(
    my_ordinal: usize,
    peer_ordinal: usize,
    peer_ctx: sys::CUcontext,
) -> Result<()> {
    let my_dev = cu_device(my_ordinal)?;
    let peer_dev = cu_device(peer_ordinal)?;

    let mut can: i32 = 0;
    cu_ok(
        unsafe { sys::cuDeviceCanAccessPeer(&raw mut can, my_dev, peer_dev) },
        "cuDeviceCanAccessPeer",
    )?;
    ensure!(
        can == 1,
        "GPU {my_ordinal} cannot NVLink-P2P access GPU {peer_ordinal}"
    );
    ensure!(
        unified_addressing(my_dev)? && unified_addressing(peer_dev)?,
        "GPU {my_ordinal}/{peer_ordinal} lack unified addressing required for peer remote stores"
    );

    let r = unsafe { sys::cuCtxEnablePeerAccess(peer_ctx, 0) };
    match r {
        CUresult::CUDA_SUCCESS | CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Ok(()),
        other => bail!("cuCtxEnablePeerAccess GPU {my_ordinal}->{peer_ordinal} failed: {other:?}"),
    }
}

fn cu_device(ordinal: usize) -> Result<sys::CUdevice> {
    let mut dev: sys::CUdevice = 0;
    cu_ok(
        unsafe { sys::cuDeviceGet(&raw mut dev, ordinal as i32) },
        "cuDeviceGet",
    )?;
    Ok(dev)
}

fn unified_addressing(dev: sys::CUdevice) -> Result<bool> {
    let mut v: i32 = 0;
    cu_ok(
        unsafe {
            sys::cuDeviceGetAttribute(
                &raw mut v,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                dev,
            )
        },
        "cuDeviceGetAttribute(UNIFIED_ADDRESSING)",
    )?;
    Ok(v == 1)
}

fn cu_ok(r: CUresult, what: &str) -> Result<()> {
    ensure!(r == CUresult::CUDA_SUCCESS, "{what} failed: {r:?}");
    Ok(())
}
