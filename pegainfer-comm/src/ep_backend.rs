//! NVLink + RDMA EP all-to-all backend.
//!
//! Thin wrapper around `p2p_all_to_all::AllToAllContext`. The four hot-path
//! methods mirror the upstream four-step pipeline so callers can run
//! host-side compute (e.g. shared-expert GEMMs) in the gap between
//! `dispatch_send` and `dispatch_recv`, or between `combine_send` and
//! `combine_recv`. All raw device pointers / strides are passed through 1:1
//! to the upstream context — no wrapper struct, no abstraction layer.

use std::ffi::c_void;
use std::sync::Arc;

use fabric_lib::{TransferEngine, api::MemoryRegionHandle};
pub use p2p_all_to_all::ScalarType;
use p2p_all_to_all::{AllToAllContext, AllToAllRankHandle};

use crate::error::{Error, Result};

/// EP rank-topology description.
///
/// Mirrors the construction-time fields the upstream context needs to size
/// its internal buffers and resolve peers. All sizes are token / element
/// counts (not bytes).
#[derive(Debug, Clone)]
pub struct EpTopology {
    /// Total number of ranks participating in the all-to-all.
    pub world_size: usize,
    /// Rank of the current process (0-based, `< world_size`).
    pub rank: usize,
    /// Ranks per node (NVLink domain size).
    pub node_size: usize,
    /// Data-parallel group size (1 if pure EP).
    pub dp_size: usize,
    /// Total number of experts across all ranks.
    pub num_experts: usize,
    /// Number of experts each token is routed to (top-k).
    pub num_experts_per_token: usize,
    /// Hidden dimension of the token tensors.
    pub hidden_dim: usize,
    /// FP8 scale stride per token (0 when not running FP8).
    pub hidden_dim_scale: usize,
    /// Maximum number of tokens any one dispatch on this rank can carry.
    pub max_num_tokens: usize,
    /// Maximum total number of tokens this rank can receive in one dispatch
    /// (sum across peers).
    pub max_recv_tokens: usize,
    /// Per-expert cap on local tokens (combine-side staging size).
    pub max_private_tokens: usize,
    /// Alignment padding applied within the combine buffer.
    pub expert_padding: usize,
}

/// Wire-format dtypes.
#[derive(Debug, Clone)]
pub struct EpDtypes {
    /// Element size (bytes) of the dispatched `x` payload (e.g. 2 for BF16).
    pub in_elemsize: usize,
    /// Element size (bytes) of the combine output payload.
    pub out_elemsize: usize,
    /// Combine output dtype (used at construction time; `combine_recv`
    /// receives both an `in_dtype` and the configured `out_dtype`).
    pub out_dtype: ScalarType,
    /// Element size (bytes) of the per-token FP8 scale buffer (0 when not
    /// running FP8).
    pub scale_elemsize: usize,
}

/// Pre-allocated, MR-registered staging buffers + per-rank pointer tables
/// owned by the caller and handed to the backend at construction time.
///
/// # Safety / lifetime contract
///
/// The pointers and memory-region handles must remain valid for the entire
/// lifetime of the resulting [`EpBackend`]. The backend reuses them across
/// every dispatch / combine; the caller MUST NOT reallocate or deregister
/// while the backend is alive.
pub struct EpRankBuffers {
    /// Device pointer to the per-expert routed-token counter (one u32 per
    /// expert).
    pub num_routed_ptr: *mut u32,
    /// MR handle for `num_routed_ptr`.
    pub num_routed_mr: MemoryRegionHandle,
    /// Device pointer to the send staging buffer.
    pub send_buffer_ptr: *mut c_void,
    /// MR handle for `send_buffer_ptr`.
    pub send_buffer_mr: MemoryRegionHandle,
    /// Device pointer to the recv staging buffer.
    pub recv_buffer_ptr: *mut c_void,
    /// MR handle for `recv_buffer_ptr`.
    pub recv_buffer_mr: MemoryRegionHandle,
    /// Per-rank table of remote sync-flag device pointers (one per peer).
    pub sync_ptrs: Vec<u64>,
    /// Per-rank table of remote send-buffer device pointers (one per peer).
    pub send_ptrs: Vec<u64>,
    /// Per-rank table of remote recv-buffer device pointers (one per peer).
    pub recv_ptrs: Vec<u64>,
}

// EpRankBuffers carries raw device pointers; the unsafety is in the
// caller's lifetime contract above. Mark Send + Sync so the owning
// `EpBackend` can be moved across threads — internal serialization is
// handled by the upstream worker.
unsafe impl Send for EpRankBuffers {}
unsafe impl Sync for EpRankBuffers {}

/// Constructor parameters for [`EpBackend`].
pub struct EpBackendParams {
    /// Rank topology / sizing.
    pub topology: EpTopology,
    /// Payload dtypes.
    pub dtypes: EpDtypes,
    /// Caller-owned staging buffers + peer pointer tables.
    pub buffers: EpRankBuffers,
    /// Per-peer rank handles obtained from the upstream rendezvous.
    pub rank_handles: Vec<AllToAllRankHandle>,
    /// Shared fabric transfer engine.
    pub transfer_engine: Arc<TransferEngine>,
    /// CUDA device ordinal. Must match the active CUDA context.
    pub device: u8,
    /// Imm-data base used to disambiguate this all-to-all from others on
    /// the same fabric.
    pub imm_base: u32,
    /// Optional CPU core to pin the backend worker thread on.
    pub worker_cpu: Option<u16>,
}

/// Concrete EP backend wrapping the upstream `AllToAllContext`.
///
/// Construct with [`EpBackend::new`]; drive the four-step pipeline through
/// the inherent methods below. Each method maps 1:1 onto the upstream
/// context method of the same name; the upstream `anyhow::Error` is erased
/// into [`Error::Backend`].
///
/// # Concurrency
///
/// The upstream methods take `&mut self`, so [`EpBackend`] must be driven
/// from a single owning lane (PegaInfer's rank worker thread). The
/// `AllToAllContext`'s internal worker thread handles all cross-rank
/// progress; the caller does not poll.
pub struct EpBackend {
    inner: AllToAllContext,
}

// `AllToAllContext` wraps an `Arc<TransferEngine>` + worker thread handle +
// pre-allocated CUDA memory (all upstream types that are individually
// `Send + Sync`). The struct is moved across threads through PegaInfer's
// `RankCommand::EnablePplx` channel; mark it `Send` explicitly so the
// channel does not require upstream auto-traits we don't control.
unsafe impl Send for EpBackend {}

impl EpBackend {
    /// Construct a new EP backend by handing the parameters off to the
    /// upstream `AllToAllContext::new`.
    ///
    /// The upstream constructor spawns a worker thread and pins it to
    /// `worker_cpu` if provided. Failures from the upstream context are
    /// erased into [`Error::Backend`].
    pub fn new(params: EpBackendParams) -> Result<Self> {
        let EpBackendParams {
            topology,
            dtypes,
            buffers,
            rank_handles,
            transfer_engine,
            device,
            imm_base,
            worker_cpu,
        } = params;

        let inner = AllToAllContext::new(
            topology.hidden_dim,
            topology.hidden_dim_scale,
            dtypes.in_elemsize,
            dtypes.out_elemsize,
            dtypes.out_dtype,
            dtypes.scale_elemsize,
            topology.max_num_tokens,
            topology.max_recv_tokens,
            topology.max_private_tokens,
            topology.num_experts,
            topology.expert_padding,
            topology.num_experts_per_token,
            topology.rank,
            topology.dp_size,
            topology.node_size,
            topology.world_size,
            buffers.num_routed_ptr,
            buffers.num_routed_mr,
            buffers.send_buffer_ptr,
            buffers.send_buffer_mr,
            buffers.recv_buffer_ptr,
            buffers.recv_buffer_mr,
            buffers.sync_ptrs,
            buffers.send_ptrs,
            buffers.recv_ptrs,
            device,
            imm_base,
            rank_handles,
            transfer_engine,
            worker_cpu,
        )
        .map_err(Error::from_anyhow)?;
        Ok(Self { inner })
    }

    /// Step 1 of dispatch: scatter tokens onto the fabric.
    ///
    /// Enqueues the dispatch-send kernel onto `stream` and hands the
    /// post-kernel notifications off to the worker thread. Returns once
    /// the kernel is launched; completion is signaled when a matching
    /// [`Self::dispatch_recv`] reports success.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_send(
        &mut self,
        num_tokens: usize,
        x_ptr: *const c_void,
        x_stride: usize,
        x_scale_ptr: *const c_void,
        x_scale_stride_elem: usize,
        x_scale_stride_token: usize,
        indices: *const i32,
        indices_stride: usize,
        weights: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        stream: u64,
    ) -> Result<()> {
        self.inner
            .dispatch_send(
                num_tokens,
                x_ptr,
                x_stride,
                x_scale_ptr,
                x_scale_stride_elem,
                x_scale_stride_token,
                indices,
                indices_stride,
                weights,
                weights_stride,
                bound_m_ptr,
                stream,
            )
            .map_err(Error::from_anyhow)
    }

    /// Step 2 of dispatch: pull token payloads in from peers.
    ///
    /// Writes `out_x_ptr` / `out_x_scale_ptr` and reports the local
    /// received-token count via `out_num_tokens_ptr`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_recv(
        &mut self,
        out_num_tokens_ptr: *mut i32,
        out_x_ptr: *mut c_void,
        out_x_stride: usize,
        out_x_scale_ptr: *mut c_void,
        out_x_scale_stride_elem: usize,
        out_x_scale_stride_token: usize,
        stream: u64,
    ) -> Result<()> {
        self.inner
            .dispatch_recv(
                out_num_tokens_ptr,
                out_x_ptr,
                out_x_stride,
                out_x_scale_ptr,
                out_x_scale_stride_elem,
                out_x_scale_stride_token,
                stream,
            )
            .map_err(Error::from_anyhow)
    }

    /// Step 3 (combine): scatter expert outputs back onto the fabric.
    pub fn combine_send(
        &mut self,
        expert_x_ptr: *const c_void,
        expert_x_stride: usize,
        stream: u64,
    ) -> Result<()> {
        self.inner
            .combine_send(expert_x_ptr, expert_x_stride, stream)
            .map_err(Error::from_anyhow)
    }

    /// Device pointer to the per-local-expert received-token counter
    /// (sized `num_local_experts` u32). Populated by [`Self::dispatch_recv`];
    /// the caller typically does a D2H copy + exclusive prefix sum to drive
    /// a downstream local grouped-expert GEMM.
    pub fn tokens_per_expert_ptr(&self) -> *const u32 {
        self.inner.tokens_per_expert_ptr()
    }

    /// Step 4 (combine): reduce remote contributions into the per-token
    /// output buffer.
    ///
    /// `in_dtype` is the dtype the experts produced (per call); the
    /// `out_dtype` was fixed at construction time on [`EpDtypes`].
    #[allow(clippy::too_many_arguments)]
    pub fn combine_recv(
        &mut self,
        num_tokens: usize,
        num_recv_tokens: usize,
        in_dtype: ScalarType,
        out_tokens_ptr: *mut c_void,
        out_tokens_stride: usize,
        indices_ptr: *const i32,
        indices_stride: usize,
        weights_ptr: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        accumulate: bool,
        stream: u64,
    ) -> Result<()> {
        self.inner
            .combine_recv(
                num_tokens,
                num_recv_tokens,
                in_dtype,
                out_tokens_ptr,
                out_tokens_stride,
                indices_ptr,
                indices_stride,
                weights_ptr,
                weights_stride,
                bound_m_ptr,
                accumulate,
                stream,
            )
            .map_err(Error::from_anyhow)
    }
}
