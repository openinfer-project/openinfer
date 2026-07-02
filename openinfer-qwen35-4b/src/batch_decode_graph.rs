//! CUDA Graph state for Qwen3.5 batched decode with bucket padding.
//!
//! Allocates MAX_BATCH=64 recurrent-state "slots" with stable GPU addresses.
//! Callers pack active requests into positions 0..batch_size; the graph
//! always replays over 0..bucket_size (padded), so GPU pointers never change.

use anyhow::Result;

use openinfer_core::cuda_graph::CudaGraphState;
use openinfer_core::kv_pool::KvPool;
use openinfer_core::tensor::DeviceContext;

use super::config::Config35;
use super::decode_buffers::BatchDecodeBuffers35;
use super::recurrent_state::RecurrentState;

/// Bucket sizes for CUDA Graph capture. Actual batch is padded to nearest bucket.
pub(crate) const BATCH_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

/// Maximum supported batch size (= largest bucket).
pub const MAX_BATCH: usize = 64;

/// Find the smallest bucket >= `bs`. Panics if `bs` > MAX_BATCH.
pub(crate) fn bucket_for(bs: usize) -> usize {
    for &b in BATCH_BUCKETS {
        if b >= bs {
            return b;
        }
    }
    panic!(
        "batch size {bs} exceeds largest bucket {}",
        BATCH_BUCKETS.last().unwrap()
    );
}

/// CUDA Graph state for Qwen3.5 batch decode.
///
/// Owns MAX_BATCH pre-allocated `RecurrentState` slots and shared decode
/// buffers. Slot `i` always maps to position `i` in the batch — when a request
/// occupies slot `i`, its recurrent state lives at `slot_states[i]` for the
/// entire lifetime of that request in the batch. The CUDA Graph captured for
/// a given bucket size always accesses `slot_states[0..bucket_size]`, so GPU
/// pointer addresses are identical on every replay.
///
/// # Slot management
///
/// Callers must ensure that positions 0..batch_size are active requests and
/// positions batch_size..padded_batch_size are padding. When a request
/// finishes mid-batch, move the last slot's data to fill the gap:
/// ```text
/// copy_state_to_slot(ctx, last_slot_src, vacated_slot_idx)
/// ```
/// and update the caller's slot-to-request mapping accordingly.
pub(crate) struct BatchDecodeGraphState {
    /// Shared decode buffers sized to MAX_BATCH.
    pub(crate) buffers: BatchDecodeBuffers35,
    /// Stable-address per-slot recurrent state; slot_states[i] is always at
    /// the same GPU address regardless of which request occupies slot i.
    pub(crate) slot_states: Vec<RecurrentState>,
    /// One `CudaGraphState` per BATCH_BUCKETS entry (indexed by position).
    pub(crate) graphs: Vec<CudaGraphState>,
    /// One capture-enabled decode graph per bucket. DFlash hidden capture adds
    /// copy kernels to the decode body, so it must not reuse the plain graph.
    pub(crate) capture_graphs: Vec<CudaGraphState>,
}

impl BatchDecodeGraphState {
    /// Create a graph state with a custom maximum batch size.
    ///
    /// `max_batch` is clamped to `MAX_BATCH` (64). Use this when GPU memory is
    /// limited and fewer concurrent decode slots are acceptable.
    pub(crate) fn with_capacity(
        ctx: &DeviceContext,
        config: &Config35,
        kv_pool: &KvPool,
        max_batch: usize,
    ) -> Result<Self> {
        let cap = max_batch.min(MAX_BATCH);
        let padding_page_id = kv_pool.padding_page_id();
        let max_total_pages = kv_pool.capacity_pages();

        let buffers =
            BatchDecodeBuffers35::new(ctx, config, cap, max_total_pages, padding_page_id)?;

        let mut slot_states = Vec::with_capacity(cap);
        for _ in 0..cap {
            slot_states.push(RecurrentState::new(ctx, config)?);
        }

        let graphs = BATCH_BUCKETS
            .iter()
            .map(|_| CudaGraphState::new())
            .collect();
        let capture_graphs = BATCH_BUCKETS
            .iter()
            .map(|_| CudaGraphState::new())
            .collect();

        Ok(Self {
            buffers,
            slot_states,
            graphs,
            capture_graphs,
        })
    }

    /// D2D copy `src` recurrent state into slot `slot_idx`.
    ///
    /// Call once when a request joins the batch (after prefill finishes).
    /// After this call, `slot_states[slot_idx]` IS the canonical state; the
    /// original `src` is no longer used.
    pub(crate) fn copy_state_to_slot(
        &mut self,
        ctx: &DeviceContext,
        src: &RecurrentState,
        slot_idx: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            slot_idx < self.slot_states.len(),
            "Qwen3.5 graph slot {slot_idx} exceeds capacity {}",
            self.slot_states.len()
        );
        let dst = &mut self.slot_states[slot_idx];
        dst.copy_from(ctx, src)
            .map_err(|e| anyhow::anyhow!("copy recurrent state to slot {slot_idx}: {e}"))?;
        Ok(())
    }

    /// D2D copy slot `slot_idx` recurrent state into a standalone state.
    pub(crate) fn copy_slot_to_state(
        &self,
        ctx: &DeviceContext,
        slot_idx: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        anyhow::ensure!(
            slot_idx < self.slot_states.len(),
            "Qwen3.5 graph slot {slot_idx} exceeds capacity {}",
            self.slot_states.len()
        );
        dst.copy_from(ctx, &self.slot_states[slot_idx])
            .map_err(|e| anyhow::anyhow!("copy recurrent slot {slot_idx} to state: {e}"))?;
        Ok(())
    }

    /// D2D copy one graph slot's recurrent/conv state into another slot.
    pub(crate) fn copy_slot_to_slot(
        &mut self,
        ctx: &DeviceContext,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            src_slot_idx < self.slot_states.len(),
            "Qwen3.5 recurrent source slot {src_slot_idx} out of range {}",
            self.slot_states.len()
        );
        anyhow::ensure!(
            dst_slot_idx < self.slot_states.len(),
            "Qwen3.5 recurrent destination slot {dst_slot_idx} out of range {}",
            self.slot_states.len()
        );
        if src_slot_idx == dst_slot_idx {
            return Ok(());
        }
        if src_slot_idx < dst_slot_idx {
            let (left, right) = self.slot_states.split_at_mut(dst_slot_idx);
            let src = &left[src_slot_idx];
            let dst = &mut right[0];
            dst.copy_from(ctx, src)
        } else {
            let (left, right) = self.slot_states.split_at_mut(src_slot_idx);
            let dst = &mut left[dst_slot_idx];
            let src = &right[0];
            dst.copy_from(ctx, src)
        }
        .map_err(|e| {
            anyhow::anyhow!("copy Qwen3.5 recurrent slot {src_slot_idx} to {dst_slot_idx}: {e}")
        })
    }
}
