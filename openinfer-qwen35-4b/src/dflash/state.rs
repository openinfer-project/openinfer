use anyhow::{Context, Result};

use openinfer_core::ops;
use openinfer_core::tensor::{DeviceContext, HiddenStates};

pub(crate) struct DFlashRequestState {
    pub(super) layers: Vec<DFlashLayerCache>,
    pub(super) pending_context: DFlashPendingContext,
    /// Projected target context for the current draft round. Computed once from
    /// `pending_context` and read by every layer's tail concat, so it lives with
    /// the request (the batched scratch only holds one request's varlen tail).
    pub(super) context: DFlashContextScratch,
    pub(super) committed_len: usize,
    pub(super) max_cache_len: usize,
}

pub(super) struct DFlashLayerCache {
    pub(super) k: HiddenStates,
    pub(super) v: HiddenStates,
}

pub(super) struct DFlashPendingContext {
    pub(super) buffer: HiddenStates,
    pub(super) len: usize,
    capacity: usize,
}

/// Per-request projected context. The fc projection + hidden_norm turn the
/// captured target hidden context into draft hidden space once per draft round;
/// every layer's tail concat reads `context_hidden`, so it must persist across
/// the layer loop and therefore lives in the request (not the shared scratch).
pub(super) struct DFlashContextScratch {
    max_context_len: usize,
    pub(super) context_projected: HiddenStates,
    pub(super) context_hidden: HiddenStates,
}

impl DFlashRequestState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        kv_dim: usize,
        context_feature_dim: usize,
        hidden_size: usize,
        block_size: usize,
        max_cache_len: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DFlashLayerCache {
                k: HiddenStates::zeros(ctx, kv_dim, max_cache_len)?,
                v: HiddenStates::zeros(ctx, kv_dim, max_cache_len)?,
            });
        }
        Ok(Self {
            layers,
            pending_context: DFlashPendingContext::new(
                ctx,
                context_feature_dim,
                block_size.min(max_cache_len),
            )?,
            context: DFlashContextScratch::new(ctx, hidden_size, block_size)?,
            committed_len: 0,
            max_cache_len,
        })
    }

    pub(crate) fn pending_context_len(&self) -> Option<usize> {
        (self.pending_context.len > 0).then_some(self.pending_context.len)
    }
}

impl DFlashPendingContext {
    fn new(ctx: &DeviceContext, hidden_dim: usize, capacity: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity > 0,
            "DFlash pending context capacity must be non-zero"
        );
        let mut buffer = HiddenStates::zeros(ctx, hidden_dim, capacity)?;
        buffer.seq_len = 0;
        Ok(Self {
            buffer,
            len: 0,
            capacity,
        })
    }

    pub(super) fn append_from(
        &mut self,
        ctx: &DeviceContext,
        src: &HiddenStates,
        src_token_offset: usize,
        token_count: usize,
        max_capacity: usize,
    ) -> Result<()> {
        let required_len = self
            .len
            .checked_add(token_count)
            .context("DFlash pending context length overflow")?;
        anyhow::ensure!(
            required_len <= max_capacity,
            "DFlash pending context length {} exceeds request capacity {}",
            required_len,
            max_capacity
        );
        self.ensure_capacity(ctx, required_len, max_capacity)?;
        self.buffer.seq_len = self.capacity;
        ops::copy_hidden_token_range_into(
            ctx,
            src,
            src_token_offset,
            &mut self.buffer,
            self.len,
            token_count,
        )?;
        self.len = required_len;
        self.buffer.seq_len = self.len;
        Ok(())
    }

    fn ensure_capacity(
        &mut self,
        ctx: &DeviceContext,
        required_len: usize,
        max_capacity: usize,
    ) -> Result<()> {
        if required_len <= self.capacity {
            return Ok(());
        }
        let doubled = self
            .capacity
            .checked_mul(2)
            .context("DFlash pending context capacity overflow")?;
        let new_capacity = required_len.max(doubled).min(max_capacity);
        anyhow::ensure!(
            new_capacity >= required_len,
            "DFlash pending context capacity {} cannot fit {} tokens",
            new_capacity,
            required_len
        );
        let mut next = HiddenStates::zeros(ctx, self.buffer.hidden_dim, new_capacity)?;
        if self.len > 0 {
            self.buffer.seq_len = self.capacity;
            ops::copy_hidden_token_range_into(ctx, &self.buffer, 0, &mut next, 0, self.len)?;
        }
        next.seq_len = self.len;
        self.buffer = next;
        self.capacity = new_capacity;
        Ok(())
    }

    pub(super) fn activate_for_read(&mut self) {
        self.buffer.seq_len = self.len;
    }

    pub(super) fn clear(&mut self) {
        self.len = 0;
        self.buffer.seq_len = 0;
    }
}

impl DFlashContextScratch {
    fn new(ctx: &DeviceContext, hidden_size: usize, max_context_len: usize) -> Result<Self> {
        Ok(Self {
            max_context_len,
            context_projected: HiddenStates::zeros(ctx, hidden_size, max_context_len)?,
            context_hidden: HiddenStates::zeros(ctx, hidden_size, max_context_len)?,
        })
    }

    pub(super) fn ensure_capacity(
        &mut self,
        ctx: &DeviceContext,
        hidden_size: usize,
        context_len: usize,
    ) -> Result<()> {
        if context_len > self.max_context_len {
            *self = Self::new(ctx, hidden_size, context_len)?;
        }
        self.context_projected.seq_len = context_len;
        self.context_hidden.seq_len = context_len;
        Ok(())
    }
}
