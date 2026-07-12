use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::dflash::config::DFlashConfig;
use openinfer_core::tensor::{DeviceContext, HiddenStates};

/// Lane-level batched draft scratch, allocated once for the whole decode batch.
///
/// Dense buffers (`hidden`, `normed`, `q_batch`, `attn_output`, the MLP buffers,
/// and `logits`) hold `max_batch * block_size` rows so the GEMM / rms_norm /
/// silu / add / logits / embedding ops run once over the batched buffer. The
/// varlen tail buffers (`tail_input`, `k_tail`, `v_tail`) stay sized for a single
/// request and are reused inside the per-request loop.
pub(crate) struct DFlashBatchScratch {
    max_batch_block_rows: usize,
    max_tail_len: usize,
    pub(super) block_token_ids_h: Vec<u32>,
    pub(super) token_ids_d: CudaSlice<u32>,
    pub(super) hidden: HiddenStates,
    pub(super) hidden_out: HiddenStates,
    pub(super) normed: HiddenStates,
    pub(super) q_batch: HiddenStates,
    pub(super) attn_output: HiddenStates,
    pub(super) o_buf: HiddenStates,
    pub(super) gate_out: HiddenStates,
    pub(super) up_out: HiddenStates,
    pub(super) act_out: HiddenStates,
    pub(super) logits_normed: HiddenStates,
    pub(super) logits: HiddenStates,
    // Shared single-request varlen tail scratch (reused inside the per-request loop).
    pub(super) tail_input: HiddenStates,
    pub(super) k_tail: HiddenStates,
    pub(super) v_tail: HiddenStates,
}

impl DFlashBatchScratch {
    pub(crate) fn estimate_bytes(
        config: &DFlashConfig,
        max_decode_batch_size: usize,
        max_tail_len: usize,
    ) -> usize {
        let batch_rows = config.block_size.saturating_mul(max_decode_batch_size);
        let q_dim = config.num_attention_heads.saturating_mul(config.head_dim);
        let kv_dim = config.num_key_value_heads.saturating_mul(config.head_dim);
        let dense_bf16_per_row = config
            .vocab_size
            .saturating_add(config.hidden_size.saturating_mul(5))
            .saturating_add(q_dim.saturating_mul(2))
            .saturating_add(config.intermediate_size.saturating_mul(3));
        batch_rows
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(
                batch_rows
                    .saturating_mul(dense_bf16_per_row)
                    .saturating_mul(std::mem::size_of::<half::bf16>()),
            )
            .saturating_add(
                max_tail_len
                    .saturating_mul(config.hidden_size.saturating_add(kv_dim.saturating_mul(2)))
                    .saturating_mul(std::mem::size_of::<half::bf16>()),
            )
    }

    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DFlashConfig,
        max_decode_batch_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            max_decode_batch_size > 0,
            "DFlash batch scratch needs a non-zero batch size"
        );
        let block_size = config.block_size;
        let hidden_size = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter_dim = config.intermediate_size;
        // Dense buffers span the whole decode batch so the dense ops run once.
        let batch_rows = block_size * max_decode_batch_size;
        // The shared varlen tail starts at one block (no committed context yet)
        // and grows on demand via `ensure_tail_capacity`.
        let tail_capacity = block_size;
        Ok(Self {
            max_batch_block_rows: batch_rows,
            max_tail_len: tail_capacity,
            block_token_ids_h: vec![config.mask_token_id; batch_rows],
            token_ids_d: ctx.stream.alloc_zeros(batch_rows)?,
            hidden: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            hidden_out: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            normed: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, batch_rows)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, batch_rows)?,
            o_buf: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, batch_rows)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, batch_rows)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, batch_rows)?,
            logits_normed: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            logits: HiddenStates::zeros(ctx, config.vocab_size, batch_rows)?,
            tail_input: HiddenStates::zeros(ctx, hidden_size, tail_capacity)?,
            k_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
            v_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
        })
    }

    /// Point every dense buffer at the active `batch_block_rows = active_batch *
    /// block_size` prefix. Allocated for the max decode batch, so this only
    /// shrinks `seq_len`; it never reallocates.
    pub(super) fn activate_dense(&mut self, batch_block_rows: usize) {
        assert!(
            batch_block_rows <= self.max_batch_block_rows,
            "DFlash batched draft {} block rows exceeds scratch capacity {}",
            batch_block_rows,
            self.max_batch_block_rows
        );
        self.hidden.seq_len = batch_block_rows;
        self.hidden_out.seq_len = batch_block_rows;
        self.normed.seq_len = batch_block_rows;
        self.q_batch.seq_len = batch_block_rows;
        self.attn_output.seq_len = batch_block_rows;
        self.o_buf.seq_len = batch_block_rows;
        self.gate_out.seq_len = batch_block_rows;
        self.up_out.seq_len = batch_block_rows;
        self.act_out.seq_len = batch_block_rows;
        self.logits_normed.seq_len = batch_block_rows;
        self.logits.seq_len = batch_block_rows;
    }

    /// Size the shared varlen tail buffers for one request's `tail_len =
    /// context_len + block_size`, growing the allocation if needed.
    pub(super) fn ensure_tail_capacity(
        &mut self,
        ctx: &DeviceContext,
        config: &DFlashConfig,
        tail_len: usize,
    ) -> Result<()> {
        if tail_len > self.max_tail_len {
            let hidden_size = config.hidden_size;
            let kv_dim = config.num_key_value_heads * config.head_dim;
            self.tail_input = HiddenStates::zeros(ctx, hidden_size, tail_len)?;
            self.k_tail = HiddenStates::zeros(ctx, kv_dim, tail_len)?;
            self.v_tail = HiddenStates::zeros(ctx, kv_dim, tail_len)?;
            self.max_tail_len = tail_len;
        }
        self.tail_input.seq_len = tail_len;
        self.k_tail.seq_len = tail_len;
        self.v_tail.seq_len = tail_len;
        Ok(())
    }
}
