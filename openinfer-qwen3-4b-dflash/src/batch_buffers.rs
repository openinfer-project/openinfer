use anyhow::Result;
use cudarc::driver::CudaSlice;
use openinfer_core::ops::RaggedPrefillPlan;
use openinfer_core::tensor::HiddenStates;

use crate::weights::DFlashDraftModel;

pub struct DFlashBatchBuffers {
    pub(crate) max_batch_size: usize,
    pub(crate) max_q_len: usize,
    pub(crate) max_ctx_len: usize,
    /// Active shape for the current batch — set by `set_active_shape` before
    /// each forward. `q_len`/`ctx_len` may shrink below `max_*`; the physical
    /// buffers are sized for the max, so the active values only narrow the view.
    pub(crate) q_len: usize,
    pub(crate) ctx_len: usize,
    pub(crate) total_q_len: usize,
    pub(crate) total_ctx_len: usize,
    pub(crate) total_kv_len: usize,
    pub(crate) noise: HiddenStates,
    pub(crate) target_hidden: HiddenStates,
    pub(crate) target_projected: HiddenStates,
    pub(crate) target_normed: HiddenStates,
    pub(crate) hidden: HiddenStates,
    pub(crate) hidden_out: HiddenStates,
    pub(crate) normed: HiddenStates,
    pub(crate) q: HiddenStates,
    pub(crate) q_ctx_scratch: HiddenStates,
    pub(crate) k_ctx: HiddenStates,
    pub(crate) k_noise: HiddenStates,
    pub(crate) v_ctx: HiddenStates,
    pub(crate) v_noise: HiddenStates,
    pub(crate) k_all: HiddenStates,
    pub(crate) v_all: HiddenStates,
    pub(crate) attn_out: HiddenStates,
    pub(crate) o_buf: HiddenStates,
    pub(crate) gate_up: HiddenStates,
    pub(crate) act_out: HiddenStates,
    pub(crate) positions_q: CudaSlice<i32>,
    pub(crate) positions_ctx: CudaSlice<i32>,
    pub(crate) ragged_plan: Option<CachedRaggedPlan>,
}

pub(crate) struct CachedRaggedPlan {
    pub(crate) batch_size: usize,
    pub(crate) q_len: usize,
    pub(crate) ctx_len: usize,
    pub(crate) plan: RaggedPrefillPlan,
}

impl DFlashBatchBuffers {
    /// Allocate a single-instance buffer sized for the worst case
    /// (`max_batch_size × max_q_len` / `× max_ctx_len`). Each forward narrows
    /// the active shape via `set_active_shape`, mirroring Qwen3's
    /// `BatchDecodeBuffers` (one allocation, dynamic `set_batch_size`).
    pub(crate) fn new(
        model: &DFlashDraftModel,
        max_batch_size: usize,
        max_q_len: usize,
        max_ctx_len: usize,
    ) -> Result<Self> {
        anyhow::ensure!(max_batch_size > 0, "max_batch_size must be positive");
        anyhow::ensure!(max_q_len > 0, "max_q_len must be positive");
        anyhow::ensure!(max_ctx_len > 0, "max_ctx_len must be positive");
        let config = model.config();
        let ctx = model.device_context();
        let hidden = config.hidden_size;
        let target_hidden_dim = config.hidden_size * config.target_layer_count();
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let total_q_len = max_batch_size * max_q_len;
        let total_ctx_len = max_batch_size * max_ctx_len;
        let total_kv_len = max_batch_size * (max_ctx_len + max_q_len);
        Ok(Self {
            max_batch_size,
            max_q_len,
            max_ctx_len,
            q_len: max_q_len,
            ctx_len: max_ctx_len,
            total_q_len,
            total_ctx_len,
            total_kv_len,
            noise: HiddenStates::zeros(ctx, hidden, total_q_len)?,
            target_hidden: HiddenStates::zeros(ctx, target_hidden_dim, total_ctx_len)?,
            target_projected: HiddenStates::zeros(ctx, hidden, total_ctx_len)?,
            target_normed: HiddenStates::zeros(ctx, hidden, total_ctx_len)?,
            hidden: HiddenStates::zeros(ctx, hidden, total_q_len)?,
            hidden_out: HiddenStates::zeros(ctx, hidden, total_q_len)?,
            normed: HiddenStates::zeros(ctx, hidden, total_q_len)?,
            q: HiddenStates::zeros(ctx, q_dim, total_q_len)?,
            q_ctx_scratch: HiddenStates::zeros(ctx, q_dim, total_ctx_len)?,
            k_ctx: HiddenStates::zeros(ctx, kv_dim, total_ctx_len)?,
            k_noise: HiddenStates::zeros(ctx, kv_dim, total_q_len)?,
            v_ctx: HiddenStates::zeros(ctx, kv_dim, total_ctx_len)?,
            v_noise: HiddenStates::zeros(ctx, kv_dim, total_q_len)?,
            k_all: HiddenStates::zeros(ctx, kv_dim, total_kv_len)?,
            v_all: HiddenStates::zeros(ctx, kv_dim, total_kv_len)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, total_q_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden, total_q_len)?,
            gate_up: HiddenStates::zeros(ctx, 2 * config.intermediate_size, total_q_len)?,
            act_out: HiddenStates::zeros(ctx, config.intermediate_size, total_q_len)?,
            positions_q: ctx.stream.alloc_zeros(total_q_len)?,
            positions_ctx: ctx.stream.alloc_zeros(total_ctx_len)?,
            ragged_plan: None,
        })
    }

    /// Narrow the active shape for this forward: sets `q_len`/`ctx_len` and
    /// recomputes every buffer's `seq_len` to `batch_size × (q|ctx)`. Buffers
    /// stay sized for the max, so callers can freely vary batch/q/ctx below it.
    pub(crate) fn set_active_shape(&mut self, batch_size: usize, q_len: usize, ctx_len: usize) {
        debug_assert!(batch_size <= self.max_batch_size);
        debug_assert!(q_len <= self.max_q_len);
        debug_assert!(ctx_len <= self.max_ctx_len);
        self.q_len = q_len;
        self.ctx_len = ctx_len;
        self.total_q_len = batch_size * q_len;
        self.total_ctx_len = batch_size * ctx_len;
        self.total_kv_len = batch_size * (ctx_len + q_len);
        self.noise.seq_len = self.total_q_len;
        self.target_hidden.seq_len = self.total_ctx_len;
        self.target_projected.seq_len = self.total_ctx_len;
        self.target_normed.seq_len = self.total_ctx_len;
        self.hidden.seq_len = self.total_q_len;
        self.hidden_out.seq_len = self.total_q_len;
        self.normed.seq_len = self.total_q_len;
        self.q.seq_len = self.total_q_len;
        self.q_ctx_scratch.seq_len = self.total_ctx_len;
        self.k_ctx.seq_len = self.total_ctx_len;
        self.k_noise.seq_len = self.total_q_len;
        self.v_ctx.seq_len = self.total_ctx_len;
        self.v_noise.seq_len = self.total_q_len;
        self.k_all.seq_len = self.total_kv_len;
        self.v_all.seq_len = self.total_kv_len;
        self.attn_out.seq_len = self.total_q_len;
        self.o_buf.seq_len = self.total_q_len;
        self.gate_up.seq_len = self.total_q_len;
        self.act_out.seq_len = self.total_q_len;
    }

    pub(crate) fn prepare_ragged_plan(
        &mut self,
        model: &DFlashDraftModel,
        batch_size: usize,
    ) -> Result<()> {
        // The plan depends on (batch_size, q_len, ctx_len); with a single
        // instance buffer any of them can change between forwards, so all three
        // must be part of the cache key.
        let needs_rebuild = self
            .ragged_plan
            .as_ref()
            .map(|cached| {
                cached.batch_size != batch_size
                    || cached.q_len != self.q_len
                    || cached.ctx_len != self.ctx_len
            })
            .unwrap_or(true);
        if needs_rebuild {
            let config = model.config();
            let q_lens = vec![self.q_len; batch_size];
            let kv_lens = vec![self.ctx_len + self.q_len; batch_size];
            let plan = RaggedPrefillPlan::new(
                model.device_context(),
                &q_lens,
                &kv_lens,
                config.num_attention_heads / config.num_key_value_heads,
            )?;
            self.ragged_plan = Some(CachedRaggedPlan {
                batch_size,
                q_len: self.q_len,
                ctx_len: self.ctx_len,
                plan,
            });
        }
        Ok(())
    }
}
