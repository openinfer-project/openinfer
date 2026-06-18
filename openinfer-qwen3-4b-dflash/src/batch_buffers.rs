use anyhow::Result;
use cudarc::driver::CudaSlice;
use openinfer_core::ops::RaggedPrefillPlan;
use openinfer_core::tensor::HiddenStates;

use crate::weights::DFlashDraftModel;

pub struct DFlashBatchBuffers {
    pub(crate) max_batch_size: usize,
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
    pub(crate) plan: RaggedPrefillPlan,
}

impl DFlashBatchBuffers {
    pub(crate) fn new(
        model: &DFlashDraftModel,
        max_batch_size: usize,
        q_len: usize,
        ctx_len: usize,
    ) -> Result<Self> {
        anyhow::ensure!(max_batch_size > 0, "max_batch_size must be positive");
        anyhow::ensure!(q_len > 0, "q_len must be positive");
        anyhow::ensure!(ctx_len > 0, "ctx_len must be positive");
        let config = model.config();
        let ctx = model.device_context();
        let hidden = config.hidden_size;
        let target_hidden_dim = config.hidden_size * config.target_layer_count();
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let total_q_len = max_batch_size * q_len;
        let total_ctx_len = max_batch_size * ctx_len;
        let total_kv_len = max_batch_size * (ctx_len + q_len);
        Ok(Self {
            max_batch_size,
            q_len,
            ctx_len,
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

    pub(crate) fn set_active_batch(&mut self, batch_size: usize) {
        debug_assert!(batch_size <= self.max_batch_size);
        self.total_q_len = batch_size * self.q_len;
        self.total_ctx_len = batch_size * self.ctx_len;
        self.total_kv_len = batch_size * (self.ctx_len + self.q_len);
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
        let needs_rebuild = self
            .ragged_plan
            .as_ref()
            .map(|cached| cached.batch_size != batch_size)
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
            self.ragged_plan = Some(CachedRaggedPlan { batch_size, plan });
        }
        Ok(())
    }
}
