use anyhow::Result;
use cudarc::driver::CudaSlice;
use rand::RngExt;
use rand::rngs::StdRng;

use super::weights::Qwen3Model;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::model::{GenerationState, ModelForward};
use pegainfer_core::ops;
use pegainfer_core::sampler::SamplingParams;
use pegainfer_core::tensor::{DeviceContext, DeviceVec};

/// Per-request mutable state for Qwen3.
///
/// Used by the legacy `ModelForward` compatibility path. The production
/// scheduler and model benchmarks use the executor phase API instead.
pub(crate) struct Qwen3State {
    /// Paged KV state — used by the prefill path (FlashInfer).
    pub(super) kv_state: KvState,
    /// Logits from the last forward pass.
    logits: Option<DeviceVec>,
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size).
    sample_probs: CudaSlice<f32>,
    /// One-element scratch for FlashInfer top-1 selection.
    sample_top1_value: CudaSlice<half::bf16>,
    /// Row-state scratch for FlashInfer radix top-k helpers.
    sample_row_states: CudaSlice<u8>,
    /// Per-request validity scratch used by FlashInfer sampling.
    sample_valid: CudaSlice<u8>,
    /// Pre-allocated sampling output (1 element).
    sample_out: CudaSlice<i32>,
}

// SAFETY: Contains raw CUDA pointers (CudaSlice, etc.) that are not Send by default.
// We only access state from the single inference thread.
unsafe impl Send for Qwen3State {}

impl Qwen3State {
    fn new(ctx: &DeviceContext, kv_state: KvState, vocab_size: usize) -> Result<Self> {
        Ok(Self {
            kv_state,
            logits: None,
            sample_probs: ctx
                .stream
                .alloc_zeros(vocab_size)
                .map_err(|e| anyhow::anyhow!("Alloc sample_probs failed: {e}"))?,
            sample_top1_value: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc sample_top1_value failed: {e}"))?,
            sample_row_states: ctx
                .stream
                .alloc_zeros(pegainfer_core::ops::flashinfer_topk_row_states_bytes())
                .map_err(|e| anyhow::anyhow!("Alloc sample_row_states failed: {e}"))?,
            sample_valid: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc sample_valid failed: {e}"))?,
            sample_out: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc sample_out failed: {e}"))?,
        })
    }
}

impl GenerationState for Qwen3State {
    fn logits(&self) -> &DeviceVec {
        self.logits.as_ref().expect("forward() not called yet")
    }

    fn reset(&mut self) -> Result<()> {
        self.kv_state.reset();
        self.logits = None;
        Ok(())
    }
}

impl ModelForward for Qwen3Model {
    type State = Qwen3State;

    fn create_state(&self) -> Result<Self::State> {
        Qwen3State::new(&self.ctx, self.kv_pool.alloc(), self.config.vocab_size)
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        let start_pos = state.kv_state.seq_len();
        let hidden = self.get_embeddings_batch(tokens)?;
        let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.kv_state)?;
        let logits = self.compute_logits_batch(&hidden)?;
        state.logits = Some(logits);
        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        let logits = state.logits.as_ref().expect("forward() not called yet");
        ops::gpu_sample_into(
            &self.ctx,
            logits,
            &mut state.sample_probs,
            &mut state.sample_top1_value,
            &mut state.sample_row_states,
            &mut state.sample_valid,
            &mut state.sample_out,
            params,
            random_val,
        )
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }
}
