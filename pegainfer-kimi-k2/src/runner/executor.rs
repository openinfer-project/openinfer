mod tp1_dp8;
mod tp8_dp1;

pub(super) use tp1_dp8::Tp1Dp8ForwardExecutor;
pub(super) use tp8_dp1::Tp8Dp1ForwardExecutor;

use anyhow::Result;

use super::worker::KimiOneTokenForwardReport;

pub(super) trait ForwardExecutor {
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        slot: usize,
        decode_batch_size: usize,
        ep_max_seq_len: usize,
    ) -> Result<KimiOneTokenForwardReport>;

    fn forward_prompt_len1_batch(
        &self,
        token_ids: &[u32],
        slots: &[usize],
        decode_batch_size: usize,
    ) -> Result<Vec<KimiOneTokenForwardReport>>;

    fn forward_decode_batch(
        &self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
    ) -> Result<Vec<KimiOneTokenForwardReport>>;

    fn worker_count(&self) -> usize;

    fn gpu_weight_ready_count(&self) -> usize;
}
