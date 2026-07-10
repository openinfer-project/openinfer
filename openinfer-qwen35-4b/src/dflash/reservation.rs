use anyhow::Result;

use crate::config::Config35;
use crate::recurrent_state::RecurrentState;
use crate::verify_buffers::VerifyBuffers35;

use super::config::DFlashConfig;
use super::scratch::DFlashBatchScratch;
use super::{DFLASH_MAX_ACTIVE_REQUESTS, DFLASH_MAX_VERIFIED_CONTEXT_TOKENS};

/// GPU memory DFlash needs outside the target paged-KV allocation.
///
/// The reservation is split by the value that drives each allocation:
/// - `bytes_per_target_page`: verify-plan metadata sized with the target page pool;
/// - `fixed_bytes`: weights, one request's draft state, and shared single-active
///   execution scratch.
pub(crate) struct DFlashMemoryReservation {
    pub(crate) bytes_per_target_page: usize,
    pub(crate) fixed_bytes: usize,
}

impl DFlashMemoryReservation {
    pub(crate) fn from_path(
        draft_path: &str,
        target: &Config35,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        let config = DFlashConfig::from_file(draft_path)?;
        Self::from_config(&config, target, max_prefill_tokens)
    }

    pub(crate) fn from_config(
        config: &DFlashConfig,
        target: &Config35,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        const BF16: usize = std::mem::size_of::<half::bf16>();

        config.validate_for_target(target)?;
        anyhow::ensure!(
            max_prefill_tokens > 0,
            "Qwen3.5 DFlash reservation requires max_prefill_tokens > 0"
        );
        let hidden = config.hidden_size;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let q_dim = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;
        let capture_layers = config.target_layer_ids.len();
        let verified_context =
            DFLASH_MAX_VERIFIED_CONTEXT_TOKENS.min(target.max_position_embeddings);

        let draft_kv = config.num_hidden_layers * 2 * kv_dim * BF16;
        let pending = hidden * capture_layers * BF16;
        let request_state_bytes_per_token = draft_kv.saturating_add(pending);

        let verify_span = if config.anchor_first() {
            config.block_size + 1
        } else {
            config.block_size
        };
        let max_tail_len = verified_context.saturating_add(config.block_size);
        let draft_scratch =
            DFlashBatchScratch::estimate_bytes(config, DFLASH_MAX_ACTIVE_REQUESTS, max_tail_len);
        let verify_scratch = VerifyBuffers35::estimate_bytes(
            target,
            DFLASH_MAX_ACTIVE_REQUESTS,
            verify_span,
            capture_layers,
            0,
        );
        let recurrent_scratch = RecurrentState::estimate_bytes(target)
            .saturating_mul(DFLASH_MAX_ACTIVE_REQUESTS)
            .saturating_mul(2);

        let per_layer = BF16
            * (hidden * (q_dim + 2 * kv_dim)
                + q_dim * hidden
                + hidden * 2 * inter
                + inter * hidden);
        let fc = BF16 * hidden * (hidden * capture_layers);
        let weights = per_layer * config.num_hidden_layers + fc;
        let weights = weights + weights / 10;

        // The scheduler admits at most one captured DFlash request. Its draft KV
        // and pending target features only need the verified context plus the
        // transient in-fill block.
        let request_cache_tokens = verified_context.saturating_add(config.block_size);
        let request_state = request_cache_tokens.saturating_mul(request_state_bytes_per_token);

        let context_bytes_per_token = 2 * hidden * BF16;
        let request_context = verified_context.saturating_mul(context_bytes_per_token);

        // Prefill capture keeps the aggregate output and the current chunk
        // output alive at the same time before copying into request state.
        let prefill_capture = max_prefill_tokens.saturating_mul(pending).saturating_mul(2);

        // Dynamic buffers grow by replacement. Reserve the old allocation that
        // can coexist briefly with the new one during D2D copy.
        let tail_bytes_per_token = (hidden + 2 * kv_dim) * BF16;
        let growth_slack = verified_context
            .saturating_mul(pending)
            .saturating_add(verified_context.saturating_mul(context_bytes_per_token))
            .saturating_add(config.block_size.saturating_mul(tail_bytes_per_token));

        Ok(Self {
            bytes_per_target_page: std::mem::size_of::<i32>(),
            fixed_bytes: weights
                .saturating_add(draft_scratch)
                .saturating_add(verify_scratch)
                .saturating_add(recurrent_scratch)
                .saturating_add(request_state)
                .saturating_add(request_context)
                .saturating_add(prefill_capture)
                .saturating_add(growth_slack),
        })
    }
}
