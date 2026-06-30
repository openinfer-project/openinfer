use anyhow::Result;

use crate::config::Eagle3Config;

/// GPU memory the EAGLE-3 draft needs on top of the target KV pool, derived from
/// the draft config so the KV budget can reserve it *before* the draft loads (the
/// draft buffers live outside the paged `KvCacheManager`). Mirrors
/// [`crate::dflash::DFlashMemoryReservation`], split by how each part scales:
///
/// - `kv_bytes_per_token` scales with the KV pool (billed by shrinking the target
///   block count): the draft's single-layer K/V cache, and nothing else. EAGLE-3
///   keeps a standard autoregressive KV cache and consumes the captured 3-layer
///   target features at the fusion step (they are forward *inputs*, not persisted),
///   so — unlike DFlash, which re-projects a context tail every round — no scratch
///   scales per pool token.
/// - `fixed_bytes` does not scale with the pool (billed via the memory margin): the
///   draft weights, the batched-prefill / per-request-decode dense scratch, and the
///   captured-feature buffer (`[3*hidden, N]`) that feeds prefill. These are bounded
///   by the prefill span / decode batch, not the sequence length.
pub(crate) struct Eagle3MemoryReservation {
    pub(crate) kv_bytes_per_token: usize,
    pub(crate) fixed_bytes: usize,
}

impl Eagle3MemoryReservation {
    /// Load the draft config from `model_path` and compute the reservation, without
    /// loading the draft weights — the KV budget reserves this footprint before the
    /// drafter loads (mirrors `DFlashMemoryReservation::from_path`).
    pub(crate) fn from_path(
        model_path: &str,
        max_prefill_tokens: usize,
        max_decode_batch_size: usize,
    ) -> Result<Self> {
        let config = Eagle3Config::from_file(model_path)?;
        Ok(Self::from_config(
            &config,
            max_prefill_tokens,
            max_decode_batch_size,
        ))
    }

    pub(crate) fn from_config(
        config: &Eagle3Config,
        max_prefill_tokens: usize,
        max_decode_batch_size: usize,
    ) -> Self {
        const BF16: usize = 2;
        let hidden = config.hidden_size;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let q_dim = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;
        // EAGLE-3 fuses low/mid/high target hidden, so `fc` takes 3 * hidden in.
        let fc_in = 3 * hidden;

        // Per-pool-token, pool-scaling: the single-layer draft K/V (k + v).
        // `num_hidden_layers == 1`; captured features do not persist (see above).
        let kv_bytes_per_token = config.num_hidden_layers * 2 * kv_dim * BF16;

        // Draft weights: the `midlayer` block + `fc` + draft head + norms. The
        // embedding is reused from the target (the checkpoint ships none), and the
        // `d2t`/`t2d` tables are host-resident, so neither is counted here. +10%
        // slack for the rope caches and allocator alignment.
        let midlayer = BF16
            * ((q_dim + 2 * kv_dim) * (2 * hidden) // qkv_proj (2*hidden input cols)
                + q_dim * hidden                    // o_proj
                + (2 * inter) * hidden              // gate_up_proj
                + inter * hidden                    // down_proj
                + 3 * hidden); // input_layernorm + hidden_norm + post_attention_layernorm
        let fc = BF16 * hidden * fc_in;
        let head = BF16 * (config.draft_vocab_size * hidden + hidden); // lm_head + final norm
        let weights = midlayer + fc + head;
        let weights = weights + weights / 10;

        // Dense forward scratch, summed over the buffers `prefill_batched` /
        // `draft_step` allocate, per token column. Conservative upper bound: the
        // prefill (one-shot, `max_prefill_tokens` wide) and decode (single-token,
        // `max_decode_batch_size` requests) peaks are summed rather than max'd.
        let dense_per_token = BF16
            * (10 * hidden            // embed, hidden, normed_embed/_hidden/_post/_final, o, mlp_out (8) + attn_input (2)
                + 2 * q_dim           // q, attn_out
                + 2 * kv_dim          // k, v
                + 3 * inter           // gate, up, act
                + config.draft_vocab_size); // logits
        let prefill_scratch = dense_per_token * max_prefill_tokens;
        let decode_scratch = dense_per_token * max_decode_batch_size;

        // Captured 3-layer target features `[3*hidden, N]` that feed `prefill_batched`.
        // The draft owns this buffer (the target writes its captured hidden into it,
        // as DFlash does with `pending_context`); it is consumed by prefill and
        // bounded by the prefill span, so it's a fixed term, not per pool token.
        let capture_scratch = fc_in * BF16 * max_prefill_tokens;

        Self {
            kv_bytes_per_token,
            fixed_bytes: weights + prefill_scratch + decode_scratch + capture_scratch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `AngelSlim/Qwen3-4B_eagle3` geometry (matches the shipped `config.json`).
    fn qwen3_4b_eagle3_config() -> Eagle3Config {
        Eagle3Config {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 1,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            draft_vocab_size: 32000,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            max_position_embeddings: 40960,
        }
    }

    #[test]
    fn eagle3_reservation_pins_geometry() {
        let config = qwen3_4b_eagle3_config();

        // Per-pool-token term = single-layer draft K/V: 1 * 2 * (8 * 128) * 2B.
        // This is the number the KV budget bills via effective_bytes_per_block; an
        // off-by-a-layer or geometry regression would silently over/under-reserve.
        let r = Eagle3MemoryReservation::from_config(&config, 2048, 256);
        assert_eq!(
            r.kv_bytes_per_token, 4096,
            "single-layer draft K/V (k+v) per token"
        );

        // Weights alone (no scratch): midlayer + fc + lm_head ≈ 437 MB — matches
        // the shipped model.safetensors (436,899,680 B) — and +10% slack ≈ 480 MB.
        let weights_only = Eagle3MemoryReservation::from_config(&config, 0, 0).fixed_bytes;
        assert!(
            (470_000_000..500_000_000).contains(&weights_only),
            "draft weights+10% ~480MB, got {weights_only}"
        );

        // Scratch adds on top of weights and scales with the prefill/decode bounds.
        assert!(
            r.fixed_bytes > weights_only,
            "fixed_bytes {} should exceed weights-only {}",
            r.fixed_bytes,
            weights_only
        );
    }
}
