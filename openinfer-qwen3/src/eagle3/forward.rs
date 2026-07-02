use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;

use crate::weights::Qwen3Model;
use openinfer_core::ops;
use openinfer_core::tensor::{DeviceContext, HiddenStates};

use super::Eagle3DraftModel;

/// Per-request state the EAGLE-3 draft carries *between* forwards.
/// To record kv cache size for draft memory allocation
pub(crate) struct Eagle3RequestState {
    k: HiddenStates,
    v: HiddenStates,
    cached_len: usize,
    max_cache_len: usize,
    /// Boundary target feature `[3 * hidden, 1]` at the last committed position
    /// For target injection
    seed_feature: Option<HiddenStates>,
}

impl Eagle3RequestState {
    pub(crate) fn cached_len(&self) -> usize {
        self.cached_len
    }

    /// The boundary target feature `[3 * hidden, 1]` (`None` until captured).
    pub(crate) fn seed_feature(&self) -> Option<&HiddenStates> {
        self.seed_feature.as_ref()
    }
}

/// Single-token draft scratch (`seq_len == 1` everywhere). v1 runs one request,
/// one token at a time — batching the draft chain is a follow-up
pub(crate) struct Eagle3Scratch {
    token_id_d: CudaSlice<u32>,
    embed: HiddenStates,         // [hidden, 1]
    hidden: HiddenStates,        // [hidden, 1] residual stream
    normed_embed: HiddenStates,  // [hidden, 1]
    normed_hidden: HiddenStates, // [hidden, 1]
    attn_input: HiddenStates,    // [2 * hidden, 1]
    q: HiddenStates,             // [q_dim, 1]
    k: HiddenStates,             // [kv_dim, 1]
    v: HiddenStates,             // [kv_dim, 1]
    attn_out: HiddenStates,      // [q_dim, 1]
    o: HiddenStates,             // [hidden, 1]
    normed_post: HiddenStates,   // [hidden, 1]
    gate: HiddenStates,          // [inter, 1]
    up: HiddenStates,            // [inter, 1]
    act: HiddenStates,           // [inter, 1]
    mlp_out: HiddenStates,       // [hidden, 1]
    normed_final: HiddenStates,  // [hidden, 1]
    logits: HiddenStates,        // [draft_vocab, 1]
}

impl Eagle3DraftModel {
    fn q_dim(&self) -> usize {
        self.midlayer.q_dim
    }

    fn kv_dim(&self) -> usize {
        self.midlayer.kv_dim
    }

    /// Allocate the single-layer K/V cache for one request. `max_cache_len` bounds
    /// the total drafted+committed positions and must fit the rope cache.
    pub(crate) fn new_request_state(
        &self,
        ctx: &DeviceContext,
        max_cache_len: usize,
    ) -> Result<Eagle3RequestState> {
        anyhow::ensure!(
            max_cache_len > 0 && max_cache_len <= self.config.max_position_embeddings,
            "EAGLE-3 request cache length {} must be in 1..={}",
            max_cache_len,
            self.config.max_position_embeddings
        );
        let kv_dim = self.kv_dim();
        let k = HiddenStates::zeros(ctx, kv_dim, max_cache_len)?;
        let v = HiddenStates::zeros(ctx, kv_dim, max_cache_len)?;
        Ok(Eagle3RequestState {
            k,
            v,
            cached_len: 0,
            max_cache_len,
            seed_feature: None,
        })
    }

    /// Allocate the single-token draft scratch.
    pub(crate) fn new_scratch(&self, ctx: &DeviceContext) -> Result<Eagle3Scratch> {
        let hidden = self.config.hidden_size;
        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let inter = self.config.intermediate_size;
        Ok(Eagle3Scratch {
            token_id_d: ctx.stream.alloc_zeros(1)?,
            embed: HiddenStates::zeros(ctx, hidden, 1)?,
            hidden: HiddenStates::zeros(ctx, hidden, 1)?,
            normed_embed: HiddenStates::zeros(ctx, hidden, 1)?,
            normed_hidden: HiddenStates::zeros(ctx, hidden, 1)?,
            attn_input: HiddenStates::zeros(ctx, 2 * hidden, 1)?,
            q: HiddenStates::zeros(ctx, q_dim, 1)?,
            k: HiddenStates::zeros(ctx, kv_dim, 1)?,
            v: HiddenStates::zeros(ctx, kv_dim, 1)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, 1)?,
            o: HiddenStates::zeros(ctx, hidden, 1)?,
            normed_post: HiddenStates::zeros(ctx, hidden, 1)?,
            gate: HiddenStates::zeros(ctx, inter, 1)?,
            up: HiddenStates::zeros(ctx, inter, 1)?,
            act: HiddenStates::zeros(ctx, inter, 1)?,
            mlp_out: HiddenStates::zeros(ctx, hidden, 1)?,
            normed_final: HiddenStates::zeros(ctx, hidden, 1)?,
            logits: HiddenStates::zeros(ctx, self.config.draft_vocab_size, 1)?,
        })
    }

    /// hidden state fuser for test
    pub(crate) fn seed_hidden_from_context(
        &self,
        ctx: &DeviceContext,
        context_features: &HiddenStates,
        scratch: &mut Eagle3Scratch,
    ) -> Result<()> {
        anyhow::ensure!(
            context_features.hidden_dim == self.fc_input_dim(),
            "EAGLE-3 context feature dim {} does not match fc input {}",
            context_features.hidden_dim,
            self.fc_input_dim()
        );
        anyhow::ensure!(
            context_features.seq_len == 1,
            "EAGLE-3 seed expects one position, got {}",
            context_features.seq_len
        );
        ops::gemm_into(ctx, &self.fc, context_features, &mut scratch.hidden);
        Ok(())
    }

    fn fc_input_dim(&self) -> usize {
        // fc: [hidden, 3 * hidden] , 3 * hidden for fuser input
        self.fc.cols
    }

    /// One EAGLE-3 draft step: consume `token_id` (the current token) and the
    /// residual stream in `scratch.hidden`, update `scratch.hidden` as output
    pub(crate) fn draft_step<'s>(
        &self,
        target: &Qwen3Model,
        state: &mut Eagle3RequestState,
        scratch: &'s mut Eagle3Scratch,
        token_id: u32,
        position: usize,
    ) -> Result<&'s HiddenStates> {
        let ctx = target.device_ctx();
        let hidden = self.config.hidden_size;
        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let inter = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;
        let num_q = self.config.num_attention_heads;
        let num_kv = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        anyhow::ensure!(
            position < state.max_cache_len,
            "EAGLE-3 draft position {} exceeds cache {}",
            position,
            state.max_cache_len
        );
        anyhow::ensure!(
            position == state.cached_len,
            "EAGLE-3 draft step expects position {} == cached_len {}",
            position,
            state.cached_len
        );

        // 1. Embed the current token (reuses the target's embed_tokens).
        {
            let mut dst = scratch.token_id_d.slice_mut(..1);
            ctx.stream.memcpy_htod(&[token_id], &mut dst)?;
        }
        target.get_embeddings_batch_into(&scratch.token_id_d, &mut scratch.embed)?;

        // 2. Norm the embedding and the fused hidden separately.
        ops::rms_norm_batch_into(
            ctx,
            &scratch.embed,
            &self.midlayer.input_layernorm,
            eps,
            &mut scratch.normed_embed,
        );
        ops::rms_norm_batch_into(
            ctx,
            &scratch.hidden,
            &self.midlayer.hidden_norm,
            eps,
            &mut scratch.normed_hidden,
        );

        // 3. attn_input = [normed_embed (rows 0..hidden) | normed_hidden (rows hidden..2h)].
        ops::copy_hidden_rows_into(ctx, &scratch.normed_embed, &mut scratch.attn_input, 0)?;
        ops::copy_hidden_rows_into(ctx, &scratch.normed_hidden, &mut scratch.attn_input, hidden)?;

        // 4. q/k/v projections (qkv_proj input is 2 * hidden).
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            0,
            q_dim,
            &scratch.attn_input,
            &mut scratch.q,
        );
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            q_dim,
            kv_dim,
            &scratch.attn_input,
            &mut scratch.k,
        );
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &scratch.attn_input,
            &mut scratch.v,
        );

        // 5. Plain RoPE (no QK-norm) on the single q/k token at `position`.
        ops::eagle3_rope_into(
            ctx,
            &mut scratch.q,
            0,
            1,
            &mut scratch.k,
            &self.cos_cache,
            &self.sin_cache,
            num_q,
            num_kv,
            head_dim,
            position,
            position,
        )?;

        // 6. Append the rotated K/V into the cache at `position`.
        ops::copy_hidden_token_range_into(ctx, &scratch.k, 0, &mut state.k, position, 1)?;
        ops::copy_hidden_token_range_into(ctx, &scratch.v, 0, &mut state.v, position, 1)?;
        let kv_len = position + 1;

        // 7. Attention: single-query decode — the one draft query attends the whole
        // [0, kv_len) prefix of the draft's contiguous KV.
        ops::single_decode_nhd_into(
            ctx,
            &scratch.q,
            &state.k,
            &state.v,
            &mut scratch.attn_out,
            num_q,
            num_kv,
            head_dim,
            kv_len,
        )?;

        // 8. Output projection.
        ops::gemm_into(
            ctx,
            &self.midlayer.o_proj,
            &scratch.attn_out,
            &mut scratch.o,
        );

        // 9. Residual + post-attention norm (hidden += o in place; normed_post = norm(hidden)).
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            ctx,
            &mut scratch.hidden,
            &scratch.o,
            &self.midlayer.post_attention_layernorm,
            eps,
            &mut scratch.normed_post,
        )?;

        // 10. MLP (SwiGLU).
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.gate_up_proj,
            0,
            inter,
            &scratch.normed_post,
            &mut scratch.gate,
        );
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.gate_up_proj,
            inter,
            inter,
            &scratch.normed_post,
            &mut scratch.up,
        );
        ops::silu_mul_batch_into(ctx, &scratch.gate, &scratch.up, &mut scratch.act)?;
        ops::gemm_into(
            ctx,
            &self.midlayer.down_proj,
            &scratch.act,
            &mut scratch.mlp_out,
        );

        // 11. Residual + final norm.
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            ctx,
            &mut scratch.hidden,
            &scratch.mlp_out,
            &self.norm,
            eps,
            &mut scratch.normed_final,
        )?;

        // 12. Draft head over the reduced vocabulary.
        ops::gemm_into(
            ctx,
            &self.lm_head,
            &scratch.normed_final,
            &mut scratch.logits,
        );

        state.cached_len = kv_len;
        Ok(&scratch.logits)
    }

    /// Teacher-forced prefil for EAGLE Draft
    /// Returns `(logits [draft_vocab, N], last_hidden [hidden, 1])`
    /// Buffers are allocated inline (prefill is one-shot, `N` varies). Does not use
    /// the single-token `Eagle3Scratch`. `tokens[i]` sits at `start_position + i`.
    pub(crate) fn prefill_batched(
        &self,
        target: &Qwen3Model,
        state: &mut Eagle3RequestState,
        features: &HiddenStates,
        tokens: &[u32],
        start_position: usize,
    ) -> Result<(HiddenStates, HiddenStates)> {
        let n = tokens.len();
        anyhow::ensure!(n > 0, "EAGLE-3 prefill needs tokens");
        anyhow::ensure!(
            features.hidden_dim == self.fc_input_dim() && features.seq_len == n,
            "EAGLE-3 batched prefill needs features [{}, {}], got [{}, {}]",
            self.fc_input_dim(),
            n,
            features.hidden_dim,
            features.seq_len
        );
        anyhow::ensure!(
            start_position == state.cached_len,
            "EAGLE-3 batched prefill expects start {} == cached_len {}",
            start_position,
            state.cached_len
        );
        anyhow::ensure!(
            start_position + n <= state.max_cache_len,
            "EAGLE-3 batched prefill overflows cache: {} + {} > {}",
            start_position,
            n,
            state.max_cache_len
        );

        let ctx = target.device_ctx();
        let hidden = self.config.hidden_size;
        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let inter = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;
        let num_q = self.config.num_attention_heads;
        let num_kv = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        // Embed all N tokens.
        let mut token_ids_d = ctx.stream.alloc_zeros::<u32>(n)?;
        ctx.stream.memcpy_htod(tokens, &mut token_ids_d)?;
        let mut embed = HiddenStates::zeros(ctx, hidden, n)?;
        target.get_embeddings_batch_into(&token_ids_d, &mut embed)?;

        // Residual stream = fc(per-position target features) — teacher forcing.
        let mut residual = HiddenStates::zeros(ctx, hidden, n)?;
        ops::gemm_into(ctx, &self.fc, features, &mut residual);

        let mut normed_embed = HiddenStates::zeros(ctx, hidden, n)?;
        let mut normed_hidden = HiddenStates::zeros(ctx, hidden, n)?;
        ops::rms_norm_batch_into(
            ctx,
            &embed,
            &self.midlayer.input_layernorm,
            eps,
            &mut normed_embed,
        );
        ops::rms_norm_batch_into(
            ctx,
            &residual,
            &self.midlayer.hidden_norm,
            eps,
            &mut normed_hidden,
        );

        // attn_input = [normed_embed (rows 0..h) | normed_hidden (rows h..2h)].
        let mut attn_input = HiddenStates::zeros(ctx, 2 * hidden, n)?;
        ops::copy_hidden_rows_into(ctx, &normed_embed, &mut attn_input, 0)?;
        ops::copy_hidden_rows_into(ctx, &normed_hidden, &mut attn_input, hidden)?;

        let mut q = HiddenStates::zeros(ctx, q_dim, n)?;
        let mut k = HiddenStates::zeros(ctx, kv_dim, n)?;
        let mut v = HiddenStates::zeros(ctx, kv_dim, n)?;
        ops::gemm_rows_into(ctx, &self.midlayer.qkv_proj, 0, q_dim, &attn_input, &mut q);
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            q_dim,
            kv_dim,
            &attn_input,
            &mut k,
        );
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &attn_input,
            &mut v,
        );

        // RoPE all N q/k at positions [start, start+N).
        ops::eagle3_rope_into(
            ctx,
            &mut q,
            0,
            n,
            &mut k,
            &self.cos_cache,
            &self.sin_cache,
            num_q,
            num_kv,
            head_dim,
            start_position,
            start_position,
        )?;

        // Append all N k/v into the cache, then one causal attention over [0, kv_len).
        ops::copy_hidden_token_range_into(ctx, &k, 0, &mut state.k, start_position, n)?;
        ops::copy_hidden_token_range_into(ctx, &v, 0, &mut state.v, start_position, n)?;
        let kv_len = start_position + n;

        let mut attn_out = HiddenStates::zeros(ctx, q_dim, n)?;
        ops::single_prefill_nhd_causal_into(
            ctx,
            &q,
            0,
            n,
            &state.k,
            &state.v,
            &mut attn_out,
            num_q,
            num_kv,
            head_dim,
            kv_len,
        )?;

        let mut o = HiddenStates::zeros(ctx, hidden, n)?;
        ops::gemm_into(ctx, &self.midlayer.o_proj, &attn_out, &mut o);

        // residual += o; normed_post = norm(residual).
        let mut normed_post = HiddenStates::zeros(ctx, hidden, n)?;
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            ctx,
            &mut residual,
            &o,
            &self.midlayer.post_attention_layernorm,
            eps,
            &mut normed_post,
        )?;

        let mut gate = HiddenStates::zeros(ctx, inter, n)?;
        let mut up = HiddenStates::zeros(ctx, inter, n)?;
        let mut act = HiddenStates::zeros(ctx, inter, n)?;
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.gate_up_proj,
            0,
            inter,
            &normed_post,
            &mut gate,
        );
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.gate_up_proj,
            inter,
            inter,
            &normed_post,
            &mut up,
        );
        ops::silu_mul_batch_into(ctx, &gate, &up, &mut act)?;
        let mut mlp_out = HiddenStates::zeros(ctx, hidden, n)?;
        ops::gemm_into(ctx, &self.midlayer.down_proj, &act, &mut mlp_out);

        // residual += mlp_out; normed_final = norm(residual).
        let mut normed_final = HiddenStates::zeros(ctx, hidden, n)?;
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            ctx,
            &mut residual,
            &mlp_out,
            &self.norm,
            eps,
            &mut normed_final,
        )?;

        let mut logits = HiddenStates::zeros(ctx, self.config.draft_vocab_size, n)?;
        ops::gemm_into(ctx, &self.lm_head, &normed_final, &mut logits);

        // The last position's decoder output (post-mlp residual) seeds the chain.
        let mut last_hidden = HiddenStates::zeros(ctx, hidden, 1)?;
        ops::copy_hidden_token_range_into(ctx, &residual, n - 1, &mut last_hidden, 0, 1)?;

        state.cached_len = kv_len;
        Ok((logits, last_hidden))
    }

    /// Capture hook: build the draft KV for a freshly-prefilled prompt and record
    /// the boundary feature, applying the EAGLE feature↔token **shift**.
    ///
    /// EAGLE pairs target feature `f_j` with the *next* token's embedding
    /// `e_{j+1}` (predicting `t_{j+2}`). So over a prompt `t_0..t_{P-1}` with
    /// captured features `f_0..f_{P-1}` we teacher-force the `P-1` pairs
    /// `(f_j, e_{j+1})` for `j = 0..P-2` (features = captured cols `0..P-1`, tokens
    /// = `prompt[1..P]`) into draft slots `0..P-2`, and keep the last feature
    /// `f_{P-1}` as the chain's boundary seed (it pairs with the first *generated*
    /// token in the first chain step). `captured_all` is the batch-wide capture;
    /// `token_offset` is this request's first column.
    ///
    /// v1 requires the whole prompt in one prefill chunk (`state.cached_len == 0`),
    /// so the shift never crosses a chunk boundary; longer prompts are skipped by
    /// the caller and fall back to plain decode.
    pub(crate) fn prefill_prompt(
        &self,
        target: &Qwen3Model,
        state: &mut Eagle3RequestState,
        captured_all: &HiddenStates,
        token_offset: usize,
        prompt_tokens: &[u32],
    ) -> Result<()> {
        let p = prompt_tokens.len();
        anyhow::ensure!(p >= 2, "EAGLE-3 prefill needs >= 2 prompt tokens");
        anyhow::ensure!(
            state.cached_len == 0,
            "EAGLE-3 prefill_prompt expects a fresh state (cached_len {})",
            state.cached_len
        );
        anyhow::ensure!(
            captured_all.hidden_dim == self.fc_input_dim(),
            "EAGLE-3 capture features have dim {} but fc expects {}",
            captured_all.hidden_dim,
            self.fc_input_dim()
        );
        anyhow::ensure!(
            token_offset + p <= captured_all.seq_len,
            "EAGLE-3 capture slice [{}, {}) overflows {} captured rows",
            token_offset,
            token_offset + p,
            captured_all.seq_len
        );
        let ctx = target.device_ctx();
        let dim = self.fc_input_dim();
        // Teacher-force the shifted pairs (f_j, e_{j+1}) for j = 0..P-2.
        let mut feat = HiddenStates::zeros(ctx, dim, p - 1)?;
        ops::copy_hidden_token_range_into(ctx, captured_all, token_offset, &mut feat, 0, p - 1)?;
        self.prefill_batched(target, state, &feat, &prompt_tokens[1..p], 0)?;
        // Boundary seed = the last target feature f_{P-1}, kept pre-fc.
        let mut seed = HiddenStates::zeros(ctx, dim, 1)?;
        ops::copy_hidden_token_range_into(
            ctx,
            captured_all,
            token_offset + p - 1,
            &mut seed,
            0,
            1,
        )?;
        state.seed_feature = Some(seed);
        Ok(())
    }

    /// Autoregressive **chain** draft (v1, top-1): from the prefill's last decoder
    /// output (`seed_hidden`) and the last committed token, draft `k` tokens one at
    /// a time — each `draft_step` produces logits, greedy-argmax picks a draft id,
    /// `d2t` maps it to the target vocab, and that token feeds the next step while
    /// the residual stream carries forward. Returns the `k` drafted target-vocab
    /// tokens (the tail of the verify span `[last_token, draft_1, …, draft_k]`).
    ///
    /// `start_position` must equal `state.cached_len` (drafting continues the KV).
    /// v1 syncs logits to host per step for the argmax; device-side sampling (à la
    /// DFlash `select_step_tokens`) is a perf follow-up, as is tree drafting.
    pub(crate) fn draft_chain(
        &self,
        target: &Qwen3Model,
        state: &mut Eagle3RequestState,
        scratch: &mut Eagle3Scratch,
        seed_hidden: &HiddenStates,
        last_token: u32,
        start_position: usize,
        k: usize,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(k > 0, "EAGLE-3 draft chain needs k > 0");
        anyhow::ensure!(
            seed_hidden.hidden_dim == self.config.hidden_size && seed_hidden.seq_len == 1,
            "EAGLE-3 chain seed must be [hidden, 1]"
        );
        let ctx = target.device_ctx();

        // Seed the residual stream with the prefill's last decoder output.
        ops::copy_hidden_token_range_into(ctx, seed_hidden, 0, &mut scratch.hidden, 0, 1)?;

        let mut span = Vec::with_capacity(k);
        let mut token = last_token;
        for i in 0..k {
            // Scope the logits borrow so the next step can re-borrow `scratch`.
            let host = {
                let logits = self.draft_step(target, state, scratch, token, start_position + i)?;
                logits.to_host(ctx)?
            };
            let draft_id = host
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .expect("non-empty draft logits");
            let target_id = self
                .draft_to_target_id(draft_id)
                .context("EAGLE-3 draft id maps outside target vocab")?;
            span.push(target_id);
            token = target_id;
        }
        Ok(span)
    }

    /// One speculative draft round for a single request: fuse the boundary feature
    /// (`fc(seed_feature)`) into the chain seed, draft `k` tokens, then **rewind**
    /// the draft KV to the round-start slot so the round is side-effect-free except
    /// for the returned tokens.
    ///
    /// The chain's KV writes (slots `[C, C+k)`) are speculative; `reseed_after_verify`
    /// rebuilds the accepted prefix teacher-forced from the verify's captured target
    /// hidden, so we discard them here by resetting `cached_len` to `C` (the seed
    /// feature is kept for the re-seed's boundary column). Returns the `k` drafted
    /// target-vocab tokens (the verify span's tail).
    pub(crate) fn chain_round(
        &self,
        target: &Qwen3Model,
        state: &mut Eagle3RequestState,
        scratch: &mut Eagle3Scratch,
        current_token: u32,
        k: usize,
    ) -> Result<Vec<u32>> {
        let ctx = target.device_ctx();
        // Chain seed = fc(boundary target feature). Scope the immutable borrow of
        // `state.seed_feature` so the `&mut state` for `draft_chain` is free after.
        let mut seed = HiddenStates::zeros(ctx, self.config.hidden_size, 1)?;
        {
            let feature = state
                .seed_feature
                .as_ref()
                .context("EAGLE-3 draft chain has no seed feature (prompt not captured?)")?;
            ops::gemm_into(ctx, &self.fc, feature, &mut seed);
        }
        let start = state.cached_len;
        let result = self.draft_chain(target, state, scratch, &seed, current_token, start, k);
        // Discard the speculative chain KV; the re-seed rebuilds the accepted
        // prefix teacher-forced. On a draft error the request is dropped.
        state.cached_len = start;
        result
    }

    /// Re-seed the chain after a verify step, applying the EAGLE feature↔token
    /// shift. Let `pos` be the position of `span_tokens[0]` (the round's current
    /// token) and `n = matched_draft_tokens`; the round committed `t_{pos+1..pos+n}`
    /// (matched drafts) plus the bonus `t_{pos+n+1}`.
    ///
    /// To predict `t_{pos+n+2}` next, the chain needs `(f_{pos+n}, e(t_{pos+n+1}))`.
    /// So we (1) teacher-force the `n+1` shifted pairs `(f_{pos-1+i}, t_{pos+i})` for
    /// `i = 0..n` — features `[boundary f_{pos-1} ‖ captured f_pos..f_{pos+n-1}]`,
    /// tokens `span_tokens[0..n+1]` — into slots `[C, C+n+1)` (`C == cached_len`,
    /// rewound by `chain_round`), rebuilding the committed prefix's draft KV; and
    /// (2) set the new boundary feature to `f_{pos+n} = captured[:, n]`. The
    /// boundary `f_{pos-1}` is the previous round's `seed_feature` (kept pre-`fc`).
    pub(crate) fn reseed_after_verify(
        &self,
        target: &Qwen3Model,
        state: &mut Eagle3RequestState,
        captured: &HiddenStates,
        token_offset: usize,
        span_tokens: &[u32],
        matched_draft_tokens: usize,
    ) -> Result<()> {
        let n = matched_draft_tokens;
        let len = n + 1;
        anyhow::ensure!(
            len <= span_tokens.len(),
            "EAGLE-3 re-seed needs {} span tokens, got {}",
            len,
            span_tokens.len()
        );
        anyhow::ensure!(
            captured.hidden_dim == self.fc_input_dim(),
            "EAGLE-3 verify capture has dim {} but fc expects {}",
            captured.hidden_dim,
            self.fc_input_dim()
        );
        // Need captured columns 0..n for the matched-feature pairs plus column n
        // for the new boundary feature.
        anyhow::ensure!(
            token_offset + n < captured.seq_len,
            "EAGLE-3 re-seed slice [{}, {}] overflows {} captured rows",
            token_offset,
            token_offset + n,
            captured.seq_len
        );
        let ctx = target.device_ctx();
        let dim = self.fc_input_dim();

        // Shifted feature row: [boundary f_{pos-1} ‖ f_pos..f_{pos+n-1}] = n+1 cols.
        let mut feat = HiddenStates::zeros(ctx, dim, len)?;
        {
            let boundary = state
                .seed_feature
                .as_ref()
                .context("EAGLE-3 re-seed has no boundary feature")?;
            ops::copy_hidden_token_range_into(ctx, boundary, 0, &mut feat, 0, 1)?;
        }
        if n > 0 {
            ops::copy_hidden_token_range_into(ctx, captured, token_offset, &mut feat, 1, n)?;
        }
        // Teacher-force the committed prefix into draft slots [C, C+n+1).
        let start_position = state.cached_len;
        self.prefill_batched(target, state, &feat, &span_tokens[..len], start_position)?;

        // New boundary feature = f_{pos+n} (the target feature at the last committed
        // position), kept pre-fc for the next round's re-seed.
        let mut new_seed = HiddenStates::zeros(ctx, dim, 1)?;
        ops::copy_hidden_token_range_into(ctx, captured, token_offset + n, &mut new_seed, 0, 1)?;
        state.seed_feature = Some(new_seed);
        Ok(())
    }

    /// Map a draft-vocabulary id back to the target vocabulary:
    /// `target_id = draft_id + d2t[draft_id]`.
    pub(crate) fn draft_to_target_id(&self, draft_id: usize) -> Option<u32> {
        let offset = *self.d2t.get(draft_id)?;
        let target_id = draft_id as i64 + offset;
        (0..self.config.vocab_size as i64)
            .contains(&target_id)
            .then_some(target_id as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{ModelRuntimeConfig, Qwen3Model};
    use std::path::Path;

    /// Load the target + EAGLE-3 drafter, or return `None` (with a skip message) if
    /// the weights are absent. `OPENINFER_TEST_MODEL_PATH` / `OPENINFER_EAGLE3_TEST_MODEL_PATH`.
    fn load_or_skip() -> Option<(Qwen3Model, Eagle3DraftModel)> {
        let target_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "models/Qwen3-4B".to_string());
        let eagle_path = std::env::var("OPENINFER_EAGLE3_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "models/Qwen3-4B_eagle3".to_string());
        if !Path::new(&target_path).join("config.json").exists()
            || !Path::new(&eagle_path).join("config.json").exists()
        {
            eprintln!(
                "skipping EAGLE-3 forward test; set OPENINFER_TEST_MODEL_PATH and OPENINFER_EAGLE3_TEST_MODEL_PATH"
            );
            return None;
        }
        let target = Qwen3Model::from_safetensors_with_runtime(
            &target_path,
            ModelRuntimeConfig {
                enable_cuda_graph: false,
                tensor_parallel: None,
                device_ordinal: 0,
                ..Default::default()
            },
        )
        .expect("load target");
        let drafter = {
            let ctx = target.device_ctx();
            Eagle3DraftModel::from_safetensors_for_target(ctx, &eagle_path, &target)
                .expect("load EAGLE-3 drafter")
        };
        Some((target, drafter))
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .expect("argmax")
    }

    /// Runtime sanity: a batched prefill produces finite, correctly-shaped
    /// per-position draft logits whose last-position argmax maps back into the
    /// target vocab. No exact-logit reference yet (HF EAGLE-3 parity is later).
    #[test]
    #[ignore = "requires GPU + Qwen3-4B target and EAGLE-3 drafter weights"]
    fn eagle3_forward_smoke() {
        let Some((target, drafter)) = load_or_skip() else {
            return;
        };
        let ctx = target.device_ctx();
        let dvoc = drafter.config.draft_vocab_size;

        let mut state = drafter.new_request_state(ctx, 64).expect("request state");

        // Synthetic captured features — one column per token (teacher forcing).
        let prompt: Vec<u32> = vec![1, 2, 3, 4];
        let n = prompt.len();
        let features = HiddenStates::zeros(ctx, drafter.fc_input_dim(), n).expect("features");

        let (logits, _seed) = drafter
            .prefill_batched(&target, &mut state, &features, &prompt, 0)
            .expect("prefill");
        assert_eq!(state.cached_len(), n);

        let host = logits.to_host(ctx).expect("logits to host");
        assert_eq!(host.len(), dvoc * n);
        assert!(
            host.iter().all(|v| v.is_finite()),
            "draft logits must be finite"
        );

        // The last position's prediction maps back into the target vocab.
        let last = &host[(n - 1) * dvoc..n * dvoc];
        let target_id = drafter
            .draft_to_target_id(argmax(last))
            .expect("d2t maps in range");
        assert!((target_id as usize) < drafter.config.vocab_size);
    }

    /// The batched prefill (one causal `single_prefill_nhd_causal_into`) must match
    /// a per-token sequential reference (a `draft_step` loop = N single-query
    /// attentions over the growing prefix): both are teacher-forced and causal, so
    /// they're numerically equivalent up to bf16 accumulation order. Compares the
    /// last position's logits.
    #[test]
    #[ignore = "requires GPU + Qwen3-4B target and EAGLE-3 drafter weights"]
    fn eagle3_batched_prefill_matches_sequential() {
        let Some((target, drafter)) = load_or_skip() else {
            return;
        };
        let ctx = target.device_ctx();
        let dvoc = drafter.config.draft_vocab_size;
        let feat_dim = drafter.fc_input_dim();
        let prompt: Vec<u32> = vec![11, 22, 33, 44, 55];
        let n = prompt.len();

        // Non-trivial features so each position's fc/hidden path actually differs.
        let feat_host: Vec<half::bf16> = (0..feat_dim * n)
            .map(|i| half::bf16::from_f32(((i % 17) as f32 - 8.0) * 0.01))
            .collect();
        let features = HiddenStates::from_host(ctx, &feat_host, feat_dim, n).expect("features");

        // Batched prefill → per-position logits [dvoc, n] (column i at i*dvoc).
        let mut state_b = drafter.new_request_state(ctx, 64).expect("state b");
        let (logits_b, _seed) = drafter
            .prefill_batched(&target, &mut state_b, &features, &prompt, 0)
            .expect("batched prefill");
        assert_eq!(state_b.cached_len(), n);
        let host_b = logits_b.to_host(ctx).expect("host b");
        assert_eq!(host_b.len(), dvoc * n);

        // Sequential per-token teacher-forced reference; keep the last position.
        let mut state_s = drafter.new_request_state(ctx, 64).expect("state s");
        let mut scratch = drafter.new_scratch(ctx).expect("scratch");
        let mut feat_col = HiddenStates::zeros(ctx, feat_dim, 1).expect("feat col");
        let mut last_seq = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            ops::copy_hidden_token_range_into(ctx, &features, i, &mut feat_col, 0, 1)
                .expect("feature column");
            drafter
                .seed_hidden_from_context(ctx, &feat_col, &mut scratch)
                .expect("seed");
            let lg = drafter
                .draft_step(&target, &mut state_s, &mut scratch, tok, i)
                .expect("seq step");
            last_seq = lg.to_host(ctx).expect("seq host");
        }

        let last_b = &host_b[(n - 1) * dvoc..n * dvoc];
        assert_eq!(last_b.len(), last_seq.len());
        let max_abs = last_b
            .iter()
            .zip(last_seq.iter())
            .map(|(a, b)| {
                assert!(a.is_finite() && b.is_finite());
                (a - b).abs()
            })
            .fold(0f32, f32::max);
        let scale = last_b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        eprintln!(
            "batched vs sequential last-pos: max|Δ|={max_abs} (rel {:.4}), argmax_b={}, argmax_s={}",
            max_abs / scale,
            argmax(last_b),
            argmax(&last_seq),
        );
        // Primary correctness signal: both paths predict the same token. The raw
        // logits differ only by bf16 accumulation across two different kernel paths
        // (batched cuBLAS + causal FlashInfer vs per-token gemv + N non-causal
        // calls), so gate the magnitude relatively — a real bug (wrong causal
        // alignment, KV, etc.) diverges by many multiples of the logit scale.
        assert_eq!(
            argmax(last_b),
            argmax(&last_seq),
            "batched and sequential prefill disagree on the argmax token"
        );
        assert!(
            max_abs < 0.05 * scale,
            "batched vs sequential last-pos logits diverge: max|Δ|={max_abs}, scale={scale}"
        );
    }

    /// Runtime sanity for the autoregressive chain proposer: prefill, then draft a
    /// `k`-token chain from the prefill seed. Asserts the span has `k` tokens, all
    /// in the target vocab, and that the chain appended `k` KV positions. (Synthetic
    /// features, so this checks the proposer mechanics, not acceptance quality.)
    #[test]
    #[ignore = "requires GPU + Qwen3-4B target and EAGLE-3 drafter weights"]
    fn eagle3_draft_chain_produces_valid_span() {
        let Some((target, drafter)) = load_or_skip() else {
            return;
        };
        let ctx = target.device_ctx();
        let feat_dim = drafter.fc_input_dim();
        let prompt: Vec<u32> = vec![11, 22, 33, 44];
        let n = prompt.len();
        let k = 5usize;

        let features = HiddenStates::zeros(ctx, feat_dim, n).expect("features");
        let mut state = drafter.new_request_state(ctx, 64).expect("state");
        let mut scratch = drafter.new_scratch(ctx).expect("scratch");

        let (_logits, seed) = drafter
            .prefill_batched(&target, &mut state, &features, &prompt, 0)
            .expect("prefill");
        assert_eq!(state.cached_len(), n);

        // Draft k tokens from the prefill seed + an (arbitrary) last token.
        let span = drafter
            .draft_chain(&target, &mut state, &mut scratch, &seed, 99u32, n, k)
            .expect("draft chain");

        assert_eq!(span.len(), k, "chain must produce k draft tokens");
        assert!(
            span.iter()
                .all(|&t| (t as usize) < drafter.config.vocab_size),
            "every drafted token maps into the target vocab"
        );
        assert_eq!(
            state.cached_len(),
            n + k,
            "chain appends one KV position per drafted token"
        );
    }

    /// EAGLE-3 drafter **golden gate**: replay a seed-pinned batched prefill and
    /// compare per-position draft logits against the OFFICIAL SafeAILab/EAGLE
    /// drafter's reference (fixture from `tools/accuracy/dump_qwen3_4b_eagle3_golden.py`).
    ///
    /// This is the real correctness check — it validates the whole drafter forward
    /// (fc fusion, `eagle3_rope`, the NHD attention kernels, mlp, head) against an
    /// INDEPENDENT implementation. A numeric bug that only degrades acceptance is
    /// invisible to the losslessness gate (verify is drafter-agnostic) and to a
    /// kernel-vs-kernel consistency check (a shared bug cancels); it is caught here.
    #[test]
    #[ignore = "requires GPU + Qwen3-4B target, EAGLE-3 drafter, and the golden fixture"]
    fn eagle3_drafter_golden_gate() {
        const GOLDEN: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../test_data/qwen3-4b-eagle3-golden.safetensors"
        );
        if !std::path::Path::new(GOLDEN).exists() {
            eprintln!(
                "skipping eagle3 drafter golden gate: {GOLDEN} missing \
                 (regenerate with tools/accuracy/dump_qwen3_4b_eagle3_golden.py)"
            );
            return;
        }
        let Some((target, drafter)) = load_or_skip() else {
            return;
        };
        let ctx = target.device_ctx();

        let bytes = std::fs::read(GOLDEN).expect("read golden");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse golden");
        let read_i32 = |name: &str| -> Vec<i32> {
            st.tensor(name)
                .unwrap_or_else(|_| panic!("golden missing {name}"))
                .data()
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let read_f32 = |name: &str| -> (Vec<f32>, Vec<usize>) {
            let t = st
                .tensor(name)
                .unwrap_or_else(|_| panic!("golden missing {name}"));
            let v = t
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (v, t.shape().to_vec())
        };
        let read_bf16 = |name: &str| -> Vec<half::bf16> {
            st.tensor(name)
                .unwrap_or_else(|_| panic!("golden missing {name}"))
                .data()
                .chunks_exact(2)
                .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        };

        let tokens: Vec<u32> = read_i32("tokens").iter().map(|&t| t as u32).collect();
        let features_host = read_bf16("features"); // [N, 3h] token-major flat == HiddenStates layout
        let (ref_logits, ref_shape) = read_f32("logits"); // [N, dvoc]
        let n = tokens.len();
        let dvoc = drafter.config.draft_vocab_size;
        let feat_dim = 3 * drafter.config.hidden_size; // fc input = 3 aux layers
        assert_eq!(ref_shape, vec![n, dvoc], "golden logits shape");
        assert_eq!(features_host.len(), n * feat_dim, "golden features size");

        let features = HiddenStates::from_host(ctx, &features_host, feat_dim, n).expect("features");
        let mut state = drafter
            .new_request_state(ctx, (n + 4).max(8))
            .expect("request state");
        let (logits, _last) = drafter
            .prefill_batched(&target, &mut state, &features, &tokens, 0)
            .expect("prefill");
        let host = logits.to_host(ctx).expect("logits host"); // [dvoc, N] token-major
        assert_eq!(host.len(), n * dvoc);

        // Per position, score the Rust drafter's argmax pick *in the reference
        // distribution* — its regret below the reference's own top logit. regret==0
        // is an exact argmax match; a tiny regret is a benign bf16 tie the two
        // implementations resolve differently (robust where strict argmax equality
        // would be brittle). Also track the worst full-vector logit deviation.
        let mut max_regret_rel = 0f32;
        let mut max_rel = 0f32;
        for s in 0..n {
            let rust = &host[s * dvoc..(s + 1) * dvoc];
            let refl = &ref_logits[s * dvoc..(s + 1) * dvoc];
            let (a_rust, a_ref) = (argmax(rust), argmax(refl));
            let ref_top = refl[a_ref];
            let scale = refl.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            let regret_rel = (ref_top - refl[a_rust]).max(0.0) / scale;
            let d = rust
                .iter()
                .zip(refl)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            max_regret_rel = max_regret_rel.max(regret_rel);
            max_rel = max_rel.max(d / scale);
            eprintln!(
                "pos {s}: rust argmax={a_rust} ref argmax={a_ref} regret/scale={regret_rel:.4} \
                 max|Δ|/scale={:.4}",
                d / scale
            );
        }
        eprintln!(
            "eagle3 drafter golden: max argmax-regret={max_regret_rel:.4}, max rel logit Δ={max_rel:.4}"
        );
        // Acceptance-relevant invariant: the Rust drafter's pick is the reference's
        // own top (or a bf16 tie with it).
        assert!(
            max_regret_rel < 0.02,
            "drafter argmax diverged from the official EAGLE reference: max regret/scale={max_regret_rel}"
        );
        // Full-vector logits differ only by bf16 accumulation across two implementations.
        assert!(
            max_rel < 0.15,
            "drafter logits diverge from the official EAGLE reference: max rel Δ={max_rel}"
        );
    }
}
