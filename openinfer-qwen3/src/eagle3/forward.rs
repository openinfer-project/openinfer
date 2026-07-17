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
    /// Boundary target aux hidden state `[3 * hidden, 1]` at the last committed
    /// position, kept pre-`fc` (vLLM's `aux_hidden_states`). Seeds the next chain
    /// round via `fc(..)`.
    aux_hidden_states: Option<HiddenStates>,
}

impl Eagle3RequestState {
    pub(crate) fn cached_len(&self) -> usize {
        self.cached_len
    }

    /// The boundary target feature `[3 * hidden, 1]` (`None` until captured).
    pub(crate) fn aux_hidden_states(&self) -> Option<&HiddenStates> {
        self.aux_hidden_states.as_ref()
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

    /// Assert a request state was allocated for this drafter's geometry. The
    /// forward path writes K/V into `state` through *unchecked* gemms/copies, so a
    /// state built for a differently-shaped drafter would silently corrupt the KV
    /// cache instead of erroring.
    fn ensure_state_geometry(&self, state: &Eagle3RequestState) -> Result<()> {
        let kv_dim = self.kv_dim();
        anyhow::ensure!(
            state.k.hidden_dim == kv_dim && state.v.hidden_dim == kv_dim,
            "EAGLE-3 request-state K/V dim [{}, {}] does not match drafter kv_dim {}",
            state.k.hidden_dim,
            state.v.hidden_dim,
            kv_dim
        );
        Ok(())
    }

    /// Assert a scratch buffer set was allocated for this drafter's geometry. One
    /// representative buffer per distinct dimension is enough: `new_scratch`
    /// allocates them all from the same config, so any mismatch means a foreign
    /// scratch. Same rationale as `ensure_state_geometry` — the gemms don't check.
    fn ensure_scratch_geometry(&self, scratch: &Eagle3Scratch) -> Result<()> {
        let hidden = self.config.hidden_size;
        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let inter = self.config.intermediate_size;
        let vocab = self.config.draft_vocab_size;
        anyhow::ensure!(
            scratch.q.hidden_dim == q_dim
                && scratch.k.hidden_dim == kv_dim
                && scratch.v.hidden_dim == kv_dim
                && scratch.attn_input.hidden_dim == 2 * hidden
                && scratch.hidden.hidden_dim == hidden
                && scratch.gate.hidden_dim == inter
                && scratch.logits.hidden_dim == vocab,
            "EAGLE-3 scratch geometry does not match drafter \
             (hidden {hidden}, q_dim {q_dim}, kv_dim {kv_dim}, inter {inter}, vocab {vocab})"
        );
        Ok(())
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
            aux_hidden_states: None,
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
        self.ensure_scratch_geometry(scratch)?;
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
        self.ensure_state_geometry(state)?;
        self.ensure_scratch_geometry(scratch)?;

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
        let num_tokens = tokens.len();
        anyhow::ensure!(num_tokens > 0, "EAGLE-3 prefill needs tokens");
        anyhow::ensure!(
            features.hidden_dim == self.fc_input_dim() && features.seq_len == num_tokens,
            "EAGLE-3 batched prefill needs features [{}, {}], got [{}, {}]",
            self.fc_input_dim(),
            num_tokens,
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
            start_position + num_tokens <= state.max_cache_len,
            "EAGLE-3 batched prefill overflows cache: {} + {} > {}",
            start_position,
            num_tokens,
            state.max_cache_len
        );
        self.ensure_state_geometry(state)?;

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
        let mut token_ids_d = ctx.stream.alloc_zeros::<u32>(num_tokens)?;
        ctx.stream.memcpy_htod(tokens, &mut token_ids_d)?;
        let mut embed = HiddenStates::zeros(ctx, hidden, num_tokens)?;
        target.get_embeddings_batch_into(&token_ids_d, &mut embed)?;

        // Residual stream = fc(per-position target features) — teacher forcing.
        let mut residual = HiddenStates::zeros(ctx, hidden, num_tokens)?;
        ops::gemm_into(ctx, &self.fc, features, &mut residual);

        let mut normed_embed = HiddenStates::zeros(ctx, hidden, num_tokens)?;
        let mut normed_hidden = HiddenStates::zeros(ctx, hidden, num_tokens)?;
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
        let mut attn_input = HiddenStates::zeros(ctx, 2 * hidden, num_tokens)?;
        ops::copy_hidden_rows_into(ctx, &normed_embed, &mut attn_input, 0)?;
        ops::copy_hidden_rows_into(ctx, &normed_hidden, &mut attn_input, hidden)?;

        let mut query = HiddenStates::zeros(ctx, q_dim, num_tokens)?;
        let mut key = HiddenStates::zeros(ctx, kv_dim, num_tokens)?;
        let mut value = HiddenStates::zeros(ctx, kv_dim, num_tokens)?;
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            0,
            q_dim,
            &attn_input,
            &mut query,
        );
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            q_dim,
            kv_dim,
            &attn_input,
            &mut key,
        );
        ops::gemm_rows_into(
            ctx,
            &self.midlayer.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &attn_input,
            &mut value,
        );

        // RoPE all N q/k at positions [start, start+N).
        ops::eagle3_rope_into(
            ctx,
            &mut query,
            0,
            num_tokens,
            &mut key,
            &self.cos_cache,
            &self.sin_cache,
            num_q,
            num_kv,
            head_dim,
            start_position,
            start_position,
        )?;

        // Append all N k/v into the cache, then one causal attention over [0, kv_len).
        ops::copy_hidden_token_range_into(ctx, &key, 0, &mut state.k, start_position, num_tokens)?;
        ops::copy_hidden_token_range_into(
            ctx,
            &value,
            0,
            &mut state.v,
            start_position,
            num_tokens,
        )?;
        let kv_len = start_position + num_tokens;

        let mut attn_out = HiddenStates::zeros(ctx, q_dim, num_tokens)?;
        ops::single_prefill_nhd_causal_into(
            ctx,
            &query,
            0,
            num_tokens,
            &state.k,
            &state.v,
            &mut attn_out,
            num_q,
            num_kv,
            head_dim,
            kv_len,
        )?;

        let mut attn_proj = HiddenStates::zeros(ctx, hidden, num_tokens)?;
        ops::gemm_into(ctx, &self.midlayer.o_proj, &attn_out, &mut attn_proj);

        // residual += attn_proj; normed_post = norm(residual).
        let mut normed_post = HiddenStates::zeros(ctx, hidden, num_tokens)?;
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            ctx,
            &mut residual,
            &attn_proj,
            &self.midlayer.post_attention_layernorm,
            eps,
            &mut normed_post,
        )?;

        let mut gate = HiddenStates::zeros(ctx, inter, num_tokens)?;
        let mut up = HiddenStates::zeros(ctx, inter, num_tokens)?;
        let mut act = HiddenStates::zeros(ctx, inter, num_tokens)?;
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
        let mut mlp_out = HiddenStates::zeros(ctx, hidden, num_tokens)?;
        ops::gemm_into(ctx, &self.midlayer.down_proj, &act, &mut mlp_out);

        // residual += mlp_out; normed_final = norm(residual).
        let mut normed_final = HiddenStates::zeros(ctx, hidden, num_tokens)?;
        openinfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            ctx,
            &mut residual,
            &mlp_out,
            &self.norm,
            eps,
            &mut normed_final,
        )?;

        let mut logits = HiddenStates::zeros(ctx, self.config.draft_vocab_size, num_tokens)?;
        ops::gemm_into(ctx, &self.lm_head, &normed_final, &mut logits);

        // The last position's decoder output (post-mlp residual) seeds the chain.
        let mut last_hidden = HiddenStates::zeros(ctx, hidden, 1)?;
        ops::copy_hidden_token_range_into(ctx, &residual, num_tokens - 1, &mut last_hidden, 0, 1)?;

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
        state.aux_hidden_states = Some(seed);
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
        // Fail before writing any speculative KV rather than mid-chain in `draft_step`.
        anyhow::ensure!(
            start_position + k <= state.max_cache_len,
            "EAGLE-3 draft chain [{}, {}) overflows cache {}",
            start_position,
            start_position + k,
            state.max_cache_len
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
    /// (`fc(aux_hidden_states)`) into the chain seed, draft `k` tokens, then **rewind**
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
        // `state.aux_hidden_states` so the `&mut state` for `draft_chain` is free after.
        let mut seed = HiddenStates::zeros(ctx, self.config.hidden_size, 1)?;
        {
            let feature = state
                .aux_hidden_states
                .as_ref()
                .context("EAGLE-3 draft chain has no seed feature (prompt not captured?)")?;
            debug_assert_eq!(
                feature.hidden_dim,
                self.fc_input_dim(),
                "aux_hidden_states must be [fc_input_dim, 1]"
            );
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
    /// boundary `f_{pos-1}` is the previous round's `aux_hidden_states` (kept pre-`fc`).
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
                .aux_hidden_states
                .as_ref()
                .context("EAGLE-3 re-seed has no boundary feature")?;
            debug_assert_eq!(
                boundary.hidden_dim, dim,
                "aux_hidden_states must be [fc_input_dim, 1]"
            );
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
        state.aux_hidden_states = Some(new_seed);
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
