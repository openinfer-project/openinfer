//! GLM5.2 DSpark draft lane (rank-local): the community
//! `RedHatAI/GLM-5.2-speculator.dspark` drafter — a 5-layer qwen3-architecture
//! dense backbone at GLM's hidden 6144 plus a rank-256 Markov head — proposing
//! 7 greedy draft tokens per round from the target's captured aux hidden
//! states. No collectives anywhere: the draft is replicated on every rank and
//! runs between global steps, DP over that rank's slots.
//!
//! Layout facts are pinned against the `vllm-project/speculators` source
//! (docs/models/glm52/dspark-mtp.md "Layout pinned against speculators
//! source"): the block input is `[anchor, mask x 7]` and block position 0 is
//! the anchor, NOT a draft (anchor-drop) — drafts are read from block
//! positions 1..=7, and the Markov loop starts at position 1 with
//! `prev(1) = anchor`. The context rows the fc projection consumes are the
//! target's residual stream captured AFTER `GLM52_DSPARK_AUX_LAYERS` (see the
//! constant's comment for the off-by-one against the checkpoint's ids), and
//! the pending context always ends one row before the anchor: the anchor's
//! own hidden is only captured when the anchor is fed to the target.
//!
//! The draft's `embed_tokens`/`lm_head` are byte-identical to the target's
//! (sha256-compared at checkpoint inspection) and are NOT loaded — the block
//! embedding and the draft logits reuse the target's matrices.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::CudaSlice;

use openinfer_core::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_1d, load_tensor_2d, mmap_shards,
    precompute_rope,
};
use openinfer_kernels::ops::{
    add_batch_into, argmax_batch_bf16_split_partials_len, copy_hidden_token_range_into,
    dflash_qk_norm_rope_into, embedding_batch, fused_add_rms_norm_round_batch_into,
    gemm_into_checked, gemm_rows_into_checked, markov_step_argmax_into, rms_norm_batch_into,
    silu_mul_batch_into, single_prefill_nhd_noncausal_into,
};
use openinfer_kernels::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

use crate::config::{GLM52_HIDDEN, GLM52_VOCAB};
use crate::model::GLM52_MAX_BATCH_PER_RANK;

/// Draft block width: anchor + 7 mask positions. Equals the top decode bucket
/// — a full verify span is one bucket-8 step.
pub(crate) const GLM52_DSPARK_BLOCK: usize = 8;

/// Drafts proposed per round (anchor-drop: block position 0 is the anchor).
pub(crate) const GLM52_DSPARK_DRAFTS: usize = GLM52_DSPARK_BLOCK - 1;

/// Target layers whose post-layer residual stream feeds the fc context
/// projection, in fc column order — OUR 0-based "residual after layer L"
/// indexing. The checkpoint config says `aux_hidden_state_layer_ids = [8, 23,
/// 39, 55, 70]`, but vLLM (which produced the training data) captures
/// `hidden + residual` BEFORE running layer `idx` — id k is the residual
/// stream after k layers, i.e. after layer k-1. Capturing after layers
/// {8,23,...} (the qwen3 DFlash-checkpoint convention) would be off by one
/// here.
pub(crate) const GLM52_DSPARK_AUX_LAYERS: [usize; 5] = [7, 22, 38, 54, 69];

/// fc input width: the 5 captured layers' hidden states concatenated per
/// token, in `GLM52_DSPARK_AUX_LAYERS` order.
pub(crate) const GLM52_DSPARK_CONTEXT_DIM: usize = GLM52_DSPARK_AUX_LAYERS.len() * GLM52_HIDDEN;

/// The draft writes `block` transient rows past the committed+context length,
/// and the last verifiable anchor sits at position `max_model_len - 1` — so
/// the draft KV/rope tables need `block` positions of headroom past the
/// target's context cap.
pub(crate) fn dspark_cache_len(max_model_len: usize) -> usize {
    max_model_len + GLM52_DSPARK_BLOCK
}

/// Exact GPU bytes the DSpark lane allocates on a rank for a given context
/// cap — the same terms `load`/`Glm52DsparkScratch::new`/
/// `Glm52DsparkSlotState::new` allocate (everything is preallocated to the
/// cap at load: a mid-serving draft round never touches the allocator, so
/// this ledger IS the allocation, not a shadow of it). Per slot: the
/// per-layer draft KV, the pending captured-context rows, and the projected
/// context pair; rank-wide: the varlen tail scratch and the rope tables.
/// The launch-time VRAM probe charges this before deciding `max_model_len`.
pub(crate) fn glm52_dspark_arena_bytes(max_model_len: usize) -> usize {
    let cache_len = dspark_cache_len(max_model_len);
    let bf16 = size_of::<half::bf16>();
    let kv = DSPARK_LAYERS * 2 * DSPARK_QKV_DIM * cache_len * bf16;
    let pending = GLM52_DSPARK_CONTEXT_DIM * cache_len * bf16;
    let context = 2 * GLM52_HIDDEN * cache_len * bf16;
    let tail = (GLM52_HIDDEN + 2 * DSPARK_QKV_DIM) * cache_len * bf16;
    let rope = 2 * cache_len * DSPARK_HEAD_DIM * bf16;
    GLM52_MAX_BATCH_PER_RANK * (kv + pending + context) + tail + rope
}

const DSPARK_LAYERS: usize = 5;
const DSPARK_HEADS: usize = 64;
const DSPARK_HEAD_DIM: usize = 64;
/// 64 q-heads == 64 kv-heads (MHA): q and kv projections are both 4096 wide.
const DSPARK_QKV_DIM: usize = DSPARK_HEADS * DSPARK_HEAD_DIM;
const DSPARK_INTER: usize = 12_288;
const DSPARK_MARKOV_RANK: usize = 256;
const DSPARK_MASK_TOKEN: u32 = 154_856;
const DSPARK_ROPE_THETA: f32 = 8_000_000.0;
const DSPARK_RMS_EPS: f32 = 1.0e-5;

struct DsparkLayer {
    input_ln: DeviceVec,
    /// vstacked `[q; k; v]` `[3 * 4096, 6144]`.
    qkv: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    post_ln: DeviceVec,
    /// vstacked `[gate; up]` `[2 * 12288, 6144]`.
    gate_up: DeviceMatrix,
    down: DeviceMatrix,
}

pub(crate) struct Glm52DsparkModel {
    layers: Vec<DsparkLayer>,
    /// Draft final norm (before the reused target lm_head).
    norm: DeviceVec,
    /// Norm applied to the fc-projected context rows.
    hidden_norm: DeviceVec,
    /// Context projection `[6144, 30720]`.
    fc: DeviceMatrix,
    /// Markov head: `bias(prev) = w2 @ w1[prev]`, both `[154880, 256]`.
    markov_w1: DeviceMatrix,
    markov_w2: DeviceMatrix,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// Draft KV/rope capacity: the target's `max_model_len` plus one block of
    /// transient headroom (see [`dspark_cache_len`]).
    cache_len: usize,
}

/// Crash-early config pin: this module hardcodes the checkpoint's geometry,
/// so a different checkpoint dir must fail at load, not produce garbage.
fn validate_config(path: &Path) -> Result<()> {
    let config_path = path.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow::anyhow!("read {}: {err}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse {}: {err}", config_path.display()))?;
    let expect = |field: &str, want: serde_json::Value| -> Result<()> {
        let got = json
            .get(field)
            .with_context(|| format!("dspark config missing `{field}`"))?;
        ensure!(
            *got == want,
            "dspark config `{field}` = {got}, this build expects {want}"
        );
        Ok(())
    };
    expect("speculators_model_type", "dspark".into())?;
    expect("block_size", 8.into())?;
    expect("mask_token_id", (DSPARK_MASK_TOKEN as u64).into())?;
    expect("markov_rank", (DSPARK_MARKOV_RANK as u64).into())?;
    expect("markov_head_type", "vanilla".into())?;
    expect(
        "aux_hidden_state_layer_ids",
        serde_json::json!([8, 23, 39, 55, 70]),
    )?;
    // DeepSpec-style anchor-first checkpoints carry a `num_anchors` marker;
    // this module implements anchor-drop only.
    ensure!(
        json.get("num_anchors").is_none(),
        "dspark config has `num_anchors` (a DeepSpec anchor-first marker); \
         this module implements the speculators anchor-drop layout only"
    );
    let tl = json
        .get("transformer_layer_config")
        .context("dspark config missing transformer_layer_config")?;
    for (field, want) in [
        ("model_type", serde_json::json!("qwen3")),
        ("num_hidden_layers", (DSPARK_LAYERS as u64).into()),
        ("hidden_size", (GLM52_HIDDEN as u64).into()),
        ("num_attention_heads", (DSPARK_HEADS as u64).into()),
        ("num_key_value_heads", (DSPARK_HEADS as u64).into()),
        ("head_dim", (DSPARK_HEAD_DIM as u64).into()),
        ("intermediate_size", (DSPARK_INTER as u64).into()),
        ("vocab_size", (GLM52_VOCAB as u64).into()),
        ("use_sliding_window", false.into()),
    ] {
        let got = tl
            .get(field)
            .with_context(|| format!("dspark transformer_layer_config missing `{field}`"))?;
        ensure!(
            *got == want,
            "dspark transformer_layer_config `{field}` = {got}, this build expects {want}"
        );
    }
    Ok(())
}

fn ensure_matrix(m: &DeviceMatrix, name: &str, rows: usize, cols: usize) -> Result<()> {
    ensure!(
        m.rows == rows && m.cols == cols,
        "dspark tensor {name} is [{}, {}], expected [{rows}, {cols}]",
        m.rows,
        m.cols
    );
    Ok(())
}

impl Glm52DsparkModel {
    pub(crate) fn load(ctx: &DeviceContext, path: &Path, max_model_len: usize) -> Result<Self> {
        validate_config(path)?;
        let cache_len = dspark_cache_len(max_model_len);
        let path_str = path
            .to_str()
            .context("dspark model path is not valid UTF-8")?;
        let (shard_paths, weight_map) = load_shard_info(path_str)?;
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let mut layers = Vec::with_capacity(DSPARK_LAYERS);
        for layer in 0..DSPARK_LAYERS {
            let p = format!("layers.{layer}");
            let q = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.q_proj.weight"),
            )?;
            let k = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.k_proj.weight"),
            )?;
            let v = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.v_proj.weight"),
            )?;
            ensure_matrix(&q, "q_proj", DSPARK_QKV_DIM, GLM52_HIDDEN)?;
            ensure_matrix(&k, "k_proj", DSPARK_QKV_DIM, GLM52_HIDDEN)?;
            ensure_matrix(&v, "v_proj", DSPARK_QKV_DIM, GLM52_HIDDEN)?;
            let qkv = DeviceMatrix::vstack(ctx, &[&q, &k, &v])?;
            drop((q, k, v));
            let gate = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.mlp.gate_proj.weight"),
            )?;
            let up = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.mlp.up_proj.weight"),
            )?;
            ensure_matrix(&gate, "gate_proj", DSPARK_INTER, GLM52_HIDDEN)?;
            ensure_matrix(&up, "up_proj", DSPARK_INTER, GLM52_HIDDEN)?;
            let gate_up = DeviceMatrix::vstack(ctx, &[&gate, &up])?;
            drop((gate, up));
            let o_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.o_proj.weight"),
            )?;
            ensure_matrix(&o_proj, "o_proj", GLM52_HIDDEN, DSPARK_QKV_DIM)?;
            let down = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.mlp.down_proj.weight"),
            )?;
            ensure_matrix(&down, "down_proj", GLM52_HIDDEN, DSPARK_INTER)?;
            layers.push(DsparkLayer {
                input_ln: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.input_layernorm.weight"),
                )?,
                qkv,
                o_proj,
                q_norm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.self_attn.q_norm.weight"),
                )?,
                k_norm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.self_attn.k_norm.weight"),
                )?,
                post_ln: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                gate_up,
                down,
            });
        }

        let fc = load_tensor_2d(ctx, &shards, &weight_map, "fc.weight")?;
        ensure_matrix(&fc, "fc", GLM52_HIDDEN, GLM52_DSPARK_CONTEXT_DIM)?;
        let markov_w1 = load_tensor_2d(ctx, &shards, &weight_map, "markov_head.markov_w1.weight")?;
        let markov_w2 = load_tensor_2d(ctx, &shards, &weight_map, "markov_head.markov_w2.weight")?;
        ensure_matrix(&markov_w1, "markov_w1", GLM52_VOCAB, DSPARK_MARKOV_RANK)?;
        ensure_matrix(&markov_w2, "markov_w2", GLM52_VOCAB, DSPARK_MARKOV_RANK)?;

        // embed_tokens / lm_head / confidence_head are intentionally not
        // loaded: the first two are byte-identical to the target's, the
        // confidence head is Phase 2.
        let (cos_cache, sin_cache) =
            precompute_rope(ctx, DSPARK_HEAD_DIM, cache_len, DSPARK_ROPE_THETA)?;
        ctx.sync()?;

        Ok(Self {
            layers,
            norm: load_tensor_1d(ctx, &shards, &weight_map, "norm.weight")?,
            hidden_norm: load_tensor_1d(ctx, &shards, &weight_map, "hidden_norm.weight")?,
            fc,
            markov_w1,
            markov_w2,
            cos_cache,
            sin_cache,
            cache_len,
        })
    }

    /// The draft KV/rope capacity slot states must be allocated with.
    pub(crate) fn cache_len(&self) -> usize {
        self.cache_len
    }

    /// Propose `GLM52_DSPARK_DRAFTS` draft tokens for each state, batched.
    ///
    /// Dense ops (embedding, norms, q/o/mlp GEMMs, logits) run once over the
    /// `active * 8` batched buffers; the varlen ops (context projection, tail
    /// concat, k/v GEMMs, rope, KV copy, attention) loop per request — the
    /// same split as the qwen3 DFlash lane. Each state's pending context is
    /// drained into its draft KV here (`committed_len` advances by the
    /// context length).
    ///
    /// `anchors[i] = (token, position)`: the verified token each block extends
    /// and its sequence position — asserted against the state's own
    /// `committed + pending` walk, so scheduler/draft position drift crashes
    /// instead of silently proposing from the wrong rope phase.
    pub(crate) fn propose(
        &self,
        ctx: &DeviceContext,
        embed: &DeviceMatrix,
        lm_head: &DeviceMatrix,
        states: &mut [&mut Glm52DsparkSlotState],
        anchors: &[(u32, usize)],
        scratch: &mut Glm52DsparkScratch,
    ) -> Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>> {
        let active = states.len();
        ensure!(active > 0, "dspark propose needs at least one request");
        ensure!(
            anchors.len() == active,
            "dspark propose: {} states vs {} anchors",
            active,
            anchors.len()
        );
        let block = GLM52_DSPARK_BLOCK;
        let block_rows = active * block;

        let mut context_lens = Vec::with_capacity(active);
        for (i, state) in states.iter().enumerate() {
            let context_len = state.pending_len;
            ensure!(
                context_len > 0,
                "dspark propose before any captured context (slot index {i})"
            );
            let (_, anchor_pos) = anchors[i];
            ensure!(
                anchor_pos == state.committed_len + context_len,
                "dspark anchor position {anchor_pos} != committed {} + pending {} (slot index {i})",
                state.committed_len,
                context_len
            );
            let tail_len = context_len + block;
            ensure!(
                state.committed_len + tail_len <= self.cache_len,
                "dspark draft cache overflow: committed={}, tail={tail_len}, cap={}",
                state.committed_len,
                self.cache_len
            );
            context_lens.push(context_len);
        }

        scratch.activate(block_rows);

        // Block token ids: [anchor, mask x 7] per request.
        scratch.block_token_ids_h[..block_rows].fill(DSPARK_MASK_TOKEN);
        for (i, &(anchor, _)) in anchors.iter().enumerate() {
            scratch.block_token_ids_h[i * block] = anchor;
        }
        {
            let mut dst = scratch.token_ids_d.slice_mut(..block_rows);
            ctx.stream
                .memcpy_htod(&scratch.block_token_ids_h[..block_rows], &mut dst)?;
        }
        embedding_batch(ctx, embed, &scratch.token_ids_d, &mut scratch.hidden)?;

        // Per-request context projection: fc over the pending captured rows,
        // then hidden_norm — persisted in the state so every layer's tail
        // concat can read it.
        for (i, state) in states.iter_mut().enumerate() {
            state.set_context_len(context_lens[i])?;
            state.pending.seq_len = context_lens[i];
            gemm_into_checked(ctx, &self.fc, &state.pending, &mut state.context_projected)?;
            rms_norm_batch_into(
                ctx,
                &state.context_projected,
                &self.hidden_norm,
                DSPARK_RMS_EPS,
                &mut state.context_hidden,
            );
            state.pending_len = 0;
            state.pending.seq_len = 0;
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rms_norm_batch_into(
                ctx,
                &scratch.hidden,
                &layer.input_ln,
                DSPARK_RMS_EPS,
                &mut scratch.normed,
            );
            gemm_rows_into_checked(
                ctx,
                &layer.qkv,
                0,
                DSPARK_QKV_DIM,
                &scratch.normed,
                &mut scratch.q_batch,
            )?;

            for (i, state) in states.iter_mut().enumerate() {
                let context_len = context_lens[i];
                let tail_len = context_len + block;
                let row_offset = i * block;
                scratch.set_tail_len(tail_len)?;

                // tail_input = [context_hidden | normed block rows].
                copy_hidden_token_range_into(
                    ctx,
                    &state.context_hidden,
                    0,
                    &mut scratch.tail_input,
                    0,
                    context_len,
                )?;
                copy_hidden_token_range_into(
                    ctx,
                    &scratch.normed,
                    row_offset,
                    &mut scratch.tail_input,
                    context_len,
                    block,
                )?;

                gemm_rows_into_checked(
                    ctx,
                    &layer.qkv,
                    DSPARK_QKV_DIM,
                    DSPARK_QKV_DIM,
                    &scratch.tail_input,
                    &mut scratch.k_tail,
                )?;
                gemm_rows_into_checked(
                    ctx,
                    &layer.qkv,
                    2 * DSPARK_QKV_DIM,
                    DSPARK_QKV_DIM,
                    &scratch.tail_input,
                    &mut scratch.v_tail,
                )?;

                // Q rows sit at the block positions (anchor_pos..+block); the
                // K tail starts at the first uncached context position.
                dflash_qk_norm_rope_into(
                    ctx,
                    &mut scratch.q_batch,
                    row_offset,
                    block,
                    &mut scratch.k_tail,
                    &layer.q_norm,
                    &layer.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    DSPARK_HEADS,
                    DSPARK_HEADS,
                    DSPARK_HEAD_DIM,
                    state.committed_len + context_len,
                    state.committed_len,
                    DSPARK_RMS_EPS,
                )?;

                let cache = &mut state.layers[layer_idx];
                copy_hidden_token_range_into(
                    ctx,
                    &scratch.k_tail,
                    0,
                    &mut cache.k,
                    state.committed_len,
                    tail_len,
                )?;
                copy_hidden_token_range_into(
                    ctx,
                    &scratch.v_tail,
                    0,
                    &mut cache.v,
                    state.committed_len,
                    tail_len,
                )?;
                // Bidirectional within the block (speculators non_causal);
                // the cached context is everything before the anchor.
                single_prefill_nhd_noncausal_into(
                    ctx,
                    &scratch.q_batch,
                    row_offset,
                    block,
                    &cache.k,
                    &cache.v,
                    &mut scratch.attn_output,
                    DSPARK_HEADS,
                    DSPARK_HEADS,
                    DSPARK_HEAD_DIM,
                    state.committed_len + tail_len,
                )?;
            }

            gemm_into_checked(ctx, &layer.o_proj, &scratch.attn_output, &mut scratch.o_buf)?;
            fused_add_rms_norm_round_batch_into(
                ctx,
                &mut scratch.hidden,
                &scratch.o_buf,
                &layer.post_ln,
                DSPARK_RMS_EPS,
                &mut scratch.normed,
            )?;

            gemm_rows_into_checked(
                ctx,
                &layer.gate_up,
                0,
                DSPARK_INTER,
                &scratch.normed,
                &mut scratch.gate_out,
            )?;
            gemm_rows_into_checked(
                ctx,
                &layer.gate_up,
                DSPARK_INTER,
                DSPARK_INTER,
                &scratch.normed,
                &mut scratch.up_out,
            )?;
            silu_mul_batch_into(
                ctx,
                &scratch.gate_out,
                &scratch.up_out,
                &mut scratch.act_out,
            )?;
            gemm_into_checked(ctx, &layer.down, &scratch.act_out, &mut scratch.o_buf)?;
            add_batch_into(
                ctx,
                &scratch.hidden,
                &scratch.o_buf,
                &mut scratch.hidden_out,
            )?;
            std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_out);
        }

        for (i, state) in states.iter_mut().enumerate() {
            state.committed_len += context_lens[i];
        }

        // Draft logits through the reused target head.
        rms_norm_batch_into(
            ctx,
            &scratch.hidden,
            &self.norm,
            DSPARK_RMS_EPS,
            &mut scratch.logits_normed,
        );
        gemm_into_checked(ctx, lm_head, &scratch.logits_normed, &mut scratch.logits)?;

        self.markov_propose(ctx, anchors, scratch)
    }

    /// Anchor-drop Markov sampling: 7 sequential steps at block positions
    /// 1..=7, `prev(1) = anchor`, `prev(k) = draft k-1`; each step is one
    /// embedding gather + one rank-256 GEMM + one strided argmax-with-bias.
    fn markov_propose(
        &self,
        ctx: &DeviceContext,
        anchors: &[(u32, usize)],
        scratch: &mut Glm52DsparkScratch,
    ) -> Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>> {
        let rows = anchors.len();
        let block = GLM52_DSPARK_BLOCK;
        scratch.w1emb.seq_len = rows;
        scratch.bias.seq_len = rows;

        let anchor_tokens: Vec<u32> = anchors.iter().map(|&(token, _)| token).collect();
        {
            let mut prev = scratch.prev_tokens.slice_mut(..rows);
            ctx.stream.memcpy_htod(&anchor_tokens, &mut prev)?;
        }
        for step in 1..block {
            embedding_batch(
                ctx,
                &self.markov_w1,
                &scratch.prev_tokens,
                &mut scratch.w1emb,
            )?;
            gemm_into_checked(ctx, &self.markov_w2, &scratch.w1emb, &mut scratch.bias)?;
            markov_step_argmax_into(
                ctx,
                &scratch.logits,
                &scratch.bias,
                block,
                step,
                rows,
                &mut scratch.partial_values,
                &mut scratch.partial_indices,
                &mut scratch.next_tokens,
                &mut scratch.sampled_tokens,
            )?;
            std::mem::swap(&mut scratch.prev_tokens, &mut scratch.next_tokens);
        }
        let sampled_view = scratch.sampled_tokens.slice(..rows * block);
        let sampled = ctx.stream.clone_dtoh(&sampled_view)?;
        Ok((0..rows)
            .map(|i| std::array::from_fn(|k| sampled[i * block + 1 + k]))
            .collect())
    }
}

/// Prefix-match speculative acceptance — ported from
/// `openinfer-qwen3/src/speculative.rs::accept_prefix_match`.
///
/// * `proposed` — the `K` draft tokens fed after the anchor.
/// * `target_tokens` — the verify step's committed token after each of the
///   `K + 1` span rows (`target_tokens[0]` follows the anchor,
///   `target_tokens[K]` is the model's continuation after the whole run):
///   the fused argmax for a greedy request, the per-row sampled token for a
///   non-greedy one.
///
/// Returns the longest accepted prefix of `proposed` followed by exactly one
/// model token (the correction at the first divergence, or the bonus
/// continuation when every draft is accepted) — always `1..=K + 1` tokens, so
/// a verify step always makes at least one token of progress.
///
/// With sampled `target_tokens` this rule is LOSSLESS speculative sampling
/// for a deterministic (greedy) draft: row `k`'s token is a true sample from
/// the target distribution given the accepted prefix, and it is committed
/// whether or not it matches `proposed[k]` — the match only decides whether
/// the round keeps riding. Every committed token is therefore distributed
/// exactly as plain sampled decode (measured A/B on jz-38 2026-07-06: the
/// full rejection-sampling variant buys ≤ 1.5% throughput over this rule on
/// code at temperature 1.0 — not worth its complexity).
#[must_use]
pub(crate) fn accept_prefix_match(proposed: &[u32], target_tokens: &[u32]) -> Vec<u32> {
    debug_assert_eq!(
        target_tokens.len(),
        proposed.len() + 1,
        "verify must produce one committed token per draft plus a bonus"
    );
    let n = proposed
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count();
    let mut committed = Vec::with_capacity(n + 1);
    committed.extend_from_slice(&proposed[..n]);
    // `n <= proposed.len() < target_tokens.len()`, so this index is valid.
    committed.push(target_tokens[n]);
    committed
}

struct DsparkLayerKv {
    k: HiddenStates,
    v: HiddenStates,
}

/// Per-slot draft state: the draft KV over committed tokens, the pending
/// captured-context rows not yet projected, and the per-round projected
/// context (persists across the layer loop, so it lives here, not in the
/// shared scratch). Everything is preallocated to `cache_len` at load — a
/// mid-serving draft round must never hit the allocator (a transient OOM
/// there would tear the whole engine down), and the launch-time VRAM probe
/// already charged the full-cap footprint ([`glm52_dspark_arena_bytes`]).
pub(crate) struct Glm52DsparkSlotState {
    layers: Vec<DsparkLayerKv>,
    /// Captured target hidden `[pending_len, 30720]` awaiting projection.
    pending: HiddenStates,
    pending_len: usize,
    committed_len: usize,
    context_projected: HiddenStates,
    context_hidden: HiddenStates,
    /// The drafter's KV capacity ([`Glm52DsparkModel::cache_len`]) — the
    /// pending-context growth cap and the overflow guard bound.
    cache_len: usize,
}

impl Glm52DsparkSlotState {
    pub(crate) fn new(ctx: &DeviceContext, cache_len: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(DSPARK_LAYERS);
        for _ in 0..DSPARK_LAYERS {
            layers.push(DsparkLayerKv {
                k: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, cache_len)?,
                v: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, cache_len)?,
            });
        }
        let mut pending = HiddenStates::zeros(ctx, GLM52_DSPARK_CONTEXT_DIM, cache_len)?;
        pending.seq_len = 0;
        Ok(Self {
            layers,
            pending,
            pending_len: 0,
            committed_len: 0,
            context_projected: HiddenStates::zeros(ctx, GLM52_HIDDEN, cache_len)?,
            context_hidden: HiddenStates::zeros(ctx, GLM52_HIDDEN, cache_len)?,
            cache_len,
        })
    }

    /// Clear the slot for a new request. The KV/pending contents need no
    /// scrubbing: `committed_len`/`pending_len` gate every read, and new
    /// rows overwrite in place.
    pub(crate) fn reset(&mut self) {
        self.committed_len = 0;
        self.pending_len = 0;
        self.pending.seq_len = 0;
    }

    /// Append one step row's captured hidden (a `[30720]` row of the step
    /// capture buffer) to the pending context. The buffer holds `cache_len`
    /// rows from birth — allocation-free by construction.
    pub(crate) fn append_captured_row(
        &mut self,
        ctx: &DeviceContext,
        captured: &CudaSlice<half::bf16>,
        row: usize,
    ) -> Result<()> {
        let required = self.pending_len + 1;
        ensure!(
            self.committed_len + required + GLM52_DSPARK_BLOCK <= self.cache_len,
            "dspark pending context would exceed the draft cache: committed={}, pending={required}",
            self.committed_len
        );
        let src =
            captured.slice(row * GLM52_DSPARK_CONTEXT_DIM..(row + 1) * GLM52_DSPARK_CONTEXT_DIM);
        let mut dst = self.pending.data.slice_mut(
            self.pending_len * GLM52_DSPARK_CONTEXT_DIM..required * GLM52_DSPARK_CONTEXT_DIM,
        );
        ctx.stream.memcpy_dtod(&src, &mut dst)?;
        self.pending_len = required;
        self.pending.seq_len = required;
        Ok(())
    }

    /// Point the projected-context pair at this round's rows. Preallocated to
    /// `cache_len` — the bound is already enforced by the caller's overflow
    /// guard, so exceeding it here is a bug, not a growth request.
    fn set_context_len(&mut self, context_len: usize) -> Result<()> {
        ensure!(
            context_len <= self.context_projected.data.len() / GLM52_HIDDEN,
            "dspark context length {context_len} exceeds the preallocated cap"
        );
        self.context_projected.seq_len = context_len;
        self.context_hidden.seq_len = context_len;
        Ok(())
    }
}

/// Rank-level draft scratch, allocated once for the whole slot batch. Dense
/// buffers hold `GLM52_MAX_BATCH_PER_RANK * block` rows; the varlen tail
/// buffers hold one request's `context + block` rows and grow on demand.
pub(crate) struct Glm52DsparkScratch {
    block_token_ids_h: Vec<u32>,
    token_ids_d: CudaSlice<u32>,
    hidden: HiddenStates,
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    attn_output: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,
    logits_normed: HiddenStates,
    logits: HiddenStates,
    tail_input: HiddenStates,
    k_tail: HiddenStates,
    v_tail: HiddenStates,
    // Markov sample-loop scratch.
    w1emb: HiddenStates,
    bias: HiddenStates,
    partial_values: CudaSlice<f32>,
    partial_indices: CudaSlice<i32>,
    prev_tokens: CudaSlice<u32>,
    next_tokens: CudaSlice<u32>,
    sampled_tokens: CudaSlice<u32>,
}

impl Glm52DsparkScratch {
    pub(crate) fn new(ctx: &DeviceContext, cache_len: usize) -> Result<Self> {
        let max_rows = GLM52_MAX_BATCH_PER_RANK * GLM52_DSPARK_BLOCK;
        // The varlen tail holds one request's context + block rows; context
        // is bounded by the draft cache, so `cache_len` covers every round —
        // preallocated so a draft round never touches the allocator (and the
        // VRAM probe's ledger charged exactly this).
        let tail_capacity = cache_len;
        let partials = argmax_batch_bf16_split_partials_len(GLM52_MAX_BATCH_PER_RANK, GLM52_VOCAB);
        Ok(Self {
            block_token_ids_h: vec![DSPARK_MASK_TOKEN; max_rows],
            token_ids_d: ctx.stream.alloc_zeros(max_rows)?,
            hidden: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            hidden_out: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            normed: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            q_batch: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, max_rows)?,
            attn_output: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, max_rows)?,
            o_buf: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            gate_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            up_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            act_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            logits_normed: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            logits: HiddenStates::zeros(ctx, GLM52_VOCAB, max_rows)?,
            tail_input: HiddenStates::zeros(ctx, GLM52_HIDDEN, tail_capacity)?,
            k_tail: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, tail_capacity)?,
            v_tail: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, tail_capacity)?,
            w1emb: HiddenStates::zeros(ctx, DSPARK_MARKOV_RANK, GLM52_MAX_BATCH_PER_RANK)?,
            bias: HiddenStates::zeros(ctx, GLM52_VOCAB, GLM52_MAX_BATCH_PER_RANK)?,
            partial_values: ctx.stream.alloc_zeros(partials)?,
            partial_indices: ctx.stream.alloc_zeros(partials)?,
            prev_tokens: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            next_tokens: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            sampled_tokens: ctx.stream.alloc_zeros(max_rows)?,
        })
    }

    /// Point the dense buffers at the active prefix (never reallocates).
    fn activate(&mut self, block_rows: usize) {
        assert!(
            block_rows <= GLM52_MAX_BATCH_PER_RANK * GLM52_DSPARK_BLOCK,
            "dspark batch {block_rows} rows exceeds scratch capacity"
        );
        self.hidden.seq_len = block_rows;
        self.hidden_out.seq_len = block_rows;
        self.normed.seq_len = block_rows;
        self.q_batch.seq_len = block_rows;
        self.attn_output.seq_len = block_rows;
        self.o_buf.seq_len = block_rows;
        self.gate_out.seq_len = block_rows;
        self.up_out.seq_len = block_rows;
        self.act_out.seq_len = block_rows;
        self.logits_normed.seq_len = block_rows;
        self.logits.seq_len = block_rows;
    }

    /// Point the varlen tail buffers at this request's `context + block`
    /// rows. Preallocated to the draft cache length — the caller's overflow
    /// guard already bounds `tail_len`, so exceeding it is a bug.
    fn set_tail_len(&mut self, tail_len: usize) -> Result<()> {
        ensure!(
            tail_len <= self.tail_input.data.len() / GLM52_HIDDEN,
            "dspark tail length {tail_len} exceeds the preallocated cap"
        );
        self.tail_input.seq_len = tail_len;
        self.k_tail.seq_len = tail_len;
        self.v_tail.seq_len = tail_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::accept_prefix_match;

    #[test]
    fn accepts_full_run_plus_bonus() {
        assert_eq!(
            accept_prefix_match(&[10, 11, 12], &[10, 11, 12, 13]),
            vec![10, 11, 12, 13]
        );
    }

    #[test]
    fn accepts_prefix_then_correction() {
        assert_eq!(
            accept_prefix_match(&[10, 11, 99], &[10, 11, 22, 33]),
            vec![10, 11, 22]
        );
    }

    #[test]
    fn rejects_first_draft_commits_the_correction() {
        assert_eq!(accept_prefix_match(&[10, 11, 12], &[7, 8, 9, 10]), vec![7]);
    }

    #[test]
    fn empty_proposal_commits_the_model_token() {
        assert_eq!(accept_prefix_match(&[], &[42]), vec![42]);
    }
}
