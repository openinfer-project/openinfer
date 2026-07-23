//! Pre-allocated scratch buffers for Qwen3.5 prefill-only chunk-wise operators.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::tensor::DeviceContext;
use openinfer_core::tensor::HiddenStates;

use super::config::Config35;

fn checked_product(factors: &[usize], label: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 {label} size overflows usize"))
    })
}

fn checked_sum(terms: &[usize], label: &str) -> Result<usize> {
    terms.iter().try_fold(0usize, |sum, &term| {
        sum.checked_add(term)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 {label} size overflows usize"))
    })
}

/// Scratch buffers for a single Qwen3.5 linear-attention chunk-wise GDR prefill call.
///
/// The first implementation target is intentionally narrow:
/// - batch size = 1
/// - fixed Qwen3.5 linear-attention shapes
/// - forward-only
/// - chunk_size = 64
///
/// Buffers are explicit because the chunk-wise path is naturally a multi-stage
/// pipeline rather than one opaque kernel launch.
pub struct GdrChunkwiseScratch35 {
    /// Chunk-local cumulative gate, fp32: [seq_len, num_value_heads]
    pub(crate) g_cumsum: CudaSlice<f32>,
    /// Beta values, fp32: [seq_len, num_value_heads]
    pub(crate) beta: CudaSlice<f32>,

    /// Expanded + normalized q in token-major layout: [seq_len, num_value_heads * key_dim]
    pub(crate) q_expanded: HiddenStates,
    /// Expanded + normalized k in token-major layout: [seq_len, num_value_heads * key_dim]
    pub(crate) k_expanded: HiddenStates,
    /// Raw v in token-major layout: [seq_len, num_value_heads * value_dim]
    pub(crate) v_raw: HiddenStates,

    /// Chunk attention matrix storage, fp32: [seq_len, num_value_heads, chunk_size]
    pub(crate) a_tril: CudaSlice<f32>,
    /// Inverse (I + A)^-1 in bf16: [seq_len, num_value_heads, chunk_size]
    pub(crate) a_inv: CudaSlice<bf16>,

    /// Prepared W tensor in token-major layout: [seq_len, num_value_heads * key_dim]
    pub(crate) w: HiddenStates,
    /// Prepared U tensor in token-major layout: [seq_len, num_value_heads * value_dim]
    pub(crate) u: HiddenStates,
    /// New value tensor consumed by chunk output stage: [seq_len, num_value_heads * value_dim]
    pub(crate) v_new: HiddenStates,

    /// Per-chunk recurrent state snapshots, fp32: [num_chunks, num_value_heads, key_dim, value_dim]
    pub(crate) chunk_state: CudaSlice<f32>,
}

impl GdrChunkwiseScratch35 {
    pub(crate) const CHUNK_SIZE: usize = 64;

    pub(crate) fn new(ctx: &DeviceContext, config: &Config35, seq_len: usize) -> Result<Self> {
        Self::from_dims(
            ctx,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            seq_len,
        )
    }

    pub fn from_dims(
        ctx: &DeviceContext,
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let kv_hidden_dim = num_value_heads * key_dim;
        let vv_hidden_dim = num_value_heads * value_dim;
        let num_chunks = seq_len.div_ceil(Self::CHUNK_SIZE);

        let g_cumsum: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc g_cumsum failed: {}", e))?;
        let beta: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc beta failed: {}", e))?;
        let a_tril: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_tril failed: {}", e))?;
        let a_inv: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_inv failed: {}", e))?;
        let chunk_state: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(num_chunks * num_value_heads * value_dim * key_dim)
            .map_err(|e| anyhow::anyhow!("Alloc chunk_state failed: {}", e))?;

        Ok(Self {
            g_cumsum,
            beta,
            q_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            k_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            v_raw: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            a_tril,
            a_inv,
            w: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            u: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            v_new: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            chunk_state,
        })
    }

    pub(crate) fn num_chunks(seq_len: usize) -> usize {
        seq_len.div_ceil(Self::CHUNK_SIZE)
    }

    /// Estimate peak GPU memory (bytes) for prefill scratch at a given seq_len.
    ///
    /// Accounts for:
    /// 1. GDR chunkwise scratch (persists across all linear attention layers)
    /// 2. Per-layer transient peak — max of full-attention or MLP intermediates,
    ///    plus shared hidden-state buffers (temporaries freed between layers)
    ///
    /// Direct-paged prefill writes full-attention K/V into the paged pool, so
    /// HND KVCache staging buffers are no longer part of the prefill scratch.
    pub(crate) fn estimate_bytes(config: &Config35, max_seq_len: usize) -> Result<usize> {
        let num_vh = config.linear_num_value_heads;
        let key_dim = config.linear_key_head_dim;
        let val_dim = config.linear_value_head_dim;
        let chunk_sz = Self::CHUNK_SIZE;
        let num_chunks = max_seq_len.div_ceil(chunk_sz);
        let seq = max_seq_len;

        let kv_hidden = checked_product(&[num_vh, key_dim], "prefill KV hidden")?;
        let vv_hidden = checked_product(&[num_vh, val_dim], "prefill value hidden")?;

        // 1. GDR scratch (bf16 = 2 bytes, f32 = 4 bytes)
        let gdr_bytes = {
            let seq_vh = checked_product(&[seq, num_vh], "prefill per-head scratch")?;
            let seq_vh_chunk = checked_product(&[seq, num_vh, chunk_sz], "prefill chunk matrix")?;
            let chunk_state = checked_product(
                &[num_chunks, num_vh, val_dim, key_dim],
                "prefill chunk state",
            )?;
            let kv_seq = checked_product(&[kv_hidden, seq], "prefill KV sequence")?;
            let vv_seq = checked_product(&[vv_hidden, seq], "prefill value sequence")?;
            let f32_elems = checked_sum(
                &[seq_vh, seq_vh, seq_vh_chunk, chunk_state],
                "prefill f32 scratch",
            )?;
            let bf16_elems = checked_sum(
                &[seq_vh_chunk, kv_seq, kv_seq, vv_seq, kv_seq, vv_seq, vv_seq],
                "prefill bf16 scratch",
            )?;
            checked_sum(
                &[
                    checked_product(
                        &[f32_elems, std::mem::size_of::<f32>()],
                        "prefill f32 bytes",
                    )?,
                    checked_product(
                        &[bf16_elems, std::mem::size_of::<bf16>()],
                        "prefill bf16 bytes",
                    )?,
                ],
                "prefill GDR scratch bytes",
            )?
        };

        // 2. Per-layer transient peak (all bf16 = 2 bytes).
        //    Attention and MLP temps don't coexist — MLP runs after attention.
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;

        // Shared: hidden_batch + normed + hidden_plus_attn + normed_for_mlp
        let shared_layer = checked_product(&[hidden_dim, seq, 4], "prefill shared layer scratch")?;

        // Full attention: q_full(with gate) + k + v + attn_out + q_prepped
        let full_qkv = checked_product(
            &[config.num_attention_heads, config.head_dim, 2],
            "prefill full-attention Q",
        )?;
        let full_kv = checked_product(
            &[config.num_key_value_heads, config.head_dim],
            "prefill full-attention KV",
        )?;
        let full_out = checked_product(
            &[config.num_attention_heads, config.head_dim],
            "prefill full-attention output",
        )?;
        let full_attn_width = checked_sum(
            &[
                full_qkv,
                checked_product(&[full_kv, 2], "prefill full-attention KV pair")?,
                checked_product(&[full_out, 2], "prefill full-attention outputs")?,
            ],
            "prefill full-attention width",
        )?;
        let full_attn_temps =
            checked_product(&[full_attn_width, seq], "prefill full-attention scratch")?;

        // MLP: gate_up_out + act_out (same peak footprint as separate gate/up)
        let mlp_temps = checked_product(&[intermediate, seq, 3], "prefill MLP scratch")?;

        let peak_layer = checked_sum(
            &[shared_layer, full_attn_temps.max(mlp_temps)],
            "prefill layer peak",
        )?;
        let per_layer_bytes = checked_product(
            &[peak_layer, std::mem::size_of::<bf16>()],
            "prefill layer bytes",
        )?;

        checked_sum(&[gdr_bytes, per_layer_bytes], "prefill scratch reserve")
    }
}
