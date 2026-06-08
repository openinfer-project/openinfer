//! Sampling scratch and batched decode buffers for Qwen3.5.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use super::config::Config35;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated GPU buffers for Qwen3.5 batch decode (N requests, 1 token each).
pub(crate) struct BatchDecodeBuffers35 {
    pub(crate) max_batch_size: usize,

    // Shared hidden-state flow [dim, batch]
    pub(crate) hidden: HiddenStates,
    pub(crate) normed: HiddenStates,
    pub(crate) attn_results: HiddenStates,
    pub(crate) hidden_mid: HiddenStates,
    pub(crate) gate_out: HiddenStates,
    pub(crate) up_out: HiddenStates,
    pub(crate) act_out: HiddenStates,
    pub(crate) mlp_out: HiddenStates,
    pub(crate) logits: HiddenStates,

    // Full attention [dim, batch]
    pub(crate) q_full: HiddenStates,
    pub(crate) q_attn: HiddenStates,
    pub(crate) k_attn: HiddenStates,
    pub(crate) v_attn: HiddenStates,
    pub(crate) attn_out_full: HiddenStates,

    // Linear attention [dim, batch]
    pub(crate) qkv: HiddenStates,
    pub(crate) z: HiddenStates,
    pub(crate) b_proj: HiddenStates,
    pub(crate) a_proj: HiddenStates,
    pub(crate) gdr_out: HiddenStates,
    pub(crate) normed_gated: HiddenStates,

    // Per-request reusable scratch for linear-attention serial decode
    pub(crate) qkv_tmp: DeviceVec,
    pub(crate) qkv_conv_tmp: DeviceVec,
    pub(crate) b_tmp: DeviceVec,
    pub(crate) a_tmp: DeviceVec,
    pub(crate) gdr_tmp: DeviceVec,

    // Metadata
    pub(crate) token_ids_d: CudaSlice<u32>,
    pub(crate) positions_d: CudaSlice<i32>,
    pub(crate) page_indices_d: CudaSlice<i32>,
    pub(crate) page_indptr_d: CudaSlice<i32>,
    pub(crate) last_page_len_d: CudaSlice<i32>,
    pub(crate) request_indices_d: CudaSlice<i32>,
    pub(crate) kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) kv_chunk_size_d: CudaSlice<i32>,

    // Sampling scratch
    pub(crate) sample_row_indices: CudaSlice<i32>,
    pub(crate) sample_probs: CudaSlice<f32>,
    pub(crate) sample_top1_value: CudaSlice<half::bf16>,
    pub(crate) sample_row_states: CudaSlice<u8>,
    pub(crate) sample_valid: CudaSlice<u8>,
    pub(crate) sample_out: CudaSlice<i32>,

    /// Page index reserved for CUDA Graph padding slots. Padding entries point
    /// here with seq_len=1 so FlashInfer accesses valid (but discarded) memory.
    padding_page_id: i32,
}

impl BatchDecodeBuffers35 {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        max_batch_size: usize,
        max_total_pages: usize,
        padding_page_id: i32,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let bs = max_batch_size;
        let q_proj_dim = config.full_attn_q_proj_dim();
        let q_dim = config.full_attn_q_dim();
        let kv_dim = config.full_attn_kv_dim();
        let qkv_dim = config.linear_attn_qkv_dim();
        let z_dim = config.linear_attn_z_dim();
        let b_dim = config.linear_num_value_heads;
        let a_dim = b_dim;

        Ok(Self {
            max_batch_size: bs,
            hidden: HiddenStates::zeros(ctx, h, bs)?,
            normed: HiddenStates::zeros(ctx, h, bs)?,
            attn_results: HiddenStates::zeros(ctx, h, bs)?,
            hidden_mid: HiddenStates::zeros(ctx, h, bs)?,
            gate_out: HiddenStates::zeros(ctx, config.intermediate_size, bs)?,
            up_out: HiddenStates::zeros(ctx, config.intermediate_size, bs)?,
            act_out: HiddenStates::zeros(ctx, config.intermediate_size, bs)?,
            mlp_out: HiddenStates::zeros(ctx, h, bs)?,
            logits: HiddenStates::zeros(ctx, config.vocab_size, bs)?,

            q_full: HiddenStates::zeros(ctx, q_proj_dim, bs)?,
            q_attn: HiddenStates::zeros(ctx, q_dim, bs)?,
            k_attn: HiddenStates::zeros(ctx, kv_dim, bs)?,
            v_attn: HiddenStates::zeros(ctx, kv_dim, bs)?,
            attn_out_full: HiddenStates::zeros(ctx, q_dim, bs)?,

            qkv: HiddenStates::zeros(ctx, qkv_dim, bs)?,
            z: HiddenStates::zeros(ctx, z_dim, bs)?,
            b_proj: HiddenStates::zeros(ctx, b_dim, bs)?,
            a_proj: HiddenStates::zeros(ctx, a_dim, bs)?,
            gdr_out: HiddenStates::zeros(ctx, z_dim, bs)?,
            normed_gated: HiddenStates::zeros(ctx, z_dim, bs)?,

            qkv_tmp: DeviceVec::zeros(ctx, qkv_dim)?,
            qkv_conv_tmp: DeviceVec::zeros(ctx, qkv_dim)?,
            b_tmp: DeviceVec::zeros(ctx, b_dim)?,
            a_tmp: DeviceVec::zeros(ctx, a_dim)?,
            gdr_tmp: DeviceVec::zeros(ctx, z_dim)?,

            token_ids_d: ctx.stream.alloc_zeros(bs)?,
            positions_d: ctx.stream.alloc_zeros(bs)?,
            // Extra capacity for padding slots (at most max_batch_size padding entries).
            page_indices_d: ctx.stream.alloc_zeros(max_total_pages + bs)?,
            page_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            last_page_len_d: ctx.stream.alloc_zeros(bs)?,
            request_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_tile_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_chunk_size_d: ctx.stream.alloc_zeros(bs)?,

            sample_row_indices: ctx.stream.alloc_zeros(bs)?,
            sample_probs: ctx.stream.alloc_zeros(config.vocab_size)?,
            sample_top1_value: ctx.stream.alloc_zeros(bs)?,
            sample_row_states: ctx
                .stream
                .alloc_zeros(crate::ops::flashinfer_topk_row_states_bytes())?,
            sample_valid: ctx.stream.alloc_zeros(1)?,
            sample_out: ctx.stream.alloc_zeros(bs)?,

            padding_page_id,
        })
    }

    pub(crate) fn set_batch_size(&mut self, bs: usize) {
        assert!(bs <= self.max_batch_size);
        self.hidden.seq_len = bs;
        self.normed.seq_len = bs;
        self.attn_results.seq_len = bs;
        self.hidden_mid.seq_len = bs;
        self.gate_out.seq_len = bs;
        self.up_out.seq_len = bs;
        self.act_out.seq_len = bs;
        self.mlp_out.seq_len = bs;
        self.logits.seq_len = bs;

        self.q_full.seq_len = bs;
        self.q_attn.seq_len = bs;
        self.k_attn.seq_len = bs;
        self.v_attn.seq_len = bs;
        self.attn_out_full.seq_len = bs;

        self.qkv.seq_len = bs;
        self.z.seq_len = bs;
        self.b_proj.seq_len = bs;
        self.a_proj.seq_len = bs;
        self.gdr_out.seq_len = bs;
        self.normed_gated.seq_len = bs;
    }

    /// Sync paged attention metadata to GPU.
    ///
    /// `padded_bs` >= `kv_states.len()`: padding slots (if any) point to the
    /// reserved padding page with seq_len=1 so FlashInfer accesses valid memory.
    pub(crate) fn sync_paged_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_states: &[&KvState],
        padded_bs: usize,
    ) -> Result<()> {
        let real_bs = kv_states.len();
        debug_assert!(padded_bs >= real_bs);

        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(padded_bs);
        let mut chunk_sizes = Vec::with_capacity(padded_bs);

        for kv in kv_states {
            let pages = kv.page_indices_i32();
            all_page_indices.extend_from_slice(&pages);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(kv.last_page_len() as i32);
            chunk_sizes.push(kv.seq_len() as i32);
        }

        // Padding slots: 1 page (the padding page), seq_len=1, last_page_len=1.
        for _ in real_bs..padded_bs {
            all_page_indices.push(self.padding_page_id);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(1);
            chunk_sizes.push(1);
        }

        let request_indices: Vec<i32> = (0..padded_bs as i32).collect();
        let kv_tile_indices = vec![0i32; padded_bs];

        ctx.stream
            .memcpy_htod(&all_page_indices, &mut self.page_indices_d)?;
        ctx.stream.memcpy_htod(&indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_lens, &mut self.last_page_len_d)?;
        ctx.stream
            .memcpy_htod(&chunk_sizes, &mut self.kv_chunk_size_d)?;
        ctx.stream
            .memcpy_htod(&request_indices, &mut self.request_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_tile_indices, &mut self.kv_tile_indices_d)?;

        Ok(())
    }
}
