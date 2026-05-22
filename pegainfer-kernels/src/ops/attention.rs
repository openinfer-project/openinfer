use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::paged_kv::PagedKvLayout;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

// ============================================================================
// Paged prefill (FlashInfer BatchPrefillWithPagedKVCache)
// ============================================================================

/// Pre-computed GPU metadata for paged prefill attention.
///
/// Built once per prefill call, shared across all layers.
/// Supports both single-request (`new`) and multi-request (`new_batch`) prefill.
pub struct PrefillPagedPlan {
    page_indices_d: CudaSlice<i32>,
    page_indptr_d: CudaSlice<i32>,
    last_page_len_d: CudaSlice<i32>,
    batch_indices_d: CudaSlice<i32>,
    positions_d: CudaSlice<i32>,
    q_indptr_d: CudaSlice<i32>,
    request_indices_d: CudaSlice<i32>,
    qo_tile_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    kv_chunk_size_d: CudaSlice<i32>,
    total_num_rows_d: CudaSlice<u32>,
    num_tiles: i32,
    batch_size: i32,
    total_tokens: usize,
    cta_tile_q: i32,
}

impl PrefillPagedPlan {
    pub fn page_indices_d(&self) -> &CudaSlice<i32> {
        &self.page_indices_d
    }
    pub fn page_indptr_d(&self) -> &CudaSlice<i32> {
        &self.page_indptr_d
    }
    pub fn last_page_len_d(&self) -> &CudaSlice<i32> {
        &self.last_page_len_d
    }
    pub fn batch_indices_d(&self) -> &CudaSlice<i32> {
        &self.batch_indices_d
    }
    pub fn positions_d(&self) -> &CudaSlice<i32> {
        &self.positions_d
    }
    pub fn q_indptr_d(&self) -> &CudaSlice<i32> {
        &self.q_indptr_d
    }
    pub fn request_indices_d(&self) -> &CudaSlice<i32> {
        &self.request_indices_d
    }
    pub fn qo_tile_indices_d(&self) -> &CudaSlice<i32> {
        &self.qo_tile_indices_d
    }
    pub fn kv_tile_indices_d(&self) -> &CudaSlice<i32> {
        &self.kv_tile_indices_d
    }
    pub fn kv_chunk_size_d(&self) -> &CudaSlice<i32> {
        &self.kv_chunk_size_d
    }
    pub fn total_num_rows_d(&self) -> &CudaSlice<u32> {
        &self.total_num_rows_d
    }
    pub fn batch_size(&self) -> i32 {
        self.batch_size
    }
    pub fn num_tiles(&self) -> i32 {
        self.num_tiles
    }
    pub fn cta_tile_q(&self) -> i32 {
        self.cta_tile_q
    }

    pub fn new(
        ctx: &DeviceContext,
        page_indices_i32: &[i32],
        last_page_len: usize,
        start_pos: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new_with_cta_tile_q(
            ctx,
            page_indices_i32,
            last_page_len,
            start_pos,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_cta_tile_q(
        ctx: &DeviceContext,
        page_indices_i32: &[i32],
        last_page_len: usize,
        start_pos: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        let kv_len = start_pos + seq_len;

        let page_indices_d = ctx.stream.clone_htod(page_indices_i32)?;
        let page_indptr_d = ctx
            .stream
            .clone_htod(&[0i32, page_indices_i32.len() as i32])?;
        let last_page_len_d = ctx.stream.clone_htod(&[last_page_len as i32])?;

        let batch_indices_d = ctx.stream.clone_htod(&vec![0i32; seq_len])?;
        let positions: Vec<i32> = (start_pos as i32..(start_pos + seq_len) as i32).collect();
        let positions_d = ctx.stream.clone_htod(&positions)?;

        let num_tiles = unsafe {
            ffi::batch_prefill_paged_num_tiles_with_cta_tile_q(
                seq_len as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                cta_tile_q_override,
            )
        };
        anyhow::ensure!(
            num_tiles > 0,
            "invalid prefill CTA tile override {cta_tile_q_override}"
        );
        let cta_tile_q = unsafe {
            ffi::batch_prefill_cta_tile_q_with_override(
                seq_len as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                cta_tile_q_override,
            )
        };
        anyhow::ensure!(
            cta_tile_q > 0,
            "invalid prefill CTA tile override {cta_tile_q_override}"
        );

        let q_indptr_d = ctx.stream.clone_htod(&[0i32, seq_len as i32])?;
        let request_indices_d = ctx.stream.clone_htod(&vec![0i32; num_tiles as usize])?;
        let qo_tile_indices: Vec<i32> = (0..num_tiles).collect();
        let qo_tile_indices_d = ctx.stream.clone_htod(&qo_tile_indices)?;
        let kv_tile_indices_d = ctx.stream.clone_htod(&vec![0i32; num_tiles as usize])?;
        let kv_chunk_size_d = ctx.stream.clone_htod(&[kv_len as i32])?;
        let total_num_rows_d = ctx.stream.clone_htod(&[seq_len as u32])?;

        Ok(Self {
            page_indices_d,
            page_indptr_d,
            last_page_len_d,
            batch_indices_d,
            positions_d,
            q_indptr_d,
            request_indices_d,
            qo_tile_indices_d,
            kv_tile_indices_d,
            kv_chunk_size_d,
            total_num_rows_d,
            num_tiles,
            batch_size: 1,
            total_tokens: seq_len,
            cta_tile_q,
        })
    }

    /// Build plan for multiple requests (batch prefill).
    ///
    /// Page lists must already reflect the post-advance state (pages allocated,
    /// seq_len advanced) for each request.
    pub fn new_batch(
        ctx: &DeviceContext,
        page_indices: &[Vec<i32>],
        last_page_lens: &[usize],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new_batch_with_cta_tile_q(
            ctx,
            page_indices,
            last_page_lens,
            start_positions,
            seq_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_batch_with_cta_tile_q(
        ctx: &DeviceContext,
        page_indices: &[Vec<i32>],
        last_page_lens: &[usize],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        let batch_size = page_indices.len();
        assert_eq!(batch_size, last_page_lens.len());
        assert_eq!(batch_size, start_positions.len());
        assert_eq!(batch_size, seq_lens.len());
        let total_tokens: usize = seq_lens.iter().sum();
        let group_size = num_q_heads / num_kv_heads;

        // Page metadata (concatenated across requests, CSR format)
        let mut all_page_indices = Vec::new();
        let mut page_indptr = vec![0i32];
        let mut last_page_lens_i32 = Vec::with_capacity(batch_size);
        let mut kv_chunk_sizes = Vec::with_capacity(batch_size);

        for (i, pages) in page_indices.iter().enumerate() {
            all_page_indices.extend_from_slice(pages);
            page_indptr.push(all_page_indices.len() as i32);
            last_page_lens_i32.push(last_page_lens[i] as i32);
            kv_chunk_sizes.push((start_positions[i] + seq_lens[i]) as i32);
        }

        // Per-token metadata
        let mut batch_indices = Vec::with_capacity(total_tokens);
        let mut positions = Vec::with_capacity(total_tokens);
        for (i, &seq_len) in seq_lens.iter().enumerate() {
            let start = start_positions[i];
            batch_indices.extend(std::iter::repeat_n(i as i32, seq_len));
            positions.extend((start..start + seq_len).map(|p| p as i32));
        }

        // Q token boundaries (CSR)
        let mut q_indptr = vec![0i32];
        for &seq_len in seq_lens {
            let prev = *q_indptr.last().unwrap();
            q_indptr.push(prev + seq_len as i32);
        }

        // Tile plan: use global cta_tile_q for consistent tiling
        let cta_tile_q = unsafe {
            ffi::batch_prefill_cta_tile_q_with_override(
                total_tokens as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                cta_tile_q_override,
            )
        } as usize;
        anyhow::ensure!(
            cta_tile_q > 0,
            "invalid prefill CTA tile override {cta_tile_q_override}"
        );

        let mut request_indices_v = Vec::new();
        let mut qo_tile_indices_v = Vec::new();
        let mut kv_tile_indices_v = Vec::new();
        for (req_idx, &seq_len) in seq_lens.iter().enumerate() {
            let packed_qo_len = seq_len * group_size;
            let num_tiles_req = packed_qo_len.div_ceil(cta_tile_q);
            for tile in 0..num_tiles_req {
                request_indices_v.push(req_idx as i32);
                qo_tile_indices_v.push(tile as i32);
                kv_tile_indices_v.push(0i32);
            }
        }
        let num_tiles = request_indices_v.len() as i32;

        // Upload all to GPU
        Ok(Self {
            page_indices_d: ctx.stream.clone_htod(&all_page_indices)?,
            page_indptr_d: ctx.stream.clone_htod(&page_indptr)?,
            last_page_len_d: ctx.stream.clone_htod(&last_page_lens_i32)?,
            batch_indices_d: ctx.stream.clone_htod(&batch_indices)?,
            positions_d: ctx.stream.clone_htod(&positions)?,
            q_indptr_d: ctx.stream.clone_htod(&q_indptr)?,
            request_indices_d: ctx.stream.clone_htod(&request_indices_v)?,
            qo_tile_indices_d: ctx.stream.clone_htod(&qo_tile_indices_v)?,
            kv_tile_indices_d: ctx.stream.clone_htod(&kv_tile_indices_v)?,
            kv_chunk_size_d: ctx.stream.clone_htod(&kv_chunk_sizes)?,
            total_num_rows_d: ctx.stream.clone_htod(&[total_tokens as u32])?,
            num_tiles,
            batch_size: batch_size as i32,
            total_tokens,
            cta_tile_q: cta_tile_q as i32,
        })
    }
}

/// Per-layer paged prefill: QK norm + RoPE, append K/V to paged, batch prefill attention.
///
/// Supports both single-request and multi-request plans. For single-request,
/// uses scalar start_pos for RoPE. For multi-request, uses per-token positions.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_into(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    rms_eps: f32,
) -> Result<()> {
    let total_tokens = plan.total_tokens;
    let kv_dim = num_kv_heads * head_dim;
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
    let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _) = plan.page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _) = plan.page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _) = plan.last_page_len_d.device_ptr(&ctx.stream);
    let (bi_ptr, _) = plan.batch_indices_d.device_ptr(&ctx.stream);
    let (pos_ptr, _) = plan.positions_d.device_ptr(&ctx.stream);
    let (qi_ptr, _) = plan.q_indptr_d.device_ptr(&ctx.stream);
    let (ri_ptr, _) = plan.request_indices_d.device_ptr(&ctx.stream);
    let (qti_ptr, _) = plan.qo_tile_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _) = plan.kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _) = plan.kv_chunk_size_d.device_ptr(&ctx.stream);
    let (tnr_ptr, _) = plan.total_num_rows_d.device_ptr(&ctx.stream);

    let stream = ctx.stream.cu_stream();

    unsafe {
        if plan.batch_size == 1 {
            // Single-request: scalar start_pos (no GPU positions array needed)
            ffi::prefill_qk_norm_rope_only_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                total_tokens as i32,
                start_pos as i32,
                rms_eps,
                stream,
            );
        } else {
            // Multi-request: per-token positions from plan
            ffi::qk_norm_rope_batched_decode_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                pos_ptr as *const i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                total_tokens as i32,
                rms_eps,
                stream,
            );
        }

        let src_stride_n = kv_dim as i64;
        let src_stride_h = head_dim as i64;

        let result = ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            bi_ptr as *const i32,
            pos_ptr as *const i32,
            total_tokens as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            stream,
        );
        if result != 0 {
            anyhow::bail!("paged_kv_scatter_cuda failed for layer {layer} with error {result}");
        }

        let result = ffi::batch_prefill_paged_cuda_with_cta_tile_q(
            q_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            qi_ptr as *const i32,
            ri_ptr as *const i32,
            qti_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            tnr_ptr as *const u32,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            total_tokens as i32,
            plan.batch_size,
            plan.num_tiles,
            stride_page,
            sm_scale,
            plan.cta_tile_q(),
            stream,
        );
        if result != 0 {
            anyhow::bail!("batch_prefill_paged_cuda failed for layer {layer} with error {result}");
        }
    }

    Ok(())
}

// ============================================================================
// Paged attention decode (FlashInfer)
// ============================================================================

/// Batched QK RMSNorm + RoPE for decode: per-request positions from GPU array.
///
/// Q: HiddenStates [q_dim, batch_size], K: HiddenStates [kv_dim, batch_size].
/// Both modified in-place.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_batch_decode_into(
    ctx: &DeviceContext,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    positions_d: &CudaSlice<i32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) {
    let batch_size = q.seq_len;
    assert_eq!(k.seq_len, batch_size);

    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _gp) = positions_d.device_ptr(&ctx.stream);

    unsafe {
        ffi::qk_norm_rope_batched_decode_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            batch_size as i32,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }
}

/// Batched QK RMSNorm + partial RoPE for Qwen3.5 HD256 decode.
///
/// Reads Q from interleaved `q_full` ([q, gate] per head), writes prepared Q into `q`,
/// and normalizes/applies partial RoPE to `k` in-place using per-request positions.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_partial_rope_batched_decode_hd256_into(
    ctx: &DeviceContext,
    q_full: &HiddenStates,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    positions_d: &CudaSlice<i32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
    rms_eps: f32,
) {
    let batch_size = q.seq_len;
    debug_assert_eq!(q_full.seq_len, batch_size);
    debug_assert_eq!(k.seq_len, batch_size);

    let (qf_ptr, _gqf) = q_full.data.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _gp) = positions_d.device_ptr(&ctx.stream);

    unsafe {
        ffi::qk_norm_partial_rope_batched_decode_hd256_cuda(
            qf_ptr as *const ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            q_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            batch_size as i32,
            rotary_dim as i32,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }
}

/// Batched paged attention decode: append K/V + FlashInfer BatchDecode for batch_size >= 1.
///
/// Q: HiddenStates [q_dim, batch_size], output: HiddenStates [q_dim, batch_size].
/// Metadata arrays are concatenated across requests (CSR format).
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (pos_ptr, _gpos) = positions_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = kv_chunk_size_d.device_ptr(&ctx.stream);

    let stream = ctx.stream.cu_stream();

    // Step 1: Append K/V to paged cache (batched) using the same generic
    // scatter path as prefill, with explicit request indices and positions.
    let src_stride_n = (num_kv_heads * head_dim) as i64;
    let src_stride_h = head_dim as i64;
    let result = unsafe {
        ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            ri_ptr as *const i32,
            pos_ptr as *const i32,
            batch_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_kv_scatter_cuda (batch decode) failed with error {result}");
    }

    // Step 2: Paged attention decode (batched)
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_attention_decode_cuda (batch) failed with error {result}");
    }

    Ok(())
}

/// Batched paged attention decode using FlashInfer partition-KV/split-K.
///
/// This is intended for low-batch, long-context decode where the non-partition
/// grid `(batch, kv_heads)` does not expose enough CTAs.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_split_kv_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    split_request_indices_d: &CudaSlice<i32>,
    split_kv_tile_indices_d: &CudaSlice<i32>,
    split_kv_chunk_size_d: &CudaSlice<i32>,
    split_o_indptr_d: &CudaSlice<i32>,
    split_block_valid_mask_d: &CudaSlice<u8>,
    split_tmp_v: &mut CudaSlice<bf16>,
    split_tmp_s: &mut CudaSlice<f32>,
    split_padded_slots: usize,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (pos_ptr, _gpos) = positions_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);
    let (split_ri_ptr, _gsri) = split_request_indices_d.device_ptr(&ctx.stream);
    let (split_kti_ptr, _gskti) = split_kv_tile_indices_d.device_ptr(&ctx.stream);
    let (split_kcs_ptr, _gskcs) = split_kv_chunk_size_d.device_ptr(&ctx.stream);
    let (split_o_indptr_ptr, _gsoi) = split_o_indptr_d.device_ptr(&ctx.stream);
    let (split_valid_ptr, _gsv) = split_block_valid_mask_d.device_ptr(&ctx.stream);
    let (split_tmp_v_ptr, _gstmpv) = split_tmp_v.device_ptr_mut(&ctx.stream);
    let (split_tmp_s_ptr, _gstmps) = split_tmp_s.device_ptr_mut(&ctx.stream);

    let stream = ctx.stream.cu_stream();

    let src_stride_n = (num_kv_heads * head_dim) as i64;
    let src_stride_h = head_dim as i64;
    let result = unsafe {
        ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            ri_ptr as *const i32,
            pos_ptr as *const i32,
            batch_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_kv_scatter_cuda (batch split-K decode) failed with error {result}");
    }

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_split_kv_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            split_ri_ptr as *const i32,
            split_kti_ptr as *const i32,
            split_kcs_ptr as *const i32,
            split_o_indptr_ptr as *const i32,
            split_valid_ptr as *const u8,
            split_tmp_v_ptr as *mut ffi::Half,
            split_tmp_s_ptr as *mut f32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            split_padded_slots as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_attention_decode_split_kv_cuda (batch) failed with error {result}");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_hd256_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    debug_assert_eq!(head_dim, 256);
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (pos_ptr, _gpos) = positions_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = kv_chunk_size_d.device_ptr(&ctx.stream);

    let stream = ctx.stream.cu_stream();

    let src_stride_n = (num_kv_heads * head_dim) as i64;
    let src_stride_h = head_dim as i64;
    let result = unsafe {
        ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            ri_ptr as *const i32,
            pos_ptr as *const i32,
            batch_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_kv_scatter_cuda (batch hd256 decode) failed with error {result}");
    }

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_cuda_hd256(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!("paged_attention_decode_cuda_hd256 (batch) failed with error {result}");
    }

    Ok(())
}
