use anyhow::{Result, bail, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::{
    ffi,
    tensor::{DeviceContext, DeviceMatrix, HiddenStates},
};

pub const KIMI_K2_MLA_LOCAL_HEADS_TP8: usize = 8;
pub const KIMI_K2_MLA_Q_HEAD_DIM: usize = 192;
pub const KIMI_K2_MLA_V_HEAD_DIM: usize = 128;
pub const KIMI_K2_MLA_ROPE_DIM: usize = KIMI_K2_MLA_Q_HEAD_DIM - KIMI_K2_MLA_V_HEAD_DIM;
pub const KIMI_K2_MLA_NOPE_DIM: usize = KIMI_K2_MLA_Q_HEAD_DIM - KIMI_K2_MLA_ROPE_DIM;
const KIMI_K2_MLA_Q_LORA_RANK: usize = 1536;
pub const KIMI_K2_MLA_KV_LORA_RANK: usize = 512;
pub const KIMI_K2_MLA_KV_A_OUT: usize = 576;
pub const KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8: usize = 2048;
pub const KIMI_K2_MLA_Q_LOCAL_OUT_TP8: usize = KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_Q_HEAD_DIM;
pub const KIMI_K2_MLA_O_LOCAL_IN_TP8: usize = KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_V_HEAD_DIM;
pub const KIMI_K2_MLA_QKV_A_OUT: usize = KIMI_K2_MLA_Q_LORA_RANK + KIMI_K2_MLA_KV_A_OUT;
pub const KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8: usize =
    KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_KV_LORA_RANK;
pub const KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8: usize =
    KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_ROPE_DIM;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiMlaPagedKvLayout {
    pub max_pages: usize,
    pub page_size: usize,
    pub batch_size: usize,
    pub ckv_stride_page: usize,
    pub ckv_stride_n: usize,
    pub kpe_stride_page: usize,
    pub kpe_stride_n: usize,
}

impl KimiMlaPagedKvLayout {
    pub fn separate_contiguous(max_pages: usize, page_size: usize, batch_size: usize) -> Self {
        Self {
            max_pages,
            page_size,
            batch_size,
            ckv_stride_page: page_size * KIMI_K2_MLA_KV_LORA_RANK,
            ckv_stride_n: KIMI_K2_MLA_KV_LORA_RANK,
            kpe_stride_page: page_size * KIMI_K2_MLA_ROPE_DIM,
            kpe_stride_n: KIMI_K2_MLA_ROPE_DIM,
        }
    }

    pub fn required_ckv_len(&self) -> Result<usize> {
        required_cache_len(
            self.max_pages,
            self.page_size,
            self.ckv_stride_page,
            self.ckv_stride_n,
            KIMI_K2_MLA_KV_LORA_RANK,
        )
    }

    pub fn required_kpe_len(&self) -> Result<usize> {
        required_cache_len(
            self.max_pages,
            self.page_size,
            self.kpe_stride_page,
            self.kpe_stride_n,
            KIMI_K2_MLA_ROPE_DIM,
        )
    }
}

fn required_cache_len(
    max_pages: usize,
    page_size: usize,
    stride_page: usize,
    stride_n: usize,
    dim: usize,
) -> Result<usize> {
    if max_pages == 0 || page_size == 0 {
        return Ok(0);
    }
    let page_offset = (max_pages - 1)
        .checked_mul(stride_page)
        .ok_or_else(|| anyhow::anyhow!("Kimi MLA paged cache page stride overflows"))?;
    let token_offset = (page_size - 1)
        .checked_mul(stride_n)
        .ok_or_else(|| anyhow::anyhow!("Kimi MLA paged cache token stride overflows"))?;
    page_offset
        .checked_add(token_offset)
        .and_then(|offset| offset.checked_add(dim))
        .ok_or_else(|| anyhow::anyhow!("Kimi MLA paged cache length overflows"))
}

fn validate_paged_layout(
    layout: KimiMlaPagedKvLayout,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
) -> Result<()> {
    ensure!(layout.max_pages > 0, "Kimi MLA max_pages must be positive");
    ensure!(layout.page_size > 0, "Kimi MLA page_size must be positive");
    ensure!(
        layout.batch_size > 0,
        "Kimi MLA batch_size must be positive"
    );
    ensure!(
        layout.ckv_stride_n >= KIMI_K2_MLA_KV_LORA_RANK
            && layout.kpe_stride_n >= KIMI_K2_MLA_ROPE_DIM,
        "Kimi MLA cache token strides must cover ckv={} and kpe={}",
        KIMI_K2_MLA_KV_LORA_RANK,
        KIMI_K2_MLA_ROPE_DIM
    );
    ensure!(
        layout.ckv_stride_page >= layout.page_size * layout.ckv_stride_n
            && layout.kpe_stride_page >= layout.page_size * layout.kpe_stride_n,
        "Kimi MLA cache page strides must cover page_size * token_stride"
    );
    ensure!(
        page_indices_d.len() > 0,
        "Kimi MLA page_indices must contain active decode pages"
    );
    ensure!(
        page_indptr_d.len() >= layout.batch_size + 1,
        "Kimi MLA page_indptr too small: got {}, need {}",
        page_indptr_d.len(),
        layout.batch_size + 1
    );
    ensure!(
        last_page_len_d.len() >= layout.batch_size,
        "Kimi MLA last_page_len too small: got {}, need {}",
        last_page_len_d.len(),
        layout.batch_size
    );
    Ok(())
}

pub fn kimi_mla_split_qkv_a(
    ctx: &DeviceContext,
    qkv_a: &HiddenStates,
    q_a: &mut HiddenStates,
    compressed: &mut HiddenStates,
    k_rope: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        qkv_a.hidden_dim == KIMI_K2_MLA_QKV_A_OUT,
        "Kimi MLA qkv_a hidden dim must be {}, got {}",
        KIMI_K2_MLA_QKV_A_OUT,
        qkv_a.hidden_dim
    );
    ensure!(
        q_a.hidden_dim == KIMI_K2_MLA_Q_LORA_RANK && q_a.seq_len == qkv_a.seq_len,
        "Kimi MLA q_a split shape mismatch: got [{}, {}], expected [{}, {}]",
        q_a.hidden_dim,
        q_a.seq_len,
        KIMI_K2_MLA_Q_LORA_RANK,
        qkv_a.seq_len
    );
    ensure!(
        compressed.hidden_dim == KIMI_K2_MLA_KV_LORA_RANK && compressed.seq_len == qkv_a.seq_len,
        "Kimi MLA compressed split shape mismatch: got [{}, {}], expected [{}, {}]",
        compressed.hidden_dim,
        compressed.seq_len,
        KIMI_K2_MLA_KV_LORA_RANK,
        qkv_a.seq_len
    );
    ensure!(
        k_rope.hidden_dim == KIMI_K2_MLA_ROPE_DIM && k_rope.seq_len == qkv_a.seq_len,
        "Kimi MLA k_rope split shape mismatch: got [{}, {}], expected [{}, {}]",
        k_rope.hidden_dim,
        k_rope.seq_len,
        KIMI_K2_MLA_ROPE_DIM,
        qkv_a.seq_len
    );

    let (qkv_a_ptr, _qkv_a_guard) = qkv_a.data.device_ptr(&ctx.stream);
    let (q_a_ptr, _q_a_guard) = q_a.data.device_ptr_mut(&ctx.stream);
    let (compressed_ptr, _compressed_guard) = compressed.data.device_ptr_mut(&ctx.stream);
    let (k_rope_ptr, _k_rope_guard) = k_rope.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_split_qkv_a_cuda(
            qkv_a_ptr as *const ffi::Half,
            q_a_ptr as *mut ffi::Half,
            compressed_ptr as *mut ffi::Half,
            k_rope_ptr as *mut ffi::Half,
            qkv_a.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_mla_rope_assemble_prefill(
    ctx: &DeviceContext,
    q_proj: &HiddenStates,
    k_rope: &HiddenStates,
    kv_b: &HiddenStates,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    q_attn: &mut HiddenStates,
    k_cache: &mut CudaSlice<half::bf16>,
    v_cache: &mut CudaSlice<half::bf16>,
) -> Result<()> {
    let seq_len = q_proj.seq_len;
    ensure!(seq_len > 0, "Kimi MLA seq_len must be positive");
    ensure!(
        q_proj.hidden_dim == KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        "Kimi MLA q local hidden dim must be {}, got {}",
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        q_proj.hidden_dim
    );
    ensure!(
        q_attn.hidden_dim == q_proj.hidden_dim && q_attn.seq_len == seq_len,
        "Kimi MLA q_attn shape mismatch: got [{}, {}], expected [{}, {}]",
        q_attn.hidden_dim,
        q_attn.seq_len,
        q_proj.hidden_dim,
        seq_len
    );
    ensure!(
        k_rope.hidden_dim == KIMI_K2_MLA_Q_HEAD_DIM - KIMI_K2_MLA_V_HEAD_DIM
            && k_rope.seq_len == seq_len,
        "Kimi MLA k_rope shape mismatch"
    );
    ensure!(
        kv_b.hidden_dim == KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8 && kv_b.seq_len == seq_len,
        "Kimi MLA kv_b shape mismatch: got [{}, {}], expected [{}, {}]",
        kv_b.hidden_dim,
        kv_b.seq_len,
        KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        seq_len
    );
    let rope_elems = seq_len * (KIMI_K2_MLA_Q_HEAD_DIM - KIMI_K2_MLA_V_HEAD_DIM);
    ensure!(
        cos.len() >= rope_elems && sin.len() >= rope_elems,
        "Kimi MLA RoPE cache too small: cos={}, sin={}, need {}",
        cos.len(),
        sin.len(),
        rope_elems
    );
    ensure!(
        k_cache.len() >= seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_Q_HEAD_DIM,
        "Kimi MLA k_cache too small"
    );
    ensure!(
        v_cache.len() >= seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_V_HEAD_DIM,
        "Kimi MLA v_cache too small"
    );

    let (q_ptr, _q_guard) = q_proj.data.device_ptr(&ctx.stream);
    let (k_rope_ptr, _k_rope_guard) = k_rope.data.device_ptr(&ctx.stream);
    let (kv_b_ptr, _kv_b_guard) = kv_b.data.device_ptr(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.device_ptr(&ctx.stream);
    let (q_attn_ptr, _q_attn_guard) = q_attn.data.device_ptr_mut(&ctx.stream);
    let (k_cache_ptr, _k_cache_guard) = k_cache.device_ptr_mut(&ctx.stream);
    let (v_cache_ptr, _v_cache_guard) = v_cache.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_rope_assemble_prefill_cuda(
            q_ptr as *const ffi::Half,
            k_rope_ptr as *const ffi::Half,
            kv_b_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            q_attn_ptr as *mut ffi::Half,
            k_cache_ptr as *mut ffi::Half,
            v_cache_ptr as *mut ffi::Half,
            seq_len as i32,
            KIMI_K2_MLA_LOCAL_HEADS_TP8 as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kimi_mla_rope_split_decode(
    ctx: &DeviceContext,
    q_proj: &HiddenStates,
    k_rope: &HiddenStates,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    positions_d: &CudaSlice<i32>,
    q_nope: &mut HiddenStates,
    q_pe: &mut HiddenStates,
    append_kpe: &mut HiddenStates,
) -> Result<()> {
    let batch_size = q_proj.seq_len;
    ensure!(batch_size > 0, "Kimi MLA decode batch must be positive");
    ensure!(
        q_proj.hidden_dim == KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        "Kimi MLA q_proj hidden dim must be {}, got {}",
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        q_proj.hidden_dim
    );
    ensure!(
        k_rope.hidden_dim == KIMI_K2_MLA_ROPE_DIM && k_rope.seq_len == batch_size,
        "Kimi MLA decode k_rope shape mismatch: got [{}, {}], expected [{}, {}]",
        k_rope.hidden_dim,
        k_rope.seq_len,
        KIMI_K2_MLA_ROPE_DIM,
        batch_size
    );
    ensure!(
        q_nope.hidden_dim == KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_NOPE_DIM
            && q_nope.seq_len == batch_size,
        "Kimi MLA q_nope shape mismatch: got [{}, {}], expected [{}, {}]",
        q_nope.hidden_dim,
        q_nope.seq_len,
        KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_NOPE_DIM,
        batch_size
    );
    ensure!(
        q_pe.hidden_dim == KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8 && q_pe.seq_len == batch_size,
        "Kimi MLA q_pe shape mismatch: got [{}, {}], expected [{}, {}]",
        q_pe.hidden_dim,
        q_pe.seq_len,
        KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8,
        batch_size
    );
    ensure!(
        append_kpe.hidden_dim == KIMI_K2_MLA_ROPE_DIM && append_kpe.seq_len == batch_size,
        "Kimi MLA append_kpe shape mismatch: got [{}, {}], expected [{}, {}]",
        append_kpe.hidden_dim,
        append_kpe.seq_len,
        KIMI_K2_MLA_ROPE_DIM,
        batch_size
    );
    ensure!(
        positions_d.len() >= batch_size,
        "Kimi MLA positions too small: got {}, need {}",
        positions_d.len(),
        batch_size
    );
    ensure!(
        cos.len() > 0 && cos.len() == sin.len(),
        "Kimi MLA RoPE cos/sin cache must be non-empty and same length"
    );

    let (q_proj_ptr, _q_proj_guard) = q_proj.data.device_ptr(&ctx.stream);
    let (k_rope_ptr, _k_rope_guard) = k_rope.data.device_ptr(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.device_ptr(&ctx.stream);
    let (positions_ptr, _positions_guard) = positions_d.device_ptr(&ctx.stream);
    let (q_nope_ptr, _q_nope_guard) = q_nope.data.device_ptr_mut(&ctx.stream);
    let (q_pe_ptr, _q_pe_guard) = q_pe.data.device_ptr_mut(&ctx.stream);
    let (append_kpe_ptr, _append_kpe_guard) = append_kpe.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::kimi_mla_rope_split_decode_cuda(
            q_proj_ptr as *const ffi::Half,
            k_rope_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            positions_ptr as *const i32,
            q_nope_ptr as *mut ffi::Half,
            q_pe_ptr as *mut ffi::Half,
            append_kpe_ptr as *mut ffi::Half,
            batch_size as i32,
            KIMI_K2_MLA_LOCAL_HEADS_TP8 as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_mla_rope_apply_kpe(
    ctx: &DeviceContext,
    k_rope: &HiddenStates,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    positions_d: &CudaSlice<i32>,
    append_kpe: &mut HiddenStates,
) -> Result<()> {
    let seq_len = k_rope.seq_len;
    ensure!(seq_len > 0, "Kimi MLA prefill KPE seq_len must be positive");
    ensure!(
        k_rope.hidden_dim == KIMI_K2_MLA_ROPE_DIM,
        "Kimi MLA prefill k_rope hidden dim must be {}, got {}",
        KIMI_K2_MLA_ROPE_DIM,
        k_rope.hidden_dim
    );
    ensure!(
        append_kpe.hidden_dim == KIMI_K2_MLA_ROPE_DIM && append_kpe.seq_len == seq_len,
        "Kimi MLA prefill append_kpe shape mismatch: got [{}, {}], expected [{}, {}]",
        append_kpe.hidden_dim,
        append_kpe.seq_len,
        KIMI_K2_MLA_ROPE_DIM,
        seq_len
    );
    ensure!(
        positions_d.len() >= seq_len,
        "Kimi MLA prefill positions too small: got {}, need {}",
        positions_d.len(),
        seq_len
    );
    ensure!(
        cos.len() >= seq_len * KIMI_K2_MLA_ROPE_DIM && cos.len() == sin.len(),
        "Kimi MLA prefill RoPE cache too small: cos={}, sin={}, need {}",
        cos.len(),
        sin.len(),
        seq_len * KIMI_K2_MLA_ROPE_DIM
    );

    let (k_rope_ptr, _k_rope_guard) = k_rope.data.device_ptr(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.device_ptr(&ctx.stream);
    let (positions_ptr, _positions_guard) = positions_d.device_ptr(&ctx.stream);
    let (append_kpe_ptr, _append_kpe_guard) = append_kpe.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::kimi_mla_rope_apply_kpe_cuda(
            k_rope_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            positions_ptr as *const i32,
            append_kpe_ptr as *mut ffi::Half,
            seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_flashinfer_single_prefill_mla(
    ctx: &DeviceContext,
    q_attn: &HiddenStates,
    k_cache: &CudaSlice<half::bf16>,
    v_cache: &CudaSlice<half::bf16>,
    output: &mut HiddenStates,
    sm_scale: f32,
) -> Result<()> {
    let seq_len = q_attn.seq_len;
    ensure!(
        q_attn.hidden_dim == KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        "Kimi MLA q_attn hidden dim must be {}, got {}",
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        q_attn.hidden_dim
    );
    ensure!(
        output.hidden_dim == KIMI_K2_MLA_O_LOCAL_IN_TP8 && output.seq_len == seq_len,
        "Kimi MLA output shape mismatch: got [{}, {}], expected [{}, {}]",
        output.hidden_dim,
        output.seq_len,
        KIMI_K2_MLA_O_LOCAL_IN_TP8,
        seq_len
    );
    ensure!(
        k_cache.len() >= seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_Q_HEAD_DIM,
        "Kimi MLA k_cache too small"
    );
    ensure!(
        v_cache.len() >= seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_V_HEAD_DIM,
        "Kimi MLA v_cache too small"
    );

    let (q_ptr, _q_guard) = q_attn.data.device_ptr(&ctx.stream);
    let (k_ptr, _k_guard) = k_cache.device_ptr(&ctx.stream);
    let (v_ptr, _v_guard) = v_cache.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_flashinfer_single_prefill_mla_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            KIMI_K2_MLA_LOCAL_HEADS_TP8 as i32,
            seq_len as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!("kimi_flashinfer_single_prefill_mla_cuda failed with cudaError={result}");
    }
    Ok(())
}

pub fn kimi_mla_absorb_q_nope(
    ctx: &DeviceContext,
    kv_b_proj: &DeviceMatrix,
    q_nope: &HiddenStates,
    q_abs_nope: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        kv_b_proj.rows == KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8
            && kv_b_proj.cols == KIMI_K2_MLA_KV_LORA_RANK,
        "Kimi MLA kv_b_proj shape mismatch: got [{}, {}], expected [{}, {}]",
        kv_b_proj.rows,
        kv_b_proj.cols,
        KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        KIMI_K2_MLA_KV_LORA_RANK
    );
    ensure!(q_nope.seq_len > 0, "Kimi MLA q_nope batch must be positive");
    ensure!(
        q_nope.hidden_dim == KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_NOPE_DIM,
        "Kimi MLA q_nope hidden dim must be {}, got {}",
        KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_NOPE_DIM,
        q_nope.hidden_dim
    );
    ensure!(
        q_abs_nope.hidden_dim == KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8
            && q_abs_nope.seq_len == q_nope.seq_len,
        "Kimi MLA q_abs_nope shape mismatch: got [{}, {}], expected [{}, {}]",
        q_abs_nope.hidden_dim,
        q_abs_nope.seq_len,
        KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8,
        q_nope.seq_len
    );

    let (weight_ptr, _weight_guard) = kv_b_proj.data.device_ptr(&ctx.stream);
    let (q_ptr, _q_guard) = q_nope.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = q_abs_nope.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_absorb_q_nope_cuda(
            weight_ptr as *const ffi::Half,
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            q_nope.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!("kimi_mla_absorb_q_nope_cuda failed with cudaError={result}");
    }
    Ok(())
}

pub fn kimi_mla_v_up(
    ctx: &DeviceContext,
    kv_b_proj: &DeviceMatrix,
    latent: &HiddenStates,
    output: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        kv_b_proj.rows == KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8
            && kv_b_proj.cols == KIMI_K2_MLA_KV_LORA_RANK,
        "Kimi MLA kv_b_proj shape mismatch: got [{}, {}], expected [{}, {}]",
        kv_b_proj.rows,
        kv_b_proj.cols,
        KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        KIMI_K2_MLA_KV_LORA_RANK
    );
    ensure!(latent.seq_len > 0, "Kimi MLA latent batch must be positive");
    ensure!(
        latent.hidden_dim == KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8,
        "Kimi MLA latent hidden dim must be {}, got {}",
        KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8,
        latent.hidden_dim
    );
    ensure!(
        output.hidden_dim == KIMI_K2_MLA_O_LOCAL_IN_TP8 && output.seq_len == latent.seq_len,
        "Kimi MLA v-up output shape mismatch: got [{}, {}], expected [{}, {}]",
        output.hidden_dim,
        output.seq_len,
        KIMI_K2_MLA_O_LOCAL_IN_TP8,
        latent.seq_len
    );

    let (weight_ptr, _weight_guard) = kv_b_proj.data.device_ptr(&ctx.stream);
    let (latent_ptr, _latent_guard) = latent.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_v_up_cuda(
            weight_ptr as *const ffi::Half,
            latent_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            latent.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!("kimi_mla_v_up_cuda failed with cudaError={result}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kimi_mla_paged_kv_append(
    ctx: &DeviceContext,
    ckv_cache: &mut CudaSlice<half::bf16>,
    kpe_cache: &mut CudaSlice<half::bf16>,
    layout: KimiMlaPagedKvLayout,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    append_ckv: &HiddenStates,
    append_kpe: &HiddenStates,
    batch_indices_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
) -> Result<()> {
    validate_paged_layout(layout, page_indices_d, page_indptr_d, last_page_len_d)?;
    ensure!(
        ckv_cache.len() >= layout.required_ckv_len()?,
        "Kimi MLA ckv_cache too small: got {}, need {}",
        ckv_cache.len(),
        layout.required_ckv_len()?
    );
    ensure!(
        kpe_cache.len() >= layout.required_kpe_len()?,
        "Kimi MLA kpe_cache too small: got {}, need {}",
        kpe_cache.len(),
        layout.required_kpe_len()?
    );
    ensure!(
        append_ckv.hidden_dim == KIMI_K2_MLA_KV_LORA_RANK,
        "Kimi MLA append_ckv hidden dim must be {}, got {}",
        KIMI_K2_MLA_KV_LORA_RANK,
        append_ckv.hidden_dim
    );
    ensure!(
        append_kpe.hidden_dim == KIMI_K2_MLA_ROPE_DIM && append_kpe.seq_len == append_ckv.seq_len,
        "Kimi MLA append_kpe shape mismatch: got [{}, {}], expected [{}, {}]",
        append_kpe.hidden_dim,
        append_kpe.seq_len,
        KIMI_K2_MLA_ROPE_DIM,
        append_ckv.seq_len
    );
    ensure!(
        batch_indices_d.len() >= append_ckv.seq_len && positions_d.len() >= append_ckv.seq_len,
        "Kimi MLA append metadata too small for nnz={}",
        append_ckv.seq_len
    );

    let (ckv_cache_ptr, _ckv_cache_guard) = ckv_cache.device_ptr_mut(&ctx.stream);
    let (kpe_cache_ptr, _kpe_cache_guard) = kpe_cache.device_ptr_mut(&ctx.stream);
    let (page_indices_ptr, _page_indices_guard) = page_indices_d.device_ptr(&ctx.stream);
    let (page_indptr_ptr, _page_indptr_guard) = page_indptr_d.device_ptr(&ctx.stream);
    let (last_page_len_ptr, _last_page_len_guard) = last_page_len_d.device_ptr(&ctx.stream);
    let (append_ckv_ptr, _append_ckv_guard) = append_ckv.data.device_ptr(&ctx.stream);
    let (append_kpe_ptr, _append_kpe_guard) = append_kpe.data.device_ptr(&ctx.stream);
    let (batch_indices_ptr, _batch_indices_guard) = batch_indices_d.device_ptr(&ctx.stream);
    let (positions_ptr, _positions_guard) = positions_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::kimi_mla_paged_kv_append_cuda(
            ckv_cache_ptr as *mut ffi::Half,
            kpe_cache_ptr as *mut ffi::Half,
            page_indices_ptr as *const i32,
            page_indptr_ptr as *const i32,
            last_page_len_ptr as *const i32,
            append_ckv_ptr as *const ffi::Half,
            append_kpe_ptr as *const ffi::Half,
            batch_indices_ptr as *const i32,
            positions_ptr as *const i32,
            append_ckv.seq_len as i32,
            layout.ckv_stride_page as i64,
            layout.ckv_stride_n as i64,
            layout.kpe_stride_page as i64,
            layout.kpe_stride_n as i64,
            layout.page_size as i32,
            layout.batch_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!("kimi_mla_paged_kv_append_cuda failed with cudaError={result}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kimi_flashinfer_batch_decode_mla(
    ctx: &DeviceContext,
    q_abs_nope: &HiddenStates,
    q_pe: &HiddenStates,
    output: &mut HiddenStates,
    ckv_cache: &CudaSlice<half::bf16>,
    kpe_cache: &CudaSlice<half::bf16>,
    layout: KimiMlaPagedKvLayout,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    sm_scale: f32,
) -> Result<()> {
    validate_paged_layout(layout, page_indices_d, page_indptr_d, last_page_len_d)?;
    ensure!(
        ckv_cache.len() >= layout.required_ckv_len()?,
        "Kimi MLA ckv_cache too small: got {}, need {}",
        ckv_cache.len(),
        layout.required_ckv_len()?
    );
    ensure!(
        kpe_cache.len() >= layout.required_kpe_len()?,
        "Kimi MLA kpe_cache too small: got {}, need {}",
        kpe_cache.len(),
        layout.required_kpe_len()?
    );
    ensure!(
        q_abs_nope.hidden_dim == KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8
            && q_abs_nope.seq_len == layout.batch_size,
        "Kimi MLA q_abs_nope shape mismatch: got [{}, {}], expected [{}, {}]",
        q_abs_nope.hidden_dim,
        q_abs_nope.seq_len,
        KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8,
        layout.batch_size
    );
    ensure!(
        q_pe.hidden_dim == KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8 && q_pe.seq_len == layout.batch_size,
        "Kimi MLA q_pe shape mismatch: got [{}, {}], expected [{}, {}]",
        q_pe.hidden_dim,
        q_pe.seq_len,
        KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8,
        layout.batch_size
    );
    ensure!(
        output.hidden_dim == KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8 && output.seq_len == layout.batch_size,
        "Kimi MLA output shape mismatch: got [{}, {}], expected [{}, {}]",
        output.hidden_dim,
        output.seq_len,
        KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8,
        layout.batch_size
    );
    ensure!(
        request_indices_d.len() >= layout.batch_size
            && kv_tile_indices_d.len() >= layout.batch_size
            && kv_chunk_size_d.len() >= layout.batch_size,
        "Kimi MLA decode plan metadata too small for batch_size={}",
        layout.batch_size
    );

    let (q_abs_nope_ptr, _q_abs_nope_guard) = q_abs_nope.data.device_ptr(&ctx.stream);
    let (q_pe_ptr, _q_pe_guard) = q_pe.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.data.device_ptr_mut(&ctx.stream);
    let (ckv_cache_ptr, _ckv_cache_guard) = ckv_cache.device_ptr(&ctx.stream);
    let (kpe_cache_ptr, _kpe_cache_guard) = kpe_cache.device_ptr(&ctx.stream);
    let (page_indices_ptr, _page_indices_guard) = page_indices_d.device_ptr(&ctx.stream);
    let (page_indptr_ptr, _page_indptr_guard) = page_indptr_d.device_ptr(&ctx.stream);
    let (last_page_len_ptr, _last_page_len_guard) = last_page_len_d.device_ptr(&ctx.stream);
    let (request_indices_ptr, _request_indices_guard) = request_indices_d.device_ptr(&ctx.stream);
    let (kv_tile_indices_ptr, _kv_tile_indices_guard) = kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kv_chunk_size_ptr, _kv_chunk_size_guard) = kv_chunk_size_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::kimi_flashinfer_batch_decode_mla_cuda(
            q_abs_nope_ptr as *const ffi::Half,
            q_pe_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            ckv_cache_ptr as *const ffi::Half,
            kpe_cache_ptr as *const ffi::Half,
            page_indices_ptr as *const i32,
            page_indptr_ptr as *const i32,
            last_page_len_ptr as *const i32,
            request_indices_ptr as *const i32,
            kv_tile_indices_ptr as *const i32,
            kv_chunk_size_ptr as *const i32,
            KIMI_K2_MLA_LOCAL_HEADS_TP8 as i32,
            layout.ckv_stride_page as i64,
            layout.ckv_stride_n as i64,
            layout.kpe_stride_page as i64,
            layout.kpe_stride_n as i64,
            layout.page_size as i32,
            layout.batch_size as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!("kimi_flashinfer_batch_decode_mla_cuda failed with cudaError={result}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "H20-only: validates FlashInfer MLA decode wrapper and paged compressed KV append"]
    fn h20_kimi_flashinfer_batch_decode_mla_bs4_smoke() {
        let ctx = DeviceContext::new().expect("CUDA context");
        let batch_size = 4usize;
        let page_size = 4usize;
        let max_pages = 4usize;
        let layout = KimiMlaPagedKvLayout::separate_contiguous(max_pages, page_size, batch_size);
        let heads = KIMI_K2_MLA_LOCAL_HEADS_TP8;
        let q_nope_hidden = heads * KIMI_K2_MLA_NOPE_DIM;
        let q_abs_hidden = KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8;
        let q_pe_hidden = KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8;
        let attn_out_hidden = KIMI_K2_MLA_O_LOCAL_IN_TP8;
        let seq_lens = [1usize, 2, 3, 4];
        let nnz = batch_size;

        let mut ckv_cache = ctx
            .stream
            .alloc_zeros::<half::bf16>(layout.required_ckv_len().expect("ckv len"))
            .expect("ckv cache");
        let mut kpe_cache = ctx
            .stream
            .alloc_zeros::<half::bf16>(layout.required_kpe_len().expect("kpe len"))
            .expect("kpe cache");

        let page_indices_d = ctx
            .stream
            .clone_htod(&[0i32, 1, 2, 3])
            .expect("page indices");
        let page_indptr_d = ctx
            .stream
            .clone_htod(&[0i32, 1, 2, 3, 4])
            .expect("page indptr");
        let last_page_len_d = ctx
            .stream
            .clone_htod(&[1i32, 2, 3, 4])
            .expect("last page len");

        let batch_indices = (0..batch_size)
            .map(|batch| batch as i32)
            .collect::<Vec<_>>();
        let positions = seq_lens
            .iter()
            .map(|seq_len| (seq_len - 1) as i32)
            .collect::<Vec<_>>();
        let batch_indices_d = ctx
            .stream
            .clone_htod(&batch_indices)
            .expect("batch indices");
        let positions_d = ctx.stream.clone_htod(&positions).expect("positions");

        let append_ckv_host = (0..nnz * KIMI_K2_MLA_KV_LORA_RANK)
            .map(|idx| {
                let value = ((idx % 127) as f32 - 63.0) * 0.0017;
                half::bf16::from_f32(value)
            })
            .collect::<Vec<_>>();
        let mut append_ckv =
            HiddenStates::zeros(&ctx, KIMI_K2_MLA_KV_LORA_RANK, nnz).expect("append ckv");
        ctx.stream
            .memcpy_htod(&append_ckv_host, &mut append_ckv.data)
            .expect("append ckv H2D");

        let kv_b_proj_host = (0..KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8 * KIMI_K2_MLA_KV_LORA_RANK)
            .map(|idx| {
                let value = ((idx % 131) as f32 - 65.0) * 0.0009;
                half::bf16::from_f32(value)
            })
            .collect::<Vec<_>>();
        let kv_b_proj = DeviceMatrix::from_host(
            &ctx,
            &kv_b_proj_host,
            KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
            KIMI_K2_MLA_KV_LORA_RANK,
        )
        .expect("kv_b_proj");

        let q_proj_host = (0..batch_size * KIMI_K2_MLA_Q_LOCAL_OUT_TP8)
            .map(|idx| {
                let value = ((idx % 113) as f32 - 56.0) * 0.0013;
                half::bf16::from_f32(value)
            })
            .collect::<Vec<_>>();
        let k_rope_host = (0..batch_size * KIMI_K2_MLA_ROPE_DIM)
            .map(|idx| {
                let value = ((idx % 67) as f32 - 33.0) * 0.0019;
                half::bf16::from_f32(value)
            })
            .collect::<Vec<_>>();
        let rope_elems = page_size * KIMI_K2_MLA_ROPE_DIM;
        let cos_host = vec![half::bf16::from_f32(1.0); rope_elems];
        let sin_host = vec![half::bf16::from_f32(0.0); rope_elems];
        let mut q_proj =
            HiddenStates::zeros(&ctx, KIMI_K2_MLA_Q_LOCAL_OUT_TP8, batch_size).expect("q_proj");
        let mut k_rope =
            HiddenStates::zeros(&ctx, KIMI_K2_MLA_ROPE_DIM, batch_size).expect("k_rope");
        ctx.stream
            .memcpy_htod(&q_proj_host, &mut q_proj.data)
            .expect("q_proj H2D");
        ctx.stream
            .memcpy_htod(&k_rope_host, &mut k_rope.data)
            .expect("k_rope H2D");
        let cos_d = ctx.stream.clone_htod(&cos_host).expect("cos H2D");
        let sin_d = ctx.stream.clone_htod(&sin_host).expect("sin H2D");
        let mut q_nope = HiddenStates::zeros(&ctx, q_nope_hidden, batch_size).expect("q_nope");
        let mut q_pe = HiddenStates::zeros(&ctx, q_pe_hidden, batch_size).expect("q_pe");
        let mut append_kpe =
            HiddenStates::zeros(&ctx, KIMI_K2_MLA_ROPE_DIM, batch_size).expect("append kpe");
        kimi_mla_rope_split_decode(
            &ctx,
            &q_proj,
            &k_rope,
            &cos_d,
            &sin_d,
            &positions_d,
            &mut q_nope,
            &mut q_pe,
            &mut append_kpe,
        )
        .expect("decode q/k rope split");

        kimi_mla_paged_kv_append(
            &ctx,
            &mut ckv_cache,
            &mut kpe_cache,
            layout,
            &page_indices_d,
            &page_indptr_d,
            &last_page_len_d,
            &append_ckv,
            &append_kpe,
            &batch_indices_d,
            &positions_d,
        )
        .expect("MLA paged append");

        let mut q_abs_nope = HiddenStates::zeros(&ctx, q_abs_hidden, batch_size).expect("q_abs");
        kimi_mla_absorb_q_nope(&ctx, &kv_b_proj, &q_nope, &mut q_abs_nope).expect("q absorption");

        let request_indices_d = ctx
            .stream
            .clone_htod(&[0i32, 1, 2, 3])
            .expect("request indices");
        let kv_tile_indices_d = ctx
            .stream
            .clone_htod(&[0i32, 0, 0, 0])
            .expect("kv tile indices");
        let kv_chunk_size_d = ctx
            .stream
            .clone_htod(&[1i32, 2, 3, page_size as i32])
            .expect("kv chunk size");
        let mut latent = HiddenStates::zeros(&ctx, q_abs_hidden, batch_size).expect("latent");
        let sm_scale = 1.0f32 / ((KIMI_K2_MLA_KV_LORA_RANK + KIMI_K2_MLA_ROPE_DIM) as f32).sqrt();

        kimi_flashinfer_batch_decode_mla(
            &ctx,
            &q_abs_nope,
            &q_pe,
            &mut latent,
            &ckv_cache,
            &kpe_cache,
            layout,
            &page_indices_d,
            &page_indptr_d,
            &last_page_len_d,
            &request_indices_d,
            &kv_tile_indices_d,
            &kv_chunk_size_d,
            sm_scale,
        )
        .expect("MLA decode");

        let mut attn_out =
            HiddenStates::zeros(&ctx, attn_out_hidden, batch_size).expect("v_up out");
        kimi_mla_v_up(&ctx, &kv_b_proj, &latent, &mut attn_out).expect("v-up");

        let latent_host = ctx.stream.clone_dtoh(&latent.data).expect("latent D2H");
        let got = ctx.stream.clone_dtoh(&attn_out.data).expect("output D2H");
        ctx.sync().expect("sync");
        assert_eq!(
            latent_host.len(),
            batch_size * heads * KIMI_K2_MLA_KV_LORA_RANK
        );
        assert_eq!(got.len(), batch_size * heads * KIMI_K2_MLA_V_HEAD_DIM);
        assert!(
            latent_host.iter().all(|value| value.to_f32().is_finite()),
            "MLA decode latent output must be finite"
        );
        assert!(
            got.iter().all(|value| value.to_f32().is_finite()),
            "MLA v-up output must be finite"
        );
        assert!(
            latent_host.iter().any(|value| value.to_f32().abs() > 0.0),
            "MLA decode latent output should not be all zero"
        );
        assert!(
            got.iter().any(|value| value.to_f32().abs() > 0.0),
            "MLA v-up output should not be all zero"
        );
    }
}
