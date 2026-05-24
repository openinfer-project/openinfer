use anyhow::{Result, bail, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::{
    ffi,
    tensor::{DeviceContext, GpuTensor, GpuWeight},
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
    qkv_a: &GpuTensor<KIMI_K2_MLA_QKV_A_OUT>,
    q_a: &mut GpuTensor<KIMI_K2_MLA_Q_LORA_RANK>,
    compressed: &mut GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    k_rope: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
) -> Result<()> {
    ensure!(
        q_a.seq_len == qkv_a.seq_len
            && compressed.seq_len == qkv_a.seq_len
            && k_rope.seq_len == qkv_a.seq_len,
        "Kimi MLA split seq_len mismatch: qkv_a={}, q_a={}, compressed={}, k_rope={}",
        qkv_a.seq_len,
        q_a.seq_len,
        compressed.seq_len,
        k_rope.seq_len
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
    q_proj: &GpuTensor<KIMI_K2_MLA_Q_LOCAL_OUT_TP8>,
    k_rope: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    kv_b: &GpuTensor<KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8>,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    q_attn: &mut GpuTensor<KIMI_K2_MLA_Q_LOCAL_OUT_TP8>,
    k_cache: &mut CudaSlice<half::bf16>,
    v_cache: &mut CudaSlice<half::bf16>,
) -> Result<()> {
    let seq_len = q_proj.seq_len;
    ensure!(seq_len > 0, "Kimi MLA seq_len must be positive");
    ensure!(
        k_rope.seq_len == seq_len && kv_b.seq_len == seq_len && q_attn.seq_len == seq_len,
        "Kimi MLA prefill assemble seq_len mismatch: q_proj={}, k_rope={}, kv_b={}, q_attn={}",
        q_proj.seq_len,
        k_rope.seq_len,
        kv_b.seq_len,
        q_attn.seq_len
    );
    let rope_elems = seq_len * KIMI_K2_MLA_ROPE_DIM;
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

pub const KIMI_K2_MLA_Q_NOPE_LOCAL_OUT_TP8: usize =
    KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_NOPE_DIM;

#[allow(clippy::too_many_arguments)]
pub fn kimi_mla_rope_split_decode(
    ctx: &DeviceContext,
    q_proj: &GpuTensor<KIMI_K2_MLA_Q_LOCAL_OUT_TP8>,
    k_rope: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    positions_d: &CudaSlice<i32>,
    q_nope: &mut GpuTensor<KIMI_K2_MLA_Q_NOPE_LOCAL_OUT_TP8>,
    q_pe: &mut GpuTensor<KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8>,
    append_kpe: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
) -> Result<()> {
    let batch_size = q_proj.seq_len;
    ensure!(
        k_rope.seq_len == batch_size
            && q_nope.seq_len == batch_size
            && q_pe.seq_len == batch_size
            && append_kpe.seq_len == batch_size,
        "Kimi MLA decode RoPE split seq_len mismatch: q_proj={}, k_rope={}, q_nope={}, q_pe={}, append_kpe={}",
        q_proj.seq_len,
        k_rope.seq_len,
        q_nope.seq_len,
        q_pe.seq_len,
        append_kpe.seq_len
    );
    ensure!(
        cos.len() >= KIMI_K2_MLA_ROPE_DIM && sin.len() >= KIMI_K2_MLA_ROPE_DIM,
        "Kimi MLA decode RoPE cache too small: cos={}, sin={}, need at least {}",
        cos.len(),
        sin.len(),
        KIMI_K2_MLA_ROPE_DIM
    );
    ensure!(
        positions_d.len() >= batch_size,
        "Kimi MLA decode positions too small: got {}, need {}",
        positions_d.len(),
        batch_size
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
    k_rope: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    positions_d: &CudaSlice<i32>,
    append_kpe: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
) -> Result<()> {
    let seq_len = k_rope.seq_len;
    ensure!(
        append_kpe.seq_len == seq_len,
        "Kimi MLA apply KPE seq_len mismatch: k_rope={}, append_kpe={}",
        k_rope.seq_len,
        append_kpe.seq_len
    );
    ensure!(
        cos.len() >= seq_len * KIMI_K2_MLA_ROPE_DIM && sin.len() >= seq_len * KIMI_K2_MLA_ROPE_DIM,
        "Kimi MLA apply KPE RoPE cache too small: cos={}, sin={}, need {}",
        cos.len(),
        sin.len(),
        seq_len * KIMI_K2_MLA_ROPE_DIM
    );
    ensure!(
        positions_d.len() >= seq_len,
        "Kimi MLA apply KPE positions too small: got {}, need {}",
        positions_d.len(),
        seq_len
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
    q_attn: &GpuTensor<KIMI_K2_MLA_Q_LOCAL_OUT_TP8>,
    k_cache: &CudaSlice<half::bf16>,
    v_cache: &CudaSlice<half::bf16>,
    output: &mut GpuTensor<KIMI_K2_MLA_O_LOCAL_IN_TP8>,
    sm_scale: f32,
) -> Result<()> {
    let seq_len = q_attn.seq_len;
    ensure!(
        output.seq_len == seq_len,
        "Kimi MLA single prefill output seq_len mismatch: q_attn={}, output={}",
        q_attn.seq_len,
        output.seq_len
    );
    ensure!(
        k_cache.len() >= seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_Q_HEAD_DIM,
        "Kimi MLA single prefill k_cache too small"
    );
    ensure!(
        v_cache.len() >= seq_len * KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_V_HEAD_DIM,
        "Kimi MLA single prefill v_cache too small"
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
    kv_b_proj: &GpuWeight<KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK>,
    q_nope: &GpuTensor<KIMI_K2_MLA_Q_NOPE_LOCAL_OUT_TP8>,
    q_abs_nope: &mut GpuTensor<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>,
) -> Result<()> {
    ensure!(
        q_abs_nope.seq_len == q_nope.seq_len,
        "Kimi MLA absorb q seq_len mismatch: q_nope={}, q_abs_nope={}",
        q_nope.seq_len,
        q_abs_nope.seq_len
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
    kv_b_proj: &GpuWeight<KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK>,
    latent: &GpuTensor<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>,
    output: &mut GpuTensor<KIMI_K2_MLA_O_LOCAL_IN_TP8>,
) -> Result<()> {
    ensure!(
        output.seq_len == latent.seq_len,
        "Kimi MLA v_up seq_len mismatch: latent={}, output={}",
        latent.seq_len,
        output.seq_len
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
    append_ckv: &GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    append_kpe: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
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
        batch_indices_d.len() >= append_ckv.seq_len && positions_d.len() >= append_ckv.seq_len,
        "Kimi MLA append metadata too small for nnz={}",
        append_ckv.seq_len
    );
    ensure!(
        append_kpe.seq_len == append_ckv.seq_len,
        "Kimi MLA append seq_len mismatch: append_ckv={}, append_kpe={}",
        append_ckv.seq_len,
        append_kpe.seq_len
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
    q_abs_nope: &GpuTensor<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>,
    q_pe: &GpuTensor<KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8>,
    output: &mut GpuTensor<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>,
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
        request_indices_d.len() >= layout.batch_size
            && kv_tile_indices_d.len() >= layout.batch_size
            && kv_chunk_size_d.len() >= layout.batch_size,
        "Kimi MLA decode plan metadata too small for batch_size={}",
        layout.batch_size
    );
    ensure!(
        q_abs_nope.seq_len == layout.batch_size
            && q_pe.seq_len == layout.batch_size
            && output.seq_len == layout.batch_size,
        "Kimi MLA batch decode seq_len must match layout batch_size {}: q_abs_nope={}, q_pe={}, output={}",
        layout.batch_size,
        q_abs_nope.seq_len,
        q_pe.seq_len,
        output.seq_len
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
            GpuTensor::<KIMI_K2_MLA_KV_LORA_RANK>::zeros(&ctx, nnz).expect("append ckv");
        ctx.stream
            .memcpy_htod(&append_ckv_host, &mut append_ckv.data)
            .expect("append ckv H2D");

        let kv_b_proj_host = (0..KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8 * KIMI_K2_MLA_KV_LORA_RANK)
            .map(|idx| {
                let value = ((idx % 131) as f32 - 65.0) * 0.0009;
                half::bf16::from_f32(value)
            })
            .collect::<Vec<_>>();
        let kv_b_proj = crate::tensor::DeviceMatrix::from_host(
            &ctx,
            &kv_b_proj_host,
            KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
            KIMI_K2_MLA_KV_LORA_RANK,
        )
        .expect("kv_b_proj");
        let kv_b_proj = GpuWeight::from_device_matrix(kv_b_proj).expect("typed kv_b_proj");

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
            GpuTensor::<KIMI_K2_MLA_Q_LOCAL_OUT_TP8>::zeros(&ctx, batch_size).expect("q_proj");
        let mut k_rope =
            GpuTensor::<KIMI_K2_MLA_ROPE_DIM>::zeros(&ctx, batch_size).expect("k_rope");
        ctx.stream
            .memcpy_htod(&q_proj_host, &mut q_proj.data)
            .expect("q_proj H2D");
        ctx.stream
            .memcpy_htod(&k_rope_host, &mut k_rope.data)
            .expect("k_rope H2D");
        let cos_d = ctx.stream.clone_htod(&cos_host).expect("cos H2D");
        let sin_d = ctx.stream.clone_htod(&sin_host).expect("sin H2D");
        let mut q_nope =
            GpuTensor::<KIMI_K2_MLA_Q_NOPE_LOCAL_OUT_TP8>::zeros(&ctx, batch_size).expect("q_nope");
        let mut q_pe =
            GpuTensor::<KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8>::zeros(&ctx, batch_size).expect("q_pe");
        let mut append_kpe =
            GpuTensor::<KIMI_K2_MLA_ROPE_DIM>::zeros(&ctx, batch_size).expect("append kpe");
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

        let mut q_abs_nope =
            GpuTensor::<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>::zeros(&ctx, batch_size).expect("q_abs");
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
        let mut latent =
            GpuTensor::<KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8>::zeros(&ctx, batch_size).expect("latent");
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
            GpuTensor::<KIMI_K2_MLA_O_LOCAL_IN_TP8>::zeros(&ctx, batch_size).expect("v_up out");
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
