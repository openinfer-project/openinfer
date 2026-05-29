use anyhow::{Result, bail, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::{
    ffi,
    tensor::{DeviceContext, GpuTensor, GpuWeight, NormWeight},
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

pub(super) fn validate_paged_layout(
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

pub fn kimi_mla_split_qkv_a_norm(
    ctx: &DeviceContext,
    qkv_a: &GpuTensor<KIMI_K2_MLA_QKV_A_OUT>,
    q_a_weight: &NormWeight<KIMI_K2_MLA_Q_LORA_RANK>,
    ckv_weight: &NormWeight<KIMI_K2_MLA_KV_LORA_RANK>,
    q_a_normed: &mut GpuTensor<KIMI_K2_MLA_Q_LORA_RANK>,
    ckv_normed: &mut GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    k_rope: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    eps: f32,
) -> Result<()> {
    ensure!(
        q_a_normed.seq_len == qkv_a.seq_len
            && ckv_normed.seq_len == qkv_a.seq_len
            && k_rope.seq_len == qkv_a.seq_len,
        "Kimi MLA split+norm seq_len mismatch: qkv_a={}, q_a_normed={}, ckv_normed={}, k_rope={}",
        qkv_a.seq_len,
        q_a_normed.seq_len,
        ckv_normed.seq_len,
        k_rope.seq_len
    );
    let (qkv_a_ptr, _g0) = qkv_a.data.device_ptr(&ctx.stream);
    let (q_w_ptr, _g1) = q_a_weight.data.device_ptr(&ctx.stream);
    let (ckv_w_ptr, _g2) = ckv_weight.data.device_ptr(&ctx.stream);
    let (q_out_ptr, _g3) = q_a_normed.data.device_ptr_mut(&ctx.stream);
    let (ckv_out_ptr, _g4) = ckv_normed.data.device_ptr_mut(&ctx.stream);
    let (rope_ptr, _g5) = k_rope.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_split_qkv_a_norm_cuda(
            qkv_a_ptr as *const ffi::Half,
            q_w_ptr as *const ffi::Half,
            ckv_w_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            ckv_out_ptr as *mut ffi::Half,
            rope_ptr as *mut ffi::Half,
            eps,
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
            KIMI_K2_MLA_LOCAL_HEADS_TP8 as i32,
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
            KIMI_K2_MLA_LOCAL_HEADS_TP8 as i32,
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
