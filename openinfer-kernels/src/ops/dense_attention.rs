use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, HiddenStates};

#[allow(clippy::too_many_arguments)]
pub fn single_prefill_nhd_noncausal_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    out: &mut HiddenStates,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let q_dim = num_qo_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(q.hidden_dim, q_dim);
    assert_eq!(k.hidden_dim, kv_dim);
    assert_eq!(v.hidden_dim, kv_dim);
    assert_eq!(v.seq_len, k.seq_len);
    assert_eq!(out.hidden_dim, q_dim);
    assert_eq!(out.seq_len, q.seq_len);
    assert_eq!(
        head_dim, 128,
        "FlashInfer wrapper is instantiated for head_dim=128"
    );

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let status = unsafe {
        ffi::single_prefill_nhd_noncausal_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            q.seq_len as i32,
            k.seq_len as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if status != 0 {
        anyhow::bail!(
            "single_prefill_nhd_noncausal_cuda failed: status={}, q_len={}, kv_len={}, q_heads={}, kv_heads={}, head_dim={}",
            status,
            q.seq_len,
            k.seq_len,
            num_qo_heads,
            num_kv_heads,
            head_dim
        );
    }
    Ok(())
}

pub struct RaggedPrefillPlan {
    q_indptr: CudaSlice<i32>,
    kv_indptr: CudaSlice<i32>,
    request_indices: CudaSlice<i32>,
    qo_tile_indices: CudaSlice<i32>,
    kv_tile_indices: CudaSlice<i32>,
    kv_chunk_size: CudaSlice<i32>,
    total_num_rows: CudaSlice<u32>,
    batch_size: usize,
    total_q_len: usize,
}

impl RaggedPrefillPlan {
    pub fn new(
        ctx: &DeviceContext,
        q_lens: &[usize],
        kv_lens: &[usize],
        group_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(!q_lens.is_empty(), "ragged prefill batch is empty");
        anyhow::ensure!(
            q_lens.len() == kv_lens.len(),
            "q_lens len {} != kv_lens len {}",
            q_lens.len(),
            kv_lens.len()
        );
        anyhow::ensure!(group_size > 0, "group_size must be positive");
        let mut q_indptr = Vec::with_capacity(q_lens.len() + 1);
        let mut kv_indptr = Vec::with_capacity(kv_lens.len() + 1);
        q_indptr.push(0i32);
        kv_indptr.push(0i32);
        for (&q_len, &kv_len) in q_lens.iter().zip(kv_lens.iter()) {
            anyhow::ensure!(q_len > 0, "ragged prefill q_len must be positive");
            anyhow::ensure!(kv_len > 0, "ragged prefill kv_len must be positive");
            q_indptr.push(q_indptr.last().copied().unwrap() + q_len as i32);
            kv_indptr.push(kv_indptr.last().copied().unwrap() + kv_len as i32);
        }
        let total_q_len = *q_indptr.last().unwrap() as usize;
        let mut request_indices = Vec::new();
        let mut qo_tile_indices = Vec::new();
        let mut kv_tile_indices = Vec::new();
        const CTA_TILE_Q: usize = 16;
        for (req_idx, &q_len) in q_lens.iter().enumerate() {
            let packed_q_len = q_len * group_size;
            let tiles = packed_q_len.div_ceil(CTA_TILE_Q);
            for tile in 0..tiles {
                request_indices.push(req_idx as i32);
                qo_tile_indices.push(tile as i32);
                kv_tile_indices.push(0i32);
            }
        }
        let kv_chunk_size: Vec<i32> = kv_lens.iter().map(|&len| len as i32).collect();
        Ok(Self {
            q_indptr: ctx.stream.clone_htod(&q_indptr)?,
            kv_indptr: ctx.stream.clone_htod(&kv_indptr)?,
            request_indices: ctx.stream.clone_htod(&request_indices)?,
            qo_tile_indices: ctx.stream.clone_htod(&qo_tile_indices)?,
            kv_tile_indices: ctx.stream.clone_htod(&kv_tile_indices)?,
            kv_chunk_size: ctx.stream.clone_htod(&kv_chunk_size)?,
            total_num_rows: ctx.stream.clone_htod(&[total_q_len as u32])?,
            batch_size: q_lens.len(),
            total_q_len,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn batch_prefill_ragged_nhd_noncausal_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    out: &mut HiddenStates,
    plan: &RaggedPrefillPlan,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let q_dim = num_qo_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(q.hidden_dim, q_dim);
    assert_eq!(k.hidden_dim, kv_dim);
    assert_eq!(v.hidden_dim, kv_dim);
    assert_eq!(v.seq_len, k.seq_len);
    assert_eq!(out.hidden_dim, q_dim);
    assert_eq!(out.seq_len, q.seq_len);
    assert_eq!(q.seq_len, plan.total_q_len);
    assert_eq!(
        head_dim, 128,
        "FlashInfer ragged wrapper is instantiated for head_dim=128"
    );

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let (q_indptr, _) = plan.q_indptr.device_ptr(&ctx.stream);
    let (kv_indptr, _) = plan.kv_indptr.device_ptr(&ctx.stream);
    let (request_indices, _) = plan.request_indices.device_ptr(&ctx.stream);
    let (qo_tile_indices, _) = plan.qo_tile_indices.device_ptr(&ctx.stream);
    let (kv_tile_indices, _) = plan.kv_tile_indices.device_ptr(&ctx.stream);
    let (kv_chunk_size, _) = plan.kv_chunk_size.device_ptr(&ctx.stream);
    let (total_num_rows, _) = plan.total_num_rows.device_ptr(&ctx.stream);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let status = unsafe {
        ffi::batch_prefill_ragged_nhd_noncausal_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_indptr as *const i32,
            kv_indptr as *const i32,
            request_indices as *const i32,
            qo_tile_indices as *const i32,
            kv_tile_indices as *const i32,
            kv_chunk_size as *const i32,
            total_num_rows as *const u32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            q.seq_len as i32,
            plan.batch_size as i32,
            plan.request_indices.len() as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if status != 0 {
        anyhow::bail!(
            "batch_prefill_ragged_nhd_noncausal_cuda failed: status={}, total_q_len={}, batch_size={}, q_heads={}, kv_heads={}, head_dim={}",
            status,
            q.seq_len,
            plan.batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim
        );
    }
    Ok(())
}
