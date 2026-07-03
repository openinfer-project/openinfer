//! GLM5.2 MoE local route/scatter/combine glue for bs=1 decode (EP1).
//!
//! These three small kernels bracket the grouped FP8 GEMM expert FFN. They build
//! the grouped expert-major row layout from the router's top-k expert ids, fan the
//! single quantized token row out into each selected expert's slot, and fold the
//! per-expert outputs back into one routed vector. The route weight is applied
//! inside the weighted SwiGLU quant between the two GEMMs, so combine is a plain
//! sum. See `csrc/glm52/glm52_moe_route.cu`.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Build the grouped-GEMM `expert_offsets[n_experts + 1]` (i64) directly from the
/// router's top-k expert ids. Each expert's block starts at `align_up(running,
/// alignment)`; for bs=1 every selected (distinct) expert owns exactly one row, so
/// `expert_offsets[n_experts]` (the spanned row count) is at most `topk *
/// alignment` — the m_capacity the caller must allocate.
pub fn glm52_moe_route_offsets_launch(
    ctx: &DeviceContext,
    n_experts: usize,
    topk: usize,
    alignment: usize,
    topk_idx: &CudaSlice<i32>,
    expert_offsets: &mut CudaSlice<i64>,
) -> Result<()> {
    ensure!(
        n_experts > 0 && topk > 0 && alignment > 0,
        "GLM5.2 MoE route offsets needs positive n_experts/topk/alignment, got {n_experts}/{topk}/{alignment}"
    );
    ensure!(
        topk_idx.len() >= topk,
        "GLM5.2 MoE route offsets topk_idx too small: have {}, need {topk}",
        topk_idx.len()
    );
    ensure!(
        expert_offsets.len() > n_experts,
        "GLM5.2 MoE route offsets expert_offsets too small: have {}, need {}",
        expert_offsets.len(),
        n_experts + 1
    );
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_moe_route_offsets_cuda(
            idx_ptr as *const i32,
            offsets_ptr as *mut i64,
            n_experts as i32,
            topk as i32,
            alignment as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 MoE route offsets launch failed: {err}"))
}

/// Replicate the single quantized hidden row (fp8 + per-group scale) into each
/// selected expert's activation slot at `expert_offsets[topk_idx[j]]`, and write
/// the expert-major per-row route weight for the weighted SwiGLU quant.
///
/// Pad rows (and stale rows from a previous token) need NOT be zeroed: every
/// kernel between scatter and combine is row-independent, and combine reads
/// only the top-k experts' base rows via `expert_offsets` — garbage in unused
/// rows never reaches the output (see `moe_decode.rs` module docs).
#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_route_scatter_launch(
    ctx: &DeviceContext,
    m_capacity: usize,
    n_experts: usize,
    topk: usize,
    k: usize,
    scale_cols: usize,
    hidden_fp8: &CudaSlice<u8>,
    hidden_scale: &CudaSlice<f32>,
    topk_idx: &CudaSlice<i32>,
    topk_weight: &CudaSlice<f32>,
    expert_offsets: &CudaSlice<i64>,
    act: &mut CudaSlice<u8>,
    act_scale: &mut CudaSlice<f32>,
    row_weight: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        m_capacity > 0 && n_experts > 0 && topk > 0 && k > 0 && scale_cols > 0,
        "GLM5.2 MoE scatter needs positive m_capacity/n_experts/topk/k/scale_cols, got {m_capacity}/{n_experts}/{topk}/{k}/{scale_cols}"
    );
    ensure!(
        hidden_fp8.len() >= k && hidden_scale.len() >= scale_cols,
        "GLM5.2 MoE scatter hidden row too small: fp8 {} (need {k}), scale {} (need {scale_cols})",
        hidden_fp8.len(),
        hidden_scale.len()
    );
    // The kernel indexes expert_offsets by EXPERT ID (0..n_experts), not by slot
    // — the buffer must span the full grouped table, `n_experts + 1` entries.
    ensure!(
        topk_idx.len() >= topk && topk_weight.len() >= topk && expert_offsets.len() > n_experts,
        "GLM5.2 MoE scatter route buffers too small: idx {}, weight {}, offsets {} (need {})",
        topk_idx.len(),
        topk_weight.len(),
        expert_offsets.len(),
        n_experts + 1
    );
    ensure!(
        act.len() >= m_capacity * k
            && act_scale.len() >= m_capacity * scale_cols
            && row_weight.len() >= m_capacity,
        "GLM5.2 MoE scatter dest buffers too small for m_capacity={m_capacity}: act {}, scale {}, weight {}",
        act.len(),
        act_scale.len(),
        row_weight.len()
    );
    let (hidden_ptr, _hidden_guard) = hidden_fp8.device_ptr(&ctx.stream);
    let (hscale_ptr, _hscale_guard) = hidden_scale.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = topk_weight.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr(&ctx.stream);
    let (act_ptr, _act_guard) = act.device_ptr_mut(&ctx.stream);
    let (ascale_ptr, _ascale_guard) = act_scale.device_ptr_mut(&ctx.stream);
    let (rweight_ptr, _rweight_guard) = row_weight.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_moe_route_scatter_cuda(
            hidden_ptr as *const u8,
            hscale_ptr as *const f32,
            idx_ptr as *const i32,
            weight_ptr as *const f32,
            offsets_ptr as *const i64,
            act_ptr as *mut u8,
            ascale_ptr as *mut f32,
            rweight_ptr as *mut f32,
            topk as i32,
            k as i32,
            scale_cols as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 MoE scatter launch failed: {err}"))
}

/// Sum the selected experts' W2 output rows (at `expert_offsets[topk_idx[j]]`)
/// into the single token's routed output `[n]`. The route weight was already
/// folded into the W2 input, so this is a plain sum.
#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_combine_launch(
    ctx: &DeviceContext,
    m_capacity: usize,
    n_experts: usize,
    n: usize,
    topk: usize,
    w2_out: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    expert_offsets: &CudaSlice<i64>,
    routed: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        m_capacity > 0 && n_experts > 0 && n > 0 && topk > 0,
        "GLM5.2 MoE combine needs positive m_capacity/n_experts/n/topk, got {m_capacity}/{n_experts}/{n}/{topk}"
    );
    ensure!(
        w2_out.len() >= m_capacity * n,
        "GLM5.2 MoE combine w2_out too small: have {}, need {}",
        w2_out.len(),
        m_capacity * n
    );
    // Same expert-id-indexed contract as scatter: `n_experts + 1` entries.
    ensure!(
        topk_idx.len() >= topk && expert_offsets.len() > n_experts && routed.len() >= n,
        "GLM5.2 MoE combine route/out buffers too small: idx {}, offsets {} (need {}), routed {}",
        topk_idx.len(),
        expert_offsets.len(),
        n_experts + 1,
        routed.len()
    );
    let (w2_ptr, _w2_guard) = w2_out.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr(&ctx.stream);
    let (routed_ptr, _routed_guard) = routed.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_moe_combine_cuda(
            w2_ptr as *const ffi::Half,
            idx_ptr as *const i32,
            offsets_ptr as *const i64,
            routed_ptr as *mut ffi::Half,
            n as i32,
            topk as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 MoE combine launch failed: {err}"))
}
