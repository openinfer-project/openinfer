use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates, HiddenStatesRef};

/// Batched element-wise add: out = a + b (same shape HiddenStates)
pub fn add_batch(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    add_batch_into(ctx, a, b, &mut out)?;
    Ok(out)
}

/// Batched element-wise add into pre-allocated output buffer (zero allocation).
pub fn add_batch_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(a.hidden_dim, b.hidden_dim);
    assert_eq!(a.seq_len, b.seq_len);
    assert_eq!(out.hidden_dim, a.hidden_dim);
    assert_eq!(out.seq_len, a.seq_len);

    let n = a.hidden_dim * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// In-place scaled add into a row range of `out`: out[row_offset..] += scale * delta.
pub fn scaled_add_rows_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
    row_offset: usize,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "scaled_add_rows_into scale must be finite"
    );
    assert_eq!(
        delta.seq_len, out.seq_len,
        "delta seq_len {} != out seq_len {}",
        delta.seq_len, out.seq_len
    );
    assert!(
        row_offset + delta.hidden_dim <= out.hidden_dim,
        "row range [{}..{}) exceeds out hidden_dim {}",
        row_offset,
        row_offset + delta.hidden_dim,
        out.hidden_dim
    );

    let (delta_ptr, _gd) = delta.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::scaled_add_rows_cuda(
            delta_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            row_offset as i32,
            delta.hidden_dim as i32,
            delta.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// In-place scaled add into a row range of a contiguous token range in `out`.
pub fn scaled_add_rows_token_range_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
    row_offset: usize,
    token_offset: usize,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "scaled_add_rows_token_range_into scale must be finite"
    );
    assert!(
        row_offset + delta.hidden_dim <= out.hidden_dim,
        "row range [{}..{}) exceeds out hidden_dim {}",
        row_offset,
        row_offset + delta.hidden_dim,
        out.hidden_dim
    );
    assert!(
        token_offset + delta.seq_len <= out.seq_len,
        "token range [{}..{}) exceeds out seq_len {}",
        token_offset,
        token_offset + delta.seq_len,
        out.seq_len
    );

    let (delta_ptr, _gd) = delta.data.device_ptr(&ctx.stream);
    let (out_base, _go) = out.data.device_ptr_mut(&ctx.stream);
    let out_ptr =
        out_base + (token_offset * out.hidden_dim * std::mem::size_of::<half::bf16>()) as u64;
    let result = unsafe {
        ffi::scaled_add_rows_cuda(
            delta_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            row_offset as i32,
            delta.hidden_dim as i32,
            delta.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

pub fn gather_hidden_tokens_into(
    ctx: &DeviceContext,
    input: &HiddenStates,
    token_indices: &CudaSlice<i32>,
    token_count: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        out.hidden_dim, input.hidden_dim,
        "gather output hidden_dim {} != input hidden_dim {}",
        out.hidden_dim, input.hidden_dim
    );
    assert_eq!(
        out.seq_len, token_count,
        "gather output seq_len {} != token_count {}",
        out.seq_len, token_count
    );
    assert!(
        token_count <= token_indices.len(),
        "token_count {} exceeds indices len {}",
        token_count,
        token_indices.len()
    );
    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (indices_ptr, _gidx) = token_indices.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gather_hidden_tokens_cuda(
            input_ptr as *const ffi::Half,
            indices_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            input.hidden_dim as i32,
            token_count as i32,
            input.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn scaled_add_rows_indexed_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    token_indices: &CudaSlice<i32>,
    token_count: usize,
    out: &mut HiddenStates,
    row_offset: usize,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "scaled_add_rows_indexed_into scale must be finite"
    );
    assert_eq!(
        delta.seq_len, token_count,
        "delta seq_len {} != token_count {}",
        delta.seq_len, token_count
    );
    assert!(
        token_count <= token_indices.len(),
        "token_count {} exceeds indices len {}",
        token_count,
        token_indices.len()
    );
    assert!(
        row_offset + delta.hidden_dim <= out.hidden_dim,
        "row range [{}..{}) exceeds out hidden_dim {}",
        row_offset,
        row_offset + delta.hidden_dim,
        out.hidden_dim
    );

    let (delta_ptr, _gd) = delta.data.device_ptr(&ctx.stream);
    let (indices_ptr, _gidx) = token_indices.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::scaled_add_rows_indexed_cuda(
            delta_ptr as *const ffi::Half,
            scale,
            indices_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            row_offset as i32,
            delta.hidden_dim as i32,
            token_count as i32,
            out.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// In-place scaled add for tensors with identical shape.
pub fn scaled_add_batch_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(delta.hidden_dim, out.hidden_dim);
    scaled_add_rows_into(ctx, delta, scale, out, 0)
}

pub fn bf16_hidden_to_f32_into(
    ctx: &DeviceContext,
    input: &HiddenStates,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    assert!(
        output.len() >= input.data.len(),
        "f32 output len {} < bf16 input len {}",
        output.len(),
        input.data.len()
    );
    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (output_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::bf16_to_f32_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut f32,
            input.data.len() as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn f32_to_bf16_hidden_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    output: &mut HiddenStates,
) -> Result<()> {
    assert!(
        input.len() >= output.data.len(),
        "f32 input len {} < bf16 output len {}",
        input.len(),
        output.data.len()
    );
    let n = output.data.len();
    let (input_ptr, _gi) = input.device_ptr(&ctx.stream);
    let (output_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::f32_to_bf16_cuda(
            input_ptr as *const f32,
            output_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn scale_f32_in_place(
    ctx: &DeviceContext,
    values: &mut CudaSlice<f32>,
    len: usize,
    scale: f32,
) -> Result<()> {
    assert!(
        len <= values.len(),
        "scale_f32_in_place len {} exceeds values len {}",
        len,
        values.len()
    );
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::scale_f32_cuda(
            values_ptr as *mut f32,
            scale,
            len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn accumulate_bf16_token_scaled_to_f32_into(
    ctx: &DeviceContext,
    token: &HiddenStates,
    scale: f32,
    token_idx: usize,
    seq_len: usize,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "accumulate_bf16_token_scaled_to_f32_into scale must be finite"
    );
    assert_eq!(
        token.seq_len, 1,
        "accumulate_bf16_token_scaled_to_f32_into expects one token, got seq_len={}",
        token.seq_len
    );
    assert!(
        token_idx < seq_len,
        "accumulate token_idx {} exceeds seq_len {}",
        token_idx,
        seq_len
    );
    assert!(
        out.len() >= token.hidden_dim * seq_len,
        "f32 output len {} < hidden_dim {} * seq_len {}",
        out.len(),
        token.hidden_dim,
        seq_len
    );
    let (token_ptr, _gt) = token.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::accumulate_bf16_token_scaled_to_f32_cuda(
            token_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut f32,
            token.hidden_dim as i32,
            token_idx as i32,
            seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn repeat_f32_for_reduce_scatter_into(
    ctx: &DeviceContext,
    local: &CudaSlice<f32>,
    repeated: &mut CudaSlice<f32>,
    local_elems: usize,
    world_size: usize,
) -> Result<()> {
    assert!(
        local_elems <= local.len(),
        "repeat_f32 local_elems {} exceeds local len {}",
        local_elems,
        local.len()
    );
    assert!(
        repeated.len() >= local_elems * world_size,
        "repeat_f32 repeated len {} < local_elems {} * world_size {}",
        repeated.len(),
        local_elems,
        world_size
    );
    let (local_ptr, _local_guard) = local.device_ptr(&ctx.stream);
    let (repeated_ptr, _repeated_guard) = repeated.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::repeat_f32_for_reduce_scatter_cuda(
            local_ptr as *const f32,
            repeated_ptr as *mut f32,
            local_elems as i32,
            world_size as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Batched SiLU+mul: out[i] = silu(gate[i]) * up[i]
pub fn silu_mul_batch(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;
    silu_mul_batch_into(ctx, gate, up, &mut out)?;
    Ok(out)
}

/// Batched SiLU+mul into pre-allocated output buffer (zero allocation).
pub fn silu_mul_batch_into(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(gate.hidden_dim, up.hidden_dim);
    assert_eq!(gate.seq_len, up.seq_len);
    assert_eq!(out.hidden_dim, gate.hidden_dim);
    assert_eq!(out.seq_len, gate.seq_len);

    let n = gate.hidden_dim * gate.seq_len;
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_triton_aot_cuda(
            g_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// Fused SiLU-mul from combined [2*I, bs] gate+up buffer → [I, bs] output.
/// Reads gate and up from interleaved column-major layout, no deinterleave needed.
pub fn silu_mul_fused_batch_into(
    ctx: &DeviceContext,
    gate_up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let intermediate_size = out.hidden_dim;
    let bs = gate_up.seq_len;
    assert_eq!(
        gate_up.hidden_dim,
        2 * intermediate_size,
        "gate_up dim {} != 2 * out dim {}",
        gate_up.hidden_dim,
        intermediate_size
    );
    assert_eq!(out.seq_len, bs);

    let (gu_ptr, _g0) = gate_up.data.device_ptr(&ctx.stream);
    let (out_ptr, _g1) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_fused_cuda(
            gu_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            intermediate_size as i32,
            bs as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        return Err(anyhow!(
            "silu_mul_fused CUDA launch failed: cuda_status={}, intermediate={}, batch={}",
            result,
            intermediate_size,
            bs
        ));
    }
    Ok(())
}

/// Extract a single token's vector from a HiddenStates batch (GPU copy)
pub fn extract_vec(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
) -> Result<DeviceVec> {
    extract_vec_ref(ctx, batch.as_ref(), token_idx)
}

/// Extract a single token's vector from a borrowed HiddenStates batch.
pub fn extract_vec_ref(
    ctx: &DeviceContext,
    batch: HiddenStatesRef<'_>,
    token_idx: usize,
) -> Result<DeviceVec> {
    let len = batch.hidden_dim;
    let mut out = DeviceVec::zeros(ctx, len)?;
    extract_vec_ref_into(ctx, batch, token_idx, &mut out)?;
    Ok(out)
}

/// Copy one column from `batch` into a pre-allocated `out`.
pub fn extract_vec_into(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    extract_vec_ref_into(ctx, batch.as_ref(), token_idx, out)
}

/// Copy one column from a borrowed `batch` into a pre-allocated `out`.
pub fn extract_vec_ref_into(
    ctx: &DeviceContext,
    batch: HiddenStatesRef<'_>,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    let len = batch.hidden_dim;
    anyhow::ensure!(out.len == len, "extract_vec_into len mismatch");
    anyhow::ensure!(
        token_idx < batch.seq_len,
        "extract_vec_into token index {token_idx} out of bounds for seq_len {}",
        batch.seq_len
    );
    let offset = token_idx * batch.hidden_dim;
    let src_view = batch.data.slice(offset..offset + len);
    ctx.stream
        .memcpy_dtod(&src_view, &mut out.data)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;
    Ok(())
}

/// Copy `src` into one column of `batch`.
pub fn write_vec_into(
    ctx: &DeviceContext,
    src: &DeviceVec,
    batch: &mut HiddenStates,
    token_idx: usize,
) -> Result<()> {
    anyhow::ensure!(src.len == batch.hidden_dim, "write_vec_into len mismatch");
    let offset = token_idx * batch.hidden_dim;
    let mut dst_view = batch.data.slice_mut(offset..offset + src.len);
    ctx.stream
        .memcpy_dtod(&src.data, &mut dst_view)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DeviceMatrix;
    use half::bf16;

    fn hidden_from_host(
        ctx: &DeviceContext,
        data: &[bf16],
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        Ok(HiddenStates {
            data: ctx.stream.clone_htod(data)?,
            hidden_dim,
            seq_len,
        })
    }

    fn hidden_to_host(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<bf16>> {
        let host = ctx.stream.clone_dtoh(&hidden.data)?;
        ctx.sync()?;
        Ok(host)
    }

    #[test]
    fn silu_mul_fused_matches_split_bf16_rounding() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden_dim = 4;
        let seq_len = 3;
        let gate: Vec<_> = [
            -3.0, -1.5, 0.0, 2.0, 0.25, 1.0, 3.5, -0.75, 8.0, -8.0, 0.125, -0.5,
        ]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
        let up: Vec<_> = [
            0.5, -2.0, 4.0, 1.25, -1.0, 0.75, 2.5, -3.0, 0.25, 1.5, -0.625, 5.0,
        ]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
        let mut gate_up = Vec::with_capacity(2 * hidden_dim * seq_len);
        for token in 0..seq_len {
            let offset = token * hidden_dim;
            gate_up.extend_from_slice(&gate[offset..offset + hidden_dim]);
            gate_up.extend_from_slice(&up[offset..offset + hidden_dim]);
        }

        let gate_hidden = hidden_from_host(&ctx, &gate, hidden_dim, seq_len)?;
        let up_hidden = hidden_from_host(&ctx, &up, hidden_dim, seq_len)?;
        let gate_up_hidden = hidden_from_host(&ctx, &gate_up, 2 * hidden_dim, seq_len)?;
        let split = silu_mul_batch(&ctx, &gate_hidden, &up_hidden)?;
        let mut fused = HiddenStates::zeros(&ctx, hidden_dim, seq_len)?;

        silu_mul_fused_batch_into(&ctx, &gate_up_hidden, &mut fused)?;

        let split_host = hidden_to_host(&ctx, &split)?;
        let fused_host = hidden_to_host(&ctx, &fused)?;
        assert_eq!(split_host.len(), fused_host.len());
        for (idx, (split_value, fused_value)) in
            split_host.iter().zip(fused_host.iter()).enumerate()
        {
            assert_eq!(
                split_value.to_bits(),
                fused_value.to_bits(),
                "fused/split silu_mul mismatch at index {idx}"
            );
        }
        Ok(())
    }

    #[test]
    fn vstacked_gate_up_gemm_matches_split_mlp_projection() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden_dim = 2;
        let intermediate = 4;
        let seq_len = 3;

        let gate_w: Vec<_> = [1.0, -0.5, -2.0, 0.25, 0.75, 1.5, -1.25, -0.75]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let up_w: Vec<_> = [-0.25, 2.0, 1.25, -1.0, 0.5, 0.75, -1.5, 1.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let input: Vec<_> = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let gate_w = DeviceMatrix::from_host(&ctx, &gate_w, intermediate, hidden_dim)?;
        let up_w = DeviceMatrix::from_host(&ctx, &up_w, intermediate, hidden_dim)?;
        let gate_up_w = DeviceMatrix::vstack(&ctx, &[&gate_w, &up_w])?;
        let input = hidden_from_host(&ctx, &input, hidden_dim, seq_len)?;

        let gate = crate::ops::gemm(&ctx, &gate_w, &input)?;
        let up = crate::ops::gemm(&ctx, &up_w, &input)?;
        let split = silu_mul_batch(&ctx, &gate, &up)?;

        let gate_up = crate::ops::gemm(&ctx, &gate_up_w, &input)?;
        let mut fused = HiddenStates::zeros(&ctx, intermediate, seq_len)?;
        silu_mul_fused_batch_into(&ctx, &gate_up, &mut fused)?;

        let split_host = hidden_to_host(&ctx, &split)?;
        let fused_host = hidden_to_host(&ctx, &fused)?;
        assert_eq!(split_host.len(), fused_host.len());
        for (idx, (split_value, fused_value)) in
            split_host.iter().zip(fused_host.iter()).enumerate()
        {
            let split_f32 = split_value.to_f32();
            let fused_f32 = fused_value.to_f32();
            let diff = (split_f32 - fused_f32).abs();
            assert!(
                diff <= 0.02,
                "vstacked gate_up GEMM mismatch at index {idx}: split={split_f32} fused={fused_f32} diff={diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn extract_vec_ref_rejects_out_of_bounds_token() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = hidden_from_host(&ctx, &[bf16::from_f32(1.0), bf16::from_f32(2.0)], 2, 1)?;
        let mut out = DeviceVec::zeros(&ctx, 2)?;

        let err = extract_vec_ref_into(&ctx, hidden.as_ref(), 1, &mut out).unwrap_err();

        assert!(
            err.to_string().contains("out of bounds"),
            "unexpected error: {err}"
        );
        Ok(())
    }
}
