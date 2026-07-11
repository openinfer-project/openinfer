use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, HiddenStates, HiddenStatesRef};

pub struct Dsv2LiteRouterOutput<'a> {
    pub topk_weight: &'a mut CudaSlice<f32>,
    pub topk_idx: &'a mut CudaSlice<i32>,
}

pub fn dsv2_lite_router_logits_into(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    gate_weight: &DeviceMatrix,
    logits: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == gate_weight.cols,
        "DSV2-Lite router hidden_dim {} must match gate cols {}",
        hidden.hidden_dim,
        gate_weight.cols
    );
    let logits_elems = hidden
        .seq_len
        .checked_mul(gate_weight.rows)
        .ok_or_else(|| anyhow!("DSV2-Lite router logits element count overflow"))?;
    ensure!(
        logits.len() >= logits_elems,
        "DSV2-Lite router logits output too small: have {}, need {logits_elems}",
        logits.len()
    );

    let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
    let (gate_ptr, _gate_guard) = gate_weight.data.device_ptr(&ctx.stream);
    let (logits_ptr, _logits_guard) = logits.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_router_logits_cuda(
            hidden_ptr as *const ffi::Half,
            gate_ptr as *const ffi::Half,
            logits_ptr as *mut f32,
            hidden.seq_len as i32,
            hidden.hidden_dim as i32,
            gate_weight.rows as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite router logits CUDA launch failed: {err}"))
}

pub fn dsv2_lite_accumulate_route_row_into(
    ctx: &DeviceContext,
    rows: HiddenStatesRef<'_>,
    source_row: usize,
    scale: f32,
    token_idx: usize,
    seq_len: usize,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(scale.is_finite(), "DSV2-Lite route scale must be finite");
    ensure!(
        source_row < rows.seq_len,
        "DSV2-Lite route source row {source_row} exceeds seq_len {}",
        rows.seq_len
    );
    ensure!(
        token_idx < seq_len,
        "DSV2-Lite route token {token_idx} exceeds seq_len {seq_len}"
    );
    let output_elems = rows
        .hidden_dim
        .checked_mul(seq_len)
        .ok_or_else(|| anyhow!("DSV2-Lite route accumulation output size overflow"))?;
    ensure!(
        out.len() >= output_elems,
        "DSV2-Lite route accumulation output too small: have {}, need {output_elems}",
        out.len()
    );
    let source_offset = source_row
        .checked_mul(rows.hidden_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite route source offset overflow"))?;
    let (rows_ptr, _rows_guard) = rows.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::accumulate_bf16_token_scaled_to_f32_cuda(
            (rows_ptr as *const ffi::Half).add(source_offset),
            scale,
            out_ptr as *mut f32,
            rows.hidden_dim as i32,
            token_idx as i32,
            seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite route accumulation CUDA launch failed: {err}"))
}

#[derive(Clone, Copy, Debug)]
pub struct Dsv2LiteAttentionConfig {
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<Dsv2LiteRopeScalingConfig>,
}

#[derive(Clone, Copy, Debug)]
pub struct Dsv2LiteRopeScalingConfig {
    pub factor: f32,
    pub mscale: f32,
    pub mscale_all_dim: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position_embeddings: usize,
}

pub fn dsv2_lite_router_softmax_topk_into(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    gate_weight: &DeviceMatrix,
    topk: usize,
    output: &mut Dsv2LiteRouterOutput<'_>,
) -> Result<()> {
    dsv2_lite_router_softmax_topk_ref_into(ctx, hidden.as_ref(), gate_weight, topk, output)
}

pub fn dsv2_lite_router_softmax_topk_ref_into(
    ctx: &DeviceContext,
    hidden: HiddenStatesRef<'_>,
    gate_weight: &DeviceMatrix,
    topk: usize,
    output: &mut Dsv2LiteRouterOutput<'_>,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == gate_weight.cols,
        "DSV2-Lite router hidden_dim {} must match gate cols {}",
        hidden.hidden_dim,
        gate_weight.cols
    );
    ensure!(
        gate_weight.rows > 0 && topk > 0 && topk <= gate_weight.rows,
        "DSV2-Lite router invalid n_experts={} topk={topk}",
        gate_weight.rows
    );
    let route_elems = hidden
        .seq_len
        .checked_mul(topk)
        .ok_or_else(|| anyhow!("DSV2-Lite router route element count overflow"))?;
    ensure!(
        output.topk_weight.len() >= route_elems,
        "DSV2-Lite router topk_weight too small: have {}, need {route_elems}",
        output.topk_weight.len()
    );
    ensure!(
        output.topk_idx.len() >= route_elems,
        "DSV2-Lite router topk_idx too small: have {}, need {route_elems}",
        output.topk_idx.len()
    );

    let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
    let (gate_ptr, _gate_guard) = gate_weight.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = output.topk_weight.device_ptr_mut(&ctx.stream);
    let (idx_ptr, _idx_guard) = output.topk_idx.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_router_softmax_topk_cuda(
            hidden_ptr as *const ffi::Half,
            gate_ptr as *const ffi::Half,
            weight_ptr as *mut f32,
            idx_ptr as *mut i32,
            hidden.seq_len as i32,
            hidden.hidden_dim as i32,
            gate_weight.rows as i32,
            topk as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite router CUDA launch failed: {err}"))
}

pub fn dsv2_lite_accumulate_fixed_expert_into(
    ctx: &DeviceContext,
    expert_output: &HiddenStates,
    topk_weight: &CudaSlice<f32>,
    topk_idx: &CudaSlice<i32>,
    global_expert: usize,
    topk: usize,
    accum: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        expert_output.hidden_dim > 0 && expert_output.seq_len > 0,
        "DSV2-Lite fixed-expert accumulate requires non-empty expert output"
    );
    let route_elems = expert_output
        .seq_len
        .checked_mul(topk)
        .ok_or_else(|| anyhow!("DSV2-Lite fixed-expert route element count overflow"))?;
    let hidden_elems = expert_output
        .hidden_dim
        .checked_mul(expert_output.seq_len)
        .ok_or_else(|| anyhow!("DSV2-Lite fixed-expert hidden element count overflow"))?;
    ensure!(
        topk_weight.len() >= route_elems && topk_idx.len() >= route_elems,
        "DSV2-Lite fixed-expert route buffers too small: weights={}, idx={}, need {route_elems}",
        topk_weight.len(),
        topk_idx.len()
    );
    ensure!(
        accum.len() >= hidden_elems,
        "DSV2-Lite fixed-expert accum too small: have {}, need {hidden_elems}",
        accum.len()
    );

    let (expert_ptr, _expert_guard) = expert_output.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = topk_weight.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (accum_ptr, _accum_guard) = accum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_accumulate_fixed_expert_cuda(
            expert_ptr as *const ffi::Half,
            weight_ptr as *const f32,
            idx_ptr as *const i32,
            accum_ptr as *mut f32,
            global_expert as i32,
            expert_output.seq_len as i32,
            expert_output.hidden_dim as i32,
            topk as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite fixed-expert accumulate CUDA launch failed: {err}"))
}

pub fn dsv2_lite_kv_norm_into(
    ctx: &DeviceContext,
    kv_a: &HiddenStates,
    norm_weight: &CudaSlice<bf16>,
    kv_lora_rank: usize,
    eps: f32,
    compressed: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        kv_a.hidden_dim >= kv_lora_rank && compressed.hidden_dim == kv_lora_rank,
        "DSV2-Lite kv norm shape mismatch: kv_a hidden_dim={}, kv_lora_rank={kv_lora_rank}, compressed hidden_dim={}",
        kv_a.hidden_dim,
        compressed.hidden_dim
    );
    ensure!(
        compressed.seq_len == kv_a.seq_len,
        "DSV2-Lite kv norm seq_len mismatch: kv_a={}, compressed={}",
        kv_a.seq_len,
        compressed.seq_len
    );
    ensure!(
        norm_weight.len() >= kv_lora_rank,
        "DSV2-Lite kv norm weight too small: have {}, need {kv_lora_rank}",
        norm_weight.len()
    );
    let (kv_a_ptr, _kv_a_guard) = kv_a.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = norm_weight.device_ptr(&ctx.stream);
    let (compressed_ptr, _compressed_guard) = compressed.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_kv_norm_cuda(
            kv_a_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            compressed_ptr as *mut ffi::Half,
            kv_lora_rank as i32,
            kv_a.hidden_dim as i32,
            kv_a.seq_len as i32,
            eps,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite kv norm CUDA launch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dsv2_lite_decode_attention_into(
    ctx: &DeviceContext,
    cfg: Dsv2LiteAttentionConfig,
    q: &HiddenStates,
    kv_a: &HiddenStates,
    kv_b: &HiddenStates,
    position: usize,
    key_cache: &mut CudaSlice<f32>,
    value_cache: &mut CudaSlice<f32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let query_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    let kv_b_stride = cfg.qk_nope_head_dim + cfg.v_head_dim;
    ensure!(
        q.hidden_dim == cfg.num_heads * query_head_dim && q.seq_len == 1,
        "DSV2-Lite attention q shape mismatch: got [{} x {}], expected [{} x 1]",
        q.hidden_dim,
        q.seq_len,
        cfg.num_heads * query_head_dim
    );
    ensure!(
        kv_a.hidden_dim == cfg.kv_lora_rank + cfg.qk_rope_head_dim && kv_a.seq_len == 1,
        "DSV2-Lite attention kv_a shape mismatch: got [{} x {}]",
        kv_a.hidden_dim,
        kv_a.seq_len
    );
    ensure!(
        kv_b.hidden_dim == cfg.num_heads * kv_b_stride && kv_b.seq_len == 1,
        "DSV2-Lite attention kv_b shape mismatch: got [{} x {}]",
        kv_b.hidden_dim,
        kv_b.seq_len
    );
    ensure!(
        out.hidden_dim == cfg.num_heads * cfg.v_head_dim && out.seq_len == 1,
        "DSV2-Lite attention out shape mismatch: got [{} x {}]",
        out.hidden_dim,
        out.seq_len
    );
    ensure!(
        position < cfg.max_seq_len,
        "DSV2-Lite attention position {position} exceeds max_seq_len {}",
        cfg.max_seq_len
    );
    let key_elems = cfg
        .max_seq_len
        .checked_mul(cfg.num_heads)
        .and_then(|value| value.checked_mul(query_head_dim))
        .ok_or_else(|| anyhow!("DSV2-Lite attention key cache element overflow"))?;
    let value_elems = cfg
        .max_seq_len
        .checked_mul(cfg.num_heads)
        .and_then(|value| value.checked_mul(cfg.v_head_dim))
        .ok_or_else(|| anyhow!("DSV2-Lite attention value cache element overflow"))?;
    ensure!(
        key_cache.len() >= key_elems,
        "DSV2-Lite attention key cache too small: have {}, need {key_elems}",
        key_cache.len()
    );
    ensure!(
        value_cache.len() >= value_elems,
        "DSV2-Lite attention value cache too small: have {}, need {value_elems}",
        value_cache.len()
    );

    let (
        rope_factor,
        rope_mscale,
        rope_mscale_all_dim,
        rope_beta_fast,
        rope_beta_slow,
        rope_original,
        has_rope_scaling,
    ) = cfg
        .rope_scaling
        .map_or((1.0, 1.0, 1.0, 1.0, 1.0, cfg.max_seq_len, 0), |rope| {
            (
                rope.factor,
                rope.mscale,
                rope.mscale_all_dim,
                rope.beta_fast,
                rope.beta_slow,
                rope.original_max_position_embeddings,
                1,
            )
        });

    let (q_ptr, _q_guard) = q.data.device_ptr(&ctx.stream);
    let (kv_a_ptr, _kv_a_guard) = kv_a.data.device_ptr(&ctx.stream);
    let (kv_b_ptr, _kv_b_guard) = kv_b.data.device_ptr(&ctx.stream);
    let (key_cache_ptr, _key_cache_guard) = key_cache.device_ptr_mut(&ctx.stream);
    let (value_cache_ptr, _value_cache_guard) = value_cache.device_ptr_mut(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_decode_attention_cuda(
            q_ptr as *const ffi::Half,
            kv_a_ptr as *const ffi::Half,
            kv_b_ptr as *const ffi::Half,
            key_cache_ptr as *mut f32,
            value_cache_ptr as *mut f32,
            out_ptr as *mut ffi::Half,
            position as i32,
            cfg.num_heads as i32,
            cfg.qk_nope_head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            cfg.v_head_dim as i32,
            cfg.kv_lora_rank as i32,
            kv_a.hidden_dim as i32,
            kv_b.hidden_dim as i32,
            cfg.max_seq_len as i32,
            cfg.rope_theta,
            rope_factor,
            rope_mscale,
            rope_mscale_all_dim,
            rope_beta_fast,
            rope_beta_slow,
            rope_original as i32,
            has_rope_scaling,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite decode attention CUDA launch failed: {err}"))
}
