use anyhow::{Result, anyhow, bail, ensure};
use cudarc::driver::{CudaSlice, sys};
use half::bf16;
use pegainfer_kernels::{
    ops::{
        KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_INT4_GROUP_SIZE,
        KIMI_K2_LOCAL_EXPERTS, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        KIMI_K2_MLA_KV_LORA_RANK, KIMI_K2_MLA_O_LOCAL_IN_TP8, KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_MLA_ROPE_DIM, KIMI_K2_TOPK,
        KimiInt4ExpertRole, KimiInt4NibbleOrder, KimiInt4WeightManifest,
        KimiMarlinFusedW13Int4Weight, KimiMarlinInt4Weight, KimiMarlinRouteWorkspace,
        KimiMarlinWna16Workspace, KimiMlaPagedKvLayout, KimiRouterBatch, KimiRouterConfig,
        KimiRouterOutput, KimiRouterScratch, add_batch_into, embedding_batch_vocab_shard,
        flashinfer_top1_batch_into, flashinfer_topk_row_states_bytes, gemm_graphsafe_into_checked,
        kimi_add_f32_bf16_to_bf16, kimi_flashinfer_batch_decode_mla, kimi_marlin_sum_topk_rows_f32,
        kimi_marlin_w13_swiglu, kimi_marlin_wna16_w2_gemm, kimi_marlin_wna16_w13_gemm,
        kimi_mla_absorb_q_nope, kimi_mla_rope_split_decode, kimi_mla_split_qkv_a, kimi_mla_v_up,
        kimi_moe_marlin_align_block_size, kimi_router_noaux_tc_launch,
        kimi_scaled_add_f32_bf16_to_bf16, repeat_f32_for_reduce_scatter_into, rms_norm_batch_into,
        scale_f32_in_place, silu_mul_batch_into,
    },
    tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, KernelCall, TensorSpec},
};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct LatencyStats {
    pub iters: u64,
    pub mean_us: f64,
    pub stddev_us: f64,
    pub min_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MeasuredCall {
    pub supported: bool,
    pub reason: Option<String>,
    pub stats: Option<LatencyStats>,
}

pub fn measure_call(call: &KernelCall, iters: u64) -> Result<MeasuredCall> {
    let stats = match call.op.as_str() {
        "gemm_graphsafe" => Some(measure_gemm(call, iters)?),
        "rms_norm_batch" => Some(measure_rms_norm(call, iters)?),
        "silu_mul_batch" => Some(measure_silu(call, iters)?),
        "add_batch" => Some(measure_add(call, iters)?),
        "scale_f32_in_place" => Some(measure_scale_f32(call, iters)?),
        "kimi_add_f32_bf16_to_bf16" => Some(measure_add_f32_bf16(call, iters)?),
        "kimi_scaled_add_f32_bf16_to_bf16" => Some(measure_scaled_add_f32_bf16(call, iters)?),
        "embedding_batch_vocab_shard" => Some(measure_embedding(call, iters)?),
        "top1_batch" => Some(measure_top1(call, iters)?),
        "kimi_mla_split_qkv_a" => Some(measure_mla_split_qkv_a(call, iters)?),
        "kimi_mla_rope_split_decode" => Some(measure_mla_rope_split(call, iters)?),
        "kimi_mla_absorb_q_nope" => Some(measure_mla_absorb_q(call, iters)?),
        "kimi_mla_v_up" => Some(measure_mla_v_up(call, iters)?),
        "kimi_router_noaux_tc" => Some(measure_router(call, iters)?),
        "kimi_moe_marlin_align_block_size" => Some(measure_marlin_align(call, iters)?),
        "kimi_marlin_sum_topk_rows_f32" => Some(measure_sum_topk(call, iters)?),
        "kimi_marlin_w13_swiglu" => Some(measure_marlin_swiglu(call, iters)?),
        "kimi_flashinfer_batch_decode_mla" => Some(measure_mla_decode(call, iters)?),
        "repeat_f32_for_reduce_scatter" => Some(measure_repeat_f32_for_rs(call, iters)?),
        "all_gather" | "reduce_scatter" => {
            let rank_hint = call
                .attrs
                .iter()
                .find(|attr| attr.name == "ep_world_size" || attr.name == "world_size")
                .map(|attr| attr.value.as_str())
                .unwrap_or("unknown");
            return Ok(MeasuredCall {
                supported: false,
                reason: Some(format!(
                    "NCCL AG/RS provider needs multi-rank H20 harness; rank participation hint={rank_hint}; counted but not timed locally"
                )),
                stats: None,
            });
        }
        "all_reduce" => {
            let rank_hint = call
                .attrs
                .iter()
                .find(|attr| attr.name == "tp_world_size" || attr.name == "world_size")
                .map(|attr| attr.value.as_str())
                .unwrap_or("unknown");
            return Ok(MeasuredCall {
                supported: false,
                reason: Some(format!(
                    "NCCL provider needs multi-rank H20 harness; rank participation hint={rank_hint}; counted but not timed locally"
                )),
                stats: None,
            });
        }
        "kimi_marlin_wna16_gemm" => Some(measure_marlin_wna16(call, iters)?),
        other => {
            return Ok(MeasuredCall {
                supported: false,
                reason: Some(format!("no Kimi provider for op `{other}`")),
                stats: None,
            });
        }
    };

    Ok(MeasuredCall {
        supported: true,
        reason: None,
        stats,
    })
}

pub fn bench_key(call: &KernelCall) -> Result<String> {
    Ok(serde_json::to_string(&serde_json::json!({
        "op": call.op,
        "inputs": call.inputs,
        "outputs": call.outputs,
        "attrs": call.attrs,
    }))?)
}

pub fn axis(spec: &TensorSpec, name: &str) -> Result<usize> {
    spec.axes
        .iter()
        .find(|axis| axis.name == name)
        .map(|axis| axis.size)
        .ok_or_else(|| anyhow!("missing axis `{name}` in {}", spec.compact()))
}

pub fn input<'a>(call: &'a KernelCall, name: &str) -> Result<&'a TensorSpec> {
    call.inputs
        .iter()
        .find(|arg| arg.name == name)
        .map(|arg| &arg.spec)
        .ok_or_else(|| anyhow!("{} missing input `{name}`", call.label))
}

pub fn output<'a>(call: &'a KernelCall, name: &str) -> Result<&'a TensorSpec> {
    call.outputs
        .iter()
        .find(|arg| arg.name == name)
        .map(|arg| &arg.spec)
        .ok_or_else(|| anyhow!("{} missing output `{name}`", call.label))
}

pub fn attr_usize(call: &KernelCall, name: &str) -> Result<usize> {
    call.attrs
        .iter()
        .find(|attr| attr.name == name)
        .ok_or_else(|| anyhow!("{} missing attr `{name}`", call.label))?
        .value
        .parse()
        .map_err(|err| anyhow!("{} invalid attr `{name}`: {err}", call.label))
}

fn measure_gemm(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let weight = input(call, "weight")?;
    let x = input(call, "x")?;
    let out_dim = axis(weight, "out")?;
    let in_dim = axis(weight, "in")?;
    let batch = axis(x, "batch")?;
    let ctx = DeviceContext::new()?;
    let weight = zero_matrix(&ctx, out_dim, in_dim)?;
    let x = HiddenStates::zeros(&ctx, in_dim, batch)?;
    let mut out = HiddenStates::zeros(&ctx, out_dim, batch)?;
    measure_loop(&ctx, iters, || {
        gemm_graphsafe_into_checked(&ctx, &weight, &x, &mut out)?;
        Ok(())
    })
}

fn measure_rms_norm(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let x = input(call, "x")?;
    let hidden = axis(x, "hidden")?;
    let batch = axis(x, "batch")?;
    let ctx = DeviceContext::new()?;
    let x = HiddenStates::zeros(&ctx, hidden, batch)?;
    let weight = DeviceVec::zeros(&ctx, hidden)?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || {
        rms_norm_batch_into(
            &ctx,
            &x,
            &weight,
            crate::config::KIMI_K2_RMS_NORM_EPS,
            &mut out,
        );
        Ok(())
    })
}

fn measure_silu(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let gate = input(call, "gate")?;
    let hidden = axis(gate, "hidden")?;
    let batch = axis(gate, "batch")?;
    let ctx = DeviceContext::new()?;
    let gate = HiddenStates::zeros(&ctx, hidden, batch)?;
    let up = HiddenStates::zeros(&ctx, hidden, batch)?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || {
        silu_mul_batch_into(&ctx, &gate, &up, &mut out)?;
        Ok(())
    })
}

fn measure_add(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let a = input(call, "a")?;
    let hidden = axis(a, "hidden")?;
    let batch = axis(a, "batch")?;
    let ctx = DeviceContext::new()?;
    let a = HiddenStates::zeros(&ctx, hidden, batch)?;
    let b = HiddenStates::zeros(&ctx, hidden, batch)?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || add_batch_into(&ctx, &a, &b, &mut out))
}

fn measure_scale_f32(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let values = input(call, "values")?;
    let elems = axis(values, "elem")?;
    let ctx = DeviceContext::new()?;
    let mut values: CudaSlice<f32> = ctx.stream.alloc_zeros(elems)?;
    measure_loop(&ctx, iters, || {
        scale_f32_in_place(&ctx, &mut values, elems, 2.827)?;
        Ok(())
    })
}

fn measure_repeat_f32_for_rs(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let local = input(call, "local")?;
    let global = output(call, "global")?;
    let local_elems = axis(local, "elem")?;
    let global_elems = axis(global, "elem")?;
    ensure!(
        local_elems > 0 && global_elems.is_multiple_of(local_elems),
        "{} repeat-f32 shape must be a positive multiple: local={local_elems}, global={global_elems}",
        call.label
    );
    let world_size = global_elems / local_elems;
    let ctx = DeviceContext::new()?;
    let local: CudaSlice<f32> = ctx.stream.alloc_zeros(local_elems)?;
    let mut global: CudaSlice<f32> = ctx.stream.alloc_zeros(global_elems)?;
    measure_loop(&ctx, iters, || {
        repeat_f32_for_reduce_scatter_into(&ctx, &local, &mut global, local_elems, world_size)
    })
}

fn measure_add_f32_bf16(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let b = input(call, "b")?;
    let hidden = axis(b, "hidden")?;
    let batch = axis(b, "batch")?;
    let ctx = DeviceContext::new()?;
    let a: CudaSlice<f32> = ctx.stream.alloc_zeros(hidden * batch)?;
    let b = HiddenStates::zeros(&ctx, hidden, batch)?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || {
        kimi_add_f32_bf16_to_bf16(&ctx, &a, &b, &mut out)
    })
}

fn measure_scaled_add_f32_bf16(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let b = input(call, "b")?;
    let hidden = axis(b, "hidden")?;
    let batch = axis(b, "batch")?;
    let scale = call
        .attrs
        .iter()
        .find(|attr| attr.name == "scale")
        .and_then(|attr| attr.value.parse::<f32>().ok())
        .unwrap_or(2.827);
    let ctx = DeviceContext::new()?;
    let a: CudaSlice<f32> = ctx.stream.alloc_zeros(hidden * batch)?;
    let b = HiddenStates::zeros(&ctx, hidden, batch)?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || {
        kimi_scaled_add_f32_bf16_to_bf16(&ctx, &a, scale, &b, &mut out)
    })
}

fn measure_embedding(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let weight = input(call, "weight")?;
    let part_vocab = axis(weight, "out")?;
    let hidden = axis(weight, "in")?;
    let token_ids = input(call, "token_ids")?;
    let batch = axis(token_ids, "batch")?;
    let ctx = DeviceContext::new()?;
    let embed = zero_matrix(&ctx, part_vocab, hidden)?;
    let token_ids_d = ctx.stream.clone_htod(&vec![0_u32; batch])?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || {
        embedding_batch_vocab_shard(&ctx, &embed, &token_ids_d, &mut out, 0, part_vocab as u32)
    })
}

fn measure_top1(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let logits = input(call, "logits")?;
    let vocab = axis(logits, "hidden")?;
    let batch = axis(logits, "batch")?;
    let ctx = DeviceContext::new()?;
    let logits = HiddenStates::zeros(&ctx, vocab, batch)?;
    let mut top1_values: CudaSlice<bf16> = ctx.stream.alloc_zeros(batch)?;
    let mut row_states: CudaSlice<u8> =
        ctx.stream.alloc_zeros(flashinfer_topk_row_states_bytes())?;
    let mut out: CudaSlice<i32> = ctx.stream.alloc_zeros(batch)?;
    measure_loop(&ctx, iters, || {
        flashinfer_top1_batch_into(&ctx, &logits, &mut top1_values, &mut row_states, &mut out)
    })
}

fn measure_mla_split_qkv_a(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let qkv_a_spec = input(call, "qkv_a")?;
    let batch = axis(qkv_a_spec, "batch")?;
    let ctx = DeviceContext::new()?;
    let qkv_a = HiddenStates::zeros(&ctx, KIMI_K2_MLA_QKV_A_OUT, batch)?;
    let mut q_a = HiddenStates::zeros(&ctx, crate::config::KIMI_K2_Q_LORA_RANK, batch)?;
    let mut compressed = HiddenStates::zeros(&ctx, KIMI_K2_MLA_KV_LORA_RANK, batch)?;
    let mut k_rope = HiddenStates::zeros(&ctx, KIMI_K2_MLA_ROPE_DIM, batch)?;
    measure_loop(&ctx, iters, || {
        kimi_mla_split_qkv_a(&ctx, &qkv_a, &mut q_a, &mut compressed, &mut k_rope)
    })
}

fn measure_mla_rope_split(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let q_proj_spec = input(call, "q_proj")?;
    let batch = axis(q_proj_spec, "batch")?;
    let ctx = DeviceContext::new()?;
    let q_proj = HiddenStates::zeros(&ctx, KIMI_K2_MLA_Q_LOCAL_OUT_TP8, batch)?;
    let k_rope = HiddenStates::zeros(&ctx, KIMI_K2_MLA_ROPE_DIM, batch)?;
    let cos: CudaSlice<bf16> = ctx.stream.alloc_zeros(KIMI_K2_MLA_ROPE_DIM)?;
    let sin: CudaSlice<bf16> = ctx.stream.alloc_zeros(KIMI_K2_MLA_ROPE_DIM)?;
    let positions_d = ctx.stream.clone_htod(&vec![0_i32; batch])?;
    let mut q_nope = HiddenStates::zeros(
        &ctx,
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8 - KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8,
        batch,
    )?;
    let mut q_pe = HiddenStates::zeros(&ctx, KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, batch)?;
    let mut append_kpe = HiddenStates::zeros(&ctx, KIMI_K2_MLA_ROPE_DIM, batch)?;
    measure_loop(&ctx, iters, || {
        kimi_mla_rope_split_decode(
            &ctx,
            &q_proj,
            &k_rope,
            &cos,
            &sin,
            &positions_d,
            &mut q_nope,
            &mut q_pe,
            &mut append_kpe,
        )
    })
}

fn measure_mla_absorb_q(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let q_nope_spec = input(call, "q_nope")?;
    let batch = axis(q_nope_spec, "batch")?;
    let ctx = DeviceContext::new()?;
    let kv_b_proj = zero_matrix(
        &ctx,
        KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        KIMI_K2_MLA_KV_LORA_RANK,
    )?;
    let q_nope = HiddenStates::zeros(
        &ctx,
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8 - KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8,
        batch,
    )?;
    let mut q_abs_nope = HiddenStates::zeros(&ctx, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch)?;
    measure_loop(&ctx, iters, || {
        kimi_mla_absorb_q_nope(&ctx, &kv_b_proj, &q_nope, &mut q_abs_nope)
    })
}

fn measure_mla_v_up(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let latent_spec = input(call, "latent")?;
    let batch = axis(latent_spec, "batch")?;
    let ctx = DeviceContext::new()?;
    let kv_b_proj = zero_matrix(
        &ctx,
        KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8,
        KIMI_K2_MLA_KV_LORA_RANK,
    )?;
    let latent = HiddenStates::zeros(&ctx, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch)?;
    let mut out = HiddenStates::zeros(&ctx, KIMI_K2_MLA_O_LOCAL_IN_TP8, batch)?;
    measure_loop(&ctx, iters, || {
        kimi_mla_v_up(&ctx, &kv_b_proj, &latent, &mut out)
    })
}

fn measure_router(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let hidden_spec = input(call, "hidden")?;
    let batch = axis(hidden_spec, "batch")?;
    let ctx = DeviceContext::new()?;
    let hidden = HiddenStates::zeros(&ctx, KIMI_K2_HIDDEN, batch)?;
    let gate_weight = zero_matrix(&ctx, crate::config::KIMI_K2_ROUTED_EXPERTS, KIMI_K2_HIDDEN)?;
    let bias: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(crate::config::KIMI_K2_ROUTED_EXPERTS)?;
    let mut logits: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(batch * crate::config::KIMI_K2_ROUTED_EXPERTS)?;
    let mut scores: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(batch * crate::config::KIMI_K2_ROUTED_EXPERTS)?;
    let mut choice_scores: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(batch * crate::config::KIMI_K2_ROUTED_EXPERTS)?;
    let mut topk_weight: CudaSlice<f32> = ctx.stream.alloc_zeros(batch * KIMI_K2_TOPK)?;
    let mut topk_idx: CudaSlice<i32> = ctx.stream.alloc_zeros(batch * KIMI_K2_TOPK)?;
    measure_loop(&ctx, iters, || {
        let mut scratch = KimiRouterScratch {
            logits: &mut logits,
            scores: &mut scores,
            choice_scores: &mut choice_scores,
        };
        let mut output = KimiRouterOutput {
            topk_weight: &mut topk_weight,
            topk_idx: &mut topk_idx,
        };
        kimi_router_noaux_tc_launch(
            &ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: batch,
                active_tokens: batch,
                padded_tokens: batch,
            },
            &hidden,
            &gate_weight,
            &bias,
            &mut scratch,
            &mut output,
        )
    })
}

fn measure_marlin_align(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let routes = input(call, "topk_idx")?;
    let route_elems = axis(routes, "route")?;
    let batch = route_elems / KIMI_K2_TOPK;
    let ctx = DeviceContext::new()?;
    let topk_idx: CudaSlice<i32> = ctx.stream.alloc_zeros(route_elems)?;
    let mut workspace = KimiMarlinRouteWorkspace::new(&ctx, batch, 64)?;
    measure_loop(&ctx, iters, || {
        let _routing =
            kimi_moe_marlin_align_block_size(&ctx, &mut workspace, &topk_idx, batch, batch, 0)?;
        Ok(())
    })
}

fn measure_sum_topk(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let expert_output = input(call, "expert_output")?;
    let hidden = axis(expert_output, "hidden")?;
    let routed_rows = axis(expert_output, "batch")?;
    let active = routed_rows / KIMI_K2_TOPK;
    let ctx = DeviceContext::new()?;
    let route_output = HiddenStates::zeros(&ctx, hidden, routed_rows)?;
    let mut out: CudaSlice<f32> = ctx.stream.alloc_zeros(hidden * active)?;
    measure_loop(&ctx, iters, || {
        kimi_marlin_sum_topk_rows_f32(&ctx, &route_output, active, &mut out)
    })
}

fn measure_marlin_swiglu(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let gate = input(call, "gate").or_else(|_| input(call, "x"))?;
    let batch = axis(gate, "batch")?;
    let ctx = DeviceContext::new()?;
    let w13 = HiddenStates::zeros(&ctx, 2 * crate::config::KIMI_K2_EXPERT_INTERMEDIATE, batch)?;
    let mut out = HiddenStates::zeros(&ctx, crate::config::KIMI_K2_EXPERT_INTERMEDIATE, batch)?;
    measure_loop(&ctx, iters, || kimi_marlin_w13_swiglu(&ctx, &w13, &mut out))
}

fn measure_marlin_wna16(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let x = input(call, "x")?;
    let out = output(call, "out")?;
    let in_dim = axis(x, "hidden")?;
    let input_rows = axis(x, "batch")?;
    let out_dim = axis(out, "hidden")?;
    let route_elems = axis(out, "batch")?;
    let active_tokens = match (in_dim, out_dim) {
        (KIMI_K2_HIDDEN, dim) if dim == 2 * KIMI_K2_EXPERT_INTERMEDIATE => input_rows,
        (KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN) => route_elems / KIMI_K2_TOPK,
        _ => bail!(
            "{} unsupported Marlin WNA16 shape: in_dim={in_dim} input_rows={input_rows} out_dim={out_dim} output_rows={route_elems}",
            call.label
        ),
    };
    ensure!(
        active_tokens > 0 && route_elems == active_tokens * KIMI_K2_TOPK,
        "{} Marlin route rows must equal active_tokens * topk: active_tokens={active_tokens}, route_elems={route_elems}",
        call.label
    );

    let ctx = DeviceContext::new()?;
    let topk_idx_host = synthetic_local_topk_idx(active_tokens);
    let topk_weight_host = synthetic_topk_weight(active_tokens);
    let topk_idx = ctx.stream.clone_htod(&topk_idx_host)?;
    let topk_weight = ctx.stream.clone_htod(&topk_weight_host)?;
    let mut route_workspace = KimiMarlinRouteWorkspace::new(&ctx, active_tokens, 64)?;
    let routing = kimi_moe_marlin_align_block_size(
        &ctx,
        &mut route_workspace,
        &topk_idx,
        active_tokens,
        active_tokens,
        0,
    )?;
    ensure!(
        routing.route_elems == route_elems,
        "{} Marlin routing produced {} route elems, expected {route_elems}",
        call.label,
        routing.route_elems
    );
    let mut workspace = KimiMarlinWna16Workspace::new(&ctx, routing.max_m_blocks, out_dim, 64)?;

    match (in_dim, out_dim) {
        (KIMI_K2_HIDDEN, dim) if dim == 2 * KIMI_K2_EXPERT_INTERMEDIATE => {
            let input = HiddenStates::zeros(&ctx, KIMI_K2_HIDDEN, active_tokens)?;
            let mut output =
                HiddenStates::zeros(&ctx, 2 * KIMI_K2_EXPERT_INTERMEDIATE, route_elems)?;
            let packed_len = KIMI_K2_LOCAL_EXPERTS
                * (KIMI_K2_HIDDEN / 16)
                * (KIMI_K2_EXPERT_INTERMEDIATE * 4)
                * std::mem::size_of::<u32>();
            let scale_len = KIMI_K2_LOCAL_EXPERTS
                * (KIMI_K2_HIDDEN / KIMI_K2_INT4_GROUP_SIZE)
                * (2 * KIMI_K2_EXPERT_INTERMEDIATE);
            let packed = ctx.stream.alloc_zeros::<u8>(packed_len)?;
            let scale = ctx.stream.alloc_zeros::<bf16>(scale_len)?;
            let weight = KimiMarlinFusedW13Int4Weight {
                local_experts: KIMI_K2_LOCAL_EXPERTS,
                in_dim: KIMI_K2_HIDDEN,
                intermediate_dim: KIMI_K2_EXPERT_INTERMEDIATE,
                group_size: KIMI_K2_INT4_GROUP_SIZE,
                weight_packed_uint4b8: &packed,
                weight_scale_permuted: &scale,
            };
            measure_loop(&ctx, iters, || {
                ctx.stream.memset_zeros(&mut output.data)?;
                ctx.stream.memset_zeros(&mut workspace.locks)?;
                kimi_marlin_wna16_w13_gemm(
                    &ctx,
                    &mut workspace,
                    &routing,
                    &input,
                    &weight,
                    &topk_weight,
                    &mut output,
                )
            })
        }
        (KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN) => {
            let input = HiddenStates::zeros(&ctx, KIMI_K2_EXPERT_INTERMEDIATE, route_elems)?;
            let mut output = HiddenStates::zeros(&ctx, KIMI_K2_HIDDEN, route_elems)?;
            let manifest = KimiInt4WeightManifest::ep8(
                KimiInt4ExpertRole::W2Down,
                0,
                KimiInt4NibbleOrder::LowThenHigh,
            );
            let packed = ctx
                .stream
                .alloc_zeros::<u8>(manifest.packed_shape.elements())?;
            let scale = ctx
                .stream
                .alloc_zeros::<bf16>(manifest.scale_shape.elements())?;
            let weight = KimiMarlinInt4Weight {
                manifest,
                weight_packed_uint4b8: &packed,
                weight_scale_permuted: &scale,
            };
            measure_loop(&ctx, iters, || {
                ctx.stream.memset_zeros(&mut output.data)?;
                ctx.stream.memset_zeros(&mut workspace.locks)?;
                kimi_marlin_wna16_w2_gemm(
                    &ctx,
                    &mut workspace,
                    &routing,
                    &input,
                    &weight,
                    &topk_weight,
                    &mut output,
                )
            })
        }
        _ => unreachable!("shape checked above"),
    }
}

fn synthetic_local_topk_idx(active_tokens: usize) -> Vec<i32> {
    (0..active_tokens * KIMI_K2_TOPK)
        .map(|idx| {
            let token = idx / KIMI_K2_TOPK;
            let route = idx % KIMI_K2_TOPK;
            ((token * 13 + route * 5) % KIMI_K2_LOCAL_EXPERTS) as i32
        })
        .collect()
}

fn synthetic_topk_weight(active_tokens: usize) -> Vec<f32> {
    let denom = (KIMI_K2_TOPK * (KIMI_K2_TOPK + 1) / 2) as f32;
    (0..active_tokens * KIMI_K2_TOPK)
        .map(|idx| ((idx % KIMI_K2_TOPK) + 1) as f32 / denom)
        .collect()
}

fn measure_mla_decode(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let q = input(call, "q_abs_nope")?;
    let batch = axis(q, "batch")?;
    let kv_len = attr_usize(call, "kv_len")?;
    let page_size = 16usize;
    let pages_per_request = kv_len.div_ceil(page_size);
    let max_pages = pages_per_request * batch;
    let ctx = DeviceContext::new()?;
    let layout = KimiMlaPagedKvLayout::separate_contiguous(max_pages, page_size, batch);
    let q_abs = HiddenStates::zeros(&ctx, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch)?;
    let q_pe = HiddenStates::zeros(&ctx, KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, batch)?;
    let mut out = HiddenStates::zeros(&ctx, KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch)?;
    let ckv_cache = ctx.stream.alloc_zeros::<bf16>(layout.required_ckv_len()?)?;
    let kpe_cache = ctx.stream.alloc_zeros::<bf16>(layout.required_kpe_len()?)?;
    let mut page_indices = Vec::with_capacity(max_pages);
    let mut page_indptr = Vec::with_capacity(batch + 1);
    page_indptr.push(0);
    for request in 0..batch {
        for page in 0..pages_per_request {
            page_indices.push((request * pages_per_request + page) as i32);
        }
        page_indptr.push(page_indices.len() as i32);
    }
    let last_page = match kv_len % page_size {
        0 => page_size,
        rem => rem,
    } as i32;
    let page_indices_d = ctx.stream.clone_htod(&page_indices)?;
    let page_indptr_d = ctx.stream.clone_htod(&page_indptr)?;
    let last_page_len_d = ctx.stream.clone_htod(&vec![last_page; batch])?;
    let request_indices_d = ctx
        .stream
        .clone_htod(&(0..batch as i32).collect::<Vec<_>>())?;
    let kv_tile_indices_d = ctx.stream.clone_htod(&vec![0_i32; batch])?;
    let kv_chunk_size_d = ctx.stream.clone_htod(&vec![kv_len as i32; batch])?;
    let sm_scale = 1.0f32 / ((KIMI_K2_MLA_KV_LORA_RANK + KIMI_K2_MLA_ROPE_DIM) as f32).sqrt();
    measure_loop(&ctx, iters, || {
        kimi_flashinfer_batch_decode_mla(
            &ctx,
            &q_abs,
            &q_pe,
            &mut out,
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
    })
}

fn measure_loop(
    ctx: &DeviceContext,
    iters: u64,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<LatencyStats> {
    if iters == 0 {
        bail!("iters must be greater than zero");
    }
    for _ in 0..3 {
        launch()?;
    }
    ctx.sync()?;
    let start = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        start.record(&ctx.stream)?;
        launch()?;
        end.record(&ctx.stream)?;
        samples.push(f64::from(start.elapsed_ms(&end)?) * 1_000.0);
    }
    ctx.sync()?;
    LatencyStats::from_samples(iters, samples)
}

impl LatencyStats {
    pub fn from_samples(iters: u64, mut samples: Vec<f64>) -> Result<Self> {
        if samples.is_empty() {
            bail!("latency sample set is empty");
        }
        samples.sort_by(f64::total_cmp);
        let mean_us = samples.iter().sum::<f64>() / samples.len() as f64;
        let stddev_us = if samples.len() > 1 {
            let variance = samples
                .iter()
                .map(|sample| {
                    let delta = sample - mean_us;
                    delta * delta
                })
                .sum::<f64>()
                / (samples.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        Ok(Self {
            iters,
            mean_us,
            stddev_us,
            min_us: samples[0],
            p50_us: percentile(&samples, 0.50),
            p95_us: percentile(&samples, 0.95),
            p99_us: percentile(&samples, 0.99),
            max_us: samples[samples.len() - 1],
        })
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * quantile).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn zero_matrix(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<DeviceMatrix> {
    Ok(DeviceMatrix {
        data: ctx.stream.alloc_zeros(rows * cols)?,
        rows,
        cols,
    })
}
