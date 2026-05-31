use crate::config::{
    KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_INT4_GROUP_SIZE, KIMI_K2_MOE_LAYERS,
    KIMI_K2_ROUTED_EXPERTS, KIMI_K2_TOPK,
};
use pegainfer_kernels::ops::{KIMI_K2_EP_WORLD, KIMI_K2_LOCAL_EXPERTS, KIMI_K2_ROUTER_SCALE};

use super::{BenchSpec, BoundKind, Stage};

const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;
const I32_BYTES: usize = 4;
const PPLX_EXPERT_PADDING: usize = 8;
const SHARED_GATE_UP: usize = 2 * KIMI_K2_EXPERT_INTERMEDIATE;
const MARLIN_W13_OUT: usize = 2 * KIMI_K2_EXPERT_INTERMEDIATE;
const CALLS_PER_DECODE_STEP: usize = 60;
const _: [(); CALLS_PER_DECODE_STEP] = [(); KIMI_K2_MOE_LAYERS];

pub(crate) fn specs(active_rows: usize, arena_rows: usize, ctx_len: usize) -> Vec<BenchSpec> {
    let routed_rows = pplx_recv_capacity_rows(arena_rows);

    vec![
        spec(
            "kimi_router_noaux_tc",
            "decode.moe.router",
            Stage::MoeRouter,
            BoundKind::Control,
            active_rows,
            arena_rows,
            ctx_len,
            active_rows,
            None,
            0,
            router_bytes(active_rows),
            format!(
                "router reads active_rows hidden states, scores {KIMI_K2_ROUTED_EXPERTS} experts, and writes topk metadata on the aux stream"
            ),
        ),
        spec(
            "kimi_shared_gate_up_cublaslt",
            "decode.moe.shared_gate_up",
            Stage::MoeShared,
            BoundKind::Compute,
            active_rows,
            arena_rows,
            ctx_len,
            active_rows,
            Some((active_rows, SHARED_GATE_UP, KIMI_K2_HIDDEN)),
            gemm_flops(active_rows, SHARED_GATE_UP, KIMI_K2_HIDDEN),
            bf16_gemm_bytes(active_rows, SHARED_GATE_UP, KIMI_K2_HIDDEN),
            "Kimi TP1 shared expert gate/up uses active_rows as batch_size because set_moe_seq_len(active_len) is applied before PPLX MoE",
        ),
        spec(
            "silu_mul_hs_fused_into",
            "decode.moe.shared_swiglu",
            Stage::MoeShared,
            BoundKind::Memory,
            active_rows,
            arena_rows,
            ctx_len,
            active_rows,
            None,
            0,
            swiglu_bytes(active_rows),
            "shared expert activation over gate/up halves",
        ),
        spec(
            "gemm_dm_hs_to_typed_graphsafe",
            "decode.moe.shared_down",
            Stage::MoeShared,
            BoundKind::Compute,
            active_rows,
            arena_rows,
            ctx_len,
            active_rows,
            Some((active_rows, KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE)),
            gemm_flops(active_rows, KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE),
            bf16_gemm_bytes(active_rows, KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE),
            "TP1 shared expert down projection; no TP all-reduce is part of this PPLX manifest",
        ),
        spec(
            "kimi_pplx_build_marlin_routing_on_stream",
            "decode.moe.pplx_build_marlin_routing",
            Stage::MoePplxCompute,
            BoundKind::Control,
            active_rows,
            arena_rows,
            ctx_len,
            routed_rows,
            None,
            0,
            pplx_routing_bytes(routed_rows),
            "PPLX routing uses recv capacity; device-side num_tokens_post_padded may be lower for the actual routed count",
        ),
        spec(
            "kimi_marlin_wna16_pplx_w13_gemm",
            "decode.moe.pplx_marlin_w13",
            Stage::MoePplxCompute,
            BoundKind::Compute,
            active_rows,
            arena_rows,
            ctx_len,
            routed_rows,
            Some((routed_rows, MARLIN_W13_OUT, KIMI_K2_HIDDEN)),
            gemm_flops(routed_rows, MARLIN_W13_OUT, KIMI_K2_HIDDEN),
            marlin_gemm_bytes(routed_rows, MARLIN_W13_OUT, KIMI_K2_HIDDEN),
            "PPLX routed W13 consumes expert-major recv rows; actual device-side rows may be lower than capacity",
        ),
        spec(
            "kimi_marlin_w13_swiglu_pplx",
            "decode.moe.pplx_swiglu",
            Stage::MoePplxCompute,
            BoundKind::Memory,
            active_rows,
            arena_rows,
            ctx_len,
            routed_rows,
            None,
            0,
            swiglu_bytes(routed_rows),
            "PPLX SwiGLU reads num_tokens_post_padded on device and skips sentinel rows within the host capacity",
        ),
        spec(
            "kimi_marlin_wna16_pplx_w2_gemm",
            "decode.moe.pplx_marlin_w2",
            Stage::MoePplxCompute,
            BoundKind::Compute,
            active_rows,
            arena_rows,
            ctx_len,
            routed_rows,
            Some((routed_rows, KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE)),
            gemm_flops(routed_rows, KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE),
            marlin_gemm_bytes(routed_rows, KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE),
            "PPLX routed W2 applies received topk weights locally; capacity rows may exceed the device-side actual routed count; dispatch/combine communication is owned by the comm manifest",
        ),
        spec(
            "kimi_residual_add_scaled_f32",
            "decode.moe.residual_add_scaled",
            Stage::MoePplxCompute,
            BoundKind::Memory,
            active_rows,
            arena_rows,
            ctx_len,
            active_rows,
            None,
            0,
            residual_add_scaled_bytes(active_rows),
            format!(
                "adds residual hidden + shared projection + routed f32 using scale={KIMI_K2_ROUTER_SCALE}"
            ),
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn spec(
    op: &'static str,
    label: &'static str,
    stage: Stage,
    bound: BoundKind,
    active_rows: usize,
    arena_rows: usize,
    ctx_len: usize,
    rows: usize,
    mnk: Option<(usize, usize, usize)>,
    flops_per_step: u128,
    bytes_per_step: u128,
    note: impl Into<String>,
) -> BenchSpec {
    let spec = BenchSpec::new(op, stage, active_rows, arena_rows, ctx_len)
        .label(label)
        .calls(CALLS_PER_DECODE_STEP)
        .elements(rows)
        .bytes(bytes_per_step)
        .flops(flops_per_step)
        .bound(bound)
        .note(note);
    let spec = match op {
        "kimi_router_noaux_tc"
        | "kimi_shared_gate_up_cublaslt"
        | "gemm_dm_typed_to_hs_graphsafe"
        | "silu_mul_hs_fused_into"
        | "gemm_dm_hs_to_typed_graphsafe"
        | "kimi_residual_add_scaled_f32" => spec.measured(),
        _ => spec.estimate_only(),
    };

    if let Some((m, n, k)) = mnk {
        spec.shape_mnk(m, n, k)
    } else {
        spec.shape(match op {
            "kimi_router_noaux_tc" => {
                format!("rows={rows}, experts={KIMI_K2_ROUTED_EXPERTS}, topk={KIMI_K2_TOPK}")
            }
            "silu_mul_hs_fused_into" | "kimi_marlin_w13_swiglu_pplx" => {
                format!("rows={rows}, gate_up={MARLIN_W13_OUT}, out={KIMI_K2_EXPERT_INTERMEDIATE}")
            }
            "kimi_pplx_build_marlin_routing_on_stream" => {
                format!("recv_capacity={rows}, local_experts={KIMI_K2_LOCAL_EXPERTS}")
            }
            "kimi_residual_add_scaled_f32" => {
                format!("rows={rows}, hidden={KIMI_K2_HIDDEN}")
            }
            _ => format!("rows={rows}"),
        })
    }
}

fn pplx_recv_capacity_rows(arena_rows: usize) -> usize {
    let max_routes = arena_rows * KIMI_K2_EP_WORLD * KIMI_K2_TOPK;
    let active_experts = max_routes.min(KIMI_K2_LOCAL_EXPERTS);
    max_routes + active_experts * (PPLX_EXPERT_PADDING - 1)
}

fn gemm_flops(m: usize, n: usize, k: usize) -> u128 {
    2 * m as u128 * n as u128 * k as u128 * CALLS_PER_DECODE_STEP as u128
}

fn bf16_gemm_bytes(m: usize, n: usize, k: usize) -> u128 {
    CALLS_PER_DECODE_STEP as u128
        * BF16_BYTES as u128
        * (m as u128 * k as u128 + n as u128 * k as u128 + m as u128 * n as u128)
}

fn marlin_gemm_bytes(m: usize, n: usize, k: usize) -> u128 {
    CALLS_PER_DECODE_STEP as u128
        * (m as u128 * k as u128 * BF16_BYTES as u128
            + int4_weight_bytes(n, k)
            + m as u128 * n as u128 * BF16_BYTES as u128)
}

fn int4_weight_bytes(out_dim: usize, in_dim: usize) -> u128 {
    let packed = KIMI_K2_LOCAL_EXPERTS as u128 * out_dim as u128 * in_dim.div_ceil(2) as u128;
    let scales = KIMI_K2_LOCAL_EXPERTS as u128
        * out_dim as u128
        * (in_dim / KIMI_K2_INT4_GROUP_SIZE) as u128
        * BF16_BYTES as u128;
    packed + scales
}

fn router_bytes(rows: usize) -> u128 {
    let input = rows * KIMI_K2_HIDDEN * BF16_BYTES;
    let gate_weight = KIMI_K2_ROUTED_EXPERTS * KIMI_K2_HIDDEN * BF16_BYTES;
    let scratch_scores = rows * KIMI_K2_ROUTED_EXPERTS * F32_BYTES * 3;
    let topk = rows * KIMI_K2_TOPK * (F32_BYTES + I32_BYTES);
    CALLS_PER_DECODE_STEP as u128
        * (input as u128 + gate_weight as u128 + scratch_scores as u128 + topk as u128)
}

fn pplx_routing_bytes(routed_rows: usize) -> u128 {
    let counts = KIMI_K2_LOCAL_EXPERTS * I32_BYTES;
    let sorted_token_ids = routed_rows * I32_BYTES;
    let expert_ids = routed_rows.div_ceil(PPLX_EXPERT_PADDING) * I32_BYTES;
    let num_tokens_post_padded = I32_BYTES;
    CALLS_PER_DECODE_STEP as u128
        * (counts as u128
            + sorted_token_ids as u128
            + expert_ids as u128
            + num_tokens_post_padded as u128)
}

fn swiglu_bytes(rows: usize) -> u128 {
    CALLS_PER_DECODE_STEP as u128
        * rows as u128
        * (MARLIN_W13_OUT + KIMI_K2_EXPERT_INTERMEDIATE) as u128
        * BF16_BYTES as u128
}

fn residual_add_scaled_bytes(rows: usize) -> u128 {
    let elems = rows * KIMI_K2_HIDDEN;
    CALLS_PER_DECODE_STEP as u128
        * elems as u128
        * (BF16_BYTES + BF16_BYTES + F32_BYTES + BF16_BYTES) as u128
}
