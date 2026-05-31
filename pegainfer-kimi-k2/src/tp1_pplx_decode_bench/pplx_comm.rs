use super::{BenchSpec, BoundKind, Stage};

const CALLS_PER_DECODE_STEP: usize = 60;
const EP_WORLD: usize = 8;
const LOCAL_EXPERTS: usize = 48;
const TOPK: usize = 8;
const HIDDEN: usize = 7168;
const EXPERT_PADDING: usize = 8;
const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;

pub(crate) fn specs(active_rows: usize, arena_rows: usize, ctx_len: usize) -> Vec<BenchSpec> {
    let active_routes = active_rows * TOPK;
    let recv_capacity = pplx_recv_capacity(arena_rows);
    let bf16_row_bytes = HIDDEN * BF16_BYTES;
    let f32_row_bytes = HIDDEN * F32_BYTES;
    let route_weight_bytes = F32_BYTES;

    vec![
        spec(
            "dispatch_send",
            "decode.moe.pplx.dispatch_send",
            active_rows,
            arena_rows,
            ctx_len,
            active_routes * HIDDEN,
            active_routes * (bf16_row_bytes + route_weight_bytes),
            format!(
                "active_routes={active_routes}; outbound accounting estimate: {active_rows} local decode rows x topk BF16 input rows plus F32 route weights; ctx_len={ctx_len} is metadata only and bytes are decode-token dependent; use a real all-rank PPLX harness for measured transfer bytes"
            ),
        ),
        spec(
            "dispatch_recv",
            "decode.moe.pplx.dispatch_recv",
            active_rows,
            arena_rows,
            ctx_len,
            recv_capacity * HIDDEN,
            recv_capacity * (bf16_row_bytes + route_weight_bytes),
            format!(
                "recv_capacity={recv_capacity}, max_total_tokens={}; inbound accounting estimate sized by PPLX receive capacity from arena_rows, including BF16 input rows plus F32 route weights; ctx_len={ctx_len} is metadata only and bytes are decode-token dependent; use a real all-rank PPLX harness for measured transfer bytes",
                arena_rows * EP_WORLD,
            ),
        ),
        spec(
            "combine_send",
            "decode.moe.pplx.combine_send",
            active_rows,
            arena_rows,
            ctx_len,
            recv_capacity * HIDDEN,
            recv_capacity * bf16_row_bytes,
            format!(
                "recv_capacity={recv_capacity}, max_total_tokens={}; outbound accounting estimate sized by PPLX receive capacity from arena_rows, sending BF16 expert rows; ctx_len={ctx_len} is metadata only and bytes are decode-token dependent; use a real all-rank PPLX harness for measured transfer bytes",
                arena_rows * EP_WORLD,
            ),
        ),
        spec(
            "combine_recv",
            "decode.moe.pplx.combine_recv",
            active_rows,
            arena_rows,
            ctx_len,
            active_routes * HIDDEN + active_rows * HIDDEN,
            active_routes * bf16_row_bytes + active_rows * f32_row_bytes,
            format!(
                "active_routes={active_routes}; inbound accounting estimate for BF16 expert rows plus F32 routed-output writes; ctx_len={ctx_len} is metadata only and bytes are decode-token dependent; use a real all-rank PPLX harness for measured transfer bytes"
            ),
        ),
    ]
}

fn spec(
    op: &'static str,
    label: &'static str,
    active_rows: usize,
    arena_rows: usize,
    ctx_len: usize,
    elem_count: usize,
    bytes_per_call: usize,
    notes: String,
) -> BenchSpec {
    BenchSpec::new(op, Stage::MoePplxComm, active_rows, arena_rows, ctx_len)
        .label(label)
        .calls(CALLS_PER_DECODE_STEP)
        .elements(elem_count)
        .bytes(bytes_per_call as u128 * CALLS_PER_DECODE_STEP as u128)
        .flops(0)
        .bound(BoundKind::Comm)
        .estimate_only()
        .note(notes)
}

fn pplx_recv_capacity(arena_rows: usize) -> usize {
    let max_total_tokens = arena_rows * EP_WORLD;
    let max_routes = max_total_tokens * TOPK;
    max_routes + max_routes.min(LOCAL_EXPERTS) * (EXPERT_PADDING - 1)
}
