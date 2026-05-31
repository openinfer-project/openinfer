use pegainfer_kernels::ops::{
    KIMI_K2_MLA_KV_LORA_RANK, KIMI_K2_MLA_NOPE_DIM, KIMI_K2_MLA_Q_HEAD_DIM, KIMI_K2_MLA_QKV_A_OUT,
    KIMI_K2_MLA_ROPE_DIM, KIMI_K2_MLA_V_HEAD_DIM,
};

use crate::config::{
    KIMI_K2_HEADS, KIMI_K2_HIDDEN, KIMI_K2_LAYERS, KIMI_K2_Q_LORA_RANK, KIMI_K2_VOCAB,
};

use super::{BenchSpec, BoundKind, Stage};

const BF16_BYTES: usize = 2;
const I32_BYTES: usize = 4;

#[allow(clippy::vec_init_then_push)]
pub(crate) fn specs(active_rows: usize, arena_rows: usize, ctx_len: usize) -> Vec<BenchSpec> {
    let local_heads = KIMI_K2_HEADS;
    let q_proj_out = local_heads * KIMI_K2_MLA_Q_HEAD_DIM;
    let q_nope_out = local_heads * KIMI_K2_MLA_NOPE_DIM;
    let q_pe_out = local_heads * KIMI_K2_MLA_ROPE_DIM;
    let abs_q_out = local_heads * KIMI_K2_MLA_KV_LORA_RANK;
    let o_proj_in = local_heads * KIMI_K2_MLA_V_HEAD_DIM;

    let layer_calls = KIMI_K2_LAYERS;
    let mut specs = Vec::with_capacity(14);

    specs.push(
        base(
            "rms_norm_batch",
            "decode.attention.input_norm",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .elem(arena_rows * KIMI_K2_HIDDEN)
        .bytes(rms_norm_bytes(arena_rows, KIMI_K2_HIDDEN, layer_calls))
        .flops(rms_norm_flops(arena_rows, KIMI_K2_HIDDEN, layer_calls))
        .bound(BoundKind::Memory)
        .measured()
        .notes("per-layer attention input_norm over the full TP1 DP-rank decode arena"),
    );
    specs.push(gemm_spec(
        "gemm_graphsafe",
        "decode.attention.qkv_a",
        Stage::Attention,
        active_rows,
        arena_rows,
        ctx_len,
        layer_calls,
        arena_rows,
        KIMI_K2_MLA_QKV_A_OUT,
        KIMI_K2_HIDDEN,
        "fused_qkv_a_proj: hidden -> q_lora + compressed_kv + k_rope",
    ));
    specs.push(
        base(
            "kimi_mla_split_qkv_a_norm",
            "decode.attention.qkv_a_split_norm",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .elem(arena_rows * KIMI_K2_MLA_QKV_A_OUT)
        .bytes(
            arena_rows
                * (KIMI_K2_MLA_QKV_A_OUT
                    + KIMI_K2_Q_LORA_RANK
                    + KIMI_K2_MLA_KV_LORA_RANK
                    + KIMI_K2_MLA_ROPE_DIM
                    + KIMI_K2_Q_LORA_RANK
                    + KIMI_K2_MLA_KV_LORA_RANK)
                * BF16_BYTES
                * layer_calls,
        )
        .flops(
            (rms_norm_flops(arena_rows, KIMI_K2_Q_LORA_RANK, 1)
                + rms_norm_flops(arena_rows, KIMI_K2_MLA_KV_LORA_RANK, 1))
                * layer_calls,
        )
        .bound(BoundKind::Memory)
        .measured()
        .notes("split qkv_a, RMS-normalize q_lora and compressed_kv, and keep k_rope"),
    );
    specs.push(gemm_spec(
        "gemm_dm_typed_to_hs_graphsafe",
        "decode.attention.q_b",
        Stage::Attention,
        active_rows,
        arena_rows,
        ctx_len,
        layer_calls,
        arena_rows,
        q_proj_out,
        KIMI_K2_Q_LORA_RANK,
        "q_b_proj: q_lora rank -> TP1 all-head q projection",
    ));
    specs.push(
        base(
            "kimi_mla_rope_split_decode_rt",
            "decode.attention.rope_split",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .elem(arena_rows * (q_proj_out + KIMI_K2_MLA_ROPE_DIM))
        .bytes(
            arena_rows
                * (q_proj_out + KIMI_K2_MLA_ROPE_DIM + q_nope_out + q_pe_out)
                * BF16_BYTES
                * layer_calls
                + arena_rows * I32_BYTES * layer_calls,
        )
        .flops(arena_rows * q_pe_out * 6 * layer_calls)
        .bound(BoundKind::Memory)
        .measured()
        .notes("split q_proj into q_nope/q_pe and apply decode RoPE to q_pe plus append_kpe"),
    );
    specs.push(
        base(
            "kimi_mla_absorb_q_nope_rt",
            "decode.attention.absorb_q_nope",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .m(arena_rows)
        .n(abs_q_out)
        .k(KIMI_K2_MLA_NOPE_DIM)
        .elem(arena_rows * abs_q_out)
        .bytes(gemm_bytes(
            arena_rows,
            abs_q_out,
            KIMI_K2_MLA_NOPE_DIM,
            layer_calls,
        ))
        .flops(gemm_flops(
            arena_rows,
            abs_q_out,
            KIMI_K2_MLA_NOPE_DIM,
            layer_calls,
        ))
        .bound(BoundKind::Compute)
        .measured()
        .notes("absorbed-K projection: per-head q_nope x kv_b K slice -> latent attention query"),
    );
    specs.push(
        base(
            "kimi_mla_paged_kv_append",
            "decode.attention.paged_kv_append",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .elem(arena_rows * (KIMI_K2_MLA_KV_LORA_RANK + KIMI_K2_MLA_ROPE_DIM))
        .bytes(
            arena_rows
                * (KIMI_K2_MLA_KV_LORA_RANK + KIMI_K2_MLA_ROPE_DIM)
                * BF16_BYTES
                * 2
                * layer_calls
                + arena_rows * I32_BYTES * 4 * layer_calls,
        )
        .flops(0)
        .bound(BoundKind::Control)
        .measured()
        .notes("append compressed_kv and k_rope into paged MLA cache for the arena rows"),
    );
    specs.push(
        base(
            "kimi_flashinfer_batch_decode_mla_rt",
            "decode.attention.flashinfer_mla_decode",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .elem(arena_rows * local_heads * ctx_len)
        .bytes(
            arena_rows
                * local_heads
                * ctx_len
                * (KIMI_K2_MLA_KV_LORA_RANK + KIMI_K2_MLA_ROPE_DIM)
                * BF16_BYTES
                * layer_calls
                + arena_rows * abs_q_out * BF16_BYTES * 3 * layer_calls,
        )
        .flops(
            2 * arena_rows
                * local_heads
                * ctx_len
                * (2 * KIMI_K2_MLA_KV_LORA_RANK + KIMI_K2_MLA_ROPE_DIM)
                * layer_calls,
        )
        .bound(BoundKind::Mixed)
        .measured()
        .notes("FlashInfer MLA decode; ctx_len-sensitive cache traffic dominates, softmax/control overhead not fully represented"),
    );
    specs.push(
        base(
            "kimi_mla_v_up_rt",
            "decode.attention.v_up",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .m(arena_rows)
        .n(o_proj_in)
        .k(KIMI_K2_MLA_KV_LORA_RANK)
        .elem(arena_rows * o_proj_in)
        .bytes(gemm_bytes(
            arena_rows,
            o_proj_in,
            KIMI_K2_MLA_KV_LORA_RANK,
            layer_calls,
        ))
        .flops(gemm_flops(
            arena_rows,
            o_proj_in,
            KIMI_K2_MLA_KV_LORA_RANK,
            layer_calls,
        ))
        .bound(BoundKind::Compute)
        .measured()
        .notes("absorbed-V projection: latent MLA output x kv_b V slice -> TP1 attention output"),
    );
    specs.push(gemm_spec(
        "kimi_o_proj_cublaslt",
        "decode.attention.o_proj",
        Stage::Attention,
        active_rows,
        arena_rows,
        ctx_len,
        layer_calls,
        arena_rows,
        KIMI_K2_HIDDEN,
        o_proj_in,
        "o_proj: TP1 all-head attention output -> hidden",
    ));
    specs.push(
        base(
            "fused_add_rms_norm_round_batch",
            "decode.attention.post_attn_add_norm",
            Stage::Attention,
            active_rows,
            arena_rows,
            ctx_len,
            layer_calls,
        )
        .elem(arena_rows * KIMI_K2_HIDDEN)
        .bytes(fused_add_rms_bytes(arena_rows, KIMI_K2_HIDDEN, layer_calls))
        .flops(fused_add_rms_flops(arena_rows, KIMI_K2_HIDDEN, layer_calls))
        .bound(BoundKind::Memory)
        .measured()
        .notes(
            "per-layer post-attention residual add, RMS norm, and BF16 rounding for the next MLP",
        ),
    );
    specs.push(
        base(
            "rms_norm_batch",
            "decode.final.norm",
            Stage::Final,
            active_rows,
            arena_rows,
            ctx_len,
            1,
        )
        .elem(arena_rows * KIMI_K2_HIDDEN)
        .bytes(rms_norm_bytes(arena_rows, KIMI_K2_HIDDEN, 1))
        .flops(rms_norm_flops(arena_rows, KIMI_K2_HIDDEN, 1))
        .bound(BoundKind::Memory)
        .measured()
        .notes("final_norm runs over the TP1 DP-rank arena before full-vocab logits"),
    );
    specs.push(gemm_spec(
        "gemm_graphsafe",
        "decode.final.lm_head",
        Stage::Final,
        active_rows,
        arena_rows,
        ctx_len,
        1,
        arena_rows,
        KIMI_K2_VOCAB,
        KIMI_K2_HIDDEN,
        "TP1 lm_head uses the full vocabulary shard on each DP rank",
    ));
    specs.push(
        base(
            "argmax_batch_bf16",
            "decode.final.argmax",
            Stage::Final,
            active_rows,
            arena_rows,
            ctx_len,
            1,
        )
        .elem(active_rows * KIMI_K2_VOCAB)
        .bytes(active_rows * KIMI_K2_VOCAB * BF16_BYTES + active_rows * (BF16_BYTES + I32_BYTES))
        .flops(0)
        .bound(BoundKind::Memory)
        .measured()
        .notes("current TP1 code launches local top1 over active rows of the full-vocab logits"),
    );

    specs
}

fn base(
    op: &'static str,
    label: &'static str,
    stage: Stage,
    active_rows: usize,
    arena_rows: usize,
    ctx_len: usize,
    calls_per_decode_step: usize,
) -> BenchSpec {
    BenchSpec::new(op, stage, active_rows, arena_rows, ctx_len)
        .label(label)
        .calls_per_decode_step(calls_per_decode_step)
}

fn gemm_spec(
    op: &'static str,
    label: &'static str,
    stage: Stage,
    active_rows: usize,
    arena_rows: usize,
    ctx_len: usize,
    calls_per_decode_step: usize,
    m: usize,
    n: usize,
    k: usize,
    notes: &'static str,
) -> BenchSpec {
    base(
        op,
        label,
        stage,
        active_rows,
        arena_rows,
        ctx_len,
        calls_per_decode_step,
    )
    .m(m)
    .n(n)
    .k(k)
    .elem(m * n)
    .bytes(gemm_bytes(m, n, k, calls_per_decode_step))
    .flops(gemm_flops(m, n, k, calls_per_decode_step))
    .bound(BoundKind::Compute)
    .measured()
    .notes(notes)
}

fn gemm_flops(m: usize, n: usize, k: usize, calls: usize) -> usize {
    2 * m * n * k * calls
}

fn gemm_bytes(m: usize, n: usize, k: usize, calls: usize) -> usize {
    (m * k + k * n + m * n) * BF16_BYTES * calls
}

fn rms_norm_flops(rows: usize, dim: usize, calls: usize) -> usize {
    5 * rows * dim * calls
}

fn rms_norm_bytes(rows: usize, dim: usize, calls: usize) -> usize {
    rows * dim * BF16_BYTES * 4 * calls
}

fn fused_add_rms_flops(rows: usize, dim: usize, calls: usize) -> usize {
    7 * rows * dim * calls
}

fn fused_add_rms_bytes(rows: usize, dim: usize, calls: usize) -> usize {
    rows * dim * BF16_BYTES * 6 * calls
}
