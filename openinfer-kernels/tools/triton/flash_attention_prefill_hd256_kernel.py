import triton
import triton.language as tl


@triton.jit
def flash_attention_prefill_hd256_kernel(
    Q_ptr,              # [num_q_heads * HEAD_DIM, seq_len] col-major
    K_cache_ptr,        # [num_kv_heads * max_seq * HEAD_DIM] row-major per head
    V_cache_ptr,        # [num_kv_heads * max_seq * HEAD_DIM] row-major per head
    Output_ptr,         # [num_q_heads * HEAD_DIM, seq_len] col-major
    num_q_heads,
    num_kv_heads,
    gqa_ratio,
    seq_len,
    start_pos_ptr,      # *i32 pointer — GPU-resident for CUDA Graph safety
    q_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """FlashAttention-2 for prefill with HEAD_DIM=256.

    Grid: (cdiv(seq_len, BLOCK_M), num_q_heads, 1)
    Each block computes BLOCK_M query rows for one Q head.

    HEAD_DIM=256 is split into four 64-wide chunks to keep register pressure
    manageable while preserving the same Q / cache layouts used by the HD128 path.
    """
    start_pos = tl.load(start_pos_ptr).to(tl.int32)

    MAX_SEQ: tl.constexpr = 4096
    QTR_HD: tl.constexpr = HEAD_DIM // 4
    scale = 1.44269504 / tl.sqrt(float(HEAD_DIM))  # log2(e) / sqrt(HEAD_DIM)

    tile_m = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // gqa_ratio

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    total_seq = start_pos + seq_len

    q_head_offset = q_head * HEAD_DIM
    offs_d = tl.arange(0, QTR_HD)
    q_mask = offs_m[:, None] < seq_len
    q_base = Q_ptr + q_head_offset + offs_m[:, None] * q_dim

    q_q0 = tl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0).to(tl.float32)
    q_q1 = tl.load(q_base + QTR_HD + offs_d[None, :], mask=q_mask, other=0.0).to(tl.float32)
    q_q2 = tl.load(
        q_base + 2 * QTR_HD + offs_d[None, :], mask=q_mask, other=0.0
    ).to(tl.float32)
    q_q3 = tl.load(
        q_base + 3 * QTR_HD + offs_d[None, :], mask=q_mask, other=0.0
    ).to(tl.float32)

    kv_cache_base = kv_head * MAX_SEQ * HEAD_DIM

    m_i = tl.full([BLOCK_M], -1e38, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_q0 = tl.zeros([BLOCK_M, QTR_HD], dtype=tl.float32)
    acc_q1 = tl.zeros([BLOCK_M, QTR_HD], dtype=tl.float32)
    acc_q2 = tl.zeros([BLOCK_M, QTR_HD], dtype=tl.float32)
    acc_q3 = tl.zeros([BLOCK_M, QTR_HD], dtype=tl.float32)

    max_query_in_tile = tl.minimum(tile_m * BLOCK_M + BLOCK_M, seq_len) - 1
    max_kv_pos = start_pos + max_query_in_tile
    num_kv_blocks = tl.cdiv(max_kv_pos + 1, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)

    for kv_block in tl.range(0, num_kv_blocks):
        kv_start = kv_block * BLOCK_N
        kv_pos = kv_start + offs_n

        k_base = kv_cache_base + kv_pos[:, None] * HEAD_DIM
        kv_mask = kv_pos[:, None] < total_seq
        k_q0 = tl.load(K_cache_ptr + k_base + offs_d[None, :], mask=kv_mask, other=0.0).to(
            tl.float32
        )
        k_q1 = tl.load(
            K_cache_ptr + k_base + QTR_HD + offs_d[None, :], mask=kv_mask, other=0.0
        ).to(tl.float32)
        k_q2 = tl.load(
            K_cache_ptr + k_base + 2 * QTR_HD + offs_d[None, :], mask=kv_mask, other=0.0
        ).to(tl.float32)
        k_q3 = tl.load(
            K_cache_ptr + k_base + 3 * QTR_HD + offs_d[None, :], mask=kv_mask, other=0.0
        ).to(tl.float32)

        qk = tl.dot(q_q0.to(tl.bfloat16), tl.trans(k_q0.to(tl.bfloat16)), out_dtype=tl.float32)
        qk += tl.dot(q_q1.to(tl.bfloat16), tl.trans(k_q1.to(tl.bfloat16)), out_dtype=tl.float32)
        qk += tl.dot(q_q2.to(tl.bfloat16), tl.trans(k_q2.to(tl.bfloat16)), out_dtype=tl.float32)
        qk += tl.dot(q_q3.to(tl.bfloat16), tl.trans(k_q3.to(tl.bfloat16)), out_dtype=tl.float32)
        qk *= scale

        causal_bound = start_pos + offs_m
        causal_mask = kv_pos[None, :] <= causal_bound[:, None]
        valid_mask = (offs_m[:, None] < seq_len) & (kv_pos[None, :] < total_seq)
        full_mask = causal_mask & valid_mask
        qk = tl.where(full_mask, qk, float("-inf"))

        block_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        p = tl.where(full_mask, p, 0.0)
        p_bf16 = p.to(tl.bfloat16)

        v_q0 = tl.load(V_cache_ptr + k_base + offs_d[None, :], mask=kv_mask, other=0.0).to(
            tl.float32
        )
        v_q1 = tl.load(
            V_cache_ptr + k_base + QTR_HD + offs_d[None, :], mask=kv_mask, other=0.0
        ).to(tl.float32)
        v_q2 = tl.load(
            V_cache_ptr + k_base + 2 * QTR_HD + offs_d[None, :], mask=kv_mask, other=0.0
        ).to(tl.float32)
        v_q3 = tl.load(
            V_cache_ptr + k_base + 3 * QTR_HD + offs_d[None, :], mask=kv_mask, other=0.0
        ).to(tl.float32)

        acc_q0 = acc_q0 * alpha[:, None] + tl.dot(
            p_bf16, v_q0.to(tl.bfloat16), out_dtype=tl.float32
        )
        acc_q1 = acc_q1 * alpha[:, None] + tl.dot(
            p_bf16, v_q1.to(tl.bfloat16), out_dtype=tl.float32
        )
        acc_q2 = acc_q2 * alpha[:, None] + tl.dot(
            p_bf16, v_q2.to(tl.bfloat16), out_dtype=tl.float32
        )
        acc_q3 = acc_q3 * alpha[:, None] + tl.dot(
            p_bf16, v_q3.to(tl.bfloat16), out_dtype=tl.float32
        )

        l_block = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_block
        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-6)
    acc_q0 = acc_q0 * inv_l[:, None]
    acc_q1 = acc_q1 * inv_l[:, None]
    acc_q2 = acc_q2 * inv_l[:, None]
    acc_q3 = acc_q3 * inv_l[:, None]

    out_base = Output_ptr + q_head_offset + offs_m[:, None] * q_dim
    out_mask = offs_m[:, None] < seq_len
    tl.store(out_base + offs_d[None, :], acc_q0.to(tl.bfloat16), mask=out_mask)
    tl.store(out_base + QTR_HD + offs_d[None, :], acc_q1.to(tl.bfloat16), mask=out_mask)
    tl.store(
        out_base + 2 * QTR_HD + offs_d[None, :], acc_q2.to(tl.bfloat16), mask=out_mask
    )
    tl.store(
        out_base + 3 * QTR_HD + offs_d[None, :], acc_q3.to(tl.bfloat16), mask=out_mask
    )
