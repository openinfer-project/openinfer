#include "common.cuh"

#define HEAD_DIM 128

// ============================================================================
// Kernel 1: Per-head QK RMSNorm + RoPE (in-place on Q and K batches)
//
// Grid: (num_q_heads + num_kv_heads, seq_len)
// Block: head_dim (128) threads
// Each block normalizes one head of one token, then applies RoPE.
// ============================================================================
__global__ void prefill_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q,        // [q_dim, seq_len] modified in-place
    __nv_bfloat16* __restrict__ k,        // [kv_dim, seq_len] modified in-place
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_cache,      // [max_pos * head_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int q_dim, int kv_dim, int start_pos,
    const int* start_pos_d,  // if non-null, *start_pos_d overrides start_pos (CUDA Graph safe)
    float eps
) {
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = (head_global < num_q_heads);
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    __nv_bfloat16* data = is_q ? q : k;
    int dim_stride = is_q ? q_dim : kv_dim;
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    int offset = head_local * head_dim + d + token * dim_stride;
    float val = __bfloat162float(data[offset]);

    // RMSNorm: sum of squares via warp reduction
    float sq = val * val;
    sq = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[4];  // head_dim/32 = 4 warps
    if (lane_id == 0) warp_sums[warp_id] = sq;
    __syncthreads();

    __shared__ float s_inv_rms;
    if (warp_id == 0) {
        float v = (lane_id < 4) ? warp_sums[lane_id] : 0.0f;
        float total = warp_reduce_sum(v);
        if (lane_id == 0) s_inv_rms = rsqrtf(total / head_dim + eps);
    }
    __syncthreads();

    // Match HF precision: round to bf16 after norm, then multiply weight
    __nv_bfloat16 normed = __float2bfloat16(val * s_inv_rms);
    float normed_f = __bfloat162float(normed) * __bfloat162float(norm_w[d]);

    // RoPE via shared memory exchange
    __shared__ __nv_bfloat16 smem[HEAD_DIM];
    smem[d] = __float2bfloat16(normed_f);
    __syncthreads();

    int half = head_dim / 2;
    // When start_pos_d is non-null, each token reads its own position from the array.
    // Single decode: start_pos_d = &decode_meta[1], token=0 → start_pos_d[0].
    // Batched decode: start_pos_d = positions, token=batch_idx → positions[batch_idx].
    // Prefill: start_pos_d = nullptr → start_pos + token (sequential within sequence).
    int pos = start_pos_d ? __ldg(start_pos_d + token) : (start_pos + token);

    __nv_bfloat16 result;
    if (d < half) {
        float lo = __bfloat162float(smem[d]);
        float hi = __bfloat162float(smem[d + half]);
        float c = __bfloat162float(cos_cache[pos * head_dim + d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + d]);
        float lo_cos = __bfloat162float(__float2bfloat16(lo * c));
        float hi_sin = __bfloat162float(__float2bfloat16(hi * s));
        result = __float2bfloat16(lo_cos - hi_sin);
    } else {
        int pair_d = d - half;
        float lo = __bfloat162float(smem[pair_d]);
        float hi = __bfloat162float(smem[d]);
        float c = __bfloat162float(cos_cache[pos * head_dim + pair_d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + pair_d]);
        float lo_sin = __bfloat162float(__float2bfloat16(lo * s));
        float hi_cos = __bfloat162float(__float2bfloat16(hi * c));
        result = __float2bfloat16(lo_sin + hi_cos);
    }

    data[offset] = result;
}

extern "C" {

// ============================================================================
// C API: QK norm + RoPE only (no cache write).
//
// Same as prefill_attention_prep_cuda but skips the KV cache write kernel.
// Used when KV is written to paged layout separately (via AppendPagedKVCache).
// ============================================================================
void prefill_qk_norm_rope_only_cuda(
    __nv_bfloat16* q_batch,          // [q_dim, seq_len] modified in-place (normed+RoPE'd)
    __nv_bfloat16* k_batch,          // [kv_dim, seq_len] modified in-place (normed+RoPE'd)
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    float rms_eps,
    cudaStream_t stream
) {
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    dim3 norm_grid(num_q_heads + num_kv_heads, seq_len);
    prefill_qk_norm_rope_kernel<<<norm_grid, head_dim, 0, stream>>>(
        q_batch, k_batch, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim,
        seq_len, q_dim, kv_dim, start_pos, /*start_pos_d=*/nullptr, rms_eps
    );
}

// ============================================================================
// Batched QK norm + RoPE for decode: per-request positions from GPU array.
//
// Q layout: [q_dim, batch_size], K layout: [kv_dim, batch_size]
// Grid: (num_q_heads + num_kv_heads, batch_size), Block: head_dim
// ============================================================================
void qk_norm_rope_batched_decode_cuda(
    __nv_bfloat16* q,                    // [q_dim * batch_size] in-place
    __nv_bfloat16* k,                    // [kv_dim * batch_size] in-place
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* positions,                // [batch_size] per-request positions on GPU
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int batch_size,
    float rms_eps,
    cudaStream_t stream
) {
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Single launch: blockIdx.y = batch index, positions[batch_idx] via start_pos_d array.
    dim3 grid(num_q_heads + num_kv_heads, batch_size);
    prefill_qk_norm_rope_kernel<<<grid, head_dim, 0, stream>>>(
        q, k, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim,
        /*seq_len=*/batch_size, q_dim, kv_dim,
        /*start_pos=*/0, /*start_pos_d=*/positions,
        rms_eps
    );
}

} // extern "C"
