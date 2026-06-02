#include "common.cuh"
#include <cmath>

// ============================================================================
// Gated Delta Rule — Recurrent decode step for linear attention
//
// Each block handles one value head (32 blocks total for Qwen3.5-4B).
// State layout: [key_dim, val_dim] f32 per head — val_dim is contiguous.
//   Matches FLA convention [H, K, V] where V stride = 1.
//
// Parallelism strategy: 512 threads per block, split into J_SLICES=4 groups.
//   Each group handles key_dim/4 = 32 rows of the j-loop.
//   val_idx = threadIdx.x % 128, j_slice = threadIdx.x / 128.
//   This gives 16 warps/block → better latency hiding on the 32-block grid.
//   Partial kv_mem and output reductions across j_slices via shared memory.
//
// Thread mapping: val_idx ∈ [0, val_dim), j_slice ∈ [0, J_SLICES).
// GQA: num_key_heads < num_value_heads, so multiple value heads share one key head.
// ============================================================================

#define GDR_KEY_DIM 128
#define GDR_VAL_DIM 128
#define GDR_J_SLICES 4
#define GDR_BLOCK_DIM (GDR_VAL_DIM * GDR_J_SLICES)  // 512
#define GDR_J_PER_SLICE (GDR_KEY_DIM / GDR_J_SLICES) // 32

__global__ void gated_delta_rule_decode_kernel(
    const __nv_bfloat16* __restrict__ qkv,   // [q_dim + k_dim + v_dim] after conv1d+SiLU
    const __nv_bfloat16* __restrict__ b_proj, // [num_value_heads]
    const __nv_bfloat16* __restrict__ a_proj, // [num_value_heads]
    const __nv_bfloat16* __restrict__ dt_bias,// [num_value_heads] bf16
    const float* __restrict__ A_log,          // [num_value_heads] f32
    float* __restrict__ state,                // [num_value_heads, key_dim, val_dim] f32 (V contiguous)
    __nv_bfloat16* __restrict__ output,       // [num_value_heads * val_dim] bf16
    int num_key_heads,
    int num_value_heads,
    int key_dim,    // 128
    int val_dim     // 128
) {
    int v_head = blockIdx.x;
    int val_idx = threadIdx.x & 0x7F;    // threadIdx.x % 128
    int j_slice = threadIdx.x >> 7;       // threadIdx.x / 128  (0..3)
    int warp_id = threadIdx.x >> 5;       // threadIdx.x / 32
    int lane_id = threadIdx.x & 0x1F;     // threadIdx.x % 32

    int k_head = v_head * num_key_heads / num_value_heads;
    int q_dim_total = key_dim * num_key_heads;
    int k_dim_total = q_dim_total;

    __shared__ float smem_q[GDR_KEY_DIM];
    __shared__ float smem_k[GDR_KEY_DIM];
    __shared__ float smem_norm[2];
    __shared__ float warp_norms[GDR_BLOCK_DIM / WARP_SIZE];  // 16
    __shared__ float s_exp_g;
    __shared__ float s_beta;
    __shared__ float smem_kv_partial[GDR_J_SLICES][GDR_VAL_DIM];
    __shared__ float smem_out_partial[GDR_J_SLICES][GDR_VAL_DIM];

    // All j_slices load the same q/k/v (duplicated but cheap)
    float q_val = __bfloat162float(qkv[k_head * key_dim + val_idx]);
    float k_val = __bfloat162float(qkv[q_dim_total + k_head * key_dim + val_idx]);
    float v_val = __bfloat162float(qkv[q_dim_total + k_dim_total + v_head * val_dim + val_idx]);

    // ========================================================================
    // L2 normalize q and k — only j_slice=0 contributes to avoid 4× counting
    // ========================================================================
    float q_sq = (j_slice == 0) ? q_val * q_val : 0.0f;
    q_sq = warp_reduce_sum(q_sq);
    if (lane_id == 0) warp_norms[warp_id] = q_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
        smem_norm[0] = rsqrtf(total + 1e-12f);
    }

    float k_sq = (j_slice == 0) ? k_val * k_val : 0.0f;
    k_sq = warp_reduce_sum(k_sq);
    if (lane_id == 0) warp_norms[warp_id] = k_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
        smem_norm[1] = rsqrtf(total + 1e-12f);
    }
    __syncthreads();

    q_val *= smem_norm[0];
    k_val *= smem_norm[1];
    q_val *= rsqrtf((float)key_dim);

    // j_slice=0 stores normalized q/k to shared memory for all slices to use
    if (j_slice == 0) {
        smem_q[val_idx] = q_val;
        smem_k[val_idx] = k_val;
    }

    // ========================================================================
    // Compute g and beta for this value head
    // ========================================================================
    if (threadIdx.x == 0) {
        float a_val = __bfloat162float(a_proj[v_head]);
        float b_val = __bfloat162float(b_proj[v_head]);
        float bias = __bfloat162float(dt_bias[v_head]);
        float a_log = A_log[v_head];

        float x = a_val + bias;
        float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));
        float g = -expf(a_log) * softplus_x;
        s_exp_g = expf(g);
        s_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float exp_g = s_exp_g;
    float beta = s_beta;

    // ========================================================================
    // State pointer — layout [key_dim, val_dim], val_dim contiguous
    // ========================================================================
    float* my_state = state + v_head * key_dim * val_dim;

    int j_start = j_slice * GDR_J_PER_SLICE;
    int j_end = j_start + GDR_J_PER_SLICE;

    // ========================================================================
    // Pass 1: Decay + partial kv_mem (each j_slice handles 32 j-iterations)
    // ========================================================================
    float partial_kv = 0.0f;
    for (int j = j_start; j < j_end; j++) {
        float s = my_state[j * val_dim + val_idx];
        s *= exp_g;
        my_state[j * val_dim + val_idx] = s;
        partial_kv += s * smem_k[j];
    }

    // Reduce partial kv_mem across j_slices
    smem_kv_partial[j_slice][val_idx] = partial_kv;
    __syncthreads();

    float kv_mem = smem_kv_partial[0][val_idx] + smem_kv_partial[1][val_idx]
                 + smem_kv_partial[2][val_idx] + smem_kv_partial[3][val_idx];

    float my_delta = (v_val - kv_mem) * beta;

    // ========================================================================
    // Pass 2: Rank-1 update + partial output
    // ========================================================================
    float partial_out = 0.0f;
    for (int j = j_start; j < j_end; j++) {
        float s = my_state[j * val_dim + val_idx];
        s += my_delta * smem_k[j];
        my_state[j * val_dim + val_idx] = s;
        partial_out += s * smem_q[j];
    }

    // Reduce partial output across j_slices, j_slice=0 writes result
    smem_out_partial[j_slice][val_idx] = partial_out;
    __syncthreads();

    if (j_slice == 0) {
        float out = smem_out_partial[0][val_idx] + smem_out_partial[1][val_idx]
                   + smem_out_partial[2][val_idx] + smem_out_partial[3][val_idx];
        output[v_head * val_dim + val_idx] = __float2bfloat16(out);
    }
}

extern "C" {

void gated_delta_rule_decode_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    float* state,
    __nv_bfloat16* output,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    cudaStream_t stream
) {
    // One block per value head, 512 threads (128 val_dim × 4 j_slices)
    gated_delta_rule_decode_kernel<<<num_value_heads, GDR_BLOCK_DIM, 0, stream>>>(
        qkv, b_proj, a_proj, dt_bias, A_log,
        state, output,
        num_key_heads, num_value_heads, key_dim, val_dim
    );
}

} // extern "C"
