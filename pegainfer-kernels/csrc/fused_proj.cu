#include "common.cuh"

// ============================================================================
// Fused SiLU-mul from combined [2*I, bs] gate+up buffer.
// Column-major: token j at offset j * 2*I.
//   gate = combined[j * 2*I + i]     for i in [0, I)
//   up   = combined[j * 2*I + I + i] for i in [0, I)
//   out[j * I + i] = bf16(silu(gate)) * up, rounded to bf16
// ============================================================================

__global__ void silu_mul_fused_kernel(
    const __nv_bfloat16 *__restrict__ gate_up, // [2*I, bs] col-major
    __nv_bfloat16 *__restrict__ out,            // [I, bs] col-major
    int intermediate_size, int bs) {

  int total = intermediate_size * bs;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int col = idx / intermediate_size;
    int row = idx % intermediate_size;

    int src_offset = col * 2 * intermediate_size;
    float g = __bfloat162float(gate_up[src_offset + row]);
    float u = __bfloat162float(gate_up[src_offset + intermediate_size + row]);

    float silu_g = g / (1.0f + expf(-g));
    float silu_bf16 = __bfloat162float(__float2bfloat16(silu_g));
    out[idx] = __float2bfloat16(silu_bf16 * u);
  }
}

extern "C" {

void silu_mul_fused_cuda(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *out,
    int intermediate_size, int bs, cudaStream_t stream) {
  int total = intermediate_size * bs;
  int block = 256;
  int grid = (total + block - 1) / block;
  silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
      gate_up, out, intermediate_size, bs);
}

} // extern "C"
