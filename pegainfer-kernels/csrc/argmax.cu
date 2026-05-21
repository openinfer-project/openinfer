#include "common.cuh"

#define SAMPLE_BLOCK 256

__global__ void argmax_kernel(const __nv_bfloat16* __restrict__ x,
                              int* __restrict__ out, int n) {
  extern __shared__ char shared_mem[];
  float* shared_vals = (float*)shared_mem;
  int* shared_idxs = (int*)(shared_mem + blockDim.x * sizeof(float));

  int tid = threadIdx.x;
  int stride = blockDim.x;

  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = tid; i < n; i += stride) {
    float val = __bfloat162float(x[i]);
    if (val > local_max || (val == local_max && i < local_idx)) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (shared_vals[tid + s] > shared_vals[tid] ||
          (shared_vals[tid + s] == shared_vals[tid] &&
           shared_idxs[tid + s] < shared_idxs[tid])) {
        shared_vals[tid] = shared_vals[tid + s];
        shared_idxs[tid] = shared_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[0] = shared_idxs[0];
  }
}

__global__ void argmax_batch_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ values,
    int* __restrict__ indices,
    int rows,
    int n) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* row_x = x + static_cast<size_t>(row) * n;
  int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    float val = __bfloat162float(row_x[i]);
    if (val > local_max || (val == local_max && i < local_idx)) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float rhs_val = shared_vals[tid + s];
      int rhs_idx = shared_idxs[tid + s];
      if (rhs_val > shared_vals[tid] ||
          (rhs_val == shared_vals[tid] && rhs_idx < shared_idxs[tid])) {
        shared_vals[tid] = rhs_val;
        shared_idxs[tid] = rhs_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    indices[row] = shared_idxs[0];
    values[row] = __float2bfloat16(shared_vals[0]);
  }
}

extern "C" {
void argmax_cuda(const __nv_bfloat16* x, int* out, int n, cudaStream_t stream) {
  argmax_kernel<<<1, SAMPLE_BLOCK,
                  SAMPLE_BLOCK * (sizeof(float) + sizeof(int)), stream>>>(x, out, n);
}

void argmax_batch_bf16_cuda(const __nv_bfloat16* x, __nv_bfloat16* values,
                            int* indices, int rows, int n,
                            cudaStream_t stream) {
  argmax_batch_bf16_kernel<<<rows, SAMPLE_BLOCK,
                             SAMPLE_BLOCK * (sizeof(float) + sizeof(int)),
                             stream>>>(x, values, indices, rows, n);
}
}
