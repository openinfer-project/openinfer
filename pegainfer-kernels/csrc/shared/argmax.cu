#include "common.cuh"

#define SAMPLE_BLOCK 256
#define ARGMAX_BATCH_TILE_ELEMS 4096

__device__ __forceinline__ bool argmax_better(float lhs_val, int lhs_idx,
                                              float rhs_val, int rhs_idx) {
  return lhs_val > rhs_val || (lhs_val == rhs_val && lhs_idx < rhs_idx);
}

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
    if (argmax_better(val, i, local_max, local_idx)) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (argmax_better(shared_vals[tid + s], shared_idxs[tid + s],
                        shared_vals[tid], shared_idxs[tid])) {
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
    if (argmax_better(val, i, local_max, local_idx)) {
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
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
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

__global__ void argmax_batch_bf16_partial_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ partial_values,
    int* __restrict__ partial_indices,
    int rows,
    int n,
    int tiles_per_row) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int tile = blockIdx.x;
  int row = blockIdx.y;
  if (row >= rows || tile >= tiles_per_row) return;

  int start = tile * ARGMAX_BATCH_TILE_ELEMS;
  int end = start + ARGMAX_BATCH_TILE_ELEMS;
  if (end > n) end = n;
  const __nv_bfloat16* row_x = x + static_cast<size_t>(row) * n;
  int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = start + tid; i < end; i += blockDim.x) {
    float val = __bfloat162float(row_x[i]);
    if (argmax_better(val, i, local_max, local_idx)) {
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
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
        shared_vals[tid] = rhs_val;
        shared_idxs[tid] = rhs_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int out = row * tiles_per_row + tile;
    partial_values[out] = shared_vals[0];
    partial_indices[out] = shared_idxs[0];
  }
}

__global__ void argmax_batch_bf16_finalize_kernel(
    const float* __restrict__ partial_values,
    const int* __restrict__ partial_indices,
    __nv_bfloat16* __restrict__ values,
    int* __restrict__ indices,
    int rows,
    int tiles_per_row) {
  extern __shared__ char shared_mem[];
  float* shared_vals = reinterpret_cast<float*>(shared_mem);
  int* shared_idxs =
      reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int base = row * tiles_per_row;
  float local_max = -INFINITY;
  int local_idx = 0;
  for (int tile = tid; tile < tiles_per_row; tile += blockDim.x) {
    float val = partial_values[base + tile];
    int idx = partial_indices[base + tile];
    if (argmax_better(val, idx, local_max, local_idx)) {
      local_max = val;
      local_idx = idx;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float rhs_val = shared_vals[tid + s];
      int rhs_idx = shared_idxs[tid + s];
      if (argmax_better(rhs_val, rhs_idx, shared_vals[tid], shared_idxs[tid])) {
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

void argmax_batch_bf16_split_cuda(const __nv_bfloat16* x, __nv_bfloat16* values,
                                  int* indices, float* partial_values,
                                  int* partial_indices, int rows, int n,
                                  cudaStream_t stream) {
  int tiles_per_row = (n + ARGMAX_BATCH_TILE_ELEMS - 1) / ARGMAX_BATCH_TILE_ELEMS;
  size_t smem = SAMPLE_BLOCK * (sizeof(float) + sizeof(int));
  argmax_batch_bf16_partial_kernel<<<dim3(tiles_per_row, rows), SAMPLE_BLOCK, smem, stream>>>(
      x, partial_values, partial_indices, rows, n, tiles_per_row);
  argmax_batch_bf16_finalize_kernel<<<rows, SAMPLE_BLOCK, smem, stream>>>(
      partial_values, partial_indices, values, indices, rows, tiles_per_row);
}
}
