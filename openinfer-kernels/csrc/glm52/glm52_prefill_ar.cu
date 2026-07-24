#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstring>

namespace {

constexpr int kHidden = 6144;
constexpr int kMaxRows = 512;
constexpr int kThreads = 256;
static_assert(sizeof(ncclUniqueId) == 128);

struct ReduceArgs {
  const __nv_bfloat16* peers[8];
  const unsigned long long* epoch;
  __nv_bfloat16* output;
  int rows;
  int ranks;
};

__global__ void publish_epoch(unsigned long long* flag,
                              const unsigned long long* epoch) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    atomicExch(flag, *epoch);
  }
}

__global__ void wait_for_consumers(unsigned long long* flag,
                                   unsigned long long* consumed, int ranks) {
  // Do not reuse a payload until every peer has finished reading it.
  if (threadIdx.x == 0 && atomicAdd(flag, 0ULL) != 0) {
    while (atomicAdd(consumed, 0ULL) != static_cast<unsigned long long>(ranks)) {
      __nanosleep(64);
    }
    atomicExch(consumed, 0ULL);
  }
}

__global__ void mark_consumed(ReduceArgs args) {
  const int rank = threadIdx.x;
  if (rank < args.ranks) {
    auto* consumed = reinterpret_cast<unsigned long long*>(
        const_cast<__nv_bfloat16*>(args.peers[rank]) + kMaxRows * kHidden) + 1;
    atomicAdd(consumed, 1ULL);
  }
}

__global__ void reduce_rows(ReduceArgs args) {
  const unsigned long long want = *args.epoch;
  for (int rank = 0; rank < args.ranks; ++rank) {
    auto* flag = reinterpret_cast<unsigned long long*>(
        const_cast<__nv_bfloat16*>(args.peers[rank]) + kMaxRows * kHidden);
    while (atomicAdd(flag, 0ULL) != want) {
      __nanosleep(64);
    }
  }
  const int elements = args.rows * kHidden;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int rank = 0; rank < args.ranks; ++rank) {
      sum += __bfloat162float(args.peers[rank][i]);
    }
    args.output[i] = __float2bfloat16_rn(sum);
  }
}

__global__ void gather_rows(const __nv_bfloat16* input, const int* rows,
                            __nv_bfloat16* output, int count) {
  const int elements = count * kHidden;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x) {
    output[i] = input[static_cast<size_t>(rows[i / kHidden]) * kHidden +
                      i % kHidden];
  }
}

__global__ void scatter_weighted_rows(const __nv_bfloat16* input,
                                      const int* rows, const float* weights,
                                      __nv_bfloat16* output, int count) {
  const int elements = count * kHidden;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x) {
    const int route = i / kHidden;
    const int dst = rows[route] * kHidden + i % kHidden;
    const float value = __bfloat162float(output[dst]) +
                        weights[route] * __bfloat162float(input[i]);
    output[dst] = __float2bfloat16_rn(value);
  }
}

__global__ void reduce_weighted_routes(const __nv_bfloat16* input,
                                       const int* route_slots,
                                       const float* weights,
                                       __nv_bfloat16* output, int rows,
                                       int routes_per_row) {
  const int elements = rows * kHidden;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x) {
    const int row = i / kHidden;
    const int col = i % kHidden;
    float sum = 0.0f;
    for (int route = 0; route < routes_per_row; ++route) {
      const int slot = route_slots[row * routes_per_row + route];
      sum += weights[slot] *
             __bfloat162float(input[slot * kHidden + col]);
    }
    output[i] = __float2bfloat16_rn(sum);
  }
}

}  // namespace

extern "C" int glm52_prefill_nccl_unique_id(uint8_t out[128]) {
  if (!out) return static_cast<int>(ncclInvalidArgument);
  ncclUniqueId id;
  const ncclResult_t result = ncclGetUniqueId(&id);
  if (result == ncclSuccess) std::memcpy(out, &id, sizeof(id));
  return static_cast<int>(result);
}

extern "C" int glm52_prefill_nccl_comm_create(
    const uint8_t unique_id[128], int rank, int ranks, void** out) {
  if (!unique_id || !out || rank < 0 || rank >= ranks) {
    return static_cast<int>(ncclInvalidArgument);
  }
  ncclUniqueId id;
  std::memcpy(&id, unique_id, sizeof(id));
  ncclComm_t comm = nullptr;
  const ncclResult_t result = ncclCommInitRank(&comm, ranks, id, rank);
  if (result == ncclSuccess) *out = comm;
  return static_cast<int>(result);
}

extern "C" int glm52_prefill_nccl_all_reduce_bf16(
    void* comm, const __nv_bfloat16* input, __nv_bfloat16* output,
    size_t count, cudaStream_t stream) {
  if (!comm || !input || !output || count == 0) {
    return static_cast<int>(ncclInvalidArgument);
  }
  return static_cast<int>(ncclAllReduce(input, output, count, ncclBfloat16,
                                        ncclSum, static_cast<ncclComm_t>(comm),
                                        stream));
}

extern "C" int glm52_prefill_nccl_comm_destroy(void* comm) {
  return comm ? static_cast<int>(ncclCommDestroy(static_cast<ncclComm_t>(comm)))
              : static_cast<int>(ncclSuccess);
}

extern "C" CUresult glm52_prefill_ar_cuda(
    const __nv_bfloat16* partial, __nv_bfloat16* output, void* local_buffer,
    const void* const* peer_buffers, const unsigned long long* epoch, int rows,
    int ranks, cudaStream_t stream) {
  if (!partial || !output || !local_buffer || !peer_buffers || !epoch ||
      rows <= 0 || rows > kMaxRows || ranks <= 1 || ranks > 8) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  OPENINFER_FFI_GUARD_BEGIN
  auto* flag = reinterpret_cast<unsigned long long*>(
      static_cast<__nv_bfloat16*>(local_buffer) + kMaxRows * kHidden);
  auto* consumed = flag + 1;
  wait_for_consumers<<<1, 1, 0, stream>>>(flag, consumed, ranks);
  const size_t bytes = static_cast<size_t>(rows) * kHidden * sizeof(__nv_bfloat16);
  auto err = cudaMemcpyAsync(local_buffer, partial, bytes, cudaMemcpyDeviceToDevice,
                             stream);
  if (err != cudaSuccess) return static_cast<CUresult>(err);
  publish_epoch<<<1, 1, 0, stream>>>(flag, epoch);
  ReduceArgs args = {};
  for (int rank = 0; rank < ranks; ++rank) {
    args.peers[rank] =
        reinterpret_cast<const __nv_bfloat16*>(peer_buffers[rank]);
  }
  args.epoch = epoch;
  args.output = output;
  args.rows = rows;
  args.ranks = ranks;
  const int blocks = (rows * kHidden + kThreads - 1) / kThreads;
  reduce_rows<<<blocks, kThreads, 0, stream>>>(args);
  mark_consumed<<<1, ranks, 0, stream>>>(args);
  return static_cast<CUresult>(cudaGetLastError());
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

extern "C" CUresult glm52_prefill_moe_gather_cuda(
    const __nv_bfloat16* input, const int* rows, __nv_bfloat16* output,
    int count, cudaStream_t stream) {
  if (!input || !rows || !output || count <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  OPENINFER_FFI_GUARD_BEGIN
  const int blocks = (count * kHidden + kThreads - 1) / kThreads;
  gather_rows<<<blocks, kThreads, 0, stream>>>(input, rows, output, count);
  return static_cast<CUresult>(cudaGetLastError());
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

extern "C" CUresult glm52_prefill_moe_reduce_cuda(
    const __nv_bfloat16* input, const int* route_slots,
    const float* weights, __nv_bfloat16* output, int rows,
    int routes_per_row, cudaStream_t stream) {
  if (!input || !route_slots || !weights || !output || rows <= 0 ||
      rows > kMaxRows || routes_per_row <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  OPENINFER_FFI_GUARD_BEGIN
  const int blocks = (rows * kHidden + kThreads - 1) / kThreads;
  reduce_weighted_routes<<<blocks, kThreads, 0, stream>>>(
      input, route_slots, weights, output, rows, routes_per_row);
  return static_cast<CUresult>(cudaGetLastError());
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

extern "C" CUresult glm52_prefill_moe_scatter_cuda(
    const __nv_bfloat16* input, const int* rows, const float* weights,
    __nv_bfloat16* output, int count, cudaStream_t stream) {
  if (!input || !rows || !weights || !output || count <= 0 ||
      count > kMaxRows) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  OPENINFER_FFI_GUARD_BEGIN
  const int blocks = (count * kHidden + kThreads - 1) / kThreads;
  scatter_weighted_rows<<<blocks, kThreads, 0, stream>>>(
      input, rows, weights, output, count);
  return static_cast<CUresult>(cudaGetLastError());
  OPENINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}
