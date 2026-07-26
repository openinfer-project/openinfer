#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstdint>
#include <cstring>

static_assert(sizeof(ncclUniqueId) == 128);

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
