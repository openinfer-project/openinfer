// Device-side decode-step input feed: advance the per-row step inputs in
// place so the next whole-step graph replay can be enqueued without a host
// round-trip. Runs as an eager launch between two replays on the decode
// stream; on a non-leased step the host prologue rewrites all four buffers,
// so an unused advance is harmless by construction.
#include <cstdint>
#include <cuda_runtime.h>

__global__ void glm52_decode_feed_kernel(const int32_t* __restrict__ argmax_indices,
                                         uint32_t* __restrict__ token_ids,
                                         uint32_t* __restrict__ positions,
                                         int64_t* __restrict__ slot_mapping,
                                         int32_t* __restrict__ seq_lens,
                                         int rows) {
    const int row = static_cast<int>(threadIdx.x);
    if (row >= rows) {
        return;
    }
    token_ids[row] = static_cast<uint32_t>(argmax_indices[row]);
    positions[row] += 1u;
    slot_mapping[row] += 1;
    seq_lens[row] += 1;
}

extern "C" cudaError_t glm52_decode_feed_launch_cuda(const int32_t* argmax_indices,
                                                     uint32_t* token_ids,
                                                     uint32_t* positions,
                                                     int64_t* slot_mapping,
                                                     int32_t* seq_lens,
                                                     int rows,
                                                     cudaStream_t stream) {
    if (rows <= 0 || rows > 32) {
        return cudaErrorInvalidValue;
    }
    glm52_decode_feed_kernel<<<1, 32, 0, stream>>>(argmax_indices, token_ids, positions,
                                                   slot_mapping, seq_lens, rows);
    return cudaGetLastError();
}
