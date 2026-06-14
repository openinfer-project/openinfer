// Green Context SM partition demo v2.
// Uses driver API launch to ensure kernels land on the correct partition.
//
// Build:
//   /usr/local/cuda-13.3/bin/nvcc -o green_ctx_demo green_ctx_demo.cu \
//       -lcuda -arch=sm_120
//
// Run:
//   ./green_ctx_demo
//   nsys profile --trace=cuda ./green_ctx_demo

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *str = "unknown";                                             \
      cuGetErrorString(err, &str);                                             \
      fprintf(stderr, "CUDA driver error at %s:%d: %s (code %d)\n",           \
              __FILE__, __LINE__, str, (int)err);                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA runtime error at %s:%d: %s\n", __FILE__,           \
              __LINE__, cudaGetErrorString(err));                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Busy-spin kernel
__global__ void busy_kernel(float *out, int iters) {
  float val = 1.0f;
  for (int i = 0; i < iters; i++) {
    val = val * 1.0001f + 0.0001f;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = val;
  }
}

__global__ void stamp_kernel(long long *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = clock64();
  }
}

// Helper: launch a kernel via driver API on a specific stream
void launch_busy(CUfunction func, CUstream stream, float *out, int iters,
                 int blocks) {
  void *args[] = {&out, &iters};
  CU_CHECK(cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, stream, args, NULL));
}

void launch_stamp(CUfunction func, CUstream stream, long long *out) {
  void *args[] = {&out};
  CU_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, NULL));
}

int main() {
  CU_CHECK(cuInit(0));
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0));

  int sm_count;
  CU_CHECK(cuDeviceGetAttribute(&sm_count,
                                 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                 device));
  printf("Device SM count: %d\n", sm_count);

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(0));
  CUcontext primary_ctx;
  CU_CHECK(cuDevicePrimaryCtxRetain(&primary_ctx, device));

  // ── Get device SM resource ─────────────────────────────────────────
  CUdevResource dev_resource;
  CU_CHECK(cuDeviceGetDevResource(device, &dev_resource,
                                   CU_DEV_RESOURCE_TYPE_SM));
  unsigned int alignment = dev_resource.sm.smCoscheduledAlignment;
  printf("SM resource: smCount=%u, alignment=%u\n",
         dev_resource.sm.smCount, alignment);

  // ── Split into 2 equal groups ─────────────────────────────────────
  // Use a larger minCount so each group gets more SMs.
  // With 70 SMs and minCount=32, we should get 2 groups of 32 (remainder=6).
  unsigned int minCount = (sm_count / 2 / alignment) * alignment;
  if (minCount < alignment) minCount = alignment;
  printf("Using minCount=%u for split\n", minCount);

  unsigned int nbGroups = 2;
  CUdevResource groups[2];
  CUdevResource remainder;
  memset(groups, 0, sizeof(groups));
  memset(&remainder, 0, sizeof(remainder));

  // Query first
  unsigned int maxGroups = 0;
  CU_CHECK(cuDevSmResourceSplitByCount(NULL, &maxGroups, &dev_resource,
                                        NULL, 0, minCount));
  printf("Max groups with minCount=%u: %u\n", minCount, maxGroups);
  if (maxGroups < 2) {
    printf("Cannot create 2 groups, trying minCount=%u\n", alignment);
    minCount = alignment;
    CU_CHECK(cuDevSmResourceSplitByCount(NULL, &maxGroups, &dev_resource,
                                          NULL, 0, minCount));
    printf("Max groups with minCount=%u: %u\n", minCount, maxGroups);
  }

  nbGroups = 2;
  CU_CHECK(cuDevSmResourceSplitByCount(groups, &nbGroups, &dev_resource,
                                        &remainder, 0, minCount));
  printf("Got %u groups: [%u SMs] [%u SMs] remainder=%u\n",
         nbGroups, groups[0].sm.smCount, groups[1].sm.smCount,
         remainder.sm.smCount);

  // ── Generate descriptors & create green contexts ──────────────────
  CUdevResourceDesc desc_a, desc_b;
  CU_CHECK(cuDevResourceGenerateDesc(&desc_a, &groups[0], 1));
  CU_CHECK(cuDevResourceGenerateDesc(&desc_b, &groups[1], 1));

  CUgreenCtx green_a, green_b;
  CU_CHECK(cuGreenCtxCreate(&green_a, desc_a, device,
                             CU_GREEN_CTX_DEFAULT_STREAM));
  CU_CHECK(cuGreenCtxCreate(&green_b, desc_b, device,
                             CU_GREEN_CTX_DEFAULT_STREAM));
  printf("Green contexts created\n");

  CUcontext ctx_a, ctx_b;
  CU_CHECK(cuCtxFromGreenCtx(&ctx_a, green_a));
  CU_CHECK(cuCtxFromGreenCtx(&ctx_b, green_b));

  // ── Create streams via cuGreenCtxStreamCreate ─────────────────────
  CUstream stream_a, stream_b;
  CU_CHECK(cuGreenCtxStreamCreate(&stream_a, green_a, CU_STREAM_NON_BLOCKING, 0));
  CU_CHECK(cuGreenCtxStreamCreate(&stream_b, green_b, CU_STREAM_NON_BLOCKING, 0));
  printf("Streams created via cuGreenCtxStreamCreate\n");

  CU_CHECK(cuCtxSetCurrent(primary_ctx));

  // ── Allocate ─────────────────────────────────────────────────────
  float *d_out_a, *d_out_b;
  long long *d_ts_a0, *d_ts_a1, *d_ts_b0, *d_ts_b1;
  CUDA_CHECK(cudaMalloc(&d_out_a, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_b, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ts_a0, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_ts_a1, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_ts_b0, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_ts_b1, sizeof(long long)));

  int iters = 2000000;
  int blocks_a = groups[0].sm.smCount;
  int blocks_b = groups[1].sm.smCount;
  printf("\nLaunching: A=%d blocks, B=%d blocks, iters=%d\n",
         blocks_a, blocks_b, iters);

  CUDA_CHECK(cudaDeviceSynchronize());

  // Launch using runtime <<<>>> with current context set to green ctx
  CU_CHECK(cuCtxSetCurrent(ctx_a));
  stamp_kernel<<<1, 1, 0, (cudaStream_t)stream_a>>>(d_ts_a0);
  busy_kernel<<<blocks_a, 256, 0, (cudaStream_t)stream_a>>>(d_out_a, iters);
  stamp_kernel<<<1, 1, 0, (cudaStream_t)stream_a>>>(d_ts_a1);

  CU_CHECK(cuCtxSetCurrent(ctx_b));
  stamp_kernel<<<1, 1, 0, (cudaStream_t)stream_b>>>(d_ts_b0);
  busy_kernel<<<blocks_b, 256, 0, (cudaStream_t)stream_b>>>(d_out_b, iters);
  stamp_kernel<<<1, 1, 0, (cudaStream_t)stream_b>>>(d_ts_b1);

  // ── Sync & read ───────────────────────────────────────────────────
  CU_CHECK(cuCtxSetCurrent(ctx_a));
  CU_CHECK(cuStreamSynchronize(stream_a));
  CU_CHECK(cuCtxSetCurrent(ctx_b));
  CU_CHECK(cuStreamSynchronize(stream_b));
  CU_CHECK(cuCtxSetCurrent(primary_ctx));

  long long ts_a0, ts_a1, ts_b0, ts_b1;
  CUDA_CHECK(cudaMemcpy(&ts_a0, d_ts_a0, sizeof(long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&ts_a1, d_ts_a1, sizeof(long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&ts_b0, d_ts_b0, sizeof(long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&ts_b1, d_ts_b1, sizeof(long long), cudaMemcpyDeviceToHost));

  long long dur_a = ts_a1 - ts_a0;
  long long dur_b = ts_b1 - ts_b0;
  printf("\n── Green Context parallel ──\n");
  printf("  A: %lld ticks (%u SMs)\n", dur_a, groups[0].sm.smCount);
  printf("  B: %lld ticks (%u SMs)\n", dur_b, groups[1].sm.smCount);

  long long overlap_start = (ts_a0 > ts_b0) ? ts_a0 : ts_b0;
  long long overlap_end   = (ts_a1 < ts_b1) ? ts_a1 : ts_b1;
  long long overlap = overlap_end - overlap_start;
  long long wall = ((ts_a1 > ts_b1) ? ts_a1 : ts_b1) -
                   ((ts_a0 < ts_b0) ? ts_a0 : ts_b0);

  if (overlap > 0) {
    printf("  OVERLAP: %lld ticks (%.1f%% of max)\n",
           overlap, 100.0 * overlap / ((dur_a > dur_b) ? dur_a : dur_b));
    printf("  Wall: %lld ticks\n", wall);
    printf("  => SM partition overlap confirmed!\n");
  } else {
    printf("  NO OVERLAP (diff=%lld ticks)\n", overlap_start - overlap_end);
  }

  // ── Serial baseline ───────────────────────────────────────────────
  printf("\n── Serial baseline ──\n");
  long long *d_s0, *d_s1, *d_s2;
  CUDA_CHECK(cudaMalloc(&d_s0, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_s1, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_s2, sizeof(long long)));

  stamp_kernel<<<1, 1>>>(d_s0);
  busy_kernel<<<blocks_a, 256>>>(d_out_a, iters);
  stamp_kernel<<<1, 1>>>(d_s1);
  busy_kernel<<<blocks_b, 256>>>(d_out_b, iters);
  stamp_kernel<<<1, 1>>>(d_s2);
  CUDA_CHECK(cudaDeviceSynchronize());

  long long s0, s1, s2;
  CUDA_CHECK(cudaMemcpy(&s0, d_s0, sizeof(long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&s1, d_s1, sizeof(long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&s2, d_s2, sizeof(long long), cudaMemcpyDeviceToHost));

  printf("  Kernel A: %lld ticks\n", s1 - s0);
  printf("  Kernel B: %lld ticks\n", s2 - s1);
  printf("  Total:    %lld ticks\n", s2 - s0);
  if (wall > 0)
    printf("\n  Speedup (serial/parallel): %.2fx\n", (double)(s2 - s0) / wall);

  // ── Cleanup ───────────────────────────────────────────────────────
  CU_CHECK(cuCtxSetCurrent(ctx_a));
  CU_CHECK(cuStreamDestroy(stream_a));
  CU_CHECK(cuCtxSetCurrent(ctx_b));
  CU_CHECK(cuStreamDestroy(stream_b));
  CU_CHECK(cuCtxSetCurrent(primary_ctx));
  CU_CHECK(cuGreenCtxDestroy(green_a));
  CU_CHECK(cuGreenCtxDestroy(green_b));
  CU_CHECK(cuDevicePrimaryCtxRelease(device));

  cudaFree(d_out_a); cudaFree(d_out_b);
  cudaFree(d_ts_a0); cudaFree(d_ts_a1);
  cudaFree(d_ts_b0); cudaFree(d_ts_b1);
  cudaFree(d_s0); cudaFree(d_s1); cudaFree(d_s2);

  printf("\nDone.\n");
  return 0;
}
