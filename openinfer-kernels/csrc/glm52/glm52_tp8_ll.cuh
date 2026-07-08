// Low-latency (LL) packet primitives shared by the GLM5.2 TP8 collective
// kernels (MoE AG/RS in glm52_moe_tp8.cu, attention allreduce in
// glm52_tp8_ar.cu).
//
// LL packet contract: the tag rides in the 4th word of one 16 B aligned
// vector store, and readers treat tag-match as "payload words are valid".
// PTX only guarantees PER-ELEMENT atomicity for vector accesses; the
// tag-guards-payload protocol additionally assumes the interconnect delivers
// an aligned 16 B write as one observable unit. That holds on NVLink (the
// same assumption NCCL's LL128 protocol makes — NCCL disables LL128 on PCIe
// paths for exactly this reason) and is enforced at buffer alloc time via
// CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED (the NVLink-vs-PCIe
// discriminator): see glm52_moe_tp8_alloc_ll_cuda. On an unprobed PCIe P2P
// topology a torn packet would be SILENT data corruption.
#pragma once

#include <cuda_bf16.h>

static __device__ __forceinline__ void glm52_tp8_st_ll(uint4* p, uint2 v,
                                                       unsigned tag) {
  asm volatile("st.volatile.global.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(p),
                   "r"(v.x), "r"(v.y), "r"(0u), "r"(tag));
}

// 12 B payload variant: 3 data words + tag in one 16 B packet (the 8 B form
// above wastes its third word). Same atomicity contract.
static __device__ __forceinline__ void glm52_tp8_st_ll12(uint4* p, unsigned x,
                                                         unsigned y, unsigned z,
                                                         unsigned tag) {
  asm volatile("st.volatile.global.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(p),
                   "r"(x), "r"(y), "r"(z), "r"(tag));
}

static __device__ __forceinline__ uint4 glm52_tp8_ld_ll(const uint4* p) {
  uint4 q;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
               : "l"(p));
  return q;
}

static __device__ __forceinline__ void glm52_tp8_ll_wait(const uint4* p,
                                                         unsigned tag,
                                                         uint4* out) {
  uint4 q;
  long c = 0;
  do {
    q = glm52_tp8_ld_ll(p);
    if (++c > 200000000L) __trap();
  } while (q.w != tag);
  *out = q;
}
