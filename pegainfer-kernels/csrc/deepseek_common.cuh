#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static __device__ __forceinline__ float fp8_e4m3_to_float(unsigned char value) {
  __half_raw raw = __nv_cvt_fp8_to_halfraw(value, __NV_E4M3);
  __half half_value(raw);
  return __half2float(half_value);
}

static __device__ __forceinline__ float e8m0_to_float(unsigned char value) {
  __nv_bfloat16_raw raw = __nv_cvt_e8m0_to_bf16raw(value);
  __nv_bfloat16 bf16_value(raw);
  return __bfloat162float(bf16_value);
}

static __device__ __forceinline__ unsigned char float_to_e8m0(float value) {
  return __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf);
}

static __device__ __forceinline__ unsigned char float_to_fp8_e4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

static __device__ __forceinline__ float round_to_bf16_float(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

static __device__ __forceinline__ float fp4_e2m1_to_float(unsigned char nibble) {
  switch (nibble & 0x0f) {
    case 0x0:
    case 0x8:
      return 0.0f;
    case 0x1:
      return 0.5f;
    case 0x2:
      return 1.0f;
    case 0x3:
      return 1.5f;
    case 0x4:
      return 2.0f;
    case 0x5:
      return 3.0f;
    case 0x6:
      return 4.0f;
    case 0x7:
      return 6.0f;
    case 0x9:
      return -0.5f;
    case 0xa:
      return -1.0f;
    case 0xb:
      return -1.5f;
    case 0xc:
      return -2.0f;
    case 0xd:
      return -3.0f;
    case 0xe:
      return -4.0f;
    case 0xf:
      return -6.0f;
  }
  return 0.0f;
}
