// GLM5.2 single-node 8×H200 DeepEP config (DP1/EP8): compile-time constants
// plus the shared warp-count derivations (deepep_config_derived.cuh).
//
// Differences from the Kimi config: 256 routed experts (32 local), hidden
// 6144, and expert_alignment 64 — the TRTLLM grouped-GEMM M-tile
// (GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT), so the dispatch's expert-major
// recv segments land pre-aligned for the grouped FP8 GEMM chain.
#pragma once

#include <cstdint>

namespace deepep_shim::cfg_glm52 {

// Model / topology (GLM5.2, DP1/EP8).
inline constexpr int kNumRanks = 8;
inline constexpr int kNumExperts = 256;
inline constexpr int kNumLocalExperts = kNumExperts / kNumRanks;  // 32
inline constexpr int kNumTopk = 8;
inline constexpr int kHidden = 6144;
inline constexpr int kHiddenBytes = kHidden * 2;  // bf16 payload, no FP8 SF
inline constexpr int kExpertAlignment = 64;       // TRTLLM grouped-GEMM M-tile

// Comm tuning and device facts: identical to the Kimi config (same H200
// node class, same direct single-node mode).
inline constexpr int kKernelQPs = 9;
inline constexpr int kAllocatedQPs = 17;
inline constexpr int64_t kTimeoutCycles = 200'000'000'000;
inline constexpr int kDeviceSms = 132;
inline constexpr int kSmemBytes = 232448;

inline constexpr int kDecodeMaxTokens = 128;
inline constexpr int kDecodeNumSms = 32;
// GLM5.2 prefill rides the decode path token-by-token (see
// docs/models/glm52/ep8-deepep-moe.md); the shim's prefill entry points are
// never called, so size them at the decode cap instead of reserving ~200 MB
// of symmetric memory per rank for a dead path. Rebake if a real chunked
// prefill dispatch ever lands.
inline constexpr int kPrefillMaxTokens = kDecodeMaxTokens;
inline constexpr int kPrefillNumSms = 64;

#include "deepep_config_derived.cuh"

}  // namespace deepep_shim::cfg_glm52
