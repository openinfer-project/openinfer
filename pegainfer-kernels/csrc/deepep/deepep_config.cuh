// Kimi-K2 single-node 8×H200 DeepEP config: compile-time constants and the
// warp-count derivations that feed kernel template parameters.
//
// The derivations mirror the upstream JIT host logic exactly
// (csrc/kernels/elastic/{dispatch,combine}.hpp); deepep_ctx_create
// runtime-asserts the constexpr mirrors against the real layout classes so
// upstream layout changes fail loudly instead of corrupting buffers.
#pragma once

#include <cstdint>

namespace deepep_shim::cfg {

// Model / topology (Kimi-K2, TP1/DP8/EP8).
inline constexpr int kNumRanks = 8;
inline constexpr int kNumExperts = 384;
inline constexpr int kNumLocalExperts = kNumExperts / kNumRanks;  // 48
inline constexpr int kNumTopk = 8;
inline constexpr int kHidden = 7168;
inline constexpr int kHiddenBytes = kHidden * 2;  // bf16, no FP8 SF
inline constexpr int kExpertAlignment = 8;        // Marlin routing block size

// Comm tuning, mirroring upstream defaults for direct (single-node) mode:
// kernel QPs = min(num_sms, 8 + 1); allocated GIN contexts = 17 (non-hybrid).
inline constexpr int kKernelQPs = 9;
inline constexpr int kAllocatedQPs = 17;
// GPU-side timeout in cycles. Upstream bakes 100 s × device clock rate at JIT
// time; we fix ~100 s at the H200's ~2 GHz. Only affects hang detection.
inline constexpr int64_t kTimeoutCycles = 200'000'000'000;

// Device facts baked at compile time (H200 SXM, sm_90). ctx_create asserts
// the actual device meets these; epilogue/prologue grids are templated on
// kDeviceSms so it must be <= the real SM count for cooperative launches.
inline constexpr int kDeviceSms = 132;
inline constexpr int kSmemBytes = 232448;  // sharedMemPerBlockOptin

// Per-path configs. Decode runs without CPU sync at fixed worst-case shapes
// (graph-capturable); prefill syncs the CPU on receive counts so buffers are
// allocated at actual size. Prefill max tokens matches the retired PPLX cap
// (PPLX_MAX_DISPATCH_TOKENS = 2048).
inline constexpr int kDecodeMaxTokens = 128;
inline constexpr int kDecodeNumSms = 32;
inline constexpr int kPrefillMaxTokens = 2048;
inline constexpr int kPrefillNumSms = 64;

// ---------------------------------------------------------------------------
// Derivations (constexpr mirrors of the upstream host logic).
// ---------------------------------------------------------------------------

inline constexpr int kTMAAlign = 32;  // ptx::kNumTMAAlignBytes

constexpr int align_up(int a, int b) { return (a + b - 1) / b * b; }
constexpr int min_i(int a, int b) { return a < b ? a : b; }

// layout::TokenLayout::get_num_bytes<kWithMBarrier>() for our shapes
// (sf bytes are always 0; mbarrier is 8 bytes aligned up to 32).
constexpr int token_smem_bytes(int hidden_bytes, int topk, bool with_metadata,
                               bool with_mbarrier) {
    const int metadata_bytes = topk * 8 + (with_metadata ? (1 + topk) * 4 : 0);
    return align_up(hidden_bytes, kTMAAlign) + align_up(metadata_bytes, kTMAAlign) +
           (with_mbarrier ? kTMAAlign : 0);
}

inline constexpr int kDispatchTokenSmem = token_smem_bytes(kHiddenBytes, kNumTopk, true, true);
inline constexpr int kCombineTokenSmem = token_smem_bytes(kHiddenBytes, kNumTopk, false, true);
inline constexpr int kReduceTokenSmem = align_up(kHiddenBytes, kTMAAlign);

// get_num_notify_smem_bytes(num_ranks, num_experts) with kNumNotifyWarps = 4.
inline constexpr int kNumNotifyWarps = 4;
inline constexpr int kNotifySmemBytes =
    align_up(kNumRanks + kNumExperts, kNumNotifyWarps * 32) * static_cast<int>(sizeof(int));

// launch_dispatch warp derivation (direct mode).
constexpr int dispatch_warps(int num_sms) {
    return min_i(min_i((kSmemBytes - kNotifySmemBytes) / kDispatchTokenSmem,
                       32 - kNumNotifyWarps),
                 (512 + num_sms - 1) / num_sms);
}

inline constexpr int kCombineWarps = min_i(kSmemBytes / kCombineTokenSmem, 32);
inline constexpr int kCopyEpilogueWarps = min_i(kSmemBytes / kDispatchTokenSmem, 32);
inline constexpr int kReduceEpilogueWarps = min_i(kSmemBytes / kReduceTokenSmem, 32);
inline constexpr int kPrologueWarps = 8;
inline constexpr int kBarrierThreads = 512;

// Worst-case decode capacities (no CPU sync ⇒ fixed shapes), mirroring
// buffer.hpp's non-cached/no-sync branch.
inline constexpr int kDecodeWorstRecvTokens = kNumRanks * kDecodeMaxTokens;
inline constexpr int kDecodeWorstExpandedTokens = align_up(
    kNumRanks * kDecodeMaxTokens * min_i(kNumTopk, kNumLocalExperts) +
        (kExpertAlignment - 1) * kNumLocalExperts,
    kExpertAlignment);

inline constexpr int kPrefillWorstRecvTokens = kNumRanks * kPrefillMaxTokens;

}  // namespace deepep_shim::cfg
