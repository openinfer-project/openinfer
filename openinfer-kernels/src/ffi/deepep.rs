use std::ffi::{c_char, c_int, c_void};

/// Raw bindings for the DeepEP elastic shim (csrc/deepep/deepep.h).
///
/// All functions return 0 on success; on failure the thread-local message is
/// readable via [`deepep_last_error`]. Use the safe wrapper in
/// `ops::deepep` instead of calling these directly.
#[repr(C)]
pub struct DeepEpCtx {
    _opaque: [u8; 0],
}

/// Compile-time capacities of the baked Kimi-K2 config (see deepep.h).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeepEpInfo {
    pub num_ranks: i32,
    pub num_experts: i32,
    pub num_local_experts: i32,
    pub num_topk: i32,
    pub hidden: i32,
    pub expert_alignment: i32,
    pub decode_max_tokens_per_rank: i32,
    pub decode_worst_recv_tokens: i32,
    pub decode_worst_expanded_tokens: i32,
    pub prefill_max_tokens_per_rank: i32,
    pub prefill_worst_recv_tokens: i32,
    pub prologue_rank_count_len: i32,
    pub buffer_bytes: i64,
    pub workspace_bytes: i64,
}

unsafe extern "C" {
    pub fn deepep_last_error() -> *const c_char;

    pub fn deepep_info(out: *mut DeepEpInfo);

    pub fn deepep_unique_id(out: *mut u8) -> c_int;

    pub fn deepep_ctx_create(
        unique_id: *const u8,
        num_ranks: i32,
        rank_idx: i32,
        out: *mut *mut DeepEpCtx,
    ) -> c_int;

    pub fn deepep_ctx_destroy(ctx: *mut DeepEpCtx) -> c_int;

    pub fn deepep_decode_dispatch(
        ctx: *mut DeepEpCtx,
        stream: *mut c_void,
        x: *const c_void,
        topk_idx: *const i32,
        topk_weights: *const f32,
        num_tokens: i32,
        rank_count_scratch: *mut i32,
        dst_slot_scratch: *mut i32,
        psum_rank: *mut i32,
        psum_expert: *mut i32,
        recv_x: *mut c_void,
        recv_topk_weights: *mut f32,
        recv_src_metadata: *mut i32,
    ) -> c_int;

    pub fn deepep_decode_combine(
        ctx: *mut DeepEpCtx,
        stream: *mut c_void,
        x: *const c_void,
        src_metadata: *const i32,
        psum_rank: *const i32,
        combined_topk_idx: *const i32,
        num_tokens: i32,
        combined_x: *mut c_void,
    ) -> c_int;

    pub fn deepep_prefill_dispatch_send(
        ctx: *mut DeepEpCtx,
        stream: *mut c_void,
        x: *const c_void,
        topk_idx: *const i32,
        topk_weights: *const f32,
        num_tokens: i32,
        rank_count_scratch: *mut i32,
        dst_slot_scratch: *mut i32,
        psum_rank: *mut i32,
        psum_expert: *mut i32,
    ) -> c_int;

    pub fn deepep_prefill_wait_counts(
        ctx: *mut DeepEpCtx,
        num_recv_tokens: *mut i32,
        num_expanded_tokens: *mut i32,
    ) -> c_int;

    pub fn deepep_prefill_dispatch_recv(
        ctx: *mut DeepEpCtx,
        stream: *mut c_void,
        num_recv_tokens: i32,
        psum_rank: *const i32,
        psum_expert: *const i32,
        recv_x: *mut c_void,
        recv_topk_weights: *mut f32,
        recv_src_metadata: *mut i32,
    ) -> c_int;

    pub fn deepep_prefill_combine(
        ctx: *mut DeepEpCtx,
        stream: *mut c_void,
        x: *const c_void,
        src_metadata: *const i32,
        psum_rank: *const i32,
        num_recv_tokens: i32,
        combined_topk_idx: *const i32,
        num_tokens: i32,
        combined_x: *mut c_void,
    ) -> c_int;
}

/// GLM5.2 shim instantiation (csrc/deepep/deepep_glm52.h): same contract as
/// the Kimi symbols above with a distinct baked config and opaque context.
#[cfg(feature = "glm52")]
#[repr(C)]
pub struct Glm52DeepEpCtx {
    _opaque: [u8; 0],
}

#[cfg(feature = "glm52")]
unsafe extern "C" {
    pub fn glm52_deepep_last_error() -> *const c_char;

    pub fn glm52_deepep_info(out: *mut DeepEpInfo);

    pub fn glm52_deepep_unique_id(out: *mut u8) -> c_int;

    pub fn glm52_deepep_ctx_create(
        unique_id: *const u8,
        num_ranks: i32,
        rank_idx: i32,
        out: *mut *mut Glm52DeepEpCtx,
    ) -> c_int;

    pub fn glm52_deepep_ctx_destroy(ctx: *mut Glm52DeepEpCtx) -> c_int;

    pub fn glm52_deepep_decode_dispatch(
        ctx: *mut Glm52DeepEpCtx,
        stream: *mut c_void,
        x: *const c_void,
        topk_idx: *const i32,
        topk_weights: *const f32,
        num_tokens: i32,
        rank_count_scratch: *mut i32,
        dst_slot_scratch: *mut i32,
        psum_rank: *mut i32,
        psum_expert: *mut i32,
        recv_x: *mut c_void,
        recv_topk_weights: *mut f32,
        recv_src_metadata: *mut i32,
    ) -> c_int;

    pub fn glm52_deepep_decode_combine(
        ctx: *mut Glm52DeepEpCtx,
        stream: *mut c_void,
        x: *const c_void,
        src_metadata: *const i32,
        psum_rank: *const i32,
        combined_topk_idx: *const i32,
        num_tokens: i32,
        combined_x: *mut c_void,
    ) -> c_int;
}
