use std::ffi::{c_char, c_int, c_void};

/// Raw bindings for the GLM5.2 DeepEP elastic shim
/// (`csrc/glm52_deepep/glm52_deepep.h`).
///
/// All functions return 0 on success; on failure the thread-local message is
/// readable via [`glm52_deepep_last_error`]. Use the safe wrapper in
/// `openinfer_kernels::ops` instead of calling these directly.
#[repr(C)]
pub struct Glm52DeepEpCtx {
    _opaque: [u8; 0],
}

/// Compile-time capacities of the baked GLM5.2 config.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Glm52DeepEpInfo {
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
    pub fn glm52_deepep_last_error() -> *const c_char;

    pub fn glm52_deepep_info(out: *mut Glm52DeepEpInfo);

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

    pub fn glm52_deepep_prefill_dispatch_send(
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
    ) -> c_int;

    pub fn glm52_deepep_prefill_wait_counts(
        ctx: *mut Glm52DeepEpCtx,
        num_recv_tokens: *mut i32,
        num_expanded_tokens: *mut i32,
    ) -> c_int;

    pub fn glm52_deepep_prefill_dispatch_recv(
        ctx: *mut Glm52DeepEpCtx,
        stream: *mut c_void,
        num_recv_tokens: i32,
        psum_rank: *const i32,
        psum_expert: *const i32,
        recv_x: *mut c_void,
        recv_topk_weights: *mut f32,
        recv_src_metadata: *mut i32,
    ) -> c_int;

    pub fn glm52_deepep_prefill_combine(
        ctx: *mut Glm52DeepEpCtx,
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
