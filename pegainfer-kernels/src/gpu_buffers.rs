//! `gpu_buffers!` — declarative macro that generates buffer structs + allocation.
//!
//! Given a list of typed fields, the macro generates:
//! 1. The struct definition
//! 2. `fn new(ctx, batch_size) -> Result<Self>` — allocates every field
//! 3. `fn set_batch_size(&mut self, bs)` — updates `seq_len` on all `GpuTensor` fields
//!
//! Dimension expressions must be wrapped in `{ }` braces:
//!
//! ```ignore
//! gpu_buffers! {
//!     pub(crate) struct MlaScratch {
//!         hidden:    GpuTensor<{ KIMI_K2_HIDDEN }>,
//!         normed:    GpuTensor<{ KIMI_K2_HIDDEN }>,
//!         qkv_a:     GpuTensor<{ KIMI_K2_MLA_QKV_A_OUT }>,
//!         router_logits: GpuRawSlice<{ KIMI_K2_ROUTED_EXPERTS }>,
//!         router_idx:    GpuRawSliceI32<{ KIMI_K2_TOPK }>,
//!     }
//! }
//! ```

/// Declares a GPU buffer struct with auto-generated `new()` and `set_batch_size()`.
///
/// Supports three field kinds:
/// - `GpuTensor<{ EXPR }>` — bf16 activation buffer
/// - `GpuRawSlice<{ EXPR }>` — f32 raw buffer
/// - `GpuRawSliceI32<{ EXPR }>` — i32 raw buffer
#[macro_export]
macro_rules! gpu_buffers {
    (
        $(#[$struct_meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident : $kind:ident < { $dim:expr } >
            ),* $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        $vis struct $name {
            $(
                $(#[$field_meta])*
                $field_vis $field : $crate::tensor:: $kind < { $dim } >,
            )*
        }

        impl $name {
            $vis fn new(
                ctx: &$crate::tensor::DeviceContext,
                batch_size: usize,
            ) -> ::anyhow::Result<Self> {
                Ok(Self {
                    $(
                        $field: $crate::gpu_buffers!(@alloc ctx, batch_size, $kind, $dim),
                    )*
                })
            }

            $vis fn set_batch_size(&mut self, bs: usize) {
                $(
                    $crate::gpu_buffers!(@set_bs self, bs, $field, $kind);
                )*
            }
        }
    };

    // ── internal: allocation dispatch ──
    (@alloc $ctx:ident, $bs:ident, GpuTensor, $dim:expr) => {
        $crate::tensor::GpuTensor::<{ $dim }>::zeros($ctx, $bs)?
    };
    (@alloc $ctx:ident, $bs:ident, GpuRawSlice, $dim:expr) => {
        $crate::tensor::GpuRawSlice::<{ $dim }>::zeros($ctx, $bs)?
    };
    (@alloc $ctx:ident, $bs:ident, GpuRawSliceI32, $dim:expr) => {
        $crate::tensor::GpuRawSliceI32::<{ $dim }>::zeros($ctx, $bs)?
    };

    // ── internal: set_batch_size dispatch ──
    (@set_bs $self:ident, $bs:ident, $field:ident, GpuTensor) => {
        $self.$field.seq_len = $bs;
    };
    (@set_bs $self:ident, $bs:ident, $field:ident, GpuRawSlice) => {
        $self.$field.batch_size = $bs;
    };
    (@set_bs $self:ident, $bs:ident, $field:ident, GpuRawSliceI32) => {
        $self.$field.batch_size = $bs;
    };
}
