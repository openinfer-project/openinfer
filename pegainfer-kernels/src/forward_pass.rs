//! `typed_forward_pass!` — declarative DSL for typed forward passes.
//!
//! Expands a sequence of `op (inputs => outputs, extras);` into typed_ops calls.
//! Each line is a single GPU kernel dispatch; the macro handles `?` propagation
//! and borrow ordering.
//!
//! # Supported ops
//!
//! | DSL op | Expands to | Signature |
//! |--------|-----------|-----------|
//! | `gemm (x => y, w)` | `typed_ops::gemm_graphsafe_into(ctx, w, x, y)?` | W @ X → Y |
//! | `rms_norm (x => y, w)` | `typed_ops::rms_norm_into(ctx, x, w, eps, y)` | RMSNorm(X, W) → Y |
//! | `fused_add_norm (h, r => y, w)` | `typed_ops::fused_add_rms_norm_into(ctx, h, r, w, eps, y)` | H += R; Norm(H) → Y |
//! | `add (a, b => y)` | `typed_ops::add_into(ctx, a, b, y)?` | A + B → Y |
//! | `silu_mul<I> (gu => y)` | `typed_ops::silu_mul_fused_into::<I>(ctx, gu, y)` | SiLU(gate) * up → Y |
//! | `swap (a, b)` | `std::mem::swap(a, b)` | Swap two buffers |
//!
//! # Example
//!
//! ```ignore
//! typed_forward_pass! {
//!     ctx, eps;
//!     rms_norm      (s.hidden    => s.normed,      w.input_norm);
//!     gemm          (s.normed    => s.qkv_a,       w.fused_qkv_a_proj);
//!     rms_norm      (s.q_a       => s.q_a_normed,  w.q_a_norm);
//!     gemm          (s.q_a_normed => s.q_proj,     w.q_b_proj);
//!     gemm          (s.attn_out  => s.projected,   w.o_proj);
//! }
//! ```

#[macro_export]
macro_rules! typed_forward_pass {
    // Entry: parse context + eps, then process steps
    (
        $ctx:expr, $eps:expr;
        $($rest:tt)*
    ) => {
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };

    // Base case: no more steps
    (@steps $ctx:expr, $eps:expr; ) => {};

    // ── gemm: Y = W @ X ──
    (@steps $ctx:expr, $eps:expr;
        gemm ( $x:expr => $y:expr, $w:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::gemm_graphsafe_into($ctx, &$w, $x, $y)?;
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };

    // ── gemm_prefill: Y = W @ X (uses workspace cuBLAS handle) ──
    (@steps $ctx:expr, $eps:expr;
        gemm_prefill ( $x:expr => $y:expr, $w:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::gemm_into($ctx, &$w, $x, $y)?;
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };

    // ── rms_norm: Y = RMSNorm(X, W) ──
    (@steps $ctx:expr, $eps:expr;
        rms_norm ( $x:expr => $y:expr, $w:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::rms_norm_into($ctx, $x, &$w, $eps, $y);
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };

    // ── fused_add_norm: H += R; Y = RMSNorm(H, W) ──
    (@steps $ctx:expr, $eps:expr;
        fused_add_norm ( $h:expr, $r:expr => $y:expr, $w:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::fused_add_rms_norm_into($ctx, $h, $r, &$w, $eps, $y);
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };

    // ── add: Y = A + B ──
    (@steps $ctx:expr, $eps:expr;
        add ( $a:expr, $b:expr => $y:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::add_into($ctx, $a, $b, $y)?;
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };

    // ── silu_mul: Y = SiLU(gate) * up, from fused [2*I, bs] buffer ──
    (@steps $ctx:expr, $eps:expr;
        silu_mul < $inter:ident > ( $gu:expr => $y:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::silu_mul_fused_into::<$inter>($ctx, $gu, $y);
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };

    // ── swap: exchange two buffers (for residual ping-pong) ──
    (@steps $ctx:expr, $eps:expr;
        swap ( $a:expr, $b:expr );
        $($rest:tt)*
    ) => {
        ::std::mem::swap($a, $b);
        $crate::typed_forward_pass!(@steps $ctx, $eps; $($rest)*)
    };
}
