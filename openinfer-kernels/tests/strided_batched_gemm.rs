//! GPU gate for the generic `gemm_strided_batched_bf16` op (one
//! `cublasGemmStridedBatchedEx`). Covers BOTH op modes that MLA absorption uses:
//! N (q_nope @ W_UK) and T (latent @ W_UV). The host reference replays cuBLAS's
//! column-major strided-batched semantics so the test pins the FFI plumbing
//! (dims, leading dims, per-batch strides, transpose flags), not just one shape.
//!
//!   cargo test --release -p openinfer-kernels --test strided_batched_gemm -- --nocapture

use half::bf16;
use openinfer_kernels::ops::gemm_strided_batched_bf16;
use openinfer_kernels::tensor::DeviceContext;

/// Deterministic value in [-1, 1) from a flat index — keeps inputs reproducible
/// without an RNG dependency.
fn g(seed: u64, i: usize) -> f32 {
    let mut h = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    ((h & 0xFFFF) as f32 / 32768.0) - 1.0
}

fn host_vec(seed: u64, len: usize) -> Vec<bf16> {
    (0..len).map(|i| bf16::from_f32(g(seed, i))).collect()
}

/// Column-major strided-batched GEMM reference matching cuBLAS semantics:
/// `C[b] = op_a(A[b]) @ op_b(B[b])`, each operand column-major with the given
/// leading dim and per-batch element stride.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn reference(
    transpose_a: bool,
    transpose_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &[bf16],
    lda: usize,
    stride_a: usize,
    b: &[bf16],
    ldb: usize,
    stride_b: usize,
    ldc: usize,
    stride_c: usize,
    batch: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; stride_c * batch];
    for bi in 0..batch {
        for col in 0..n {
            for row in 0..m {
                let mut acc = 0.0f32;
                for j in 0..k {
                    // op_a(A)[row, j]
                    let a_idx = if transpose_a {
                        bi * stride_a + j + row * lda
                    } else {
                        bi * stride_a + row + j * lda
                    };
                    // op_b(B)[j, col]
                    let b_idx = if transpose_b {
                        bi * stride_b + col + j * ldb
                    } else {
                        bi * stride_b + j + col * ldb
                    };
                    acc += a[a_idx].to_f32() * b[b_idx].to_f32();
                }
                c[bi * stride_c + row + col * ldc] = acc;
            }
        }
    }
    c
}

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn run_case(
    ctx: &DeviceContext,
    label: &str,
    transpose_a: bool,
    transpose_b: bool,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) {
    // Tightly packed column-major operands: lda/stride derive from the op shape.
    let (lda, stride_a) = if transpose_a { (k, k * m) } else { (m, m * k) };
    let (ldb, stride_b) = if transpose_b { (n, n * k) } else { (k, k * n) };
    let (ldc, stride_c) = (m, m * n);

    let a_host = host_vec(0x1111 ^ label.len() as u64, stride_a * batch);
    let b_host = host_vec(0x2222 ^ label.len() as u64, stride_b * batch);

    let mut a_dev = ctx.stream.alloc_zeros::<bf16>(a_host.len()).unwrap();
    let mut b_dev = ctx.stream.alloc_zeros::<bf16>(b_host.len()).unwrap();
    let mut c_dev = ctx.stream.alloc_zeros::<bf16>(stride_c * batch).unwrap();
    ctx.stream.memcpy_htod(&a_host, &mut a_dev).unwrap();
    ctx.stream.memcpy_htod(&b_host, &mut b_dev).unwrap();

    gemm_strided_batched_bf16(
        ctx,
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        &a_dev,
        lda,
        stride_a,
        &b_dev,
        ldb,
        stride_b,
        &mut c_dev,
        ldc,
        stride_c,
        batch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let c_host = ctx.stream.clone_dtoh(&c_dev).unwrap();
    let want = reference(
        transpose_a,
        transpose_b,
        m,
        n,
        k,
        &a_host,
        lda,
        stride_a,
        &b_host,
        ldb,
        stride_b,
        ldc,
        stride_c,
        batch,
    );

    let mut max_abs = 0.0f32;
    for (got, exp) in c_host.iter().zip(want.iter()) {
        max_abs = max_abs.max((got.to_f32() - exp).abs());
    }
    println!("{label}: max|Δ| = {max_abs:.5} (m={m} n={n} k={k} batch={batch})");
    // bf16 inputs + f32 accumulate over k terms; abs tol scales with k.
    let tol = 0.02 * k as f32;
    assert!(max_abs <= tol, "{label} drift {max_abs} exceeds tol {tol}");
}

#[test]
fn strided_batched_bf16_matches_host_reference() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("no CUDA device; skipping");
        return;
    };

    // op_a = N: the MLA absorb-q_nope mode (ql_nope = q_nope @ W_UK), batch = heads.
    run_case(&ctx, "absorb_q_nope(N,N)", false, false, 4, 2, 3, 5);
    // op_a = T: the MLA v-up mode (v = latent @ W_UV^T), batch = heads.
    run_case(&ctx, "v_up(T,N)", true, false, 4, 2, 3, 5);
    // Wider, GLM-ish single-token decode shapes (n = 1) to exercise real strides.
    run_case(&ctx, "absorb_glm(N,N)", false, false, 512, 1, 192, 8);
    run_case(&ctx, "v_up_glm(T,N)", true, false, 256, 1, 512, 8);
}
