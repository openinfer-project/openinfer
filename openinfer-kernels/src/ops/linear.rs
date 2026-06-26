use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, HiddenStatesRef};

/// Generic strided-batched bf16 GEMM — one `cublasGemmStridedBatchedEx` on the
/// graph-safe (workspace-free, decode/capture) cuBLAS handle. `lda`/`ldb`/`ldc`
/// are cuBLAS column-major leading dims and `stride_*` are per-batch element
/// strides; each buffer must hold at least `stride * batch` elements. This is
/// the shared primitive for per-head batched GEMMs (MLA absorption
/// `q_nope @ W_UK` and `latent @ W_UV`, batch = head count) so model crates do
/// not hand-roll bespoke batched-GEMM kernels. `transpose_*` map to CUBLAS_OP_T.
///
/// Args use BLAS notation (a/b/c, m/n/k) to match the cuBLAS call it wraps.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn gemm_strided_batched_bf16(
    ctx: &DeviceContext,
    transpose_a: bool,
    transpose_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &CudaSlice<bf16>,
    lda: usize,
    stride_a: usize,
    b: &CudaSlice<bf16>,
    ldb: usize,
    stride_b: usize,
    c: &mut CudaSlice<bf16>,
    ldc: usize,
    stride_c: usize,
    batch: usize,
) -> Result<()> {
    ensure!(
        m > 0 && n > 0 && k > 0 && batch > 0,
        "gemm_strided_batched_bf16 empty dims: m={m} n={n} k={k} batch={batch}"
    );
    ensure!(
        a.len() >= stride_a * batch,
        "gemm_strided_batched_bf16 A too small: have {}, need {}",
        a.len(),
        stride_a * batch
    );
    ensure!(
        b.len() >= stride_b * batch,
        "gemm_strided_batched_bf16 B too small: have {}, need {}",
        b.len(),
        stride_b * batch
    );
    ensure!(
        c.len() >= stride_c * batch,
        "gemm_strided_batched_bf16 C too small: have {}, need {}",
        c.len(),
        stride_c * batch
    );
    let (a_ptr, _ga) = a.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.device_ptr(&ctx.stream);
    let (c_ptr, _gc) = c.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::gemm_strided_batched_bf16_cuda(
            i32::from(transpose_a),
            i32::from(transpose_b),
            m as i32,
            n as i32,
            k as i32,
            a_ptr as *const ffi::Half,
            lda as i32,
            stride_a as i64,
            b_ptr as *const ffi::Half,
            ldb as i32,
            stride_b as i64,
            c_ptr as *mut ffi::Half,
            ldc as i32,
            stride_c as i64,
            batch as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(
        status == 0,
        "gemm_strided_batched_bf16 failed: status={status} (m={m} n={n} k={k} batch={batch})"
    );
    Ok(())
}

/// GEMMs at or below this N consult the cublasLt plan cache. 32 covers the
/// decode buckets where cuBLAS's GemmEx heuristic picks badly (worst at
/// N in [8, 16], where it skips split-K entirely); above that the GEMMs are
/// wide enough that the default selection is fine.
pub const GEMM_LT_MAX_N: usize = 32;

/// Mirrors GEMM_LT_UNTUNED in csrc/shared/linear.cu.
const GEMM_LT_UNTUNED: i32 = -1;

/// Select the fastest cublasLt algo for a (num_rows, n, cols) GEMM and cache
/// it for this thread's subsequent decode GEMMs of that shape.
///
/// `samples` are (weight, row_offset) pairs of identically shaped slices.
/// Passing several layers' weights keeps the timing loop out of L2, so the
/// ranking matches steady-state decode where weights stream from DRAM. Must
/// run on the executor thread before CUDA Graph capture.
pub fn gemm_lt_tune(
    ctx: &DeviceContext,
    samples: &[(&DeviceMatrix, usize)],
    num_rows: usize,
    n: usize,
) -> Result<()> {
    let (first, _) = samples
        .first()
        .ok_or_else(|| anyhow::anyhow!("gemm_lt_tune requires at least one weight sample"))?;
    let cols = first.cols;
    let mut guards = Vec::with_capacity(samples.len());
    let mut ptrs: Vec<*const ffi::Half> = Vec::with_capacity(samples.len());
    for (weight, row_offset) in samples {
        assert_eq!(weight.cols, cols);
        assert!(row_offset + num_rows <= weight.rows);
        let (w_ptr, guard) = weight.data.device_ptr(&ctx.stream);
        ptrs.push(
            (w_ptr + (row_offset * cols * std::mem::size_of::<half::bf16>()) as u64)
                as *const ffi::Half,
        );
        guards.push(guard);
    }
    let status = unsafe {
        ffi::gemm_lt_tune_cuda(
            ptrs.as_ptr(),
            ptrs.len() as i32,
            num_rows as i32,
            n as i32,
            cols as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if status != 0 {
        bail!(
            "cublasLt GEMM tuning failed: status={}, m={}, n={}, k={}",
            status,
            num_rows,
            n,
            cols
        );
    }
    Ok(())
}

/// Mirrors the GEMM_LT_PIN_* sentinels in csrc/shared/linear.cu.
const GEMM_LT_PIN_UNTUNED: i32 = -1;
const GEMM_LT_PIN_UNSUPPORTED: i32 = -2;

/// Pin one cublasLt algo for `(num_rows, cols)`, selected by the heuristic at `rep_n` (shape-only),
/// reused for every N. Run on the GEMM-issuing thread (the plan cache is thread-local).
pub fn gemm_lt_pin_tune(num_rows: usize, rep_n: usize, cols: usize) -> Result<()> {
    let status = unsafe { ffi::gemm_lt_pin_tune_cuda(num_rows as i32, rep_n as i32, cols as i32) };
    if status != 0 {
        bail!(
            "cublasLt pin tuning failed: status={}, m={}, rep_n={}, k={}",
            status,
            num_rows,
            rep_n,
            cols
        );
    }
    Ok(())
}

/// Run the pinned `(rows, cols)` algo at this N. `Ok(false)` = algo can't serve this N (caller
/// falls back); bails if never pinned.
pub fn gemm_lt_pin_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<bool> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    let status = unsafe {
        ffi::gemm_lt_pin_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            weight.rows as i32,
            x.seq_len as i32,
            weight.cols as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    match status {
        0 => Ok(true),
        GEMM_LT_PIN_UNSUPPORTED => Ok(false),
        GEMM_LT_PIN_UNTUNED => bail!(
            "gemm_lt_pin_into_checked: (m={}, k={}) was never pinned — call gemm_lt_pin_tune first",
            weight.rows,
            weight.cols
        ),
        s if s >= 100_000 => bail!(
            "cublasLt pin GEMM failed: cublas_status={}, m={}, n={}, k={}",
            s - 100_000,
            weight.rows,
            x.seq_len,
            weight.cols
        ),
        s => bail!(
            "cublasLt pin GEMM launch failed: cuda_status={}, m={}, n={}, k={}",
            s,
            weight.rows,
            x.seq_len,
            weight.cols
        ),
    }
}

/// Diagnostics: the pinned algo's `[tile_id, stages_id, splitk_num, reduction_scheme]` for
/// `(rows, cols)`, or `None` if never pinned.
pub fn gemm_lt_pin_inspect(rows: usize, cols: usize) -> Option<[i32; 4]> {
    let mut out = [0i32; 4];
    let status =
        unsafe { ffi::gemm_lt_pin_inspect_cuda(rows as i32, cols as i32, out.as_mut_ptr()) };
    if status == 0 { Some(out) } else { None }
}

/// GEMM on a row sub-range of a weight matrix: Y = W[row_offset..row_offset+M, :] @ X
pub fn gemm_rows_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_rows_into_checked(ctx, weight, row_offset, num_rows, x, out)
        .expect("GEMM row-range launch failed");
}

/// Checked row-range GEMM. New hot paths should use this form so cuBLAS launch
/// failures surface at the operator boundary instead of at a later collective.
pub fn gemm_rows_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert!(row_offset + num_rows <= weight.rows);
    assert_eq!(weight.cols, x.hidden_dim);
    assert_eq!(out.hidden_dim, num_rows);
    assert_eq!(out.seq_len, x.seq_len);

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let w_sub = w_ptr + (row_offset * weight.cols * std::mem::size_of::<half::bf16>()) as u64;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_sub as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        num_rows,
        x.seq_len,
        weight.cols,
        x.seq_len == 1,
        ctx,
    )
}

/// Matrix-vector multiplication: y = A @ x (via cuBLAS GEMM with N=1)
/// A: (M, K) row-major, x: (K,), y: (M,)
pub fn gemv(ctx: &DeviceContext, a: &DeviceMatrix, x: &DeviceVec, y: &mut DeviceVec) -> Result<()> {
    assert_eq!(a.cols, x.len, "A cols {} != x len {}", a.cols, x.len);
    assert_eq!(a.rows, y.len, "A rows {} != y len {}", a.rows, y.len);

    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        a_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        a.rows,
        1,
        a.cols,
        true,
        ctx,
    )
}
/// Linear layer: y = weight @ x
pub fn linear(ctx: &DeviceContext, x: &DeviceVec, weight: &DeviceMatrix) -> Result<DeviceVec> {
    let mut y = DeviceVec::zeros(ctx, weight.rows)?;
    gemv(ctx, weight, x, &mut y)?;
    Ok(y)
}

/// GEMM: Y = weight @ X (batched linear projection)
/// weight: [out_dim, in_dim] row-major, X: HiddenStates [in_dim, seq_len], Y: HiddenStates [out_dim, seq_len]
pub fn gemm(ctx: &DeviceContext, weight: &DeviceMatrix, x: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, weight.rows, x.seq_len)?;
    gemm_into_checked(ctx, weight, x, &mut out)?;
    Ok(out)
}

/// Per-token GEMM: each row is computed through the same N=1 cuBLAS boundary
/// used by decode. This is useful for narrow batch-shaped accuracy gates where
/// row-wise parity with the serial decode oracle matters more than throughput.
pub fn gemm_per_token(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, weight.rows, x.seq_len)?;
    gemm_per_token_into_checked(ctx, weight, x, &mut out)?;
    Ok(out)
}

/// GEMM into pre-allocated output buffer (zero allocation).
/// For seq_len=1, uses the graph-safe cuBLAS handle (no workspace) for lower
/// latency while preserving numerical parity with the prefill path.
pub fn gemm_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_into_checked(ctx, weight, x, out).expect("GEMM launch failed");
}

/// Checked GEMM using the default policy: graph-safe handle for single-token
/// decode, workspace-backed handle for prefill/batched projections.
pub fn gemm_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_into_with_policy(ctx, weight, x, out, x.seq_len == 1)
}

/// GEMM for a contiguous token range in `x` into a compact output buffer.
pub fn gemm_token_range_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    token_offset: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert!(
        token_offset + out.seq_len <= x.seq_len,
        "token range [{}..{}) exceeds input seq_len {}",
        token_offset,
        token_offset + out.seq_len,
        x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_base, _gx) = x.data.device_ptr(&ctx.stream);
    let x_ptr = x_base + (token_offset * x.hidden_dim * std::mem::size_of::<half::bf16>()) as u64;
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        weight.rows,
        out.seq_len,
        weight.cols,
        out.seq_len == 1,
        ctx,
    )
}

/// Checked GEMM that always uses the workspace-free cuBLAS handle. Kimi decode
/// uses this for active-batch sizes 1..=4 so graph-readiness is not tied to a
/// bs1-only condition.
pub fn gemm_graphsafe_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_ref_into_with_policy(ctx, weight, x.as_ref(), out, true)
}

pub fn gemm_graphsafe_ref_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: HiddenStatesRef<'_>,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_ref_into_with_policy(ctx, weight, x, out, true)
}

pub fn gemm_per_token_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        let status = ffi::gemm_per_token_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            weight.rows as i32,
            x.seq_len as i32,
            weight.cols as i32,
            crate::tensor::active_cu_stream(ctx),
        );
        if status != 0 {
            if status >= 100_000 {
                bail!(
                    "cuBLAS per-token GEMM failed: cublas_status={}, m={}, batch={}, k={}",
                    status - 100_000,
                    weight.rows,
                    x.seq_len,
                    weight.cols
                );
            }
            bail!(
                "CUDA per-token GEMM launch failed: cuda_status={}, m={}, batch={}, k={}",
                status,
                weight.rows,
                x.seq_len,
                weight.cols
            );
        }
    }
    Ok(())
}

fn gemm_into_with_policy(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
    graphsafe: bool,
) -> Result<()> {
    gemm_ref_into_with_policy(ctx, weight, x.as_ref(), out, graphsafe)
}

fn gemm_ref_into_with_policy(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: HiddenStatesRef<'_>,
    out: &mut HiddenStates,
    graphsafe: bool,
) -> Result<()> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        weight.rows,
        x.seq_len,
        weight.cols,
        graphsafe,
        ctx,
    )
}

/// Process-global numeric policy for projection GEMMs (atomics, not thread-local: set on the main
/// thread before worker construction, read on the worker). `Tuned` (default) = production path;
/// `Pin` = batch-invariant pinned algo (per-token fallback when it can't serve an N, or under a
/// stream override); `PerToken` = N=1 oracle. Set via `set_numeric_policy` before executor
/// construction / graph capture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NumericPolicy {
    Tuned = 0,
    Pin = 1,
    PerToken = 2,
}

static NUMERIC_POLICY: AtomicU8 = AtomicU8::new(NumericPolicy::Tuned as u8);
static PIN_SERVED: AtomicU64 = AtomicU64::new(0);
static PIN_FALLBACK: AtomicU64 = AtomicU64::new(0);
type FallbackShapeMap = Mutex<BTreeMap<(usize, usize, usize), u64>>;
/// Per-(m, n, k) fallback tally; only the fallback branch touches it (served path stays lock-free).
static PIN_FALLBACK_SHAPES: OnceLock<FallbackShapeMap> = OnceLock::new();

fn pin_fallback_shapes_map() -> &'static FallbackShapeMap {
    PIN_FALLBACK_SHAPES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Set the process-global policy. Must run before executor construction / graph capture (the
/// captured algo is the one live at capture). Tests drive baseline/pin/per-token through it.
pub fn set_numeric_policy(p: NumericPolicy) {
    NUMERIC_POLICY.store(p as u8, Ordering::Release);
}

pub fn numeric_policy() -> NumericPolicy {
    match NUMERIC_POLICY.load(Ordering::Acquire) {
        1 => NumericPolicy::Pin,
        2 => NumericPolicy::PerToken,
        _ => NumericPolicy::Tuned,
    }
}

/// `(pin_served, pin_fallback)` projection-GEMM counts: process-global, mixed across prefill +
/// decode + unified and across TP ranks (not decode-specific). Decode counts the graph capture,
/// not replays; read quiesced.
pub fn pin_counters() -> (u64, u64) {
    (
        PIN_SERVED.load(Ordering::Relaxed),
        PIN_FALLBACK.load(Ordering::Relaxed),
    )
}

pub fn reset_pin_counters() {
    PIN_SERVED.store(0, Ordering::Relaxed);
    PIN_FALLBACK.store(0, Ordering::Relaxed);
    pin_fallback_shapes_map().lock().unwrap().clear();
}

/// `((m, n, k), count)` per shape that fell back to per-token. Quiesced and single-threaded, an
/// empty list means no recorded fallback; under TP/concurrency the map and counter can momentarily
/// disagree.
pub fn pin_fallback_shapes() -> Vec<((usize, usize, usize), u64)> {
    pin_fallback_shapes_map()
        .lock()
        .unwrap()
        .iter()
        .map(|(&shape, &count)| (shape, count))
        .collect()
}

fn record_pin_fallback(m: usize, n: usize, k: usize, reason: &str) {
    PIN_FALLBACK.fetch_add(1, Ordering::Relaxed);
    let mut shapes = pin_fallback_shapes_map().lock().unwrap();
    let count = shapes.entry((m, n, k)).or_insert(0);
    if *count == 0 {
        log::warn!("batch-invariant pin fell back to per-token at (m={m}, n={n}, k={k}): {reason}");
    }
    *count += 1;
}

/// Fixed representative N at which every (M,K) pin is resolved — one algo for ALL
/// N (NOT first-call-N, which would make the pinned algo call-order-dependent).
const PIN_REP_N: i32 = 32;

/// Pin one `(num_rows, cols)` algo at the canonical `PIN_REP_N`, reused for all N.
pub fn gemm_lt_pin_warmup(num_rows: usize, cols: usize) -> Result<()> {
    gemm_lt_pin_tune(num_rows, PIN_REP_N as usize, cols)
}

/// `Pin` policy: lazily pin (m,k) at PIN_REP_N, run at live N, per-token fallback if it can't serve N.
fn launch_gemm_pin(
    w_ptr: *const ffi::Half,
    x_ptr: *const ffi::Half,
    y_ptr: *mut ffi::Half,
    m: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
) -> Result<()> {
    // cuBLASLt (gemm_lt_pin) risks Xid 31 under a stream override; fall back to per-token.
    if crate::tensor::has_stream_override() {
        record_pin_fallback(
            m,
            n,
            k,
            "stream override active (cuBLASLt would risk Xid 31)",
        );
        return launch_gemm_pertoken(w_ptr, x_ptr, y_ptr, m, n, k, ctx);
    }
    // The eager warmup (gemm_lt_pin_warmup in tune_decode_gemm_algos) pins every decode shape before
    // capture; a lazy tune here allocates, illegal mid-capture — so refuse under capture, don't 900.
    if gemm_lt_pin_inspect(m, k).is_none() {
        match unsafe { ffi::stream_is_capturing_cuda(crate::tensor::active_cu_stream(ctx)) } {
            0 => gemm_lt_pin_tune(m, PIN_REP_N as usize, k)?,
            s if s < 0 => bail!(
                "cudaStreamIsCapturing query failed: status={}, m={m}, k={k}",
                -s
            ),
            _ => bail!(
                "batch-invariant pin: ({m},{k}) reached graph capture unpinned — the decode pin warmup must run before capture"
            ),
        }
    }
    unsafe {
        let status = ffi::gemm_lt_pin_cuda(
            w_ptr,
            x_ptr,
            y_ptr,
            m as i32,
            n as i32,
            k as i32,
            crate::tensor::active_cu_stream(ctx),
        );
        match status {
            0 => {
                PIN_SERVED.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            GEMM_LT_PIN_UNSUPPORTED | GEMM_LT_PIN_UNTUNED => {
                record_pin_fallback(m, n, k, "pinned algo cannot serve this N");
                let st = ffi::gemm_per_token_cuda(
                    w_ptr,
                    x_ptr,
                    y_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    crate::tensor::active_cu_stream(ctx),
                );
                if st != 0 {
                    bail!("pin per-token fallback failed: status={st}, m={m}, n={n}, k={k}");
                }
                Ok(())
            }
            s if s >= 100_000 => {
                bail!(
                    "cuBLAS pin GEMM failed: cublas_status={}, m={m}, n={n}, k={k}",
                    s - 100_000
                )
            }
            s => bail!("CUDA pin GEMM launch failed: cuda_status={s}, m={m}, n={n}, k={k}"),
        }
    }
}

/// `PerToken` policy: the N=1-per-column oracle (invariant by construction).
fn launch_gemm_pertoken(
    w_ptr: *const ffi::Half,
    x_ptr: *const ffi::Half,
    y_ptr: *mut ffi::Half,
    m: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
) -> Result<()> {
    unsafe {
        let status = ffi::gemm_per_token_cuda(
            w_ptr,
            x_ptr,
            y_ptr,
            m as i32,
            n as i32,
            k as i32,
            crate::tensor::active_cu_stream(ctx),
        );
        if status != 0 {
            if status >= 100_000 {
                bail!(
                    "cuBLAS per-token GEMM failed: cublas_status={}, m={m}, n={n}, k={k}",
                    status - 100_000
                );
            }
            bail!("CUDA per-token GEMM launch failed: cuda_status={status}, m={m}, n={n}, k={k}");
        }
    }
    Ok(())
}

fn launch_gemm(
    w_ptr: *const ffi::Half,
    x_ptr: *const ffi::Half,
    y_ptr: *mut ffi::Half,
    m: usize,
    n: usize,
    k: usize,
    graphsafe: bool,
    ctx: &DeviceContext,
) -> Result<()> {
    // Non-Tuned policies route every N through the pinned/per-token path; Tuned falls through.
    match numeric_policy() {
        NumericPolicy::Pin => return launch_gemm_pin(w_ptr, x_ptr, y_ptr, m, n, k, ctx),
        NumericPolicy::PerToken => return launch_gemm_pertoken(w_ptr, x_ptr, y_ptr, m, n, k, ctx),
        NumericPolicy::Tuned => {}
    }
    unsafe {
        // Small-N projections run the cublasLt algo selected by gemm_lt_tune.
        // Shapes this thread never tuned report GEMM_LT_UNTUNED and keep their
        // existing cublasGemmEx path (and its capture behavior) unchanged.
        //
        // NOTE: gemm_lt is disabled when a stream override is active (SM-partition
        // concurrent mode). cuBLASLt has device-global state that conflicts when
        // two green-ctx streams run cublasLtMatmul concurrently, causing Xid 31.
        let mut status = if n <= GEMM_LT_MAX_N && !crate::tensor::has_stream_override() {
            ffi::gemm_lt_cuda(
                w_ptr,
                x_ptr,
                y_ptr,
                m as i32,
                n as i32,
                k as i32,
                crate::tensor::active_cu_stream(ctx),
            )
        } else {
            GEMM_LT_UNTUNED
        };
        if status == GEMM_LT_UNTUNED {
            status = if graphsafe {
                ffi::gemm_graphsafe_cuda(
                    w_ptr,
                    x_ptr,
                    y_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    crate::tensor::active_cu_stream(ctx),
                )
            } else {
                ffi::gemm_cuda(
                    w_ptr,
                    x_ptr,
                    y_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    crate::tensor::active_cu_stream(ctx),
                )
            };
        }
        if status != 0 {
            if status >= 100_000 {
                bail!(
                    "cuBLAS GEMM failed: cublas_status={}, m={}, n={}, k={}, graphsafe={}",
                    status - 100_000,
                    m,
                    n,
                    k,
                    graphsafe
                );
            }
            bail!(
                "CUDA GEMM launch failed: cuda_status={}, m={}, n={}, k={}, graphsafe={}",
                status,
                m,
                n,
                k,
                graphsafe
            );
        }
    }
    Ok(())
}
