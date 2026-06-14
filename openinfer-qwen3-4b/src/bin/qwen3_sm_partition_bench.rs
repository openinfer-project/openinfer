//! Green Context SM-partition concurrency benchmark using real Qwen3
//! attention kernels (paged decode + paged prefill).
//!
//! Measures speedup of running prefill and decode concurrently on separate SM
//! partitions vs serial execution. Uses the same FlashInfer kernels as prod.
//!
//! Build:
//!   cargo build --release -p openinfer-qwen3-4b --bin qwen3_sm_partition_bench
//!
//! Run:
//!   cargo run --release -p openinfer-qwen3-4b --bin qwen3_sm_partition_bench

use std::ptr;

use anyhow::{Result, bail};
use cudarc::driver::sys::{self, CUdevice, CUresult, CUstream};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use openinfer_kernels::ffi;
use openinfer_kernels::ops::PrefillPagedPlan;
use openinfer_kernels::paged_kv::PagedKvLayout;
use openinfer_kernels::tensor::{DeviceContext, DeviceVec, HiddenStates};
use openinfer_qwen3_4b::kernel_bench::{HEAD_DIM, NUM_KV_HEADS, NUM_QO_HEADS, PAGE_SIZE};

// ─── Configuration ──────────────────────────────────────────────────────────

const REPS: u64 = 10;
const WARMUP: u64 = 3;

// Decode workload configs: (batch_size, kv_len)
const DECODE_CONFIGS: &[(usize, usize)] = &[(1, 2048), (4, 2048), (8, 2048), (4, 4096), (8, 4096)];

// Prefill workload configs: (batch_size, seq_len)
const PREFILL_CONFIGS: &[(usize, usize)] = &[(1, 256), (1, 512), (1, 1024), (1, 2048)];

// ─── Helpers ────────────────────────────────────────────────────────────────

fn check_cu(result: CUresult, msg: &str) -> Result<()> {
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("{msg}: CUresult = {result:?}");
    }
    Ok(())
}

fn patterned_bf16(len: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| bf16::from_f32((((i % 251) as f32) - 125.0) * scale))
        .collect()
}

fn rope_cache_bf16(seq_len: usize, cos: bool) -> Vec<bf16> {
    let half_dim = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / 1_000_000.0f32.powf(i as f32 * 2.0 / HEAD_DIM as f32))
        .collect();
    let mut out = vec![bf16::ZERO; seq_len * HEAD_DIM];
    for pos in 0..seq_len {
        let base = pos * HEAD_DIM;
        for (i, inv_freq) in inv_freq.iter().copied().enumerate() {
            let angle = pos as f32 * inv_freq;
            let value = if cos { angle.cos() } else { angle.sin() };
            let value = bf16::from_f32(value);
            out[base + i] = value;
            out[base + i + half_dim] = value;
        }
    }
    out
}

// ─── Green Context partition ────────────────────────────────────────────────

struct SmPartition {
    sm_decode: u32,
    sm_prefill: u32,
    gctx_decode: sys::CUgreenCtx,
    gctx_prefill: sys::CUgreenCtx,
    stream_decode: CUstream,
    stream_prefill: CUstream,
}

impl SmPartition {
    fn create(device: CUdevice, sm_res: &sys::CUdevResource, sm_for_decode: u32) -> Result<Self> {
        let mut nb: u32 = 1;
        let mut grp_decode: sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut grp_prefill: sys::CUdevResource = unsafe { std::mem::zeroed() };

        check_cu(
            unsafe {
                sys::cuDevSmResourceSplitByCount(
                    &mut grp_decode,
                    &mut nb,
                    sm_res,
                    &mut grp_prefill,
                    0,
                    sm_for_decode,
                )
            },
            "cuDevSmResourceSplitByCount",
        )?;

        let sm_decode = unsafe { grp_decode.__bindgen_anon_1.sm.smCount };
        let sm_prefill = unsafe { grp_prefill.__bindgen_anon_1.sm.smCount };

        // Generate resource descriptors
        let mut desc_decode: sys::CUdevResourceDesc = ptr::null_mut();
        let mut desc_prefill: sys::CUdevResourceDesc = ptr::null_mut();
        check_cu(
            unsafe { sys::cuDevResourceGenerateDesc(&mut desc_decode, &mut grp_decode, 1) },
            "cuDevResourceGenerateDesc (decode)",
        )?;
        check_cu(
            unsafe { sys::cuDevResourceGenerateDesc(&mut desc_prefill, &mut grp_prefill, 1) },
            "cuDevResourceGenerateDesc (prefill)",
        )?;

        // Create green contexts
        let mut gctx_decode: sys::CUgreenCtx = ptr::null_mut();
        let mut gctx_prefill: sys::CUgreenCtx = ptr::null_mut();
        check_cu(
            unsafe {
                sys::cuGreenCtxCreate(
                    &mut gctx_decode,
                    desc_decode,
                    device,
                    sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32,
                )
            },
            "cuGreenCtxCreate (decode)",
        )?;
        check_cu(
            unsafe {
                sys::cuGreenCtxCreate(
                    &mut gctx_prefill,
                    desc_prefill,
                    device,
                    sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32,
                )
            },
            "cuGreenCtxCreate (prefill)",
        )?;

        // Get CUcontext from green contexts and create streams
        let mut ctx_decode: sys::CUcontext = ptr::null_mut();
        let mut ctx_prefill: sys::CUcontext = ptr::null_mut();
        check_cu(
            unsafe { sys::cuCtxFromGreenCtx(&mut ctx_decode, gctx_decode) },
            "cuCtxFromGreenCtx (decode)",
        )?;
        check_cu(
            unsafe { sys::cuCtxFromGreenCtx(&mut ctx_prefill, gctx_prefill) },
            "cuCtxFromGreenCtx (prefill)",
        )?;

        // Create streams on decode green context
        let mut stream_decode: CUstream = ptr::null_mut();
        unsafe { sys::cuCtxPushCurrent_v2(ctx_decode) };
        check_cu(
            unsafe {
                sys::cuStreamCreate(
                    &mut stream_decode,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "cuStreamCreate (decode)",
        )?;
        unsafe { sys::cuCtxPopCurrent_v2(ptr::null_mut()) };

        // Create streams on prefill green context
        let mut stream_prefill: CUstream = ptr::null_mut();
        unsafe { sys::cuCtxPushCurrent_v2(ctx_prefill) };
        check_cu(
            unsafe {
                sys::cuStreamCreate(
                    &mut stream_prefill,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "cuStreamCreate (prefill)",
        )?;
        unsafe { sys::cuCtxPopCurrent_v2(ptr::null_mut()) };

        Ok(Self {
            sm_decode,
            sm_prefill,
            gctx_decode,
            gctx_prefill,
            stream_decode,
            stream_prefill,
        })
    }
}

impl Drop for SmPartition {
    fn drop(&mut self) {
        unsafe {
            sys::cuStreamDestroy_v2(self.stream_decode);
            sys::cuStreamDestroy_v2(self.stream_prefill);
            sys::cuGreenCtxDestroy(self.gctx_decode);
            sys::cuGreenCtxDestroy(self.gctx_prefill);
        }
    }
}

// ─── Decode case (memory-bound) ─────────────────────────────────────────────

struct DecodeCase {
    // Keep allocations alive
    _q: HiddenStates,
    _output: HiddenStates,
    _kv_buffer: CudaSlice<bf16>,
    _page_indices_d: CudaSlice<i32>,
    _page_indptr_d: CudaSlice<i32>,
    _last_page_len_d: CudaSlice<i32>,
    _request_indices_d: CudaSlice<i32>,
    _kv_tile_indices_d: CudaSlice<i32>,
    _kv_chunk_size_d: CudaSlice<i32>,
    // Cached raw pointers
    q_ptr: u64,
    out_ptr: u64,
    kv_ptr: u64,
    pi_ptr: u64,
    pip_ptr: u64,
    lpl_ptr: u64,
    ri_ptr: u64,
    kti_ptr: u64,
    kcs_ptr: u64,
    // Params
    k_offset: i64,
    v_offset: i64,
    stride_page: i64,
    sm_scale: f32,
    batch_size: i32,
}

impl DecodeCase {
    fn new(ctx: &DeviceContext, batch_size: usize, kv_len: usize) -> Result<Self> {
        let layout = PagedKvLayout::new(1, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE);
        let q_dim = NUM_QO_HEADS * HEAD_DIM;
        let pages_per_request = kv_len.div_ceil(PAGE_SIZE);
        let total_pages = pages_per_request * batch_size;

        let q = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(q_dim * batch_size, 0.01))?,
            hidden_dim: q_dim,
            seq_len: batch_size,
        };
        let mut output = HiddenStates::zeros(ctx, q_dim, batch_size)?;
        let kv_buffer = ctx
            .stream
            .clone_htod(&patterned_bf16(total_pages * layout.page_stride, 0.001))?;

        let mut page_indices = Vec::with_capacity(total_pages);
        let mut page_indptr = Vec::with_capacity(batch_size + 1);
        page_indptr.push(0);
        for req in 0..batch_size {
            for p in 0..pages_per_request {
                page_indices.push((req * pages_per_request + p) as i32);
            }
            page_indptr.push(page_indices.len() as i32);
        }
        let last_page_len = match kv_len % PAGE_SIZE {
            0 => PAGE_SIZE,
            rem => rem,
        };
        let last_page_lens = vec![last_page_len as i32; batch_size];
        let request_indices: Vec<i32> = (0..batch_size as i32).collect();
        let kv_tile_indices = vec![0i32; batch_size];
        let kv_chunk_sizes = vec![kv_len as i32; batch_size];

        let page_indices_d = ctx.stream.clone_htod(&page_indices)?;
        let page_indptr_d = ctx.stream.clone_htod(&page_indptr)?;
        let last_page_len_d = ctx.stream.clone_htod(&last_page_lens)?;
        let request_indices_d = ctx.stream.clone_htod(&request_indices)?;
        let kv_tile_indices_d = ctx.stream.clone_htod(&kv_tile_indices)?;
        let kv_chunk_size_d = ctx.stream.clone_htod(&kv_chunk_sizes)?;
        ctx.sync()?;

        // Cache raw pointers (safe: buffers are pinned device memory)
        let (q_ptr, _) = q.data.device_ptr(&ctx.stream);
        let (out_ptr, _) = output.data.device_ptr_mut(&ctx.stream);
        let (kv_ptr, _) = kv_buffer.device_ptr(&ctx.stream);
        let (pi_ptr, _) = page_indices_d.device_ptr(&ctx.stream);
        let (pip_ptr, _) = page_indptr_d.device_ptr(&ctx.stream);
        let (lpl_ptr, _) = last_page_len_d.device_ptr(&ctx.stream);
        let (ri_ptr, _) = request_indices_d.device_ptr(&ctx.stream);
        let (kti_ptr, _) = kv_tile_indices_d.device_ptr(&ctx.stream);
        let (kcs_ptr, _) = kv_chunk_size_d.device_ptr(&ctx.stream);

        Ok(Self {
            _q: q,
            _output: output,
            _kv_buffer: kv_buffer,
            _page_indices_d: page_indices_d,
            _page_indptr_d: page_indptr_d,
            _last_page_len_d: last_page_len_d,
            _request_indices_d: request_indices_d,
            _kv_tile_indices_d: kv_tile_indices_d,
            _kv_chunk_size_d: kv_chunk_size_d,
            q_ptr,
            out_ptr,
            kv_ptr,
            pi_ptr,
            pip_ptr,
            lpl_ptr,
            ri_ptr,
            kti_ptr,
            kcs_ptr,
            k_offset: 0,
            v_offset: layout.kv_block_len as i64,
            stride_page: layout.page_stride as i64,
            sm_scale: 1.0f32 / (HEAD_DIM as f32).sqrt(),
            batch_size: batch_size as i32,
        })
    }

    #[inline]
    unsafe fn launch_on(&self, stream: CUstream) -> Result<()> {
        let result = unsafe {
            ffi::paged_attention_decode_cuda(
                self.q_ptr as *const ffi::Half,
                self.out_ptr as *mut ffi::Half,
                self.kv_ptr as *const ffi::Half,
                self.k_offset,
                self.v_offset,
                self.pi_ptr as *const i32,
                self.pip_ptr as *const i32,
                self.lpl_ptr as *const i32,
                self.ri_ptr as *const i32,
                self.kti_ptr as *const i32,
                self.kcs_ptr as *const i32,
                NUM_QO_HEADS as i32,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                PAGE_SIZE as i32,
                self.batch_size,
                self.stride_page,
                self.sm_scale,
                stream,
            )
        };
        if result != 0 {
            bail!("paged_attention_decode_cuda failed: {result}");
        }
        Ok(())
    }
}

// ─── Prefill case (compute-bound) ────────────────────────────────────────────

struct PrefillCase {
    // Keep allocations alive
    _q: HiddenStates,
    _k: HiddenStates,
    _v: HiddenStates,
    _output: HiddenStates,
    _q_norm: DeviceVec,
    _k_norm: DeviceVec,
    _cos_cache: DeviceVec,
    _sin_cache: DeviceVec,
    _kv_buffer: CudaSlice<bf16>,
    _plan: PrefillPagedPlan,
    // Cached raw pointers
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    out_ptr: u64,
    qn_ptr: u64,
    kn_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    kv_ptr: u64,
    pi_ptr: u64,
    pip_ptr: u64,
    lpl_ptr: u64,
    pos_ptr: u64,
    bi_ptr: u64,
    qi_ptr: u64,
    ri_ptr: u64,
    qti_ptr: u64,
    kti_ptr: u64,
    kcs_ptr: u64,
    tnr_ptr: u64,
    // Params
    total_tokens: i32,
    cos_max_pos: i32,
    k_offset: i64,
    v_offset: i64,
    stride_page: i64,
    sm_scale: f32,
    kv_dim: i64,
    plan_batch_size: i32,
    plan_num_tiles: i32,
}

impl PrefillCase {
    fn new(ctx: &DeviceContext, batch_size: usize, seq_len: usize) -> Result<Self> {
        let layout = PagedKvLayout::new(1, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE);
        let q_dim = NUM_QO_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;
        let total_tokens = batch_size * seq_len;
        let pages_per_request = seq_len.div_ceil(PAGE_SIZE);
        let total_pages = pages_per_request * batch_size;

        let mut q = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(q_dim * total_tokens, 0.01))?,
            hidden_dim: q_dim,
            seq_len: total_tokens,
        };
        let mut k = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(kv_dim * total_tokens, 0.001))?,
            hidden_dim: kv_dim,
            seq_len: total_tokens,
        };
        let v = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(kv_dim * total_tokens, 0.002))?,
            hidden_dim: kv_dim,
            seq_len: total_tokens,
        };
        let mut output = HiddenStates::zeros(ctx, q_dim, total_tokens)?;
        let q_norm = DeviceVec::from_host(ctx, &vec![bf16::from_f32(1.0); HEAD_DIM])?;
        let k_norm = DeviceVec::from_host(ctx, &vec![bf16::from_f32(1.0); HEAD_DIM])?;
        let cos_cache = DeviceVec::from_host(ctx, &rope_cache_bf16(seq_len, true))?;
        let sin_cache = DeviceVec::from_host(ctx, &rope_cache_bf16(seq_len, false))?;
        let kv_buffer = ctx
            .stream
            .clone_htod(&patterned_bf16(total_pages * layout.page_stride, 0.001))?;

        // Build plan
        let last_page_len = match seq_len % PAGE_SIZE {
            0 => PAGE_SIZE,
            rem => rem,
        };
        let page_indices: Vec<Vec<i32>> = (0..batch_size)
            .map(|req| {
                (0..pages_per_request)
                    .map(|p| (req * pages_per_request + p) as i32)
                    .collect()
            })
            .collect();
        let last_page_lens = vec![last_page_len; batch_size];
        let start_positions = vec![0usize; batch_size];
        let seq_lens_vec = vec![seq_len; batch_size];
        let plan = PrefillPagedPlan::new_batch_with_cta_tile_q(
            ctx,
            &page_indices,
            &last_page_lens,
            &start_positions,
            &seq_lens_vec,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            0,
        )?;
        ctx.sync()?;

        // Cache raw pointers
        let (q_ptr, _) = q.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _) = k.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _) = v.data.device_ptr(&ctx.stream);
        let (out_ptr, _) = output.data.device_ptr_mut(&ctx.stream);
        let (qn_ptr, _) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _) = sin_cache.data.device_ptr(&ctx.stream);
        let (kv_ptr, _) = kv_buffer.device_ptr(&ctx.stream);
        let (pi_ptr, _) = plan.page_indices_d().device_ptr(&ctx.stream);
        let (pip_ptr, _) = plan.page_indptr_d().device_ptr(&ctx.stream);
        let (lpl_ptr, _) = plan.last_page_len_d().device_ptr(&ctx.stream);
        let (pos_ptr, _) = plan.positions_d().device_ptr(&ctx.stream);
        let (bi_ptr, _) = plan.batch_indices_d().device_ptr(&ctx.stream);
        let (qi_ptr, _) = plan.q_indptr_d().device_ptr(&ctx.stream);
        let (ri_ptr, _) = plan.request_indices_d().device_ptr(&ctx.stream);
        let (qti_ptr, _) = plan.qo_tile_indices_d().device_ptr(&ctx.stream);
        let (kti_ptr, _) = plan.kv_tile_indices_d().device_ptr(&ctx.stream);
        let (kcs_ptr, _) = plan.kv_chunk_size_d().device_ptr(&ctx.stream);
        let (tnr_ptr, _) = plan.total_num_rows_d().device_ptr(&ctx.stream);

        let cos_max_pos = (cos_cache.len / HEAD_DIM) as i32;
        let plan_batch_size = plan.batch_size();
        let plan_num_tiles = plan.num_tiles();

        Ok(Self {
            _q: q,
            _k: k,
            _v: v,
            _output: output,
            _q_norm: q_norm,
            _k_norm: k_norm,
            _cos_cache: cos_cache,
            _sin_cache: sin_cache,
            _kv_buffer: kv_buffer,
            _plan: plan,
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            qn_ptr,
            kn_ptr,
            cos_ptr,
            sin_ptr,
            kv_ptr,
            pi_ptr,
            pip_ptr,
            lpl_ptr,
            pos_ptr,
            bi_ptr,
            qi_ptr,
            ri_ptr,
            qti_ptr,
            kti_ptr,
            kcs_ptr,
            tnr_ptr,
            total_tokens: total_tokens as i32,
            cos_max_pos,
            k_offset: 0,
            v_offset: layout.kv_block_len as i64,
            stride_page: layout.page_stride as i64,
            sm_scale: 1.0f32 / (HEAD_DIM as f32).sqrt(),
            kv_dim: kv_dim as i64,
            plan_batch_size,
            plan_num_tiles,
        })
    }

    #[inline]
    unsafe fn launch_on(&self, stream: CUstream) -> Result<()> {
        // 1. QK norm + RoPE
        unsafe {
            ffi::qk_norm_rope_batched_decode_cuda(
                self.q_ptr as *mut ffi::Half,
                self.k_ptr as *mut ffi::Half,
                self.qn_ptr as *const ffi::Half,
                self.kn_ptr as *const ffi::Half,
                self.cos_ptr as *const ffi::Half,
                self.sin_ptr as *const ffi::Half,
                self.pos_ptr as *const i32,
                NUM_QO_HEADS as i32,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                self.total_tokens,
                1.0e-6,
                self.cos_max_pos,
                stream,
            );
        }

        // 2. KV scatter
        let result = unsafe {
            ffi::paged_kv_scatter_cuda(
                self.kv_ptr as *const ffi::Half,
                self.k_offset,
                self.v_offset,
                self.pi_ptr as *const i32,
                self.pip_ptr as *const i32,
                self.lpl_ptr as *const i32,
                self.k_ptr as *const ffi::Half,
                self.v_ptr as *const ffi::Half,
                self.bi_ptr as *const i32,
                self.pos_ptr as *const i32,
                self.total_tokens,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                PAGE_SIZE as i32,
                self.stride_page,
                self.kv_dim,
                HEAD_DIM as i64,
                stream,
            )
        };
        if result != 0 {
            bail!("paged_kv_scatter_cuda failed: {result}");
        }

        // 3. Attention core
        let result = unsafe {
            ffi::batch_prefill_paged_cuda_with_cta_tile_q(
                self.q_ptr as *const ffi::Half,
                self.out_ptr as *mut ffi::Half,
                self.kv_ptr as *const ffi::Half,
                self.k_offset,
                self.v_offset,
                self.pi_ptr as *const i32,
                self.pip_ptr as *const i32,
                self.lpl_ptr as *const i32,
                self.qi_ptr as *const i32,
                self.ri_ptr as *const i32,
                self.qti_ptr as *const i32,
                self.kti_ptr as *const i32,
                self.kcs_ptr as *const i32,
                self.tnr_ptr as *const u32,
                NUM_QO_HEADS as i32,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                PAGE_SIZE as i32,
                self.total_tokens,
                self.plan_batch_size,
                self.plan_num_tiles,
                self.stride_page,
                self.sm_scale,
                0, // default cta_tile_q
                stream,
            )
        };
        if result != 0 {
            bail!("batch_prefill_paged_cuda failed: {result}");
        }
        Ok(())
    }
}

// ─── Timing ───────────────────────────────────────────────────────────────────

fn stream_sync(stream: CUstream) -> Result<()> {
    check_cu(
        unsafe { sys::cuStreamSynchronize(stream) },
        "cuStreamSynchronize",
    )
}

fn event_elapsed_ms(start: sys::CUevent, end: sys::CUevent) -> Result<f32> {
    let mut ms: f32 = 0.0;
    check_cu(
        unsafe { sys::cuEventElapsedTime(&mut ms, start, end) },
        "cuEventElapsedTime",
    )?;
    Ok(ms)
}

struct TimingEvents {
    start: sys::CUevent,
    end: sys::CUevent,
}

impl TimingEvents {
    fn new() -> Result<Self> {
        let mut start: sys::CUevent = ptr::null_mut();
        let mut end: sys::CUevent = ptr::null_mut();
        check_cu(
            unsafe { sys::cuEventCreate(&mut start, 0) },
            "cuEventCreate",
        )?;
        check_cu(unsafe { sys::cuEventCreate(&mut end, 0) }, "cuEventCreate")?;
        Ok(Self { start, end })
    }

    fn record_start(&self, stream: CUstream) -> Result<()> {
        check_cu(
            unsafe { sys::cuEventRecord(self.start, stream) },
            "cuEventRecord start",
        )
    }

    fn record_end(&self, stream: CUstream) -> Result<()> {
        check_cu(
            unsafe { sys::cuEventRecord(self.end, stream) },
            "cuEventRecord end",
        )
    }

    fn elapsed_ms(&self) -> Result<f32> {
        event_elapsed_ms(self.start, self.end)
    }
}

impl Drop for TimingEvents {
    fn drop(&mut self) {
        unsafe {
            sys::cuEventDestroy_v2(self.start);
            sys::cuEventDestroy_v2(self.end);
        }
    }
}

/// Time a kernel on a given stream (warmup + measured reps).
fn time_kernel_on(
    stream: CUstream,
    launch: &mut dyn FnMut(CUstream) -> Result<()>,
    warmup: u64,
    reps: u64,
) -> Result<f32> {
    for _ in 0..warmup {
        launch(stream)?;
    }
    stream_sync(stream)?;

    let ev = TimingEvents::new()?;
    ev.record_start(stream)?;
    for _ in 0..reps {
        launch(stream)?;
    }
    ev.record_end(stream)?;
    stream_sync(stream)?;
    Ok(ev.elapsed_ms()? / reps as f32)
}

/// Time two kernels launched concurrently on separate streams.
fn time_concurrent(
    stream_a: CUstream,
    stream_b: CUstream,
    launch_a: &mut dyn FnMut(CUstream) -> Result<()>,
    launch_b: &mut dyn FnMut(CUstream) -> Result<()>,
    warmup: u64,
    reps: u64,
) -> Result<f32> {
    // Warmup
    for _ in 0..warmup {
        launch_a(stream_a)?;
        launch_b(stream_b)?;
    }
    stream_sync(stream_a)?;
    stream_sync(stream_b)?;

    let mut total_ms = 0.0f32;
    for _ in 0..reps {
        // Use events to synchronize start and measure wall time
        let mut anchor: sys::CUevent = ptr::null_mut();
        let mut done: sys::CUevent = ptr::null_mut();
        let mut done_b: sys::CUevent = ptr::null_mut();
        unsafe {
            sys::cuEventCreate(&mut anchor, 0);
            sys::cuEventCreate(&mut done, 0);
            sys::cuEventCreate(&mut done_b, 0);
        }

        // Sync start
        check_cu(unsafe { sys::cuEventRecord(anchor, stream_a) }, "anchor")?;
        check_cu(
            unsafe { sys::cuStreamWaitEvent(stream_b, anchor, 0) },
            "stream_b wait anchor",
        )?;

        launch_a(stream_a)?;
        launch_b(stream_b)?;

        // Wait for both, measure from anchor
        check_cu(unsafe { sys::cuEventRecord(done_b, stream_b) }, "done_b")?;
        check_cu(
            unsafe { sys::cuStreamWaitEvent(stream_a, done_b, 0) },
            "stream_a wait done_b",
        )?;
        check_cu(unsafe { sys::cuEventRecord(done, stream_a) }, "done")?;
        stream_sync(stream_a)?;

        total_ms += event_elapsed_ms(anchor, done)?;

        unsafe {
            sys::cuEventDestroy_v2(anchor);
            sys::cuEventDestroy_v2(done);
            sys::cuEventDestroy_v2(done_b);
        }
    }
    Ok(total_ms / reps as f32)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Initialize CUDA driver
    check_cu(unsafe { sys::cuInit(0) }, "cuInit")?;

    let ctx = DeviceContext::new()?;
    let default_stream = ctx.stream.cu_stream();

    // Query device info
    let mut device: CUdevice = 0;
    check_cu(unsafe { sys::cuDeviceGet(&mut device, 0) }, "cuDeviceGet")?;

    let mut sm_count: i32 = 0;
    check_cu(
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut sm_count,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device,
            )
        },
        "SM count",
    )?;

    let mut name = [0u8; 256];
    check_cu(
        unsafe { sys::cuDeviceGetName(name.as_mut_ptr() as *mut i8, 256, device) },
        "cuDeviceGetName",
    )?;
    let name_str = std::str::from_utf8(&name)
        .unwrap_or("unknown")
        .trim_end_matches('\0');
    println!("GPU: {}  SMs: {}", name_str, sm_count);

    // Query SM resource
    let mut sm_res: sys::CUdevResource = unsafe { std::mem::zeroed() };
    check_cu(
        unsafe {
            sys::cuDeviceGetDevResource(
                device,
                &mut sm_res,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM,
            )
        },
        "cuDeviceGetDevResource",
    )?;
    let total_sm = unsafe { sm_res.__bindgen_anon_1.sm.smCount };

    // Get min SM count (alignment)
    let mut nb: u32 = 1;
    let mut probe_grp: sys::CUdevResource = unsafe { std::mem::zeroed() };
    let mut probe_rem: sys::CUdevResource = unsafe { std::mem::zeroed() };
    check_cu(
        unsafe {
            sys::cuDevSmResourceSplitByCount(&mut probe_grp, &mut nb, &sm_res, &mut probe_rem, 0, 1)
        },
        "probe split",
    )?;
    let min_sm = unsafe { probe_grp.__bindgen_anon_1.sm.smCount };
    println!("SM partition: total={total_sm} min={min_sm}");
    println!();

    // Create partitions to sweep
    let split_pcts: &[u32] = &[20, 30, 40, 50];
    let mut partitions: Vec<SmPartition> = Vec::new();
    for &decode_pct in split_pcts {
        let sm_decode_target = (total_sm * decode_pct / 100 / min_sm) * min_sm;
        if sm_decode_target < min_sm || total_sm - sm_decode_target < min_sm {
            continue;
        }
        match SmPartition::create(device, &sm_res, sm_decode_target) {
            Ok(p) => {
                println!(
                    "  Partition: decode={}SM prefill={}SM (target {}%)",
                    p.sm_decode, p.sm_prefill, decode_pct
                );
                partitions.push(p);
            }
            Err(e) => {
                eprintln!("  Skip {}% split: {e}", decode_pct);
            }
        }
    }
    if partitions.is_empty() {
        bail!("No valid SM partitions could be created. Green Contexts may not be supported.");
    }
    println!();

    // Also create a second plain stream for multi-stream baseline
    let mut plain_stream_b: CUstream = ptr::null_mut();
    check_cu(
        unsafe {
            sys::cuStreamCreate(
                &mut plain_stream_b,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        },
        "cuStreamCreate (plain B)",
    )?;

    // ─── Run benchmark matrix ────────────────────────────────────────────────────

    println!(
        "{:<22} {:>8} {:>8} {:>8} {:>10}",
        "workload", "decode", "prefill", "serial", "multi-s"
    );
    for p in &partitions {
        print!(" {:>3}/{:<3}", p.sm_decode, p.sm_prefill);
    }
    println!("  best");
    println!("{}", "-".repeat(22 + 8 * 4 + 10 + partitions.len() * 8 + 6));

    for &(pf_bs, pf_seq) in PREFILL_CONFIGS {
        let mut prefill = PrefillCase::new(&ctx, pf_bs, pf_seq)?;

        for &(dec_bs, dec_kv) in DECODE_CONFIGS {
            let mut decode = DecodeCase::new(&ctx, dec_bs, dec_kv)?;
            ctx.sync()?;

            let label = format!("P{pf_bs}x{pf_seq} D{dec_bs}x{dec_kv}");

            // Measure decode alone
            let t_decode = time_kernel_on(
                default_stream,
                &mut |s| unsafe { decode.launch_on(s) },
                WARMUP,
                REPS,
            )?;

            // Measure prefill alone
            let t_prefill = time_kernel_on(
                default_stream,
                &mut |s| unsafe { prefill.launch_on(s) },
                WARMUP,
                REPS,
            )?;

            let t_serial = t_decode + t_prefill;

            // Multi-stream baseline (no partition)
            let t_ms = time_concurrent(
                default_stream,
                plain_stream_b,
                &mut |s| unsafe { prefill.launch_on(s) },
                &mut |s| unsafe { decode.launch_on(s) },
                WARMUP,
                REPS,
            )?;
            let sp_ms = t_serial / t_ms;

            print!(
                "{:<22} {:>7.3} {:>7.3} {:>7.3} {:>7.2}x",
                label, t_decode, t_prefill, t_serial, sp_ms
            );

            let mut best_sp = sp_ms;
            let mut best_label = "multi-s".to_string();

            for p in &partitions {
                let t_gc = time_concurrent(
                    p.stream_prefill,
                    p.stream_decode,
                    &mut |s| unsafe { prefill.launch_on(s) },
                    &mut |s| unsafe { decode.launch_on(s) },
                    WARMUP,
                    REPS,
                )?;
                let sp = t_serial / t_gc;
                if sp > best_sp {
                    best_sp = sp;
                    best_label = format!("{}/{}", p.sm_decode, p.sm_prefill);
                }
                print!(" {:>6.2}x", sp);
            }
            println!("  {best_label} {best_sp:.2}x");
        }
    }

    // Cleanup
    unsafe { sys::cuStreamDestroy_v2(plain_stream_b) };
    drop(partitions);

    Ok(())
}
