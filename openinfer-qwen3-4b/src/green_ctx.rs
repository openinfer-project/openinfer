//! Green Context SM-partition support for concurrent prefill/decode.
//!
//! When enabled, the executor uses two CUDA streams backed by Green Context
//! SM partitions: a decode partition (fewer SMs, memory-bound workload) and a
//! prefill partition (more SMs, compute-bound workload). When no prefill is
//! pending, decode runs on the original full-SM stream.

use std::ptr;

use anyhow::{Result, bail};
use cudarc::driver::sys::{self, CUdevice, CUstream};

/// SM partition configuration.
#[derive(Clone, Copy, Debug)]
pub struct SmPartitionConfig {
    /// Percentage of total SMs assigned to decode (e.g. 20 means 20%).
    pub decode_pct: u32,
}

impl Default for SmPartitionConfig {
    fn default() -> Self {
        Self { decode_pct: 20 }
    }
}

/// A `CUstream` wrapper that is `Send + Sync + Clone + Copy`.
/// SAFETY: CUDA streams are thread-safe when used on the same device.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct SendStream(pub CUstream);

unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

/// Active Green Context SM partition with two streams.
pub(crate) struct SmPartition {
    pub decode_stream: SendStream,
    pub prefill_stream: SendStream,
    pub sm_decode: u32,
    pub sm_prefill: u32,
    gctx_decode: sys::CUgreenCtx,
    gctx_prefill: sys::CUgreenCtx,
    #[allow(dead_code)]
    ctx_decode: sys::CUcontext,
    #[allow(dead_code)]
    ctx_prefill: sys::CUcontext,
}

fn check_cu(result: sys::CUresult, msg: &str) -> Result<()> {
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("{msg}: CUresult = {result:?}");
    }
    Ok(())
}

impl SmPartition {
    /// Create an SM partition on the given device.
    /// `decode_pct` is the percentage of SMs to assign to decode.
    pub fn create(device_ordinal: usize, config: SmPartitionConfig) -> Result<Self> {
        let device: CUdevice = device_ordinal as i32;

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

        // Get minimum SM granularity
        let mut nb: u32 = 1;
        let mut probe_grp: sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut probe_rem: sys::CUdevResource = unsafe { std::mem::zeroed() };
        check_cu(
            unsafe {
                sys::cuDevSmResourceSplitByCount(
                    &mut probe_grp,
                    &mut nb,
                    &sm_res,
                    &mut probe_rem,
                    0,
                    1,
                )
            },
            "probe split",
        )?;
        let min_sm = unsafe { probe_grp.__bindgen_anon_1.sm.smCount };

        // Compute decode SM count aligned to minimum
        let sm_for_decode = (total_sm * config.decode_pct / 100 / min_sm) * min_sm;
        if sm_for_decode < min_sm || total_sm - sm_for_decode < min_sm {
            bail!(
                "SM partition not viable: total={total_sm} min={min_sm} \
                 decode_target={sm_for_decode}"
            );
        }

        // Split
        let mut grp_decode: sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut grp_prefill: sys::CUdevResource = unsafe { std::mem::zeroed() };
        nb = 1;
        check_cu(
            unsafe {
                sys::cuDevSmResourceSplitByCount(
                    &mut grp_decode,
                    &mut nb,
                    &sm_res,
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

        // Get CUcontext from green contexts
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

        // Create streams
        let mut decode_stream: CUstream = ptr::null_mut();
        unsafe { sys::cuCtxPushCurrent_v2(ctx_decode) };
        check_cu(
            unsafe {
                sys::cuStreamCreate(
                    &mut decode_stream,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "cuStreamCreate (decode)",
        )?;
        unsafe { sys::cuCtxPopCurrent_v2(ptr::null_mut()) };

        let mut prefill_stream: CUstream = ptr::null_mut();
        unsafe { sys::cuCtxPushCurrent_v2(ctx_prefill) };
        check_cu(
            unsafe {
                sys::cuStreamCreate(
                    &mut prefill_stream,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "cuStreamCreate (prefill)",
        )?;
        unsafe { sys::cuCtxPopCurrent_v2(ptr::null_mut()) };

        log::info!(
            "Green Context SM partition created: decode={sm_decode}SM \
             prefill={sm_prefill}SM (total={total_sm})"
        );

        Ok(Self {
            decode_stream: SendStream(decode_stream),
            prefill_stream: SendStream(prefill_stream),
            sm_decode,
            sm_prefill,
            gctx_decode,
            gctx_prefill,
            ctx_decode,
            ctx_prefill,
        })
    }

    /// Synchronize the decode partition stream.
    pub fn sync_decode(&self) -> Result<()> {
        check_cu(
            unsafe { sys::cuStreamSynchronize(self.decode_stream.0) },
            "sync decode stream",
        )
    }

    /// Synchronize the prefill partition stream.
    pub fn sync_prefill(&self) -> Result<()> {
        check_cu(
            unsafe { sys::cuStreamSynchronize(self.prefill_stream.0) },
            "sync prefill stream",
        )
    }
}

impl Drop for SmPartition {
    fn drop(&mut self) {
        unsafe {
            sys::cuStreamDestroy_v2(self.decode_stream.0);
            sys::cuStreamDestroy_v2(self.prefill_stream.0);
            sys::cuGreenCtxDestroy(self.gctx_decode);
            sys::cuGreenCtxDestroy(self.gctx_prefill);
        }
    }
}

// SAFETY: SmPartition is only used from the executor's single GPU worker thread.
unsafe impl Send for SmPartition {}
