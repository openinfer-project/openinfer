use std::{
    collections::HashSet,
    env,
    ffi::{CStr, c_char, c_int, c_void},
    fs,
    path::{Path, PathBuf},
    ptr,
    sync::{Arc, Mutex, MutexGuard},
    thread,
};

use anyhow::{Context, Result, bail, ensure};
use cudarc::{
    driver::{
        CudaSlice, DevicePtr, DevicePtrMut,
        sys::{
            CUdeviceptr, CUgraph, CUgraphExec, CUstream,
            CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        },
    },
    nccl::sys::{ncclComm_t, ncclDataType_t, ncclRedOp_t, ncclResult_t},
};
use half::bf16;
use libloading::Library;
use openinfer_core::{
    ffi as core_ffi, ops,
    tensor::{DeviceContext, HiddenStates, HiddenStatesRef},
};
use openinfer_kernels::ops::dsv2_lite_accumulate_fixed_expert_into;
use serde::Serialize;

use crate::device::{activate, activate_graph_capture, graph_capture_activation_guard};

#[cfg(test)]
mod tests;

type NcclCommInitAll = unsafe extern "C" fn(*mut ncclComm_t, c_int, *const c_int) -> ncclResult_t;
type NcclCommCount = unsafe extern "C" fn(ncclComm_t, *mut c_int) -> ncclResult_t;
type NcclCommCuDevice = unsafe extern "C" fn(ncclComm_t, *mut c_int) -> ncclResult_t;
type NcclCommAbort = unsafe extern "C" fn(ncclComm_t) -> ncclResult_t;
type NcclGroupStart = unsafe extern "C" fn() -> ncclResult_t;
type NcclGroupEnd = unsafe extern "C" fn() -> ncclResult_t;
type NcclAllReduce = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    ncclDataType_t,
    ncclRedOp_t,
    ncclComm_t,
    CUstream,
) -> ncclResult_t;
type NcclGetErrorString = unsafe extern "C" fn(ncclResult_t) -> *const c_char;

// Keep the correctness-first NCCL bridge below the long-prompt failure band
// observed on the 2x RTX 5090 validation host. These are not hardware limits;
// they are conservative per-call caps that preserve the short-shape path as one
// collective while splitting the long bf16 dense exchange and f32 combine rows
// that previously failed in prefill.
const NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL: usize = 64 * 1024;
const NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL: usize = 48 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AllReduceChunk {
    offset: usize,
    len: usize,
}

pub(crate) struct NaiveNcclEp2Backend {
    lib: Arc<RawNcclLib>,
    comms: Vec<ncclComm_t>,
    dense_exchange_scratch: Mutex<DeviceDenseExchangeScratch>,
    combine_scratch: Mutex<DeviceCombineScratch>,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Serialize)]
pub(crate) struct NcclGraphSmokeReport {
    attempted: bool,
    captured: bool,
    replayed: bool,
    verified: bool,
    count: usize,
    expected_sum: f32,
    rank0_value: Option<f32>,
    rank1_value: Option<f32>,
    capture_error: Option<String>,
    replay_error: Option<String>,
    verification_error: Option<String>,
    capture_mode: &'static str,
}

impl NcclGraphSmokeReport {
    pub(crate) fn coverage_status(&self) -> &'static str {
        if self.verified {
            "captured_replayed_verified"
        } else if self.replayed {
            "replayed_but_not_verified"
        } else if self.captured {
            "captured_but_not_replayed"
        } else {
            "failed"
        }
    }

    pub(crate) fn verified(&self) -> bool {
        self.verified
    }

    pub(crate) fn failure_summary(&self) -> String {
        format!(
            "status={}, capture_error={:?}, replay_error={:?}, verification_error={:?}",
            self.coverage_status(),
            self.capture_error,
            self.replay_error,
            self.verification_error
        )
    }
}

struct RawNcclLib {
    _library: Library,
    source: String,
    comm_init_all: NcclCommInitAll,
    comm_count: NcclCommCount,
    comm_cu_device: NcclCommCuDevice,
    comm_abort: NcclCommAbort,
    group_start: NcclGroupStart,
    group_end: NcclGroupEnd,
    all_reduce: NcclAllReduce,
    get_error_string: NcclGetErrorString,
}

#[derive(Default)]
struct DeviceDenseExchangeScratch {
    hidden_dim: usize,
    seq_len: usize,
    rank0_recv: Option<CudaSlice<bf16>>,
    rank1_send_zero: Option<CudaSlice<bf16>>,
    rank1_recv: Option<CudaSlice<bf16>>,
}

impl DeviceDenseExchangeScratch {
    fn ensure(
        &mut self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<usize> {
        let elems = dense_exchange_elems(hidden_dim, seq_len)?;
        if self.hidden_dim == hidden_dim
            && self.seq_len == seq_len
            && self
                .rank0_recv
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
            && self
                .rank1_send_zero
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
            && self
                .rank1_recv
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
        {
            return Ok(elems);
        }

        activate(rank0)?;
        drop(self.rank0_recv.take());
        let rank0_recv = rank0.stream.alloc_zeros::<bf16>(elems)?;
        activate(rank1)?;
        drop(self.rank1_send_zero.take());
        drop(self.rank1_recv.take());
        let rank1_send_zero = rank1.stream.alloc_zeros::<bf16>(elems)?;
        let rank1_recv = rank1.stream.alloc_zeros::<bf16>(elems)?;

        self.hidden_dim = hidden_dim;
        self.seq_len = seq_len;
        self.rank0_recv = Some(rank0_recv);
        self.rank1_send_zero = Some(rank1_send_zero);
        self.rank1_recv = Some(rank1_recv);
        Ok(elems)
    }

    fn rank1_hidden_ref(&self) -> Result<HiddenStatesRef<'_>> {
        Ok(HiddenStatesRef {
            data: self
                .rank1_recv
                .as_ref()
                .context("DeepSeek-V2-Lite NCCL rank1 dense exchange recv scratch is missing")?,
            hidden_dim: self.hidden_dim,
            seq_len: self.seq_len,
        })
    }
}

pub(crate) struct DenseExchangeOutput<'a> {
    scratch: MutexGuard<'a, DeviceDenseExchangeScratch>,
}

impl DenseExchangeOutput<'_> {
    pub(crate) fn rank1_hidden(&self) -> Result<HiddenStatesRef<'_>> {
        self.scratch.rank1_hidden_ref()
    }
}

#[derive(Default)]
struct DeviceCombineScratch {
    hidden_dim: usize,
    seq_len: usize,
    rank0_send: Option<CudaSlice<f32>>,
    rank0_recv: Option<CudaSlice<f32>>,
    rank1_send: Option<CudaSlice<f32>>,
    rank1_recv: Option<CudaSlice<f32>>,
}

impl DeviceCombineScratch {
    fn ensure(
        &mut self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        let elems = combine_elems(hidden_dim, seq_len)?;
        if self.hidden_dim == hidden_dim
            && self.seq_len == seq_len
            && self
                .rank0_send
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
            && self
                .rank0_recv
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
            && self
                .rank1_send
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
            && self
                .rank1_recv
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
        {
            return Ok(());
        }

        activate(rank0)?;
        drop(self.rank0_send.take());
        drop(self.rank0_recv.take());
        let rank0_send = rank0.stream.alloc_zeros::<f32>(elems)?;
        let rank0_recv = rank0.stream.alloc_zeros::<f32>(elems)?;
        activate(rank1)?;
        drop(self.rank1_send.take());
        drop(self.rank1_recv.take());
        let rank1_send = rank1.stream.alloc_zeros::<f32>(elems)?;
        let rank1_recv = rank1.stream.alloc_zeros::<f32>(elems)?;

        self.hidden_dim = hidden_dim;
        self.seq_len = seq_len;
        self.rank0_send = Some(rank0_send);
        self.rank0_recv = Some(rank0_recv);
        self.rank1_send = Some(rank1_send);
        self.rank1_recv = Some(rank1_recv);
        Ok(())
    }

    fn ensure_shape(&self, hidden_dim: usize, seq_len: usize) -> Result<usize> {
        let elems = combine_elems(hidden_dim, seq_len)?;
        ensure!(
            self.hidden_dim == hidden_dim && self.seq_len == seq_len,
            "DeepSeek-V2-Lite NCCL device combine scratch shape mismatch: scratch=[{}, {}], requested=[{}, {}]",
            self.hidden_dim,
            self.seq_len,
            hidden_dim,
            seq_len
        );
        ensure!(
            self.rank0_send
                .as_ref()
                .is_some_and(|buf| buf.len() >= elems)
                && self
                    .rank0_recv
                    .as_ref()
                    .is_some_and(|buf| buf.len() >= elems)
                && self
                    .rank1_send
                    .as_ref()
                    .is_some_and(|buf| buf.len() >= elems)
                && self
                    .rank1_recv
                    .as_ref()
                    .is_some_and(|buf| buf.len() >= elems),
            "DeepSeek-V2-Lite NCCL device combine scratch is not initialized for {elems} elements"
        );
        Ok(elems)
    }

    fn send_mut(&mut self, rank: usize) -> Result<&mut CudaSlice<f32>> {
        match rank {
            0 => self.rank0_send.as_mut(),
            1 => self.rank1_send.as_mut(),
            other => bail!("DeepSeek-V2-Lite NCCL device combine unsupported EP rank {other}"),
        }
        .context("DeepSeek-V2-Lite NCCL device combine send scratch is missing")
    }
}

impl NaiveNcclEp2Backend {
    pub(crate) fn new(rank0: &DeviceContext, rank1: &DeviceContext) -> Result<Self> {
        ensure!(
            rank0.device_ordinal != rank1.device_ordinal,
            "DeepSeek-V2-Lite NCCL EP=2 requires distinct CUDA devices, got {:?}",
            [rank0.device_ordinal, rank1.device_ordinal]
        );
        let lib = Arc::new(RawNcclLib::load()?);
        let ordinals = [rank0.device_ordinal as i32, rank1.device_ordinal as i32];
        let mut comms = vec![ptr::null_mut(); 2];
        let status = unsafe {
            // SAFETY: `comms` has space for two communicator handles and
            // `ordinals` names the two distinct CUDA devices validated above.
            (lib.comm_init_all)(comms.as_mut_ptr(), comms.len() as i32, ordinals.as_ptr())
        };
        lib.check(
            status,
            "DeepSeek-V2-Lite NCCL EP=2 communicator initialization",
        )?;
        ensure!(
            comms.iter().all(|comm| !comm.is_null()),
            "DeepSeek-V2-Lite NCCL EP=2 communicator initialization returned a null communicator"
        );
        let backend = Self {
            lib,
            comms,
            dense_exchange_scratch: Mutex::new(DeviceDenseExchangeScratch::default()),
            combine_scratch: Mutex::new(DeviceCombineScratch::default()),
        };
        backend.validate_communicators(&ordinals)?;
        backend.smoke_all_reduce_f32(rank0, rank1)?;
        Ok(backend)
    }

    pub(crate) fn dense_all_reduce_rank0_hidden_to_rank1(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        input: &HiddenStates,
    ) -> Result<DenseExchangeOutput<'_>> {
        ensure!(
            input.hidden_dim > 0 && input.seq_len > 0,
            "DeepSeek-V2-Lite NCCL dense hidden exchange requires non-empty hidden states"
        );
        let mut scratch = self.dense_exchange_scratch()?;
        let elems = scratch.ensure(rank0, rank1, input.hidden_dim, input.seq_len)?;
        activate(rank1)?;
        rank1
            .stream
            .memset_zeros(scratch.rank1_send_zero.as_mut().context(
                "DeepSeek-V2-Lite NCCL rank1 dense exchange zero-send scratch is missing",
            )?)
            .context("clear DeepSeek-V2-Lite NCCL rank1 dense exchange zero-send scratch")?;

        let DeviceDenseExchangeScratch {
            rank0_recv,
            rank1_send_zero,
            rank1_recv,
            ..
        } = &mut *scratch;
        let rank0_recv = rank0_recv
            .as_mut()
            .context("DeepSeek-V2-Lite NCCL rank0 dense exchange recv scratch is missing")?;
        let rank1_send_zero = rank1_send_zero
            .as_ref()
            .context("DeepSeek-V2-Lite NCCL rank1 dense exchange zero-send scratch is missing")?;
        let rank1_recv = rank1_recv
            .as_mut()
            .context("DeepSeek-V2-Lite NCCL rank1 dense exchange recv scratch is missing")?;

        // Correctness-first dense exchange: rank0 contributes the hidden state
        // and rank1 contributes zeros. This makes rank0 hidden visible on rank1
        // without pretending to be sparse routed dispatch.
        self.all_reduce_bf16_pair_chunked(
            rank0,
            rank1,
            &input.data,
            rank0_recv,
            rank1_send_zero,
            rank1_recv,
            elems,
            "DeepSeek-V2-Lite NCCL dense hidden all-reduce",
        )?;
        Ok(DenseExchangeOutput { scratch })
    }

    pub(crate) fn clear_device_combine(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        let mut scratch = self.combine_scratch()?;
        let elems = combine_elems(hidden_dim, seq_len)?;
        scratch.ensure(rank0, rank1, hidden_dim, seq_len)?;
        activate(rank0)?;
        rank0
            .stream
            .memset_zeros(scratch.send_mut(0)?)
            .context("clear DeepSeek-V2-Lite NCCL rank0 combine send scratch")?;
        activate(rank1)?;
        rank1
            .stream
            .memset_zeros(scratch.send_mut(1)?)
            .context("clear DeepSeek-V2-Lite NCCL rank1 combine send scratch")?;
        scratch.ensure_shape(hidden_dim, seq_len)?;
        ensure!(
            elems > 0,
            "DeepSeek-V2-Lite NCCL device combine requires non-empty scratch"
        );
        Ok(())
    }

    pub(crate) fn accumulate_device_contribution(
        &self,
        rank: usize,
        ctx: &DeviceContext,
        expert_output: &HiddenStates,
        token_idx: usize,
        seq_len: usize,
        weight: f32,
    ) -> Result<()> {
        ensure!(
            expert_output.seq_len == 1,
            "DeepSeek-V2-Lite NCCL device combine expects one-token expert output, got seq_len={}",
            expert_output.seq_len
        );
        let mut scratch = self.combine_scratch()?;
        scratch.ensure_shape(expert_output.hidden_dim, seq_len)?;
        activate(ctx)?;
        ops::accumulate_bf16_token_scaled_to_f32_into(
            ctx,
            expert_output,
            weight,
            token_idx,
            seq_len,
            scratch.send_mut(rank)?,
        )
    }

    pub(crate) fn accumulate_fixed_expert_contribution(
        &self,
        rank: usize,
        ctx: &DeviceContext,
        expert_output: &HiddenStates,
        topk_weight: &CudaSlice<f32>,
        topk_idx: &CudaSlice<i32>,
        global_expert: usize,
        topk: usize,
    ) -> Result<()> {
        let mut scratch = self.combine_scratch()?;
        // Graph capture relies on `prepare_graph_shape` sizing this scratch
        // before the capture window; this call is only a shape assertion.
        scratch.ensure_shape(expert_output.hidden_dim, expert_output.seq_len)?;
        activate(ctx)?;
        dsv2_lite_accumulate_fixed_expert_into(
            ctx,
            expert_output,
            topk_weight,
            topk_idx,
            global_expert,
            topk,
            scratch.send_mut(rank)?,
        )
    }

    pub(crate) fn combine_device_contributions_to_rank0(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let mut scratch = self.combine_scratch()?;
        let elems = scratch.ensure_shape(hidden_dim, seq_len)?;

        let DeviceCombineScratch {
            rank0_send,
            rank0_recv,
            rank1_send,
            rank1_recv,
            ..
        } = &mut *scratch;
        let rank0_send = rank0_send
            .as_ref()
            .context("DeepSeek-V2-Lite NCCL rank0 combine send scratch is missing")?;
        let rank0_recv = rank0_recv
            .as_mut()
            .context("DeepSeek-V2-Lite NCCL rank0 combine recv scratch is missing")?;
        let rank1_send = rank1_send
            .as_ref()
            .context("DeepSeek-V2-Lite NCCL rank1 combine send scratch is missing")?;
        let rank1_recv = rank1_recv
            .as_mut()
            .context("DeepSeek-V2-Lite NCCL rank1 combine recv scratch is missing")?;

        self.all_reduce_f32_pair_chunked(
            rank0,
            rank1,
            rank0_send,
            rank0_recv,
            rank1_send,
            rank1_recv,
            elems,
            "DeepSeek-V2-Lite NCCL combine all-reduce",
        )?;

        activate(rank0)?;
        let mut routed = HiddenStates::zeros(rank0, hidden_dim, seq_len)?;
        ops::f32_to_bf16_hidden_into(rank0, rank0_recv, &mut routed)?;
        Ok(routed)
    }

    pub(crate) fn prepare_graph_shape(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.dense_exchange_scratch()?
            .ensure(rank0, rank1, hidden_dim, seq_len)?;
        self.combine_scratch()?
            .ensure(rank0, rank1, hidden_dim, seq_len)?;
        Ok(())
    }

    pub(crate) fn combine_device_contributions_to_rank0_into(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        hidden_dim: usize,
        seq_len: usize,
        out: &mut HiddenStates,
    ) -> Result<()> {
        ensure!(
            out.hidden_dim == hidden_dim && out.seq_len == seq_len,
            "DeepSeek-V2-Lite NCCL combine output shape mismatch: out=[{}, {}], requested=[{}, {}]",
            out.hidden_dim,
            out.seq_len,
            hidden_dim,
            seq_len
        );
        let mut scratch = self.combine_scratch()?;
        let elems = scratch.ensure_shape(hidden_dim, seq_len)?;

        let DeviceCombineScratch {
            rank0_send,
            rank0_recv,
            rank1_send,
            rank1_recv,
            ..
        } = &mut *scratch;
        let rank0_send = rank0_send
            .as_ref()
            .context("DeepSeek-V2-Lite NCCL rank0 combine send scratch is missing")?;
        let rank0_recv = rank0_recv
            .as_mut()
            .context("DeepSeek-V2-Lite NCCL rank0 combine recv scratch is missing")?;
        let rank1_send = rank1_send
            .as_ref()
            .context("DeepSeek-V2-Lite NCCL rank1 combine send scratch is missing")?;
        let rank1_recv = rank1_recv
            .as_mut()
            .context("DeepSeek-V2-Lite NCCL rank1 combine recv scratch is missing")?;

        self.all_reduce_f32_pair_chunked(
            rank0,
            rank1,
            rank0_send,
            rank0_recv,
            rank1_send,
            rank1_recv,
            elems,
            "DeepSeek-V2-Lite NCCL combine all-reduce",
        )?;

        activate(rank0)?;
        ops::f32_to_bf16_hidden_into(rank0, rank0_recv, out)
    }

    fn smoke_all_reduce_f32(&self, rank0: &DeviceContext, rank1: &DeviceContext) -> Result<()> {
        activate(rank0)?;
        let rank0_send = rank0.stream.clone_htod(&[1.0f32])?;
        let mut rank0_recv = rank0.stream.alloc_zeros::<f32>(1)?;
        activate(rank1)?;
        let rank1_send = rank1.stream.clone_htod(&[2.0f32])?;
        let mut rank1_recv = rank1.stream.alloc_zeros::<f32>(1)?;

        self.grouped("DeepSeek-V2-Lite NCCL EP=2 init smoke all-reduce", || {
            activate(rank0)?;
            self.all_reduce_f32(
                0,
                &rank0_send,
                &mut rank0_recv,
                1,
                rank0.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL init smoke rank0 all-reduce",
            )?;
            activate(rank1)?;
            self.all_reduce_f32(
                1,
                &rank1_send,
                &mut rank1_recv,
                1,
                rank1.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL init smoke rank1 all-reduce",
            )?;
            Ok(())
        })?;
        rank0.sync()?;
        rank1.sync()?;

        activate(rank0)?;
        let rank0_value = rank0.stream.clone_dtoh(&rank0_recv)?;
        rank0.sync()?;
        activate(rank1)?;
        let rank1_value = rank1.stream.clone_dtoh(&rank1_recv)?;
        rank1.sync()?;
        ensure!(
            rank0_value == [3.0] && rank1_value == [3.0],
            "DeepSeek-V2-Lite NCCL EP=2 init smoke all-reduce returned rank0={rank0_value:?}, rank1={rank1_value:?}, expected [3.0]"
        );
        Ok(())
    }

    pub(crate) fn graph_smoke_all_reduce_f32(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
    ) -> NcclGraphSmokeReport {
        let mut report = NcclGraphSmokeReport {
            attempted: true,
            captured: false,
            replayed: false,
            verified: false,
            count: 1,
            expected_sum: 3.0,
            rank0_value: None,
            rank1_value: None,
            capture_error: None,
            replay_error: None,
            verification_error: None,
            capture_mode: "thread_local",
        };

        if let Err(err) = self.graph_smoke_all_reduce_f32_inner(rank0, rank1, &mut report) {
            let message = format!("{err:#}");
            if report.captured {
                report.replay_error = Some(message);
            } else {
                report.capture_error = Some(message);
            }
        }
        report
    }

    fn graph_smoke_all_reduce_f32_inner(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        report: &mut NcclGraphSmokeReport,
    ) -> Result<()> {
        activate(rank0)?;
        let rank0_send = rank0.stream.clone_htod(&[1.0f32])?;
        let mut rank0_recv = rank0.stream.alloc_zeros::<f32>(report.count)?;
        activate(rank1)?;
        let rank1_send = rank1.stream.clone_htod(&[2.0f32])?;
        let mut rank1_recv = rank1.stream.alloc_zeros::<f32>(report.count)?;
        rank0.sync()?;
        rank1.sync()?;

        let rank0_stream = rank0.stream.clone();
        let rank1_stream = rank1.stream.clone();
        let (rank0_send_ptr, rank0_send_guard) = rank0_send.device_ptr(&rank0_stream);
        let (rank0_recv_ptr, rank0_recv_guard) = rank0_recv.device_ptr_mut(&rank0_stream);
        let (rank1_send_ptr, rank1_send_guard) = rank1_send.device_ptr(&rank1_stream);
        let (rank1_recv_ptr, rank1_recv_guard) = rank1_recv.device_ptr_mut(&rank1_stream);

        let graph0;
        let graph1;
        let mut rank0_capture_started = false;
        let mut rank1_capture_started = false;
        let capture_result = (|| -> Result<(RawCudaGraph, RawCudaGraph)> {
            let _activation_guard = graph_capture_activation_guard();
            activate_graph_capture(rank0)?;
            begin_capture(rank0.stream.cu_stream(), "rank0")?;
            rank0_capture_started = true;
            activate_graph_capture(rank1)?;
            begin_capture(rank1.stream.cu_stream(), "rank1")?;
            rank1_capture_started = true;

            self.grouped(
                "DeepSeek-V2-Lite NCCL graph smoke all-reduce capture",
                || {
                    activate_graph_capture(rank0)?;
                    self.all_reduce_f32_raw(
                        0,
                        rank0_send_ptr,
                        rank0_recv_ptr,
                        report.count,
                        rank0.stream.cu_stream(),
                        "DeepSeek-V2-Lite NCCL graph smoke rank0 all-reduce",
                    )?;
                    activate_graph_capture(rank1)?;
                    self.all_reduce_f32_raw(
                        1,
                        rank1_send_ptr,
                        rank1_recv_ptr,
                        report.count,
                        rank1.stream.cu_stream(),
                        "DeepSeek-V2-Lite NCCL graph smoke rank1 all-reduce",
                    )?;
                    Ok(())
                },
            )?;

            activate_graph_capture(rank0)?;
            let captured0 = end_capture(rank0.stream.cu_stream(), "rank0")?;
            rank0_capture_started = false;
            activate_graph_capture(rank1)?;
            let captured1 = end_capture(rank1.stream.cu_stream(), "rank1")?;
            rank1_capture_started = false;
            report.captured = true;
            activate_graph_capture(rank0)?;
            let graph0 = captured0.instantiate("rank0")?;
            activate_graph_capture(rank1)?;
            let graph1 = captured1.instantiate("rank1")?;
            Ok((graph0, graph1))
        })();

        match capture_result {
            Ok((captured0, captured1)) => {
                graph0 = captured0;
                graph1 = captured1;
            }
            Err(err) => {
                cleanup_capture(rank0, rank0_capture_started);
                cleanup_capture(rank1, rank1_capture_started);
                return Err(err);
            }
        }

        launch_graph_pair_and_sync(
            &graph0,
            &graph1,
            rank0.device_ordinal,
            rank0.stream.cu_stream(),
            rank1.device_ordinal,
            rank1.stream.cu_stream(),
        )
        .context("launch paired NCCL CUDA Graph smoke graphs")?;
        report.replayed = true;
        drop(rank0_send_guard);
        drop(rank0_recv_guard);
        drop(rank1_send_guard);
        drop(rank1_recv_guard);

        activate(rank0)?;
        let rank0_values = rank0.stream.clone_dtoh(&rank0_recv)?;
        rank0.sync()?;
        activate(rank1)?;
        let rank1_values = rank1.stream.clone_dtoh(&rank1_recv)?;
        rank1.sync()?;
        report.rank0_value = rank0_values.first().copied();
        report.rank1_value = rank1_values.first().copied();
        if rank0_values == [report.expected_sum] && rank1_values == [report.expected_sum] {
            report.verified = true;
        } else {
            report.verification_error = Some(format!(
                "expected [{expected}], got rank0={rank0_values:?}, rank1={rank1_values:?}",
                expected = report.expected_sum
            ));
        }
        Ok(())
    }

    fn validate_communicators(&self, expected_ordinals: &[c_int; 2]) -> Result<()> {
        for (rank, expected_ordinal) in expected_ordinals.iter().copied().enumerate() {
            let comm = self.comm(rank)?;
            let count = self.lib.query_comm_count(
                comm,
                &format!("DeepSeek-V2-Lite NCCL communicator rank {rank} world-size query"),
            )?;
            ensure!(
                count == self.comms.len() as c_int,
                "DeepSeek-V2-Lite NCCL communicator rank {rank} world size mismatch: got {count}, expected {}",
                self.comms.len()
            );
            let device = self.lib.query_comm_cu_device(
                comm,
                &format!("DeepSeek-V2-Lite NCCL communicator rank {rank} device query"),
            )?;
            ensure!(
                device == expected_ordinal,
                "DeepSeek-V2-Lite NCCL communicator rank {rank} CUDA device mismatch: got {device}, expected {expected_ordinal}"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn all_reduce_bf16_pair_chunked(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        rank0_send: &CudaSlice<bf16>,
        rank0_recv: &mut CudaSlice<bf16>,
        rank1_send: &CudaSlice<bf16>,
        rank1_recv: &mut CudaSlice<bf16>,
        count: usize,
        context: &str,
    ) -> Result<()> {
        for chunk in bf16_all_reduce_chunks(count) {
            let chunk_context =
                format!("{context} chunk offset={} len={}", chunk.offset, chunk.len);
            self.grouped(&chunk_context, || {
                activate(rank0)?;
                self.all_reduce_bf16_range(
                    0,
                    rank0_send,
                    rank0_recv,
                    chunk.offset,
                    chunk.len,
                    rank0.stream.cu_stream(),
                    "DeepSeek-V2-Lite NCCL dense hidden rank0 all-reduce",
                )?;
                activate(rank1)?;
                self.all_reduce_bf16_range(
                    1,
                    rank1_send,
                    rank1_recv,
                    chunk.offset,
                    chunk.len,
                    rank1.stream.cu_stream(),
                    "DeepSeek-V2-Lite NCCL dense hidden rank1 all-reduce",
                )?;
                Ok(())
            })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn all_reduce_bf16_range(
        &self,
        rank: usize,
        send: &CudaSlice<bf16>,
        recv: &mut CudaSlice<bf16>,
        offset: usize,
        count: usize,
        stream: CUstream,
        context: &str,
    ) -> Result<()> {
        ensure!(
            offset
                .checked_add(count)
                .is_some_and(|end| end <= send.len() && end <= recv.len()),
            "{context}: dense exchange buffer range out of bounds: offset={offset}, count={count}, send={}, recv={}",
            send.len(),
            recv.len()
        );
        let stream_ref = recv.stream().clone();
        let (send_ptr, _send_guard) = send.device_ptr(&stream_ref);
        let (recv_ptr, _recv_guard) = recv.device_ptr_mut(&stream_ref);
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<bf16>())
            .context("DeepSeek-V2-Lite NCCL bf16 all-reduce byte offset overflow")?;
        let status = unsafe {
            // SAFETY: Device pointers come from cudarc allocations on the
            // active CUDA devices, and `count` plus `offset` were checked
            // against both buffers. `CUdeviceptr` arithmetic uses byte offsets.
            (self.lib.all_reduce)(
                (send_ptr + byte_offset as u64) as *const c_void,
                (recv_ptr + byte_offset as u64) as *mut c_void,
                count,
                ncclDataType_t::ncclBfloat16,
                ncclRedOp_t::ncclSum,
                self.comm(rank)?,
                stream,
            )
        };
        self.lib.check(status, context)
    }

    fn all_reduce_f32(
        &self,
        rank: usize,
        send: &CudaSlice<f32>,
        recv: &mut CudaSlice<f32>,
        count: usize,
        stream: CUstream,
        context: &str,
    ) -> Result<()> {
        ensure!(
            send.len() >= count && recv.len() >= count,
            "{context}: contribution buffer too small: send={}, recv={}, required={count}",
            send.len(),
            recv.len()
        );
        self.all_reduce_f32_range(rank, send, recv, 0, count, stream, context)
    }

    #[allow(clippy::too_many_arguments)]
    fn all_reduce_f32_pair_chunked(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        rank0_send: &CudaSlice<f32>,
        rank0_recv: &mut CudaSlice<f32>,
        rank1_send: &CudaSlice<f32>,
        rank1_recv: &mut CudaSlice<f32>,
        count: usize,
        context: &str,
    ) -> Result<()> {
        for chunk in f32_all_reduce_chunks(count) {
            let chunk_context =
                format!("{context} chunk offset={} len={}", chunk.offset, chunk.len);
            self.grouped(&chunk_context, || {
                activate(rank0)?;
                self.all_reduce_f32_range(
                    0,
                    rank0_send,
                    rank0_recv,
                    chunk.offset,
                    chunk.len,
                    rank0.stream.cu_stream(),
                    "DeepSeek-V2-Lite NCCL combine rank0 all-reduce",
                )?;
                activate(rank1)?;
                self.all_reduce_f32_range(
                    1,
                    rank1_send,
                    rank1_recv,
                    chunk.offset,
                    chunk.len,
                    rank1.stream.cu_stream(),
                    "DeepSeek-V2-Lite NCCL combine rank1 all-reduce",
                )?;
                Ok(())
            })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn all_reduce_f32_range(
        &self,
        rank: usize,
        send: &CudaSlice<f32>,
        recv: &mut CudaSlice<f32>,
        offset: usize,
        count: usize,
        stream: CUstream,
        context: &str,
    ) -> Result<()> {
        ensure!(
            offset
                .checked_add(count)
                .is_some_and(|end| end <= send.len() && end <= recv.len()),
            "{context}: contribution buffer range out of bounds: offset={offset}, count={count}, send={}, recv={}",
            send.len(),
            recv.len()
        );
        let stream_ref = recv.stream().clone();
        let (send_ptr, _send_guard) = send.device_ptr(&stream_ref);
        let (recv_ptr, _recv_guard) = recv.device_ptr_mut(&stream_ref);
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<f32>())
            .context("DeepSeek-V2-Lite NCCL f32 all-reduce byte offset overflow")?;
        let status = unsafe {
            // SAFETY: Device pointers come from cudarc allocations and `count`
            // plus `offset` were checked against both buffers before enqueueing
            // the collective. `CUdeviceptr` arithmetic uses byte offsets.
            self.enqueue_all_reduce_f32(
                rank,
                send_ptr + byte_offset as u64,
                recv_ptr + byte_offset as u64,
                count,
                stream,
            )?
        };
        self.lib.check(status, context)
    }

    fn all_reduce_f32_raw(
        &self,
        rank: usize,
        send_ptr: CUdeviceptr,
        recv_ptr: CUdeviceptr,
        count: usize,
        stream: CUstream,
        context: &str,
    ) -> Result<()> {
        let status = unsafe {
            // SAFETY: The caller pre-validates that pointers come from live
            // device allocations with at least `count` f32 elements and keeps
            // the cudarc access guards alive until capture/enqueue completes.
            self.enqueue_all_reduce_f32(rank, send_ptr, recv_ptr, count, stream)?
        };
        self.lib.check(status, context)
    }

    unsafe fn enqueue_all_reduce_f32(
        &self,
        rank: usize,
        send_ptr: CUdeviceptr,
        recv_ptr: CUdeviceptr,
        count: usize,
        stream: CUstream,
    ) -> Result<ncclResult_t> {
        Ok(unsafe {
            (self.lib.all_reduce)(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                count,
                ncclDataType_t::ncclFloat32,
                ncclRedOp_t::ncclSum,
                self.comm(rank)?,
                stream,
            )
        })
    }

    fn comm(&self, rank: usize) -> Result<ncclComm_t> {
        let comm = *self.comms.get(rank).ok_or_else(|| {
            anyhow::anyhow!("DeepSeek-V2-Lite NCCL communicator rank {rank} is missing")
        })?;
        ensure!(
            !comm.is_null(),
            "DeepSeek-V2-Lite NCCL communicator rank {rank} is null"
        );
        Ok(comm)
    }

    fn combine_scratch(&self) -> Result<MutexGuard<'_, DeviceCombineScratch>> {
        self.combine_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("DeepSeek-V2-Lite NCCL device combine scratch poisoned"))
    }

    fn dense_exchange_scratch(&self) -> Result<MutexGuard<'_, DeviceDenseExchangeScratch>> {
        self.dense_exchange_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("DeepSeek-V2-Lite NCCL dense exchange scratch poisoned"))
    }

    fn grouped(&self, context: &str, f: impl FnOnce() -> Result<()>) -> Result<()> {
        let start = unsafe {
            // SAFETY: NCCL group state is process-global and entered/exited on
            // this single host thread for the paired rank0/rank1 calls.
            (self.lib.group_start)()
        };
        self.lib.check(start, &format!("{context}: group_start"))?;
        let op_result = f();
        let end = unsafe {
            // SAFETY: Matches the successful `group_start` above.
            (self.lib.group_end)()
        };
        let end_result = self.lib.check(end, &format!("{context}: group_end"));
        op_result?;
        end_result
    }
}

fn combine_elems(hidden_dim: usize, seq_len: usize) -> Result<usize> {
    ensure!(
        hidden_dim > 0 && seq_len > 0,
        "DeepSeek-V2-Lite NCCL device combine requires non-empty shape, got hidden_dim={hidden_dim}, seq_len={seq_len}"
    );
    hidden_dim.checked_mul(seq_len).with_context(|| {
        format!(
            "DeepSeek-V2-Lite NCCL device combine shape overflow: hidden_dim={hidden_dim}, seq_len={seq_len}"
        )
    })
}

fn f32_all_reduce_chunks(count: usize) -> Vec<AllReduceChunk> {
    all_reduce_chunks(count, NCCL_F32_ALL_REDUCE_MAX_ELEMS_PER_CALL)
}

fn bf16_all_reduce_chunks(count: usize) -> Vec<AllReduceChunk> {
    all_reduce_chunks(count, NCCL_BF16_ALL_REDUCE_MAX_ELEMS_PER_CALL)
}

fn all_reduce_chunks(count: usize, max_elems_per_call: usize) -> Vec<AllReduceChunk> {
    debug_assert!(max_elems_per_call > 0);
    if count == 0 {
        return Vec::new();
    }
    let mut chunks = Vec::with_capacity(count.div_ceil(max_elems_per_call));
    let mut offset = 0usize;
    while offset < count {
        let len = (count - offset).min(max_elems_per_call);
        chunks.push(AllReduceChunk { offset, len });
        offset += len;
    }
    chunks
}

fn dense_exchange_elems(hidden_dim: usize, seq_len: usize) -> Result<usize> {
    ensure!(
        hidden_dim > 0 && seq_len > 0,
        "DeepSeek-V2-Lite NCCL dense exchange requires non-empty shape, got hidden_dim={hidden_dim}, seq_len={seq_len}"
    );
    hidden_dim.checked_mul(seq_len).with_context(|| {
        format!(
            "DeepSeek-V2-Lite NCCL dense exchange shape overflow: hidden_dim={hidden_dim}, seq_len={seq_len}"
        )
    })
}

fn cleanup_capture(ctx: &DeviceContext, capture_started: bool) {
    if capture_started {
        let _ = activate(ctx);
        let _ = end_capture(ctx.stream.cu_stream(), "cleanup");
    }
}

pub(crate) struct CapturedCudaGraph {
    graph: CUgraph,
}

impl CapturedCudaGraph {
    pub(crate) fn instantiate(mut self, rank_label: &str) -> Result<RawCudaGraph> {
        let mut exec = ptr::null_mut();
        let status = unsafe {
            // SAFETY: `graph` was returned by `cuStreamEndCapture` and is
            // still owned by this captured graph wrapper.
            cudarc::driver::sys::cuGraphInstantiateWithFlags(&raw mut exec, self.graph, 0)
        };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS || exec.is_null() {
            bail!("instantiate CUDA Graph on {rank_label} stream failed with {status:?}");
        }
        let graph = self.graph;
        self.graph = ptr::null_mut();
        Ok(RawCudaGraph { graph, exec })
    }
}

impl Drop for CapturedCudaGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            let _ = unsafe {
                // SAFETY: Best-effort destruction for an uninstantiated graph
                // owned by the smoke helper.
                cudarc::driver::sys::cuGraphDestroy(self.graph)
            };
            self.graph = ptr::null_mut();
        }
    }
}

pub(crate) struct RawCudaGraph {
    graph: CUgraph,
    exec: CUgraphExec,
}

impl RawCudaGraph {
    fn exec(&self) -> CUgraphExec {
        self.exec
    }
}

fn launch_exec_and_sync_on_device(
    exec: usize,
    device_ordinal: usize,
    stream: usize,
    label: &'static str,
) -> Result<()> {
    let err = unsafe { core_ffi::cuda_set_device(device_ordinal as i32) };
    ensure!(
        err == 0,
        "{label}: failed to activate CUDA device {device_ordinal}: cudaError={err}"
    );
    let stream = stream as CUstream;
    let status = unsafe {
        // SAFETY: `exec` is a live graph exec owned by the caller for the
        // duration of the paired replay, and `stream` is that rank's stream.
        cudarc::driver::sys::cuGraphLaunch(exec as CUgraphExec, stream)
    };
    ensure!(
        status == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label}: cuGraphLaunch failed with {status:?}"
    );
    let status = unsafe {
        // SAFETY: `stream` is the live rank stream used for graph replay.
        cudarc::driver::sys::cuStreamSynchronize(stream)
    };
    ensure!(
        status == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label}: cuStreamSynchronize failed with {status:?}"
    );
    Ok(())
}

pub(crate) fn launch_graph_pair_and_sync(
    graph0: &RawCudaGraph,
    graph1: &RawCudaGraph,
    rank0_device_ordinal: usize,
    rank0_stream: CUstream,
    rank1_device_ordinal: usize,
    rank1_stream: CUstream,
) -> Result<()> {
    ensure!(
        !graph0.exec().is_null() && !graph1.exec().is_null(),
        "cannot launch destroyed CUDA Graph exec"
    );
    let rank0_exec = graph0.exec() as usize;
    let rank1_exec = graph1.exec() as usize;
    let rank0_stream = rank0_stream as usize;
    let rank1_stream = rank1_stream as usize;
    let rank0 = thread::Builder::new()
        .name("dsv2-lite-rank0-graph-replay".to_string())
        .spawn(move || {
            launch_exec_and_sync_on_device(
                rank0_exec,
                rank0_device_ordinal,
                rank0_stream,
                "rank0 CUDA Graph replay",
            )
        })
        .context("spawn rank0 CUDA Graph replay thread")?;
    let rank1 = thread::Builder::new()
        .name("dsv2-lite-rank1-graph-replay".to_string())
        .spawn(move || {
            launch_exec_and_sync_on_device(
                rank1_exec,
                rank1_device_ordinal,
                rank1_stream,
                "rank1 CUDA Graph replay",
            )
        })
        .context("spawn rank1 CUDA Graph replay thread")?;

    let rank0_result = rank0
        .join()
        .map_err(|_| anyhow::anyhow!("rank0 CUDA Graph replay thread panicked"))?;
    let rank1_result = rank1
        .join()
        .map_err(|_| anyhow::anyhow!("rank1 CUDA Graph replay thread panicked"))?;
    rank0_result?;
    rank1_result?;
    Ok(())
}

impl Drop for RawCudaGraph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let _ = unsafe {
                // SAFETY: Best-effort destruction for graph exec owned by this
                // smoke helper.
                cudarc::driver::sys::cuGraphExecDestroy(self.exec)
            };
            self.exec = ptr::null_mut();
        }
        if !self.graph.is_null() {
            let _ = unsafe {
                // SAFETY: Best-effort destruction for graph owned by this
                // smoke helper.
                cudarc::driver::sys::cuGraphDestroy(self.graph)
            };
            self.graph = ptr::null_mut();
        }
    }
}

pub(crate) fn begin_capture(stream: CUstream, rank_label: &str) -> Result<()> {
    let status = unsafe {
        // SAFETY: `stream` is a live rank stream. This smoke intentionally
        // avoids context rebinding inside the capture window to match
        // nccl-tests' per-stream capture shape.
        cudarc::driver::sys::cuStreamBeginCapture_v2(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
    };
    ensure!(
        status == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "begin CUDA Graph capture on {rank_label} stream failed with {status:?}"
    );
    Ok(())
}

pub(crate) fn end_capture(stream: CUstream, rank_label: &str) -> Result<CapturedCudaGraph> {
    let mut graph = ptr::null_mut();
    let status = unsafe {
        // SAFETY: Matches `begin_capture` on the same live rank stream.
        cudarc::driver::sys::cuStreamEndCapture(stream, &raw mut graph)
    };
    ensure!(
        status == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !graph.is_null(),
        "end CUDA Graph capture on {rank_label} stream failed with {status:?}"
    );
    Ok(CapturedCudaGraph { graph })
}

impl Drop for NaiveNcclEp2Backend {
    fn drop(&mut self) {
        for comm in &mut self.comms {
            if !comm.is_null() {
                let _ = unsafe {
                    // SAFETY: Abort is non-collective and safe for
                    // best-effort teardown in Drop.
                    (self.lib.comm_abort)(*comm)
                };
                *comm = ptr::null_mut();
            }
        }
    }
}

impl RawNcclLib {
    fn load() -> Result<Self> {
        let mut tried = Vec::new();
        for candidate in nccl_library_candidates() {
            tried.push(candidate.clone());
            let Ok(library) = (unsafe {
                // SAFETY: Loading NCCL is required to create the selected
                // runtime backend. All symbols are validated immediately below.
                Library::new(&candidate)
            }) else {
                continue;
            };
            return unsafe {
                // SAFETY: The library is kept alive inside `RawNcclLib`; copied
                // function pointers do not outlive it.
                Self::from_library(library, candidate.clone())
            }
            .with_context(|| format!("load DeepSeek-V2-Lite NCCL backend from {candidate}"));
        }
        bail!(
            "DeepSeek-V2-Lite NCCL backend could not load libnccl; tried {}",
            tried.join(", ")
        )
    }

    unsafe fn from_library(library: Library, source: String) -> Result<Self> {
        Ok(Self {
            comm_init_all: unsafe { load_symbol(&library, b"ncclCommInitAll\0")? },
            comm_count: unsafe { load_symbol(&library, b"ncclCommCount\0")? },
            comm_cu_device: unsafe { load_symbol(&library, b"ncclCommCuDevice\0")? },
            comm_abort: unsafe { load_symbol(&library, b"ncclCommAbort\0")? },
            group_start: unsafe { load_symbol(&library, b"ncclGroupStart\0")? },
            group_end: unsafe { load_symbol(&library, b"ncclGroupEnd\0")? },
            all_reduce: unsafe { load_symbol(&library, b"ncclAllReduce\0")? },
            get_error_string: unsafe { load_symbol(&library, b"ncclGetErrorString\0")? },
            source,
            _library: library,
        })
    }

    fn check(&self, status: ncclResult_t, context: &str) -> Result<()> {
        if status == ncclResult_t::ncclSuccess {
            return Ok(());
        }
        let message = unsafe {
            // SAFETY: NCCL returns a static null-terminated string for known
            // result codes; null is handled defensively.
            let ptr = (self.get_error_string)(status);
            if ptr.is_null() {
                format!("{status:?}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        bail!(
            "{context} failed with NCCL library {}: {message} ({status:?})",
            self.source
        )
    }

    fn query_comm_count(&self, comm: ncclComm_t, context: &str) -> Result<c_int> {
        let mut count = 0;
        let status = unsafe {
            // SAFETY: `count` is a valid out pointer and `comm` was validated
            // by the caller as a non-null communicator handle.
            (self.comm_count)(comm, &raw mut count)
        };
        self.check(status, context)?;
        Ok(count)
    }

    fn query_comm_cu_device(&self, comm: ncclComm_t, context: &str) -> Result<c_int> {
        let mut device = -1;
        let status = unsafe {
            // SAFETY: `device` is a valid out pointer and `comm` was validated
            // by the caller as a non-null communicator handle.
            (self.comm_cu_device)(comm, &raw mut device)
        };
        self.check(status, context)?;
        Ok(device)
    }
}

fn nccl_library_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    add_env_file_candidates(&mut candidates, &mut seen, "OPENINFER_NCCL_LIB");
    add_env_dir_candidates(&mut candidates, &mut seen, "OPENINFER_NCCL_LIB_DIR");
    add_env_dir_candidates(&mut candidates, &mut seen, "OPENINFER_NCCL_LIBRARY_PATH");
    for lib_dir in nccl_python_wheel_lib_dirs() {
        add_nccl_dir_candidates(&mut candidates, &mut seen, &lib_dir);
    }

    add_candidate(&mut candidates, &mut seen, "libnccl.so.2".to_string());
    add_candidate(&mut candidates, &mut seen, "libnccl.so".to_string());
    candidates
}

fn add_env_file_candidates(candidates: &mut Vec<String>, seen: &mut HashSet<String>, key: &str) {
    let Ok(value) = env::var(key) else {
        return;
    };
    for path in env::split_paths(&value) {
        add_candidate(candidates, seen, path.to_string_lossy().into_owned());
    }
}

fn add_env_dir_candidates(candidates: &mut Vec<String>, seen: &mut HashSet<String>, key: &str) {
    let Ok(value) = env::var(key) else {
        return;
    };
    for dir in env::split_paths(&value) {
        add_nccl_dir_candidates(candidates, seen, &dir);
    }
}

fn add_nccl_dir_candidates(candidates: &mut Vec<String>, seen: &mut HashSet<String>, dir: &Path) {
    add_candidate(
        candidates,
        seen,
        dir.join("libnccl.so.2").to_string_lossy().into_owned(),
    );
    add_candidate(
        candidates,
        seen,
        dir.join("libnccl.so").to_string_lossy().into_owned(),
    );
}

fn add_candidate(candidates: &mut Vec<String>, seen: &mut HashSet<String>, candidate: String) {
    if !candidate.is_empty() && seen.insert(candidate.clone()) {
        candidates.push(candidate);
    }
}

fn nccl_python_wheel_lib_dirs() -> Vec<PathBuf> {
    python_env_roots()
        .into_iter()
        .flat_map(|root| nccl_python_wheel_lib_dirs_from_root(&root))
        .collect()
}

fn python_env_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut seen = HashSet::new();

    for key in ["OPENINFER_NCCL_PYTHON", "OPENINFER_TRITON_PYTHON"] {
        if let Ok(value) = env::var(key) {
            add_python_env_root(&mut roots, &mut seen, Path::new(&value));
        }
    }
    for key in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Ok(value) = env::var(key) {
            add_pathbuf_once(&mut roots, &mut seen, PathBuf::from(value));
        }
    }
    roots
}

fn add_python_env_root(roots: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, python: &Path) {
    if python.is_dir() {
        add_pathbuf_once(roots, seen, python.to_path_buf());
        return;
    }
    if let Some(parent) = python.parent()
        && parent.file_name().is_some_and(|name| name == "bin")
        && let Some(root) = parent.parent()
    {
        add_pathbuf_once(roots, seen, root.to_path_buf());
    }
}

fn add_pathbuf_once(paths: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    if !path.as_os_str().is_empty() && seen.insert(path.clone()) {
        paths.push(path);
    }
}

fn nccl_python_wheel_lib_dirs_from_root(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut seen = HashSet::new();
    add_python_wheel_lib_dir(
        &mut dirs,
        &mut seen,
        root.join("site-packages/nvidia/nccl/lib"),
    );

    let lib_root = root.join("lib");
    if let Ok(entries) = fs::read_dir(&lib_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("python") {
                add_python_wheel_lib_dir(
                    &mut dirs,
                    &mut seen,
                    path.join("site-packages/nvidia/nccl/lib"),
                );
            }
        }
    }

    add_python_wheel_lib_dir(
        &mut dirs,
        &mut seen,
        root.join("Lib/site-packages/nvidia/nccl/lib"),
    );
    dirs
}

fn add_python_wheel_lib_dir(dirs: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, dir: PathBuf) {
    if nccl_lib_dir_exists(&dir) && seen.insert(dir.clone()) {
        dirs.push(dir);
    }
}

fn nccl_lib_dir_exists(dir: &Path) -> bool {
    dir.join("libnccl.so.2").exists() || dir.join("libnccl.so").exists()
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &'static [u8]) -> Result<T> {
    unsafe { library.get::<T>(symbol) }
        .map(|symbol| *symbol)
        .with_context(|| {
            format!(
                "DeepSeek-V2-Lite NCCL backend missing required symbol {}",
                String::from_utf8_lossy(symbol).trim_end_matches('\0')
            )
        })
}
