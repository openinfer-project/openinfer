use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr,
    sync::Arc,
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
use pegainfer_core::tensor::{DeviceContext, HiddenStates};
use serde::Serialize;

use crate::device::activate;

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

pub(crate) struct NaiveNcclEp2Backend {
    lib: Arc<RawNcclLib>,
    comms: Vec<ncclComm_t>,
}

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
    comm_init_all: NcclCommInitAll,
    comm_count: NcclCommCount,
    comm_cu_device: NcclCommCuDevice,
    comm_abort: NcclCommAbort,
    group_start: NcclGroupStart,
    group_end: NcclGroupEnd,
    all_reduce: NcclAllReduce,
    get_error_string: NcclGetErrorString,
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
        let backend = Self { lib, comms };
        backend.validate_communicators(&ordinals)?;
        backend.smoke_all_reduce_f32(rank0, rank1)?;
        Ok(backend)
    }

    pub(crate) fn dense_all_reduce_rank0_hidden_to_rank1(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        input: &HiddenStates,
    ) -> Result<HiddenStates> {
        ensure!(
            input.hidden_dim > 0 && input.seq_len > 0,
            "DeepSeek-V2-Lite NCCL dense hidden exchange requires non-empty hidden states"
        );
        activate(rank0)?;
        let mut rank0_recv = HiddenStates::zeros(rank0, input.hidden_dim, input.seq_len)?;
        activate(rank1)?;
        let rank1_send = HiddenStates::zeros(rank1, input.hidden_dim, input.seq_len)?;
        let mut rank1_recv = HiddenStates::zeros(rank1, input.hidden_dim, input.seq_len)?;

        // Correctness-first dense exchange: rank0 contributes the hidden state
        // and rank1 contributes zeros. This makes rank0 hidden visible on rank1
        // without pretending to be sparse routed dispatch.
        let count = input.hidden_dim * input.seq_len;
        self.grouped("DeepSeek-V2-Lite NCCL dense hidden all-reduce", || {
            activate(rank0)?;
            self.all_reduce_bf16(
                0,
                &input.data,
                &mut rank0_recv.data,
                count,
                rank0.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL dense hidden rank0 all-reduce",
            )?;
            activate(rank1)?;
            self.all_reduce_bf16(
                1,
                &rank1_send.data,
                &mut rank1_recv.data,
                count,
                rank1.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL dense hidden rank1 all-reduce",
            )?;
            Ok(())
        })?;
        rank0.sync()?;
        rank1.sync()?;
        Ok(rank1_recv)
    }

    pub(crate) fn combine_f32_contributions_to_rank0(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        rank0_contrib: &[f32],
        rank1_contrib: &[f32],
    ) -> Result<Vec<f32>> {
        ensure!(
            !rank0_contrib.is_empty(),
            "DeepSeek-V2-Lite NCCL combine requires non-empty rank0 contribution"
        );
        ensure!(
            rank0_contrib.len() == rank1_contrib.len(),
            "DeepSeek-V2-Lite NCCL combine contribution length mismatch: rank0={}, rank1={}",
            rank0_contrib.len(),
            rank1_contrib.len()
        );
        activate(rank0)?;
        let rank0_send = rank0.stream.clone_htod(rank0_contrib)?;
        let mut rank0_recv = rank0.stream.alloc_zeros::<f32>(rank0_contrib.len())?;
        activate(rank1)?;
        let rank1_send = rank1.stream.clone_htod(rank1_contrib)?;
        let mut rank1_recv = rank1.stream.alloc_zeros::<f32>(rank1_contrib.len())?;

        self.grouped("DeepSeek-V2-Lite NCCL combine all-reduce", || {
            activate(rank0)?;
            self.all_reduce_f32(
                0,
                &rank0_send,
                &mut rank0_recv,
                rank0_contrib.len(),
                rank0.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL combine rank0 all-reduce",
            )?;
            activate(rank1)?;
            self.all_reduce_f32(
                1,
                &rank1_send,
                &mut rank1_recv,
                rank1_contrib.len(),
                rank1.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL combine rank1 all-reduce",
            )?;
            Ok(())
        })?;
        rank0.sync()?;
        rank1.sync()?;

        let combined = rank0.stream.clone_dtoh(&rank0_recv)?;
        rank0.sync()?;
        Ok(combined)
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
            activate(rank0)?;
            begin_capture(rank0.stream.cu_stream(), "rank0")?;
            rank0_capture_started = true;
            activate(rank1)?;
            begin_capture(rank1.stream.cu_stream(), "rank1")?;
            rank1_capture_started = true;

            self.grouped(
                "DeepSeek-V2-Lite NCCL graph smoke all-reduce capture",
                || {
                    activate(rank0)?;
                    self.all_reduce_f32_raw(
                        0,
                        rank0_send_ptr,
                        rank0_recv_ptr,
                        report.count,
                        rank0.stream.cu_stream(),
                        "DeepSeek-V2-Lite NCCL graph smoke rank0 all-reduce",
                    )?;
                    activate(rank1)?;
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

            activate(rank0)?;
            let captured0 = end_capture(rank0.stream.cu_stream(), "rank0")?;
            rank0_capture_started = false;
            activate(rank1)?;
            let captured1 = end_capture(rank1.stream.cu_stream(), "rank1")?;
            rank1_capture_started = false;
            report.captured = true;
            activate(rank0)?;
            let graph0 = captured0.instantiate("rank0")?;
            activate(rank1)?;
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

        activate(rank0)?;
        graph0
            .launch(rank0.stream.cu_stream(), "rank0")
            .context("launch captured rank0 NCCL CUDA Graph")?;
        activate(rank1)?;
        graph1
            .launch(rank1.stream.cu_stream(), "rank1")
            .context("launch captured rank1 NCCL CUDA Graph")?;
        report.replayed = true;
        rank0.sync()?;
        rank1.sync()?;
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

    fn all_reduce_bf16(
        &self,
        rank: usize,
        send: &CudaSlice<bf16>,
        recv: &mut CudaSlice<bf16>,
        count: usize,
        stream: CUstream,
        context: &str,
    ) -> Result<()> {
        ensure!(
            recv.len() >= count,
            "{context}: recv buffer too small: recv={}, required={count}",
            recv.len()
        );
        ensure!(
            send.len() >= count,
            "{context}: send buffer too small: send={}, required={count}",
            send.len()
        );
        let stream_ref = recv.stream().clone();
        let (send_ptr, _send_guard) = send.device_ptr(&stream_ref);
        let (recv_ptr, _recv_guard) = recv.device_ptr_mut(&stream_ref);
        let status = unsafe {
            // SAFETY: Device pointers come from cudarc allocations on the
            // active CUDA devices, and `count` was checked against both buffers.
            (self.lib.all_reduce)(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
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
        let stream_ref = recv.stream().clone();
        let (send_ptr, _send_guard) = send.device_ptr(&stream_ref);
        let (recv_ptr, _recv_guard) = recv.device_ptr_mut(&stream_ref);
        let status = unsafe {
            // SAFETY: Device pointers come from cudarc allocations and `count`
            // was checked against both buffers before enqueueing the collective.
            self.enqueue_all_reduce_f32(rank, send_ptr, recv_ptr, count, stream)?
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

fn cleanup_capture(ctx: &DeviceContext, capture_started: bool) {
    if capture_started {
        let _ = activate(ctx);
        let _ = end_capture(ctx.stream.cu_stream(), "cleanup");
    }
}

struct CapturedCudaGraph {
    graph: CUgraph,
}

impl CapturedCudaGraph {
    fn instantiate(mut self, rank_label: &str) -> Result<RawCudaGraph> {
        let mut exec = ptr::null_mut();
        let status = unsafe {
            // SAFETY: `graph` was returned by `cuStreamEndCapture` and is
            // still owned by this captured graph wrapper.
            cudarc::driver::sys::cuGraphInstantiateWithFlags(&mut exec, self.graph, 0)
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

struct RawCudaGraph {
    graph: CUgraph,
    exec: CUgraphExec,
}

impl RawCudaGraph {
    fn launch(&self, stream: CUstream, label: &str) -> Result<()> {
        let status = unsafe {
            // SAFETY: `exec` is instantiated from the graph captured on this
            // rank stream, and launch is part of the paired NCCL graph smoke.
            cudarc::driver::sys::cuGraphLaunch(self.exec, stream)
        };
        ensure!(
            status == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{label}: cuGraphLaunch failed with {status:?}"
        );
        Ok(())
    }
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

fn begin_capture(stream: CUstream, rank_label: &str) -> Result<()> {
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

fn end_capture(stream: CUstream, rank_label: &str) -> Result<CapturedCudaGraph> {
    let mut graph = ptr::null_mut();
    let status = unsafe {
        // SAFETY: Matches `begin_capture` on the same live rank stream.
        cudarc::driver::sys::cuStreamEndCapture(stream, &mut graph)
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
        for candidate in ["libnccl.so.2", "libnccl.so"] {
            tried.push(candidate);
            let library = match unsafe {
                // SAFETY: Loading NCCL is required to create the selected
                // runtime backend. All symbols are validated immediately below.
                Library::new(candidate)
            } {
                Ok(library) => library,
                Err(_) => continue,
            };
            return unsafe {
                // SAFETY: The library is kept alive inside `RawNcclLib`; copied
                // function pointers do not outlive it.
                Self::from_library(library)
            }
            .with_context(|| format!("load DeepSeek-V2-Lite NCCL backend from {candidate}"));
        }
        bail!(
            "DeepSeek-V2-Lite NCCL backend could not load libnccl; tried {}",
            tried.join(", ")
        )
    }

    unsafe fn from_library(library: Library) -> Result<Self> {
        Ok(Self {
            comm_init_all: unsafe { load_symbol(&library, b"ncclCommInitAll\0")? },
            comm_count: unsafe { load_symbol(&library, b"ncclCommCount\0")? },
            comm_cu_device: unsafe { load_symbol(&library, b"ncclCommCuDevice\0")? },
            comm_abort: unsafe { load_symbol(&library, b"ncclCommAbort\0")? },
            group_start: unsafe { load_symbol(&library, b"ncclGroupStart\0")? },
            group_end: unsafe { load_symbol(&library, b"ncclGroupEnd\0")? },
            all_reduce: unsafe { load_symbol(&library, b"ncclAllReduce\0")? },
            get_error_string: unsafe { load_symbol(&library, b"ncclGetErrorString\0")? },
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
        bail!("{context} failed: {message} ({status:?})")
    }

    fn query_comm_count(&self, comm: ncclComm_t, context: &str) -> Result<c_int> {
        let mut count = 0;
        let status = unsafe {
            // SAFETY: `count` is a valid out pointer and `comm` was validated
            // by the caller as a non-null communicator handle.
            (self.comm_count)(comm, &mut count)
        };
        self.check(status, context)?;
        Ok(count)
    }

    fn query_comm_cu_device(&self, comm: ncclComm_t, context: &str) -> Result<c_int> {
        let mut device = -1;
        let status = unsafe {
            // SAFETY: `device` is a valid out pointer and `comm` was validated
            // by the caller as a non-null communicator handle.
            (self.comm_cu_device)(comm, &mut device)
        };
        self.check(status, context)?;
        Ok(device)
    }
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
