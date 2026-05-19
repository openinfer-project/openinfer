use std::{
    ffi::{CStr, c_char, c_void},
    ptr,
    sync::Arc,
};

use anyhow::{Context, Result, bail, ensure};
use cudarc::{
    driver::{CudaSlice, DevicePtr, DevicePtrMut, sys::CUstream},
    nccl::sys::{ncclComm_t, ncclDataType_t, ncclRedOp_t, ncclResult_t},
};
use half::bf16;
use libloading::Library;
use pegainfer_core::tensor::{DeviceContext, HiddenStates};

use crate::device::activate;

type NcclCommInitAll = unsafe extern "C" fn(
    *mut ncclComm_t,
    ::core::ffi::c_int,
    *const ::core::ffi::c_int,
) -> ncclResult_t;
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

struct RawNcclLib {
    _library: Library,
    comm_init_all: NcclCommInitAll,
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
        Ok(Self { lib, comms })
    }

    pub(crate) fn dispatch_rank0_hidden_to_rank1(
        &self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        input: &HiddenStates,
    ) -> Result<HiddenStates> {
        ensure!(
            input.hidden_dim > 0 && input.seq_len > 0,
            "DeepSeek-V2-Lite NCCL dispatch requires non-empty hidden states"
        );
        activate(rank0)?;
        let mut rank0_recv = HiddenStates::zeros(rank0, input.hidden_dim, input.seq_len)?;
        activate(rank1)?;
        let rank1_send = HiddenStates::zeros(rank1, input.hidden_dim, input.seq_len)?;
        let mut rank1_recv = HiddenStates::zeros(rank1, input.hidden_dim, input.seq_len)?;

        let count = input.hidden_dim * input.seq_len;
        self.grouped("DeepSeek-V2-Lite NCCL dispatch all-reduce", || {
            activate(rank0)?;
            self.all_reduce_bf16(
                0,
                &input.data,
                &mut rank0_recv.data,
                count,
                rank0.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL dispatch rank0 all-reduce",
            )?;
            activate(rank1)?;
            self.all_reduce_bf16(
                1,
                &rank1_send.data,
                &mut rank1_recv.data,
                count,
                rank1.stream.cu_stream(),
                "DeepSeek-V2-Lite NCCL dispatch rank1 all-reduce",
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
            (self.lib.all_reduce)(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                count,
                ncclDataType_t::ncclFloat32,
                ncclRedOp_t::ncclSum,
                self.comm(rank)?,
                stream,
            )
        };
        self.lib.check(status, context)
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

impl Drop for NaiveNcclEp2Backend {
    fn drop(&mut self) {
        for comm in &mut self.comms {
            if !comm.is_null() {
                let _ = unsafe {
                    // SAFETY: Best-effort teardown for communicator handles
                    // returned by `ncclCommInitAll`.
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
