//! Capture-only CUDA graph for a PP stage.
//!
//! This is a GLM-local sibling of `openinfer_core::cuda_graph::CudaGraphState`
//! with one critical difference: `CudaGraphState::run_or_capture` does a live
//! `cuGraphLaunch` at the end of capture to warm the graph. A PP stage graph
//! contains a spin-wait (`wait_hidden`), so that bundled first launch would
//! HANG -- the upstream stage has not released epoch-0 yet. Here capture stops
//! at instantiate; the coordinator drives explicit `launch`es once every stage
//! is captured and all rings are zeroed back to epoch 0.

use std::sync::Arc;

use anyhow::{Result, bail};
use cudarc::driver::CudaContext;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::sys::{self, CUgraph, CUgraphExec};
use openinfer_kernels::tensor::DeviceContext;

pub(crate) struct Glm52StageGraph {
    graph: CUgraph,
    exec: CUgraphExec,
    /// Anchors the context so `Drop` can destroy the graph handles regardless of
    /// field-drop order (same rationale as `CudaGraphState::_ctx`).
    _ctx: Arc<CudaContext>,
}

// SAFETY: the handles are only ever touched from the single stage thread that
// owns this graph's context.
unsafe impl Send for Glm52StageGraph {}

impl Glm52StageGraph {
    /// Capture `kernels` into a replayable graph, skipping the warm-up launch.
    /// Capture runs in `THREAD_LOCAL` mode so the eight stages can capture
    /// concurrently on their own threads without cross-contaminating each
    /// other's streams.
    pub(crate) fn capture<F>(ctx: &DeviceContext, kernels: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        let stream = ctx.stream.cu_stream();
        check(
            unsafe { sys::cuStreamBeginCapture_v2(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )?;

        // On kernel-enqueue error, end the in-progress capture so the stream is
        // not left stuck capturing, then propagate the original error.
        if let Err(e) = kernels() {
            let mut aborted: CUgraph = std::ptr::null_mut();
            unsafe { sys::cuStreamEndCapture(stream, &raw mut aborted) };
            if !aborted.is_null() {
                unsafe { sys::cuGraphDestroy(aborted) };
            }
            return Err(e);
        }

        let mut graph: CUgraph = std::ptr::null_mut();
        check(
            unsafe { sys::cuStreamEndCapture(stream, &raw mut graph) },
            "cuStreamEndCapture",
        )?;
        let mut exec: CUgraphExec = std::ptr::null_mut();
        check(
            unsafe { sys::cuGraphInstantiateWithFlags(&raw mut exec, graph, 0) },
            "cuGraphInstantiateWithFlags",
        )?;
        Ok(Self {
            graph,
            exec,
            _ctx: ctx.ctx.clone(),
        })
    }

    /// Enqueue one replay on the stage's stream. Async: returns immediately,
    /// the device flags serialize this replay against the neighbouring stages.
    pub(crate) fn launch(&self, ctx: &DeviceContext) -> Result<()> {
        check(
            unsafe { sys::cuGraphLaunch(self.exec, ctx.stream.cu_stream()) },
            "cuGraphLaunch",
        )
    }
}

impl Drop for Glm52StageGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                sys::cuGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                sys::cuGraphDestroy(self.graph);
            }
        }
    }
}

fn check(r: sys::CUresult, what: &str) -> Result<()> {
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("{what} failed: {r:?}");
    }
    Ok(())
}
