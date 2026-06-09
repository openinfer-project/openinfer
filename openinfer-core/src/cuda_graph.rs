use anyhow::Result;
use log::debug;

use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;

use crate::tensor::DeviceContext;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaGraphPhase {
    BeforeBeginCapture,
    AfterBeginCapture,
    BeforeEndCapture,
    AfterEndCapture,
    BeforeLaunch,
    AfterLaunch,
}

/// CUDA Graph state for decode path.
/// First decode call captures the graph; subsequent calls replay it.
pub struct CudaGraphState {
    graph: Option<CudaGraph>,
}

// SAFETY: CudaGraph contains raw CUDA pointers that are not Send by default.
// We only access the graph from the single inference thread that owns the model.
unsafe impl Send for CudaGraphState {}

impl CudaGraphState {
    pub fn new() -> Self {
        Self { graph: None }
    }

    /// Run kernel closure directly, or capture into a graph and replay.
    ///
    /// `kernels` must be a pure GPU kernel sequence — no CPU-GPU sync, no allocation.
    pub fn run_or_capture<F>(&mut self, ctx: &DeviceContext, kernels: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        self.run_or_capture_synchronized(ctx, |_| {}, kernels)
    }

    pub fn run_or_capture_synchronized<F, S>(
        &mut self,
        ctx: &DeviceContext,
        mut synchronize: S,
        kernels: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
        S: FnMut(CudaGraphPhase),
    {
        if let Some(graph) = &self.graph {
            synchronize(CudaGraphPhase::BeforeLaunch);
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("CUDA Graph launch failed: {}", e))?;
            synchronize(CudaGraphPhase::AfterLaunch);
        } else {
            debug!("Capturing CUDA Graph for decode path...");
            synchronize(CudaGraphPhase::BeforeBeginCapture);
            ctx.stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| anyhow::anyhow!("begin_capture failed: {}", e))?;
            synchronize(CudaGraphPhase::AfterBeginCapture);

            kernels()?;

            synchronize(CudaGraphPhase::BeforeEndCapture);
            self.graph = ctx
                .stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|e| anyhow::anyhow!("end_capture failed: {}", e))?;
            synchronize(CudaGraphPhase::AfterEndCapture);
            debug!("CUDA Graph captured successfully");

            if let Some(ref graph) = self.graph {
                synchronize(CudaGraphPhase::BeforeLaunch);
                graph
                    .launch()
                    .map_err(|e| anyhow::anyhow!("CUDA Graph first launch failed: {}", e))?;
                synchronize(CudaGraphPhase::AfterLaunch);
            }
        }
        Ok(())
    }
}

impl Default for CudaGraphState {
    fn default() -> Self {
        Self::new()
    }
}
