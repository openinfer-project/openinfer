use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::PinnedHostSlice;
use cudarc::driver::result::memcpy_htod_async;
use cudarc::driver::sys::CUevent_flags;
use half::bf16;
use log::error;

use crate::tensor::DeviceContext;

/// bf16 elements per pinned staging buffer: 64 MiB amortizes the per-chunk
/// event sync while capping pinned memory at 128 MiB across both buffers.
const STAGE_ELEMS: usize = (64 << 20) / std::mem::size_of::<bf16>();
/// A single memcpy thread cannot keep up with the pinned H2D copy rate.
const FILL_THREADS: usize = 4;

struct StagingBuf {
    pinned: PinnedHostSlice<bf16>,
    dma_done: CudaEvent,
}

/// Pageable `clone_htod` serializes the page-cache read with the DMA inside
/// the driver; double-buffered pinned staging overlaps them.
pub(crate) struct WeightStager {
    stream: Arc<CudaStream>,
    bufs: [StagingBuf; 2],
    next: usize,
}

impl WeightStager {
    pub(crate) fn new(ctx: &DeviceContext) -> Result<Self> {
        let make = || -> Result<StagingBuf> {
            // SAFETY: every byte a DMA reads is written by `fill_pinned` first;
            // buffer reuse is gated on `dma_done`.
            let pinned = unsafe { ctx.ctx.alloc_pinned::<bf16>(STAGE_ELEMS) }
                .map_err(|e| anyhow::anyhow!("pinned staging alloc failed: {e}"))?;
            let dma_done = ctx
                .ctx
                .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| anyhow::anyhow!("staging event create failed: {e}"))?;
            Ok(StagingBuf { pinned, dma_done })
        };
        Ok(Self {
            stream: ctx.stream.clone(),
            bufs: [make()?, make()?],
            next: 0,
        })
    }

    /// Copy `src` to `dst[dst_offset..dst_offset + src.len()]` through the
    /// staging pipeline. The copy is asynchronous on the stager's stream, but
    /// the in-flight DMA reads only the pinned buffers: `src` is copied out
    /// synchronously and its lifetime is not extended past the call.
    pub(crate) fn upload(
        &mut self,
        src: &[bf16],
        dst: &mut CudaSlice<bf16>,
        dst_offset: usize,
    ) -> Result<()> {
        // Both the events and `dst`'s stream-ordered allocation are only
        // ordered against work on the stager's own stream.
        anyhow::ensure!(
            Arc::ptr_eq(dst.stream(), &self.stream),
            "staged upload into a buffer allocated on a different stream than the stager's"
        );
        anyhow::ensure!(
            !crate::tensor::has_stream_override(),
            "staged upload under a thread-local stream override is unsupported"
        );
        anyhow::ensure!(
            dst_offset
                .checked_add(src.len())
                .is_some_and(|end| end <= dst.len()),
            "staged upload out of bounds: dst_offset {dst_offset} + src len {} > dst len {}",
            src.len(),
            dst.len()
        );
        let (dst_ptr, _dst_order) = dst.device_ptr_mut(&self.stream);
        for (i, chunk) in src.chunks(STAGE_ELEMS).enumerate() {
            let idx = self.next;
            self.next = (self.next + 1) % self.bufs.len();
            let buf = &mut self.bufs[idx];
            buf.dma_done
                .synchronize()
                .map_err(|e| anyhow::anyhow!("staging drain failed: {e}"))?;
            let stage = buf
                .pinned
                .as_mut_ptr()
                .map_err(|e| anyhow::anyhow!("staging pointer failed: {e}"))?;
            // SAFETY: the staging buffer holds STAGE_ELEMS elements and
            // `chunk.len() <= STAGE_ELEMS` by construction of the chunk loop;
            // it is privately owned by `self.bufs` and never lent out, so the
            // caller-borrowed `chunk` cannot overlap it.
            unsafe { fill_pinned(chunk, stage) };
            let dst_at =
                dst_ptr + ((dst_offset + i * STAGE_ELEMS) * std::mem::size_of::<bf16>()) as u64;
            // SAFETY: `stage` holds `chunk.len()` freshly written elements.
            // The destination range starts at element `dst_offset + i *
            // STAGE_ELEMS` and stays inside `dst`: the entry ensure bounds
            // `dst_offset + src.len() <= dst.len()` and the chunks partition
            // `src`. The buffer outlives the copy on every path — retired via
            // `dma_done` (recorded below) on success, or by the
            // drain-or-abort branches when the copy or record call fails.
            // The event synchronize above bound the context to this thread.
            let copied = unsafe {
                let staged = std::slice::from_raw_parts(stage.cast_const(), chunk.len());
                memcpy_htod_async(dst_at, staged, self.stream.cu_stream())
            };
            if let Err(copy_err) = copied {
                // An async-API error can stem from earlier work on the stream
                // and does not prove this copy never started.
                drain_or_abort(
                    &self.stream,
                    &format!("staged H2D copy failed ({copy_err})"),
                );
                return Err(anyhow::anyhow!("staged H2D copy failed: {copy_err}"));
            }
            if let Err(record_err) = buf.dma_done.record(&self.stream) {
                // The copy is in flight with no event covering it.
                drain_or_abort(
                    &self.stream,
                    &format!("staging record failed ({record_err})"),
                );
                return Err(anyhow::anyhow!("staging record failed: {record_err}"));
            }
        }
        Ok(())
    }
}

impl Drop for WeightStager {
    fn drop(&mut self) {
        // PinnedHostSlice's own drop only waits on its embedded event, which
        // this pipeline never records; drain our events instead, and fail
        // closed when the DMA state is unknown.
        for buf in &self.bufs {
            if let Err(err) = buf.dma_done.synchronize() {
                error!(
                    "staging DMA drain failed on drop ({err}); aborting instead of freeing pinned memory under an in-flight DMA"
                );
                std::process::abort();
            }
        }
    }
}

fn drain_or_abort(stream: &CudaStream, context: &str) {
    if let Err(sync_err) = stream.synchronize() {
        error!(
            "{context}; stream drain failed ({sync_err}); aborting instead of freeing pinned memory under an in-flight DMA"
        );
        std::process::abort();
    }
}

/// Write-combined destination: forward streaming writes only, never read.
///
/// # Safety
/// `dst` must be valid for writes of at least `src.len()` elements and must
/// not overlap `src`.
unsafe fn fill_pinned(src: &[bf16], dst: *mut bf16) {
    if src.is_empty() {
        return;
    }
    let per = src.len().div_ceil(FILL_THREADS);
    let dst_addr = dst as usize;
    std::thread::scope(|scope| {
        for (i, part) in src.chunks(per).enumerate() {
            scope.spawn(move || {
                // SAFETY: each thread writes a disjoint range of the staging
                // buffer, which is at least `src.len()` elements long.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        part.as_ptr(),
                        (dst_addr as *mut bf16).add(i * per),
                        part.len(),
                    );
                }
            });
        }
    });
}
