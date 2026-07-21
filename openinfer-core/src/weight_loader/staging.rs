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
            // SAFETY: every byte a DMA reads is initialized by `stage_chunk`'s
            // fill callback first; buffer reuse is gated on `dma_done`.
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
        self.ensure_uploadable(dst)?;
        anyhow::ensure!(
            dst_offset
                .checked_add(src.len())
                .is_some_and(|end| end <= dst.len()),
            "staged upload out of bounds: dst_offset {dst_offset} + src len {} > dst len {}",
            src.len(),
            dst.len()
        );
        let stream = self.stream.clone();
        let (dst_ptr, _dst_order) = dst.device_ptr_mut(&stream);
        for (i, chunk) in src.chunks(STAGE_ELEMS).enumerate() {
            let dst_at =
                dst_ptr + ((dst_offset + i * STAGE_ELEMS) * std::mem::size_of::<bf16>()) as u64;
            let fill = |stage: *mut bf16| {
                // SAFETY: the pinned buffer holds STAGE_ELEMS elements and is
                // privately owned by `self.bufs`, so `chunk` cannot overlap
                // it.
                unsafe { fill_pinned(chunk, stage) };
            };
            // SAFETY: the chunks partition `src`, so `dst_at` addresses
            // `chunk.len()` elements inside `dst` per the entry ensure, and
            // `chunk.len() <= STAGE_ELEMS` by construction.
            unsafe { self.stage_chunk(chunk.len(), dst_at, fill) }?;
        }
        Ok(())
    }

    /// Strided variant of [`Self::upload`]: gathers `take`-column row
    /// segments of a row-major `total_cols`-wide source straight into the
    /// pinned buffers, with no intermediate host copy.
    pub(crate) fn upload_cols(
        &mut self,
        src: &[bf16],
        total_cols: usize,
        col_offset: usize,
        take: usize,
        dst: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        self.ensure_uploadable(dst)?;
        anyhow::ensure!(
            total_cols > 0 && src.len().is_multiple_of(total_cols),
            "strided upload source of {} elements is not a multiple of {total_cols} columns",
            src.len()
        );
        anyhow::ensure!(
            col_offset
                .checked_add(take)
                .is_some_and(|end| end <= total_cols),
            "col range out of bounds: col_offset={col_offset} take={take} total_cols={total_cols}"
        );
        anyhow::ensure!(
            (1..=STAGE_ELEMS).contains(&take),
            "column shard width {take} outside 1..={STAGE_ELEMS}"
        );
        let rows = src.len() / total_cols;
        anyhow::ensure!(
            rows.checked_mul(take).is_some_and(|n| n == dst.len()),
            "staged upload shape mismatch: {rows} rows x {take} cols vs dst len {}",
            dst.len()
        );
        let rows_per_chunk = STAGE_ELEMS / take;
        let stream = self.stream.clone();
        let (dst_ptr, _dst_order) = dst.device_ptr_mut(&stream);
        let mut row = 0;
        while row < rows {
            let chunk_rows = rows_per_chunk.min(rows - row);
            let dst_at = dst_ptr + (row * take * std::mem::size_of::<bf16>()) as u64;
            let fill = |stage: *mut bf16| {
                // SAFETY: the pinned buffer is privately owned by
                // `self.bufs`, so `src` cannot overlap it; the subslice
                // holds `(rows - row) * total_cols` elements, covering
                // `(chunk_rows - 1) * total_cols + col_offset + take` since
                // `col_offset + take <= total_cols`.
                unsafe {
                    fill_pinned_strided(
                        &src[row * total_cols..],
                        total_cols,
                        col_offset,
                        take,
                        chunk_rows,
                        stage,
                    );
                }
            };
            // SAFETY: `[row * take, (row + chunk_rows) * take)` lies inside
            // `dst` by the rows x take bound above, and `chunk_rows * take <=
            // STAGE_ELEMS` by construction of `rows_per_chunk`.
            unsafe { self.stage_chunk(chunk_rows * take, dst_at, fill) }?;
            row += chunk_rows;
        }
        Ok(())
    }

    // Both the events and `dst`'s stream-ordered allocation are only ordered
    // against work on the stager's own stream.
    fn ensure_uploadable(&self, dst: &CudaSlice<bf16>) -> Result<()> {
        anyhow::ensure!(
            Arc::ptr_eq(dst.stream(), &self.stream),
            "staged upload into a buffer allocated on a different stream than the stager's"
        );
        anyhow::ensure!(
            !crate::tensor::has_stream_override(),
            "staged upload under a thread-local stream override is unsupported"
        );
        Ok(())
    }

    /// One staging step: wait out the next buffer's previous DMA, fill it,
    /// issue its copy.
    ///
    /// # Safety
    /// `dst_at` must address at least `elems` elements of device memory
    /// writable on the stager's stream, `elems <= STAGE_ELEMS`, and `fill`
    /// must initialize the `elems` elements at the pointer it is given.
    unsafe fn stage_chunk(
        &mut self,
        elems: usize,
        dst_at: u64,
        fill: impl FnOnce(*mut bf16),
    ) -> Result<()> {
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
        fill(stage);
        // SAFETY: `fill` initialized `elems` elements and `dst_at` is valid
        // per this function's contract. The buffer outlives the copy on every
        // path — retired via `dma_done` on success, or by the drain-or-abort
        // branches below. The event synchronize above bound the context to
        // this thread.
        let copied = unsafe {
            let staged = std::slice::from_raw_parts(stage.cast_const(), elems);
            memcpy_htod_async(dst_at, staged, self.stream.cu_stream())
        };
        if let Err(copy_err) = copied {
            // An async-API error can stem from earlier work on the stream and
            // does not prove this copy never started.
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
/// `dst` must be valid for writes of at least `rows * take` elements and must
/// not overlap `src`; when `rows > 0`, `src` must hold at least
/// `(rows - 1) * total_cols + col_offset + take` elements.
unsafe fn fill_pinned_strided(
    src: &[bf16],
    total_cols: usize,
    col_offset: usize,
    take: usize,
    rows: usize,
    dst: *mut bf16,
) {
    if rows == 0 {
        return;
    }
    let rows_per = rows.div_ceil(FILL_THREADS);
    let dst_addr = dst as usize;
    std::thread::scope(|scope| {
        for t in 0..FILL_THREADS.min(rows) {
            let start = t * rows_per;
            let end = rows.min(start + rows_per);
            if start >= end {
                break;
            }
            scope.spawn(move || {
                for row in start..end {
                    // SAFETY: each thread writes the disjoint row range
                    // `[start * take, end * take)` of a destination sized for
                    // `rows * take`; source indices are bounded per the
                    // function contract.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(row * total_cols + col_offset),
                            (dst_addr as *mut bf16).add(row * take),
                            take,
                        );
                    }
                }
            });
        }
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_pinned_strided_matches_scalar_gather() {
        for &(rows, total_cols, col_offset, take) in &[
            (1usize, 7usize, 0usize, 7usize),
            (3, 8, 2, 5),
            (4, 5, 1, 4),
            (9, 6, 3, 3),
            (17, 4, 0, 1),
        ] {
            let src: Vec<bf16> = (0..rows * total_cols)
                .map(|i| bf16::from_f32(i as f32))
                .collect();
            let mut dst = vec![bf16::ZERO; rows * take];
            // SAFETY: `dst` holds `rows * take` elements and does not overlap
            // `src`; `src` holds `rows * total_cols` elements, covering
            // `(rows - 1) * total_cols + col_offset + take` since
            // `col_offset + take <= total_cols`.
            unsafe {
                fill_pinned_strided(&src, total_cols, col_offset, take, rows, dst.as_mut_ptr());
            }
            let mut expect = Vec::with_capacity(rows * take);
            for r in 0..rows {
                for c in 0..take {
                    expect.push(src[r * total_cols + col_offset + c]);
                }
            }
            assert_eq!(
                dst, expect,
                "rows={rows} total_cols={total_cols} col_offset={col_offset} take={take}"
            );
        }
    }
}
