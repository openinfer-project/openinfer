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

const BF16_SIZE: usize = std::mem::size_of::<bf16>();
/// Bytes per pinned staging buffer: 64 MiB amortizes the per-chunk event sync
/// while capping pinned memory at 128 MiB across both buffers.
const STAGE_BYTES: usize = 64 << 20;
/// A single memcpy thread cannot keep up with the pinned H2D copy rate.
const FILL_THREADS: usize = 4;

struct StagingBuf {
    pinned: PinnedHostSlice<bf16>,
    dma_done: CudaEvent,
}

/// Pinned double-buffering overlaps the source read with the H2D copy that
/// pageable `clone_htod` would serialize; sources are raw bytes.
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
            let pinned = unsafe { ctx.ctx.alloc_pinned::<bf16>(STAGE_BYTES / BF16_SIZE) }
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

    /// Stage `src` into `dst` at element `dst_offset`. The copy is async on
    /// the stager's stream; `src` is copied out synchronously and not read
    /// after return.
    pub(crate) fn upload(
        &mut self,
        src: &[u8],
        dst: &mut CudaSlice<bf16>,
        dst_offset: usize,
    ) -> Result<()> {
        self.ensure_uploadable(dst)?;
        anyhow::ensure!(
            src.len().is_multiple_of(BF16_SIZE),
            "staged upload source of {} bytes is not a whole number of bf16 elements",
            src.len()
        );
        anyhow::ensure!(
            dst_offset
                .checked_mul(BF16_SIZE)
                .and_then(|off| off.checked_add(src.len()))
                .is_some_and(|end| end <= dst.len() * BF16_SIZE),
            "staged upload out of bounds: dst_offset {dst_offset} + src bytes {} > dst len {}",
            src.len(),
            dst.len()
        );
        let stream = self.stream.clone();
        let (dst_ptr, _dst_order) = dst.device_ptr_mut(&stream);
        for (i, chunk) in src.chunks(STAGE_BYTES).enumerate() {
            let dst_at = dst_ptr + (dst_offset * BF16_SIZE + i * STAGE_BYTES) as u64;
            let fill = |stage: *mut u8| {
                // SAFETY: `chunk.len() <= STAGE_BYTES`, and the privately
                // owned buffer cannot overlap `chunk`.
                unsafe { fill_pinned(chunk, stage) };
            };
            // SAFETY: the chunks partition `src`, so `dst_at` stays inside
            // `dst` per the entry ensure, with `chunk.len() <= STAGE_BYTES`.
            unsafe { self.stage_chunk(chunk.len(), dst_at, fill) }?;
        }
        Ok(())
    }

    /// Strided variant of [`Self::upload`]: gathers `take`-column row
    /// segments straight into the pinned buffers (column counts in
    /// elements).
    pub(crate) fn upload_cols(
        &mut self,
        src: &[u8],
        total_cols: usize,
        col_offset: usize,
        take: usize,
        dst: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        self.ensure_uploadable(dst)?;
        let stride_b = total_cols
            .checked_mul(BF16_SIZE)
            .filter(|&s| s > 0 && src.len().is_multiple_of(s));
        anyhow::ensure!(
            stride_b.is_some(),
            "strided upload source of {} bytes is not a multiple of {total_cols} bf16 columns",
            src.len()
        );
        let stride_b = stride_b.unwrap();
        anyhow::ensure!(
            col_offset
                .checked_add(take)
                .is_some_and(|end| end <= total_cols),
            "col range out of bounds: col_offset={col_offset} take={take} total_cols={total_cols}"
        );
        let take_b = take * BF16_SIZE;
        anyhow::ensure!(
            (1..=STAGE_BYTES).contains(&take_b),
            "column shard width {take} outside 1..={}",
            STAGE_BYTES / BF16_SIZE
        );
        let rows = src.len() / stride_b;
        anyhow::ensure!(
            rows.checked_mul(take).is_some_and(|n| n == dst.len()),
            "staged upload shape mismatch: {rows} rows x {take} cols vs dst len {}",
            dst.len()
        );
        let rows_per_chunk = STAGE_BYTES / take_b;
        let off_b = col_offset * BF16_SIZE;
        let stream = self.stream.clone();
        let (dst_ptr, _dst_order) = dst.device_ptr_mut(&stream);
        let mut row = 0;
        while row < rows {
            let chunk_rows = rows_per_chunk.min(rows - row);
            let dst_at = dst_ptr + (row * take_b) as u64;
            let fill = |stage: *mut u8| {
                // SAFETY: the privately owned buffer cannot overlap `src`,
                // and the subslice covers `chunk_rows` full rows since
                // `off_b + take_b <= stride_b`.
                unsafe {
                    fill_pinned_strided(
                        &src[row * stride_b..],
                        stride_b,
                        off_b,
                        take_b,
                        chunk_rows,
                        stage,
                    );
                }
            };
            // SAFETY: the destination rows lie inside `dst` per the
            // rows x take bound, with `chunk_rows * take_b <= STAGE_BYTES`.
            unsafe { self.stage_chunk(chunk_rows * take_b, dst_at, fill) }?;
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

    /// # Safety
    /// `dst_at` must address `bytes <= STAGE_BYTES` writable bytes on the
    /// stager's stream, and `fill` must initialize the `bytes` it is given.
    unsafe fn stage_chunk(
        &mut self,
        bytes: usize,
        dst_at: u64,
        fill: impl FnOnce(*mut u8),
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
            .map_err(|e| anyhow::anyhow!("staging pointer failed: {e}"))?
            .cast::<u8>();
        fill(stage);
        // SAFETY: `fill` initialized `bytes` at `stage` and `dst_at` is valid
        // per the contract; the buffer outlives the copy (`dma_done` or the
        // drain-or-abort branches), and the event synchronize above bound the
        // context to this thread.
        let copied = unsafe {
            let staged = std::slice::from_raw_parts(stage.cast_const(), bytes);
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
        // PinnedHostSlice's drop only waits on its embedded, never-recorded
        // event; drain ours instead and fail closed when the DMA state is
        // unknown.
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

/// # Safety
/// `dst` must hold `src.len()` writable bytes without overlapping `src`.
unsafe fn fill_pinned(src: &[u8], dst: *mut u8) {
    if src.is_empty() {
        return;
    }
    let per = src.len().div_ceil(FILL_THREADS);
    let dst_addr = dst as usize;
    std::thread::scope(|scope| {
        for (i, part) in src.chunks(per).enumerate() {
            scope.spawn(move || {
                // SAFETY: disjoint per-thread ranges within `dst`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        part.as_ptr(),
                        (dst_addr as *mut u8).add(i * per),
                        part.len(),
                    );
                }
            });
        }
    });
}

/// # Safety
/// `dst` must hold `rows * take_b` writable bytes without overlapping `src`;
/// every requested source row slice must exist.
unsafe fn fill_pinned_strided(
    src: &[u8],
    stride_b: usize,
    off_b: usize,
    take_b: usize,
    rows: usize,
    dst: *mut u8,
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
                    // SAFETY: disjoint per-thread row ranges; source rows
                    // exist per the contract.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(row * stride_b + off_b),
                            (dst_addr as *mut u8).add(row * take_b),
                            take_b,
                        );
                    }
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
            let (stride_b, off_b, take_b) = (
                total_cols * BF16_SIZE,
                col_offset * BF16_SIZE,
                take * BF16_SIZE,
            );
            let mut buf = vec![0u8; rows * stride_b];
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
            let src = &buf[..];
            let mut dst = vec![0u8; rows * take_b];
            // SAFETY: `dst` holds `rows * take_b` bytes and does not overlap
            // `src`; `src` holds `rows * stride_b` bytes, covering
            // `(rows - 1) * stride_b + off_b + take_b` since
            // `off_b + take_b <= stride_b`.
            unsafe {
                fill_pinned_strided(src, stride_b, off_b, take_b, rows, dst.as_mut_ptr());
            }
            let mut expect = Vec::with_capacity(rows * take_b);
            for r in 0..rows {
                for c in 0..take_b {
                    expect.push(src[r * stride_b + off_b + c]);
                }
            }
            assert_eq!(
                dst, expect,
                "rows={rows} total_cols={total_cols} col_offset={col_offset} take={take}"
            );
        }
    }
}
