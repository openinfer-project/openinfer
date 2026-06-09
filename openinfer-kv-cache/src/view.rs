use cudarc::driver::CudaSlice;
use half::bf16;

use crate::buffer::KvBuffer;
use crate::layout::KvLayout;

/// Lightweight, non-owning view of a request's KV state.
///
/// Built from a `SchedulableSequence`'s assigned block IDs before each
/// forward pass. Block lifecycle is managed externally by `BlockManager`.
#[derive(Clone)]
pub struct KvView {
    page_indices: Vec<i32>,
    seq_len: usize,
    page_size: usize,
}

impl KvView {
    pub fn new(page_indices: Vec<i32>, seq_len: usize, page_size: usize) -> Self {
        Self {
            page_indices,
            seq_len,
            page_size,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn num_pages(&self) -> usize {
        self.page_indices.len()
    }

    pub fn last_page_len(&self) -> usize {
        if self.seq_len == 0 {
            0
        } else {
            let rem = self.seq_len % self.page_size;
            if rem == 0 { self.page_size } else { rem }
        }
    }

    pub fn page_indices(&self) -> &[i32] {
        &self.page_indices
    }

    /// Verify pre-allocated capacity covers the requested token count.
    pub fn ensure_capacity(&self, token_count: usize) -> anyhow::Result<()> {
        let needed = token_count.div_ceil(self.page_size);
        anyhow::ensure!(
            needed <= self.page_indices.len(),
            "KvView: need {needed} pages but only {} allocated",
            self.page_indices.len(),
        );
        Ok(())
    }

    pub fn advance(&mut self, count: usize) {
        self.seq_len += count;
    }

    pub fn desc<'a>(&'a self, buffer: &'a KvBuffer) -> KvViewDesc<'a> {
        KvViewDesc {
            layout: *buffer.layout(),
            buffer: buffer.buffer(),
            pages: &self.page_indices,
            seq_len: self.seq_len,
            last_page_len: self.last_page_len(),
        }
    }
}

/// Kernel-facing metadata bundle.
pub struct KvViewDesc<'a> {
    layout: KvLayout,
    buffer: &'a CudaSlice<bf16>,
    pages: &'a [i32],
    seq_len: usize,
    last_page_len: usize,
}

impl KvViewDesc<'_> {
    pub fn layout(&self) -> &KvLayout {
        &self.layout
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn last_page_len(&self) -> usize {
        self.last_page_len
    }

    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    pub fn page_indices(&self) -> &[i32] {
        self.pages
    }

    pub fn buffer(&self) -> &CudaSlice<bf16> {
        self.buffer
    }
}
