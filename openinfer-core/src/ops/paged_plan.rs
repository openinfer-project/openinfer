use std::ops::Deref;

use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::kv_pool::KvDesc;
use crate::tensor::DeviceContext;

pub struct PrefillPagedPlan {
    inner: openinfer_kernels::ops::PrefillPagedPlan,
}

impl PrefillPagedPlan {
    pub fn new(
        ctx: &DeviceContext,
        desc: &KvDesc<'_>,
        start_pos: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new_with_cta_tile_q(
            ctx,
            desc,
            start_pos,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_cta_tile_q(
        ctx: &DeviceContext,
        desc: &KvDesc<'_>,
        start_pos: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        let page_indices: Vec<i32> = desc
            .page_indices()
            .iter()
            .map(|p| p.index() as i32)
            .collect();
        Ok(Self {
            inner: openinfer_kernels::ops::PrefillPagedPlan::new_with_cta_tile_q(
                ctx,
                &page_indices,
                desc.last_page_len(),
                start_pos,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                cta_tile_q_override,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_batch_with_cta_tile_q(
        ctx: &DeviceContext,
        page_indices: &[Vec<i32>],
        last_page_lens: &[usize],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        Ok(Self {
            inner: openinfer_kernels::ops::PrefillPagedPlan::new_batch_with_cta_tile_q(
                ctx,
                page_indices,
                last_page_lens,
                start_positions,
                seq_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                cta_tile_q_override,
            )?,
        })
    }

    /// Pre-allocate a worst-case-sized plan to be refilled in place by
    /// [`Self::update_batch_with_cta_tile_q`] (graph-stable buffer reuse).
    pub fn new_preallocated(
        ctx: &DeviceContext,
        max_total_tokens: usize,
        max_total_pages: usize,
        max_batch: usize,
        max_tiles: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: openinfer_kernels::ops::PrefillPagedPlan::new_preallocated(
                ctx,
                max_total_tokens,
                max_total_pages,
                max_batch,
                max_tiles,
            )?,
        })
    }

    /// Refill a pre-allocated plan in place (no allocation, pointers unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn update_batch_with_cta_tile_q(
        &mut self,
        ctx: &DeviceContext,
        page_indices: &[Vec<i32>],
        last_page_lens: &[usize],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<()> {
        self.inner.update_batch_with_cta_tile_q(
            ctx,
            page_indices,
            last_page_lens,
            start_positions,
            seq_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            cta_tile_q_override,
        )
    }

    pub fn page_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.page_indices_d()
    }
    pub fn page_indptr_d(&self) -> &CudaSlice<i32> {
        self.inner.page_indptr_d()
    }
    pub fn last_page_len_d(&self) -> &CudaSlice<i32> {
        self.inner.last_page_len_d()
    }
    pub fn q_indptr_d(&self) -> &CudaSlice<i32> {
        self.inner.q_indptr_d()
    }
    pub fn request_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.request_indices_d()
    }
    pub fn qo_tile_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.qo_tile_indices_d()
    }
    pub fn kv_tile_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.kv_tile_indices_d()
    }
    pub fn kv_chunk_size_d(&self) -> &CudaSlice<i32> {
        self.inner.kv_chunk_size_d()
    }
    pub fn total_num_rows_d(&self) -> &CudaSlice<u32> {
        self.inner.total_num_rows_d()
    }
    pub fn batch_size(&self) -> i32 {
        self.inner.batch_size()
    }
    pub fn num_tiles(&self) -> i32 {
        self.inner.num_tiles()
    }
}

impl Deref for PrefillPagedPlan {
    type Target = openinfer_kernels::ops::PrefillPagedPlan;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
