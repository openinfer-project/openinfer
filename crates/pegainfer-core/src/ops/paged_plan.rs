use std::ops::Deref;

use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::kv_pool::KvDesc;
use crate::tensor::DeviceContext;

pub struct PrefillPagedPlan {
    inner: pegainfer_kernels::ops::PrefillPagedPlan,
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
        let page_indices: Vec<i32> = desc
            .page_indices()
            .iter()
            .map(|p| p.index() as i32)
            .collect();
        Ok(Self {
            inner: pegainfer_kernels::ops::PrefillPagedPlan::new(
                ctx,
                &page_indices,
                desc.last_page_len(),
                start_pos,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
            )?,
        })
    }

    pub fn new_batch(
        ctx: &DeviceContext,
        descs: &[KvDesc<'_>],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let page_indices: Vec<Vec<i32>> = descs
            .iter()
            .map(|desc| {
                desc.page_indices()
                    .iter()
                    .map(|p| p.index() as i32)
                    .collect()
            })
            .collect();
        let last_page_lens: Vec<usize> = descs.iter().map(KvDesc::last_page_len).collect();
        Ok(Self {
            inner: pegainfer_kernels::ops::PrefillPagedPlan::new_batch(
                ctx,
                &page_indices,
                &last_page_lens,
                start_positions,
                seq_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
            )?,
        })
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
    pub fn batch_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.batch_indices_d()
    }
    pub fn positions_d(&self) -> &CudaSlice<i32> {
        self.inner.positions_d()
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
    type Target = pegainfer_kernels::ops::PrefillPagedPlan;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
