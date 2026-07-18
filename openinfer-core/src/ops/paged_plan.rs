use std::ops::Deref;

use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::kv_pool::KvDesc;
use crate::tensor::DeviceContext;

/// Checked dimensions for a reusable [`PrefillPagedPlan`].
///
/// `max_page_indices` counts logical page-table entries across every request,
/// not unique physical KV pages. Prefix-cached requests may reference the same
/// physical page, so the logical bound is `max_batch * pages_per_request`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillPagedPlanCapacity {
    total_tokens: usize,
    page_indices: usize,
    batch: usize,
    tiles: usize,
}

impl PrefillPagedPlanCapacity {
    pub fn for_context(
        max_total_tokens: usize,
        max_batch: usize,
        max_context_tokens: usize,
        page_size: usize,
        gqa_group_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            max_total_tokens > 0,
            "prefill plan token capacity must be positive"
        );
        anyhow::ensure!(
            max_batch > 0,
            "prefill plan batch capacity must be positive"
        );
        anyhow::ensure!(
            max_context_tokens > 0,
            "prefill plan context capacity must be positive"
        );
        anyhow::ensure!(page_size > 0, "prefill plan page size must be positive");
        anyhow::ensure!(
            gqa_group_size > 0,
            "prefill plan GQA group size must be positive"
        );

        let max_pages_per_request = max_context_tokens.div_ceil(page_size);
        let max_page_indices = max_batch
            .checked_mul(max_pages_per_request)
            .ok_or_else(|| anyhow::anyhow!("prefill plan logical page capacity overflow"))?;
        let max_tiles = max_total_tokens
            .checked_mul(gqa_group_size)
            .ok_or_else(|| anyhow::anyhow!("prefill plan tile capacity overflow"))?;
        let capacity = Self {
            total_tokens: max_total_tokens,
            page_indices: max_page_indices,
            batch: max_batch,
            tiles: max_tiles,
        };
        capacity.preallocated_bytes()?;
        Ok(capacity)
    }

    pub fn preallocated_bytes(self) -> Result<usize> {
        PrefillPagedPlan::preallocated_bytes(
            self.total_tokens,
            self.page_indices,
            self.batch,
            self.tiles,
        )
    }

    pub fn max_total_tokens(self) -> usize {
        self.total_tokens
    }

    pub fn max_page_indices(self) -> usize {
        self.page_indices
    }

    pub fn max_batch(self) -> usize {
        self.batch
    }

    pub fn max_tiles(self) -> usize {
        self.tiles
    }
}

pub struct PrefillPagedPlan {
    inner: openinfer_kernels::ops::PrefillPagedPlan,
}

impl PrefillPagedPlan {
    /// Exact device bytes reserved by [`Self::new_preallocated`].
    pub fn preallocated_bytes(
        max_total_tokens: usize,
        max_page_indices: usize,
        max_batch: usize,
        max_tiles: usize,
    ) -> Result<usize> {
        openinfer_kernels::ops::PrefillPagedPlan::preallocated_bytes(
            max_total_tokens,
            max_page_indices,
            max_batch,
            max_tiles,
        )
    }

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
        max_page_indices: usize,
        max_batch: usize,
        max_tiles: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: openinfer_kernels::ops::PrefillPagedPlan::new_preallocated(
                ctx,
                max_total_tokens,
                max_page_indices,
                max_batch,
                max_tiles,
            )?,
        })
    }

    /// Allocate from one checked capacity value so allocation and memory
    /// accounting cannot silently derive different dimensions.
    pub fn new_preallocated_for_capacity(
        ctx: &DeviceContext,
        capacity: PrefillPagedPlanCapacity,
    ) -> Result<Self> {
        Self::new_preallocated(
            ctx,
            capacity.total_tokens,
            capacity.page_indices,
            capacity.batch,
            capacity.tiles,
        )
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::PrefillPagedPlanCapacity;

    #[test]
    fn logical_page_capacity_counts_shared_prefix_for_each_request() {
        let capacity = PrefillPagedPlanCapacity::for_context(16, 3, 32, 16, 5)
            .expect("valid reusable plan capacity");
        let page_lists = [vec![7, 8], vec![7, 8], vec![7, 8]];
        let logical_pages: usize = page_lists.iter().map(Vec::len).sum();
        let physical_pages: HashSet<i32> = page_lists.into_iter().flatten().collect();

        assert_eq!(physical_pages.len(), 2);
        assert_eq!(logical_pages, 6);
        assert_eq!(capacity.max_page_indices(), logical_pages);
    }

    #[test]
    fn capacity_rejects_zero_page_size() {
        let error = PrefillPagedPlanCapacity::for_context(1, 1, 1, 0, 1)
            .expect_err("zero page size must be rejected");
        assert!(error.to_string().contains("page size"));
    }
}
