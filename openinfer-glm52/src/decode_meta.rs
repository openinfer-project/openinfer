//! GLM5.2 fixed-bucket decode metadata owner.
//!
//! vLLM contract fields:
//! - `seq_lens_2d`: non-MTP decode context lengths as `[B, 1]`.
//! - `block_table`: dense row-major `[B, max_blocks_per_row]` physical page table.
//! - `slot_mapping`: one cache-write slot per decode row, matching vLLM's
//!   flattened `block_id * block_size + offset` convention.
//! - `decode_lens`: fixed one-token decode rows.
//! - `schedule_metadata`: backend scheduler metadata for paged MQA/indexer logits.
//!
//! OpenInfer-owned fields:
//! - `positions`: absolute decode positions used by RoPE/KPE and cache append.
//! - `active_rows_d` / `active_mask`: fixed-bucket activity metadata.
//! - geometry and padding-page invariants tying FlashMLA sparse attention,
//!   indexer cache insert, indexer logits, and sparse top-k to the same page view.

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;

use crate::{
    config::{GLM52_DECODE_BATCH_CAP, GLM52_DECODE_DEVICE_SMS},
    weights::Glm52RankGpuContext,
};

/// One active decode row supplied by the scheduler.
///
/// `seq_len` is the effective KV length after this decode token is appended.
/// Therefore `position == seq_len - 1`, and `block_ids` must cover exactly
/// `ceil(seq_len / block_size)` physical cache blocks. Padding rows are owned
/// by [`Glm52DecodeBatchGeometry::padding_block_id`], not represented here.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Glm52DecodePageRow<'a> {
    pub(crate) seq_len: usize,
    pub(crate) block_ids: &'a [i32],
}

impl<'a> Glm52DecodePageRow<'a> {
    pub(crate) fn new(seq_len: usize, block_ids: &'a [i32]) -> Self {
        Self { seq_len, block_ids }
    }
}

/// Fixed GLM5.2 decode-bucket geometry.
///
/// The first decode-forward bucket is intentionally fixed at `B=128`. Active
/// rows may be fewer, but all device buffers keep the same base addresses so
/// later CUDA Graph captures can replay against updated contents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DecodeBatchGeometry {
    pub(crate) batch_capacity: usize,
    pub(crate) block_size: usize,
    pub(crate) max_model_len: usize,
    pub(crate) max_blocks_per_row: usize,
    pub(crate) pool_blocks: usize,
    pub(crate) padding_block_id: i32,
    pub(crate) num_sms: usize,
}

impl Glm52DecodeBatchGeometry {
    pub(crate) fn fixed_h200(
        block_size: usize,
        max_model_len: usize,
        pool_blocks: usize,
        padding_block_id: i32,
    ) -> Result<Self> {
        Self::new(
            GLM52_DECODE_BATCH_CAP,
            block_size,
            max_model_len,
            pool_blocks,
            padding_block_id,
            GLM52_DECODE_DEVICE_SMS,
        )
    }

    pub(crate) fn new(
        batch_capacity: usize,
        block_size: usize,
        max_model_len: usize,
        pool_blocks: usize,
        padding_block_id: i32,
        num_sms: usize,
    ) -> Result<Self> {
        ensure!(
            batch_capacity == GLM52_DECODE_BATCH_CAP,
            "GLM5.2 decode metadata is fixed at B={GLM52_DECODE_BATCH_CAP}, got {batch_capacity}"
        );
        ensure!(block_size > 0, "GLM5.2 decode block_size must be positive");
        ensure!(
            max_model_len > 0,
            "GLM5.2 decode max_model_len must be positive"
        );
        ensure!(pool_blocks > 0, "GLM5.2 decode metadata needs a cache pool");
        ensure!(
            padding_block_id >= 0,
            "GLM5.2 decode padding block must be non-negative, got {padding_block_id}"
        );
        ensure!(
            (padding_block_id as usize) < pool_blocks,
            "GLM5.2 decode padding block {padding_block_id} exceeds pool blocks {pool_blocks}"
        );
        ensure!(num_sms > 0, "GLM5.2 decode num_sms must be positive");
        let max_blocks_per_row = max_model_len.div_ceil(block_size);
        ensure!(
            max_blocks_per_row > 0,
            "GLM5.2 decode block table must have at least one block column"
        );
        Ok(Self {
            batch_capacity,
            block_size,
            max_model_len,
            max_blocks_per_row,
            pool_blocks,
            padding_block_id,
            num_sms,
        })
    }

    fn schedule_metadata_len(self) -> usize {
        (self.num_sms + 1) * 2
    }

    fn block_table_len(self) -> Result<usize> {
        self.batch_capacity
            .checked_mul(self.max_blocks_per_row)
            .ok_or_else(|| anyhow::anyhow!("GLM5.2 decode block table length overflow"))
    }

    fn validate_row(self, row_idx: usize, row: Glm52DecodePageRow<'_>) -> Result<()> {
        ensure!(
            row.seq_len > 0,
            "GLM5.2 decode row {row_idx} has zero KV length"
        );
        ensure!(
            row.seq_len <= self.max_model_len,
            "GLM5.2 decode row {row_idx} seq_len {} exceeds max_model_len {}",
            row.seq_len,
            self.max_model_len
        );
        let needed = row.seq_len.div_ceil(self.block_size);
        ensure!(
            row.block_ids.len() == needed,
            "GLM5.2 decode row {row_idx} has {} blocks for {} tokens, needs {needed}",
            row.block_ids.len(),
            row.seq_len
        );
        for (block_idx, &block_id) in row.block_ids.iter().enumerate() {
            ensure!(
                block_id >= 0,
                "GLM5.2 decode row {row_idx} block {block_idx} is negative: {block_id}"
            );
            ensure!(
                (block_id as usize) < self.pool_blocks,
                "GLM5.2 decode row {row_idx} block {block_id} exceeds pool blocks {}",
                self.pool_blocks
            );
        }
        Ok(())
    }

    fn slot_mapping(self, seq_len: usize, block_ids: &[i32]) -> Result<i64> {
        let position = seq_len - 1;
        let block = block_ids[position / self.block_size];
        let offset = position % self.block_size;
        Ok(i64::from(block) * self.block_size as i64 + offset as i64)
    }
}

/// CPU staging for one fixed-bucket metadata upload.
///
/// This is one batched page-table build and one set of device uploads. It does
/// not own per-request execution and must not grow into a loop of bs=1 forwards.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DecodeBatchHostMetadata {
    pub(crate) active_rows: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) total_active_blocks: usize,
    pub(crate) seq_lens_2d: Vec<i32>,
    pub(crate) block_table: Vec<i32>,
    pub(crate) slot_mapping: Vec<i64>,
    pub(crate) positions: Vec<i32>,
    pub(crate) decode_lens: Vec<i32>,
    pub(crate) active_rows_scalar: Vec<i32>,
    pub(crate) active_mask: Vec<i32>,
}

impl Glm52DecodeBatchHostMetadata {
    pub(crate) fn build(
        geometry: Glm52DecodeBatchGeometry,
        rows: &[Glm52DecodePageRow<'_>],
    ) -> Result<Self> {
        ensure!(
            rows.len() <= geometry.batch_capacity,
            "GLM5.2 decode active rows {} exceed fixed bucket {}",
            rows.len(),
            geometry.batch_capacity
        );

        let mut seq_lens_2d = vec![1i32; geometry.batch_capacity];
        let mut block_table = vec![geometry.padding_block_id; geometry.block_table_len()?];
        let mut slot_mapping = vec![
            i64::from(geometry.padding_block_id)
                * geometry.block_size as i64;
            geometry.batch_capacity
        ];
        let mut positions = vec![0i32; geometry.batch_capacity];
        let decode_lens = vec![1i32; geometry.batch_capacity];
        let mut active_mask = vec![0i32; geometry.batch_capacity];

        let mut max_seq_len = 1usize;
        let mut total_active_blocks = 0usize;
        for (row_idx, &row) in rows.iter().enumerate() {
            geometry.validate_row(row_idx, row)?;
            let row_start = row_idx * geometry.max_blocks_per_row;
            let row_end = row_start + row.block_ids.len();
            block_table[row_start..row_end].copy_from_slice(row.block_ids);
            seq_lens_2d[row_idx] = usize_to_i32(row.seq_len, "seq_len")?;
            slot_mapping[row_idx] = geometry.slot_mapping(row.seq_len, row.block_ids)?;
            positions[row_idx] = usize_to_i32(row.seq_len - 1, "position")?;
            active_mask[row_idx] = 1;
            max_seq_len = max_seq_len.max(row.seq_len);
            total_active_blocks += row.block_ids.len();
        }

        Ok(Self {
            active_rows: rows.len(),
            max_seq_len,
            total_active_blocks,
            seq_lens_2d,
            block_table,
            slot_mapping,
            positions,
            decode_lens,
            active_rows_scalar: vec![usize_to_i32(rows.len(), "active_rows")?],
            active_mask,
        })
    }
}

/// Graph-stable device owner for one GLM5.2 decode bucket.
pub(crate) struct Glm52DecodeBatchMetadata {
    pub(crate) geometry: Glm52DecodeBatchGeometry,
    pub(crate) active_rows: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) total_active_blocks: usize,
    pub(crate) seq_lens_2d: CudaSlice<i32>,
    pub(crate) block_table: CudaSlice<i32>,
    pub(crate) slot_mapping: CudaSlice<i64>,
    pub(crate) positions: CudaSlice<i32>,
    pub(crate) decode_lens: CudaSlice<i32>,
    pub(crate) active_rows_d: CudaSlice<i32>,
    pub(crate) active_mask: CudaSlice<i32>,
    pub(crate) schedule_metadata: CudaSlice<i32>,
}

impl Glm52DecodeBatchMetadata {
    pub(crate) fn new(
        ctx: &Glm52RankGpuContext,
        geometry: Glm52DecodeBatchGeometry,
    ) -> Result<Self> {
        let host = Glm52DecodeBatchHostMetadata::build(geometry, &[])?;
        let schedule_metadata = vec![0i32; geometry.schedule_metadata_len()];
        let stream = ctx.stream();
        Ok(Self {
            geometry,
            active_rows: host.active_rows,
            max_seq_len: host.max_seq_len,
            total_active_blocks: host.total_active_blocks,
            seq_lens_2d: stream.clone_htod(&host.seq_lens_2d)?,
            block_table: stream.clone_htod(&host.block_table)?,
            slot_mapping: stream.clone_htod(&host.slot_mapping)?,
            positions: stream.clone_htod(&host.positions)?,
            decode_lens: stream.clone_htod(&host.decode_lens)?,
            active_rows_d: stream.clone_htod(&host.active_rows_scalar)?,
            active_mask: stream.clone_htod(&host.active_mask)?,
            schedule_metadata: stream.clone_htod(&schedule_metadata)?,
        })
    }

    pub(crate) fn sync_from_rows(
        &mut self,
        ctx: &Glm52RankGpuContext,
        rows: &[Glm52DecodePageRow<'_>],
    ) -> Result<()> {
        let host = Glm52DecodeBatchHostMetadata::build(self.geometry, rows)?;
        let stream = ctx.stream();
        stream.memcpy_htod(&host.seq_lens_2d, &mut self.seq_lens_2d)?;
        stream.memcpy_htod(&host.block_table, &mut self.block_table)?;
        stream.memcpy_htod(&host.slot_mapping, &mut self.slot_mapping)?;
        stream.memcpy_htod(&host.positions, &mut self.positions)?;
        stream.memcpy_htod(&host.decode_lens, &mut self.decode_lens)?;
        stream.memcpy_htod(&host.active_rows_scalar, &mut self.active_rows_d)?;
        stream.memcpy_htod(&host.active_mask, &mut self.active_mask)?;
        self.active_rows = host.active_rows;
        self.max_seq_len = host.max_seq_len;
        self.total_active_blocks = host.total_active_blocks;
        Ok(())
    }

    pub(crate) fn upload_schedule_metadata(
        &mut self,
        ctx: &Glm52RankGpuContext,
        schedule_metadata: &[i32],
    ) -> Result<()> {
        ensure!(
            schedule_metadata.len() == self.geometry.schedule_metadata_len(),
            "GLM5.2 schedule_metadata length {} does not match [{}, 2]",
            schedule_metadata.len(),
            self.geometry.num_sms + 1
        );
        ctx.stream()
            .memcpy_htod(schedule_metadata, &mut self.schedule_metadata)?;
        Ok(())
    }
}

fn usize_to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| anyhow::anyhow!("GLM5.2 decode {name} {value} exceeds i32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_geometry() -> Glm52DecodeBatchGeometry {
        Glm52DecodeBatchGeometry::new(GLM52_DECODE_BATCH_CAP, 64, 1024, 100, 99, 132).unwrap()
    }

    #[test]
    fn host_metadata_builds_dense_vllm_contract_and_padding() {
        let geometry = test_geometry();
        let row0_blocks = [3, 4];
        let row1_blocks = [7];
        let rows = [
            Glm52DecodePageRow::new(65, &row0_blocks),
            Glm52DecodePageRow::new(64, &row1_blocks),
        ];

        let host = Glm52DecodeBatchHostMetadata::build(geometry, &rows).unwrap();

        assert_eq!(host.active_rows, 2);
        assert_eq!(host.max_seq_len, 65);
        assert_eq!(host.total_active_blocks, 3);
        assert_eq!(host.seq_lens_2d[0], 65);
        assert_eq!(host.seq_lens_2d[1], 64);
        assert_eq!(host.seq_lens_2d[2], 1);
        assert_eq!(host.positions[0], 64);
        assert_eq!(host.positions[1], 63);
        assert_eq!(host.slot_mapping[0], 4 * 64);
        assert_eq!(host.slot_mapping[1], 7 * 64 + 63);
        assert_eq!(host.slot_mapping[2], 99 * 64);
        assert_eq!(host.active_rows_scalar, vec![2]);
        assert_eq!(&host.active_mask[0..4], &[1, 1, 0, 0]);

        let stride = geometry.max_blocks_per_row;
        assert_eq!(&host.block_table[0..3], &[3, 4, 99]);
        assert_eq!(&host.block_table[stride..stride + 2], &[7, 99]);
        assert_eq!(host.block_table[2 * stride], 99);
        assert!(host.decode_lens.iter().all(|&len| len == 1));
    }

    #[test]
    fn host_metadata_rejects_page_accounting_drift() {
        let geometry = test_geometry();
        let one_block = [3];
        let rows = [Glm52DecodePageRow::new(65, &one_block)];

        let err = Glm52DecodeBatchHostMetadata::build(geometry, &rows).unwrap_err();
        assert!(
            err.to_string().contains("needs 2"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn geometry_rejects_non_fixed_batch_and_bad_padding() {
        assert!(Glm52DecodeBatchGeometry::new(64, 64, 1024, 100, 99, 132).is_err());
        assert!(
            Glm52DecodeBatchGeometry::new(GLM52_DECODE_BATCH_CAP, 64, 1024, 100, 100, 132).is_err()
        );
    }
}
