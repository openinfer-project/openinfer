use kvbm_logical::SequenceHash;
use kvbm_logical::integrations::{DecodeOutcome, SchedulableSequence, ScheduleError};
use kvbm_logical::manager::BlockManager;
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;

use crate::view::KvView;

/// Logical KV block pool: a `BlockManager` plus the reserved padding block.
///
/// Owns no GPU memory — the physical layout (full-attention `KvBuffer`,
/// MLA dual ckv/kpe buffers, ...) lives with the consumer and is indexed
/// by the block IDs this pool hands out.
pub struct BlockPool {
    block_manager: BlockManager<()>,
    block_size: usize,
    padding_block_id: usize,
}

impl BlockPool {
    pub fn new(block_size: usize, num_blocks: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(num_blocks >= 2, "need at least 2 blocks (1 for padding)");

        let registry = BlockRegistry::builder().build();
        let block_manager = BlockManager::builder()
            .block_count(num_blocks)
            .block_size(block_size)
            .registry(registry)
            .duplication_policy(BlockDuplicationPolicy::Allow)
            .with_lru_backend()
            .build()
            .map_err(|e| anyhow::anyhow!("BlockManager build failed: {e}"))?;

        // Reserve block 0 as CUDA Graph padding slot.
        let padding_blocks = block_manager
            .allocate_blocks(1)
            .ok_or_else(|| anyhow::anyhow!("failed to allocate padding block"))?;
        let padding_block_id = padding_blocks[0].block_id();
        let padding_complete = padding_blocks
            .into_iter()
            .next()
            .unwrap()
            .stage(SequenceHash::default(), block_size)
            .map_err(|e| anyhow::anyhow!("padding block stage failed: {e}"))?;
        // Register so it stays alive (ImmutableBlock RAII keeps it out of the
        // free pool), then leak — padding lives for the lifetime of the engine.
        std::mem::forget(block_manager.register_block(padding_complete));

        Ok(Self {
            block_manager,
            block_size,
            padding_block_id,
        })
    }

    pub fn block_manager(&self) -> &BlockManager<()> {
        &self.block_manager
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn available_blocks(&self) -> usize {
        self.block_manager.available_blocks()
    }

    pub fn total_blocks(&self) -> usize {
        self.block_manager.total_blocks()
    }

    pub fn padding_block_id(&self) -> i32 {
        self.padding_block_id as i32
    }

    /// Maximum blocks a single request can consume (total minus padding).
    pub fn max_request_blocks(&self) -> usize {
        self.block_manager.total_blocks().saturating_sub(1)
    }

    /// `lora_name` scopes the prefix cache: blocks registered under one
    /// adapter (or the base model, `None`) never match a request running
    /// under a different adapter — the name is folded into the block-hash
    /// chain as a salt, so K/V computed with different weights can't be
    /// silently reused.
    pub fn new_request(
        &self,
        prompt_tokens: Vec<u32>,
        max_output_tokens: usize,
        lora_name: Option<&str>,
    ) -> RequestKv {
        let salt_hash = dynamo_kv_hashing::compute_salt_hash(None, lora_name)
            .expect("salt hash from lora name is infallible");
        let seq = SchedulableSequence::new(
            prompt_tokens,
            max_output_tokens,
            self.block_size as u32,
            None,
            Some(salt_hash),
        );
        RequestKv { seq }
    }
}

/// Per-request KV state wrapping `SchedulableSequence`.
///
/// Lifecycle: `schedule_prefill → prefill_view/pages → forward → apply_prefill`,
/// then `schedule_decode → decode_view/pages → forward → apply_decode` in a loop.
pub struct RequestKv {
    seq: SchedulableSequence<()>,
}

impl RequestKv {
    // ── Prefix cache ───────────────────────────────────────────────────

    /// Match the prompt's full blocks against registered blocks and skip
    /// their prefill. Returns the number of cached tokens; `kv_position()`
    /// advances by the same amount. Must be called on a fresh request,
    /// before the first `schedule_prefill`.
    ///
    /// Matching always leaves at least one prompt token uncached so the
    /// final prefill chunk can emit the first generated token.
    pub fn match_and_add_prefix(&mut self, pool: &BlockPool) -> anyhow::Result<usize> {
        let blocks = self
            .seq
            .match_and_add_prefix(&pool.block_manager)
            .map_err(|e| anyhow::anyhow!("match_and_add_prefix: {e}"))?;
        Ok(blocks * self.seq.block_size())
    }

    // ── Scheduling (allocates blocks) ──────────────────────────────────

    pub fn schedule_prefill(
        &mut self,
        num_tokens: usize,
        pool: &BlockPool,
    ) -> Result<(), ScheduleError> {
        self.seq.schedule_prefill(num_tokens, &pool.block_manager)
    }

    pub fn schedule_decode(&mut self, pool: &BlockPool) -> Result<(), ScheduleError> {
        self.seq.schedule_decode(&pool.block_manager)
    }

    // ── Views (for forward pass) ───────────────────────────────────────

    /// Build an immutable `KvView` for prefill.
    ///
    /// `prompt_len` tokens will be appended starting at `kv_position()`.
    /// The view's seq_len = kv_position + prompt_len (post-advance state
    /// that FlashInfer attention metadata expects).
    pub fn prefill_view(&self, prompt_len: usize) -> KvView {
        let target_seq_len = self.seq.kv_position() + prompt_len;
        KvView::new(self.page_indices(), target_seq_len, self.seq.block_size())
    }

    /// Build an immutable `KvView` for decode (one new token).
    pub fn decode_view(&self) -> KvView {
        let target_seq_len = self.seq.kv_position() + 1;
        KvView::new(self.page_indices(), target_seq_len, self.seq.block_size())
    }

    // ── Apply (register blocks, advance position) ──────────────────────

    pub fn apply_prefill(&mut self, token: u32, pool: &BlockPool) -> anyhow::Result<()> {
        self.seq
            .apply_prefill(Some(token), &pool.block_manager)
            .map_err(|e| anyhow::anyhow!("apply_prefill: {e}"))
    }

    pub fn apply_decode(&mut self, token: u32, pool: &BlockPool) -> anyhow::Result<DecodeOutcome> {
        self.seq
            .apply_decode(token, &pool.block_manager)
            .map_err(|e| anyhow::anyhow!("apply_decode: {e}"))
    }

    pub fn release(&mut self) -> anyhow::Result<()> {
        self.seq
            .release()
            .map_err(|e| anyhow::anyhow!("release: {e}"))
    }

    // ── Queries ────────────────────────────────────────────────────────

    /// Tokens with KV already computed.
    pub fn kv_position(&self) -> usize {
        self.seq.kv_position()
    }

    pub fn assigned_blocks(&self) -> usize {
        self.seq.assigned_blocks()
    }

    pub fn is_complete(&self) -> bool {
        self.seq.is_complete()
    }

    pub fn generated_tokens(&self) -> usize {
        self.seq.generated_tokens()
    }

    pub fn block_size(&self) -> usize {
        self.seq.block_size()
    }

    /// Physical page IDs assigned to this request, in sequence order.
    /// Includes every block the request currently holds — which can be one
    /// more than the KV tokens need (see `step_page_indices`).
    pub fn page_indices(&self) -> Vec<i32> {
        self.seq
            .inner()
            .assignments()
            .all_block_ids()
            .map(|&id| id as i32)
            .collect()
    }

    /// Page IDs covering exactly the KV tokens present after this step
    /// appends `new_tokens` (`kv_position + new_tokens`). `page_indices()`
    /// can hold one block more: kvbm's `schedule_decode` eagerly allocates
    /// the next generation block whenever this step's token fills the last
    /// slot of the current block. Page tables built from the raw list make
    /// the kernel see a longer sequence than exists — use this for any
    /// per-step page row handed to a forward pass.
    pub fn step_page_indices(&self, new_tokens: usize) -> Vec<i32> {
        assert!(new_tokens > 0, "a forward step appends at least one token");
        let kv_tokens = self.seq.kv_position() + new_tokens;
        let mut pages = self.page_indices();
        pages.truncate(kv_tokens.div_ceil(self.seq.block_size()));
        pages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// kvbm's `schedule_decode` allocates the next generation block when the
    /// appended token fills the current block (`need = pending + 1`), so the
    /// raw `page_indices()` exceeds `ceil(kv_tokens / block_size)` at every
    /// block boundary. `step_page_indices` must hand the forward pass an
    /// exact page row at every step — this deadlocked Kimi DP8 on H200 when
    /// the raw list reached the worker's exact-match page-table check.
    #[test]
    fn step_page_indices_exact_at_block_boundaries() {
        let mut raw_overshoots = 0usize;
        for prompt_len in [1usize, 15, 16, 17, 31, 32, 33, 40, 47, 48] {
            let pool = BlockPool::new(16, 256).unwrap();
            let mut kv =
                pool.new_request((0..prompt_len as u32).map(|i| 100 + i).collect(), 24, None);
            kv.schedule_prefill(prompt_len, &pool).unwrap();
            assert_eq!(
                kv.step_page_indices(prompt_len).len(),
                prompt_len.div_ceil(16),
                "prefill page row P={prompt_len}"
            );
            kv.apply_prefill(1000, &pool).unwrap();
            for step in 0..23u32 {
                kv.schedule_decode(&pool).unwrap();
                let need = (kv.kv_position() + 1).div_ceil(16);
                assert_eq!(
                    kv.step_page_indices(1).len(),
                    need,
                    "decode page row P={prompt_len} step={step}"
                );
                raw_overshoots += usize::from(kv.page_indices().len() > need);
                kv.apply_decode(2000 + step, &pool).unwrap();
            }
        }
        assert!(
            raw_overshoots > 0,
            "kvbm no longer over-allocates the generation block; \
             step_page_indices and this test can be retired"
        );
    }
}
