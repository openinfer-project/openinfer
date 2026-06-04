use std::sync::Arc;

use cudarc::driver::CudaStream;
use kvbm_logical::SequenceHash;
use kvbm_logical::integrations::{DecodeOutcome, SchedulableSequence, ScheduleError};
use kvbm_logical::manager::BlockManager;
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;

use crate::buffer::KvBuffer;
use crate::view::KvView;

/// Engine-level KV cache: one `BlockManager` + one GPU buffer.
pub struct KvCacheManager {
    block_manager: BlockManager<()>,
    buffer: KvBuffer,
    block_size: usize,
    padding_block_id: usize,
}

impl KvCacheManager {
    pub fn new(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_blocks >= 2, "need at least 2 blocks (1 for padding)");

        let buffer = KvBuffer::new(
            stream,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
        )?;

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
        // Register so it stays alive (ImmutableBlock RAII keeps it out of the free pool).
        let _padding_immutable = block_manager.register_block(padding_complete);
        // Leak intentionally — padding lives for the lifetime of the engine.
        std::mem::forget(_padding_immutable);

        Ok(Self {
            block_manager,
            buffer,
            block_size,
            padding_block_id,
        })
    }

    pub fn block_manager(&self) -> &BlockManager<()> {
        &self.block_manager
    }

    pub fn buffer(&self) -> &KvBuffer {
        &self.buffer
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

    pub fn new_request(&self, prompt_tokens: Vec<u32>, max_output_tokens: usize) -> RequestKv {
        let seq = SchedulableSequence::new(
            prompt_tokens,
            max_output_tokens,
            self.block_size as u32,
            None,
        );
        RequestKv { seq }
    }
}

/// Per-request KV state wrapping `SchedulableSequence`.
///
/// Lifecycle: `schedule_prefill → prefill_view → forward → apply_prefill`,
/// then `schedule_decode → decode_view → forward → apply_decode` in a loop.
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
    pub fn match_and_add_prefix(&mut self, manager: &KvCacheManager) -> anyhow::Result<usize> {
        let blocks = self
            .seq
            .match_and_add_prefix(&manager.block_manager)
            .map_err(|e| anyhow::anyhow!("match_and_add_prefix: {e}"))?;
        Ok(blocks * self.seq.block_size())
    }

    // ── Scheduling (allocates blocks) ──────────────────────────────────

    pub fn schedule_prefill(
        &mut self,
        num_tokens: usize,
        manager: &KvCacheManager,
    ) -> Result<(), ScheduleError> {
        self.seq
            .schedule_prefill(num_tokens, &manager.block_manager)
    }

    pub fn schedule_decode(&mut self, manager: &KvCacheManager) -> Result<(), ScheduleError> {
        self.seq.schedule_decode(&manager.block_manager)
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

    pub fn apply_prefill(&mut self, token: u32, manager: &KvCacheManager) -> anyhow::Result<()> {
        self.seq
            .apply_prefill(Some(token), &manager.block_manager)
            .map_err(|e| anyhow::anyhow!("apply_prefill: {e}"))
    }

    pub fn apply_decode(
        &mut self,
        token: u32,
        manager: &KvCacheManager,
    ) -> anyhow::Result<DecodeOutcome> {
        self.seq
            .apply_decode(token, &manager.block_manager)
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

    /// Blocks needed to hold all possible output (prompt + max_output).
    pub fn max_blocks_needed(&self) -> usize {
        self.seq.num_blocks()
    }

    // ── Internal ───────────────────────────────────────────────────────

    fn page_indices(&self) -> Vec<i32> {
        self.seq
            .inner()
            .assignments()
            .all_block_ids()
            .map(|&id| id as i32)
            .collect()
    }
}
