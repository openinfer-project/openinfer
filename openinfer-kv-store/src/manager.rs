use std::sync::Arc;

use cudarc::driver::CudaStream;
use kvbm_logical::events::KvCacheEvent;
use tokio::sync::broadcast;

use crate::buffer::KvBuffer;
use crate::pool::BlockPool;

/// Engine-level KV cache for full-attention models: one logical
/// [`BlockPool`] + one GPU [`KvBuffer`] with matching geometry.
///
/// MLA models (kimi-k2) consume [`BlockPool`] directly and own their
/// physical buffers — this facade is the full-attention pairing only. The
/// pool sits in an `Arc` so the same logical pool can be shared with a
/// [`crate::KvStore`] rank: the store's resolve path probes and pins blocks
/// in exactly the pool the scheduler allocates from.
pub struct KvCacheManager {
    pool: Arc<BlockPool>,
    buffer: KvBuffer,
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
        let buffer = KvBuffer::new(
            stream,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
        )?;
        let pool = Arc::new(BlockPool::new(block_size, num_blocks));
        Ok(Self { pool, buffer })
    }

    /// Like [`new`](Self::new) but the pool emits KV block events; returns the
    /// receiver to drain. See [`BlockPool::with_events`].
    pub fn new_with_events(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<(Self, broadcast::Receiver<KvCacheEvent>)> {
        let buffer = KvBuffer::new(
            stream,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
        )?;
        let (pool, events) = BlockPool::with_events(block_size, num_blocks);
        Ok((
            Self {
                pool: Arc::new(pool),
                buffer,
            },
            events,
        ))
    }

    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    /// The shared handle a [`crate::KvStoreBuilder`] rank consumes.
    pub fn pool_arc(&self) -> Arc<BlockPool> {
        Arc::clone(&self.pool)
    }

    pub fn buffer(&self) -> &KvBuffer {
        &self.buffer
    }
}
