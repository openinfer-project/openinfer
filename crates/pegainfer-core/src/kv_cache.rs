//! KV Cache — contiguous buffers for fused attention.

use anyhow::Result;

use crate::tensor::{DeviceContext, DeviceVec};

/// KV Cache — contiguous buffers for fused attention.
pub struct KVCache {
    // [layer] -> contiguous buffer (num_kv_heads * max_seq * head_dim)
    k_cache: Vec<DeviceVec>,
    v_cache: Vec<DeviceVec>,
    seq_len: usize,
    head_dim: usize,
    num_layers: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, num_kv_heads: usize) -> Self {
        Self {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            seq_len: 0,
            head_dim: 0,
            num_layers,
            num_kv_heads,
            max_seq_len: 20_000,
        }
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Get mutable references to K/V cache for a layer
    pub fn get_cache_mut(
        &mut self,
        ctx: &DeviceContext,
        layer: usize,
    ) -> Result<(&mut DeviceVec, &mut DeviceVec)> {
        // Initialize on first access
        if self.k_cache.is_empty() {
            for _ in 0..self.num_layers {
                // Allocate max size upfront
                let cache_size = self.num_kv_heads * self.max_seq_len * self.head_dim;
                self.k_cache.push(DeviceVec::zeros(ctx, cache_size)?);
                self.v_cache.push(DeviceVec::zeros(ctx, cache_size)?);
            }
        }
        Ok((&mut self.k_cache[layer], &mut self.v_cache[layer]))
    }

    pub fn init_if_needed(&mut self, ctx: &DeviceContext, head_dim: usize) -> Result<()> {
        if self.head_dim == 0 {
            self.head_dim = head_dim;
            for _ in 0..self.num_layers {
                let cache_size = self.num_kv_heads * self.max_seq_len * head_dim;
                self.k_cache.push(DeviceVec::zeros(ctx, cache_size)?);
                self.v_cache.push(DeviceVec::zeros(ctx, cache_size)?);
            }
        }
        Ok(())
    }

    pub fn advance_seq_len(&mut self, count: usize) {
        self.seq_len += count;
    }
}
