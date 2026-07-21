//! Translate the engine's internal block lifecycle into the neutral
//! [`KvBlockEvent`] stream a cache-aware router consumes.
//!
//! Two sources feed one stream:
//! - **Store** comes from the seal site (rich u64 hashes straight off the token
//!   blocks), via [`Qwen3Executor::take_kv_store_events`](crate::executor).
//! - **Remove** comes from the block pool's eviction broadcast, which carries
//!   only the engine's 128-bit lineage hash — so each store also records a
//!   `lineage → sequence_hash` entry that the eviction is translated through.
//!
//! The router works in dynamo's u64 hash space; the lineage hash never leaves
//! this module.

use std::collections::HashMap;

use log::warn;
use openinfer_core::engine::KvBlockEvent;
use openinfer_core::engine::KvStoredBlock;
use openinfer_kv_cache::KvCacheEvent;
use openinfer_kv_cache::RegisteredBlock;
use tokio::sync::broadcast;
use tokio::sync::mpsc;

pub(super) struct KvEventProducer {
    /// Neutral feed to the engine handle's KV-event consumer (the router pump).
    /// A dropped receiver just means no one is listening yet — sends are
    /// best-effort.
    tx: mpsc::UnboundedSender<KvBlockEvent>,
    /// Pool eviction/registration broadcast. `Create` is ignored (richer store
    /// events come from the seal site); `Remove` is translated to the router's
    /// `sequence_hash`.
    removes: broadcast::Receiver<KvCacheEvent>,
    /// lineage hash → router `sequence_hash`, written on store and read on
    /// remove. Bounded by the number of currently-cached blocks.
    lineage_to_seq: HashMap<u128, u64>,
}

impl KvEventProducer {
    pub(super) fn new(
        tx: mpsc::UnboundedSender<KvBlockEvent>,
        removes: broadcast::Receiver<KvCacheEvent>,
    ) -> Self {
        Self {
            tx,
            removes,
            lineage_to_seq: HashMap::new(),
        }
    }

    /// Publish a store event per run of blocks registered since the last call.
    /// A run is a contiguous, parent-chained sequence; the router walks it as a
    /// prefix path.
    pub(super) fn emit_stores(&mut self, runs: Vec<Vec<RegisteredBlock>>) {
        for run in runs {
            let Some(first) = run.first() else { continue };
            let parent_hash = first.parent_sequence_hash;
            let mut blocks = Vec::with_capacity(run.len());
            for b in &run {
                self.lineage_to_seq.insert(b.plh, b.sequence_hash);
                blocks.push(KvStoredBlock {
                    sequence_hash: b.sequence_hash,
                    tokens_hash: b.tokens_hash,
                });
            }
            let _ = self.tx.send(KvBlockEvent::Stored {
                parent_hash,
                blocks,
            });
        }
    }

    /// Drain the pool's eviction broadcast and publish a remove for each block
    /// we previously stored. Non-blocking; call once per scheduler step.
    pub(super) fn drain_removes(&mut self) {
        loop {
            match self.removes.try_recv() {
                Ok(KvCacheEvent::Remove(lineage)) => {
                    if let Some(sequence_hash) = self.lineage_to_seq.remove(&lineage.as_u128()) {
                        let _ = self.tx.send(KvBlockEvent::Removed { sequence_hash });
                    }
                }
                // Stores are sourced from the seal site, which carries both
                // hashes; the bare lineage hash here would be lossy.
                Ok(KvCacheEvent::Create(_)) => {}
                Err(
                    broadcast::error::TryRecvError::Empty | broadcast::error::TryRecvError::Closed,
                ) => {
                    break;
                }
                Err(broadcast::error::TryRecvError::Lagged(dropped)) => {
                    // A dropped eviction leaves a stale entry in the router's
                    // prefix tree (it may route a prefix this worker already
                    // evicted — a re-prefill, never wrong output) and orphans the
                    // block's `lineage_to_seq` entry until that lineage re-evicts.
                    // The 64Ki channel makes this rare, not impossible; it stays
                    // bounded by block churn either way.
                    warn!(
                        "KV event feed lagged: {dropped} eviction(s) dropped; \
                         router prefix tree may carry stale entries"
                    );
                }
            }
        }
    }
}
