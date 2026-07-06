//! Host-tier KV offload glue: the coordinator's two touch points with the
//! shared pegaflow pool. Restore runs at admission (a step boundary — every
//! rank is joined, so blocking on the load is safe and the loaded pages race
//! nothing); save runs on request release, fire-and-forget, with block
//! guards pinning the pages until the D2H copy lands.
//!
//! Both legs are cache maintenance, never a correctness dependency: every
//! failure degrades to a full prefill (or a forfeited future hit) with a
//! warn, in contrast to the pool-invariant breaks around them that fail the
//! step. The launch-time contract (`Glm52LaunchOptions` validation) already
//! guarantees offload implies the prefix cache is on.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use openinfer_kv_cache::{BlockPool, KvBlockGuard, PrefixProbe, RequestKv};
use openinfer_kv_offload::{OffloadEngine, QueryOutcome};

/// Distinguishes concurrent queries inside pegaflow's bookkeeping; nothing
/// joins on it, so a process-local counter is enough.
static QUERY_SEQ: AtomicU64 = AtomicU64::new(0);

/// One rank's offload engine plus the pages its in-flight release saves
/// still pin. The save guards hold released blocks in the active pool
/// (unallocatable, un-evictable) until the D2H copy lands — pages the
/// admission full-lifetime math would otherwise promise to a new request,
/// turning a slow copy into a mid-request allocation failure and a
/// `fail_step` engine teardown. Admission subtracts [`Self::pinned_blocks`]
/// from the rank's usable count instead, degrading to "admit a few steps
/// later" — the same honor-or-reject posture as the rest of the scheduler.
pub(super) struct RankOffload {
    pub(super) engine: OffloadEngine,
    pinned: Arc<AtomicUsize>,
}

/// Keep-alive payload for one release save: the block guards plus the
/// pinned-page accounting. Dropped by the offload engine exactly when the
/// D2H copy lands (or on any early-error path), releasing both the pins and
/// the count together.
struct SavePin {
    _guards: Vec<KvBlockGuard>,
    pinned: Arc<AtomicUsize>,
    blocks: usize,
}

impl Drop for SavePin {
    fn drop(&mut self) {
        self.pinned.fetch_sub(self.blocks, Ordering::Release);
    }
}

impl RankOffload {
    pub(super) fn new(engine: OffloadEngine) -> Self {
        Self {
            engine,
            pinned: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Pool pages currently pinned by in-flight release saves.
    pub(super) fn pinned_blocks(&self) -> usize {
        self.pinned.load(Ordering::Acquire)
    }

    /// Send the request's freshly-sealed blocks to the host tier before its
    /// pool pages release. Skips the prefix-matched head — those blocks were
    /// restored from the host tier or saved when their producing request
    /// released, so they are already resident there. Fire-and-forget: the
    /// [`SavePin`] keeps the pages pinned (and counted) until the D2H copy
    /// lands, and the last step that wrote them has already joined, so the
    /// bytes are final.
    pub(super) fn save_sealed_on_release(&self, kv: &RequestKv) {
        let sealed = kv.assigned_block_hashes();
        let matched = kv.prefix_matched_blocks();
        if sealed.len() <= matched {
            return;
        }
        let fresh = &sealed[matched..];
        let block_ids: Vec<i32> = fresh.iter().map(|(id, _)| *id).collect();
        let block_hashes: Vec<Vec<u8>> = fresh.iter().map(|(_, hash)| hash.to_vec()).collect();
        let mut guards = kv.assigned_block_guards();
        let guards = guards.split_off(matched);
        self.pinned.fetch_add(guards.len(), Ordering::Release);
        let pin = SavePin {
            blocks: guards.len(),
            _guards: guards,
            pinned: Arc::clone(&self.pinned),
        };
        self.engine.save(&block_ids, &block_hashes, pin);
    }
}

/// Restore the prompt-prefix blocks the GPU cache no longer holds from the
/// host tier: probe → query → load into reserved pool pages → commit as
/// matchable prefix. Blocks on the load — admission is a step boundary, and
/// the request's first prefill chunk must not read half-restored pages.
///
/// Returns the probe, which holds the GPU-hit and freshly-committed blocks
/// alive; the caller keeps it across `match_and_add_prefix` so the restored
/// prefix cannot be evicted before it is re-matched.
pub(super) fn restore_host_prefix(
    engine: &OffloadEngine,
    pool: &BlockPool,
    prompt_tokens: &[u32],
) -> PrefixProbe {
    let mut probe = pool.probe_prefix(prompt_tokens.to_vec(), None);
    let hashes = probe.cpu_query_hashes();
    if hashes.is_empty() {
        return probe;
    }
    let req_key = format!("glm52-admit-{}", QUERY_SEQ.fetch_add(1, Ordering::Relaxed));
    let hit = match engine.query(&req_key, &hashes) {
        Ok(QueryOutcome::Ready(hit)) => hit,
        Ok(QueryOutcome::Loading) => {
            // Host-memory-only setup: pegaflow has no deeper tier to fetch
            // from, so an async outcome means a config drift worth seeing.
            log::warn!("GLM5.2 host-tier query went async in a host-only setup; skipping restore");
            return probe;
        }
        Err(err) => {
            log::warn!("GLM5.2 host-tier query failed (prefill from scratch): {err}");
            return probe;
        }
    };
    let Some(lease) = hit.lease else {
        return probe;
    };
    let Some(reservation) = pool.reserve_loaded_blocks(hit.num_blocks) else {
        // Block pressure: the pool cannot hold the restored prefix right
        // now. Prefill recomputes it — correct, just colder.
        engine.release_query_lease(lease);
        return probe;
    };
    let page_ids = reservation.page_ids();
    let restored = reservation.len();
    match engine.load(lease, page_ids) {
        Ok(handle) => match handle.wait() {
            Ok(()) => {
                pool.commit_loaded_blocks(&mut probe, reservation);
                // The only signal separating a host-tier restore from a plain
                // GPU prefix hit — the parity/eviction gates key on it.
                log::info!("GLM5.2 host-tier restore: {restored} blocks committed");
            }
            Err(err) => {
                log::warn!("GLM5.2 host-tier load failed (prefill from scratch): {err}");
            }
        },
        Err(err) => {
            log::warn!("GLM5.2 host-tier load submit failed (prefill from scratch): {err}");
        }
    }
    probe
}
