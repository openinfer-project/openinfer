//! Contract tests for the resolve/seal/retire orchestration. Pure
//! logical-pool + mock-tier tests: no GPU, no pegaflow. What is under test is
//! the store's coordination — probe/query/reserve/load/commit ordering, floor
//! yielding, lease hygiene on every decline path, pin accounting, and the
//! park-on-retire discipline.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use openinfer_kv_cache::BlockPool;
use openinfer_kv_store::CacheScope;
use openinfer_kv_store::CancelProbe;
use openinfer_kv_store::KvStore;
use openinfer_kv_store::KvStoreConfig;
use openinfer_kv_store::NeverCancelled;
use openinfer_kv_store::SaveClass;
use openinfer_kv_store::SaveCursor;
use openinfer_kv_store::testkit::MockQuery;
use openinfer_kv_store::testkit::MockTier;

const BLOCK_SIZE: usize = 16;
const RANK: usize = 0;

fn store() -> KvStore {
    KvStore::new(tokio::runtime::Handle::current(), KvStoreConfig::default())
}

fn pool(blocks: usize) -> Arc<BlockPool> {
    Arc::new(BlockPool::new(BLOCK_SIZE, blocks).expect("pool"))
}

/// `full_blocks` full blocks plus one forwarded token.
fn prompt(full_blocks: usize) -> Vec<u32> {
    (0..=(full_blocks * BLOCK_SIZE) as u32)
        .map(|i| i % 251)
        .collect()
}

/// Seed the GPU prefix cache with `full_blocks` sealed blocks of `prompt`.
fn seed_gpu_prefix(pool: &BlockPool, prompt: &[u32], full_blocks: usize) {
    let mut seed = pool.new_request(prompt[..full_blocks * BLOCK_SIZE].to_vec(), 1, None);
    seed.schedule_prefill(full_blocks * BLOCK_SIZE, pool)
        .expect("seed schedule");
    seed.apply_prefill(1, pool).expect("seed apply");
    seed.release().expect("seed release");
}

struct Cancelled;

impl CancelProbe for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..400 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for: {what}");
}

// ── resolve ────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn resolve_without_tier_returns_gpu_hit_with_hold() {
    let store = store();
    let pool = pool(64);
    let prompt = prompt(4);
    seed_gpu_prefix(&pool, &prompt, 4);
    store.register_rank(RANK, Arc::clone(&pool), None);

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &NeverCancelled)
        .await;
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_SIZE);
    assert!(prefix.has_hold());

    // The hold is what keeps the hit resident: a cold-cache flush must not
    // evict the held blocks, so the request's own match still sees them.
    pool.evict_inactive();
    let mut req = pool.new_request(prompt, 4, None);
    assert_eq!(
        req.match_and_add_prefix(&pool).expect("match"),
        4 * BLOCK_SIZE
    );
    drop(prefix);
    req.release().expect("release");
}

#[tokio::test(flavor = "multi_thread")]
async fn resolve_extends_gpu_hit_with_host_tier_load() {
    let store = store();
    let pool = pool(64);
    let prompt = prompt(8);
    seed_gpu_prefix(&pool, &prompt, 3);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hit(5)]));
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &NeverCancelled)
        .await;
    assert_eq!(prefix.hit_tokens(), 8 * BLOCK_SIZE);
    assert_eq!(tier.loads.lock().unwrap().len(), 1);
    assert_eq!(tier.loads.lock().unwrap()[0].len(), 5);
    assert_eq!(tier.released(), 0);

    // The loaded blocks were registered under the continuation hashes: the
    // request's match reuses the whole combined prefix.
    let mut req = pool.new_request(prompt, 4, None);
    assert_eq!(
        req.match_and_add_prefix(&pool).expect("match"),
        8 * BLOCK_SIZE
    );
    drop(prefix);
    req.release().expect("release");
}

#[tokio::test(flavor = "multi_thread")]
async fn resolve_requeries_through_loading_until_ready() {
    let store = store();
    let pool = pool(64);
    let prompt = prompt(4);
    let tier = Arc::new(MockTier::scripted([
        MockQuery::Loading,
        MockQuery::Loading,
        MockQuery::Hit(4),
    ]));
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &NeverCancelled)
        .await;
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_SIZE);
    assert_eq!(tier.loads.lock().unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn resolve_deadline_degrades_to_gpu_hit_alone() {
    let config = KvStoreConfig {
        requery_interval: Duration::from_millis(1),
        resolve_deadline: Duration::from_millis(20),
    };
    let store = KvStore::new(tokio::runtime::Handle::current(), config);
    let pool = pool(64);
    let prompt = prompt(6);
    seed_gpu_prefix(&pool, &prompt, 2);
    // Never leaves Loading: an unreachable deeper tier.
    let tier = Arc::new(MockTier::scripted(std::iter::repeat_n(
        MockQuery::Loading,
        1000,
    )));
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &NeverCancelled)
        .await;
    // Degraded is not a distinct state: just the smaller (GPU-only) hit.
    assert_eq!(prefix.hit_tokens(), 2 * BLOCK_SIZE);
    assert_eq!(store.stats().resolve_degraded.load(Ordering::Relaxed), 1);
    assert_eq!(tier.loads.lock().unwrap().len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn resolve_yields_to_admission_floor_and_releases_the_lease() {
    let store = store();
    // 16 usable blocks; the floor promises 14 of them to admission, so a
    // 5-block host hit must be declined.
    let pool = pool(17);
    let prompt = prompt(5);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hit(5)]));
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));
    store.set_admission_floor(RANK, 14);

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &NeverCancelled)
        .await;
    assert_eq!(prefix.hit_tokens(), 0);
    // Declined hit = released lease, not a TTL-stranded one.
    assert_eq!(tier.released(), 1);
    assert_eq!(tier.loads.lock().unwrap().len(), 0);
    assert_eq!(store.stats().resolve_degraded.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_resolve_skips_io_and_returns_none() {
    let store = store();
    let pool = pool(64);
    let prompt = prompt(4);
    seed_gpu_prefix(&pool, &prompt, 4);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hit(4)]));
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &Cancelled)
        .await;
    assert_eq!(prefix.hit_tokens(), 0);
    assert!(!prefix.has_hold());
    // Cancelled before the query: no lease was ever taken.
    assert_eq!(tier.loads.lock().unwrap().len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_after_query_releases_the_lease() {
    // The tier flips the cancel flag as a side effect of answering the query,
    // so the resolve holds a live lease exactly when it next observes
    // cancellation — the release-on-decline path, deterministically.
    struct FlagOnQuery {
        inner: Arc<MockTier>,
        flag: Arc<AtomicBool>,
    }
    impl openinfer_kv_store::HostTier for FlagOnQuery {
        fn query(
            &self,
            req_id: &str,
            hashes: Vec<Vec<u8>>,
        ) -> openinfer_kv_store::TierFuture<anyhow::Result<openinfer_kv_store::TierQuery>> {
            self.flag.store(true, Ordering::Release);
            self.inner.query(req_id, hashes)
        }
        fn load(
            &self,
            hit: openinfer_kv_store::TierHit,
            dst: Vec<i32>,
        ) -> openinfer_kv_store::TierFuture<anyhow::Result<()>> {
            self.inner.load(hit, dst)
        }
        fn release(&self, hit: openinfer_kv_store::TierHit) {
            self.inner.release(hit);
        }
        fn save(
            &self,
            ids: Vec<i32>,
            hashes: Vec<Vec<u8>>,
            keep_alive: Box<dyn std::any::Any + Send>,
        ) -> openinfer_kv_offload::SaveHandle {
            self.inner.save(ids, hashes, keep_alive)
        }
    }
    struct Flagged(Arc<AtomicBool>);
    impl CancelProbe for Flagged {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    let store = store();
    let pool = pool(64);
    let prompt = prompt(4);
    let mock = Arc::new(MockTier::scripted([MockQuery::Hit(4)]));
    let flag = Arc::new(AtomicBool::new(false));
    let tier = Arc::new(FlagOnQuery {
        inner: Arc::clone(&mock),
        flag: Arc::clone(&flag),
    });
    store.register_rank(RANK, Arc::clone(&pool), Some(tier));

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &Flagged(flag))
        .await;
    assert_eq!(prefix.hit_tokens(), 0);
    assert!(!prefix.has_hold());
    assert_eq!(mock.released(), 1);
    assert_eq!(mock.loads.lock().unwrap().len(), 0);
}

// ── seal / retire ──────────────────────────────────────────────────────

/// Run a request through a 3-block prefill so it owns sealed blocks.
fn sealed_request(pool: &BlockPool, prompt: &[u32]) -> openinfer_kv_cache::RequestKv {
    let mut kv = pool.new_request(prompt.to_vec(), 4, None);
    kv.schedule_prefill(prompt.len(), pool).expect("schedule");
    kv.apply_prefill(1, pool).expect("apply");
    kv
}

#[tokio::test(flavor = "multi_thread")]
async fn seal_pins_blocks_until_the_save_lands() {
    let store = store();
    let pool = pool(64);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::default().with_manual_saves());
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let kv = sealed_request(&pool, &prompt);
    let mut cursor = SaveCursor::new();
    store.seal(RANK, &kv, &mut cursor, SaveClass::Cacheable);

    assert_eq!(tier.saves.lock().unwrap().len(), 1);
    assert_eq!(tier.saves.lock().unwrap()[0].0.len(), 3);
    assert_eq!(store.pinned_blocks(RANK), 3);

    // Re-sealing with no new blocks is a no-op (the cursor advanced).
    store.seal(RANK, &kv, &mut cursor, SaveClass::Cacheable);
    assert_eq!(tier.saves.lock().unwrap().len(), 1);

    tier.complete_saves();
    wait_until("save pins released", || store.pinned_blocks(RANK) == 0).await;
    store.retire(RANK, kv, cursor, SaveClass::Cacheable);
}

#[tokio::test(flavor = "multi_thread")]
async fn retire_cacheable_releases_immediately_into_the_prefix_cache() {
    let store = store();
    let pool = pool(64);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::default());
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let kv = sealed_request(&pool, &prompt);
    store.retire(RANK, kv, SaveCursor::new(), SaveClass::Cacheable);
    assert_eq!(tier.saves.lock().unwrap().len(), 1);

    // The released blocks stay matchable: the whole point of retire-to-cache.
    let mut next = pool.new_request(prompt, 4, None);
    assert_eq!(
        next.match_and_add_prefix(&pool).expect("match"),
        3 * BLOCK_SIZE
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn retire_handoff_parks_the_kv_until_saves_settle() {
    let store = store();
    let pool = pool(16);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::default().with_manual_saves());
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let available_before = pool.available_blocks();
    let kv = sealed_request(&pool, &prompt);
    store.retire(RANK, kv, SaveCursor::new(), SaveClass::Handoff);

    // Parked: the save is in flight and the KV has not released its blocks.
    assert_eq!(tier.pending_save_count(), 1);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(pool.available_blocks() < available_before);
    assert_eq!(store.stats().retires_parked.load(Ordering::Relaxed), 1);

    tier.complete_saves();
    wait_until("parked KV released", || {
        // Released blocks land in the inactive prefix cache, which still
        // counts as available capacity.
        pool.available_blocks() >= available_before
    })
    .await;
    wait_until("save pins released", || store.pinned_blocks(RANK) == 0).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn seal_skips_prefix_hit_blocks() {
    let store = store();
    let pool = pool(64);
    let prompt = prompt(6);
    seed_gpu_prefix(&pool, &prompt, 3);
    let tier = Arc::new(MockTier::default());
    store.register_rank(RANK, Arc::clone(&pool), Some(tier.clone()));

    let mut kv = pool.new_request(prompt.to_vec(), 4, None);
    assert_eq!(
        kv.match_and_add_prefix(&pool).expect("match"),
        3 * BLOCK_SIZE
    );
    let remaining = prompt.len() - 3 * BLOCK_SIZE;
    kv.schedule_prefill(remaining, &pool).expect("schedule");
    kv.apply_prefill(1, &pool).expect("apply");

    let mut cursor = SaveCursor::new();
    store.seal(RANK, &kv, &mut cursor, SaveClass::Cacheable);
    // Only the blocks this request sealed itself — the 3 prefix-hit blocks
    // were stored by whoever first sealed them.
    let saves = tier.saves.lock().unwrap();
    assert_eq!(saves.len(), 1);
    assert_eq!(saves[0].0.len(), 3);
}
