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
use openinfer_kv_store::KvStoreBuilder;
use openinfer_kv_store::KvStoreConfig;
use openinfer_kv_store::NeverCancelled;
use openinfer_kv_store::SaveClass;
use openinfer_kv_store::SaveCursor;
use openinfer_kv_store::testkit::MockQuery;
use openinfer_kv_store::testkit::MockTier;

const BLOCK_SIZE: usize = 16;
const RANK: usize = 0;

fn builder() -> KvStoreBuilder {
    KvStoreBuilder::new(tokio::runtime::Handle::current(), KvStoreConfig::default())
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
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(4);
    seed_gpu_prefix(&pool, &prompt, 4);
    let store = b.rank(RANK, Arc::clone(&pool), None).build();

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
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(8);
    seed_gpu_prefix(&pool, &prompt, 3);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hit(5)]));
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(4);
    let tier = Arc::new(MockTier::scripted([
        MockQuery::Loading,
        MockQuery::Loading,
        MockQuery::Hit(4),
    ]));
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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
        ..KvStoreConfig::default()
    };
    let b = KvStoreBuilder::new(tokio::runtime::Handle::current(), config);
    let pool = pool(64);
    let prompt = prompt(6);
    seed_gpu_prefix(&pool, &prompt, 2);
    // Never leaves Loading: an unreachable deeper tier.
    let tier = Arc::new(MockTier::scripted(std::iter::repeat_n(
        MockQuery::Loading,
        1000,
    )));
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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
    let b = builder();
    // 16 usable blocks; the floor promises 14 of them to admission, so a
    // 5-block host hit must be declined.
    let pool = pool(17);
    let prompt = prompt(5);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hit(5)]));
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();
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
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(4);
    seed_gpu_prefix(&pool, &prompt, 4);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hit(4)]));
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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

    let b = builder();
    let pool = pool(64);
    let prompt = prompt(4);
    let mock = Arc::new(MockTier::scripted([MockQuery::Hit(4)]));
    let flag = Arc::new(AtomicBool::new(false));
    let tier = Arc::new(FlagOnQuery {
        inner: Arc::clone(&mock),
        flag: Arc::clone(&flag),
    });
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier)).build();

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &Flagged(flag))
        .await;
    assert_eq!(prefix.hit_tokens(), 0);
    assert!(!prefix.has_hold());
    assert_eq!(mock.released(), 1);
    assert_eq!(mock.loads.lock().unwrap().len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn hung_query_degrades_by_deadline() {
    // A query future that never settles (hung storage worker) must degrade
    // exactly like a slow one — the request cannot strand outside the
    // scheduler (Codex review, PR #825).
    let config = KvStoreConfig {
        resolve_deadline: Duration::from_millis(20),
        ..KvStoreConfig::default()
    };
    let b = KvStoreBuilder::new(tokio::runtime::Handle::current(), config);
    let pool = pool(64);
    let prompt = prompt(5);
    seed_gpu_prefix(&pool, &prompt, 2);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hang]));
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier)).build();

    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &NeverCancelled)
        .await;
    assert_eq!(prefix.hit_tokens(), 2 * BLOCK_SIZE);
    assert_eq!(store.stats().resolve_degraded.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn hung_load_degrades_but_never_frees_the_reservation() {
    let config = KvStoreConfig {
        resolve_deadline: Duration::from_millis(30),
        ..KvStoreConfig::default()
    };
    let b = KvStoreBuilder::new(tokio::runtime::Handle::current(), config);
    let pool = pool(64);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::scripted([MockQuery::Hit(3)]).with_hung_loads());
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier)).build();

    let usable_before = pool.available_blocks();
    let prefix = store
        .resolve_prefix(RANK, "r1", &prompt, CacheScope::default(), &NeverCancelled)
        .await;
    // Degraded to no hit — but the DMA may still be writing: the abandoned
    // reservation stays owned by the detached task, so its destination
    // blocks must NOT have returned to the pool.
    assert_eq!(prefix.hit_tokens(), 0);
    assert_eq!(store.stats().resolve_degraded.load(Ordering::Relaxed), 1);
    assert_eq!(pool.available_blocks(), usable_before - 3);
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
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::default().with_manual_saves());
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::default());
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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
    let b = builder();
    let pool = pool(16);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::default().with_manual_saves());
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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
async fn cacheable_saves_shed_under_pin_pressure_and_retry_after() {
    // Budget = 64 * 10% = 6 blocks. Two 4-block requests: the second must
    // shed (4 + 4 > 6) instead of pinning admission out of the pool; once
    // pressure clears, re-sealing submits it (the cursor did not advance).
    let config = KvStoreConfig {
        cacheable_pin_percent: 10,
        ..KvStoreConfig::default()
    };
    let b = KvStoreBuilder::new(tokio::runtime::Handle::current(), config);
    let pool = pool(64);
    let tier = Arc::new(MockTier::default().with_manual_saves());
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

    let prompt_a: Vec<u32> = (0..=(4 * BLOCK_SIZE) as u32).map(|i| i % 241).collect();
    let prompt_b: Vec<u32> = (0..=(4 * BLOCK_SIZE) as u32).map(|i| 1 + i % 239).collect();
    let kv_a = sealed_request(&pool, &prompt_a);
    let kv_b = sealed_request(&pool, &prompt_b);

    let mut cursor_a = SaveCursor::new();
    let mut cursor_b = SaveCursor::new();
    store.seal(RANK, &kv_a, &mut cursor_a, SaveClass::Cacheable);
    assert_eq!(store.pinned_blocks(RANK), 4);
    store.seal(RANK, &kv_b, &mut cursor_b, SaveClass::Cacheable);
    assert_eq!(tier.saves.lock().unwrap().len(), 1);
    assert_eq!(store.stats().saves_shed.load(Ordering::Relaxed), 1);
    assert_eq!(store.pinned_blocks(RANK), 4);

    tier.complete_saves();
    wait_until("first save's pins released", || {
        store.pinned_blocks(RANK) == 0
    })
    .await;
    store.seal(RANK, &kv_b, &mut cursor_b, SaveClass::Cacheable);
    assert_eq!(tier.saves.lock().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_handoff_save_is_counted_and_still_releases() {
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(3);
    let tier = Arc::new(MockTier::default().with_manual_saves());
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

    let usable_before = pool.available_blocks();
    let kv = sealed_request(&pool, &prompt);
    store.retire(RANK, kv, SaveCursor::new(), SaveClass::Handoff);

    tier.fail_saves();
    wait_until("handoff failure counted", || {
        store.stats().handoff_failed.load(Ordering::Relaxed) == 1
    })
    .await;
    // The blocks still return (no leak); the miss is the peer's to observe.
    wait_until("parked KV released after failure", || {
        pool.available_blocks() >= usable_before
    })
    .await;
    assert_eq!(store.stats().saves_failed.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn seal_skips_prefix_hit_blocks() {
    let b = builder();
    let pool = pool(64);
    let prompt = prompt(6);
    seed_gpu_prefix(&pool, &prompt, 3);
    let tier = Arc::new(MockTier::default());
    let store = b.rank(RANK, Arc::clone(&pool), Some(tier.clone())).build();

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
