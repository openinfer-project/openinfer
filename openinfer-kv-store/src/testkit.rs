//! CPU-only test double for the [`HostTier`] seam — the contract tests drive
//! the full resolve/seal/retire orchestration against this with no GPU and no
//! pegaflow. Loads are logical-only (the pool registration is what the store
//! orchestrates; physical bytes are the real tier's business).

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use openinfer_kv_offload::SaveHandle;
use tokio::sync::oneshot;

use crate::HostTier;
use crate::TierFuture;
use crate::TierHit;
use crate::TierQuery;

/// One scripted query outcome.
#[derive(Clone, Copy, Debug)]
pub enum MockQuery {
    Miss,
    Loading,
    Hit(usize),
    /// The query future never resolves — a hung storage worker.
    Hang,
}

type PendingSave = (
    oneshot::Sender<Result<(), openinfer_kv_offload::EngineError>>,
    Box<dyn Any + Send>,
);

/// Scripted [`HostTier`]: queries pop from a script (an exhausted script
/// answers `Miss`), loads/saves/releases are recorded for assertions.
#[derive(Default)]
pub struct MockTier {
    script: Mutex<VecDeque<MockQuery>>,
    /// Hits declined via `release`, i.e. lease returns without a load.
    pub released: AtomicUsize,
    /// Destination page ids of every completed load, in call order.
    pub loads: Mutex<Vec<Vec<i32>>>,
    /// `(block_ids, block_hashes)` of every submitted save, in call order.
    pub saves: Mutex<Vec<(Vec<i32>, Vec<Vec<u8>>)>>,
    /// When true, saves stay in flight until [`Self::complete_saves`];
    /// their keep-alive payloads are held with them (the pin semantics).
    manual_saves: bool,
    /// When true, load futures never resolve — a hung DMA.
    hang_loads: bool,
    pending_saves: Mutex<Vec<PendingSave>>,
}

impl MockTier {
    #[must_use]
    pub fn scripted(script: impl IntoIterator<Item = MockQuery>) -> Self {
        Self {
            script: Mutex::new(script.into_iter().collect()),
            ..Self::default()
        }
    }

    /// A tier whose saves settle only when [`Self::complete_saves`] runs.
    #[must_use]
    pub fn with_manual_saves(mut self) -> Self {
        self.manual_saves = true;
        self
    }

    /// A tier whose loads never settle (hung DMA).
    #[must_use]
    pub fn with_hung_loads(mut self) -> Self {
        self.hang_loads = true;
        self
    }

    pub fn released(&self) -> usize {
        self.released.load(Ordering::Acquire)
    }

    pub fn pending_save_count(&self) -> usize {
        self.pending_saves.lock().expect("pending_saves").len()
    }

    /// Settle every in-flight save successfully, dropping the keep-alive
    /// payloads (releasing the source-block pins).
    pub fn complete_saves(&self) {
        for (tx, keep_alive) in self.pending_saves.lock().expect("pending_saves").drain(..) {
            let _ = tx.send(Ok(()));
            drop(keep_alive);
        }
    }

    /// Settle every in-flight save with a storage error.
    pub fn fail_saves(&self) {
        for (tx, keep_alive) in self.pending_saves.lock().expect("pending_saves").drain(..) {
            let _ = tx.send(Err(openinfer_kv_offload::EngineError::Storage(
                "mock save failure".into(),
            )));
            drop(keep_alive);
        }
    }
}

impl HostTier for MockTier {
    fn query(&self, _req_id: &str, _hashes: Vec<Vec<u8>>) -> TierFuture<anyhow::Result<TierQuery>> {
        let step = self
            .script
            .lock()
            .expect("script")
            .pop_front()
            .unwrap_or(MockQuery::Miss);
        match step {
            MockQuery::Miss => Box::pin(std::future::ready(Ok(TierQuery::Miss))),
            MockQuery::Loading => Box::pin(std::future::ready(Ok(TierQuery::Loading))),
            MockQuery::Hit(blocks) => Box::pin(std::future::ready(Ok(TierQuery::Hit(TierHit {
                blocks,
                token: Box::new(()),
            })))),
            MockQuery::Hang => Box::pin(std::future::pending()),
        }
    }

    fn load(&self, _hit: TierHit, dst_page_ids: Vec<i32>) -> TierFuture<anyhow::Result<()>> {
        if self.hang_loads {
            return Box::pin(std::future::pending());
        }
        self.loads.lock().expect("loads").push(dst_page_ids);
        Box::pin(std::future::ready(Ok(())))
    }

    fn release(&self, _hit: TierHit) {
        self.released.fetch_add(1, Ordering::AcqRel);
    }

    fn save(
        &self,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
        keep_alive: Box<dyn Any + Send>,
    ) -> SaveHandle {
        self.saves
            .lock()
            .expect("saves")
            .push((block_ids, block_hashes));
        if self.manual_saves {
            let (handle, tx) = SaveHandle::in_flight();
            self.pending_saves
                .lock()
                .expect("pending_saves")
                .push((tx, keep_alive));
            handle
        } else {
            drop(keep_alive);
            SaveHandle::settled(Ok(()))
        }
    }
}
