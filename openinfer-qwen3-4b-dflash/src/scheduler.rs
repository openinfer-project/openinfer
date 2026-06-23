use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel as channel;

use crate::executor::{
    DFlashBatchKey, DFlashDraftHostRequest, DFlashDraftHostResponse, DFlashExecutor,
    DFlashExecutorOptions, DFlashRequestId,
};

pub struct DFlashSchedulerOptions {
    pub executor: DFlashExecutorOptions,
    pub max_wait: Duration,
    pub max_total_tokens: usize,
}

impl Default for DFlashSchedulerOptions {
    fn default() -> Self {
        Self {
            executor: DFlashExecutorOptions::default(),
            max_wait: Duration::from_micros(200),
            max_total_tokens: 512,
        }
    }
}

/// Handle to the DFlash draft scheduler thread. Mirrors the `EngineHandle`
/// pattern (`openinfer-engine::engine::EngineHandle`): the handle is cheaply
/// cloneable (shared sender), and the last clone's `Drop` closes the channel
/// and joins the scheduler thread, replying "stopped" to any in-flight
/// requests. This prevents leaking the GPU-owner thread when a caller drops
/// the handle without an explicit shutdown.
#[derive(Clone)]
pub struct DFlashSchedulerHandle {
    inner: Arc<DFlashSchedulerInner>,
}

struct DFlashSchedulerInner {
    submit_tx: Option<channel::Sender<SchedulerMessage>>,
    join_handle: Option<JoinHandle<()>>,
}

impl Drop for DFlashSchedulerInner {
    fn drop(&mut self) {
        // Drop our sender first; when the last sender goes, the scheduler
        // loop's `recv` returns `Err` and the thread flushes pending requests
        // via `send_stopped` before exiting (mirrors EngineHandle::Drop in
        // openinfer-engine/src/engine.rs).
        self.submit_tx.take();
        if let Some(join_handle) = self.join_handle.take() {
            // Never join from inside the scheduler thread itself.
            if join_handle.thread().id() != thread::current().id() {
                let _ = join_handle.join();
            }
        }
    }
}

enum SchedulerMessage {
    Submit {
        request: DFlashDraftHostRequest,
        response_tx: channel::Sender<Result<DFlashDraftHostResponse>>,
    },
    ResetCache {
        request_id: DFlashRequestId,
        response_tx: channel::Sender<Result<()>>,
    },
    DropCache {
        request_id: DFlashRequestId,
        response_tx: channel::Sender<Result<()>>,
    },
    CropCache {
        request_id: DFlashRequestId,
        seq_len: usize,
        response_tx: channel::Sender<Result<()>>,
    },
    CacheSeqLen {
        request_id: DFlashRequestId,
        response_tx: channel::Sender<Result<usize>>,
    },
}

struct PendingRequest {
    request: DFlashDraftHostRequest,
    response_tx: channel::Sender<Result<DFlashDraftHostResponse>>,
    queued_at: Instant,
}

enum PendingItem {
    Submit(PendingRequest),
    Control(SchedulerControl),
}

enum SchedulerControl {
    ResetCache {
        request_id: DFlashRequestId,
        response_tx: channel::Sender<Result<()>>,
    },
    DropCache {
        request_id: DFlashRequestId,
        response_tx: channel::Sender<Result<()>>,
    },
    CropCache {
        request_id: DFlashRequestId,
        seq_len: usize,
        response_tx: channel::Sender<Result<()>>,
    },
    CacheSeqLen {
        request_id: DFlashRequestId,
        response_tx: channel::Sender<Result<usize>>,
    },
}

impl DFlashSchedulerHandle {
    pub fn start(
        model_path: &Path,
        device_ordinal: usize,
        options: DFlashSchedulerOptions,
    ) -> Result<Self> {
        let (submit_tx, submit_rx) = channel::unbounded();
        let (init_tx, init_rx) = channel::bounded(1);
        let model_path = PathBuf::from(model_path);
        let max_wait = options.max_wait;
        let max_total_tokens = options.max_total_tokens;
        let join_handle = thread::Builder::new()
            .name("qwen3-dflash-scheduler".into())
            .spawn(move || {
                let mut executor =
                    match DFlashExecutor::load(&model_path, device_ordinal, options.executor) {
                        Ok(executor) => executor,
                        Err(err) => {
                            let _ = init_tx.send(Err(err));
                            return;
                        }
                    };
                let _ = init_tx.send(Ok(()));
                scheduler_loop(&mut executor, submit_rx, max_wait, max_total_tokens);
            })
            .expect("failed to spawn DFlash scheduler thread");
        init_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DFlash scheduler initialization channel closed"))??;
        Ok(Self {
            inner: Arc::new(DFlashSchedulerInner {
                submit_tx: Some(submit_tx),
                join_handle: Some(join_handle),
            }),
        })
    }

    fn submit_tx(&self) -> Result<&channel::Sender<SchedulerMessage>> {
        self.inner
            .submit_tx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash scheduler is closed"))
    }

    pub fn submit(&self, request: DFlashDraftHostRequest) -> Result<DFlashDraftHostResponse> {
        let (response_tx, response_rx) = channel::bounded(1);
        self.submit_tx()?
            .send(SchedulerMessage::Submit {
                request,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("DFlash scheduler is closed"))?;
        response_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DFlash scheduler response channel closed"))?
    }

    pub fn reset_cache(&self, request_id: DFlashRequestId) -> Result<()> {
        let (response_tx, response_rx) = channel::bounded(1);
        self.submit_tx()?
            .send(SchedulerMessage::ResetCache {
                request_id,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("DFlash scheduler is closed"))?;
        response_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DFlash scheduler response channel closed"))?
    }

    /// Release a request's draft cache and reclaim its GPU buffers. Mirrors
    /// Qwen3's `drop_request`: the executor removes the cache entry and RAII
    /// frees the per-layer past K/V + scratch. Idempotent — retiring a
    /// request that never created a cache is not an error. Callers should
    /// invoke this once a draft request is verified or abandoned so the
    /// `max_caches` pool does not fill with dead entries.
    pub fn drop_cache(&self, request_id: DFlashRequestId) -> Result<()> {
        let (response_tx, response_rx) = channel::bounded(1);
        self.submit_tx()?
            .send(SchedulerMessage::DropCache {
                request_id,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("DFlash scheduler is closed"))?;
        response_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DFlash scheduler response channel closed"))?
    }

    pub fn crop_cache(&self, request_id: DFlashRequestId, seq_len: usize) -> Result<()> {
        let (response_tx, response_rx) = channel::bounded(1);
        self.submit_tx()?
            .send(SchedulerMessage::CropCache {
                request_id,
                seq_len,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("DFlash scheduler is closed"))?;
        response_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DFlash scheduler response channel closed"))?
    }

    pub fn cache_seq_len(&self, request_id: DFlashRequestId) -> Result<usize> {
        let (response_tx, response_rx) = channel::bounded(1);
        self.submit_tx()?
            .send(SchedulerMessage::CacheSeqLen {
                request_id,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("DFlash scheduler is closed"))?;
        response_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DFlash scheduler response channel closed"))?
    }
}

fn scheduler_loop(
    executor: &mut DFlashExecutor,
    submit_rx: channel::Receiver<SchedulerMessage>,
    max_wait: Duration,
    max_total_tokens: usize,
) {
    let mut pending: VecDeque<PendingItem> = VecDeque::new();
    loop {
        if pending.is_empty() {
            match submit_rx.recv() {
                Ok(msg) => handle_message_or_enqueue(msg, &mut pending),
                Err(_) => break,
            }
        }
        while let Ok(msg) = submit_rx.try_recv() {
            handle_message_or_enqueue(msg, &mut pending);
        }
        if pending.is_empty() {
            continue;
        }
        let head_wait = pending
            .front()
            .and_then(PendingItem::queued_elapsed)
            .unwrap_or(max_wait);
        if pending.len() == 1 && head_wait < max_wait {
            let timeout = max_wait - head_wait;
            if let Ok(msg) = submit_rx.recv_timeout(timeout) {
                handle_message_or_enqueue(msg, &mut pending);
                while let Ok(msg) = submit_rx.try_recv() {
                    handle_message_or_enqueue(msg, &mut pending);
                }
            }
        }
        drain_one_batch(executor, &mut pending, max_total_tokens);
    }
    for pending in pending {
        pending.send_stopped();
    }
}

fn handle_message_or_enqueue(msg: SchedulerMessage, pending: &mut VecDeque<PendingItem>) {
    match msg {
        SchedulerMessage::Submit {
            request,
            response_tx,
        } => pending.push_back(PendingItem::Submit(PendingRequest {
            request,
            response_tx,
            queued_at: Instant::now(),
        })),
        SchedulerMessage::ResetCache {
            request_id,
            response_tx,
        } => pending.push_back(PendingItem::Control(SchedulerControl::ResetCache {
            request_id,
            response_tx,
        })),
        SchedulerMessage::DropCache {
            request_id,
            response_tx,
        } => pending.push_back(PendingItem::Control(SchedulerControl::DropCache {
            request_id,
            response_tx,
        })),
        SchedulerMessage::CropCache {
            request_id,
            seq_len,
            response_tx,
        } => pending.push_back(PendingItem::Control(SchedulerControl::CropCache {
            request_id,
            seq_len,
            response_tx,
        })),
        SchedulerMessage::CacheSeqLen {
            request_id,
            response_tx,
        } => pending.push_back(PendingItem::Control(SchedulerControl::CacheSeqLen {
            request_id,
            response_tx,
        })),
    }
}

fn drain_one_batch(
    executor: &mut DFlashExecutor,
    pending: &mut VecDeque<PendingItem>,
    max_total_tokens: usize,
) {
    let Some(first) = pending.pop_front() else {
        return;
    };
    let PendingItem::Submit(first) = first else {
        if let PendingItem::Control(control) = first {
            control.execute(executor);
        }
        return;
    };
    let key = match executor.host_batch_key(&first.request) {
        Ok(key) => key,
        Err(err) => {
            let _ = first.response_tx.send(Err(err));
            return;
        }
    };
    let max_batch_size = executor_max_batch_size(executor);
    let mut batch = vec![first];
    let mut total_tokens = key.q_len + key.ctx_len + key.past_len;
    if total_tokens > max_total_tokens {
        let err = anyhow::anyhow!(
            "DFlash scheduler request total tokens {} exceeds max_total_tokens {}",
            total_tokens,
            max_total_tokens
        );
        let first = batch.pop().expect("first request exists");
        let _ = first.response_tx.send(Err(err));
        return;
    }
    let mut i = 0;
    while i < pending.len() && batch.len() < max_batch_size {
        if !matches!(pending.get(i), Some(PendingItem::Submit(_))) {
            break;
        }
        let matches = pending
            .get(i)
            .map(|candidate| {
                let PendingItem::Submit(candidate) = candidate else {
                    return false;
                };
                request_matches_key(
                    executor,
                    &candidate.request,
                    key,
                    total_tokens,
                    max_total_tokens,
                )
            })
            .unwrap_or(false);
        if matches {
            total_tokens += key.q_len + key.ctx_len + key.past_len;
            match pending.remove(i).expect("pending index exists") {
                PendingItem::Submit(request) => batch.push(request),
                PendingItem::Control(_) => unreachable!("control items are batch barriers"),
            }
        } else {
            i += 1;
        }
    }
    let response_txs = batch
        .iter()
        .map(|req| req.response_tx.clone())
        .collect::<Vec<_>>();
    let requests = batch.into_iter().map(|pending| pending.request).collect();
    match executor.execute_host_batch_host(requests) {
        Ok(responses) => {
            for (response_tx, response) in response_txs.into_iter().zip(responses.into_iter()) {
                let _ = response_tx.send(Ok(response));
            }
        }
        Err(err) => {
            let message = err.to_string();
            for response_tx in response_txs {
                let _ = response_tx.send(Err(anyhow::anyhow!(message.clone())));
            }
        }
    }
}

fn request_matches_key(
    executor: &DFlashExecutor,
    request: &DFlashDraftHostRequest,
    key: DFlashBatchKey,
    current_total_tokens: usize,
    max_total_tokens: usize,
) -> bool {
    executor
        .host_batch_key(request)
        .map(|candidate| {
            let candidate_tokens = candidate.q_len + candidate.ctx_len + candidate.past_len;
            candidate == key && current_total_tokens + candidate_tokens <= max_total_tokens
        })
        .unwrap_or(false)
}

fn executor_max_batch_size(executor: &DFlashExecutor) -> usize {
    executor.max_batch_size()
}

impl PendingItem {
    fn queued_elapsed(&self) -> Option<Duration> {
        match self {
            PendingItem::Submit(request) => Some(request.queued_at.elapsed()),
            PendingItem::Control(_) => None,
        }
    }

    fn send_stopped(self) {
        match self {
            PendingItem::Submit(request) => {
                let _ = request
                    .response_tx
                    .send(Err(anyhow::anyhow!("DFlash scheduler stopped")));
            }
            PendingItem::Control(control) => control.send_stopped(),
        }
    }
}

impl SchedulerControl {
    fn execute(self, executor: &mut DFlashExecutor) {
        match self {
            SchedulerControl::ResetCache {
                request_id,
                response_tx,
            } => {
                let _ = response_tx.send(executor.reset_cache(request_id));
            }
            SchedulerControl::DropCache {
                request_id,
                response_tx,
            } => {
                let _ = response_tx.send(executor.drop_cache(request_id));
            }
            SchedulerControl::CropCache {
                request_id,
                seq_len,
                response_tx,
            } => {
                let _ = response_tx.send(executor.crop_cache(request_id, seq_len));
            }
            SchedulerControl::CacheSeqLen {
                request_id,
                response_tx,
            } => {
                let _ = response_tx.send(executor.cache_seq_len(request_id));
            }
        }
    }

    fn send_stopped(self) {
        match self {
            SchedulerControl::ResetCache { response_tx, .. }
            | SchedulerControl::DropCache { response_tx, .. }
            | SchedulerControl::CropCache { response_tx, .. } => {
                let _ = response_tx.send(Err(anyhow::anyhow!("DFlash scheduler stopped")));
            }
            SchedulerControl::CacheSeqLen { response_tx, .. } => {
                let _ = response_tx.send(Err(anyhow::anyhow!("DFlash scheduler stopped")));
            }
        }
    }
}
