//! [`OffloadEngine`]: the in-process connector that moves KV blocks between
//! pegainfer's GPU paged cache and pegaflow's host/SSD tiers.
//!
//! It owns a [`PegaEngine`] plus a small tokio runtime to drive pegaflow's
//! async save/query, and translates pegainfer's page-first [`KvLayout`] into
//! pegaflow's per-layer strided registration. Block content hashes are opaque
//! `Vec<u8>` here — the caller (scheduler) derives them from kvbm sequence
//! hashes, so this layer never depends on the logical-cache hashing scheme.

use std::sync::{Arc, Mutex};

use cudarc::driver::CudaStream;
use pegaflow_core::{
    EngineError, LayerSave, PegaEngine, PrefetchStatus, QueryLeaseId, StorageConfig,
};
use pegainfer_kv_cache::KvBuffer;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// Single-GPU, single-rank topology. The dense Qwen3-4B path runs one offload
/// engine per executor rank, each owning one GPU's KV buffer.
const NAMESPACE: &str = "pegainfer";
const TP_RANK: usize = 0;
const PP_RANK: usize = 0;
const TP_SIZE: usize = 1;
const WORLD_SIZE: usize = 1;

/// bf16 KV cache: every layout stride is counted in elements, bytes are ×2.
const ELEM_SIZE: usize = std::mem::size_of::<half::bf16>();

/// Guard the `block_on` entry points: tokio panics with an opaque message if
/// you block on a runtime from within any runtime. These methods are meant for
/// the synchronous scheduler thread — fail loud and specific if that's violated.
fn assert_outside_runtime(op: &str) {
    debug_assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "OffloadEngine::{op} drives the offload runtime with block_on and must be \
         called from a synchronous thread, never from within a tokio runtime"
    );
}

/// Tuning knobs for a new [`OffloadEngine`].
pub struct OffloadConfig {
    /// Stable identifier shared across this engine's lifetime so prefix blocks
    /// saved by one request are query-visible to the next.
    pub instance_id: String,
    /// CUDA device ordinal whose KV buffer this engine offloads.
    pub device_id: i32,
    /// Host pinned-memory pool size in bytes (the CPU KV tier capacity).
    pub pinned_pool_bytes: usize,
    /// Worker threads for the embedded runtime that drives pegaflow's async
    /// save/query. Two is plenty: save is fire-and-forget, query is a brief
    /// memory-cache lookup.
    pub runtime_threads: usize,
}

impl OffloadConfig {
    pub fn new(instance_id: impl Into<String>, device_id: i32, pinned_pool_bytes: usize) -> Self {
        Self {
            instance_id: instance_id.into(),
            device_id,
            pinned_pool_bytes,
            runtime_threads: 2,
        }
    }
}

/// A query hit: how many prefix blocks pegaflow can return from its CPU tier,
/// and the lease that owns those blocks until [`OffloadEngine::load`] consumes
/// it. `num_blocks == 0` means a full miss and `lease` is `None`.
pub struct QueryHit {
    pub lease: Option<QueryLeaseId>,
    pub num_blocks: usize,
}

/// In-flight handle for a CPU→GPU load submitted to pegaflow's worker.
///
/// The load runs on pegaflow's GPU worker thread; this resolves when the DMA
/// completes. [`Self::poll`] keeps scheduler admission non-blocking; [`Self::wait`]
/// blocks for tests and non-pipelined callers.
pub struct LoadHandle {
    rx: oneshot::Receiver<Result<(), EngineError>>,
}

impl LoadHandle {
    /// Non-blocking check for a scheduler tick. `None` while still loading.
    pub fn poll(&mut self) -> Option<Result<(), EngineError>> {
        match self.rx.try_recv() {
            Ok(result) => Some(result),
            Err(oneshot::error::TryRecvError::Empty) => None,
            Err(oneshot::error::TryRecvError::Closed) => Some(Err(EngineError::Storage(
                "load worker dropped reply".into(),
            ))),
        }
    }

    /// Block the current thread until the load settles.
    pub fn wait(self) -> Result<(), EngineError> {
        self.rx
            .blocking_recv()
            .unwrap_or_else(|_| Err(EngineError::Storage("load worker dropped reply".into())))
    }
}

/// Per-layer registration geometry derived once from a [`KvBuffer`]'s layout.
///
/// Only `data_ptrs` and `size_bytes` differ per layer; the rest are the same
/// scalar broadcast across all layers (kept as vectors only to feed pegaflow's
/// one batched registration call).
struct Registration {
    layer_names: Vec<String>,
    data_ptrs: Vec<u64>,
    size_bytes: Vec<usize>,
    num_blocks: Vec<usize>,
    bytes_per_block: Vec<usize>,
    kv_stride_bytes: Vec<usize>,
    segments: Vec<usize>,
    block_stride_bytes: Vec<usize>,
}

impl Registration {
    /// Map the fused page-first buffer to pegaflow's per-layer view.
    ///
    /// Each model layer registers as one pegaflow "layer". Within a page the
    /// layout is K then V back-to-back (`layer_stride = 2·kv_block_len`), so a
    /// layer's K and V are *contiguous* — one single segment of `layer_stride`
    /// bytes copies both, and pegaflow's K/V-split path (which needs the two
    /// segments set apart, `kv_stride > bytes_per_block`) does not apply here.
    /// What is *not* contiguous is consecutive blocks of one layer: the fused
    /// buffer interleaves all layers within a page, so they sit `page_stride`
    /// apart. That gap (stride ≠ copy size) is exactly what `block_stride_bytes`
    /// decouples.
    fn from_buffer(buffer: &KvBuffer, stream: &CudaStream) -> Self {
        let layout = buffer.layout();
        let num_blocks = buffer.num_blocks();
        let base_ptr = buffer.device_ptr(stream);

        // One block's copy unit for a layer = its whole [K|V] span in a page.
        let layer_bytes = layout.layer_stride * ELEM_SIZE;
        let page_stride_bytes = layout.page_stride * ELEM_SIZE;
        let total_bytes = num_blocks * page_stride_bytes;

        let n = layout.num_layers;
        let mut reg = Registration {
            layer_names: Vec::with_capacity(n),
            data_ptrs: Vec::with_capacity(n),
            size_bytes: Vec::with_capacity(n),
            num_blocks: vec![num_blocks; n],
            bytes_per_block: vec![layer_bytes; n],
            kv_stride_bytes: vec![0; n],
            segments: vec![1; n],
            block_stride_bytes: vec![page_stride_bytes; n],
        };
        for layer in 0..n {
            let layer_off = layer * layer_bytes;
            reg.layer_names.push(layer.to_string());
            reg.data_ptrs.push(base_ptr + layer_off as u64);
            // The layer's region runs from its [K|V] base to the end of the
            // buffer; bounds are validated against the strided last-block reach.
            reg.size_bytes.push(total_bytes - layer_off);
        }
        reg
    }
}

/// In-process bridge from pegainfer's GPU KV cache to pegaflow's offload tiers.
///
/// Dropping the engine drops its [`Runtime`], which abandons any in-flight
/// fire-and-forget [`Self::save`] tasks. That is acceptable: the host tier is a
/// cache, so a lost save only forfeits a future hit, never inference
/// correctness. Saves that must survive a handoff (eviction) use the synchronous
/// [`Self::save_blocking`] instead.
pub struct OffloadEngine {
    engine: Arc<PegaEngine>,
    runtime: Runtime,
    instance_id: String,
    device_id: i32,
    /// Owned per-layer names; load borrows these as `&[&str]`.
    layer_names: Vec<String>,
    /// In-flight fire-and-forget save tasks. [`Self::flush_saves`] awaits these
    /// before draining the write pipeline, so a flush is a true barrier — the
    /// detached D2H may not even have started when the caller flushes.
    /// Finished handles are pruned on each [`Self::save`].
    pending_saves: Mutex<Vec<JoinHandle<()>>>,
}

impl OffloadEngine {
    /// Build the engine and register `buffer` as the GPU side of the offload.
    ///
    /// `stream` must be the stream that owns `buffer` (used only to read its
    /// base device address). pegaflow attaches the device's primary CUDA
    /// context for its own worker transfers — the same context pegainfer runs
    /// on — so the registered pointers are valid across both.
    pub fn new(
        config: OffloadConfig,
        buffer: &KvBuffer,
        stream: &CudaStream,
    ) -> Result<Self, EngineError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.runtime_threads.max(1))
            .enable_all()
            .build()
            .map_err(|e| EngineError::Storage(format!("offload runtime build: {e}")))?;

        let storage_config = StorageConfig::default();
        let engine = Arc::new(PegaEngine::new_with_config(
            config.pinned_pool_bytes,
            false,
            storage_config,
        )?);

        let reg = Registration::from_buffer(buffer, stream);
        engine.register_context_layer_batch_strided(
            &config.instance_id,
            NAMESPACE,
            config.device_id,
            TP_RANK,
            PP_RANK,
            TP_SIZE,
            WORLD_SIZE,
            reg.layer_names.len(),
            &reg.layer_names,
            &reg.data_ptrs,
            &reg.size_bytes,
            &reg.num_blocks,
            &reg.bytes_per_block,
            &reg.kv_stride_bytes,
            &reg.segments,
            Some(&reg.block_stride_bytes),
        )?;

        Ok(Self {
            engine,
            runtime,
            instance_id: config.instance_id,
            device_id: config.device_id,
            layer_names: reg.layer_names,
            pending_saves: Mutex::new(Vec::new()),
        })
    }

    /// Fan one (block_id, hash) list across every layer — the device data
    /// differs per layer, the ids and hashes don't.
    fn build_saves(&self, block_ids: &[i32], block_hashes: &[Vec<u8>]) -> Vec<LayerSave> {
        self.layer_names
            .iter()
            .map(|name| LayerSave {
                layer_name: name.clone(),
                block_ids: block_ids.to_vec(),
                block_hashes: block_hashes.to_vec(),
            })
            .collect()
    }

    /// Save the named GPU blocks to the host tier — fire-and-forget.
    ///
    /// Best-effort by contract: the GPU→CPU copy runs on pegaflow's worker and
    /// any failure (pinned pool full, copy error) is logged, never surfaced.
    /// `block_hashes[i]` is the content hash of `block_ids[i]`; all layers share
    /// the same (block_id, hash) pairing — only the device data differs.
    ///
    /// ORDERING CONTRACT: pegaflow's D2H runs on *its own* stream, with no
    /// dependency on pegainfer's compute stream. The caller must therefore only
    /// save blocks whose KV writes are already complete — i.e. call this after
    /// the producing forward step has synchronized (block-seal time, which is
    /// post-step-sync in the executor). Saving a block whose attention write is
    /// still in flight reads torn data. This connector cannot enforce the
    /// invariant (it does not own the compute stream); the wiring must uphold it.
    ///
    /// REUSE CONTRACT: the copy reads the GPU block asynchronously *after* this
    /// returns, so the block must stay stable until the copy lands. `keep_alive`
    /// is an opaque payload (e.g. the source blocks' allocator guards) held for
    /// the lifetime of the spawned save and dropped only once it finishes — so
    /// the caller's blocks cannot be evicted and overwritten under the in-flight
    /// D2H (which would snapshot the wrong KV and persist it under the old hash).
    /// Pass `()` only when the blocks are owned elsewhere for the whole save.
    pub fn save<G: Send + 'static>(
        &self,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
        keep_alive: G,
    ) {
        debug_assert_eq!(block_ids.len(), block_hashes.len());
        if block_ids.is_empty() {
            return;
        }
        let saves = self.build_saves(block_ids, block_hashes);
        let engine = Arc::clone(&self.engine);
        let instance_id = self.instance_id.clone();
        let device_id = self.device_id;
        let handle = self.runtime.spawn(async move {
            if let Err(e) = engine
                .batch_save_kv_blocks_from_ipc(&instance_id, TP_RANK, PP_RANK, device_id, saves)
                .await
            {
                log::warn!("pegaflow save failed (best-effort): {e}");
            }
            // Release the source-block pins only now the D2H has landed; before
            // this point the blocks must not be reused (see REUSE CONTRACT).
            drop(keep_alive);
        });
        // Track for `flush_saves`; prune the ones that already settled so the
        // list stays bounded by the genuinely in-flight saves.
        let mut pending = self.pending_saves.lock().expect("pending_saves poisoned");
        pending.retain(|h| !h.is_finished());
        pending.push(handle);
    }

    /// Save the named GPU blocks and block until the GPU→CPU copy has captured
    /// the data into the host tier (the insert may still be in flight; pair with
    /// [`Self::flush_saves`] for cache visibility).
    ///
    /// The synchronous contract is what makes this safe at eviction handoff: the
    /// GPU block can be reused the moment this returns. Errors surface, unlike
    /// the fire-and-forget [`Self::save`]. The same compute-stream ORDERING
    /// CONTRACT as [`Self::save`] applies: blocking waits on pegaflow's D2H, not
    /// on pegainfer's compute stream, so the writes must already be complete.
    pub fn save_blocking(
        &self,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<(), EngineError> {
        debug_assert_eq!(block_ids.len(), block_hashes.len());
        if block_ids.is_empty() {
            return Ok(());
        }
        assert_outside_runtime("save_blocking");
        let saves = self.build_saves(block_ids, block_hashes);
        self.runtime
            .block_on(self.engine.batch_save_kv_blocks_from_ipc(
                &self.instance_id,
                TP_RANK,
                PP_RANK,
                self.device_id,
                saves,
            ))
    }

    /// Look up how long a prefix of `block_hashes` is resident in the CPU tier.
    ///
    /// Returns the hit-block count and a lease owning those blocks; pass the
    /// lease to [`Self::load`] to copy them to GPU. `req_id` must be non-empty
    /// and unique enough to scope an in-flight prefetch (the request id works).
    pub fn query(&self, req_id: &str, block_hashes: &[Vec<u8>]) -> Result<QueryHit, EngineError> {
        if block_hashes.is_empty() {
            return Ok(QueryHit {
                lease: None,
                num_blocks: 0,
            });
        }
        assert_outside_runtime("query");
        let status = self
            .runtime
            .block_on(self.engine.count_prefix_hit_blocks_with_prefetch(
                &self.instance_id,
                req_id,
                block_hashes,
            ))?;

        match status {
            // No SSD/RDMA tier in the dense v1 path, so a prefetch never lands
            // in flight; treat the rare `Loading` as a miss this tick.
            PrefetchStatus::Loading => Ok(QueryHit {
                lease: None,
                num_blocks: 0,
            }),
            PrefetchStatus::Ready { blocks, .. } => {
                if blocks.is_empty() {
                    return Ok(QueryHit {
                        lease: None,
                        num_blocks: 0,
                    });
                }
                let num_blocks = blocks.len();
                let lease = self.engine.create_query_lease(&self.instance_id, blocks)?;
                Ok(QueryHit {
                    lease: Some(lease),
                    num_blocks,
                })
            }
        }
    }

    /// Copy the leased CPU blocks into the GPU blocks named by `dst_block_ids`,
    /// across every registered layer. Returns a non-blocking [`LoadHandle`].
    ///
    /// `dst_block_ids.len()` must equal the lease's block count (the
    /// `num_blocks` from [`Self::query`]); pegaflow maps the i-th leased block
    /// onto `dst_block_ids[i]` for each layer.
    pub fn load(
        &self,
        lease: QueryLeaseId,
        dst_block_ids: Vec<i32>,
    ) -> Result<LoadHandle, EngineError> {
        let layer_refs: Vec<&str> = self.layer_names.iter().map(String::as_str).collect();
        let loads = [(lease, dst_block_ids)];
        let rx = self.engine.batch_load_kv_blocks_multi_layer_inproc(
            &self.instance_id,
            TP_RANK,
            self.device_id,
            &layer_refs,
            &loads,
        )?;
        Ok(LoadHandle { rx })
    }

    /// Release a query lease without loading it.
    ///
    /// [`Self::query`] pins its hit blocks behind a lease until [`Self::load`]
    /// consumes it. When the caller decides not to load (e.g. no GPU
    /// destination blocks are free), it must release the lease here — a dropped
    /// [`QueryLeaseId`] is an inert token, so without this the pinned host
    /// blocks would sit unevictable until the lease's TTL expires. A no-op if
    /// the lease was already consumed by a `load`.
    pub fn release_query_lease(&self, lease: QueryLeaseId) {
        self.engine.release_query_lease(&lease);
    }

    /// Flush pending saves into the read cache so a following [`Self::query`]
    /// can see them. A correctness barrier for tests and eviction handoff, not
    /// a steady-state call.
    ///
    /// First awaits every in-flight fire-and-forget [`Self::save`] (their D2H
    /// copy + write-pipeline submit), *then* drains the write pipeline — without
    /// the first step a detached save that has not started yet would be missed.
    pub fn flush_saves(&self) {
        assert_outside_runtime("flush_saves");
        let handles: Vec<JoinHandle<()>> = {
            let mut pending = self.pending_saves.lock().expect("pending_saves poisoned");
            pending.drain(..).collect()
        };
        self.runtime.block_on(async {
            for handle in handles {
                let _ = handle.await;
            }
            self.engine.flush_saves().await;
        });
    }

    /// Drop all resident CPU-tier blocks (test/eviction helper). Saved data in
    /// a backing store would survive; the dense v1 path has none, so this
    /// empties the CPU tier.
    pub fn evict_all(&self) {
        self.engine.cleanup_memory_cache();
    }
}
