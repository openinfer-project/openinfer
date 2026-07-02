//! [`OffloadEngine`]: the in-process connector that moves KV blocks between
//! openinfer's GPU paged cache and pegaflow's host/SSD tiers.
//!
//! It owns a [`PegaEngine`] plus a small tokio runtime to drive pegaflow's
//! async save/query, and translates openinfer's page-first [`KvLayout`] into
//! pegaflow's per-layer strided registration. Block content hashes are opaque
//! `Vec<u8>` here — the caller (scheduler) derives them from kvbm sequence
//! hashes, so this layer never depends on the logical-cache hashing scheme.

use std::sync::{Arc, Mutex};

use cudarc::driver::CudaStream;
use openinfer_kv_cache::KvBuffer;
use pegaflow_core::{
    EngineError, LayerSave, P2pTransferService, PegaEngine, PrefetchStatus, QueryLeaseId,
    StorageConfig, TransferMode,
};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// Single-GPU, single-rank topology. The dense Qwen3 path runs one offload
/// engine per executor rank, each owning one GPU's KV buffer.
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

/// Cross-instance P2P sharing over pegaflow's MetaServer + RDMA data plane.
///
/// With this set, the engine (a) registers saved block hashes with the
/// MetaServer, (b) serves peer RDMA fetches on `listen_addr`, and (c) on a
/// local host-tier miss, discovers and pulls the missing prefix from whichever
/// peer owns it (one-sided RDMA READ into the local pinned pool, then a normal
/// H2D load). This is the P/D disaggregation data plane: a decode node finds
/// the prefill node's KV by content hash — no handle protocol.
#[derive(Clone, Debug)]
pub struct P2pConfig {
    /// MetaServer gRPC address, e.g. `http://10.0.0.100:50056`.
    pub metaserver_addr: String,
    /// This engine's routable `host:port` — peers dial it for RDMA handshakes
    /// and block queries, and the MetaServer records it as the block owner.
    /// Must match `listen_addr` (same port) and must not be 0.0.0.0/127.0.0.1
    /// for cross-node use.
    pub advertise_addr: String,
    /// RDMA NIC device names to register the pinned pool on (e.g. `mlx5_0`).
    pub rdma_nics: Vec<String>,
}

/// Tuning knobs for a new [`OffloadEngine`].
pub struct OffloadConfig {
    /// Stable identifier shared across this engine's lifetime so prefix blocks
    /// saved by one request are query-visible to the next.
    pub instance_id: String,
    /// Content-addressing domain shared with P2P peers: two engines see each
    /// other's blocks iff their namespaces match. Callers derive it from
    /// whatever makes KV layouts interchange-safe (model, dtype, block
    /// geometry). Single-node offload can use any constant.
    pub namespace: String,
    /// CUDA device ordinal whose KV buffer this engine offloads.
    pub device_id: i32,
    /// Host pinned-memory pool size in bytes (the CPU KV tier capacity).
    pub pinned_pool_bytes: usize,
    /// Worker threads for the embedded runtime that drives pegaflow's async
    /// save/query. Two is plenty: save is fire-and-forget, query is a brief
    /// memory-cache lookup.
    pub runtime_threads: usize,
    /// `Some` joins the cross-instance P2P mesh (see [`P2pConfig`]).
    pub p2p: Option<P2pConfig>,
}

impl OffloadConfig {
    pub fn new(instance_id: impl Into<String>, device_id: i32, pinned_pool_bytes: usize) -> Self {
        Self {
            instance_id: instance_id.into(),
            namespace: "openinfer".to_string(),
            device_id,
            pinned_pool_bytes,
            runtime_threads: 2,
            p2p: None,
        }
    }

    #[must_use]
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    #[must_use]
    pub fn with_p2p(mut self, p2p: P2pConfig) -> Self {
        self.p2p = Some(p2p);
        self
    }
}

/// A query hit: how many prefix blocks pegaflow can return from its CPU tier,
/// and the lease that owns those blocks until [`OffloadEngine::load`] consumes
/// it. `num_blocks == 0` means a full miss and `lease` is `None`.
pub struct QueryHit {
    pub lease: Option<QueryLeaseId>,
    pub num_blocks: usize,
}

/// Outcome of [`OffloadEngine::query`].
pub enum QueryOutcome {
    /// Terminal: `hit.num_blocks` prefix blocks are host-resident and leased.
    Ready(QueryHit),
    /// pegaflow kicked off an async fetch of the missing prefix from a remote
    /// peer (P2P) or SSD. Not terminal: re-`query` with the same `req_id` next
    /// tick to poll; the fetch resolves to `Ready` (with the pulled blocks) or
    /// falls back to a plain local hit count. Only occurs with a deeper tier
    /// configured — never in the host-memory-only setup.
    Loading,
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

/// In-process bridge from openinfer's GPU KV cache to pegaflow's offload tiers.
///
/// Dropping the engine drops its [`Runtime`], which abandons any in-flight
/// fire-and-forget [`Self::save`] tasks. That is acceptable: the host tier is a
/// cache, so a lost save only forfeits a future hit, never inference
/// correctness. Saves that must survive a handoff (eviction) use the synchronous
/// [`Self::save_blocking`] instead. The P2P serving tasks (if any) stop with
/// the runtime as well; peers degrade to their own local prefill.
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
    /// `Some` when P2P is on: resolves the P2P serving tasks (gRPC transfer
    /// service + transfer-lock GC) on drop.
    p2p_shutdown: Option<oneshot::Sender<()>>,
}

impl OffloadEngine {
    /// Build the engine and register `buffer` as the GPU side of the offload.
    ///
    /// `stream` must be the stream that owns `buffer` (used only to read its
    /// base device address). pegaflow attaches the device's primary CUDA
    /// context for its own worker transfers — the same context openinfer runs
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

        let mut storage_config = StorageConfig::default();
        if let Some(p2p) = &config.p2p {
            if p2p.rdma_nics.is_empty() {
                return Err(EngineError::InvalidArgument(
                    "P2P requires at least one RDMA NIC".into(),
                ));
            }
            storage_config.rdma_nic_names = Some(p2p.rdma_nics.clone());
            storage_config.metaserver_addr = Some(p2p.metaserver_addr.clone());
            storage_config.advertise_addr = Some(p2p.advertise_addr.clone());
        }
        // pegaflow's MetaServerClient spawns its background registration loop
        // with tokio::spawn, so the engine must be built inside our runtime.
        let engine = {
            let _guard = runtime.enter();
            Arc::new(PegaEngine::new_with_config(
                config.pinned_pool_bytes,
                false,
                storage_config,
            )?)
        };

        // P2P serving side: peers discovered us via the MetaServer and dial
        // `advertise_addr` for the RDMA handshake + block queries. Same
        // lifecycle as the engine — shut down (via the oneshot) on drop.
        let p2p_shutdown = match &config.p2p {
            Some(p2p) => {
                let listen: std::net::SocketAddr = p2p.advertise_addr.parse().map_err(|e| {
                    EngineError::InvalidArgument(format!(
                        "P2P advertise_addr {:?} is not a socket address: {e}",
                        p2p.advertise_addr
                    ))
                })?;
                let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
                let serve_engine = Arc::clone(&engine);
                let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
                runtime.spawn(async move {
                    // Bind eagerly so startup fails loud on a taken port
                    // instead of P2P silently never serving.
                    let bound = tokio::net::TcpListener::bind(listen).await;
                    let listener = match bound {
                        Ok(l) => {
                            let _ = ready_tx.send(Ok(()));
                            l
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!("bind {listen}: {e}")));
                            return;
                        }
                    };
                    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
                    if let Err(e) =
                        P2pTransferService::serve_with_incoming(serve_engine, incoming, async {
                            let _ = shutdown_rx.await;
                        })
                        .await
                    {
                        log::error!("P2P transfer service exited: {e}");
                    }
                });
                ready_rx
                    .recv()
                    .map_err(|_| EngineError::Storage("P2P serve task died at startup".into()))?
                    .map_err(EngineError::Storage)?;

                // Expired-lock GC, mirroring pegaflow-server's background task:
                // a crashed peer must not pin our blocks past the lock timeout.
                let gc_engine = Arc::clone(&engine);
                runtime.spawn(async move {
                    let mut tick = tokio::time::interval(std::time::Duration::from_mins(1));
                    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        tick.tick().await;
                        let expired = gc_engine.gc_expired_transfer_locks();
                        if expired > 0 {
                            log::warn!("P2P GC released {expired} expired transfer locks");
                        }
                    }
                });
                log::info!(
                    "KV offload P2P enabled: serving on {listen}, metaserver={}",
                    p2p.metaserver_addr
                );
                Some(shutdown_tx)
            }
            None => None,
        };

        let reg = Registration::from_buffer(buffer, stream);
        engine.register_context_layer_batch_strided(
            &config.instance_id,
            &config.namespace,
            config.device_id,
            TP_RANK,
            PP_RANK,
            TP_SIZE,
            WORLD_SIZE,
            &reg.layer_names,
            &reg.data_ptrs,
            &reg.size_bytes,
            &reg.num_blocks,
            &reg.bytes_per_block,
            &reg.kv_stride_bytes,
            &reg.segments,
            Some(reg.block_stride_bytes.as_slice()),
            // Direct (cuMemcpyAsync on the DMA engines) suits this path's few,
            // large per-layer copies; the Kernel backend only wins for highly
            // fragmented batches.
            TransferMode::Direct,
            // page_first = false: openinfer registers one pegaflow layer per
            // model layer (see `Registration::from_buffer`) and expresses the
            // page-interleaved gap via `block_stride_bytes` — the layer-first
            // model. The page-first path instead collapses all layers into a
            // single page slot per block, which this per-layer registration is
            // not laid out for.
            false,
        )?;

        Ok(Self {
            engine,
            runtime,
            instance_id: config.instance_id,
            device_id: config.device_id,
            layer_names: reg.layer_names,
            pending_saves: Mutex::new(Vec::new()),
            p2p_shutdown,
        })
    }

    /// Fan one (block_id, hash) list across every layer — the device data
    /// differs per layer, the ids and hashes don't.
    fn build_saves(&self, block_ids: &[i32], block_hashes: &[Vec<u8>]) -> Vec<LayerSave> {
        // pegaflow indexes GPU blocks by `usize`; openinfer carries them as
        // `i32` (its kvbm/CUDA convention). Convert once at this boundary —
        // block ids are slot indices, always non-negative.
        let block_ids: Vec<usize> = block_ids.iter().map(|&id| id as usize).collect();
        self.layer_names
            .iter()
            .map(|name| LayerSave {
                layer_name: name.clone(),
                block_ids: block_ids.clone(),
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
    /// dependency on openinfer's compute stream. The caller must therefore only
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
    /// on openinfer's compute stream, so the writes must already be complete.
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
    /// Returns [`QueryOutcome::Ready`] with the hit-block count and a lease
    /// owning those blocks (pass the lease to [`Self::load`] to copy them to
    /// GPU), or [`QueryOutcome::Loading`] when pegaflow is fetching the missing
    /// prefix from a remote peer / SSD in the background — re-`query` with the
    /// same `req_id` to poll. `req_id` must be non-empty and unique enough to
    /// scope an in-flight prefetch (the request id works).
    pub fn query(
        &self,
        req_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<QueryOutcome, EngineError> {
        if block_hashes.is_empty() {
            return Ok(QueryOutcome::Ready(QueryHit {
                lease: None,
                num_blocks: 0,
            }));
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
            PrefetchStatus::Loading => Ok(QueryOutcome::Loading),
            PrefetchStatus::Ready { blocks, .. } => {
                if blocks.is_empty() {
                    return Ok(QueryOutcome::Ready(QueryHit {
                        lease: None,
                        num_blocks: 0,
                    }));
                }
                let num_blocks = blocks.len();
                let lease = self.engine.create_query_lease(&self.instance_id, blocks)?;
                Ok(QueryOutcome::Ready(QueryHit {
                    lease: Some(lease),
                    num_blocks,
                }))
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
        // pegaflow indexes GPU blocks by `usize` (see `build_saves`).
        let dst_block_ids: Vec<usize> = dst_block_ids.into_iter().map(|id| id as usize).collect();
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

    /// Whether this engine participates in the cross-instance P2P mesh.
    pub fn p2p_enabled(&self) -> bool {
        self.p2p_shutdown.is_some()
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
    /// can see them — and, with P2P on, so the MetaServer knows this engine
    /// owns the saved hashes. A correctness barrier for tests, eviction
    /// handoff, and the P/D KV-ready signal; not a steady-state call.
    ///
    /// First awaits every in-flight fire-and-forget [`Self::save`] (their D2H
    /// copy + write-pipeline submit), then drains the write pipeline, then
    /// waits for the queued MetaServer registrations to be delivered (or
    /// dropped after a failed attempt — registration stays best-effort, the
    /// barrier only bounds when). Without P2P the last step is a no-op.
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
            self.engine.flush_saves_and_registrations().await;
        });
    }

    /// Drop all resident CPU-tier blocks (test/eviction helper). Saved data in
    /// a backing store would survive; the dense v1 path has none, so this
    /// empties the CPU tier.
    pub fn evict_all(&self) {
        self.engine.cleanup_memory_cache();
    }
}
