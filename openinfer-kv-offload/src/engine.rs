//! [`OffloadEngine`]: the in-process connector that moves KV blocks between
//! openinfer's GPU paged cache and pegaflow's host/SSD tiers.
//!
//! It owns a [`PegaEngine`] plus a small tokio runtime to drive pegaflow's
//! async save/query, and translates openinfer's page-first [`KvLayout`] into
//! pegaflow's per-layer strided registration. Block content hashes are opaque
//! `Vec<u8>` here — the caller (scheduler) derives them from kvbm sequence
//! hashes, so this layer never depends on the logical-cache hashing scheme.

use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::CudaStream;
use openinfer_kv_cache::KvBuffer;
use pegaflow_core::EngineError;
use pegaflow_core::LayerSave;
use pegaflow_core::P2pTransferService;
use pegaflow_core::PegaEngine;
use pegaflow_core::PrefetchStatus;
use pegaflow_core::QueryLeaseId;
use pegaflow_core::StorageConfig;
use pegaflow_core::TransferMode;
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

/// Upper bound on the [`OffloadEngine::flush_saves_then`] barrier. Generous
/// for the normal case (D2H drain + a few local RPCs complete in
/// milliseconds); the cap only bites when the MetaServer connection stalls
/// mid-RPC, where the alternative is withholding finished requests' responses
/// for the TCP keepalive window.
const FLUSH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

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
    /// This engine's routable `IP:port` (a literal socket address — it doubles
    /// as the embedded transfer service's bind address, so hostnames are
    /// rejected at startup). Peers dial it for RDMA handshakes and block
    /// queries, and the MetaServer records it as the block owner. Must not be
    /// 0.0.0.0/127.0.0.1 for cross-node use.
    pub advertise_addr: String,
    /// RDMA NIC device names to register the pinned pool on (e.g. `mlx5_0`).
    pub rdma_nics: Vec<String>,
}

/// Tuning knobs for a new [`OffloadEngine`].
pub struct OffloadConfig {
    /// Stable identifier shared across this engine's lifetime so prefix blocks
    /// saved by one request are query-visible to the next.
    instance_id: String,
    /// Content-addressing domain shared with P2P peers: two engines see each
    /// other's blocks iff their namespaces match. Callers derive it from
    /// whatever makes KV layouts interchange-safe (model, dtype, block
    /// geometry). Single-node offload can use any constant.
    namespace: String,
    /// CUDA device ordinal whose KV buffer this engine offloads.
    device_id: i32,
    /// Host pinned-memory pool size in bytes (the CPU KV tier capacity).
    pinned_pool_bytes: usize,
    /// Back the pinned pool with hugepages (see [`HostConfig::use_hugepages`]).
    pub use_hugepages: bool,
    /// Worker threads for the embedded runtime that drives pegaflow's async
    /// save/query. Two is plenty: save is fire-and-forget, query is a brief
    /// memory-cache lookup.
    runtime_threads: usize,
    /// `Some` joins the cross-instance P2P mesh (see [`P2pConfig`]).
    p2p: Option<P2pConfig>,
}

impl OffloadConfig {
    pub fn new(instance_id: impl Into<String>, device_id: i32, pinned_pool_bytes: usize) -> Self {
        Self {
            instance_id: instance_id.into(),
            namespace: "openinfer".to_string(),
            device_id,
            pinned_pool_bytes,
            use_hugepages: false,
            runtime_threads: 2,
            p2p: None,
        }
    }

    /// The host-tier half of this config (private-host constructors split it
    /// off before consuming the instance fields).
    fn host(&self) -> HostConfig {
        HostConfig {
            pinned_pool_bytes: self.pinned_pool_bytes,
            use_hugepages: self.use_hugepages,
            runtime_threads: self.runtime_threads,
            p2p: self.p2p.clone(),
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

/// One strided GPU arena to register as one pegaflow "layer": `num_blocks`
/// copy units of `bytes_per_block`, sitting `block_stride_bytes` apart from
/// `base_ptr`. A fused buffer (qwen3) contributes one arena per model layer;
/// a model with sidecar caches (GLM5.2: MLA latent + index-K per layer, two
/// separate allocations sharing pool block ids) contributes several arenas
/// per model layer — pegaflow moves whatever arenas are registered under one
/// block id together, which is what keeps sidecars in lockstep with their
/// main cache.
///
/// `name` keys the arena for the whole engine lifetime (save/load fan across
/// every registered name); it must be unique within the engine.
pub struct KvArena {
    pub name: String,
    pub base_ptr: u64,
    pub num_blocks: usize,
    pub bytes_per_block: usize,
    pub block_stride_bytes: usize,
}

/// Per-layer registration geometry fed to pegaflow's one batched call.
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

        let arenas: Vec<KvArena> = (0..layout.num_layers)
            .map(|layer| KvArena {
                name: layer.to_string(),
                base_ptr: base_ptr + (layer * layer_bytes) as u64,
                num_blocks,
                bytes_per_block: layer_bytes,
                block_stride_bytes: page_stride_bytes,
            })
            .collect();
        Self::from_arenas(&arenas)
    }

    /// One pegaflow layer per arena, single-segment (an arena is one copy
    /// unit per block by definition; K/V split segments only exist for the
    /// symmetric-pair layouts vLLM registers).
    fn from_arenas(arenas: &[KvArena]) -> Self {
        let n = arenas.len();
        let mut reg = Registration {
            layer_names: Vec::with_capacity(n),
            data_ptrs: Vec::with_capacity(n),
            size_bytes: Vec::with_capacity(n),
            num_blocks: Vec::with_capacity(n),
            bytes_per_block: Vec::with_capacity(n),
            kv_stride_bytes: vec![0; n],
            segments: vec![1; n],
            block_stride_bytes: Vec::with_capacity(n),
        };
        for arena in arenas {
            assert!(
                arena.bytes_per_block <= arena.block_stride_bytes,
                "arena {} copy unit {} overruns its block stride {}",
                arena.name,
                arena.bytes_per_block,
                arena.block_stride_bytes
            );
            reg.layer_names.push(arena.name.clone());
            reg.data_ptrs.push(arena.base_ptr);
            // The arena's region must cover the strided reach of its last
            // block (pegaflow validates copies against this bound).
            reg.size_bytes
                .push((arena.num_blocks - 1) * arena.block_stride_bytes + arena.bytes_per_block);
            reg.num_blocks.push(arena.num_blocks);
            reg.bytes_per_block.push(arena.bytes_per_block);
            reg.block_stride_bytes.push(arena.block_stride_bytes);
        }
        reg
    }
}

/// Host-tier knobs for a shared [`OffloadHost`].
pub struct HostConfig {
    /// Host pinned-memory pool size in bytes (the CPU KV tier capacity).
    pub pinned_pool_bytes: usize,
    /// Back the pinned pool with hugepages (pegaflow supports it natively).
    /// Verify the box actually holds a reservation (`HugePages_Total`) —
    /// some cluster platforms re-claim it across reboots.
    pub use_hugepages: bool,
    /// Worker threads for the runtime that drives pegaflow's async save/query.
    pub runtime_threads: usize,
    /// `Some` joins the cross-instance P2P mesh (see [`P2pConfig`]).
    pub p2p: Option<P2pConfig>,
}

/// The shared side of the offload: one [`PegaEngine`] (one host pool), the
/// tokio runtime that drives it, and the optional P2P serving lifecycle.
///
/// One host serves any number of rank-level [`OffloadEngine`]s. That is the
/// DP-rank sharing model: each rank registers its own GPU arenas as its own
/// pegaflow *instance*, but blocks land in the one host tier keyed by
/// `(namespace, hash)` — with a shared namespace, any rank restores what any
/// rank saved. Callers share a namespace only when their KV is
/// interchangeable across instances: for replicated-weight DP ranks that
/// holds to the same tolerance as reusing a rank's own prefix cache (the
/// bytes may differ by FP reduction order across batch shapes, exactly like
/// two local recomputations of the same prefix would).
///
/// Dropping the last handle drops the [`Runtime`], which abandons any
/// in-flight fire-and-forget saves (acceptable — the host tier is a cache)
/// and stops the P2P serving tasks; peers degrade to their own local
/// prefill. In-flight [`OffloadEngine::flush_saves_then`] barriers are
/// cancelled too, dropping their `then` callbacks unrun.
pub struct OffloadHost {
    engine: Arc<PegaEngine>,
    runtime: Runtime,
    /// `Some` when P2P is on: resolves the P2P serving tasks (gRPC transfer
    /// service + transfer-lock GC) on drop.
    #[allow(dead_code)]
    p2p_shutdown: Option<oneshot::Sender<()>>,
}

impl OffloadHost {
    pub fn new(config: HostConfig) -> Result<Arc<Self>, EngineError> {
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
                config.use_hugepages,
                storage_config,
            )?)
        };

        // P2P serving side: peers discovered us via the MetaServer and dial
        // `advertise_addr` for the RDMA handshake + block queries. Same
        // lifecycle as the engine — shut down (via the oneshot) on drop.
        let p2p_shutdown = match config.p2p {
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

                // Background GC, mirroring pegaflow-server's task. Two sweeps:
                // expired transfer locks (a crashed peer must not pin our
                // blocks past the lock timeout) and stale prefetch state — an
                // abandoned remote fetch (request dropped mid-RemoteFetch, or
                // the executor's re-query deadline fired) leaves an orphaned
                // entry whose completed task pins its fetched blocks in the
                // pinned pool until this sweep drops it.
                let gc_engine = Arc::clone(&engine);
                runtime.spawn(async move {
                    const STALE_MAX_AGE: std::time::Duration = std::time::Duration::from_mins(5);
                    let mut tick = tokio::time::interval(std::time::Duration::from_mins(1));
                    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        tick.tick().await;
                        let expired = gc_engine.gc_expired_transfer_locks();
                        if expired > 0 {
                            log::warn!("P2P GC released {expired} expired transfer locks");
                        }
                        let (stale, failed) = gc_engine
                            .gc_stale_inflight(STALE_MAX_AGE, STALE_MAX_AGE)
                            .await;
                        if stale > 0 || failed > 0 {
                            log::info!(
                                "P2P GC dropped {stale} stale prefetch entries, \
                                 {failed} failed-remote markers"
                            );
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

        Ok(Arc::new(Self {
            engine,
            runtime,
            p2p_shutdown,
        }))
    }
}

/// In-process bridge from one rank's GPU KV cache to pegaflow's offload
/// tiers, over a shared or private [`OffloadHost`].
///
/// Save is best-effort fire-and-forget (a lost save only forfeits a future
/// hit, never inference correctness); saves that must survive a handoff
/// (eviction) use the synchronous [`Self::save_blocking`]. Runtime and P2P
/// lifetime live on the host — see [`OffloadHost`] for drop semantics.
pub struct OffloadEngine {
    host: Arc<OffloadHost>,
    instance_id: String,
    device_id: i32,
    /// Owned per-layer names; load borrows these as `&[&str]`.
    layer_names: Vec<String>,
    /// In-flight fire-and-forget save tasks plus the completion signal of the
    /// latest flush barrier. One lock so a barrier's "drain handles + chain
    /// behind the previous barrier" is atomic — two racing barriers can never
    /// each take half the coverage (see [`Self::flush_saves_then`]).
    write_barrier: Mutex<WriteBarrierState>,
}

/// Save handles and barrier chain behind [`OffloadEngine::write_barrier`].
struct WriteBarrierState {
    /// In-flight fire-and-forget save tasks; finished handles are pruned on
    /// each [`OffloadEngine::save`].
    pending_saves: Vec<JoinHandle<()>>,
    /// Completion signal of the latest spawned flush barrier (fires on
    /// success and deadline alike). `None` before the first barrier.
    prev_flush_done: Option<oneshot::Receiver<()>>,
}

impl OffloadEngine {
    /// Build a private host and register `buffer` as the GPU side of the
    /// offload.
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
        let reg = Registration::from_buffer(buffer, stream);
        let host = OffloadHost::new(config.host())?;
        Self::register(
            host,
            config.instance_id,
            &config.namespace,
            config.device_id,
            reg,
            false,
        )
    }

    /// Build the engine over explicit arenas (instead of one fused
    /// [`KvBuffer`]) onto an existing shared host: this rank becomes
    /// one more pegaflow instance over the host's single pool. Ranks that
    /// should see each other's blocks pass the same `namespace`. Every
    /// arena's device allocation must stay live and pointer-stable for the
    /// engine's lifetime (the registration bakes raw device addresses), and
    /// all arenas must be indexed by the same pool block ids.
    /// `page_first` must match how the namespace's writer stores blocks: the
    /// vLLM connector stores MLA-model blocks page-first (all layers of a
    /// block concatenated into one host page, offsets by lexicographic layer
    /// name), so joining a vLLM MLA namespace requires `true` — with layer
    /// names and per-layer block bytes identical to the writer's.
    pub fn with_arenas_on(
        host: Arc<OffloadHost>,
        instance_id: impl Into<String>,
        namespace: &str,
        device_id: i32,
        arenas: &[KvArena],
        page_first: bool,
    ) -> Result<Self, EngineError> {
        Self::register(
            host,
            instance_id.into(),
            namespace,
            device_id,
            Registration::from_arenas(arenas),
            page_first,
        )
    }

    fn register(
        host: Arc<OffloadHost>,
        instance_id: String,
        namespace: &str,
        device_id: i32,
        reg: Registration,
        page_first: bool,
    ) -> Result<Self, EngineError> {
        host.engine.register_context_layer_batch_strided(
            &instance_id,
            namespace,
            device_id,
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
            // Direct (cuMemcpyAsync on the DMA engines). The Kernel backend
            // was A/B'd for the fragmented bulk-restore batches (#704) and
            // measured WORSE for co-resident decode: its grid-strided copy
            // kernels compete for SMs with decode kernels, stretching the
            // stall (two ~110ms waves vs Direct's one).
            TransferMode::Direct,
            // Layer-first (false): one pegaflow layer per model layer, the
            // page-interleaved gap expressed via `block_stride_bytes` — the
            // native openinfer layout. Page-first (true) instead stores each
            // block as one host page holding every layer at its
            // name-sorted offset; used only to join a namespace whose writer
            // (the vLLM connector on MLA models) stores blocks that way.
            page_first,
        )?;

        Ok(Self {
            host,
            instance_id,
            device_id,
            layer_names: reg.layer_names,
            write_barrier: Mutex::new(WriteBarrierState {
                pending_saves: Vec::new(),
                prev_flush_done: None,
            }),
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
        let engine = Arc::clone(&self.host.engine);
        let instance_id = self.instance_id.clone();
        let device_id = self.device_id;
        let handle = self.host.runtime.spawn(async move {
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
        // Track for the flush barrier; prune the ones that already settled so
        // the list stays bounded by the genuinely in-flight saves.
        let mut barrier = self.write_barrier.lock().expect("write_barrier poisoned");
        barrier.pending_saves.retain(|h| !h.is_finished());
        barrier.pending_saves.push(handle);
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
        self.host
            .runtime
            .block_on(self.host.engine.batch_save_kv_blocks_from_ipc(
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
        let status =
            self.host
                .runtime
                .block_on(self.host.engine.count_prefix_hit_blocks_with_prefetch(
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
                let lease = self
                    .host
                    .engine
                    .create_query_lease(&self.instance_id, blocks)?;
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
        let rx = self.host.engine.batch_load_kv_blocks_multi_layer_inproc(
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
        self.host.engine.release_query_lease(&lease);
    }

    /// Blocking form of [`Self::flush_saves_then`], for tests and eviction
    /// handoff on synchronous threads. Bounded by the same [`FLUSH_DEADLINE`]
    /// chain.
    pub fn flush_saves(&self) {
        assert_outside_runtime("flush_saves");
        let (tx, rx) = oneshot::channel();
        self.flush_saves_then(move || {
            let _ = tx.send(());
        });
        // block_on (not a bare channel wait) keeps the call-from-a-runtime
        // misuse a loud tokio panic instead of a silently deadlocked worker.
        let _ = self.host.runtime.block_on(rx);
    }

    /// Barrier the save pipeline, then call `then` — without blocking the
    /// caller. Once `then` runs, a following [`Self::query`] (local or from a
    /// P2P peer) observes every block saved before this call: this is the P/D
    /// KV-ready signal, where the prefill node withholds a request's
    /// `Finished` event until its KV is peer-visible.
    ///
    /// The barrier first awaits every in-flight fire-and-forget [`Self::save`]
    /// (their D2H copy + write-pipeline submit), then drains the write
    /// pipeline, then waits for the queued MetaServer registrations to be
    /// delivered (or dropped after a failed attempt — registration stays
    /// best-effort, the barrier only bounds *when* delivery is attempted,
    /// never *whether* it succeeds; a peer that misses a registration
    /// degrades to recompute). Without P2P the last step is a no-op.
    ///
    /// Barriers chain: each first awaits the previous barrier's completion,
    /// so — as long as no barrier in the chain hit its deadline — it
    /// transitively covers every save submitted before its own call,
    /// including handles an earlier barrier drained whose D2H had not yet
    /// submitted into the write pipeline (e.g. a chunked prefill's early
    /// chunks flushed by another request's finish). Without the chain, the
    /// pipeline drain cannot see such saves and the barrier would falsely
    /// report them visible. A predecessor that timed out may leave its
    /// drained handles permanently uncovered; that is the same accepted
    /// degradation as the deadline itself — peers recompute.
    ///
    /// Each barrier is capped at [`FLUSH_DEADLINE`] (the wait on the
    /// predecessor counts against it, and the predecessor is itself capped,
    /// so delays never accumulate): a stalled MetaServer connection degrades
    /// to "registrations still in flight" — semantically the same as a
    /// dropped registration — and `then` still runs.
    pub fn flush_saves_then(&self, then: impl FnOnce() + Send + 'static) {
        let (done_tx, done_rx) = oneshot::channel();
        let (handles, prev_done) = {
            let mut barrier = self.write_barrier.lock().expect("write_barrier poisoned");
            let handles = std::mem::take(&mut barrier.pending_saves);
            let prev_done = barrier.prev_flush_done.replace(done_rx);
            (handles, prev_done)
        };
        let engine = Arc::clone(&self.host.engine);
        self.host.runtime.spawn(async move {
            let flushed = tokio::time::timeout(FLUSH_DEADLINE, async {
                if let Some(prev) = prev_done {
                    // A cancelled predecessor (runtime teardown) resolves as
                    // an error immediately — don't let it stall the chain.
                    let _ = prev.await;
                }
                for handle in handles {
                    let _ = handle.await;
                }
                engine.flush_saves_and_registrations().await;
            })
            .await;
            if flushed.is_err() {
                log::warn!(
                    "KV offload flush timed out after {FLUSH_DEADLINE:?}; \
                     saves/registrations still in flight (peers may recompute)"
                );
            }
            let _ = done_tx.send(());
            then();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GLM5.2 shape: two arenas per model layer (MLA latent + index-K
    /// sidecar) with different copy units, sharing pool block ids. Pins the
    /// per-arena mapping and the exact strided-reach size bound pegaflow
    /// validates copies against.
    #[test]
    fn arena_registration_geometry() {
        const MLA: usize = 656 * 64;
        const IDXK: usize = 132 * 64;
        let arenas = [
            KvArena {
                name: "0.mla".into(),
                base_ptr: 0x1000,
                num_blocks: 10,
                bytes_per_block: MLA,
                block_stride_bytes: MLA,
            },
            KvArena {
                name: "0.idxk".into(),
                base_ptr: 0x9000,
                num_blocks: 10,
                bytes_per_block: IDXK,
                block_stride_bytes: IDXK,
            },
        ];
        let reg = Registration::from_arenas(&arenas);
        assert_eq!(reg.layer_names, ["0.mla", "0.idxk"]);
        assert_eq!(reg.data_ptrs, [0x1000, 0x9000]);
        assert_eq!(reg.segments, [1, 1]);
        assert_eq!(reg.kv_stride_bytes, [0, 0]);
        assert_eq!(reg.num_blocks, [10, 10]);
        assert_eq!(reg.bytes_per_block, [MLA, IDXK]);
        assert_eq!(reg.block_stride_bytes, [MLA, IDXK]);
        assert_eq!(reg.size_bytes, [10 * MLA, 10 * IDXK]);
    }

    /// A page-interleaved arena (the qwen3 fused layout expressed as arenas):
    /// stride exceeds the copy unit, and the size bound is the reach of the
    /// last block, not `num_blocks * stride`.
    #[test]
    fn interleaved_arena_size_is_last_block_reach() {
        let reg = Registration::from_arenas(&[KvArena {
            name: "3".into(),
            base_ptr: 0x100,
            num_blocks: 4,
            bytes_per_block: 512,
            block_stride_bytes: 4096,
        }]);
        assert_eq!(reg.size_bytes, [3 * 4096 + 512]);
    }

    #[test]
    #[should_panic(expected = "overruns its block stride")]
    fn arena_copy_unit_must_fit_its_stride() {
        let _ = Registration::from_arenas(&[KvArena {
            name: "bad".into(),
            base_ptr: 0,
            num_blocks: 1,
            bytes_per_block: 4096,
            block_stride_bytes: 512,
        }]);
    }
}
