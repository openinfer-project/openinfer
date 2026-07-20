//! Client-side bridge between OpenInfer GPU KV cache and a PegaFlow server.
//!
//! OpenInfer owns the GPU allocations and logical prefix-cache policy. The
//! external server owns every deeper tier and the transfer workers. This
//! module registers CUDA IPC views, then translates scheduler save/query/load
//! operations into PegaFlow RPCs.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cudarc::driver::CudaStream;
use openinfer_kv_cache::KvBuffer;
use pegaflow_core::{EngineError, LayerSave};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::external::{ExternalClient, ExternalQuery, ExternalRegistration, ExternalSession};

const TP_RANK: usize = 0;
const PP_RANK: usize = 0;
const TP_SIZE: usize = 1;
const WORLD_SIZE: usize = 1;
const ELEM_SIZE: usize = std::mem::size_of::<half::bf16>();
const FLUSH_DEADLINE: Duration = Duration::from_secs(5);

fn assert_outside_runtime(op: &str) {
    debug_assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "OffloadEngine::{op} drives the offload runtime with block_on and must be \
         called from a synchronous thread"
    );
}

/// Connection settings for one rank-level offload engine.
pub struct OffloadConfig {
    /// Base identifier for this rank. A process-local suffix is added so a
    /// restarted client cannot collide with a session still being cleaned up.
    pub instance_id: String,
    /// Content-addressing domain shared by producers and consumers whose KV
    /// bytes and block layout are interchangeable.
    pub namespace: String,
    /// CUDA device ordinal whose allocations are exported over CUDA IPC.
    pub device_id: i32,
    /// PegaFlow gRPC endpoint, for example `http://127.0.0.1:50055`.
    pub server_addr: String,
    /// Worker threads for the client runtime.
    pub runtime_threads: usize,
}

impl OffloadConfig {
    pub fn new(
        instance_id: impl Into<String>,
        device_id: i32,
        server_addr: impl Into<String>,
    ) -> Self {
        Self {
            instance_id: instance_id.into(),
            namespace: "openinfer".to_string(),
            device_id,
            server_addr: server_addr.into(),
            runtime_threads: 2,
        }
    }

    #[must_use]
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }
}

/// Opaque ownership token returned by a PegaFlow prefix query.
pub struct QueryLeaseId(Vec<u8>);

pub struct QueryHit {
    pub lease: Option<QueryLeaseId>,
    pub num_blocks: usize,
}

pub enum QueryOutcome {
    Ready(QueryHit),
    /// PegaFlow is fetching the prefix from a deeper tier. Re-query with the
    /// same request id on a later scheduler tick.
    Loading,
}

/// In-flight server-side CPU-to-GPU load.
pub struct LoadHandle {
    submission: Option<oneshot::Receiver<Result<(), EngineError>>>,
}

impl LoadHandle {
    pub fn poll(&mut self) -> Option<Result<(), EngineError>> {
        if let Some(rx) = &mut self.submission {
            match rx.try_recv() {
                Ok(Ok(())) => self.submission = None,
                Ok(Err(err)) => return Some(Err(err)),
                Err(oneshot::error::TryRecvError::Empty) => return None,
                Err(oneshot::error::TryRecvError::Closed) => {
                    return Some(Err(EngineError::Storage(
                        "external load submission dropped reply".into(),
                    )));
                }
            }
        }
        Some(Ok(()))
    }

    pub fn wait(mut self) -> Result<(), EngineError> {
        if let Some(rx) = self.submission.take() {
            rx.blocking_recv().unwrap_or_else(|_| {
                Err(EngineError::Storage(
                    "external load submission dropped reply".into(),
                ))
            })?;
        }
        Ok(())
    }
}

/// One strided GPU arena registered as one PegaFlow layer.
pub struct KvArena {
    pub name: String,
    pub base_ptr: u64,
    pub num_blocks: usize,
    pub bytes_per_block: usize,
    pub block_stride_bytes: usize,
}

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
    fn from_buffer(buffer: &KvBuffer, stream: &CudaStream) -> Self {
        let layout = buffer.layout();
        let num_blocks = buffer.num_blocks();
        let base_ptr = buffer.device_ptr(stream);
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

    fn from_arenas(arenas: &[KvArena]) -> Self {
        assert!(!arenas.is_empty(), "KV offload requires at least one arena");
        let n = arenas.len();
        let mut reg = Self {
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
            assert!(arena.num_blocks > 0, "arena {} has no blocks", arena.name);
            assert!(
                arena.bytes_per_block > 0,
                "arena {} has an empty copy unit",
                arena.name
            );
            assert!(
                arena.bytes_per_block <= arena.block_stride_bytes,
                "arena {} copy unit {} overruns its block stride {}",
                arena.name,
                arena.bytes_per_block,
                arena.block_stride_bytes
            );
            let strided_prefix = (arena.num_blocks - 1)
                .checked_mul(arena.block_stride_bytes)
                .expect("arena byte reach overflows usize");
            let size = strided_prefix
                .checked_add(arena.bytes_per_block)
                .expect("arena byte reach overflows usize");
            reg.layer_names.push(arena.name.clone());
            reg.data_ptrs.push(arena.base_ptr);
            reg.size_bytes.push(size);
            reg.num_blocks.push(arena.num_blocks);
            reg.bytes_per_block.push(arena.bytes_per_block);
            reg.block_stride_bytes.push(arena.block_stride_bytes);
        }
        reg
    }
}

/// Shared connection and runtime for rank-level engines in one process.
pub struct OffloadHost {
    client: ExternalClient,
    runtime: Runtime,
    instance_suffix: String,
}

impl OffloadHost {
    pub fn connect(server_addr: &str, runtime_threads: usize) -> Result<Arc<Self>, EngineError> {
        if server_addr.is_empty() {
            return Err(EngineError::InvalidArgument(
                "PegaFlow server address must not be empty".into(),
            ));
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(runtime_threads.max(1))
            .enable_all()
            .build()
            .map_err(|err| EngineError::Storage(format!("offload runtime build: {err}")))?;
        let client = runtime.block_on(ExternalClient::connect(server_addr))?;
        let instance_suffix = new_instance_suffix();
        Ok(Arc::new(Self {
            client,
            runtime,
            instance_suffix,
        }))
    }

    fn instance_id(&self, base: &str) -> String {
        format!("{base}-{}", self.instance_suffix)
    }
}

fn new_instance_suffix() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Rank-level client for one set of GPU KV arenas.
pub struct OffloadEngine {
    session: Option<ExternalSession>,
    host: Arc<OffloadHost>,
    instance_id: String,
    device_id: i32,
    layer_names: Vec<String>,
    write_barrier: Mutex<WriteBarrierState>,
}

struct WriteBarrierState {
    pending_saves: Vec<JoinHandle<()>>,
    prev_flush_done: Option<oneshot::Receiver<()>>,
}

impl OffloadEngine {
    pub fn new(
        config: &OffloadConfig,
        buffer: &KvBuffer,
        stream: &CudaStream,
    ) -> Result<Self, EngineError> {
        let reg = Registration::from_buffer(buffer, stream);
        let host = OffloadHost::connect(&config.server_addr, config.runtime_threads)?;
        Self::register(
            host,
            &config.instance_id,
            &config.namespace,
            config.device_id,
            reg,
            false,
        )
    }

    pub fn with_arenas(config: &OffloadConfig, arenas: &[KvArena]) -> Result<Self, EngineError> {
        let host = OffloadHost::connect(&config.server_addr, config.runtime_threads)?;
        Self::register(
            host,
            &config.instance_id,
            &config.namespace,
            config.device_id,
            Registration::from_arenas(arenas),
            false,
        )
    }

    /// Register arenas on a connection shared by several ranks in this process.
    /// `page_first` must match the producer's host layout for the namespace.
    pub fn with_arenas_on(
        host: Arc<OffloadHost>,
        instance_id: impl Into<String>,
        namespace: &str,
        device_id: i32,
        arenas: &[KvArena],
        page_first: bool,
    ) -> Result<Self, EngineError> {
        let instance_id = instance_id.into();
        Self::register(
            host,
            &instance_id,
            namespace,
            device_id,
            Registration::from_arenas(arenas),
            page_first,
        )
    }

    fn register(
        host: Arc<OffloadHost>,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        reg: Registration,
        page_first: bool,
    ) -> Result<Self, EngineError> {
        assert_outside_runtime("register");
        let instance_id = host.instance_id(instance_id);
        let session = host
            .runtime
            .block_on(host.client.register(ExternalRegistration {
                instance_id: &instance_id,
                namespace,
                device_id,
                tp_rank: TP_RANK,
                pp_rank: PP_RANK,
                tp_size: TP_SIZE,
                world_size: WORLD_SIZE,
                layer_names: &reg.layer_names,
                data_ptrs: &reg.data_ptrs,
                size_bytes: &reg.size_bytes,
                num_blocks: &reg.num_blocks,
                bytes_per_block: &reg.bytes_per_block,
                kv_stride_bytes: &reg.kv_stride_bytes,
                block_stride_bytes: &reg.block_stride_bytes,
                segments: &reg.segments,
                page_first,
            }))?;

        Ok(Self {
            session: Some(session),
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

    fn build_saves(&self, block_ids: &[i32], block_hashes: &[Vec<u8>]) -> Vec<LayerSave> {
        let block_ids: Vec<usize> = block_ids
            .iter()
            .map(|&id| usize::try_from(id).expect("KV block id must be non-negative"))
            .collect();
        self.layer_names
            .iter()
            .map(|name| LayerSave {
                layer_name: name.clone(),
                block_ids: block_ids.clone(),
                block_hashes: block_hashes.to_vec(),
            })
            .collect()
    }

    /// Save GPU blocks without blocking the scheduler. `keep_alive` pins the
    /// source blocks until the server confirms the copy submission completed.
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
        let client = self.host.client.clone();
        let instance_id = self.instance_id.clone();
        let device_id = self.device_id;
        let handle = self.host.runtime.spawn(async move {
            if let Err(err) = client
                .save(&instance_id, TP_RANK, PP_RANK, device_id, saves)
                .await
            {
                log::warn!("PegaFlow save failed (best-effort): {err}");
            }
            drop(keep_alive);
        });
        let mut barrier = self.write_barrier.lock().expect("write_barrier poisoned");
        barrier.pending_saves.retain(|handle| !handle.is_finished());
        barrier.pending_saves.push(handle);
    }

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
        self.host.runtime.block_on(self.host.client.save(
            &self.instance_id,
            TP_RANK,
            PP_RANK,
            self.device_id,
            saves,
        ))
    }

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
        match self.host.runtime.block_on(self.host.client.query(
            &self.instance_id,
            req_id,
            block_hashes,
        ))? {
            ExternalQuery::Loading => Ok(QueryOutcome::Loading),
            ExternalQuery::Ready { num_blocks, lease } => Ok(QueryOutcome::Ready(QueryHit {
                lease: (num_blocks > 0).then_some(QueryLeaseId(lease)),
                num_blocks,
            })),
        }
    }

    pub fn load(
        &self,
        lease: &QueryLeaseId,
        dst_block_ids: Vec<i32>,
    ) -> Result<LoadHandle, EngineError> {
        let dst_block_ids: Vec<usize> = dst_block_ids
            .into_iter()
            .map(|id| usize::try_from(id).expect("KV block id must be non-negative"))
            .collect();
        let instance_id = self.instance_id.clone();
        let layer_names = self.layer_names.clone();
        let client = self.host.client.clone();
        let device_id = self.device_id;
        let lease = lease.0.clone();
        let (tx, rx) = oneshot::channel();
        self.host.runtime.spawn(async move {
            let result = client
                .load(
                    &instance_id,
                    TP_RANK,
                    device_id,
                    &layer_names,
                    lease,
                    dst_block_ids,
                )
                .await;
            let _ = tx.send(result);
        });
        Ok(LoadHandle {
            submission: Some(rx),
        })
    }

    pub fn release_query_lease(&self, lease: QueryLeaseId) {
        let client = self.host.client.clone();
        self.host.runtime.spawn(async move {
            if let Err(err) = client.release(lease.0).await {
                log::warn!("PegaFlow lease release failed: {err}");
            }
        });
    }

    pub fn flush_saves(&self) {
        assert_outside_runtime("flush_saves");
        let (tx, rx) = oneshot::channel();
        self.flush_saves_then(move || {
            let _ = tx.send(());
        });
        let _ = self.host.runtime.block_on(rx);
    }

    /// Wait for all saves submitted before this call and the server visibility
    /// barrier, then invoke `then`. The deadline prevents a failed server from
    /// withholding a finished request indefinitely.
    pub fn flush_saves_then(&self, then: impl FnOnce() + Send + 'static) {
        let (done_tx, done_rx) = oneshot::channel();
        let (handles, prev_done) = {
            let mut barrier = self.write_barrier.lock().expect("write_barrier poisoned");
            let handles = std::mem::take(&mut barrier.pending_saves);
            let prev_done = barrier.prev_flush_done.replace(done_rx);
            (handles, prev_done)
        };
        let client = self.host.client.clone();
        self.host.runtime.spawn(async move {
            let flushed = tokio::time::timeout(FLUSH_DEADLINE, async {
                if let Some(prev) = prev_done {
                    let _ = prev.await;
                }
                for handle in handles {
                    let _ = handle.await;
                }
                if let Err(err) = client.flush().await {
                    log::warn!("PegaFlow flush failed: {err}");
                }
            })
            .await;
            if flushed.is_err() {
                log::warn!(
                    "KV offload flush timed out after {FLUSH_DEADLINE:?}; peers may recompute"
                );
            }
            let _ = done_tx.send(());
            then();
        });
    }

    /// Flush submitted saves, then ask the server to close this instance's
    /// imported CUDA mappings.
    pub fn shutdown(&mut self) {
        if self.session.is_none() {
            return;
        }
        assert_outside_runtime("shutdown");
        self.flush_saves();
        if let Err(err) = self
            .host
            .runtime
            .block_on(self.host.client.unregister(&self.instance_id))
        {
            log::error!(
                "PegaFlow unregister barrier failed for {}: {err}; aborting before exported CUDA \
                 memory can be released",
                self.instance_id
            );
            std::process::abort();
        }
        // The explicit unregister owns mapping cleanup. Aborting the liveness
        // stream afterwards only removes its now-empty session entry.
        self.session.take();
    }
}

impl Drop for OffloadEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_registration_geometry() {
        const MLA: usize = 656 * 64;
        const IDXK: usize = 132 * 64;
        let reg = Registration::from_arenas(&[
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
        ]);
        assert_eq!(reg.layer_names, ["0.mla", "0.idxk"]);
        assert_eq!(reg.data_ptrs, [0x1000, 0x9000]);
        assert_eq!(reg.segments, [1, 1]);
        assert_eq!(reg.kv_stride_bytes, [0, 0]);
        assert_eq!(reg.num_blocks, [10, 10]);
        assert_eq!(reg.bytes_per_block, [MLA, IDXK]);
        assert_eq!(reg.block_stride_bytes, [MLA, IDXK]);
        assert_eq!(reg.size_bytes, [10 * MLA, 10 * IDXK]);
    }

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
    fn instance_suffix_is_process_independent_uuid() {
        let a = new_instance_suffix();
        let b = new_instance_suffix();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|ch| ch.is_ascii_hexdigit()));
        assert_ne!(a, b);
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
