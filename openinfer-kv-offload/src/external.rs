use pegaflow_core::EngineError;
use pegaflow_core::LayerSave;
use pegaflow_proto::proto::engine::FlushRequest;
use pegaflow_proto::proto::engine::HealthRequest;
use pegaflow_proto::proto::engine::LeaseLoad;
use pegaflow_proto::proto::engine::LoadRequest;
use pegaflow_proto::proto::engine::NativeKvTensor;
use pegaflow_proto::proto::engine::QueryRequest;
use pegaflow_proto::proto::engine::RegisterContextRequest;
use pegaflow_proto::proto::engine::ReleaseRequest;
use pegaflow_proto::proto::engine::ResponseStatus;
use pegaflow_proto::proto::engine::SaveLayer;
use pegaflow_proto::proto::engine::SaveRequest;
use pegaflow_proto::proto::engine::SessionRequest;
use pegaflow_proto::proto::engine::TransferMode;
use pegaflow_proto::proto::engine::UnregisterRequest;
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::query_response;
use tonic::Request;
use tonic::Streaming;
use tonic::transport::Channel;
use tonic::transport::Endpoint;

const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
const CONNECT_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);
const RPC_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
/// Save/load move data through the server's GPU/host pipelines; under load they
/// legitimately take much longer than control RPCs, and hitting this deadline
/// is fatal (see `abort_server_timeout`). Keep it far above worst-case D2H/H2D.
const DATA_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);
/// Byte length of a serialized `CUipcMemHandle`.
const IPC_HANDLE_BYTES: usize = 64;

pub(super) struct ExternalRegistration<'a> {
    pub instance_id: &'a str,
    pub namespace: &'a str,
    pub device_id: i32,
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub tp_size: usize,
    pub world_size: usize,
    pub layer_names: &'a [String],
    /// Per-layer view offsets into the server-allocated arena.
    pub offset_bytes: &'a [u64],
    pub size_bytes: &'a [usize],
    pub num_blocks: &'a [usize],
    pub bytes_per_block: &'a [usize],
    pub kv_stride_bytes: &'a [usize],
    pub block_stride_bytes: &'a [usize],
    pub segments: &'a [usize],
    pub page_first: bool,
    /// Total arena size the server allocates; every layer view fits inside.
    pub alloc_size: usize,
}

pub(super) enum ExternalQuery {
    Loading,
    Ready { num_blocks: usize, lease: Vec<u8> },
}

#[derive(Clone)]
pub(super) struct ExternalClient {
    client: EngineClient<Channel>,
}

pub(super) struct ExternalSession {
    task: tokio::task::JoinHandle<()>,
}

impl Drop for ExternalSession {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl ExternalClient {
    pub(super) async fn connect(server_addr: &str) -> Result<Self, EngineError> {
        let endpoint = Endpoint::from_shared(server_addr.to_string()).map_err(|err| {
            EngineError::InvalidArgument(format!(
                "invalid external PegaFlow server address {server_addr:?}: {err}"
            ))
        })?;
        let channel = endpoint
            .connect_timeout(CONNECT_DEADLINE)
            .connect()
            .await
            .map_err(|err| {
                EngineError::Storage(format!(
                    "connect external PegaFlow server {server_addr}: {err}"
                ))
            })?;
        let mut client = EngineClient::new(channel)
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
        let response = client
            .health(deadline_request(HealthRequest {}))
            .await
            .map_err(|err| rpc_error("health", &err))?
            .into_inner();
        require_ok("health", response.status)?;
        Ok(Self { client })
    }

    /// Register the KV layout. The server allocates the arena and returns its
    /// CUDA IPC handle.
    pub(super) async fn register(
        &self,
        registration: ExternalRegistration<'_>,
    ) -> Result<(ExternalSession, Vec<u8>), EngineError> {
        let native_kv_tensors = build_native_kv_tensors(
            registration.offset_bytes,
            registration.size_bytes,
            registration.block_stride_bytes,
        )?;
        let tp_rank = as_u32(registration.tp_rank, "tp_rank")?;
        let pp_rank = as_u32(registration.pp_rank, "pp_rank")?;
        let tp_size = as_u32(registration.tp_size, "tp_size")?;
        let world_size = as_u32(registration.world_size, "world_size")?;

        let mut session_client = self.client.clone();
        let session = tokio::time::timeout(
            RPC_DEADLINE,
            session_client.session(SessionRequest {
                instance_id: registration.instance_id.to_string(),
                namespace: registration.namespace.to_string(),
                tp_size,
                world_size,
            }),
        )
        .await
        .map_err(|_| EngineError::Storage("external PegaFlow session RPC timed out".into()))?
        .map_err(|err| rpc_error("session", &err))?;
        let session = ExternalSession {
            task: tokio::spawn(watch_session(session.into_inner())),
        };

        let request = RegisterContextRequest {
            instance_id: registration.instance_id.to_string(),
            namespace: registration.namespace.to_string(),
            tp_rank,
            tp_size,
            world_size,
            device_id: registration.device_id,
            layer_names: registration.layer_names.to_vec(),
            wrapper_bytes: Vec::new(),
            num_blocks: registration
                .num_blocks
                .iter()
                .map(|&value| as_u64(value, "num_blocks"))
                .collect::<Result<_, _>>()?,
            bytes_per_block: registration
                .bytes_per_block
                .iter()
                .map(|&value| as_u64(value, "bytes_per_block"))
                .collect::<Result<_, _>>()?,
            kv_stride_bytes: registration
                .kv_stride_bytes
                .iter()
                .map(|&value| as_u64(value, "kv_stride_bytes"))
                .collect::<Result<_, _>>()?,
            segments: registration
                .segments
                .iter()
                .map(|&value| as_u32(value, "segments"))
                .collect::<Result<_, _>>()?,
            pp_rank,
            client_version: pegaflow_proto::VERSION.to_string(),
            transfer_mode: TransferMode::Direct as i32,
            page_first: registration.page_first,
            native_kv_tensors,
            native_alloc_size: as_u64(registration.alloc_size, "alloc_size")?,
        };
        let mut client = self.client.clone();
        let response = match tokio::time::timeout(
            RPC_DEADLINE,
            client.register_context_batch(Request::new(request)),
        )
        .await
        {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(err)) => return Err(rpc_error("register_context_batch", &err)),
            Err(_) => {
                return Err(EngineError::Storage(
                    "external PegaFlow register_context_batch timed out".into(),
                ));
            }
        };
        require_ok("register_context_batch", response.status)?;
        if response.arena_ipc_handle.len() != IPC_HANDLE_BYTES {
            return Err(EngineError::Storage(format!(
                "register_context_batch returned a {}-byte arena handle, expected \
                 {IPC_HANDLE_BYTES} (is the server running the native-arena build?)",
                response.arena_ipc_handle.len()
            )));
        }
        Ok((session, response.arena_ipc_handle))
    }

    pub(super) async fn save(
        &self,
        instance_id: &str,
        tp_rank: usize,
        pp_rank: usize,
        device_id: i32,
        saves: Vec<LayerSave>,
    ) -> Result<(), EngineError> {
        let saves = saves
            .into_iter()
            .map(|save| {
                Ok(SaveLayer {
                    layer_name: save.layer_name,
                    block_ids: save
                        .block_ids
                        .into_iter()
                        .map(|id| as_u32(id, "block_id"))
                        .collect::<Result<_, EngineError>>()?,
                    block_hashes: save.block_hashes,
                })
            })
            .collect::<Result<_, EngineError>>()?;
        let mut client = self.client.clone();
        let response = match tokio::time::timeout(
            DATA_DEADLINE,
            client.save(Request::new(SaveRequest {
                instance_id: instance_id.to_string(),
                tp_rank: as_u32(tp_rank, "tp_rank")?,
                device_id,
                saves,
                pp_rank: as_u32(pp_rank, "pp_rank")?,
            })),
        )
        .await
        {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(err)) => abort_lost_server("save", &err),
            Err(_) => abort_server_timeout("save"),
        };
        require_ok("save", response.status)
    }

    pub(super) async fn query(
        &self,
        instance_id: &str,
        req_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<ExternalQuery, EngineError> {
        let mut client = self.client.clone();
        let response = client
            .query_prefetch(deadline_request(QueryRequest {
                instance_id: instance_id.to_string(),
                block_hashes: block_hashes.to_vec(),
                req_id: req_id.to_string(),
                wait_for_full_prefix: false,
            }))
            .await
            .map_err(|err| rpc_error("query_prefetch", &err))?
            .into_inner();
        match response.outcome {
            Some(query_response::Outcome::Loading(_)) => Ok(ExternalQuery::Loading),
            Some(query_response::Outcome::Ready(ready)) => {
                let num_blocks = usize::try_from(ready.num_hit_blocks).map_err(|_| {
                    EngineError::Storage(format!(
                        "query hit count {} does not fit usize",
                        ready.num_hit_blocks
                    ))
                })?;
                if num_blocks > block_hashes.len() {
                    return Err(EngineError::Storage(format!(
                        "query returned {num_blocks} blocks for {} requested hashes",
                        block_hashes.len()
                    )));
                }
                if (num_blocks == 0) != ready.lease.is_empty() {
                    return Err(EngineError::Storage(format!(
                        "query returned inconsistent hit/lease: blocks={num_blocks}, lease_bytes={}",
                        ready.lease.len()
                    )));
                }
                Ok(ExternalQuery::Ready {
                    num_blocks,
                    lease: ready.lease,
                })
            }
            None => Err(EngineError::Storage(
                "query_prefetch response omitted outcome".to_string(),
            )),
        }
    }

    pub(super) async fn load(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_names: &[String],
        lease: Vec<u8>,
        block_ids: Vec<usize>,
    ) -> Result<(), EngineError> {
        let mut client = self.client.clone();
        let response = match tokio::time::timeout(
            DATA_DEADLINE,
            client.load(Request::new(LoadRequest {
                instance_id: instance_id.to_string(),
                tp_rank: as_u32(tp_rank, "tp_rank")?,
                device_id,
                load_state_shm: String::new(),
                layer_names: layer_names.to_vec(),
                loads: vec![LeaseLoad {
                    lease,
                    block_ids: block_ids
                        .into_iter()
                        .map(|id| as_u32(id, "block_id"))
                        .collect::<Result<_, _>>()?,
                }],
                wait_for_completion: true,
            })),
        )
        .await
        {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(err)) => abort_lost_server("load", &err),
            Err(_) => abort_server_timeout("load"),
        };
        require_ok("load", response.status)
    }

    pub(super) async fn release(&self, lease: Vec<u8>) -> Result<(), EngineError> {
        let mut client = self.client.clone();
        client
            .release(deadline_request(ReleaseRequest { lease }))
            .await
            .map_err(|err| rpc_error("release", &err))?;
        Ok(())
    }

    /// Server-wide durability barrier: `Flush` has no instance scope, so on a
    /// shared server this also waits out other instances' save tails.
    pub(super) async fn flush(&self) -> Result<(), EngineError> {
        let mut client = self.client.clone();
        let response = client
            .flush(deadline_request(FlushRequest {}))
            .await
            .map_err(|err| rpc_error("flush", &err))?
            .into_inner();
        require_ok("flush", response.status)
    }

    /// Ensure the instance is gone server-side. "Not found" counts as
    /// success: dropping the liveness stream just before this call may have
    /// already triggered the server's session cleanup for the same instance.
    pub(super) async fn unregister(&self, instance_id: &str) -> Result<(), EngineError> {
        let mut client = self.client.clone();
        let response = match tokio::time::timeout(
            RPC_DEADLINE,
            client.unregister_context(Request::new(UnregisterRequest {
                instance_id: instance_id.to_string(),
            })),
        )
        .await
        {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(err))
                if matches!(
                    err.code(),
                    tonic::Code::FailedPrecondition | tonic::Code::NotFound
                ) =>
            {
                log::debug!(
                    "unregister_context: instance {instance_id} already cleaned up ({err})"
                );
                return Ok(());
            }
            Ok(Err(err)) => return Err(rpc_error("unregister_context", &err)),
            Err(_) => abort_server_timeout("unregister_context"),
        };
        require_ok("unregister_context", response.status)
    }
}

/// The liveness stream doubles as a dead-server detector. The server owns the
/// KV arena this process has mapped; if the server goes away the mapping is
/// backed by freed memory, so the only safe reaction is to exit. There is no
/// reconnect.
async fn watch_session(mut stream: Streaming<pegaflow_proto::proto::engine::SessionEvent>) {
    loop {
        match stream.message().await {
            Ok(Some(_)) => {}
            Ok(None) => {
                log::error!(
                    "external PegaFlow session closed by server; the imported KV arena is gone, \
                     exiting"
                );
                std::process::abort();
            }
            Err(err) => {
                log::error!(
                    "external PegaFlow session failed: {err}; the imported KV arena is gone, \
                     exiting"
                );
                std::process::abort();
            }
        }
    }
}

fn deadline_request<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.set_timeout(RPC_DEADLINE);
    request
}

fn abort_lost_server(operation: &str, err: &tonic::Status) -> ! {
    log::error!(
        "external PegaFlow {operation} RPC failed mid-transfer: {err}; the server owns the KV \
         arena this process has mapped, exiting"
    );
    std::process::abort();
}

fn abort_server_timeout(operation: &str) -> ! {
    log::error!(
        "external PegaFlow {operation} did not respond in time; the server owns this process's \
         KV arena, so its state is unknowable — exiting"
    );
    std::process::abort();
}

/// Build per-layer strided views for the register RPC from explicit offsets
/// into the server-allocated arena.
fn build_native_kv_tensors(
    offset_bytes: &[u64],
    size_bytes: &[usize],
    block_stride_bytes: &[usize],
) -> Result<Vec<NativeKvTensor>, EngineError> {
    if offset_bytes.is_empty()
        || offset_bytes.len() != size_bytes.len()
        || offset_bytes.len() != block_stride_bytes.len()
    {
        return Err(EngineError::InvalidArgument(format!(
            "native layer metadata length mismatch: offsets={}, sizes={}, block strides={}",
            offset_bytes.len(),
            size_bytes.len(),
            block_stride_bytes.len()
        )));
    }
    offset_bytes
        .iter()
        .copied()
        .zip(size_bytes.iter().copied())
        .zip(block_stride_bytes.iter().copied())
        .map(|((offset, view_size), block_stride)| {
            Ok(NativeKvTensor {
                offset_bytes: offset,
                size_bytes: as_u64(view_size, "size_bytes")?,
                block_stride_bytes: as_u64(block_stride, "block_stride_bytes")?,
            })
        })
        .collect()
}

fn require_ok(operation: &str, status: Option<ResponseStatus>) -> Result<(), EngineError> {
    let status = status
        .ok_or_else(|| EngineError::Storage(format!("{operation} response omitted status")))?;
    if status.ok {
        Ok(())
    } else {
        Err(EngineError::Storage(format!(
            "{operation} failed: {}",
            status.message
        )))
    }
}

fn rpc_error(operation: &str, err: &tonic::Status) -> EngineError {
    EngineError::Storage(format!("external PegaFlow {operation} RPC: {err}"))
}

fn as_u32(value: usize, field: &str) -> Result<u32, EngineError> {
    u32::try_from(value)
        .map_err(|_| EngineError::InvalidArgument(format!("{field}={value} does not fit u32")))
}

fn as_u64(value: usize, field: &str) -> Result<u64, EngineError> {
    u64::try_from(value)
        .map_err(|_| EngineError::InvalidArgument(format!("{field}={value} does not fit u64")))
}
