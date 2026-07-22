use std::os::fd::{AsRawFd, BorrowedFd};

use pegaflow_core::{EngineError, LayerSave};
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::{
    FlushRequest, HealthRequest, LeaseLoad, LoadRequest, NativeKvTensor, QueryRequest,
    RegisterContextRequest, ReleaseRequest, ResponseStatus, SaveLayer, SaveRequest, SessionRequest,
    TransferMode, UnregisterRequest, query_response,
};
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Streaming};

const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
const CONNECT_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);
const RPC_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

pub(super) struct ExternalRegistration<'a> {
    pub instance_id: &'a str,
    pub namespace: &'a str,
    pub device_id: i32,
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub tp_size: usize,
    pub world_size: usize,
    pub layer_names: &'a [String],
    pub data_ptrs: &'a [u64],
    pub size_bytes: &'a [usize],
    pub num_blocks: &'a [usize],
    pub bytes_per_block: &'a [usize],
    pub kv_stride_bytes: &'a [usize],
    pub block_stride_bytes: &'a [usize],
    pub segments: &'a [usize],
    pub page_first: bool,
    /// Exported POSIX fd for the fused VMM allocation. Sent to the server over
    /// its fd side-channel before the register RPC. `None` is a hard error for
    /// native registration (there is no other way to share the allocation).
    pub export_fd: Option<std::os::fd::BorrowedFd<'a>>,
    /// Byte size of that allocation (granularity-aligned). The server reserves
    /// and maps exactly this. `None` alongside `export_fd` is an error.
    pub alloc_size: Option<usize>,
}

pub(super) enum ExternalQuery {
    Loading,
    Ready { num_blocks: usize, lease: Vec<u8> },
}

#[derive(Clone)]
pub(super) struct ExternalClient {
    client: EngineClient<Channel>,
    /// Server's fd side-channel path, learned from the Health response. Native
    /// VMM registration sends the exported allocation fd here before the gRPC
    /// register RPC. Empty if the server has no local fd channel.
    fd_socket_path: String,
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
        Ok(Self {
            client,
            fd_socket_path: response.fd_socket_path,
        })
    }

    pub(super) async fn register(
        &self,
        registration: ExternalRegistration<'_>,
    ) -> Result<ExternalSession, EngineError> {
        // Build the per-layer strided views into the fused VMM allocation. The
        // backing fd travels out-of-band (below), not in this metadata.
        let native_kv_tensors = build_native_kv_tensors(
            registration.data_ptrs,
            registration.size_bytes,
            registration.block_stride_bytes,
        )?;
        let tp_rank = as_u32(registration.tp_rank, "tp_rank")?;
        let pp_rank = as_u32(registration.pp_rank, "pp_rank")?;
        let tp_size = as_u32(registration.tp_size, "tp_size")?;
        let world_size = as_u32(registration.world_size, "world_size")?;

        // Send the allocation fd on the server's side-channel BEFORE the
        // register RPC, so the fd is waiting when the handler claims it. A VMM
        // allocation cannot be shared any other way — fail loudly if either the
        // fd or the server's channel path is missing.
        let export_fd = registration.export_fd.ok_or_else(|| {
            EngineError::InvalidArgument(
                "native VMM registration requires an exported allocation fd".into(),
            )
        })?;
        let alloc_size = registration.alloc_size.ok_or_else(|| {
            EngineError::InvalidArgument(
                "native VMM registration requires the allocation size".into(),
            )
        })?;
        if self.fd_socket_path.is_empty() {
            return Err(EngineError::Storage(
                "external PegaFlow server did not advertise an fd side-channel path".into(),
            ));
        }
        send_export_fd(
            &self.fd_socket_path,
            registration.instance_id,
            registration.device_id,
            export_fd,
        )
        .await?;

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
            native_alloc_size: as_u64(alloc_size, "alloc_size")?,
        };
        let mut client = self.client.clone();
        let response = match tokio::time::timeout(
            RPC_DEADLINE,
            client.register_context_batch(Request::new(request)),
        )
        .await
        {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(err)) => {
                let register_error = rpc_error("register_context_batch", &err);
                if let Err(cleanup_error) = self.unregister(registration.instance_id).await {
                    abort_cleanup_failure("failed registration", &cleanup_error);
                }
                return Err(register_error);
            }
            Err(_) => abort_ownership_timeout("register_context_batch"),
        };
        if let Err(register_error) = require_ok("register_context_batch", response.status) {
            if let Err(cleanup_error) = self.unregister(registration.instance_id).await {
                abort_cleanup_failure("rejected registration", &cleanup_error);
            }
            return Err(register_error);
        }
        Ok(session)
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
            RPC_DEADLINE,
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
            Ok(Err(err)) => abort_indeterminate_transfer("save", &err),
            Err(_) => abort_ownership_timeout("save"),
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
            RPC_DEADLINE,
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
            Ok(Err(err)) => abort_indeterminate_transfer("load", &err),
            Err(_) => abort_ownership_timeout("load"),
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

    pub(super) async fn flush(&self) -> Result<(), EngineError> {
        let mut client = self.client.clone();
        let response = client
            .flush(deadline_request(FlushRequest {}))
            .await
            .map_err(|err| rpc_error("flush", &err))?
            .into_inner();
        require_ok("flush", response.status)
    }

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
            Ok(Err(err)) => return Err(rpc_error("unregister_context", &err)),
            Err(_) => abort_ownership_timeout("unregister_context"),
        };
        require_ok("unregister_context", response.status)
    }
}

async fn watch_session(mut stream: Streaming<pegaflow_proto::proto::engine::SessionEvent>) {
    loop {
        match stream.message().await {
            Ok(Some(_)) => {}
            Ok(None) => {
                log::error!("external PegaFlow session closed by server");
                break;
            }
            Err(err) => {
                log::error!("external PegaFlow session failed: {err}");
                break;
            }
        }
    }
}

fn deadline_request<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.set_timeout(RPC_DEADLINE);
    request
}

fn abort_indeterminate_transfer(operation: &str, err: &tonic::Status) -> ! {
    log::error!(
        "external PegaFlow {operation} RPC ended before the server acknowledged DMA completion: \
         {err}; aborting to keep exported CUDA memory from being reused"
    );
    std::process::abort();
}

fn abort_ownership_timeout(operation: &str) -> ! {
    log::error!(
        "external PegaFlow {operation} did not acknowledge CUDA ownership within \
         {RPC_DEADLINE:?}; aborting before exported memory can be reused"
    );
    std::process::abort();
}

fn abort_cleanup_failure(context: &str, err: &EngineError) -> ! {
    log::error!(
        "external PegaFlow cleanup after {context} failed: {err}; aborting before exported CUDA \
         memory can be released"
    );
    std::process::abort();
}

/// Build per-layer strided views for the register RPC. Each view's
/// `offset_bytes` is relative to the fused allocation base (the smallest layer
/// pointer), matching how the server offsets into its imported mapping.
fn build_native_kv_tensors(
    data_ptrs: &[u64],
    size_bytes: &[usize],
    block_stride_bytes: &[usize],
) -> Result<Vec<NativeKvTensor>, EngineError> {
    if data_ptrs.len() != size_bytes.len() || data_ptrs.len() != block_stride_bytes.len() {
        return Err(EngineError::InvalidArgument(format!(
            "native export metadata length mismatch: pointers={}, sizes={}, block strides={}",
            data_ptrs.len(),
            size_bytes.len(),
            block_stride_bytes.len()
        )));
    }
    // The allocation base is the smallest layer pointer; page-first layout lays
    // every layer out at base + layer_index * layer_bytes.
    let base = data_ptrs
        .iter()
        .copied()
        .min()
        .ok_or_else(|| EngineError::InvalidArgument("native export has no arenas".into()))?;
    data_ptrs
        .iter()
        .copied()
        .zip(size_bytes.iter().copied())
        .zip(block_stride_bytes.iter().copied())
        .map(|((data_ptr, view_size), block_stride)| {
            let offset = data_ptr.checked_sub(base).ok_or_else(|| {
                EngineError::Storage(format!(
                    "layer pointer {data_ptr:#x} precedes allocation base {base:#x}"
                ))
            })?;
            Ok(NativeKvTensor {
                offset_bytes: offset,
                size_bytes: as_u64(view_size, "size_bytes")?,
                block_stride_bytes: as_u64(block_stride, "block_stride_bytes")?,
            })
        })
        .collect()
}

/// Send `fd` to the server's fd side-channel at `socket_path`, tagged with
/// `instance_id\0device_id` so the register handler can claim it by key. Uses a
/// blocking `sendmsg`/SCM_RIGHTS on a std Unix socket (fd passing has no async
/// wrapper); the payload is one small datagram, so this does not block long.
async fn send_export_fd(
    socket_path: &str,
    instance_id: &str,
    device_id: i32,
    fd: BorrowedFd<'_>,
) -> Result<(), EngineError> {
    let socket_path = socket_path.to_string();
    let instance_id = instance_id.to_string();
    let raw_fd = fd.as_raw_fd();
    // Run the blocking syscall off the async worker. `raw_fd` stays valid: the
    // caller's `BorrowedFd` outlives this await, and the server dups it.
    tokio::task::spawn_blocking(move || {
        send_fd_blocking(&socket_path, &instance_id, device_id, raw_fd)
    })
    .await
    .map_err(|err| EngineError::Storage(format!("fd side-channel task join failed: {err}")))?
}

fn send_fd_blocking(
    socket_path: &str,
    instance_id: &str,
    device_id: i32,
    raw_fd: std::os::fd::RawFd,
) -> Result<(), EngineError> {
    use std::os::unix::net::UnixStream;

    let stream = UnixStream::connect(socket_path).map_err(|err| {
        EngineError::Storage(format!("connect fd side-channel {socket_path}: {err}"))
    })?;
    let sock = stream.as_raw_fd();

    // Payload: "<instance_id>\0<device_id>".
    let payload = format!("{instance_id}\0{device_id}");
    let mut iov = libc::iovec {
        iov_base: payload.as_ptr() as *mut libc::c_void,
        iov_len: payload.len(),
    };
    let mut cmsg_space = [0u8; 64]; // CMSG_SPACE(size_of::<RawFd>()) with margin
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_space.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_space.len();

    // SAFETY: msg has a valid control buffer; we populate exactly one
    // SCM_RIGHTS header carrying one fd.
    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<std::os::fd::RawFd>() as u32) as _;
        std::ptr::write_unaligned(libc::CMSG_DATA(cmsg).cast::<std::os::fd::RawFd>(), raw_fd);
        msg.msg_controllen = (*cmsg).cmsg_len;
    }

    // SAFETY: msg is fully initialized with a valid iov and control message.
    let sent = unsafe { libc::sendmsg(sock, &msg, 0) };
    if sent < 0 {
        return Err(EngineError::Storage(format!(
            "sendmsg fd on side-channel: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
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
