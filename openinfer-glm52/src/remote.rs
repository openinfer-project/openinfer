//! Cross-node rank transport (design: `docs/models/glm52/cross-node-scaling.md`).
//!
//! A dumb `rank-host` process exposes N unmodified [`Glm52RankWorker`]s over
//! ONE framed-TCP connection, and the coordinator drives them through
//! [`Glm52RemoteRankWorker`]s whose typed surface mirrors the local worker
//! exactly — `submit`-style `*_async` methods returning a bounded(1)
//! [`Receiver`]. The wire carries the SAME request payloads the local command
//! enum does (minus the response channel), so there is no second schema to
//! drift: a command's fields serialize as-is via serde/bincode.
//!
//! Framing: `len(u32 LE) ‖ bincode(frame)`, `TCP_NODELAY`. Responses are
//! strict FIFO **per worker** (each worker thread drains its queue serially,
//! so completion order == command order per rank); the demux thread matches
//! them against per-worker pending queues.
//!
//! Fail-stop: a connection reset, decode error, or handshake mismatch poisons
//! the node — every pending and future call errors, which the engine treats
//! exactly like a dead worker thread today. No retry, no reconnect, no
//! buffering.

use std::collections::VecDeque;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::net::TcpStream;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use anyhow::ensure;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use crossbeam_channel::unbounded;
use openinfer_kv_offload::KvArena;

use crate::Glm52MoeTopo;
use crate::dspark::GLM52_DSPARK_DRAFTS;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::Glm52StepKv;
use crate::model::Glm52StepShape;
use crate::runner::Glm52RankPlacement;
use crate::runner::Glm52RankWeightLoadReport;
use crate::runner::Glm52RankWorker;
use crate::runner::Glm52RowSample;
use crate::runner::Glm52StepFlags;
use crate::weights::Glm52WeightManifest;

/// Bump on ANY wire-visible change. Both ends ship in one repo at one
/// commit; the handshake check turns a mixed deploy into a clean reject
/// instead of a bincode decode error mid-flight.
const GLM52_WIRE_VERSION: u32 = 1;

/// Frames are small (a `Step` is < 100 KiB even at max table width); anything
/// bigger than this is a corrupted length prefix, not a real frame.
const GLM52_WIRE_MAX_FRAME: u32 = 64 * 1024 * 1024;

fn wire_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

fn write_frame<T: serde::Serialize>(writer: &mut impl Write, frame: &T) -> Result<()> {
    let bytes =
        bincode::serde::encode_to_vec(frame, wire_config()).context("GLM5.2 wire: frame encode")?;
    let len = u32::try_from(bytes.len()).context("GLM5.2 wire: frame too large")?;
    ensure!(len <= GLM52_WIRE_MAX_FRAME, "GLM5.2 wire: frame too large");
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

fn read_frame<T: serde::de::DeserializeOwned>(reader: &mut impl Read) -> Result<T> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    ensure!(
        len <= GLM52_WIRE_MAX_FRAME,
        "GLM5.2 wire: frame length {len} exceeds cap (corrupt stream?)"
    );
    let mut bytes = vec![0u8; len as usize];
    reader.read_exact(&mut bytes)?;
    let (frame, consumed) = bincode::serde::decode_from_slice(&bytes, wire_config())
        .context("GLM5.2 wire: frame decode")?;
    ensure!(
        consumed == bytes.len(),
        "GLM5.2 wire: trailing bytes in frame"
    );
    Ok(frame)
}

/// Connection opener: what this node is being asked to host. The rank-host
/// derives its own weight-load bundles from `(model_path, moe_topo)` — the
/// weights live on storage both ends can read, so only the coordinates cross
/// the wire.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct WireHello {
    version: u32,
    moe_topo: Glm52MoeTopo,
    model_path: PathBuf,
    /// Global rank of this node's first worker; local worker `i` is global
    /// rank `first_rank + i` on device ordinal `i`.
    first_rank: usize,
    rank_count: usize,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum WireHelloAck {
    Ready,
    Reject(String),
}

/// One command to one worker. Field-for-field the local
/// [`Glm52RankCommand`](crate::runner) payloads.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum WireRequest {
    LoadWeights {
        model_path: PathBuf,
        moe_topo: Glm52MoeTopo,
    },
    BuildModel {
        max_model_len: usize,
        moe_topo: Glm52MoeTopo,
        dspark_enabled: bool,
    },
    SetupComm {
        unique_id: Vec<u8>,
        moe_topo: Glm52MoeTopo,
    },
    Step {
        inputs: [(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
        kv: Glm52StepKv,
        flags: Glm52StepFlags,
        sampling: Vec<Glm52RowSample>,
        seed: u64,
    },
    LoadDspark {
        path: PathBuf,
    },
    FreeVram,
    Draft {
        bucket: usize,
        resets: Vec<usize>,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
    },
    Shutdown,
}

/// Errors cross the wire as display strings — the coordinator's failure
/// handling is fail-stop either way, so structure would be dead weight.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum WireResponse {
    LoadWeights(Result<Glm52RankWeightLoadReport, String>),
    /// The rank's KV arenas hold device pointers and MUST NOT cross the
    /// wire; remote arenas would register with a node-local offload host
    /// (not built yet — the engine rejects `kv_offload` + remote ranks at
    /// launch). The reply carries the arena count for the log line only.
    BuildModel(Result<usize, String>),
    SetupComm(Result<(), String>),
    Step(Result<[u32; GLM52_MAX_BATCH_PER_RANK], String>),
    LoadDspark(Result<(), String>),
    FreeVram(Result<usize, String>),
    Draft(Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>, String>),
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct WireCmd {
    worker: u8,
    req: WireRequest,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct WireResp {
    worker: u8,
    body: WireResponse,
}

// ---------------------------------------------------------------------------
// Coordinator side
// ---------------------------------------------------------------------------

/// A queued response slot: which typed sender the next `WireResp` for this
/// worker fulfills. Pushed under the per-worker lock in the same critical
/// section as the frame write, so queue order == wire order.
enum PendingResp {
    LoadWeights(Sender<Result<Glm52RankWeightLoadReport>>),
    BuildModel(Sender<Result<Vec<KvArena>>>),
    SetupComm(Sender<Result<()>>),
    Step(Sender<Result<[u32; GLM52_MAX_BATCH_PER_RANK]>>),
    LoadDspark(Sender<Result<()>>),
    FreeVram(Sender<Result<usize>>),
    Draft(Sender<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>),
}

impl PendingResp {
    /// Fulfill this slot from a decoded response body; a cross-kind pair is
    /// a protocol bug and poisons the node.
    fn fulfill(self, body: WireResponse) -> Result<()> {
        fn send<T>(tx: &Sender<Result<T>>, value: Result<T, String>) {
            // A dropped receiver means the caller gave up (engine teardown);
            // not a wire error.
            let _ = tx.send(value.map_err(|msg| anyhow!(msg)));
        }
        match (self, body) {
            (Self::LoadWeights(tx), WireResponse::LoadWeights(v)) => send(&tx, v),
            (Self::BuildModel(tx), WireResponse::BuildModel(v)) => {
                send(&tx, v.map(|_count| Vec::new()));
            }
            (Self::SetupComm(tx), WireResponse::SetupComm(v)) => send(&tx, v),
            (Self::Step(tx), WireResponse::Step(v)) => send(&tx, v),
            (Self::LoadDspark(tx), WireResponse::LoadDspark(v)) => send(&tx, v),
            (Self::FreeVram(tx), WireResponse::FreeVram(v)) => send(&tx, v),
            (Self::Draft(tx), WireResponse::Draft(v)) => send(&tx, v),
            (_, body) => bail!("GLM5.2 wire: response kind mismatch (got {body:?})"),
        }
        Ok(())
    }

    fn fail(self, msg: &str) {
        let err = || anyhow!("{msg}");
        match self {
            Self::LoadWeights(tx) => drop(tx.send(Err(err()))),
            Self::BuildModel(tx) => drop(tx.send(Err(err()))),
            Self::SetupComm(tx) => drop(tx.send(Err(err()))),
            Self::Step(tx) => drop(tx.send(Err(err()))),
            Self::LoadDspark(tx) => drop(tx.send(Err(err()))),
            Self::FreeVram(tx) => drop(tx.send(Err(err()))),
            Self::Draft(tx) => drop(tx.send(Err(err()))),
        }
    }
}

struct RemoteNodeShared {
    addr: String,
    /// Socket handle held ONLY for `shutdown()`: dropping the writer clone
    /// alone never sends a FIN while the demux reader clone keeps the fd
    /// alive, so poisoning must shut the socket down explicitly — that both
    /// unblocks the demux `read` and lets the rank-host see EOF and tear its
    /// workers down.
    stream: TcpStream,
    writer: Mutex<Option<BufWriter<TcpStream>>>,
    /// Per local worker index: responses fulfill front-to-back.
    pending: Vec<Mutex<VecDeque<PendingResp>>>,
    /// First poison reason; `Some` fails every later call fast.
    poison: Mutex<Option<String>>,
}

impl RemoteNodeShared {
    /// Orderly teardown: same sealing as [`Self::poison`] without the alarm —
    /// the coordinator closing its node is the normal last step of a run.
    fn close(&self) {
        self.seal("coordinator closed the connection", false);
    }

    fn poison(&self, reason: &str) {
        self.seal(reason, true);
    }

    fn seal(&self, reason: &str, failure: bool) {
        {
            let mut poison = self.poison.lock().expect("poison lock");
            if poison.is_none() {
                if failure {
                    log::error!("GLM5.2 remote node {} failed: {reason}", self.addr);
                } else {
                    log::info!("GLM5.2 remote node {}: {reason}", self.addr);
                }
                *poison = Some(reason.to_string());
            }
        }
        // Shut the socket down FIRST (unblocks the demux thread, EOFs the
        // remote end, and unblocks any writer stuck in write_all — taking the
        // writer lock before the shutdown would deadlock against it), then
        // fail everything still in flight.
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        *self.writer.lock().expect("writer lock") = None;
        for queue in &self.pending {
            let mut queue = queue.lock().expect("pending lock");
            while let Some(slot) = queue.pop_front() {
                slot.fail(reason);
            }
        }
    }

    fn poison_reason(&self) -> Option<String> {
        self.poison.lock().expect("poison lock").clone()
    }

    /// Queue the response slot and write the frame under the per-worker lock
    /// — the atomicity is what makes queue order equal wire order.
    fn submit(&self, worker: u8, req: WireRequest, slot: PendingResp) -> Result<()> {
        if let Some(reason) = self.poison_reason() {
            bail!("GLM5.2 remote node {}: {reason}", self.addr);
        }
        let mut queue = self.pending[worker as usize].lock().expect("pending lock");
        let mut writer = self.writer.lock().expect("writer lock");
        let Some(stream) = writer.as_mut() else {
            bail!("GLM5.2 remote node {}: connection closed", self.addr);
        };
        if let Err(err) = write_frame(stream, &WireCmd { worker, req }) {
            drop(writer);
            drop(queue);
            self.poison(&format!("write failed: {err:#}"));
            bail!("GLM5.2 remote node {}: write failed: {err:#}", self.addr);
        }
        queue.push_back(slot);
        Ok(())
    }
}

/// One connected rank-host node. Held behind an `Arc` by every one of its
/// [`Glm52RemoteRankWorker`]s — the LAST worker drop closes the connection
/// (which tears down the remote workers) and joins the demux thread, so the
/// node's lifetime needs no separate plumbing through the engine.
pub(crate) struct Glm52RemoteNode {
    shared: Arc<RemoteNodeShared>,
    demux: Option<thread::JoinHandle<()>>,
}

impl Glm52RemoteNode {
    /// Connect, handshake, and start the response demux thread; returns the
    /// node's workers in global-rank order. The remote end derives its own
    /// load bundles and spawns its workers before acking, so a `Ready` means
    /// the node is fully staffed.
    pub(crate) fn connect(
        addr: &str,
        model_path: &Path,
        moe_topo: Glm52MoeTopo,
        first_rank: usize,
        rank_count: usize,
    ) -> Result<Vec<Glm52RemoteRankWorker>> {
        ensure!(rank_count > 0, "GLM5.2 remote node needs at least one rank");
        let stream =
            TcpStream::connect(addr).with_context(|| format!("GLM5.2 rank-host connect {addr}"))?;
        stream.set_nodelay(true)?;
        let shutdown_handle = stream.try_clone()?;
        let mut writer = BufWriter::new(stream.try_clone()?);
        write_frame(
            &mut writer,
            &WireHello {
                version: GLM52_WIRE_VERSION,
                moe_topo,
                model_path: model_path.to_path_buf(),
                first_rank,
                rank_count,
            },
        )?;
        // Handshake-only read deadline (the ack lands after the rank-host
        // parses the weight manifest and spawns workers — seconds, not the
        // minutes-long weight load, which happens later via commands).
        // Cleared before the demux loop takes over the reader.
        stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
        let mut reader = BufReader::new(stream);
        match read_frame::<WireHelloAck>(&mut reader)
            .with_context(|| format!("GLM5.2 rank-host {addr} handshake"))?
        {
            WireHelloAck::Ready => {}
            WireHelloAck::Reject(reason) => {
                bail!("GLM5.2 rank-host {addr} rejected: {reason}")
            }
        }
        shutdown_handle.set_read_timeout(None)?;
        log::info!(
            "GLM5.2 rank-host {addr} ready: ranks {first_rank}..{}",
            first_rank + rank_count
        );
        let shared = Arc::new(RemoteNodeShared {
            addr: addr.to_string(),
            stream: shutdown_handle,
            writer: Mutex::new(Some(writer)),
            pending: (0..rank_count)
                .map(|_| Mutex::new(VecDeque::new()))
                .collect(),
            poison: Mutex::new(None),
        });
        let demux_shared = Arc::clone(&shared);
        let demux = thread::Builder::new()
            .name(format!("glm52-remote-{addr}"))
            .spawn(move || {
                loop {
                    let resp: WireResp = match read_frame(&mut reader) {
                        Ok(resp) => resp,
                        Err(err) => {
                            demux_shared.poison(&format!("read failed: {err:#}"));
                            return;
                        }
                    };
                    let Some(queue) = demux_shared.pending.get(resp.worker as usize) else {
                        demux_shared
                            .poison(&format!("response for unknown worker {}", resp.worker));
                        return;
                    };
                    let slot = queue.lock().expect("pending lock").pop_front();
                    let Some(slot) = slot else {
                        demux_shared
                            .poison(&format!("unsolicited response for worker {}", resp.worker));
                        return;
                    };
                    if let Err(err) = slot.fulfill(resp.body) {
                        demux_shared.poison(&format!("{err:#}"));
                        return;
                    }
                }
            })
            .map_err(|err| anyhow!("failed to spawn GLM5.2 remote demux: {err}"))?;
        let node = Arc::new(Self {
            shared,
            demux: Some(demux),
        });
        Ok((0..rank_count)
            .map(|index| Glm52RemoteRankWorker {
                node: Arc::clone(&node),
                worker: index as u8,
                rank: first_rank + index,
            })
            .collect())
    }
}

impl Drop for Glm52RemoteNode {
    fn drop(&mut self) {
        // The socket shutdown inside makes the rank-host see EOF and tear
        // down its workers, and unblocks our demux read; sealing fails any
        // stragglers locally.
        self.shared.close();
        if let Some(handle) = self.demux.take() {
            let _ = handle.join();
        }
    }
}

/// The coordinator-side twin of one remote [`Glm52RankWorker`]: same typed
/// `*_async` surface, wire underneath.
pub(crate) struct Glm52RemoteRankWorker {
    node: Arc<Glm52RemoteNode>,
    worker: u8,
    rank: usize,
}

impl Glm52RemoteRankWorker {
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn load_weights_async(
        &self,
        model_path: &Path,
        moe_topo: Glm52MoeTopo,
    ) -> Result<Receiver<Result<Glm52RankWeightLoadReport>>> {
        let (tx, rx) = bounded(1);
        self.node.shared.submit(
            self.worker,
            WireRequest::LoadWeights {
                model_path: model_path.to_path_buf(),
                moe_topo,
            },
            PendingResp::LoadWeights(tx),
        )?;
        Ok(rx)
    }

    pub(crate) fn build_model_async(
        &self,
        max_model_len: usize,
        moe_topo: Glm52MoeTopo,
        dspark_enabled: bool,
    ) -> Result<Receiver<Result<Vec<KvArena>>>> {
        let (tx, rx) = bounded(1);
        self.node.shared.submit(
            self.worker,
            WireRequest::BuildModel {
                max_model_len,
                moe_topo,
                dspark_enabled,
            },
            PendingResp::BuildModel(tx),
        )?;
        Ok(rx)
    }

    pub(crate) fn setup_comm_async(
        &self,
        unique_id: [u8; 128],
        moe_topo: Glm52MoeTopo,
    ) -> Result<Receiver<Result<()>>> {
        let (tx, rx) = bounded(1);
        self.node.shared.submit(
            self.worker,
            WireRequest::SetupComm {
                unique_id: unique_id.to_vec(),
                moe_topo,
            },
            PendingResp::SetupComm(tx),
        )?;
        Ok(rx)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_async(
        &self,
        inputs: [(u32, usize); GLM52_MAX_BATCH_PER_RANK],
        shape: Glm52StepShape,
        kv: Glm52StepKv,
        flags: Glm52StepFlags,
        sampling: Vec<Glm52RowSample>,
        seed: u64,
    ) -> Result<Receiver<Result<[u32; GLM52_MAX_BATCH_PER_RANK]>>> {
        let (tx, rx) = bounded(1);
        self.node.shared.submit(
            self.worker,
            WireRequest::Step {
                inputs,
                shape,
                kv,
                flags,
                sampling,
                seed,
            },
            PendingResp::Step(tx),
        )?;
        Ok(rx)
    }

    pub(crate) fn load_dspark_async(&self, path: &Path) -> Result<Receiver<Result<()>>> {
        let (tx, rx) = bounded(1);
        self.node.shared.submit(
            self.worker,
            WireRequest::LoadDspark {
                path: path.to_path_buf(),
            },
            PendingResp::LoadDspark(tx),
        )?;
        Ok(rx)
    }

    pub(crate) fn free_vram_async(&self) -> Result<Receiver<Result<usize>>> {
        let (tx, rx) = bounded(1);
        self.node.shared.submit(
            self.worker,
            WireRequest::FreeVram,
            PendingResp::FreeVram(tx),
        )?;
        Ok(rx)
    }

    pub(crate) fn draft_async(
        &self,
        bucket: usize,
        resets: Vec<usize>,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
    ) -> Result<Receiver<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>> {
        let (tx, rx) = bounded(1);
        self.node.shared.submit(
            self.worker,
            WireRequest::Draft {
                bucket,
                resets,
                appends,
                proposals,
            },
            PendingResp::Draft(tx),
        )?;
        Ok(rx)
    }

    pub(crate) fn request_shutdown(&self) -> Result<()> {
        if self.node.shared.poison_reason().is_some() {
            // Teardown after a failure: the connection is already gone and so
            // are the remote workers (rank-host fail-stop).
            return Ok(());
        }
        let mut writer = self.node.shared.writer.lock().expect("writer lock");
        if let Some(stream) = writer.as_mut() {
            write_frame(
                stream,
                &WireCmd {
                    worker: self.worker,
                    req: WireRequest::Shutdown,
                },
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// rank-host side
// ---------------------------------------------------------------------------

/// Serve exactly ONE rank-host connection, then return so the process exits.
/// One process = one fleet incarnation: worker drop does not return all of
/// the hosted GPU state (weights/arenas stay resident, measured 281 GiB/GPU
/// after a cleanly closed connection), so process exit is the only reliable
/// way to hand the GPUs back — the same reasoning as the teardown watchdog.
/// Wrap in a restart loop (`while true; do openinfer --glm52-rank-host ...;
/// done`) for a persistent node.
pub fn serve_rank_host(listen: &str) -> Result<()> {
    let listener =
        TcpListener::bind(listen).with_context(|| format!("GLM5.2 rank-host bind {listen}"))?;
    log::info!("GLM5.2 rank-host listening on {listen}");
    let (stream, peer) = listener.accept()?;
    log::info!("GLM5.2 rank-host serving coordinator {peer}");
    let result = serve_connection(stream);
    match &result {
        Ok(()) => log::info!(
            "GLM5.2 rank-host connection {peer} closed cleanly; exiting to release GPU state"
        ),
        Err(err) => log::error!("GLM5.2 rank-host connection {peer} failed: {err:#}"),
    }
    result
}

/// A worker's queued in-flight responses: the forwarder drains them in
/// command order (the worker thread completes them in that order) and writes
/// the response frames.
enum HostPending {
    LoadWeights(Receiver<Result<Glm52RankWeightLoadReport>>),
    BuildModel(Receiver<Result<Vec<KvArena>>>),
    SetupComm(Receiver<Result<()>>),
    Step(Receiver<Result<[u32; GLM52_MAX_BATCH_PER_RANK]>>),
    LoadDspark(Receiver<Result<()>>),
    FreeVram(Receiver<Result<usize>>),
    Draft(Receiver<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>),
}

impl HostPending {
    fn wait(self) -> WireResponse {
        fn flat<T>(rx: Receiver<Result<T>>) -> Result<T, String> {
            match rx.recv() {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(err)) => Err(format!("{err:#}")),
                Err(_) => Err("GLM5.2 rank worker dropped its response".to_string()),
            }
        }
        match self {
            Self::LoadWeights(rx) => WireResponse::LoadWeights(flat(rx)),
            Self::BuildModel(rx) => WireResponse::BuildModel(flat(rx).map(|arenas| arenas.len())),
            Self::SetupComm(rx) => WireResponse::SetupComm(flat(rx)),
            Self::Step(rx) => WireResponse::Step(flat(rx)),
            Self::LoadDspark(rx) => WireResponse::LoadDspark(flat(rx)),
            Self::FreeVram(rx) => WireResponse::FreeVram(flat(rx)),
            Self::Draft(rx) => WireResponse::Draft(flat(rx)),
        }
    }
}

fn serve_connection(stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true)?;
    // Handshake-only read deadline: a connect-and-hold socket must not pin
    // the (single-connection) rank-host. Cleared before the demux loop —
    // long idle stretches between commands are the normal serving state.
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut hello_writer = BufWriter::new(stream.try_clone()?);

    let hello: WireHello = read_frame(&mut reader).context("GLM5.2 rank-host hello")?;
    stream.set_read_timeout(None)?;
    if hello.version != GLM52_WIRE_VERSION {
        let reason = format!(
            "wire version mismatch: coordinator {}, rank-host {GLM52_WIRE_VERSION}",
            hello.version
        );
        write_frame(&mut hello_writer, &WireHelloAck::Reject(reason.clone()))?;
        bail!(reason);
    }

    let workers = match spawn_hosted_workers(&hello) {
        Ok(workers) => {
            write_frame(&mut hello_writer, &WireHelloAck::Ready)?;
            workers
        }
        Err(err) => {
            write_frame(&mut hello_writer, &WireHelloAck::Reject(format!("{err:#}")))?;
            return Err(err);
        }
    };
    log::info!(
        "GLM5.2 rank-host hosting ranks {}..{} ({:?}, model {})",
        hello.first_rank,
        hello.first_rank + hello.rank_count,
        hello.moe_topo,
        hello.model_path.display(),
    );

    // One forwarder thread per worker: preserves per-worker FIFO while
    // letting all workers' commands run concurrently. They share the write
    // half behind a mutex.
    let resp_writer = Arc::new(Mutex::new(BufWriter::new(stream)));
    let mut forwarders = Vec::with_capacity(workers.len());
    let mut pending_txs: Vec<Sender<HostPending>> = Vec::with_capacity(workers.len());
    for index in 0..workers.len() {
        let (tx, rx) = unbounded::<HostPending>();
        let writer = Arc::clone(&resp_writer);
        let handle = thread::Builder::new()
            .name(format!("glm52-rank-host-fwd-{index}"))
            .spawn(move || {
                for pending in rx {
                    let body = pending.wait();
                    let mut writer = writer.lock().expect("resp writer lock");
                    if let Err(err) = write_frame(
                        &mut *writer,
                        &WireResp {
                            worker: index as u8,
                            body,
                        },
                    ) {
                        // Fail-stop the CONNECTION, not just this thread: a
                        // silently lost response would leave the lockstep
                        // coordinator waiting forever on a healthy socket.
                        log::error!("GLM5.2 rank-host response write failed: {err:#}");
                        let _ = writer.get_ref().shutdown(std::net::Shutdown::Both);
                        return;
                    }
                }
            })
            .map_err(|err| anyhow!("failed to spawn GLM5.2 rank-host forwarder: {err}"))?;
        forwarders.push(handle);
        pending_txs.push(tx);
    }

    // Demux loop: every decoded command maps 1:1 onto the local worker's
    // typed surface. EOF or a decode error ends the connection (fail-stop);
    // the drops below tear the workers down.
    let result = host_demux_loop(&mut reader, &workers, &pending_txs);

    // The whole teardown — forwarder joins AND worker drops — must sit
    // behind one deadline. A forwarder blocks in `pending.wait()` while its
    // worker is stuck inside a collective (e.g. the coordinator died during
    // NCCL init and `ctx_create` waits for ranks that will never come), and
    // worker drop runs the collective DeepEP destroy, which needs the
    // coordinator-side ranks to rendezvous — and they may already be gone
    // (fail-stop kills the engine process without ceremony). A node stuck in
    // a dead fleet's barrier is worthless, so bound it: past the deadline,
    // exit and let the operator restart a clean process — the GPU state dies
    // with it either way.
    let deadline = std::time::Duration::from_secs(60);
    let (done_tx, done_rx) = bounded::<()>(0);
    let teardown = thread::Builder::new()
        .name("glm52-rank-host-teardown".to_string())
        .spawn(move || {
            drop(pending_txs); // forwarders drain and exit
            for handle in forwarders {
                let _ = handle.join();
            }
            drop(workers);
            let _ = done_tx.send(());
        })
        .map_err(|err| anyhow!("failed to spawn GLM5.2 rank-host teardown: {err}"))?;
    if done_rx.recv_timeout(deadline).is_err() {
        log::error!(
            "GLM5.2 rank-host teardown exceeded {deadline:?} (worker stuck in a \
             collective missing peers?); exiting for a clean restart"
        );
        std::process::exit(2);
    }
    let _ = teardown.join();
    result
}

fn spawn_hosted_workers(hello: &WireHello) -> Result<Vec<Glm52RankWorker>> {
    let manifest = Glm52WeightManifest::from_model_dir(&hello.model_path)?;
    let bundles = manifest.all_rank_load_bundles(hello.moe_topo)?;
    ensure!(
        hello.first_rank + hello.rank_count <= bundles.len(),
        "GLM5.2 rank-host asked for ranks {}..{} but {:?} has {} ranks",
        hello.first_rank,
        hello.first_rank + hello.rank_count,
        hello.moe_topo,
        bundles.len(),
    );
    let mut workers = Vec::with_capacity(hello.rank_count);
    for index in 0..hello.rank_count {
        let rank = hello.first_rank + index;
        let placement = Glm52RankPlacement {
            rank,
            device_ordinal: index,
        };
        workers.push(Glm52RankWorker::spawn(placement, bundles[rank].clone())?);
    }
    Ok(workers)
}

fn host_demux_loop(
    reader: &mut impl Read,
    workers: &[Glm52RankWorker],
    pending_txs: &[Sender<HostPending>],
) -> Result<()> {
    loop {
        let cmd: WireCmd = match read_frame(reader) {
            Ok(cmd) => cmd,
            Err(err) => {
                // EOF after the coordinator dropped the connection is the
                // normal end of an engine's life.
                if err
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::UnexpectedEof)
                {
                    return Ok(());
                }
                return Err(err);
            }
        };
        let index = cmd.worker as usize;
        let (Some(worker), Some(pending_tx)) = (workers.get(index), pending_txs.get(index)) else {
            bail!("GLM5.2 rank-host: command for unknown worker {index}");
        };
        let pending = match cmd.req {
            WireRequest::LoadWeights {
                model_path,
                moe_topo,
            } => HostPending::LoadWeights(worker.load_weights_async(&model_path, moe_topo)?),
            WireRequest::BuildModel {
                max_model_len,
                moe_topo,
                dspark_enabled,
            } => HostPending::BuildModel(worker.build_model_async(
                max_model_len,
                moe_topo,
                dspark_enabled,
            )?),
            WireRequest::SetupComm {
                unique_id,
                moe_topo,
            } => {
                ensure!(
                    !moe_topo.uses_tensor_replicated_moe(),
                    "GLM5.2 rank-host: tensor-replicated topologies are single-node"
                );
                let id: [u8; 128] = unique_id
                    .try_into()
                    .map_err(|_| anyhow!("GLM5.2 rank-host: unique_id must be 128 bytes"))?;
                HostPending::SetupComm(worker.setup_comm_async(id, moe_topo, None)?)
            }
            WireRequest::Step {
                inputs,
                shape,
                kv,
                flags,
                sampling,
                seed,
            } => HostPending::Step(worker.step_async(inputs, shape, kv, flags, sampling, seed)?),
            WireRequest::LoadDspark { path } => {
                HostPending::LoadDspark(worker.load_dspark_async(&path)?)
            }
            WireRequest::FreeVram => HostPending::FreeVram(worker.free_vram_async()?),
            WireRequest::Draft {
                bucket,
                resets,
                appends,
                proposals,
            } => HostPending::Draft(worker.draft_async(bucket, resets, appends, proposals)?),
            WireRequest::Shutdown => {
                // Per-worker shutdown arrives only during whole-engine
                // teardown; the connection close right after it is what
                // actually drops the workers. Nothing to queue.
                worker.request_shutdown()?;
                continue;
            }
        };
        pending_tx
            .send(pending)
            .map_err(|_| anyhow!("GLM5.2 rank-host forwarder {index} died"))?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip() -> Result<()> {
        let mut buf = Vec::new();
        let cmd = WireCmd {
            worker: 3,
            req: WireRequest::Step {
                inputs: [(7u32, 42usize); GLM52_MAX_BATCH_PER_RANK],
                shape: Glm52StepShape {
                    bucket: 8,
                    slots: [0, 1, 2, 3, 4, 5, 6, 7],
                    active_rows: 5,
                },
                kv: Glm52StepKv {
                    pages: vec![1i32; 8 * 16].into_boxed_slice(),
                    slot_mapping: [9i64; GLM52_MAX_BATCH_PER_RANK],
                },
                flags: Glm52StepFlags::plain(),
                sampling: vec![Glm52RowSample {
                    row: 2,
                    params: openinfer_sample::SamplingParams::default(),
                    step: 11,
                }],
                seed: 0xDEAD_BEEF,
            },
        };
        write_frame(&mut buf, &cmd)?;
        let decoded: WireCmd = read_frame(&mut buf.as_slice())?;
        ensure!(decoded.worker == 3);
        match decoded.req {
            WireRequest::Step {
                inputs,
                shape,
                kv,
                sampling,
                seed,
                ..
            } => {
                ensure!(inputs == [(7u32, 42usize); GLM52_MAX_BATCH_PER_RANK]);
                ensure!(shape.bucket == 8 && shape.active_rows == 5);
                ensure!(kv.pages.len() == 8 * 16 && kv.slot_mapping == [9i64; 8]);
                ensure!(sampling.len() == 1 && sampling[0].row == 2);
                ensure!(seed == 0xDEAD_BEEF);
            }
            other => bail!("decoded wrong variant: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn oversized_frame_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(GLM52_WIRE_MAX_FRAME + 1).to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        let result: Result<WireCmd> = read_frame(&mut buf.as_slice());
        assert!(result.is_err());
    }
}
