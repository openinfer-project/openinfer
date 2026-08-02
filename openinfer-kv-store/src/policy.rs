//! The caller-facing vocabulary of the store: cancellation, cache scoping,
//! per-request read policy, and save bookkeeping.

use openinfer_engine::engine::TokenSink;
use tokio::sync::oneshot;

/// Whether the request wants the resolve abandoned. Implemented by the
/// engine's `TokenSink` (the abort atomic the frontend flips); the store
/// observes it between operations and short-circuits remaining I/O.
pub trait CancelProbe: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

impl CancelProbe for TokenSink {
    fn is_cancelled(&self) -> bool {
        self.is_closed()
    }
}

/// For callers without a request context (tests, warmup).
pub struct NeverCancelled;

impl CancelProbe for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Prefix-cache identity of a resolve, mirroring
/// `BlockPool::probe_prefix_with_cache_salt`: the producer request and the
/// resolve must derive identical block hashes or the query keys are
/// unrelated.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheScope<'a> {
    pub(crate) cache_salt: Option<&'a str>,
    pub(crate) lora_name: Option<&'a str>,
}

impl<'a> CacheScope<'a> {
    /// Extra cache identity beyond the tokens (glm52's native-MTP page salt).
    #[must_use]
    pub fn cache_salt(mut self, salt: &'a str) -> Self {
        self.cache_salt = Some(salt);
        self
    }

    /// Weight identity: blocks computed under one adapter never match
    /// another's.
    #[must_use]
    pub fn lora(mut self, name: &'a str) -> Self {
        self.lora_name = Some(name);
        self
    }
}

/// Per-request read policy for [`crate::KvStore::resolve_prefix`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ResolvePolicy {
    pub(crate) wait_for_full_hit: bool,
}

impl ResolvePolicy {
    /// The caller cannot recompute a miss (P/D decode: admission asserts the
    /// hit covers the handoff's committed length), so both halves of that
    /// intent apply: the tier query is all-or-nothing (a partial hit is
    /// worthless — pegaflow's `wait_for_full_prefix`), and a `Miss` means
    /// the producer's registration has not landed yet — keep waiting under
    /// the deadline instead of concluding the cache is cold.
    #[must_use]
    pub fn wait_for_full_hit(mut self) -> Self {
        self.wait_for_full_hit = true;
        self
    }
}

/// Save quality-of-service, from the design doc's QoS split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveClass {
    /// Fire-and-forget cacheability: a lost save forfeits a future hit, never
    /// correctness. Sheddable under pressure.
    Cacheable,
    /// Must-complete (P/D handoff): [`crate::KvStore::retire`] parks the request's KV
    /// until these saves settle. Lease semantics against the consuming peer
    /// land with the glm52 P/D migration.
    Handoff,
}

/// Per-request save bookkeeping, owned by the scheduler next to the
/// `RequestKv` (no hidden map in the store). Starts past the prefix-cache
/// hit — those blocks were stored by whoever first sealed them.
#[derive(Default)]
pub struct SaveCursor {
    pub(crate) saved_blocks: usize,
    /// Completion outcomes of this request's `Handoff`-class saves, awaited
    /// by [`crate::KvStore::retire`] before the KV releases.
    pub(crate) pending: Vec<oneshot::Receiver<Result<(), String>>>,
}

impl SaveCursor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}
