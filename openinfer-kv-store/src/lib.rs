//! `openinfer-kv-store`: the shared KV read/write orchestration layer.
//!
//! Design: `docs/subsystems/kv-cache/design.md`. This crate gives the per-model
//! offload glue (qwen3's prefetch state machine, glm52's `offload.rs`) one
//! home built on the same primitives they already use: the logical
//! `BlockPool` for GPU pages and a [`HostTier`] (pegaflow via
//! `OffloadEngine`) below it.
//!
//! Three verbs:
//! - [`KvStore::resolve_prefix`] — the whole read path as one async fn:
//!   probe the GPU radix, query the host tier (re-query/deadline built in),
//!   reserve pages under the admission floor, load, and register into the
//!   radix. One terminal type: [`KvPrefix`] (re-exported from the engine
//!   contract) — degraded outcomes surface as a
//!   smaller hit plus a stats event, never a distinct variant.
//! - [`KvStore::seal`] — save freshly-sealed blocks at a checkpoint boundary.
//!   Guards pin the source pages across the async D2H (the reuse contract).
//! - [`KvStore::retire`] — final seal + release; parks the whole `RequestKv`
//!   with any must-complete saves instead of blocking anyone.
//!
//! The scheduler stays synchronous: it receives resolved requests from its
//! submit channel, reads [`KvStore::pinned_blocks`] during admission, and
//! maintains [`KvStore::set_admission_floor`]. Cancellation is the request's
//! existing abort state ([`CancelProbe`] over `TokenSink::is_closed`) observed
//! between operations; a submitted DMA is an uncancellable section.

mod builder;
mod policy;
mod stats;
mod store;
mod tier;

pub mod testkit;

pub use openinfer_engine::engine::KvPrefix;

pub use crate::builder::KvStoreBuilder;
pub use crate::policy::CacheScope;
pub use crate::policy::CancelProbe;
pub use crate::policy::NeverCancelled;
pub use crate::policy::ResolvePolicy;
pub use crate::policy::SaveClass;
pub use crate::policy::SaveCursor;
pub use crate::stats::DegradeReason;
pub use crate::stats::KvStoreStats;
pub use crate::store::KvStore;
pub use crate::tier::HostTier;
pub use crate::tier::TierFuture;
pub use crate::tier::TierHit;
pub use crate::tier::TierQuery;
