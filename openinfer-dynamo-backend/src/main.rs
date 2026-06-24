//! Dynamo backend worker serving openinfer (Qwen3).
//!
//! Standalone binary, intentionally NOT a member of the main openinfer
//! workspace — plain single-machine openinfer (`cargo run --release --
//! --model-path ...`) stays dynamo-free. Start one process per GPU and put a
//! Dynamo frontend + KV router in front; the router fans requests across the
//! replicas. `dynamo_backend_common::run` owns the runtime, discovery
//! registration, and graceful-shutdown lifecycle.

use std::sync::Arc;

mod convert;
mod engine;

fn main() -> anyhow::Result<()> {
    let (backend, config) = engine::OpeninferBackend::from_args()?;
    dynamo_backend_common::run(Arc::new(backend), config)
}
