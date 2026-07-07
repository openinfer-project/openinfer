//! `/metrics` Prometheus endpoint.
//!
//! Exposes `vllm:spec_decode_*` counters (matching vLLM's naming so that
//! `vllm bench serve` can scrape them without modification) and an
//! `openinfer:` namespace for engine-specific gauges. The handler reads
//! shared atomics from [`SpecDecodeStats`] at scrape time — no lock, no
//! background thread.
//!
//! The stats Arc is wired lazily via a shared cell because the HTTP server
//! builds its router before the engine future resolves; the bridge task fills
//! the cell once the handle is available.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tokio::sync::Mutex;

use openinfer_engine::engine::SpecDecodeStats;

pub(crate) type StatsCell = Arc<Mutex<Option<Arc<SpecDecodeStats>>>>;

pub(crate) fn stats_cell() -> StatsCell {
    Arc::new(Mutex::new(None))
}

pub(crate) fn metrics_router(stats_cell: StatsCell) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(stats_cell)
}

async fn metrics_handler(State(state): State<StatsCell>) -> Response {
    let stats = state.lock().await.clone();
    let mut buf = String::with_capacity(2048);

    match stats {
        Some(stats) => {
            render_spec_decode_metrics(&mut buf, &stats);
            let mut resp = buf.into_response();
            resp.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
            );
            resp
        }
        None => (
            StatusCode::OK,
            "# speculative decoding not enabled\n",
        )
            .into_response(),
    }
}

/// Emit `vllm:spec_decode_*` counters matching vLLM's Prometheus schema
/// so `vllm bench serve` (which hardcodes `vllm:spec_decode` prefix +
/// `_total` suffix) can scrape them.
fn render_spec_decode_metrics(buf: &mut String, stats: &SpecDecodeStats) {
    let num_drafts = stats.num_drafts();
    let num_draft_tokens = stats.num_draft_tokens();
    let num_accepted_tokens = stats.num_accepted_tokens();
    let per_pos = stats.accepted_per_pos();

    buf.push_str("# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.\n");
    buf.push_str("# TYPE vllm:spec_decode_num_drafts counter\n");
    buf.push_str(&format!("vllm:spec_decode_num_drafts_total {num_drafts}\n"));

    buf.push_str("# HELP vllm:spec_decode_num_draft_tokens Number of draft tokens.\n");
    buf.push_str("# TYPE vllm:spec_decode_num_draft_tokens counter\n");
    buf.push_str(&format!(
        "vllm:spec_decode_num_draft_tokens_total {num_draft_tokens}\n"
    ));

    buf.push_str(
        "# HELP vllm:spec_decode_num_accepted_tokens Number of accepted tokens.\n",
    );
    buf.push_str("# TYPE vllm:spec_decode_num_accepted_tokens counter\n");
    buf.push_str(&format!(
        "vllm:spec_decode_num_accepted_tokens_total {num_accepted_tokens}\n"
    ));

    buf.push_str(
        "# HELP vllm:spec_decode_num_accepted_tokens_per_pos Accepted tokens per draft position.\n",
    );
    buf.push_str("# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter\n");
    for (pos, count) in per_pos.iter().enumerate() {
        buf.push_str(&format!(
            "vllm:spec_decode_num_accepted_tokens_per_pos_total{{position=\"{pos}\"}} {count}\n"
        ));
    }
}
