//! W3C trace-context intake for distributed request tracing.
//!
//! The pinned `vllm-server` HTTP layer forwards `X-Request-Id` but drops
//! `traceparent`, so an upstream trace (e.g. vllm-router with OTel enabled)
//! cannot reach the bridge's `request` root span. Until vllm-server populates
//! `EngineCoreRequest.trace_headers` itself (upstream PR tracked in
//! docs/subsystems/tracing/e2e-router-tracing.md), this module stashes the
//! incoming `traceparent` at the axum boundary keyed by the request's
//! external id, and the bridge pops it when the matching `EngineCoreRequest`
//! arrives over ZMQ.
//!
//! Correlation key: the request's `X-Request-Id`, which vllm-server resolves
//! into `EngineCoreRequest.external_req_id`. When the caller sent none, the
//! middleware generates one and injects it into the request headers so both
//! sides agree on the key. Entries are popped on use; requests rejected
//! before reaching the engine leave stale entries that expire by TTL.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;

/// Stale-entry lifetime and the stash's hard capacity. Entries are normally
/// popped by the bridge within milliseconds of insertion; both bounds exist
/// only for requests that never reach the engine (e.g. validation rejects).
const TTL: Duration = Duration::from_secs(120);
const CAPACITY: usize = 4096;

/// Stashed `traceparent` headers awaiting pickup by the engine bridge.
///
/// Cheap to clone (inner `Arc`); one instance is shared between the axum
/// layer (insert) and every engine bridge task (take).
#[derive(Clone, Default)]
pub(crate) struct TraceContextStash {
    inner: Arc<Mutex<HashMap<String, (String, Instant)>>>,
}

impl TraceContextStash {
    fn insert(&self, request_id: &str, traceparent: &str) {
        let mut inner = self.inner.lock().expect("trace context stash poisoned");
        if inner.len() >= CAPACITY {
            let now = Instant::now();
            inner.retain(|_, (_, inserted)| now.duration_since(*inserted) < TTL);
            if inner.len() >= CAPACITY {
                // Pathological: more live unreached requests than the stash
                // holds. Dropping it splits some traces; the stash must never
                // grow unbounded or stall serving.
                inner.clear();
            }
        }
        inner.insert(
            request_id.to_owned(),
            (traceparent.to_owned(), Instant::now()),
        );
    }

    /// Pop the stashed traceparent for `request_id`. One-shot: a second call
    /// for the same id returns `None`. Entries older than [`TTL`] are dropped
    /// instead of returned, so a request reusing an id never joins a stale
    /// trace left behind by a request that never reached the engine.
    pub(crate) fn take(&self, request_id: &str) -> Option<String> {
        let (traceparent, inserted) = self
            .inner
            .lock()
            .expect("trace context stash poisoned")
            .remove(request_id)?;
        (inserted.elapsed() < TTL).then_some(traceparent)
    }

    /// Pop the traceparent for a request the bridge just received.
    ///
    /// `external_req_id` is vllm-server's external request id: the caller's
    /// `X-Request-Id` with an API-specific prefix prepended (`cmpl-` for
    /// completions, `chatcmpl-` for chat completions). The stash is keyed by
    /// the bare header value, so try the id verbatim first, then with the
    /// known prefixes stripped (longest first — `cmpl-` is a suffix of
    /// `chatcmpl-`'s tail).
    pub(crate) fn take_for_external_req_id(&self, external_req_id: &str) -> Option<String> {
        self.take(external_req_id).or_else(|| {
            let bare = external_req_id
                .strip_prefix("chatcmpl-")
                .or_else(|| external_req_id.strip_prefix("cmpl-"))?;
            self.take(bare)
        })
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("trace context stash poisoned")
            .len()
    }
}

/// Axum middleware: stash the request's `traceparent` under its external id.
///
/// Does nothing when request tracing is disabled (the bridge would never
/// look entries up) or the request carries no `traceparent`.
pub(crate) async fn stash_trace_context(
    State(stash): State<TraceContextStash>,
    mut request: Request,
    next: Next,
) -> Response {
    if openinfer_engine::tracing_state::is_enabled() {
        stash_from_headers(&stash, request.headers_mut());
    }
    next.run(request).await
}

/// Read `traceparent` from `headers` and stash it under the request's
/// external id, generating and injecting `X-Request-Id` when absent so the
/// bridge and vllm-server agree on the correlation key.
fn stash_from_headers(stash: &TraceContextStash, headers: &mut HeaderMap) {
    let Some(traceparent) = headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
    else {
        return;
    };
    let request_id = if let Some(id) = headers.get("x-request-id").and_then(|v| v.to_str().ok()) {
        id.to_owned()
    } else {
        // Mirror vllm-server's own id shape (8 hex chars) so logs read the
        // same regardless of which side generated the id.
        let mut id = uuid::Uuid::new_v4().simple().to_string();
        id.truncate(8);
        if let Ok(value) = id.parse() {
            headers.insert("x-request-id", value);
        }
        id
    };
    stash.insert(&request_id, &traceparent);
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    const TRACEPARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    #[test]
    fn stash_roundtrip_decodes_to_upstream_trace() {
        let stash = TraceContextStash::default();
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", HeaderValue::from_static(TRACEPARENT));
        headers.insert("x-request-id", HeaderValue::from_static("req-1"));

        stash_from_headers(&stash, &mut headers);

        let stashed = stash.take("req-1").expect("traceparent stashed");
        let ctx = fastrace::collector::SpanContext::decode_w3c_traceparent(&stashed)
            .expect("valid W3C traceparent");
        assert_eq!(ctx.encode_w3c_traceparent(), TRACEPARENT);
        // One-shot: the bridge must not join a second request to the same span.
        assert!(stash.take("req-1").is_none());
    }

    #[test]
    fn generates_and_injects_request_id_when_absent() {
        let stash = TraceContextStash::default();
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", HeaderValue::from_static(TRACEPARENT));

        stash_from_headers(&stash, &mut headers);

        let injected = headers
            .get("x-request-id")
            .expect("request id injected")
            .to_str()
            .expect("ascii request id");
        assert_eq!(injected.len(), 8);
        assert!(injected.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(stash.take(injected).is_some());
    }

    #[test]
    fn lookup_tolerates_vllm_api_prefixes() {
        let stash = TraceContextStash::default();
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", HeaderValue::from_static(TRACEPARENT));
        headers.insert("x-request-id", HeaderValue::from_static("dbg12345"));

        stash_from_headers(&stash, &mut headers);

        assert!(stash.take_for_external_req_id("chatcmpl-nope").is_none());
        assert_eq!(
            stash.take_for_external_req_id("cmpl-dbg12345"),
            Some(TRACEPARENT.to_owned())
        );
        assert!(stash.take_for_external_req_id("cmpl-dbg12345").is_none());

        let mut headers = HeaderMap::new();
        headers.insert("traceparent", HeaderValue::from_static(TRACEPARENT));
        headers.insert("x-request-id", HeaderValue::from_static("dbg12345"));
        stash_from_headers(&stash, &mut headers);
        assert_eq!(
            stash.take_for_external_req_id("chatcmpl-dbg12345"),
            Some(TRACEPARENT.to_owned())
        );
    }

    #[test]
    fn take_drops_expired_entries() {
        let stash = TraceContextStash::default();
        stash
            .inner
            .lock()
            .expect("trace context stash poisoned")
            .insert(
                "old".to_owned(),
                (
                    TRACEPARENT.to_owned(),
                    Instant::now() - TTL - Duration::from_secs(1),
                ),
            );

        assert!(stash.take("old").is_none());
    }

    #[test]
    fn ignores_requests_without_traceparent() {
        let stash = TraceContextStash::default();
        let mut headers = HeaderMap::new();

        stash_from_headers(&stash, &mut headers);

        assert_eq!(stash.len(), 0);
        assert!(headers.get("x-request-id").is_none());
    }
}
