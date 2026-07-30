//! W3C trace-context intake for distributed request tracing.
//!
//! The pinned `vllm-server` HTTP layer forwards `X-Request-Id` but drops
//! `traceparent`, so an upstream trace (e.g. vllm-router with OTel enabled)
//! cannot reach the bridge's `request` root span. Until vllm-server populates
//! `EngineCoreRequest.trace_headers` itself (upstream PR tracked in
//! docs/subsystems/tracing/e2e-router-tracing.md), this module stashes the
//! incoming `traceparent` at the axum boundary, and the bridge pops it when
//! the matching `EngineCoreRequest` arrives over ZMQ.
//!
//! Correlation key: `EngineCoreRequest.external_req_id`, computed at intake
//! where the route is still known (route prefix + `X-Request-Id`, mirroring
//! vllm-server's completions/chat prefixing), so the bridge lookup is exact.
//! Requests without trace context reserve their slot with an untraced marker,
//! so overlapping attempts reusing one id each consume only their own slot.
//! Entries are popped on use; entries whose request never reaches the engine
//! are retired by a middleware drop guard (error responses, client
//! disconnects), or expire by TTL.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::Method;
use axum::middleware::Next;
use axum::response::Response;

/// Stale-entry lifetime and the stash's hard capacity. Entries are normally
/// popped by the bridge within milliseconds of insertion; both bounds exist
/// only as backstops for requests that never reach the engine.
const TTL: Duration = Duration::from_secs(120);
const CAPACITY: usize = 4096;

/// Routes whose accepted requests produce an `EngineCoreRequest` (and thus
/// consume a stash entry at the bridge). Stashing anywhere else would leak
/// entries for requests that never reach the engine (e.g. traced `/metrics`
/// probes), eventually tripping the capacity clear and splitting live traces.
const GENERATION_PATHS: &[&str] = &[
    "/v1/completions",
    "/v1/chat/completions",
    "/inference/v1/generate",
];

/// `(token, traceparent, inserted-at)` entries queued under one id. A `None`
/// traceparent is an untraced attempt's marker: it reserves the attempt's
/// slot without supplying a parent to join.
type EntryQueue = VecDeque<(u64, Option<String>, Instant)>;

/// Per-insertion token: lets the HTTP layer discard exactly the entry it
/// stashed even when responses complete out of FIFO order.
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Stashed `traceparent` headers awaiting pickup by the engine bridge.
///
/// Cheap to clone (inner `Arc`); one instance is shared between the axum
/// layer (insert) and every engine bridge task (take). Entries queue FIFO per
/// id: concurrent attempts reusing one `X-Request-Id` (e.g. hedged retries)
/// each keep their own slot instead of the latest insert overwriting the
/// rest, paired best-effort in intake order (see `take`'s pairing-ceiling
/// note). A per-attempt unique key is not available — `external_req_id` is
/// the only correlation key the bridge can derive downstream.
#[derive(Clone, Default)]
pub(crate) struct TraceContextStash {
    inner: Arc<Mutex<HashMap<String, EntryQueue>>>,
}

impl TraceContextStash {
    /// Queue `traceparent` (`None` for an untraced marker) under `request_id`,
    /// returning the insertion token the HTTP layer needs to discard exactly
    /// this entry on the error path.
    fn insert(&self, request_id: &str, traceparent: Option<&str>) -> u64 {
        let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
        let mut inner = self.inner.lock().expect("trace context stash poisoned");
        let mut total: usize = inner.values().map(VecDeque::len).sum();
        if total >= CAPACITY {
            let now = Instant::now();
            inner.retain(|_, queue| {
                queue.retain(|(_, _, inserted)| now.duration_since(*inserted) < TTL);
                !queue.is_empty()
            });
            total = inner.values().map(VecDeque::len).sum();
            if total >= CAPACITY {
                // Pathological: more live unreached requests than the stash
                // holds. Dropping it splits some traces; the stash must never
                // grow unbounded or stall serving.
                inner.clear();
            }
        }
        inner.entry(request_id.to_owned()).or_default().push_back((
            token,
            traceparent.map(str::to_owned),
            Instant::now(),
        ));
        token
    }

    /// Pop the oldest unexpired entry queued for `request_id`; each queued
    /// entry is consumed at most once. A fresh parent comes back as `Some`,
    /// anything else — no entry, an expired one, or an untraced marker — as
    /// `None`, which the bridge treats as "start a fresh trace". Expired
    /// entries are dropped, never joined: a request reusing an id must not
    /// attach to a trace left behind by a request that never reached the
    /// engine.
    ///
    /// Pairing ceiling: entries queue in middleware intake order, which the
    /// bridge normally follows. Concurrent requests reusing one id (hedged
    /// retries) can be reordered downstream — e.g. a LoRA request stalled in
    /// body rewriting while a later request overtakes it — and would then
    /// consume each other's parent. Exact pairing is impossible there: the
    /// correlation key is identical by construction and the engine
    /// `request_id`'s random suffix carries no intake information. Callers
    /// needing deterministic pairing must use unique `X-Request-Id` values.
    pub(crate) fn take(&self, request_id: &str) -> Option<String> {
        let mut inner = self.inner.lock().expect("trace context stash poisoned");
        let (result, queue_empty) = {
            let queue = inner.get_mut(request_id)?;
            let mut result = None;
            while let Some((_, traceparent, inserted)) = queue.pop_front() {
                if inserted.elapsed() < TTL {
                    result = traceparent;
                    break;
                }
            }
            (result, queue.is_empty())
        };
        if queue_empty {
            inner.remove(request_id);
        }
        result
    }

    /// Drop exactly the entry tagged `token`, wherever it sits in the queue.
    /// Responses can complete out of FIFO order, so the error path must not
    /// assume the abandoned attempt is at the head.
    fn discard_entry(&self, request_id: &str, token: u64) {
        let mut inner = self.inner.lock().expect("trace context stash poisoned");
        let Some(queue) = inner.get_mut(request_id) else {
            return;
        };
        if let Some(pos) = queue.iter().position(|(t, _, _)| *t == token) {
            queue.remove(pos);
        }
        if queue.is_empty() {
            inner.remove(request_id);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("trace context stash poisoned")
            .values()
            .map(VecDeque::len)
            .sum()
    }
}

/// The id vllm-server will put into `EngineCoreRequest.external_req_id` for a
/// generation request on `path`: the route prefix plus the caller's
/// `X-Request-Id`, mirroring the pinned server's completions/chat prefixing;
/// other routes pass the id through verbatim. Computing it at intake — where
/// the route is still known — keeps the bridge lookup a plain exact match.
fn external_req_id_for(path: &str, request_id: &str) -> String {
    match path {
        "/v1/completions" => format!("cmpl-{request_id}"),
        "/v1/chat/completions" => format!("chatcmpl-{request_id}"),
        _ => request_id.to_owned(),
    }
}

/// Drop guard for one stashed insertion. The bridge consumes an entry the
/// moment its `EngineCoreRequest` arrives (accepted or rejected alike), so
/// when the middleware future ends — normally, on an error response, or by
/// cancellation when the client disconnects mid-pipeline — a still-present
/// entry can only belong to a request that never reached the engine, and is
/// discarded. For consumed entries the discard is a no-op.
struct InsertionGuard {
    stash: TraceContextStash,
    key: String,
    token: u64,
}

impl Drop for InsertionGuard {
    fn drop(&mut self) {
        self.stash.discard_entry(&self.key, self.token);
    }
}

/// Axum middleware: stash the request's `traceparent` under its external id.
///
/// Only generation routes are eligible — other routes never produce an
/// `EngineCoreRequest`, so their entries would leak until the TTL. Does
/// nothing when request tracing is disabled. An [`InsertionGuard`] retires
/// the entry if the request never reaches the engine: HTTP validation
/// rejects, or a client disconnect dropping the future before the handler
/// finishes.
pub(crate) async fn stash_trace_context(
    State(stash): State<TraceContextStash>,
    mut request: Request,
    next: Next,
) -> Response {
    if !openinfer_engine::tracing_state::is_enabled()
        || request.method() != Method::POST
        || !GENERATION_PATHS.contains(&request.uri().path())
    {
        return next.run(request).await;
    }
    let path = request.uri().path().to_owned();
    let _guard = stash_from_headers(&stash, &path, request.headers_mut()).map(|(key, token)| {
        InsertionGuard {
            stash: stash.clone(),
            key,
            token,
        }
    });
    next.run(request).await
}

/// Read `traceparent` from `headers` and stash it under the request's
/// external id, generating and injecting `X-Request-Id` when absent so the
/// bridge and vllm-server agree on the correlation key. Returns the stash key
/// and insertion token when an entry was queued, `None` otherwise.
fn stash_from_headers(
    stash: &TraceContextStash,
    path: &str,
    headers: &mut HeaderMap,
) -> Option<(String, u64)> {
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let Some(traceparent) = headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
    else {
        // A request with no trace context of its own reserves its slot with
        // an untraced marker, so an overlapping traced attempt reusing the id
        // keeps its own parent and this attempt consumes nothing upstream.
        // With no caller id at all, vllm-server mints a unique id downstream
        // and no slot is needed.
        let key = external_req_id_for(path, &request_id?);
        let token = stash.insert(&key, None);
        return Some((key, token));
    };
    let request_id = request_id.unwrap_or_else(|| {
        // Mirror vllm-server's own id shape (8 hex chars) so logs read the
        // same regardless of which side generated the id.
        let mut id = uuid::Uuid::new_v4().simple().to_string();
        id.truncate(8);
        if let Ok(value) = id.parse() {
            headers.insert("x-request-id", value);
        }
        id
    });
    let key = external_req_id_for(path, &request_id);
    let token = stash.insert(&key, Some(&traceparent));
    Some((key, token))
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    const TRACEPARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    const TRACEPARENT_B: &str = "00-1af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    const COMPLETIONS: &str = "/v1/completions";
    const CHAT: &str = "/v1/chat/completions";
    const GENERATE: &str = "/inference/v1/generate";

    fn traced_headers(id: &str, traceparent: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", HeaderValue::from_str(traceparent).unwrap());
        headers.insert("x-request-id", HeaderValue::from_str(id).unwrap());
        headers
    }

    #[test]
    fn stash_roundtrip_decodes_to_upstream_trace() {
        let stash = TraceContextStash::default();
        let mut headers = traced_headers("req-1", TRACEPARENT);

        stash_from_headers(&stash, COMPLETIONS, &mut headers);

        let stashed = stash.take("cmpl-req-1").expect("traceparent stashed");
        let ctx = fastrace::collector::SpanContext::decode_w3c_traceparent(&stashed)
            .expect("valid W3C traceparent");
        assert_eq!(ctx.encode_w3c_traceparent(), TRACEPARENT);
        // One-shot: the bridge must not join a second request to the same span.
        assert!(stash.take("cmpl-req-1").is_none());
    }

    #[test]
    fn generates_and_injects_request_id_when_absent() {
        let stash = TraceContextStash::default();
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", HeaderValue::from_static(TRACEPARENT));

        stash_from_headers(&stash, COMPLETIONS, &mut headers);

        let injected = headers
            .get("x-request-id")
            .expect("request id injected")
            .to_str()
            .expect("ascii request id");
        assert_eq!(injected.len(), 8);
        assert!(injected.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(stash.take(&format!("cmpl-{injected}")).is_some());
    }

    #[test]
    fn external_key_mirrors_route_prefixing() {
        // Each route's stash key is exactly the external_req_id the bridge
        // will see: completions/chat prepend, other routes pass through.
        let stash = TraceContextStash::default();
        for (path, expected_key) in [
            (COMPLETIONS, "cmpl-dbg12345"),
            (CHAT, "chatcmpl-dbg12345"),
            (GENERATE, "dbg12345"),
        ] {
            let mut headers = traced_headers("dbg12345", TRACEPARENT);
            stash_from_headers(&stash, path, &mut headers);
            assert_eq!(stash.take(expected_key), Some(TRACEPARENT.to_owned()));
        }
    }

    #[test]
    fn unprefixed_route_preserves_prefixed_header_ids() {
        // /inference/v1/generate prepends nothing: a header literally named
        // `cmpl-foo` must be looked up verbatim, not mistaken for a
        // completions-generated prefix of `foo`.
        let stash = TraceContextStash::default();
        let mut generated = traced_headers("cmpl-foo", TRACEPARENT_B);
        stash_from_headers(&stash, GENERATE, &mut generated);

        assert_eq!(stash.take("cmpl-foo"), Some(TRACEPARENT_B.to_owned()));
    }

    #[test]
    fn prefixed_header_ids_collide_only_with_themselves() {
        // Completions headers `foo` and `cmpl-foo` become `cmpl-foo` and
        // `cmpl-cmpl-foo` at the bridge; each consumes its own traceparent.
        let stash = TraceContextStash::default();
        for (id, tp) in [("foo", TRACEPARENT), ("cmpl-foo", TRACEPARENT_B)] {
            let mut headers = traced_headers(id, tp);
            stash_from_headers(&stash, COMPLETIONS, &mut headers);
        }

        assert_eq!(stash.take("cmpl-foo"), Some(TRACEPARENT.to_owned()));
        assert_eq!(stash.take("cmpl-cmpl-foo"), Some(TRACEPARENT_B.to_owned()));
    }

    #[test]
    fn take_drops_expired_entries() {
        let stash = TraceContextStash::default();
        let expired = Instant::now()
            .checked_sub(TTL + Duration::from_secs(1))
            .expect("TTL subtraction stays within Instant range");
        stash
            .inner
            .lock()
            .expect("trace context stash poisoned")
            .insert(
                "old".to_owned(),
                VecDeque::from([(1, Some(TRACEPARENT.to_owned()), expired)]),
            );

        assert!(stash.take("old").is_none());
    }

    #[test]
    fn duplicate_ids_keep_separate_parents_fifo() {
        // Hedged retries reusing one X-Request-Id concurrently: each attempt
        // must consume a distinct parent, oldest first.
        let stash = TraceContextStash::default();
        for tp in [TRACEPARENT, TRACEPARENT_B] {
            let mut headers = traced_headers("hedged", tp);
            stash_from_headers(&stash, COMPLETIONS, &mut headers);
        }
        assert_eq!(stash.len(), 2);

        assert_eq!(stash.take("cmpl-hedged"), Some(TRACEPARENT.to_owned()));
        assert_eq!(stash.take("cmpl-hedged"), Some(TRACEPARENT_B.to_owned()));
        assert!(stash.take("cmpl-hedged").is_none());
        assert_eq!(stash.len(), 0);
    }

    #[test]
    fn error_path_discards_the_rejected_attempts_own_entry() {
        // Two overlapping traced requests share an id; the later one errors
        // out first. Discarding must remove the later attempt's entry, not
        // the queue head.
        let stash = TraceContextStash::default();
        let mut first = traced_headers("dup", TRACEPARENT);
        stash_from_headers(&stash, COMPLETIONS, &mut first);
        let mut second = traced_headers("dup", TRACEPARENT_B);
        let (_, token_b) = stash_from_headers(&stash, COMPLETIONS, &mut second).unwrap();

        stash.discard_entry("cmpl-dup", token_b);

        assert_eq!(stash.take("cmpl-dup"), Some(TRACEPARENT.to_owned()));
        assert!(stash.take("cmpl-dup").is_none());
    }

    #[test]
    fn traced_retry_after_rejection_gets_fresh_parent() {
        // First attempt is rejected before reaching the engine; the HTTP
        // layer discards its entry on the error response, so the traced
        // retry's own parent is what the bridge consumes.
        let stash = TraceContextStash::default();
        let mut first = traced_headers("retry-2", TRACEPARENT);
        let stashed = stash_from_headers(&stash, COMPLETIONS, &mut first);
        assert_eq!(
            stashed.as_ref().map(|(key, _)| key.as_str()),
            Some("cmpl-retry-2")
        );

        let (key, token) = stashed.unwrap();
        stash.discard_entry(&key, token);
        assert_eq!(stash.len(), 0);

        let mut retry = traced_headers("retry-2", TRACEPARENT_B);
        stash_from_headers(&stash, COMPLETIONS, &mut retry);
        assert_eq!(stash.take("cmpl-retry-2"), Some(TRACEPARENT_B.to_owned()));
        assert_eq!(stash.len(), 0);
    }

    #[test]
    fn untraced_attempt_reserves_its_own_slot() {
        // An overlapping untraced request must neither consume the traced
        // attempt's parent nor delete it: it queues a marker and the bridge
        // opens a fresh trace for exactly that attempt.
        let stash = TraceContextStash::default();
        let mut traced = traced_headers("retry-1", TRACEPARENT);
        stash_from_headers(&stash, COMPLETIONS, &mut traced);
        assert_eq!(stash.len(), 1);

        let mut retry = HeaderMap::new();
        retry.insert("x-request-id", HeaderValue::from_static("retry-1"));
        stash_from_headers(&stash, COMPLETIONS, &mut retry);
        assert_eq!(stash.len(), 2);

        // The traced attempt still consumes its own parent first...
        assert_eq!(stash.take("cmpl-retry-1"), Some(TRACEPARENT.to_owned()));
        // ...and the untraced attempt pops its marker: no parent, one-shot.
        assert!(stash.take("cmpl-retry-1").is_none());
        assert_eq!(stash.len(), 0);
    }

    #[test]
    fn dropped_guard_retires_only_its_own_entry() {
        // Client disconnects mid-pipeline drop the middleware future; the
        // guard must retire exactly the cancelled attempt's entry.
        let stash = TraceContextStash::default();
        let mut first = traced_headers("g", TRACEPARENT);
        let (key_a, token_a) = stash_from_headers(&stash, COMPLETIONS, &mut first).unwrap();
        let mut second = traced_headers("g", TRACEPARENT_B);
        let (key_b, _) = stash_from_headers(&stash, COMPLETIONS, &mut second).unwrap();
        assert_eq!(key_a, key_b);

        drop(InsertionGuard {
            stash: stash.clone(),
            key: key_a,
            token: token_a,
        });

        assert_eq!(stash.take(&key_b), Some(TRACEPARENT_B.to_owned()));
        assert!(stash.take(&key_b).is_none());
    }

    #[test]
    fn ignores_requests_without_traceparent_or_id() {
        let stash = TraceContextStash::default();
        let mut headers = HeaderMap::new();

        stash_from_headers(&stash, COMPLETIONS, &mut headers);

        assert_eq!(stash.len(), 0);
        assert!(headers.get("x-request-id").is_none());
    }
}
