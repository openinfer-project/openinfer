use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, Request};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde_json::{Value, json};

mod otlp;
mod trace;

pub use otlp::{OpenTelemetryOptions, OpenTelemetrySink};
use trace::RequestRecord;
pub(crate) use trace::RequestTrace;

const PROMETHEUS_TEXT: &str = "text/plain; version=0.0.4; charset=utf-8";
const JSON: &str = "application/json";
const DEFAULT_METRIC_PREFIX: &str = "openinfer_frontend";
const LATENCY_BUCKETS_MS: &[f64] = &[
    1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0,
];

#[derive(Clone)]
pub struct Telemetry {
    inner: Arc<Inner>,
}

struct Inner {
    request_log_enabled: bool,
    http_trace_log_enabled: bool,
    trace_buffer_capacity: usize,
    trace_buffer: Option<Mutex<VecDeque<Value>>>,
    opentelemetry_sink: Option<OpenTelemetrySink>,
    active_requests: AtomicI64,
    started_requests: AtomicU64,
    finished_requests: [AtomicU64; RequestOutcome::COUNT],
    prompt_tokens: AtomicU64,
    cached_prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    metrics: MetricNames,
    queue_wait_ms: Histogram,
    ttft_ms: Histogram,
    request_duration_ms: Histogram,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestOutcome {
    Length,
    Stop,
    Error,
    Rejected,
    Aborted,
}

impl RequestOutcome {
    const COUNT: usize = 5;
    const ALL: [Self; Self::COUNT] = [
        Self::Length,
        Self::Stop,
        Self::Error,
        Self::Rejected,
        Self::Aborted,
    ];

    fn index(self) -> usize {
        match self {
            Self::Length => 0,
            Self::Stop => 1,
            Self::Error => 2,
            Self::Rejected => 3,
            Self::Aborted => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
            Self::Error => "error",
            Self::Rejected => "rejected",
            Self::Aborted => "aborted",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RequestMetrics {
    pub queue_wait_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub duration_ms: Option<f64>,
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Clone)]
pub struct TelemetryOptions {
    pub request_log_enabled: bool,
    pub http_trace_log_enabled: bool,
    pub trace_buffer_capacity: usize,
    /// Prometheus metric prefix. Defaults to `openinfer_frontend`.
    pub metric_prefix: String,
    /// Prometheus latency histogram buckets in milliseconds.
    ///
    /// Values are filtered, sorted, and deduplicated. If no usable bucket is
    /// provided, the default serving-oriented bucket set is used.
    pub latency_buckets_ms: Vec<f64>,
    /// Optional built-in OTLP payload sink.
    ///
    /// Custom integrations should subscribe to the emitted `tracing` spans and
    /// events instead of adding callbacks here.
    pub opentelemetry_sink: Option<OpenTelemetrySink>,
}

impl Default for TelemetryOptions {
    fn default() -> Self {
        Self {
            request_log_enabled: false,
            http_trace_log_enabled: false,
            trace_buffer_capacity: 0,
            metric_prefix: DEFAULT_METRIC_PREFIX.to_string(),
            latency_buckets_ms: LATENCY_BUCKETS_MS.to_vec(),
            opentelemetry_sink: None,
        }
    }
}

struct MetricNames {
    active_requests: String,
    requests_started_total: String,
    requests_finished_total: String,
    prompt_tokens_total: String,
    cached_prompt_tokens_total: String,
    completion_tokens_total: String,
    queue_wait_ms: String,
    ttft_ms: String,
    request_duration_ms: String,
}

impl MetricNames {
    fn new(prefix: &str) -> Self {
        let prefix = if prefix.trim().is_empty() {
            DEFAULT_METRIC_PREFIX
        } else {
            prefix.trim()
        };
        Self {
            active_requests: format!("{prefix}_active_requests"),
            requests_started_total: format!("{prefix}_requests_started_total"),
            requests_finished_total: format!("{prefix}_requests_finished_total"),
            prompt_tokens_total: format!("{prefix}_prompt_tokens_total"),
            cached_prompt_tokens_total: format!("{prefix}_cached_prompt_tokens_total"),
            completion_tokens_total: format!("{prefix}_completion_tokens_total"),
            queue_wait_ms: format!("{prefix}_queue_wait_ms"),
            ttft_ms: format!("{prefix}_ttft_ms"),
            request_duration_ms: format!("{prefix}_request_duration_ms"),
        }
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    pub fn new() -> Self {
        Self::with_options(TelemetryOptions {
            request_log_enabled: env_flag("OPENINFER_TELEMETRY_LOG"),
            http_trace_log_enabled: env_flag("OPENINFER_HTTP_TRACE"),
            trace_buffer_capacity: env_usize("OPENINFER_TRACE_BUFFER").unwrap_or(0),
            ..TelemetryOptions::default()
        })
    }

    pub fn with_options(options: TelemetryOptions) -> Self {
        let TelemetryOptions {
            request_log_enabled,
            http_trace_log_enabled,
            trace_buffer_capacity,
            metric_prefix,
            latency_buckets_ms,
            opentelemetry_sink,
        } = options;
        let latency_buckets_ms = normalize_latency_buckets(latency_buckets_ms);
        Self {
            inner: Arc::new(Inner {
                request_log_enabled,
                http_trace_log_enabled,
                trace_buffer_capacity,
                trace_buffer: (trace_buffer_capacity > 0)
                    .then(|| Mutex::new(VecDeque::with_capacity(trace_buffer_capacity))),
                opentelemetry_sink,
                active_requests: AtomicI64::new(0),
                started_requests: AtomicU64::new(0),
                finished_requests: std::array::from_fn(|_| AtomicU64::new(0)),
                prompt_tokens: AtomicU64::new(0),
                cached_prompt_tokens: AtomicU64::new(0),
                completion_tokens: AtomicU64::new(0),
                metrics: MetricNames::new(&metric_prefix),
                queue_wait_ms: Histogram::new(&latency_buckets_ms),
                ttft_ms: Histogram::new(&latency_buckets_ms),
                request_duration_ms: Histogram::new(&latency_buckets_ms),
            }),
        }
    }

    pub(crate) fn request_started(&self) {
        self.inner.started_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_rejected(&self, prompt_tokens: usize) {
        self.request_started();
        self.request_finished(
            RequestOutcome::Rejected,
            RequestMetrics {
                prompt_tokens,
                ..RequestMetrics::default()
            },
        );
    }

    pub(crate) fn trace_enabled(&self) -> bool {
        self.inner.request_log_enabled
            || self.inner.http_trace_log_enabled
            || self.inner.trace_buffer_capacity > 0
            || self.inner.opentelemetry_sink.is_some()
    }

    pub(crate) fn record_trace(&self, trace: Value) {
        if self.inner.request_log_enabled {
            tracing::info!("openinfer_request_log {trace}");
        }
        if self.inner.http_trace_log_enabled {
            tracing::info!("openinfer_http_trace {trace}");
        }
        if let Some(sink) = &self.inner.opentelemetry_sink {
            sink.enqueue_trace(&trace);
        }
        if let Some(buffer) = &self.inner.trace_buffer {
            if let Ok(mut buffer) = buffer.lock() {
                if buffer.len() == self.inner.trace_buffer_capacity {
                    buffer.pop_front();
                }
                buffer.push_back(trace);
            }
        }
    }

    pub(crate) fn request_finished(&self, outcome: RequestOutcome, metrics: RequestMetrics) {
        self.inner.finished_requests[outcome.index()].fetch_add(1, Ordering::Relaxed);
        self.inner.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.inner
            .prompt_tokens
            .fetch_add(metrics.prompt_tokens as u64, Ordering::Relaxed);
        self.inner
            .cached_prompt_tokens
            .fetch_add(metrics.cached_prompt_tokens as u64, Ordering::Relaxed);
        self.inner
            .completion_tokens
            .fetch_add(metrics.completion_tokens as u64, Ordering::Relaxed);

        if let Some(value) = metrics.queue_wait_ms {
            self.inner.queue_wait_ms.record(value);
        }
        if let Some(value) = metrics.ttft_ms {
            self.inner.ttft_ms.record(value);
        }
        if let Some(value) = metrics.duration_ms {
            self.inner.request_duration_ms.record(value);
        }
    }

    pub(crate) fn record_request(&self, record: RequestRecord) {
        self.request_finished(record.outcome, record.metrics);
        if let Some(trace) = record.trace {
            self.record_trace(trace);
        }
    }

    pub(crate) fn finish_request(
        &self,
        trace: &RequestTrace,
        outcome: RequestOutcome,
        terminal_at_unix_s: f64,
    ) {
        self.record_request(trace.finish(outcome, terminal_at_unix_s, self.trace_enabled()));
    }

    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        render_gauge(
            &mut out,
            &self.inner.metrics.active_requests,
            "Requests currently tracked by the local frontend bridge.",
            self.inner.active_requests.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            &self.inner.metrics.requests_started_total,
            "Requests accepted by the local frontend bridge.",
            self.inner.started_requests.load(Ordering::Relaxed),
        );

        let _ = writeln!(
            out,
            "# HELP {} Requests terminated by outcome.",
            self.inner.metrics.requests_finished_total
        );
        let _ = writeln!(
            out,
            "# TYPE {} counter",
            self.inner.metrics.requests_finished_total
        );
        for outcome in RequestOutcome::ALL {
            let _ = writeln!(
                out,
                "{}{{outcome=\"{}\"}} {}",
                self.inner.metrics.requests_finished_total,
                outcome.label(),
                self.inner.finished_requests[outcome.index()].load(Ordering::Relaxed)
            );
        }

        render_counter(
            &mut out,
            &self.inner.metrics.prompt_tokens_total,
            "Prompt tokens observed at request termination.",
            self.inner.prompt_tokens.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            &self.inner.metrics.cached_prompt_tokens_total,
            "Prompt tokens reported as prefix-cache hits.",
            self.inner.cached_prompt_tokens.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            &self.inner.metrics.completion_tokens_total,
            "Completion tokens emitted by terminated requests.",
            self.inner.completion_tokens.load(Ordering::Relaxed),
        );
        render_histogram(
            &mut out,
            &self.inner.metrics.queue_wait_ms,
            "Milliseconds between engine queue and scheduler admission.",
            &self.inner.queue_wait_ms,
        );
        render_histogram(
            &mut out,
            &self.inner.metrics.ttft_ms,
            "Milliseconds between engine queue and first token emitted by the bridge.",
            &self.inner.ttft_ms,
        );
        render_histogram(
            &mut out,
            &self.inner.metrics.request_duration_ms,
            "Milliseconds between engine queue and terminal event.",
            &self.inner.request_duration_ms,
        );
        out
    }

    pub(crate) fn render_traces(&self) -> String {
        let traces: Vec<Value> = self
            .inner
            .trace_buffer
            .as_ref()
            .and_then(|buffer| {
                buffer
                    .lock()
                    .ok()
                    .map(|buffer| buffer.iter().cloned().collect())
            })
            .unwrap_or_default();
        json!({
            "enabled": self.inner.trace_buffer_capacity > 0,
            "capacity": self.inner.trace_buffer_capacity,
            "traces": traces,
        })
        .to_string()
    }
}

pub(crate) async fn guard_metrics_request(
    State(telemetry): State<Telemetry>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if req.uri().path() == "/metrics" {
        let response = next.run(req).await;
        let (mut parts, body) = response.into_parts();
        return match to_bytes(body, usize::MAX).await {
            Ok(bytes) => {
                let mut text = String::from_utf8_lossy(&bytes).into_owned();
                if !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&telemetry.render());
                parts
                    .headers
                    .insert(CONTENT_TYPE, HeaderValue::from_static(PROMETHEUS_TEXT));
                Response::from_parts(parts, Body::from(text))
            }
            Err(_) => ([(CONTENT_TYPE, PROMETHEUS_TEXT)], telemetry.render()).into_response(),
        };
    }
    next.run(req).await
}

pub(crate) fn traces_router(telemetry: Telemetry) -> Router {
    Router::new()
        .route("/openinfer/traces", get(traces_handler))
        .with_state(telemetry)
}

async fn traces_handler(State(telemetry): State<Telemetry>) -> impl IntoResponse {
    ([(CONTENT_TYPE, JSON)], telemetry.render_traces())
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|value| !matches!(value.as_str(), "" | "0" | "false" | "False" | "FALSE"))
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn normalize_latency_buckets(mut buckets: Vec<f64>) -> Vec<f64> {
    buckets.retain(|bucket| bucket.is_finite() && *bucket >= 0.0);
    buckets.sort_by(|left, right| left.partial_cmp(right).unwrap());
    buckets.dedup_by(|left, right| left == right);
    if buckets.is_empty() {
        LATENCY_BUCKETS_MS.to_vec()
    } else {
        buckets
    }
}

struct Histogram {
    buckets: Vec<f64>,
    bucket_hits: Vec<AtomicU64>,
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl Histogram {
    fn new(buckets: &[f64]) -> Self {
        Self {
            buckets: buckets.to_vec(),
            bucket_hits: buckets.iter().map(|_| AtomicU64::new(0)).collect(),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    fn record(&self, value_ms: f64) {
        if !value_ms.is_finite() || value_ms < 0.0 {
            return;
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add((value_ms * 1_000.0).round() as u64, Ordering::Relaxed);
        if let Some(index) = self.buckets.iter().position(|bucket| value_ms <= *bucket) {
            self.bucket_hits[index].fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn render_counter(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    let _ = writeln!(out, "{name} {value}");
}

fn render_gauge(out: &mut String, name: &str, help: &str, value: i64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name} {value}");
}

fn render_histogram(out: &mut String, name: &str, help: &str, histogram: &Histogram) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} histogram");

    let mut cumulative = 0;
    for (bucket, hits) in histogram.buckets.iter().zip(&histogram.bucket_hits) {
        cumulative += hits.load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "{name}_bucket{{le=\"{}\"}} {cumulative}",
            bucket_label(*bucket)
        );
    }
    let count = histogram.count.load(Ordering::Relaxed);
    let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {count}");
    let _ = writeln!(
        out,
        "{name}_sum {:.3}",
        histogram.sum_micros.load(Ordering::Relaxed) as f64 / 1_000.0
    );
    let _ = writeln!(out, "{name}_count {count}");
}

fn bucket_label(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use tower::ServiceExt;

    use super::*;

    #[test]
    fn telemetry_renders_prometheus_counters_and_histograms() {
        let telemetry = Telemetry::new();
        telemetry.request_started();
        telemetry.request_finished(
            RequestOutcome::Length,
            RequestMetrics {
                queue_wait_ms: Some(3.0),
                ttft_ms: Some(12.5),
                duration_ms: Some(42.0),
                prompt_tokens: 11,
                cached_prompt_tokens: 4,
                completion_tokens: 2,
            },
        );

        let text = telemetry.render();
        assert!(text.contains("openinfer_frontend_active_requests 0"));
        assert!(text.contains("openinfer_frontend_requests_started_total 1"));
        assert!(text.contains("openinfer_frontend_requests_finished_total{outcome=\"length\"} 1"));
        assert!(text.contains("openinfer_frontend_prompt_tokens_total 11"));
        assert!(text.contains("openinfer_frontend_cached_prompt_tokens_total 4"));
        assert!(text.contains("openinfer_frontend_completion_tokens_total 2"));
        assert!(text.contains("openinfer_frontend_ttft_ms_count 1"));
    }

    #[test]
    fn telemetry_options_customize_metric_prefix_and_latency_buckets() {
        let telemetry = Telemetry::with_options(TelemetryOptions {
            metric_prefix: "tenant_infer".to_string(),
            latency_buckets_ms: vec![7.0, 1.0, 7.0],
            ..TelemetryOptions::default()
        });

        telemetry.request_started();
        telemetry.request_finished(
            RequestOutcome::Stop,
            RequestMetrics {
                ttft_ms: Some(6.0),
                ..RequestMetrics::default()
            },
        );

        let text = telemetry.render();
        assert!(text.contains("tenant_infer_active_requests 0"));
        assert!(text.contains("tenant_infer_ttft_ms_bucket{le=\"1\"} 0"));
        assert!(text.contains("tenant_infer_ttft_ms_bucket{le=\"7\"} 1"));
        assert!(text.contains("tenant_infer_ttft_ms_count 1"));
        assert!(!text.contains("openinfer_frontend_active_requests"));
    }

    #[tokio::test]
    async fn telemetry_exports_traces_to_builtin_otlp_sink() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let telemetry = Telemetry::with_options(TelemetryOptions {
            opentelemetry_sink: Some(OpenTelemetrySink::new(
                sender,
                OpenTelemetryOptions {
                    service_name: "openinfer-test".to_string(),
                },
            )),
            ..TelemetryOptions::default()
        });

        assert!(telemetry.trace_enabled());
        telemetry.record_trace(json!({
            "request_id":"req-1",
            "queued_at_unix_s": 1.0,
            "terminal_at_unix_s": 1.020,
            "finish_reason": "stop",
        }));

        let payload = receiver.try_recv().unwrap();
        assert_eq!(
            payload["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"],
            "openinfer-test"
        );
    }

    #[test]
    fn trace_buffer_keeps_latest_request_traces() {
        let telemetry = Telemetry::with_options(TelemetryOptions {
            trace_buffer_capacity: 2,
            ..TelemetryOptions::default()
        });
        telemetry.record_trace(json!({"request_id":"old"}));
        telemetry.record_trace(json!({"request_id":"new-a"}));
        telemetry.record_trace(json!({"request_id":"new-b"}));

        let traces: Value = serde_json::from_str(&telemetry.render_traces()).unwrap();
        assert_eq!(traces["enabled"], true);
        assert_eq!(traces["capacity"], 2);
        assert_eq!(traces["traces"].as_array().unwrap().len(), 2);
        assert!(!traces.to_string().contains("old"));
        assert!(traces.to_string().contains("new-a"));
        assert!(traces.to_string().contains("new-b"));
    }

    #[tokio::test]
    async fn metrics_route_returns_text_exposition() {
        let telemetry = Telemetry::new();
        telemetry.request_started();
        let router = Router::new()
            .route("/metrics", get(|| async { "vllm metrics" }))
            .layer(from_fn_with_state(telemetry, guard_metrics_request));

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            PROMETHEUS_TEXT
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("vllm metrics"));
        assert!(text.contains("openinfer_frontend_active_requests 1"));
    }

    #[tokio::test]
    async fn traces_route_returns_buffered_traces() {
        let telemetry = Telemetry::with_options(TelemetryOptions {
            trace_buffer_capacity: 4,
            ..TelemetryOptions::default()
        });
        telemetry.record_trace(json!({"request_id":"req-1"}));
        let router = traces_router(telemetry);

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/openinfer/traces")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers().get(CONTENT_TYPE).unwrap(), JSON);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let traces: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(traces["traces"][0]["request_id"], "req-1");
    }
}
