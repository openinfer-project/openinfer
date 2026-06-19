use serde_json::{Value, json};
use tokio::sync::mpsc;
use uuid::Uuid;

const DEFAULT_OTEL_SERVICE_NAME: &str = "openinfer";
const OTEL_SCOPE_NAME: &str = "openinfer-vllm-frontend";
const OTEL_REQUEST_SPAN_NAME: &str = "openinfer.request";

#[derive(Clone, Debug)]
pub struct OpenTelemetryOptions {
    pub service_name: String,
}

impl OpenTelemetryOptions {
    pub fn from_env() -> Self {
        Self {
            service_name: env_non_empty("OPENINFER_OTEL_SERVICE_NAME")
                .or_else(|| env_non_empty("OTEL_SERVICE_NAME"))
                .unwrap_or_else(|| DEFAULT_OTEL_SERVICE_NAME.to_string()),
        }
    }
}

impl Default for OpenTelemetryOptions {
    fn default() -> Self {
        Self {
            service_name: DEFAULT_OTEL_SERVICE_NAME.to_string(),
        }
    }
}

#[derive(Clone)]
pub struct OpenTelemetrySink {
    sender: mpsc::Sender<Value>,
    service_name: String,
}

impl OpenTelemetrySink {
    pub fn new(sender: mpsc::Sender<Value>, options: OpenTelemetryOptions) -> Self {
        Self {
            sender,
            service_name: options.service_name,
        }
    }

    pub(crate) fn enqueue_trace(&self, trace: &Value) {
        if let Ok(permit) = self.sender.try_reserve() {
            permit.send(opentelemetry_trace_payload(&self.service_name, trace));
        }
    }
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn opentelemetry_trace_payload(service_name: &str, trace: &Value) -> Value {
    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    otel_string_attr("service.name", service_name),
                    otel_string_attr("telemetry.sdk.language", "rust"),
                    otel_string_attr("telemetry.sdk.name", "openinfer"),
                ]
            },
            "scopeSpans": [{
                "scope": {"name": OTEL_SCOPE_NAME},
                "spans": [opentelemetry_request_span(trace)]
            }]
        }]
    })
}

fn opentelemetry_request_span(trace: &Value) -> Value {
    let start = trace
        .get("queued_at_unix_s")
        .and_then(Value::as_f64)
        .map(unix_s_to_nanos)
        .unwrap_or_else(|| "0".to_string());
    let end = trace
        .get("terminal_at_unix_s")
        .and_then(Value::as_f64)
        .map(unix_s_to_nanos)
        .unwrap_or_else(|| start.clone());
    let finish_reason = trace
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    json!({
        "traceId": Uuid::new_v4().simple().to_string(),
        "spanId": Uuid::new_v4().simple().to_string()[..16].to_string(),
        "name": OTEL_REQUEST_SPAN_NAME,
        "kind": 2,
        "startTimeUnixNano": start,
        "endTimeUnixNano": end,
        "attributes": opentelemetry_attributes(trace, finish_reason),
        "events": opentelemetry_events(trace),
        "status": {"code": opentelemetry_status_code(finish_reason)},
    })
}

fn opentelemetry_attributes(trace: &Value, finish_reason: &str) -> Vec<Value> {
    let mut attributes = Vec::new();
    attributes.push(otel_string_attr("openinfer.finish_reason", finish_reason));
    if let Some(request_id) = trace.get("request_id").and_then(Value::as_str) {
        attributes.push(otel_string_attr("openinfer.request_id", request_id));
    }
    for (name, key) in [
        ("openinfer.prompt_tokens", "prompt_tokens"),
        ("openinfer.cached_tokens", "cached_tokens"),
        ("openinfer.completion_tokens", "completion_tokens"),
    ] {
        if let Some(value) = trace.get(key).and_then(Value::as_u64) {
            attributes.push(otel_int_attr(name, value));
        }
    }
    if let Some(prefill_ms) = trace.get("prefill_ms").and_then(Value::as_f64) {
        attributes.push(otel_double_attr("openinfer.prefill_ms", prefill_ms));
    }
    attributes
}

fn opentelemetry_events(trace: &Value) -> Vec<Value> {
    let mut events = Vec::new();
    for (name, key) in [
        ("scheduled", "scheduled_at_unix_s"),
        ("first_token", "first_token_emit_unix_s"),
    ] {
        if let Some(timestamp) = trace.get(key).and_then(Value::as_f64) {
            events.push(json!({
                "name": name,
                "timeUnixNano": unix_s_to_nanos(timestamp),
            }));
        }
    }
    events
}

fn opentelemetry_status_code(finish_reason: &str) -> i32 {
    match finish_reason {
        "error" | "rejected" => 2,
        _ => 1,
    }
}

fn otel_string_attr(key: &str, value: &str) -> Value {
    json!({"key": key, "value": {"stringValue": value}})
}

fn otel_int_attr(key: &str, value: u64) -> Value {
    json!({"key": key, "value": {"intValue": value.to_string()}})
}

fn otel_double_attr(key: &str, value: f64) -> Value {
    json!({"key": key, "value": {"doubleValue": value}})
}

fn unix_s_to_nanos(value: f64) -> String {
    if !value.is_finite() || value <= 0.0 {
        return "0".to_string();
    }
    ((value * 1_000_000_000.0).round() as u64).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opentelemetry_payload_uses_otlp_trace_shape() {
        let payload = opentelemetry_trace_payload(
            "openinfer-test",
            &json!({
                "request_id": "req-1",
                "queued_at_unix_s": 1.0,
                "scheduled_at_unix_s": 1.001,
                "first_token_emit_unix_s": 1.010,
                "terminal_at_unix_s": 1.020,
                "finish_reason": "stop",
                "prompt_tokens": 11,
                "cached_tokens": 4,
                "completion_tokens": 2,
                "prefill_ms": 9.0,
            }),
        );

        let resource_span = &payload["resourceSpans"][0];
        assert_eq!(
            resource_span["resource"]["attributes"][0]["value"]["stringValue"],
            "openinfer-test"
        );
        let span = &resource_span["scopeSpans"][0]["spans"][0];
        assert_eq!(span["name"], OTEL_REQUEST_SPAN_NAME);
        assert_eq!(span["kind"], 2);
        assert_eq!(span["startTimeUnixNano"], "1000000000");
        assert_eq!(span["endTimeUnixNano"], "1020000000");
        assert_eq!(span["events"][0]["name"], "scheduled");
        assert_eq!(span["events"][1]["name"], "first_token");
        assert!(
            span["attributes"]
                .to_string()
                .contains("openinfer.request_id")
        );
        assert!(
            span["attributes"]
                .to_string()
                .contains("openinfer.prompt_tokens")
        );
    }

    #[tokio::test]
    async fn opentelemetry_sink_queues_payload_for_caller_exporter() {
        let (sender, mut receiver) = mpsc::channel(1);
        let sink = OpenTelemetrySink::new(
            sender,
            OpenTelemetryOptions {
                service_name: "openinfer-test".to_string(),
            },
        );

        let trace = json!({
            "request_id": "req-1",
            "queued_at_unix_s": 1.0,
            "terminal_at_unix_s": 1.020,
            "finish_reason": "stop",
        });
        sink.enqueue_trace(&trace);

        let payload = receiver.try_recv().unwrap();
        assert_eq!(
            payload["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"],
            "openinfer-test"
        );
    }
}
