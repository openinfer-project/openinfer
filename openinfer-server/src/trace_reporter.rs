//! File-based trace reporter for fastrace.
//!
//! Writes Chrome Trace Event Format JSON, one file per trace.
//! Open with `chrome://tracing` or <https://ui.perfetto.dev>.
//! Filename format: `{timestamp_ms}_{trace_id}.json`

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use fastrace::collector::{Reporter, SpanRecord};
use log::error;
use serde::Serialize;

pub struct FileReporter {
    output_dir: PathBuf,
}

impl FileReporter {
    pub fn new(output_dir: PathBuf) -> Self {
        Self { output_dir }
    }
}

/// Chrome Trace Event Format event.
/// Spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
#[derive(Serialize)]
struct TraceEvent {
    name: String,
    cat: String,
    ph: &'static str,
    /// Microseconds.
    ts: f64,
    /// Duration in microseconds (complete events only).
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>,
    pid: u64,
    tid: u64,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    args: HashMap<String, String>,
}

impl Reporter for FileReporter {
    fn report(&mut self, spans: Vec<SpanRecord>) {
        if spans.is_empty() {
            return;
        }

        let mut traces: HashMap<String, Vec<TraceEvent>> = HashMap::new();

        for span in spans {
            let trace_id = format!("{}", span.trace_id);

            let mut args = HashMap::new();
            args.insert("span_id".into(), format!("{}", span.span_id));
            args.insert("parent_id".into(), format!("{}", span.parent_id));
            for (k, v) in &span.properties {
                args.insert(k.to_string(), v.to_string());
            }

            let entry = traces.entry(trace_id).or_default();

            // Sub-span events → instant markers
            for e in &span.events {
                let mut evt_args = HashMap::new();
                for (k, v) in &e.properties {
                    evt_args.insert(k.to_string(), v.to_string());
                }
                entry.push(TraceEvent {
                    name: e.name.to_string(),
                    cat: "event".into(),
                    ph: "i",
                    ts: e.timestamp_unix_ns as f64 / 1000.0,
                    dur: None,
                    pid: 1,
                    tid: 1,
                    args: evt_args,
                });
            }

            // Span → complete event ("X")
            entry.push(TraceEvent {
                name: span.name.into_owned(),
                cat: "span".into(),
                ph: "X",
                ts: span.begin_time_unix_ns as f64 / 1000.0,
                dur: Some(span.duration_ns as f64 / 1000.0),
                pid: 1,
                tid: 1,
                args,
            });
        }

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        for (trace_id, mut events) in traces {
            events.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap());

            let filename = format!("{}_{}.json", timestamp_ms, trace_id);
            let path = self.output_dir.join(&filename);

            match serde_json::to_string_pretty(&events) {
                Ok(json) => {
                    if let Err(e) = std::fs::write(&path, json) {
                        error!("Failed to write trace file {}: {}", path.display(), e);
                    }
                }
                Err(e) => {
                    error!("Failed to serialize trace: {}", e);
                }
            }
        }
    }
}
