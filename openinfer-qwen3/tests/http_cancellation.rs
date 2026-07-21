//! HTTP-level integration test for issue #642.
//!
//! `scheduler_robustness.rs` proves the scheduler retires an orphaned request
//! when its *engine* receiver is dropped, but it drives `EngineHandle` directly.
//! @xiaguan asked for the same guarantee through the **real HTTP disconnect
//! path**: a client hangs up on a streaming `/v1/completions` response and the
//! running server must tear that request down.
//!
//! This test starts the actual `openinfer` server binary as a subprocess,
//! opens a burst of concurrent streaming completions, verifies they show up as
//! in-flight in `/metrics`, disconnects every client at once, and polls
//! `/metrics` until running+waiting collapse to zero — then confirms a clean
//! follow-up request is still served.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and a built release binary at
//! `target/release/openinfer`. It skips cleanly when the model or binary is
//! absent; point `OPENINFER_TEST_MODEL_PATH` at the weights to run it.

use std::net::TcpListener;
use std::path::Path;
use std::process::Child;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use reqwest::blocking::Client;
use serde_json::Value;
use serde_json::json;

/// Path to the release server binary, relative to this crate's manifest dir.
const SERVER_BIN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/release/openinfer");

/// Default model weights location when `OPENINFER_TEST_MODEL_PATH` is unset.
const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

/// How long to wait for the server to finish loading the model and answer
/// `/metrics`. Loading Qwen3-4B on a consumer GPU takes ~30s; leave headroom.
const SERVER_READY_TIMEOUT: Duration = Duration::from_secs(180);

/// Number of concurrent streaming clients in the disconnect burst.
const NUM_CLIENTS: usize = 8;

/// Prompt length in repeated words. Long enough that prefill keeps every
/// request firmly in-flight at the moment we disconnect.
const PROMPT_WORDS: usize = 800;

/// How long to wait for in-flight + queued requests to drain to zero after the
/// mass disconnect.
const COLLAPSE_TIMEOUT: Duration = Duration::from_secs(15);

/// How long to wait for at least one request to enter the running set.
const INFLIGHT_TIMEOUT: Duration = Duration::from_secs(15);

/// `/metrics` polling cadence.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

// ---------------------------------------------------------------------------
// Skip guards
// ---------------------------------------------------------------------------

fn model_path_or_skip() -> Option<String> {
    if let Ok(path) = std::env::var("OPENINFER_TEST_MODEL_PATH") {
        return Some(path);
    }
    if Path::new(DEFAULT_MODEL_PATH).join("config.json").exists() {
        return Some(DEFAULT_MODEL_PATH.to_string());
    }
    eprintln!(
        "skipping http_cancellation: model weights not found at {DEFAULT_MODEL_PATH}; \
         set OPENINFER_TEST_MODEL_PATH to run it"
    );
    None
}

fn server_bin_or_skip() -> Option<&'static str> {
    if Path::new(SERVER_BIN).exists() {
        Some(SERVER_BIN)
    } else {
        eprintln!(
            "skipping http_cancellation: server binary not found at {SERVER_BIN}; \
             run `cargo build --release -p openinfer` first"
        );
        None
    }
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

/// Kills the child on drop so a failing assertion cannot orphan the server.
struct ServerGuard {
    child: Child,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Bind a loopback socket to grab a free ephemeral port, then release it so the
/// server can bind it. The inherent TOCTOU race is negligible on a test box and
/// this matches the rest of the suite.
fn reserve_loopback_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .expect("failed to reserve loopback port")
        .local_addr()
        .expect("failed to read reserved port")
        .port()
}

fn wait_for_server(client: &Client, base_url: &str) -> bool {
    let metrics_url = format!("{base_url}/metrics");
    let deadline = Instant::now() + SERVER_READY_TIMEOUT;
    while Instant::now() < deadline {
        if let Ok(resp) = client
            .get(&metrics_url)
            .timeout(Duration::from_secs(2))
            .send()
        {
            if resp.status().is_success() {
                return true;
            }
        }
        thread::sleep(Duration::from_millis(500));
    }
    false
}

// ---------------------------------------------------------------------------
// Metrics parsing
// ---------------------------------------------------------------------------

/// Sum the trailing numeric value of every non-comment metrics line that
/// contains `name` (across all labelled series / engines). Prometheus emits one
/// line per series, so summing is correct for a single-engine server and robust
/// to the `engine="0"` label format.
fn metric_total(metrics_text: &str, name: &str) -> i64 {
    metrics_text
        .lines()
        .filter(|line| !line.starts_with('#') && line.contains(name))
        .filter_map(|line| {
            line.rsplit_once(' ')
                .and_then(|(_, value)| value.trim().parse::<f64>().ok())
                .map(|value| value as i64)
        })
        .sum()
}

/// Returns `(running, waiting)` summed across engines, or `(-1, -1)` if the
/// metrics endpoint is temporarily unreachable.
fn metrics_counts(client: &Client, base_url: &str) -> (i64, i64) {
    let url = format!("{base_url}/metrics");
    let Ok(resp) = client.get(&url).timeout(Duration::from_secs(5)).send() else {
        return (-1, -1);
    };
    let Ok(text) = resp.text() else {
        return (-1, -1);
    };
    (
        metric_total(&text, "num_requests_running"),
        metric_total(&text, "num_requests_waiting"),
    )
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

/// Burst of streaming completions → mass disconnect → metrics collapse → clean
/// follow-up. See the module docs for the full rationale.
#[test]
fn http_disconnect_metrics_collapse() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let Some(server_bin) = server_bin_or_skip() else {
        return;
    };

    let port = reserve_loopback_port();
    let base_url = format!("http://127.0.0.1:{port}");

    eprintln!("=== starting {server_bin} --model-path {model_path} --port {port} ===");
    let child = Command::new(server_bin)
        .arg("--model-path")
        .arg(&model_path)
        .arg("--port")
        .arg(port.to_string())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn server binary: {e}"));
    // Killing `server` on drop guarantees cleanup even on assertion failure.
    let server = ServerGuard { child };

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to build reqwest client");

    // 1. Wait for the server to finish loading and answer /metrics.
    eprintln!(
        "=== waiting for server readiness (up to {}s) ===",
        SERVER_READY_TIMEOUT.as_secs()
    );
    assert!(
        wait_for_server(&client, &base_url),
        "server did not become ready within {}s",
        SERVER_READY_TIMEOUT.as_secs()
    );
    eprintln!("=== server ready ===");

    // 2. Discover the model id from /v1/models (defaults to the model path).
    let model_id: String = {
        let resp = client
            .get(format!("{base_url}/v1/models"))
            .timeout(Duration::from_secs(10))
            .send()
            .expect("GET /v1/models failed")
            .error_for_status()
            .expect("/v1/models returned non-2xx");
        let body: Value = resp.json().expect("/v1/models did not return JSON");
        body["data"][0]["id"]
            .as_str()
            .expect("/v1/models response missing data[0].id")
            .to_string()
    };
    eprintln!("=== model id: {model_id} ===");

    // 3. Build the streaming request body shared by every burst client.
    let burst_body = json!({
        "model": model_id.as_str(),
        "prompt": "Hello ".repeat(PROMPT_WORDS),
        "max_tokens": 128,
        "stream": true,
        "temperature": 0.7,
    });

    // 4. Launch the burst. Scoped threads borrow `client`/`base_url`/`burst_body`
    //    so only the per-client `Sender` and the shared stop flag need cloning.
    let disconnect_start = thread::scope(|s| -> Instant {
        let (tx, rx) = mpsc::channel::<Result<(), String>>();
        let stop = Arc::new(AtomicBool::new(false));

        for id in 0..NUM_CLIENTS {
            let tx = tx.clone();
            let stop = stop.clone();
            let client = &client;
            let base_url = &base_url;
            let body = &burst_body;
            s.spawn(move || {
                let url = format!("{base_url}/v1/completions");
                let resp = match client.post(&url).json(body).send() {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(Err(format!("client {id}: send failed: {e}")));
                        return;
                    }
                };
                // Headers received → the request is in-flight on the server.
                let _ = tx.send(Ok(()));
                // Hold the streaming response open until told to disconnect.
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(10));
                }
                // Dropping `resp` closes the TCP connection: the server's next
                // stream write fails and it retires the request.
                drop(resp);
            });
        }
        drop(tx); // workers still hold their clones; rx sees all results.

        // 4a. Wait for every client to confirm it has an open streaming response.
        let mut connected = 0usize;
        while connected < NUM_CLIENTS {
            #[allow(clippy::match_wild_err_arm)]
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(Ok(())) => connected += 1,
                Ok(Err(e)) => panic!("streaming client failed to connect: {e}"),
                Err(_) => panic!(
                    "timed out waiting for {NUM_CLIENTS} streaming clients to connect \
                     (only {connected} connected)"
                ),
            }
        }
        eprintln!("=== all {NUM_CLIENTS} streaming clients connected ===");

        // 4b. Confirm the burst actually entered the running set.
        let inflight_deadline = Instant::now() + INFLIGHT_TIMEOUT;
        loop {
            let (running, waiting) = metrics_counts(&client, &base_url);
            if running > 0 {
                eprintln!("=== in-flight: running={running} waiting={waiting} ===");
                break;
            }
            assert!(
                Instant::now() < inflight_deadline,
                "no requests entered the running set within {}s \
                 (last running={running} waiting={waiting})",
                INFLIGHT_TIMEOUT.as_secs()
            );
            thread::sleep(POLL_INTERVAL);
        }

        // 4c. Mass disconnect.
        eprintln!("=== disconnecting all clients ===");
        let disconnect_start = Instant::now();
        stop.store(true, Ordering::Relaxed);
        // Scope exit joins the workers, i.e. waits until every `resp` is dropped.
        disconnect_start
    });

    // 5. Poll /metrics until running+waiting collapse to zero.
    eprintln!("=== polling for metrics collapse ===");
    let collapse_deadline = disconnect_start + COLLAPSE_TIMEOUT;
    loop {
        let (running, waiting) = metrics_counts(&client, &base_url);
        if running == 0 && waiting == 0 {
            eprintln!(
                "=== collapsed to running=0 waiting=0 in {}ms ===",
                disconnect_start.elapsed().as_millis()
            );
            break;
        }
        assert!(
            Instant::now() < collapse_deadline,
            "metrics did not collapse within {}s after disconnect \
             (running={running} waiting={waiting})",
            COLLAPSE_TIMEOUT.as_secs()
        );
        thread::sleep(POLL_INTERVAL);
    }

    // 6. Clean follow-up request: the engine must still serve new work.
    eprintln!("=== follow-up non-streaming request ===");
    let followup = json!({
        "model": model_id.as_str(),
        "prompt": "Say hello",
        "max_tokens": 5,
        "stream": false,
    });
    let resp = client
        .post(format!("{base_url}/v1/completions"))
        .json(&followup)
        .timeout(Duration::from_secs(30))
        .send()
        .expect("follow-up request failed");
    let status = resp.status();
    let body: Value = resp
        .json()
        .unwrap_or_else(|e| panic!("follow-up response is not JSON ({e})"));
    assert!(
        status.is_success(),
        "follow-up request returned {status}: {body}"
    );
    let text = body["choices"][0]["text"]
        .as_str()
        .expect("follow-up response missing choices[0].text");
    assert!(!text.trim().is_empty(), "follow-up returned empty text");
    eprintln!("=== follow-up ok: status={status} text={text:?} ===");

    eprintln!("=== PASS: metrics collapsed after disconnect, follow-up served ===");
    // `server` dropped here → child killed.
    drop(server);
}
