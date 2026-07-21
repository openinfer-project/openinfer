//! Engine-level Pin gate: a request's emitted token ids and per-token logprob bits must be
//! identical alone and co-batched. A `TokenEvent` stream carries no step composition, so the
//! per-phase guards below only assert the composition under test happened at all — step and chunk
//! assignment is gated in the scheduler's unit tests, which read the assignment itself.
//!
//! Needs one CUDA GPU, `OPENINFER_TEST_MODEL_PATH`, and `--test-threads=1`.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU8;

use openinfer_core::engine::EngineHandle;
use openinfer_core::engine::EngineLoadOptions;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::RequestAbortReason;
use openinfer_core::engine::RequestTag;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::TokenSink;
use openinfer_core::engine::TokenStreamReceiver;
use openinfer_core::engine::TokenStreamSender;
use openinfer_core::sampler::SamplingParams;
use openinfer_kernels::ops::NumericPolicy;
use openinfer_kernels::ops::numeric_policy;
use tokio::sync::mpsc;

const A_TOKENS: usize = 64;
const LOAD_COUNT: usize = 40;
const BURST_DEPTH: usize = 3;
const MIN_BURSTS_IN_WINDOW: usize = 8;
type Sample = (u32, Option<f32>);

static POLICY_LOCK: Mutex<()> = Mutex::new(());

#[derive(Default)]
struct Trace {
    samples: Vec<Sample>,
    token_orders: Vec<usize>,
    scheduled_orders: Vec<usize>,
    terminal: bool,
}

impl Trace {
    fn token_before(&self, order: usize) -> bool {
        self.token_orders.iter().any(|&candidate| candidate < order)
    }

    fn token_after(&self, order: usize) -> bool {
        self.token_orders.iter().any(|&candidate| candidate > order)
    }

    fn scheduled_between(&self, start: usize, end: usize) -> bool {
        self.scheduled_orders
            .iter()
            .any(|&candidate| candidate > start && candidate < end)
    }
}

struct Harness {
    handle: EngineHandle,
    tx: TokenStreamSender,
    rx: TokenStreamReceiver,
    traces: HashMap<RequestTag, Trace>,
    order: usize,
}

impl Harness {
    fn new(handle: EngineHandle) -> Self {
        let (tx, rx): (TokenStreamSender, TokenStreamReceiver) = mpsc::unbounded_channel();
        Self {
            handle,
            tx,
            rx,
            traces: HashMap::new(),
            order: 0,
        }
    }

    fn submit(
        &mut self,
        tag: impl Into<String>,
        prompt_tokens: Vec<u32>,
        output: (usize, usize),
    ) -> (RequestTag, usize) {
        self.drain();
        let cutoff = self.order;
        let request_id = tag.into();
        let tag: RequestTag = Arc::from(request_id.as_str());
        assert!(self.traces.insert(tag.clone(), Trace::default()).is_none());
        let abort = Arc::new(AtomicU8::new(RequestAbortReason::None as u8));
        let token_tx = TokenSink::new(tag.clone(), self.tx.clone(), abort);
        self.handle
            .submit(GenerateRequest {
                request_id: Some(request_id),
                queued_at_unix_s: None,
                data_parallel_rank: None,
                prompt_tokens,
                params: SamplingParams {
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
                max_tokens: output.0,
                lora_adapter: None,
                token_tx,
                logprobs: output.1,
                echo: false,
            })
            .expect("submit failed");
        (tag, cutoff)
    }

    fn dispatch(&mut self, tag: &RequestTag, event: TokenEvent) {
        self.order += 1;
        let trace = self
            .traces
            .get_mut(tag)
            .expect("event for unknown request tag");
        match event {
            TokenEvent::Scheduled { .. } => trace.scheduled_orders.push(self.order),
            TokenEvent::Token { id, logprob } => {
                trace.samples.push((id, logprob.map(|value| value.logprob)));
                trace.token_orders.push(self.order);
            }
            TokenEvent::Finished { .. } => trace.terminal = true,
            TokenEvent::PromptTokens { .. } => {}
            TokenEvent::Error { message, .. } => panic!("request {tag} failed: {message}"),
            TokenEvent::Rejected { message, .. } => panic!("request {tag} rejected: {message}"),
        }
    }

    fn recv(&mut self) {
        let (tag, event) = self
            .rx
            .blocking_recv()
            .expect("engine event channel closed");
        self.dispatch(&tag, event);
    }

    fn drain(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok((tag, event)) => self.dispatch(&tag, event),
                Err(mpsc::error::TryRecvError::Empty) => return,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    panic!("engine event channel closed")
                }
            }
        }
    }

    fn barrier(&mut self) {
        self.drain();
        while self.traces.values().any(|trace| !trace.terminal) {
            self.recv();
        }
    }

    fn collect(&mut self, tag: &RequestTag) -> Vec<Sample> {
        while !self.trace(tag).terminal {
            self.recv();
        }
        self.trace(tag).samples.clone()
    }

    fn trace(&self, tag: &RequestTag) -> &Trace {
        self.traces.get(tag).expect("missing request trace")
    }
}

fn model_path_or_skip() -> Option<String> {
    if let Ok(path) = std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Some(path)
    } else {
        eprintln!("skip batch_invariance_output: set OPENINFER_TEST_MODEL_PATH to Qwen3-4B-base");
        None
    }
}

fn start_pin_engine(model_path: &str) -> EngineHandle {
    let handle = openinfer_qwen3::start_engine_with_offload(
        Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        openinfer_qwen3::Qwen3OffloadOptions::disabled(),
        true,
        openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
        openinfer_qwen3::Qwen3MemoryOptions::default(),
        openinfer_qwen3::DecodeOverlap::Off,
        true,
        None,
        false,
    )
    .expect("failed to start engine");
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Pin,
        "--batch-invariant did not select Pin"
    );
    handle
}

fn prompt(len: usize, row: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i + row) % 1000 + 10).collect()
}

fn first_divergence(expected: &[Sample], actual: &[Sample]) -> Option<usize> {
    let common = expected.len().min(actual.len());
    (0..common)
        .find(|&i| {
            expected[i].0 != actual[i].0
                || expected[i].1.map(f32::to_bits) != actual[i].1.map(f32::to_bits)
        })
        .or((expected.len() != actual.len()).then_some(common))
}

fn status(expected: &[Sample], actual: &[Sample]) -> String {
    first_divergence(expected, actual).map_or_else(
        || "identical".into(),
        |index| format!("first-divergence={index}"),
    )
}

fn report_divergence(label: &str, expected: &[Sample], actual: &[Sample]) {
    let Some(index) = first_divergence(expected, actual) else {
        return;
    };
    let start = index.saturating_sub(4);
    let end = (start + 8).min(expected.len().max(actual.len()));
    let expected_window = &expected[start.min(expected.len())..end.min(expected.len())];
    let actual_window = &actual[start.min(actual.len())..end.min(actual.len())];
    let expected_lp = expected.get(index).and_then(|sample| sample.1);
    let actual_lp = actual.get(index).and_then(|sample| sample.1);
    let delta = expected_lp.zip(actual_lp).map(|(left, right)| right - left);
    eprintln!(
        "[{label}] first-divergence={index} expected-window={expected_window:?} actual-window={actual_window:?} logprob-delta={delta:?} expected-logprob={expected_lp:?} actual-logprob={actual_lp:?}"
    );
}

fn assert_same(label: &str, expected: &[Sample], actual: &[Sample]) {
    report_divergence(label, expected, actual);
    let expected_ids: Vec<_> = expected.iter().map(|sample| sample.0).collect();
    let actual_ids: Vec<_> = actual.iter().map(|sample| sample.0).collect();
    assert_eq!(
        actual_ids, expected_ids,
        "{label}: token-id sequence drifted"
    );
    let expected_bits: Vec<_> = expected
        .iter()
        .map(|sample| sample.1.map(f32::to_bits))
        .collect();
    let actual_bits: Vec<_> = actual
        .iter()
        .map(|sample| sample.1.map(f32::to_bits))
        .collect();
    assert_eq!(
        actual_bits, expected_bits,
        "{label}: per-token logprob bits drifted"
    );
}

fn assert_probe_shape(label: &str, samples: &[Sample]) {
    assert_eq!(samples.len(), A_TOKENS, "{label}: wrong output length");
    assert!(
        samples.iter().all(|sample| sample.1.is_some()),
        "{label}: requested logprob is missing"
    );
}

fn phase_zero(harness: &mut Harness) -> Vec<Sample> {
    let (first_tag, _) = harness.submit("p0-a-1", prompt(600, 1), (A_TOKENS, 1));
    let first = harness.collect(&first_tag);
    let (second_tag, _) = harness.submit("p0-a-2", prompt(600, 1), (A_TOKENS, 1));
    let second = harness.collect(&second_tag);
    assert_probe_shape("phase 0 first run", &first);
    assert_probe_shape("phase 0 second run", &second);
    let divergence = first_divergence(&first, &second);
    eprintln!(
        "[phase 0 determinism] seq_len={} {}",
        second.len(),
        status(&first, &second)
    );
    if divergence.is_some() {
        report_divergence("phase 0 determinism", &first, &second);
        panic!("phase 0: output changed across identical runs; the gate is ill-defined in-process");
    }
    first
}

fn wait_for_load_tokens(harness: &mut Harness, loads: &[RequestTag]) {
    while loads
        .iter()
        .any(|tag| harness.trace(tag).samples.is_empty())
    {
        harness.recv();
    }
}

fn load_span(
    harness: &mut Harness,
    loads: &[RequestTag],
    window: (usize, usize),
) -> (usize, usize) {
    while loads.iter().any(|tag| {
        let trace = harness.trace(tag);
        !trace.terminal && !trace.token_after(window.1)
    }) {
        harness.recv();
    }
    let before = loads
        .iter()
        .filter(|tag| harness.trace(tag).token_before(window.0))
        .count();
    let after = loads
        .iter()
        .filter(|tag| harness.trace(tag).token_after(window.1))
        .count();
    (before, after)
}

fn phase_one(harness: &mut Harness, baseline: &[Sample]) {
    let loads: Vec<_> = (0..LOAD_COUNT)
        .map(|i| {
            harness
                .submit(
                    format!("p1-load-{i}"),
                    prompt(550 + i * 2, 100 + i as u32),
                    (192, 0),
                )
                .0
        })
        .collect();
    wait_for_load_tokens(harness, &loads);
    let (a_tag, _) = harness.submit("p1-a", prompt(600, 1), (A_TOKENS, 1));
    let actual = harness.collect(&a_tag);
    assert_probe_shape("phase 1 A", &actual);
    let first = harness.trace(&a_tag).token_orders[0];
    let last = *harness.trace(&a_tag).token_orders.last().unwrap();
    let (before, after) = load_span(harness, &loads, (first, last));
    assert_eq!(
        before, LOAD_COUNT,
        "phase 1 non-vacuity: only {before}/{LOAD_COUNT} loads emitted before A's first token"
    );
    assert_eq!(
        after, LOAD_COUNT,
        "phase 1 non-vacuity: only {after}/{LOAD_COUNT} loads emitted after A's last token"
    );
    eprintln!(
        "[phase 1 decode-heavy] seq_len={} {} loads-before={before} loads-after={after}",
        actual.len(),
        status(baseline, &actual)
    );
    assert_same(
        "phase 1 unified-prefill first token",
        &baseline[..1],
        &actual[..1],
    );
    assert_same("phase 1 decode suffix", &baseline[1..], &actual[1..]);
}

fn submit_burst(harness: &mut Harness, id: usize) -> RequestTag {
    harness
        .submit(
            format!("p2-burst-{id}"),
            prompt(120 + id, 300 + id as u32),
            (4, 0),
        )
        .0
}

fn phase_two(harness: &mut Harness, baseline: &[Sample]) {
    let (a_tag, _) = harness.submit("p2-a", prompt(600, 1), (A_TOKENS, 1));
    while harness.trace(&a_tag).samples.is_empty() {
        harness.recv();
    }
    let mut bursts: Vec<_> = (0..BURST_DEPTH)
        .map(|id| submit_burst(harness, id))
        .collect();
    let mut advanced = HashSet::new();
    let mut next = BURST_DEPTH;
    while !harness.trace(&a_tag).terminal {
        harness.recv();
        let ready = bursts.iter().position(|tag| {
            !advanced.contains(tag)
                && (!harness.trace(tag).scheduled_orders.is_empty() || harness.trace(tag).terminal)
        });
        if let Some(index) = ready {
            let tag = bursts[index].clone();
            advanced.insert(tag);
            harness.drain();
            if !harness.trace(&a_tag).terminal {
                bursts.push(submit_burst(harness, next));
                next += 1;
            }
        }
    }
    let actual = harness.trace(&a_tag).samples.clone();
    assert_probe_shape("phase 2 A", &actual);
    let first = harness.trace(&a_tag).token_orders[0];
    let last = *harness.trace(&a_tag).token_orders.last().unwrap();
    let inside = bursts
        .iter()
        .flat_map(|tag| &harness.trace(tag).scheduled_orders)
        .filter(|&&order| order > first && order < last)
        .count();
    assert!(
        inside >= MIN_BURSTS_IN_WINDOW,
        "phase 2 liveness: only {inside} burst prefills ran inside A's decode span — the \
         decode-under-prefill-load composition never materialized here"
    );
    eprintln!(
        "[phase 2 prefill-burst] seq_len={} {} scheduled-inside={inside}",
        actual.len(),
        status(baseline, &actual)
    );
    assert_same("phase 2 output", baseline, &actual);
}

fn phase_three(harness: &mut Harness) {
    // Phase 0's control prompt prefills in one chunk; this one does not. Re-establish determinism on
    // the chunked shape, or a wobble in the shape itself would read as batch-composition drift.
    let (base_tag, _) = harness.submit("p3-a-long-alone", prompt(1500, 77), (A_TOKENS, 1));
    let baseline = harness.collect(&base_tag);
    assert_probe_shape("phase 3 A_long baseline", &baseline);
    harness.barrier();
    let (repeat_tag, _) = harness.submit("p3-a-long-alone-2", prompt(1500, 77), (A_TOKENS, 1));
    let repeat = harness.collect(&repeat_tag);
    assert_probe_shape("phase 3 A_long control", &repeat);
    eprintln!(
        "[phase 3 control] A_long alone x2: {}",
        status(&baseline, &repeat)
    );
    assert_same("phase 3 A_long determinism control", &baseline, &repeat);
    harness.barrier();

    let (b1, _) = harness.submit("p3-b1", prompt(900, 91), (4, 0));
    let (b2, _) = harness.submit("p3-b2", prompt(900, 92), (4, 0));
    let (a_tag, cutoff) = harness.submit("p3-a-long-batched", prompt(1500, 77), (A_TOKENS, 1));
    let actual = harness.collect(&a_tag);
    assert_probe_shape("phase 3 A_long batched", &actual);
    // A's first token lands only once its prefill completes, so this window is A's queue+prefill
    // span: a B scheduled inside it prefilled while A was queued or prefilling — necessary for the
    // two to contend for one step's budget, not proof of it.
    let first = harness.trace(&a_tag).token_orders[0];
    let b_scheduled = [b1, b2]
        .into_iter()
        .filter(|tag| harness.trace(tag).scheduled_between(cutoff, first))
        .count();
    assert_eq!(
        b_scheduled, 2,
        "phase 3 liveness: only {b_scheduled}/2 B prefills overlapped A_long's prefill span — A_long \
         ran alone, so this phase exercised nothing"
    );
    eprintln!(
        "[phase 3 chunked-prefill] seq_len={} {} b-scheduled-before-first={b_scheduled}",
        actual.len(),
        status(&baseline, &actual)
    );
    assert_same("phase 3 output", &baseline, &actual);
}

#[test]
fn output_sequence_batch_invariant_under_pin() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let _guard = POLICY_LOCK.lock().unwrap();
    let handle = start_pin_engine(&model_path);
    let mut harness = Harness::new(handle);
    let baseline = phase_zero(&mut harness);
    phase_one(&mut harness, &baseline);
    harness.barrier();
    phase_two(&mut harness, &baseline);
    harness.barrier();
    phase_three(&mut harness);
    harness.barrier();
}
