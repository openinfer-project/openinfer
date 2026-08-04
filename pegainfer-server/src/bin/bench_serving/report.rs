//! Serializable report and metric types emitted by the benchmark runners.

use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunInfo {
    pub(crate) command: &'static str,
    pub(crate) model_path: String,
    pub(crate) model_type: String,
    pub(crate) cuda_graph: bool,
    pub(crate) load_ms: f64,
    pub(crate) label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PromptDescriptor {
    pub(crate) source: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) prompt_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DurationStats {
    pub(crate) avg_ms: f64,
    pub(crate) p50_ms: f64,
    pub(crate) p95_ms: f64,
    pub(crate) p99_ms: f64,
    pub(crate) max_ms: f64,
    pub(crate) samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CountStats {
    pub(crate) min: usize,
    pub(crate) max: usize,
    pub(crate) avg: f64,
    pub(crate) samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeneratedTokenTrace {
    pub(crate) hash: String,
    pub(crate) prefix: Vec<u32>,
    pub(crate) len: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RequestWorkload {
    pub(crate) prompt: PromptDescriptor,
    pub(crate) output_len: usize,
    pub(crate) concurrency: usize,
    pub(crate) warmup: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RequestMetrics {
    pub(crate) ttft_ms: DurationStats,
    pub(crate) first_decode_step_ms: Option<DurationStats>,
    pub(crate) steady_tpot_ms: Option<DurationStats>,
    pub(crate) e2e_ms: DurationStats,
    pub(crate) generated_tokens: CountStats,
    #[serde(default)]
    pub(crate) generated_token_traces: Vec<GeneratedTokenTrace>,
    pub(crate) request_tok_s: Option<f64>,
    pub(crate) decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RequestIterationTiming {
    pub(crate) index: usize,
    pub(crate) ttft_ms: f64,
    pub(crate) first_decode_step_ms: Option<f64>,
    pub(crate) steady_tpot_ms: Option<DurationStats>,
    pub(crate) e2e_ms: f64,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_trace: GeneratedTokenTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SnapshotProfile {
    pub(crate) prompt_len: usize,
    pub(crate) output_len: usize,
    pub(crate) metrics: RequestMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SnapshotReport {
    pub(crate) commit: String,
    pub(crate) date: String,
    pub(crate) model: String,
    pub(crate) gpu: String,
    /// Parallel layout the snapshot was measured under (e.g. "tp1-dp8-deepep").
    /// Absent in snapshots that predate multi-GPU model lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parallel: Option<String>,
    pub(crate) prefill_heavy: SnapshotProfile,
    pub(crate) decode_heavy: SnapshotProfile,
    /// Long cold prompt arriving into a decode-heavy steady state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) mixed_itl: Option<SnapshotMixedItl>,
}

/// Mixed-load ITL profile baked into a snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SnapshotMixedItl {
    pub(crate) config: MixedLoadConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) baseline_itl: Option<DurationStats>,
    pub(crate) itl: MixedLoadItl,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RequestReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: RequestWorkload,
    pub(crate) metrics: RequestMetrics,
    pub(crate) iterations: Vec<RequestIterationTiming>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PrefillWorkload {
    pub(crate) prompt_lens: Vec<usize>,
    pub(crate) batches: Vec<usize>,
    pub(crate) distinct_prompts: usize,
    pub(crate) warmup: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
    /// Total KV pool capacity in tokens the sweep was checked against (`None`
    /// if the model did not report it, in which case no capacity guard ran).
    pub(crate) kv_capacity_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PrefillCell {
    pub(crate) prompt_len: usize,
    pub(crate) batch: usize,
    /// `prompt_len × batch` — the KV the batch holds resident during prefill.
    pub(crate) total_tokens: usize,
    pub(crate) ttft_ms: DurationStats,
    /// Batch prefill throughput: `total_tokens / ttft.p50`.
    pub(crate) prefill_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PrefillReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: PrefillWorkload,
    pub(crate) cells: Vec<PrefillCell>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct DecodeWorkload {
    pub(crate) ctxs: Vec<usize>,
    pub(crate) batches: Vec<usize>,
    pub(crate) decode_steps: usize,
    pub(crate) warmup_steps: usize,
    pub(crate) distinct_prompts: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
    /// Total KV pool capacity in tokens the sweep was checked against (`None`
    /// if the model did not report it, in which case no capacity guard ran).
    pub(crate) kv_capacity_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct DecodeCell {
    pub(crate) ctx: usize,
    pub(crate) batch: usize,
    pub(crate) decode_steps: usize,
    /// `batch × (ctx + decode_steps)` — peak resident KV during the decode.
    pub(crate) peak_tokens: usize,
    /// Steady-state per-token decode latency (prefill served from cache, so it
    /// is excluded; the leading `warmup_steps` are dropped).
    pub(crate) tpot_ms: DurationStats,
    /// Aggregate decode throughput: `batch / tpot.p50`.
    pub(crate) decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct DecodeReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: DecodeWorkload,
    pub(crate) cells: Vec<DecodeCell>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MatrixWorkload {
    pub(crate) prompt_lens: Vec<usize>,
    pub(crate) output_lens: Vec<usize>,
    pub(crate) warmup: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
    pub(crate) synthetic_pattern: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MatrixCell {
    pub(crate) prompt_len: usize,
    pub(crate) output_len: usize,
    pub(crate) ttft_ms: DurationStats,
    pub(crate) e2e_ms: DurationStats,
    pub(crate) first_decode_step_ms: Option<DurationStats>,
    pub(crate) steady_tpot_ms: Option<DurationStats>,
    pub(crate) generated_tokens: CountStats,
    pub(crate) request_tok_s: Option<f64>,
    pub(crate) decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MatrixReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: MatrixWorkload,
    pub(crate) cells: Vec<MatrixCell>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CurveWorkload {
    pub(crate) prompt: PromptDescriptor,
    pub(crate) output_len: usize,
    pub(crate) window: usize,
    pub(crate) warmup: usize,
    pub(crate) iters: usize,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CurveWindow {
    pub(crate) ctx_start: usize,
    pub(crate) ctx_end: usize,
    pub(crate) tpot_ms: DurationStats,
    pub(crate) decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CurveReport {
    pub(crate) run: RunInfo,
    pub(crate) workload: CurveWorkload,
    pub(crate) windows: Vec<CurveWindow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MixedLoadConfig {
    pub(crate) bg_prompt_len: usize,
    pub(crate) bg_concurrency: usize,
    pub(crate) bg_output_len: usize,
    pub(crate) inj_prompt_len: usize,
    pub(crate) inj_output_len: usize,
    pub(crate) qps: f64,
    pub(crate) num_injections: usize,
    pub(crate) inj_warm_frac: f64,
    pub(crate) warmup: usize,
    pub(crate) seed: u64,
    /// Scheduler concurrent-request cap (`--max-batch`) the engine was built
    /// with. Core to reproducing the #470 matrix (e.g. 5 leaves one free slot
    /// for the injector above `bg_concurrency=4`). `0` in pre-#470 snapshots.
    #[serde(default)]
    pub(crate) max_batch: usize,
    /// Per-step chunked-prefill token budget (`--max-prefill-tokens`). `None`
    /// means the model default (chunking on); a huge value means chunking off.
    #[serde(default)]
    pub(crate) max_prefill_tokens: Option<usize>,
}

/// Inter-token-latency of the background decode streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MixedLoadItl {
    /// Every background decode gap.
    pub(crate) all: DurationStats,
    /// Gaps with no overlapping injection window (decode unaffected by prefill).
    pub(crate) steady: Option<DurationStats>,
    /// Gaps overlapping an in-flight prefill (the unified-step stall tail).
    pub(crate) stall: Option<DurationStats>,
    pub(crate) stall_gap_count: usize,
    pub(crate) total_gap_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct InjectionRecord {
    pub(crate) index: usize,
    /// Whether this injection reused the shared prompt (intended prefix-cache hit).
    pub(crate) warm: bool,
    pub(crate) prefill_ms: f64,
    pub(crate) arrival_offset_ms: f64,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_trace: GeneratedTokenTrace,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MixedDecisionInputs {
    pub(crate) baseline_p50_ms: Option<f64>,
    pub(crate) baseline_p99_ms: Option<f64>,
    pub(crate) mixed_p50_ms: f64,
    pub(crate) mixed_p99_ms: f64,
    pub(crate) p99_delta_ms: Option<f64>,
    pub(crate) p99_delta_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MixedLoadReport {
    pub(crate) commit: String,
    pub(crate) date: String,
    pub(crate) gpu: String,
    pub(crate) run: RunInfo,
    pub(crate) config: MixedLoadConfig,
    pub(crate) background_generated_tokens: CountStats,
    pub(crate) background_generated_token_traces: Vec<GeneratedTokenTrace>,
    pub(crate) baseline_itl: Option<DurationStats>,
    pub(crate) mixed_itl: MixedLoadItl,
    pub(crate) injections: Vec<InjectionRecord>,
    pub(crate) decision_inputs: MixedDecisionInputs,
    /// Non-fatal measurement caveats (e.g. a background stream finished early).
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BenchReport {
    Request(Box<RequestReport>),
    Prefill(PrefillReport),
    Decode(DecodeReport),
    Matrix(MatrixReport),
    Curve(CurveReport),
    Mixed(Box<MixedLoadReport>),
}
