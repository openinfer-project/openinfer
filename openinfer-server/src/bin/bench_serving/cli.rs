//! CLI surface: global options, subcommands, and per-command argument structs.

use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use openinfer_core::engine::EpBackend;

pub(crate) const DEFAULT_MODEL_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
pub(crate) const TOP_LEVEL_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- request
  cargo run -r --bin bench_serving -- request --prompt \"Tell me a story about Rust\" --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64
  cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512,2048 --output-lens 32,128,256
  cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32
  cargo run -r --bin bench_serving -- --format json --out bench.json request --prompt-len 512 --output-len 64
  cargo run -r --bin bench_serving -- snapshot
  cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json";
pub(crate) const REQUEST_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- request
  cargo run -r --bin bench_serving -- request --prompt \"Tell me a story about Rust\" --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-file prompts/story.txt --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64 --warmup 3 --iters 10";
pub(crate) const MATRIX_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- matrix
  cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512,2048 --output-lens 32,128,256
  cargo run -r --bin bench_serving -- --format json --out matrix.json matrix --prompt-lens 128,512 --output-lens 64,256";
pub(crate) const CURVE_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- curve
  cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32
  cargo run -r --bin bench_serving -- curve --prompt \"Summarize KV cache behavior\" --output-len 128 --window 16";
pub(crate) const SNAPSHOT_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- snapshot
  cargo run -r --bin bench_serving -- snapshot --warmup 3 --iters 10";
pub(crate) const COMPARE_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json
  cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json --baseline HEAD~3";

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum CliEpBackend {
    Nccl,
    #[value(name = "deepep")]
    DeepEp,
}

impl From<CliEpBackend> for EpBackend {
    fn from(value: CliEpBackend) -> Self {
        match value {
            CliEpBackend::Nccl => Self::Nccl,
            CliEpBackend::DeepEp => Self::DeepEp,
        }
    }
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Measure one request shape end-to-end.
    #[command(after_help = REQUEST_EXAMPLES)]
    Request(RequestArgs),
    /// Sweep prompt_len x output_len and summarize each cell.
    #[command(after_help = MATRIX_EXAMPLES)]
    Matrix(MatrixArgs),
    /// Measure TPOT as context grows during decode.
    #[command(after_help = CURVE_EXAMPLES)]
    Curve(CurveArgs),
    /// Run standard profiles and write a regression-trackable snapshot.
    #[command(after_help = SNAPSHOT_EXAMPLES)]
    Snapshot(SnapshotArgs),
    /// Compare a snapshot against its git baseline.
    #[command(after_help = COMPARE_EXAMPLES)]
    Compare(CompareArgs),
}

#[derive(Parser, Debug)]
#[command(
    name = "bench_serving",
    about = "openinfer in-process inference benchmark",
    after_help = TOP_LEVEL_EXAMPLES
)]
pub(crate) struct Cli {
    /// Model directory (contains config.json, tokenizer, safetensors)
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    pub(crate) model_path: String,

    /// Enable CUDA graph on decode path
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) cuda_graph: bool,

    /// Render result to terminal as text or structured JSON
    #[arg(long, default_value = "text")]
    pub(crate) format: OutputFormat,

    /// Optional label to tag this benchmark run
    #[arg(long)]
    pub(crate) label: Option<String>,

    /// Optional output path for the rendered report
    #[arg(long)]
    pub(crate) out: Option<String>,

    /// Capture only measured iterations for nsys `-c cudaProfilerApi`
    #[arg(long, default_value_t = false)]
    pub(crate) cuda_profiler_capture: bool,

    /// Tensor-parallel world size for Kimi-K2
    #[arg(long, default_value_t = 1)]
    pub(crate) tp_size: usize,

    /// Data-parallel world size for Kimi-K2
    #[arg(long, default_value_t = 8)]
    pub(crate) dp_size: usize,

    /// Expert-parallel backend for Kimi-K2 (TP1/DP8 requires deepep; TP8/DP1 requires nccl)
    #[arg(long, default_value = "deepep")]
    pub(crate) ep_backend: CliEpBackend,

    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct PromptInputArgs {
    /// Inline prompt text
    #[arg(long, conflicts_with_all = ["prompt_file", "prompt_len"])]
    pub(crate) prompt: Option<String>,

    /// Read prompt text from file
    #[arg(long, conflicts_with_all = ["prompt", "prompt_len"])]
    pub(crate) prompt_file: Option<String>,

    /// Use a synthetic prompt with exactly this many token ids
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file"])]
    pub(crate) prompt_len: Option<usize>,
}

#[derive(Debug, Clone, ClapArgs)]
pub(crate) struct RunArgs {
    /// Warmup iterations
    #[arg(long, default_value_t = 5)]
    pub(crate) warmup: usize,

    /// Measured iterations
    #[arg(long, default_value_t = 20)]
    pub(crate) iters: usize,

    /// RNG seed (matters once sampling becomes non-greedy)
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,
}

#[derive(Debug, ClapArgs)]
pub(crate) struct RequestArgs {
    #[command(flatten)]
    pub(crate) prompt_input: PromptInputArgs,

    /// Max generated tokens
    #[arg(long, default_value_t = 64)]
    pub(crate) output_len: usize,

    /// Number of concurrent requests per measured iteration
    #[arg(long, default_value_t = 1)]
    pub(crate) concurrency: usize,

    /// Number of *distinct* synthetic prompts to tile across the concurrent
    /// batch (0 = one per request, fully diverse). `1` makes every concurrent
    /// request identical, which collapses MoE routing onto a narrow expert set
    /// and under-measures decode TPOT — sweep this to quantify the
    /// routing-diversity → TPOT curve (see the MoE bench-diversity lesson).
    #[arg(long, default_value_t = 0)]
    pub(crate) distinct_prompts: usize,

    #[command(flatten)]
    pub(crate) run: RunArgs,
}

#[derive(Debug, ClapArgs)]
pub(crate) struct MatrixArgs {
    /// Synthetic prompt lengths to sweep
    #[arg(long, value_delimiter = ',', default_value = "32,128,512,2048")]
    pub(crate) prompt_lens: Vec<usize>,

    /// Output lengths to sweep
    #[arg(long, value_delimiter = ',', default_value = "32,128,256")]
    pub(crate) output_lens: Vec<usize>,

    #[command(flatten)]
    pub(crate) run: RunArgs,
}

#[derive(Debug, ClapArgs)]
pub(crate) struct CurveArgs {
    #[command(flatten)]
    pub(crate) prompt_input: PromptInputArgs,

    /// Max generated tokens
    #[arg(long, default_value_t = 256)]
    pub(crate) output_len: usize,

    /// Group decode positions into windows of this size
    #[arg(long, default_value_t = 32)]
    pub(crate) window: usize,

    #[command(flatten)]
    pub(crate) run: RunArgs,
}

#[derive(Debug, ClapArgs)]
pub(crate) struct SnapshotArgs {
    #[command(flatten)]
    pub(crate) run: RunArgs,
}

#[derive(Debug, ClapArgs)]
pub(crate) struct CompareArgs {
    /// Path to snapshot JSON file
    pub(crate) path: String,

    /// Git ref to compare against
    #[arg(long, default_value = "HEAD")]
    pub(crate) baseline: String,
}
