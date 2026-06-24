//! Pure-Rust Dynamo OpenAI HTTP frontend launcher — the same data plane that
//! `python -m dynamo.frontend` runs, with no Python in the deploy.
//!
//! The Python frontend module is only a ~250-line argparse shim over two pyo3
//! functions (`make_engine` + `run_input`); the HTTP server (axum), the OpenAI
//! preprocessor (tokenize / chat-template / detokenize), the KV router and the
//! model-discovery watcher are all Rust in `dynamo-llm`. This binary builds the
//! same `EngineConfig::Dynamic` + `DistributedRuntime` the shim builds and
//! awaits `dynamo_llm::entrypoint::input::run_input(Input::Http)` directly.
//!
//! It discovers openinfer Qwen3 workers from etcd and routes across them. Point
//! it at the same control plane as the workers via `NATS_SERVER` /
//! `ETCD_ENDPOINTS`. KV-aware routing is on by default (`--router-mode kv`).

use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use dynamo_llm::entrypoint::input::{Input, run_input};
use dynamo_llm::entrypoint::{EngineConfig, RouterConfig};
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime, Worker, logging};

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Pure-Rust Dynamo OpenAI HTTP frontend + KV router (no Python)."
)]
struct Args {
    /// HTTP listen host.
    #[arg(long, default_value = "0.0.0.0")]
    http_host: String,

    /// HTTP listen port (serves /v1/chat/completions, /v1/completions, ...).
    #[arg(long, default_value_t = 8000)]
    http_port: u16,

    /// Router strategy. `kv` = cache-aware (steer to a warm-prefix replica),
    /// the reason this frontend exists; `round-robin` / `random` are the
    /// cache-blind fallbacks.
    #[arg(long, value_enum, default_value_t = RouterModeArg::Kv)]
    router_mode: RouterModeArg,

    /// Optional local model directory to resolve tokenizer/config from when the
    /// workers' model path is not present on this host. Workers publish their
    /// own model card; this is only the metadata-resolution overlay.
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Public model name. Defaults to whatever the workers register.
    #[arg(long)]
    model_name: Option<String>,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum RouterModeArg {
    Kv,
    RoundRobin,
    Random,
}

impl From<RouterModeArg> for RouterMode {
    fn from(mode: RouterModeArg) -> Self {
        match mode {
            RouterModeArg::Kv => RouterMode::KV,
            RouterModeArg::RoundRobin => RouterMode::RoundRobin,
            RouterModeArg::Random => RouterMode::Random,
        }
    }
}

fn main() -> anyhow::Result<()> {
    logging::init();
    let args = Args::parse();
    // `Worker` owns the tokio runtime + shutdown; it blocks until `run` returns.
    Worker::from_settings()?.execute(move |runtime| run(runtime, args))
}

async fn run(runtime: Runtime, args: Args) -> anyhow::Result<()> {
    let drt = DistributedRuntime::from_settings(runtime)
        .await
        .context("failed to build DistributedRuntime (is etcd / NATS reachable?)")?;

    let mut builder = LocalModelBuilder::default();
    builder
        .model_name(args.model_name)
        .http_host(Some(args.http_host))
        .http_port(args.http_port)
        .router_config(Some(RouterConfig {
            router_mode: args.router_mode.into(),
            ..Default::default()
        }));
    // Overlay the metadata path only if given; with none, the frontend resolves
    // tokenizer/config entirely from each worker's published model card.
    if let Some(model_path) = args.model_path {
        builder.model_path(model_path);
    }
    let local_model = builder
        .build()
        .await
        .context("failed to build LocalModel")?;

    // `Dynamic` = discover networked workers via etcd and route to them. No
    // in-process engine, no Python chat factory, no AIC load estimator — the
    // openinfer workers are the engines.
    let engine_config = EngineConfig::Dynamic {
        model: Box::new(local_model),
        chat_engine_factory: None,
        prefill_load_estimator: None,
    };

    run_input(drt, Input::Http, engine_config).await
}
