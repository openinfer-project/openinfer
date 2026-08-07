//! Kimi-K2's [`ModelLine`] implementation.

use clap::Args as ClapArgs;
use clap::FromArgMatches;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EpBackend;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

use crate::KimiLaunchOptions;

pub static MODEL_LINE: KimiK2Line = KimiK2Line;

pub struct KimiK2Line;

// Kimi-K2-exclusive CLI flags.
#[derive(ClapArgs)]
struct KimiCli {
    /// Expert-parallel backend for Kimi-K2 (TP1/DP8 requires deepep; TP8/DP1 requires nccl)
    #[arg(long, default_value = "deepep")]
    ep_backend: CliEpBackend,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliEpBackend {
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

fn config_model_type(config: &serde_json::Value) -> Option<&str> {
    config.get("model_type").and_then(serde_json::Value::as_str)
}

fn text_config_model_type(config: &serde_json::Value) -> Option<&str> {
    config
        .get("text_config")
        .and_then(|text| text.get("model_type"))
        .and_then(serde_json::Value::as_str)
}

impl ModelLine for KimiK2Line {
    fn name(&self) -> &'static str {
        "Kimi-K2.6"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let is_kimi = matches!(config_model_type(config), Some("kimi_k25" | "kimi_k2"))
            || text_config_model_type(config) == Some("kimi_k2");
        if !is_kimi {
            return Err(format!(
                "model_type {:?} is not a Kimi-K2 identity",
                config_model_type(config)
            ));
        }
        crate::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        KimiCli::augment_args(cmd)
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &["tp_size", "dp_size", "cuda_graph"]
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<EngineHandle> {
        let cli =
            KimiCli::from_arg_matches(ctx.matches).expect("KimiCli parses from the merged command");
        crate::launch(
            ctx.model_path,
            KimiLaunchOptions {
                tp_size: ctx.shared.tp_size,
                dp_size: ctx.shared.dp_size.unwrap_or(8),
                ep_backend: cli.ep_backend.into(),
                cuda_graph: ctx.shared.cuda_graph,
            },
        )
    }
}
