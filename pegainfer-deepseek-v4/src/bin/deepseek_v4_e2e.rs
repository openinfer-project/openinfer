use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use pegainfer_core::logging;
use pegainfer_deepseek_v4::e2e_runner::{
    self, DEFAULT_GROUND_TRUTH_PATH, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_PATH, E2eOptions,
};

fn main() -> Result<()> {
    logging::init_default();

    let cli = Cli::parse();
    let options = cli.options()?;

    let summary = e2e_runner::run(&options)?;
    if summary.fail > 0 {
        bail!(
            "{} / {} DeepSeek V4 exact cases failed",
            summary.fail,
            summary.pass + summary.fail
        );
    }

    Ok(())
}

#[derive(Debug, Parser)]
#[command(about = "Run DeepSeek V4 exact-text E2E validation")]
struct Cli {
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,
    #[arg(
        long,
        alias = "gt-path",
        default_value = DEFAULT_GROUND_TRUTH_PATH
    )]
    ground_truth: PathBuf,
    #[arg(long, default_value_t = 0)]
    offset: usize,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(
        long,
        default_value_t = DEFAULT_MAX_NEW_TOKENS
    )]
    max_new_tokens: usize,
    #[arg(long, value_delimiter = ',', default_value = "0,1,2,3,4,5,6,7")]
    devices: Vec<usize>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    cuda_graph: bool,
}

impl Cli {
    fn options(self) -> Result<E2eOptions> {
        if self.devices.is_empty() {
            bail!("--devices must contain at least one device ordinal");
        }
        Ok(E2eOptions {
            model_path: self.model_path,
            ground_truth_path: self.ground_truth,
            offset: self.offset,
            limit: self.limit,
            max_new_tokens: self.max_new_tokens,
            device_ordinals: self.devices,
            seed: self.seed,
            enable_cuda_graph: self.cuda_graph,
        })
    }
}
