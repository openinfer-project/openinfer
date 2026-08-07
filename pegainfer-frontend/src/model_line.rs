//! The seam between the server binary and a model crate.
//!
//! A model line is everything south of the engine contract: config probing,
//! its own CLI knobs, and a `launch` that starts scheduler threads and hands
//! back an [`EngineHandle`]. The server binary holds a feature-gated
//! [`ModelLineRegistry`] and does pure dispatch — it never names a model
//! crate's option types.
//!
//! Onboarding a model line means implementing [`ModelLine`] in the model
//! crate and adding the instance to the registry in the server binary. No
//! other server edits: argument routing, `--help` grouping, and config
//! detection all derive from the trait.

use std::path::Path;

use crate::engine::EngineHandle;

/// Everything `launch` may read besides its own CLI arguments.
pub struct LaunchContext<'a> {
    /// Model directory containing `config.json` and weights.
    pub model_path: &'a Path,
    /// Parsed `config.json` — the same value `probe` accepted.
    pub config: &'a serde_json::Value,
}

/// A servable model family. Implemented by each model crate.
pub trait ModelLine: Send + Sync {
    /// Family name used in logs, errors, and `--help` section headers
    /// (e.g. `"qwen3"`).
    fn name(&self) -> &'static str;

    /// Claim or reject a model directory by its parsed `config.json`.
    /// Return `Err` with the reason when the architecture doesn't match;
    /// exactly one registered line must accept a given config.
    fn probe(&self, config: &serde_json::Value) -> Result<(), String>;

    /// Append this line's CLI arguments to the server command. The registry
    /// diffs the command before/after to learn which argument ids belong to
    /// this line, so `--help` grouping and "flag X is not consumed by model
    /// Y" validation need no separate declaration.
    fn augment_cli(&self, cmd: clap::Command) -> clap::Command;

    /// Number of scheduler partitions (logical DP ranks) the launched engine
    /// will expose. The HTTP frontend registers one engine identity per
    /// partition *while the engine is still loading*, so this must be
    /// derivable from CLI arguments alone. Checked against
    /// [`EngineHandle::scheduler_partition_count`] after launch.
    fn scheduler_partition_count(&self, _matches: &clap::ArgMatches) -> usize {
        1
    }

    /// Start the engine: spawn scheduler threads, build the handle with its
    /// metadata (`with_kv_capacity`, `with_load_watch`, ...), return it.
    /// Reads its own arguments from `matches`.
    fn launch(
        &self,
        ctx: &LaunchContext<'_>,
        matches: &clap::ArgMatches,
    ) -> anyhow::Result<EngineHandle>;
}

/// The server binary's fixed list of compiled-in model lines.
pub struct ModelLineRegistry {
    lines: Vec<&'static dyn ModelLine>,
}

impl ModelLineRegistry {
    pub fn new(lines: Vec<&'static dyn ModelLine>) -> Self {
        Self { lines }
    }

    /// Append every registered line's CLI section to the server command.
    pub fn augment_cli(&self, mut cmd: clap::Command) -> clap::Command {
        for line in &self.lines {
            cmd = line.augment_cli(cmd);
        }
        cmd
    }

    /// Find the unique line that claims this `config.json`. Errors list every
    /// line's rejection reason so an unrecognized model names the candidates
    /// that were tried (and whether the right feature was compiled out).
    pub fn detect(&self, config: &serde_json::Value) -> Result<&'static dyn ModelLine, String> {
        let mut rejections = Vec::new();
        let mut claimed = None;
        for line in &self.lines {
            match line.probe(config) {
                Ok(()) => match claimed {
                    None => claimed = Some(*line),
                    Some(prev) => {
                        return Err(format!(
                            "config claimed by both {} and {}",
                            prev.name(),
                            line.name()
                        ));
                    }
                },
                Err(reason) => rejections.push(format!("{}: {reason}", line.name())),
            }
        }
        claimed.ok_or_else(|| {
            format!(
                "no compiled-in model line claims this config (tried {})",
                rejections.join("; ")
            )
        })
    }
}
