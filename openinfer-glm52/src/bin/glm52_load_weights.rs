use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail};
use openinfer_glm52::{Glm52LaunchOptions, launch};

const DEFAULT_MODEL_PATH: &str = "models/GLM-5.2-FP8";

#[derive(Debug)]
struct Args {
    model_path: PathBuf,
    tp_size: usize,
    dp_size: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(DEFAULT_MODEL_PATH),
            tp_size: 1,
            dp_size: 1,
        }
    }
}

fn main() -> Result<()> {
    openinfer_core::logging::init_default();
    let args = parse_args(std::env::args().skip(1))?;
    let started = Instant::now();
    let handle = launch(
        &args.model_path,
        Glm52LaunchOptions {
            tp_size: args.tp_size,
            dp_size: args.dp_size,
        },
    )
    .with_context(|| {
        format!(
            "failed to load GLM5.2 weights from {}",
            args.model_path.display()
        )
    })?;
    drop(handle);
    let elapsed_ms = started.elapsed().as_millis();
    log::info!(
        "GLM5.2 load-weight command complete: model_path={}, elapsed_ms={elapsed_ms}",
        args.model_path.display()
    );
    println!("GLM5.2 load weights complete: elapsed_ms={elapsed_ms}");
    Ok(())
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args> {
    let mut parsed = Args::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-path" => {
                parsed.model_path = PathBuf::from(next_value(&mut args, "--model-path")?);
            }
            "--tp-size" => {
                parsed.tp_size = parse_usize(&next_value(&mut args, "--tp-size")?, "--tp-size")?;
            }
            "--dp-size" => {
                parsed.dp_size = parse_usize(&next_value(&mut args, "--dp-size")?, "--dp-size")?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release -p openinfer-glm52 --bin glm52_load_weights -- [--model-path PATH] [--tp-size 1] [--dp-size 1]"
                );
                std::process::exit(0);
            }
            _ if arg.starts_with("--model-path=") => {
                parsed.model_path = PathBuf::from(value_after_equals(&arg, "--model-path")?);
            }
            _ if arg.starts_with("--tp-size=") => {
                parsed.tp_size = parse_usize(value_after_equals(&arg, "--tp-size")?, "--tp-size")?;
            }
            _ if arg.starts_with("--dp-size=") => {
                parsed.dp_size = parse_usize(value_after_equals(&arg, "--dp-size")?, "--dp-size")?;
            }
            _ => bail!("unknown argument {arg}"),
        }
    }
    Ok(parsed)
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn value_after_equals<'a>(arg: &'a str, flag: &str) -> Result<&'a str> {
    arg.strip_prefix(flag)
        .and_then(|rest| rest.strip_prefix('='))
        .filter(|value| !value.is_empty())
        .with_context(|| format!("{flag} requires a non-empty value"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("invalid {flag}: {value}"))
}
