use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use half::bf16;
use openinfer_core::tensor::HiddenStates;
use openinfer_qwen3_4b_dflash::{DFlashDraftModel, DFlashTargetHidden};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let fixture_bytes = std::fs::read(&args.fixture).with_context(|| {
        format!(
            "failed to read input fixture {}",
            args.fixture.to_string_lossy()
        )
    })?;
    let st = SafeTensors::deserialize(&fixture_bytes).context("parse input fixture")?;
    let model = DFlashDraftModel::load(&args.model_path, args.device)?;
    let config = model.config();
    let ctx = model.device_context();

    let noise = bf16_tensor(&st, "noise_embedding")?;
    let target_hidden = bf16_tensor(&st, "target_hidden")?;
    let positions = i32_tensor(&st, "position_ids")?;

    if noise.1.len() != 3 || noise.1[0] != 1 || noise.1[2] != config.hidden_size {
        bail!(
            "noise_embedding shape mismatch: expected [1, q_len, {}], got {:?}",
            config.hidden_size,
            noise.1
        );
    }
    if target_hidden.1.len() != 3
        || target_hidden.1[0] != 1
        || target_hidden.1[2] != config.hidden_size * config.target_layer_count()
    {
        bail!(
            "target_hidden shape mismatch: expected [1, ctx_len, {}], got {:?}",
            config.hidden_size * config.target_layer_count(),
            target_hidden.1
        );
    }
    let q_len = noise.1[1];
    let ctx_len = target_hidden.1[1];
    ensure_shape("position_ids", &positions.1, &[1, ctx_len + q_len])?;

    let noise_embedding = HiddenStates {
        data: ctx.stream.clone_htod(&noise.0)?,
        hidden_dim: config.hidden_size,
        seq_len: q_len,
    };
    let target_hidden = HiddenStates {
        data: ctx.stream.clone_htod(&target_hidden.0)?,
        hidden_dim: config.hidden_size * config.target_layer_count(),
        seq_len: ctx_len,
    };
    let out = model.forward(
        &noise_embedding,
        DFlashTargetHidden {
            concatenated: &target_hidden,
        },
        &positions.0,
    )?;
    ctx.sync()?;
    let out = ctx.stream.clone_dtoh(&out.data)?;
    ctx.sync()?;

    let out_bytes = bf16_bytes(&out);
    let tensors = HashMap::from([(
        "openinfer_output".to_string(),
        TensorView::new(Dtype::BF16, vec![1, q_len, config.hidden_size], &out_bytes)?,
    )]);
    safetensors::serialize_to_file(tensors, None, &args.out)?;
    Ok(())
}

struct Args {
    model_path: PathBuf,
    fixture: PathBuf,
    out: PathBuf,
    device: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model_path = None;
        let mut fixture = None;
        let mut out = None;
        let mut device = 0usize;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-path" => model_path = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                "--fixture" => fixture = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                "--out" => out = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                "--device" => device = next_value(&mut args, &arg)?.parse()?,
                _ => bail!("unknown argument {arg}"),
            }
        }
        Ok(Self {
            model_path: model_path
                .unwrap_or_else(|| PathBuf::from("/home/hezhaozhao/models/Qwen3-4B-DFlash-b16")),
            fixture: fixture.context("--fixture is required")?,
            out: out.context("--out is required")?,
            device,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn ensure_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        bail!("{name} shape mismatch: expected {expected:?}, got {got:?}");
    }
    Ok(())
}

fn bf16_tensor(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<bf16>, Vec<usize>)> {
    let view = st.tensor(name)?;
    if view.dtype() != Dtype::BF16 {
        bail!("{name} must be BF16, got {:?}", view.dtype());
    }
    let values = view
        .data()
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    Ok((values, view.shape().to_vec()))
}

fn i32_tensor(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<i32>, Vec<usize>)> {
    let view = st.tensor(name)?;
    if view.dtype() != Dtype::I32 {
        bail!("{name} must be I32, got {:?}", view.dtype());
    }
    let values = view
        .data()
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok((values, view.shape().to_vec()))
}

fn bf16_bytes(values: &[bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend(value.to_bits().to_le_bytes());
    }
    out
}
