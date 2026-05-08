use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;
use pegainfer_deepseek_v4::{
    Bf16HiddenStates, Config, HcHiddenStates, RankGpuContext, RankWeights, TensorParallelConfig,
    hc_head_bf16_hidden, load_rank_subset_to_gpu, rank_local_logits_from_hidden,
    rms_norm_bf16_hidden, score_route_bf16_hidden,
};
use pegainfer_kernels::ffi;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let config = Config::from_model_dir(&args.model_dir)?;
    let ctx = RankGpuContext::new(args.device)?;
    let route_gate_weight = format!("layers.{}.ffn.gate.weight", args.route_layer);
    let route_gate_bias = format!("layers.{}.ffn.gate.bias", args.route_layer);
    let shared_w1_weight = format!("layers.{}.ffn.shared_experts.w1.weight", args.route_layer);
    let shared_w1_scale = format!("layers.{}.ffn.shared_experts.w1.scale", args.route_layer);
    let shared_w2_weight = format!("layers.{}.ffn.shared_experts.w2.weight", args.route_layer);
    let shared_w2_scale = format!("layers.{}.ffn.shared_experts.w2.scale", args.route_layer);
    let shared_w3_weight = format!("layers.{}.ffn.shared_experts.w3.weight", args.route_layer);
    let shared_w3_scale = format!("layers.{}.ffn.shared_experts.w3.scale", args.route_layer);
    let weight_names = [
        "embed.weight",
        "hc_head_fn",
        "hc_head_scale",
        "hc_head_base",
        "norm.weight",
        "head.weight",
        route_gate_weight.as_str(),
        route_gate_bias.as_str(),
        shared_w1_weight.as_str(),
        shared_w1_scale.as_str(),
        shared_w2_weight.as_str(),
        shared_w2_scale.as_str(),
        shared_w3_weight.as_str(),
        shared_w3_scale.as_str(),
    ];
    let tensors = load_rank_subset_to_gpu(
        &ctx,
        &args.model_dir,
        &config,
        TensorParallelConfig::mp8(args.rank),
        &weight_names,
    )?;
    let total_bytes = tensors.values().map(|tensor| tensor.bytes).sum();
    let weights = RankWeights {
        rank: args.rank,
        world_size: 8,
        tensors,
        total_bytes,
    };
    let view = weights.view(&config)?;

    let fixture_path = args
        .fixture_dir
        .join(format!("rank{}.safetensors", args.rank));
    let fixture_bytes = fs::read(&fixture_path)
        .with_context(|| format!("failed to read {}", fixture_path.display()))?;
    let fixture = SafeTensors::deserialize(&fixture_bytes)
        .with_context(|| format!("failed to parse {}", fixture_path.display()))?;

    let final_h = load_bf16(
        &fixture,
        "final.h_input",
        &[1, 1, config.hc_mult, config.dim],
    )?;
    let final_h_expected = load_bf16(&fixture, "final.hc_head", &[1, 1, config.dim])?;
    let final_norm_expected = load_bf16(&fixture, "final.norm", &[1, 1, config.dim])?;
    let local_vocab = config.vocab_size / 8;
    let logits_expected = load_f32(&fixture, "final.local_logits", &[1, local_vocab])?;

    let final_h_gpu = HcHiddenStates {
        data: ctx.stream.clone_htod(&final_h)?,
        hidden_dim: config.dim,
        seq_len: 1,
        hc: config.hc_mult,
    };
    let hc_head = hc_head_bf16_hidden(
        &ctx,
        &config,
        &final_h_gpu,
        &view.hc_head_fn()?,
        &view.hc_head_scale()?,
        &view.hc_head_base()?,
    )?;
    let hc_head_actual = ctx.stream.clone_dtoh(&hc_head.data)?;
    ctx.sync()?;
    report_bf16("final.hc_head", &final_h_expected, &hc_head_actual);

    let normed = rms_norm_bf16_hidden(&ctx, &hc_head, &view.norm()?, config.rms_norm_eps)?;
    let norm_actual = ctx.stream.clone_dtoh(&normed.data)?;
    ctx.sync()?;
    report_bf16("final.norm", &final_norm_expected, &norm_actual);

    let logits = rank_local_logits_from_hidden(&ctx, &normed, &view.head()?)?;
    let logits_actual = logits.to_host(&ctx)?;
    report_f32("final.local_logits", &logits_expected, &logits_actual);

    let route_prefix = format!("layer{}.route", args.route_layer);
    let route_input = load_bf16(&fixture, &format!("{route_prefix}.input"), &[1, config.dim])?;
    let route_weights_expected = load_f32(
        &fixture,
        &format!("{route_prefix}.weights"),
        &[1, config.n_activated_experts],
    )?;
    let raw_scores_expected = load_f32(
        &fixture,
        &format!("{route_prefix}.raw_scores"),
        &[1, config.n_routed_experts],
    )?;
    let original_scores_expected = load_f32(
        &fixture,
        &format!("{route_prefix}.original_scores"),
        &[1, config.n_routed_experts],
    )?;
    let select_scores_expected = load_f32(
        &fixture,
        &format!("{route_prefix}.select_scores"),
        &[1, config.n_routed_experts],
    )?;
    let route_indices_expected = load_i32(
        &fixture,
        &format!("{route_prefix}.indices"),
        &[1, config.n_activated_experts],
    )?;
    let route_input_gpu = Bf16HiddenStates {
        data: ctx.stream.clone_htod(&route_input)?,
        hidden_dim: config.dim,
        seq_len: 1,
    };
    let ffn = view.ffn(args.route_layer)?;
    let mut raw_scores = ctx.stream.alloc_zeros(config.n_routed_experts)?;
    let mut original_scores = ctx.stream.alloc_zeros(config.n_routed_experts)?;
    let mut select_scores = ctx.stream.alloc_zeros(config.n_routed_experts)?;
    let mut debug_weights = ctx.stream.alloc_zeros(config.n_activated_experts)?;
    let mut debug_indices = ctx.stream.alloc_zeros(config.n_activated_experts)?;
    {
        let (x_ptr, _x_guard) = route_input_gpu.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = ffn.gate_weight.tensor.data.device_ptr(&ctx.stream);
        let bias = ffn
            .gate_bias
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("route debug requires gate bias"))?;
        let (bias_ptr, _bias_guard) = bias.tensor.data.device_ptr(&ctx.stream);
        let (raw_ptr, _raw_guard) = raw_scores.device_ptr_mut(&ctx.stream);
        let (original_ptr, _original_guard) = original_scores.device_ptr_mut(&ctx.stream);
        let (select_ptr, _select_guard) = select_scores.device_ptr_mut(&ctx.stream);
        let (weights_ptr, _weights_guard) = debug_weights.device_ptr_mut(&ctx.stream);
        let (indices_ptr, _indices_guard) = debug_indices.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_score_gate_debug_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                bias_ptr as *const f32,
                raw_ptr as *mut f32,
                original_ptr as *mut f32,
                select_ptr as *mut f32,
                weights_ptr as *mut f32,
                indices_ptr as *mut i32,
                1,
                config.dim as i32,
                config.n_routed_experts as i32,
                config.n_activated_experts as i32,
                config.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    let raw_scores_actual = ctx.stream.clone_dtoh(&raw_scores)?;
    let original_scores_actual = ctx.stream.clone_dtoh(&original_scores)?;
    let select_scores_actual = ctx.stream.clone_dtoh(&select_scores)?;
    let debug_weights_actual = ctx.stream.clone_dtoh(&debug_weights)?;
    let debug_indices_actual = ctx.stream.clone_dtoh(&debug_indices)?;
    ctx.sync()?;
    report_f32(
        &format!("{route_prefix}.raw_scores"),
        &raw_scores_expected,
        &raw_scores_actual,
    );
    report_f32(
        &format!("{route_prefix}.original_scores"),
        &original_scores_expected,
        &original_scores_actual,
    );
    report_f32(
        &format!("{route_prefix}.select_scores"),
        &select_scores_expected,
        &select_scores_actual,
    );
    report_f32(
        &format!("{route_prefix}.debug_weights"),
        &route_weights_expected,
        &debug_weights_actual,
    );
    report_i32(
        &format!("{route_prefix}.debug_indices"),
        &route_indices_expected,
        &debug_indices_actual,
    );

    let routed = score_route_bf16_hidden(&ctx, &config, &route_input_gpu, &ffn)?;
    let route_weights_actual = ctx.stream.clone_dtoh(&routed.weights)?;
    let route_indices_actual = ctx.stream.clone_dtoh(&routed.indices)?;
    ctx.sync()?;
    report_f32(
        &format!("{route_prefix}.weights"),
        &route_weights_expected,
        &route_weights_actual,
    );
    report_i32(
        &format!("{route_prefix}.indices"),
        &route_indices_expected,
        &route_indices_actual,
    );

    Ok(())
}

struct Args {
    model_dir: PathBuf,
    fixture_dir: PathBuf,
    rank: usize,
    device: usize,
    route_layer: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model_dir = None;
        let mut fixture_dir = None;
        let mut rank = 0usize;
        let mut device = 0usize;
        let mut route_layer = 25usize;
        let mut items = env::args().skip(1);
        while let Some(arg) = items.next() {
            match arg.as_str() {
                "--model-dir" => model_dir = Some(PathBuf::from(next_value(&mut items, &arg)?)),
                "--fixture-dir" => fixture_dir = Some(PathBuf::from(next_value(&mut items, &arg)?)),
                "--rank" => rank = next_value(&mut items, &arg)?.parse()?,
                "--device" => device = next_value(&mut items, &arg)?.parse()?,
                "--route-layer" => route_layer = next_value(&mut items, &arg)?.parse()?,
                "-h" | "--help" => {
                    println!(
                        "usage: deepseek_kernel_check --model-dir DIR --fixture-dir DIR [--rank N] [--device N] [--route-layer N]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(Self {
            model_dir: model_dir
                .or_else(|| env::var_os("PEGAINFER_TEST_MODEL_PATH").map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("models/DeepSeek-V4-Flash")),
            fixture_dir: fixture_dir
                .unwrap_or_else(|| PathBuf::from("/tmp/deepseek_kernel_fixtures")),
            rank,
            device,
            route_layer,
        })
    }
}

fn next_value(items: &mut impl Iterator<Item = String>, arg: &str) -> Result<String> {
    items
        .next()
        .ok_or_else(|| anyhow::anyhow!("{arg} requires a value"))
}

fn tensor<'a>(
    fixture: &'a SafeTensors<'a>,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<TensorView<'a>> {
    let view = fixture
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        view.dtype() == dtype,
        "tensor {name} dtype mismatch: expected {:?}, got {:?}",
        dtype,
        view.dtype()
    );
    ensure!(
        view.shape() == shape,
        "tensor {name} shape mismatch: expected {:?}, got {:?}",
        shape,
        view.shape()
    );
    Ok(view)
}

fn load_bf16(fixture: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
    let view = tensor(fixture, name, Dtype::BF16, shape)?;
    Ok(view
        .data()
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn load_f32(fixture: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = tensor(fixture, name, Dtype::F32, shape)?;
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn load_i32(fixture: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<i32>> {
    let view = tensor(fixture, name, Dtype::I32, shape)?;
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn report_bf16(stage: &str, expected: &[bf16], actual: &[bf16]) {
    assert_eq!(expected.len(), actual.len(), "{stage} length mismatch");
    let mut first = None;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for (idx, (&a, &b)) in expected.iter().zip(actual.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            mismatches += 1;
            if first.is_none() {
                first = Some((idx, a, b));
            }
        }
        max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
    }
    print!("{stage}: mismatches={mismatches} max_abs={max_abs}");
    if let Some((idx, expected, actual)) = first {
        print!(
            " first idx={} expected={} actual={} expected_bits=0x{:04x} actual_bits=0x{:04x}",
            idx,
            expected.to_f32(),
            actual.to_f32(),
            expected.to_bits(),
            actual.to_bits()
        );
    }
    println!();
}

fn report_f32(stage: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{stage} length mismatch");
    let mut first = None;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    for (idx, (&a, &b)) in expected.iter().zip(actual.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            mismatches += 1;
            if first.is_none() {
                first = Some((idx, a, b));
            }
        }
        max_abs = max_abs.max((a - b).abs());
        max_ulp = max_ulp.max(ulp_distance(a, b));
    }
    let top_expected = top1(expected);
    let top_actual = top1(actual);
    print!(
        "{stage}: mismatches={mismatches} max_abs={max_abs} max_ulp={max_ulp} top_expected=({},{}) top_actual=({},{})",
        top_expected.0, top_expected.1, top_actual.0, top_actual.1
    );
    if let Some((idx, expected, actual)) = first {
        print!(
            " first idx={} expected={} actual={} expected_bits=0x{:08x} actual_bits=0x{:08x}",
            idx,
            expected,
            actual,
            expected.to_bits(),
            actual.to_bits()
        );
    }
    println!();
}

fn report_i32(stage: &str, expected: &[i32], actual: &[i32]) {
    assert_eq!(expected.len(), actual.len(), "{stage} length mismatch");
    let mut first = None;
    let mut mismatches = 0usize;
    for (idx, (&a, &b)) in expected.iter().zip(actual.iter()).enumerate() {
        if a != b {
            mismatches += 1;
            if first.is_none() {
                first = Some((idx, a, b));
            }
        }
    }
    print!("{stage}: mismatches={mismatches}");
    if let Some((idx, expected, actual)) = first {
        print!(" first idx={idx} expected={expected} actual={actual}");
    }
    println!(" expected={expected:?} actual={actual:?}");
}

fn ordered_i32(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    if bits < 0 { i32::MIN - bits } else { bits }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    ordered_i32(a).abs_diff(ordered_i32(b))
}

fn top1(values: &[f32]) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    (best_idx, best)
}
