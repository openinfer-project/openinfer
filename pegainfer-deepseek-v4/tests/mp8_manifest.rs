use std::path::{Path, PathBuf};

use half::bf16;
use pegainfer_deepseek_v4::{
    AttentionProjections, Bf16Cache, Bf16HiddenStates, Config, DeepSeekRankModel,
    DeepSeekRopeCache, GpuRawTensor, LayerDecodeCache, QuantLinearRef, RankGpuContext, RankWeights,
    TensorParallelConfig, TensorRef, all_reduce_hidden_group, apply_rope_attention_projections,
    attention_decode_rank_local_bf16_hidden, attention_project_bf16_hidden,
    block_decode_group_bf16_hidden, block_prefill_group_bf16_hidden, compress_topk_indices_decode,
    copy_window_prefill_to_ring_cache, embedding_vocab_parallel_group,
    final_logits_group_bf16_hidden, fp4_linear_bf16_hidden, fp8_linear_bf16_hidden,
    hadamard_fp4_quant_bf16_hidden_in_place, hc_expand_bf16_hidden, load_rank_manifest,
    load_rank_subset_to_gpu, precompute_rope_cache, window_topk_indices_decode,
};
use safetensors::Dtype;

const DEFAULT_MODEL_PATH: &str = "models/DeepSeek-V4-Flash";

fn deepseek_model_path() -> PathBuf {
    std::env::var_os("PEGAINFER_TEST_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn official_deepseek_rope_pair_value(
    config: &Config,
    layer: usize,
    pos: usize,
    pair: usize,
    cos: bool,
) -> f32 {
    let dim = config.qk_rope_head_dim;
    let compress = config.compress_ratios[layer] > 0;
    let base = if compress {
        config.compress_rope_theta as f64
    } else {
        config.rope_theta as f64
    };
    let original_seq_len = if compress {
        config.rope_scaling.original_seq_len
    } else {
        0
    };
    let factor = config.rope_scaling.factor;

    let mut freq = 1.0f32 / (base as f32).powf((2 * pair) as f32 / dim as f32);
    if original_seq_len > 0 {
        let find_correction_dim = |num_rotations: usize| -> f64 {
            dim as f64
                * ((original_seq_len as f64) / (num_rotations as f64 * 2.0 * std::f64::consts::PI))
                    .ln()
                / (2.0 * base.ln())
        };
        let low = find_correction_dim(config.rope_scaling.beta_fast)
            .floor()
            .max(0.0);
        let mut high = find_correction_dim(config.rope_scaling.beta_slow)
            .ceil()
            .min((dim - 1) as f64);
        if low == high {
            high += 0.001;
        }
        let ramp = ((pair as f64 - low) / (high - low)).clamp(0.0, 1.0) as f32;
        let smooth = 1.0 - ramp;
        freq = freq / factor * (1.0 - smooth) + freq * smooth;
    }

    let angle = pos as f32 * freq;
    if cos { angle.cos() } else { angle.sin() }
}

#[test]
fn rank0_manifest_matches_deepseek_v4_mp8_layout() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 manifest test; {} does not exist",
            model_path.display()
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let manifest = load_rank_manifest(model_path, &config, TensorParallelConfig::mp8(0))
        .expect("load rank0 manifest");

    assert_eq!(manifest.tensors.len(), 10007);
    assert_eq!(manifest.tensors["embed.weight"].dtype, Dtype::BF16);
    assert_eq!(manifest.tensors["embed.weight"].shape, vec![16160, 4096]);
    assert_eq!(manifest.tensors["norm.weight"].dtype, Dtype::BF16);
    assert_eq!(manifest.tensors["norm.weight"].shape, vec![4096]);
    assert_eq!(
        manifest.tensors["layers.0.attn.wq_a.weight"].dtype,
        Dtype::F8_E4M3
    );
    assert_eq!(
        manifest.tensors["layers.0.ffn.experts.0.w1.weight"].dtype,
        Dtype::F4
    );
    assert_eq!(
        manifest.tensors["layers.0.ffn.gate.tid2eid"].dtype,
        Dtype::I64
    );
}

#[test]
fn rope_cache_matches_deepseek_official_formula_for_plain_and_compressed_layers() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 RoPE formula test; {} does not exist",
            model_path.display()
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let positions = [0usize, 1, 127, 128, 4095, 8191];
    let pairs = [0usize, 1, 15, 31];

    for layer in [0usize, 2, 3] {
        let rope = precompute_rope_cache(&ctx, &config, layer, 8192).expect("rope cache");
        let cos = ctx.stream.clone_dtoh(&rope.cos).expect("copy cos");
        let sin = ctx.stream.clone_dtoh(&rope.sin).expect("copy sin");
        ctx.sync().expect("sync CUDA stream");
        let pair_count = config.qk_rope_head_dim / 2;

        for pos in positions {
            for pair in pairs {
                let idx = pos * pair_count + pair;
                let expected_cos =
                    official_deepseek_rope_pair_value(&config, layer, pos, pair, true);
                let expected_sin =
                    official_deepseek_rope_pair_value(&config, layer, pos, pair, false);
                assert!(
                    (cos[idx] - expected_cos).abs() <= 1.0e-6,
                    "cos mismatch layer={layer} pos={pos} pair={pair}: actual={} expected={}",
                    cos[idx],
                    expected_cos
                );
                assert!(
                    (sin[idx] - expected_sin).abs() <= 1.0e-6,
                    "sin mismatch layer={layer} pos={pos} pair={pair}: actual={} expected={}",
                    sin[idx],
                    expected_sin
                );
            }
        }
    }
}

#[test]
fn rope_kernel_rotates_adjacent_tail_pairs_only() {
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let seq_len = 2usize;
    let local_heads = 2usize;
    let head_dim = 68usize;
    let rotary_dim = 4usize;
    let start_pos = 3usize;
    let max_seq_len = start_pos + seq_len;
    let pairs = rotary_dim / 2;

    let mut cos_host = vec![0.0f32; max_seq_len * pairs];
    let mut sin_host = vec![0.0f32; max_seq_len * pairs];
    for pos in 0..max_seq_len {
        for pair in 0..pairs {
            let angle = 0.125 * pos as f32 * (pair as f32 + 1.0);
            cos_host[pos * pairs + pair] = angle.cos();
            sin_host[pos * pairs + pair] = angle.sin();
        }
    }
    let rope = DeepSeekRopeCache {
        cos: ctx.stream.clone_htod(&cos_host).expect("cos h2d"),
        sin: ctx.stream.clone_htod(&sin_host).expect("sin h2d"),
        max_seq_len,
        rotary_dim,
    };

    let q_len = seq_len * local_heads * head_dim;
    let kv_len = seq_len * head_dim;
    let q_host = (0..q_len)
        .map(|idx| bf16::from_f32((idx as f32 + 1.0) * 0.125))
        .collect::<Vec<_>>();
    let kv_host = (0..kv_len)
        .map(|idx| bf16::from_f32((idx as f32 + 1.0) * -0.25))
        .collect::<Vec<_>>();

    let mut projections = AttentionProjections {
        qr: Bf16HiddenStates {
            data: ctx.stream.clone_htod(&[bf16::ZERO]).expect("qr h2d"),
            hidden_dim: 1,
            seq_len,
        },
        q: Bf16HiddenStates {
            data: ctx.stream.clone_htod(&q_host).expect("q h2d"),
            hidden_dim: local_heads * head_dim,
            seq_len,
        },
        kv: Bf16HiddenStates {
            data: ctx.stream.clone_htod(&kv_host).expect("kv h2d"),
            hidden_dim: head_dim,
            seq_len,
        },
        local_heads,
        head_dim,
    };

    apply_rope_attention_projections(&ctx, &mut projections, &rope, start_pos).expect("apply rope");
    let got_q = ctx.stream.clone_dtoh(&projections.q.data).expect("copy q");
    let got_kv = ctx
        .stream
        .clone_dtoh(&projections.kv.data)
        .expect("copy kv");
    ctx.sync().expect("sync CUDA stream");

    let rotate_pair = |x0: bf16, x1: bf16, c: f32, s: f32| -> (bf16, bf16) {
        let x0 = x0.to_f32();
        let x1 = x1.to_f32();
        (
            bf16::from_f32(x0 * c - x1 * s),
            bf16::from_f32(x0 * s + x1 * c),
        )
    };

    let nope_dim = head_dim - rotary_dim;
    let mut expected_q = q_host.clone();
    for token in 0..seq_len {
        let pos = start_pos + token;
        for head in 0..local_heads {
            for pair in 0..pairs {
                let offset = token * local_heads * head_dim + head * head_dim + nope_dim + 2 * pair;
                let (lo, hi) = rotate_pair(
                    expected_q[offset],
                    expected_q[offset + 1],
                    cos_host[pos * pairs + pair],
                    sin_host[pos * pairs + pair],
                );
                expected_q[offset] = lo;
                expected_q[offset + 1] = hi;
            }
        }
    }

    let mut expected_kv = kv_host.clone();
    for token in 0..seq_len {
        let pos = start_pos + token;
        for pair in 0..pairs {
            let offset = token * head_dim + nope_dim + 2 * pair;
            let (lo, hi) = rotate_pair(
                expected_kv[offset],
                expected_kv[offset + 1],
                cos_host[pos * pairs + pair],
                sin_host[pos * pairs + pair],
            );
            expected_kv[offset] = lo;
            expected_kv[offset + 1] = hi;
        }
    }

    assert_eq!(got_q, expected_q, "Q RoPE must use adjacent tail pairs");
    for token in 0..seq_len {
        for dim in nope_dim..head_dim {
            let idx = token * head_dim + dim;
            assert_eq!(
                got_kv[idx], expected_kv[idx],
                "KV RoPE tail mismatch token={token} dim={dim}"
            );
        }
    }
}

#[test]
fn hadamard_fp4_quant_matches_official_quant_dequant_reference() {
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let rows = 3usize;
    let groups = 2usize;
    let dim = 128usize;
    let host = (0..rows * groups * dim)
        .map(|idx| {
            let lane = (idx % dim) as f32;
            let row = (idx / dim) as f32;
            let value = ((lane * 0.03125).sin() * 1.75) + ((row + 1.0) * 0.0078125)
                - ((idx % 11) as f32 * 0.00390625);
            bf16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let expected = official_hadamard_fp4_quant_dequant_reference(&host, rows, groups, dim);
    let mut hidden = Bf16HiddenStates {
        data: ctx.stream.clone_htod(&host).expect("copy input"),
        hidden_dim: groups * dim,
        seq_len: rows,
    };

    hadamard_fp4_quant_bf16_hidden_in_place(&ctx, &mut hidden, groups, dim)
        .expect("Hadamard FP4 quant");
    let got = ctx.stream.clone_dtoh(&hidden.data).expect("copy output");
    ctx.sync().expect("sync");

    for (idx, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            *actual,
            *expected,
            "Hadamard FP4 quant mismatch idx={idx}: actual={} expected={}",
            actual.to_f32(),
            expected.to_f32()
        );
    }
}

#[test]
fn tilelang_fp8_linear_matches_power_of_two_reference() {
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let seq_len = 2usize;
    let in_dim = 4096usize;
    let out_dim = 512usize;
    let input_host = two_row_power_of_two_input(in_dim);
    let input = Bf16HiddenStates {
        data: ctx.stream.clone_htod(&input_host).expect("input h2d"),
        hidden_dim: in_dim,
        seq_len,
    };
    let weight = synthetic_gpu_tensor(
        &ctx,
        "synthetic.fp8.weight",
        Dtype::F8_E4M3,
        vec![out_dim, in_dim],
        vec![0x38; out_dim * in_dim],
    );
    let scale = synthetic_gpu_tensor(
        &ctx,
        "synthetic.fp8.scale",
        Dtype::F8_E8M0,
        vec![out_dim / 128, in_dim / 128],
        vec![0x7f; (out_dim / 128) * (in_dim / 128)],
    );
    let linear = quant_linear_ref(&weight, &scale);

    let out = fp8_linear_bf16_hidden(&ctx, &input, &linear).expect("fp8 linear");
    let got = ctx.stream.clone_dtoh(&out.data).expect("output dtoh");
    ctx.sync().expect("sync");
    assert_linear_power_of_two_output(&got, out_dim);
}

#[test]
fn tilelang_fp4_linear_matches_power_of_two_reference() {
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let seq_len = 2usize;
    let in_dim = 4096usize;
    let out_dim = 2048usize;
    let input_host = two_row_power_of_two_input(in_dim);
    let input = Bf16HiddenStates {
        data: ctx.stream.clone_htod(&input_host).expect("input h2d"),
        hidden_dim: in_dim,
        seq_len,
    };
    let weight = synthetic_gpu_tensor(
        &ctx,
        "synthetic.fp4.weight",
        Dtype::F4,
        vec![out_dim, in_dim],
        vec![0x22; out_dim * in_dim / 2],
    );
    let scale = synthetic_gpu_tensor(
        &ctx,
        "synthetic.fp4.scale",
        Dtype::F8_E8M0,
        vec![out_dim, in_dim / 32],
        vec![0x7f; out_dim * (in_dim / 32)],
    );
    let linear = quant_linear_ref(&weight, &scale);

    let out = fp4_linear_bf16_hidden(&ctx, &input, &linear).expect("fp4 linear");
    let got = ctx.stream.clone_dtoh(&out.data).expect("output dtoh");
    ctx.sync().expect("sync");
    assert_linear_power_of_two_output(&got, out_dim);
}

fn two_row_power_of_two_input(in_dim: usize) -> Vec<bf16> {
    let mut input = Vec::with_capacity(2 * in_dim);
    input.extend((0..in_dim).map(|_| bf16::from_f32(1.0)));
    input.extend((0..in_dim).map(|_| bf16::from_f32(-0.5)));
    input
}

fn synthetic_gpu_tensor(
    ctx: &RankGpuContext,
    name: &str,
    dtype: Dtype,
    shape: Vec<usize>,
    host: Vec<u8>,
) -> GpuRawTensor {
    GpuRawTensor {
        name: name.to_string(),
        dtype,
        shape,
        bytes: host.len(),
        data: ctx.stream.clone_htod(&host).expect("tensor h2d"),
    }
}

fn quant_linear_ref<'a>(weight: &'a GpuRawTensor, scale: &'a GpuRawTensor) -> QuantLinearRef<'a> {
    QuantLinearRef {
        weight: TensorRef {
            name: weight.name.as_str(),
            tensor: weight,
        },
        scale: TensorRef {
            name: scale.name.as_str(),
            tensor: scale,
        },
    }
}

fn assert_linear_power_of_two_output(got: &[bf16], out_dim: usize) {
    assert_eq!(got.len(), 2 * out_dim);
    for (idx, value) in got[..out_dim].iter().enumerate() {
        assert_eq!(
            *value,
            bf16::from_f32(4096.0),
            "row0 output mismatch idx={idx}"
        );
    }
    for (idx, value) in got[out_dim..].iter().enumerate() {
        assert_eq!(
            *value,
            bf16::from_f32(-2048.0),
            "row1 output mismatch idx={idx}"
        );
    }
}

fn official_hadamard_fp4_quant_dequant_reference(
    input: &[bf16],
    rows: usize,
    groups: usize,
    dim: usize,
) -> Vec<bf16> {
    assert_eq!(input.len(), rows * groups * dim);
    assert_eq!(dim, 128);
    let mut out = vec![bf16::ZERO; input.len()];
    let hadamard_scale = 1.0f32 / (dim as f32).sqrt();

    for row in 0..rows {
        for group in 0..groups {
            let base = row * groups * dim + group * dim;
            let mut values = (0..dim)
                .map(|idx| input[base + idx].to_f32() * hadamard_scale)
                .collect::<Vec<_>>();

            let mut stride = 1usize;
            while stride < dim {
                for idx in 0..dim {
                    if (idx & stride) == 0 {
                        let other = idx | stride;
                        let a = values[idx];
                        let b = values[other];
                        values[idx] = a + b;
                        values[other] = a - b;
                    }
                }
                stride <<= 1;
            }

            let rotated = values.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
            for block in 0..dim / 32 {
                let start = block * 32;
                let end = start + 32;
                let mut amax = 0.0f32;
                for value in &rotated[start..end] {
                    amax = amax.max(value.to_f32().abs());
                }
                amax = amax.max(6.0 * f32::MIN_POSITIVE);
                let quant_scale = official_fp4_power_of_two_scale(amax);
                for idx in start..end {
                    let value = rotated[idx].to_f32();
                    let clamped = (value / quant_scale).clamp(-6.0, 6.0);
                    let quantized = round_to_fp4_e2m1(clamped) * quant_scale;
                    out[base + idx] = bf16::from_f32(quantized);
                }
            }
        }
    }

    out
}

fn official_fp4_power_of_two_scale(amax: f32) -> f32 {
    let value = amax * (1.0 / 6.0);
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & ((1 << 23) - 1);
    let rounded_exponent = exponent + if mantissa != 0 { 1 } else { 0 };
    f32::from_bits(((rounded_exponent + 127) as u32) << 23)
}

fn round_to_fp4_e2m1(value: f32) -> f32 {
    const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let value = value.abs().min(6.0);
    let mut best = LEVELS[0];
    let mut best_dist = (value - best).abs();
    for candidate in LEVELS.into_iter().skip(1) {
        let dist = (value - candidate).abs();
        if dist <= best_dist {
            best = candidate;
            best_dist = dist;
        }
    }
    sign * best
}

#[test]
fn rank0_subset_copies_to_gpu_as_raw_bytes() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 GPU load test; {} does not exist",
            model_path.display()
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let loaded = load_rank_subset_to_gpu(
        &ctx,
        model_path,
        &config,
        TensorParallelConfig::mp8(0),
        &[
            "embed.weight",
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wq_a.scale",
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
        ],
    )
    .expect("copy subset to GPU");
    ctx.sync().expect("sync CUDA stream");

    assert_eq!(loaded["embed.weight"].bytes, 16160 * 4096 * 2);
    assert_eq!(loaded["layers.0.attn.wq_a.weight"].bytes, 1024 * 4096);
    assert_eq!(
        loaded["layers.0.ffn.experts.0.w1.weight"].bytes,
        2048 * 4096 / 2
    );
}

#[test]
fn rank0_subset_supports_typed_weight_view_for_layer0() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 typed view test; {} does not exist",
            model_path.display()
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let loaded = load_rank_subset_to_gpu(
        &ctx,
        model_path,
        &config,
        TensorParallelConfig::mp8(0),
        &[
            "embed.weight",
            "head.weight",
            "layers.0.attn_norm.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.hc_attn_fn",
            "layers.0.hc_attn_base",
            "layers.0.hc_attn_scale",
            "layers.0.hc_ffn_fn",
            "layers.0.hc_ffn_base",
            "layers.0.hc_ffn_scale",
            "layers.0.attn.attn_sink",
            "layers.0.attn.q_norm.weight",
            "layers.0.attn.kv_norm.weight",
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wq_a.scale",
            "layers.0.attn.wq_b.weight",
            "layers.0.attn.wq_b.scale",
            "layers.0.attn.wkv.weight",
            "layers.0.attn.wkv.scale",
            "layers.0.attn.wo_a.weight",
            "layers.0.attn.wo_b.weight",
            "layers.0.attn.wo_b.scale",
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.gate.tid2eid",
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.shared_experts.w1.scale",
            "layers.0.ffn.shared_experts.w2.weight",
            "layers.0.ffn.shared_experts.w2.scale",
            "layers.0.ffn.shared_experts.w3.weight",
            "layers.0.ffn.shared_experts.w3.scale",
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
            "layers.0.ffn.experts.0.w2.weight",
            "layers.0.ffn.experts.0.w2.scale",
            "layers.0.ffn.experts.0.w3.weight",
            "layers.0.ffn.experts.0.w3.scale",
        ],
    )
    .expect("copy typed-view subset to GPU");
    ctx.sync().expect("sync CUDA stream");

    let total_bytes = loaded.values().map(|tensor| tensor.bytes).sum();
    let weights = RankWeights {
        rank: 0,
        world_size: 8,
        tensors: loaded,
        total_bytes,
    };
    let view = weights.view(&config).expect("create typed view");
    assert_eq!(view.embed().expect("embed").tensor.shape, vec![16160, 4096]);
    assert_eq!(view.head().expect("head").tensor.shape, vec![16160, 4096]);
    let block0 = view.block(0).expect("layer0 block");
    assert!(block0.ffn.gate_tid2eid.is_some());
    assert!(block0.ffn.gate_bias.is_none());
    let expert0 = view.local_expert(0, 0).expect("local expert0");
    assert_eq!(expert0.w1.weight.tensor.dtype, Dtype::F4);
    assert_eq!(expert0.w2.weight.tensor.shape, vec![4096, 2048]);
}

#[test]
fn rank0_compressed_attention_weight_accessors_match_mp8_layout() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 compressed attention accessor test; {} does not exist",
            model_path.display()
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let loaded = load_rank_subset_to_gpu(
        &ctx,
        model_path,
        &config,
        TensorParallelConfig::mp8(0),
        &[
            "embed.weight",
            "head.weight",
            "layers.2.attn.compressor.ape",
            "layers.2.attn.compressor.norm.weight",
            "layers.2.attn.compressor.wgate.weight",
            "layers.2.attn.compressor.wkv.weight",
            "layers.2.attn.indexer.compressor.ape",
            "layers.2.attn.indexer.compressor.norm.weight",
            "layers.2.attn.indexer.compressor.wgate.weight",
            "layers.2.attn.indexer.compressor.wkv.weight",
            "layers.2.attn.indexer.weights_proj.weight",
            "layers.2.attn.indexer.wq_b.weight",
            "layers.2.attn.indexer.wq_b.scale",
            "layers.3.attn.compressor.ape",
            "layers.3.attn.compressor.norm.weight",
            "layers.3.attn.compressor.wgate.weight",
            "layers.3.attn.compressor.wkv.weight",
        ],
    )
    .expect("copy compressed attention subset to GPU");
    let total_bytes = loaded.values().map(|tensor| tensor.bytes).sum();
    let weights = RankWeights {
        rank: 0,
        world_size: 8,
        tensors: loaded,
        total_bytes,
    };
    let view = weights.view(&config).expect("view");

    let layer2_compressor = view
        .compressor(2)
        .expect("layer2 compressor accessor")
        .expect("layer2 compressor");
    let layer2_indexer = view
        .indexer(2)
        .expect("layer2 indexer accessor")
        .expect("layer2 indexer");
    assert_eq!(layer2_compressor.ape.tensor.shape, vec![4, 1024]);
    assert_eq!(layer2_compressor.wkv.tensor.shape, vec![1024, 4096]);
    assert_eq!(layer2_compressor.norm.tensor.shape, vec![512]);
    assert_eq!(layer2_indexer.wq_b.weight.tensor.shape, vec![1024, 1024]);
    assert_eq!(layer2_indexer.weights_proj.tensor.shape, vec![8, 4096]);
    assert_eq!(layer2_indexer.compressor.ape.tensor.shape, vec![4, 256]);
    assert_eq!(layer2_indexer.compressor.norm.tensor.shape, vec![128]);

    let layer3_compressor = view
        .compressor(3)
        .expect("layer3 compressor accessor")
        .expect("layer3 compressor");
    assert!(view.indexer(3).expect("layer3 indexer accessor").is_none());
    assert_eq!(layer3_compressor.ape.tensor.shape, vec![128, 512]);
    assert_eq!(layer3_compressor.wkv.tensor.shape, vec![512, 4096]);
    assert_eq!(layer3_compressor.wgate.tensor.shape, vec![512, 4096]);
    assert_eq!(layer3_compressor.norm.tensor.shape, vec![512]);
}

#[test]
fn rank0_full_gpu_load_builds_executor_owned_model() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 executor-owned model test; {} does not exist",
            model_path.display()
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx = RankGpuContext::new(0).expect("create CUDA context");
    let weights = pegainfer_deepseek_v4::load_rank_to_gpu(
        &ctx,
        model_path,
        &config,
        TensorParallelConfig::mp8(0),
    )
    .expect("load rank0 weights");
    ctx.sync().expect("sync CUDA stream");

    let model = DeepSeekRankModel::new(config, weights).expect("build model");
    assert_eq!(model.layers().len(), 43);
    assert_eq!(model.layers()[0].ffn.experts.len(), 32);
    assert_eq!(model.layers()[0].ffn.experts[0].global_expert, 0);
    assert_eq!(model.layers()[42].ffn.experts[31].global_expert, 31);
    assert_eq!(model.top().embed, "embed.weight");
    assert_eq!(model.top().norm, "norm.weight");
    assert_eq!(model.weights().tensors.len(), 10007);
}

#[test]
fn nccl_all_reduce_sums_bf16_hidden_states_pair() {
    if !nccl_runtime_loadable() {
        eprintln!(
            "skipping NCCL pair reduction test; cudarc cannot load libnccl from the current runtime path"
        );
        return;
    }

    let ctx0 = RankGpuContext::new(0).expect("ctx0");
    let ctx1 = RankGpuContext::new(1).expect("ctx1");
    let comms =
        cudarc::nccl::safe::Comm::from_devices(vec![ctx0.stream.clone(), ctx1.stream.clone()])
            .expect("NCCL comms");

    let mut h0 = Bf16HiddenStates {
        data: ctx0
            .stream
            .clone_htod(&[
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(2.0),
                half::bf16::from_f32(3.0),
            ])
            .expect("h0"),
        hidden_dim: 3,
        seq_len: 1,
    };
    let mut h1 = Bf16HiddenStates {
        data: ctx1
            .stream
            .clone_htod(&[
                half::bf16::from_f32(10.0),
                half::bf16::from_f32(20.0),
                half::bf16::from_f32(30.0),
            ])
            .expect("h1"),
        hidden_dim: 3,
        seq_len: 1,
    };

    let (comm0, comm1) = (&comms[0], &comms[1]);
    all_reduce_hidden_group(&mut [(comm0, &mut h0), (comm1, &mut h1)]).expect("all-reduce");
    let got0 = h0.to_host_f32(&ctx0).expect("got0");
    let got1 = h1.to_host_f32(&ctx1).expect("got1");
    assert_eq!(got0, vec![11.0, 22.0, 33.0]);
    assert_eq!(got1, vec![11.0, 22.0, 33.0]);
}

#[test]
fn bf16_window_prefill_cache_copy_uses_official_ring_layout() {
    let ctx = RankGpuContext::new(0).expect("ctx");
    let host = (0..6 * 4)
        .map(|value| bf16::from_f32(value as f32))
        .collect::<Vec<_>>();
    let src = Bf16HiddenStates {
        data: ctx.stream.clone_htod(&host).expect("copy source"),
        hidden_dim: 4,
        seq_len: 6,
    };
    let mut cache = Bf16Cache::zeros(&ctx, 4, 4).expect("cache");
    copy_window_prefill_to_ring_cache(&ctx, &src, &mut cache, 4).expect("copy ring");
    ctx.sync().expect("sync");

    let out = ctx.stream.clone_dtoh(&cache.data).expect("copy cache");
    let out = out
        .iter()
        .map(|value| value.to_f32() as i32)
        .collect::<Vec<_>>();
    assert_eq!(
        out,
        vec![
            16, 17, 18, 19, // source row 4 wraps to cache row 0
            20, 21, 22, 23, // source row 5 wraps to cache row 1
            8, 9, 10, 11, // source row 2 stays at cache row 2
            12, 13, 14, 15, // source row 3 stays at cache row 3
        ]
    );
}

#[test]
fn decode_window_indices_match_official_ring_order() {
    let ctx = RankGpuContext::new(0).expect("ctx");
    let (indices, topk) = window_topk_indices_decode(&ctx, 130, 128).expect("indices");
    assert_eq!(topk, 128);
    let host = ctx.stream.clone_dtoh(&indices).expect("copy indices");
    assert_eq!(host[0], 3);
    assert_eq!(host[124], 127);
    assert_eq!(host[125], 0);
    assert_eq!(host[126], 1);
    assert_eq!(host[127], 2);

    let (indices, _) = window_topk_indices_decode(&ctx, 3, 8).expect("early indices");
    let host = ctx.stream.clone_dtoh(&indices).expect("copy early indices");
    assert_eq!(host, vec![0, 1, 2, 3, -1, -1, -1, -1]);
}

#[test]
fn decode_compress_indices_match_official_prefix_order() {
    let ctx = RankGpuContext::new(0).expect("ctx");
    let (indices, topk) = compress_topk_indices_decode(&ctx, 130, 4, 128).expect("ratio4");
    assert_eq!(topk, 32);
    let host = ctx.stream.clone_dtoh(&indices).expect("copy ratio4");
    assert_eq!(host[0], 128);
    assert_eq!(host[31], 159);

    let (indices, topk) = compress_topk_indices_decode(&ctx, 127, 128, 128).expect("ratio128");
    assert_eq!(topk, 1);
    let host = ctx.stream.clone_dtoh(&indices).expect("copy ratio128");
    assert_eq!(host, vec![128]);
}

#[test]
fn layer0_attention_decode_rank_local_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 attention decode GPU path test; {} does not exist",
            model_path.display()
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx = RankGpuContext::new(0).expect("ctx");
    let weights = load_layer0_block_weights(model_path, &config, &ctx, 0);
    let view = weights.view(&config).expect("view");
    let attn = view.attention(0).expect("layer0 attention");
    let seq_len = 3;
    let host = (0..seq_len * config.dim)
        .map(|idx| bf16::from_f32(((idx % 97) as f32 - 48.0) * 0.001))
        .collect::<Vec<_>>();
    let full_input = Bf16HiddenStates {
        data: ctx.stream.clone_htod(&host).expect("full input"),
        hidden_dim: config.dim,
        seq_len,
    };
    let last_input = Bf16HiddenStates {
        data: ctx
            .stream
            .clone_htod(&host[(seq_len - 1) * config.dim..])
            .expect("last input"),
        hidden_dim: config.dim,
        seq_len: 1,
    };
    let rope = precompute_rope_cache(&ctx, &config, 0, seq_len).expect("rope");

    let mut full_projections =
        attention_project_bf16_hidden(&ctx, &config, &full_input, &attn).expect("full project");
    apply_rope_attention_projections(&ctx, &mut full_projections, &rope, 0).expect("full rope");
    let mut kv_cache =
        Bf16Cache::zeros(&ctx, config.head_dim, config.sliding_window).expect("decode kv cache");
    copy_window_prefill_to_ring_cache(
        &ctx,
        &full_projections.kv,
        &mut kv_cache,
        config.sliding_window,
    )
    .expect("prefill cache");
    let decode_out = attention_decode_rank_local_bf16_hidden(
        &ctx,
        &config,
        0,
        &last_input,
        &attn,
        &rope,
        seq_len - 1,
        &mut kv_cache,
    )
    .expect("decode attention");
    let decode = decode_out.to_host_f32(&ctx).expect("decode host");
    assert_eq!(decode.len(), config.dim);
    assert!(decode.iter().all(|value| value.is_finite()));
    assert!(decode.iter().any(|value| *value != 0.0));
}

#[test]
fn layer0_block_prefill_group_pair_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 group block runtime test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!("skipping DeepSeek V4 mp8 group block runtime test; cudarc cannot load libnccl");
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx0 = RankGpuContext::new(0).expect("ctx0");
    let ctx1 = RankGpuContext::new(1).expect("ctx1");
    let weights0 = load_layer0_block_weights(model_path, &config, &ctx0, 0);
    let weights1 = load_layer0_block_weights(model_path, &config, &ctx1, 1);
    let view0 = weights0.view(&config).expect("view0");
    let view1 = weights1.view(&config).expect("view1");
    let token_ids0 = ctx0.stream.clone_htod(&[0_u32]).expect("tokens0");
    let token_ids1 = ctx1.stream.clone_htod(&[0_u32]).expect("tokens1");
    let comms =
        cudarc::nccl::safe::Comm::from_devices(vec![ctx0.stream.clone(), ctx1.stream.clone()])
            .expect("NCCL comms");

    let hidden = embedding_vocab_parallel_group(
        &[
            (&ctx0, &view0, &comms[0], &token_ids0),
            (&ctx1, &view1, &comms[1], &token_ids1),
        ],
        &config,
        1,
    )
    .expect("vocab-parallel embedding");
    let hc0 = hc_expand_bf16_hidden(&ctx0, &hidden[0], config.hc_mult).expect("hc0");
    let hc1 = hc_expand_bf16_hidden(&ctx1, &hidden[1], config.hc_mult).expect("hc1");
    let rope0 = precompute_rope_cache(&ctx0, &config, 0, 1).expect("rope0");
    let rope1 = precompute_rope_cache(&ctx1, &config, 0, 1).expect("rope1");

    let out = block_prefill_group_bf16_hidden(
        &[
            (&ctx0, &view0, &comms[0], &hc0, &token_ids0),
            (&ctx1, &view1, &comms[1], &hc1, &token_ids1),
        ],
        &config,
        0,
        &[&rope0, &rope1],
        0,
    )
    .expect("group block prefill");
    let got0 = out[0].to_host_f32(&ctx0).expect("got0");
    let got1 = out[1].to_host_f32(&ctx1).expect("got1");

    assert_eq!(out[0].hidden_dim, config.dim);
    assert_eq!(out[1].hidden_dim, config.dim);
    assert_eq!(out[0].seq_len, 1);
    assert_eq!(out[1].seq_len, 1);
    assert_eq!(out[0].hc, config.hc_mult);
    assert_eq!(out[1].hc, config.hc_mult);
    assert_eq!(got0.len(), config.hc_mult * config.dim);
    assert_eq!(got1.len(), config.hc_mult * config.dim);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got1.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
    assert!(got1.iter().any(|value| *value != 0.0));
}

#[test]
fn layer0_block_decode_group_pair_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 group block decode test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!("skipping DeepSeek V4 mp8 group block decode test; cudarc cannot load libnccl");
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx0 = RankGpuContext::new(0).expect("ctx0");
    let ctx1 = RankGpuContext::new(1).expect("ctx1");
    let weights0 = load_layer0_block_weights(model_path, &config, &ctx0, 0);
    let weights1 = load_layer0_block_weights(model_path, &config, &ctx1, 1);
    let view0 = weights0.view(&config).expect("view0");
    let view1 = weights1.view(&config).expect("view1");
    let token_ids0 = ctx0.stream.clone_htod(&[1_u32]).expect("tokens0");
    let token_ids1 = ctx1.stream.clone_htod(&[1_u32]).expect("tokens1");
    let comms =
        cudarc::nccl::safe::Comm::from_devices(vec![ctx0.stream.clone(), ctx1.stream.clone()])
            .expect("NCCL comms");

    let hidden = embedding_vocab_parallel_group(
        &[
            (&ctx0, &view0, &comms[0], &token_ids0),
            (&ctx1, &view1, &comms[1], &token_ids1),
        ],
        &config,
        1,
    )
    .expect("vocab-parallel embedding");
    let hc0 = hc_expand_bf16_hidden(&ctx0, &hidden[0], config.hc_mult).expect("hc0");
    let hc1 = hc_expand_bf16_hidden(&ctx1, &hidden[1], config.hc_mult).expect("hc1");
    let rope0 = precompute_rope_cache(&ctx0, &config, 0, 2).expect("rope0");
    let rope1 = precompute_rope_cache(&ctx1, &config, 0, 2).expect("rope1");
    let mut kv_caches = [
        LayerDecodeCache::zeros(&ctx0, &config, 0).expect("cache0"),
        LayerDecodeCache::zeros(&ctx1, &config, 0).expect("cache1"),
    ];

    let out = block_decode_group_bf16_hidden(
        &[
            (&ctx0, &view0, &comms[0], &hc0, &token_ids0),
            (&ctx1, &view1, &comms[1], &hc1, &token_ids1),
        ],
        &config,
        0,
        &[&rope0, &rope1],
        1,
        &mut kv_caches,
    )
    .expect("group block decode");
    let got0 = out[0].to_host_f32(&ctx0).expect("got0");
    let got1 = out[1].to_host_f32(&ctx1).expect("got1");

    assert_eq!(out[0].hidden_dim, config.dim);
    assert_eq!(out[1].hidden_dim, config.dim);
    assert_eq!(out[0].seq_len, 1);
    assert_eq!(out[1].seq_len, 1);
    assert_eq!(out[0].hc, config.hc_mult);
    assert_eq!(out[1].hc, config.hc_mult);
    assert_eq!(got0.len(), config.hc_mult * config.dim);
    assert_eq!(got1.len(), config.hc_mult * config.dim);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got1.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
    assert!(got1.iter().any(|value| *value != 0.0));
}

#[test]
fn compressed_block_decode_group_pair_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 compressed block decode test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!(
            "skipping DeepSeek V4 mp8 compressed block decode test; cudarc cannot load libnccl"
        );
        return;
    }

    run_compressed_block_decode_pair(model_path, 2, 3, load_layer2_block_weights);
    run_compressed_block_decode_pair(model_path, 3, 127, load_layer3_block_weights);
}

#[test]
fn eight_rank_layer0_block_prefill_to_logits_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 8-rank layer0 GPU path test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!(
            "skipping DeepSeek V4 mp8 8-rank layer0 GPU path test; cudarc cannot load libnccl"
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let mut contexts = Vec::with_capacity(8);
    for device in 0..8 {
        match RankGpuContext::new(device) {
            Ok(ctx) => contexts.push(ctx),
            Err(err) => {
                eprintln!(
                    "skipping DeepSeek V4 mp8 8-rank layer0 GPU path test; device {device}: {err:?}"
                );
                return;
            }
        }
    }

    let weights = contexts
        .iter()
        .enumerate()
        .map(|(rank, ctx)| load_layer0_block_and_final_weights(model_path, &config, ctx, rank))
        .collect::<Vec<_>>();
    let views = weights
        .iter()
        .map(|weights| weights.view(&config).expect("view"))
        .collect::<Vec<_>>();
    let token_ids = contexts
        .iter()
        .map(|ctx| ctx.stream.clone_htod(&[0_u32]).expect("token ids"))
        .collect::<Vec<_>>();
    let streams = contexts
        .iter()
        .map(|ctx| ctx.stream.clone())
        .collect::<Vec<_>>();
    let comms = cudarc::nccl::safe::Comm::from_devices(streams).expect("NCCL comms");

    let embedding_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    let hidden =
        embedding_vocab_parallel_group(&embedding_inputs, &config, 1).expect("embedding group");
    let hcs = (0..8)
        .map(|rank| {
            hc_expand_bf16_hidden(&contexts[rank], &hidden[rank], config.hc_mult).expect("hc")
        })
        .collect::<Vec<_>>();
    let ropes = contexts
        .iter()
        .map(|ctx| precompute_rope_cache(ctx, &config, 0, 1).expect("rope"))
        .collect::<Vec<_>>();
    let rope_refs = ropes.iter().collect::<Vec<_>>();
    let block_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &hcs[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    let block_out =
        block_prefill_group_bf16_hidden(&block_inputs, &config, 0, &rope_refs, 0).expect("block");
    let logits_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &block_out[rank],
            )
        })
        .collect::<Vec<_>>();
    let logits = final_logits_group_bf16_hidden(&logits_inputs, &config).expect("logits");

    assert_eq!(logits.len(), 8);
    for rank in 0..8 {
        assert_eq!(logits[rank].vocab_size, config.vocab_size);
    }
    let got0 = logits[0].to_host(&contexts[0]).expect("rank0 logits");
    let got7 = logits[7].to_host(&contexts[7]).expect("rank7 logits");
    assert_eq!(got0.len(), config.vocab_size);
    assert_eq!(got7.len(), config.vocab_size);
    assert_eq!(got0, got7);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
}

#[test]
fn eight_rank_layer3_block_prefill_to_logits_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 8-rank layer3 GPU path test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!(
            "skipping DeepSeek V4 mp8 8-rank layer3 GPU path test; cudarc cannot load libnccl"
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let mut contexts = Vec::with_capacity(8);
    for device in 0..8 {
        match RankGpuContext::new(device) {
            Ok(ctx) => contexts.push(ctx),
            Err(err) => {
                eprintln!(
                    "skipping DeepSeek V4 mp8 8-rank layer3 GPU path test; device {device}: {err:?}"
                );
                return;
            }
        }
    }

    let weights = contexts
        .iter()
        .enumerate()
        .map(|(rank, ctx)| load_layer3_block_and_final_weights(model_path, &config, ctx, rank))
        .collect::<Vec<_>>();
    let views = weights
        .iter()
        .map(|weights| weights.view(&config).expect("view"))
        .collect::<Vec<_>>();
    let token_ids_host = (0..128).collect::<Vec<u32>>();
    let token_ids = contexts
        .iter()
        .map(|ctx| ctx.stream.clone_htod(&token_ids_host).expect("token ids"))
        .collect::<Vec<_>>();
    let streams = contexts
        .iter()
        .map(|ctx| ctx.stream.clone())
        .collect::<Vec<_>>();
    let comms = cudarc::nccl::safe::Comm::from_devices(streams).expect("NCCL comms");

    let embedding_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    let hidden =
        embedding_vocab_parallel_group(&embedding_inputs, &config, 128).expect("embedding group");
    let hcs = (0..8)
        .map(|rank| {
            hc_expand_bf16_hidden(&contexts[rank], &hidden[rank], config.hc_mult).expect("hc")
        })
        .collect::<Vec<_>>();
    let ropes = contexts
        .iter()
        .map(|ctx| precompute_rope_cache(ctx, &config, 3, 128).expect("rope"))
        .collect::<Vec<_>>();
    let rope_refs = ropes.iter().collect::<Vec<_>>();
    let block_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &hcs[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    let block_out =
        block_prefill_group_bf16_hidden(&block_inputs, &config, 3, &rope_refs, 0).expect("block");
    let logits_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &block_out[rank],
            )
        })
        .collect::<Vec<_>>();
    let logits = final_logits_group_bf16_hidden(&logits_inputs, &config).expect("logits");

    assert_eq!(logits.len(), 8);
    for rank in 0..8 {
        assert_eq!(logits[rank].vocab_size, config.vocab_size);
    }
    let got0 = logits[0].to_host(&contexts[0]).expect("rank0 logits");
    let got7 = logits[7].to_host(&contexts[7]).expect("rank7 logits");
    assert_eq!(got0.len(), config.vocab_size);
    assert_eq!(got7.len(), config.vocab_size);
    assert_eq!(got0, got7);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
}

#[test]
fn eight_rank_layer2_ratio4_block_prefill_to_logits_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 8-rank layer2 ratio4 GPU path test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!(
            "skipping DeepSeek V4 mp8 8-rank layer2 ratio4 GPU path test; cudarc cannot load libnccl"
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let mut contexts = Vec::with_capacity(8);
    for device in 0..8 {
        match RankGpuContext::new(device) {
            Ok(ctx) => contexts.push(ctx),
            Err(err) => {
                eprintln!(
                    "skipping DeepSeek V4 mp8 8-rank layer2 ratio4 GPU path test; device {device}: {err:?}"
                );
                return;
            }
        }
    }

    let weights = contexts
        .iter()
        .enumerate()
        .map(|(rank, ctx)| load_layer2_block_and_final_weights(model_path, &config, ctx, rank))
        .collect::<Vec<_>>();
    let views = weights
        .iter()
        .map(|weights| weights.view(&config).expect("view"))
        .collect::<Vec<_>>();
    let token_ids_host = (0..4).collect::<Vec<u32>>();
    let token_ids = contexts
        .iter()
        .map(|ctx| ctx.stream.clone_htod(&token_ids_host).expect("token ids"))
        .collect::<Vec<_>>();
    let streams = contexts
        .iter()
        .map(|ctx| ctx.stream.clone())
        .collect::<Vec<_>>();
    let comms = cudarc::nccl::safe::Comm::from_devices(streams).expect("NCCL comms");

    let embedding_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    let hidden =
        embedding_vocab_parallel_group(&embedding_inputs, &config, 4).expect("embedding group");
    let hcs = (0..8)
        .map(|rank| {
            hc_expand_bf16_hidden(&contexts[rank], &hidden[rank], config.hc_mult).expect("hc")
        })
        .collect::<Vec<_>>();
    let ropes = contexts
        .iter()
        .map(|ctx| precompute_rope_cache(ctx, &config, 2, 4).expect("rope"))
        .collect::<Vec<_>>();
    let rope_refs = ropes.iter().collect::<Vec<_>>();
    let block_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &hcs[rank],
                &token_ids[rank],
            )
        })
        .collect::<Vec<_>>();
    let block_out =
        block_prefill_group_bf16_hidden(&block_inputs, &config, 2, &rope_refs, 0).expect("block");
    let logits_inputs = (0..8)
        .map(|rank| {
            (
                &contexts[rank],
                &views[rank],
                &comms[rank],
                &block_out[rank],
            )
        })
        .collect::<Vec<_>>();
    let logits = final_logits_group_bf16_hidden(&logits_inputs, &config).expect("logits");

    assert_eq!(logits.len(), 8);
    for rank in 0..8 {
        assert_eq!(logits[rank].vocab_size, config.vocab_size);
    }
    let got0 = logits[0].to_host(&contexts[0]).expect("rank0 logits");
    let got7 = logits[7].to_host(&contexts[7]).expect("rank7 logits");
    assert_eq!(got0.len(), config.vocab_size);
    assert_eq!(got7.len(), config.vocab_size);
    assert_eq!(got0, got7);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
}

#[test]
fn layer2_ratio4_block_prefill_group_pair_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 layer2 ratio4 group block test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!(
            "skipping DeepSeek V4 mp8 layer2 ratio4 group block test; cudarc cannot load libnccl"
        );
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx0 = RankGpuContext::new(0).expect("ctx0");
    let ctx1 = RankGpuContext::new(1).expect("ctx1");
    let weights0 = load_layer2_block_weights(model_path, &config, &ctx0, 0);
    let weights1 = load_layer2_block_weights(model_path, &config, &ctx1, 1);
    let view0 = weights0.view(&config).expect("view0");
    let view1 = weights1.view(&config).expect("view1");
    let token_ids_host: Vec<u32> = (0..4).collect();
    let token_ids0 = ctx0.stream.clone_htod(&token_ids_host).expect("tokens0");
    let token_ids1 = ctx1.stream.clone_htod(&token_ids_host).expect("tokens1");
    let comms =
        cudarc::nccl::safe::Comm::from_devices(vec![ctx0.stream.clone(), ctx1.stream.clone()])
            .expect("NCCL comms");
    let hidden = embedding_vocab_parallel_group(
        &[
            (&ctx0, &view0, &comms[0], &token_ids0),
            (&ctx1, &view1, &comms[1], &token_ids1),
        ],
        &config,
        4,
    )
    .expect("vocab-parallel embedding");
    let hc0 = hc_expand_bf16_hidden(&ctx0, &hidden[0], config.hc_mult).expect("hc0");
    let hc1 = hc_expand_bf16_hidden(&ctx1, &hidden[1], config.hc_mult).expect("hc1");
    let rope0 = precompute_rope_cache(&ctx0, &config, 2, 4).expect("rope0");
    let rope1 = precompute_rope_cache(&ctx1, &config, 2, 4).expect("rope1");
    let out = block_prefill_group_bf16_hidden(
        &[
            (&ctx0, &view0, &comms[0], &hc0, &token_ids0),
            (&ctx1, &view1, &comms[1], &hc1, &token_ids1),
        ],
        &config,
        2,
        &[&rope0, &rope1],
        0,
    )
    .expect("layer2 ratio4 group block prefill");
    let got0 = out[0].to_host_f32(&ctx0).expect("got0");
    let got1 = out[1].to_host_f32(&ctx1).expect("got1");

    assert_eq!(out[0].hidden_dim, config.dim);
    assert_eq!(out[1].hidden_dim, config.dim);
    assert_eq!(out[0].seq_len, 4);
    assert_eq!(out[1].seq_len, 4);
    assert_eq!(out[0].hc, config.hc_mult);
    assert_eq!(out[1].hc, config.hc_mult);
    assert_eq!(got0.len(), config.hc_mult * config.dim * 4);
    assert_eq!(got1.len(), config.hc_mult * config.dim * 4);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got1.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
    assert!(got1.iter().any(|value| *value != 0.0));
}

#[test]
fn layer3_block_prefill_group_pair_runs_on_gpu() {
    let model_path = deepseek_model_path();
    let model_path = model_path.as_path();
    if !model_path.exists() {
        eprintln!(
            "skipping DeepSeek V4 mp8 layer3 group block test; {} does not exist",
            model_path.display()
        );
        return;
    }
    if !nccl_runtime_loadable() {
        eprintln!("skipping DeepSeek V4 mp8 layer3 group block test; cudarc cannot load libnccl");
        return;
    }

    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx0 = RankGpuContext::new(0).expect("ctx0");
    let ctx1 = RankGpuContext::new(1).expect("ctx1");
    let weights0 = load_layer3_block_weights(model_path, &config, &ctx0, 0);
    let weights1 = load_layer3_block_weights(model_path, &config, &ctx1, 1);
    let view0 = weights0.view(&config).expect("view0");
    let view1 = weights1.view(&config).expect("view1");
    let token_ids_host: Vec<u32> = (0..128).collect();
    let token_ids0 = ctx0.stream.clone_htod(&token_ids_host).expect("tokens0");
    let token_ids1 = ctx1.stream.clone_htod(&token_ids_host).expect("tokens1");
    let comms =
        cudarc::nccl::safe::Comm::from_devices(vec![ctx0.stream.clone(), ctx1.stream.clone()])
            .expect("NCCL comms");
    let hidden = embedding_vocab_parallel_group(
        &[
            (&ctx0, &view0, &comms[0], &token_ids0),
            (&ctx1, &view1, &comms[1], &token_ids1),
        ],
        &config,
        128,
    )
    .expect("vocab-parallel embedding");
    let hc0 = hc_expand_bf16_hidden(&ctx0, &hidden[0], config.hc_mult).expect("hc0");
    let hc1 = hc_expand_bf16_hidden(&ctx1, &hidden[1], config.hc_mult).expect("hc1");
    let rope0 = precompute_rope_cache(&ctx0, &config, 3, 128).expect("rope0");
    let rope1 = precompute_rope_cache(&ctx1, &config, 3, 128).expect("rope1");
    let out = block_prefill_group_bf16_hidden(
        &[
            (&ctx0, &view0, &comms[0], &hc0, &token_ids0),
            (&ctx1, &view1, &comms[1], &hc1, &token_ids1),
        ],
        &config,
        3,
        &[&rope0, &rope1],
        0,
    )
    .expect("layer3 group block prefill");
    let got0 = out[0].to_host_f32(&ctx0).expect("got0");
    let got1 = out[1].to_host_f32(&ctx1).expect("got1");

    assert_eq!(out[0].hidden_dim, config.dim);
    assert_eq!(out[1].hidden_dim, config.dim);
    assert_eq!(out[0].seq_len, 128);
    assert_eq!(out[1].seq_len, 128);
    assert_eq!(out[0].hc, config.hc_mult);
    assert_eq!(out[1].hc, config.hc_mult);
    assert_eq!(got0.len(), config.hc_mult * config.dim * 128);
    assert_eq!(got1.len(), config.hc_mult * config.dim * 128);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got1.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
    assert!(got1.iter().any(|value| *value != 0.0));
}

fn load_layer0_block_weights(
    model_path: &Path,
    config: &Config,
    ctx: &RankGpuContext,
    rank: usize,
) -> RankWeights {
    load_named_rank_weights(
        model_path,
        config,
        ctx,
        rank,
        layer0_block_tensor_names(config, rank),
        "copy layer0 block subset to GPU",
    )
}

fn load_layer0_block_and_final_weights(
    model_path: &Path,
    config: &Config,
    ctx: &RankGpuContext,
    rank: usize,
) -> RankWeights {
    let mut tensor_names = layer0_block_tensor_names(config, rank);
    for name in ["norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"] {
        tensor_names.push(name.to_string());
    }
    load_named_rank_weights(
        model_path,
        config,
        ctx,
        rank,
        tensor_names,
        "copy layer0 block and final head subset to GPU",
    )
}

fn load_layer3_block_weights(
    model_path: &Path,
    config: &Config,
    ctx: &RankGpuContext,
    rank: usize,
) -> RankWeights {
    load_named_rank_weights(
        model_path,
        config,
        ctx,
        rank,
        layer3_block_tensor_names(config, rank),
        "copy layer3 block subset to GPU",
    )
}

fn load_layer2_block_weights(
    model_path: &Path,
    config: &Config,
    ctx: &RankGpuContext,
    rank: usize,
) -> RankWeights {
    load_named_rank_weights(
        model_path,
        config,
        ctx,
        rank,
        layer2_block_tensor_names(config, rank),
        "copy layer2 block subset to GPU",
    )
}

fn run_compressed_block_decode_pair(
    model_path: &Path,
    layer: usize,
    start_pos: usize,
    load_weights: fn(&Path, &Config, &RankGpuContext, usize) -> RankWeights,
) {
    let config = Config::from_model_dir(model_path).expect("load config");
    let ctx0 = RankGpuContext::new(0).expect("ctx0");
    let ctx1 = RankGpuContext::new(1).expect("ctx1");
    let weights0 = load_weights(model_path, &config, &ctx0, 0);
    let weights1 = load_weights(model_path, &config, &ctx1, 1);
    let view0 = weights0.view(&config).expect("view0");
    let view1 = weights1.view(&config).expect("view1");
    let token_id = (start_pos % 17) as u32;
    let token_ids0 = ctx0.stream.clone_htod(&[token_id]).expect("tokens0");
    let token_ids1 = ctx1.stream.clone_htod(&[token_id]).expect("tokens1");
    let comms =
        cudarc::nccl::safe::Comm::from_devices(vec![ctx0.stream.clone(), ctx1.stream.clone()])
            .expect("NCCL comms");

    let hidden = embedding_vocab_parallel_group(
        &[
            (&ctx0, &view0, &comms[0], &token_ids0),
            (&ctx1, &view1, &comms[1], &token_ids1),
        ],
        &config,
        1,
    )
    .expect("vocab-parallel embedding");
    let hc0 = hc_expand_bf16_hidden(&ctx0, &hidden[0], config.hc_mult).expect("hc0");
    let hc1 = hc_expand_bf16_hidden(&ctx1, &hidden[1], config.hc_mult).expect("hc1");
    let rope0 = precompute_rope_cache(&ctx0, &config, layer, start_pos + 1).expect("rope0");
    let rope1 = precompute_rope_cache(&ctx1, &config, layer, start_pos + 1).expect("rope1");
    let mut caches = [
        LayerDecodeCache::zeros(&ctx0, &config, layer).expect("cache0"),
        LayerDecodeCache::zeros(&ctx1, &config, layer).expect("cache1"),
    ];

    let out = block_decode_group_bf16_hidden(
        &[
            (&ctx0, &view0, &comms[0], &hc0, &token_ids0),
            (&ctx1, &view1, &comms[1], &hc1, &token_ids1),
        ],
        &config,
        layer,
        &[&rope0, &rope1],
        start_pos,
        &mut caches,
    )
    .expect("compressed group block decode");
    let got0 = out[0].to_host_f32(&ctx0).expect("got0");
    let got1 = out[1].to_host_f32(&ctx1).expect("got1");

    assert_eq!(out[0].hidden_dim, config.dim);
    assert_eq!(out[1].hidden_dim, config.dim);
    assert_eq!(out[0].seq_len, 1);
    assert_eq!(out[1].seq_len, 1);
    assert_eq!(out[0].hc, config.hc_mult);
    assert_eq!(out[1].hc, config.hc_mult);
    assert_eq!(got0.len(), config.hc_mult * config.dim);
    assert_eq!(got1.len(), config.hc_mult * config.dim);
    assert!(got0.iter().all(|value| value.is_finite()));
    assert!(got1.iter().all(|value| value.is_finite()));
    assert!(got0.iter().any(|value| *value != 0.0));
    assert!(got1.iter().any(|value| *value != 0.0));
}

fn load_layer2_block_and_final_weights(
    model_path: &Path,
    config: &Config,
    ctx: &RankGpuContext,
    rank: usize,
) -> RankWeights {
    let mut tensor_names = layer2_block_tensor_names(config, rank);
    for name in ["norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"] {
        tensor_names.push(name.to_string());
    }
    load_named_rank_weights(
        model_path,
        config,
        ctx,
        rank,
        tensor_names,
        "copy layer2 block and final head subset to GPU",
    )
}

fn load_layer3_block_and_final_weights(
    model_path: &Path,
    config: &Config,
    ctx: &RankGpuContext,
    rank: usize,
) -> RankWeights {
    let mut tensor_names = layer3_block_tensor_names(config, rank);
    for name in ["norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"] {
        tensor_names.push(name.to_string());
    }
    load_named_rank_weights(
        model_path,
        config,
        ctx,
        rank,
        tensor_names,
        "copy layer3 block and final head subset to GPU",
    )
}

fn layer2_block_tensor_names(config: &Config, rank: usize) -> Vec<String> {
    let mut tensor_names = vec![
        "embed.weight".to_string(),
        "head.weight".to_string(),
        "layers.2.attn_norm.weight".to_string(),
        "layers.2.ffn_norm.weight".to_string(),
        "layers.2.hc_attn_fn".to_string(),
        "layers.2.hc_attn_base".to_string(),
        "layers.2.hc_attn_scale".to_string(),
        "layers.2.hc_ffn_fn".to_string(),
        "layers.2.hc_ffn_base".to_string(),
        "layers.2.hc_ffn_scale".to_string(),
        "layers.2.attn.q_norm.weight".to_string(),
        "layers.2.attn.kv_norm.weight".to_string(),
        "layers.2.attn.attn_sink".to_string(),
        "layers.2.attn.wq_a.weight".to_string(),
        "layers.2.attn.wq_a.scale".to_string(),
        "layers.2.attn.wq_b.weight".to_string(),
        "layers.2.attn.wq_b.scale".to_string(),
        "layers.2.attn.wkv.weight".to_string(),
        "layers.2.attn.wkv.scale".to_string(),
        "layers.2.attn.wo_a.weight".to_string(),
        "layers.2.attn.wo_b.weight".to_string(),
        "layers.2.attn.wo_b.scale".to_string(),
        "layers.2.attn.compressor.ape".to_string(),
        "layers.2.attn.compressor.norm.weight".to_string(),
        "layers.2.attn.compressor.wgate.weight".to_string(),
        "layers.2.attn.compressor.wkv.weight".to_string(),
        "layers.2.attn.indexer.compressor.ape".to_string(),
        "layers.2.attn.indexer.compressor.norm.weight".to_string(),
        "layers.2.attn.indexer.compressor.wgate.weight".to_string(),
        "layers.2.attn.indexer.compressor.wkv.weight".to_string(),
        "layers.2.attn.indexer.weights_proj.weight".to_string(),
        "layers.2.attn.indexer.wq_b.weight".to_string(),
        "layers.2.attn.indexer.wq_b.scale".to_string(),
        "layers.2.ffn.gate.weight".to_string(),
        "layers.2.ffn.gate.tid2eid".to_string(),
        "layers.2.ffn.shared_experts.w1.weight".to_string(),
        "layers.2.ffn.shared_experts.w1.scale".to_string(),
        "layers.2.ffn.shared_experts.w2.weight".to_string(),
        "layers.2.ffn.shared_experts.w2.scale".to_string(),
        "layers.2.ffn.shared_experts.w3.weight".to_string(),
        "layers.2.ffn.shared_experts.w3.scale".to_string(),
    ];
    let global_expert_start = rank * (config.n_routed_experts / 8);
    let global_expert_end = global_expert_start + config.n_routed_experts / 8;
    for expert in global_expert_start..global_expert_end {
        for linear in ["w1", "w2", "w3"] {
            tensor_names.push(format!("layers.2.ffn.experts.{expert}.{linear}.weight"));
            tensor_names.push(format!("layers.2.ffn.experts.{expert}.{linear}.scale"));
        }
    }
    tensor_names
}

fn layer3_block_tensor_names(config: &Config, rank: usize) -> Vec<String> {
    let mut tensor_names = vec![
        "embed.weight".to_string(),
        "head.weight".to_string(),
        "layers.3.attn_norm.weight".to_string(),
        "layers.3.ffn_norm.weight".to_string(),
        "layers.3.hc_attn_fn".to_string(),
        "layers.3.hc_attn_base".to_string(),
        "layers.3.hc_attn_scale".to_string(),
        "layers.3.hc_ffn_fn".to_string(),
        "layers.3.hc_ffn_base".to_string(),
        "layers.3.hc_ffn_scale".to_string(),
        "layers.3.attn.q_norm.weight".to_string(),
        "layers.3.attn.kv_norm.weight".to_string(),
        "layers.3.attn.attn_sink".to_string(),
        "layers.3.attn.wq_a.weight".to_string(),
        "layers.3.attn.wq_a.scale".to_string(),
        "layers.3.attn.wq_b.weight".to_string(),
        "layers.3.attn.wq_b.scale".to_string(),
        "layers.3.attn.wkv.weight".to_string(),
        "layers.3.attn.wkv.scale".to_string(),
        "layers.3.attn.wo_a.weight".to_string(),
        "layers.3.attn.wo_b.weight".to_string(),
        "layers.3.attn.wo_b.scale".to_string(),
        "layers.3.attn.compressor.ape".to_string(),
        "layers.3.attn.compressor.norm.weight".to_string(),
        "layers.3.attn.compressor.wgate.weight".to_string(),
        "layers.3.attn.compressor.wkv.weight".to_string(),
        "layers.3.ffn.gate.weight".to_string(),
        "layers.3.ffn.gate.bias".to_string(),
        "layers.3.ffn.shared_experts.w1.weight".to_string(),
        "layers.3.ffn.shared_experts.w1.scale".to_string(),
        "layers.3.ffn.shared_experts.w2.weight".to_string(),
        "layers.3.ffn.shared_experts.w2.scale".to_string(),
        "layers.3.ffn.shared_experts.w3.weight".to_string(),
        "layers.3.ffn.shared_experts.w3.scale".to_string(),
    ];
    let global_expert_start = rank * (config.n_routed_experts / 8);
    let global_expert_end = global_expert_start + config.n_routed_experts / 8;
    for expert in global_expert_start..global_expert_end {
        for linear in ["w1", "w2", "w3"] {
            tensor_names.push(format!("layers.3.ffn.experts.{expert}.{linear}.weight"));
            tensor_names.push(format!("layers.3.ffn.experts.{expert}.{linear}.scale"));
        }
    }
    tensor_names
}

fn layer0_block_tensor_names(config: &Config, rank: usize) -> Vec<String> {
    let mut tensor_names = vec![
        "embed.weight".to_string(),
        "head.weight".to_string(),
        "layers.0.attn_norm.weight".to_string(),
        "layers.0.ffn_norm.weight".to_string(),
        "layers.0.hc_attn_fn".to_string(),
        "layers.0.hc_attn_base".to_string(),
        "layers.0.hc_attn_scale".to_string(),
        "layers.0.hc_ffn_fn".to_string(),
        "layers.0.hc_ffn_base".to_string(),
        "layers.0.hc_ffn_scale".to_string(),
        "layers.0.attn.q_norm.weight".to_string(),
        "layers.0.attn.kv_norm.weight".to_string(),
        "layers.0.attn.attn_sink".to_string(),
        "layers.0.attn.wq_a.weight".to_string(),
        "layers.0.attn.wq_a.scale".to_string(),
        "layers.0.attn.wq_b.weight".to_string(),
        "layers.0.attn.wq_b.scale".to_string(),
        "layers.0.attn.wkv.weight".to_string(),
        "layers.0.attn.wkv.scale".to_string(),
        "layers.0.attn.wo_a.weight".to_string(),
        "layers.0.attn.wo_b.weight".to_string(),
        "layers.0.attn.wo_b.scale".to_string(),
        "layers.0.ffn.gate.weight".to_string(),
        "layers.0.ffn.gate.tid2eid".to_string(),
        "layers.0.ffn.shared_experts.w1.weight".to_string(),
        "layers.0.ffn.shared_experts.w1.scale".to_string(),
        "layers.0.ffn.shared_experts.w2.weight".to_string(),
        "layers.0.ffn.shared_experts.w2.scale".to_string(),
        "layers.0.ffn.shared_experts.w3.weight".to_string(),
        "layers.0.ffn.shared_experts.w3.scale".to_string(),
    ];
    let global_expert_start = rank * (config.n_routed_experts / 8);
    let global_expert_end = global_expert_start + config.n_routed_experts / 8;
    for expert in global_expert_start..global_expert_end {
        for linear in ["w1", "w2", "w3"] {
            tensor_names.push(format!("layers.0.ffn.experts.{expert}.{linear}.weight"));
            tensor_names.push(format!("layers.0.ffn.experts.{expert}.{linear}.scale"));
        }
    }
    tensor_names
}

fn load_named_rank_weights(
    model_path: &Path,
    config: &Config,
    ctx: &RankGpuContext,
    rank: usize,
    tensor_names: Vec<String>,
    expect_message: &str,
) -> RankWeights {
    let tensor_name_refs: Vec<&str> = tensor_names.iter().map(String::as_str).collect();
    let loaded = load_rank_subset_to_gpu(
        ctx,
        model_path,
        config,
        TensorParallelConfig::mp8(rank),
        &tensor_name_refs,
    )
    .expect(expect_message);
    let total_bytes = loaded.values().map(|tensor| tensor.bytes).sum();
    RankWeights {
        rank,
        world_size: 8,
        tensors: loaded,
        total_bytes,
    }
}

fn nccl_runtime_loadable() -> bool {
    use std::ffi::CString;

    [
        "libnccl.so",
        "libnccl.so.12",
        "libnccl.so.11",
        "libnccl.so.10",
        "libnccl.so.9",
        "libnccl.so.1",
    ]
    .iter()
    .any(|candidate| {
        let candidate = CString::new(*candidate).expect("static lib name");
        let handle = unsafe { libc::dlopen(candidate.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            false
        } else {
            unsafe { libc::dlclose(handle) };
            true
        }
    })
}
