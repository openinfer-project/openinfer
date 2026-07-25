//! Official-vLLM golden gates for the GLM5.2 MTP accuracy bring-up.
//!
//! The fixture is a five-row prompt forward captured from the official vLLM
//! nightly at commit `dcfebf93f4eccf30f71872283331eee757915daf`. It covers
//! the position-zero embedding mask, both input norms, the physical concat +
//! single BF16 `eh_proj` GEMM, layer 78, the shared-head recycle norm, and
//! sampled-row logits.
//!
//! Bookend operators are compared directly and should be bit-exact apart from
//! the cuBLAS GEMM tail. The full-layer reference runs vLLM's TP8 attention +
//! sequence-parallel MoE, while OpenInfer runs full attention + EP8 MoE.
//! Their reduction trees differ, so that gate bounds the hidden-state delta
//! and requires the draft top-1, top-8 set, and at least 30/32 top logits to
//! agree instead of pretending the two distributed topologies are bitwise
//! comparable.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use openinfer_kernels::ops::glm52_ep_deepep_unique_id;
use openinfer_kernels::tensor::DeviceContext;
use openinfer_kernels::tensor::DeviceMatrix;
use safetensors::Dtype;
use safetensors::SafeTensors;

use super::layer::GateLayerMlp;
use super::layer::LayerTensors;
use super::layer::load_decoder_layer;
use super::layer::load_rank_expert_bank;
use super::layer::model_path;
use super::layer_ep8::run_layer_prefill_ep8;
use super::layer_ep8::run_moe_ep8_rows;
use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_MTP_LAYER;
use crate::config::GLM52_VOCAB;
use crate::moe_ep8::Glm52MoeEp8State;
use crate::moe_ep8::glm52_moe_ep8_routed_forward;
use crate::mtp::Glm52MtpBookendWeights;
use crate::mtp::Glm52MtpScratch;
use crate::mtp::glm52_mtp_prepare_into;
use crate::mtp::glm52_mtp_recycle_into;
use crate::rows::Rows;
use crate::weights::Glm52WeightManifest;
use crate::weights::mmap_file;

const ROWS: usize = 5;
const EP_RANKS: usize = 8;
const VLLM_COMMIT: &str = "dcfebf93f4eccf30f71872283331eee757915daf";
const MODEL_CONFIG_SHA256: &str =
    "d1539d36be7546a1d827fe9cf74c55874695652efb6a5aaa3e60cde1c76ba819";
const MODEL_WEIGHT_INDEX_SHA256: &str =
    "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf";
const GOLDEN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/glm52-mtp-front-vllm-dcfebf93.safetensors"
));
const TP1_LAYER_GOLDEN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/glm52-mtp-layer78-vllm-tp1-dcfebf93.safetensors"
));

fn validate_fixture_metadata(bytes: &[u8], topology: &str) -> Result<()> {
    let (_, header) = SafeTensors::read_metadata(bytes)?;
    let metadata = header
        .metadata()
        .as_ref()
        .context("MTP fixture has no provenance metadata")?;
    for (key, expected) in [
        ("reference", "official-vllm"),
        ("vllm_commit", VLLM_COMMIT),
        ("topology", topology),
        ("model", "GLM-5.2-FP8"),
        ("model_config_sha256", MODEL_CONFIG_SHA256),
        ("model_weight_index_sha256", MODEL_WEIGHT_INDEX_SHA256),
    ] {
        ensure!(
            metadata.get(key).map(String::as_str) == Some(expected),
            "MTP fixture metadata {key:?} is {:?}, expected {expected:?}",
            metadata.get(key)
        );
    }
    Ok(())
}

fn bf16_tensor(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
    let view = tensors.tensor(name)?;
    ensure!(
        view.dtype() == Dtype::BF16 && view.shape() == shape,
        "MTP golden {name} must be BF16 {shape:?}, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    Ok(view
        .data()
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

fn positions(tensors: &SafeTensors<'_>) -> Result<Vec<u32>> {
    i64_tensor(tensors, "positions", &[ROWS])?
        .into_iter()
        .map(|position| u32::try_from(position).context("MTP golden position is outside u32"))
        .collect()
}

fn i64_tensor(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<i64>> {
    let view = tensors.tensor(name)?;
    ensure!(
        view.dtype() == Dtype::I64 && view.shape() == shape,
        "MTP golden {name} must be I64 {shape:?}, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    Ok(view
        .data()
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")))
        .collect())
}

fn i32_tensor(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<i32>> {
    let view = tensors.tensor(name)?;
    ensure!(
        view.dtype() == Dtype::I32 && view.shape() == shape,
        "MTP golden {name} must be I32 {shape:?}, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn f32_tensor(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = tensors.tensor(name)?;
    ensure!(
        view.dtype() == Dtype::F32 && view.shape() == shape,
        "MTP golden {name} must be F32 {shape:?}, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn checkpoint_tensor(
    manifest: &Glm52WeightManifest,
    model_path: &Path,
    name: &str,
) -> Result<Vec<u8>> {
    let shard = manifest.shard_for(name)?;
    let mmap = mmap_file(&model_path.join(shard))?;
    let tensors = SafeTensors::deserialize(mmap.as_ref())?;
    Ok(tensors.tensor(name)?.data().to_vec())
}

fn load_mtp_head_weights(
    ctx: &DeviceContext,
    manifest: &Glm52WeightManifest,
    model_path: &Path,
) -> Result<Glm52MtpBookendWeights> {
    let prefix = "model.layers.78";
    Glm52MtpBookendWeights::from_host(
        ctx,
        &checkpoint_tensor(manifest, model_path, &format!("{prefix}.enorm.weight"))?,
        &checkpoint_tensor(manifest, model_path, &format!("{prefix}.hnorm.weight"))?,
        &checkpoint_tensor(manifest, model_path, &format!("{prefix}.eh_proj.weight"))?,
        &checkpoint_tensor(
            manifest,
            model_path,
            &format!("{prefix}.shared_head.norm.weight"),
        )?,
    )
}

fn upload_rows<const C: usize>(ctx: &DeviceContext, host: &[bf16]) -> Result<Rows<C>> {
    ensure!(
        host.len().is_multiple_of(C),
        "MTP golden host tensor length {} is not divisible by {C}",
        host.len()
    );
    let mut rows = Rows::zeros(ctx, host.len() / C)?;
    ctx.stream.memcpy_htod(host, rows.data_mut())?;
    Ok(rows)
}

fn assert_close(
    ctx: &DeviceContext,
    label: &str,
    actual: &Rows<GLM52_HIDDEN>,
    expected: &[bf16],
    rms_limit: f32,
    p99_limit: f32,
) -> Result<()> {
    let actual = ctx
        .stream
        .clone_dtoh(actual.data())?
        .into_iter()
        .map(bf16::to_f32)
        .collect::<Vec<_>>();
    assert_close_values(label, &actual, expected, rms_limit, p99_limit)
}

fn assert_close_values(
    label: &str,
    actual: &[f32],
    expected: &[bf16],
    rms_limit: f32,
    p99_limit: f32,
) -> Result<()> {
    let expected = expected
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    assert_close_f32_values(label, actual, &expected, rms_limit, p99_limit)
}

fn assert_close_f32_values(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    rms_limit: f32,
    p99_limit: f32,
) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "{label}: actual length {} != expected {}",
        actual.len(),
        expected.len()
    );
    let exact = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let mut diffs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<_>>();
    diffs.sort_by(f32::total_cmp);
    let rms = (diffs.iter().map(|diff| diff * diff).sum::<f32>() / diffs.len() as f32).sqrt();
    let p99 = diffs[(diffs.len() * 99 / 100).min(diffs.len() - 1)];
    let max = diffs[diffs.len() - 1];
    println!(
        "{label}: exact={exact}/{} ({:.2}%) rms={rms:.6e} p99={p99:.6e} max={max:.6e}",
        diffs.len(),
        exact as f64 * 100.0 / diffs.len() as f64
    );
    ensure!(
        rms <= rms_limit && p99 <= p99_limit,
        "{label}: rms {rms:.6e} / p99 {p99:.6e} exceed limits \
         {rms_limit:.6e} / {p99_limit:.6e}"
    );
    Ok(())
}

fn assert_vllm_topk(
    actual_logits: &[bf16],
    expected_ids: &[i64],
    expected_values: &[bf16],
) -> Result<()> {
    ensure!(
        expected_ids.len() == expected_values.len() && !expected_ids.is_empty(),
        "MTP logits golden ids/values shape mismatch"
    );
    let mut ranked = (0..actual_logits.len()).collect::<Vec<_>>();
    ranked.sort_unstable_by(|&left, &right| {
        actual_logits[right]
            .to_f32()
            .total_cmp(&actual_logits[left].to_f32())
            .then_with(|| left.cmp(&right))
    });
    let actual_topk = &ranked[..expected_ids.len()];
    let expected_ids = expected_ids
        .iter()
        .map(|&id| usize::try_from(id).context("negative MTP golden token id"))
        .collect::<Result<Vec<_>>>()?;
    let overlap = actual_topk
        .iter()
        .filter(|id| expected_ids.contains(id))
        .count();
    let top8_overlap = actual_topk[..8]
        .iter()
        .filter(|id| expected_ids[..8].contains(id))
        .count();
    println!(
        "mtp/logits: top1 actual={} expected={}, top8 overlap={top8_overlap}, \
         top{} overlap={overlap}, actual_top8={:?}",
        actual_topk[0],
        expected_ids[0],
        expected_ids.len(),
        &actual_topk[..8.min(actual_topk.len())],
    );
    let actual_expected_values = expected_ids
        .iter()
        .map(|&id| actual_logits[id].to_f32())
        .collect::<Vec<_>>();
    assert_close_values(
        "mtp/logits/expected_ids",
        &actual_expected_values,
        expected_values,
        2.5e-1,
        5.0e-1,
    )?;
    ensure!(
        actual_topk[0] == expected_ids[0],
        "MTP draft top-1 differs from official vLLM"
    );
    ensure!(
        top8_overlap == 8,
        "MTP draft top-8 set overlap with official vLLM is {top8_overlap}/8"
    );
    ensure!(
        overlap + 2 >= expected_ids.len(),
        "MTP draft top-{} overlap with official vLLM is {overlap}/{}",
        expected_ids.len(),
        expected_ids.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA + GLM-5.2-FP8 checkpoint"]
fn mtp_front_vllm_golden_gate() -> Result<()> {
    validate_fixture_metadata(GOLDEN, "tp8-ep0")?;
    let fixture = SafeTensors::deserialize(GOLDEN)?;
    let model_path = model_path();
    let manifest = Glm52WeightManifest::from_model_dir(&model_path)?;
    let ctx = DeviceContext::new()?;
    let weights = load_mtp_head_weights(&ctx, &manifest, &model_path)?;

    let positions = ctx.stream.clone_htod(&positions(&fixture)?)?;
    let inputs_embeds = upload_rows::<GLM52_HIDDEN>(
        &ctx,
        &bf16_tensor(&fixture, "inputs_embeds_raw", &[ROWS, GLM52_HIDDEN])?,
    )?;
    let previous_hidden = upload_rows::<GLM52_HIDDEN>(
        &ctx,
        &bf16_tensor(&fixture, "previous_hidden_raw", &[ROWS, GLM52_HIDDEN])?,
    )?;
    let mut scratch = Glm52MtpScratch::new(&ctx, ROWS)?;
    let mut decoder_input = Rows::zeros(&ctx, ROWS)?;
    glm52_mtp_prepare_into(
        &ctx,
        &weights,
        &positions,
        &inputs_embeds,
        &previous_hidden,
        &mut scratch,
        &mut decoder_input,
    )?;

    assert_close(
        &ctx,
        "mtp/enorm",
        scratch.normed_embed(),
        &bf16_tensor(&fixture, "inputs_embeds_norm", &[ROWS, GLM52_HIDDEN])?,
        1.0e-3,
        3.90625e-3,
    )?;
    assert_close(
        &ctx,
        "mtp/hnorm",
        scratch.normed_previous(),
        &bf16_tensor(&fixture, "previous_hidden_norm", &[ROWS, GLM52_HIDDEN])?,
        1.0e-3,
        3.90625e-3,
    )?;
    assert_close(
        &ctx,
        "mtp/eh_proj",
        &decoder_input,
        &bf16_tensor(&fixture, "eh_proj", &[ROWS, GLM52_HIDDEN])?,
        1.0e-3,
        3.90625e-3,
    )?;

    let raw_hidden = upload_rows::<GLM52_HIDDEN>(
        &ctx,
        &bf16_tensor(&fixture, "raw_hidden", &[ROWS, GLM52_HIDDEN])?,
    )?;
    let mut recycle_hidden = Rows::zeros(&ctx, ROWS)?;
    glm52_mtp_recycle_into(&ctx, &weights, &raw_hidden, &mut recycle_hidden)?;
    assert_close(
        &ctx,
        "mtp/shared_norm",
        &recycle_hidden,
        &bf16_tensor(&fixture, "recycle_hidden", &[ROWS, GLM52_HIDDEN])?,
        1.0e-3,
        3.90625e-3,
    )
}

#[test]
#[ignore = "requires 8×H200 + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4 + DeepGEMM env"]
fn mtp_layer78_vllm_ep8_golden_gate() -> Result<()> {
    validate_fixture_metadata(GOLDEN, "tp8-ep0")?;
    validate_fixture_metadata(TP1_LAYER_GOLDEN, "tp1-ep0")?;
    let fixture = SafeTensors::deserialize(GOLDEN)?;
    let tp1_fixture = SafeTensors::deserialize(TP1_LAYER_GOLDEN)?;
    let expected_post_attention = bf16_tensor(&fixture, "decoder_residual", &[ROWS, GLM52_HIDDEN])?;
    let expected_tp8_mlp = bf16_tensor(&fixture, "decoder_hidden", &[ROWS, GLM52_HIDDEN])?;
    let expected_tp1_mlp = bf16_tensor(&tp1_fixture, "decoder_hidden", &[ROWS, GLM52_HIDDEN])?;
    let expected_tp1_normed =
        bf16_tensor(&tp1_fixture, "post_attention_norm", &[ROWS, GLM52_HIDDEN])?;
    let expected_tp1_topk_ids = i32_tensor(&tp1_fixture, "topk_ids", &[ROWS, 8])?;
    let expected_tp1_topk_weights = f32_tensor(&tp1_fixture, "topk_weights", &[ROWS, 8])?;
    let expected_tp1_routed = bf16_tensor(&tp1_fixture, "routed_hidden", &[ROWS, GLM52_HIDDEN])?;
    let expected_tp1_shared_gate_up = bf16_tensor(
        &tp1_fixture,
        "shared_gate_up",
        &[
            ROWS,
            2 * crate::moe_decode::GLM52_SHARED_EXPERT_INTERMEDIATE,
        ],
    )?;
    let expected_tp1_shared_silu = bf16_tensor(
        &tp1_fixture,
        "shared_silu",
        &[ROWS, crate::moe_decode::GLM52_SHARED_EXPERT_INTERMEDIATE],
    )?;
    let expected_tp1_shared = bf16_tensor(&tp1_fixture, "shared_hidden", &[ROWS, GLM52_HIDDEN])?;
    let expected_tp1_recycle = bf16_tensor(&tp1_fixture, "recycle_hidden", &[ROWS, GLM52_HIDDEN])?;
    let expected = bf16_tensor(&fixture, "raw_hidden", &[ROWS, GLM52_HIDDEN])?;
    let model_path = model_path();
    let unique_id = glm52_ep_deepep_unique_id(EP_RANKS)?;
    let tensors = Arc::new(LayerTensors::load(&model_path, GLM52_MTP_LAYER)?);

    let handles: Vec<_> = (1..EP_RANKS)
        .map(|rank| {
            let tensors = Arc::clone(&tensors);
            std::thread::Builder::new()
                .name(format!("mtp-ep8-gate-rank-{rank}"))
                .spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank =
                        load_rank_expert_bank(&ctx, &tensors, GLM52_MTP_LAYER, rank, EP_RANKS)?;
                    let mut ep8 = Glm52MoeEp8State::new(&ctx, &unique_id, EP_RANKS, rank)?;
                    for _ in 0..2 * ROWS {
                        let dispatched =
                            glm52_moe_ep8_routed_forward(&ctx, &mut ep8, &bank, None, EP_RANKS)?;
                        ensure!(!dispatched, "expert rank produced a combined output");
                    }
                    Ok(())
                })
                .expect("spawn MTP EP8 gate rank thread")
        })
        .collect();

    let ctx = DeviceContext::new_with_device(0)?;
    let manifest = Glm52WeightManifest::from_model_dir(&model_path)?;
    let mtp_weights = load_mtp_head_weights(&ctx, &manifest, &model_path)?;
    let positions = ctx.stream.clone_htod(&positions(&fixture)?)?;
    let inputs_embeds = upload_rows::<GLM52_HIDDEN>(
        &ctx,
        &bf16_tensor(&fixture, "inputs_embeds_raw", &[ROWS, GLM52_HIDDEN])?,
    )?;
    let previous_hidden = upload_rows::<GLM52_HIDDEN>(
        &ctx,
        &bf16_tensor(&fixture, "previous_hidden_raw", &[ROWS, GLM52_HIDDEN])?,
    )?;
    let mut mtp_scratch = Glm52MtpScratch::new(&ctx, ROWS)?;
    let mut prepared_decoder_input = Rows::zeros(&ctx, ROWS)?;
    glm52_mtp_prepare_into(
        &ctx,
        &mtp_weights,
        &positions,
        &inputs_embeds,
        &previous_hidden,
        &mut mtp_scratch,
        &mut prepared_decoder_input,
    )?;
    let prepared_decoder_input = ctx.stream.clone_dtoh(prepared_decoder_input.data())?;
    let weights = load_decoder_layer(
        &ctx,
        &model_path,
        GLM52_MTP_LAYER,
        GateLayerMlp::MoeEp8Rank0,
    )?;
    let mut ep8 = Glm52MoeEp8State::new(&ctx, &unique_id, EP_RANKS, 0)?;
    let actual = run_layer_prefill_ep8(
        &ctx,
        &weights,
        &mut ep8,
        &prepared_decoder_input,
        ROWS,
        EP_RANKS,
    );
    let isolated = run_moe_ep8_rows(
        &ctx,
        &weights,
        &mut ep8,
        &expected_post_attention,
        &expected_tp1_normed,
        ROWS,
        EP_RANKS,
    );

    drop(ep8);
    for (rank, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .expect("MTP EP8 gate rank thread panicked")
            .with_context(|| format!("MTP EP8 gate rank {}", rank + 1))?;
    }
    let actual = actual?;
    let isolated = isolated?;
    let matching_topk = isolated
        .topk_ids
        .iter()
        .zip(&expected_tp1_topk_ids)
        .filter(|(actual, expected)| actual == expected)
        .count();
    println!(
        "mtp/layer78/topk_ids_vs_vllm_tp1: exact={matching_topk}/{} actual={:?} expected={:?}",
        expected_tp1_topk_ids.len(),
        isolated.topk_ids,
        expected_tp1_topk_ids
    );
    ensure!(
        isolated.topk_ids == expected_tp1_topk_ids,
        "MTP layer 78 router top-k IDs differ from official vLLM TP1"
    );
    assert_close_values(
        "mtp/layer78/post_attention_norm_vs_vllm_tp1",
        &isolated.normed,
        &expected_tp1_normed,
        1.0e-3,
        4.0e-3,
    )?;
    assert_close_f32_values(
        "mtp/layer78/topk_weights_vs_vllm_tp1",
        &isolated.topk_weights,
        &expected_tp1_topk_weights,
        1.0e-6,
        1.0e-6,
    )?;
    assert_close_values(
        "mtp/layer78/shared_gate_up_vs_vllm_tp1",
        &isolated.shared_gate_up,
        &expected_tp1_shared_gate_up,
        1.25e-2,
        4.0e-2,
    )?;
    assert_close_values(
        "mtp/layer78/routed_vs_vllm_tp1",
        &isolated.routed,
        &expected_tp1_routed,
        7.0e-3,
        2.1e-2,
    )?;
    assert_close_values(
        "mtp/layer78/shared_silu_vs_vllm_tp1",
        &isolated.shared_silu,
        &expected_tp1_shared_silu,
        6.0e-3,
        2.1e-2,
    )?;
    assert_close_values(
        "mtp/layer78/shared_vs_vllm_tp1",
        &isolated.shared,
        &expected_tp1_shared,
        6.0e-3,
        1.8e-2,
    )?;
    assert_close_values(
        "mtp/layer78/post_attention",
        &actual.post_attention,
        &expected_post_attention,
        1.2e-2,
        3.125e-2,
    )?;
    assert_close_values(
        "mtp/layer78/mlp",
        &actual.mlp,
        &expected_tp8_mlp,
        2.5e-2,
        7.8125e-2,
    )?;
    assert_close_values(
        "mtp/layer78/mlp_from_vllm_tp8_residual",
        &isolated.mlp,
        &expected_tp8_mlp,
        1.2e-2,
        3.125e-2,
    )?;
    assert_close_values(
        "mtp/layer78/mlp_vs_vllm_tp1",
        &isolated.mlp,
        &expected_tp1_mlp,
        1.0e-2,
        3.125e-2,
    )?;
    for row in 0..ROWS {
        let range = row * GLM52_HIDDEN..(row + 1) * GLM52_HIDDEN;
        assert_close_values(
            &format!("mtp/layer78/row{row}"),
            &actual.hidden[range.clone()],
            &expected[range],
            3.5e-2,
            9.375e-2,
        )?;
    }
    let sampled_row = usize::try_from(i64_tensor(&fixture, "logits_sampled_row", &[1])?[0])
        .context("negative MTP logits sampled row")?;
    ensure!(
        sampled_row < ROWS,
        "MTP logits sampled row {sampled_row} is outside {ROWS} rows"
    );
    let lm_head = DeviceMatrix::from_safetensors(
        &ctx,
        &checkpoint_tensor(&manifest, &model_path, "lm_head.weight")?,
        GLM52_VOCAB,
        GLM52_HIDDEN,
    )?;
    let raw_hidden = actual
        .hidden
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let raw_hidden = upload_rows::<GLM52_HIDDEN>(&ctx, &raw_hidden)?;
    let mut recycle_hidden = Rows::<GLM52_HIDDEN>::zeros(&ctx, ROWS)?;
    glm52_mtp_recycle_into(&ctx, &mtp_weights, &raw_hidden, &mut recycle_hidden)?;
    let recycle_host = ctx
        .stream
        .clone_dtoh(recycle_hidden.data())?
        .into_iter()
        .map(bf16::to_f32)
        .collect::<Vec<_>>();
    assert_close_values(
        "mtp/recycle_vllm_tp1_vs_tp8",
        &expected_tp1_recycle
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>(),
        &bf16_tensor(&fixture, "recycle_hidden", &[ROWS, GLM52_HIDDEN])?,
        2.5e-2,
        7.8125e-2,
    )?;
    // The shared RMSNorm amplifies the EP8/full-attention raw-hidden delta.
    // Bound that state explicitly, then let the stricter top-k checks below
    // decide whether the amplified delta changes draft-token decisions.
    assert_close_values(
        "mtp/recycle_chained_vs_vllm_tp1",
        &recycle_host,
        &expected_tp1_recycle,
        1.1e-1,
        3.125e-1,
    )?;
    assert_close_values(
        "mtp/recycle_chained_vs_vllm_tp8",
        &recycle_host,
        &bf16_tensor(&fixture, "recycle_hidden", &[ROWS, GLM52_HIDDEN])?,
        1.1e-1,
        3.125e-1,
    )?;
    let mut logits = Rows::<GLM52_VOCAB>::zeros(&ctx, ROWS)?;
    glm52_lm_head_into(&ctx, &recycle_hidden, &lm_head, &mut logits)?;
    let logits = ctx.stream.clone_dtoh(logits.data())?;
    let logits = &logits[sampled_row * GLM52_VOCAB..(sampled_row + 1) * GLM52_VOCAB];
    assert_vllm_topk(
        logits,
        &i64_tensor(&fixture, "logits_topk_ids", &[1, 32])?,
        &bf16_tensor(&fixture, "logits_topk_values", &[1, 32])?,
    )?;
    assert_close_values("mtp/layer78", &actual.hidden, &expected, 2.5e-2, 7.8125e-2)
}
