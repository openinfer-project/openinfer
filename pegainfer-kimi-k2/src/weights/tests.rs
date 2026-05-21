use super::*;
use crate::config::{
    KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_INT4_GROUP_SIZE, KIMI_K2_MOE_LAYERS,
    KIMI_K2_Q_LORA_RANK, KIMI_K2_TOPK, KIMI_K2_VOCAB,
};
use half::bf16;
use pegainfer_kernels::{
    ops::{
        KimiMarlinRouteWorkspace, KimiMarlinWna16Workspace, kimi_marlin_sum_topk_rows_f32,
        kimi_marlin_w13_swiglu, kimi_marlin_wna16_w2_gemm, kimi_marlin_wna16_w13_gemm,
        kimi_moe_marlin_align_block_size,
    },
    tensor::HiddenStates,
};
use safetensors::tensor::{TensorView, serialize};
use serde_json::json;
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn real_kimi_index_scans_when_present() {
    let path = Path::new("/data/models/Kimi-K2.6/model.safetensors.index.json");
    if !path.exists() {
        return;
    }
    let manifest = KimiK2WeightManifest::from_index_file(path).unwrap();
    assert_eq!(manifest.layers.len(), KIMI_K2_LAYERS);
    assert_eq!(manifest.text_tensor_count, 208_215);
    assert_eq!(manifest.ignored_non_text_tensor_count, 335);
    assert_eq!(manifest.shard_count, 64);
    let rank7 = manifest.rank_plan(7).unwrap();
    assert_eq!(rank7.attention_head_range, 56..64);
    assert_eq!(rank7.vocab_range, 143_360..163_840);
    assert_eq!(rank7.local_expert_range, 336..384);
    assert_eq!(rank7.tensor_count, 26_775);
    let shard_plan = manifest.rank_shard_plan(7).unwrap();
    assert_eq!(shard_plan.rank, 7);
    assert_eq!(shard_plan.shards.len(), 62);
    assert_eq!(shard_plan.tensor_count, rank7.tensor_count);
    assert!(shard_plan.shards.iter().any(|shard| {
        shard.shard == "model-00062-of-000064.safetensors"
            && shard
                .tensors
                .iter()
                .any(|name| name == "language_model.lm_head.weight")
    }));
}

#[test]
#[ignore = "H20-only: loads rank0 sliced Kimi-K2.5 payload into GPU memory"]
fn h20_kimi_k25_rank0_sliced_payload_loads_typed_gpu_view() {
    let model_path = Path::new("/data/models/Kimi-K2.5");
    assert!(
        model_path.join(KIMI_K2_WEIGHT_INDEX).exists(),
        "missing H20 Kimi-K2.5 weights at {}",
        model_path.display()
    );

    let manifest = KimiK2WeightManifest::from_model_dir(model_path).unwrap();
    let names = manifest.rank_weight_names(0).unwrap();
    let load_plan = manifest.rank_sliced_load_plan(0).unwrap();
    assert_eq!(load_plan.rank, 0);
    assert_eq!(load_plan.tensor_count, names.plan.tensor_count);
    assert_eq!(names.plan.attention_head_range, 0..8);
    assert_eq!(names.plan.vocab_range, 0..(KIMI_K2_VOCAB / 8));
    assert_eq!(names.plan.local_expert_range, 0..48);

    let ctx = KimiRankGpuContext::new(0).unwrap();
    let weights = load_rank_sliced_weights_to_gpu(&ctx, model_path, &load_plan).unwrap();
    assert_eq!(weights.rank, 0);
    assert_eq!(weights.tensors.len(), names.plan.tensor_count);
    assert!(weights.total_bytes > 0);

    let typed = weights.typed_view(&names).unwrap();
    assert_eq!(typed.rank, 0);
    assert_eq!(typed.layers.len(), KIMI_K2_LAYERS);
    assert_eq!(
        typed.top.token_embedding.shape.as_slice(),
        &[KIMI_K2_VOCAB / 8, KIMI_K2_HIDDEN]
    );
    assert_eq!(
        typed.layers[0].attention.q_b_proj.shape.as_slice(),
        &[KIMI_K2_Q_LORA_RANK, KIMI_K2_Q_LORA_RANK]
    );
    match &typed.layers[1].kind {
        KimiLayerKindGpuWeights::Moe(moe) => {
            assert_eq!(moe.routed_experts.len(), 48);
            assert_eq!(moe.routed_experts[0].global_expert, 0);
            assert_eq!(moe.routed_experts[47].global_expert, 47);
        }
        KimiLayerKindGpuWeights::Dense(_) => panic!("layer1 must be MoE"),
    };
    let expert_major = typed.expert_major_weight_plan().unwrap();
    assert_eq!(expert_major.rank, 0);
    assert_eq!(expert_major.layers.len(), KIMI_K2_MOE_LAYERS);
    assert_eq!(expert_major.local_expert_range, 0..48);
    assert_eq!(expert_major.layers[0].layer_idx, 1);
    assert_eq!(
        expert_major.layers[0].gate.packed_i32_shape_per_expert,
        [KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN / 8]
    );
    assert_eq!(
        expert_major.layers[0].down.packed_i32_shape_per_expert,
        [KIMI_K2_HIDDEN, KIMI_K2_EXPERT_INTERMEDIATE / 8]
    );
    let marlin_layer = typed
        .pack_expert_major_layer_marlin_weights(&ctx, expert_major.layers[0].layer_idx)
        .unwrap();
    ctx.sync().unwrap();
    assert_eq!(
        marlin_layer.raw_source_bytes,
        expert_major.layers[0].total_bytes
    );
    assert!(marlin_layer.total_bytes < marlin_layer.raw_source_bytes);
    assert_eq!(
        marlin_layer.w13.weight_packed_marlin_uint4b8.len(),
        2 * 48 * KIMI_K2_EXPERT_INTERMEDIATE * (KIMI_K2_HIDDEN / 2)
    );
    assert_eq!(
        marlin_layer.w13.weight_scale_marlin_permuted.len(),
        2 * 48 * KIMI_K2_EXPERT_INTERMEDIATE * (KIMI_K2_HIDDEN / KIMI_K2_INT4_GROUP_SIZE)
    );
    marlin_layer.as_marlin_weights().validate().unwrap();
}

#[test]
#[ignore = "H20-only: packages rank0 Kimi-K2.5 routed experts into Marlin WNA16 runtime buffers"]
fn h20_kimi_k25_rank0_marlin_expert_package_loads() {
    let model_path = Path::new("/data/models/Kimi-K2.5");
    assert!(
        model_path.join(KIMI_K2_WEIGHT_INDEX).exists(),
        "missing H20 Kimi-K2.5 weights at {}",
        model_path.display()
    );

    let manifest = KimiK2WeightManifest::from_model_dir(model_path).unwrap();
    let names = manifest.rank_weight_names(0).unwrap();
    let load_plan = manifest.rank_sliced_load_plan(0).unwrap();
    let ctx = KimiRankGpuContext::new(0).unwrap();
    let mut weights = load_rank_sliced_weights_to_gpu(&ctx, model_path, &load_plan).unwrap();
    let original_total_bytes = weights.total_bytes;
    let marlin_weights = weights
        .pack_rank_expert_marlin_weights(&ctx, &names)
        .unwrap();
    ctx.sync().unwrap();

    assert_eq!(marlin_weights.rank, 0);
    assert_eq!(marlin_weights.local_expert_range, 0..48);
    assert_eq!(marlin_weights.layers.len(), KIMI_K2_MOE_LAYERS);
    assert_eq!(marlin_weights.layers[0].layer_idx, 1);
    assert_eq!(marlin_weights.layers[59].layer_idx, 60);
    assert_eq!(
        original_total_bytes,
        weights.total_bytes + marlin_weights.raw_source_bytes
    );
    assert!(marlin_weights.total_bytes < marlin_weights.raw_source_bytes);
    let layer1 = &marlin_weights.layers[0];
    layer1.as_marlin_weights().validate().unwrap();
    assert_eq!(
        layer1.w13.weight_packed_marlin_uint4b8.len(),
        2 * 48 * KIMI_K2_EXPERT_INTERMEDIATE * (KIMI_K2_HIDDEN / 2)
    );
    assert_eq!(
        layer1.w13.weight_scale_marlin_permuted.len(),
        2 * 48 * KIMI_K2_EXPERT_INTERMEDIATE * (KIMI_K2_HIDDEN / KIMI_K2_INT4_GROUP_SIZE)
    );
    assert!(
        !weights
            .tensors
            .keys()
            .any(|name| name.contains(".mlp.experts."))
    );
}

#[test]
#[ignore = "H20-only: compares real Kimi-K2.5 rank0 layer1 Marlin routed expert against vLLM"]
fn h20_kimi_k25_rank0_layer1_marlin_wna16_matches_vllm_reference() {
    use std::path::PathBuf;

    const TOKENS: usize = 4;
    const BLOCK_SIZE: usize = 8;
    const LAYER_IDX: usize = 1;

    let model_path = Path::new("/data/models/Kimi-K2.5");
    assert!(
        model_path.join(KIMI_K2_WEIGHT_INDEX).exists(),
        "missing H20 Kimi-K2.5 weights at {}",
        model_path.display()
    );
    let reference_dir = std::env::var("PEGAINFER_KIMI_K25_MARLIN_LAYER_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/data/fixtures/kimi-k2/k25_rank0_layer1_marlin_vllm"));
    assert!(
        reference_dir.join("metadata.json").exists(),
        "missing vLLM real layer Marlin reference at {}",
        reference_dir.display()
    );

    let manifest = KimiK2WeightManifest::from_model_dir(model_path).unwrap();
    let names = manifest.rank_weight_names(0).unwrap();
    let load_plan = manifest.rank_sliced_load_plan(0).unwrap();
    let ctx = KimiRankGpuContext::new(0).unwrap();
    let weights = load_rank_sliced_weights_to_gpu(&ctx, model_path, &load_plan).unwrap();
    let typed = weights.typed_view(&names).unwrap();
    let layer1 = typed
        .pack_expert_major_layer_marlin_weights(&ctx, LAYER_IDX)
        .unwrap();
    ctx.sync().unwrap();
    layer1.as_marlin_weights().validate().unwrap();
    assert_eq!(layer1.first_global_expert, 0);
    assert_eq!(layer1.local_experts, 48);

    let device_ctx = ctx.as_device_context();
    let route_elems = TOKENS * KIMI_K2_TOPK;
    let topk_host = (0..route_elems)
        .map(|idx| {
            let token = idx / KIMI_K2_TOPK;
            let route = idx % KIMI_K2_TOPK;
            i32::try_from((token * 13 + route * 5) % 48).unwrap()
        })
        .collect::<Vec<_>>();
    let denom = (KIMI_K2_TOPK * (KIMI_K2_TOPK + 1) / 2) as f32;
    let topk_weight_host = (0..route_elems)
        .map(|idx| ((idx % KIMI_K2_TOPK) + 1) as f32 / denom)
        .collect::<Vec<_>>();
    let hidden_host = deterministic_bf16(TOKENS * KIMI_K2_HIDDEN, 23, 1.0 / 32.0, -11.0);

    let topk_dev = device_ctx.stream.clone_htod(&topk_host).unwrap();
    let topk_weight_dev = device_ctx.stream.clone_htod(&topk_weight_host).unwrap();
    let hidden_data = device_ctx.stream.clone_htod(&hidden_host).unwrap();
    let mut route_workspace =
        KimiMarlinRouteWorkspace::new(&device_ctx, TOKENS, BLOCK_SIZE).unwrap();
    let routing = kimi_moe_marlin_align_block_size(
        &device_ctx,
        &mut route_workspace,
        &topk_dev,
        TOKENS,
        TOKENS,
        0,
    )
    .unwrap();
    let mut gemm_workspace = KimiMarlinWna16Workspace::new(
        &device_ctx,
        routing.max_m_blocks,
        KIMI_K2_HIDDEN,
        BLOCK_SIZE,
    )
    .unwrap();
    let hidden = HiddenStates {
        data: hidden_data,
        hidden_dim: KIMI_K2_HIDDEN,
        seq_len: TOKENS,
    };
    let layer1_weights = layer1.as_marlin_weights();

    let mut w13_out =
        HiddenStates::zeros(&device_ctx, 2 * KIMI_K2_EXPERT_INTERMEDIATE, route_elems).unwrap();
    kimi_marlin_wna16_w13_gemm(
        &device_ctx,
        &mut gemm_workspace,
        &routing,
        &hidden,
        &layer1_weights.w13,
        &topk_weight_dev,
        &mut w13_out,
    )
    .unwrap();
    let w13_got = device_ctx.stream.clone_dtoh(&w13_out.data).unwrap();
    let w13_ref = read_bf16_file(
        &reference_dir.join("w13_out_bf16.bin"),
        route_elems * 2 * KIMI_K2_EXPERT_INTERMEDIATE,
    );
    assert_bf16_close("k25_layer1_w13_out", &w13_got, &w13_ref, 0.5, 0.03);

    let mut activated =
        HiddenStates::zeros(&device_ctx, KIMI_K2_EXPERT_INTERMEDIATE, route_elems).unwrap();
    kimi_marlin_w13_swiglu(&device_ctx, &w13_out, &mut activated).unwrap();
    let mut route_output = HiddenStates::zeros(&device_ctx, KIMI_K2_HIDDEN, route_elems).unwrap();
    kimi_marlin_wna16_w2_gemm(
        &device_ctx,
        &mut gemm_workspace,
        &routing,
        &activated,
        &layer1_weights.w2_down,
        &topk_weight_dev,
        &mut route_output,
    )
    .unwrap();
    let route_got = device_ctx.stream.clone_dtoh(&route_output.data).unwrap();
    let route_ref = read_bf16_file(
        &reference_dir.join("route_output_bf16.bin"),
        route_elems * KIMI_K2_HIDDEN,
    );
    assert_bf16_close(
        "k25_layer1_route_output",
        &route_got,
        &route_ref,
        16.0,
        0.03,
    );

    let mut final_out = device_ctx
        .stream
        .alloc_zeros::<f32>(TOKENS * KIMI_K2_HIDDEN)
        .unwrap();
    kimi_marlin_sum_topk_rows_f32(&device_ctx, &route_output, TOKENS, &mut final_out).unwrap();
    let final_got_f32 = device_ctx.stream.clone_dtoh(&final_out).unwrap();
    let final_got = final_got_f32
        .iter()
        .map(|value| bf16::from_f32(*value))
        .collect::<Vec<_>>();
    let final_ref = read_bf16_file(
        &reference_dir.join("final_bf16.bin"),
        TOKENS * KIMI_K2_HIDDEN,
    );
    assert_bf16_close("k25_layer1_final", &final_got, &final_ref, 128.0, 0.25);
}

#[test]
fn rank_tensor_names_filter_local_experts() {
    let manifest = tiny_manifest();
    let rank0 = manifest.rank_tensor_names(0).unwrap();
    assert!(rank0.iter().any(|entry| entry.name.contains("experts.0.")));
    assert!(rank0.iter().any(|entry| entry.name.contains("experts.47.")));
    assert!(!rank0.iter().any(|entry| entry.name.contains("experts.48.")));
    let rank1 = manifest.rank_tensor_names(1).unwrap();
    assert!(rank1.iter().any(|entry| entry.name.contains("experts.48.")));
    assert!(!rank1.iter().any(|entry| entry.name.contains("experts.47.")));
}

#[test]
fn rank_weight_names_are_local_and_typed() {
    let manifest = tiny_manifest();
    let names = manifest.rank_weight_names(1).unwrap();
    assert_eq!(names.rank, 1);
    assert_eq!(names.plan.local_expert_range, 48..96);
    assert_eq!(
        names.top.token_embedding,
        "language_model.model.embed_tokens.weight"
    );
    assert_eq!(names.layers.len(), KIMI_K2_LAYERS);
    match &names.layers[0].kind {
        KimiLayerWeightKindNames::Dense(mlp) => {
            assert_eq!(
                mlp.gate_proj,
                "language_model.model.layers.0.mlp.gate_proj.weight"
            );
        }
        KimiLayerWeightKindNames::Moe(_) => panic!("layer0 must be dense"),
    }
    match &names.layers[1].kind {
        KimiLayerWeightKindNames::Moe(moe) => {
            assert_eq!(moe.routed_experts.len(), 48);
            assert_eq!(moe.routed_experts[0].global_expert, 48);
            assert_eq!(moe.routed_experts[47].global_expert, 95);
        }
        KimiLayerWeightKindNames::Dense(_) => panic!("layer1 must be MoE"),
    }
}

#[test]
fn rank_sliced_load_plan_applies_tp8_ep8_slices() {
    let manifest = tiny_manifest();
    let load_plan = manifest.rank_sliced_load_plan(3).unwrap();
    assert_eq!(load_plan.rank, 3);
    assert_eq!(load_plan.tensor_count, 26_775);

    assert_eq!(
        find_load_spec(&load_plan, "language_model.model.embed_tokens.weight").slice,
        KimiTensorLoadSlice::RowRange {
            start: 61_440,
            end: 81_920
        }
    );
    assert_eq!(
        find_load_spec(&load_plan, "language_model.lm_head.weight").slice,
        KimiTensorLoadSlice::RowRange {
            start: 61_440,
            end: 81_920
        }
    );
    assert_eq!(
        find_load_spec(&load_plan, "language_model.model.norm.weight").slice,
        KimiTensorLoadSlice::Full
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.self_attn.q_b_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::RowRange {
            start: 4_608,
            end: 6_144
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.self_attn.kv_b_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::RowRange {
            start: 6_144,
            end: 8_192
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.self_attn.o_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::ColRange {
            start: 3_072,
            end: 4_096
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.mlp.gate_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::RowRange {
            start: 6_912,
            end: 9_216
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.1.mlp.shared_experts.down_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::ColRange {
            start: 768,
            end: 1_024
        }
    );

    assert!(
        find_load_spec_opt(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.143.gate_proj.weight_packed"
        )
        .is_none()
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.144.gate_proj.weight_packed"
        )
        .slice,
        KimiTensorLoadSlice::Full
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.191.down_proj.weight_shape"
        )
        .slice,
        KimiTensorLoadSlice::Full
    );
    assert!(
        find_load_spec_opt(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.192.gate_proj.weight_packed"
        )
        .is_none()
    );
}

#[test]
fn rank_weight_headers_validate_typed_gpu_view_contract() {
    let manifest = tiny_manifest();
    let names = manifest.rank_weight_names(1).unwrap();
    let headers = headers_for_names(&names);
    headers.validate_typed_names(&names).unwrap();
    assert_eq!(names.required_tensor_names().unwrap().len(), 26_775);
}

#[test]
fn rank_weight_headers_reject_wrong_typed_dtype() {
    let manifest = tiny_manifest();
    let names = manifest.rank_weight_names(1).unwrap();
    let mut headers = headers_for_names(&names);
    let bias_name = match &names.layers[1].kind {
        KimiLayerWeightKindNames::Moe(moe) => &moe.router.e_score_correction_bias,
        KimiLayerWeightKindNames::Dense(_) => panic!("layer1 must be MoE"),
    };
    headers.tensors.get_mut(bias_name).unwrap().dtype = Dtype::BF16;
    let err = headers.validate_typed_names(&names).unwrap_err();
    assert!(err.to_string().contains("expected F32"));
}

#[test]
fn load_rank_weight_headers_reads_planned_shards() {
    let dir = make_temp_dir();
    let shard0 = dir.join("model-00001-of-000002.safetensors");
    let shard1 = dir.join("model-00002-of-000002.safetensors");
    write_safetensor(
        &shard0,
        &[
            ("a.weight", Dtype::BF16, vec![2], vec![1, 2, 3, 4]),
            ("b.weight", Dtype::F32, vec![1], vec![5, 6, 7, 8]),
        ],
    );
    write_safetensor(
        &shard1,
        &[("c.weight", Dtype::U8, vec![3], vec![9, 10, 11])],
    );
    let shard_plan = KimiRankShardPlan {
        rank: 3,
        shards: vec![
            KimiShardTensorPlan {
                shard: "model-00001-of-000002.safetensors".to_owned(),
                tensors: vec!["a.weight".to_owned(), "b.weight".to_owned()],
            },
            KimiShardTensorPlan {
                shard: "model-00002-of-000002.safetensors".to_owned(),
                tensors: vec!["c.weight".to_owned()],
            },
        ],
        tensor_count: 3,
    };
    let headers = load_rank_weight_headers(&dir, &shard_plan).unwrap();
    assert_eq!(headers.rank, 3);
    assert_eq!(headers.total_bytes, 11);
    assert_eq!(headers.tensors["a.weight"].dtype, Dtype::BF16);
    assert_eq!(headers.tensors["a.weight"].shape, vec![2]);
    assert_eq!(
        headers.tensors["c.weight"].shard,
        "model-00002-of-000002.safetensors"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn load_rank_sliced_weight_headers_reports_local_shapes_and_bytes() {
    let dir = make_temp_dir();
    write_safetensor(
        &dir.join("model-00001-of-000001.safetensors"),
        &[
            ("row.weight", Dtype::BF16, vec![6, 4], (0..48).collect()),
            ("col.weight", Dtype::BF16, vec![3, 6], (0..36).collect()),
            ("full.weight", Dtype::U8, vec![5], (0..5).collect()),
        ],
    );
    let load_plan = KimiRankSlicedLoadPlan {
        rank: 2,
        shards: vec![KimiShardTensorLoadPlan {
            shard: "model-00001-of-000001.safetensors".to_owned(),
            tensors: vec![
                KimiTensorLoadSpec {
                    name: "row.weight".to_owned(),
                    shard: "model-00001-of-000001.safetensors".to_owned(),
                    slice: KimiTensorLoadSlice::RowRange { start: 2, end: 5 },
                },
                KimiTensorLoadSpec {
                    name: "col.weight".to_owned(),
                    shard: "model-00001-of-000001.safetensors".to_owned(),
                    slice: KimiTensorLoadSlice::ColRange { start: 1, end: 5 },
                },
                KimiTensorLoadSpec {
                    name: "full.weight".to_owned(),
                    shard: "model-00001-of-000001.safetensors".to_owned(),
                    slice: KimiTensorLoadSlice::Full,
                },
            ],
        }],
        tensor_count: 3,
    };
    let headers = load_rank_sliced_weight_headers(&dir, &load_plan).unwrap();
    assert_eq!(headers.rank, 2);
    assert_eq!(headers.tensors["row.weight"].shape, vec![3, 4]);
    assert_eq!(headers.tensors["row.weight"].bytes, 24);
    assert_eq!(headers.tensors["col.weight"].shape, vec![3, 4]);
    assert_eq!(headers.tensors["col.weight"].bytes, 24);
    assert_eq!(headers.tensors["full.weight"].shape, vec![5]);
    assert_eq!(headers.total_bytes, 53);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sliced_tensor_bytes_packs_col_slice_as_row_major() {
    let data = (0u8..24).collect::<Vec<_>>();
    let out = sliced_tensor_bytes(
        &data,
        &[3, 4],
        Dtype::BF16,
        &KimiTensorLoadSlice::ColRange { start: 1, end: 3 },
    )
    .unwrap();
    assert_eq!(out, vec![2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21]);
}

#[test]
fn load_rank_weight_headers_rejects_missing_tensor() {
    let dir = make_temp_dir();
    write_safetensor(
        &dir.join("model-00001-of-000001.safetensors"),
        &[("present", Dtype::U8, vec![1], vec![1])],
    );
    let shard_plan = KimiRankShardPlan {
        rank: 0,
        shards: vec![KimiShardTensorPlan {
            shard: "model-00001-of-000001.safetensors".to_owned(),
            tensors: vec!["missing".to_owned()],
        }],
        tensor_count: 1,
    };
    let err = load_rank_weight_headers(&dir, &shard_plan).unwrap_err();
    assert!(err.to_string().contains("missing tensor missing"));
    fs::remove_dir_all(dir).unwrap();
}

fn find_load_spec<'a>(plan: &'a KimiRankSlicedLoadPlan, name: &str) -> &'a KimiTensorLoadSpec {
    find_load_spec_opt(plan, name).unwrap_or_else(|| panic!("missing load spec {name}"))
}

fn find_load_spec_opt<'a>(
    plan: &'a KimiRankSlicedLoadPlan,
    name: &str,
) -> Option<&'a KimiTensorLoadSpec> {
    plan.shards
        .iter()
        .flat_map(|shard| shard.tensors.iter())
        .find(|spec| spec.name == name)
}

fn tiny_manifest() -> KimiK2WeightManifest {
    let mut layers = Vec::new();
    for layer_idx in 0..KIMI_K2_LAYERS {
        let attention = KimiAttentionManifest {
            input_layernorm: fake(layer_idx, "input_layernorm.weight"),
            q_a_proj: fake(layer_idx, "self_attn.q_a_proj.weight"),
            q_a_layernorm: fake(layer_idx, "self_attn.q_a_layernorm.weight"),
            q_b_proj: fake(layer_idx, "self_attn.q_b_proj.weight"),
            kv_a_proj_with_mqa: fake(layer_idx, "self_attn.kv_a_proj_with_mqa.weight"),
            kv_a_layernorm: fake(layer_idx, "self_attn.kv_a_layernorm.weight"),
            kv_b_proj: fake(layer_idx, "self_attn.kv_b_proj.weight"),
            o_proj: fake(layer_idx, "self_attn.o_proj.weight"),
            post_attention_layernorm: fake(layer_idx, "post_attention_layernorm.weight"),
        };
        let kind = if layer_idx == 0 {
            KimiLayerKindManifest::Dense(KimiDenseMlpManifest {
                gate_proj: fake(layer_idx, "mlp.gate_proj.weight"),
                up_proj: fake(layer_idx, "mlp.up_proj.weight"),
                down_proj: fake(layer_idx, "mlp.down_proj.weight"),
            })
        } else {
            KimiLayerKindManifest::Moe(KimiMoeLayerManifest {
                router: KimiRouterManifest {
                    gate_weight: fake(layer_idx, "mlp.gate.weight"),
                    e_score_correction_bias: fake(layer_idx, "mlp.gate.e_score_correction_bias"),
                },
                shared_experts: KimiSharedExpertManifest {
                    gate_proj: fake(layer_idx, "mlp.shared_experts.gate_proj.weight"),
                    up_proj: fake(layer_idx, "mlp.shared_experts.up_proj.weight"),
                    down_proj: fake(layer_idx, "mlp.shared_experts.down_proj.weight"),
                },
                routed_experts: (0..KIMI_K2_ROUTED_EXPERTS)
                    .map(|expert_idx| KimiRoutedExpertManifest {
                        expert_idx,
                        gate_proj: fake_int4(layer_idx, expert_idx, "gate_proj"),
                        up_proj: fake_int4(layer_idx, expert_idx, "up_proj"),
                        down_proj: fake_int4(layer_idx, expert_idx, "down_proj"),
                    })
                    .collect(),
            })
        };
        layers.push(KimiLayerManifest {
            layer_idx,
            attention,
            kind,
        });
    }
    KimiK2WeightManifest {
        total_size: Some(1),
        text_tensor_count: 208_215,
        ignored_non_text_tensor_count: 0,
        shard_count: 64,
        token_embedding: top("language_model.model.embed_tokens.weight"),
        final_norm: top("language_model.model.norm.weight"),
        lm_head: top("language_model.lm_head.weight"),
        layers,
        parallel: KimiK2ParallelShape::tp8_ep8(),
    }
}

fn fake(layer_idx: usize, suffix: &str) -> KimiTensorEntry {
    top(&format!("language_model.model.layers.{layer_idx}.{suffix}"))
}

fn fake_int4(layer_idx: usize, expert_idx: usize, projection: &str) -> KimiInt4ProjectionManifest {
    let prefix =
        format!("language_model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}");
    KimiInt4ProjectionManifest {
        weight_packed: top(&format!("{prefix}.weight_packed")),
        weight_scale: top(&format!("{prefix}.weight_scale")),
        weight_shape: top(&format!("{prefix}.weight_shape")),
    }
}

fn top(name: &str) -> KimiTensorEntry {
    KimiTensorEntry {
        name: name.to_owned(),
        shard: "model-00001-of-000064.safetensors".to_owned(),
    }
}

fn headers_for_names(names: &KimiRankWeightNames) -> KimiRankWeightHeaders {
    let mut tensors = BTreeMap::new();
    insert_header(&mut tensors, &names.top.token_embedding, Dtype::BF16);
    insert_header(&mut tensors, &names.top.final_norm, Dtype::BF16);
    insert_header(&mut tensors, &names.top.lm_head, Dtype::BF16);
    for layer in &names.layers {
        insert_attention_headers(&mut tensors, &layer.attention);
        match &layer.kind {
            KimiLayerWeightKindNames::Dense(mlp) => {
                insert_header(&mut tensors, &mlp.gate_proj, Dtype::BF16);
                insert_header(&mut tensors, &mlp.up_proj, Dtype::BF16);
                insert_header(&mut tensors, &mlp.down_proj, Dtype::BF16);
            }
            KimiLayerWeightKindNames::Moe(moe) => {
                insert_header(&mut tensors, &moe.router.gate_weight, Dtype::BF16);
                insert_header(
                    &mut tensors,
                    &moe.router.e_score_correction_bias,
                    Dtype::F32,
                );
                insert_header(&mut tensors, &moe.shared_experts.gate_proj, Dtype::BF16);
                insert_header(&mut tensors, &moe.shared_experts.up_proj, Dtype::BF16);
                insert_header(&mut tensors, &moe.shared_experts.down_proj, Dtype::BF16);
                for expert in &moe.routed_experts {
                    insert_int4_projection_headers(&mut tensors, &expert.gate_proj);
                    insert_int4_projection_headers(&mut tensors, &expert.up_proj);
                    insert_int4_projection_headers(&mut tensors, &expert.down_proj);
                }
            }
        }
    }
    KimiRankWeightHeaders {
        rank: names.rank,
        total_bytes: tensors.len(),
        tensors,
    }
}

fn insert_attention_headers(
    tensors: &mut BTreeMap<String, KimiTensorHeader>,
    attention: &KimiAttentionWeightNames,
) {
    insert_header(tensors, &attention.input_layernorm, Dtype::BF16);
    insert_header(tensors, &attention.q_a_proj, Dtype::BF16);
    insert_header(tensors, &attention.q_a_layernorm, Dtype::BF16);
    insert_header(tensors, &attention.q_b_proj, Dtype::BF16);
    insert_header(tensors, &attention.kv_a_proj_with_mqa, Dtype::BF16);
    insert_header(tensors, &attention.kv_a_layernorm, Dtype::BF16);
    insert_header(tensors, &attention.kv_b_proj, Dtype::BF16);
    insert_header(tensors, &attention.o_proj, Dtype::BF16);
    insert_header(tensors, &attention.post_attention_layernorm, Dtype::BF16);
}

fn insert_int4_projection_headers(
    tensors: &mut BTreeMap<String, KimiTensorHeader>,
    projection: &KimiInt4ProjectionWeightNames,
) {
    insert_header(tensors, &projection.weight_packed, Dtype::I32);
    insert_header(tensors, &projection.weight_scale, Dtype::BF16);
    insert_header(tensors, &projection.weight_shape, Dtype::I32);
}

fn insert_header(tensors: &mut BTreeMap<String, KimiTensorHeader>, name: &str, dtype: Dtype) {
    let previous = tensors.insert(
        name.to_owned(),
        KimiTensorHeader {
            name: name.to_owned(),
            shard: "model-00001-of-000064.safetensors".to_owned(),
            dtype,
            shape: vec![1],
            bytes: 1,
        },
    );
    assert!(previous.is_none(), "duplicate test tensor {name}");
}

fn make_temp_dir() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "pegainfer-kimi-k2-weights-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_safetensor(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
    let views = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                *name,
                TensorView::new(*dtype, shape.clone(), data.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let bytes = serialize(views, None).unwrap();
    fs::write(path, bytes).unwrap();
}

fn deterministic_bf16(len: usize, modulo: usize, scale: f32, offset: f32) -> Vec<bf16> {
    (0..len)
        .map(|idx| bf16::from_f32(((idx % modulo) as f32 + offset) * scale))
        .collect()
}

fn read_bf16_file(path: &Path, expected: usize) -> Vec<bf16> {
    let bytes =
        fs::read(path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    assert_eq!(
        bytes.len(),
        expected * std::mem::size_of::<u16>(),
        "{} len mismatch",
        path.display()
    );
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn assert_bf16_close(name: &str, got: &[bf16], expected: &[bf16], max_limit: f32, mean_limit: f32) {
    assert_eq!(got.len(), expected.len(), "{name} len mismatch");
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut max_idx = 0usize;
    let mut max_got = 0.0f32;
    let mut max_expected = 0.0f32;
    for (idx, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        let diff = (actual - expected).abs();
        sum_diff += diff;
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
            max_got = actual;
            max_expected = expected;
        }
    }
    let mean_diff = sum_diff / got.len() as f32;
    println!(
        "{name}: max_diff={max_diff} mean_diff={mean_diff} max_idx={max_idx} got={max_got} expected={max_expected}"
    );
    assert!(
        max_diff <= max_limit && mean_diff <= mean_limit,
        "{name} diff too large: max_diff={max_diff} mean_diff={mean_diff} limits=({max_limit}, {mean_limit}) max_idx={max_idx} got={max_got} expected={max_expected}"
    );
}

#[test]
fn scanner_rejects_missing_required_text_tensor() {
    let json = json!({
        "metadata": {"total_size": 1},
        "weight_map": {
            "language_model.model.embed_tokens.weight": "model-00001-of-000064.safetensors"
        }
    });
    let err = KimiK2WeightManifest::from_index_json(&json).unwrap_err();
    assert!(err.to_string().contains("language_model.model.norm.weight"));
}
