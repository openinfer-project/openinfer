use super::*;
use safetensors::tensor::{TensorView, serialize};
use serde_json::json;
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

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
