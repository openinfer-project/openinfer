use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorInfo};
use serde_json::Value;

const GLM52_MODEL_PATH: &str = "/data/models/GLM-5.2-FP8";
const HIDDEN_SIZE: usize = 6144;
const CONFIG_HEADS: usize = 64;
const Q_LORA_RANK: usize = 2048;
const KV_LORA_RANK: usize = 512;
const QK_NOPE_HEAD_DIM: usize = 192;
const QK_ROPE_HEAD_DIM: usize = 64;
const V_HEAD_DIM: usize = 256;
const FP8_BLOCK_SIZE: usize = 128;

#[test]
#[ignore]
fn jiuzhang_checkpoint_mla_projection_layout_metadata() {
    let model_path = Path::new(GLM52_MODEL_PATH);
    assert!(
        model_path.join("model.safetensors.index.json").exists(),
        "GLM5.2 checkpoint missing at {}",
        model_path.display()
    );

    let config = read_json(&model_path.join("config.json"));
    assert_eq!(config["model_type"].as_str(), Some("glm_moe_dsa"));
    assert_eq!(usize_field(&config, "num_attention_heads"), CONFIG_HEADS);
    assert_eq!(usize_field(&config, "num_key_value_heads"), CONFIG_HEADS);
    assert_eq!(usize_field(&config, "q_lora_rank"), Q_LORA_RANK);
    assert_eq!(usize_field(&config, "kv_lora_rank"), KV_LORA_RANK);
    assert_eq!(usize_field(&config, "qk_nope_head_dim"), QK_NOPE_HEAD_DIM);
    assert_eq!(usize_field(&config, "qk_rope_head_dim"), QK_ROPE_HEAD_DIM);
    assert_eq!(usize_field(&config, "v_head_dim"), V_HEAD_DIM);

    let weight_map = read_weight_map(model_path);
    let q_b = tensor_info(
        model_path,
        &weight_map,
        "model.layers.0.self_attn.q_b_proj.weight",
    );
    let q_b_scale = tensor_info(
        model_path,
        &weight_map,
        "model.layers.0.self_attn.q_b_proj.weight_scale_inv",
    );
    let kv_b = tensor_info(
        model_path,
        &weight_map,
        "model.layers.0.self_attn.kv_b_proj.weight",
    );
    let kv_b_scale = tensor_info(
        model_path,
        &weight_map,
        "model.layers.0.self_attn.kv_b_proj.weight_scale_inv",
    );

    // Standard 64-head DeepSeek-DSA MLA -- no packing/fold. Every projection is
    // exactly `heads * per_head_dim`. (The earlier "factor-4" was a corrupt
    // vendor repack; the official zai-org/GLM-5.2-FP8 is plain 64-head.)
    let qk_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
    let q_b_out = CONFIG_HEADS * qk_head_dim;
    let kv_b_out = CONFIG_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM);
    let o_proj_in = CONFIG_HEADS * V_HEAD_DIM;
    assert_eq!(q_b_out, 16_384);
    assert_eq!(kv_b_out, 28_672);
    assert_eq!(o_proj_in, 16_384);

    assert_tensor(&q_b, Dtype::F8_E4M3, &[q_b_out, Q_LORA_RANK]);
    assert_tensor(
        &q_b_scale,
        Dtype::F32,
        &[q_b_out / FP8_BLOCK_SIZE, Q_LORA_RANK / FP8_BLOCK_SIZE],
    );
    assert_tensor(&kv_b, Dtype::F8_E4M3, &[kv_b_out, KV_LORA_RANK]);
    assert_tensor(
        &kv_b_scale,
        Dtype::F32,
        &[kv_b_out / FP8_BLOCK_SIZE, KV_LORA_RANK / FP8_BLOCK_SIZE],
    );

    let o_proj = tensor_info(
        model_path,
        &weight_map,
        "model.layers.0.self_attn.o_proj.weight",
    );
    assert_tensor(&o_proj, Dtype::F8_E4M3, &[HIDDEN_SIZE, o_proj_in]);

    // FlashMLA sparse FP8 decode contract (DeepSeek-V3.2 cache): absorbed query
    // width 576 and 656-byte cache token (512 fp8 ckv + 16 f32 scale + 64 bf16 k_pe).
    let flashmla_query_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
    let flashmla_fp8_cache_token_bytes =
        KV_LORA_RANK + (KV_LORA_RANK / FP8_BLOCK_SIZE) * 4 + QK_ROPE_HEAD_DIM * 2;
    assert_eq!(flashmla_query_dim, 576);
    assert_eq!(flashmla_fp8_cache_token_bytes, 656);
}

fn read_json(path: &Path) -> Value {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&content).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

fn read_weight_map(model_path: &Path) -> HashMap<String, String> {
    let index = read_json(&model_path.join("model.safetensors.index.json"));
    index["weight_map"]
        .as_object()
        .expect("model.safetensors.index.json has no weight_map")
        .iter()
        .map(|(name, shard)| {
            (
                name.clone(),
                shard
                    .as_str()
                    .unwrap_or_else(|| panic!("weight_map[{name}] is not a string"))
                    .to_string(),
            )
        })
        .collect()
}

fn tensor_info(
    model_path: &Path,
    weight_map: &HashMap<String, String>,
    tensor_name: &str,
) -> TensorInfo {
    let shard = weight_map
        .get(tensor_name)
        .unwrap_or_else(|| panic!("checkpoint index missing {tensor_name}"));
    let shard_path: PathBuf = model_path.join(shard);
    let file = File::open(&shard_path)
        .unwrap_or_else(|err| panic!("open {}: {err}", shard_path.display()));
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .unwrap_or_else(|err| panic!("mmap {}: {err}", shard_path.display()))
    };
    let (_, metadata) = SafeTensors::read_metadata(&mmap)
        .unwrap_or_else(|err| panic!("read metadata {}: {err}", shard_path.display()));
    metadata
        .info(tensor_name)
        .unwrap_or_else(|| panic!("{} does not contain {tensor_name}", shard_path.display()))
        .clone()
}

fn assert_tensor(info: &TensorInfo, dtype: Dtype, shape: &[usize]) {
    assert_eq!(info.dtype, dtype);
    assert_eq!(info.shape, shape);
}

fn usize_field(config: &Value, key: &str) -> usize {
    config[key]
        .as_u64()
        .unwrap_or_else(|| panic!("config.{key} is not an unsigned integer")) as usize
}
