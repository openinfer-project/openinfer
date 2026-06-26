use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    ops::Range,
    path::Path,
};

use anyhow::Context;
use anyhow::{Result, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::config::{
    GLM52_DENSE_INTERMEDIATE, GLM52_EXPERT_INTERMEDIATE, GLM52_HIDDEN, GLM52_INDEX_HEAD_DIM,
    GLM52_INDEX_HEADS, GLM52_KV_A_OUT, GLM52_KV_B_OUT, GLM52_KV_LORA_RANK, GLM52_O_PROJ_IN,
    GLM52_Q_B_OUT, GLM52_Q_LORA_RANK, GLM52_VOCAB,
};
use crate::config::{GLM52_DENSE_LAYERS, GLM52_LAYERS, GLM52_MOE_LAYERS, GLM52_ROUTED_EXPERTS};
use crate::pp::{Glm52StagePlan, glm52_pp_stage_plans};

const GLM52_WEIGHT_INDEX: &str = "model.safetensors.index.json";
const NEXTN_LAYER_PREFIX: &str = "model.layers.78.";
const FULL_INDEXER_LAYER_COUNT: usize = 21;
const FP8_BLOCK_SIZE: usize = 128;

mod context;
mod load;
mod package;
mod view;

pub(crate) use context::Glm52RankGpuContext;
pub(crate) use load::{Glm52StageGpuWeights, load_stage_sliced_weights_to_gpu};
pub(crate) use package::Glm52StageExpertFp8Weights;
pub(crate) use view::Glm52NonExpertWeightContractReport;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52TensorEntry {
    pub(crate) name: String,
    pub(crate) shard: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52Fp8ProjectionManifest {
    pub(crate) weight: Glm52TensorEntry,
    pub(crate) weight_scale_inv: Glm52TensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52IndexerManifest {
    pub(crate) k_norm_weight: Glm52TensorEntry,
    pub(crate) k_norm_bias: Glm52TensorEntry,
    pub(crate) weights_proj: Glm52TensorEntry,
    pub(crate) wk: Glm52Fp8ProjectionManifest,
    pub(crate) wq_b: Glm52Fp8ProjectionManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52AttentionManifest {
    pub(crate) input_layernorm: Glm52TensorEntry,
    pub(crate) q_a_proj: Glm52Fp8ProjectionManifest,
    pub(crate) q_a_layernorm: Glm52TensorEntry,
    pub(crate) q_b_proj: Glm52Fp8ProjectionManifest,
    pub(crate) kv_a_proj_with_mqa: Glm52Fp8ProjectionManifest,
    pub(crate) kv_a_layernorm: Glm52TensorEntry,
    pub(crate) kv_b_proj: Glm52Fp8ProjectionManifest,
    pub(crate) o_proj: Glm52Fp8ProjectionManifest,
    pub(crate) post_attention_layernorm: Glm52TensorEntry,
    pub(crate) indexer: Option<Glm52IndexerManifest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DenseMlpManifest {
    pub(crate) gate_proj: Glm52Fp8ProjectionManifest,
    pub(crate) up_proj: Glm52Fp8ProjectionManifest,
    pub(crate) down_proj: Glm52Fp8ProjectionManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RouterManifest {
    pub(crate) gate_weight: Glm52TensorEntry,
    pub(crate) e_score_correction_bias: Glm52TensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52SharedExpertManifest {
    pub(crate) gate_proj: Glm52Fp8ProjectionManifest,
    pub(crate) up_proj: Glm52Fp8ProjectionManifest,
    pub(crate) down_proj: Glm52Fp8ProjectionManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RoutedExpertManifest {
    pub(crate) expert_idx: usize,
    pub(crate) gate_proj: Glm52Fp8ProjectionManifest,
    pub(crate) up_proj: Glm52Fp8ProjectionManifest,
    pub(crate) down_proj: Glm52Fp8ProjectionManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeLayerManifest {
    pub(crate) router: Glm52RouterManifest,
    pub(crate) shared_experts: Glm52SharedExpertManifest,
    pub(crate) routed_experts: Vec<Glm52RoutedExpertManifest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Glm52LayerKindManifest {
    Dense(Glm52DenseMlpManifest),
    Moe(Glm52MoeLayerManifest),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52LayerManifest {
    pub(crate) layer_idx: usize,
    pub(crate) attention: Glm52AttentionManifest,
    pub(crate) kind: Glm52LayerKindManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52WeightManifest {
    pub(crate) total_tensor_count: usize,
    pub(crate) runtime_tensor_count: usize,
    pub(crate) nextn_tensor_count: usize,
    pub(crate) token_embedding: Glm52TensorEntry,
    pub(crate) final_norm: Glm52TensorEntry,
    pub(crate) lm_head: Glm52TensorEntry,
    pub(crate) layers: Vec<Glm52LayerManifest>,
}

/// One pipeline stage's weight residency plan: the contiguous layer range it
/// owns, whether it carries the embedding / final-norm+lm_head bookends, and the
/// routed-expert range — always all [`GLM52_ROUTED_EXPERTS`] under EP1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StageWeightPlan {
    pub(crate) stage: usize,
    pub(crate) layers: Range<usize>,
    pub(crate) owns_embed: bool,
    pub(crate) owns_head: bool,
    pub(crate) expert_range: Range<usize>,
    pub(crate) tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Glm52TensorLoadSlice {
    Full,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52TensorLoadSpec {
    pub(crate) name: String,
    pub(crate) shard: String,
    pub(crate) slice: Glm52TensorLoadSlice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52ShardTensorLoadPlan {
    pub(crate) shard: String,
    pub(crate) tensors: Vec<Glm52TensorLoadSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StageSlicedLoadPlan {
    pub(crate) stage: usize,
    pub(crate) shards: Vec<Glm52ShardTensorLoadPlan>,
    pub(crate) tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StageLoadBundle {
    pub(crate) plan: Glm52StageWeightPlan,
    pub(crate) names: view::Glm52StageWeightNames,
    pub(crate) load_plan: Glm52StageSlicedLoadPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StageHeaderStats {
    pub(crate) tensor_count: usize,
    pub(crate) total_bytes: usize,
}

impl Glm52WeightManifest {
    pub(crate) fn from_model_dir(model_path: &Path) -> Result<Self> {
        Self::from_index_file(&model_path.join(GLM52_WEIGHT_INDEX))
    }

    fn from_index_file(index_path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let json: Value = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;
        Self::from_index_json(&json)
    }

    pub(crate) fn from_index_json(json: &Value) -> Result<Self> {
        let weight_map = json
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("GLM5.2 safetensors index missing weight_map"))?;
        let mut tensors = BTreeMap::new();
        for (name, shard) in weight_map {
            let shard = shard
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("weight_map entry {name} is not a shard string"))?;
            tensors.insert(name.as_str(), shard);
        }

        let token_embedding = tensor(&tensors, "model.embed_tokens.weight")?;
        let final_norm = tensor(&tensors, "model.norm.weight")?;
        let lm_head = tensor(&tensors, "lm_head.weight")?;
        let mut layers = Vec::with_capacity(GLM52_LAYERS);
        for layer_idx in 0..GLM52_LAYERS {
            let attention = attention_manifest(&tensors, layer_idx)?;
            let kind = if layer_idx < GLM52_DENSE_LAYERS {
                Glm52LayerKindManifest::Dense(dense_mlp_manifest(&tensors, layer_idx)?)
            } else {
                Glm52LayerKindManifest::Moe(moe_layer_manifest(&tensors, layer_idx)?)
            };
            layers.push(Glm52LayerManifest {
                layer_idx,
                attention,
                kind,
            });
        }

        let nextn_tensor_count = weight_map
            .keys()
            .filter(|name| name.starts_with(NEXTN_LAYER_PREFIX))
            .count();
        let mut manifest = Self {
            total_tensor_count: weight_map.len(),
            runtime_tensor_count: 0,
            nextn_tensor_count,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        };
        manifest.runtime_tensor_count = manifest.runtime_tensor_entries()?.len();
        manifest.validate()?;
        Ok(manifest)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.layers.len() == GLM52_LAYERS,
            "GLM5.2 manifest expected {GLM52_LAYERS} runtime layers, got {}",
            self.layers.len()
        );
        let moe_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.kind, Glm52LayerKindManifest::Moe(_)))
            .count();
        ensure!(
            moe_layers == GLM52_MOE_LAYERS,
            "GLM5.2 manifest expected {GLM52_MOE_LAYERS} MoE layers, got {moe_layers}"
        );
        let full_indexer_layers = self
            .layers
            .iter()
            .filter(|layer| layer.attention.indexer.is_some())
            .count();
        ensure!(
            full_indexer_layers == FULL_INDEXER_LAYER_COUNT,
            "GLM5.2 manifest expected {FULL_INDEXER_LAYER_COUNT} full-indexer runtime layers, got {full_indexer_layers}"
        );
        ensure!(
            self.runtime_tensor_count + self.nextn_tensor_count == self.total_tensor_count,
            "GLM5.2 manifest parsed {} runtime tensors + {} nextn tensors, but index contains {} tensors",
            self.runtime_tensor_count,
            self.nextn_tensor_count,
            self.total_tensor_count
        );
        Ok(())
    }

    pub(crate) fn stage_plan(&self, stage: &Glm52StagePlan) -> Result<Glm52StageWeightPlan> {
        ensure!(
            stage.layers.end <= GLM52_LAYERS,
            "GLM5.2 stage {} layer range {:?} exceeds {GLM52_LAYERS} layers",
            stage.stage,
            stage.layers
        );
        let tensor_count = self.stage_tensor_names(stage).len();
        Ok(Glm52StageWeightPlan {
            stage: stage.stage,
            layers: stage.layers.clone(),
            owns_embed: stage.owns_embed,
            owns_head: stage.owns_head,
            expert_range: 0..GLM52_ROUTED_EXPERTS,
            tensor_count,
        })
    }

    pub(crate) fn stage_tensor_names(&self, stage: &Glm52StagePlan) -> Vec<&Glm52TensorEntry> {
        let mut names = Vec::new();
        if stage.owns_embed {
            names.push(&self.token_embedding);
        }
        if stage.owns_head {
            names.push(&self.final_norm);
            names.push(&self.lm_head);
        }
        for layer in &self.layers {
            if !stage.layers.contains(&layer.layer_idx) {
                continue;
            }
            push_attention(&mut names, &layer.attention);
            match &layer.kind {
                Glm52LayerKindManifest::Dense(mlp) => push_dense_mlp(&mut names, mlp),
                Glm52LayerKindManifest::Moe(moe) => {
                    names.push(&moe.router.gate_weight);
                    names.push(&moe.router.e_score_correction_bias);
                    push_shared_expert(&mut names, &moe.shared_experts);
                    // EP1: every stage holds all routed experts for its MoE layers.
                    for expert in &moe.routed_experts {
                        push_routed_expert(&mut names, expert);
                    }
                }
            }
        }
        names
    }

    pub(crate) fn stage_sliced_load_plan(
        &self,
        stage: &Glm52StagePlan,
    ) -> Glm52StageSlicedLoadPlan {
        let mut by_shard: BTreeMap<String, Vec<Glm52TensorLoadSpec>> = BTreeMap::new();
        for entry in self.stage_tensor_names(stage) {
            by_shard
                .entry(entry.shard.clone())
                .or_default()
                .push(Glm52TensorLoadSpec {
                    name: entry.name.clone(),
                    shard: entry.shard.clone(),
                    slice: Glm52TensorLoadSlice::Full,
                });
        }
        let tensor_count = by_shard.values().map(Vec::len).sum();
        let shards = by_shard
            .into_iter()
            .map(|(shard, tensors)| Glm52ShardTensorLoadPlan { shard, tensors })
            .collect();
        Glm52StageSlicedLoadPlan {
            stage: stage.stage,
            shards,
            tensor_count,
        }
    }

    pub(crate) fn stage_load_bundle(&self, stage: &Glm52StagePlan) -> Result<Glm52StageLoadBundle> {
        let plan = self.stage_plan(stage)?;
        let names = self.stage_weight_names(stage)?;
        let load_plan = self.stage_sliced_load_plan(stage);
        ensure!(
            plan.tensor_count == load_plan.tensor_count,
            "GLM5.2 stage {} tensor plan {} disagrees with load plan {}",
            stage.stage,
            plan.tensor_count,
            load_plan.tensor_count
        );
        ensure!(
            plan == names.plan,
            "GLM5.2 stage {} tensor plan disagrees with typed weight names",
            stage.stage
        );
        Ok(Glm52StageLoadBundle {
            plan,
            names,
            load_plan,
        })
    }

    pub(crate) fn all_stage_load_bundles(
        &self,
        pp_world: usize,
    ) -> Result<Vec<Glm52StageLoadBundle>> {
        let stage_plans = glm52_pp_stage_plans(pp_world);
        // Crash-early: the stages must contiguously partition every layer exactly
        // once (debug_assert inside the partitioner is a no-op in release).
        let mut next = 0usize;
        for stage in &stage_plans {
            ensure!(
                stage.layers.start == next,
                "GLM5.2 PP{pp_world} stage {} layer range {:?} is not contiguous after layer {next}",
                stage.stage,
                stage.layers
            );
            next = stage.layers.end;
        }
        ensure!(
            next == GLM52_LAYERS,
            "GLM5.2 PP{pp_world} stage partition covers {next} layers, expected {GLM52_LAYERS}"
        );
        // Bookend stages carry fewer/more tensors than mid stages, so there is no
        // equal-tensor-count cross-stage invariant to assert here.
        stage_plans
            .iter()
            .map(|stage| self.stage_load_bundle(stage))
            .collect()
    }

    fn runtime_tensor_entries(&self) -> Result<Vec<&Glm52TensorEntry>> {
        let mut entries = Vec::new();
        entries.push(&self.token_embedding);
        entries.push(&self.final_norm);
        entries.push(&self.lm_head);
        for layer in &self.layers {
            push_attention(&mut entries, &layer.attention);
            match &layer.kind {
                Glm52LayerKindManifest::Dense(mlp) => push_dense_mlp(&mut entries, mlp),
                Glm52LayerKindManifest::Moe(moe) => {
                    entries.push(&moe.router.gate_weight);
                    entries.push(&moe.router.e_score_correction_bias);
                    push_shared_expert(&mut entries, &moe.shared_experts);
                    for expert in &moe.routed_experts {
                        push_routed_expert(&mut entries, expert);
                    }
                }
            }
        }
        let unique = entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>();
        ensure!(
            unique.len() == entries.len(),
            "GLM5.2 manifest contains duplicate runtime tensor names"
        );
        Ok(entries)
    }
}

pub(crate) fn validate_stage_safetensor_headers(
    model_path: &Path,
    load_plan: &Glm52StageSlicedLoadPlan,
) -> Result<Glm52StageHeaderStats> {
    let mut tensor_count = 0usize;
    let mut total_bytes = 0usize;
    for shard in &load_plan.shards {
        let path = model_path.join(&shard.shard);
        let mmap = mmap_file(&path)?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .map_err(|err| anyhow::anyhow!("failed to deserialize {}: {err}", path.display()))?;
        for spec in &shard.tensors {
            let view = safetensors.tensor(&spec.name).map_err(|err| {
                anyhow::anyhow!("missing tensor {} in {}: {err}", spec.name, path.display())
            })?;
            ensure!(
                spec.slice == Glm52TensorLoadSlice::Full,
                "GLM5.2 TP1 loader only supports full tensor loads, got {:?} for {}",
                spec.slice,
                spec.name
            );
            let contract = expected_tensor_contract(&spec.name)?;
            ensure!(
                view.dtype() == contract.dtype,
                "GLM5.2 tensor {} dtype mismatch: got {:?}, expected {:?}",
                spec.name,
                view.dtype(),
                contract.dtype
            );
            ensure!(
                view.shape() == contract.shape.as_slice(),
                "GLM5.2 tensor {} shape mismatch: got {:?}, expected {:?}",
                spec.name,
                view.shape(),
                contract.shape
            );
            tensor_count += 1;
            total_bytes += view.data().len();
        }
    }
    ensure!(
        tensor_count == load_plan.tensor_count,
        "GLM5.2 stage {} header validation visited {tensor_count} tensors, expected {}",
        load_plan.stage,
        load_plan.tensor_count
    );
    Ok(Glm52StageHeaderStats {
        tensor_count,
        total_bytes,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Glm52TensorContract {
    dtype: Dtype,
    shape: Vec<usize>,
}

fn expected_tensor_contract(name: &str) -> Result<Glm52TensorContract> {
    if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        return Ok(contract(Dtype::BF16, [GLM52_VOCAB, GLM52_HIDDEN]));
    }
    if name == "model.norm.weight" {
        return Ok(contract(Dtype::BF16, [GLM52_HIDDEN]));
    }

    let layer_idx = parse_layer_index(name)?;
    ensure!(
        layer_idx < GLM52_LAYERS,
        "GLM5.2 runtime tensor contract excludes layer {layer_idx}: {name}"
    );

    if name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
    {
        return Ok(contract(Dtype::BF16, [GLM52_HIDDEN]));
    }
    if name.ends_with(".self_attn.q_a_layernorm.weight") {
        return Ok(contract(Dtype::BF16, [GLM52_Q_LORA_RANK]));
    }
    if name.ends_with(".self_attn.kv_a_layernorm.weight") {
        return Ok(contract(Dtype::BF16, [GLM52_KV_LORA_RANK]));
    }

    if let Some(projection) = attention_projection_contract(name) {
        return Ok(projection);
    }
    if let Some(indexer) = indexer_contract(name) {
        return Ok(indexer);
    }
    if let Some(mlp) = mlp_contract(layer_idx, name) {
        return Ok(mlp);
    }

    anyhow::bail!("no GLM5.2 tensor contract for {name}")
}

fn attention_projection_contract(name: &str) -> Option<Glm52TensorContract> {
    projection_contract(name, ".self_attn.q_a_proj", GLM52_Q_LORA_RANK, GLM52_HIDDEN)
        .or_else(|| {
            projection_contract(
                name,
                ".self_attn.q_b_proj",
                GLM52_Q_B_OUT,
                GLM52_Q_LORA_RANK,
            )
        })
        .or_else(|| {
            projection_contract(
                name,
                ".self_attn.kv_a_proj_with_mqa",
                GLM52_KV_A_OUT,
                GLM52_HIDDEN,
            )
        })
        .or_else(|| {
            projection_contract(
                name,
                ".self_attn.kv_b_proj",
                GLM52_KV_B_OUT,
                GLM52_KV_LORA_RANK,
            )
        })
        .or_else(|| projection_contract(name, ".self_attn.o_proj", GLM52_HIDDEN, GLM52_O_PROJ_IN))
}

fn indexer_contract(name: &str) -> Option<Glm52TensorContract> {
    if name.ends_with(".self_attn.indexer.k_norm.weight")
        || name.ends_with(".self_attn.indexer.k_norm.bias")
    {
        return Some(contract(Dtype::BF16, [GLM52_INDEX_HEAD_DIM]));
    }
    if name.ends_with(".self_attn.indexer.weights_proj.weight") {
        return Some(contract(Dtype::BF16, [GLM52_INDEX_HEADS, GLM52_HIDDEN]));
    }
    projection_contract(
        name,
        ".self_attn.indexer.wk",
        GLM52_INDEX_HEAD_DIM,
        GLM52_HIDDEN,
    )
    .or_else(|| {
        projection_contract(
            name,
            ".self_attn.indexer.wq_b",
            GLM52_INDEX_HEADS * GLM52_INDEX_HEAD_DIM,
            GLM52_Q_LORA_RANK,
        )
    })
}

fn mlp_contract(layer_idx: usize, name: &str) -> Option<Glm52TensorContract> {
    if name.ends_with(".mlp.gate.weight") {
        return Some(contract(Dtype::BF16, [GLM52_ROUTED_EXPERTS, GLM52_HIDDEN]));
    }
    if name.ends_with(".mlp.gate.e_score_correction_bias") {
        return Some(contract(Dtype::F32, [GLM52_ROUTED_EXPERTS]));
    }

    let intermediate = if layer_idx < GLM52_DENSE_LAYERS {
        GLM52_DENSE_INTERMEDIATE
    } else {
        GLM52_EXPERT_INTERMEDIATE
    };
    projection_contract(name, ".gate_proj", intermediate, GLM52_HIDDEN)
        .or_else(|| projection_contract(name, ".up_proj", intermediate, GLM52_HIDDEN))
        .or_else(|| projection_contract(name, ".down_proj", GLM52_HIDDEN, intermediate))
}

fn projection_contract(
    name: &str,
    stem: &str,
    rows: usize,
    cols: usize,
) -> Option<Glm52TensorContract> {
    if name.ends_with(&format!("{stem}.weight")) {
        return Some(contract(Dtype::F8_E4M3, [rows, cols]));
    }
    if name.ends_with(&format!("{stem}.weight_scale_inv")) {
        return Some(contract(
            Dtype::F32,
            [rows.div_ceil(FP8_BLOCK_SIZE), cols.div_ceil(FP8_BLOCK_SIZE)],
        ));
    }
    None
}

fn contract<const N: usize>(dtype: Dtype, shape: [usize; N]) -> Glm52TensorContract {
    Glm52TensorContract {
        dtype,
        shape: shape.to_vec(),
    }
}

fn parse_layer_index(name: &str) -> Result<usize> {
    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 tensor is not a layer tensor: {name}"))?;
    let (idx, _) = rest
        .split_once('.')
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 tensor has malformed layer prefix: {name}"))?;
    idx.parse::<usize>()
        .map_err(|err| anyhow::anyhow!("GLM5.2 tensor has invalid layer index in {name}: {err}"))
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = fs::File::open(path)
        .map_err(|err| anyhow::anyhow!("failed to open {}: {err}", path.display()))?;
    // SAFETY: checkpoint shards are opened read-only and the mapping is only
    // consumed while reading safetensors metadata.
    unsafe { Mmap::map(&file) }
        .map_err(|err| anyhow::anyhow!("failed to mmap {}: {err}", path.display()))
}

fn tensor(tensors: &BTreeMap<&str, &str>, name: &str) -> Result<Glm52TensorEntry> {
    let shard = tensors
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 safetensors index missing tensor {name}"))?;
    Ok(Glm52TensorEntry {
        name: name.to_owned(),
        shard: (*shard).to_owned(),
    })
}

fn fp8_projection(
    tensors: &BTreeMap<&str, &str>,
    prefix: &str,
) -> Result<Glm52Fp8ProjectionManifest> {
    Ok(Glm52Fp8ProjectionManifest {
        weight: tensor(tensors, &format!("{prefix}.weight"))?,
        weight_scale_inv: tensor(tensors, &format!("{prefix}.weight_scale_inv"))?,
    })
}

fn attention_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
) -> Result<Glm52AttentionManifest> {
    let prefix = format!("model.layers.{layer_idx}");
    let indexer = tensors
        .contains_key(format!("{prefix}.self_attn.indexer.k_norm.weight").as_str())
        .then(|| indexer_manifest(tensors, &prefix))
        .transpose()?;
    Ok(Glm52AttentionManifest {
        input_layernorm: tensor(tensors, &format!("{prefix}.input_layernorm.weight"))?,
        q_a_proj: fp8_projection(tensors, &format!("{prefix}.self_attn.q_a_proj"))?,
        q_a_layernorm: tensor(tensors, &format!("{prefix}.self_attn.q_a_layernorm.weight"))?,
        q_b_proj: fp8_projection(tensors, &format!("{prefix}.self_attn.q_b_proj"))?,
        kv_a_proj_with_mqa: fp8_projection(
            tensors,
            &format!("{prefix}.self_attn.kv_a_proj_with_mqa"),
        )?,
        kv_a_layernorm: tensor(
            tensors,
            &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
        )?,
        kv_b_proj: fp8_projection(tensors, &format!("{prefix}.self_attn.kv_b_proj"))?,
        o_proj: fp8_projection(tensors, &format!("{prefix}.self_attn.o_proj"))?,
        post_attention_layernorm: tensor(
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?,
        indexer,
    })
}

fn indexer_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_prefix: &str,
) -> Result<Glm52IndexerManifest> {
    let prefix = format!("{layer_prefix}.self_attn.indexer");
    Ok(Glm52IndexerManifest {
        k_norm_weight: tensor(tensors, &format!("{prefix}.k_norm.weight"))?,
        k_norm_bias: tensor(tensors, &format!("{prefix}.k_norm.bias"))?,
        weights_proj: tensor(tensors, &format!("{prefix}.weights_proj.weight"))?,
        wk: fp8_projection(tensors, &format!("{prefix}.wk"))?,
        wq_b: fp8_projection(tensors, &format!("{prefix}.wq_b"))?,
    })
}

fn dense_mlp_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
) -> Result<Glm52DenseMlpManifest> {
    let prefix = format!("model.layers.{layer_idx}.mlp");
    Ok(Glm52DenseMlpManifest {
        gate_proj: fp8_projection(tensors, &format!("{prefix}.gate_proj"))?,
        up_proj: fp8_projection(tensors, &format!("{prefix}.up_proj"))?,
        down_proj: fp8_projection(tensors, &format!("{prefix}.down_proj"))?,
    })
}

fn moe_layer_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
) -> Result<Glm52MoeLayerManifest> {
    let prefix = format!("model.layers.{layer_idx}.mlp");
    let routed_experts = (0..GLM52_ROUTED_EXPERTS)
        .map(|expert_idx| routed_expert_manifest(tensors, layer_idx, expert_idx))
        .collect::<Result<Vec<_>>>()?;
    Ok(Glm52MoeLayerManifest {
        router: Glm52RouterManifest {
            gate_weight: tensor(tensors, &format!("{prefix}.gate.weight"))?,
            e_score_correction_bias: tensor(
                tensors,
                &format!("{prefix}.gate.e_score_correction_bias"),
            )?,
        },
        shared_experts: Glm52SharedExpertManifest {
            gate_proj: fp8_projection(tensors, &format!("{prefix}.shared_experts.gate_proj"))?,
            up_proj: fp8_projection(tensors, &format!("{prefix}.shared_experts.up_proj"))?,
            down_proj: fp8_projection(tensors, &format!("{prefix}.shared_experts.down_proj"))?,
        },
        routed_experts,
    })
}

fn routed_expert_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
    expert_idx: usize,
) -> Result<Glm52RoutedExpertManifest> {
    let prefix = format!("model.layers.{layer_idx}.mlp.experts.{expert_idx}");
    Ok(Glm52RoutedExpertManifest {
        expert_idx,
        gate_proj: fp8_projection(tensors, &format!("{prefix}.gate_proj"))?,
        up_proj: fp8_projection(tensors, &format!("{prefix}.up_proj"))?,
        down_proj: fp8_projection(tensors, &format!("{prefix}.down_proj"))?,
    })
}

fn push_attention<'a>(out: &mut Vec<&'a Glm52TensorEntry>, attention: &'a Glm52AttentionManifest) {
    out.push(&attention.input_layernorm);
    push_fp8_projection(out, &attention.q_a_proj);
    out.push(&attention.q_a_layernorm);
    push_fp8_projection(out, &attention.q_b_proj);
    push_fp8_projection(out, &attention.kv_a_proj_with_mqa);
    out.push(&attention.kv_a_layernorm);
    push_fp8_projection(out, &attention.kv_b_proj);
    push_fp8_projection(out, &attention.o_proj);
    out.push(&attention.post_attention_layernorm);
    if let Some(indexer) = &attention.indexer {
        out.push(&indexer.k_norm_weight);
        out.push(&indexer.k_norm_bias);
        out.push(&indexer.weights_proj);
        push_fp8_projection(out, &indexer.wk);
        push_fp8_projection(out, &indexer.wq_b);
    }
}

fn push_dense_mlp<'a>(out: &mut Vec<&'a Glm52TensorEntry>, mlp: &'a Glm52DenseMlpManifest) {
    push_fp8_projection(out, &mlp.gate_proj);
    push_fp8_projection(out, &mlp.up_proj);
    push_fp8_projection(out, &mlp.down_proj);
}

fn push_shared_expert<'a>(
    out: &mut Vec<&'a Glm52TensorEntry>,
    expert: &'a Glm52SharedExpertManifest,
) {
    push_fp8_projection(out, &expert.gate_proj);
    push_fp8_projection(out, &expert.up_proj);
    push_fp8_projection(out, &expert.down_proj);
}

fn push_routed_expert<'a>(
    out: &mut Vec<&'a Glm52TensorEntry>,
    expert: &'a Glm52RoutedExpertManifest,
) {
    push_fp8_projection(out, &expert.gate_proj);
    push_fp8_projection(out, &expert.up_proj);
    push_fp8_projection(out, &expert.down_proj);
}

fn push_fp8_projection<'a>(
    out: &mut Vec<&'a Glm52TensorEntry>,
    projection: &'a Glm52Fp8ProjectionManifest,
) {
    out.push(&projection.weight);
    out.push(&projection.weight_scale_inv);
}
