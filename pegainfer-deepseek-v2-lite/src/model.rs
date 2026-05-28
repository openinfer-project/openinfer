use std::{collections::HashMap, path::Path};

use anyhow::{Result, bail, ensure};
use half::bf16;
use pegainfer_core::{
    ops,
    tensor::{DeviceContext, DeviceMatrix, HiddenStates},
    weight_loader::{deserialize_shards, load_shard_info, load_tensor_2d, mmap_shards},
};
use safetensors::Dtype;

use crate::{Config, device::activate, ep::ExpertParallelLayout};

pub(crate) struct DriverRankModel {
    pub(crate) ctx: DeviceContext,
    pub(crate) layout: ExpertParallelLayout,
    pub(crate) embed_tokens: DeviceMatrix,
    pub(crate) lm_head: DeviceMatrix,
    pub(crate) norm_host: Vec<f32>,
    pub(crate) layers: Vec<LayerWeights>,
}

pub(crate) struct ExpertRankModel {
    pub(crate) ctx: DeviceContext,
    pub(crate) layout: ExpertParallelLayout,
    layers: Vec<Option<Vec<ExpertMlp>>>,
}

pub(crate) struct LayerWeights {
    pub(crate) input_layernorm_host: Vec<f32>,
    pub(crate) post_attention_layernorm_host: Vec<f32>,
    pub(crate) attention: AttentionWeights,
    pub(crate) mlp: MlpWeights,
}

pub(crate) struct AttentionWeights {
    pub(crate) q_proj: DeviceMatrix,
    pub(crate) kv_a_proj: DeviceMatrix,
    pub(crate) kv_a_norm_host: Vec<f32>,
    pub(crate) kv_b_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
}

pub(crate) enum MlpWeights {
    Dense(DenseMlp),
    Moe(MoeMlp),
}

pub(crate) struct DenseMlp {
    gate_up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

pub(crate) struct MoeMlp {
    pub(crate) gate_host: Vec<f32>,
    pub(crate) shared: DenseMlp,
    pub(crate) experts: Vec<ExpertMlp>,
}

pub(crate) struct ExpertMlp {
    pub(crate) global_expert: usize,
    pub(crate) dense: DenseMlp,
}

impl DriverRankModel {
    pub(crate) fn load(
        model_path: &Path,
        config: &Config,
        layout: ExpertParallelLayout,
        device_ordinal: usize,
    ) -> Result<Self> {
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        activate(&ctx)?;

        with_weight_shards(model_path, |shards, weight_map| {
            let embed_tokens =
                load_tensor_2d(&ctx, shards, weight_map, "model.embed_tokens.weight")?;
            ensure!(
                !config.tie_word_embeddings,
                "DeepSeek-V2-Lite first gate expects untied lm_head"
            );
            let lm_head = load_tensor_2d(&ctx, shards, weight_map, "lm_head.weight")?;
            let norm_host = load_tensor_1d_host(shards, weight_map, "model.norm.weight")?;

            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            for layer_idx in 0..config.num_hidden_layers {
                let prefix = format!("model.layers.{layer_idx}");
                let input_layernorm_host = load_tensor_1d_host(
                    shards,
                    weight_map,
                    &format!("{prefix}.input_layernorm.weight"),
                )?;
                let post_attention_layernorm_host = load_tensor_1d_host(
                    shards,
                    weight_map,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?;
                let attn = format!("{prefix}.self_attn");
                let kv_a_norm_host = load_tensor_1d_host(
                    shards,
                    weight_map,
                    &format!("{attn}.kv_a_layernorm.weight"),
                )?;
                let attention = AttentionWeights {
                    q_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.q_proj.weight"),
                    )?,
                    kv_a_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.kv_a_proj_with_mqa.weight"),
                    )?,
                    kv_a_norm_host,
                    kv_b_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.kv_b_proj.weight"),
                    )?,
                    o_proj: load_tensor_2d(
                        &ctx,
                        shards,
                        weight_map,
                        &format!("{attn}.o_proj.weight"),
                    )?,
                };
                let mlp_prefix = format!("{prefix}.mlp");
                let mlp = if config.is_moe_layer(layer_idx) {
                    MlpWeights::Moe(load_moe_mlp(
                        &ctx,
                        shards,
                        weight_map,
                        config,
                        &layout,
                        &mlp_prefix,
                    )?)
                } else {
                    MlpWeights::Dense(load_dense_mlp(&ctx, shards, weight_map, &mlp_prefix)?)
                };
                layers.push(LayerWeights {
                    input_layernorm_host,
                    post_attention_layernorm_host,
                    attention,
                    mlp,
                });
            }

            Ok(Self {
                ctx,
                layout,
                embed_tokens,
                lm_head,
                norm_host,
                layers,
            })
        })
    }

    pub(crate) fn routed_expert(
        &self,
        layer_idx: usize,
        global_expert: usize,
    ) -> Result<&ExpertMlp> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?;
        let MlpWeights::Moe(moe) = &layer.mlp else {
            bail!("layer {layer_idx} is not a MoE layer");
        };
        routed_expert_from_slice(&self.layout, &moe.experts, global_expert)
    }
}

impl ExpertRankModel {
    pub(crate) fn load(
        model_path: &Path,
        config: &Config,
        layout: ExpertParallelLayout,
        device_ordinal: usize,
    ) -> Result<Self> {
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        activate(&ctx)?;

        with_weight_shards(model_path, |shards, weight_map| {
            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            for layer_idx in 0..config.num_hidden_layers {
                if config.is_moe_layer(layer_idx) {
                    let prefix = format!("model.layers.{layer_idx}.mlp");
                    layers.push(Some(load_owned_experts(
                        &ctx, shards, weight_map, config, &layout, &prefix,
                    )?));
                } else {
                    layers.push(None);
                }
            }

            Ok(Self {
                ctx,
                layout,
                layers,
            })
        })
    }

    pub(crate) fn routed_expert(
        &self,
        layer_idx: usize,
        global_expert: usize,
    ) -> Result<&ExpertMlp> {
        let experts = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} is not a MoE layer"))?;
        routed_expert_from_slice(&self.layout, experts, global_expert)
    }
}

fn with_weight_shards<T>(
    model_path: &Path,
    load: impl FnOnce(&[safetensors::SafeTensors<'_>], &HashMap<String, usize>) -> Result<T>,
) -> Result<T> {
    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    let (shard_paths, weight_map) = load_shard_info(model_path_str)?;
    let mmaps = mmap_shards(&shard_paths)?;
    let shards = deserialize_shards(&mmaps)?;
    load(&shards, &weight_map)
}

fn routed_expert_from_slice<'a>(
    layout: &ExpertParallelLayout,
    experts: &'a [ExpertMlp],
    global_expert: usize,
) -> Result<&'a ExpertMlp> {
    let local_expert = layout.local_expert(global_expert)?;
    let expert = experts.get(local_expert).ok_or_else(|| {
        anyhow::anyhow!(
            "rank {} local expert {} missing for global expert {}",
            layout.rank(),
            local_expert,
            global_expert
        )
    })?;
    ensure!(
        expert.global_expert == global_expert,
        "rank {} local expert {} expected global {}, got {}",
        layout.rank(),
        local_expert,
        global_expert,
        expert.global_expert
    );
    Ok(expert)
}

fn load_dense_mlp(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    prefix: &str,
) -> Result<DenseMlp> {
    let gate_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.gate_proj.weight"),
    )?;
    let up_proj = load_tensor_2d(ctx, shards, weight_map, &format!("{prefix}.up_proj.weight"))?;
    let gate_up_proj = DeviceMatrix::vstack(ctx, &[&gate_proj, &up_proj])?;
    let down_proj = load_tensor_2d(
        ctx,
        shards,
        weight_map,
        &format!("{prefix}.down_proj.weight"),
    )?;
    Ok(DenseMlp {
        gate_up_proj,
        down_proj,
    })
}

fn load_moe_mlp(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    config: &Config,
    layout: &ExpertParallelLayout,
    prefix: &str,
) -> Result<MoeMlp> {
    let gate_host = load_tensor_2d_host(shards, weight_map, &format!("{prefix}.gate.weight"))?;
    let shared = load_dense_mlp(ctx, shards, weight_map, &format!("{prefix}.shared_experts"))?;
    let experts = load_owned_experts(ctx, shards, weight_map, config, layout, prefix)?;
    Ok(MoeMlp {
        gate_host,
        shared,
        experts,
    })
}

fn load_tensor_1d_host(
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<f32>> {
    load_bf16_tensor_host(shards, weight_map, name, 1)
}

fn load_tensor_2d_host(
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<f32>> {
    load_bf16_tensor_host(shards, weight_map, name, 2)
}

fn load_bf16_tensor_host(
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    name: &str,
    expected_rank: usize,
) -> Result<Vec<f32>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    ensure!(
        tensor.dtype() == Dtype::BF16,
        "tensor {name} expected BF16, got {:?}",
        tensor.dtype()
    );
    ensure!(
        shape.len() == expected_rank,
        "tensor {name} expected {expected_rank}D, got {shape:?}"
    );
    let elem_count = shape.iter().product::<usize>();
    let expected_bytes = elem_count * std::mem::size_of::<bf16>();
    ensure!(
        tensor.data().len() == expected_bytes,
        "tensor {name} expected {expected_bytes} bf16 bytes, got {}",
        tensor.data().len()
    );
    Ok(tensor
        .data()
        .chunks_exact(std::mem::size_of::<bf16>())
        .map(|bytes| {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            bf16::from_bits(bits).to_f32()
        })
        .collect())
}

fn find_tensor<'a>(
    shards: &'a [safetensors::SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        return shards[idx]
            .tensor(name)
            .map_err(|err| anyhow::anyhow!("failed to load tensor {name}: {err}"));
    }
    for shard in shards {
        if let Ok(tensor) = shard.tensor(name) {
            return Ok(tensor);
        }
    }
    bail!("tensor {name} not found in any shard")
}

fn load_owned_experts(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    config: &Config,
    layout: &ExpertParallelLayout,
    prefix: &str,
) -> Result<Vec<ExpertMlp>> {
    let mut experts = Vec::with_capacity(layout.experts_per_rank());
    for global_expert in layout.owned_experts() {
        let dense = load_dense_mlp(
            ctx,
            shards,
            weight_map,
            &format!("{prefix}.experts.{global_expert}"),
        )?;
        experts.push(ExpertMlp {
            global_expert,
            dense,
        });
    }
    ensure!(
        experts.len() == config.n_routed_experts / layout.ep_size(),
        "rank {} loaded {} routed experts, expected {}",
        layout.rank(),
        experts.len(),
        config.n_routed_experts / layout.ep_size()
    );
    Ok(experts)
}

pub(crate) fn dense_mlp_forward(
    ctx: &DeviceContext,
    mlp: &DenseMlp,
    input: &HiddenStates,
) -> Result<HiddenStates> {
    activate(ctx)?;
    let gate_up = ops::gemm(ctx, &mlp.gate_up_proj, input)?;
    let mut act = HiddenStates::zeros(ctx, gate_up.hidden_dim / 2, input.seq_len)?;
    ops::silu_mul_fused_batch_into(ctx, &gate_up, &mut act);
    ops::gemm(ctx, &mlp.down_proj, &act)
}

pub(crate) fn dense_mlp_forward_per_token(
    ctx: &DeviceContext,
    mlp: &DenseMlp,
    input: &HiddenStates,
) -> Result<HiddenStates> {
    activate(ctx)?;
    let gate_up = ops::gemm_per_token(ctx, &mlp.gate_up_proj, input)?;
    let mut act = HiddenStates::zeros(ctx, gate_up.hidden_dim / 2, input.seq_len)?;
    ops::silu_mul_fused_batch_into(ctx, &gate_up, &mut act);
    ops::gemm_per_token(ctx, &mlp.down_proj, &act)
}
