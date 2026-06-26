use anyhow::{Context, Result, ensure};
use safetensors::Dtype;

use super::load::Glm52GpuRawTensor;
use super::{
    FP8_BLOCK_SIZE, FULL_INDEXER_LAYER_COUNT, GLM52_DENSE_LAYERS, GLM52_KV_B_OUT, GLM52_LAYERS,
    GLM52_MOE_LAYERS, GLM52_O_PROJ_IN, Glm52AttentionManifest, Glm52DenseMlpManifest,
    Glm52Fp8ProjectionManifest, Glm52IndexerManifest, Glm52LayerKindManifest, Glm52RankGpuWeights,
    Glm52RankWeightPlan, Glm52RoutedExpertManifest, Glm52RouterManifest, Glm52SharedExpertManifest,
    Glm52WeightManifest, expected_tensor_contract,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52TopWeightNames {
    token_embedding: String,
    final_norm: String,
    lm_head: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52Fp8ProjectionWeightNames {
    pub(crate) weight: String,
    pub(crate) weight_scale_inv: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52IndexerWeightNames {
    k_norm_weight: String,
    k_norm_bias: String,
    weights_proj: String,
    wk: Glm52Fp8ProjectionWeightNames,
    wq_b: Glm52Fp8ProjectionWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52AttentionWeightNames {
    input_layernorm: String,
    q_a_proj: Glm52Fp8ProjectionWeightNames,
    q_a_layernorm: String,
    q_b_proj: Glm52Fp8ProjectionWeightNames,
    kv_a_proj_with_mqa: Glm52Fp8ProjectionWeightNames,
    kv_a_layernorm: String,
    kv_b_proj: Glm52Fp8ProjectionWeightNames,
    o_proj: Glm52Fp8ProjectionWeightNames,
    post_attention_layernorm: String,
    indexer: Option<Glm52IndexerWeightNames>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DenseMlpWeightNames {
    gate_proj: Glm52Fp8ProjectionWeightNames,
    up_proj: Glm52Fp8ProjectionWeightNames,
    down_proj: Glm52Fp8ProjectionWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RouterWeightNames {
    gate_weight: String,
    e_score_correction_bias: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52SharedExpertWeightNames {
    gate_proj: Glm52Fp8ProjectionWeightNames,
    up_proj: Glm52Fp8ProjectionWeightNames,
    down_proj: Glm52Fp8ProjectionWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RoutedExpertWeightNames {
    pub(crate) global_expert: usize,
    pub(crate) gate_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) up_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) down_proj: Glm52Fp8ProjectionWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52MoeLayerWeightNames {
    pub(crate) router: Glm52RouterWeightNames,
    pub(crate) shared_experts: Glm52SharedExpertWeightNames,
    pub(crate) routed_experts: Vec<Glm52RoutedExpertWeightNames>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Glm52LayerWeightKindNames {
    Dense(Glm52DenseMlpWeightNames),
    Moe(Glm52MoeLayerWeightNames),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52LayerWeightNames {
    pub(crate) layer_idx: usize,
    pub(crate) attention: Glm52AttentionWeightNames,
    pub(crate) kind: Glm52LayerWeightKindNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankWeightNames {
    pub(crate) rank: usize,
    pub(crate) plan: Glm52RankWeightPlan,
    pub(crate) top: Glm52TopWeightNames,
    pub(crate) layers: Vec<Glm52LayerWeightNames>,
}

pub(crate) struct Glm52Fp8ProjectionGpuWeights<'a> {
    pub(crate) weight: &'a Glm52GpuRawTensor,
    pub(crate) weight_scale_inv: &'a Glm52GpuRawTensor,
}

pub(crate) struct Glm52IndexerGpuWeights<'a> {
    pub(crate) k_norm_weight: &'a Glm52GpuRawTensor,
    pub(crate) k_norm_bias: &'a Glm52GpuRawTensor,
    pub(crate) weights_proj: &'a Glm52GpuRawTensor,
    pub(crate) wk: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) wq_b: Glm52Fp8ProjectionGpuWeights<'a>,
}

pub(crate) struct Glm52AttentionGpuWeights<'a> {
    pub(crate) input_layernorm: &'a Glm52GpuRawTensor,
    pub(crate) q_a_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) q_a_layernorm: &'a Glm52GpuRawTensor,
    pub(crate) q_b_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) kv_a_proj_with_mqa: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) kv_a_layernorm: &'a Glm52GpuRawTensor,
    pub(crate) kv_b_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) o_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) post_attention_layernorm: &'a Glm52GpuRawTensor,
    pub(crate) indexer: Option<Glm52IndexerGpuWeights<'a>>,
}

pub(crate) struct Glm52DenseMlpGpuWeights<'a> {
    pub(crate) gate_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) up_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) down_proj: Glm52Fp8ProjectionGpuWeights<'a>,
}

pub(crate) struct Glm52RouterGpuWeights<'a> {
    pub(crate) gate_weight: &'a Glm52GpuRawTensor,
    pub(crate) e_score_correction_bias: &'a Glm52GpuRawTensor,
}

pub(crate) struct Glm52SharedExpertGpuWeights<'a> {
    pub(crate) gate_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) up_proj: Glm52Fp8ProjectionGpuWeights<'a>,
    pub(crate) down_proj: Glm52Fp8ProjectionGpuWeights<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52NonExpertWeightContractReport {
    pub(crate) rank: usize,
    pub(crate) tensor_count: usize,
    pub(crate) total_bytes: usize,
    pub(crate) dense_layers: usize,
    pub(crate) moe_layers: usize,
    pub(crate) full_indexer_layers: usize,
    pub(crate) attention_fp8_projections: usize,
    pub(crate) dense_fp8_projections: usize,
    pub(crate) shared_fp8_projections: usize,
    pub(crate) total_fp8_projections: usize,
    pub(crate) fp8_weight_bytes: usize,
    pub(crate) fp8_scale_bytes: usize,
    pub(crate) max_out_dim: usize,
    pub(crate) max_in_dim: usize,
    pub(crate) max_scale_rows: usize,
    pub(crate) max_scale_cols: usize,
}

impl Glm52WeightManifest {
    pub(crate) fn rank_weight_names(&self, rank: usize) -> Result<Glm52RankWeightNames> {
        let plan = self.rank_plan(rank)?;
        let local_expert_range = self.rank_local_expert_range(rank)?;
        let top = Glm52TopWeightNames {
            token_embedding: self.token_embedding.name.clone(),
            final_norm: self.final_norm.name.clone(),
            lm_head: self.lm_head.name.clone(),
        };
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let attention = Glm52AttentionWeightNames::from_manifest(&layer.attention);
            let kind = match &layer.kind {
                Glm52LayerKindManifest::Dense(mlp) => {
                    Glm52LayerWeightKindNames::Dense(Glm52DenseMlpWeightNames::from_manifest(mlp))
                }
                Glm52LayerKindManifest::Moe(moe) => {
                    let routed_experts = moe
                        .routed_experts
                        .iter()
                        .filter(|expert| local_expert_range.contains(&expert.expert_idx))
                        .map(Glm52RoutedExpertWeightNames::from_manifest)
                        .collect::<Vec<_>>();
                    Glm52LayerWeightKindNames::Moe(Glm52MoeLayerWeightNames {
                        router: Glm52RouterWeightNames::from_manifest(&moe.router),
                        shared_experts: Glm52SharedExpertWeightNames::from_manifest(
                            &moe.shared_experts,
                        ),
                        routed_experts,
                    })
                }
            };
            layers.push(Glm52LayerWeightNames {
                layer_idx: layer.layer_idx,
                attention,
                kind,
            });
        }
        Ok(Glm52RankWeightNames {
            rank,
            plan,
            top,
            layers,
        })
    }
}

impl Glm52RankGpuWeights {
    pub(crate) fn validate_non_expert_weight_contract(
        &self,
        names: &Glm52RankWeightNames,
    ) -> Result<Glm52NonExpertWeightContractReport> {
        ensure!(
            self.rank == names.rank,
            "GLM5.2 GPU rank {} does not match typed names rank {}",
            self.rank,
            names.rank
        );
        ensure!(
            names.layers.len() == GLM52_LAYERS,
            "GLM5.2 typed names expected {GLM52_LAYERS} layers, got {}",
            names.layers.len()
        );

        let mut summary = TensorSummary::default();
        let mut dense_layers = 0usize;
        let mut moe_layers = 0usize;
        let mut full_indexer_layers = 0usize;
        let mut attention_fp8_projections = 0usize;
        let mut dense_fp8_projections = 0usize;
        let mut shared_fp8_projections = 0usize;
        for tensor in [
            self.expect_tensor(&names.top.token_embedding)?,
            self.expect_tensor(&names.top.final_norm)?,
            self.expect_tensor(&names.top.lm_head)?,
        ] {
            summary.add(tensor);
        }

        for layer in &names.layers {
            let attention = self.attention_view(&layer.attention)?;
            let attention_summary = attention.summary()?;
            attention_fp8_projections += attention_summary.fp8_projection_count;
            summary.merge(&attention_summary);
            if attention.indexer.is_some() {
                full_indexer_layers += 1;
            }

            match &layer.kind {
                Glm52LayerWeightKindNames::Dense(mlp) => {
                    dense_layers += 1;
                    let mlp = self.dense_mlp_view(mlp)?;
                    let mlp_summary = mlp.summary()?;
                    dense_fp8_projections += mlp_summary.fp8_projection_count;
                    summary.merge(&mlp_summary);
                }
                Glm52LayerWeightKindNames::Moe(moe) => {
                    moe_layers += 1;
                    let router = Glm52RouterGpuWeights {
                        gate_weight: self.expect_tensor(&moe.router.gate_weight)?,
                        e_score_correction_bias: self
                            .expect_tensor(&moe.router.e_score_correction_bias)?,
                    };
                    let shared_experts = Glm52SharedExpertGpuWeights {
                        gate_proj: self.fp8_projection_view(&moe.shared_experts.gate_proj)?,
                        up_proj: self.fp8_projection_view(&moe.shared_experts.up_proj)?,
                        down_proj: self.fp8_projection_view(&moe.shared_experts.down_proj)?,
                    };
                    summary.add(router.gate_weight);
                    summary.add(router.e_score_correction_bias);
                    let shared_summary = shared_experts.summary()?;
                    shared_fp8_projections += shared_summary.fp8_projection_count;
                    summary.merge(&shared_summary);
                }
            }
        }

        ensure!(
            dense_layers == GLM52_DENSE_LAYERS && moe_layers == GLM52_MOE_LAYERS,
            "GLM5.2 rank {} non-expert weight contract has dense/moe layer counts {}/{}, expected {GLM52_DENSE_LAYERS}/{GLM52_MOE_LAYERS}",
            self.rank,
            dense_layers,
            moe_layers
        );
        ensure!(
            full_indexer_layers == FULL_INDEXER_LAYER_COUNT,
            "GLM5.2 rank {} non-expert weight contract has {} full-indexer layers, expected {FULL_INDEXER_LAYER_COUNT}",
            self.rank,
            full_indexer_layers
        );
        let expected_attention_fp8 = GLM52_LAYERS * 5 + FULL_INDEXER_LAYER_COUNT * 2;
        let expected_dense_fp8 = GLM52_DENSE_LAYERS * 3;
        let expected_shared_fp8 = GLM52_MOE_LAYERS * 3;
        ensure!(
            attention_fp8_projections == expected_attention_fp8,
            "GLM5.2 rank {} attention FP8 projection count {} != expected {}",
            self.rank,
            attention_fp8_projections,
            expected_attention_fp8
        );
        ensure!(
            dense_fp8_projections == expected_dense_fp8,
            "GLM5.2 rank {} dense FP8 projection count {} != expected {}",
            self.rank,
            dense_fp8_projections,
            expected_dense_fp8
        );
        ensure!(
            shared_fp8_projections == expected_shared_fp8,
            "GLM5.2 rank {} shared-expert FP8 projection count {} != expected {}",
            self.rank,
            shared_fp8_projections,
            expected_shared_fp8
        );
        ensure!(
            summary.tensor_count == self.tensors.len(),
            "GLM5.2 rank {} non-expert weight contract covers {} tensors, resident map has {}",
            self.rank,
            summary.tensor_count,
            self.tensors.len()
        );
        ensure!(
            summary.total_bytes == self.total_bytes,
            "GLM5.2 rank {} non-expert weight contract covers {} bytes, resident map has {}",
            self.rank,
            summary.total_bytes,
            self.total_bytes
        );
        ensure!(
            summary.max_out_dim == GLM52_KV_B_OUT && summary.max_in_dim == GLM52_O_PROJ_IN,
            "GLM5.2 rank {} non-expert FP8 projection max_out/max_in are {}/{}, expected {}/{}",
            self.rank,
            summary.max_out_dim,
            summary.max_in_dim,
            GLM52_KV_B_OUT,
            GLM52_O_PROJ_IN
        );
        ensure!(
            summary.max_scale_rows == GLM52_KV_B_OUT.div_ceil(FP8_BLOCK_SIZE)
                && summary.max_scale_cols == GLM52_O_PROJ_IN.div_ceil(FP8_BLOCK_SIZE),
            "GLM5.2 rank {} non-expert FP8 scale max grid is {}x{}, expected {}x{}",
            self.rank,
            summary.max_scale_rows,
            summary.max_scale_cols,
            GLM52_KV_B_OUT.div_ceil(FP8_BLOCK_SIZE),
            GLM52_O_PROJ_IN.div_ceil(FP8_BLOCK_SIZE)
        );
        let total_fp8_projections =
            attention_fp8_projections + dense_fp8_projections + shared_fp8_projections;
        Ok(Glm52NonExpertWeightContractReport {
            rank: self.rank,
            tensor_count: summary.tensor_count,
            total_bytes: summary.total_bytes,
            dense_layers,
            moe_layers,
            full_indexer_layers,
            attention_fp8_projections,
            dense_fp8_projections,
            shared_fp8_projections,
            total_fp8_projections,
            fp8_weight_bytes: summary.fp8_weight_bytes,
            fp8_scale_bytes: summary.fp8_scale_bytes,
            max_out_dim: summary.max_out_dim,
            max_in_dim: summary.max_in_dim,
            max_scale_rows: summary.max_scale_rows,
            max_scale_cols: summary.max_scale_cols,
        })
    }

    pub(crate) fn first_moe_router<'a>(
        &'a self,
        names: &'a Glm52RankWeightNames,
    ) -> Result<Glm52RouterGpuWeights<'a>> {
        ensure!(
            self.rank == names.rank,
            "GLM5.2 GPU rank {} does not match typed names rank {}",
            self.rank,
            names.rank
        );
        for layer in &names.layers {
            if let Glm52LayerWeightKindNames::Moe(moe) = &layer.kind {
                return Ok(Glm52RouterGpuWeights {
                    gate_weight: self.expect_tensor(&moe.router.gate_weight)?,
                    e_score_correction_bias: self
                        .expect_tensor(&moe.router.e_score_correction_bias)?,
                });
            }
        }
        anyhow::bail!("GLM5.2 typed rank weights have no MoE layer")
    }

    pub(crate) fn first_attention<'a>(
        &'a self,
        names: &'a Glm52RankWeightNames,
    ) -> Result<Glm52AttentionGpuWeights<'a>> {
        ensure!(
            self.rank == names.rank,
            "GLM5.2 GPU rank {} does not match typed names rank {}",
            self.rank,
            names.rank
        );
        let layer = names
            .layers
            .first()
            .ok_or_else(|| anyhow::anyhow!("GLM5.2 typed rank weights have no layers"))?;
        ensure!(
            layer.layer_idx == 0,
            "GLM5.2 first attention smoke expects layer 0, got {}",
            layer.layer_idx
        );
        self.attention_view(&layer.attention)
    }

    fn attention_view<'a>(
        &'a self,
        names: &'a Glm52AttentionWeightNames,
    ) -> Result<Glm52AttentionGpuWeights<'a>> {
        Ok(Glm52AttentionGpuWeights {
            input_layernorm: self.expect_tensor(&names.input_layernorm)?,
            q_a_proj: self.fp8_projection_view(&names.q_a_proj)?,
            q_a_layernorm: self.expect_tensor(&names.q_a_layernorm)?,
            q_b_proj: self.fp8_projection_view(&names.q_b_proj)?,
            kv_a_proj_with_mqa: self.fp8_projection_view(&names.kv_a_proj_with_mqa)?,
            kv_a_layernorm: self.expect_tensor(&names.kv_a_layernorm)?,
            kv_b_proj: self.fp8_projection_view(&names.kv_b_proj)?,
            o_proj: self.fp8_projection_view(&names.o_proj)?,
            post_attention_layernorm: self.expect_tensor(&names.post_attention_layernorm)?,
            indexer: names
                .indexer
                .as_ref()
                .map(|indexer| self.indexer_view(indexer))
                .transpose()?,
        })
    }

    fn indexer_view<'a>(
        &'a self,
        names: &'a Glm52IndexerWeightNames,
    ) -> Result<Glm52IndexerGpuWeights<'a>> {
        Ok(Glm52IndexerGpuWeights {
            k_norm_weight: self.expect_tensor(&names.k_norm_weight)?,
            k_norm_bias: self.expect_tensor(&names.k_norm_bias)?,
            weights_proj: self.expect_tensor(&names.weights_proj)?,
            wk: self.fp8_projection_view(&names.wk)?,
            wq_b: self.fp8_projection_view(&names.wq_b)?,
        })
    }

    fn dense_mlp_view<'a>(
        &'a self,
        names: &'a Glm52DenseMlpWeightNames,
    ) -> Result<Glm52DenseMlpGpuWeights<'a>> {
        Ok(Glm52DenseMlpGpuWeights {
            gate_proj: self.fp8_projection_view(&names.gate_proj)?,
            up_proj: self.fp8_projection_view(&names.up_proj)?,
            down_proj: self.fp8_projection_view(&names.down_proj)?,
        })
    }

    fn fp8_projection_view<'a>(
        &'a self,
        names: &'a Glm52Fp8ProjectionWeightNames,
    ) -> Result<Glm52Fp8ProjectionGpuWeights<'a>> {
        Ok(Glm52Fp8ProjectionGpuWeights {
            weight: self.expect_tensor(&names.weight)?,
            weight_scale_inv: self.expect_tensor(&names.weight_scale_inv)?,
        })
    }

    fn expect_tensor(&self, name: &str) -> Result<&Glm52GpuRawTensor> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("missing GLM5.2 GPU tensor {name}"))?;
        let contract = expected_tensor_contract(name)?;
        ensure!(
            tensor.dtype == contract.dtype,
            "GLM5.2 GPU tensor {name} dtype {:?} does not match expected {:?}",
            tensor.dtype,
            contract.dtype
        );
        ensure!(
            tensor.shape == contract.shape,
            "GLM5.2 GPU tensor {name} shape {:?} does not match expected {:?}",
            tensor.shape,
            contract.shape
        );
        Ok(tensor)
    }
}

impl Glm52AttentionGpuWeights<'_> {
    fn summary(&self) -> Result<TensorSummary> {
        let mut out = TensorSummary::default();
        out.add(self.input_layernorm);
        out.add_projection(&self.q_a_proj)?;
        out.add(self.q_a_layernorm);
        out.add_projection(&self.q_b_proj)?;
        out.add_projection(&self.kv_a_proj_with_mqa)?;
        out.add(self.kv_a_layernorm);
        out.add_projection(&self.kv_b_proj)?;
        out.add_projection(&self.o_proj)?;
        out.add(self.post_attention_layernorm);
        if let Some(indexer) = &self.indexer {
            out.add(indexer.k_norm_weight);
            out.add(indexer.k_norm_bias);
            out.add(indexer.weights_proj);
            out.add_projection(&indexer.wk)?;
            out.add_projection(&indexer.wq_b)?;
        }
        Ok(out)
    }
}

impl Glm52DenseMlpGpuWeights<'_> {
    fn summary(&self) -> Result<TensorSummary> {
        let mut out = TensorSummary::default();
        out.add_projection(&self.gate_proj)?;
        out.add_projection(&self.up_proj)?;
        out.add_projection(&self.down_proj)?;
        Ok(out)
    }
}

impl Glm52SharedExpertGpuWeights<'_> {
    fn summary(&self) -> Result<TensorSummary> {
        let mut out = TensorSummary::default();
        out.add_projection(&self.gate_proj)?;
        out.add_projection(&self.up_proj)?;
        out.add_projection(&self.down_proj)?;
        Ok(out)
    }
}

#[derive(Default)]
struct TensorSummary {
    tensor_count: usize,
    total_bytes: usize,
    fp8_projection_count: usize,
    fp8_weight_bytes: usize,
    fp8_scale_bytes: usize,
    max_out_dim: usize,
    max_in_dim: usize,
    max_scale_rows: usize,
    max_scale_cols: usize,
}

impl TensorSummary {
    fn add(&mut self, tensor: &Glm52GpuRawTensor) {
        self.tensor_count += 1;
        self.total_bytes += tensor.bytes;
    }

    fn add_projection(&mut self, projection: &Glm52Fp8ProjectionGpuWeights<'_>) -> Result<()> {
        ensure!(
            projection.weight.dtype == Dtype::F8_E4M3,
            "GLM5.2 FP8 projection {} has weight dtype {:?}, expected F8_E4M3",
            projection.weight.name,
            projection.weight.dtype
        );
        ensure!(
            projection.weight_scale_inv.dtype == Dtype::F32,
            "GLM5.2 FP8 projection {} has scale dtype {:?}, expected F32",
            projection.weight_scale_inv.name,
            projection.weight_scale_inv.dtype
        );
        let [out_dim, in_dim] = projection.weight.shape.as_slice() else {
            anyhow::bail!(
                "GLM5.2 FP8 projection {} weight shape must be rank-2, got {:?}",
                projection.weight.name,
                projection.weight.shape
            );
        };
        let [scale_rows, scale_cols] = projection.weight_scale_inv.shape.as_slice() else {
            anyhow::bail!(
                "GLM5.2 FP8 projection {} scale shape must be rank-2, got {:?}",
                projection.weight_scale_inv.name,
                projection.weight_scale_inv.shape
            );
        };
        let out_dim = *out_dim;
        let in_dim = *in_dim;
        let scale_rows = *scale_rows;
        let scale_cols = *scale_cols;
        let expected_scale_rows = out_dim.div_ceil(FP8_BLOCK_SIZE);
        let expected_scale_cols = in_dim.div_ceil(FP8_BLOCK_SIZE);
        ensure!(
            scale_rows == expected_scale_rows && scale_cols == expected_scale_cols,
            "GLM5.2 FP8 projection {} scale grid is {}x{}, expected {}x{} from weight {}x{}",
            projection.weight_scale_inv.name,
            scale_rows,
            scale_cols,
            expected_scale_rows,
            expected_scale_cols,
            out_dim,
            in_dim
        );
        self.add(projection.weight);
        self.add(projection.weight_scale_inv);
        self.fp8_projection_count += 1;
        self.fp8_weight_bytes += projection.weight.bytes;
        self.fp8_scale_bytes += projection.weight_scale_inv.bytes;
        self.max_out_dim = self.max_out_dim.max(out_dim);
        self.max_in_dim = self.max_in_dim.max(in_dim);
        self.max_scale_rows = self.max_scale_rows.max(scale_rows);
        self.max_scale_cols = self.max_scale_cols.max(scale_cols);
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.tensor_count += other.tensor_count;
        self.total_bytes += other.total_bytes;
        self.fp8_projection_count += other.fp8_projection_count;
        self.fp8_weight_bytes += other.fp8_weight_bytes;
        self.fp8_scale_bytes += other.fp8_scale_bytes;
        self.max_out_dim = self.max_out_dim.max(other.max_out_dim);
        self.max_in_dim = self.max_in_dim.max(other.max_in_dim);
        self.max_scale_rows = self.max_scale_rows.max(other.max_scale_rows);
        self.max_scale_cols = self.max_scale_cols.max(other.max_scale_cols);
    }
}

impl Glm52AttentionWeightNames {
    fn from_manifest(manifest: &Glm52AttentionManifest) -> Self {
        Self {
            input_layernorm: manifest.input_layernorm.name.clone(),
            q_a_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.q_a_proj),
            q_a_layernorm: manifest.q_a_layernorm.name.clone(),
            q_b_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.q_b_proj),
            kv_a_proj_with_mqa: Glm52Fp8ProjectionWeightNames::from_manifest(
                &manifest.kv_a_proj_with_mqa,
            ),
            kv_a_layernorm: manifest.kv_a_layernorm.name.clone(),
            kv_b_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.kv_b_proj),
            o_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.o_proj),
            post_attention_layernorm: manifest.post_attention_layernorm.name.clone(),
            indexer: manifest
                .indexer
                .as_ref()
                .map(Glm52IndexerWeightNames::from_manifest),
        }
    }
}

impl Glm52IndexerWeightNames {
    fn from_manifest(manifest: &Glm52IndexerManifest) -> Self {
        Self {
            k_norm_weight: manifest.k_norm_weight.name.clone(),
            k_norm_bias: manifest.k_norm_bias.name.clone(),
            weights_proj: manifest.weights_proj.name.clone(),
            wk: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.wk),
            wq_b: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.wq_b),
        }
    }
}

impl Glm52DenseMlpWeightNames {
    fn from_manifest(manifest: &Glm52DenseMlpManifest) -> Self {
        Self {
            gate_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.gate_proj),
            up_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.up_proj),
            down_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.down_proj),
        }
    }
}

impl Glm52RouterWeightNames {
    fn from_manifest(manifest: &Glm52RouterManifest) -> Self {
        Self {
            gate_weight: manifest.gate_weight.name.clone(),
            e_score_correction_bias: manifest.e_score_correction_bias.name.clone(),
        }
    }
}

impl Glm52SharedExpertWeightNames {
    fn from_manifest(manifest: &Glm52SharedExpertManifest) -> Self {
        Self {
            gate_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.gate_proj),
            up_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.up_proj),
            down_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.down_proj),
        }
    }
}

impl Glm52RoutedExpertWeightNames {
    fn from_manifest(manifest: &Glm52RoutedExpertManifest) -> Self {
        Self {
            global_expert: manifest.expert_idx,
            gate_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.gate_proj),
            up_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.up_proj),
            down_proj: Glm52Fp8ProjectionWeightNames::from_manifest(&manifest.down_proj),
        }
    }
}

impl Glm52Fp8ProjectionWeightNames {
    fn from_manifest(manifest: &Glm52Fp8ProjectionManifest) -> Self {
        Self {
            weight: manifest.weight.name.clone(),
            weight_scale_inv: manifest.weight_scale_inv.name.clone(),
        }
    }
}
