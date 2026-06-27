use anyhow::{Context, Result, ensure};
use safetensors::Dtype;

use super::load::Glm52GpuRawTensor;
use super::{
    FP8_BLOCK_SIZE, GLM52_DENSE_LAYERS, GLM52_KV_B_OUT, GLM52_O_PROJ_IN, Glm52AttentionManifest,
    Glm52DenseMlpManifest, Glm52Fp8ProjectionManifest, Glm52IndexerManifest,
    Glm52LayerKindManifest, Glm52RoutedExpertManifest, Glm52RouterManifest,
    Glm52SharedExpertManifest, Glm52StageGpuWeights, Glm52StageWeightPlan, Glm52WeightManifest,
    expected_tensor_contract,
};
use crate::pp::Glm52StagePlan;

/// Bookend tensor names. Each is present only on its owning stage: the token
/// embedding on stage 0, the final norm + lm_head on the last stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52TopWeightNames {
    pub(crate) token_embedding: Option<String>,
    pub(crate) final_norm: Option<String>,
    pub(crate) lm_head: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52Fp8ProjectionWeightNames {
    pub(crate) weight: String,
    pub(crate) weight_scale_inv: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52IndexerWeightNames {
    pub(crate) k_norm_weight: String,
    pub(crate) k_norm_bias: String,
    pub(crate) weights_proj: String,
    pub(crate) wk: Glm52Fp8ProjectionWeightNames,
    pub(crate) wq_b: Glm52Fp8ProjectionWeightNames,
}

impl Glm52IndexerWeightNames {
    /// All resident tensor names for this indexer (k-norm gamma/beta, the score
    /// projection, and the wk / wq_b fp8 projection weight+scale pairs).
    pub(crate) fn tensor_names(&self) -> [&str; 7] {
        [
            &self.k_norm_weight,
            &self.k_norm_bias,
            &self.weights_proj,
            &self.wk.weight,
            &self.wk.weight_scale_inv,
            &self.wq_b.weight,
            &self.wq_b.weight_scale_inv,
        ]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52AttentionWeightNames {
    pub(crate) input_layernorm: String,
    pub(crate) q_a_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) q_a_layernorm: String,
    pub(crate) q_b_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) kv_a_proj_with_mqa: Glm52Fp8ProjectionWeightNames,
    pub(crate) kv_a_layernorm: String,
    pub(crate) kv_b_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) o_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) post_attention_layernorm: String,
    pub(crate) indexer: Option<Glm52IndexerWeightNames>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DenseMlpWeightNames {
    pub(crate) gate_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) up_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) down_proj: Glm52Fp8ProjectionWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RouterWeightNames {
    pub(crate) gate_weight: String,
    pub(crate) e_score_correction_bias: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52SharedExpertWeightNames {
    pub(crate) gate_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) up_proj: Glm52Fp8ProjectionWeightNames,
    pub(crate) down_proj: Glm52Fp8ProjectionWeightNames,
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

/// Typed tensor names for one pipeline stage. `layers` holds only the stage's
/// own layers; bookends in `top` are present only on the owning stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StageWeightNames {
    pub(crate) stage: usize,
    pub(crate) plan: Glm52StageWeightPlan,
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
    pub(crate) stage: usize,
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
    pub(crate) fn stage_weight_names(
        &self,
        stage: &Glm52StagePlan,
    ) -> Result<Glm52StageWeightNames> {
        let plan = self.stage_plan(stage)?;
        let top = Glm52TopWeightNames {
            token_embedding: stage.owns_embed.then(|| self.token_embedding.name.clone()),
            final_norm: stage.owns_head.then(|| self.final_norm.name.clone()),
            lm_head: stage.owns_head.then(|| self.lm_head.name.clone()),
        };
        let mut layers = Vec::with_capacity(stage.layers.len());
        for layer in &self.layers {
            if !stage.layers.contains(&layer.layer_idx) {
                continue;
            }
            let attention = Glm52AttentionWeightNames::from_manifest(&layer.attention);
            let kind = match &layer.kind {
                Glm52LayerKindManifest::Dense(mlp) => {
                    Glm52LayerWeightKindNames::Dense(Glm52DenseMlpWeightNames::from_manifest(mlp))
                }
                Glm52LayerKindManifest::Moe(moe) => {
                    // EP1: every stage holds all routed experts for its MoE layers.
                    let routed_experts = moe
                        .routed_experts
                        .iter()
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
        Ok(Glm52StageWeightNames {
            stage: stage.stage,
            plan,
            top,
            layers,
        })
    }
}

impl Glm52StageGpuWeights {
    pub(crate) fn validate_non_expert_weight_contract(
        &self,
        names: &Glm52StageWeightNames,
    ) -> Result<Glm52NonExpertWeightContractReport> {
        ensure!(
            self.stage == names.stage,
            "GLM5.2 GPU stage {} does not match typed names stage {}",
            self.stage,
            names.stage
        );
        ensure!(
            names.layers.len() == names.plan.layers.len(),
            "GLM5.2 stage {} typed names expected {} resident layers, got {}",
            self.stage,
            names.plan.layers.len(),
            names.layers.len()
        );

        let mut summary = TensorSummary::default();
        let mut dense_layers = 0usize;
        let mut moe_layers = 0usize;
        let mut full_indexer_layers = 0usize;
        let mut attention_fp8_projections = 0usize;
        let mut dense_fp8_projections = 0usize;
        let mut shared_fp8_projections = 0usize;
        // Bookends are resolved only on their owning stage.
        for name in [
            names.top.token_embedding.as_deref(),
            names.top.final_norm.as_deref(),
            names.top.lm_head.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            summary.add(self.expect_tensor(name)?);
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

        // Dense layers are the contiguous prefix [0, GLM52_DENSE_LAYERS); for a
        // stage's contiguous layer range the expected dense/MoE split is exact.
        let stage_layers = names.plan.layers.len();
        let expected_dense = names
            .plan
            .layers
            .clone()
            .filter(|layer_idx| *layer_idx < GLM52_DENSE_LAYERS)
            .count();
        let expected_moe = stage_layers - expected_dense;
        ensure!(
            dense_layers == expected_dense && moe_layers == expected_moe,
            "GLM5.2 stage {} non-expert weight contract has dense/moe layer counts {}/{}, expected {expected_dense}/{expected_moe}",
            self.stage,
            dense_layers,
            moe_layers
        );
        // Every resident layer carries attention (5 FP8 projections); indexer
        // layers add wk + wq_b. Couple the GPU-resolved count to the indexer flag.
        let expected_attention_fp8 = stage_layers * 5 + full_indexer_layers * 2;
        let expected_dense_fp8 = dense_layers * 3;
        let expected_shared_fp8 = moe_layers * 3;
        ensure!(
            attention_fp8_projections == expected_attention_fp8,
            "GLM5.2 stage {} attention FP8 projection count {} != expected {} ({stage_layers} layers, {full_indexer_layers} indexer)",
            self.stage,
            attention_fp8_projections,
            expected_attention_fp8
        );
        ensure!(
            dense_fp8_projections == expected_dense_fp8,
            "GLM5.2 stage {} dense FP8 projection count {} != expected {}",
            self.stage,
            dense_fp8_projections,
            expected_dense_fp8
        );
        ensure!(
            shared_fp8_projections == expected_shared_fp8,
            "GLM5.2 stage {} shared-expert FP8 projection count {} != expected {}",
            self.stage,
            shared_fp8_projections,
            expected_shared_fp8
        );
        ensure!(
            summary.tensor_count == self.tensors.len(),
            "GLM5.2 stage {} non-expert weight contract covers {} tensors, resident map has {}",
            self.stage,
            summary.tensor_count,
            self.tensors.len()
        );
        ensure!(
            summary.total_bytes == self.total_bytes,
            "GLM5.2 stage {} non-expert weight contract covers {} bytes, resident map has {}",
            self.stage,
            summary.total_bytes,
            self.total_bytes
        );
        // Attention's kv_b_proj / o_proj are the largest non-expert FP8
        // projections and live on every stage, so these maxima hold per-stage.
        ensure!(
            summary.max_out_dim == GLM52_KV_B_OUT && summary.max_in_dim == GLM52_O_PROJ_IN,
            "GLM5.2 stage {} non-expert FP8 projection max_out/max_in are {}/{}, expected {}/{}",
            self.stage,
            summary.max_out_dim,
            summary.max_in_dim,
            GLM52_KV_B_OUT,
            GLM52_O_PROJ_IN
        );
        ensure!(
            summary.max_scale_rows == GLM52_KV_B_OUT.div_ceil(FP8_BLOCK_SIZE)
                && summary.max_scale_cols == GLM52_O_PROJ_IN.div_ceil(FP8_BLOCK_SIZE),
            "GLM5.2 stage {} non-expert FP8 scale max grid is {}x{}, expected {}x{}",
            self.stage,
            summary.max_scale_rows,
            summary.max_scale_cols,
            GLM52_KV_B_OUT.div_ceil(FP8_BLOCK_SIZE),
            GLM52_O_PROJ_IN.div_ceil(FP8_BLOCK_SIZE)
        );
        let total_fp8_projections =
            attention_fp8_projections + dense_fp8_projections + shared_fp8_projections;
        Ok(Glm52NonExpertWeightContractReport {
            stage: self.stage,
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
