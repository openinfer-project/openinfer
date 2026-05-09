use anyhow::{Context, Result, ensure};
use safetensors::Dtype;

use crate::{
    config::Config,
    weights::{GpuRawTensor, RankWeights},
};

pub struct DeepSeekRankModel {
    config: Config,
    weights: RankWeights,
    top: TopLevelWeightNames,
    layers: Vec<BlockWeightNames>,
}

#[derive(Clone, Debug)]
pub struct TopLevelWeightNames {
    pub embed: String,
    pub head: String,
    pub norm: String,
    pub hc_head_fn: String,
    pub hc_head_base: String,
    pub hc_head_scale: String,
}

#[derive(Clone, Debug)]
pub struct BlockWeightNames {
    pub attn_norm: String,
    pub ffn_norm: String,
    pub hc_attn_fn: String,
    pub hc_attn_base: String,
    pub hc_attn_scale: String,
    pub hc_ffn_fn: String,
    pub hc_ffn_base: String,
    pub hc_ffn_scale: String,
    pub attn: AttentionWeightNames,
    pub ffn: FfnWeightNames,
}

#[derive(Clone, Debug)]
pub struct AttentionWeightNames {
    pub attn_sink: String,
    pub q_norm: String,
    pub kv_norm: String,
    pub wq_a: QuantLinearNames,
    pub wq_b: QuantLinearNames,
    pub wkv: QuantLinearNames,
    pub wo_a: String,
    pub wo_b: QuantLinearNames,
    pub compressor: Option<CompressorWeightNames>,
    pub indexer: Option<IndexerWeightNames>,
}

#[derive(Clone, Debug)]
pub struct CompressorWeightNames {
    pub ape: String,
    pub wkv: String,
    pub wgate: String,
    pub norm: String,
}

#[derive(Clone, Debug)]
pub struct IndexerWeightNames {
    pub wq_b: QuantLinearNames,
    pub weights_proj: String,
    pub compressor: CompressorWeightNames,
}

#[derive(Clone, Debug)]
pub struct FfnWeightNames {
    pub gate_weight: String,
    pub gate_bias: Option<String>,
    pub gate_tid2eid: Option<String>,
    pub shared_w1: QuantLinearNames,
    pub shared_w2: QuantLinearNames,
    pub shared_w3: QuantLinearNames,
    pub experts: Vec<ExpertWeightNames>,
}

#[derive(Clone, Debug)]
pub struct ExpertWeightNames {
    pub global_expert: usize,
    pub w1: QuantLinearNames,
    pub w2: QuantLinearNames,
    pub w3: QuantLinearNames,
}

#[derive(Clone, Debug)]
pub struct QuantLinearNames {
    pub weight: String,
    pub scale: String,
}

pub struct TensorRef<'a> {
    pub name: &'a str,
    pub tensor: &'a GpuRawTensor,
}

#[derive(Clone)]
pub struct RankWeightView<'a> {
    config: &'a Config,
    weights: &'a RankWeights,
}

pub struct AttentionWeights<'a> {
    pub attn_sink: TensorRef<'a>,
    pub q_norm: TensorRef<'a>,
    pub kv_norm: TensorRef<'a>,
    pub wq_a: QuantLinearRef<'a>,
    pub wq_b: QuantLinearRef<'a>,
    pub wkv: QuantLinearRef<'a>,
    pub wo_a: TensorRef<'a>,
    pub wo_b: QuantLinearRef<'a>,
    pub compressor: Option<CompressorWeights<'a>>,
    pub indexer: Option<IndexerWeights<'a>>,
}

pub struct CompressorWeights<'a> {
    pub ape: TensorRef<'a>,
    pub wkv: TensorRef<'a>,
    pub wgate: TensorRef<'a>,
    pub norm: TensorRef<'a>,
}

pub struct IndexerWeights<'a> {
    pub wq_b: QuantLinearRef<'a>,
    pub weights_proj: TensorRef<'a>,
    pub compressor: CompressorWeights<'a>,
}

pub struct FfnWeights<'a> {
    pub gate_weight: TensorRef<'a>,
    pub gate_bias: Option<TensorRef<'a>>,
    pub gate_tid2eid: Option<TensorRef<'a>>,
    pub shared_w1: QuantLinearRef<'a>,
    pub shared_w2: QuantLinearRef<'a>,
    pub shared_w3: QuantLinearRef<'a>,
}

pub struct ExpertWeights<'a> {
    pub w1: QuantLinearRef<'a>,
    pub w2: QuantLinearRef<'a>,
    pub w3: QuantLinearRef<'a>,
}

pub struct BlockWeights<'a> {
    pub attn_norm: TensorRef<'a>,
    pub ffn_norm: TensorRef<'a>,
    pub hc_attn_fn: TensorRef<'a>,
    pub hc_attn_base: TensorRef<'a>,
    pub hc_attn_scale: TensorRef<'a>,
    pub hc_ffn_fn: TensorRef<'a>,
    pub hc_ffn_base: TensorRef<'a>,
    pub hc_ffn_scale: TensorRef<'a>,
    pub attn: AttentionWeights<'a>,
    pub ffn: FfnWeights<'a>,
}

pub struct QuantLinearRef<'a> {
    pub weight: TensorRef<'a>,
    pub scale: TensorRef<'a>,
}

impl DeepSeekRankModel {
    pub fn new(config: Config, weights: RankWeights) -> Result<Self> {
        let view = weights.view(&config)?;
        let top = TopLevelWeightNames {
            embed: view.embed()?.name.to_string(),
            head: view.head()?.name.to_string(),
            norm: view.norm()?.name.to_string(),
            hc_head_fn: view.hc_head_fn()?.name.to_string(),
            hc_head_base: view.hc_head_base()?.name.to_string(),
            hc_head_scale: view.hc_head_scale()?.name.to_string(),
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            let block = view
                .block(layer)
                .with_context(|| format!("validate layer {layer} rank-local weights"))?;
            let mut experts = Vec::with_capacity(view.local_experts());
            for local_expert in 0..view.local_experts() {
                let expert = view.local_expert(layer, local_expert).with_context(|| {
                    format!("validate layer {layer} local expert {local_expert}")
                })?;
                experts.push(ExpertWeightNames {
                    global_expert: view.rank() * view.local_experts() + local_expert,
                    w1: QuantLinearNames::from_ref(expert.w1),
                    w2: QuantLinearNames::from_ref(expert.w2),
                    w3: QuantLinearNames::from_ref(expert.w3),
                });
            }

            layers.push(BlockWeightNames {
                attn_norm: block.attn_norm.name.to_string(),
                ffn_norm: block.ffn_norm.name.to_string(),
                hc_attn_fn: block.hc_attn_fn.name.to_string(),
                hc_attn_base: block.hc_attn_base.name.to_string(),
                hc_attn_scale: block.hc_attn_scale.name.to_string(),
                hc_ffn_fn: block.hc_ffn_fn.name.to_string(),
                hc_ffn_base: block.hc_ffn_base.name.to_string(),
                hc_ffn_scale: block.hc_ffn_scale.name.to_string(),
                attn: AttentionWeightNames {
                    attn_sink: block.attn.attn_sink.name.to_string(),
                    q_norm: block.attn.q_norm.name.to_string(),
                    kv_norm: block.attn.kv_norm.name.to_string(),
                    wq_a: QuantLinearNames::from_ref(block.attn.wq_a),
                    wq_b: QuantLinearNames::from_ref(block.attn.wq_b),
                    wkv: QuantLinearNames::from_ref(block.attn.wkv),
                    wo_a: block.attn.wo_a.name.to_string(),
                    wo_b: QuantLinearNames::from_ref(block.attn.wo_b),
                    compressor: block
                        .attn
                        .compressor
                        .as_ref()
                        .map(CompressorWeightNames::from_ref),
                    indexer: block
                        .attn
                        .indexer
                        .as_ref()
                        .map(IndexerWeightNames::from_ref),
                },
                ffn: FfnWeightNames {
                    gate_weight: block.ffn.gate_weight.name.to_string(),
                    gate_bias: block.ffn.gate_bias.map(|tensor| tensor.name.to_string()),
                    gate_tid2eid: block.ffn.gate_tid2eid.map(|tensor| tensor.name.to_string()),
                    shared_w1: QuantLinearNames::from_ref(block.ffn.shared_w1),
                    shared_w2: QuantLinearNames::from_ref(block.ffn.shared_w2),
                    shared_w3: QuantLinearNames::from_ref(block.ffn.shared_w3),
                    experts,
                },
            });
        }

        drop(view);

        Ok(Self {
            config,
            weights,
            top,
            layers,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn weights(&self) -> &RankWeights {
        &self.weights
    }

    pub fn top(&self) -> &TopLevelWeightNames {
        &self.top
    }

    pub fn layers(&self) -> &[BlockWeightNames] {
        &self.layers
    }

    pub fn view(&self) -> Result<RankWeightView<'_>> {
        self.weights.view(&self.config)
    }
}

impl QuantLinearNames {
    fn from_ref(linear: QuantLinearRef<'_>) -> Self {
        Self::from_ref_ref(&linear)
    }

    fn from_ref_ref(linear: &QuantLinearRef<'_>) -> Self {
        Self {
            weight: linear.weight.name.to_string(),
            scale: linear.scale.name.to_string(),
        }
    }
}

impl RankWeights {
    pub fn view<'a>(&'a self, config: &'a Config) -> Result<RankWeightView<'a>> {
        ensure!(
            self.world_size == 8,
            "DeepSeek V4 mp8 view expects world_size=8"
        );
        let view = RankWeightView {
            config,
            weights: self,
        };
        view.embed()?;
        view.head()?;
        Ok(view)
    }
}

impl<'a> RankWeightView<'a> {
    pub fn rank(&self) -> usize {
        self.weights.rank
    }

    pub fn world_size(&self) -> usize {
        self.weights.world_size
    }

    pub fn embed(&self) -> Result<TensorRef<'a>> {
        self.tensor(
            "embed.weight",
            Dtype::BF16,
            &[self.local_vocab(), self.config.dim],
        )
    }

    pub fn head(&self) -> Result<TensorRef<'a>> {
        self.tensor(
            "head.weight",
            Dtype::BF16,
            &[self.local_vocab(), self.config.dim],
        )
    }

    pub fn norm(&self) -> Result<TensorRef<'a>> {
        self.tensor("norm.weight", Dtype::BF16, &[self.config.dim])
    }

    pub fn hc_head_fn(&self) -> Result<TensorRef<'a>> {
        self.tensor(
            "hc_head_fn",
            Dtype::F32,
            &[self.config.hc_mult, self.config.hc_mult * self.config.dim],
        )
    }

    pub fn hc_head_base(&self) -> Result<TensorRef<'a>> {
        self.tensor("hc_head_base", Dtype::F32, &[self.config.hc_mult])
    }

    pub fn hc_head_scale(&self) -> Result<TensorRef<'a>> {
        self.tensor("hc_head_scale", Dtype::F32, &[1])
    }

    pub fn attn_norm(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.tensor(
            &format!("layers.{layer}.attn_norm.weight"),
            Dtype::BF16,
            &[self.config.dim],
        )
    }

    pub fn ffn_norm(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.tensor(
            &format!("layers.{layer}.ffn_norm.weight"),
            Dtype::BF16,
            &[self.config.dim],
        )
    }

    pub fn hc_attn_fn(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.hc_fn(&format!("layers.{layer}.hc_attn_fn"))
    }

    pub fn hc_attn_base(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.hc_base(&format!("layers.{layer}.hc_attn_base"))
    }

    pub fn hc_attn_scale(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.hc_scale(&format!("layers.{layer}.hc_attn_scale"), 3)
    }

    pub fn hc_ffn_fn(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.hc_fn(&format!("layers.{layer}.hc_ffn_fn"))
    }

    pub fn hc_ffn_base(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.hc_base(&format!("layers.{layer}.hc_ffn_base"))
    }

    pub fn hc_ffn_scale(&self, layer: usize) -> Result<TensorRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.hc_scale(&format!("layers.{layer}.hc_ffn_scale"), 3)
    }

    pub fn attn_wq_a(&self, layer: usize) -> Result<QuantLinearRef<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        self.fp8_linear(
            &format!("layers.{layer}.attn.wq_a"),
            self.config.q_lora_rank,
            self.config.dim,
        )
    }

    pub fn block(&self, layer: usize) -> Result<BlockWeights<'a>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        Ok(BlockWeights {
            attn_norm: self.attn_norm(layer)?,
            ffn_norm: self.ffn_norm(layer)?,
            hc_attn_fn: self.hc_attn_fn(layer)?,
            hc_attn_base: self.hc_attn_base(layer)?,
            hc_attn_scale: self.hc_attn_scale(layer)?,
            hc_ffn_fn: self.hc_ffn_fn(layer)?,
            hc_ffn_base: self.hc_ffn_base(layer)?,
            hc_ffn_scale: self.hc_ffn_scale(layer)?,
            attn: self.attention(layer)?,
            ffn: self.ffn(layer)?,
        })
    }

    pub fn attention(&self, layer: usize) -> Result<AttentionWeights<'a>> {
        let prefix = format!("layers.{layer}.attn");
        Ok(AttentionWeights {
            attn_sink: self.tensor(
                &format!("{prefix}.attn_sink"),
                Dtype::F32,
                &[self.local_heads()],
            )?,
            q_norm: self.tensor(
                &format!("{prefix}.q_norm.weight"),
                Dtype::BF16,
                &[self.config.q_lora_rank],
            )?,
            kv_norm: self.tensor(
                &format!("{prefix}.kv_norm.weight"),
                Dtype::BF16,
                &[self.config.head_dim],
            )?,
            wq_a: self.attn_wq_a(layer)?,
            wq_b: self.fp8_linear(
                &format!("{prefix}.wq_b"),
                self.local_heads() * self.config.head_dim,
                self.config.q_lora_rank,
            )?,
            wkv: self.fp8_linear(
                &format!("{prefix}.wkv"),
                self.config.head_dim,
                self.config.dim,
            )?,
            wo_a: self.tensor(
                &format!("{prefix}.wo_a.weight"),
                Dtype::BF16,
                &[
                    self.local_groups() * self.config.o_lora_rank,
                    self.local_heads() * self.config.head_dim / self.local_groups(),
                ],
            )?,
            wo_b: self.fp8_linear(
                &format!("{prefix}.wo_b"),
                self.config.dim,
                self.local_groups() * self.config.o_lora_rank,
            )?,
            compressor: self.compressor(layer)?,
            indexer: self.indexer(layer)?,
        })
    }

    pub fn compressor(&self, layer: usize) -> Result<Option<CompressorWeights<'a>>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        let ratio = self.config.compress_ratios[layer];
        if ratio == 0 {
            return Ok(None);
        }
        self.compressor_with_prefix(
            &format!("layers.{layer}.attn.compressor"),
            ratio,
            self.config.head_dim,
        )
        .map(Some)
    }

    pub fn indexer(&self, layer: usize) -> Result<Option<IndexerWeights<'a>>> {
        ensure!(layer < self.config.n_layers, "layer {layer} out of range");
        let ratio = self.config.compress_ratios[layer];
        if ratio != 4 {
            return Ok(None);
        }
        Ok(Some(IndexerWeights {
            wq_b: self.fp8_linear(
                &format!("layers.{layer}.attn.indexer.wq_b"),
                self.local_index_heads() * self.config.index_head_dim,
                self.config.q_lora_rank,
            )?,
            weights_proj: self.tensor(
                &format!("layers.{layer}.attn.indexer.weights_proj.weight"),
                Dtype::BF16,
                &[self.local_index_heads(), self.config.dim],
            )?,
            compressor: self.compressor_with_prefix(
                &format!("layers.{layer}.attn.indexer.compressor"),
                ratio,
                self.config.index_head_dim,
            )?,
        }))
    }

    pub fn ffn(&self, layer: usize) -> Result<FfnWeights<'a>> {
        let prefix = format!("layers.{layer}.ffn");
        Ok(FfnWeights {
            gate_weight: self.tensor(
                &format!("{prefix}.gate.weight"),
                Dtype::BF16,
                &[self.config.n_routed_experts, self.config.dim],
            )?,
            gate_bias: if layer < self.config.n_hash_layers {
                None
            } else {
                Some(self.tensor(
                    &format!("{prefix}.gate.bias"),
                    Dtype::F32,
                    &[self.config.n_routed_experts],
                )?)
            },
            gate_tid2eid: if layer < self.config.n_hash_layers {
                Some(self.tensor(
                    &format!("{prefix}.gate.tid2eid"),
                    Dtype::I64,
                    &[self.config.vocab_size, self.config.n_activated_experts],
                )?)
            } else {
                None
            },
            shared_w1: self.fp8_linear(
                &format!("{prefix}.shared_experts.w1"),
                self.config.moe_inter_dim,
                self.config.dim,
            )?,
            shared_w2: self.fp8_linear(
                &format!("{prefix}.shared_experts.w2"),
                self.config.dim,
                self.config.moe_inter_dim,
            )?,
            shared_w3: self.fp8_linear(
                &format!("{prefix}.shared_experts.w3"),
                self.config.moe_inter_dim,
                self.config.dim,
            )?,
        })
    }

    pub fn local_expert(&self, layer: usize, local_expert: usize) -> Result<ExpertWeights<'a>> {
        ensure!(
            local_expert < self.local_experts(),
            "local_expert {} out of range {}",
            local_expert,
            self.local_experts()
        );
        Ok(ExpertWeights {
            w1: self.local_expert_w1(layer, local_expert)?,
            w2: self.local_expert_w2(layer, local_expert)?,
            w3: self.local_expert_w3(layer, local_expert)?,
        })
    }

    pub fn local_expert_w1(&self, layer: usize, local_expert: usize) -> Result<QuantLinearRef<'a>> {
        self.local_expert_fp4_linear(
            layer,
            local_expert,
            "w1",
            self.config.moe_inter_dim,
            self.config.dim,
        )
    }

    pub fn local_expert_w2(&self, layer: usize, local_expert: usize) -> Result<QuantLinearRef<'a>> {
        self.local_expert_fp4_linear(
            layer,
            local_expert,
            "w2",
            self.config.dim,
            self.config.moe_inter_dim,
        )
    }

    pub fn local_expert_w3(&self, layer: usize, local_expert: usize) -> Result<QuantLinearRef<'a>> {
        self.local_expert_fp4_linear(
            layer,
            local_expert,
            "w3",
            self.config.moe_inter_dim,
            self.config.dim,
        )
    }

    fn local_expert_fp4_linear(
        &self,
        layer: usize,
        local_expert: usize,
        name: &str,
        out: usize,
        input: usize,
    ) -> Result<QuantLinearRef<'a>> {
        ensure!(
            local_expert < self.local_experts(),
            "local_expert {} out of range {}",
            local_expert,
            self.local_experts()
        );
        let global_expert = self.weights.rank * self.local_experts() + local_expert;
        self.fp4_linear(
            &format!("layers.{layer}.ffn.experts.{global_expert}.{name}"),
            out,
            input,
        )
    }

    fn fp8_linear(&self, prefix: &str, out: usize, input: usize) -> Result<QuantLinearRef<'a>> {
        Ok(QuantLinearRef {
            weight: self.tensor(&format!("{prefix}.weight"), Dtype::F8_E4M3, &[out, input])?,
            scale: self.tensor(
                &format!("{prefix}.scale"),
                Dtype::F8_E8M0,
                &[out.div_ceil(128), input.div_ceil(128)],
            )?,
        })
    }

    fn fp4_linear(
        &self,
        prefix: &str,
        out: usize,
        logical_input: usize,
    ) -> Result<QuantLinearRef<'a>> {
        Ok(QuantLinearRef {
            weight: self.tensor(
                &format!("{prefix}.weight"),
                Dtype::F4,
                &[out, logical_input],
            )?,
            scale: self.tensor(
                &format!("{prefix}.scale"),
                Dtype::F8_E8M0,
                &[out, logical_input / 32],
            )?,
        })
    }

    fn compressor_with_prefix(
        &self,
        prefix: &str,
        ratio: usize,
        head_dim: usize,
    ) -> Result<CompressorWeights<'a>> {
        ensure!(ratio > 0, "compressor ratio must be positive");
        let coff = if ratio == 4 { 2 } else { 1 };
        let out_dim = coff * head_dim;
        Ok(CompressorWeights {
            ape: self.tensor(&format!("{prefix}.ape"), Dtype::F32, &[ratio, out_dim])?,
            wkv: self.tensor(
                &format!("{prefix}.wkv.weight"),
                Dtype::BF16,
                &[out_dim, self.config.dim],
            )?,
            wgate: self.tensor(
                &format!("{prefix}.wgate.weight"),
                Dtype::BF16,
                &[out_dim, self.config.dim],
            )?,
            norm: self.tensor(&format!("{prefix}.norm.weight"), Dtype::BF16, &[head_dim])?,
        })
    }

    fn tensor(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<TensorRef<'a>> {
        let tensor = self
            .weights
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing GPU tensor {name}"))?;
        ensure!(
            tensor.dtype == dtype,
            "GPU tensor {name} dtype mismatch: expected {:?}, got {:?}",
            dtype,
            tensor.dtype
        );
        ensure!(
            tensor.shape == shape,
            "GPU tensor {name} shape mismatch: expected {:?}, got {:?}",
            shape,
            tensor.shape
        );
        Ok(TensorRef {
            name: &tensor.name,
            tensor,
        })
    }

    fn hc_fn(&self, name: &str) -> Result<TensorRef<'a>> {
        let hc_mix = (2 + self.config.hc_mult) * self.config.hc_mult;
        let hc_dim = self.config.hc_mult * self.config.dim;
        self.tensor(name, Dtype::F32, &[hc_mix, hc_dim])
    }

    fn hc_base(&self, name: &str) -> Result<TensorRef<'a>> {
        let hc_mix = (2 + self.config.hc_mult) * self.config.hc_mult;
        self.tensor(name, Dtype::F32, &[hc_mix])
    }

    fn hc_scale(&self, name: &str, len: usize) -> Result<TensorRef<'a>> {
        self.tensor(name, Dtype::F32, &[len])
    }

    fn local_vocab(&self) -> usize {
        self.config.vocab_size / self.weights.world_size
    }

    fn local_heads(&self) -> usize {
        self.config.num_attention_heads / self.weights.world_size
    }

    fn local_groups(&self) -> usize {
        self.config.o_groups / self.weights.world_size
    }

    fn local_index_heads(&self) -> usize {
        self.config.index_n_heads / self.weights.world_size
    }

    pub fn local_experts(&self) -> usize {
        self.config.n_routed_experts / self.weights.world_size
    }
}

impl CompressorWeightNames {
    fn from_ref(compressor: &CompressorWeights<'_>) -> Self {
        Self {
            ape: compressor.ape.name.to_string(),
            wkv: compressor.wkv.name.to_string(),
            wgate: compressor.wgate.name.to_string(),
            norm: compressor.norm.name.to_string(),
        }
    }
}

impl IndexerWeightNames {
    fn from_ref(indexer: &IndexerWeights<'_>) -> Self {
        Self {
            wq_b: QuantLinearNames::from_ref_ref(&indexer.wq_b),
            weights_proj: indexer.weights_proj.name.to_string(),
            compressor: CompressorWeightNames::from_ref(&indexer.compressor),
        }
    }
}
