use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use memmap2::Mmap;
use pegainfer_kernels::ffi;
use safetensors::{Dtype, SafeTensors};
use std::sync::Arc;

use crate::config::{Config, TensorParallelConfig};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct RankManifest {
    pub rank: usize,
    pub world_size: usize,
    pub path: PathBuf,
    pub tensors: BTreeMap<String, TensorInfo>,
}

pub struct GpuRawTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub data: CudaSlice<u8>,
}

pub struct RankWeights {
    pub rank: usize,
    pub world_size: usize,
    pub tensors: BTreeMap<String, GpuRawTensor>,
    pub total_bytes: usize,
}

#[derive(Clone)]
pub struct RankGpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub device_ordinal: usize,
}

// SAFETY: A rank context is pinned to one CUDA device. The parallel runtime only
// drives each rank's context from one worker at a time; different ranks use
// distinct contexts/streams/devices.
unsafe impl Send for RankGpuContext {}
unsafe impl Sync for RankGpuContext {}

impl RankGpuContext {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        Self::set_current_device(device_ordinal)?;
        let ctx = CudaContext::new(device_ordinal).with_context(|| {
            format!("failed to create CUDA context for device {device_ordinal}")
        })?;
        unsafe {
            ctx.disable_event_tracking();
        }
        let stream = ctx
            .new_stream()
            .with_context(|| format!("failed to create CUDA stream for device {device_ordinal}"))?;
        Ok(Self {
            ctx,
            stream,
            device_ordinal,
        })
    }

    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .with_context(|| format!("failed to synchronize device {}", self.device_ordinal))
    }

    pub fn set_current(&self) -> Result<()> {
        Self::set_current_device(self.device_ordinal)?;
        self.ctx.bind_to_thread().with_context(|| {
            format!(
                "failed to bind CUDA context for device {} to current thread",
                self.device_ordinal
            )
        })?;
        Ok(())
    }

    fn set_current_device(device_ordinal: usize) -> Result<()> {
        let err = unsafe { ffi::cuda_set_device(device_ordinal as i32) };
        ensure!(
            err == 0,
            "failed to set CUDA device {device_ordinal}: cudaError={err}"
        );
        Ok(())
    }
}

impl RankManifest {
    pub fn require(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&TensorInfo> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name} in {}", self.path.display()))?;
        ensure!(
            info.dtype == dtype,
            "tensor {name} dtype mismatch: expected {:?}, got {:?}",
            dtype,
            info.dtype
        );
        ensure!(
            info.shape == shape,
            "tensor {name} shape mismatch: expected {:?}, got {:?}",
            shape,
            info.shape
        );
        Ok(info)
    }
}

pub fn mp_rank_path(model_path: impl AsRef<Path>, tp: TensorParallelConfig) -> PathBuf {
    model_path
        .as_ref()
        .join("mp8")
        .join(format!("model{}-mp8.safetensors", tp.rank))
}

pub fn load_rank_manifest(
    model_path: impl AsRef<Path>,
    config: &Config,
    tp: TensorParallelConfig,
) -> Result<RankManifest> {
    tp.validate_for(config)?;
    let path = mp_rank_path(model_path, tp);
    let mmap = mmap_file(&path)?;
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("failed to deserialize {}", path.display()))?;

    let mut infos = BTreeMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name)?;
        infos.insert(
            name.to_string(),
            TensorInfo {
                name: name.to_string(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                bytes: view.data().len(),
            },
        );
    }

    let manifest = RankManifest {
        rank: tp.rank,
        world_size: tp.world_size,
        path,
        tensors: infos,
    };
    validate_mp8_manifest(config, &manifest)?;
    Ok(manifest)
}

pub fn load_rank_subset_to_gpu(
    ctx: &RankGpuContext,
    model_path: impl AsRef<Path>,
    config: &Config,
    tp: TensorParallelConfig,
    tensor_names: &[&str],
) -> Result<BTreeMap<String, GpuRawTensor>> {
    ctx.set_current()?;
    tp.validate_for(config)?;
    let path = mp_rank_path(model_path, tp);
    let requested: BTreeSet<&str> = tensor_names.iter().copied().collect();
    let mmap = mmap_file(&path)?;
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("failed to deserialize {}", path.display()))?;

    let mut out = BTreeMap::new();
    for name in tensor_names {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("missing tensor {name} in {}", path.display()))?;
        let gpu_data = ctx
            .stream
            .clone_htod(view.data())
            .with_context(|| format!("failed to copy tensor {name} to GPU"))?;
        out.insert(
            (*name).to_string(),
            GpuRawTensor {
                name: (*name).to_string(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                bytes: view.data().len(),
                data: gpu_data,
            },
        );
    }

    for name in requested {
        ensure!(
            out.contains_key(name),
            "requested tensor {name} was not loaded"
        );
    }
    ctx.sync()
        .with_context(|| format!("failed to finish GPU tensor copies for {}", path.display()))?;
    Ok(out)
}

pub fn load_rank_to_gpu(
    ctx: &RankGpuContext,
    model_path: impl AsRef<Path>,
    config: &Config,
    tp: TensorParallelConfig,
) -> Result<RankWeights> {
    tp.validate_for(config)?;
    let manifest = load_rank_manifest(&model_path, config, tp)?;
    let tensor_names: Vec<&str> = manifest.tensors.keys().map(String::as_str).collect();
    let tensors = load_rank_subset_to_gpu(ctx, model_path, config, tp, &tensor_names)?;
    let total_bytes = tensors.values().map(|tensor| tensor.bytes).sum();
    Ok(RankWeights {
        rank: tp.rank,
        world_size: tp.world_size,
        tensors,
        total_bytes,
    })
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: the checkpoint file is kept read-only and the mmap is only used
    // while deserializing/copying tensor bytes.
    unsafe { Mmap::map(&file) }.with_context(|| format!("failed to mmap {}", path.display()))
}

fn validate_mp8_manifest(config: &Config, manifest: &RankManifest) -> Result<()> {
    let local_vocab = config.vocab_size / manifest.world_size;
    let local_heads = config.num_attention_heads / manifest.world_size;
    let local_groups = config.o_groups / manifest.world_size;
    let local_experts = config.n_routed_experts / manifest.world_size;

    manifest.require("embed.weight", Dtype::BF16, &[local_vocab, config.dim])?;
    manifest.require("head.weight", Dtype::BF16, &[local_vocab, config.dim])?;
    manifest.require("norm.weight", Dtype::BF16, &[config.dim])?;
    manifest.require(
        "hc_head_fn",
        Dtype::F32,
        &[config.hc_mult, config.hc_mult * config.dim],
    )?;
    manifest.require("hc_head_base", Dtype::F32, &[config.hc_mult])?;
    manifest.require("hc_head_scale", Dtype::F32, &[1])?;

    for layer in 0..config.n_layers {
        validate_layer(
            config,
            manifest,
            layer,
            local_heads,
            local_groups,
            local_experts,
        )?;
    }

    Ok(())
}

fn validate_layer(
    config: &Config,
    manifest: &RankManifest,
    layer: usize,
    local_heads: usize,
    local_groups: usize,
    local_experts: usize,
) -> Result<()> {
    let prefix = format!("layers.{layer}");
    let hc_mix = (2 + config.hc_mult) * config.hc_mult;
    let hc_dim = config.hc_mult * config.dim;

    manifest.require(
        &format!("{prefix}.attn_norm.weight"),
        Dtype::BF16,
        &[config.dim],
    )?;
    manifest.require(
        &format!("{prefix}.ffn_norm.weight"),
        Dtype::BF16,
        &[config.dim],
    )?;
    manifest.require(
        &format!("{prefix}.hc_attn_fn"),
        Dtype::F32,
        &[hc_mix, hc_dim],
    )?;
    manifest.require(&format!("{prefix}.hc_attn_base"), Dtype::F32, &[hc_mix])?;
    manifest.require(&format!("{prefix}.hc_attn_scale"), Dtype::F32, &[3])?;
    manifest.require(
        &format!("{prefix}.hc_ffn_fn"),
        Dtype::F32,
        &[hc_mix, hc_dim],
    )?;
    manifest.require(&format!("{prefix}.hc_ffn_base"), Dtype::F32, &[hc_mix])?;
    manifest.require(&format!("{prefix}.hc_ffn_scale"), Dtype::F32, &[3])?;

    let attn = format!("{prefix}.attn");
    manifest.require(&format!("{attn}.attn_sink"), Dtype::F32, &[local_heads])?;
    manifest.require(
        &format!("{attn}.q_norm.weight"),
        Dtype::BF16,
        &[config.q_lora_rank],
    )?;
    manifest.require(
        &format!("{attn}.kv_norm.weight"),
        Dtype::BF16,
        &[config.head_dim],
    )?;
    manifest.require(
        &format!("{attn}.wq_a.weight"),
        Dtype::F8_E4M3,
        &[config.q_lora_rank, config.dim],
    )?;
    manifest.require(
        &format!("{attn}.wq_a.scale"),
        Dtype::F8_E8M0,
        &[config.q_lora_rank / 128, config.dim / 128],
    )?;
    manifest.require(
        &format!("{attn}.wq_b.weight"),
        Dtype::F8_E4M3,
        &[local_heads * config.head_dim, config.q_lora_rank],
    )?;
    manifest.require(
        &format!("{attn}.wq_b.scale"),
        Dtype::F8_E8M0,
        &[
            (local_heads * config.head_dim) / 128,
            config.q_lora_rank / 128,
        ],
    )?;
    manifest.require(
        &format!("{attn}.wkv.weight"),
        Dtype::F8_E4M3,
        &[config.head_dim, config.dim],
    )?;
    manifest.require(
        &format!("{attn}.wkv.scale"),
        Dtype::F8_E8M0,
        &[config.head_dim / 128, config.dim / 128],
    )?;
    manifest.require(
        &format!("{attn}.wo_a.weight"),
        Dtype::BF16,
        &[
            local_groups * config.o_lora_rank,
            local_heads * config.head_dim / local_groups,
        ],
    )?;
    manifest.require(
        &format!("{attn}.wo_b.weight"),
        Dtype::F8_E4M3,
        &[config.dim, local_groups * config.o_lora_rank],
    )?;
    manifest.require(
        &format!("{attn}.wo_b.scale"),
        Dtype::F8_E8M0,
        &[config.dim / 128, (local_groups * config.o_lora_rank) / 128],
    )?;

    let ffn = format!("{prefix}.ffn");
    manifest.require(
        &format!("{ffn}.gate.weight"),
        Dtype::BF16,
        &[config.n_routed_experts, config.dim],
    )?;
    if layer < config.n_hash_layers {
        manifest.require(
            &format!("{ffn}.gate.tid2eid"),
            Dtype::I64,
            &[config.vocab_size, config.n_activated_experts],
        )?;
    } else {
        manifest.require(
            &format!("{ffn}.gate.bias"),
            Dtype::F32,
            &[config.n_routed_experts],
        )?;
    }
    validate_fp8_linear(
        manifest,
        &format!("{ffn}.shared_experts.w1"),
        config.moe_inter_dim,
        config.dim,
    )?;
    validate_fp8_linear(
        manifest,
        &format!("{ffn}.shared_experts.w2"),
        config.dim,
        config.moe_inter_dim,
    )?;
    validate_fp8_linear(
        manifest,
        &format!("{ffn}.shared_experts.w3"),
        config.moe_inter_dim,
        config.dim,
    )?;

    let local_start = manifest.rank * local_experts;
    for expert in local_start..local_start + local_experts {
        validate_fp4_linear(
            manifest,
            &format!("{ffn}.experts.{expert}.w1"),
            config.moe_inter_dim,
            config.dim,
        )?;
        validate_fp4_linear(
            manifest,
            &format!("{ffn}.experts.{expert}.w2"),
            config.dim,
            config.moe_inter_dim,
        )?;
        validate_fp4_linear(
            manifest,
            &format!("{ffn}.experts.{expert}.w3"),
            config.moe_inter_dim,
            config.dim,
        )?;
    }

    match config.compress_ratios.get(layer).copied() {
        Some(0) => {}
        Some(4) => validate_compressed_attention(config, manifest, layer, true)?,
        Some(128) => validate_compressed_attention(config, manifest, layer, false)?,
        Some(other) => bail!("unsupported compress ratio {other} at layer {layer}"),
        None => bail!("missing compress ratio for layer {layer}"),
    }

    Ok(())
}

fn validate_fp8_linear(
    manifest: &RankManifest,
    prefix: &str,
    out: usize,
    input: usize,
) -> Result<()> {
    manifest.require(&format!("{prefix}.weight"), Dtype::F8_E4M3, &[out, input])?;
    manifest.require(
        &format!("{prefix}.scale"),
        Dtype::F8_E8M0,
        &[out.div_ceil(128), input.div_ceil(128)],
    )?;
    Ok(())
}

fn validate_fp4_linear(
    manifest: &RankManifest,
    prefix: &str,
    out: usize,
    logical_input: usize,
) -> Result<()> {
    manifest.require(
        &format!("{prefix}.weight"),
        Dtype::F4,
        &[out, logical_input],
    )?;
    manifest.require(
        &format!("{prefix}.scale"),
        Dtype::F8_E8M0,
        &[out, logical_input / 32],
    )?;
    Ok(())
}

fn validate_compressed_attention(
    config: &Config,
    manifest: &RankManifest,
    layer: usize,
    has_indexer: bool,
) -> Result<()> {
    let prefix = format!("layers.{layer}.attn");
    let compress_ratio = config.compress_ratios[layer];
    let coff = if compress_ratio == 4 { 2 } else { 1 };
    let compressed_dim = coff * config.head_dim;

    manifest.require(
        &format!("{prefix}.compressor.ape"),
        Dtype::F32,
        &[compress_ratio, compressed_dim],
    )?;
    manifest.require(
        &format!("{prefix}.compressor.wkv.weight"),
        Dtype::BF16,
        &[compressed_dim, config.dim],
    )?;
    manifest.require(
        &format!("{prefix}.compressor.wgate.weight"),
        Dtype::BF16,
        &[compressed_dim, config.dim],
    )?;
    manifest.require(
        &format!("{prefix}.compressor.norm.weight"),
        Dtype::BF16,
        &[config.head_dim],
    )?;

    if has_indexer {
        let indexer = format!("{prefix}.indexer");
        manifest.require(
            &format!("{indexer}.wq_b.weight"),
            Dtype::F8_E4M3,
            &[
                config.index_n_heads / manifest.world_size * config.index_head_dim,
                config.q_lora_rank,
            ],
        )?;
        manifest.require(
            &format!("{indexer}.wq_b.scale"),
            Dtype::F8_E8M0,
            &[
                (config.index_n_heads / manifest.world_size * config.index_head_dim) / 128,
                config.q_lora_rank / 128,
            ],
        )?;
        manifest.require(
            &format!("{indexer}.weights_proj.weight"),
            Dtype::BF16,
            &[config.index_n_heads / manifest.world_size, config.dim],
        )?;
        manifest.require(
            &format!("{indexer}.compressor.ape"),
            Dtype::F32,
            &[compress_ratio, 2 * config.index_head_dim],
        )?;
        manifest.require(
            &format!("{indexer}.compressor.wkv.weight"),
            Dtype::BF16,
            &[2 * config.index_head_dim, config.dim],
        )?;
        manifest.require(
            &format!("{indexer}.compressor.wgate.weight"),
            Dtype::BF16,
            &[2 * config.index_head_dim, config.dim],
        )?;
        manifest.require(
            &format!("{indexer}.compressor.norm.weight"),
            Dtype::BF16,
            &[config.index_head_dim],
        )?;
    }

    Ok(())
}
