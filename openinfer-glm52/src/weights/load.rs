use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use bytesize::ByteSize;
use cudarc::driver::CudaSlice;
use log::debug;
use safetensors::Dtype;

use super::{
    GLM52_MOE_LAYERS, Glm52NonExpertWeightContractReport, Glm52RankExpertFp8Weights,
    Glm52RankGpuContext, Glm52RankLoadBundle, Glm52TensorLoadSlice, expected_tensor_contract,
    mmap_file,
};

pub(crate) struct Glm52GpuRawTensor {
    pub(crate) name: String,
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) bytes: usize,
    pub(crate) data: CudaSlice<u8>,
}

pub(crate) struct Glm52RankGpuWeights {
    pub(crate) rank: usize,
    pub(crate) tensors: BTreeMap<String, Glm52GpuRawTensor>,
    pub(crate) total_bytes: usize,
}

pub(crate) struct Glm52RankSlicedLoadOutput {
    pub(crate) weights: Glm52RankGpuWeights,
    pub(crate) expert_kernel_weights: Glm52RankExpertFp8Weights,
    pub(crate) non_expert_weight_contract: Glm52NonExpertWeightContractReport,
    pub(crate) loaded_tensor_count: usize,
    pub(crate) loaded_total_bytes: usize,
}

pub(crate) fn load_rank_sliced_weights_to_gpu(
    ctx: &Glm52RankGpuContext,
    model_path: &Path,
    bundle: &Glm52RankLoadBundle,
) -> Result<Glm52RankSlicedLoadOutput> {
    ctx.set_current()?;
    ensure!(
        bundle.plan.tensor_count == bundle.load_plan.tensor_count,
        "GLM5.2 rank {} tensor plan {} disagrees with load plan {}",
        bundle.load_plan.rank,
        bundle.plan.tensor_count,
        bundle.load_plan.tensor_count
    );

    let mut weights = Glm52RankGpuWeights {
        rank: bundle.load_plan.rank,
        tensors: BTreeMap::new(),
        total_bytes: 0,
    };
    let mut packed_moe_layers = BTreeSet::new();
    let mut expert_layers = Vec::with_capacity(GLM52_MOE_LAYERS);
    let mut loaded_tensor_count = 0usize;
    let mut loaded_total_bytes = 0usize;
    let load_started = Instant::now();
    let mut slowest_shard: Option<(String, f64)> = None;
    debug!(
        "GLM5.2 rank {} start weight load: tensors={}, shards={}",
        bundle.load_plan.rank,
        bundle.load_plan.tensor_count,
        bundle.load_plan.shards.len()
    );

    for shard in &bundle.load_plan.shards {
        let path = model_path.join(&shard.shard);
        let shard_started = Instant::now();
        let mmap = mmap_file(&path)?;
        let safetensors = safetensors::SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for spec in &shard.tensors {
            ensure!(
                spec.slice == Glm52TensorLoadSlice::Full,
                "GLM5.2 first-cut TP1 loader only supports full tensor loads, got {:?} for {}",
                spec.slice,
                spec.name
            );
            let view = safetensors
                .tensor(&spec.name)
                .with_context(|| format!("missing tensor {} in {}", spec.name, path.display()))?;
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
            let data = ctx
                .stream()
                .clone_htod(view.data())
                .with_context(|| format!("failed to copy GLM5.2 tensor {} to GPU", spec.name))?;
            let tensor = Glm52GpuRawTensor {
                name: spec.name.clone(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                bytes: view.data().len(),
                data,
            };
            weights.total_bytes += tensor.bytes;
            loaded_total_bytes += tensor.bytes;
            loaded_tensor_count += 1;
            ensure!(
                weights.tensors.insert(spec.name.clone(), tensor).is_none(),
                "duplicate GLM5.2 tensor {} in rank {} load plan",
                spec.name,
                bundle.load_plan.rank
            );
        }
        weights.pack_loaded_expert_fp8_layers(
            ctx,
            &bundle.names,
            &mut packed_moe_layers,
            &mut expert_layers,
        )?;
        let shard_secs = shard_started.elapsed().as_secs_f64();
        match &slowest_shard {
            Some((_, slowest_secs)) if *slowest_secs >= shard_secs => {}
            _ => slowest_shard = Some((shard.shard.clone(), shard_secs)),
        }
    }

    ensure!(
        loaded_tensor_count == bundle.load_plan.tensor_count,
        "GLM5.2 rank {} loaded {} tensors but load plan has {}",
        bundle.load_plan.rank,
        loaded_tensor_count,
        bundle.load_plan.tensor_count
    );
    ensure!(
        expert_layers.len() == GLM52_MOE_LAYERS,
        "GLM5.2 rank {} expected {GLM52_MOE_LAYERS} streamed MoE FP8 expert packages, got {}",
        bundle.load_plan.rank,
        expert_layers.len()
    );
    expert_layers.sort_by_key(|layer| layer.layer_idx);
    let expert_kernel_total_bytes = expert_layers.iter().map(|layer| layer.total_bytes).sum();
    let expert_kernel_weights = Glm52RankExpertFp8Weights {
        rank: bundle.load_plan.rank,
        local_expert_range: bundle.names.plan.local_expert_range.clone(),
        layers: expert_layers,
        total_bytes: expert_kernel_total_bytes,
    };
    expert_kernel_weights.validate()?;
    let non_expert_weight_contract = weights.validate_non_expert_weight_contract(&bundle.names)?;
    ctx.sync().with_context(|| {
        format!(
            "failed to finish GLM5.2 rank {} H2D tensor copies",
            bundle.load_plan.rank
        )
    })?;

    let (slowest_shard, slowest_secs) = slowest_shard.unwrap_or_else(|| ("none".to_owned(), 0.0));
    debug!(
        "GLM5.2 rank {} weight load cost {:.2}s: loaded_tensors={}, loaded_bytes={}, resident_non_expert_raw_bytes={}, expert_package_bytes={}, non_expert_fp8_projections={}, packed_moe_layers={}, slowest_shard={} {:.2}s",
        bundle.load_plan.rank,
        load_started.elapsed().as_secs_f64(),
        loaded_tensor_count,
        ByteSize(loaded_total_bytes as u64),
        ByteSize(weights.total_bytes as u64),
        ByteSize(expert_kernel_weights.total_bytes as u64),
        non_expert_weight_contract.total_fp8_projections,
        packed_moe_layers.len(),
        slowest_shard,
        slowest_secs
    );

    Ok(Glm52RankSlicedLoadOutput {
        weights,
        expert_kernel_weights,
        non_expert_weight_contract,
        loaded_tensor_count,
        loaded_total_bytes,
    })
}
