use super::*;

pub fn ensure_text_only_model_index(model_path: &Path) -> Result<KimiK2WeightManifest> {
    let manifest = KimiK2WeightManifest::from_model_dir(model_path)?;
    if manifest.text_tensor_count == 0 {
        bail!("Kimi safetensors index contains no language_model tensors");
    }
    Ok(manifest)
}

pub fn load_rank_weight_headers(
    model_path: &Path,
    shard_plan: &KimiRankShardPlan,
) -> Result<KimiRankWeightHeaders> {
    let mut tensors = BTreeMap::new();
    let mut total_bytes = 0usize;
    for shard in &shard_plan.shards {
        let path = model_path.join(&shard.shard);
        let mmap = mmap_file(&path)?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for name in &shard.tensors {
            let view = safetensors
                .tensor(name)
                .with_context(|| format!("missing tensor {name} in {}", path.display()))?;
            let header = KimiTensorHeader {
                name: name.clone(),
                shard: shard.shard.clone(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                bytes: view.data().len(),
            };
            total_bytes += header.bytes;
            ensure!(
                tensors.insert(name.clone(), header).is_none(),
                "duplicate Kimi tensor {name} in rank {} shard plan",
                shard_plan.rank
            );
        }
    }
    ensure!(
        tensors.len() == shard_plan.tensor_count,
        "Kimi rank {} header count {} does not match shard plan {}",
        shard_plan.rank,
        tensors.len(),
        shard_plan.tensor_count
    );
    Ok(KimiRankWeightHeaders {
        rank: shard_plan.rank,
        tensors,
        total_bytes,
    })
}

pub fn load_rank_sliced_weight_headers(
    model_path: &Path,
    load_plan: &KimiRankSlicedLoadPlan,
) -> Result<KimiRankWeightHeaders> {
    let mut tensors = BTreeMap::new();
    let mut total_bytes = 0usize;
    for shard in &load_plan.shards {
        let path = model_path.join(&shard.shard);
        let mmap = mmap_file(&path)?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for spec in &shard.tensors {
            let view = safetensors
                .tensor(&spec.name)
                .with_context(|| format!("missing tensor {} in {}", spec.name, path.display()))?;
            let shape = spec.slice.local_shape(view.shape())?;
            let bytes = spec.slice.local_bytes(view.shape(), view.dtype())?;
            let header = KimiTensorHeader {
                name: spec.name.clone(),
                shard: shard.shard.clone(),
                dtype: view.dtype(),
                shape,
                bytes,
            };
            total_bytes += header.bytes;
            ensure!(
                tensors.insert(spec.name.clone(), header).is_none(),
                "duplicate Kimi tensor {} in rank {} sliced load plan",
                spec.name,
                load_plan.rank
            );
        }
    }
    ensure!(
        tensors.len() == load_plan.tensor_count,
        "Kimi rank {} sliced header count {} does not match load plan {}",
        load_plan.rank,
        tensors.len(),
        load_plan.tensor_count
    );
    Ok(KimiRankWeightHeaders {
        rank: load_plan.rank,
        tensors,
        total_bytes,
    })
}

pub fn load_rank_weights_to_gpu(
    ctx: &KimiRankGpuContext,
    model_path: &Path,
    shard_plan: &KimiRankShardPlan,
) -> Result<KimiRankGpuWeights> {
    ctx.set_current()?;
    let mut tensors = BTreeMap::new();
    let mut total_bytes = 0usize;
    for shard in &shard_plan.shards {
        let path = model_path.join(&shard.shard);
        let mmap = mmap_file(&path)?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for name in &shard.tensors {
            let view = safetensors
                .tensor(name)
                .with_context(|| format!("missing tensor {name} in {}", path.display()))?;
            let data = ctx
                .stream
                .clone_htod(view.data())
                .with_context(|| format!("failed to copy Kimi tensor {name} to GPU"))?;
            let tensor = KimiGpuRawTensor {
                name: name.clone(),
                shard: shard.shard.clone(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                bytes: view.data().len(),
                data,
            };
            total_bytes += tensor.bytes;
            ensure!(
                tensors.insert(name.clone(), tensor).is_none(),
                "duplicate Kimi tensor {name} in rank {} shard plan",
                shard_plan.rank
            );
        }
    }
    ensure!(
        tensors.len() == shard_plan.tensor_count,
        "Kimi rank {} GPU tensor count {} does not match shard plan {}",
        shard_plan.rank,
        tensors.len(),
        shard_plan.tensor_count
    );
    ctx.sync().with_context(|| {
        format!(
            "failed to finish Kimi rank {} GPU tensor copies",
            shard_plan.rank
        )
    })?;
    Ok(KimiRankGpuWeights {
        rank: shard_plan.rank,
        tensors,
        total_bytes,
    })
}

pub fn load_rank_sliced_weights_to_gpu(
    ctx: &KimiRankGpuContext,
    model_path: &Path,
    load_plan: &KimiRankSlicedLoadPlan,
) -> Result<KimiRankGpuWeights> {
    ctx.set_current()?;
    let mut tensors = BTreeMap::new();
    let mut total_bytes = 0usize;
    for shard in &load_plan.shards {
        let path = model_path.join(&shard.shard);
        let mmap = mmap_file(&path)?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for spec in &shard.tensors {
            let view = safetensors
                .tensor(&spec.name)
                .with_context(|| format!("missing tensor {} in {}", spec.name, path.display()))?;
            let shape = spec.slice.local_shape(view.shape())?;
            let bytes = spec.slice.local_bytes(view.shape(), view.dtype())?;
            let data = match spec.slice {
                KimiTensorLoadSlice::Full => ctx
                    .stream
                    .clone_htod(view.data())
                    .with_context(|| format!("failed to copy Kimi tensor {} to GPU", spec.name))?,
                _ => {
                    let sliced =
                        sliced_tensor_bytes(view.data(), view.shape(), view.dtype(), &spec.slice)
                            .with_context(|| format!("failed to slice Kimi tensor {}", spec.name))?;
                    ctx.stream.clone_htod(sliced.as_slice()).with_context(|| {
                        format!("failed to copy Kimi tensor {} to GPU", spec.name)
                    })?
                }
            };
            let tensor = KimiGpuRawTensor {
                name: spec.name.clone(),
                shard: shard.shard.clone(),
                dtype: view.dtype(),
                shape,
                bytes,
                data,
            };
            total_bytes += tensor.bytes;
            ensure!(
                tensors.insert(spec.name.clone(), tensor).is_none(),
                "duplicate Kimi tensor {} in rank {} sliced load plan",
                spec.name,
                load_plan.rank
            );
        }
    }
    ensure!(
        tensors.len() == load_plan.tensor_count,
        "Kimi rank {} sliced GPU tensor count {} does not match load plan {}",
        load_plan.rank,
        tensors.len(),
        load_plan.tensor_count
    );
    ctx.sync().with_context(|| {
        format!(
            "failed to finish Kimi rank {} sliced GPU tensor copies",
            load_plan.rank
        )
    })?;
    Ok(KimiRankGpuWeights {
        rank: load_plan.rank,
        tensors,
        total_bytes,
    })
}

pub(super) fn sliced_tensor_bytes(
    data: &[u8],
    shape: &[usize],
    dtype: Dtype,
    slice: &KimiTensorLoadSlice,
) -> Result<Vec<u8>> {
    let element_bytes = dtype_element_bytes(dtype)?;
    match *slice {
        KimiTensorLoadSlice::Full => Ok(data.to_vec()),
        KimiTensorLoadSlice::RowRange { start, end } => {
            ensure!(
                shape.len() == 2 && start <= end && end <= shape[0],
                "invalid row slice [{start}..{end}) for shape {:?}",
                shape
            );
            let row_bytes = shape[1] * element_bytes;
            let start_byte = start * row_bytes;
            let end_byte = end * row_bytes;
            ensure!(
                end_byte <= data.len(),
                "row slice byte range [{start_byte}..{end_byte}) exceeds tensor bytes {}",
                data.len()
            );
            Ok(data[start_byte..end_byte].to_vec())
        }
        KimiTensorLoadSlice::ColRange { start, end } => {
            ensure!(
                shape.len() == 2 && start <= end && end <= shape[1],
                "invalid col slice [{start}..{end}) for shape {:?}",
                shape
            );
            let rows = shape[0];
            let cols = shape[1];
            let row_bytes = cols * element_bytes;
            let local_cols = end - start;
            let local_row_bytes = local_cols * element_bytes;
            let mut out = vec![0u8; rows * local_row_bytes];
            for row in 0..rows {
                let src = row * row_bytes + start * element_bytes;
                let dst = row * local_row_bytes;
                out[dst..dst + local_row_bytes].copy_from_slice(&data[src..src + local_row_bytes]);
            }
            Ok(out)
        }
    }
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: checkpoint shards are opened read-only and the mapping is only
    // consumed while reading safetensors metadata or copying tensor bytes.
    unsafe { Mmap::map(&file) }.with_context(|| format!("failed to mmap {}", path.display()))
}
