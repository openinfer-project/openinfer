//! Safetensors weight loading and RoPE precomputation.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::info;
use log::warn;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::SafeTensors;

use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::DeviceVec;

mod staging;
use staging::WeightStager;

/// Load shard metadata. Returns (shard_file_paths, weight_map: tensor_name -> shard_index)
pub fn load_shard_info(model_path: &str) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let single_path = format!("{}/model.safetensors", model_path);
    if std::path::Path::new(&single_path).exists() {
        return Ok((vec![single_path], HashMap::new()));
    }

    let index_path = format!("{}/model.safetensors.index.json", model_path);
    let index_content = fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    let weight_map_json = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid index.json: missing weight_map"))?;

    let mut shard_files: Vec<String> = Vec::new();
    let mut file_to_idx: HashMap<String, usize> = HashMap::new();
    let mut weight_map: HashMap<String, usize> = HashMap::new();

    for (tensor_name, shard_file_val) in weight_map_json {
        let shard_file = shard_file_val.as_str().unwrap().to_string();
        let idx = if let Some(&idx) = file_to_idx.get(&shard_file) {
            idx
        } else {
            let idx = shard_files.len();
            shard_files.push(format!("{model_path}/{shard_file}"));
            file_to_idx.insert(shard_file, idx);
            idx
        };
        weight_map.insert(tensor_name.clone(), idx);
    }

    Ok((shard_files, weight_map))
}

/// Advisory parallel page-cache prefetch for a whole-checkpoint load;
/// the loader never depends on it. Dropping cancels and joins the workers, and
/// failures are aggregated into one warning with the first cause retained.
pub struct WeightPrefetch {
    cancel: Arc<AtomicBool>,
    stats: Arc<PrefetchStats>,
    unreadable_shards: usize,
    spawn_failures: usize,
    workers: Vec<std::thread::JoinHandle<()>>,
}

#[derive(Default)]
struct PrefetchStats {
    read_errors: AtomicUsize,
    first_error: Mutex<Option<String>>,
}

impl PrefetchStats {
    fn record_first_error(&self, message: impl FnOnce() -> String) {
        let mut slot = match self.first_error.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if slot.is_none() {
            *slot = Some(message());
        }
    }
}

impl WeightPrefetch {
    pub fn spawn(shard_paths: &[String]) -> Self {
        const CHUNK: u64 = 16 << 20;
        const THREADS: usize = 8;

        let stats = Arc::new(PrefetchStats::default());
        let mut files: Vec<(Arc<fs::File>, u64, String)> = Vec::new();
        let mut chunks: Vec<(usize, u64)> = Vec::new();
        let mut unreadable_shards = 0usize;
        for path in shard_paths {
            let meta = fs::File::open(path).and_then(|file| {
                let len = file.metadata()?.len();
                Ok((file, len))
            });
            match meta {
                Ok((file, len)) => {
                    let idx = files.len();
                    files.push((Arc::new(file), len, path.clone()));
                    chunks.extend((0..len).step_by(CHUNK as usize).map(|off| (idx, off)));
                }
                Err(err) => {
                    unreadable_shards += 1;
                    stats.record_first_error(|| format!("{path}: {err}"));
                }
            }
        }

        let cancel = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::new();
        let mut spawn_failures = 0usize;
        let threads = THREADS.min(chunks.len());
        if !chunks.is_empty() {
            let total_bytes: u64 = files.iter().map(|(_, len, _)| len).sum();
            let num_files = files.len();
            let files = Arc::new(files);
            let chunks = Arc::new(chunks);
            let next = Arc::new(AtomicUsize::new(0));
            for _ in 0..threads {
                let worker = {
                    let (files, chunks, next) = (files.clone(), chunks.clone(), next.clone());
                    let (cancel, stats) = (cancel.clone(), stats.clone());
                    std::thread::Builder::new()
                        .name("weight-prefetch".into())
                        .spawn(move || {
                            let mut buf = vec![0u8; CHUNK as usize];
                            while !cancel.load(Ordering::Relaxed) {
                                let i = next.fetch_add(1, Ordering::Relaxed);
                                let Some(&(file_idx, off)) = chunks.get(i) else {
                                    break;
                                };
                                let (file, len, path) = &files[file_idx];
                                let want = CHUNK.min(len - off) as usize;
                                if let Err(err) = file.read_exact_at(&mut buf[..want], off) {
                                    stats.read_errors.fetch_add(1, Ordering::Relaxed);
                                    stats.record_first_error(|| format!("{path}@{off}: {err}"));
                                }
                            }
                        })
                };
                match worker {
                    Ok(handle) => workers.push(handle),
                    Err(err) => {
                        spawn_failures += 1;
                        stats.record_first_error(|| format!("worker spawn: {err}"));
                    }
                }
            }
            if !workers.is_empty() {
                info!(
                    "Prefetching {num_files} weight shard(s) ({:.1} GB) on {} threads",
                    total_bytes as f64 / 1e9,
                    workers.len()
                );
            }
        }
        Self {
            cancel,
            stats,
            unreadable_shards,
            spawn_failures,
            workers,
        }
    }
}

impl Drop for WeightPrefetch {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        let mut panicked = 0usize;
        for worker in self.workers.drain(..) {
            if let Err(payload) = worker.join() {
                panicked += 1;
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                self.stats
                    .record_first_error(|| format!("worker panic: {message}"));
            }
        }
        let read_errors = self.stats.read_errors.load(Ordering::Relaxed);
        if self.unreadable_shards + self.spawn_failures + read_errors + panicked > 0 {
            let first = match self.stats.first_error.lock() {
                Ok(guard) => guard.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            };
            warn!(
                "weight prefetch incomplete: {} unreadable shard(s), {} worker spawn failure(s), {} chunk read error(s), {} panic(s); first error: {}",
                self.unreadable_shards,
                self.spawn_failures,
                read_errors,
                panicked,
                first.as_deref().unwrap_or("unknown")
            );
        }
    }
}

/// Memory-map shard files and return the mmaps.
///
/// Typically chained with [`deserialize_shards`] to get `SafeTensors` views:
/// ```ignore
/// let mmaps = mmap_shards(&paths)?;
/// let shards = deserialize_shards(&mmaps)?;
/// ```
pub fn mmap_shards(shard_paths: &[String]) -> Result<Vec<Mmap>> {
    let mmaps: Vec<Mmap> = shard_paths
        .iter()
        .map(|p| {
            let file = fs::File::open(p)?;
            // SAFETY: we keep the Mmap alive for the duration of model loading,
            // and the file is not modified concurrently.
            unsafe { Mmap::map(&file) }
        })
        .collect::<std::io::Result<_>>()?;

    let total_bytes: usize = mmaps.iter().map(|m| m.len()).sum();
    info!(
        "Memory-mapped {} shard(s) ({:.1} MB)",
        mmaps.len(),
        total_bytes as f64 / 1e6
    );
    Ok(mmaps)
}

/// Deserialize memory-mapped shard data into `SafeTensors` views.
pub fn deserialize_shards(mmaps: &[Mmap]) -> Result<Vec<SafeTensors<'_>>> {
    mmaps
        .iter()
        .map(|m| {
            SafeTensors::deserialize(m).map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))
        })
        .collect()
}

fn find_tensor<'a>(
    shards: &'a [SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        shards[idx]
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Failed to load tensor '{}': {}", name, e))
    } else {
        // Fallback: try all shards (single-file case)
        for shard in shards {
            if let Ok(t) = shard.tensor(name) {
                return Ok(t);
            }
        }
        Err(anyhow::anyhow!("Tensor '{}' not found in any shard", name))
    }
}

fn tensor_bf16_bytes<'d>(
    tensor: &safetensors::tensor::TensorView<'d>,
    name: &str,
) -> Result<&'d [u8]> {
    anyhow::ensure!(
        tensor.dtype() == Dtype::BF16,
        "Tensor '{name}': expected dtype BF16, got {:?}",
        tensor.dtype()
    );
    let data = tensor.data();
    anyhow::ensure!(
        data.len().is_multiple_of(std::mem::size_of::<bf16>()),
        "Tensor '{name}': {} bytes is not a whole number of bf16 elements",
        data.len()
    );
    Ok(data)
}

/// Aligned payloads borrow zero-copy; misaligned ones (legal in safetensors)
/// decode into an owned buffer, since a misaligned bf16 view is UB.
#[allow(clippy::cast_ptr_alignment)]
fn tensor_bf16_cow<'d>(
    tensor: &safetensors::tensor::TensorView<'d>,
    name: &str,
) -> Result<Cow<'d, [bf16]>> {
    let data = tensor_bf16_bytes(tensor, name)?;
    if (data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<bf16>()) {
        // SAFETY: alignment checked; any bit pattern is a valid bf16.
        Ok(Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().cast::<bf16>(),
                data.len() / std::mem::size_of::<bf16>(),
            )
        }))
    } else {
        Ok(Cow::Owned(
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16::from_bits(u16::from_le_bytes(b)))
                .collect(),
        ))
    }
}

/// One row-consecutive part of a fused matrix: `rows` rows starting at
/// `row_offset` of a source tensor that must have exactly `src_rows` rows.
pub struct FusedPart<'a> {
    pub name: &'a str,
    pub src_rows: usize,
    pub row_offset: usize,
    pub rows: usize,
}

/// Validating BF16 checkpoint loader backed by pinned staging: every method
/// checks dtype and config-derived dimensions at the load boundary; payload
/// alignment is not required.
pub struct StagedWeightLoader<'a> {
    ctx: &'a DeviceContext,
    stager: WeightStager,
    shards: &'a [SafeTensors<'a>],
    weight_map: &'a HashMap<String, usize>,
}

impl<'a> StagedWeightLoader<'a> {
    pub fn new(
        ctx: &'a DeviceContext,
        shards: &'a [SafeTensors<'a>],
        weight_map: &'a HashMap<String, usize>,
    ) -> Result<Self> {
        Ok(Self {
            ctx,
            stager: WeightStager::new(ctx)?,
            shards,
            weight_map,
        })
    }

    fn tensor_2d(&self, name: &str, rows: usize, cols: usize) -> Result<&'a [u8]> {
        let tensor = find_tensor(self.shards, self.weight_map, name)?;
        let shape = tensor.shape();
        anyhow::ensure!(
            shape.len() == 2,
            "Tensor '{name}' expected 2D, got shape {shape:?}"
        );
        anyhow::ensure!(
            shape[0] == rows && shape[1] == cols,
            "Tensor '{name}' has shape {shape:?}, config expects [{rows}, {cols}]"
        );
        tensor_bf16_bytes(&tensor, name)
    }

    pub fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<DeviceMatrix> {
        let src = self.tensor_2d(name, rows, cols)?;
        // SAFETY: fully overwritten by the staged upload below.
        let mut data = unsafe { self.ctx.stream.alloc::<bf16>(rows * cols) }
            .map_err(|e| anyhow::anyhow!("Alloc failed for '{name}': {e}"))?;
        self.stager.upload(src, &mut data, 0)?;
        Ok(DeviceMatrix { data, rows, cols })
    }

    /// Row-concatenation of `parts`, each a validated row range of its
    /// source tensor; all sources must have exactly `cols` columns.
    pub fn fused_rows(&mut self, cols: usize, parts: &[FusedPart]) -> Result<DeviceMatrix> {
        anyhow::ensure!(!parts.is_empty(), "fused load needs at least one part");
        let mut total_rows = 0usize;
        let mut srcs = Vec::with_capacity(parts.len());
        for part in parts {
            let name = part.name;
            let full = self.tensor_2d(name, part.src_rows, cols)?;
            anyhow::ensure!(
                part.row_offset
                    .checked_add(part.rows)
                    .is_some_and(|end| end <= part.src_rows),
                "row range out of bounds for '{name}': row_offset={} rows={} total_rows={}",
                part.row_offset,
                part.rows,
                part.src_rows
            );
            let elem = std::mem::size_of::<bf16>();
            srcs.push(
                &full[part.row_offset * cols * elem..(part.row_offset + part.rows) * cols * elem],
            );
            total_rows += part.rows;
        }
        // SAFETY: every element is overwritten by the staged uploads below.
        let mut data =
            unsafe { self.ctx.stream.alloc::<bf16>(total_rows * cols) }.map_err(|e| {
                anyhow::anyhow!("Alloc failed for fused '{}' group: {e}", parts[0].name)
            })?;
        let mut dst_offset = 0;
        for src in srcs {
            self.stager.upload(src, &mut data, dst_offset)?;
            dst_offset += src.len() / std::mem::size_of::<bf16>();
        }
        Ok(DeviceMatrix {
            data,
            rows: total_rows,
            cols,
        })
    }

    /// `take` columns starting at `col_offset` of a source tensor validated
    /// to exactly `src_rows` x `src_cols` before anything is allocated.
    pub fn col_shard(
        &mut self,
        name: &str,
        src_rows: usize,
        src_cols: usize,
        col_offset: usize,
        take: usize,
    ) -> Result<DeviceMatrix> {
        let full = self.tensor_2d(name, src_rows, src_cols)?;
        anyhow::ensure!(
            col_offset
                .checked_add(take)
                .is_some_and(|end| end <= src_cols),
            "col range out of bounds for '{name}': col_offset={col_offset} take={take} total_cols={src_cols}"
        );
        // SAFETY: fully overwritten by the staged upload below.
        let mut data = unsafe { self.ctx.stream.alloc::<bf16>(src_rows * take) }
            .map_err(|e| anyhow::anyhow!("Alloc failed for '{name}': {e}"))?;
        self.stager
            .upload_cols(full, src_cols, col_offset, take, &mut data)?;
        Ok(DeviceMatrix {
            data,
            rows: src_rows,
            cols: take,
        })
    }

    /// Small tensors; plain pageable copy, no staging.
    pub fn vector(&mut self, name: &str, len: usize) -> Result<DeviceVec> {
        let tensor = find_tensor(self.shards, self.weight_map, name)?;
        let shape = tensor.shape();
        anyhow::ensure!(
            shape.len() == 1 && shape[0] == len,
            "Tensor '{name}' has shape {shape:?}, config expects [{len}]"
        );
        let src = tensor_bf16_cow(&tensor, name)?;
        DeviceVec::from_host(self.ctx, &src)
    }
}

pub fn load_tensor_1d(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceVec> {
    let tensor = find_tensor(shards, weight_map, name)?;
    DeviceVec::from_safetensors(ctx, tensor.data())
}

pub fn load_tensor_2d(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    DeviceMatrix::from_safetensors(ctx, tensor.data(), shape[0], shape[1])
}

pub fn load_tensor_2d_row_shard(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    row_offset: usize,
    rows: usize,
) -> Result<DeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(anyhow::anyhow!(
            "Tensor '{}' expected 2D, got shape {:?}",
            name,
            shape
        ));
    }
    let total_rows = shape[0];
    let cols = shape[1];
    if row_offset + rows > total_rows {
        return Err(anyhow::anyhow!(
            "2D row shard out of bounds for '{}': row_offset={} rows={} total_rows={}",
            name,
            row_offset,
            rows,
            total_rows
        ));
    }
    let elems = tensor_bf16_cow(&tensor, name)?;
    let start = row_offset * cols;
    let end = (row_offset + rows) * cols;
    DeviceMatrix::from_host(ctx, &elems[start..end], rows, cols)
}

fn gather_cols(
    elems: &[bf16],
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    take: usize,
) -> Vec<bf16> {
    let mut host = vec![bf16::ZERO; rows * take];
    for row in 0..rows {
        let src = row * total_cols + col_offset;
        let dst = row * take;
        host[dst..dst + take].copy_from_slice(&elems[src..src + take]);
    }
    host
}

pub fn load_tensor_2d_col_shard(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    col_offset: usize,
    cols: usize,
) -> Result<DeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(anyhow::anyhow!(
            "Tensor '{}' expected 2D, got shape {:?}",
            name,
            shape
        ));
    }
    let rows = shape[0];
    let total_cols = shape[1];
    if col_offset + cols > total_cols {
        return Err(anyhow::anyhow!(
            "2D col shard out of bounds for '{}': col_offset={} cols={} total_cols={}",
            name,
            col_offset,
            cols,
            total_cols
        ));
    }
    let elems = tensor_bf16_cow(&tensor, name)?;
    let host = gather_cols(&elems, rows, total_cols, col_offset, cols);
    DeviceMatrix::from_host(ctx, &host, rows, cols)
}

/// Precompute RoPE cos/sin cache as contiguous GPU buffers.
/// Layout: [max_seq_len * head_dim] — position `pos` at offset `pos * head_dim`.
pub fn precompute_rope(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(DeviceVec, DeviceVec)> {
    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();

    let total = max_seq_len * head_dim;
    let mut cos_host = vec![bf16::ZERO; total];
    let mut sin_host = vec![bf16::ZERO; total];

    for pos in 0..max_seq_len {
        let base = pos * head_dim;
        for i in 0..half_dim {
            let freq = pos as f32 * inv_freq[i];
            let cos_val = bf16::from_f32(freq.cos());
            let sin_val = bf16::from_f32(freq.sin());
            // Half-split layout: [cos(0)..cos(63), cos(0)..cos(63)]
            cos_host[base + i] = cos_val;
            cos_host[base + i + half_dim] = cos_val;
            sin_host[base + i] = sin_val;
            sin_host[base + i + half_dim] = sin_val;
        }
    }

    let cos_cache = DeviceVec::from_host(ctx, &cos_host)?;
    let sin_cache = DeviceVec::from_host(ctx, &sin_host)?;

    Ok((cos_cache, sin_cache))
}

#[allow(clippy::cast_ptr_alignment)]
/// Load a 1D F32 tensor to GPU as CudaSlice<f32>.
/// For weights stored in float32 (e.g., A_log, norm.weight in linear attention).
pub fn load_tensor_1d_f32(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<CudaSlice<f32>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let data = tensor.data();
    if data.len() % 4 != 0 {
        return Err(anyhow::anyhow!(
            "F32 tensor '{}': data length {} not multiple of 4",
            name,
            data.len()
        ));
    }
    let len = data.len() / 4;
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), len) };
    let gpu_data = ctx
        .stream
        .clone_htod(slice)
        .map_err(|e| anyhow::anyhow!("H2D copy failed for '{}': {}", name, e))?;
    Ok(gpu_data)
}

/// Load a 1D I64 tensor into a host `Vec<i64>`.
///
/// For small integer lookup tables that live on the host (e.g. EAGLE-3's `d2t`
/// draft→target vocab offset map), not weights destined for a GEMM.
pub fn load_tensor_i64_host(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<i64>> {
    let tensor = find_tensor(shards, weight_map, name)?;

    if tensor.dtype() != Dtype::I64 {
        return Err(anyhow::anyhow!(
            "I64 tensor '{}': expected dtype I64, got {:?}",
            name,
            tensor.dtype()
        ));
    }
    if tensor.shape().len() != 1 {
        return Err(anyhow::anyhow!(
            "I64 tensor '{}': expected 1D, got shape {:?}",
            name,
            tensor.shape()
        ));
    }
    let data = tensor.data();
    if data.len() % 8 != 0 {
        return Err(anyhow::anyhow!(
            "I64 tensor '{}': data length {} not multiple of 8",
            name,
            data.len()
        ));
    }
    Ok(data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&b| i64::from_le_bytes(b))
        .collect())
}

/// Load a 1D BOOL tensor into a host `Vec<bool>` (safetensors stores BOOL as one
/// byte per element, `0`/`1`). For mask tables like EAGLE-3's `t2d`.
pub fn load_tensor_bool_host(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<bool>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    if tensor.dtype() != Dtype::BOOL {
        return Err(anyhow::anyhow!(
            "BOOL tensor '{}': expected dtype BOOL, got {:?}",
            name,
            tensor.dtype()
        ));
    }
    if tensor.shape().len() != 1 {
        return Err(anyhow::anyhow!(
            "BOOL tensor '{}': expected 1D, got shape {:?}",
            name,
            tensor.shape()
        ));
    }
    Ok(tensor.data().iter().map(|&b| b != 0).collect())
}

/// Load shard info with fixup for mismatched shard filenames in index.json.
///
/// Some models (e.g., Qwen3.5) have index.json with shard filenames like
/// `model.safetensors-00001-of-00002.safetensors` while actual files are
/// `model-00001-of-00002.safetensors`. This function detects and fixes that.
pub fn load_shard_info_fixed(model_path: &str) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let (mut shard_files, weight_map) = load_shard_info(model_path)?;

    for path in &mut shard_files {
        if !std::path::Path::new(path).exists() {
            // Try replacing "model.safetensors-" with "model-" in filename
            let filename = std::path::Path::new(path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();
            if let Some(rest) = filename.strip_prefix("model.safetensors-") {
                let fixed = format!("{}/model-{}", model_path, rest);
                if std::path::Path::new(&fixed).exists() {
                    log::info!(
                        "Fixed shard path: {} -> {}",
                        filename,
                        std::path::Path::new(&fixed)
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                    );
                    *path = fixed;
                    continue;
                }
            }
            return Err(anyhow::anyhow!("Shard file not found: {}", path));
        }
    }

    Ok((shard_files, weight_map))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use safetensors::Dtype;
    use safetensors::tensor::TensorView;

    use super::tensor_bf16_cow;

    #[test]
    fn tensor_bf16_cow_borrows_aligned_and_decodes_unaligned() {
        let vals: [u16; 4] = [0x3f80, 0x0001, 0xbf12, 0x7fff];
        let mut bytes = vec![0u8; vals.len() * 2 + 3];
        // A Vec<u8> base has no alignment guarantee; derive both offsets from
        // the actual address so each branch is forced deterministically.
        let base = bytes.as_ptr() as usize;
        let aligned_off = base.next_multiple_of(2) - base;
        for (off, expect_borrowed) in [(aligned_off, true), (aligned_off + 1, false)] {
            for (i, v) in vals.iter().enumerate() {
                bytes[off + i * 2..off + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
            let view = TensorView::new(
                Dtype::BF16,
                vec![vals.len()],
                &bytes[off..off + vals.len() * 2],
            )
            .unwrap();
            let cow = tensor_bf16_cow(&view, "w").unwrap();
            assert_eq!(
                matches!(cow, Cow::Borrowed(_)),
                expect_borrowed,
                "off={off}"
            );
            let got: Vec<u16> = cow.iter().map(|b| b.to_bits()).collect();
            assert_eq!(got, vals, "off={off}");
        }
    }
}
