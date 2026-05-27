use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use pegainfer_core::ops;
use pegainfer_core::tensor::{DeviceContext, DeviceMatrix, HiddenStates};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::config::{Config, TensorParallelConfig};

const ADAPTER_CONFIG_FILE: &str = "adapter_config.json";
const ADAPTER_WEIGHTS_FILE: &str = "adapter_model.safetensors";
const SUPPORTED_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct LoraAdapterManifest {
    pub(crate) path: PathBuf,
    pub(crate) rank: usize,
    pub(crate) alpha: usize,
    pub(crate) target_modules: Vec<String>,
    pub(crate) tensor_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct LoraAdapter {
    pub(crate) manifest: LoraAdapterManifest,
    pub(crate) layers: Vec<LoraLayer>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LoraLayer {
    pub(crate) projections: BTreeMap<String, LoraProjection>,
}

#[derive(Debug, Clone)]
pub(crate) struct LoraProjection {
    pub(crate) a: LoraMatrix,
    pub(crate) b: LoraMatrix,
}

#[derive(Debug, Clone)]
pub(crate) struct LoraMatrix {
    pub(crate) data: Vec<bf16>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub(crate) struct DeviceLoraAdapter {
    pub(crate) name: String,
    pub(crate) manifest: LoraAdapterManifest,
    pub(crate) scale: f32,
    pub(crate) layers: Vec<DeviceLoraLayer>,
}

#[derive(Default)]
pub(crate) struct DeviceLoraLayer {
    pub(crate) q_proj: Option<DeviceLoraProjection>,
    pub(crate) k_proj: Option<DeviceLoraProjection>,
    pub(crate) v_proj: Option<DeviceLoraProjection>,
    pub(crate) o_proj: Option<DeviceLoraProjection>,
    pub(crate) gate_proj: Option<DeviceLoraProjection>,
    pub(crate) up_proj: Option<DeviceLoraProjection>,
    pub(crate) down_proj: Option<DeviceLoraProjection>,
}

pub(crate) struct DeviceLoraProjection {
    pub(crate) a: DeviceMatrix,
    pub(crate) b: DeviceMatrix,
}

#[derive(Debug, Deserialize)]
struct PeftAdapterConfig {
    #[serde(alias = "r")]
    lora_rank: usize,
    #[serde(alias = "lora_alpha")]
    alpha: usize,
    target_modules: TargetModules,
    #[serde(default)]
    peft_type: Option<String>,
    #[serde(default)]
    task_type: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TargetModules {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Copy)]
struct ProjectionSpec {
    path_segment: &'static str,
    in_dim: usize,
    out_dim: usize,
}

impl TargetModules {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(target) => vec![target],
            Self::Many(targets) => targets,
        }
    }
}

impl LoraAdapter {
    pub(crate) fn shard_for_tensor_parallel(
        &self,
        config: &Config,
        tensor_parallel: TensorParallelConfig,
    ) -> Result<Self> {
        tensor_parallel.validate_for(config)?;
        if !tensor_parallel.is_sharded() {
            return Ok(self.clone());
        }

        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut sharded_layer = LoraLayer::default();
            for (target, projection) in &layer.projections {
                sharded_layer.projections.insert(
                    target.clone(),
                    shard_projection_for_tensor_parallel(
                        config,
                        tensor_parallel,
                        target,
                        projection,
                    )?,
                );
            }
            layers.push(sharded_layer);
        }

        Ok(Self {
            manifest: self.manifest.clone(),
            layers,
        })
    }
}

pub(crate) fn load_lora_adapter(path: &Path, config: &Config) -> Result<LoraAdapter> {
    let (manifest, raw_weights) = inspect_lora_adapter(path, config)?;
    let tensors = SafeTensors::deserialize(&raw_weights).with_context(|| {
        format!(
            "failed to parse {}",
            path.join(ADAPTER_WEIGHTS_FILE).display()
        )
    })?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let mut layer = LoraLayer::default();
        for target in &manifest.target_modules {
            let spec = projection_spec(config, target)?;
            let a_name = tensor_name(layer_idx, spec.path_segment, "lora_A");
            let b_name = tensor_name(layer_idx, spec.path_segment, "lora_B");
            let a = load_matrix(&tensors, &a_name)?;
            let b = load_matrix(&tensors, &b_name)?;
            layer
                .projections
                .insert(target.clone(), LoraProjection { a, b });
        }
        layers.push(layer);
    }

    Ok(LoraAdapter { manifest, layers })
}

pub(crate) fn load_device_lora_adapter(
    ctx: &DeviceContext,
    name: String,
    adapter: LoraAdapter,
) -> Result<DeviceLoraAdapter> {
    let scale = adapter.manifest.alpha as f32 / adapter.manifest.rank as f32;
    let mut layers = Vec::with_capacity(adapter.layers.len());
    for layer in adapter.layers {
        let mut device_layer = DeviceLoraLayer::default();
        for (target, projection) in layer.projections {
            let device_projection = DeviceLoraProjection {
                a: projection.a.to_device(ctx)?,
                b: projection.b.to_device(ctx)?,
            };
            match target.as_str() {
                "q_proj" => device_layer.q_proj = Some(device_projection),
                "k_proj" => device_layer.k_proj = Some(device_projection),
                "v_proj" => device_layer.v_proj = Some(device_projection),
                "o_proj" => device_layer.o_proj = Some(device_projection),
                "gate_proj" => device_layer.gate_proj = Some(device_projection),
                "up_proj" => device_layer.up_proj = Some(device_projection),
                "down_proj" => device_layer.down_proj = Some(device_projection),
                _ => bail!("unsupported Qwen3 LoRA target module {target}"),
            }
        }
        layers.push(device_layer);
    }

    Ok(DeviceLoraAdapter {
        name,
        manifest: adapter.manifest,
        scale,
        layers,
    })
}

pub(crate) fn apply_lora_projection_delta(
    ctx: &DeviceContext,
    projection: &DeviceLoraProjection,
    input: &HiddenStates,
    out: &mut HiddenStates,
    row_offset: usize,
    scale: f32,
) -> Result<()> {
    let mut rank_out = HiddenStates::zeros(ctx, projection.a.rows, input.seq_len)?;
    ops::gemm_into(ctx, &projection.a, input, &mut rank_out);
    let mut delta = HiddenStates::zeros(ctx, projection.b.rows, input.seq_len)?;
    ops::gemm_into(ctx, &projection.b, &rank_out, &mut delta);
    ops::scaled_add_rows_into(ctx, &delta, scale, out, row_offset)
}

fn inspect_lora_adapter(path: &Path, config: &Config) -> Result<(LoraAdapterManifest, Vec<u8>)> {
    let adapter_config = load_adapter_config(path)?;
    let rank = adapter_config.lora_rank;
    let alpha = adapter_config.alpha;
    ensure!(rank > 0, "LoRA rank must be > 0");
    ensure!(alpha > 0, "LoRA alpha must be > 0");
    if let Some(peft_type) = &adapter_config.peft_type {
        ensure!(
            peft_type.eq_ignore_ascii_case("LORA"),
            "unsupported peft_type={peft_type}; expected LORA"
        );
    }
    let _task_type = adapter_config.task_type.as_deref();

    let target_modules = normalize_target_modules(adapter_config.target_modules.into_vec())?;
    let raw_weights = fs::read(path.join(ADAPTER_WEIGHTS_FILE)).with_context(|| {
        format!(
            "failed to read LoRA safetensors file {}",
            path.join(ADAPTER_WEIGHTS_FILE).display()
        )
    })?;
    let tensors = SafeTensors::deserialize(&raw_weights).with_context(|| {
        format!(
            "failed to parse {}",
            path.join(ADAPTER_WEIGHTS_FILE).display()
        )
    })?;

    validate_tensor_catalog(&tensors, config, rank, &target_modules)?;

    let manifest = LoraAdapterManifest {
        path: path.to_path_buf(),
        rank,
        alpha,
        target_modules,
        tensor_count: tensors.len(),
    };

    Ok((manifest, raw_weights))
}

fn load_adapter_config(path: &Path) -> Result<PeftAdapterConfig> {
    let config_path = path.join(ADAPTER_CONFIG_FILE);
    let content = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {}", config_path.display()))
}

fn normalize_target_modules(target_modules: Vec<String>) -> Result<Vec<String>> {
    ensure!(
        !target_modules.is_empty(),
        "LoRA adapter_config.json target_modules must not be empty"
    );

    let supported: BTreeSet<&str> = SUPPORTED_TARGET_MODULES.iter().copied().collect();
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(target_modules.len());
    for target in target_modules {
        ensure!(
            supported.contains(target.as_str()),
            "unsupported Qwen3 LoRA target module {target}; supported modules: {}",
            SUPPORTED_TARGET_MODULES.join(", ")
        );
        if seen.insert(target.clone()) {
            normalized.push(target);
        }
    }
    Ok(normalized)
}

fn validate_tensor_catalog(
    tensors: &SafeTensors<'_>,
    config: &Config,
    rank: usize,
    target_modules: &[String],
) -> Result<()> {
    let mut expected = BTreeMap::new();
    for layer_idx in 0..config.num_hidden_layers {
        for target in target_modules {
            let spec = projection_spec(config, target)?;
            expected.insert(
                tensor_name(layer_idx, spec.path_segment, "lora_A"),
                vec![rank, spec.in_dim],
            );
            expected.insert(
                tensor_name(layer_idx, spec.path_segment, "lora_B"),
                vec![spec.out_dim, rank],
            );
        }
    }

    let actual: BTreeSet<String> = tensors.names().into_iter().map(str::to_owned).collect();
    for (name, shape) in &expected {
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("missing LoRA tensor {name}"))?;
        ensure_lora_dtype(name, tensor.dtype())?;
        ensure!(
            tensor.shape() == shape.as_slice(),
            "LoRA tensor {name} shape mismatch: expected {:?}, got {:?}",
            shape,
            tensor.shape()
        );
    }

    for name in actual {
        if !expected.contains_key(&name) {
            bail!("unexpected LoRA tensor {name}");
        }
    }

    Ok(())
}

fn projection_spec(config: &Config, target: &str) -> Result<ProjectionSpec> {
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    match target {
        "q_proj" => Ok(ProjectionSpec {
            path_segment: "self_attn.q_proj",
            in_dim: config.hidden_size,
            out_dim: q_dim,
        }),
        "k_proj" => Ok(ProjectionSpec {
            path_segment: "self_attn.k_proj",
            in_dim: config.hidden_size,
            out_dim: kv_dim,
        }),
        "v_proj" => Ok(ProjectionSpec {
            path_segment: "self_attn.v_proj",
            in_dim: config.hidden_size,
            out_dim: kv_dim,
        }),
        "o_proj" => Ok(ProjectionSpec {
            path_segment: "self_attn.o_proj",
            in_dim: q_dim,
            out_dim: config.hidden_size,
        }),
        "gate_proj" => Ok(ProjectionSpec {
            path_segment: "mlp.gate_proj",
            in_dim: config.hidden_size,
            out_dim: config.intermediate_size,
        }),
        "up_proj" => Ok(ProjectionSpec {
            path_segment: "mlp.up_proj",
            in_dim: config.hidden_size,
            out_dim: config.intermediate_size,
        }),
        "down_proj" => Ok(ProjectionSpec {
            path_segment: "mlp.down_proj",
            in_dim: config.intermediate_size,
            out_dim: config.hidden_size,
        }),
        _ => bail!("unsupported Qwen3 LoRA target module {target}"),
    }
}

fn tensor_name(layer_idx: usize, path_segment: &str, lora_side: &str) -> String {
    format!("base_model.model.model.layers.{layer_idx}.{path_segment}.{lora_side}.weight")
}

fn ensure_lora_dtype(name: &str, dtype: Dtype) -> Result<()> {
    ensure!(
        matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32),
        "LoRA tensor {name} has unsupported dtype {dtype:?}; expected F16, BF16, or F32"
    );
    Ok(())
}

fn load_matrix(tensors: &SafeTensors<'_>, name: &str) -> Result<LoraMatrix> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing LoRA tensor {name}"))?;
    ensure!(
        tensor.shape().len() == 2,
        "LoRA tensor {name} expected 2D, got {:?}",
        tensor.shape()
    );
    Ok(LoraMatrix {
        data: tensor_to_bf16(&tensor, name)?,
        rows: tensor.shape()[0],
        cols: tensor.shape()[1],
    })
}

fn tensor_to_bf16(tensor: &TensorView<'_>, name: &str) -> Result<Vec<bf16>> {
    ensure_lora_dtype(name, tensor.dtype())?;
    let elems = tensor.shape().iter().product::<usize>();
    match tensor.dtype() {
        Dtype::BF16 => {
            ensure!(
                tensor.data().len() == elems * 2,
                "LoRA tensor {name} BF16 byte length mismatch"
            );
            Ok(tensor
                .data()
                .chunks_exact(2)
                .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
                .collect())
        }
        Dtype::F16 => {
            ensure!(
                tensor.data().len() == elems * 2,
                "LoRA tensor {name} F16 byte length mismatch"
            );
            Ok(tensor
                .data()
                .chunks_exact(2)
                .map(|bytes| {
                    let value = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
                    bf16::from_f32(value.to_f32())
                })
                .collect())
        }
        Dtype::F32 => {
            ensure!(
                tensor.data().len() == elems * 4,
                "LoRA tensor {name} F32 byte length mismatch"
            );
            Ok(tensor
                .data()
                .chunks_exact(4)
                .map(|bytes| {
                    bf16::from_f32(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                })
                .collect())
        }
        dtype => bail!("LoRA tensor {name} has unsupported dtype {dtype:?}"),
    }
}

impl LoraMatrix {
    fn to_device(&self, ctx: &DeviceContext) -> Result<DeviceMatrix> {
        DeviceMatrix::from_host(ctx, &self.data, self.rows, self.cols)
    }

    fn row_shard(&self, row_offset: usize, rows: usize) -> Result<Self> {
        ensure!(
            row_offset + rows <= self.rows,
            "LoRA row shard out of bounds: row_offset={} rows={} total_rows={}",
            row_offset,
            rows,
            self.rows
        );
        let start = row_offset * self.cols;
        let end = (row_offset + rows) * self.cols;
        Ok(Self {
            data: self.data[start..end].to_vec(),
            rows,
            cols: self.cols,
        })
    }

    fn col_shard(&self, col_offset: usize, cols: usize) -> Result<Self> {
        ensure!(
            col_offset + cols <= self.cols,
            "LoRA col shard out of bounds: col_offset={} cols={} total_cols={}",
            col_offset,
            cols,
            self.cols
        );
        let mut data = Vec::with_capacity(self.rows * cols);
        for row in 0..self.rows {
            let start = row * self.cols + col_offset;
            data.extend_from_slice(&self.data[start..start + cols]);
        }
        Ok(Self {
            data,
            rows: self.rows,
            cols,
        })
    }
}

fn shard_projection_for_tensor_parallel(
    config: &Config,
    tensor_parallel: TensorParallelConfig,
    target: &str,
    projection: &LoraProjection,
) -> Result<LoraProjection> {
    match target {
        "q_proj" => {
            let (row_offset, rows) =
                tensor_parallel.shard_range(config.num_attention_heads * config.head_dim);
            Ok(LoraProjection {
                a: projection.a.clone(),
                b: projection.b.row_shard(row_offset, rows)?,
            })
        }
        "k_proj" | "v_proj" => {
            let (row_offset, rows) =
                tensor_parallel.shard_range(config.num_key_value_heads * config.head_dim);
            Ok(LoraProjection {
                a: projection.a.clone(),
                b: projection.b.row_shard(row_offset, rows)?,
            })
        }
        "gate_proj" | "up_proj" => {
            let (row_offset, rows) = tensor_parallel.shard_range(config.intermediate_size);
            Ok(LoraProjection {
                a: projection.a.clone(),
                b: projection.b.row_shard(row_offset, rows)?,
            })
        }
        "o_proj" => {
            let (col_offset, cols) =
                tensor_parallel.shard_range(config.num_attention_heads * config.head_dim);
            Ok(LoraProjection {
                a: projection.a.col_shard(col_offset, cols)?,
                b: projection.b.clone(),
            })
        }
        "down_proj" => {
            let (col_offset, cols) = tensor_parallel.shard_range(config.intermediate_size);
            Ok(LoraProjection {
                a: projection.a.col_shard(col_offset, cols)?,
                b: projection.b.clone(),
            })
        }
        _ => bail!("unsupported Qwen3 LoRA target module {target}"),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use safetensors::View;

    use super::*;

    static NEXT_TEST_DIR: AtomicUsize = AtomicUsize::new(0);

    fn tiny_config() -> Config {
        Config {
            hidden_size: 4,
            intermediate_size: 6,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 2,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            eos_token_id: 151645,
            tie_word_embeddings: false,
            stop_token_ids: vec![151645],
        }
    }

    fn temp_adapter_dir(test_name: &str) -> PathBuf {
        let id = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "pegainfer-qwen3-lora-{test_name}-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp adapter dir");
        path
    }

    fn write_adapter_config(path: &Path, targets: &[&str], rank: usize) {
        let targets = targets
            .iter()
            .map(|target| format!("\"{target}\""))
            .collect::<Vec<_>>()
            .join(", ");
        fs::write(
            path.join(ADAPTER_CONFIG_FILE),
            format!(
                r#"{{
  "peft_type": "LORA",
  "r": {rank},
  "lora_alpha": 16,
  "target_modules": [{targets}]
}}"#
            ),
        )
        .expect("write adapter config");
    }

    fn write_adapter_weights(path: &Path, config: &Config, targets: &[&str], rank: usize) {
        let mut tensors = BTreeMap::new();
        for layer_idx in 0..config.num_hidden_layers {
            for target in targets {
                let spec = projection_spec(config, target).expect("projection spec");
                push_tensor(
                    &mut tensors,
                    tensor_name(layer_idx, spec.path_segment, "lora_A"),
                    vec![rank, spec.in_dim],
                );
                push_tensor(
                    &mut tensors,
                    tensor_name(layer_idx, spec.path_segment, "lora_B"),
                    vec![spec.out_dim, rank],
                );
            }
        }
        safetensors::serialize_to_file(tensors, None, &path.join(ADAPTER_WEIGHTS_FILE))
            .expect("write safetensors");
    }

    #[derive(Clone)]
    struct TestTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for TestTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn push_tensor(tensors: &mut BTreeMap<String, TestTensor>, name: String, shape: Vec<usize>) {
        push_tensor_with_dtype(tensors, name, shape, Dtype::BF16, 0.0);
    }

    fn push_tensor_with_dtype(
        tensors: &mut BTreeMap<String, TestTensor>,
        name: String,
        shape: Vec<usize>,
        dtype: Dtype,
        value: f32,
    ) {
        let elems = shape.iter().product::<usize>();
        let data = match dtype {
            Dtype::BF16 => bf16::from_f32(value).to_bits().to_le_bytes().repeat(elems),
            Dtype::F16 => f16::from_f32(value).to_bits().to_le_bytes().repeat(elems),
            Dtype::F32 => value.to_le_bytes().repeat(elems),
            _ => panic!("unsupported test dtype {dtype:?}"),
        };
        tensors.insert(name, TestTensor { dtype, shape, data });
    }

    fn matrix(rows: usize, cols: usize) -> LoraMatrix {
        let data = (0..rows * cols)
            .map(|idx| bf16::from_f32(idx as f32))
            .collect();
        LoraMatrix { data, rows, cols }
    }

    fn values(matrix: &LoraMatrix) -> Vec<f32> {
        matrix.data.iter().map(|value| value.to_f32()).collect()
    }

    #[test]
    fn validates_supported_qwen3_lora_adapter() {
        let config = tiny_config();
        let path = temp_adapter_dir("valid");
        let targets = SUPPORTED_TARGET_MODULES;
        write_adapter_config(&path, targets, 2);
        write_adapter_weights(&path, &config, targets, 2);

        let manifest = load_lora_adapter(&path, &config)
            .expect("load adapter")
            .manifest;

        assert_eq!(manifest.rank, 2);
        assert_eq!(manifest.alpha, 16);
        assert_eq!(manifest.target_modules, targets);
        assert_eq!(
            manifest.tensor_count,
            config.num_hidden_layers * targets.len() * 2
        );
    }

    #[test]
    fn loads_lora_tensors_grouped_by_layer_and_target() {
        let config = tiny_config();
        let path = temp_adapter_dir("load");
        write_adapter_config(&path, &["q_proj", "down_proj"], 2);
        write_adapter_weights(&path, &config, &["q_proj", "down_proj"], 2);

        let adapter = load_lora_adapter(&path, &config).expect("load adapter");

        assert_eq!(adapter.manifest.rank, 2);
        assert_eq!(adapter.layers.len(), config.num_hidden_layers);
        let layer0 = &adapter.layers[0];
        let q_proj = layer0.projections.get("q_proj").expect("q_proj");
        assert_eq!((q_proj.a.rows, q_proj.a.cols), (2, config.hidden_size));
        assert_eq!(
            (q_proj.b.rows, q_proj.b.cols),
            (config.num_attention_heads * config.head_dim, 2)
        );
        let down_proj = layer0.projections.get("down_proj").expect("down_proj");
        assert_eq!(
            (down_proj.a.rows, down_proj.a.cols),
            (2, config.intermediate_size)
        );
        assert_eq!(
            (down_proj.b.rows, down_proj.b.cols),
            (config.hidden_size, 2)
        );
    }

    #[test]
    fn loads_supported_lora_tensor_dtypes_as_bf16() {
        let config = tiny_config();
        let path = temp_adapter_dir("dtype-load");
        write_adapter_config(&path, &["q_proj"], 2);

        let mut tensors = BTreeMap::new();
        for layer_idx in 0..config.num_hidden_layers {
            push_tensor_with_dtype(
                &mut tensors,
                tensor_name(layer_idx, "self_attn.q_proj", "lora_A"),
                vec![2, config.hidden_size],
                Dtype::F16,
                1.5,
            );
            push_tensor_with_dtype(
                &mut tensors,
                tensor_name(layer_idx, "self_attn.q_proj", "lora_B"),
                vec![config.hidden_size, 2],
                Dtype::F32,
                2.25,
            );
        }
        safetensors::serialize_to_file(tensors, None, &path.join(ADAPTER_WEIGHTS_FILE))
            .expect("write safetensors");

        let adapter = load_lora_adapter(&path, &config).expect("load adapter");
        let q_proj = adapter.layers[0].projections.get("q_proj").expect("q_proj");

        assert_eq!(q_proj.a.data[0].to_f32(), bf16::from_f32(1.5).to_f32());
        assert_eq!(q_proj.b.data[0].to_f32(), bf16::from_f32(2.25).to_f32());
    }

    #[test]
    fn rejects_unsupported_target_module() {
        let config = tiny_config();
        let path = temp_adapter_dir("unsupported-target");
        write_adapter_config(&path, &["q_proj", "embed_tokens"], 2);
        write_adapter_weights(&path, &config, &["q_proj"], 2);

        let error = load_lora_adapter(&path, &config).expect_err("unsupported target");

        assert!(error.to_string().contains("unsupported Qwen3 LoRA target"));
    }

    #[test]
    fn rejects_wrong_lora_tensor_shape() {
        let config = tiny_config();
        let path = temp_adapter_dir("bad-shape");
        write_adapter_config(&path, &["q_proj"], 2);

        let mut tensors = BTreeMap::new();
        for layer_idx in 0..config.num_hidden_layers {
            push_tensor(
                &mut tensors,
                tensor_name(layer_idx, "self_attn.q_proj", "lora_A"),
                vec![2, config.hidden_size],
            );
            push_tensor(
                &mut tensors,
                tensor_name(layer_idx, "self_attn.q_proj", "lora_B"),
                if layer_idx == 0 {
                    vec![config.hidden_size + 1, 2]
                } else {
                    vec![config.hidden_size, 2]
                },
            );
        }
        safetensors::serialize_to_file(tensors, None, &path.join(ADAPTER_WEIGHTS_FILE))
            .expect("write safetensors");

        let error = load_lora_adapter(&path, &config).expect_err("bad tensor shape");

        assert!(error.to_string().contains("shape mismatch"));
    }

    #[test]
    fn shards_column_parallel_lora_b_rows_for_tp_rank() {
        let config = tiny_config();
        let tp = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };
        let projection = LoraProjection {
            a: matrix(2, config.hidden_size),
            b: matrix(config.intermediate_size, 2),
        };

        let sharded = shard_projection_for_tensor_parallel(&config, tp, "gate_proj", &projection)
            .expect("shard gate_proj");

        assert_eq!((sharded.a.rows, sharded.a.cols), (2, config.hidden_size));
        assert_eq!(values(&sharded.a), values(&projection.a));
        assert_eq!((sharded.b.rows, sharded.b.cols), (3, 2));
        assert_eq!(values(&sharded.b), vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn shards_row_parallel_lora_a_cols_for_tp_rank() {
        let config = tiny_config();
        let tp = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };
        let projection = LoraProjection {
            a: matrix(2, config.intermediate_size),
            b: matrix(config.hidden_size, 2),
        };

        let sharded = shard_projection_for_tensor_parallel(&config, tp, "down_proj", &projection)
            .expect("shard down_proj");

        assert_eq!((sharded.a.rows, sharded.a.cols), (2, 3));
        assert_eq!(values(&sharded.a), vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]);
        assert_eq!((sharded.b.rows, sharded.b.cols), (config.hidden_size, 2));
        assert_eq!(values(&sharded.b), values(&projection.b));
    }

    #[test]
    fn shards_full_adapter_for_tensor_parallel() {
        let config = tiny_config();
        let path = temp_adapter_dir("tp-shard");
        write_adapter_config(&path, &["q_proj", "down_proj"], 2);
        write_adapter_weights(&path, &config, &["q_proj", "down_proj"], 2);
        let adapter = load_lora_adapter(&path, &config).expect("load adapter");

        let sharded = adapter
            .shard_for_tensor_parallel(
                &config,
                TensorParallelConfig {
                    rank: 1,
                    world_size: 2,
                },
            )
            .expect("shard adapter");

        let q_proj = sharded.layers[0].projections.get("q_proj").expect("q_proj");
        assert_eq!((q_proj.a.rows, q_proj.a.cols), (2, config.hidden_size));
        assert_eq!((q_proj.b.rows, q_proj.b.cols), (2, 2));

        let down_proj = sharded.layers[0]
            .projections
            .get("down_proj")
            .expect("down_proj");
        assert_eq!((down_proj.a.rows, down_proj.a.cols), (2, 3));
        assert_eq!(
            (down_proj.b.rows, down_proj.b.cols),
            (config.hidden_size, 2)
        );
    }
}
