use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::config::Config;

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

pub(crate) fn validate_lora_adapter(path: &Path, config: &Config) -> Result<LoraAdapterManifest> {
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

    Ok(LoraAdapterManifest {
        path: path.to_path_buf(),
        rank,
        alpha,
        target_modules,
        tensor_count: tensors.len(),
    })
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

    let actual: BTreeSet<&str> = tensors.names().into_iter().collect();
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
        if !expected.contains_key(name) {
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
            num_key_value_heads: 1,
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
        let bytes = vec![0_u8; shape.iter().product::<usize>() * 2];
        tensors.insert(
            name,
            TestTensor {
                dtype: Dtype::BF16,
                shape,
                data: bytes,
            },
        );
    }

    #[test]
    fn validates_supported_qwen3_lora_adapter() {
        let config = tiny_config();
        let path = temp_adapter_dir("valid");
        let targets = SUPPORTED_TARGET_MODULES;
        write_adapter_config(&path, targets, 2);
        write_adapter_weights(&path, &config, targets, 2);

        let manifest = validate_lora_adapter(&path, &config).expect("validate adapter");

        assert_eq!(manifest.rank, 2);
        assert_eq!(manifest.alpha, 16);
        assert_eq!(manifest.target_modules, targets);
        assert_eq!(
            manifest.tensor_count,
            config.num_hidden_layers * targets.len() * 2
        );
    }

    #[test]
    fn rejects_unsupported_target_module() {
        let config = tiny_config();
        let path = temp_adapter_dir("unsupported-target");
        write_adapter_config(&path, &["q_proj", "embed_tokens"], 2);
        write_adapter_weights(&path, &config, &["q_proj"], 2);

        let error = validate_lora_adapter(&path, &config).expect_err("unsupported target");

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

        let error = validate_lora_adapter(&path, &config).expect_err("bad tensor shape");

        assert!(error.to_string().contains("shape mismatch"));
    }
}
