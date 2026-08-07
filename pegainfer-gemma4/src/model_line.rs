//! Gemma 4's [`ModelLine`] implementation (registration only — the engine
//! itself is not yet available).

use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

pub static MODEL_LINE: Gemma4Line = Gemma4Line;

pub struct Gemma4Line;

fn config_model_type(config: &serde_json::Value) -> Option<&str> {
    config.get("model_type").and_then(serde_json::Value::as_str)
}

fn text_config_model_type(config: &serde_json::Value) -> Option<&str> {
    config
        .get("text_config")
        .and_then(|text| text.get("model_type"))
        .and_then(serde_json::Value::as_str)
}

impl ModelLine for Gemma4Line {
    fn name(&self) -> &'static str {
        "Gemma 4"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let is_gemma4 = matches!(config_model_type(config), Some("gemma4" | "gemma4_unified"))
            || matches!(
                text_config_model_type(config),
                Some("gemma4_text" | "gemma4_unified_text")
            );
        if !is_gemma4 {
            return Err(format!(
                "model_type {:?} is not a Gemma 4 identity",
                config_model_type(config)
            ));
        }
        crate::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &["cuda_graph"]
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<EngineHandle> {
        crate::start_engine(ctx.model_path, EngineLoadOptions::default())
    }
}
