//! DeepSeek-V2-Lite's [`ModelLine`] implementation.

use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

pub static MODEL_LINE: DeepSeekV2LiteLine = DeepSeekV2LiteLine;

pub struct DeepSeekV2LiteLine;

impl ModelLine for DeepSeekV2LiteLine {
    fn name(&self) -> &'static str {
        "DeepSeek-V2-Lite"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        match crate::probe_config_json(config) {
            Ok(true) => Ok(()),
            Ok(false) => Err(format!(
                "model_type {:?} is not \"deepseek_v2\"",
                config.get("model_type").and_then(serde_json::Value::as_str)
            )),
            Err(error) => Err(error.to_string()),
        }
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &["cuda_graph"]
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<EngineHandle> {
        crate::launch(ctx.model_path, ctx.shared.cuda_graph)
    }
}
