mod config;

use std::path::Path;

use anyhow::Result;
pub use config::probe_config_json;
use pegainfer_engine::engine::EngineHandle;
use pegainfer_engine::engine::EngineLoadOptions;

#[cfg(feature = "gemma4")]
pub fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!("Gemma 4 engine is not implemented yet (registration only)")
}

#[cfg(not(feature = "gemma4"))]
pub fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!(
        "Gemma 4 support is feature-gated; rebuild pegainfer-server with --features gemma4"
    )
}
