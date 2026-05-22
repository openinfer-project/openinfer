use std::sync::{Arc, Mutex};

use once_cell::sync::OnceCell;
use tokio::runtime::{Builder, Runtime};
use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};
use vllm_text::{Error, Result};
use vllm_tokenizer::{DynTokenizer, HuggingFaceTokenizer, TekkenTokenizer, TiktokenTokenizer};

static TOKENIZER_RESOLVER_RUNTIME: OnceCell<Mutex<Runtime>> = OnceCell::new();

pub fn load_tokenizer(model_id: &str) -> Result<DynTokenizer> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(Error::Tokenizer(
            "pegainfer_vllm_support::load_tokenizer is synchronous and cannot be called from \
             inside an active Tokio runtime; use load_tokenizer_async instead"
                .to_string(),
        ));
    }

    let files = resolve_model_files(model_id)?;
    tokenizer_from_source(&files.tokenizer)
}

pub async fn load_tokenizer_async(model_id: &str) -> Result<DynTokenizer> {
    let files = ResolvedModelFiles::new(model_id).await?;
    tokenizer_from_source(&files.tokenizer)
}

pub fn tokenizer_from_source(source: &TokenizerSource) -> Result<DynTokenizer> {
    match source {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}

fn resolve_model_files(model_id: &str) -> Result<ResolvedModelFiles> {
    let runtime = TOKENIZER_RESOLVER_RUNTIME.get_or_try_init(|| {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .map(Mutex::new)
            .map_err(|err| {
                Error::Tokenizer(format!(
                    "failed to create tokenizer resolver runtime: {err}"
                ))
            })
    })?;
    let runtime = runtime
        .lock()
        .map_err(|_| Error::Tokenizer("tokenizer resolver runtime mutex poisoned".to_string()))?;

    runtime.block_on(ResolvedModelFiles::new(model_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_loader_rejects_active_tokio_runtime() {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime should build");

        let err = runtime.block_on(async {
            match load_tokenizer("unused-model-id") {
                Ok(_) => panic!("sync tokenizer loader should reject active Tokio runtime"),
                Err(err) => err,
            }
        });

        assert!(
            err.to_string().contains("load_tokenizer_async"),
            "unexpected error: {err}"
        );
    }
}
