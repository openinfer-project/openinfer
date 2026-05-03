use std::path::Path;
use std::sync::Arc;

use vllm_text::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

pub(crate) fn load_tokenizer(model_path: &str) -> DynTokenizer {
    let tokenizer_path = Path::new(model_path).join("tokenizer.json");
    Arc::new(
        HuggingFaceTokenizer::new(&tokenizer_path)
            .unwrap_or_else(|e| panic!("Failed to load {}: {e}", tokenizer_path.display())),
    )
}
