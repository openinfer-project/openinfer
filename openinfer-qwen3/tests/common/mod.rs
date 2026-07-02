use vllm_text::tokenizer::DynTokenizer;

pub(crate) fn load_tokenizer(model_path: &str) -> DynTokenizer {
    openinfer_vllm_support::load_tokenizer(model_path)
        .unwrap_or_else(|err| panic!("Failed to load tokenizer for {model_path}: {err}"))
}
