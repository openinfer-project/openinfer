use vllm_text::tokenizer::DynTokenizer;

#[allow(dead_code)]
pub(crate) fn load_tokenizer(model_path: &str) -> DynTokenizer {
    openinfer_vllm_support::load_tokenizer(model_path)
        .unwrap_or_else(|err| panic!("Failed to load tokenizer for {model_path}: {err}"))
}

pub(crate) fn tp2_device_ordinals() -> Vec<usize> {
    const ENV: &str = "OPENINFER_TEST_TP_DEVICES";
    let value = match std::env::var(ENV) {
        Ok(value) => value,
        Err(_) => return vec![0, 1],
    };

    let devices: Vec<usize> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .unwrap_or_else(|err| panic!("{ENV} must be comma-separated CUDA ordinals: {err}"))
        })
        .collect();

    assert_eq!(
        devices.len(),
        2,
        "{ENV} must specify exactly two CUDA ordinals for TP2, e.g. 0,1 or 2,3"
    );
    assert_ne!(
        devices[0], devices[1],
        "{ENV} must specify two distinct CUDA ordinals for TP2"
    );
    devices
}
