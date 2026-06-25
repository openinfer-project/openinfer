//! `--batch-invariant` is rejected at the Qwen3 builder boundary for unsupported combos:
//! stream overlap, the DFlash drafter, and LoRA each run GEMMs the base-projection pin does not cover; the builder rejects all three.

use std::path::Path;
use std::sync::Mutex;

use openinfer_core::engine::EngineLoadOptions;
use openinfer_kernels::ops::{NumericPolicy, numeric_policy, set_numeric_policy};
use openinfer_qwen3_4b::{
    DEFAULT_MAX_PREFILL_TOKENS, DecodeOverlap, Qwen3LoraOptions, Qwen3MemoryOptions,
    Qwen3OffloadOptions, start_engine_with_lora_control, start_engine_with_offload,
};

// Serialize the reject #[test]s — they share the process-global numeric policy.
static POLICY_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn batch_invariant_rejects_decode_overlap() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_offload(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3OffloadOptions::disabled(),
        false,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::SharedSm,
        true,
        None,
        false,
    )
    .err()
    .expect("--batch-invariant + --decode-overlap must be rejected");
    assert!(
        format!("{err}").contains("decode-overlap"),
        "unexpected error: {err}"
    );
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}

#[test]
fn batch_invariant_rejects_dflash() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_offload(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3OffloadOptions::disabled(),
        false,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        true,
        Some(Path::new("/nonexistent-draft")),
        false,
    )
    .err()
    .expect("--batch-invariant + DFlash must be rejected");
    assert!(
        format!("{err}").contains("DFlash"),
        "unexpected error: {err}"
    );
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}

#[test]
fn batch_invariant_rejects_lora() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_lora_control(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3LoraOptions::default(),
        Qwen3OffloadOptions::disabled(),
        false,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        true,
    )
    .err()
    .expect("--batch-invariant + LoRA must be rejected");
    assert!(format!("{err}").contains("LoRA"), "unexpected error: {err}");
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}
