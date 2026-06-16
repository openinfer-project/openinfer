/// Slot that a Higgs checkpoint tensor name maps to in the backbone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackboneSlot {
    /// `tied.embedding.text_embedding.weight` — used as both embed_tokens and lm_head (tied).
    EmbedTokens,
    /// `tied.head.text_head.weight` — lm_head (tied with embed_tokens, value should match).
    LmHead,
    /// `body.layers.{i}.{rest}` — transformer block at layer index i.
    Layer { layer_idx: usize, rest: String },
    /// `body.norm.weight` — final RMSNorm.
    FinalNorm,
}

/// Map a Higgs checkpoint tensor name to its backbone slot.
/// Returns `None` for non-backbone tensors (audio/codec/modality).
pub fn map_backbone(name: &str) -> Option<BackboneSlot> {
    // tied.embedding.text_embedding.weight
    if name == "tied.embedding.text_embedding.weight" {
        return Some(BackboneSlot::EmbedTokens);
    }
    // tied.head.text_head.weight
    if name == "tied.head.text_head.weight" {
        return Some(BackboneSlot::LmHead);
    }
    // body.norm.weight
    if name == "body.norm.weight" {
        return Some(BackboneSlot::FinalNorm);
    }
    // body.layers.{i}.{rest}
    if let Some(rest) = name.strip_prefix("body.layers.") {
        // rest looks like "0.self_attn.q_proj.weight" or "0.input_layernorm.weight"
        if let Some(dot_pos) = rest.find('.') {
            let layer_str = &rest[..dot_pos];
            if let Ok(layer_idx) = layer_str.parse::<usize>() {
                let rest = rest[dot_pos + 1..].to_string();
                return Some(BackboneSlot::Layer { layer_idx, rest });
            }
        }
    }
    // Everything else: audio/codec/modality — skip.
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn test_model_path() -> PathBuf {
        let env_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("docs/private/higgs-audio-v3-tts-4b"));
        if env_path.is_absolute() {
            env_path
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root")
                .join(&env_path)
        }
    }

    fn load_index_tensor_names(model_path: &PathBuf) -> Vec<String> {
        let index_path = model_path.join("model.safetensors.index.json");
        let content = std::fs::read_to_string(&index_path).expect("failed to read index");
        let index: serde_json::Value =
            serde_json::from_str(&content).expect("failed to parse index");
        index["weight_map"]
            .as_object()
            .expect("weight_map not an object")
            .keys()
            .cloned()
            .collect()
    }

    #[test]
    fn backbone_tensors_map_to_exactly_398_unique_slots() {
        let names = load_index_tensor_names(&test_model_path());

        let mut backbone_count = 0usize;
        let mut seen_slots = HashSet::new();
        let mut embed_seen = false;
        let mut norm_seen = false;

        for name in &names {
            if let Some(slot) = map_backbone(name) {
                backbone_count += 1;
                match &slot {
                    BackboneSlot::EmbedTokens => embed_seen = true,
                    BackboneSlot::LmHead => {} // may or may not exist
                    BackboneSlot::FinalNorm => norm_seen = true,
                    BackboneSlot::Layer { layer_idx, rest } => {
                        seen_slots.insert((*layer_idx, rest.clone()));
                    }
                }
            }
        }

        // 36 layers × 11 tensors per layer = 396 + embed + norm = 398
        // (lm_head is tied with embed_tokens, no separate tied.head.text_head.weight
        //  in this checkpoint)
        assert_eq!(backbone_count, 398, "expected exactly 398 backbone tensors");
        assert!(embed_seen, "embed_tokens not found");
        assert!(norm_seen, "final norm not found");

        // Check all 36 layers present with all 11 components
        let expected_components: &[&str] = &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];

        for layer_idx in 0..36 {
            for comp in expected_components {
                assert!(
                    seen_slots.contains(&(layer_idx, comp.to_string())),
                    "missing layer {layer_idx} component {comp}"
                );
            }
        }

        assert_eq!(seen_slots.len(), 396, "expected 396 unique layer slots");
    }

    #[test]
    fn audio_codec_tensors_all_skip() {
        let names = load_index_tensor_names(&test_model_path());

        for name in &names {
            if name.starts_with("body.")
                || name == "tied.embedding.text_embedding.weight"
                || name == "tied.head.text_head.weight"
            {
                // These are backbone — should map to Some
                assert!(
                    map_backbone(name).is_some(),
                    "backbone tensor {name} returned None"
                );
            } else {
                // Everything else (audio/codec/modality) should be None
                assert!(
                    map_backbone(name).is_none(),
                    "non-backbone tensor {name} should return None"
                );
            }
        }
    }
}
