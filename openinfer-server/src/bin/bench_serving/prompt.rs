//! Prompt resolution: inline/file/synthetic inputs and per-request salting.

use std::fs;

use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use vllm_text::tokenizer::DynTokenizer;

use crate::cli::PromptInputArgs;
use crate::report::PromptDescriptor;

pub(crate) const SYNTHETIC_PATTERN: &str = "token_id = 100 + (idx % 1000)";

pub(crate) fn truncate_preview(text: &str, limit: usize) -> String {
    let one_line = text.replace('\n', "\\n");
    if one_line.chars().count() <= limit {
        return one_line;
    }
    let mut truncated = String::new();
    for ch in one_line.chars().take(limit) {
        truncated.push(ch);
    }
    truncated.push_str("...");
    truncated
}

pub(crate) fn synthetic_prompt_tokens(len: usize) -> Vec<u32> {
    (0..len).map(|i| ((i % 1000) + 100) as u32).collect()
}

/// Like `synthetic_prompt_tokens`, but offset by `salt` so each prompt of the
/// same prompt length gets a different token sequence and so misses the prefix
/// cache when intended (e.g. `--inj-warm-frac 0.0`).
pub(crate) fn synthetic_prompt_tokens_salted(len: usize, salt: usize) -> Vec<u32> {
    let shift = salt.wrapping_mul(7919);
    (0..len)
        .map(|i| (((i + shift) % 1000) + 100) as u32)
        .collect()
}

/// Token-id bounds for synthetic concurrent prompts: above the low special
/// tokens and well under the smallest supported vocab (DeepSeek-V2-Lite ≈
/// 102 400), so every drawn id is an ordinary token on any model line.
pub(crate) const SYNTHETIC_TOKEN_LO: u32 = 100;
pub(crate) const SYNTHETIC_TOKEN_HI: u32 = 100_000;

/// One synthetic prompt of `len` random tokens, seeded per request so the
/// concurrent decode streams diverge. Identical concurrent prompts route a MoE
/// batch onto a narrow expert set, packing the Marlin expert GEMM into fat
/// tiles and under-measuring decode TPOT by ~7–15% (measured on Kimi-K2 via a
/// `--distinct-prompts` sweep; the bench trap behind the misread #225 "+51%
/// HTTP" gap). Distinct prompts exercise realistic broad expert routing. See
/// docs/lessons/moe-bench-prompt-diversity.md.
pub(crate) fn synthetic_random_prompt(len: usize, seed: u64, request_idx: usize) -> Vec<u32> {
    let mut rng =
        StdRng::seed_from_u64(seed ^ (request_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (0..len)
        .map(|_| rng.random_range(SYNTHETIC_TOKEN_LO..SYNTHETIC_TOKEN_HI))
        .collect()
}

#[derive(Debug, Clone)]
pub(crate) struct PromptSpec {
    pub(crate) descriptor: PromptDescriptor,
    pub(crate) tokens: Vec<u32>,
}

pub(crate) fn resolve_prompt_input(
    args: &PromptInputArgs,
    tokenizer: &DynTokenizer,
    default_text: Option<&str>,
    default_prompt_len: Option<usize>,
) -> Result<PromptSpec> {
    match (&args.prompt, &args.prompt_file, args.prompt_len) {
        (Some(prompt), None, None) => Ok(PromptSpec {
            descriptor: PromptDescriptor {
                source: "text".to_string(),
                prompt_tokens: tokenizer.encode(prompt, false)?.len(),
                prompt_preview: Some(truncate_preview(prompt, 96)),
            },
            tokens: tokenizer.encode(prompt, false)?,
        }),
        (None, Some(path), None) => {
            let prompt = fs::read_to_string(path)
                .with_context(|| format!("failed to read prompt file: {path}"))?;
            let tokens = tokenizer.encode(&prompt, false)?;
            Ok(PromptSpec {
                descriptor: PromptDescriptor {
                    source: format!("file:{path}"),
                    prompt_tokens: tokens.len(),
                    prompt_preview: Some(truncate_preview(&prompt, 96)),
                },
                tokens,
            })
        }
        (None, None, Some(prompt_len)) => {
            ensure!(prompt_len > 0, "--prompt-len must be > 0");
            Ok(PromptSpec {
                descriptor: PromptDescriptor {
                    source: format!("synthetic:{SYNTHETIC_PATTERN}"),
                    prompt_tokens: prompt_len,
                    prompt_preview: None,
                },
                tokens: synthetic_prompt_tokens(prompt_len),
            })
        }
        (None, None, None) => {
            if let Some(prompt) = default_text {
                let tokens = tokenizer.encode(prompt, false)?;
                Ok(PromptSpec {
                    descriptor: PromptDescriptor {
                        source: "text".to_string(),
                        prompt_tokens: tokens.len(),
                        prompt_preview: Some(truncate_preview(prompt, 96)),
                    },
                    tokens,
                })
            } else if let Some(prompt_len) = default_prompt_len {
                Ok(PromptSpec {
                    descriptor: PromptDescriptor {
                        source: format!("synthetic:{SYNTHETIC_PATTERN}"),
                        prompt_tokens: prompt_len,
                        prompt_preview: None,
                    },
                    tokens: synthetic_prompt_tokens(prompt_len),
                })
            } else {
                unreachable!("default prompt source must be provided");
            }
        }
        _ => unreachable!("clap enforces prompt input conflicts"),
    }
}
