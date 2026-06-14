//! N-gram (prompt-lookup) speculative proposer for Qwen3.
//!
//! This proposes candidate continuation tokens **without a draft model**: it
//! looks for the most recent earlier occurrence of the current token suffix in
//! the running context (prompt + tokens generated so far) and proposes the
//! tokens that followed that occurrence. The target model then verifies these
//! candidates in a single forward pass and accepts the longest matching prefix.
//!
//! This is effective for repetitive / structured text (code, quoting, JSON,
//! long-form copying) at zero extra model cost. When no n-gram match is found
//! it proposes nothing and decoding falls back to the normal one-token step.
//!
//! Only the proposer is implemented here; verification and KV rollback are
//! handled by the decode path and are intentionally out of scope for this
//! module so the lookup logic can be unit-tested in isolation.

/// Configuration for [`NgramProposer`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NgramConfig {
    /// Largest suffix length to match. Longer suffixes are tried first because
    /// they are more specific (a longer match is a stronger predictor).
    pub max_ngram: usize,
    /// Smallest suffix length to match before giving up. Must be `>= 1`.
    pub min_ngram: usize,
    /// Maximum number of speculative tokens proposed per step.
    pub num_speculative: usize,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            max_ngram: 3,
            min_ngram: 1,
            num_speculative: 4,
        }
    }
}

impl std::fmt::Display for NgramConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "K={}, max_ngram={}",
            self.num_speculative, self.max_ngram
        )
    }
}

impl NgramConfig {
    /// Read the n-gram-specific knobs from the environment, falling back to
    /// [`Default`] for anything unset. The generic on/off switch lives in
    /// [`crate::speculative::SpeculativeConfig::from_env`]; this only owns the
    /// proposer's own parameters:
    ///
    /// * `OPENINFER_QWEN3_NGRAM_TOKENS` = draft count `K`.
    /// * `OPENINFER_QWEN3_NGRAM_MAX_NGRAM` = longest suffix to match.
    #[must_use]
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            max_ngram: env_usize("OPENINFER_QWEN3_NGRAM_MAX_NGRAM").unwrap_or(defaults.max_ngram),
            min_ngram: defaults.min_ngram,
            num_speculative: env_usize("OPENINFER_QWEN3_NGRAM_TOKENS")
                .unwrap_or(defaults.num_speculative),
        }
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Stateless n-gram / prompt-lookup proposer.
///
/// The proposer keeps no history of its own; each call scans the supplied
/// context, so the caller is free to reuse a single instance across requests.
#[derive(Clone, Copy, Debug)]
pub struct NgramProposer {
    config: NgramConfig,
}

impl NgramProposer {
    /// Create a proposer. `min_ngram` is clamped to at least 1 and to at most
    /// `max_ngram`, and `max_ngram` to at least 1, so the config is always
    /// usable regardless of caller input.
    #[must_use]
    pub fn new(config: NgramConfig) -> Self {
        let max_ngram = config.max_ngram.max(1);
        let min_ngram = config.min_ngram.clamp(1, max_ngram);
        Self {
            config: NgramConfig {
                max_ngram,
                min_ngram,
                num_speculative: config.num_speculative,
            },
        }
    }

    /// Propose up to `num_speculative` continuation tokens for `context`
    /// (the full token sequence so far: prompt followed by generated tokens).
    ///
    /// Returns an empty `Vec` when no usable n-gram match exists or when
    /// `num_speculative == 0`. Tries the longest configured suffix first and
    /// falls back to shorter suffixes.
    #[must_use]
    pub fn propose(&self, context: &[u32]) -> Vec<u32> {
        if self.config.num_speculative == 0 {
            return Vec::new();
        }
        let len = context.len();
        // A match plus at least one following token needs `n + 1` tokens.
        let max_n = self.config.max_ngram.min(len.saturating_sub(1));
        for n in (self.config.min_ngram..=max_n).rev() {
            let suffix = &context[len - n..];
            if let Some(start) = latest_earlier_match(context, suffix) {
                let pred_start = start + n;
                let end = (pred_start + self.config.num_speculative).min(len);
                debug_assert!(pred_start < end, "match must leave a following token");
                return context[pred_start..end].to_vec();
            }
        }
        Vec::new()
    }
}

impl crate::speculative::SpeculativeProposer for NgramProposer {
    fn propose(&self, context: &[u32]) -> Vec<u32> {
        NgramProposer::propose(self, context)
    }
}

/// Find the latest start index `i < len - n` such that `context[i..i + n]`
/// equals `suffix` (the trailing `n` tokens). "Latest" prefers the most recent
/// context. Guaranteed to leave at least one following token at `i + n`.
fn latest_earlier_match(context: &[u32], suffix: &[u32]) -> Option<usize> {
    let n = suffix.len();
    let len = context.len();
    if n == 0 || len < n + 1 {
        return None;
    }
    // Candidate starts run from `len - n - 1` (immediately before the trailing
    // suffix) down to 0; the trailing occurrence itself (start `len - n`) is
    // excluded so we never "predict" from the suffix we are matching.
    (0..len - n).rev().find(|&i| &context[i..i + n] == suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposer(max_ngram: usize, min_ngram: usize, k: usize) -> NgramProposer {
        NgramProposer::new(NgramConfig {
            max_ngram,
            min_ngram,
            num_speculative: k,
        })
    }

    #[test]
    fn proposes_continuation_after_recent_match() {
        // ...1 2 3 1 2 3 1 2 ; suffix [1,2] last matched at index 3 -> follow [3,1,2]
        let ctx = [1u32, 2, 3, 1, 2, 3, 1, 2];
        let got = proposer(2, 1, 4).propose(&ctx);
        assert_eq!(got, vec![3, 1, 2]);
    }

    #[test]
    fn prefers_longest_suffix_match() {
        // Both [9] (len1) and [4,9] (len2) recur, but the len-2 match is more
        // specific and must win.
        let ctx = [4u32, 9, 7, 1, 9, 0, 4, 9];
        // suffix len2 = [4,9] earlier at index 0 -> follow [7,1,...]
        let got = proposer(2, 1, 2).propose(&ctx);
        assert_eq!(got, vec![7, 1]);
    }

    #[test]
    fn falls_back_to_shorter_suffix() {
        // No repeated 2-gram suffix, but the last token 5 occurred earlier.
        let ctx = [5u32, 8, 2, 7, 5];
        // suffix len2 = [7,5] has no earlier match; len1 = [5] matches at 0 -> [8]
        let got = proposer(2, 1, 3).propose(&ctx);
        assert_eq!(got, vec![8, 2, 7]);
    }

    #[test]
    fn returns_empty_when_no_match() {
        let ctx = [1u32, 2, 3, 4, 5];
        assert!(proposer(3, 1, 4).propose(&ctx).is_empty());
    }

    #[test]
    fn caps_proposal_at_num_speculative() {
        let ctx = [1u32, 2, 1, 2, 1, 2, 1, 2];
        let got = proposer(2, 1, 1).propose(&ctx);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn handles_short_context() {
        assert!(proposer(3, 1, 4).propose(&[]).is_empty());
        assert!(proposer(3, 1, 4).propose(&[7]).is_empty());
    }

    #[test]
    fn zero_speculative_proposes_nothing() {
        let ctx = [1u32, 2, 1, 2];
        assert!(proposer(2, 1, 0).propose(&ctx).is_empty());
    }

    #[test]
    fn new_clamps_invalid_config() {
        // min_ngram > max_ngram and zero max are clamped, not panicked.
        let p = NgramProposer::new(NgramConfig {
            max_ngram: 0,
            min_ngram: 5,
            num_speculative: 2,
        });
        // With max_ngram clamped to 1, a single-token recurrence still works:
        // suffix [3] recurs at index 0, so the follower [1, 3] is proposed.
        let ctx = [3u32, 1, 3];
        assert_eq!(p.propose(&ctx), vec![1, 3]);
    }
}
