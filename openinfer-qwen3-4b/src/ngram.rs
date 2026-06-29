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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgramConfig {
    /// Largest suffix length to match. Longer suffixes are tried first because
    /// they are more specific (a longer match is a stronger predictor).
    pub max_ngram: usize,
    /// Smallest suffix length to match before giving up. Must be `>= 1`.
    pub min_ngram: usize,
    /// Maximum number of speculative tokens proposed per step.
    pub num_speculative: usize,
    /// Mean accepted-draft-tokens-per-step below which speculation is judged not
    /// worth its verify forward, so the engine falls back to plain decode (see
    /// [`NgramGate`]). On non-repetitive text (e.g. prose) prompt-lookup drafts
    /// are almost never accepted, and at higher concurrency the wasted verify
    /// compute is a net throughput loss; gating recovers it. `0.0` disables the
    /// gate (always speculate).
    pub accept_threshold: f32,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            max_ngram: 3,
            min_ngram: 1,
            num_speculative: 4,
            accept_threshold: 0.3,
        }
    }
}

impl std::fmt::Display for NgramConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "K={}, max_ngram={}, accept_threshold={}",
            self.num_speculative, self.max_ngram, self.accept_threshold
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
    /// * `OPENINFER_QWEN3_NGRAM_ACCEPT_THRESHOLD` = gate threshold (`0` disables).
    #[must_use]
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            max_ngram: env_usize("OPENINFER_QWEN3_NGRAM_MAX_NGRAM").unwrap_or(defaults.max_ngram),
            min_ngram: defaults.min_ngram,
            num_speculative: env_usize("OPENINFER_QWEN3_NGRAM_TOKENS")
                .unwrap_or(defaults.num_speculative),
            accept_threshold: env_f32("OPENINFER_QWEN3_NGRAM_ACCEPT_THRESHOLD")
                .unwrap_or(defaults.accept_threshold),
        }
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Stateless n-gram / prompt-lookup proposer.
///
/// The proposer keeps no history of its own; each call scans the supplied
/// context, so the caller is free to reuse a single instance across requests.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NgramProposer {
    config: NgramConfig,
}

impl NgramProposer {
    /// Create a proposer. `min_ngram` is clamped to at least 1 and to at most
    /// `max_ngram`, and `max_ngram` to at least 1, so the config is always
    /// usable regardless of caller input.
    #[must_use]
    pub(crate) fn new(config: NgramConfig) -> Self {
        let max_ngram = config.max_ngram.max(1);
        let min_ngram = config.min_ngram.clamp(1, max_ngram);
        Self {
            config: NgramConfig {
                max_ngram,
                min_ngram,
                ..config
            },
        }
    }

    /// The (clamped) configuration this proposer was built with.
    #[must_use]
    pub(crate) fn config(&self) -> NgramConfig {
        self.config
    }

    /// Propose up to `num_speculative` continuation tokens for `context`
    /// (the full token sequence so far: prompt followed by generated tokens).
    ///
    /// Returns an empty `Vec` when no usable n-gram match exists or when
    /// `num_speculative == 0`. Tries the longest configured suffix first and
    /// falls back to shorter suffixes.
    ///
    /// Cost: each call is a linear reverse scan of `context` per suffix length,
    /// so a miss is `O(context_len * max_ngram)` on the executor thread, every
    /// speculative step. Fine for the target case (repetitive / bounded
    /// contexts); a very long context could erode the speedup it buys. If that
    /// becomes a concern, bound the look-back window (as vLLM's prompt-lookup
    /// does) rather than scanning the whole history.
    #[must_use]
    pub(crate) fn propose(&self, context: &[u32]) -> Vec<u32> {
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

/// Engine-wide switch that decides whether n-gram speculation is currently
/// paying for itself.
///
/// Speculation is only a win when enough drafted tokens are accepted to offset
/// the verify forward (which costs ~`K + 1`× a plain decode). On repetitive text
/// acceptance is high and this stays open; on prose it collapses to near zero, so
/// the gate closes and decoding falls back to plain — recovering the throughput
/// the wasted verify would otherwise burn, which matters most at the higher batch
/// sizes where the GPU has no spare compute to hide it.
///
/// The estimate is a single EWMA of accepted draft tokens per step, updated only
/// when we actually drafted. While closed the gate **probes** once every
/// `PROBE_INTERVAL` steps so a shift into repetitive text re-opens it. An initial
/// warmup keeps it open until there is enough data to judge.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NgramGate {
    /// EWMA of accepted draft tokens per drafted step.
    accept_ewma: f32,
    /// Drafted steps observed so far (capped at `WARMUP_STEPS`).
    warmup_left: u32,
    /// Steps elapsed since the gate last allowed a draft (drives probing).
    since_draft: u32,
}

impl NgramGate {
    /// Smoothing for the acceptance EWMA: low enough to react within a handful
    /// of steps when a request switches between repetitive and prose regions.
    const EWMA_ALPHA: f32 = 0.2;
    /// Always speculate for this many drafted steps before trusting the EWMA.
    const WARMUP_STEPS: u32 = 8;
    /// While closed, allow one probing draft every this many steps.
    const PROBE_INTERVAL: u32 = 32;

    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            accept_ewma: 0.0,
            warmup_left: Self::WARMUP_STEPS,
            since_draft: 0,
        }
    }

    /// Whether this step should draft. Open during warmup, while recent
    /// acceptance clears the threshold, or on a periodic probe. A non-positive
    /// threshold disables gating (always open).
    #[must_use]
    pub(crate) fn should_draft(&self, threshold: f32) -> bool {
        threshold <= 0.0
            || self.warmup_left > 0
            || self.accept_ewma >= threshold
            || self.since_draft >= Self::PROBE_INTERVAL
    }

    /// Record the outcome of a step that drafted: `accepted` is the mean number
    /// of draft tokens verify took across the step's drafted requests. Resets the
    /// probe clock and folds the count into the EWMA.
    pub(crate) fn record_drafted(&mut self, accepted: f32) {
        self.since_draft = 0;
        self.warmup_left = self.warmup_left.saturating_sub(1);
        self.accept_ewma += Self::EWMA_ALPHA * (accepted - self.accept_ewma);
    }

    /// Record a step that did not draft (gate closed, or no match found), so the
    /// probe clock advances toward the next re-check.
    pub(crate) fn record_skipped(&mut self) {
        self.since_draft = self.since_draft.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposer(max_ngram: usize, min_ngram: usize, k: usize) -> NgramProposer {
        NgramProposer::new(NgramConfig {
            max_ngram,
            min_ngram,
            num_speculative: k,
            ..NgramConfig::default()
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
            ..NgramConfig::default()
        });
        // With max_ngram clamped to 1, a single-token recurrence still works:
        // suffix [3] recurs at index 0, so the follower [1, 3] is proposed.
        let ctx = [3u32, 1, 3];
        assert_eq!(p.propose(&ctx), vec![1, 3]);
    }

    #[test]
    fn gate_disabled_with_nonpositive_threshold() {
        let mut gate = NgramGate::new();
        // Drive acceptance to zero; a zero threshold must keep speculating.
        for _ in 0..50 {
            gate.record_drafted(0.0);
        }
        assert!(gate.should_draft(0.0));
    }

    #[test]
    fn gate_closes_after_warmup_on_low_acceptance() {
        let mut gate = NgramGate::new();
        // Warmup keeps it open even while acceptance reads zero.
        for _ in 0..NgramGate::WARMUP_STEPS {
            assert!(gate.should_draft(0.3));
            gate.record_drafted(0.0);
        }
        // Warmup spent + EWMA below threshold -> closed.
        assert!(!gate.should_draft(0.3));
    }

    #[test]
    fn gate_stays_open_on_high_acceptance() {
        let mut gate = NgramGate::new();
        for _ in 0..20 {
            gate.record_drafted(3.0);
        }
        assert!(gate.should_draft(0.3));
    }

    #[test]
    fn gate_probes_after_cooldown_then_recovers() {
        let mut gate = NgramGate::new();
        for _ in 0..NgramGate::WARMUP_STEPS {
            gate.record_drafted(0.0);
        }
        assert!(!gate.should_draft(0.3), "closed after low-acceptance warmup");
        // Skipping for PROBE_INTERVAL steps re-opens it for one probe.
        for _ in 0..NgramGate::PROBE_INTERVAL {
            gate.record_skipped();
        }
        assert!(gate.should_draft(0.3), "probe re-opens the gate");
        // A probe that finds repetitive text (high acceptance) keeps it open.
        gate.record_drafted(4.0);
        assert!(gate.should_draft(0.3));
    }
}
