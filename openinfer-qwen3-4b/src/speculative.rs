//! Method-agnostic core of speculative decoding for Qwen3.
//!
//! Three pieces live here, none tied to a specific proposer:
//! * [`SpeculativeProposer`] — the draft-token source (the one axis that varies
//!   between methods; [`crate::ngram::NgramProposer`] is the first impl).
//! * [`SpeculativeConfig`] / [`SpeculativeMethod`] — what to run and how to
//!   build the proposer ([`SpeculativeConfig::build_proposer`]).
//! * [`accept_greedy`] — greedy acceptance: given the `K` drafts and the target
//!   model's greedy token at each of the `K + 1` verify positions, return the
//!   tokens to commit.
//!
//! The GPU verify forward and KV reserve/rollback that feed `accept_greedy`
//! live in the decode path (executor) and are intentionally not part of this
//! module, so the acceptance rule stays pure and unit-tested.

use crate::ngram::{NgramConfig, NgramProposer};

/// A draft-token source for speculative decoding.
///
/// This is the single axis of variation between speculative methods: n-gram /
/// prompt-lookup today, draft-model or EAGLE/Medusa later. Everything around it
/// — the verify forward, KV reserve/rollback, greedy acceptance, and the
/// scheduler step that streams committed tokens — is method-agnostic and is
/// reused unchanged.
///
/// The context is the request's full token sequence so far (prompt + generated
/// tokens). Returning an empty `Vec` means "no draft this step", and the decode
/// path falls back to a single-token decode. A token-only context fits n-gram
/// and draft-model proposers; hidden-state proposers (EAGLE/Medusa) will need a
/// richer context (request id + hidden states), which is a deliberate future
/// extension rather than something modelled speculatively here.
pub trait SpeculativeProposer: Send + Sync {
    /// Propose up to `K` continuation tokens for `context`.
    fn propose(&self, context: &[u32]) -> Vec<u32>;
}

/// Which proposer to build. Add a variant per new speculative method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculativeMethod {
    /// N-gram / prompt-lookup (no draft model).
    Ngram(NgramConfig),
}

impl Default for SpeculativeMethod {
    fn default() -> Self {
        Self::Ngram(NgramConfig::default())
    }
}

/// Speculative-decoding configuration.
///
/// `Default` is disabled with the default method (`enabled: false`), so the
/// decode path runs the normal one-token step until explicitly turned on.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct SpeculativeConfig {
    /// Master switch. When `false` the decode path runs the normal one-token
    /// step and never builds a proposer.
    pub enabled: bool,
    /// Which proposer to use (ignored when `enabled` is `false`).
    pub method: SpeculativeMethod,
}

impl SpeculativeConfig {
    /// Read the config from the environment (the operational switch until a
    /// first-class server/config knob is wired). This owns only the generic
    /// switch; each method parses its own knobs (see [`NgramConfig::from_env`]):
    ///
    /// * `OPENINFER_QWEN3_SPEC` = `1`/`true` enables speculation (default off).
    ///
    /// The method defaults to n-gram; add a `OPENINFER_QWEN3_SPEC_METHOD`
    /// dispatch here when a second proposer lands.
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var("OPENINFER_QWEN3_SPEC")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        Self {
            enabled,
            method: SpeculativeMethod::Ngram(NgramConfig::from_env()),
        }
    }

    /// Construct the proposer described by [`Self::method`]. The scheduler calls
    /// this once at startup and reuses the boxed proposer across all requests.
    #[must_use]
    pub fn build_proposer(&self) -> Box<dyn SpeculativeProposer> {
        match self.method {
            SpeculativeMethod::Ngram(cfg) => Box::new(NgramProposer::new(cfg)),
        }
    }

    /// Human-readable one-liner for startup logging.
    #[must_use]
    pub fn describe(&self) -> String {
        match self.method {
            SpeculativeMethod::Ngram(cfg) => format!(
                "n-gram (K={}, max_ngram={})",
                cfg.num_speculative, cfg.max_ngram
            ),
        }
    }
}

/// Greedy speculative acceptance.
///
/// * `proposed` — the `K` candidate tokens from the proposer.
/// * `target_argmax` — the target model's greedy token at each of the `K + 1`
///   verify positions. `target_argmax[i]` is the model's prediction *after*
///   consuming verify input `i`; `target_argmax[0]` is the token that follows
///   the last confirmed token, and `target_argmax[K]` is the model's own
///   continuation after the whole candidate run.
///
/// Returns the tokens to commit: the longest prefix of `proposed` that the
/// model agrees with, followed by exactly one model token — the correction at
/// the first divergence, or the bonus continuation when every candidate is
/// accepted. The result is therefore always between `1` and `K + 1` tokens, so
/// a verify step always makes at least one token of progress.
///
/// # Panics
/// Panics (debug builds) if `target_argmax.len() != proposed.len() + 1`.
#[must_use]
pub fn accept_greedy(proposed: &[u32], target_argmax: &[u32]) -> Vec<u32> {
    debug_assert_eq!(
        target_argmax.len(),
        proposed.len() + 1,
        "verify must produce one greedy token per candidate plus a bonus"
    );
    let mut committed = Vec::with_capacity(proposed.len() + 1);
    let mut i = 0;
    while i < proposed.len() && proposed[i] == target_argmax[i] {
        committed.push(proposed[i]);
        i += 1;
    }
    // The model's own token at the first divergence (or the bonus continuation
    // when the whole run was accepted). `i <= proposed.len() < target_argmax.len()`
    // so this index is always valid.
    committed.push(target_argmax[i]);
    committed
}

/// Number of candidate tokens accepted before the committed bonus token, i.e.
/// `accept_greedy(..).len() - 1`. Useful for KV rollback bookkeeping (how many
/// of the speculatively written candidate positions to keep).
#[must_use]
pub fn num_accepted(proposed: &[u32], target_argmax: &[u32]) -> usize {
    let mut i = 0;
    while i < proposed.len() && proposed[i] == target_argmax[i] {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_full_run_plus_bonus() {
        // Model agrees with all 3 candidates, then emits its own bonus token.
        let proposed = [10u32, 11, 12];
        let argmax = [10u32, 11, 12, 13];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![10, 11, 12, 13]);
        assert_eq!(num_accepted(&proposed, &argmax), 3);
    }

    #[test]
    fn accepts_prefix_then_correction() {
        // Candidates 10,11 match; 99 diverges (model wanted 22) -> commit 10,11,22.
        let proposed = [10u32, 11, 99];
        let argmax = [10u32, 11, 22, 33];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![10, 11, 22]);
        assert_eq!(num_accepted(&proposed, &argmax), 2);
    }

    #[test]
    fn rejects_first_candidate_commits_one() {
        // First candidate already wrong -> only the model's own token commits.
        let proposed = [10u32, 11, 12];
        let argmax = [7u32, 8, 9, 10];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![7]);
        assert_eq!(num_accepted(&proposed, &argmax), 0);
    }

    #[test]
    fn empty_proposal_commits_model_token() {
        // No candidates (proposer returned nothing): plain one-token decode.
        let proposed: [u32; 0] = [];
        let argmax = [42u32];
        assert_eq!(accept_greedy(&proposed, &argmax), vec![42]);
        assert_eq!(num_accepted(&proposed, &argmax), 0);
    }

    #[test]
    fn always_commits_at_least_one_token() {
        let proposed = [1u32, 2];
        let argmax = [9u32, 9, 9];
        assert!(!accept_greedy(&proposed, &argmax).is_empty());
    }

    #[test]
    fn config_default_is_disabled() {
        assert!(!SpeculativeConfig::default().enabled);
    }
}
