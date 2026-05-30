//! Kimi-K2 expert-parallel (EP) topology constants.

use crate::config::KIMI_K2_ROUTED_EXPERTS;

/// EP world size for the Kimi-K2 expert-parallel contract (EP8).
pub const KIMI_K2_EP_WORLD: usize = 8;

/// Routed experts owned by each EP8 rank.
pub const KIMI_K2_EP8_LOCAL_EXPERTS: usize = KIMI_K2_ROUTED_EXPERTS / KIMI_K2_EP_WORLD;
