use anyhow::Result;
use rand::rngs::StdRng;

use crate::sampler::SamplingParams;
use crate::tensor::DeviceVec;

/// Per-request mutable generation state.
pub trait GenerationState {
    fn logits(&self) -> &DeviceVec;
    fn reset(&mut self) -> Result<()>;
}

/// Minimal direct-request model interface used by tests and in-process tools.
pub trait ModelForward: Send {
    type State: GenerationState + Send;

    fn create_state(&self) -> Result<Self::State>;
    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;
    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32>;
    fn is_stop_token(&self, token_id: u32) -> bool;
}
