#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    /// Greedy means argmax: temperature below the sampling epsilon (the
    /// temperature -> 0 limit is argmax regardless of top_p, and 1/temperature
    /// overflows long before that; vLLM draws the same line at 1e-5) or
    /// top_k == 1 (a single token survives the mask). Everything else requires
    /// a real sampling pass.
    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-5 || self.top_k == 1
    }
}
