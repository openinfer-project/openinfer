use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, ensure};
use pegainfer_engine::engine::{
    EngineHandle, FinishReason, GenerateRequest, TokenEvent, TokenLogprob,
};
use tokio::sync::mpsc;

#[derive(Clone, Copy, Debug)]
pub struct SimulatedEngineConfig {
    base_ttft_ms: f64,
    prefill_tokens_per_ms: f64,
    tpot_ms: f64,
    fallback_token_id: u32,
}

impl SimulatedEngineConfig {
    pub fn new(
        base_ttft_ms: f64,
        prefill_tokens_per_ms: f64,
        tpot_ms: f64,
        fallback_token_id: u32,
    ) -> Result<Self> {
        ensure!(
            base_ttft_ms.is_finite() && base_ttft_ms >= 0.0,
            "base TTFT must be finite and non-negative"
        );
        ensure!(
            prefill_tokens_per_ms.is_finite() && prefill_tokens_per_ms > 0.0,
            "prefill throughput must be finite and positive"
        );
        ensure!(
            tpot_ms.is_finite() && tpot_ms >= 0.0,
            "TPOT must be finite and non-negative"
        );

        Ok(Self {
            base_ttft_ms,
            prefill_tokens_per_ms,
            tpot_ms,
            fallback_token_id,
        })
    }

    fn ttft(&self, prompt_tokens: usize) -> Duration {
        duration_from_ms(self.base_ttft_ms + prompt_tokens as f64 / self.prefill_tokens_per_ms)
    }

    fn tpot(&self) -> Duration {
        duration_from_ms(self.tpot_ms)
    }
}

impl Default for SimulatedEngineConfig {
    fn default() -> Self {
        Self {
            base_ttft_ms: 5.0,
            prefill_tokens_per_ms: 100.0,
            tpot_ms: 12.0,
            fallback_token_id: 0,
        }
    }
}

pub fn start_engine(config: SimulatedEngineConfig) -> EngineHandle {
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(req) = submit_rx.recv().await {
            tokio::spawn(run_simulated_request(req, config));
        }
    });
    EngineHandle::new(submit_tx)
}

async fn run_simulated_request(req: GenerateRequest, config: SimulatedEngineConfig) {
    let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(now_secs_f64);
    let prompt_len = req.prompt_tokens.len();
    let mut completion_tokens = 0;

    if req
        .token_tx
        .send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s: now_secs_f64(),
            prompt_tokens: prompt_len,
        })
        .is_err()
    {
        return;
    }

    if req.echo
        && req
            .token_tx
            .send(TokenEvent::PromptTokens {
                ids: req.prompt_tokens.clone(),
                logprobs: vec![None; req.prompt_tokens.len()],
            })
            .is_err()
    {
        return;
    }

    if req.max_tokens > 0 {
        tokio::time::sleep(config.ttft(prompt_len)).await;
    }

    for index in 0..req.max_tokens {
        if index > 0 {
            tokio::time::sleep(config.tpot()).await;
        }

        let logprob = (req.logprobs > 0).then_some(TokenLogprob {
            logprob: 0.0,
            top_logprobs: Vec::new(),
        });
        if req
            .token_tx
            .send(TokenEvent::Token {
                id: fake_token_id(&req.prompt_tokens, index, config.fallback_token_id),
                logprob,
            })
            .is_err()
        {
            return;
        }
        completion_tokens += 1;
    }

    let _ = req.token_tx.send(TokenEvent::Finished {
        finish_reason: FinishReason::Length,
        prompt_tokens: prompt_len,
        completion_tokens,
    });
}

fn fake_token_id(prompt_tokens: &[u32], index: usize, fallback_token_id: u32) -> u32 {
    if prompt_tokens.is_empty() {
        return fallback_token_id;
    }
    prompt_tokens[index % prompt_tokens.len()]
}

fn duration_from_ms(ms: f64) -> Duration {
    Duration::from_secs_f64(ms / 1000.0)
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use pegainfer_engine::sampler::SamplingParams;

    use super::*;

    #[test]
    fn fake_token_id_cycles_prompt_tokens() {
        assert_eq!(fake_token_id(&[7, 9], 0, 42), 7);
        assert_eq!(fake_token_id(&[7, 9], 1, 42), 9);
        assert_eq!(fake_token_id(&[7, 9], 2, 42), 7);
        assert_eq!(fake_token_id(&[], 0, 42), 42);
    }

    #[test]
    fn ttft_combines_floor_and_prefill_time() {
        let config = SimulatedEngineConfig::new(5.0, 100.0, 12.0, 0).unwrap();

        assert_eq!(config.ttft(1_024), Duration::from_micros(15_240));
        assert_eq!(config.tpot(), Duration::from_millis(12));
    }

    #[test]
    fn config_rejects_invalid_timing_values() {
        assert!(SimulatedEngineConfig::new(-1.0, 100.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 0.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 100.0, -1.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(f64::NAN, 100.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, f64::INFINITY, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 100.0, f64::INFINITY, 0).is_err());
    }

    #[tokio::test]
    async fn simulated_request_emits_scheduled_tokens_and_finished() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 42).unwrap();
        let (token_tx, mut token_rx) = mpsc::unbounded_channel();

        run_simulated_request(
            GenerateRequest {
                request_id: Some("req-1".to_string()),
                queued_at_unix_s: Some(1.0),
                prompt_tokens: vec![7, 9],
                params: SamplingParams::default(),
                max_tokens: 3,
                token_tx,
                logprobs: 1,
                echo: false,
            },
            config,
        )
        .await;

        assert!(matches!(
            token_rx.recv().await,
            Some(TokenEvent::Scheduled {
                prompt_tokens: 2,
                ..
            })
        ));
        assert!(matches!(
            token_rx.recv().await,
            Some(TokenEvent::Token {
                id: 7,
                logprob: Some(_)
            })
        ));
        assert!(matches!(
            token_rx.recv().await,
            Some(TokenEvent::Token {
                id: 9,
                logprob: Some(_)
            })
        ));
        assert!(matches!(
            token_rx.recv().await,
            Some(TokenEvent::Token {
                id: 7,
                logprob: Some(_)
            })
        ));
        assert!(matches!(
            token_rx.recv().await,
            Some(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: 2,
                completion_tokens: 3
            })
        ));
    }
}
