use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use anyhow::ensure;
use openinfer_engine::engine::EngineHandle;
use openinfer_engine::engine::FinishReason;
use openinfer_engine::engine::GenerateRequest;
use openinfer_engine::engine::TokenEvent;
use openinfer_engine::engine::TokenLogprob;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct SimulatedEngineConfig {
    base_ttft_ms: f64,
    prefill_tokens_per_ms: f64,
    tpot_ms: f64,
    fallback_token_id: u32,
    /// Explicit completion token-id sequence to replay verbatim. Empty (the
    /// default) keeps the legacy behaviour of cycling the prompt tokens.
    scripted_completion: Vec<u32>,
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
            scripted_completion: Vec::new(),
        })
    }

    /// Replay `ids` verbatim as the completion for every request
    #[must_use]
    pub fn with_scripted_completion(mut self, ids: Vec<u32>) -> Self {
        self.scripted_completion = ids;
        self
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
            scripted_completion: Vec::new(),
        }
    }
}

pub fn start_engine(config: SimulatedEngineConfig) -> EngineHandle {
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(req) = submit_rx.recv().await {
            tokio::spawn(run_simulated_request(req, config.clone()));
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
            cached_tokens: 0,
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

    let script = &config.scripted_completion;
    let emit_count = if script.is_empty() {
        req.max_tokens
    } else {
        req.max_tokens.min(script.len())
    };

    if emit_count > 0 {
        tokio::time::sleep(config.ttft(prompt_len)).await;
    }

    for index in 0..emit_count {
        if index > 0 {
            tokio::time::sleep(config.tpot()).await;
        }

        let logprob = (req.logprobs > 0).then_some(TokenLogprob {
            logprob: 0.0,
            top_logprobs: Vec::new(),
        });
        let id = if script.is_empty() {
            fake_token_id(&req.prompt_tokens, index, config.fallback_token_id)
        } else {
            script[index]
        };
        if req
            .token_tx
            .send(TokenEvent::Token { id, logprob })
            .is_err()
        {
            return;
        }
        completion_tokens += 1;
    }

    let finish_reason = if !script.is_empty() && emit_count == script.len() {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };
    let _ = req.token_tx.send(TokenEvent::Finished {
        finish_reason,
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
    use openinfer_engine::engine::TokenSink;
    use openinfer_engine::sampler::SamplingParams;

    use super::*;

    #[test]
    fn fake_token_id_cycles_prompt_tokens() {
        assert_eq!(fake_token_id(&[7, 9], 0, 42), 7);
        assert_eq!(fake_token_id(&[7, 9], 1, 42), 9);
        assert_eq!(fake_token_id(&[7, 9], 2, 42), 7);
        assert_eq!(fake_token_id(&[], 0, 42), 42);
    }

    #[tokio::test]
    async fn scripted_completion_replays_ids_and_stops() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 0)
            .unwrap()
            .with_scripted_completion(vec![11, 22, 33]);
        let (token_tx, mut token_rx) = TokenSink::standalone();

        run_simulated_request(
            GenerateRequest {
                trace_parent: None,
                request_id: Some("req-scripted".to_string()),
                queued_at_unix_s: Some(1.0),
                data_parallel_rank: None,
                // Prompt is irrelevant in scripted mode; output ignores it.
                prompt_tokens: vec![7, 9],
                params: SamplingParams::default(),
                max_tokens: 8,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            config,
        )
        .await;

        assert!(matches!(
            token_rx.recv().await.map(|(_, e)| e),
            Some(TokenEvent::Scheduled { .. })
        ));
        for expected in [11u32, 22, 33] {
            assert!(matches!(
                token_rx.recv().await.map(|(_, e)| e),
                Some(TokenEvent::Token { id, .. }) if id == expected
            ));
        }
        // Whole script fit inside max_tokens -> Stop, not Length.
        assert!(matches!(
            token_rx.recv().await.map(|(_, e)| e),
            Some(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                completion_tokens: 3,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn scripted_completion_truncated_by_max_tokens_is_length() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 0)
            .unwrap()
            .with_scripted_completion(vec![11, 22, 33]);
        let (token_tx, mut token_rx) = TokenSink::standalone();

        run_simulated_request(
            GenerateRequest {
                trace_parent: None,
                request_id: Some("req-trunc".to_string()),
                queued_at_unix_s: Some(1.0),
                data_parallel_rank: None,
                prompt_tokens: vec![7],
                params: SamplingParams::default(),
                max_tokens: 2,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            config,
        )
        .await;

        assert!(matches!(
            token_rx.recv().await.map(|(_, e)| e),
            Some(TokenEvent::Scheduled { .. })
        ));
        for expected in [11u32, 22] {
            assert!(matches!(
                token_rx.recv().await.map(|(_, e)| e),
                Some(TokenEvent::Token { id, .. }) if id == expected
            ));
        }
        assert!(matches!(
            token_rx.recv().await.map(|(_, e)| e),
            Some(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                completion_tokens: 2,
                ..
            })
        ));
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
        let (token_tx, mut token_rx) = TokenSink::standalone();

        run_simulated_request(
            GenerateRequest {
                trace_parent: None,
                request_id: Some("req-1".to_string()),
                queued_at_unix_s: Some(1.0),
                data_parallel_rank: None,
                prompt_tokens: vec![7, 9],
                params: SamplingParams::default(),
                max_tokens: 3,
                lora_adapter: None,
                token_tx,
                logprobs: 1,
                echo: false,
            },
            config,
        )
        .await;

        assert!(matches!(
            token_rx.recv().await.map(|(_, event)| event),
            Some(TokenEvent::Scheduled {
                prompt_tokens: 2,
                ..
            })
        ));
        assert!(matches!(
            token_rx.recv().await.map(|(_, event)| event),
            Some(TokenEvent::Token {
                id: 7,
                logprob: Some(_)
            })
        ));
        assert!(matches!(
            token_rx.recv().await.map(|(_, event)| event),
            Some(TokenEvent::Token {
                id: 9,
                logprob: Some(_)
            })
        ));
        assert!(matches!(
            token_rx.recv().await.map(|(_, event)| event),
            Some(TokenEvent::Token {
                id: 7,
                logprob: Some(_)
            })
        ));
        assert!(matches!(
            token_rx.recv().await.map(|(_, event)| event),
            Some(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: 2,
                completion_tokens: 3
            })
        ));
    }
}
