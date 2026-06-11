use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use openinfer_engine::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent,
};
use tokio::sync::mpsc;

use crate::runtime::{DeepSeekV2LiteEp2Generator, GenerationResult};

pub(crate) fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let mut generator = DeepSeekV2LiteEp2Generator::load(model_path, options)?;
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel();

    let join_handle = std::thread::Builder::new()
        .name("deepseek-v2-lite-ep2".to_string())
        .spawn(move || {
            while let Some(req) = submit_rx.blocking_recv() {
                handle_request(&mut generator, &req);
            }
        })
        .context("spawn DeepSeek-V2-Lite EP=2 engine thread")?;

    Ok(EngineHandle::new_with_join_handle(submit_tx, join_handle))
}

fn handle_request(generator: &mut DeepSeekV2LiteEp2Generator, req: &GenerateRequest) {
    let prompt_tokens = req.prompt_tokens.len();
    let now = unix_time_secs();
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s: req.queued_at_unix_s.unwrap_or(now),
        scheduled_at_unix_s: now,
        prompt_tokens,
        cached_tokens: 0,
    });
    if req.echo {
        let _ = req.token_tx.send(TokenEvent::PromptTokens {
            ids: req.prompt_tokens.clone(),
            logprobs: vec![None; prompt_tokens],
        });
    }
    if !req.params.is_greedy() {
        reject_request(
            req,
            prompt_tokens,
            format!(
                "DeepSeek-V2-Lite EP=2 first gate serves greedy decoding only; requested temperature={}, top_k={}, top_p={}",
                req.params.temperature, req.params.top_k, req.params.top_p
            ),
        );
        return;
    }
    if req.logprobs > 0 {
        reject_request(
            req,
            prompt_tokens,
            "DeepSeek-V2-Lite EP=2 first gate does not return logprobs yet".to_string(),
        );
        return;
    }
    if req.max_tokens == 0 {
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens,
            completion_tokens: 0,
        });
        return;
    }

    match generator.generate_greedy(&req.prompt_tokens, req.max_tokens, req.params.ignore_eos) {
        Ok(result) => {
            emit_generation_result(&req.token_tx, prompt_tokens, &result);
        }
        Err(err) => {
            let _ = req.token_tx.send(TokenEvent::Error {
                message: err.to_string(),
                prompt_tokens,
                completion_tokens: 0,
            });
        }
    }
}

fn reject_request(req: &GenerateRequest, prompt_tokens: usize, message: String) {
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message,
        prompt_tokens,
        completion_tokens: 0,
    });
}

fn emit_generation_result(
    token_tx: &mpsc::UnboundedSender<TokenEvent>,
    prompt_tokens: usize,
    result: &GenerationResult,
) {
    for token in &result.tokens {
        let _ = token_tx.send(TokenEvent::Token {
            id: *token,
            logprob: None,
        });
    }
    let _ = token_tx.send(TokenEvent::Finished {
        finish_reason: result.finish_reason,
        prompt_tokens,
        completion_tokens: result.tokens.len(),
    });
}

fn unix_time_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::GenerationStats;

    #[test]
    fn stop_generation_streams_tokens_and_stop_finish() {
        let (tx, mut rx) = mpsc::unbounded_channel();

        emit_generation_result(
            &tx,
            3,
            &GenerationResult {
                tokens: vec![10, 11],
                finish_reason: FinishReason::Stop,
                stats: GenerationStats::default(),
            },
        );
        drop(tx);

        match rx.try_recv().expect("expected first token") {
            TokenEvent::Token { id, .. } => assert_eq!(id, 10),
            _ => panic!("expected first token event"),
        }
        match rx.try_recv().expect("expected second token") {
            TokenEvent::Token { id, .. } => assert_eq!(id, 11),
            _ => panic!("expected second token event"),
        }
        match rx.try_recv().expect("expected finished event") {
            TokenEvent::Finished {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                assert_eq!(finish_reason, FinishReason::Stop);
                assert_eq!(prompt_tokens, 3);
                assert_eq!(completion_tokens, 2);
            }
            _ => panic!("expected finished event"),
        }
        assert!(rx.try_recv().is_err());
    }
}
