use std::{path::Path, sync::mpsc as std_mpsc, thread};

use anyhow::{Context, Result, bail};
use log::{info, warn};
use pegainfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent,
};
use tokio::sync::mpsc;

use super::worker::{
    FullDirectRuntime, ensure_direct_decode_caches, load_full_direct_runtime,
    run_direct_decode_logits, run_prefill_logits_and_seed_decode_cache,
};
use crate::Config;

pub struct DirectGeneration {
    pub generated: Vec<u32>,
    pub finish_reason: FinishReason,
}

pub struct DeepSeekV4RequestState {
    request_epoch: u64,
    prompt_len: usize,
    max_new_tokens: usize,
    ignore_eos: bool,
    generated: Vec<u32>,
    next_logits: Option<Vec<f32>>,
    finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug)]
pub struct DirectDecodeStep {
    request_epoch: u64,
    generated_len_before: usize,
    prompt_len: usize,
    token: Option<u32>,
    finish_reason: Option<FinishReason>,
}

impl DirectDecodeStep {
    pub fn token(&self) -> Option<u32> {
        self.token
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    pub fn generated_len_before(&self) -> usize {
        self.generated_len_before
    }

    pub fn start_pos(&self) -> usize {
        self.prompt_len + self.generated_len_before
    }
}

impl DeepSeekV4RequestState {
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    pub fn completion_tokens(&self) -> usize {
        self.generated.len()
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

pub struct DeepSeekV4DirectGenerator {
    config: &'static Config,
    runtime: FullDirectRuntime,
    next_request_epoch: u64,
}

impl DeepSeekV4DirectGenerator {
    pub fn from_model_dir(model_path: &Path) -> Result<Self> {
        let config = Box::leak(Box::new(Config::from_model_dir(model_path).with_context(
            || {
                format!(
                    "failed to load DeepSeek V4 config from {}",
                    model_path.display()
                )
            },
        )?));
        let runtime = load_full_direct_runtime(model_path, config)?;
        Ok(Self {
            config,
            runtime,
            next_request_epoch: 0,
        })
    }

    pub fn eos_token_id(&self) -> usize {
        self.config.eos_token_id
    }

    pub fn start_greedy_request(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<DeepSeekV4RequestState> {
        if prompt_tokens.is_empty() {
            bail!("DeepSeek V4 request produced an empty prompt");
        }
        let request_epoch = self.next_request_epoch;
        self.next_request_epoch = self
            .next_request_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 request epoch exhausted"))?;

        if max_new_tokens == 0 {
            return Ok(DeepSeekV4RequestState {
                request_epoch,
                prompt_len: prompt_tokens.len(),
                max_new_tokens,
                ignore_eos,
                generated: Vec::new(),
                next_logits: None,
                finish_reason: Some(FinishReason::Length),
            });
        }

        ensure_direct_decode_caches(
            &mut self.runtime,
            self.config,
            prompt_tokens.len() + max_new_tokens,
        )?;

        let next_logits = run_prefill_logits_and_seed_decode_cache(
            &mut self.runtime,
            self.config,
            prompt_tokens,
        )?;

        Ok(DeepSeekV4RequestState {
            request_epoch,
            prompt_len: prompt_tokens.len(),
            max_new_tokens,
            ignore_eos,
            generated: Vec::with_capacity(max_new_tokens),
            next_logits: Some(next_logits),
            finish_reason: None,
        })
    }

    pub fn sample_greedy_step(&self, state: &DeepSeekV4RequestState) -> Result<DirectDecodeStep> {
        if let Some(finish_reason) = state.finish_reason {
            return Ok(DirectDecodeStep {
                request_epoch: state.request_epoch,
                generated_len_before: state.generated.len(),
                prompt_len: state.prompt_len,
                token: None,
                finish_reason: Some(finish_reason),
            });
        }

        let next_logits = state
            .next_logits
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 request state missing next logits"))?;
        let token = argmax_f32(&next_logits) as u32;
        if !state.ignore_eos && token as usize == self.config.eos_token_id {
            return Ok(DirectDecodeStep {
                request_epoch: state.request_epoch,
                generated_len_before: state.generated.len(),
                prompt_len: state.prompt_len,
                token: None,
                finish_reason: Some(FinishReason::Stop),
            });
        }

        let finish_reason =
            (state.generated.len() + 1 == state.max_new_tokens).then_some(FinishReason::Length);
        Ok(DirectDecodeStep {
            request_epoch: state.request_epoch,
            generated_len_before: state.generated.len(),
            prompt_len: state.prompt_len,
            token: Some(token),
            finish_reason,
        })
    }

    pub fn advance_greedy_step(
        &mut self,
        state: &mut DeepSeekV4RequestState,
        step: &DirectDecodeStep,
    ) -> Result<()> {
        ensure_step_matches_state(state, step)?;
        if state.is_finished() {
            return Ok(());
        }

        if let Some(finish_reason) = step.finish_reason()
            && step.token().is_none()
        {
            state.next_logits = None;
            state.finish_reason = Some(finish_reason);
            return Ok(());
        }

        let Some(token) = step.token() else {
            bail!("DeepSeek V4 decode step without token or finish reason");
        };
        state
            .next_logits
            .take()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 request state missing consumed logits"))?;
        state.generated.push(token);
        if let Some(finish_reason) = step.finish_reason() {
            state.finish_reason = Some(finish_reason);
            return Ok(());
        }
        state.next_logits = Some(run_direct_decode_logits(
            &mut self.runtime,
            token,
            step.start_pos(),
        )?);

        Ok(())
    }

    pub fn decode_greedy_step(
        &mut self,
        state: &mut DeepSeekV4RequestState,
    ) -> Result<DirectDecodeStep> {
        let step = self.sample_greedy_step(state)?;
        self.advance_greedy_step(state, &step)?;
        Ok(step)
    }

    pub fn generate_greedy<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        mut on_token: F,
    ) -> Result<DirectGeneration>
    where
        F: FnMut(u32) -> Result<()>,
    {
        let mut state = self.start_greedy_request(prompt_tokens, max_new_tokens, ignore_eos)?;
        while !state.is_finished() {
            let step = self.sample_greedy_step(&state)?;
            if let Some(token) = step.token() {
                on_token(token)?;
            }
            self.advance_greedy_step(&mut state, &step)?;
        }
        Ok(DirectGeneration {
            generated: state.generated,
            finish_reason: state
                .finish_reason
                .expect("DeepSeek V4 request state must finish after greedy generation"),
        })
    }
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    if options.device_ordinals != (0..8).collect::<Vec<_>>() {
        bail!(
            "DeepSeek V4 MP8 currently requires device_ordinals=0..7, got {:?}",
            options.device_ordinals
        );
    }
    if options.enable_cuda_graph {
        warn!("DeepSeek V4 direct engine does not use CUDA graph yet");
    }
    let model_path = model_path.to_path_buf();
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (init_tx, init_rx) = std_mpsc::channel::<Result<()>>();
    thread::Builder::new()
        .name("deepseek-v4-scheduler".into())
        .spawn(move || {
            let mut generator = match DeepSeekV4DirectGenerator::from_model_dir(&model_path) {
                Ok(generator) => {
                    let _ = init_tx.send(Ok(()));
                    generator
                }
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                    return;
                }
            };
            info!("DeepSeek V4 scheduler ready");
            while let Some(req) = submit_rx.blocking_recv() {
                handle_request(&mut generator, req);
            }
            info!("DeepSeek V4 scheduler exiting");
        })
        .expect("failed to spawn DeepSeek V4 scheduler thread");
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 engine init channel closed: {err}"))??;
    Ok(EngineHandle::new(submit_tx))
}

fn handle_request(generator: &mut DeepSeekV4DirectGenerator, req: GenerateRequest) {
    let prompt_len = req.prompt_tokens.len();
    if req.echo {
        let _ = req.token_tx.send(TokenEvent::PromptTokens {
            ids: req.prompt_tokens.clone(),
            logprobs: vec![None; prompt_len],
        });
    }
    if req.params.temperature > 0.0 || req.params.top_k != -1 || req.params.top_p < 1.0 {
        reject_request(
            &req,
            prompt_len,
            format!(
                "DeepSeek V4 direct engine currently serves greedy decoding only; requested temperature={}, top_k={}, top_p={}",
                req.params.temperature, req.params.top_k, req.params.top_p
            ),
        );
        return;
    }
    if req.logprobs > 0 {
        reject_request(
            &req,
            prompt_len,
            "DeepSeek V4 direct engine does not return logprobs yet".to_string(),
        );
        return;
    }

    let token_tx = req.token_tx.clone();
    let result = generator.generate_greedy(
        &req.prompt_tokens,
        req.max_tokens,
        req.params.ignore_eos,
        |token| {
            token_tx
                .send(TokenEvent::Token {
                    id: token,
                    logprob: None,
                })
                .map_err(|_| anyhow::anyhow!("request receiver dropped"))?;
            Ok(())
        },
    );
    match result {
        Ok(generation) => {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: generation.finish_reason,
                prompt_tokens: prompt_len,
                completion_tokens: generation.generated.len(),
            });
        }
        Err(err) => {
            let message = format!("DeepSeek V4 direct request failed: {err:#}");
            warn!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
        }
    }
}

fn ensure_step_matches_state(
    state: &DeepSeekV4RequestState,
    step: &DirectDecodeStep,
) -> Result<()> {
    if step.request_epoch != state.request_epoch {
        bail!(
            "DeepSeek V4 decode step request epoch mismatch: step={}, state={}",
            step.request_epoch,
            state.request_epoch
        );
    }
    if step.prompt_len != state.prompt_len {
        bail!(
            "DeepSeek V4 decode step prompt length mismatch: step={}, state={}",
            step.prompt_len,
            state.prompt_len
        );
    }
    if step.generated_len_before != state.generated.len() {
        bail!(
            "DeepSeek V4 decode step generated length mismatch: step={}, state={}",
            step.generated_len_before,
            state.generated.len()
        );
    }
    Ok(())
}

fn reject_request(req: &GenerateRequest, prompt_len: usize, reason: String) {
    warn!("{reason}");
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message: reason,
        prompt_tokens: prompt_len,
        completion_tokens: 0,
    });
}

fn argmax_f32(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}
