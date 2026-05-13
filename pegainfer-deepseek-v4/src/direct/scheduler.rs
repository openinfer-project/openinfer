use std::{error::Error, fmt, path::Path, sync::mpsc as std_mpsc, thread};

use anyhow::{Context, Result, bail, ensure};
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectKvCacheRejectReason {
    ActiveRequest,
    CapacityExceeded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectKvCacheReject {
    reason: DirectKvCacheRejectReason,
    requested_seq_len: usize,
    capacity_seq_len: usize,
}

impl DirectKvCacheReject {
    pub fn reason(&self) -> DirectKvCacheRejectReason {
        self.reason
    }

    pub fn requested_seq_len(&self) -> usize {
        self.requested_seq_len
    }

    pub fn capacity_seq_len(&self) -> usize {
        self.capacity_seq_len
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectKvCacheActiveSnapshot {
    request_epoch: u64,
    prompt_len: usize,
    max_new_tokens: usize,
    reserved_seq_len: usize,
    attached: bool,
}

impl DirectKvCacheActiveSnapshot {
    pub fn request_epoch(&self) -> u64 {
        self.request_epoch
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    pub fn reserved_seq_len(&self) -> usize {
        self.reserved_seq_len
    }

    pub fn attached(&self) -> bool {
        self.attached
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectKvCacheSnapshot {
    capacity_seq_len: usize,
    allocated_seq_len: usize,
    active: Option<DirectKvCacheActiveSnapshot>,
    total_reservations: u64,
    total_releases: u64,
    total_rejections: u64,
    total_allocations: u64,
    total_resets: u64,
    total_reuses: u64,
    last_reject: Option<DirectKvCacheReject>,
}

impl DirectKvCacheSnapshot {
    pub fn capacity_seq_len(&self) -> usize {
        self.capacity_seq_len
    }

    pub fn allocated_seq_len(&self) -> usize {
        self.allocated_seq_len
    }

    pub fn active(&self) -> Option<&DirectKvCacheActiveSnapshot> {
        self.active.as_ref()
    }

    pub fn total_reservations(&self) -> u64 {
        self.total_reservations
    }

    pub fn total_releases(&self) -> u64 {
        self.total_releases
    }

    pub fn total_rejections(&self) -> u64 {
        self.total_rejections
    }

    pub fn total_allocations(&self) -> u64 {
        self.total_allocations
    }

    pub fn total_resets(&self) -> u64 {
        self.total_resets
    }

    pub fn total_reuses(&self) -> u64 {
        self.total_reuses
    }

    pub fn last_reject(&self) -> Option<&DirectKvCacheReject> {
        self.last_reject.as_ref()
    }
}

#[derive(Clone, Debug)]
pub struct DirectKvCacheLease {
    request_epoch: u64,
    prompt_len: usize,
    max_new_tokens: usize,
    reserved_seq_len: usize,
}

impl DirectKvCacheLease {
    pub fn request_epoch(&self) -> u64 {
        self.request_epoch
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    pub fn reserved_seq_len(&self) -> usize {
        self.reserved_seq_len
    }
}

#[derive(Debug)]
struct DirectKvCacheReservationError {
    reject: DirectKvCacheReject,
    message: String,
}

impl fmt::Display for DirectKvCacheReservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for DirectKvCacheReservationError {}

pub struct DeepSeekV4RequestState {
    request_epoch: u64,
    kv_cache: Option<DirectKvCacheLease>,
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

    pub fn kv_cache_lease(&self) -> Option<&DirectKvCacheLease> {
        self.kv_cache.as_ref()
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
    kv_cache: DirectKvCacheManager,
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
            kv_cache: DirectKvCacheManager::new(config.max_position_embeddings),
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
                kv_cache: None,
                prompt_len: prompt_tokens.len(),
                max_new_tokens,
                ignore_eos,
                generated: Vec::new(),
                next_logits: None,
                finish_reason: Some(FinishReason::Length),
            });
        }

        let kv_cache = self
            .kv_cache
            .reserve(request_epoch, prompt_tokens.len(), max_new_tokens)?;
        if let Err(err) =
            ensure_direct_decode_caches(&mut self.runtime, self.config, kv_cache.reserved_seq_len())
        {
            if let Err(release_err) = self.kv_cache.release(&kv_cache) {
                warn!(
                    "failed to release DeepSeek V4 KV cache after cache prepare error: {release_err:#}"
                );
            }
            return Err(err);
        }
        if let Err(err) = self.kv_cache.attach_prepared(&kv_cache) {
            if let Err(release_err) = self.kv_cache.release(&kv_cache) {
                warn!(
                    "failed to release DeepSeek V4 KV cache after cache attach error: {release_err:#}"
                );
            }
            return Err(err);
        }

        let next_logits = match run_prefill_logits_and_seed_decode_cache(
            &mut self.runtime,
            self.config,
            prompt_tokens,
        ) {
            Ok(next_logits) => next_logits,
            Err(err) => {
                if let Err(release_err) = self.kv_cache.release(&kv_cache) {
                    warn!(
                        "failed to release DeepSeek V4 KV cache after prefill error: {release_err:#}"
                    );
                }
                return Err(err);
            }
        };

        Ok(DeepSeekV4RequestState {
            request_epoch,
            kv_cache: Some(kv_cache),
            prompt_len: prompt_tokens.len(),
            max_new_tokens,
            ignore_eos,
            generated: Vec::with_capacity(max_new_tokens),
            next_logits: Some(next_logits),
            finish_reason: None,
        })
    }

    pub fn kv_cache_snapshot(&self) -> DirectKvCacheSnapshot {
        self.kv_cache.snapshot()
    }

    pub fn release_greedy_request(&mut self, state: &mut DeepSeekV4RequestState) -> Result<()> {
        release_greedy_request_from(&mut self.kv_cache, state)
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
        ensure!(
            state.kv_cache.is_some(),
            "DeepSeek V4 active request state missing KV cache lease"
        );
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
        let runtime = &mut self.runtime;
        advance_greedy_step_with_decode(&mut self.kv_cache, state, step, |token, start_pos| {
            run_direct_decode_logits(runtime, token, start_pos)
        })
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
                if let Err(err) = on_token(token) {
                    if let Err(release_err) = self.release_greedy_request(&mut state) {
                        warn!(
                            "failed to release DeepSeek V4 KV cache after token callback error: {release_err:#}"
                        );
                    }
                    return Err(err);
                }
            }
            if let Err(err) = self.advance_greedy_step(&mut state, &step) {
                if let Err(release_err) = self.release_greedy_request(&mut state) {
                    warn!(
                        "failed to release DeepSeek V4 KV cache after decode error: {release_err:#}"
                    );
                }
                return Err(err);
            }
        }
        Ok(DirectGeneration {
            generated: state.generated,
            finish_reason: state
                .finish_reason
                .expect("DeepSeek V4 request state must finish after greedy generation"),
        })
    }
}

#[derive(Clone, Debug)]
struct DirectKvCacheActive {
    request_epoch: u64,
    prompt_len: usize,
    max_new_tokens: usize,
    reserved_seq_len: usize,
    attached: bool,
    reused_capacity: bool,
}

struct DirectKvCacheManager {
    capacity_seq_len: usize,
    allocated_seq_len: usize,
    active: Option<DirectKvCacheActive>,
    total_reservations: u64,
    total_releases: u64,
    total_rejections: u64,
    total_allocations: u64,
    total_resets: u64,
    total_reuses: u64,
    last_reject: Option<DirectKvCacheReject>,
}

impl DirectKvCacheManager {
    fn new(capacity_seq_len: usize) -> Self {
        Self {
            capacity_seq_len,
            allocated_seq_len: 0,
            active: None,
            total_reservations: 0,
            total_releases: 0,
            total_rejections: 0,
            total_allocations: 0,
            total_resets: 0,
            total_reuses: 0,
            last_reject: None,
        }
    }

    fn reserve(
        &mut self,
        request_epoch: u64,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Result<DirectKvCacheLease> {
        let reserved_seq_len = prompt_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 KV cache reservation length overflow"))?;
        if self.active.is_some() {
            return self.reject(
                DirectKvCacheRejectReason::ActiveRequest,
                reserved_seq_len,
                "DeepSeek V4 KV cache already has an active request",
            );
        }
        if reserved_seq_len > self.capacity_seq_len {
            return self.reject(
                DirectKvCacheRejectReason::CapacityExceeded,
                reserved_seq_len,
                &format!(
                    "DeepSeek V4 KV cache reservation {reserved_seq_len} exceeds capacity {}",
                    self.capacity_seq_len
                ),
            );
        }

        let lease = DirectKvCacheLease {
            request_epoch,
            prompt_len,
            max_new_tokens,
            reserved_seq_len,
        };
        self.active = Some(DirectKvCacheActive {
            request_epoch,
            prompt_len,
            max_new_tokens,
            reserved_seq_len,
            attached: false,
            reused_capacity: self.allocated_seq_len >= reserved_seq_len,
        });
        self.total_reservations += 1;
        Ok(lease)
    }

    fn attach_prepared(&mut self, lease: &DirectKvCacheLease) -> Result<()> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 KV cache attach without reservation"))?;
        ensure!(
            active.request_epoch == lease.request_epoch,
            "DeepSeek V4 KV cache attach epoch mismatch: active={}, lease={}",
            active.request_epoch,
            lease.request_epoch
        );
        ensure!(
            active.reserved_seq_len == lease.reserved_seq_len,
            "DeepSeek V4 KV cache attach length mismatch: active={}, lease={}",
            active.reserved_seq_len,
            lease.reserved_seq_len
        );
        active.attached = true;
        self.total_resets += 1;
        if active.reused_capacity {
            self.total_reuses += 1;
        } else {
            self.total_allocations += 1;
            self.allocated_seq_len = self.allocated_seq_len.max(active.reserved_seq_len);
        }
        Ok(())
    }

    fn release(&mut self, lease: &DirectKvCacheLease) -> Result<()> {
        let active = self
            .active
            .take()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 KV cache release without active lease"))?;
        ensure!(
            active.request_epoch == lease.request_epoch,
            "DeepSeek V4 KV cache release epoch mismatch: active={}, lease={}",
            active.request_epoch,
            lease.request_epoch
        );
        ensure!(
            active.reserved_seq_len == lease.reserved_seq_len,
            "DeepSeek V4 KV cache release length mismatch: active={}, lease={}",
            active.reserved_seq_len,
            lease.reserved_seq_len
        );
        self.total_releases += 1;
        Ok(())
    }

    fn snapshot(&self) -> DirectKvCacheSnapshot {
        DirectKvCacheSnapshot {
            capacity_seq_len: self.capacity_seq_len,
            allocated_seq_len: self.allocated_seq_len,
            active: self
                .active
                .as_ref()
                .map(|active| DirectKvCacheActiveSnapshot {
                    request_epoch: active.request_epoch,
                    prompt_len: active.prompt_len,
                    max_new_tokens: active.max_new_tokens,
                    reserved_seq_len: active.reserved_seq_len,
                    attached: active.attached,
                }),
            total_reservations: self.total_reservations,
            total_releases: self.total_releases,
            total_rejections: self.total_rejections,
            total_allocations: self.total_allocations,
            total_resets: self.total_resets,
            total_reuses: self.total_reuses,
            last_reject: self.last_reject.clone(),
        }
    }

    fn reject<T>(
        &mut self,
        reason: DirectKvCacheRejectReason,
        requested_seq_len: usize,
        message: &str,
    ) -> Result<T> {
        self.total_rejections += 1;
        let reject = DirectKvCacheReject {
            reason,
            requested_seq_len,
            capacity_seq_len: self.capacity_seq_len,
        };
        self.last_reject = Some(reject.clone());
        Err(DirectKvCacheReservationError {
            reject,
            message: message.to_string(),
        }
        .into())
    }
}

fn release_greedy_request_from(
    kv_cache: &mut DirectKvCacheManager,
    state: &mut DeepSeekV4RequestState,
) -> Result<()> {
    if let Some(lease) = state.kv_cache.take() {
        kv_cache.release(&lease)?;
    }
    state.next_logits = None;
    Ok(())
}

fn advance_greedy_step_with_decode<F>(
    kv_cache: &mut DirectKvCacheManager,
    state: &mut DeepSeekV4RequestState,
    step: &DirectDecodeStep,
    decode_next_logits: F,
) -> Result<()>
where
    F: FnOnce(u32, usize) -> Result<Vec<f32>>,
{
    ensure_step_matches_state(state, step)?;
    if state.is_finished() {
        return Ok(());
    }

    if let Some(finish_reason) = step.finish_reason()
        && step.token().is_none()
    {
        state.next_logits = None;
        state.finish_reason = Some(finish_reason);
        release_greedy_request_from(kv_cache, state)?;
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
        release_greedy_request_from(kv_cache, state)?;
        return Ok(());
    }

    match decode_next_logits(token, step.start_pos()) {
        Ok(next_logits) => {
            state.next_logits = Some(next_logits);
            Ok(())
        }
        Err(err) => {
            release_greedy_request_from(kv_cache, state)?;
            Err(err)
        }
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
            if let Some(kv_err) = err.downcast_ref::<DirectKvCacheReservationError>() {
                reject_request(
                    &req,
                    prompt_len,
                    format!(
                        "DeepSeek V4 direct request rejected by KV cache ownership gate ({:?}): {}",
                        kv_err.reject.reason(),
                        kv_err
                    ),
                );
                return;
            }
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
    if !state.is_finished() {
        let lease = state.kv_cache.as_ref().ok_or_else(|| {
            anyhow::anyhow!("DeepSeek V4 active request state missing KV cache lease")
        })?;
        ensure!(
            lease.request_epoch == state.request_epoch,
            "DeepSeek V4 KV cache lease epoch mismatch: lease={}, state={}",
            lease.request_epoch,
            state.request_epoch
        );
        ensure!(
            lease.prompt_len == state.prompt_len,
            "DeepSeek V4 KV cache lease prompt length mismatch: lease={}, state={}",
            lease.prompt_len,
            state.prompt_len
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_manager_rejects_active_request_until_release() {
        let mut manager = DirectKvCacheManager::new(16);
        let lease = manager.reserve(1, 4, 4).unwrap();
        manager.attach_prepared(&lease).unwrap();

        let err = manager.reserve(2, 4, 4).unwrap_err().to_string();
        assert!(err.contains("active request"));
        let snapshot = manager.snapshot();
        assert_eq!(snapshot.total_reservations(), 1);
        assert_eq!(snapshot.total_rejections(), 1);
        assert_eq!(
            snapshot.last_reject().unwrap().reason(),
            DirectKvCacheRejectReason::ActiveRequest
        );

        manager.release(&lease).unwrap();
        let snapshot = manager.snapshot();
        assert!(snapshot.active().is_none());
        assert_eq!(snapshot.total_releases(), 1);
    }

    #[test]
    fn kv_cache_manager_rejects_over_capacity_request() {
        let mut manager = DirectKvCacheManager::new(8);
        let err = manager.reserve(1, 6, 3).unwrap_err().to_string();
        assert!(err.contains("exceeds capacity"));

        let snapshot = manager.snapshot();
        assert_eq!(snapshot.total_reservations(), 0);
        assert_eq!(snapshot.total_rejections(), 1);
        assert_eq!(
            snapshot.last_reject().unwrap().reason(),
            DirectKvCacheRejectReason::CapacityExceeded
        );
        assert_eq!(snapshot.last_reject().unwrap().requested_seq_len(), 9);
    }

    #[test]
    fn kv_cache_manager_tracks_allocate_reset_and_reuse() {
        let mut manager = DirectKvCacheManager::new(16);
        let first = manager.reserve(1, 4, 4).unwrap();
        manager.attach_prepared(&first).unwrap();
        manager.release(&first).unwrap();

        let second = manager.reserve(2, 2, 2).unwrap();
        manager.attach_prepared(&second).unwrap();
        manager.release(&second).unwrap();

        let snapshot = manager.snapshot();
        assert_eq!(snapshot.allocated_seq_len(), 8);
        assert_eq!(snapshot.total_reservations(), 2);
        assert_eq!(snapshot.total_releases(), 2);
        assert_eq!(snapshot.total_allocations(), 1);
        assert_eq!(snapshot.total_resets(), 2);
        assert_eq!(snapshot.total_reuses(), 1);
    }

    #[test]
    fn decode_runtime_error_releases_active_kv_lease() {
        let mut manager = DirectKvCacheManager::new(16);
        let lease = manager.reserve(1, 4, 4).unwrap();
        manager.attach_prepared(&lease).unwrap();
        let mut state = DeepSeekV4RequestState {
            request_epoch: 1,
            kv_cache: Some(lease),
            prompt_len: 4,
            max_new_tokens: 4,
            ignore_eos: true,
            generated: Vec::new(),
            next_logits: Some(vec![0.0, 1.0]),
            finish_reason: None,
        };
        let step = DirectDecodeStep {
            request_epoch: 1,
            generated_len_before: 0,
            prompt_len: 4,
            token: Some(1),
            finish_reason: None,
        };

        let err = advance_greedy_step_with_decode(&mut manager, &mut state, &step, |_, _| {
            Err(anyhow::anyhow!("synthetic decode failure"))
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("synthetic decode failure"));
        assert!(state.kv_cache_lease().is_none());
        assert!(state.next_logits.is_none());
        assert_eq!(state.generated(), &[1]);
        let snapshot = manager.snapshot();
        assert!(snapshot.active().is_none());
        assert_eq!(snapshot.total_releases(), 1);

        let next = manager.reserve(2, 2, 2).unwrap();
        manager.attach_prepared(&next).unwrap();
        assert!(manager.snapshot().active().is_some());
    }
}
