use std::{
    error::Error,
    fmt,
    path::Path,
    sync::mpsc as std_mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use log::{info, warn};
use pegainfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent,
};
use tokio::sync::mpsc::{self, error::TryRecvError};

#[cfg(test)]
use super::worker::reset_direct_decode_cache_slot_for_test;
use super::worker::{
    DIRECT_BATCH_DECODE_CAPACITY, DirectBatchDecodeEntry, FullDirectRuntime,
    clone_direct_decode_cache_slot, ensure_direct_decode_batch_caches, ensure_direct_decode_caches,
    load_full_direct_runtime, run_direct_decode_batch_logits, run_direct_decode_logits,
    run_prefill_logits_and_seed_decode_cache,
};
use crate::Config;

const DIRECT_HTTP_BATCH_ADMISSION_WAIT: Duration = Duration::from_millis(2);

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
    request_slots: usize,
    active_count: usize,
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

    pub fn request_slots(&self) -> usize {
        self.request_slots
    }

    pub fn active_count(&self) -> usize {
        self.active_count
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
    slot_id: usize,
    request_epoch: u64,
    prompt_len: usize,
    max_new_tokens: usize,
    reserved_seq_len: usize,
}

impl DirectKvCacheLease {
    pub fn slot_id(&self) -> usize {
        self.slot_id
    }

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
        Self::from_model_dir_with_prefill_profile(model_path, false)
    }

    pub fn from_model_dir_with_prefill_profile(
        model_path: &Path,
        enable_prefill_profile: bool,
    ) -> Result<Self> {
        let config = Box::leak(Box::new(Config::from_model_dir(model_path).with_context(
            || {
                format!(
                    "failed to load DeepSeek V4 config from {}",
                    model_path.display()
                )
            },
        )?));
        let runtime = load_full_direct_runtime(model_path, config, enable_prefill_profile)?;
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

    /// Activate the pplx-garden NVLink + RDMA EP backend on every rank.
    ///
    /// `ep_backends` must have length equal to the model's tensor-parallel
    /// world size; element `i` is moved into rank `i`'s worker. After this
    /// returns Ok, decode-time MoE routing uses dispatch/combine instead
    /// of NCCL AG/RS. CUDA Graph decode capture must be disabled in the
    /// caller's run loop when pplx is active (the pplx worker thread
    /// participates in host-side bookkeeping that is incompatible with
    /// graph capture/replay).
    #[cfg(feature = "pplx-ep")]
    pub fn enable_pplx(&self, ep_backends: Vec<pegainfer_comm::EpBackend>) -> Result<()> {
        self.runtime.enable_pplx(ep_backends)
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

        self.kv_cache.set_request_slots(1)?;
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

    fn prepare_active_set_slots(&mut self, max_seq_len: usize, request_slots: usize) -> Result<()> {
        self.kv_cache.set_request_slots(request_slots)?;
        ensure_direct_decode_batch_caches(&mut self.runtime, self.config, max_seq_len)
    }

    fn start_greedy_request_in_slot(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        slot_id: usize,
    ) -> Result<DeepSeekV4RequestState> {
        if prompt_tokens.is_empty() {
            bail!("DeepSeek V4 request produced an empty prompt");
        }
        let request_epoch = self.next_request_epoch;
        self.next_request_epoch = self
            .next_request_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 request epoch exhausted"))?;
        let kv_cache = self.kv_cache.reserve_in_slot(
            request_epoch,
            prompt_tokens.len(),
            max_new_tokens,
            Some(slot_id),
        )?;
        start_greedy_request_from_reserved_slot(
            &mut self.kv_cache,
            kv_cache,
            request_epoch,
            prompt_tokens.len(),
            max_new_tokens,
            ignore_eos,
            || {
                let next_logits = run_prefill_logits_and_seed_decode_cache(
                    &mut self.runtime,
                    self.config,
                    prompt_tokens,
                )?;
                if slot_id != 0 {
                    clone_direct_decode_cache_slot(&mut self.runtime, 0, slot_id)?;
                }
                Ok(next_logits)
            },
        )
    }

    #[cfg(test)]
    fn decode_greedy_batch_step(
        &mut self,
        states: &mut [&mut DeepSeekV4RequestState],
    ) -> Result<Vec<DirectDecodeStep>> {
        let steps = states
            .iter()
            .map(|state| self.sample_greedy_step(state))
            .collect::<Result<Vec<_>>>()?;
        let mut pairs = states
            .iter_mut()
            .zip(steps.iter().cloned())
            .map(|(state, step)| (&mut **state, step))
            .collect::<Vec<_>>();
        advance_greedy_batch_steps_with_decode(&mut self.kv_cache, &mut pairs, |entries| {
            run_direct_decode_batch_logits(&mut self.runtime, entries)
        })?;
        Ok(steps)
    }

    fn decode_greedy_batch_step_from_steps(
        &mut self,
        states_and_steps: &mut [(&mut DeepSeekV4RequestState, DirectDecodeStep)],
    ) -> Result<()> {
        advance_greedy_batch_steps_with_decode(&mut self.kv_cache, states_and_steps, |entries| {
            run_direct_decode_batch_logits(&mut self.runtime, entries)
        })
    }

    #[allow(dead_code)] // PR A lands the runtime batch path before scheduler wiring.
    pub(crate) fn decode_batch_logits_for_test(
        &mut self,
        entries: &[(u32, usize, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        let entries = entries
            .iter()
            .map(|(token_id, start_pos, slot_id)| DirectBatchDecodeEntry {
                token_id: *token_id,
                start_pos: *start_pos,
                slot_id: *slot_id,
            })
            .collect::<Vec<_>>();
        run_direct_decode_batch_logits(&mut self.runtime, &entries)
    }

    #[cfg(test)]
    fn prepare_batch_decode_slots_for_test(&mut self, max_seq_len: usize) -> Result<()> {
        ensure_direct_decode_batch_caches(&mut self.runtime, self.config, max_seq_len)
    }

    #[cfg(test)]
    fn seed_decode_slot_for_test(
        &mut self,
        prompt_tokens: &[u32],
        slot_id: usize,
    ) -> Result<Vec<f32>> {
        reset_direct_decode_cache_slot_for_test(&mut self.runtime, 0)?;
        let logits = run_prefill_logits_and_seed_decode_cache(
            &mut self.runtime,
            self.config,
            prompt_tokens,
        )?;
        if slot_id != 0 {
            clone_direct_decode_cache_slot(&mut self.runtime, 0, slot_id)?;
        }
        Ok(logits)
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
    slot_id: usize,
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
    active: Vec<Option<DirectKvCacheActive>>,
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
            active: vec![None],
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
        self.reserve_in_slot(request_epoch, prompt_len, max_new_tokens, None)
    }

    fn reserve_in_slot(
        &mut self,
        request_epoch: u64,
        prompt_len: usize,
        max_new_tokens: usize,
        preferred_slot: Option<usize>,
    ) -> Result<DirectKvCacheLease> {
        let reserved_seq_len = prompt_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 KV cache reservation length overflow"))?;
        let slot_id = match preferred_slot {
            Some(slot_id) => {
                ensure!(
                    slot_id < self.active.len(),
                    "DeepSeek V4 KV cache preferred slot {slot_id} out of range {}",
                    self.active.len()
                );
                slot_id
            }
            None => self
                .active
                .iter()
                .position(Option::is_none)
                .ok_or_else(|| {
                    self.total_rejections += 1;
                    let reject = DirectKvCacheReject {
                        reason: DirectKvCacheRejectReason::ActiveRequest,
                        requested_seq_len: reserved_seq_len,
                        capacity_seq_len: self.capacity_seq_len,
                    };
                    self.last_reject = Some(reject.clone());
                    DirectKvCacheReservationError {
                        reject,
                        message: "DeepSeek V4 KV cache already has all request slots active"
                            .to_string(),
                    }
                })?,
        };
        if self.active[slot_id].is_some() {
            return self.reject(
                DirectKvCacheRejectReason::ActiveRequest,
                reserved_seq_len,
                "DeepSeek V4 KV cache requested slot already has an active request",
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
            slot_id,
            request_epoch,
            prompt_len,
            max_new_tokens,
            reserved_seq_len,
        };
        self.active[slot_id] = Some(DirectKvCacheActive {
            slot_id,
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

    fn set_request_slots(&mut self, request_slots: usize) -> Result<()> {
        ensure!(
            request_slots > 0,
            "DeepSeek V4 KV cache request slots must be positive"
        );
        let active_count = self.active.iter().filter(|slot| slot.is_some()).count();
        ensure!(
            active_count == 0,
            "DeepSeek V4 KV cache cannot resize request slots while {active_count} request(s) are active"
        );
        self.active.resize_with(request_slots, || None);
        Ok(())
    }

    fn attach_prepared(&mut self, lease: &DirectKvCacheLease) -> Result<()> {
        let active = self
            .active
            .get_mut(lease.slot_id)
            .and_then(Option::as_mut)
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
            .get_mut(lease.slot_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 KV cache release slot {} out of range",
                    lease.slot_id
                )
            })?
            .take()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 KV cache release without active lease"))?;
        ensure!(
            active.slot_id == lease.slot_id,
            "DeepSeek V4 KV cache release slot mismatch: active={}, lease={}",
            active.slot_id,
            lease.slot_id
        );
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
            request_slots: self.active.len(),
            active_count: self.active.iter().filter(|slot| slot.is_some()).count(),
            active: self.active.iter().find_map(Option::as_ref).map(|active| {
                DirectKvCacheActiveSnapshot {
                    request_epoch: active.request_epoch,
                    prompt_len: active.prompt_len,
                    max_new_tokens: active.max_new_tokens,
                    reserved_seq_len: active.reserved_seq_len,
                    attached: active.attached,
                }
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

fn start_greedy_request_from_reserved_slot<F>(
    kv_cache_manager: &mut DirectKvCacheManager,
    kv_cache: DirectKvCacheLease,
    request_epoch: u64,
    prompt_len: usize,
    max_new_tokens: usize,
    ignore_eos: bool,
    seed_next_logits: F,
) -> Result<DeepSeekV4RequestState>
where
    F: FnOnce() -> Result<Vec<f32>>,
{
    if let Err(err) = kv_cache_manager.attach_prepared(&kv_cache) {
        if let Err(release_err) = kv_cache_manager.release(&kv_cache) {
            warn!(
                "failed to release DeepSeek V4 KV cache after slot cache attach error: {release_err:#}"
            );
        }
        return Err(err);
    }

    let next_logits = match seed_next_logits() {
        Ok(next_logits) => next_logits,
        Err(err) => {
            if let Err(release_err) = kv_cache_manager.release(&kv_cache) {
                warn!(
                    "failed to release DeepSeek V4 KV cache after slot request start error: {release_err:#}"
                );
            }
            return Err(err);
        }
    };

    Ok(DeepSeekV4RequestState {
        request_epoch,
        kv_cache: Some(kv_cache),
        prompt_len,
        max_new_tokens,
        ignore_eos,
        generated: Vec::with_capacity(max_new_tokens),
        next_logits: Some(next_logits),
        finish_reason: None,
    })
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

#[allow(dead_code)] // PR B exposes this through the scheduler active-set path.
fn advance_greedy_batch_steps_with_decode<F>(
    kv_cache: &mut DirectKvCacheManager,
    states_and_steps: &mut [(&mut DeepSeekV4RequestState, DirectDecodeStep)],
    decode_next_logits: F,
) -> Result<()>
where
    F: FnOnce(&[DirectBatchDecodeEntry]) -> Result<Vec<Vec<f32>>>,
{
    for (state, step) in states_and_steps.iter() {
        if let Err(err) = ensure_step_matches_state(state, step) {
            release_batch_states_from(kv_cache, states_and_steps, "batch step provenance error");
            return Err(err);
        }
    }

    let mut decode_indices = Vec::new();
    let mut decode_entries = Vec::new();
    for (idx, (state, step)) in states_and_steps.iter_mut().enumerate() {
        if state.is_finished() {
            continue;
        }

        if let Some(finish_reason) = step.finish_reason()
            && step.token().is_none()
        {
            state.next_logits = None;
            state.finish_reason = Some(finish_reason);
            release_greedy_request_from(kv_cache, state)?;
            continue;
        }

        let Some(token) = step.token() else {
            bail!("DeepSeek V4 batch decode step without token or finish reason");
        };
        state
            .next_logits
            .take()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 request state missing consumed logits"))?;
        state.generated.push(token);
        if let Some(finish_reason) = step.finish_reason() {
            state.finish_reason = Some(finish_reason);
            release_greedy_request_from(kv_cache, state)?;
            continue;
        }

        let lease = state.kv_cache.as_ref().ok_or_else(|| {
            anyhow::anyhow!("DeepSeek V4 active request state missing KV cache lease")
        })?;
        decode_indices.push(idx);
        decode_entries.push(DirectBatchDecodeEntry {
            token_id: token,
            start_pos: step.start_pos(),
            slot_id: lease.slot_id(),
        });
    }

    if decode_entries.is_empty() {
        return Ok(());
    }

    let rows = match decode_next_logits(&decode_entries) {
        Ok(rows) => rows,
        Err(err) => {
            release_batch_states_from(kv_cache, states_and_steps, "batch decode error");
            return Err(err);
        }
    };
    if rows.len() != decode_indices.len() {
        let err = anyhow::anyhow!(
            "DeepSeek V4 batch decode returned {} rows for {} requests",
            rows.len(),
            decode_indices.len()
        );
        release_batch_states_from(kv_cache, states_and_steps, "batch decode row-count error");
        return Err(err);
    }
    for (idx, next_logits) in decode_indices.into_iter().zip(rows) {
        states_and_steps[idx].0.next_logits = Some(next_logits);
    }
    Ok(())
}

fn release_batch_states_from(
    kv_cache: &mut DirectKvCacheManager,
    states_and_steps: &mut [(&mut DeepSeekV4RequestState, DirectDecodeStep)],
    context: &str,
) {
    for (state, _) in states_and_steps.iter_mut() {
        if let Err(release_err) = release_greedy_request_from(kv_cache, state) {
            warn!("failed to release DeepSeek V4 KV cache after {context}: {release_err:#}");
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
            let mut generator = match DeepSeekV4DirectGenerator::from_model_dir_with_prefill_profile(
                &model_path,
                options.enable_prefill_profile,
            ) {
                Ok(generator) => generator,
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                    return;
                }
            };
            super::affinity::pin_scheduler_thread(generator.runtime.thread_placement());
            #[cfg(feature = "pplx-ep")]
            if std::env::var("PEGAINFER_DSV4_PPLX").is_ok_and(|v| v != "0" && !v.is_empty()) {
                info!("PEGAINFER_DSV4_PPLX set; building pplx EP backends");
                let ep_shape = pegainfer_comm::bootstrap::EpModelShape {
                    n_routed_experts: generator.config.n_routed_experts,
                    n_activated_experts: generator.config.n_activated_experts,
                    hidden_dim: generator.config.dim,
                };
                match pegainfer_comm::bootstrap::build_intra_node_backends(
                    ep_shape,
                    &(0..8).collect::<Vec<_>>(),
                    generator.runtime.thread_placement(),
                    pegainfer_comm::bootstrap::PplxBootstrapParams::default(),
                ) {
                    Ok((backends, resources)) => {
                        // Leak resources for process lifetime — bootstrap is one-shot.
                        std::mem::forget(resources);
                        if let Err(err) = generator.enable_pplx(backends) {
                            let _ = init_tx.send(Err(err));
                            return;
                        }
                        info!("pplx EP backends installed on all 8 ranks");
                    }
                    Err(err) => {
                        let _ = init_tx.send(Err(err));
                        return;
                    }
                }
            }
            let _ = init_tx.send(Ok(()));
            info!("DeepSeek V4 scheduler ready");
            while let Some(req) = submit_rx.blocking_recv() {
                let mut wave = collect_request_wave(&mut submit_rx, req);
                if wave.len() == 1 {
                    handle_request(
                        &mut generator,
                        wave.pop()
                            .expect("DeepSeek V4 scheduler wave must contain request"),
                    );
                } else {
                    handle_request_wave(&mut generator, wave);
                }
            }
            info!("DeepSeek V4 scheduler exiting");
        })
        .expect("failed to spawn DeepSeek V4 scheduler thread");
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 engine init channel closed: {err}"))??;
    Ok(EngineHandle::new(submit_tx))
}

fn collect_request_wave(
    submit_rx: &mut mpsc::UnboundedReceiver<GenerateRequest>,
    first: GenerateRequest,
) -> Vec<GenerateRequest> {
    let mut wave = vec![first];
    let deadline = Instant::now() + DIRECT_HTTP_BATCH_ADMISSION_WAIT;
    while wave.len() < DIRECT_BATCH_DECODE_CAPACITY {
        match submit_rx.try_recv() {
            Ok(req) => wave.push(req),
            Err(TryRecvError::Empty) => {
                if Instant::now() >= deadline {
                    break;
                }
                thread::sleep(Duration::from_micros(100));
            }
            Err(TryRecvError::Disconnected) => break,
        }
    }
    wave
}

fn handle_request(generator: &mut DeepSeekV4DirectGenerator, req: GenerateRequest) {
    let prompt_len = req.prompt_tokens.len();
    let request_id = req
        .request_id
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_secs_f64);
    let scheduled_at_unix_s = unix_secs_f64();
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s,
        scheduled_at_unix_s,
        prompt_tokens: prompt_len,
    });
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

    let prefill_start = Instant::now();
    let mut state = match generator.start_greedy_request(
        &req.prompt_tokens,
        req.max_tokens,
        req.params.ignore_eos,
    ) {
        Ok(state) => state,
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
            return;
        }
    };
    let prefill_done_unix_s = unix_secs_f64();
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let mut first_token_emit_unix_s = None;
    let mut first_decode_ms = None;

    while !state.is_finished() {
        let step = match generator.sample_greedy_step(&state) {
            Ok(step) => step,
            Err(err) => {
                if let Err(release_err) = generator.release_greedy_request(&mut state) {
                    warn!(
                        "failed to release DeepSeek V4 KV cache after sample error: {release_err:#}"
                    );
                }
                let message = format!("DeepSeek V4 direct request failed: {err:#}");
                warn!("{message}");
                let _ = req.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: prompt_len,
                    completion_tokens: state.generated().len(),
                });
                return;
            }
        };
        if let Some(token) = step.token() {
            if first_token_emit_unix_s.is_none() {
                first_token_emit_unix_s = Some(unix_secs_f64());
            }
            let emit_result = req.token_tx.send(TokenEvent::Token {
                id: token,
                logprob: None,
            });
            if emit_result.is_err() {
                if let Err(release_err) = generator.release_greedy_request(&mut state) {
                    warn!(
                        "failed to release DeepSeek V4 KV cache after receiver drop: {release_err:#}"
                    );
                }
                return;
            }
        }

        let decode_start = Instant::now();
        let advance_result = generator.advance_greedy_step(&mut state, &step);
        if let Err(err) = advance_result {
            let message = format!("DeepSeek V4 direct request failed: {err:#}");
            warn!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: prompt_len,
                completion_tokens: state.generated().len(),
            });
            return;
        }
        if first_decode_ms.is_none() && step.token().is_some() && step.finish_reason().is_none() {
            first_decode_ms = Some(decode_start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    info!(
        "pegainfer_http_trace {}",
        serde_json::json!({
            "request_id": request_id,
            "queued_at_unix_s": queued_at_unix_s,
            "scheduled_at_unix_s": scheduled_at_unix_s,
            "prefill_done_unix_s": prefill_done_unix_s,
            "first_token_emit_unix_s": first_token_emit_unix_s,
            "prefill_ms": prefill_ms,
            "first_decode_ms": first_decode_ms,
            "prompt_tokens": prompt_len,
            "completion_tokens": state.generated().len(),
            "active_set_size": 1,
            "decode_batch_size_max": 1,
            "batch_decode_steps": 0,
        })
    );
    let _ = req.token_tx.send(TokenEvent::Finished {
        finish_reason: state
            .finish_reason()
            .expect("DeepSeek V4 request state must finish after greedy generation"),
        prompt_tokens: prompt_len,
        completion_tokens: state.generated().len(),
    });
}

struct PendingDirectRequest {
    req: GenerateRequest,
    request_id: String,
    queued_at_unix_s: f64,
    scheduled_at_unix_s: f64,
    prompt_len: usize,
    slot_id: usize,
}

struct ActiveDirectRequest {
    req: GenerateRequest,
    request_id: String,
    queued_at_unix_s: f64,
    scheduled_at_unix_s: f64,
    prefill_done_unix_s: f64,
    prefill_ms: f64,
    first_token_emit_unix_s: Option<f64>,
    first_decode_ms: Option<f64>,
    initial_active_set_size: usize,
    max_decode_batch_size: usize,
    batch_decode_steps: usize,
    prompt_len: usize,
    state: DeepSeekV4RequestState,
}

fn handle_request_wave(generator: &mut DeepSeekV4DirectGenerator, requests: Vec<GenerateRequest>) {
    let mut pending = Vec::new();
    for req in requests {
        let prompt_len = req.prompt_tokens.len();
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_secs_f64);
        let scheduled_at_unix_s = unix_secs_f64();
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s,
            prompt_tokens: prompt_len,
        });
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
            continue;
        }
        if req.logprobs > 0 {
            reject_request(
                &req,
                prompt_len,
                "DeepSeek V4 direct engine does not return logprobs yet".to_string(),
            );
            continue;
        }
        if req.max_tokens == 0 {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            continue;
        }
        let slot_id = pending.len();
        let request_id = req
            .request_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        pending.push(PendingDirectRequest {
            req,
            request_id,
            queued_at_unix_s,
            scheduled_at_unix_s,
            prompt_len,
            slot_id,
        });
    }
    if pending.is_empty() {
        return;
    }

    let max_seq_len = pending
        .iter()
        .map(|pending| pending.prompt_len + pending.req.max_tokens)
        .max()
        .expect("pending direct request set must not be empty")
        .max(generator.config.sliding_window);
    if let Err(err) = generator.prepare_active_set_slots(max_seq_len, pending.len()) {
        for pending in pending {
            send_request_error(
                &pending.req,
                pending.prompt_len,
                0,
                format!("DeepSeek V4 direct batch scheduler prepare failed: {err:#}"),
            );
        }
        return;
    }

    let mut active = Vec::with_capacity(pending.len());
    for pending in pending.into_iter().rev() {
        let prefill_start = Instant::now();
        match generator.start_greedy_request_in_slot(
            &pending.req.prompt_tokens,
            pending.req.max_tokens,
            pending.req.params.ignore_eos,
            pending.slot_id,
        ) {
            Ok(state) => {
                let prefill_done_unix_s = unix_secs_f64();
                let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
                active.push(ActiveDirectRequest {
                    req: pending.req,
                    request_id: pending.request_id,
                    queued_at_unix_s: pending.queued_at_unix_s,
                    scheduled_at_unix_s: pending.scheduled_at_unix_s,
                    prefill_done_unix_s,
                    prefill_ms,
                    first_token_emit_unix_s: None,
                    first_decode_ms: None,
                    initial_active_set_size: 0,
                    max_decode_batch_size: 0,
                    batch_decode_steps: 0,
                    prompt_len: pending.prompt_len,
                    state,
                });
            }
            Err(err) => {
                send_request_error(
                    &pending.req,
                    pending.prompt_len,
                    0,
                    format!("DeepSeek V4 direct batch scheduler admission failed: {err:#}"),
                );
                for active in &mut active {
                    if let Err(release_err) = generator.release_greedy_request(&mut active.state) {
                        warn!(
                            "failed to release DeepSeek V4 KV cache after batch admission error: {release_err:#}"
                        );
                    }
                    send_request_error(
                        &active.req,
                        active.prompt_len,
                        active.state.generated().len(),
                        format!(
                            "DeepSeek V4 direct batch scheduler admission failed after this request was admitted: {err:#}"
                        ),
                    );
                }
                return;
            }
        }
    }
    let initial_active_set_size = active.len();
    for active_req in &mut active {
        active_req.initial_active_set_size = initial_active_set_size;
    }

    while !active.is_empty() {
        let mut steps = Vec::with_capacity(active.len());
        let mut dropped = Vec::new();
        for (idx, active_req) in active.iter_mut().enumerate() {
            match generator.sample_greedy_step(&active_req.state) {
                Ok(step) => {
                    if let Some(token) = step.token() {
                        if active_req.first_token_emit_unix_s.is_none() {
                            active_req.first_token_emit_unix_s = Some(unix_secs_f64());
                        }
                        if active_req
                            .req
                            .token_tx
                            .send(TokenEvent::Token {
                                id: token,
                                logprob: None,
                            })
                            .is_err()
                        {
                            if let Err(release_err) =
                                generator.release_greedy_request(&mut active_req.state)
                            {
                                warn!(
                                    "failed to release DeepSeek V4 KV cache after receiver drop: {release_err:#}"
                                );
                            }
                            dropped.push(idx);
                            steps.push(None);
                            continue;
                        }
                    }
                    steps.push(Some(step));
                }
                Err(err) => {
                    send_request_error(
                        &active_req.req,
                        active_req.prompt_len,
                        active_req.state.generated().len(),
                        format!("DeepSeek V4 direct batch scheduler sample failed: {err:#}"),
                    );
                    if let Err(release_err) =
                        generator.release_greedy_request(&mut active_req.state)
                    {
                        warn!(
                            "failed to release DeepSeek V4 KV cache after sample error: {release_err:#}"
                        );
                    }
                    dropped.push(idx);
                    steps.push(None);
                }
            }
        }
        for idx in dropped.into_iter().rev() {
            active.swap_remove(idx);
            steps.swap_remove(idx);
        }
        if active.is_empty() {
            break;
        }

        let decode_batch_size = steps
            .iter()
            .filter(|step| {
                step.as_ref()
                    .is_some_and(|step| step.token().is_some() && step.finish_reason().is_none())
            })
            .count();
        let mut pairs = active
            .iter_mut()
            .zip(steps.into_iter())
            .filter_map(|(active_req, step)| step.map(|step| (&mut active_req.state, step)))
            .collect::<Vec<_>>();
        let decode_start = Instant::now();
        let batch_result = generator.decode_greedy_batch_step_from_steps(&mut pairs);
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        drop(pairs);
        if let Err(err) = batch_result {
            for active_req in &mut active {
                if let Err(release_err) = generator.release_greedy_request(&mut active_req.state) {
                    warn!(
                        "failed to release DeepSeek V4 KV cache after batch scheduler decode error: {release_err:#}"
                    );
                }
                send_request_error(
                    &active_req.req,
                    active_req.prompt_len,
                    active_req.state.generated().len(),
                    format!("DeepSeek V4 direct batch scheduler decode failed: {err:#}"),
                );
            }
            active.clear();
            break;
        }
        if decode_batch_size > 0 {
            for active_req in &mut active {
                active_req.max_decode_batch_size =
                    active_req.max_decode_batch_size.max(decode_batch_size);
                active_req.batch_decode_steps += 1;
                active_req.first_decode_ms.get_or_insert(decode_ms);
            }
        }

        let mut finished = Vec::new();
        for (idx, active_req) in active.iter().enumerate() {
            if let Some(finish_reason) = active_req.state.finish_reason() {
                log_http_trace_for_active_request(active_req);
                let _ = active_req.req.token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens: active_req.prompt_len,
                    completion_tokens: active_req.state.generated().len(),
                });
                finished.push(idx);
            }
        }
        for idx in finished.into_iter().rev() {
            active.swap_remove(idx);
        }
    }
}

fn log_http_trace_for_active_request(active_req: &ActiveDirectRequest) {
    info!(
        "pegainfer_http_trace {}",
        serde_json::json!({
            "request_id": active_req.request_id,
            "queued_at_unix_s": active_req.queued_at_unix_s,
            "scheduled_at_unix_s": active_req.scheduled_at_unix_s,
            "prefill_done_unix_s": active_req.prefill_done_unix_s,
            "first_token_emit_unix_s": active_req.first_token_emit_unix_s,
            "prefill_ms": active_req.prefill_ms,
            "first_decode_ms": active_req.first_decode_ms,
            "prompt_tokens": active_req.prompt_len,
            "completion_tokens": active_req.state.generated().len(),
            "active_set_size": active_req.initial_active_set_size,
            "decode_batch_size_max": active_req.max_decode_batch_size,
            "batch_decode_steps": active_req.batch_decode_steps,
        })
    );
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

fn send_request_error(
    req: &GenerateRequest,
    prompt_len: usize,
    completion_tokens: usize,
    message: String,
) {
    warn!("{message}");
    let _ = req.token_tx.send(TokenEvent::Error {
        message,
        prompt_tokens: prompt_len,
        completion_tokens,
    });
}

fn unix_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
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
    use std::{env, path::PathBuf};

    #[test]
    fn kv_cache_manager_rejects_active_request_until_release() {
        let mut manager = DirectKvCacheManager::new(16);
        let lease = manager.reserve(1, 4, 4).unwrap();
        manager.attach_prepared(&lease).unwrap();

        let err = manager.reserve(2, 4, 4).unwrap_err().to_string();
        assert!(err.contains("request slots active"));
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

    #[test]
    fn kv_cache_manager_releases_only_matching_slot() {
        let mut manager = DirectKvCacheManager::new(16);
        manager.set_request_slots(2).unwrap();
        let slot0 = manager.reserve_in_slot(1, 4, 4, Some(0)).unwrap();
        let slot1 = manager.reserve_in_slot(2, 3, 3, Some(1)).unwrap();
        manager.attach_prepared(&slot0).unwrap();
        manager.attach_prepared(&slot1).unwrap();

        let snapshot = manager.snapshot();
        assert_eq!(snapshot.request_slots(), 2);
        assert_eq!(snapshot.active_count(), 2);
        assert_eq!(slot0.slot_id(), 0);
        assert_eq!(slot1.slot_id(), 1);

        manager.release(&slot1).unwrap();
        let snapshot = manager.snapshot();
        assert_eq!(snapshot.active_count(), 1);
        assert_eq!(snapshot.total_releases(), 1);
        assert_eq!(snapshot.active().unwrap().request_epoch(), 1);

        let replacement = manager.reserve_in_slot(3, 2, 2, Some(1)).unwrap();
        assert_eq!(replacement.slot_id(), 1);
        manager.release(&slot0).unwrap();
        manager.attach_prepared(&replacement).unwrap();
        manager.release(&replacement).unwrap();
        let snapshot = manager.snapshot();
        assert_eq!(snapshot.active_count(), 0);
        assert_eq!(snapshot.total_releases(), 3);
    }

    #[test]
    fn slot_start_error_releases_only_failed_slot() {
        let mut manager = DirectKvCacheManager::new(16);
        manager.set_request_slots(2).unwrap();
        let slot0 = manager.reserve_in_slot(1, 4, 4, Some(0)).unwrap();
        manager.attach_prepared(&slot0).unwrap();
        let slot1 = manager.reserve_in_slot(2, 3, 3, Some(1)).unwrap();

        let err = match start_greedy_request_from_reserved_slot(
            &mut manager,
            slot1,
            2,
            3,
            3,
            true,
            || Err(anyhow::anyhow!("synthetic prefill failure")),
        ) {
            Ok(_) => panic!("synthetic slot start failure unexpectedly succeeded"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("synthetic prefill failure"));
        let snapshot = manager.snapshot();
        assert_eq!(snapshot.active_count(), 1);
        assert_eq!(snapshot.active().unwrap().request_epoch(), 1);
        assert_eq!(snapshot.total_releases(), 1);

        let replacement = manager.reserve_in_slot(3, 2, 2, Some(1)).unwrap();
        assert_eq!(replacement.slot_id(), 1);
        manager.attach_prepared(&replacement).unwrap();
        assert_eq!(manager.snapshot().active_count(), 2);
        manager.release(&slot0).unwrap();
        manager.release(&replacement).unwrap();
        assert_eq!(manager.snapshot().active_count(), 0);
    }

    #[test]
    fn batch_row_count_error_releases_all_active_slots() {
        let mut manager = DirectKvCacheManager::new(16);
        manager.set_request_slots(2).unwrap();
        let slot0 = manager.reserve_in_slot(1, 4, 4, Some(0)).unwrap();
        let slot1 = manager.reserve_in_slot(2, 4, 4, Some(1)).unwrap();
        manager.attach_prepared(&slot0).unwrap();
        manager.attach_prepared(&slot1).unwrap();
        let mut state0 = DeepSeekV4RequestState {
            request_epoch: 1,
            kv_cache: Some(slot0),
            prompt_len: 4,
            max_new_tokens: 4,
            ignore_eos: true,
            generated: Vec::new(),
            next_logits: Some(vec![0.0, 1.0]),
            finish_reason: None,
        };
        let mut state1 = DeepSeekV4RequestState {
            request_epoch: 2,
            kv_cache: Some(slot1),
            prompt_len: 4,
            max_new_tokens: 4,
            ignore_eos: true,
            generated: Vec::new(),
            next_logits: Some(vec![1.0, 0.0]),
            finish_reason: None,
        };
        let step0 = DirectDecodeStep {
            request_epoch: 1,
            generated_len_before: 0,
            prompt_len: 4,
            token: Some(1),
            finish_reason: None,
        };
        let step1 = DirectDecodeStep {
            request_epoch: 2,
            generated_len_before: 0,
            prompt_len: 4,
            token: Some(0),
            finish_reason: None,
        };
        let mut pairs = vec![(&mut state0, step0), (&mut state1, step1)];

        let err = match advance_greedy_batch_steps_with_decode(&mut manager, &mut pairs, |_| {
            Ok(vec![vec![0.0, 1.0]])
        }) {
            Ok(_) => panic!("synthetic row-count mismatch unexpectedly succeeded"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("returned 1 rows for 2 requests"));
        assert_eq!(state0.generated(), &[1]);
        assert_eq!(state1.generated(), &[0]);
        assert!(state0.kv_cache_lease().is_none());
        assert!(state1.kv_cache_lease().is_none());
        let snapshot = manager.snapshot();
        assert_eq!(snapshot.active_count(), 0);
        assert_eq!(snapshot.total_releases(), 2);

        let next = manager.reserve_in_slot(3, 2, 2, Some(1)).unwrap();
        manager.attach_prepared(&next).unwrap();
        assert_eq!(manager.snapshot().active_count(), 1);
    }

    #[test]
    #[ignore = "requires 8 GPUs and DeepSeek-V4-Flash weights"]
    fn batch_decode_logits_match_two_single_decode_rows() -> Result<()> {
        let model_path = env::var_os("PEGAINFER_TEST_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/DeepSeek-V4-Flash"));
        let mut generator = DeepSeekV4DirectGenerator::from_model_dir(&model_path)?;
        let prompt_a = (1000..1128).collect::<Vec<u32>>();
        let prompt_b = (2000..2128).collect::<Vec<u32>>();

        generator.prepare_batch_decode_slots_for_test(prompt_a.len() + 2)?;
        let prefill_a = generator.seed_decode_slot_for_test(&prompt_a, 1)?;
        let prefill_b = generator.seed_decode_slot_for_test(&prompt_b, 0)?;
        let token_a = argmax_f32(&prefill_a) as u32;
        let token_b = argmax_f32(&prefill_b) as u32;

        let batch = generator.decode_batch_logits_for_test(&[
            (token_a, prompt_a.len(), 1),
            (token_b, prompt_b.len(), 0),
        ])?;
        assert_eq!(batch.len(), 2);

        let single_a = decode_logits_for_token(&mut generator, &prompt_a, token_a)?;
        let single_b = decode_logits_for_token(&mut generator, &prompt_b, token_b)?;
        assert_logits_close("row0 prompt A", &single_a, &batch[0]);
        assert_logits_close("row1 prompt B", &single_b, &batch[1]);
        Ok(())
    }

    #[test]
    #[ignore = "requires 8 GPUs and DeepSeek-V4-Flash weights"]
    fn active_set_batch_tick_releases_each_slot() -> Result<()> {
        let model_path = env::var_os("PEGAINFER_TEST_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/DeepSeek-V4-Flash"));
        let mut generator = DeepSeekV4DirectGenerator::from_model_dir(&model_path)?;
        let prompt_a = (1000..1128).collect::<Vec<u32>>();
        let prompt_b = (2000..2128).collect::<Vec<u32>>();

        generator.prepare_active_set_slots(prompt_a.len() + 2, 2)?;
        let mut state_a = generator.start_greedy_request_in_slot(&prompt_a, 2, true, 1)?;
        let mut state_b = generator.start_greedy_request_in_slot(&prompt_b, 2, true, 0)?;
        assert_eq!(state_a.kv_cache_lease().unwrap().slot_id(), 1);
        assert_eq!(state_b.kv_cache_lease().unwrap().slot_id(), 0);
        assert_eq!(generator.kv_cache_snapshot().active_count(), 2);

        let steps = generator.decode_greedy_batch_step(&mut [&mut state_a, &mut state_b])?;
        let token_a = steps[0]
            .token()
            .ok_or_else(|| anyhow::anyhow!("active-set prompt A unexpectedly finished"))?;
        let token_b = steps[1]
            .token()
            .ok_or_else(|| anyhow::anyhow!("active-set prompt B unexpectedly finished"))?;
        assert_eq!(state_a.generated(), &[token_a]);
        assert_eq!(state_b.generated(), &[token_b]);
        assert_eq!(generator.kv_cache_snapshot().active_count(), 2);
        let batch_a = state_a
            .next_logits
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("active-set row A missing next logits"))?
            .clone();
        let batch_b = state_b
            .next_logits
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("active-set row B missing next logits"))?
            .clone();

        generator.decode_greedy_batch_step(&mut [&mut state_a, &mut state_b])?;
        assert!(state_a.is_finished());
        assert!(state_b.is_finished());
        let snapshot = generator.kv_cache_snapshot();
        assert_eq!(snapshot.active_count(), 0);
        assert_eq!(snapshot.total_releases(), 2);

        let single_a = decode_logits_for_token(&mut generator, &prompt_a, token_a)?;
        let single_b = decode_logits_for_token(&mut generator, &prompt_b, token_b)?;
        assert_logits_close("active-set row0 prompt A", &single_a, &batch_a);
        assert_logits_close("active-set row1 prompt B", &single_b, &batch_b);
        Ok(())
    }

    fn decode_logits_for_token(
        generator: &mut DeepSeekV4DirectGenerator,
        prompt_tokens: &[u32],
        token: u32,
    ) -> Result<Vec<f32>> {
        let mut state = generator.start_greedy_request(prompt_tokens, 2, true)?;
        let step = generator.sample_greedy_step(&state)?;
        assert_eq!(step.start_pos(), prompt_tokens.len());
        let logits = run_direct_decode_logits(&mut generator.runtime, token, prompt_tokens.len())?;
        generator.release_greedy_request(&mut state)?;
        Ok(logits)
    }

    fn assert_logits_close(label: &str, expected: &[f32], actual: &[f32]) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "{label} vocab length mismatch"
        );
        assert_eq!(
            argmax_f32(expected),
            argmax_f32(actual),
            "{label} top token mismatch"
        );
        let max_abs = expected
            .iter()
            .zip(actual)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = expected
            .iter()
            .zip(actual)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / expected.len() as f32;
        assert!(max_abs <= 1.5, "{label} max_abs diff too large: {max_abs}");
        assert!(
            mean_abs <= 0.2,
            "{label} mean_abs diff too large: {mean_abs}"
        );
    }
}
