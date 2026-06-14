//! Scheduler: dedicated GPU thread that batches concurrent requests.
//!
//! Frontend handlers tokenize prompts and submit `GenerateRequest` via channel.
//! The scheduler batch-prefills all pending requests in one forward pass, then
//! batch-decodes all active requests. Per-request tokens flow back through
//! individual channels.

mod effects;
mod plan;
mod resolve;

use std::collections::{HashSet, VecDeque};
use std::thread;

use anyhow::Result;
use log::{debug, info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::executor::{ModelExecutor, Qwen3Executor, RequestId};
use crate::{Qwen3LoraOptions, Qwen3OffloadOptions};
use openinfer_core::engine::{
    EngineCommand, EngineControlRequest, EngineHandle, GenerateRequest, KvCapacity, TokenEvent,
};
use openinfer_core::sampler::SamplingParams;

use self::effects::apply_effects;
use self::plan::{build_next_plan, execute_plan};
use self::resolve::resolve_step;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded.
pub(super) struct ActiveRequestState {
    pub(super) request_id: RequestId,
    pub(super) lora_adapter: Option<String>,
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) last_token: u32,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
    pub(super) prompt_len: usize,
    pub(super) params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    pub(super) logprobs: usize,
}

pub(super) struct PendingRequest {
    pub(super) request_id: RequestId,
    pub(super) lora_adapter: Option<String>,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) params: SamplingParams,
    pub(super) max_tokens: usize,
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) logprobs: usize,
    pub(super) echo: bool,
    pub(super) queued_at_unix_s: Option<f64>,
    /// Whether this request has already been offered to async KV prefetch.
    /// Offered at most once; a no-hit offer leaves the request in the normal
    /// admission flow with this set so it isn't re-probed every tick.
    pub(super) prefetch_offered: bool,
    /// Prompt tokens whose KV is already computed (prefix-cache hits plus
    /// chunks applied in earlier steps). Updated from the executor's
    /// authoritative position after every chunk.
    pub(super) prefill_pos: usize,
    /// Prompt tokens to forward in the upcoming step. Set by
    /// `take_prefill_chunks` when the request is packed into a step.
    pub(super) step_chunk: usize,
    /// Prefix-cache hits reported by the first chunk, carried across later
    /// chunks so the final result still reports them truthfully.
    pub(super) cached_tokens: usize,
}

impl PendingRequest {
    fn from_scheduler_request(request_id: RequestId, req: GenerateRequest) -> Self {
        Self {
            request_id,
            lora_adapter: req.lora_adapter,
            prompt_tokens: req.prompt_tokens,
            params: req.params,
            max_tokens: req.max_tokens,
            token_tx: req.token_tx,
            logprobs: req.logprobs,
            echo: req.echo,
            queued_at_unix_s: req.queued_at_unix_s,
            prefetch_offered: false,
            prefill_pos: 0,
            step_chunk: 0,
            cached_tokens: 0,
        }
    }

    fn remaining_prompt_tokens(&self) -> usize {
        self.prompt_tokens.len() - self.prefill_pos
    }
}

/// Pull the next prefill step set off the front of `prefilling`, capping the
/// step's total forwarded tokens at `max_prefill_tokens`. Each taken request
/// gets its per-step chunk recorded in `step_chunk`. Echo requests need
/// logits for every prompt position in one forward, so they only run when
/// their whole remainder fits — or alone at the head of an empty step, which
/// also guarantees the queue always makes progress.
fn take_prefill_chunks(
    prefilling: &mut Vec<PendingRequest>,
    max_prefill_tokens: usize,
) -> Vec<PendingRequest> {
    let mut budget = max_prefill_tokens;
    let mut taken: Vec<PendingRequest> = Vec::new();
    let mut i = 0;
    while i < prefilling.len() && budget > 0 {
        let remaining = prefilling[i].remaining_prompt_tokens();
        let chunk = if prefilling[i].echo {
            if remaining > budget && !taken.is_empty() {
                i += 1;
                continue;
            }
            remaining
        } else {
            remaining.min(budget)
        };
        let mut req = prefilling.remove(i);
        req.step_chunk = chunk;
        budget = budget.saturating_sub(chunk);
        taken.push(req);
    }
    // Echo skips can take items out of arrival order; results come back
    // sorted by request id, so the step set must be too.
    taken.sort_by_key(|req| req.request_id);
    taken
}

// ── Entry point ─────────────────────────────────────────────────────────

pub(crate) fn start_qwen3(
    model_path: &str,
    enable_cuda_graph: bool,
    device_ordinals: &[usize],
    seed: u64,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    let mut executor = Qwen3Executor::from_runtime_with_lora_options(
        model_path,
        enable_cuda_graph,
        device_ordinals,
        Qwen3LoraOptions::default(),
        offload_options,
    )?;
    executor.set_no_prefix_cache(no_prefix_cache);
    Ok(start_with_executor(executor, seed, max_prefill_tokens))
}

pub(crate) fn start_qwen3_with_lora_control(
    model_path: &str,
    enable_cuda_graph: bool,
    device_ordinals: &[usize],
    seed: u64,
    lora_options: Qwen3LoraOptions,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    let mut executor = Qwen3Executor::from_runtime_with_lora_options(
        model_path,
        enable_cuda_graph,
        device_ordinals,
        lora_options,
        offload_options,
    )?;
    executor.set_no_prefix_cache(no_prefix_cache);
    Ok(start_with_executor_with_lora_control(
        executor,
        seed,
        max_prefill_tokens,
    ))
}

fn servable_len(max_context: usize, max_blocks: usize, block_size: usize) -> u32 {
    max_context
        .min(max_blocks.saturating_mul(block_size))
        .try_into()
        .unwrap_or(u32::MAX)
}

pub(crate) fn start_with_executor<E>(
    executor: E,
    seed: u64,
    max_prefill_tokens: usize,
) -> EngineHandle
where
    E: ModelExecutor + 'static,
{
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    let servable = servable_len(
        executor.max_context_tokens(),
        executor.max_request_blocks(),
        executor.block_size(),
    );
    // Executor just built: the only committed block is the leaked CUDA-graph
    // padding slot, so available_blocks() is total − 1. Conservative by one
    // block, which is the right side to err on for a capacity ceiling.
    let kv_capacity = KvCapacity {
        total_blocks: executor.available_blocks(),
        block_size: executor.block_size(),
    };
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop(executor, submit_rx, seed, max_prefill_tokens);
        })
        .expect("failed to spawn scheduler thread");

    EngineHandle::new(submit_tx)
        .with_servable_len(servable)
        .with_kv_capacity(kv_capacity)
}

pub(crate) fn start_with_executor_with_lora_control<E>(
    executor: E,
    seed: u64,
    max_prefill_tokens: usize,
) -> EngineHandle
where
    E: ModelExecutor + 'static,
{
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    let servable = servable_len(
        executor.max_context_tokens(),
        executor.max_request_blocks(),
        executor.block_size(),
    );
    // Executor just built: the only committed block is the leaked CUDA-graph
    // padding slot, so available_blocks() is total − 1. Conservative by one
    // block, which is the right side to err on for a capacity ceiling.
    let kv_capacity = KvCapacity {
        total_blocks: executor.available_blocks(),
        block_size: executor.block_size(),
    };
    let (command_tx, command_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop_with_lora_control(executor, command_rx, seed, max_prefill_tokens);
        })
        .expect("failed to spawn scheduler thread");

    EngineHandle::new_with_command_channel(command_tx)
        .with_servable_len(servable)
        .with_kv_capacity(kv_capacity)
}

// ── KV-offload prefetch admission helpers ────────────────────────────────

/// Move requests whose async CPU-tier prefetch just settled from `loading`
/// back to the front of `deferred` — their KV is hot, so admit them first.
fn reclaim_ready_prefetch<E: ModelExecutor>(
    executor: &mut E,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
) {
    promote_ready(executor.drain_ready_prefetch(), deferred, loading);
}

/// Offer each not-yet-offered `deferred` request to async CPU-tier prefetch,
/// moving the ones that start loading out of `deferred` into `loading`. A
/// request that doesn't start a load (pure GPU hit, miss, or block pressure)
/// stays in `deferred`, flagged so it isn't re-probed next tick.
///
/// Echo requests are never offered: their prefill forwards the whole prompt to
/// recover prompt logprobs and so skips `match_and_add_prefix` (see
/// `execute_prefill`). Prefetched blocks would never be matched/reused — they
/// would only park restored KV that admission credits but prefill can't spend,
/// starving the request under tight budgets. Leaving `prefetch_offered` unset
/// for echo is harmless: the `!req.echo` guard keeps them from being probed.
fn offer_prefetch<E: ModelExecutor>(
    executor: &mut E,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
    // Free blocks already promised to admitted requests; the prefetch must
    // leave them untouched (see `ModelExecutor::begin_kv_prefetch`).
    reserve_floor: usize,
) {
    let mut keep = Vec::with_capacity(deferred.len());
    for mut req in deferred.drain(..) {
        if !req.prefetch_offered && !req.echo {
            req.prefetch_offered = true;
            if executor.begin_kv_prefetch(
                req.request_id,
                &req.prompt_tokens,
                req.lora_adapter.as_deref(),
                reserve_floor,
            ) {
                loading.push(req);
                continue;
            }
        }
        keep.push(req);
    }
    *deferred = keep;
}

/// Block until at least one in-flight prefetch settles, then promote the
/// settled requests to `deferred`. Called only when the scheduler is otherwise
/// idle, so blocking on the DMA costs nothing.
fn block_on_loading<E: ModelExecutor>(
    executor: &mut E,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
) {
    promote_ready(executor.wait_ready_prefetch(), deferred, loading);
}

fn promote_ready(
    ready: Vec<RequestId>,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
) {
    for id in ready {
        if let Some(pos) = loading.iter().position(|p| p.request_id == id) {
            deferred.insert(0, loading.remove(pos));
        }
    }
}

/// Release any executor-side state a request accumulated before it was rejected
/// at admission. A rejected request never prefills, so the only state it can
/// hold is a settled KV prefetch — committed prefix blocks parked in the
/// executor while the request waited in `deferred`. Without this they would
/// leak (blocks pinned, map entry stranded) for the engine's lifetime. Idempotent
/// and harmless for requests that were never prefetched.
fn release_rejected<E: ModelExecutor>(executor: &mut E, req: &PendingRequest) {
    if let Err(e) = executor.drop_request(req.request_id) {
        warn!(
            "failed to release state for rejected {:?}: {e}",
            req.request_id
        );
    }
}

// ── Main loop ───────────────────────────────────────────────────────────

fn scheduler_loop<E>(
    mut executor: E,
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    seed: u64,
    max_prefill_tokens: usize,
) where
    E: ModelExecutor,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequestState> = Vec::new();
    let mut next_request_id = 0u64;
    // Requests that could not be admitted due to KV budget pressure.
    // Held here so they aren't lost; re-evaluated every loop iteration.
    let mut deferred: Vec<PendingRequest> = Vec::new();
    // Requests parked while their async CPU-tier KV prefetch loads.
    let mut loading: Vec<PendingRequest> = Vec::new();
    // Admitted requests whose prompts are not fully prefilled yet (chunked
    // prefill). FIFO by request id; each step takes chunks off the front.
    let mut prefilling: Vec<PendingRequest> = Vec::new();

    info!("Scheduler ready");

    loop {
        // 1. Drain all incoming requests into deferred.
        while let Ok(req) = submit_rx.try_recv() {
            deferred.push(PendingRequest::from_scheduler_request(
                RequestId(next_request_id),
                req,
            ));
            next_request_id += 1;
        }

        // 2. Reclaim settled prefetches, then offer fresh requests to prefetch.
        reclaim_ready_prefetch(&mut executor, &mut deferred, &mut loading);
        let reserve_floor = admitted_future_blocks(&executor, &active, &prefilling);
        offer_prefetch(&mut executor, &mut deferred, &mut loading, reserve_floor);

        // 3. Nothing active and nothing admittable → block. Prefer blocking on
        // an in-flight load (so its request prefills next) over a new submit;
        // only truly idle (no loads either) do we block on the channel.
        if active.is_empty() && deferred.is_empty() && prefilling.is_empty() {
            if !loading.is_empty() {
                block_on_loading(&mut executor, &mut deferred, &mut loading);
                continue;
            }
            if let Some(req) = submit_rx.blocking_recv() {
                deferred.push(PendingRequest::from_scheduler_request(
                    RequestId(next_request_id),
                    req,
                ));
                next_request_id += 1;
            } else {
                info!("Scheduler: all handles dropped, exiting");
                return;
            }
            while let Ok(req) = submit_rx.try_recv() {
                deferred.push(PendingRequest::from_scheduler_request(
                    RequestId(next_request_id),
                    req,
                ));
                next_request_id += 1;
            }
            continue;
        }

        let lora_validation = reject_unknown_lora_requests(deferred, &executor);
        for rejected in &lora_validation.rejected {
            send_unknown_lora_rejection(rejected);
            release_rejected(&mut executor, rejected);
        }

        let admission = admit_deferred_requests(
            lora_validation.accepted,
            &active,
            &prefilling,
            executor.block_size(),
            executor.available_blocks(),
            executor.max_request_blocks(),
            executor.max_context_tokens(),
            executor.max_decode_batch_size(),
            |id| executor.prefetched_blocks(id),
        );
        for (rejected, reason) in &admission.rejected {
            send_rejection(rejected, *reason);
            release_rejected(&mut executor, rejected);
        }
        prefilling.extend(admission.pending);
        deferred = admission.deferred;
        let pending = take_prefill_chunks(&mut prefilling, max_prefill_tokens);

        let Some(plan) = build_next_plan(!active.is_empty(), pending) else {
            continue;
        };
        let failure_targets = failure_targets_for(&active, &plan);
        let artifacts = match execute_plan(&mut executor, &mut active, plan, &mut rng) {
            Ok(v) => v,
            Err(e) => {
                warn!("Execution step failed: {e}");
                fail_touched_requests(&mut executor, &mut active, failure_targets, &e.to_string());
                continue;
            }
        };
        let effects = resolve_step(&executor, &active, artifacts);
        apply_effects(&mut executor, &mut active, &mut prefilling, effects);
    }
}

fn scheduler_loop_with_lora_control<E>(
    mut executor: E,
    mut command_rx: mpsc::UnboundedReceiver<EngineCommand>,
    seed: u64,
    max_prefill_tokens: usize,
) where
    E: ModelExecutor,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequestState> = Vec::new();
    let mut next_request_id = 0u64;
    let mut deferred: Vec<PendingRequest> = Vec::new();
    let mut loading: Vec<PendingRequest> = Vec::new();
    let mut prefilling: Vec<PendingRequest> = Vec::new();
    let mut pending_control: VecDeque<EngineControlRequest> = VecDeque::new();
    let mut post_control_deferred: Vec<PendingRequest> = Vec::new();

    info!("Scheduler ready with LoRA control");

    loop {
        // 1. Drain incoming commands. Generation submitted after a pending
        // control command waits until that control command is handled at idle.
        while let Ok(command) = command_rx.try_recv() {
            enqueue_engine_command(
                command,
                &mut deferred,
                &mut pending_control,
                &mut post_control_deferred,
                &mut next_request_id,
            );
        }

        // 1b. Reclaim settled prefetches and offer fresh requests. Control
        // commands gate generation, so only offer once no control is pending
        // (a prefetch must not race ahead of an adapter load it depends on).
        reclaim_ready_prefetch(&mut executor, &mut deferred, &mut loading);
        if pending_control.is_empty() {
            let reserve_floor = admitted_future_blocks(&executor, &active, &prefilling);
            offer_prefetch(&mut executor, &mut deferred, &mut loading, reserve_floor);
        }

        // 2. Once idle, apply pending control commands before admitting newer
        // generation requests that arrived behind them.
        if active.is_empty() && deferred.is_empty() && prefilling.is_empty() {
            drain_idle_control(&mut executor, &mut pending_control);
            if pending_control.is_empty() && !post_control_deferred.is_empty() {
                deferred.append(&mut post_control_deferred);
            }
        }

        // 3. Nothing active and no deferred generation → block. An in-flight
        // load takes priority over waiting on a new command.
        if active.is_empty() && deferred.is_empty() && prefilling.is_empty() {
            if !loading.is_empty() {
                block_on_loading(&mut executor, &mut deferred, &mut loading);
                continue;
            }
            if let Some(command) = command_rx.blocking_recv() {
                enqueue_engine_command(
                    command,
                    &mut deferred,
                    &mut pending_control,
                    &mut post_control_deferred,
                    &mut next_request_id,
                );
            } else {
                info!("Scheduler: all handles dropped, exiting");
                return;
            }
            while let Ok(command) = command_rx.try_recv() {
                enqueue_engine_command(
                    command,
                    &mut deferred,
                    &mut pending_control,
                    &mut post_control_deferred,
                    &mut next_request_id,
                );
            }
            if active.is_empty() && deferred.is_empty() {
                drain_idle_control(&mut executor, &mut pending_control);
                if pending_control.is_empty() && !post_control_deferred.is_empty() {
                    deferred.append(&mut post_control_deferred);
                }
            }
        }

        let lora_validation = reject_unknown_lora_requests(deferred, &executor);
        for rejected in &lora_validation.rejected {
            send_unknown_lora_rejection(rejected);
            release_rejected(&mut executor, rejected);
        }

        let admission = admit_deferred_requests(
            lora_validation.accepted,
            &active,
            &prefilling,
            executor.block_size(),
            executor.available_blocks(),
            executor.max_request_blocks(),
            executor.max_context_tokens(),
            executor.max_decode_batch_size(),
            |id| executor.prefetched_blocks(id),
        );
        for (rejected, reason) in &admission.rejected {
            send_rejection(rejected, *reason);
            release_rejected(&mut executor, rejected);
        }
        prefilling.extend(admission.pending);
        deferred = admission.deferred;
        let pending = take_prefill_chunks(&mut prefilling, max_prefill_tokens);

        if active.is_empty() && pending.is_empty() {
            // A parked load must still be polled to completion before we block.
            if !loading.is_empty() {
                block_on_loading(&mut executor, &mut deferred, &mut loading);
                continue;
            }
            if let Some(command) = command_rx.blocking_recv() {
                enqueue_engine_command(
                    command,
                    &mut deferred,
                    &mut pending_control,
                    &mut post_control_deferred,
                    &mut next_request_id,
                );
            } else {
                info!("Scheduler: all handles dropped, exiting");
                return;
            }
            continue;
        }

        let Some(plan) = build_next_plan(!active.is_empty(), pending) else {
            continue;
        };
        let failure_targets = failure_targets_for(&active, &plan);
        let artifacts = match execute_plan(&mut executor, &mut active, plan, &mut rng) {
            Ok(v) => v,
            Err(e) => {
                warn!("Execution step failed: {e}");
                fail_touched_requests(&mut executor, &mut active, failure_targets, &e.to_string());
                continue;
            }
        };
        let effects = resolve_step(&executor, &active, artifacts);
        apply_effects(&mut executor, &mut active, &mut prefilling, effects);
    }
}

fn enqueue_engine_command(
    command: EngineCommand,
    deferred: &mut Vec<PendingRequest>,
    pending_control: &mut VecDeque<EngineControlRequest>,
    post_control_deferred: &mut Vec<PendingRequest>,
    next_request_id: &mut u64,
) {
    match command {
        EngineCommand::Generate(req) => {
            let pending = PendingRequest::from_scheduler_request(RequestId(*next_request_id), req);
            *next_request_id += 1;
            if pending_control.is_empty() {
                deferred.push(pending);
            } else {
                post_control_deferred.push(pending);
            }
        }
        EngineCommand::Control(control) => pending_control.push_back(control),
    }
}

fn drain_idle_control(
    executor: &mut impl ModelExecutor,
    pending_control: &mut VecDeque<EngineControlRequest>,
) {
    while let Some(control) = pending_control.pop_front() {
        handle_control_request(executor, control);
    }
}

fn handle_control_request(executor: &mut impl ModelExecutor, control: EngineControlRequest) {
    match control {
        EngineControlRequest::LoadLoraAdapter {
            request,
            response_tx,
        } => {
            info!(
                "LoRA adapter load requested while scheduler is idle: name={}, path={}",
                request.lora_name,
                request.lora_path.display()
            );
            let _ = response_tx.send(
                executor
                    .load_lora_adapter(&request)
                    .map_err(|error| error.to_string()),
            );
        }
        EngineControlRequest::UnloadLoraAdapter {
            request,
            response_tx,
        } => {
            info!(
                "LoRA adapter unload requested while scheduler is idle: name={}",
                request.lora_name
            );
            let _ = response_tx.send(
                executor
                    .unload_lora_adapter(&request)
                    .map_err(|error| error.to_string()),
            );
        }
        EngineControlRequest::ListLoraAdapters { response_tx } => {
            let _ = response_tx.send(Ok(executor.list_lora_adapters()));
        }
    }
}

#[derive(Clone)]
struct RequestFailureTarget {
    request_id: RequestId,
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Why a request was rejected at admission, so the client gets an accurate error.
#[derive(Clone, Copy)]
enum RejectReason {
    /// Worst-case length exceeds the model's position-encoding window.
    ContextLength { limit: usize },
    /// Worst-case length needs more KV blocks than this instance can ever provide.
    KvBudget,
}

struct AdmissionOutcome {
    pending: Vec<PendingRequest>,
    deferred: Vec<PendingRequest>,
    rejected: Vec<(PendingRequest, RejectReason)>,
}

struct LoraValidationOutcome {
    accepted: Vec<PendingRequest>,
    rejected: Vec<PendingRequest>,
}

fn reject_unknown_lora_requests(
    deferred: Vec<PendingRequest>,
    executor: &impl ModelExecutor,
) -> LoraValidationOutcome {
    if !deferred.iter().any(|req| req.lora_adapter.is_some()) {
        return LoraValidationOutcome {
            accepted: deferred,
            rejected: Vec::new(),
        };
    }

    let loaded_lora_adapters = executor.list_lora_adapters();
    let loaded_lora_adapters: HashSet<_> = loaded_lora_adapters.into_iter().collect();
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for req in deferred {
        match req.lora_adapter.as_ref() {
            Some(adapter) if !loaded_lora_adapters.contains(adapter) => rejected.push(req),
            _ => accepted.push(req),
        }
    }

    LoraValidationOutcome { accepted, rejected }
}

fn blocks_needed(token_count: usize, block_size: usize) -> usize {
    token_count.div_ceil(block_size)
}

// Prefill samples the first output token but does not write its KV. A generated
// token's KV is written only when it is fed as the next decode input. Therefore
// N returned completion tokens occupy at most N - 1 generated-token KV slots.
fn max_request_tokens(req: &PendingRequest) -> usize {
    req.prompt_tokens
        .len()
        .saturating_add(req.max_tokens.saturating_sub(1))
}

#[cfg(test)]
fn max_active_tokens(req: &ActiveRequestState) -> usize {
    req.prompt_len
        .saturating_add(req.max_tokens.saturating_sub(1))
}

fn current_active_tokens(req: &ActiveRequestState) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

// Pool blocks a request can draw over its lifetime. One-token completions
// finish after prefill, so schedule_decode never provisions a dangling block.
// Multi-token requests can draw that final dangling decode block, so admission
// reserves prompt + max_tokens for them.
fn request_lifetime_blocks(prompt_len: usize, max_tokens: usize, block_size: usize) -> usize {
    let lifetime_tokens = if max_tokens <= 1 {
        prompt_len
    } else {
        prompt_len.saturating_add(max_tokens)
    };
    lifetime_tokens.div_ceil(block_size).max(1)
}

fn pending_lifetime_blocks(req: &PendingRequest, block_size: usize) -> usize {
    request_lifetime_blocks(req.prompt_tokens.len(), req.max_tokens, block_size)
}

fn active_lifetime_blocks(req: &ActiveRequestState, block_size: usize) -> usize {
    request_lifetime_blocks(req.prompt_len, req.max_tokens, block_size)
}

fn active_future_blocks(active: &[ActiveRequestState], block_size: usize) -> usize {
    active
        .iter()
        .map(|req| {
            active_lifetime_blocks(req, block_size)
                .saturating_sub(blocks_needed(current_active_tokens(req), block_size))
        })
        .sum()
}

/// Free blocks already promised to admitted requests (active decode growth +
/// remaining prefill chunks). A KV prefetch reservation must stay out of this
/// floor or a later chunk/decode fails allocation and kills the whole step.
fn admitted_future_blocks<E: ModelExecutor>(
    executor: &E,
    active: &[ActiveRequestState],
    prefilling: &[PendingRequest],
) -> usize {
    let block_size = executor.block_size();
    active_future_blocks(active, block_size)
        + prefilling_future_blocks(prefilling, block_size, |id| executor.prefetched_blocks(id))
}

fn prefilling_future_blocks(
    prefilling: &[PendingRequest],
    block_size: usize,
    // Blocks a request already holds via a settled prefetch (zero once its
    // first chunk absorbs them). They are out of the free pool, so counting
    // them as future need would double-charge the budget.
    prefetch_credit: impl Fn(RequestId) -> usize,
) -> usize {
    prefilling
        .iter()
        .map(|req| {
            pending_lifetime_blocks(req, block_size)
                .saturating_sub(blocks_needed(req.prefill_pos, block_size))
                .saturating_sub(prefetch_credit(req.request_id))
        })
        .sum()
}

/// Default for `max_prefill_tokens`: prompt tokens forwarded in a single step
/// (chunked prefill). Prefill activation scratch scales with the step's total
/// prompt tokens (~22 KB/token measured on Qwen3-4B), so an unbounded prefill
/// batch can eat the post-KV-pool VRAM headroom and OOM mid-serving under a
/// request burst. Prompts longer than the budget are split across steps, so
/// long prompts can't monopolize a step and starve running decodes.
/// Exception: echo requests need all-position logits in one forward and run
/// whole regardless of the budget — an oversized echo prompt still spikes
/// activation memory.
///
/// A unified step's duration scales with its prefill tokens, and every decode
/// request in the batch stalls for the whole step — the budget bounds that
/// stall. 1024 halves ITL p99 vs 2048 at mid-load with the same mean TPOT;
/// 512 chunks no longer amortize the per-step fixed cost, so prefill falls
/// behind arrivals and TTFT queues up.
pub const DEFAULT_MAX_PREFILL_TOKENS: usize = 1024;

fn admit_deferred_requests(
    deferred: Vec<PendingRequest>,
    active: &[ActiveRequestState],
    // Admitted requests still mid-prefill: they hold KV for their applied
    // chunks and will take a decode slot when they promote, so admission
    // must reserve both or completing chunks can overshoot capacity.
    prefilling: &[PendingRequest],
    block_size: usize,
    available_blocks: usize,
    max_request_blocks: usize,
    max_context_tokens: usize,
    max_decode_batch_size: usize,
    // Blocks a request already holds from a settled prefetch. These are already
    // out of `available_blocks`, so they must be credited against the request's
    // need or admission double-counts them and can wedge a near-budget CPU-hit
    // request forever (never admitted, prefetch never released).
    prefetch_credit: impl Fn(RequestId) -> usize,
) -> AdmissionOutcome {
    let mut budget = available_blocks
        .saturating_sub(active_future_blocks(active, block_size))
        .saturating_sub(prefilling_future_blocks(
            prefilling,
            block_size,
            &prefetch_credit,
        ));
    let mut decode_slots = max_decode_batch_size
        .saturating_sub(active.len())
        .saturating_sub(prefilling.len());
    let mut pending = Vec::new();
    let mut still_deferred = Vec::new();
    let mut rejected = Vec::new();

    for req in deferred {
        // Reject if the full sequence overflows the position-encoding window
        if req.prompt_tokens.len().saturating_add(req.max_tokens) > max_context_tokens {
            rejected.push((
                req,
                RejectReason::ContextLength {
                    limit: max_context_tokens,
                },
            ));
            continue;
        }

        // Full physical footprint gates the per-request cap (a request occupies
        // all of it, prefetched or not)…
        let footprint = pending_lifetime_blocks(&req, block_size);
        if footprint > max_request_blocks {
            rejected.push((req, RejectReason::KvBudget));
            continue;
        }

        // …but only the blocks not already held by this request's prefetch must
        // come from the free-pool budget.
        let fresh_needed = footprint.saturating_sub(prefetch_credit(req.request_id));
        if fresh_needed <= budget && decode_slots > 0 {
            budget -= fresh_needed;
            decode_slots -= 1;
            debug!(
                "request admitted: request_id={:?} prompt_len={} max_tokens={}",
                req.request_id,
                req.prompt_tokens.len(),
                req.max_tokens
            );
            pending.push(req);
        } else {
            still_deferred.push(req);
        }
    }

    AdmissionOutcome {
        pending,
        deferred: still_deferred,
        rejected,
    }
}

fn send_rejection(req: &PendingRequest, reason: RejectReason) {
    let message = match reason {
        RejectReason::ContextLength { limit } => format!(
            "request exceeds this model's maximum context length of {} tokens: requested {} (prompt={} + max_tokens={})",
            limit,
            req.prompt_tokens.len().saturating_add(req.max_tokens),
            req.prompt_tokens.len(),
            req.max_tokens
        ),
        RejectReason::KvBudget => format!(
            "request requires more KV blocks than this model instance can provide: prompt_tokens={}, max_request_tokens={}",
            req.prompt_tokens.len(),
            max_request_tokens(req)
        ),
    };
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message,
        prompt_tokens: req.prompt_tokens.len(),
        completion_tokens: 0,
    });
}

fn send_unknown_lora_rejection(req: &PendingRequest) {
    let adapter = req.lora_adapter.as_deref().unwrap_or("<missing>");
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message: format!("LoRA adapter is not loaded: {adapter}"),
        prompt_tokens: req.prompt_tokens.len(),
        completion_tokens: 0,
    });
}

fn failure_targets_for(
    active: &[ActiveRequestState],
    plan: &self::plan::ExecutionPlan,
) -> Vec<RequestFailureTarget> {
    let mut targets = Vec::new();
    match plan {
        self::plan::ExecutionPlan::Prefill { pending } => {
            targets.extend(pending.iter().map(pending_failure_target));
        }
        self::plan::ExecutionPlan::Decode => {
            targets.extend(active.iter().map(active_failure_target));
        }
        self::plan::ExecutionPlan::Unified { pending } => {
            targets.extend(active.iter().map(active_failure_target));
            targets.extend(pending.iter().map(pending_failure_target));
        }
    }
    targets
}

fn active_failure_target(req: &ActiveRequestState) -> RequestFailureTarget {
    RequestFailureTarget {
        request_id: req.request_id,
        token_tx: req.token_tx.clone(),
        prompt_tokens: req.prompt_len,
        completion_tokens: req.generated_count,
    }
}

fn pending_failure_target(req: &PendingRequest) -> RequestFailureTarget {
    RequestFailureTarget {
        request_id: req.request_id,
        token_tx: req.token_tx.clone(),
        prompt_tokens: req.prompt_tokens.len(),
        completion_tokens: 0,
    }
}

fn fail_touched_requests(
    executor: &mut impl ModelExecutor,
    active: &mut Vec<ActiveRequestState>,
    targets: Vec<RequestFailureTarget>,
    message: &str,
) {
    for target in targets {
        let _ = target.token_tx.send(TokenEvent::Error {
            message: message.to_string(),
            prompt_tokens: target.prompt_tokens,
            completion_tokens: target.completion_tokens,
        });
        if let Err(error) = executor.drop_request(target.request_id) {
            warn!(
                "failed to drop request state after execution error for {:?}: {error}",
                target.request_id
            );
        }
    }
    active.clear();
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use anyhow::Result;
    use openinfer_core::engine::{
        EngineControlError, LoadLoraAdapterRequest, UnloadLoraAdapterRequest,
    };
    use openinfer_kv_cache::BlockPool;

    use super::*;
    use crate::executor::{
        DecodePlan, DecodeRequestResult, PrefillPlan, PrefillRequestResult, PrefillResult,
        PrefillStepItem, UnifiedPlan, UnifiedResult,
    };

    struct FakeExecutor {
        block_size: usize,
        cached_tokens: usize,
        max_request_blocks: usize,
        max_context_tokens: usize,
        available_blocks: usize,
        held_tokens: HashMap<RequestId, usize>,
        // Prompt progress of requests mid-chunked-prefill (mirrors the real
        // executor's kv_position so multi-chunk scheduling is exercised).
        prefill_positions: HashMap<RequestId, usize>,
        fail_decode_once: bool,
        decode_delay: Duration,
        loaded_lora_adapters: HashSet<String>,
        dropped: Arc<Mutex<Vec<u64>>>,
        prefetch_offers: Arc<Mutex<Vec<u64>>>,
        prefill_batches: Arc<Mutex<Vec<Vec<RequestId>>>>,
        decode_batches: Arc<Mutex<Vec<Vec<RequestId>>>>,
        prefill_lora_batches: Arc<Mutex<Vec<Vec<Option<String>>>>>,
        decode_lora_batches: Arc<Mutex<Vec<Vec<Option<String>>>>>,
    }

    impl FakeExecutor {
        fn new(max_request_blocks: usize, dropped: Arc<Mutex<Vec<u64>>>) -> Self {
            Self {
                block_size: 16,
                cached_tokens: 0,
                max_request_blocks,
                max_context_tokens: usize::MAX,
                available_blocks: max_request_blocks,
                held_tokens: HashMap::new(),
                prefill_positions: HashMap::new(),
                fail_decode_once: false,
                decode_delay: Duration::ZERO,
                loaded_lora_adapters: HashSet::new(),
                dropped,
                prefetch_offers: Arc::new(Mutex::new(Vec::new())),
                prefill_batches: Arc::new(Mutex::new(Vec::new())),
                decode_batches: Arc::new(Mutex::new(Vec::new())),
                prefill_lora_batches: Arc::new(Mutex::new(Vec::new())),
                decode_lora_batches: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_cached_tokens(mut self, cached_tokens: usize) -> Self {
            self.cached_tokens = cached_tokens;
            self
        }

        fn with_decode_failure(mut self) -> Self {
            self.fail_decode_once = true;
            self
        }

        fn with_max_context_tokens(mut self, max_context_tokens: usize) -> Self {
            self.max_context_tokens = max_context_tokens;
            self
        }

        fn with_decode_delay(mut self, delay: Duration) -> Self {
            self.decode_delay = delay;
            self
        }

        fn with_lora_adapters(mut self, names: &[&str]) -> Self {
            self.loaded_lora_adapters = names.iter().map(|name| (*name).to_string()).collect();
            self
        }

        fn with_batch_records(
            mut self,
            prefill_batches: Arc<Mutex<Vec<Vec<RequestId>>>>,
            decode_batches: Arc<Mutex<Vec<Vec<RequestId>>>>,
        ) -> Self {
            self.prefill_batches = prefill_batches;
            self.decode_batches = decode_batches;
            self
        }

        fn with_lora_batch_records(
            mut self,
            prefill_lora_batches: Arc<Mutex<Vec<Vec<Option<String>>>>>,
            decode_lora_batches: Arc<Mutex<Vec<Vec<Option<String>>>>>,
        ) -> Self {
            self.prefill_lora_batches = prefill_lora_batches;
            self.decode_lora_batches = decode_lora_batches;
            self
        }

        /// Advance a request's prompt by one chunk, mirroring the real
        /// executor: clamp the scheduler's budget to the tokens remaining
        /// and report the new authoritative position.
        fn fake_prefill_result(&mut self, req: &PrefillStepItem) -> PrefillRequestResult {
            let start = self
                .prefill_positions
                .get(&req.request_id)
                .copied()
                .unwrap_or(0);
            let chunk = (req.prompt_tokens.len() - start).min(req.chunk_budget);
            let prefill_pos = start + chunk;
            let completed = prefill_pos == req.prompt_tokens.len();
            if completed {
                self.prefill_positions.remove(&req.request_id);
            } else {
                self.prefill_positions.insert(req.request_id, prefill_pos);
            }
            PrefillRequestResult {
                request_id: req.request_id,
                first_token: 100 + req.request_id.get() as u32,
                first_token_logprob: None,
                prompt_logprobs: None,
                cached_tokens: self.cached_tokens,
                completed,
                prefill_pos,
            }
        }

        fn ensure_request_tokens(
            &mut self,
            request_id: RequestId,
            token_count: usize,
        ) -> Result<()> {
            let current_tokens = self.held_tokens.get(&request_id).copied().unwrap_or(0);
            let current_blocks = blocks_needed(current_tokens, self.block_size);
            let needed_blocks = blocks_needed(token_count, self.block_size);
            let grow = needed_blocks.saturating_sub(current_blocks);
            if grow > self.available_blocks {
                anyhow::bail!("fake KV capacity exhausted");
            }
            self.available_blocks -= grow;
            self.held_tokens.insert(request_id, token_count);
            Ok(())
        }
    }

    impl ModelExecutor for FakeExecutor {
        fn block_size(&self) -> usize {
            self.block_size
        }

        fn max_request_blocks(&self) -> usize {
            self.max_request_blocks
        }

        fn max_context_tokens(&self) -> usize {
            self.max_context_tokens
        }

        fn max_decode_batch_size(&self) -> usize {
            64
        }

        fn available_blocks(&self) -> usize {
            self.available_blocks
        }

        fn is_stop_token(&self, _token_id: u32) -> bool {
            false
        }

        fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
            if let Some(tokens) = self.held_tokens.remove(&request_id) {
                self.available_blocks += blocks_needed(tokens, self.block_size);
            }
            self.prefill_positions.remove(&request_id);
            self.dropped.lock().unwrap().push(request_id.get());
            Ok(())
        }

        fn begin_kv_prefetch(
            &mut self,
            request_id: RequestId,
            _prompt_tokens: &[u32],
            _lora_adapter: Option<&str>,
            _reserve_floor: usize,
        ) -> bool {
            self.prefetch_offers.lock().unwrap().push(request_id.get());
            false
        }

        fn list_lora_adapters(&self) -> Vec<String> {
            let mut names: Vec<_> = self.loaded_lora_adapters.iter().cloned().collect();
            names.sort();
            names
        }

        fn unload_lora_adapter(&mut self, request: &UnloadLoraAdapterRequest) -> Result<()> {
            anyhow::ensure!(
                self.loaded_lora_adapters.remove(&request.lora_name),
                "LoRA adapter is not loaded: {}",
                request.lora_name
            );
            Ok(())
        }

        fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
            self.prefill_batches.lock().unwrap().push(
                plan.requests
                    .iter()
                    .map(|request| request.request_id)
                    .collect(),
            );
            self.prefill_lora_batches.lock().unwrap().push(
                plan.requests
                    .iter()
                    .map(|request| request.lora_adapter.clone())
                    .collect(),
            );
            for req in plan.requests {
                self.ensure_request_tokens(req.request_id, req.prompt_tokens.len())?;
            }
            Ok(PrefillResult {
                requests: plan
                    .requests
                    .iter()
                    .map(|req| self.fake_prefill_result(req))
                    .collect(),
            })
        }

        fn execute_decode(
            &mut self,
            plan: DecodePlan<'_>,
        ) -> Result<crate::executor::DecodeResult> {
            if !self.decode_delay.is_zero() {
                std::thread::sleep(self.decode_delay);
            }
            if self.fail_decode_once {
                self.fail_decode_once = false;
                anyhow::bail!("fake decode KV capacity exhausted");
            }

            self.decode_batches.lock().unwrap().push(
                plan.requests
                    .iter()
                    .map(|request| request.request_id)
                    .collect(),
            );
            self.decode_lora_batches.lock().unwrap().push(
                plan.requests
                    .iter()
                    .map(|request| request.lora_adapter.clone())
                    .collect(),
            );
            for req in plan.requests {
                let current_tokens = self
                    .held_tokens
                    .get(&req.request_id)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing fake request state"))?;
                self.ensure_request_tokens(req.request_id, current_tokens + 1)?;
            }

            Ok(crate::executor::DecodeResult {
                requests: plan
                    .requests
                    .iter()
                    .map(|req| DecodeRequestResult {
                        request_id: req.request_id,
                        token: 200 + req.request_id.get() as u32,
                        logprob: None,
                    })
                    .collect(),
            })
        }

        fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
            self.prefill_batches.lock().unwrap().push(
                plan.prefill_requests
                    .iter()
                    .map(|request| request.request_id)
                    .collect(),
            );
            self.prefill_lora_batches.lock().unwrap().push(
                plan.prefill_requests
                    .iter()
                    .map(|request| request.lora_adapter.clone())
                    .collect(),
            );
            self.decode_batches.lock().unwrap().push(
                plan.decode_requests
                    .iter()
                    .map(|request| request.request_id)
                    .collect(),
            );
            self.decode_lora_batches.lock().unwrap().push(
                plan.decode_requests
                    .iter()
                    .map(|request| request.lora_adapter.clone())
                    .collect(),
            );
            for req in plan.prefill_requests {
                self.ensure_request_tokens(req.request_id, req.prompt_tokens.len())?;
            }
            for req in plan.decode_requests {
                let current_tokens = self
                    .held_tokens
                    .get(&req.request_id)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing fake request state"))?;
                self.ensure_request_tokens(req.request_id, current_tokens + 1)?;
            }

            Ok(UnifiedResult {
                prefill_requests: plan
                    .prefill_requests
                    .iter()
                    .map(|req| self.fake_prefill_result(req))
                    .collect(),
                decode_requests: plan
                    .decode_requests
                    .iter()
                    .map(|req| DecodeRequestResult {
                        request_id: req.request_id,
                        token: 200 + req.request_id.get() as u32,
                        logprob: None,
                    })
                    .collect(),
            })
        }
    }

    #[test]
    fn kv_budget_distinguishes_written_tokens_from_lifetime_blocks() {
        let (pending_req, _pending_rx) = request(16, 1);
        let pending = PendingRequest::from_scheduler_request(RequestId(7), pending_req);
        assert_eq!(max_request_tokens(&pending), 16);
        assert_eq!(blocks_needed(max_request_tokens(&pending), 16), 1);
        assert_eq!(pending_lifetime_blocks(&pending, 16), 1);

        let (pending_req, _pending_rx) = request(16, 17);
        let pending = PendingRequest::from_scheduler_request(RequestId(8), pending_req);
        assert_eq!(max_request_tokens(&pending), 32);
        assert_eq!(blocks_needed(max_request_tokens(&pending), 16), 2);
        assert_eq!(pending_lifetime_blocks(&pending, 16), 3);

        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        let after_prefill = ActiveRequestState {
            request_id: RequestId(8),
            lora_adapter: None,
            token_tx,
            last_token: 100,
            generated_count: 1,
            max_tokens: 17,
            prompt_len: 16,
            params: SamplingParams::default(),
            logprobs: 0,
        };
        assert_eq!(current_active_tokens(&after_prefill), 16);
        assert_eq!(max_active_tokens(&after_prefill), 32);
        assert_eq!(active_lifetime_blocks(&after_prefill, 16), 3);

        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        let after_one_decode = ActiveRequestState {
            request_id: RequestId(9),
            lora_adapter: None,
            token_tx,
            last_token: 200,
            generated_count: 2,
            max_tokens: 17,
            prompt_len: 16,
            params: SamplingParams::default(),
            logprobs: 0,
        };
        assert_eq!(current_active_tokens(&after_one_decode), 17);
        assert_eq!(max_active_tokens(&after_one_decode), 32);
        assert_eq!(active_lifetime_blocks(&after_one_decode, 16), 3);
    }

    #[test]
    fn admission_splits_deferred_into_pending_deferred_and_rejected() {
        // block_size 16, per-request cap 4 blocks (max 64 tokens). One active
        // request is mid-flight and will grow into 2 more blocks, so it
        // pre-reserves them out of the budget.
        let (token_tx, _rx) = mpsc::unbounded_channel();
        let active = [ActiveRequestState {
            request_id: RequestId(0),
            lora_adapter: None,
            token_tx,
            last_token: 1,
            generated_count: 1, // current tokens = prompt_len (16) -> 1 block
            max_tokens: 18,     // lifetime tokens = 16 + 18 = 34 -> 3 blocks; future growth = 2
            prompt_len: 16,
            params: SamplingParams::default(),
            logprobs: 0,
        }];

        let mk = |id: u64, prompt_len, max_tokens| {
            PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, max_tokens).0)
        };
        let deferred = vec![
            mk(1, 16, 1), // one-token completion on a page boundary: admitted
            mk(2, 16, 1), // 1 block: admitted, budget now 0
            mk(3, 16, 1), // 1 block: no budget left -> stays deferred
            mk(4, 80, 1), // 80 prompt tokens -> 5 blocks > cap of 4 -> rejected outright
        ];

        // available 4 blocks - 2 reserved for active growth = budget of 2.
        let outcome =
            admit_deferred_requests(deferred, &active, &[], 16, 4, 4, usize::MAX, 64, |_| 0);

        let ids =
            |reqs: &[PendingRequest]| reqs.iter().map(|r| r.request_id.get()).collect::<Vec<_>>();
        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "admit in order until the budget is spent"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "budget-starved requests stay deferred, not dropped"
        );
        let rejected_ids = outcome
            .rejected
            .iter()
            .map(|(r, _)| r.request_id.get())
            .collect::<Vec<_>>();
        assert_eq!(
            rejected_ids,
            vec![4],
            "requests larger than the per-request cap are rejected outright"
        );
    }

    #[test]
    fn requests_exceeding_context_window_are_rejected() {
        let active: [ActiveRequestState; 0] = [];
        let mk = |id: u64, prompt_len, max_tokens| {
            PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, max_tokens).0)
        };

        let deferred = vec![
            mk(1, 16, 16), // request 1: 16 prompt + 16 max = 32 total: admitted
            mk(2, 16, 17), // request 2: 16 prompt + 17 max = 33 total: overflows by 1 token → rejected
            mk(3, 40, 1), // request 3: 40 prompt + 1 max = 41 total: overflows by 9 tokens → rejected
        ];

        let outcome =
            admit_deferred_requests(deferred, &active, &[], 16, 1000, 1000, 32, 64, |_| 0);

        let pending_ids = outcome
            .pending
            .iter()
            .map(|r| r.request_id.get())
            .collect::<Vec<_>>();
        assert_eq!(
            pending_ids,
            vec![1],
            "only the request that fits the window is admitted; overflows are rejected, not clamped"
        );

        let rejected_ids = outcome
            .rejected
            .iter()
            .map(|(r, _)| r.request_id.get())
            .collect::<Vec<_>>();
        assert_eq!(rejected_ids, vec![2, 3]);
        for (_, reason) in &outcome.rejected {
            assert!(
                matches!(reason, RejectReason::ContextLength { limit: 32 }),
                "rejected on the context window, not the KV budget"
            );
        }
    }

    #[test]
    fn admission_respects_decode_batch_capacity() {
        let mut active = Vec::new();
        for id in 0..64 {
            let (token_tx, _rx) = mpsc::unbounded_channel();
            active.push(ActiveRequestState {
                request_id: RequestId(id),
                lora_adapter: None,
                token_tx,
                last_token: 1,
                generated_count: 1,
                max_tokens: 2,
                prompt_len: 16,
                params: SamplingParams::default(),
                logprobs: 0,
            });
        }
        let pending = PendingRequest::from_scheduler_request(RequestId(64), request(16, 1).0);

        let outcome = admit_deferred_requests(
            vec![pending],
            &active,
            &[],
            16,
            1024,
            1024,
            usize::MAX,
            64,
            |_| 0,
        );

        assert!(
            outcome.pending.is_empty(),
            "new request must not be admitted past decode scratch capacity"
        );
        assert_eq!(
            outcome.deferred[0].request_id,
            RequestId(64),
            "capacity-starved request should stay deferred"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn prefill_chunking_caps_step_tokens_and_keeps_fifo_progress() {
        let mk = |id: u64, prompt_len, max_tokens| {
            PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, max_tokens).0)
        };

        // A prompt larger than the budget is split: the head request gets a
        // budget-sized chunk and everyone behind it waits.
        let mut prefilling = vec![mk(1, 64, 1), mk(2, 16, 1)];
        let taken = take_prefill_chunks(&mut prefilling, 32);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].request_id, RequestId(1));
        assert_eq!(taken[0].step_chunk, 32, "chunk is capped at the budget");
        assert_eq!(
            prefilling[0].request_id,
            RequestId(2),
            "follow-up waits for the next step once the budget is spent"
        );

        // Requests pack until the budget is filled exactly; the overflow stays
        // queued in arrival order.
        let mut prefilling = vec![mk(3, 16, 1), mk(4, 16, 1), mk(5, 16, 1)];
        let taken = take_prefill_chunks(&mut prefilling, 32);
        assert_eq!(
            taken.iter().map(|r| r.step_chunk).collect::<Vec<_>>(),
            vec![16, 16],
            "16 + 16 fills the 32-token budget"
        );
        assert_eq!(prefilling[0].request_id, RequestId(5));

        // A partially-prefilled head request only consumes its remainder.
        let mut head = mk(6, 64, 1);
        head.prefill_pos = 48;
        let mut prefilling = vec![head, mk(7, 16, 1)];
        let taken = take_prefill_chunks(&mut prefilling, 32);
        assert_eq!(
            taken.iter().map(|r| r.step_chunk).collect::<Vec<_>>(),
            vec![16, 16],
            "remainder of the chunked head + the next request share the step"
        );
        assert!(prefilling.is_empty());
    }

    #[test]
    fn echo_requests_never_split_but_run_alone_when_oversized() {
        let mk_echo = |id: u64, prompt_len| {
            let (req, _rx) = request(prompt_len, 1);
            let mut pending = PendingRequest::from_scheduler_request(RequestId(id), req);
            pending.echo = true;
            pending
        };
        let mk = |id: u64, prompt_len| {
            PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, 1).0)
        };

        // Oversized echo at the head of an empty step runs whole — chunking it
        // would lose the all-position logits echo needs.
        let mut prefilling = vec![mk_echo(1, 64), mk(2, 16)];
        let taken = take_prefill_chunks(&mut prefilling, 32);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].step_chunk, 64, "echo takes its full prompt");
        assert_eq!(prefilling[0].request_id, RequestId(2));

        // An echo that doesn't fit behind earlier work is skipped, not split;
        // later requests may still fill the leftover budget, and the step set
        // stays sorted by request id.
        let mut prefilling = vec![mk(3, 24), mk_echo(4, 16), mk(5, 8)];
        let taken = take_prefill_chunks(&mut prefilling, 32);
        assert_eq!(
            taken
                .iter()
                .map(|r| (r.request_id.get(), r.step_chunk))
                .collect::<Vec<_>>(),
            vec![(3, 24), (5, 8)],
            "echo skipped, leftover budget goes to the next non-echo request"
        );
        assert_eq!(prefilling[0].request_id, RequestId(4));
    }

    #[test]
    fn long_prompt_chunks_across_steps_and_still_completes() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let prefill_batches = Arc::new(Mutex::new(Vec::new()));
        let decode_batches = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(64, Arc::clone(&dropped))
            .with_batch_records(Arc::clone(&prefill_batches), Arc::clone(&decode_batches));
        // 8-token chunk budget: a 32-token prompt needs 4 prefill steps.
        let handle = start_with_executor(executor, 42, 8);

        let (req, mut rx) = request(32, 2);
        handle.submit(req).expect("submit chunked request");
        match rx.blocking_recv() {
            Some(TokenEvent::Scheduled { prompt_tokens, .. }) => {
                assert_eq!(
                    prompt_tokens, 32,
                    "Scheduled fires once, on the first chunk"
                );
            }
            _ => panic!("stream opens with Scheduled"),
        }
        assert!(
            matches!(rx.blocking_recv(), Some(TokenEvent::Token { id: 100, .. })),
            "first token arrives only after the final chunk, with no duplicate Scheduled"
        );
        assert!(
            matches!(rx.blocking_recv(), Some(TokenEvent::Token { id: 200, .. })),
            "decode continues normally after promotion"
        );
        assert!(matches!(
            rx.blocking_recv(),
            Some(TokenEvent::Finished { .. })
        ));

        let batches = prefill_batches.lock().unwrap();
        assert_eq!(
            batches.len(),
            4,
            "32-token prompt under an 8-token budget takes exactly 4 chunk steps"
        );
        assert!(
            batches.iter().all(|b| b == &vec![RequestId(0)]),
            "every chunk step carries the same request"
        );
    }

    #[test]
    fn disconnect_mid_chunk_drops_the_request() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(64, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42, 8);

        let (req, rx) = request(32, 2);
        handle.submit(req).expect("submit chunked request");
        drop(rx); // client goes away while chunks are still in flight

        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if dropped.lock().unwrap().contains(&0) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("request KV was not dropped after a mid-prefill disconnect");
    }

    #[test]
    fn one_token_completion_on_page_boundary_fits_one_page() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(1, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (fits_exactly, mut rx) = request(16, 1);
        handle.submit(fits_exactly).expect("submit fits_exactly");
        assert!(
            matches!(
                recv_skipping_scheduled(&mut rx),
                Some(TokenEvent::Token { id: 100, .. })
            ),
            "prefill should emit the sampled token"
        );
        assert!(
            matches!(
                recv_skipping_scheduled(&mut rx),
                Some(TokenEvent::Finished { .. })
            ),
            "one-token completion should finish without a decode KV page"
        );
        assert!(
            dropped.lock().unwrap().contains(&0),
            "finished request should release its one prompt page"
        );
    }

    /// The engine stream opens with exactly one `Scheduled` event carrying
    /// the executor-reported prefix-cache hit count (#246) — before any
    /// token, and never again.
    #[test]
    fn scheduled_event_reports_cached_tokens_exactly_once() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped)).with_cached_tokens(7);
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (req, mut rx) = request(32, 2);
        handle.submit(req).expect("submit");

        match rx.blocking_recv() {
            Some(TokenEvent::Scheduled {
                prompt_tokens,
                cached_tokens,
                queued_at_unix_s,
                scheduled_at_unix_s,
            }) => {
                assert_eq!(prompt_tokens, 32);
                assert_eq!(cached_tokens, 7);
                assert!(queued_at_unix_s <= scheduled_at_unix_s);
            }
            _ => panic!("first event must be Scheduled"),
        }

        loop {
            match rx.blocking_recv() {
                Some(TokenEvent::Scheduled { .. }) => {
                    panic!("Scheduled must be emitted exactly once")
                }
                Some(TokenEvent::Finished { .. }) => break,
                Some(_) => {}
                None => panic!("stream closed without Finished"),
            }
        }
    }

    #[test]
    fn page_boundary_lifetime_blocks_gate_admission() {
        let active: [ActiveRequestState; 0] = [];
        let mk = |id: u64, prompt_len, max_tokens| {
            PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, max_tokens).0)
        };

        let under_reserved = admit_deferred_requests(
            vec![mk(1, 16, 17)],
            &active,
            &[],
            16,
            2,
            2,
            usize::MAX,
            64,
            |_| 0,
        );
        assert!(
            under_reserved.pending.is_empty(),
            "old prompt + max_tokens - 1 arithmetic would admit this request with 2 blocks"
        );
        assert_eq!(under_reserved.rejected.len(), 1);
        assert!(
            matches!(under_reserved.rejected[0].1, RejectReason::KvBudget),
            "request needs 3 lifetime blocks: ceil((16 + 17) / 16)"
        );

        let exactly_reserved = admit_deferred_requests(
            vec![mk(2, 16, 17)],
            &active,
            &[],
            16,
            3,
            3,
            usize::MAX,
            64,
            |_| 0,
        );
        assert_eq!(
            exactly_reserved.pending[0].request_id,
            RequestId(2),
            "ceil((prompt + max_tokens) / block_size) admits the request"
        );
        assert!(exactly_reserved.rejected.is_empty());
    }

    fn kvbm_peak_draw(prompt_len: usize, max_tokens: usize, block_size: usize) -> usize {
        let pool = BlockPool::new(block_size, 512).expect("test block pool");
        let base = pool.available_blocks();
        let mut peak = 0usize;
        let mut kv = pool.new_request(vec![1; prompt_len], max_tokens, None);

        kv.schedule_prefill(prompt_len, &pool)
            .expect("schedule prefill");
        peak = peak.max(base - pool.available_blocks());
        kv.apply_prefill(100, &pool).expect("apply prefill");
        peak = peak.max(base - pool.available_blocks());

        for step in 1..max_tokens {
            kv.schedule_decode(&pool).expect("schedule decode");
            peak = peak.max(base - pool.available_blocks());
            kv.apply_decode(100 + step as u32, &pool)
                .expect("apply decode");
            peak = peak.max(base - pool.available_blocks());
        }

        kv.release().expect("release request kv");
        assert_eq!(
            pool.available_blocks(),
            base,
            "probe must release every block it draws"
        );
        peak
    }

    #[test]
    fn lifetime_blocks_match_kvbm_peak_draw_at_issue_boundaries() {
        let block_size = 16;
        for (prompt_len, max_tokens) in [(16usize, 17usize), (1, 16), (17, 16)] {
            let reserved = request_lifetime_blocks(prompt_len, max_tokens, block_size);
            let peak = kvbm_peak_draw(prompt_len, max_tokens, block_size);
            let old = blocks_needed(
                prompt_len.saturating_add(max_tokens.saturating_sub(1)),
                block_size,
            );
            assert_eq!(
                peak, reserved,
                "prompt={prompt_len} max_tokens={max_tokens}"
            );
            assert_eq!(
                old + 1,
                peak,
                "old prompt + max_tokens - 1 arithmetic under-reserved by one block"
            );
        }

        let prompt_len = 33usize;
        let max_tokens = 100usize;
        let reserved = request_lifetime_blocks(prompt_len, max_tokens, block_size);
        let peak = kvbm_peak_draw(prompt_len, max_tokens, block_size);
        let old = blocks_needed(
            prompt_len.saturating_add(max_tokens.saturating_sub(1)),
            block_size,
        );
        assert_eq!(peak, reserved);
        assert_eq!(
            old, reserved,
            "non-boundary case should not reserve more than the old arithmetic"
        );
    }

    #[test]
    fn lifetime_blocks_never_under_reserve_kvbm_peak_draw() {
        let block_size = 16;
        for prompt_len in 1usize..=64 {
            for max_tokens in 1usize..=64 {
                let reserved = request_lifetime_blocks(prompt_len, max_tokens, block_size);
                let peak = kvbm_peak_draw(prompt_len, max_tokens, block_size);
                assert!(
                    peak <= reserved,
                    "prompt={prompt_len} max_tokens={max_tokens}: peak={peak}, reserved={reserved}"
                );
            }
        }
    }

    #[test]
    fn request_waits_for_full_kv_budget_before_prefill() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (long_running, mut long_rx) = request(16, 18);
        handle.submit(long_running).expect("submit long_running");
        assert!(
            matches!(
                recv_skipping_scheduled(&mut long_rx),
                Some(TokenEvent::Token { id: 100, .. })
            ),
            "first request should prefill"
        );

        let (must_wait, mut wait_rx) = request(17, 1);
        handle.submit(must_wait).expect("submit must_wait");

        assert!(
            matches!(
                recv_skipping_scheduled(&mut wait_rx),
                Some(TokenEvent::Token { id: 101, .. })
            ),
            "waiting request should start once the active request releases its full KV budget"
        );
        assert!(
            dropped.lock().unwrap().contains(&0),
            "second request was admitted before the first request released KV"
        );
        assert!(
            matches!(
                recv_skipping_scheduled(&mut wait_rx),
                Some(TokenEvent::Finished { .. })
            ),
            "waiting request should finish after admission"
        );
    }

    /// Engine streams now open with `TokenEvent::Scheduled` (#246); these
    /// tests assert on the token/terminal events, so skip past it.
    fn recv_skipping_scheduled(rx: &mut mpsc::UnboundedReceiver<TokenEvent>) -> Option<TokenEvent> {
        loop {
            match rx.blocking_recv() {
                Some(TokenEvent::Scheduled { .. }) => {}
                other => return other,
            }
        }
    }

    fn pending(request_id: u64, echo: bool) -> PendingRequest {
        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        PendingRequest {
            request_id: RequestId::new(request_id),
            lora_adapter: None,
            prompt_tokens: vec![1; 32],
            params: SamplingParams::default(),
            max_tokens: 1,
            token_tx,
            logprobs: 0,
            echo,
            queued_at_unix_s: None,
            prefetch_offered: false,
            prefill_pos: 0,
            step_chunk: 0,
            cached_tokens: 0,
        }
    }

    #[test]
    fn echo_requests_are_never_offered_to_prefetch() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let mut executor = FakeExecutor::new(64, dropped);
        let offers = Arc::clone(&executor.prefetch_offers);

        let mut deferred = vec![pending(1, true), pending(2, false)];
        let mut loading = Vec::new();
        offer_prefetch(&mut executor, &mut deferred, &mut loading, 0);

        // The plain request is probed; the echo request is skipped entirely, so
        // its prefill forwards the whole prompt without parking unspendable KV.
        assert_eq!(*offers.lock().unwrap(), vec![2]);
        let echo = deferred.iter().find(|r| r.request_id.get() == 1).unwrap();
        assert!(!echo.prefetch_offered, "echo request must stay un-probed");
        let plain = deferred.iter().find(|r| r.request_id.get() == 2).unwrap();
        assert!(
            plain.prefetch_offered,
            "plain request must be marked probed"
        );
    }

    fn request(
        prompt_len: usize,
        max_tokens: usize,
    ) -> (GenerateRequest, mpsc::UnboundedReceiver<TokenEvent>) {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        (
            GenerateRequest {
                request_id: None,
                queued_at_unix_s: None,
                prompt_tokens: vec![1; prompt_len],
                params: SamplingParams::default(),
                max_tokens,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            token_rx,
        )
    }

    fn request_with_lora(
        prompt_len: usize,
        max_tokens: usize,
        lora_adapter: Option<&str>,
    ) -> (GenerateRequest, mpsc::UnboundedReceiver<TokenEvent>) {
        let (mut request, token_rx) = request(prompt_len, max_tokens);
        request.lora_adapter = lora_adapter.map(ToString::to_string);
        (request, token_rx)
    }

    fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if predicate() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn impossible_request_is_rejected_without_blocking_later_work() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(2, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (too_large, mut too_large_rx) = request(16, 34);
        handle.submit(too_large).expect("submit too_large");
        match too_large_rx.blocking_recv() {
            Some(TokenEvent::Rejected {
                prompt_tokens,
                completion_tokens,
                message,
            }) => {
                assert_eq!(prompt_tokens, 16);
                assert_eq!(completion_tokens, 0);
                assert!(message.contains("requires more KV blocks"));
            }
            _ => panic!("oversized request should be rejected"),
        }

        let (fits, mut fits_rx) = request(16, 1);
        handle.submit(fits).expect("submit fits");
        match recv_skipping_scheduled(&mut fits_rx) {
            Some(TokenEvent::Token { id, .. }) => assert_eq!(id, 101),
            _ => panic!("later fitting request should emit a token"),
        }
        assert!(
            matches!(
                recv_skipping_scheduled(&mut fits_rx),
                Some(TokenEvent::Finished { .. })
            ),
            "later fitting request should finish"
        );
    }

    /// End-to-end through the real scheduler loop (no GPU): a request whose
    /// prompt + max_tokens exceeds the context window is rejected with a context-length error
    #[test]
    fn over_context_window_request_is_rejected_through_scheduler_loop() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        // max_positional_encoding_tokens = 32
        let executor = FakeExecutor::new(1000, Arc::clone(&dropped)).with_max_context_tokens(32);
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        // prompt=16, max_new=100
        let (too_long, mut too_long_rx) = request(16, 100);
        handle.submit(too_long).expect("submit too_long");
        match too_long_rx.blocking_recv() {
            Some(TokenEvent::Rejected {
                prompt_tokens,
                completion_tokens,
                message,
            }) => {
                assert_eq!(prompt_tokens, 16);
                assert_eq!(completion_tokens, 0);
                assert!(
                    message.contains("context length"),
                    "expected a context-length rejection, got: {message}"
                );
            }
            _ => panic!("over-context request should be rejected"),
        }

        // The loop must keep serving a request that fits the window.
        let (fits, mut fits_rx) = request(16, 1);
        handle.submit(fits).expect("submit fits");
        assert!(
            matches!(
                recv_skipping_scheduled(&mut fits_rx),
                Some(TokenEvent::Token { .. })
            ),
            "later fitting request should emit a token"
        );
        assert!(
            matches!(
                recv_skipping_scheduled(&mut fits_rx),
                Some(TokenEvent::Finished { .. })
            ),
            "later fitting request should finish"
        );
    }

    #[test]
    fn mixed_lora_prefill_requests_run_in_one_batch() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let prefill_batches = Arc::new(Mutex::new(Vec::new()));
        let decode_batches = Arc::new(Mutex::new(Vec::new()));
        let prefill_lora_batches = Arc::new(Mutex::new(Vec::new()));
        let decode_lora_batches = Arc::new(Mutex::new(Vec::new()));
        let mut executor = FakeExecutor::new(4, Arc::clone(&dropped))
            .with_lora_adapters(&["adapter-a", "adapter-b"])
            .with_batch_records(Arc::clone(&prefill_batches), Arc::clone(&decode_batches))
            .with_lora_batch_records(
                Arc::clone(&prefill_lora_batches),
                Arc::clone(&decode_lora_batches),
            );
        let mut rng = StdRng::seed_from_u64(42);
        let mut active = Vec::new();

        let (base, _base_rx) = request_with_lora(16, 1, None);
        let (adapter_a, _adapter_a_rx) = request_with_lora(16, 1, Some("adapter-a"));
        let (adapter_b, _adapter_b_rx) = request_with_lora(16, 1, Some("adapter-b"));
        let pending = vec![
            PendingRequest::from_scheduler_request(RequestId(0), adapter_b),
            PendingRequest::from_scheduler_request(RequestId(1), base),
            PendingRequest::from_scheduler_request(RequestId(2), adapter_a),
        ];

        let artifacts = plan::execute_plan(
            &mut executor,
            &mut active,
            plan::ExecutionPlan::Prefill { pending },
            &mut rng,
        )
        .expect("execute mixed-LoRA prefill");
        let plan::ExecutionArtifacts::Prefill { result, .. } = artifacts else {
            panic!("expected prefill artifacts");
        };

        assert_eq!(
            result
                .requests
                .iter()
                .map(|request| request.request_id)
                .collect::<Vec<_>>(),
            vec![RequestId(0), RequestId(1), RequestId(2)]
        );
        assert_eq!(
            *prefill_batches.lock().unwrap(),
            vec![vec![RequestId(0), RequestId(1), RequestId(2)]],
            "one execution plan should run as one mixed-LoRA prefill batch"
        );
        assert_eq!(
            *prefill_lora_batches.lock().unwrap(),
            vec![vec![
                Some("adapter-b".to_string()),
                None,
                Some("adapter-a".to_string())
            ]],
            "mixed-LoRA batch should preserve per-request adapter metadata"
        );
        assert!(
            decode_batches.lock().unwrap().is_empty(),
            "prefill-only plan should not execute decode batches"
        );
        assert!(
            decode_lora_batches.lock().unwrap().is_empty(),
            "prefill-only plan should not record decode LoRA metadata"
        );
    }

    #[test]
    fn unknown_lora_request_is_rejected_without_blocking_base_request() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (unknown, mut unknown_rx) = request_with_lora(16, 1, Some("missing-adapter"));
        let (base, mut base_rx) = request_with_lora(16, 1, None);
        handle.submit(unknown).expect("submit unknown adapter");
        handle.submit(base).expect("submit base");

        match unknown_rx.blocking_recv() {
            Some(TokenEvent::Rejected {
                message,
                prompt_tokens,
                completion_tokens,
            }) => {
                assert!(message.contains("LoRA adapter is not loaded: missing-adapter"));
                assert_eq!(prompt_tokens, 16);
                assert_eq!(completion_tokens, 0);
            }
            _ => panic!("unknown adapter request should be rejected"),
        }

        assert!(
            matches!(
                recv_skipping_scheduled(&mut base_rx),
                Some(TokenEvent::Token { id: 101, .. })
            ),
            "base request should still run after unknown adapter rejection"
        );
        assert!(
            matches!(
                recv_skipping_scheduled(&mut base_rx),
                Some(TokenEvent::Finished { .. })
            ),
            "base request should finish"
        );
    }

    #[test]
    fn decode_error_drops_request_state_and_scheduler_recovers() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped)).with_decode_failure();
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (will_fail, mut fail_rx) = request(16, 2);
        handle.submit(will_fail).expect("submit will_fail");
        assert!(
            matches!(
                recv_skipping_scheduled(&mut fail_rx),
                Some(TokenEvent::Token { id: 100, .. })
            ),
            "first token should be emitted before decode failure"
        );
        match recv_skipping_scheduled(&mut fail_rx) {
            Some(TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens,
            }) => {
                assert!(message.contains("fake decode KV capacity exhausted"));
                assert_eq!(prompt_tokens, 16);
                assert_eq!(completion_tokens, 1);
            }
            _ => panic!("decode failure should surface as TokenEvent::Error"),
        }
        assert!(
            wait_until(Duration::from_secs(1), || dropped
                .lock()
                .unwrap()
                .contains(&0)),
            "failed request state should be dropped"
        );

        let (after_failure, mut after_rx) = request(16, 1);
        handle.submit(after_failure).expect("submit after_failure");
        assert!(
            matches!(
                recv_skipping_scheduled(&mut after_rx),
                Some(TokenEvent::Token { id: 101, .. })
            ),
            "scheduler should accept new work after a decode error"
        );
        assert!(
            matches!(
                recv_skipping_scheduled(&mut after_rx),
                Some(TokenEvent::Finished { .. })
            ),
            "request after failure should finish"
        );
    }

    #[test]
    fn active_receiver_drop_releases_request_state() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (will_disconnect, mut token_rx) = request(16, 3);
        handle
            .submit(will_disconnect)
            .expect("submit will_disconnect");
        assert!(
            matches!(
                recv_skipping_scheduled(&mut token_rx),
                Some(TokenEvent::Token { id: 100, .. })
            ),
            "prefill should emit the first token"
        );
        drop(token_rx);

        assert!(
            wait_until(Duration::from_secs(1), || dropped
                .lock()
                .unwrap()
                .contains(&0)),
            "dropping an active receiver should release request state"
        );
    }

    #[test]
    fn retiring_multiple_active_requests_tolerates_unsorted_indices() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let mut executor = FakeExecutor::new(8, Arc::clone(&dropped));
        let mut active = Vec::new();

        for request_id in [RequestId(10), RequestId(1), RequestId(7)] {
            let (token_tx, _token_rx) = mpsc::unbounded_channel();
            active.push(ActiveRequestState {
                request_id,
                lora_adapter: None,
                token_tx,
                last_token: 100,
                generated_count: 1,
                max_tokens: 2,
                prompt_len: 16,
                params: SamplingParams::default(),
                logprobs: 0,
            });
            executor
                .ensure_request_tokens(request_id, 16)
                .expect("seed fake request state");
        }

        apply_effects(
            &mut executor,
            &mut active,
            &mut Vec::new(),
            effects::StepEffects {
                scheduled: Vec::new(),
                prompt_echoes: Vec::new(),
                pending: Vec::new(),
                decode: vec![
                    effects::DecodeEffect::EmitAndFinish {
                        request_id: RequestId(1),
                        token: 201,
                        logprob: None,
                        finish_reason: openinfer_core::engine::FinishReason::Length,
                        completion_tokens: 2,
                    },
                    effects::DecodeEffect::EmitAndFinish {
                        request_id: RequestId(10),
                        token: 210,
                        logprob: None,
                        finish_reason: openinfer_core::engine::FinishReason::Length,
                        completion_tokens: 2,
                    },
                    effects::DecodeEffect::EmitAndFinish {
                        request_id: RequestId(7),
                        token: 207,
                        logprob: None,
                        finish_reason: openinfer_core::engine::FinishReason::Length,
                        completion_tokens: 2,
                    },
                ],
            },
        );

        assert!(
            active.is_empty(),
            "all finished requests should retire without index drift"
        );
        let mut dropped = dropped.lock().unwrap().clone();
        dropped.sort_unstable();
        assert_eq!(dropped, vec![1, 7, 10]);
    }

    #[test]
    fn lora_control_reports_unimplemented_when_idle() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped));
        let handle =
            start_with_executor_with_lora_control(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let error = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build runtime")
            .block_on(handle.load_lora_adapter(LoadLoraAdapterRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: "/tmp/adapter-a".into(),
                load_inplace: false,
            }))
            .expect_err("adapter load should be a stub error");

        match error {
            EngineControlError::OperationFailed(message) => {
                assert!(message.contains("not implemented yet"));
                assert!(message.contains("adapter-a"));
            }
            other => panic!("unexpected control error: {other:?}"),
        }
    }

    #[test]
    fn lora_control_unloads_adapter_when_idle() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor =
            FakeExecutor::new(4, Arc::clone(&dropped)).with_lora_adapters(&["adapter-a"]);
        let handle =
            start_with_executor_with_lora_control(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build runtime");
        runtime
            .block_on(handle.unload_lora_adapter(UnloadLoraAdapterRequest {
                lora_name: "adapter-a".to_string(),
                lora_int_id: None,
            }))
            .expect("unload adapter");
        assert_eq!(
            runtime
                .block_on(handle.list_lora_adapters())
                .expect("list adapters"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn lora_control_waits_until_scheduler_idle() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor =
            FakeExecutor::new(4, Arc::clone(&dropped)).with_decode_delay(Duration::from_millis(80));
        let handle =
            start_with_executor_with_lora_control(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

        let (long_running, mut token_rx) = request(16, 3);
        handle.submit(long_running).expect("submit long_running");
        assert!(
            matches!(
                recv_skipping_scheduled(&mut token_rx),
                Some(TokenEvent::Token { id: 100, .. })
            ),
            "first token should be emitted before decode"
        );

        let load_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let load_done_thread = Arc::clone(&load_done);
        let load_handle = handle;
        let load_thread = thread::spawn(move || {
            let result = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("build runtime")
                .block_on(load_handle.load_lora_adapter(LoadLoraAdapterRequest {
                    lora_name: "adapter-a".to_string(),
                    lora_path: "/tmp/adapter-a".into(),
                    load_inplace: false,
                }));
            load_done_thread.store(true, std::sync::atomic::Ordering::SeqCst);
            result
        });

        std::thread::sleep(Duration::from_millis(20));
        assert!(
            !load_done.load(std::sync::atomic::Ordering::SeqCst),
            "load_lora_adapter should wait while generation is active"
        );

        while !matches!(
            recv_skipping_scheduled(&mut token_rx),
            Some(TokenEvent::Finished { .. })
        ) {}

        let error = load_thread
            .join()
            .expect("join load thread")
            .expect_err("adapter load should be a stub error");
        assert!(matches!(error, EngineControlError::OperationFailed(_)));
    }
}
