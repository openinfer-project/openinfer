//! Scheduler: dedicated GPU thread that batches concurrent requests.
//!
//! Frontend handlers tokenize prompts and submit `GenerateRequest` via channel.
//! The scheduler batch-prefills all pending requests in one forward pass, then
//! batch-decodes all active requests. Per-request tokens flow back through
//! individual channels.

mod effects;
mod kv_events;
mod plan;
mod resolve;

use std::collections::{HashSet, VecDeque};
use std::thread;

use anyhow::Result;
use log::{debug, info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::{mpsc, watch};

use crate::executor::{ModelExecutor, Qwen3Executor, RequestId};
use crate::weights::Qwen3MemoryOptions;
use crate::{Qwen3LoraOptions, Qwen3OffloadOptions};
use openinfer_core::engine::{
    EngineCommand, EngineControlRequest, EngineHandle, GenerateRequest, KvCapacity, LoadSnapshot,
    TokenEvent, TokenSink,
};
use openinfer_core::sampler::SamplingParams;

use self::effects::apply_effects;
use self::kv_events::KvEventProducer;
use self::plan::{
    ExecutionArtifacts, ExecutionPlan, build_next_plan, execute_plan, should_speculative_decode,
};
use self::resolve::resolve_step;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded.
pub(super) struct ActiveRequestState {
    pub(super) request_id: RequestId,
    pub(super) lora_adapter: Option<String>,
    pub(super) token_tx: TokenSink,
    pub(super) last_token: u32,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
    pub(super) prompt_len: usize,
    pub(super) params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    pub(super) logprobs: usize,
}

#[derive(Clone)]
pub(super) struct PendingRequest {
    pub(super) request_id: RequestId,
    pub(super) lora_adapter: Option<String>,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) params: SamplingParams,
    pub(super) max_tokens: usize,
    pub(super) token_tx: TokenSink,
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
/// their whole remainder fits the profiled prefill bound.
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
            if remaining > budget {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_qwen3(
    model_path: &str,
    enable_cuda_graph: bool,
    device_ordinals: &[usize],
    seed: u64,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
    decode_overlap: crate::DecodeOverlap,
    dflash_draft_model_path: Option<&str>,
    enable_kv_events: bool,
) -> Result<EngineHandle> {
    let mut executor = Qwen3Executor::from_runtime_with_lora_options(
        model_path,
        enable_cuda_graph,
        device_ordinals,
        Qwen3LoraOptions::default(),
        offload_options,
        max_prefill_tokens,
        dflash_draft_model_path,
        memory_options,
        enable_kv_events,
    )?;
    executor.set_no_prefix_cache(no_prefix_cache);
    executor.enable_decode_overlap(decode_overlap)?;
    // Speculative decoding loads its draft model after the target is up (the
    // draft is built against the target's embeddings/head) and forces the
    // prefix cache off, so it must follow set_no_prefix_cache. Its GPU footprint
    // was already reserved during profiling from the draft path passed above.
    if let Some(draft_path) = dflash_draft_model_path {
        executor.load_dflash_draft_model(draft_path)?;
    }

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
    memory_options: Qwen3MemoryOptions,
    decode_overlap: crate::DecodeOverlap,
) -> Result<EngineHandle> {
    let mut executor = Qwen3Executor::from_runtime_with_lora_options(
        model_path,
        enable_cuda_graph,
        device_ordinals,
        lora_options,
        offload_options,
        max_prefill_tokens,
        None,
        memory_options,
        // LoRA serving never emits KV events: the router-facing cache feed is
        // base-model single-rank only.
        false,
    )?;
    executor.set_no_prefix_cache(no_prefix_cache);
    executor.enable_decode_overlap(decode_overlap)?;
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
    mut executor: E,
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
    let kv_total = executor.available_blocks() as u64;
    let kv_capacity = KvCapacity {
        total_blocks: kv_total as usize,
        block_size: executor.block_size(),
    };
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (load_tx, load_rx) = watch::channel(LoadSnapshot {
        kv_total_blocks: kv_total,
        ..LoadSnapshot::default()
    });

    // Opt-in KV block-event feed: `Some` only when the executor was built with
    // events on (single-GPU, no LoRA). The producer runs on the scheduler
    // thread; the neutral receiver is handed to the engine handle.
    let (producer, kv_events_rx) = match executor.take_kv_event_receiver() {
        Some(removes) => {
            let (event_tx, event_rx) = mpsc::unbounded_channel();
            (
                Some(KvEventProducer::new(event_tx, removes)),
                Some(event_rx),
            )
        }
        None => (None, None),
    };

    let join_handle = thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop(
                executor,
                submit_rx,
                seed,
                max_prefill_tokens,
                kv_total,
                &load_tx,
                producer,
            );
        })
        .expect("failed to spawn scheduler thread");

    let handle = EngineHandle::new_with_join_handle(submit_tx, join_handle)
        .with_servable_len(servable)
        .with_kv_capacity(kv_capacity)
        .with_load_watch(load_rx);
    match kv_events_rx {
        Some(rx) => handle.with_kv_events(rx),
        None => handle,
    }
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
    let kv_total = executor.available_blocks() as u64;
    let kv_capacity = KvCapacity {
        total_blocks: kv_total as usize,
        block_size: executor.block_size(),
    };
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (load_tx, load_rx) = watch::channel(LoadSnapshot {
        kv_total_blocks: kv_total,
        ..LoadSnapshot::default()
    });

    let join_handle = thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop_with_lora_control(
                executor,
                command_rx,
                seed,
                max_prefill_tokens,
                kv_total,
                &load_tx,
            );
        })
        .expect("failed to spawn scheduler thread");

    EngineHandle::new_with_command_channel_and_join_handle(command_tx, join_handle)
        .with_servable_len(servable)
        .with_kv_capacity(kv_capacity)
        .with_load_watch(load_rx)
}

// ── KV-offload prefetch admission helpers ────────────────────────────────

/// Move requests whose async CPU-tier prefetch just settled from `loading`
/// back to the front of `deferred` — their KV is hot, so admit them first.
fn reclaim_ready_prefetch<E: ModelExecutor>(
    executor: &mut E,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
    // Free blocks already promised to admitted requests; a remote fetch that
    // resolves during this sweep must not reserve into them.
    reserve_floor: usize,
) {
    promote_ready(
        executor.drain_ready_prefetch(reserve_floor),
        deferred,
        loading,
    );
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
    reserve_floor: usize,
) {
    promote_ready(
        executor.wait_ready_prefetch(reserve_floor),
        deferred,
        loading,
    );
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

/// Republish live KV occupancy to the load-watch feed. Called once at the top of
/// every loop iteration (before this step admits/allocates), so it reports the
/// resident occupancy *between* steps — the steady-state load a router wants,
/// not a transient in-step peak. Top-of-loop placement guarantees exactly one
/// publish per iteration regardless of which `continue` the step takes, and the
/// post-completion free shows up at the next iteration's top before the loop
/// parks idle. `watch` coalesces (a consumer wakes at most once per step and
/// reads the latest); `send_replace` ignores a dropped receiver, so the
/// scheduler runs whether or not anyone is watching.
/// `num_waiting_reqs` folds every not-yet-running queue (KV-deferred,
/// prefetch-loading, post-control) into one number; the vLLM frontend exports
/// it as `num_requests_waiting` and attributes all of it to
/// `reason="capacity"` — the `reason="deferred"` gauge is driven by a separate
/// skipped-requests counter this scheduler does not report, so it reads 0 by
/// design, not because the local `deferred` queue is invisible.
fn publish_load<E: ModelExecutor>(
    load_tx: &watch::Sender<LoadSnapshot>,
    kv_total: u64,
    executor: &E,
    num_running_reqs: u64,
    num_waiting_reqs: u64,
) {
    load_tx.send_replace(LoadSnapshot {
        kv_used_blocks: kv_total.saturating_sub(executor.available_blocks() as u64),
        kv_total_blocks: kv_total,
        num_running_reqs,
        num_waiting_reqs,
    });
}

fn scheduler_loop<E>(
    mut executor: E,
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    seed: u64,
    max_prefill_tokens: usize,
    kv_total: u64,
    load_tx: &watch::Sender<LoadSnapshot>,
    mut kv_producer: Option<KvEventProducer>,
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
    // Decode-overlap async prefill: pending requests whose prefill is in-flight
    // on the prefill overlap stream. `None` when no async prefill is running.
    let mut inflight_prefill_pending: Option<Vec<PendingRequest>> = None;

    info!("Scheduler ready");

    loop {
        publish_load(
            load_tx,
            kv_total,
            &executor,
            (active.len()
                + prefilling.len()
                + inflight_prefill_pending.as_ref().map_or(0, Vec::len)) as u64,
            (deferred.len() + loading.len()) as u64,
        );
        // Flush the prior step's cache changes to a router (no-op unless the
        // event feed is on). Top-of-loop, like `publish_load`: one pass per
        // iteration regardless of which branch the step takes, at the cost of a
        // one-iteration announcement lag the router tolerates. Stores first so a
        // block evicted the same step it registered is announced before removed.
        if let Some(producer) = kv_producer.as_mut() {
            producer.emit_stores(executor.take_kv_store_events());
            producer.drain_removes();
        }

        // 0. Poll in-flight async prefill (decode-overlap mode).
        if inflight_prefill_pending.is_some() {
            if let Some(prefill_result) = executor.poll_async_prefill() {
                let pending = inflight_prefill_pending.take().unwrap();
                info!(
                    "decode-overlap: async prefill completed ({} reqs)",
                    pending.len()
                );
                let scheduled_at_unix_s = openinfer_core::engine::unix_now_s();
                let artifacts = ExecutionArtifacts::Prefill {
                    pending,
                    result: prefill_result,
                    scheduled_at_unix_s,
                };
                let effects = resolve_step(&executor, &active, artifacts);
                apply_effects(&mut executor, &mut active, &mut prefilling, effects);
            }
        }

        // 1. Drain all incoming requests into deferred.
        while let Ok(req) = submit_rx.try_recv() {
            deferred.push(PendingRequest::from_scheduler_request(
                RequestId(next_request_id),
                req,
            ));
            next_request_id += 1;
        }

        // 2. Reclaim settled prefetches, then offer fresh requests to prefetch.
        let reserve_floor = admitted_future_blocks(&executor, &active, &prefilling);
        reclaim_ready_prefetch(&mut executor, &mut deferred, &mut loading, reserve_floor);
        offer_prefetch(&mut executor, &mut deferred, &mut loading, reserve_floor);

        // 3. Nothing active and nothing admittable → block. Prefer blocking on
        // an in-flight load (so its request prefills next) over a new submit;
        // only truly idle (no loads either) do we block on the channel.
        if active.is_empty() && deferred.is_empty() && prefilling.is_empty() {
            if !loading.is_empty() {
                let reserve_floor = admitted_future_blocks(&executor, &active, &prefilling);
                block_on_loading(&mut executor, &mut deferred, &mut loading, reserve_floor);
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
            max_prefill_tokens,
            |id| executor.prefetched_blocks(id),
        );
        for (rejected, reason) in &admission.rejected {
            send_rejection(rejected, *reason);
            release_rejected(&mut executor, rejected);
        }
        prefilling.extend(admission.pending);
        deferred = admission.deferred;
        // If there's an in-flight async prefill, skip taking new prefill chunks
        // (only do decode until the async prefill completes).
        let pending = if inflight_prefill_pending.is_some() {
            Vec::new()
        } else {
            take_prefill_chunks(&mut prefilling, max_prefill_tokens)
        };

        let Some(plan) = runtime_plan(&executor, &active, pending) else {
            continue;
        };

        // Decode-overlap path: when a Unified plan appears and overlap streams
        // are active (and no async prefill is already in-flight), execute_unified
        // internally uses SplitConcurrent which only syncs decode and defers
        // the prefill sync. This lets the scheduler advance decode immediately.
        // The prefill result is polled at the top of the next iteration.
        if executor.has_decode_overlap()
            && inflight_prefill_pending.is_none()
            && matches!(plan, ExecutionPlan::Unified { .. })
        {
            if let ExecutionPlan::Unified { pending } = plan {
                let prefill_tokens: usize = pending.iter().map(|r| r.step_chunk).sum();
                info!(
                    "decode-overlap: unified step with async prefill ({} reqs, ~{} tokens)",
                    pending.len(),
                    prefill_tokens
                );

                // Save pending for later poll resolution.
                let pending_for_poll = pending.clone();

                // execute_plan(Unified) will internally SplitConcurrent:
                // - sync decode immediately
                // - defer prefill sync
                // It returns Unified artifacts with empty prefill_requests.
                let unified_plan = ExecutionPlan::Unified { pending };
                let failure_targets = failure_targets_for(&active, &unified_plan);
                let artifacts =
                    match execute_plan(&mut executor, &mut active, unified_plan, &mut rng) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Execution step failed: {e}");
                            fail_touched_requests(
                                &mut executor,
                                &mut active,
                                failure_targets,
                                &e.to_string(),
                            );
                            continue;
                        }
                    };

                // Only apply decode effects from the unified result.
                let effects = resolve_step(&executor, &active, artifacts);
                apply_effects(&mut executor, &mut active, &mut prefilling, effects);

                // Track the pending prefill for next-iteration polling.
                inflight_prefill_pending = Some(pending_for_poll);
                continue;
            }
        }

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
    kv_total: u64,
    load_tx: &watch::Sender<LoadSnapshot>,
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
        publish_load(
            load_tx,
            kv_total,
            &executor,
            (active.len() + prefilling.len()) as u64,
            (deferred.len() + loading.len() + post_control_deferred.len()) as u64,
        );

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
        let reserve_floor = admitted_future_blocks(&executor, &active, &prefilling);
        reclaim_ready_prefetch(&mut executor, &mut deferred, &mut loading, reserve_floor);
        if pending_control.is_empty() {
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
                let reserve_floor = admitted_future_blocks(&executor, &active, &prefilling);
                block_on_loading(&mut executor, &mut deferred, &mut loading, reserve_floor);
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
                // Back to the top like the non-LoRA loop: republish the load
                // snapshot (the new request must show as waiting before its
                // prefill runs) and let steps 1–2 drain the burst and any
                // idle-control work instead of repeating them here.
                continue;
            }
            info!("Scheduler: all handles dropped, exiting");
            return;
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
            max_prefill_tokens,
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
                let reserve_floor = admitted_future_blocks(&executor, &active, &prefilling);
                block_on_loading(&mut executor, &mut deferred, &mut loading, reserve_floor);
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

        let Some(plan) = runtime_plan(&executor, &active, pending) else {
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
    token_tx: TokenSink,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Why a request was rejected at admission, so the client gets an accurate error.
#[derive(Clone, Copy)]
enum RejectReason {
    /// Worst-case length exceeds the model's position-encoding window.
    ContextLength { limit: usize },
    /// Echo needs all-position logits in one forward, so it must fit the
    /// profiled prefill bound.
    EchoPrefillTokens { limit: usize },
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

fn echo_exceeds_prefill_bound(req: &PendingRequest, max_prefill_tokens: usize) -> bool {
    req.echo && req.prompt_tokens.len() > max_prefill_tokens
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
/// Echo requests need all-position logits in one forward and are rejected when
/// their prompt exceeds this bound.
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
    max_prefill_tokens: usize,
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

        if echo_exceeds_prefill_bound(&req, max_prefill_tokens) {
            rejected.push((
                req,
                RejectReason::EchoPrefillTokens {
                    limit: max_prefill_tokens,
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
        RejectReason::EchoPrefillTokens { limit } => format!(
            "echo request prompt exceeds the profiled prefill limit of {} tokens: prompt_tokens={}",
            limit,
            req.prompt_tokens.len()
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

/// Choose the step plan, preferring a speculative-decode step when the whole
/// active batch is draft-ready. Prefill of new arrivals still takes priority —
/// a speculative step only runs when there is nothing to prefill, so the two
/// never mix in one step.
fn runtime_plan(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    pending: Vec<PendingRequest>,
) -> Option<ExecutionPlan> {
    if should_speculative_decode(executor, active) {
        if pending.is_empty() {
            Some(ExecutionPlan::SpeculativeDecode)
        } else {
            Some(ExecutionPlan::Prefill { pending })
        }
    } else {
        build_next_plan(!active.is_empty(), pending, executor.speculative_enabled())
    }
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
        self::plan::ExecutionPlan::Decode | self::plan::ExecutionPlan::SpeculativeDecode => {
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
mod tests;
