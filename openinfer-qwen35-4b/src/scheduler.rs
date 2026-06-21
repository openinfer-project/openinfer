//! Scheduler for Qwen3.5: dedicated GPU thread that batches concurrent requests.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - `RecurrentState` alongside `KvState` (linear attention layers)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)

mod plan;

use std::sync::mpsc as std_mpsc;
use std::thread;

use anyhow::Result;
use log::{debug, info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::logprobs::snapshot_requested_logprobs;
use crate::recurrent_state::RecurrentState;
use crate::weights::Qwen35Model;
use openinfer_core::engine::{
    EngineHandle as SchedulerHandle, FinishReason, GenerateRequest as SchedulerRequest, TokenEvent,
    TokenLogprob, TokenSink,
};
use openinfer_core::kv_pool::KvState;
use openinfer_core::sampler::SamplingParams;
use openinfer_core::tensor::HiddenStates;

use self::plan::{
    ActiveKvBudget, ExecutionPlan, PrefillKvBudget, RejectReason, admit_pending_requests,
    compaction_after_retire, max_kv_tokens, plan_prefill_chunks, prefilling_future_pages,
    slot_for_new_request,
};

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded. Recurrent state lives in the
/// `BatchDecodeGraphState` at `graph_slot_idx` — NOT owned here.
struct ActiveRequest35 {
    request_id: Option<String>,
    token_tx: TokenSink,
    kv: KvState,
    /// Index into `BatchDecodeGraphState.slot_states`.
    graph_slot_idx: usize,
    last_token: u32,
    generated_count: usize,
    max_tokens: usize,
    prompt_len: usize,
    params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    logprobs: usize,
}

/// A request whose prompt is being prefilled across multiple scheduler steps.
/// It owns its growing KV and recurrent state until the prompt is exhausted,
/// at which point it is promoted into the decode batch.
struct PrefillingRequest35 {
    req: SchedulerRequest,
    kv: KvState,
    rec: RecurrentState,
    /// Prompt tokens prefilled so far.
    cursor: usize,
    /// Tokens to prefill in the step currently scheduled (set by `take_prefill_chunks`).
    step_chunk: usize,
}

pub const DEFAULT_MAX_PREFILL_TOKENS: usize = 1024;

// ── Entry point ─────────────────────────────────────────────────────────

/// Start the Qwen3.5 scheduler thread with a custom max batch size.
///
/// Lower `max_batch` reduces GPU memory usage (each slot holds a full
/// RecurrentState for all linear attention layers).
pub fn start_with_capacity(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<SchedulerHandle> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    // Static instance cap for the vLLM bridge's max_model_len. Live admission
    // still uses the current page budget inside the scheduler loop.
    let servable = servable_len(
        model.config().max_position_embeddings,
        model.kv_pool().capacity_pages().saturating_sub(1),
        model.kv_pool().layout().page_size,
    );
    let graph_state = model.create_batch_decode_graph_state_with_capacity(max_batch)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (startup_tx, startup_rx) = std_mpsc::channel();

    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35".into())
        .spawn(move || match bind_model_thread(&model) {
            Ok(_guard) => {
                let _ = startup_tx.send(Ok(()));
                scheduler_loop(model, graph_state, submit_rx, seed, max_prefill_tokens);
            }
            Err(err) => {
                let _ = startup_tx.send(Err(err));
            }
        })
        .expect("failed to spawn Qwen3.5 scheduler thread");

    let startup = startup_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("Qwen3.5 scheduler exited during startup"))?;
    if let Err(err) = startup {
        let _ = join_handle.join();
        return Err(err);
    }
    Ok(SchedulerHandle::new_with_join_handle(submit_tx, join_handle).with_servable_len(servable))
}

fn servable_len(max_context: usize, max_pages: usize, page_size: usize) -> u32 {
    max_context
        .min(max_pages.saturating_mul(page_size))
        .try_into()
        .unwrap_or(u32::MAX)
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

fn bind_model_thread(model: &Qwen35Model) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 scheduler thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 scheduler thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    model.tune_decode_gemm_algos()?;
    Ok(CublasThreadGuard)
}

// ── Main loop ───────────────────────────────────────────────────────────

#[allow(clippy::needless_pass_by_value)]
fn scheduler_loop(
    model: Qwen35Model,
    mut graph_state: BatchDecodeGraphState,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    seed: u64,
    prefill_budget: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest35> = Vec::new();
    let mut deferred: Vec<SchedulerRequest> = Vec::new();
    let mut prefilling: Vec<PrefillingRequest35> = Vec::new();
    let max_batch = graph_state.slot_states.len();

    info!("scheduler ready (max_batch={})", max_batch);

    loop {
        // 1. Drain all pending requests (deferred from last iteration + channel)
        let mut pending = std::mem::take(&mut deferred);
        while let Ok(req) = submit_rx.try_recv() {
            pending.push(req);
        }

        // 2. Nothing in flight (no decode, no in-progress prefill) and nothing
        //    pending → block until a request arrives.
        if active.is_empty() && prefilling.is_empty() && pending.is_empty() {
            if let Some(req) = submit_rx.blocking_recv() {
                pending.push(req);
            } else {
                info!("scheduler: all handles dropped, exiting");
                return;
            }
            while let Ok(req) = submit_rx.try_recv() {
                pending.push(req);
            }
        }

        // 3. Admit new prompts. In-flight prefills reserve their promotion slot
        //    and future KV growth, so shrink the slot/page budgets accordingly
        let active_budget: Vec<ActiveKvBudget> = active
            .iter()
            .map(|req| ActiveKvBudget {
                prompt_len: req.prompt_len,
                generated_count: req.generated_count,
                max_tokens: req.max_tokens,
            })
            .collect();
        let page_size = model.kv_pool().layout().page_size;
        let prefilling_budget: Vec<PrefillKvBudget> = prefilling
            .iter()
            .map(|p| PrefillKvBudget {
                current_tokens: p.cursor,
                prompt_len: p.req.prompt_tokens.len(),
                max_tokens: p.req.max_tokens,
            })
            .collect();
        let page_budget = model
            .kv_pool()
            .available_pages()
            .saturating_sub(prefilling_future_pages(&prefilling_budget, page_size));
        let decode_batching_slot = max_batch.saturating_sub(prefilling.len());
        let admission = admit_pending_requests(
            pending,
            &active_budget,
            decode_batching_slot,
            page_size,
            page_budget,
            // KvPool capacity includes the CUDA Graph padding page reserved at
            // construction, so a real request can use at most the remaining pages.
            model.kv_pool().capacity_pages().saturating_sub(1),
            model.config().max_position_embeddings,
            |req| req.prompt_tokens.len(),
            |req| req.max_tokens,
        );
        for (rejected, reason) in &admission.rejected {
            send_rejection(rejected, *reason);
        }

        // 4. Move freshly admitted prompts into the chunked-prefill queue.
        for req in admission.pending {
            debug!(
                "request admitted: request_id={:?} prompt_len={} max_tokens={}",
                req.request_id,
                req.prompt_tokens.len(),
                req.max_tokens
            );
            match RecurrentState::new(model.device_ctx(), model.config()) {
                Ok(rec) => prefilling.push(PrefillingRequest35 {
                    kv: model.alloc_kv(),
                    rec,
                    cursor: 0,
                    step_chunk: 0,
                    req,
                }),
                Err(e) => {
                    warn!("failed to allocate recurrent state for new request: {e}");
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: e.to_string(),
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                }
            }
        }

        deferred = admission.deferred;

        // 5. Take this step's budgeted prefill chunk off the front of the queue,
        //    then dispatch by plan.
        let scheduled = take_prefill_chunks(&mut prefilling, prefill_budget);
        if let Some(plan) = plan::build_next_plan(!active.is_empty(), scheduled) {
            match plan {
                ExecutionPlan::Unified { pending } => unified_step_sched(
                    &model,
                    &mut active,
                    pending,
                    &mut prefilling,
                    &mut graph_state,
                    &mut rng,
                ),
                ExecutionPlan::Prefill { pending } => prefill_batch(
                    &model,
                    &mut active,
                    pending,
                    &mut prefilling,
                    &mut graph_state,
                    &mut rng,
                ),
                ExecutionPlan::Decode => {
                    decode_step(&model, &mut active, &mut graph_state, &mut rng);
                }
            }
        }
    }
}

fn send_rejection(req: &SchedulerRequest, reason: RejectReason) {
    let message = match reason {
        RejectReason::ContextLength { limit } => format!(
            "request exceeds this model's maximum context length of {limit} tokens: requested {} (prompt={} + max_tokens={})",
            req.prompt_tokens.len().saturating_add(req.max_tokens),
            req.prompt_tokens.len(),
            req.max_tokens
        ),
        RejectReason::KvBudget => {
            let max_request_tokens = max_kv_tokens(req.prompt_tokens.len(), req.max_tokens);
            format!(
                "request requires more KV pages than this model instance can provide: prompt_tokens={}, max_request_tokens={max_request_tokens}",
                req.prompt_tokens.len()
            )
        }
    };
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message,
        prompt_tokens: req.prompt_tokens.len(),
        completion_tokens: 0,
    });
}

// ── Batch prefill ───────────────────────────────────────────────────────

fn prefill_batch(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    rng: &mut StdRng,
) {
    let mut chunk = ScheduledChunk::from(scheduled);
    // Scope the borrows of `chunk` to the executor call so the error path can
    // move `chunk` into `fail_chunk`.
    let result = {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(|w| w.as_slice()).collect();
        let mut rec_refs: Vec<&mut RecurrentState> = chunk.recs.iter_mut().collect();
        model.batch_prefill_logits(&window_refs, &mut chunk.kvs, &mut rec_refs)
    };
    let logits = match result {
        Ok(v) => v,
        Err(e) => {
            warn!("batch prefill failed: {e}");
            fail_chunk(chunk, &e.to_string());
            return;
        }
    };

    let (tokens, logprobs_vec) =
        match sample_prefill_logits(model, &chunk.reqs, &logits, graph_state, rng) {
            Ok(v) => v,
            Err(e) => {
                warn!("prefill sampling failed: {e}");
                fail_chunk(chunk, &e.to_string());
                return;
            }
        };

    promote_or_requeue(
        model,
        active,
        prefilling,
        graph_state,
        chunk,
        &tokens,
        &logprobs_vec,
    );
}

fn sample_prefill_logits(
    model: &Qwen35Model,
    pending: &[SchedulerRequest],
    logits: &HiddenStates,
    graph_state: &mut BatchDecodeGraphState,
    rng: &mut StdRng,
) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
    debug_assert_eq!(
        logits.seq_len,
        pending.len(),
        "Qwen3.5 prefill logits rows must preserve pending request order"
    );
    let requested_logprobs: Vec<usize> = pending.iter().map(|r| r.logprobs).collect();
    let cpu_logits = snapshot_requested_logprobs(model.device_ctx(), logits, &requested_logprobs)?;
    let params_refs: Vec<&SamplingParams> = pending.iter().map(|r| &r.params).collect();
    let sample_seed = rand::RngExt::random(rng);
    let tokens = model.select_tokens_from_logits_varied(
        logits,
        &mut graph_state.buffers,
        &params_refs,
        sample_seed,
    )?;

    let logprobs = cpu_logits
        .into_iter()
        .enumerate()
        .map(|(i, logits_opt)| {
            logits_opt.and_then(|logits_f32| {
                openinfer_sample::token_logprob_from_row(
                    &logits_f32,
                    tokens[i],
                    pending[i].logprobs,
                )
            })
        })
        .collect();
    Ok((tokens, logprobs))
}

// ── Unified step (prefill chunk + decode in one forward pass) ──────────────

fn unified_step_sched(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    rng: &mut StdRng,
) {
    let mut chunk = ScheduledChunk::from(scheduled);
    // Scope the borrows of `chunk` / `active` to the executor call so the error
    // and decode-processing paths can use them afterwards.
    let result = {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(|w| w.as_slice()).collect();
        let mut rec_refs: Vec<&mut RecurrentState> = chunk.recs.iter_mut().collect();
        let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let mut decode_kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();
        model.unified_step(
            &window_refs,
            &mut chunk.kvs,
            &mut rec_refs,
            &decode_tokens,
            &mut decode_kv_refs,
            graph_state,
        )
    };
    let output = match result {
        Ok(v) => v,
        Err(e) => {
            warn!("unified step failed: {e}");
            let message = e.to_string();
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            fail_chunk(chunk, &message);
            return;
        }
    };

    // Process decode results FIRST (it may retire requests and free graph slots
    // that promotion then fills densely).
    if output.decoded {
        process_decode_logits(model, active, graph_state, rng);
    }

    let prefill_logits = output
        .prefill_logits
        .as_ref()
        .expect("scheduled prefill chunk must return prefill logits");
    let (tokens, logprobs_vec) =
        match sample_prefill_logits(model, &chunk.reqs, prefill_logits, graph_state, rng) {
            Ok(v) => v,
            Err(e) => {
                warn!("unified prefill sampling failed: {e}");
                fail_chunk(chunk, &e.to_string());
                return;
            }
        };

    promote_or_requeue(
        model,
        active,
        prefilling,
        graph_state,
        chunk,
        &tokens,
        &logprobs_vec,
    );
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

fn decode_step(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    rng: &mut StdRng,
) {
    let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
    let mut kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

    if let Err(e) = model.batch_decode_graph(&token_ids, &mut kv_refs, graph_state) {
        warn!("batch_decode_graph error: {e}");
        let message = e.to_string();
        for req in active.drain(..) {
            let _ = req.token_tx.send(TokenEvent::Error {
                message: message.clone(),
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
        }
        return;
    }

    // Snapshot logits to CPU BEFORE sampling (sampling may modify bufs.logits)
    let requested_logprobs: Vec<usize> = active.iter().map(|r| r.logprobs).collect();
    let cpu_logits = match snapshot_requested_logprobs(
        model.device_ctx(),
        &graph_state.buffers.logits,
        &requested_logprobs,
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!("logprobs snapshot error: {e}");
            let message = e.to_string();
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            return;
        }
    };

    let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
    let sample_seed = rand::RngExt::random(rng);
    let tokens =
        match model.select_tokens_batch_varied(&mut graph_state.buffers, &params_refs, sample_seed)
        {
            Ok(t) => t,
            Err(e) => {
                warn!("sampling error: {e}");
                let message = e.to_string();
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                return;
            }
        };

    let logprobs_vec: Vec<Option<TokenLogprob>> = cpu_logits
        .into_iter()
        .enumerate()
        .map(|(i, logits_opt)| {
            logits_opt.and_then(|logits_f32| {
                openinfer_sample::token_logprob_from_row(&logits_f32, tokens[i], active[i].logprobs)
            })
        })
        .collect();

    dispatch_decode_tokens(model, active, &tokens, &logprobs_vec, graph_state);
}

/// Process decode logits from unified step: sample, extract logprobs, dispatch.
fn process_decode_logits(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    rng: &mut StdRng,
) {
    let requested_logprobs: Vec<usize> = active.iter().map(|r| r.logprobs).collect();
    let cpu_logits = match snapshot_requested_logprobs(
        model.device_ctx(),
        &graph_state.buffers.logits,
        &requested_logprobs,
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!("decode logprobs snapshot error: {e}");
            let message = e.to_string();
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            return;
        }
    };

    let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
    let sample_seed = rand::RngExt::random(rng);
    let tokens =
        match model.select_tokens_batch_varied(&mut graph_state.buffers, &params_refs, sample_seed)
        {
            Ok(t) => t,
            Err(e) => {
                warn!("decode sampling error: {e}");
                let message = e.to_string();
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                return;
            }
        };

    let logprobs_vec: Vec<Option<TokenLogprob>> = cpu_logits
        .into_iter()
        .enumerate()
        .map(|(i, logits_opt)| {
            logits_opt.and_then(|logits_f32| {
                openinfer_sample::token_logprob_from_row(&logits_f32, tokens[i], active[i].logprobs)
            })
        })
        .collect();

    dispatch_decode_tokens(model, active, &tokens, &logprobs_vec, graph_state);
}

/// Dispatch sampled decode tokens: send events, check EOS/limits, retire finished.
///
/// `tokens` and `logprobs` are indexed by original position in `active`.
/// Retirements collected first, then compacted in reverse order.
fn dispatch_decode_tokens(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
    graph_state: &mut BatchDecodeGraphState,
) {
    let n = active.len();
    let mut to_retire = Vec::new();

    for i in 0..n {
        let token = tokens[i];
        let logprob = logprobs[i].clone();
        let req = &mut active[i];
        req.generated_count += 1;

        let is_eos = !req.params.ignore_eos && model.is_stop_token(token);
        let at_limit = req.generated_count >= req.max_tokens;

        if is_eos {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                req.prompt_len,
                req.generated_count,
                FinishReason::Stop
            );
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            to_retire.push(i);
        } else if at_limit {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                req.prompt_len,
                req.generated_count,
                FinishReason::Length
            );
            let _ = req.token_tx.send(TokenEvent::Token { id: token, logprob });
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            to_retire.push(i);
        } else if req
            .token_tx
            .send(TokenEvent::Token { id: token, logprob })
            .is_err()
        {
            debug!(
                "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                req.request_id, req.generated_count
            );
            to_retire.push(i);
        } else {
            req.last_token = token;
        }
    }

    // Remove in reverse order so compact_slot indices stay valid
    for &i in to_retire.iter().rev() {
        compact_slot(model, active, graph_state, i);
    }
}

/// Remove request at `idx` via swap_remove and compact graph slots.
///
/// After swap_remove, the element that was at `active.len()-1` (before remove)
/// now sits at `idx`. Its graph slot must be copied into the vacated slot so
/// that slots 0..active.len() remain dense.
fn compact_slot(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    idx: usize,
) {
    let compaction = compaction_after_retire(active.len(), idx);
    active.swap_remove(idx);

    if let Some(compaction) = compaction {
        // The element that was at `last` is now at `idx`.
        // Copy its recurrent state from slot `last` to slot `idx`.
        let src_slot = active[idx].graph_slot_idx;
        debug_assert_eq!(src_slot, compaction.moved_from);

        // D2D copy: graph_state.slot_states[src] -> graph_state.slot_states[dst]
        // We can't borrow two slots mutably at once, so use raw index copy.
        let ctx = model.device_ctx();
        let src = &graph_state.slot_states[compaction.moved_from];
        // Copy layer by layer using the public fields
        for layer_idx in 0..src.layers.len() {
            let (src_part, dst_part) = if compaction.moved_to < compaction.moved_from {
                let (left, right) = graph_state.slot_states.split_at_mut(compaction.moved_from);
                (
                    &right[0].layers[layer_idx],
                    &mut left[compaction.moved_to].layers[layer_idx],
                )
            } else {
                unreachable!("idx < active.len() <= last");
            };

            ctx.stream
                .memcpy_dtod(&src_part.state, &mut dst_part.state)
                .expect("compact slot state copy failed");
            ctx.stream
                .memcpy_dtod(&src_part.conv_state.data, &mut dst_part.conv_state.data)
                .expect("compact slot conv_state copy failed");
        }
        graph_state.slot_states[compaction.moved_to].seq_len =
            graph_state.slot_states[compaction.moved_from].seq_len;

        active[compaction.moved_to].graph_slot_idx = compaction.moved_to;
    }
}

// ── Chunked-prefill helpers ────────────────────────────────────────────────

/// Step's scheduled prefill set
struct ScheduledChunk {
    reqs: Vec<SchedulerRequest>,
    kvs: Vec<KvState>,
    recs: Vec<RecurrentState>,
    /// Prompt cursor after this step's chunk
    ends: Vec<usize>,
    /// This step's chunked token slice per request
    windows: Vec<Vec<u32>>,
}

impl From<Vec<PrefillingRequest35>> for ScheduledChunk {
    fn from(scheduled: Vec<PrefillingRequest35>) -> Self {
        let n = scheduled.len();
        let mut chunk = ScheduledChunk {
            reqs: Vec::with_capacity(n),
            kvs: Vec::with_capacity(n),
            recs: Vec::with_capacity(n),
            ends: Vec::with_capacity(n),
            windows: Vec::with_capacity(n),
        };
        for p in scheduled {
            let end = p.cursor + p.step_chunk;
            chunk
                .windows
                .push(p.req.prompt_tokens[p.cursor..end].to_vec());
            chunk.ends.push(end);
            chunk.reqs.push(p.req);
            chunk.kvs.push(p.kv);
            chunk.recs.push(p.rec);
        }
        chunk
    }
}

/// Pull this step's prefill set off the FRONT of `prefilling`, capping the
/// step's total forwarded prompt tokens at `prefill_budget`.
fn take_prefill_chunks(
    prefilling: &mut Vec<PrefillingRequest35>,
    prefill_budget: usize,
) -> Vec<PrefillingRequest35> {
    let remaining: Vec<usize> = prefilling
        .iter()
        .map(|p| p.req.prompt_tokens.len() - p.cursor)
        .collect();
    let chunks = plan_prefill_chunks(&remaining, prefill_budget);
    let mut scheduled: Vec<PrefillingRequest35> = prefilling.drain(0..chunks.len()).collect();
    for (p, chunk) in scheduled.iter_mut().zip(&chunks) {
        p.step_chunk = *chunk;
    }
    scheduled
}

/// Report a forward/sampling failure to every request in the failed chunk.
fn fail_chunk(chunk: ScheduledChunk, message: &str) {
    for req in chunk.reqs {
        let _ = req.token_tx.send(TokenEvent::Error {
            message: message.to_string(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
    }
}

/// For each request in the just-prefilled chunk: if its prompt is now exhausted,
/// sample its first token, emit events, and move it into the decode batch;
/// otherwise re-queue it (with an advanced cursor) at the FRONT of `prefilling`.
/// `tokens` / `logprobs` are indexed by request order in `chunk`.
fn promote_or_requeue(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    chunk: ScheduledChunk,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
) {
    let ScheduledChunk {
        reqs,
        kvs,
        recs,
        ends,
        ..
    } = chunk;
    let mut still_prefilling: Vec<PrefillingRequest35> = Vec::new();

    for (i, (((req, kv), rec), end)) in reqs.into_iter().zip(kvs).zip(recs).zip(ends).enumerate() {
        // Not finished: re-queue with the advanced cursor
        if end < req.prompt_tokens.len() {
            still_prefilling.push(PrefillingRequest35 {
                req,
                kv,
                rec,
                cursor: end,
                step_chunk: 0,
            });
            continue;
        }

        let prompt_len = req.prompt_tokens.len();
        let first_token = tokens[i];
        let logprob = logprobs[i].clone();

        if req.echo {
            let echo_logprobs = vec![None; req.prompt_tokens.len()];
            let _ = req.token_tx.send(TokenEvent::PromptTokens {
                ids: req.prompt_tokens.clone(),
                logprobs: echo_logprobs,
            });
        }

        if !req.params.ignore_eos && model.is_stop_token(first_token) {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                prompt_len,
                0,
                FinishReason::Stop
            );
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            continue;
        }

        if req
            .token_tx
            .send(TokenEvent::Token {
                id: first_token,
                logprob,
            })
            .is_err()
        {
            debug!(
                "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                req.request_id, 0
            );
            continue;
        }

        if req.max_tokens <= 1 {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                prompt_len,
                1,
                FinishReason::Length
            );
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
            continue;
        }

        // Assign a graph slot and copy recurrent state into it.
        let slot_idx = slot_for_new_request(active.len(), graph_state.slot_states.len())
            .expect("admission must reserve a graph slot");
        graph_state
            .copy_state_to_slot(model.device_ctx(), &rec, slot_idx)
            .expect("copy recurrent state to slot failed");
        active.push(ActiveRequest35 {
            request_id: req.request_id,
            token_tx: req.token_tx,
            kv,
            graph_slot_idx: slot_idx,
            last_token: first_token,
            generated_count: 1,
            max_tokens: req.max_tokens,
            prompt_len,
            params: req.params,
            logprobs: req.logprobs,
        });
    }

    prefilling.splice(0..0, still_prefilling);
}

#[cfg(test)]
mod tests;
