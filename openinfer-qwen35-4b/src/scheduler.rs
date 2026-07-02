//! Scheduler for Qwen3.5: dedicated GPU thread that batches concurrent requests.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - `RecurrentState` alongside `KvState` (linear attention layers)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)

mod plan;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::thread;

use anyhow::Result;
use log::{debug, info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::dflash::{DFlashBatchScratch, DFlashDraftModel, DFlashRequestState};
use crate::logprobs::snapshot_requested_logprobs;
use crate::recurrent_state::RecurrentState;
use crate::speculative::{VerifiedToken, accept_greedy};
use crate::verify_buffers::VerifyBuffers35;
use crate::weights::Qwen35Model;
use openinfer_core::engine::{
    EngineHandle as SchedulerHandle, FinishReason, GenerateRequest as SchedulerRequest, KvCapacity,
    TokenEvent, TokenLogprob, TokenSink,
};
use openinfer_core::kv_pool::KvState;
use openinfer_core::sampler::SamplingParams;
use openinfer_core::tensor::HiddenStates;

use self::plan::{
    ActiveKvBudget, ExecutionPlan, PrefillKvBudget, RejectReason, admit_pending_requests,
    compaction_after_retire, max_kv_tokens, plan_prefill_chunks, prefilling_future_pages,
    slot_for_new_request,
};

const DFLASH_MIN_STATS_DRAFT_TOKENS: usize = 4;
const DFLASH_MIN_ACCEPT_RATE: f64 = 0.15;
const DFLASH_PROBE_DRAFT_TOKENS: usize = 4;
const DFLASH_FULL_BLOCK_ACCEPT_RATE: f64 = 0.60;
const DFLASH_MIN_CONTEXT_TOKENS: usize = 16;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded. Recurrent state lives in the
/// `BatchDecodeGraphState` at `graph_slot_idx` — NOT owned here.
struct ActiveRequest35 {
    local_id: usize,
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

struct DFlashSchedulerState {
    model: DFlashDraftModel,
    requests: HashMap<usize, DFlashRequestState>,
    stats: HashMap<usize, DFlashRequestStats>,
    scratch: DFlashBatchScratch,
    sample: openinfer_sample::SampleScratch,
    verify_bufs: VerifyBuffers35,
    backup_states: Vec<RecurrentState>,
    verify_scratch_states: Vec<RecurrentState>,
    commit_scratch_states: Vec<RecurrentState>,
    verified_draft_tokens: usize,
    accepted_draft_tokens: usize,
    commit_stats: DFlashCommitStats,
}

#[derive(Clone, Copy, Default)]
struct DFlashRequestStats {
    verified_draft_tokens: usize,
    accepted_draft_tokens: usize,
}

#[derive(Clone, Copy, Default)]
struct DFlashCommitStats {
    full_span_rounds: usize,
    same_partial_rounds: usize,
    batched_replay_rounds: usize,
    batched_replay_rows: usize,
    prefix_state_rounds: usize,
    prefix_state_rows: usize,
}

impl DFlashSchedulerState {
    fn new(target: &Qwen35Model, draft_path: &str, max_batch: usize) -> Result<Self> {
        let model =
            DFlashDraftModel::from_safetensors_for_target(target.device_ctx(), draft_path, target)?;
        if std::env::var_os("OPENINFER_QWEN35_DFLASH_TUNE_GEMM").is_some() {
            model.tune_gemm_algos(target)?;
        }
        let scratch = model.new_batch_scratch(target.device_ctx(), max_batch)?;
        let sample = openinfer_sample::SampleScratch::new(
            target.device_ctx(),
            target.config().vocab_size,
            max_batch * model.verify_span(),
        )?;
        let verify_bufs = VerifyBuffers35::new(
            target.device_ctx(),
            target.config(),
            max_batch,
            model.verify_span(),
            model.target_layer_ids().len(),
            target.kv_pool().capacity_pages(),
        )?;
        Ok(Self {
            model,
            requests: HashMap::new(),
            stats: HashMap::new(),
            scratch,
            sample,
            verify_bufs,
            backup_states: Vec::new(),
            verify_scratch_states: Vec::new(),
            commit_scratch_states: Vec::new(),
            verified_draft_tokens: 0,
            accepted_draft_tokens: 0,
            commit_stats: DFlashCommitStats::default(),
        })
    }

    fn capture_layer_ids(&self) -> &[usize] {
        self.model.target_layer_ids()
    }

    fn usable_context_tokens(&self, target_max_position_embeddings: usize) -> usize {
        target_max_position_embeddings.min(
            self.model
                .max_position_embeddings()
                .saturating_sub(self.model.block_size()),
        )
    }

    fn drop_request(&mut self, local_id: usize) {
        self.requests.remove(&local_id);
        self.stats.remove(&local_id);
    }

    fn pending_context_len(&self, local_id: usize) -> Option<usize> {
        self.requests
            .get(&local_id)
            .and_then(DFlashRequestState::pending_context_len)
    }

    fn ready_for_draft(&self, local_id: usize) -> bool {
        self.pending_context_len(local_id)
            .is_some_and(|len| len >= DFLASH_MIN_CONTEXT_TOKENS)
    }

    fn ensure_state_scratch(
        &mut self,
        ctx: &openinfer_core::tensor::DeviceContext,
        config: &crate::config::Config35,
        batch: usize,
    ) -> Result<()> {
        while self.backup_states.len() < batch {
            self.backup_states.push(RecurrentState::new(ctx, config)?);
        }
        while self.verify_scratch_states.len() < batch {
            self.verify_scratch_states
                .push(RecurrentState::new(ctx, config)?);
        }
        while self.commit_scratch_states.len() < batch {
            self.commit_scratch_states
                .push(RecurrentState::new(ctx, config)?);
        }
        Ok(())
    }
}

/// A request whose prompt is being prefilled across multiple scheduler steps.
/// It owns its growing KV and recurrent state until the prompt is exhausted,
/// at which point it is promoted into the decode batch.
struct PrefillingRequest35 {
    local_id: usize,
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
    start_with_capacity_and_dflash(model, seed, max_batch, max_prefill_tokens, None)
}

pub(crate) fn start_with_capacity_and_dflash(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
    max_prefill_tokens: usize,
    dflash_draft_model_path: Option<PathBuf>,
) -> Result<SchedulerHandle> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    // Static instance cap for the vLLM bridge's max_model_len. Live admission
    // still uses the current page budget inside the scheduler loop.
    let total_blocks = model.kv_pool().capacity_pages().saturating_sub(1);
    let block_size = model.kv_pool().layout().page_size;
    let servable = servable_len(
        model.config().max_position_embeddings,
        total_blocks,
        block_size,
    );
    let graph_state = model.create_batch_decode_graph_state_with_capacity(max_batch)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (startup_tx, startup_rx) = std_mpsc::channel();

    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35".into())
        .spawn(move || match bind_model_thread(&model) {
            Ok(_guard) => {
                let dflash = match dflash_draft_model_path
                    .as_ref()
                    .map(|path| {
                        path.to_str()
                            .ok_or_else(|| {
                                anyhow::anyhow!("DFlash draft model path must be valid UTF-8")
                            })
                            .and_then(|path| DFlashSchedulerState::new(&model, path, max_batch))
                    })
                    .transpose()
                {
                    Ok(dflash) => dflash,
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                        return;
                    }
                };
                let _ = startup_tx.send(Ok(()));
                scheduler_loop(
                    model,
                    graph_state,
                    submit_rx,
                    seed,
                    max_prefill_tokens,
                    dflash,
                );
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
    Ok(
        SchedulerHandle::new_with_join_handle(submit_tx, join_handle)
            .with_servable_len(servable)
            .with_kv_capacity(KvCapacity {
                total_blocks,
                block_size,
            }),
    )
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
    mut dflash: Option<DFlashSchedulerState>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest35> = Vec::new();
    let mut deferred: Vec<SchedulerRequest> = Vec::new();
    let mut prefilling: Vec<PrefillingRequest35> = Vec::new();
    let max_batch = graph_state.slot_states.len();
    let mut next_local_id = 0usize;

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
            dflash
                .as_ref()
                .map_or(model.config().max_position_embeddings, |state| {
                    state.usable_context_tokens(model.config().max_position_embeddings)
                }),
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
                    local_id: {
                        let id = next_local_id;
                        next_local_id = next_local_id
                            .checked_add(1)
                            .expect("Qwen3.5 scheduler local request id exhausted");
                        id
                    },
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
        let may_capture_dflash_prefill = true;
        let force_prefill_for_dflash = dflash.is_some()
            && may_capture_dflash_prefill
            && scheduled
                .iter()
                .any(|pending| should_capture_dflash_prefill_context(&pending.req));
        if let Some(plan) =
            plan::build_next_plan(!active.is_empty() && !force_prefill_for_dflash, scheduled)
        {
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
                    dflash.as_mut(),
                ),
                ExecutionPlan::Decode => {
                    if !decode_step_speculative(
                        &model,
                        &mut active,
                        &mut graph_state,
                        dflash.as_mut(),
                    ) {
                        decode_step(
                            &model,
                            &mut active,
                            &mut graph_state,
                            &mut rng,
                            dflash.as_mut(),
                        );
                    }
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
    dflash: Option<&mut DFlashSchedulerState>,
) {
    let mut chunk = ScheduledChunk::from(scheduled);
    let may_capture_dflash_prefill = true;
    let should_capture_dflash = dflash.is_some()
        && may_capture_dflash_prefill
        && chunk.reqs.iter().any(should_capture_dflash_prefill_context);
    let capture_layer_ids = dflash
        .as_ref()
        .filter(|_| should_capture_dflash)
        .map(|d| d.capture_layer_ids());
    // Scope the borrows of `chunk` to the executor call so the error path can
    // move `chunk` into `fail_chunk`.
    let result = {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(|w| w.as_slice()).collect();
        let mut rec_refs: Vec<&mut RecurrentState> = chunk.recs.iter_mut().collect();
        model.batch_prefill_logits_with_capture(
            &window_refs,
            &mut chunk.kvs,
            &mut rec_refs,
            capture_layer_ids,
        )
    };
    let (logits, captured_hidden) = match result {
        Ok(v) => v,
        Err(e) => {
            warn!("batch prefill failed: {e}");
            fail_chunk(chunk, &e.to_string());
            return;
        }
    };
    let mut dflash = dflash;
    if let Some(dflash) = dflash.as_mut() {
        if should_capture_dflash {
            if let Err(e) =
                record_dflash_prefill_context(model, &mut chunk, dflash, captured_hidden.as_ref())
            {
                warn!("DFlash prefill context failed: {e}");
                fail_chunk(chunk, &e.to_string());
                return;
            }
        } else {
            for local_id in &chunk.local_ids {
                dflash.drop_request(*local_id);
            }
        }
    }

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
        dflash,
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

fn record_dflash_prefill_context(
    model: &Qwen35Model,
    chunk: &mut ScheduledChunk,
    dflash: &mut DFlashSchedulerState,
    captured_hidden: Option<&HiddenStates>,
) -> Result<()> {
    let captured_hidden = captured_hidden.ok_or_else(|| {
        anyhow::anyhow!("DFlash prefill capture requested but no hidden returned")
    })?;
    let expected_tokens: usize = chunk.windows.iter().map(Vec::len).sum();
    anyhow::ensure!(
        captured_hidden.seq_len == expected_tokens,
        "Qwen3.5 DFlash captured {} hidden rows for {} scheduled tokens",
        captured_hidden.seq_len,
        expected_tokens
    );
    let mut token_offset = 0usize;
    for (i, req) in chunk.reqs.iter().enumerate() {
        let local_id = chunk.local_ids[i];
        let chunk_start = chunk.ends[i] - chunk.windows[i].len();
        let capture_supported = req.logprobs == 0 && !req.echo && req.params.is_greedy();
        if capture_supported {
            let max_cache_len =
                (req.prompt_tokens.len() + req.max_tokens + dflash.model.block_size())
                    .min(dflash.model.max_position_embeddings());
            let mut state = match dflash.requests.remove(&local_id) {
                Some(state) => state,
                None => dflash
                    .model
                    .new_request_state(model.device_ctx(), max_cache_len)?,
            };
            let pending_len = state.pending_context_len().unwrap_or(0);
            anyhow::ensure!(
                pending_len == chunk_start,
                "Qwen3.5 DFlash prefill context for local request {local_id} is discontinuous: pending={pending_len}, chunk_start={chunk_start}"
            );
            dflash.model.append_pending_context(
                model.device_ctx(),
                &mut state,
                captured_hidden,
                token_offset,
                chunk.windows[i].len(),
            )?;
            dflash.requests.insert(local_id, state);
        } else {
            dflash.requests.remove(&local_id);
        }
        token_offset += chunk.windows[i].len();
    }
    Ok(())
}

fn should_capture_dflash_prefill_context(req: &SchedulerRequest) -> bool {
    req.logprobs == 0 && !req.echo && req.params.is_greedy()
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
        None,
    );
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

fn decode_step(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    rng: &mut StdRng,
    mut dflash: Option<&mut DFlashSchedulerState>,
) {
    let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
    let may_capture_dflash_decode = true;
    let capture_layer_ids = dflash
        .as_ref()
        .filter(|d| {
            may_capture_dflash_decode
                && active
                    .iter()
                    .any(|req| d.requests.contains_key(&req.local_id))
        })
        .map(|d| d.capture_layer_ids().to_vec());
    let mut kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

    let decode_result = if let Some(capture_layer_ids) = capture_layer_ids.as_deref() {
        model.batch_decode_graph_with_capture(
            &token_ids,
            &mut kv_refs,
            graph_state,
            Some(capture_layer_ids),
        )
    } else {
        model.batch_decode_graph(&token_ids, &mut kv_refs, graph_state)
    };
    if let Err(e) = decode_result {
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

    if let Some(dflash) = dflash.as_mut() {
        if capture_layer_ids.is_some() {
            if let Err(e) = record_dflash_decode_context(
                model,
                active,
                dflash,
                &graph_state.buffers.captured_hidden,
            ) {
                warn!("DFlash decode context failed: {e}");
                for req in active.iter() {
                    dflash.drop_request(req.local_id);
                }
            }
        }
    }
    dispatch_decode_tokens(model, active, &tokens, &logprobs_vec, graph_state, dflash);
}

fn decode_step_speculative(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    dflash: Option<&mut DFlashSchedulerState>,
) -> bool {
    let Some(dflash) = dflash else {
        return false;
    };
    // Multi-active Qwen3.5 DFlash is enabled for greedy/logprobs-free rows once
    // each row has enough captured hidden context for the drafter.
    if active.iter().any(|req| {
        req.logprobs != 0 || !req.params.is_greedy() || !dflash.ready_for_draft(req.local_id)
    }) {
        return false;
    }
    let draft_spans = match execute_dflash_draft(model, active, dflash) {
        Ok(spans) => spans,
        Err(e) => {
            warn!("Qwen3.5 DFlash draft failed, falling back to decode: {e}");
            return false;
        }
    };
    let verify = match verify_dflash_spans(model, active, graph_state, dflash, &draft_spans) {
        Ok(verify) => verify,
        Err(e) => {
            warn!("Qwen3.5 DFlash verify failed: {e}");
            let message = e.to_string();
            for req in active.drain(..) {
                dflash.drop_request(req.local_id);
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            return true;
        }
    };
    dispatch_speculative_tokens(model, active, &verify, graph_state, dflash);
    true
}

fn record_dflash_decode_context(
    model: &Qwen35Model,
    active: &[ActiveRequest35],
    dflash: &mut DFlashSchedulerState,
    captured_hidden: &HiddenStates,
) -> Result<()> {
    anyhow::ensure!(
        captured_hidden.seq_len >= active.len(),
        "Qwen3.5 DFlash decode captured {} rows for {} active requests",
        captured_hidden.seq_len,
        active.len()
    );
    for (slot_idx, req) in active.iter().enumerate() {
        if req.logprobs != 0 || !req.params.is_greedy() {
            dflash.drop_request(req.local_id);
            continue;
        }
        let Some(state) = dflash.requests.get_mut(&req.local_id) else {
            continue;
        };
        dflash.model.append_pending_context(
            model.device_ctx(),
            state,
            captured_hidden,
            slot_idx,
            1,
        )?;
    }
    Ok(())
}

fn execute_dflash_draft(
    model: &Qwen35Model,
    active: &[ActiveRequest35],
    dflash: &mut DFlashSchedulerState,
) -> Result<Vec<Vec<u32>>> {
    let block_size = dflash.model.block_size();
    let current_tokens: Vec<u32> = active.iter().map(|req| req.last_token).collect();
    let mut taken = Vec::with_capacity(active.len());
    for req in active {
        let state = dflash
            .requests
            .remove(&req.local_id)
            .ok_or_else(|| anyhow::anyhow!("missing Qwen3.5 DFlash state for {}", req.local_id))?;
        taken.push((req.local_id, state));
    }
    let result = (|| -> Result<Vec<Vec<u32>>> {
        let mut state_refs: Vec<&mut DFlashRequestState> =
            taken.iter_mut().map(|(_, state)| state).collect();
        let logits = dflash.model.draft_logits_batched(
            model,
            &mut state_refs,
            &current_tokens,
            &mut dflash.scratch,
        )?;
        anyhow::ensure!(
            logits.seq_len == active.len() * block_size,
            "Qwen3.5 DFlash logits rows {} != active {} x block {}",
            logits.seq_len,
            active.len(),
            block_size
        );
        let greedy = SamplingParams::default();
        let params: Vec<&SamplingParams> = vec![&greedy; logits.seq_len];
        let sampled = openinfer_sample::select_batch(
            model.device_ctx(),
            logits,
            &params,
            0,
            &mut dflash.sample,
        )?;
        let drafts_start = if dflash.model.anchor_first() { 0 } else { 1 };
        let mut spans = Vec::with_capacity(active.len());
        for (i, req) in active.iter().enumerate() {
            let remaining = req.max_tokens.saturating_sub(req.generated_count);
            if remaining == 0 {
                spans.push(vec![req.last_token]);
                continue;
            }
            let block = &sampled[i * block_size..(i + 1) * block_size];
            let drafts = &block[drafts_start..];
            let draft_limit =
                dflash_verify_draft_limit(dflash, req.local_id, drafts.len()).min(remaining);
            let mut span = Vec::with_capacity(drafts.len() + 1);
            span.push(req.last_token);
            span.extend(drafts.iter().take(draft_limit).copied());
            spans.push(span);
        }
        Ok(spans)
    })();
    for (local_id, state) in taken {
        dflash.requests.insert(local_id, state);
    }
    result
}

fn dflash_verify_draft_limit(
    dflash: &DFlashSchedulerState,
    local_id: usize,
    max_drafts: usize,
) -> usize {
    if max_drafts == 0 {
        return 0;
    }
    let Some(stats) = dflash.stats.get(&local_id) else {
        return max_drafts.min(DFLASH_PROBE_DRAFT_TOKENS);
    };
    if stats.verified_draft_tokens < DFLASH_MIN_STATS_DRAFT_TOKENS {
        return max_drafts.min(DFLASH_PROBE_DRAFT_TOKENS);
    }
    let rate = if stats.verified_draft_tokens == 0 {
        0.0
    } else {
        stats.accepted_draft_tokens as f64 / stats.verified_draft_tokens as f64
    };
    if rate >= DFLASH_FULL_BLOCK_ACCEPT_RATE {
        max_drafts
    } else {
        max_drafts.min(DFLASH_PROBE_DRAFT_TOKENS)
    }
}

fn verify_dflash_spans(
    model: &Qwen35Model,
    active: &mut [ActiveRequest35],
    graph_state: &mut BatchDecodeGraphState,
    dflash: &mut DFlashSchedulerState,
    spans: &[Vec<u32>],
) -> Result<Vec<Vec<VerifiedToken>>> {
    dflash.ensure_state_scratch(model.device_ctx(), model.config(), active.len())?;
    copy_recurrent_states_into(
        model,
        active,
        graph_state,
        &mut dflash.backup_states[..active.len()],
    )?;
    let capture_layer_ids = dflash.capture_layer_ids().to_vec();
    let original_seq_lens: Vec<usize> = active.iter().map(|req| req.kv.seq_len()).collect();
    let result = (|| -> Result<Vec<Vec<VerifiedToken>>> {
        for (slot_idx, span) in spans.iter().enumerate() {
            anyhow::ensure!(
                span.len() >= 2,
                "Qwen3.5 DFlash verify span for local request {} is too short",
                active[slot_idx].local_id
            );
            dflash.verify_scratch_states[slot_idx]
                .copy_from(model.device_ctx(), &dflash.backup_states[slot_idx])?;
        }

        let span_refs: Vec<&[u32]> = spans.iter().map(Vec::as_slice).collect();
        let active_len = active.len();
        let mut kv_refs: Vec<&mut KvState> = active.iter_mut().map(|req| &mut req.kv).collect();
        let mut rec_refs: Vec<&mut RecurrentState> = dflash.verify_scratch_states[..active_len]
            .iter_mut()
            .collect();
        model.prefill_verify_into(
            &span_refs,
            &mut kv_refs,
            &mut rec_refs,
            &capture_layer_ids,
            &mut dflash.verify_bufs,
        )?;

        let total_rows: usize = spans.iter().map(Vec::len).sum();
        let greedy = SamplingParams::default();
        let params: Vec<&SamplingParams> = vec![&greedy; total_rows];
        let target_tokens = openinfer_sample::select_batch(
            model.device_ctx(),
            &dflash.verify_bufs.logits,
            &params,
            0,
            &mut dflash.verify_bufs.sample,
        )?;

        let mut accepted_rows = Vec::with_capacity(active.len());
        let mut matched_draft_rows = Vec::with_capacity(active.len());
        let mut row_offset = 0usize;
        for (slot_idx, req) in active.iter_mut().enumerate() {
            let span = &spans[slot_idx];
            let row_end = row_offset + span.len();
            let target_slice = &target_tokens[row_offset..row_end];
            let (matched, accepted_ids) = accept_greedy(&span[1..], target_slice);
            if std::env::var_os("OPENINFER_QWEN35_DFLASH_TRACE").is_some() {
                log::info!(
                    "Qwen3.5 DFlash trace local_request={} span_head={:?} target_head={:?} matched_drafts={} accepted_head={:?}",
                    req.local_id,
                    &span[..span.len().min(12)],
                    &target_slice[..target_slice.len().min(12)],
                    matched,
                    &accepted_ids[..accepted_ids.len().min(12)],
                );
            }
            let accepted: Vec<VerifiedToken> = accepted_ids
                .into_iter()
                .map(|token| VerifiedToken {
                    token,
                    logprob: None,
                })
                .collect();
            matched_draft_rows.push(matched);
            accepted_rows.push(accepted);
            row_offset = row_end;
        }

        let mut local_ids = Vec::with_capacity(active.len());
        let mut span_lens = Vec::with_capacity(active.len());
        for (slot_idx, req) in active.iter_mut().enumerate() {
            let span = &spans[slot_idx];
            let accepted_len = accepted_rows[slot_idx].len();
            anyhow::ensure!(
                accepted_len > 0 && accepted_len <= span.len(),
                "Qwen3.5 DFlash accepted span {} outside verify span {} for local request {}",
                accepted_len,
                span.len(),
                req.local_id
            );
            let committed_drafts = matched_draft_rows[slot_idx];
            dflash.verified_draft_tokens += span.len().saturating_sub(1);
            dflash.accepted_draft_tokens += committed_drafts;
            let stats = dflash.stats.entry(req.local_id).or_default();
            stats.verified_draft_tokens += span.len().saturating_sub(1);
            stats.accepted_draft_tokens += committed_drafts;
            local_ids.push(req.local_id);
            span_lens.push(span.len());
        }

        let accepted_lens: Vec<usize> = accepted_rows.iter().map(Vec::len).collect();
        let accepted_token_ids: Vec<Vec<u32>> = accepted_rows
            .iter()
            .map(|row| row.iter().map(|token| token.token).collect())
            .collect();
        let commit_input_token_ids: Vec<Vec<u32>> = spans
            .iter()
            .zip(accepted_token_ids.iter())
            .zip(accepted_lens.iter())
            .map(|((span, accepted_ids), &accepted_len)| {
                let mut ids = Vec::with_capacity(accepted_len);
                ids.push(span[0]);
                ids.extend(
                    accepted_ids
                        .iter()
                        .take(accepted_len.saturating_sub(1))
                        .copied(),
                );
                ids
            })
            .collect();
        let mut commit_captured_offsets: Vec<usize> = (0..active.len())
            .map(|slot_idx| row_offset_for_span(spans, slot_idx))
            .collect();

        let batch_keep_speculating = if active.len() > 1 {
            let batch_verified: usize = active
                .iter()
                .filter_map(|req| dflash.stats.get(&req.local_id))
                .map(|stats| stats.verified_draft_tokens)
                .sum();
            let batch_accepted: usize = active
                .iter()
                .filter_map(|req| dflash.stats.get(&req.local_id))
                .map(|stats| stats.accepted_draft_tokens)
                .sum();
            batch_verified < DFLASH_MIN_STATS_DRAFT_TOKENS * active.len()
                || (batch_accepted as f64 / batch_verified as f64) >= DFLASH_MIN_ACCEPT_RATE
        } else {
            true
        };
        let keep_speculating_by_slot: Vec<bool> = (0..active.len())
            .map(|slot_idx| {
                if active.len() > 1 {
                    batch_keep_speculating
                } else {
                    match dflash.stats.get(&local_ids[slot_idx]) {
                        Some(stats)
                            if stats.verified_draft_tokens >= DFLASH_MIN_STATS_DRAFT_TOKENS =>
                        {
                            let rate = stats.accepted_draft_tokens as f64
                                / stats.verified_draft_tokens as f64;
                            rate >= DFLASH_MIN_ACCEPT_RATE
                        }
                        _ => true,
                    }
                }
            })
            .collect();

        let full_span_commit = accepted_lens
            .iter()
            .zip(spans.iter())
            .all(|(&accepted_len, span)| accepted_len == span.len());
        let mut append_after_commit = vec![true; active.len()];
        let commit_path = if full_span_commit {
            dflash.commit_stats.full_span_rounds += 1;
            for (slot_idx, req) in active.iter_mut().enumerate() {
                graph_state.copy_state_to_slot(
                    model.device_ctx(),
                    &dflash.verify_scratch_states[slot_idx],
                    req.graph_slot_idx,
                )?;
            }
            "full_span_state_commit"
        } else {
            dflash.commit_stats.prefix_state_rounds += 1;
            dflash.commit_stats.prefix_state_rows += accepted_lens.iter().sum::<usize>();
            let partial_indices: Vec<usize> = accepted_lens
                .iter()
                .zip(spans.iter())
                .enumerate()
                .filter_map(|(slot_idx, (&accepted_len, span))| {
                    (accepted_len < span.len()).then_some(slot_idx)
                })
                .collect();
            for slot_idx in 0..active.len() {
                if accepted_lens[slot_idx] == spans[slot_idx].len() {
                    append_after_commit[slot_idx] = false;
                    if keep_speculating_by_slot[slot_idx]
                        && let Some(state) = dflash.requests.get_mut(&local_ids[slot_idx])
                    {
                        dflash.model.append_pending_context(
                            model.device_ctx(),
                            state,
                            &dflash.verify_bufs.captured_hidden,
                            row_offset_for_span(spans, slot_idx),
                            accepted_lens[slot_idx],
                        )?;
                    }
                }
            }
            let mut offset = 0usize;
            for &slot_idx in &partial_indices {
                commit_captured_offsets[slot_idx] = offset;
                offset += accepted_lens[slot_idx];
                dflash.commit_scratch_states[slot_idx]
                    .copy_from(model.device_ctx(), &dflash.backup_states[slot_idx])?;
                active[slot_idx]
                    .kv
                    .truncate_to(original_seq_lens[slot_idx])?;
            }
            if !partial_indices.is_empty() {
                let prefix_refs: Vec<&[u32]> = partial_indices
                    .iter()
                    .map(|&slot_idx| commit_input_token_ids[slot_idx].as_slice())
                    .collect();
                let mut kv_refs: Vec<&mut KvState> = active
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(slot_idx, req)| {
                        partial_indices.contains(&slot_idx).then_some(&mut req.kv)
                    })
                    .collect();
                let mut rec_refs: Vec<&mut RecurrentState> = dflash
                    .commit_scratch_states
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(slot_idx, state)| {
                        partial_indices.contains(&slot_idx).then_some(state)
                    })
                    .collect();
                model.prefill_verify_into(
                    &prefix_refs,
                    &mut kv_refs,
                    &mut rec_refs,
                    &capture_layer_ids,
                    &mut dflash.verify_bufs,
                )?;
            }
            for (slot_idx, req) in active.iter_mut().enumerate() {
                let state = if accepted_lens[slot_idx] == spans[slot_idx].len() {
                    &dflash.verify_scratch_states[slot_idx]
                } else {
                    &dflash.commit_scratch_states[slot_idx]
                };
                graph_state.copy_state_to_slot(model.device_ctx(), state, req.graph_slot_idx)?;
            }
            if accepted_lens
                .first()
                .is_some_and(|&first| accepted_lens.iter().all(|&len| len == first))
            {
                dflash.commit_stats.same_partial_rounds += 1;
                "same_len_prefix_state_commit"
            } else {
                dflash.commit_stats.batched_replay_rounds += 1;
                dflash.commit_stats.batched_replay_rows += partial_indices
                    .iter()
                    .map(|&slot_idx| accepted_lens[slot_idx])
                    .sum::<usize>();
                "heterogeneous_prefix_state_commit"
            }
        };
        for slot_idx in 0..active.len() {
            if append_after_commit[slot_idx]
                && keep_speculating_by_slot[slot_idx]
                && let Some(state) = dflash.requests.get_mut(&local_ids[slot_idx])
            {
                dflash.model.append_pending_context(
                    model.device_ctx(),
                    state,
                    &dflash.verify_bufs.captured_hidden,
                    commit_captured_offsets[slot_idx],
                    accepted_lens[slot_idx],
                )?;
            }
        }

        for slot_idx in 0..active.len() {
            let accepted_len = accepted_lens[slot_idx];
            let local_id = local_ids[slot_idx];
            let span_len = span_lens[slot_idx];
            let keep_speculating = keep_speculating_by_slot[slot_idx];
            if !keep_speculating {
                dflash.drop_request(local_id);
            }
            let rate = if dflash.verified_draft_tokens == 0 {
                0.0
            } else {
                dflash.accepted_draft_tokens as f64 / dflash.verified_draft_tokens as f64
            };
            let request_rate = dflash.stats.get(&local_id).map_or(0.0, |stats| {
                if stats.verified_draft_tokens == 0 {
                    0.0
                } else {
                    stats.accepted_draft_tokens as f64 / stats.verified_draft_tokens as f64
                }
            });
            log::debug!(
                "Qwen3.5 DFlash local_request={} accepted_tokens={} verified_span={} request_accept_rate={:.3} cumulative_accept_rate={:.3} keep_speculating={} captured_offset={}",
                local_id,
                accepted_len,
                span_len,
                request_rate,
                rate,
                keep_speculating,
                commit_captured_offsets[slot_idx],
            );
        }
        if std::env::var_os("OPENINFER_QWEN35_DFLASH_COMMIT_TRACE").is_some() {
            log::info!(
                "Qwen3.5 DFlash commit path={} active={} accepted_lens={:?} span_lens={:?} full={} same_partial={} batched_replay={} batched_replay_rows={} prefix_state={} prefix_state_rows={}",
                commit_path,
                active.len(),
                accepted_lens,
                span_lens,
                dflash.commit_stats.full_span_rounds,
                dflash.commit_stats.same_partial_rounds,
                dflash.commit_stats.batched_replay_rounds,
                dflash.commit_stats.batched_replay_rows,
                dflash.commit_stats.prefix_state_rounds,
                dflash.commit_stats.prefix_state_rows,
            );
        }

        Ok(accepted_rows)
    })();
    if result.is_err() {
        for ((req, backup_state), &seq_len) in active
            .iter_mut()
            .zip(dflash.backup_states.iter())
            .zip(original_seq_lens.iter())
        {
            let _ = req.kv.truncate_to(seq_len);
            let _ = graph_state.copy_state_to_slot(
                model.device_ctx(),
                backup_state,
                req.graph_slot_idx,
            );
        }
    }
    result
}

fn row_offset_for_span(spans: &[Vec<u32>], slot_idx: usize) -> usize {
    spans[..slot_idx].iter().map(Vec::len).sum()
}

fn copy_recurrent_states_into(
    model: &Qwen35Model,
    active: &[ActiveRequest35],
    graph_state: &BatchDecodeGraphState,
    states: &mut [RecurrentState],
) -> Result<()> {
    anyhow::ensure!(
        states.len() >= active.len(),
        "Qwen3.5 DFlash backup state capacity {} < active {}",
        states.len(),
        active.len()
    );
    for (req, state) in active.iter().zip(states.iter_mut()) {
        graph_state.copy_slot_to_state(model.device_ctx(), req.graph_slot_idx, state)?;
    }
    Ok(())
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

    dispatch_decode_tokens(model, active, &tokens, &logprobs_vec, graph_state, None);
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
    mut dflash: Option<&mut DFlashSchedulerState>,
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
        if let Some(dflash) = dflash.as_mut() {
            dflash.drop_request(active[i].local_id);
        }
        compact_slot(model, active, graph_state, i);
    }
}

fn dispatch_speculative_tokens(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    accepted: &[Vec<VerifiedToken>],
    graph_state: &mut BatchDecodeGraphState,
    dflash: &mut DFlashSchedulerState,
) {
    let n = active.len();
    let mut to_retire = Vec::new();

    for i in 0..n {
        let req = &mut active[i];
        let mut should_retire = None;
        for token in &accepted[i] {
            req.generated_count += 1;
            let is_eos = !req.params.ignore_eos && model.is_stop_token(token.token);
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
                should_retire = Some(i);
                break;
            }
            if req
                .token_tx
                .send(TokenEvent::Token {
                    id: token.token,
                    logprob: token.logprob.clone(),
                })
                .is_err()
            {
                debug!(
                    "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                    req.request_id, req.generated_count
                );
                should_retire = Some(i);
                break;
            }
            if at_limit {
                debug!(
                    "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                    req.request_id,
                    req.prompt_len,
                    req.generated_count,
                    FinishReason::Length
                );
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Length,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
                should_retire = Some(i);
                break;
            }
            req.last_token = token.token;
        }
        if let Some(idx) = should_retire {
            to_retire.push(idx);
        }
    }

    for &i in to_retire.iter().rev() {
        let local_id = active[i].local_id;
        dflash.drop_request(local_id);
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

        graph_state
            .copy_slot_to_slot(
                model.device_ctx(),
                compaction.moved_from,
                compaction.moved_to,
            )
            .expect("compact slot recurrent state copy failed");

        active[compaction.moved_to].graph_slot_idx = compaction.moved_to;
    }
}

// ── Chunked-prefill helpers ────────────────────────────────────────────────

/// Step's scheduled prefill set
struct ScheduledChunk {
    local_ids: Vec<usize>,
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
            local_ids: Vec::with_capacity(n),
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
            chunk.local_ids.push(p.local_id);
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
    dflash: Option<&mut DFlashSchedulerState>,
) {
    let ScheduledChunk {
        local_ids,
        reqs,
        kvs,
        recs,
        ends,
        ..
    } = chunk;
    let mut still_prefilling: Vec<PrefillingRequest35> = Vec::new();
    let mut dflash = dflash;

    for (i, ((((req, kv), rec), end), local_id)) in reqs
        .into_iter()
        .zip(kvs)
        .zip(recs)
        .zip(ends)
        .zip(local_ids)
        .enumerate()
    {
        // Not finished: re-queue with the advanced cursor
        if end < req.prompt_tokens.len() {
            still_prefilling.push(PrefillingRequest35 {
                local_id,
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
            if let Some(dflash) = dflash.as_mut() {
                dflash.drop_request(local_id);
            }
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
            if let Some(dflash) = dflash.as_mut() {
                dflash.drop_request(local_id);
            }
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
            if let Some(dflash) = dflash.as_mut() {
                dflash.drop_request(local_id);
            }
            continue;
        }

        // Assign a graph slot and copy recurrent state into it.
        let slot_idx = slot_for_new_request(active.len(), graph_state.slot_states.len())
            .expect("admission must reserve a graph slot");
        graph_state
            .copy_state_to_slot(model.device_ctx(), &rec, slot_idx)
            .expect("copy recurrent state to slot failed");
        active.push(ActiveRequest35 {
            local_id,
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
