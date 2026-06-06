//! Scheduler for Qwen3.5: dedicated GPU thread that batches concurrent requests.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - `RecurrentState` alongside `KvState` (linear attention layers)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)
//!
//! Prefill allocates temporary `RecurrentState`s, then D2D-copies them into
//! graph slots. On request retirement, swap-remove compaction keeps slots dense.

mod plan;

use std::sync::mpsc as std_mpsc;
use std::thread;

use anyhow::Result;
use log::{info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::recurrent_state::RecurrentState;
use crate::weights::Qwen35Model;
use pegainfer_core::engine::{
    EngineHandle as SchedulerHandle, FinishReason, GenerateRequest as SchedulerRequest, TokenEvent,
    TokenLogprob,
};
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::sampler::SamplingParams;
use pegainfer_core::tensor::DeviceVec;

use self::plan::{
    ExecutionPlan, admit_pending_requests, compaction_after_retire, slot_for_new_request,
};

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded. Recurrent state lives in the
/// `BatchDecodeGraphState` at `graph_slot_idx` — NOT owned here.
struct ActiveRequest35 {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
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

/// Scratch buffers for GPU sampling (reused across prefill sampling).
struct SampleScratch {
    probs: cudarc::driver::CudaSlice<f32>,
    top1_value: cudarc::driver::CudaSlice<half::bf16>,
    row_states: cudarc::driver::CudaSlice<u8>,
    valid: cudarc::driver::CudaSlice<u8>,
    out: cudarc::driver::CudaSlice<i32>,
}

impl SampleScratch {
    fn new(model: &Qwen35Model) -> Result<Self> {
        let vocab_size = model.config().vocab_size;
        let ctx = model.device_ctx();
        Ok(Self {
            probs: ctx.stream.alloc_zeros(vocab_size)?,
            top1_value: ctx.stream.alloc_zeros(1)?,
            row_states: ctx
                .stream
                .alloc_zeros(crate::ops::flashinfer_topk_row_states_bytes())?,
            valid: ctx.stream.alloc_zeros(1)?,
            out: ctx.stream.alloc_zeros(1)?,
        })
    }
}

// ── Entry point ─────────────────────────────────────────────────────────

/// Start the Qwen3.5 scheduler thread with default max batch size (64).
pub fn start(model: Qwen35Model, seed: u64) -> Result<SchedulerHandle> {
    start_with_capacity(model, seed, crate::batch_decode_graph::MAX_BATCH)
}

/// Start the Qwen3.5 scheduler thread with a custom max batch size.
///
/// Lower `max_batch` reduces GPU memory usage (each slot holds a full
/// RecurrentState for all linear attention layers).
pub fn start_with_capacity(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
) -> Result<SchedulerHandle> {
    let graph_state = model.create_batch_decode_graph_state_with_capacity(max_batch)?;
    let sample_scratch = SampleScratch::new(&model)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (startup_tx, startup_rx) = std_mpsc::channel();

    thread::Builder::new()
        .name("scheduler-qwen35".into())
        .spawn(move || match bind_model_thread(&model) {
            Ok(_guard) => {
                let _ = startup_tx.send(Ok(()));
                scheduler_loop(
                    model,
                    submit_rx,
                    graph_state,
                    sample_scratch,
                    seed,
                    max_batch,
                );
            }
            Err(err) => {
                let _ = startup_tx.send(Err(err));
            }
        })
        .expect("failed to spawn Qwen3.5 scheduler thread");

    startup_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("Qwen3.5 scheduler exited during startup"))??;
    Ok(SchedulerHandle::new(submit_tx))
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
    Ok(CublasThreadGuard)
}

// ── Main loop ───────────────────────────────────────────────────────────

#[allow(clippy::needless_pass_by_value)]
fn scheduler_loop(
    model: Qwen35Model,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    mut graph_state: BatchDecodeGraphState,
    mut sample_scratch: SampleScratch,
    seed: u64,
    max_batch: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest35> = Vec::new();
    let mut deferred: Vec<SchedulerRequest> = Vec::new();

    info!("Qwen3.5 scheduler ready (max_batch={})", max_batch);

    loop {
        // 1. Drain all pending requests (deferred from last iteration + channel)
        let mut pending = std::mem::take(&mut deferred);
        while let Ok(req) = submit_rx.try_recv() {
            pending.push(req);
        }

        // 2. Nothing active and nothing pending → block until a request arrives
        if active.is_empty() && pending.is_empty() {
            if let Some(req) = submit_rx.blocking_recv() {
                pending.push(req);
            } else {
                info!("Qwen3.5 scheduler: all handles dropped, exiting");
                return;
            }
            while let Ok(req) = submit_rx.try_recv() {
                pending.push(req);
            }
        }

        let admission = admit_pending_requests(
            pending,
            active.len(),
            graph_state.slot_states.len(),
            model.kv_pool().layout().page_size,
            model.kv_pool().available_pages(),
            |req| req.prompt_tokens.len(),
        );
        let pending = admission.pending;
        deferred = admission.deferred;

        match plan::build_next_plan(!active.is_empty(), pending) {
            Some(ExecutionPlan::Unified { pending }) => unified_step_sched(
                &model,
                &mut active,
                pending,
                &mut graph_state,
                &mut sample_scratch,
                &mut rng,
            ),
            Some(ExecutionPlan::Prefill { pending }) => prefill_batch(
                &model,
                &mut active,
                pending,
                &mut graph_state,
                &mut sample_scratch,
                &mut rng,
            ),
            Some(ExecutionPlan::Decode) => {
                decode_step(&model, &mut active, &mut graph_state, &mut rng)
            }
            None => {}
        }
    }
}

// ── Batch prefill ───────────────────────────────────────────────────────

fn prefill_batch(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    pending: Vec<SchedulerRequest>,
    graph_state: &mut BatchDecodeGraphState,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
    let mut kv_states: Vec<KvState> = (0..pending.len()).map(|_| model.alloc_kv()).collect();

    // Allocate temporary recurrent states for prefill
    let mut rec_states: Vec<RecurrentState> = (0..pending.len())
        .map(|_| RecurrentState::new(model.device_ctx(), model.config()).unwrap())
        .collect();
    let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();

    let logits_vec = match model.batch_prefill(&prompts, &mut kv_states, &mut rec_refs) {
        Ok(v) => v,
        Err(e) => {
            warn!("Qwen3.5 batch prefill failed: {e}");
            for req in pending {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            return;
        }
    };

    for (i, req) in pending.into_iter().enumerate() {
        let prompt_len = req.prompt_tokens.len();

        let first_token = match sample_from_logits(model, &logits_vec[i], scratch, &req.params, rng)
        {
            Ok(t) => t,
            Err(e) => {
                warn!("First token sampling failed for request {i}: {e}");
                continue;
            }
        };

        let logprob = if req.logprobs > 0 {
            extract_logprobs(model, &logits_vec[i], first_token, req.logprobs).ok()
        } else {
            None
        };

        if req.echo {
            let echo_logprobs = vec![None; req.prompt_tokens.len()];
            let _ = req.token_tx.send(TokenEvent::PromptTokens {
                ids: req.prompt_tokens.clone(),
                logprobs: echo_logprobs,
            });
        }

        if !req.params.ignore_eos && model.is_stop_token(first_token) {
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
            continue;
        }

        if req.max_tokens <= 1 {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
            continue;
        }

        // Assign a graph slot and copy recurrent state into it
        let slot_idx = slot_for_new_request(active.len(), graph_state.slot_states.len())
            .expect("admission must reserve a graph slot");
        graph_state
            .copy_state_to_slot(model.device_ctx(), &rec_states[i], slot_idx)
            .expect("copy recurrent state to slot failed");

        let kv = std::mem::replace(&mut kv_states[i], model.alloc_kv());
        active.push(ActiveRequest35 {
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
}

// ── Unified step (prefill + decode in one forward pass) ────────────────

fn unified_step_sched(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    pending: Vec<SchedulerRequest>,
    graph_state: &mut BatchDecodeGraphState,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    // Build prefill inputs
    let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
    let mut prefill_kv_states: Vec<KvState> =
        (0..pending.len()).map(|_| model.alloc_kv()).collect();

    // Allocate temporary recurrent states for prefill
    let mut rec_states: Vec<RecurrentState> = (0..pending.len())
        .map(|_| RecurrentState::new(model.device_ctx(), model.config()).unwrap())
        .collect();
    let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();

    // Build decode inputs
    let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
    let mut decode_kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

    // Run unified forward pass
    let (prefill_logits, decode_logits) = match model.unified_step(
        &prompts,
        &mut prefill_kv_states,
        &mut rec_refs,
        &decode_tokens,
        &mut decode_kv_refs,
        graph_state,
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!("Qwen3.5 unified step failed: {e}");
            // Notify all pending requests
            for req in pending {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            // Notify all active decode requests
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            return;
        }
    };

    // Process decode results FIRST (before adding prefill results to active)
    process_decode_logits(model, active, &decode_logits, graph_state, scratch, rng);

    // Process prefill results: sample first token, add to active
    for (i, req) in pending.into_iter().enumerate() {
        let prompt_len = req.prompt_tokens.len();

        let first_token =
            match sample_from_logits(model, &prefill_logits[i], scratch, &req.params, rng) {
                Ok(t) => t,
                Err(e) => {
                    warn!("First token sampling failed for request {i}: {e}");
                    continue;
                }
            };

        let logprob = if req.logprobs > 0 {
            extract_logprobs(model, &prefill_logits[i], first_token, req.logprobs).ok()
        } else {
            None
        };

        if req.echo {
            let echo_logprobs = vec![None; req.prompt_tokens.len()];
            let _ = req.token_tx.send(TokenEvent::PromptTokens {
                ids: req.prompt_tokens.clone(),
                logprobs: echo_logprobs,
            });
        }

        if !req.params.ignore_eos && model.is_stop_token(first_token) {
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
            continue;
        }

        if req.max_tokens <= 1 {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
            continue;
        }

        // Assign graph slot and copy recurrent state
        let slot_idx = slot_for_new_request(active.len(), graph_state.slot_states.len())
            .expect("admission must reserve a graph slot");
        graph_state
            .copy_state_to_slot(model.device_ctx(), &rec_states[i], slot_idx)
            .expect("copy recurrent state to slot failed");

        let kv = std::mem::replace(&mut prefill_kv_states[i], model.alloc_kv());
        active.push(ActiveRequest35 {
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
        warn!("Qwen3.5 batch_decode_graph error: {e}");
        for req in active.drain(..) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
        }
        return;
    }

    // Snapshot logits to CPU BEFORE sampling (sampling may modify bufs.logits)
    let any_logprobs = active.iter().any(|r| r.logprobs > 0);
    let cpu_logits: Vec<Option<Vec<f32>>> = if any_logprobs {
        (0..active.len())
            .map(|i| {
                if active[i].logprobs > 0 {
                    crate::ops::extract_vec(model.device_ctx(), &graph_state.buffers.logits, i)
                        .ok()
                        .and_then(|v| v.to_host(model.device_ctx()).ok())
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![None; active.len()]
    };

    let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
    let tokens = match model.select_tokens_batch_varied(&mut graph_state.buffers, &params_refs, rng)
    {
        Ok(t) => t,
        Err(e) => {
            warn!("Qwen3.5 sampling error: {e}");
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
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
                compute_logprobs_from_cpu(&logits_f32, tokens[i], active[i].logprobs)
            })
        })
        .collect();

    dispatch_decode_tokens(model, active, &tokens, &logprobs_vec, graph_state);
}

/// Process decode logits from unified step: sample, extract logprobs, dispatch.
fn process_decode_logits(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    decode_logits: &[DeviceVec],
    graph_state: &mut BatchDecodeGraphState,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    let mut tokens = Vec::with_capacity(active.len());
    let mut logprobs_vec: Vec<Option<TokenLogprob>> = Vec::with_capacity(active.len());
    for (i, logits) in decode_logits.iter().enumerate() {
        match sample_from_logits(model, logits, scratch, &active[i].params, rng) {
            Ok(t) => {
                let lp = if active[i].logprobs > 0 {
                    extract_logprobs(model, logits, t, active[i].logprobs).ok()
                } else {
                    None
                };
                tokens.push(t);
                logprobs_vec.push(lp);
            }
            Err(e) => {
                warn!("decode sampling error: {e}");
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Stop,
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                return;
            }
        }
    }

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
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            to_retire.push(i);
        } else if at_limit {
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

fn sample_from_logits(
    model: &Qwen35Model,
    logits: &DeviceVec,
    scratch: &mut SampleScratch,
    params: &SamplingParams,
    rng: &mut StdRng,
) -> Result<u32> {
    let random_val: f32 = rand::RngExt::random(rng);
    crate::ops::gpu_sample_into(
        model.device_ctx(),
        logits,
        &mut scratch.probs,
        &mut scratch.top1_value,
        &mut scratch.row_states,
        &mut scratch.valid,
        &mut scratch.out,
        params,
        random_val,
    )
}

fn extract_logprobs(
    model: &Qwen35Model,
    logits: &DeviceVec,
    sampled_token: u32,
    top_k: usize,
) -> Result<TokenLogprob> {
    let logits_f32 = logits.to_host(model.device_ctx())?;
    compute_logprobs_from_cpu(&logits_f32, sampled_token, top_k)
        .ok_or_else(|| anyhow::anyhow!("logprobs computation failed"))
}

fn compute_logprobs_from_cpu(
    logits_f32: &[f32],
    sampled_token: u32,
    top_k: usize,
) -> Option<TokenLogprob> {
    if logits_f32.is_empty() {
        return None;
    }

    let max_val = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits_f32.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();

    let sampled_logprob = logits_f32[sampled_token as usize] - log_sum_exp;

    let k = top_k.min(logits_f32.len());
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    if k > 0 {
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (idx, &val) in logits_f32.iter().enumerate() {
            if best.len() < k || val > best.last().unwrap().1 {
                let pos = best.partition_point(|&(_, v)| v > val);
                best.insert(pos, (idx as u32, val));
                if best.len() > k {
                    best.pop();
                }
            }
        }
        for (idx, val) in best {
            top.push((idx, val - log_sum_exp));
        }
    }

    Some(TokenLogprob {
        logprob: sampled_logprob,
        top_logprobs: top,
    })
}
