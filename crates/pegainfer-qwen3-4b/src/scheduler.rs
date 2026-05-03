//! Scheduler: dedicated GPU thread that batches concurrent requests.
//!
//! Frontend handlers tokenize prompts and submit `GenerateRequest` via channel.
//! The scheduler batch-prefills all pending requests in one forward pass, then
//! batch-decodes all active requests. Per-request tokens flow back through
//! individual channels.

mod effects;
mod plan;
mod resolve;

use std::thread;

use anyhow::Result;
use log::{info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::executor::{Qwen3Executor, RequestId};
use pegainfer_core::engine::{EngineHandle, FinishReason, GenerateRequest, TokenEvent};
use pegainfer_core::sampler::SamplingParams;

use self::effects::apply_effects;
use self::plan::{build_next_plan, execute_plan};
use self::resolve::resolve_step;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded.
pub(super) struct ActiveRequestState {
    pub(super) request_id: RequestId,
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
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) params: SamplingParams,
    pub(super) max_tokens: usize,
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) logprobs: usize,
    pub(super) echo: bool,
}

impl PendingRequest {
    fn from_scheduler_request(request_id: RequestId, req: GenerateRequest) -> Self {
        Self {
            request_id,
            prompt_tokens: req.prompt_tokens,
            params: req.params,
            max_tokens: req.max_tokens,
            token_tx: req.token_tx,
            logprobs: req.logprobs,
            echo: req.echo,
        }
    }
}

// ── Entry point ─────────────────────────────────────────────────────────

pub(crate) fn start_qwen3(
    model_path: &str,
    enable_cuda_graph: bool,
    device_ordinals: &[usize],
    seed: u64,
) -> Result<EngineHandle> {
    let executor = Qwen3Executor::from_runtime(model_path, enable_cuda_graph, device_ordinals)?;
    Ok(start_with_executor(executor, seed))
}

pub(crate) fn start_with_executor(executor: Qwen3Executor, seed: u64) -> EngineHandle {
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop(executor, submit_rx, seed);
        })
        .expect("failed to spawn scheduler thread");

    EngineHandle::new(submit_tx)
}

// ── Main loop ───────────────────────────────────────────────────────────

fn scheduler_loop(
    mut executor: Qwen3Executor,
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequestState> = Vec::new();
    let mut next_request_id = 0u64;
    // Requests that could not be admitted due to KV budget pressure.
    // Held here so they aren't lost; re-evaluated every loop iteration.
    let mut deferred: Vec<PendingRequest> = Vec::new();

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

        // 2. Nothing active and nothing deferred → block until a request arrives.
        if active.is_empty() && deferred.is_empty() {
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
        }

        // 3. Admission control: admit deferred requests only if the KV pool
        //    has enough pages for their prefill, after reserving one page per
        //    active request for its next decode step.
        let page_size = executor.page_size();
        let decode_reserve = active.len(); // one page per active request
        let mut budget = executor.available_pages().saturating_sub(decode_reserve);
        let mut pending: Vec<PendingRequest> = Vec::new();
        let mut still_deferred: Vec<PendingRequest> = Vec::new();
        for req in deferred.drain(..) {
            let needed = req.prompt_tokens.len().div_ceil(page_size);
            if needed <= budget {
                budget -= needed;
                pending.push(req);
            } else {
                still_deferred.push(req);
            }
        }
        deferred = still_deferred;

        let Some(plan) = build_next_plan(!active.is_empty(), pending) else {
            continue;
        };
        let artifacts = match execute_plan(&mut executor, &mut active, plan, &mut rng) {
            Ok(v) => v,
            Err(e) => {
                warn!("Execution step failed: {e}");
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Stop,
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                continue;
            }
        };
        let effects = resolve_step(&executor, &active, artifacts);
        apply_effects(&mut executor, &mut active, effects);
    }
}
