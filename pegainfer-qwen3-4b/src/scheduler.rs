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

use crate::executor::{ModelExecutor, Qwen3Executor, RequestId};
use pegainfer_core::engine::{EngineHandle, GenerateRequest, TokenEvent};
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

pub(crate) fn start_with_executor<E>(executor: E, seed: u64) -> EngineHandle
where
    E: ModelExecutor + 'static,
{
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

fn scheduler_loop<E>(
    mut executor: E,
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    seed: u64,
) where
    E: ModelExecutor,
{
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

        let admission = admit_deferred_requests(
            deferred,
            &active,
            executor.page_size(),
            executor.available_pages(),
            executor.max_request_pages(),
        );
        for rejected in &admission.rejected {
            send_rejection(rejected);
        }
        let pending = admission.pending;
        deferred = admission.deferred;

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
        apply_effects(&mut executor, &mut active, effects);
    }
}

#[derive(Clone)]
struct RequestFailureTarget {
    request_id: RequestId,
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_tokens: usize,
    completion_tokens: usize,
}

struct AdmissionOutcome {
    pending: Vec<PendingRequest>,
    deferred: Vec<PendingRequest>,
    rejected: Vec<PendingRequest>,
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

// Prefill samples the first output token but does not append it to KV. A
// generated token occupies KV only when it is fed as the next decode input.
// Therefore N returned completion tokens occupy at most N - 1 generated-token
// KV slots.
fn max_request_tokens(req: &PendingRequest) -> usize {
    req.prompt_tokens
        .len()
        .saturating_add(req.max_tokens.saturating_sub(1))
}

fn max_active_tokens(req: &ActiveRequestState) -> usize {
    req.prompt_len
        .saturating_add(req.max_tokens.saturating_sub(1))
}

fn current_active_tokens(req: &ActiveRequestState) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

fn active_future_pages(active: &[ActiveRequestState], page_size: usize) -> usize {
    active
        .iter()
        .map(|req| {
            pages_needed(max_active_tokens(req), page_size)
                .saturating_sub(pages_needed(current_active_tokens(req), page_size))
        })
        .sum()
}

fn admit_deferred_requests(
    deferred: Vec<PendingRequest>,
    active: &[ActiveRequestState],
    page_size: usize,
    available_pages: usize,
    max_request_pages: usize,
) -> AdmissionOutcome {
    let mut budget = available_pages.saturating_sub(active_future_pages(active, page_size));
    let mut pending = Vec::new();
    let mut still_deferred = Vec::new();
    let mut rejected = Vec::new();

    for req in deferred {
        let max_needed = pages_needed(max_request_tokens(&req), page_size);
        if max_needed > max_request_pages {
            rejected.push(req);
            continue;
        }

        if max_needed <= budget {
            budget -= max_needed;
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

fn send_rejection(req: &PendingRequest) {
    let max_tokens = max_request_tokens(req);
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message: format!(
            "request requires more KV pages than this model instance can provide: prompt_tokens={}, max_context_tokens={}",
            req.prompt_tokens.len(),
            max_tokens
        ),
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

    use super::*;
    use crate::executor::{
        DecodePlan, DecodeRequestResult, PrefillPlan, PrefillRequestResult, PrefillResult,
        UnifiedPlan, UnifiedResult,
    };

    struct FakeExecutor {
        page_size: usize,
        max_request_pages: usize,
        available_pages: usize,
        held_tokens: HashMap<RequestId, usize>,
        fail_decode_once: bool,
        dropped: Arc<Mutex<Vec<u64>>>,
    }

    impl FakeExecutor {
        fn new(max_request_pages: usize, dropped: Arc<Mutex<Vec<u64>>>) -> Self {
            Self {
                page_size: 16,
                max_request_pages,
                available_pages: max_request_pages,
                held_tokens: HashMap::new(),
                fail_decode_once: false,
                dropped,
            }
        }

        fn with_decode_failure(mut self) -> Self {
            self.fail_decode_once = true;
            self
        }

        fn ensure_request_tokens(
            &mut self,
            request_id: RequestId,
            token_count: usize,
        ) -> Result<()> {
            let current_tokens = self.held_tokens.get(&request_id).copied().unwrap_or(0);
            let current_pages = pages_needed(current_tokens, self.page_size);
            let needed_pages = pages_needed(token_count, self.page_size);
            let grow = needed_pages.saturating_sub(current_pages);
            if grow > self.available_pages {
                anyhow::bail!("fake KV capacity exhausted");
            }
            self.available_pages -= grow;
            self.held_tokens.insert(request_id, token_count);
            Ok(())
        }
    }

    impl ModelExecutor for FakeExecutor {
        fn page_size(&self) -> usize {
            self.page_size
        }

        fn max_request_pages(&self) -> usize {
            self.max_request_pages
        }

        fn available_pages(&self) -> usize {
            self.available_pages
        }

        fn is_stop_token(&self, _token_id: u32) -> bool {
            false
        }

        fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
            if let Some(tokens) = self.held_tokens.remove(&request_id) {
                self.available_pages += pages_needed(tokens, self.page_size);
            }
            self.dropped.lock().unwrap().push(request_id.get());
            Ok(())
        }

        fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
            for req in plan.requests {
                self.ensure_request_tokens(req.request_id, req.prompt_tokens.len())?;
            }
            Ok(PrefillResult {
                requests: plan
                    .requests
                    .iter()
                    .map(|req| PrefillRequestResult {
                        request_id: req.request_id,
                        first_token: 100 + req.request_id.get() as u32,
                        first_token_logprob: None,
                        prompt_logprobs: None,
                    })
                    .collect(),
            })
        }

        fn execute_decode(
            &mut self,
            plan: DecodePlan<'_>,
        ) -> Result<crate::executor::DecodeResult> {
            if self.fail_decode_once {
                self.fail_decode_once = false;
                anyhow::bail!("fake decode KV capacity exhausted");
            }

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
                    .map(|req| PrefillRequestResult {
                        request_id: req.request_id,
                        first_token: 100 + req.request_id.get() as u32,
                        first_token_logprob: None,
                        prompt_logprobs: None,
                    })
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
    fn kv_budget_counts_only_tokens_written_to_cache() {
        let (pending_req, _pending_rx) = request(16, 1);
        let pending = PendingRequest::from_scheduler_request(RequestId(7), pending_req);
        assert_eq!(max_request_tokens(&pending), 16);
        assert_eq!(pages_needed(max_request_tokens(&pending), 16), 1);

        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        let after_prefill = ActiveRequestState {
            request_id: RequestId(8),
            token_tx,
            last_token: 100,
            generated_count: 1,
            max_tokens: 3,
            prompt_len: 16,
            params: SamplingParams::default(),
            logprobs: 0,
        };
        assert_eq!(current_active_tokens(&after_prefill), 16);
        assert_eq!(max_active_tokens(&after_prefill), 18);

        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        let after_one_decode = ActiveRequestState {
            request_id: RequestId(9),
            token_tx,
            last_token: 200,
            generated_count: 2,
            max_tokens: 3,
            prompt_len: 16,
            params: SamplingParams::default(),
            logprobs: 0,
        };
        assert_eq!(current_active_tokens(&after_one_decode), 17);
        assert_eq!(max_active_tokens(&after_one_decode), 18);
    }

    #[test]
    fn one_token_completion_on_page_boundary_fits_one_page() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(1, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42);

        let (fits_exactly, mut rx) = request(16, 1);
        handle.submit(fits_exactly).expect("submit fits_exactly");
        assert!(
            matches!(rx.blocking_recv(), Some(TokenEvent::Token { id: 100, .. })),
            "prefill should emit the sampled token"
        );
        assert!(
            matches!(rx.blocking_recv(), Some(TokenEvent::Finished { .. })),
            "one-token completion should finish without a decode KV page"
        );
        assert!(
            dropped.lock().unwrap().contains(&0),
            "finished request should release its one prompt page"
        );
    }

    #[test]
    fn request_waits_for_full_kv_budget_before_prefill() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42);

        let (long_running, mut long_rx) = request(16, 18);
        handle.submit(long_running).expect("submit long_running");
        assert!(
            matches!(
                long_rx.blocking_recv(),
                Some(TokenEvent::Token { id: 100, .. })
            ),
            "first request should prefill"
        );

        let (must_wait, mut wait_rx) = request(17, 1);
        handle.submit(must_wait).expect("submit must_wait");

        assert!(
            matches!(
                wait_rx.blocking_recv(),
                Some(TokenEvent::Token { id: 101, .. })
            ),
            "waiting request should start once the active request releases its full KV budget"
        );
        assert!(
            dropped.lock().unwrap().contains(&0),
            "second request was admitted before the first request released KV"
        );
        assert!(
            matches!(wait_rx.blocking_recv(), Some(TokenEvent::Finished { .. })),
            "waiting request should finish after admission"
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
                token_tx,
                logprobs: 0,
                echo: false,
            },
            token_rx,
        )
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
        let handle = start_with_executor(executor, 42);

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
                assert!(message.contains("requires more KV pages"));
            }
            _ => panic!("oversized request should be rejected"),
        }

        let (fits, mut fits_rx) = request(16, 1);
        handle.submit(fits).expect("submit fits");
        match fits_rx.blocking_recv() {
            Some(TokenEvent::Token { id, .. }) => assert_eq!(id, 101),
            _ => panic!("later fitting request should emit a token"),
        }
        assert!(
            matches!(fits_rx.blocking_recv(), Some(TokenEvent::Finished { .. })),
            "later fitting request should finish"
        );
    }

    #[test]
    fn decode_error_drops_request_state_and_scheduler_recovers() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped)).with_decode_failure();
        let handle = start_with_executor(executor, 42);

        let (will_fail, mut fail_rx) = request(16, 2);
        handle.submit(will_fail).expect("submit will_fail");
        assert!(
            matches!(
                fail_rx.blocking_recv(),
                Some(TokenEvent::Token { id: 100, .. })
            ),
            "first token should be emitted before decode failure"
        );
        match fail_rx.blocking_recv() {
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
                after_rx.blocking_recv(),
                Some(TokenEvent::Token { id: 101, .. })
            ),
            "scheduler should accept new work after a decode error"
        );
        assert!(
            matches!(after_rx.blocking_recv(), Some(TokenEvent::Finished { .. })),
            "request after failure should finish"
        );
    }

    #[test]
    fn active_receiver_drop_releases_request_state() {
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let executor = FakeExecutor::new(4, Arc::clone(&dropped));
        let handle = start_with_executor(executor, 42);

        let (will_disconnect, mut token_rx) = request(16, 3);
        handle
            .submit(will_disconnect)
            .expect("submit will_disconnect");
        assert!(
            matches!(
                token_rx.blocking_recv(),
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
}
