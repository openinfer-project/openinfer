use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use openinfer_core::engine::EngineControlError;
use openinfer_core::engine::LoadLoraAdapterRequest;
use openinfer_core::engine::UnloadLoraAdapterRequest;
use openinfer_kv_cache::BlockPool;

use super::*;
use crate::executor::DecodePlan;
use crate::executor::DecodeRequestResult;
use crate::executor::PrefillPlan;
use crate::executor::PrefillRequestResult;
use crate::executor::PrefillResult;
use crate::executor::PrefillStepItem;
use crate::executor::UnifiedPlan;
use crate::executor::UnifiedResult;

struct FakeExecutor {
    block_size: usize,
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
    stop_token: Option<u32>,
}

impl FakeExecutor {
    fn new(max_request_blocks: usize, dropped: Arc<Mutex<Vec<u64>>>) -> Self {
        Self {
            block_size: 16,
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
            stop_token: None,
        }
    }

    fn with_stop_token(mut self, token: u32) -> Self {
        self.stop_token = Some(token);
        self
    }

    fn with_decode_failure(mut self) -> Self {
        self.fail_decode_once = true;
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
            cached_tokens: 0,
            completed,
            prefill_pos,
        }
    }

    fn ensure_request_tokens(&mut self, request_id: RequestId, token_count: usize) -> Result<()> {
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

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token == Some(token_id)
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
        for req in plan.requests {
            self.ensure_request_tokens(req.request_id, req.prompt_tokens.len())?;
        }
        Ok(PrefillResult {
            requests: plan
                .requests
                .iter()
                .map(|req| self.fake_prefill_result(req))
                .collect(),
            dflash_context_captured_requests: Vec::new(),
        })
    }

    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<crate::executor::DecodeResult> {
        if !self.decode_delay.is_zero() {
            std::thread::sleep(self.decode_delay);
        }
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

    let (token_tx, _token_rx) = TokenSink::standalone();
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

    let (token_tx, _token_rx) = TokenSink::standalone();
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
    let (token_tx, _rx) = TokenSink::standalone();
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
        admit_deferred_requests(deferred, &active, &[], 16, 4, 4, usize::MAX, 64, 32, |_| 0);

    let ids = |reqs: &[PendingRequest]| reqs.iter().map(|r| r.request_id.get()).collect::<Vec<_>>();
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
        mk(3, 40, 1),  // request 3: 40 prompt + 1 max = 41 total: overflows by 9 tokens → rejected
    ];

    let outcome =
        admit_deferred_requests(deferred, &active, &[], 16, 1000, 1000, 32, 64, 64, |_| 0);

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
        let (token_tx, _rx) = TokenSink::standalone();
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
        32,
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
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
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
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
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
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
    assert_eq!(
        taken.iter().map(|r| r.step_chunk).collect::<Vec<_>>(),
        vec![16, 16],
        "remainder of the chunked head + the next request share the step"
    );
    assert!(prefilling.is_empty());
}

// TokenEvent does not expose per-step chunk assignments, so gate them here.
#[test]
fn request_local_chunks_are_independent_of_earlier_requests() {
    let mk = |id: u64, prompt_len| {
        PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, 1).0)
    };
    let simulate = |mut prefilling: Vec<PendingRequest>, request_local: bool| {
        let mut target_chunks = Vec::new();
        while prefilling.iter().any(|req| req.request_id == RequestId(3)) {
            let taken = take_prefill_chunks(&mut prefilling, 32, request_local);
            let mut continued = Vec::new();
            for mut req in taken {
                if req.request_id == RequestId(3) {
                    target_chunks.push(req.step_chunk);
                }
                req.prefill_pos += req.step_chunk;
                if req.remaining_prompt_tokens() > 0 {
                    continued.push(req);
                }
            }
            prefilling.splice(0..0, continued);
        }
        target_chunks
    };
    let alone = || vec![mk(3, 80)];
    let behind_others = || vec![mk(1, 24), mk(2, 16), mk(3, 80)];

    assert_eq!(simulate(alone(), true), vec![32, 32, 16]);
    assert_eq!(simulate(behind_others(), true), vec![32, 32, 16]);

    // Without request-local chunking the same request is cut differently once it queues behind
    // others — the drift this pins. Its absence would make the two asserts above vacuous.
    assert_eq!(simulate(alone(), false), vec![32, 32, 16]);
    assert_ne!(
        simulate(behind_others(), false),
        vec![32, 32, 16],
        "shared-budget chunking must still be batch-dependent, or the request-local asserts prove nothing"
    );
}

#[test]
fn echo_requests_run_only_when_their_prompt_fits_the_prefill_bound() {
    let mk_echo = |id: u64, prompt_len| {
        let (req, _rx) = request(prompt_len, 1);
        let mut pending = PendingRequest::from_scheduler_request(RequestId(id), req);
        pending.echo = true;
        pending
    };
    let mk = |id: u64, prompt_len| {
        PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, 1).0)
    };

    // Oversized echo is rejected by admission. If a caller bypasses
    // admission, the chunk picker must still keep it out of the profiled
    // prefill shape instead of running it whole.
    let mut prefilling = vec![mk_echo(1, 64), mk(2, 16)];
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
    assert_eq!(taken.len(), 1);
    assert_eq!(taken[0].request_id, RequestId(2));
    assert_eq!(taken[0].step_chunk, 16);
    assert_eq!(
        prefilling[0].request_id,
        RequestId(1),
        "oversized echo stays queued if admission was bypassed"
    );

    // An echo that doesn't fit behind earlier work is skipped, not split;
    // later requests may still fill the leftover budget, and the step set
    // stays sorted by request id.
    let mut prefilling = vec![mk(3, 24), mk_echo(4, 16), mk(5, 8)];
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
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
fn oversized_echo_request_is_rejected_at_admission() {
    let active: [ActiveRequestState; 0] = [];
    let mk_echo = |id: u64, prompt_len| {
        let (mut req, _rx) = request(prompt_len, 1);
        req.echo = true;
        PendingRequest::from_scheduler_request(RequestId(id), req)
    };
    let mk = |id: u64, prompt_len| {
        PendingRequest::from_scheduler_request(RequestId(id), request(prompt_len, 1).0)
    };

    let outcome = admit_deferred_requests(
        vec![mk_echo(1, 33), mk(2, 64)],
        &active,
        &[],
        16,
        1024,
        1024,
        usize::MAX,
        64,
        32,
        |_| 0,
    );

    assert_eq!(
        outcome
            .pending
            .iter()
            .map(|r| r.request_id.get())
            .collect::<Vec<_>>(),
        vec![2],
        "non-echo oversized prompts can still be admitted and chunked"
    );
    assert_eq!(outcome.rejected.len(), 1);
    assert_eq!(outcome.rejected[0].0.request_id, RequestId(1));
    assert!(
        matches!(
            outcome.rejected[0].1,
            RejectReason::EchoPrefillTokens { limit: 32 }
        ),
        "oversized echo should be rejected against the profiled prefill bound"
    );
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
        32,
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
        32,
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

/// Engine streams now open with `TokenEvent::Scheduled` (#246); these
/// tests assert on the token/terminal events, so skip past it.
fn recv_skipping_scheduled(
    rx: &mut openinfer_core::engine::TokenStreamReceiver,
) -> Option<TokenEvent> {
    loop {
        match rx.blocking_recv() {
            Some((_, TokenEvent::Scheduled { .. })) => {}
            Some((_, event)) => return Some(event),
            None => return None,
        }
    }
}

fn pending(request_id: u64, echo: bool) -> PendingRequest {
    let (token_tx, _token_rx) = TokenSink::standalone();
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
) -> (GenerateRequest, openinfer_core::engine::TokenStreamReceiver) {
    let (token_tx, token_rx) = TokenSink::standalone();
    (
        GenerateRequest {
            trace_parent: None,
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
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
) -> (GenerateRequest, openinfer_core::engine::TokenStreamReceiver) {
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
fn unknown_lora_request_is_rejected_without_blocking_base_request() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, Arc::clone(&dropped));
    let handle = start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

    let (unknown, mut unknown_rx) = request_with_lora(16, 1, Some("missing-adapter"));
    let (base, mut base_rx) = request_with_lora(16, 1, None);
    handle.submit(unknown).expect("submit unknown adapter");
    handle.submit(base).expect("submit base");

    match unknown_rx.blocking_recv() {
        Some((
            _,
            TokenEvent::Rejected {
                message,
                prompt_tokens,
                completion_tokens,
            },
        )) => {
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
fn retiring_multiple_active_requests_tolerates_unsorted_indices() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let mut executor = FakeExecutor::new(8, Arc::clone(&dropped));
    let mut active = Vec::new();

    for request_id in [RequestId(10), RequestId(1), RequestId(7)] {
        let (token_tx, _token_rx) = TokenSink::standalone();
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
        &mut super::phase_trace::PhaseTracker::default(),
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
fn lora_control_unloads_adapter_when_idle() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, Arc::clone(&dropped)).with_lora_adapters(&["adapter-a"]);
    let handle = start_with_executor_with_lora_control(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

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
    let handle = start_with_executor_with_lora_control(executor, 42, DEFAULT_MAX_PREFILL_TOKENS);

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

// ── Speculative span resolution (multi-token emission) ──────────────────────
// resolve_speculative_outputs walks an accepted span [t_0, .., t_m] and decides,
// per request, whether to emit-and-continue or truncate at a stop token / the
// max-output budget. These were GPU-test-only; the truth table below pins the
// branch behaviour with the existing FakeExecutor (only is_stop_token matters).

use openinfer_core::engine::FinishReason;

use crate::speculative::VerifyRequestResult;

fn spec_active(
    id: u64,
    generated_count: usize,
    max_tokens: usize,
    ignore_eos: bool,
) -> ActiveRequestState {
    let (token_tx, _rx) = TokenSink::standalone();
    ActiveRequestState {
        request_id: RequestId(id),
        lora_adapter: None,
        token_tx,
        last_token: 1,
        generated_count,
        max_tokens,
        prompt_len: 16,
        params: SamplingParams {
            ignore_eos,
            ..SamplingParams::default()
        },
        logprobs: 0,
    }
}

fn spec_result(id: u64, accepted: Vec<u32>) -> VerifyRequestResult {
    VerifyRequestResult {
        request_id: RequestId(id),
        matched_draft_tokens: accepted.len().saturating_sub(1),
        accepted_tokens: accepted,
    }
}

const SPEC_EOS: u32 = 99;

#[test]
fn speculative_full_span_accept_continues() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 3, 100, false)];
    let results = [spec_result(1, vec![10, 11, 12, 13])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndContinue {
                request_id,
                tokens,
                completion_tokens,
            },
        ] => {
            assert_eq!(*request_id, RequestId(1));
            assert_eq!(tokens, &vec![10, 11, 12, 13]);
            assert_eq!(
                *completion_tokens,
                3 + 4,
                "completion = prior generated + span len"
            );
        }
        _ => panic!("expected EmitManyAndContinue"),
    }
}

#[test]
fn speculative_stop_token_midspan_finishes_and_suppresses_eos() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 5, 100, false)];
    // EOS lands at span position 2; tokens before it are emitted, EOS is not.
    let results = [spec_result(1, vec![10, 11, SPEC_EOS, 13])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndFinish {
                tokens,
                finish_reason,
                completion_tokens,
                ..
            },
        ] => {
            assert_eq!(
                tokens,
                &vec![10, 11],
                "EOS itself is suppressed from emission"
            );
            assert!(matches!(finish_reason, FinishReason::Stop));
            assert_eq!(
                *completion_tokens,
                5 + 3,
                "the EOS still counts toward completion length"
            );
        }
        _ => panic!("expected EmitManyAndFinish(Stop)"),
    }
}

#[test]
fn speculative_stop_token_at_span_start_emits_nothing() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 5, 100, false)];
    let results = [spec_result(1, vec![SPEC_EOS, 11, 12])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndFinish {
                tokens,
                finish_reason,
                completion_tokens,
                ..
            },
        ] => {
            assert!(tokens.is_empty(), "stop at position 0 emits no tokens");
            assert!(matches!(finish_reason, FinishReason::Stop));
            assert_eq!(*completion_tokens, 5 + 1);
        }
        _ => panic!("expected EmitManyAndFinish(Stop)"),
    }
}

#[test]
fn speculative_max_tokens_truncates_midspan() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    // generated 8, budget 10 -> only 2 more tokens fit; the span offers 4.
    let active = [spec_active(1, 8, 10, false)];
    let results = [spec_result(1, vec![10, 11, 12, 13])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndFinish {
                tokens,
                finish_reason,
                completion_tokens,
                ..
            },
        ] => {
            assert_eq!(
                tokens,
                &vec![10, 11],
                "the budget-hitting token is emitted, the rest dropped"
            );
            assert!(matches!(finish_reason, FinishReason::Length));
            assert_eq!(
                *completion_tokens, 10,
                "completion stops exactly at max_tokens"
            );
        }
        _ => panic!("expected EmitManyAndFinish(Length)"),
    }
}

#[test]
fn speculative_ignore_eos_does_not_stop() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 0, 100, true)];
    let results = [spec_result(1, vec![SPEC_EOS, SPEC_EOS])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [effects::DecodeEffect::EmitManyAndContinue { tokens, .. }] => {
            assert_eq!(
                tokens,
                &vec![SPEC_EOS, SPEC_EOS],
                "ignore_eos passes stop tokens through"
            );
        }
        _ => panic!("expected EmitManyAndContinue"),
    }
}

#[test]
fn speculative_resolves_each_request_independently() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 0, 100, false), spec_active(2, 0, 100, false)];
    let results = [
        spec_result(1, vec![10, 11]),       // continues
        spec_result(2, vec![20, SPEC_EOS]), // finishes on EOS
    ];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    assert!(matches!(
        &effects[0],
        effects::DecodeEffect::EmitManyAndContinue { request_id, .. } if *request_id == RequestId(1)
    ));
    assert!(matches!(
        &effects[1],
        effects::DecodeEffect::EmitManyAndFinish { request_id, finish_reason: FinishReason::Stop, .. }
            if *request_id == RequestId(2)
    ));
}
