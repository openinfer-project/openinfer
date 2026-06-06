use super::*;

struct FakeActive {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_len: usize,
    generated_count: usize,
    max_tokens: usize,
}

struct FakeSchedulerDriver {
    max_batch: usize,
    page_size: usize,
    available_pages: usize,
    max_request_pages: usize,
    fail_next_execution: Option<&'static str>,
}

impl FakeSchedulerDriver {
    fn new(
        max_batch: usize,
        page_size: usize,
        available_pages: usize,
        max_request_pages: usize,
    ) -> Self {
        Self {
            max_batch,
            page_size,
            available_pages,
            max_request_pages,
            fail_next_execution: None,
        }
    }

    fn with_next_execution_failure(mut self, message: &'static str) -> Self {
        self.fail_next_execution = Some(message);
        self
    }
}

impl SchedulerDriver for FakeSchedulerDriver {
    type Active = FakeActive;

    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn available_pages(&self) -> usize {
        self.available_pages
    }

    fn max_request_pages(&self) -> usize {
        self.max_request_pages
    }

    fn active_budget(&self, req: &Self::Active) -> ActiveKvBudget {
        ActiveKvBudget {
            prompt_len: req.prompt_len,
            generated_count: req.generated_count,
            max_tokens: req.max_tokens,
        }
    }

    fn execute_plan(
        &mut self,
        active: &mut Vec<Self::Active>,
        plan: ExecutionPlan<SchedulerRequest>,
        _rng: &mut StdRng,
    ) {
        if let Some(message) = self.fail_next_execution.take() {
            fail_fake_plan(active, plan, message);
            return;
        }

        match plan {
            ExecutionPlan::Prefill { pending } => {
                for req in pending {
                    fake_prefill_success(active, req);
                }
            }
            ExecutionPlan::Unified { pending } => {
                fake_decode_success(active);
                for req in pending {
                    fake_prefill_success(active, req);
                }
            }
            ExecutionPlan::Decode => fake_decode_success(active),
        }
    }
}

fn fail_fake_plan(
    active: &mut Vec<FakeActive>,
    plan: ExecutionPlan<SchedulerRequest>,
    message: &str,
) {
    match plan {
        ExecutionPlan::Prefill { pending } | ExecutionPlan::Unified { pending } => {
            for req in pending {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.to_string(),
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
        }
        ExecutionPlan::Decode => {}
    }
    for req in active.drain(..) {
        let _ = req.token_tx.send(TokenEvent::Error {
            message: message.to_string(),
            prompt_tokens: req.prompt_len,
            completion_tokens: req.generated_count,
        });
    }
}

fn fake_prefill_success(active: &mut Vec<FakeActive>, req: SchedulerRequest) {
    let prompt_len = req.prompt_tokens.len();
    if req
        .token_tx
        .send(TokenEvent::Token {
            id: 11,
            logprob: None,
        })
        .is_err()
    {
        return;
    }

    if req.max_tokens <= 1 {
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: prompt_len,
            completion_tokens: 1,
        });
        return;
    }

    active.push(FakeActive {
        token_tx: req.token_tx,
        prompt_len,
        generated_count: 1,
        max_tokens: req.max_tokens,
    });
}

fn fake_decode_success(active: &mut Vec<FakeActive>) {
    for mut req in active.drain(..) {
        req.generated_count += 1;
        let _ = req.token_tx.send(TokenEvent::Token {
            id: 12,
            logprob: None,
        });
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: req.prompt_len,
            completion_tokens: req.generated_count,
        });
    }
}

fn start_fake_scheduler(driver: FakeSchedulerDriver) -> SchedulerHandle {
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35-fake".into())
        .spawn(move || scheduler_loop(driver, submit_rx, 0))
        .expect("spawn fake scheduler");
    SchedulerHandle::new_with_join_handle(submit_tx, join_handle)
}

fn fake_request(
    prompt_len: usize,
    max_tokens: usize,
) -> (SchedulerRequest, mpsc::UnboundedReceiver<TokenEvent>) {
    let (token_tx, token_rx) = mpsc::unbounded_channel();
    (
        SchedulerRequest {
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

fn recv_event(rx: &mut mpsc::UnboundedReceiver<TokenEvent>) -> TokenEvent {
    rx.blocking_recv().expect("token event")
}

#[test]
fn send_rejection_reports_kv_lifetime_context() {
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    let req = SchedulerRequest {
        request_id: Some("too-large".to_string()),
        queued_at_unix_s: None,
        prompt_tokens: vec![1; 16],
        params: SamplingParams::default(),
        max_tokens: 65,
        lora_adapter: None,
        token_tx,
        logprobs: 0,
        echo: false,
    };

    send_rejection(&req);

    match token_rx.blocking_recv() {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 0);
            assert!(
                message.contains("max_context_tokens=80"),
                "rejection should report the full lifetime KV context"
            );
        }
        _ => panic!("expected rejection event"),
    }
}

#[test]
fn scheduler_loop_rejects_impossible_request_without_blocking_later_fit() {
    let handle = start_fake_scheduler(FakeSchedulerDriver::new(1, 16, 4, 4));
    let (too_large, mut too_large_rx) = fake_request(16, 65);
    let (fits, mut fits_rx) = fake_request(16, 1);

    handle.submit(too_large).expect("submit too-large request");
    handle.submit(fits).expect("submit fitting request");

    match recv_event(&mut too_large_rx) {
        TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        } => {
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 0);
            assert!(message.contains("max_context_tokens=80"));
        }
        _ => panic!("expected rejection for impossible request"),
    }

    assert!(matches!(
        recv_event(&mut fits_rx),
        TokenEvent::Token { id: 11, .. }
    ));
    assert!(matches!(
        recv_event(&mut fits_rx),
        TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: 16,
            completion_tokens: 1,
        }
    ));
    drop(handle);
}

#[test]
fn scheduler_loop_reports_execution_error_and_accepts_next_request() {
    let handle = start_fake_scheduler(
        FakeSchedulerDriver::new(1, 16, 4, 4).with_next_execution_failure("fake prefill failed"),
    );
    let (first, mut first_rx) = fake_request(16, 1);
    let (second, mut second_rx) = fake_request(16, 1);

    handle.submit(first).expect("submit first request");
    match recv_event(&mut first_rx) {
        TokenEvent::Error {
            message,
            prompt_tokens,
            completion_tokens,
        } => {
            assert_eq!(message, "fake prefill failed");
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 0);
        }
        _ => panic!("expected execution error"),
    }

    handle.submit(second).expect("submit second request");
    assert!(matches!(
        recv_event(&mut second_rx),
        TokenEvent::Token { id: 11, .. }
    ));
    assert!(matches!(
        recv_event(&mut second_rx),
        TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: 16,
            completion_tokens: 1,
        }
    ));
    drop(handle);
}
