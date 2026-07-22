//! Generation timing harness: the BenchModel trait, the timed-run loop,
//! the scheduler stream helper, and the per-model bench adapters.

use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::ensure;
use openinfer::sampler::SamplingParams;
use openinfer::scheduler::SchedulerHandle;
use openinfer::scheduler::SchedulerRequest;
use openinfer::scheduler::TokenEvent;
use openinfer::scheduler::TokenSink;
use rand::rngs::StdRng;

pub(crate) struct GenTimings {
    pub(crate) ttft: Duration,
    pub(crate) tbt: Vec<Duration>,
    pub(crate) total: Duration,
    pub(crate) emitted_tokens: usize,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) decode_tokens_for_rate: usize,
    pub(crate) decode_time_for_rate: Duration,
}

pub(crate) trait BenchModel {
    fn validate_concurrency(&self, concurrency: usize) -> Result<()> {
        ensure!(concurrency > 0, "--concurrency must be > 0");
        Ok(())
    }

    /// Scheduler handle for open-loop mixed-load benchmarking.
    fn scheduler_handle(&self) -> Option<SchedulerHandle> {
        None
    }

    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        rng: &mut StdRng,
    ) -> GenTimings;

    /// Run one request per prompt; the slice length is the concurrency. Each
    /// prompt is independent, so MoE models must be handed *distinct* prompts
    /// to exercise realistic expert routing (see `synthetic_random_prompt`).
    fn timed_generation_batch(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        rng: &mut StdRng,
    ) -> Vec<GenTimings> {
        prompts
            .iter()
            .map(|prompt| self.timed_generation(prompt, max_new_tokens, sampling, rng))
            .collect()
    }
}

pub(crate) fn run_timed<F>(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    mut generate: F,
) -> GenTimings
where
    F: FnMut(&[u32], usize, &mut dyn FnMut(u32) -> bool) -> Result<()>,
{
    let start = Instant::now();
    let mut first_at: Option<Instant> = None;
    let mut prev_at: Option<Instant> = None;
    let mut emitted_tokens = 0usize;
    let mut tbt = Vec::with_capacity(max_new_tokens.saturating_sub(1));
    let mut generated_tokens = Vec::with_capacity(max_new_tokens);

    generate(prompt_tokens, max_new_tokens, &mut |tok| {
        let now = Instant::now();
        emitted_tokens += 1;
        generated_tokens.push(tok);
        if first_at.is_none() {
            first_at = Some(now);
        } else if let Some(prev) = prev_at {
            tbt.push(now - prev);
        }
        prev_at = Some(now);
        true
    })
    .expect("generation failed");

    let total = start.elapsed();
    let ttft = first_at.map_or(total, |t| t - start);
    let decode_tokens_for_rate = emitted_tokens.saturating_sub(1);
    let decode_time_for_rate = tbt.iter().copied().sum();
    GenTimings {
        ttft,
        tbt,
        total,
        emitted_tokens,
        generated_tokens,
        decode_tokens_for_rate,
        decode_time_for_rate,
    }
}

/// Outcome of draining a scheduler request's token stream.
pub(crate) struct StreamOutcome {
    /// True if the stream ended on `TokenEvent::Finished`; false if `on_token`
    /// requested an early stop by returning false.
    pub(crate) finished: bool,
}

/// Submit a single request to the scheduler and drain its token stream,
/// invoking `on_token` for each generated token id. Returns when the request
/// finishes, `on_token` returns false (early stop), or an error/closed event
/// arrives. Owns its args and borrows the handle so it composes inside a
/// `thread::spawn(move)` worker with a cloned `SchedulerHandle`.
pub(crate) fn run_scheduler_stream(
    handle: &SchedulerHandle,
    request_id: Option<String>,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    max_tokens: usize,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<StreamOutcome> {
    let (token_tx, mut token_rx) = TokenSink::standalone();
    handle
        .submit(SchedulerRequest {
            trace_parent: None,
            request_id,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params,
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .map_err(|e| anyhow::anyhow!("scheduler submit failed: {e}"))?;

    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, .. }) => {
                if !on_token(id) {
                    return Ok(StreamOutcome { finished: false });
                }
            }
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => return Ok(StreamOutcome { finished: true }),
            Some(TokenEvent::Error { message, .. }) => {
                anyhow::bail!("scheduler request failed: {message}");
            }
            Some(TokenEvent::Rejected { message, .. }) => {
                anyhow::bail!("scheduler request rejected: {message}");
            }
            None => anyhow::bail!("scheduler channel closed"),
        }
    }
}

pub(crate) struct SchedulerBenchModel {
    pub(crate) handle: SchedulerHandle,
}

impl BenchModel for SchedulerBenchModel {
    fn scheduler_handle(&self) -> Option<SchedulerHandle> {
        Some(self.handle.clone())
    }

    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> GenTimings {
        run_timed(prompt_tokens, max_new_tokens, |toks, n, cb| {
            run_scheduler_stream(&self.handle, None, toks.to_vec(), *sampling, n, &mut *cb)?;
            Ok(())
        })
    }

    fn timed_generation_batch(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> Vec<GenTimings> {
        let mut workers = Vec::with_capacity(prompts.len());
        for (idx, prompt) in prompts.iter().enumerate() {
            let handle = self.handle.clone();
            let prompt_tokens = prompt.clone();
            let sampling = *sampling;
            workers.push(thread::spawn(move || {
                run_timed(&prompt_tokens, max_new_tokens, |toks, n, cb| {
                    run_scheduler_stream(
                        &handle,
                        Some(format!("bench-serving-{idx}")),
                        toks.to_vec(),
                        sampling,
                        n,
                        &mut *cb,
                    )?;
                    Ok(())
                })
            }));
        }

        workers
            .into_iter()
            .map(|worker| worker.join().expect("bench request worker panicked"))
            .collect()
    }
}

#[cfg(feature = "deepseek-v2-lite")]
pub(crate) struct DeepSeekV2LiteBenchModel {
    pub(crate) generator: openinfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator,
}

#[cfg(feature = "deepseek-v2-lite")]
impl BenchModel for DeepSeekV2LiteBenchModel {
    fn validate_concurrency(&self, concurrency: usize) -> Result<()> {
        ensure!(
            concurrency > 0 && concurrency <= 8,
            "DeepSeek-V2-Lite direct benchmark supports --concurrency 1..=8; concurrency=1 is the single-row control and >1 uses the narrow same-prompt batched decode path, got {concurrency}"
        );
        Ok(())
    }

    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> GenTimings {
        assert_dsv2_lite_sampling_contract(sampling);
        let (result, attribution) = self
            .generator
            .generate_greedy_with_attribution(prompt_tokens, max_new_tokens, sampling.ignore_eos)
            .expect("DeepSeek-V2-Lite generation failed");
        timings_from_dsv2_lite_attribution(
            result.tokens,
            max_new_tokens,
            attribution.total_generation_us(),
            attribution.prefill_next_token_us(),
            attribution.per_token_decode_us(),
        )
    }

    fn timed_generation_batch(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> Vec<GenTimings> {
        assert_dsv2_lite_sampling_contract(sampling);
        if prompts.len() == 1 {
            return vec![self.timed_generation(&prompts[0], max_new_tokens, sampling, _rng)];
        }

        // This generator drives a narrow same-prompt batched decode kernel:
        // every row shares `prompts[0]`. Distinct per-request prompts are a
        // scheduler-path concern; this microbench takes one prompt by design.
        let result = self
            .generator
            .generate_greedy_batch_same_prompt_with_timings(
                &prompts[0],
                prompts.len(),
                max_new_tokens,
                sampling.ignore_eos,
            )
            .expect("DeepSeek-V2-Lite batched generation failed");
        timings_from_dsv2_lite_batched_generation(result, max_new_tokens)
    }
}

#[cfg(feature = "deepseek-v2-lite")]
pub(crate) fn assert_dsv2_lite_sampling_contract(sampling: &SamplingParams) {
    assert!(
        sampling.ignore_eos,
        "DeepSeek-V2-Lite direct attribution benchmark requires ignore_eos=true so output_len maps to an exact generated-token count"
    );
    assert!(
        (sampling.temperature <= 0.0 || sampling.top_k == 1) && sampling.top_p >= 1.0,
        "DeepSeek-V2-Lite direct attribution benchmark supports greedy decoding only; requested temperature={}, top_k={}, top_p={}",
        sampling.temperature,
        sampling.top_k,
        sampling.top_p
    );
}

#[cfg(feature = "deepseek-v2-lite")]
pub(crate) fn timings_from_dsv2_lite_attribution(
    generated_token_ids: Vec<u32>,
    expected_generated_tokens: usize,
    total_generation_us: u64,
    prefill_next_token_us: Option<u64>,
    per_token_decode_us: &[u64],
) -> GenTimings {
    // This bench helper intentionally panics on corrupted attribution data rather
    // than synthesizing a result. The surrounding trait does not carry errors,
    // and emitting bogus TPOT would be worse than aborting the benchmark.
    let emitted_tokens = generated_token_ids.len();
    assert_eq!(
        emitted_tokens, expected_generated_tokens,
        "DeepSeek-V2-Lite generated token count mismatch: got {} tokens for requested output_len={}",
        emitted_tokens, expected_generated_tokens
    );
    let expected_decode_steps = expected_generated_tokens.saturating_sub(1);
    assert_eq!(
        per_token_decode_us.len(),
        expected_decode_steps,
        "DeepSeek-V2-Lite timing count mismatch: got {} decode samples for {} generated tokens",
        per_token_decode_us.len(),
        emitted_tokens
    );
    assert!(
        total_generation_us > 0,
        "DeepSeek-V2-Lite total generation timing is zero; refusing to report TPOT"
    );
    if emitted_tokens > 0 {
        assert!(
            prefill_next_token_us.is_some_and(|us| us > 0),
            "DeepSeek-V2-Lite TTFT timing is missing or zero; refusing to report TPOT"
        );
    }
    if expected_decode_steps > 0 {
        assert!(
            per_token_decode_us.iter().all(|us| *us > 0),
            "DeepSeek-V2-Lite decode timing contains a zero-duration sample; refusing to report TPOT"
        );
    }
    let tbt: Vec<_> = per_token_decode_us
        .iter()
        .map(|us| Duration::from_micros(*us))
        .collect();
    let decode_time_for_rate = tbt.iter().copied().sum();
    GenTimings {
        ttft: Duration::from_micros(prefill_next_token_us.unwrap_or(total_generation_us)),
        tbt,
        total: Duration::from_micros(total_generation_us),
        emitted_tokens,
        generated_tokens: generated_token_ids,
        decode_tokens_for_rate: emitted_tokens.saturating_sub(1),
        decode_time_for_rate,
    }
}

#[cfg(feature = "deepseek-v2-lite")]
pub(crate) fn timings_from_dsv2_lite_batched_generation(
    result: openinfer_deepseek_v2_lite::BatchedGenerationResult,
    expected_generated_tokens: usize,
) -> Vec<GenTimings> {
    let batch_size = result.tokens.len();
    assert!(
        batch_size > 0,
        "DeepSeek-V2-Lite batch result must contain at least one row"
    );
    assert_eq!(
        result.prefill_next_token_us.len(),
        batch_size,
        "DeepSeek-V2-Lite batch result TTFT count mismatch"
    );
    assert!(
        result.total_generation_us > 0,
        "DeepSeek-V2-Lite batch total generation timing is zero; refusing to report TPOT"
    );
    assert!(
        result.prefill_next_token_us.iter().all(|us| *us > 0),
        "DeepSeek-V2-Lite batch TTFT timing contains a zero-duration sample; refusing to report TPOT"
    );
    let expected_decode_steps = expected_generated_tokens.saturating_sub(1);
    assert_eq!(
        result.per_token_decode_us.len(),
        expected_decode_steps,
        "DeepSeek-V2-Lite batch timing count mismatch: got {} decode samples for {} generated tokens",
        result.per_token_decode_us.len(),
        expected_generated_tokens
    );
    if expected_decode_steps > 0 {
        assert!(
            result.per_token_decode_us.iter().all(|us| *us > 0),
            "DeepSeek-V2-Lite batch decode timing contains a zero-duration sample; refusing to report TPOT"
        );
    }

    let tbt: Vec<_> = result
        .per_token_decode_us
        .iter()
        .map(|us| Duration::from_micros(*us))
        .collect();
    let decode_time_for_rate: Duration = tbt.iter().copied().sum();
    let decode_tokens_for_rate = batch_size * expected_decode_steps;

    result
        .tokens
        .into_iter()
        .zip(result.prefill_next_token_us)
        .enumerate()
        .map(|(idx, (generated_token_ids, prefill_us))| {
            let emitted_tokens = generated_token_ids.len();
            assert_eq!(
                emitted_tokens, expected_generated_tokens,
                "DeepSeek-V2-Lite batch row {idx} generated token count mismatch: got {} tokens for requested output_len={}",
                emitted_tokens, expected_generated_tokens
            );
            GenTimings {
                ttft: Duration::from_micros(prefill_us),
                tbt: tbt.clone(),
                total: Duration::from_micros(result.total_generation_us),
                emitted_tokens,
                generated_tokens: generated_token_ids,
                decode_tokens_for_rate: if idx == 0 { decode_tokens_for_rate } else { 0 },
                decode_time_for_rate: if idx == 0 {
                    decode_time_for_rate
                } else {
                    Duration::ZERO
                },
            }
        })
        .collect()
}
