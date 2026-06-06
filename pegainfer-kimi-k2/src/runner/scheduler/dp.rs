use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use log::error;
use pegainfer_core::engine::{FinishReason, GenerateRequest, TokenEvent};
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::runner::{
    executor::{DP_MAX_BATCH_PER_RANK, ForwardExecutor},
    load_balancer::DpLoadBalancer,
    worker::{KimiOneTokenForwardReport, KimiRowOptions},
};

use super::lifecycle::{preflight_prefill_candidate, send_scheduled};
use super::row_options;

/// Stable per-rank decode arena capacity. Logical slot IDs are arena rows, so
/// every TP1 DP8 decode/prefill command must keep this capacity stable.
const MAX_BATCH_PER_DP: usize = DP_MAX_BATCH_PER_RANK;

/// Coordinated DP engine: one coordinator thread drives all DP ranks in
/// lock-step. Every decode step, ALL ranks execute forward simultaneously
/// (active ranks with real tokens, idle ranks with padding). This satisfies
/// the PPLX EP contract that requires all ranks to participate in every
/// MoE layer's dispatch/combine collective.
pub(in crate::runner) struct DpCoordinator {
    dp_world: usize,
    ranks: Vec<DpRankState>,
    executors: Vec<Box<dyn ForwardExecutor + Send>>,
    step_txs: Vec<Sender<StepCommand>>,
    result_rxs: Vec<Receiver<StepResult>>,
    stop_token_ids: Vec<u32>,
    /// Drives non-greedy sampling: every step command carries a fresh philox
    /// seed per rank (rows within a rank decorrelate through the philox
    /// subsequence; ranks must not share a seed or same-index rows across
    /// ranks would draw the same uniform).
    rng: StdRng,
}

pub(in crate::runner) struct DpRankState {
    slots: Vec<Option<RequestState>>,
}

struct RequestState {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    prompt_len: usize,
    completion_tokens: usize,
    max_tokens: usize,
    last_token: u32,
    options: KimiRowOptions,
}

struct DecodeAdmission {
    slot: usize,
    req: GenerateRequest,
}

#[derive(Clone, Copy)]
struct DecodeInput {
    token_id: u32,
    append_position: usize,
    slot: usize,
    options: KimiRowOptions,
}

enum DecodeBatchRow {
    Active(DecodeInput),
    Admission(DecodeAdmission),
}

impl DecodeBatchRow {
    fn token_id(&self) -> u32 {
        match self {
            Self::Active(input) => input.token_id,
            Self::Admission(admission) => admission.req.prompt_tokens[0],
        }
    }

    fn append_position(&self) -> usize {
        match self {
            Self::Active(input) => input.append_position,
            Self::Admission(_) => 0,
        }
    }

    fn slot(&self) -> usize {
        match self {
            Self::Active(input) => input.slot,
            Self::Admission(admission) => admission.slot,
        }
    }

    fn options(&self) -> KimiRowOptions {
        match self {
            Self::Active(input) => input.options,
            Self::Admission(admission) => row_options(&admission.req),
        }
    }
}

enum StepCommand {
    Decode {
        token_ids: Vec<u32>,
        positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
        rows: Vec<KimiRowOptions>,
        seed: u64,
    },
    Prefill {
        input_ids: Vec<u32>,
        slot: usize,
        decode_batch_size: usize,
        ep_max_seq_len: usize,
        row: KimiRowOptions,
        seed: u64,
    },
    Shutdown,
}

enum StepResult {
    Decode(Result<Vec<KimiOneTokenForwardReport>>),
    Prefill(Result<KimiOneTokenForwardReport>),
}

impl DpCoordinator {
    pub(in crate::runner) fn new(
        executors: Vec<Box<dyn ForwardExecutor + Send>>,
        stop_token_ids: Vec<u32>,
        seed: u64,
    ) -> Self {
        let dp_world = executors.len();
        let mut ranks = Vec::with_capacity(dp_world);
        for _ in 0..dp_world {
            ranks.push(DpRankState {
                slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
            });
        }

        Self {
            dp_world,
            ranks,
            executors,
            step_txs: Vec::new(),
            result_rxs: Vec::new(),
            stop_token_ids,
            rng: rand::SeedableRng::seed_from_u64(seed),
        }
    }

    fn next_step_seed(&mut self) -> u64 {
        rand::RngExt::random(&mut self.rng)
    }

    /// Spawn per-rank forward threads and run the coordinated decode loop.
    /// This consumes self and blocks until shutdown.
    pub(in crate::runner) fn run(
        mut self,
        mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
        lb: DpLoadBalancer,
    ) {
        let mut step_txs = Vec::with_capacity(self.dp_world);
        let mut result_rxs = Vec::with_capacity(self.dp_world);
        let mut handles = Vec::with_capacity(self.dp_world);

        for (dp_rank, executor) in self.executors.drain(..).enumerate() {
            let (cmd_tx, cmd_rx) = bounded::<StepCommand>(1);
            let (res_tx, res_rx) = bounded::<StepResult>(1);
            step_txs.push(cmd_tx);
            result_rxs.push(res_rx);

            let handle = std::thread::Builder::new()
                .name(format!("kimi-k2-dp-fwd-{dp_rank}"))
                .spawn(move || {
                    rank_forward_loop(executor, cmd_rx, res_tx);
                })
                .expect("failed to spawn DP rank forward thread");
            handles.push(handle);
        }

        self.step_txs = step_txs;
        self.result_rxs = result_rxs;

        let mut pending_reqs: Vec<GenerateRequest> = Vec::new();

        loop {
            // 1. Drain new requests from submit channel
            if self.global_active_count() == 0 && pending_reqs.is_empty() {
                match submit_rx.blocking_recv() {
                    Some(req) => pending_reqs.push(req),
                    None => break,
                }
            }
            while let Ok(req) = submit_rx.try_recv() {
                pending_reqs.push(req);
            }

            // 2. Admit pending requests to DP ranks via load balancer
            self.admit_pending_requests(&mut pending_reqs, lb);

            // 3. Run one synchronized step across ALL ranks
            if self.global_active_count() > 0 {
                self.synchronized_decode_step();
            }
        }

        // Shutdown all forward threads
        for tx in &self.step_txs {
            let _ = tx.send(StepCommand::Shutdown);
        }
        for handle in handles {
            let _ = handle.join();
        }
    }

    fn global_active_count(&self) -> usize {
        self.ranks.iter().map(DpRankState::active_count).sum()
    }

    fn admit_pending_requests(
        &mut self,
        pending_reqs: &mut Vec<GenerateRequest>,
        lb: DpLoadBalancer,
    ) {
        let mut still_pending = Vec::new();
        let mut decode_admissions = self.empty_decode_admissions();
        let mut reserved_free_slots = self.free_slot_lists();

        for req in pending_reqs.drain(..) {
            let Some(req) = preflight_prefill_candidate(req) else {
                continue;
            };

            if req.prompt_tokens.len() == 1 {
                let Some(rank) = pick_rank_from_free_slots(&reserved_free_slots) else {
                    still_pending.push(req);
                    continue;
                };
                let slot = reserved_free_slots[rank].remove(0);
                send_scheduled(&req);
                decode_admissions[rank].push(DecodeAdmission { slot, req });
                continue;
            }

            self.flush_decode_admissions(&mut decode_admissions);
            reserved_free_slots = self.free_slot_lists();

            let dp_rank = lb.pick_rank(&self.ranks);
            match dp_rank {
                Some(rank) => {
                    let Some(slot) = self.ranks[rank].find_free_slot() else {
                        still_pending.push(req);
                        continue;
                    };
                    let Some(prefill_slots) = self.prefill_slots_for(rank, slot) else {
                        still_pending.push(req);
                        continue;
                    };
                    self.admit_request(rank, slot, &prefill_slots, req);
                    reserved_free_slots = self.free_slot_lists();
                }
                None => still_pending.push(req),
            }
        }

        self.flush_decode_admissions(&mut decode_admissions);
        *pending_reqs = still_pending;
    }

    fn empty_decode_admissions(&self) -> Vec<Vec<DecodeAdmission>> {
        (0..self.dp_world).map(|_| Vec::new()).collect()
    }

    fn free_slot_lists(&self) -> Vec<Vec<usize>> {
        self.ranks.iter().map(DpRankState::free_slots).collect()
    }

    fn prefill_slots_for(&self, owning_rank: usize, owning_slot: usize) -> Option<Vec<usize>> {
        let mut slots = Vec::with_capacity(self.dp_world);
        for dp_rank in 0..self.dp_world {
            if dp_rank == owning_rank {
                slots.push(owning_slot);
            } else {
                slots.push(self.ranks[dp_rank].find_free_slot()?);
            }
        }
        Some(slots)
    }

    fn flush_decode_admissions(&mut self, batch: &mut Vec<Vec<DecodeAdmission>>) {
        if batch.iter().all(Vec::is_empty) {
            return;
        }
        let ready = std::mem::replace(batch, self.empty_decode_admissions());
        self.synchronized_decode_admissions(ready);
    }

    fn admit_request(
        &mut self,
        dp_rank: usize,
        slot: usize,
        prefill_slots: &[usize],
        req: GenerateRequest,
    ) {
        send_scheduled(&req);

        // Prefill: all ranks run prefill in lock-step so PPLX collectives
        // align. Owning rank processes real tokens; padding ranks process a
        // single dummy token into a free slot (output discarded).
        self.synchronized_prefill(dp_rank, prefill_slots, &req);

        let prompt_len = req.prompt_tokens.len();

        let owner_result = self.result_rxs[dp_rank].recv();
        let mut padding_errors = Vec::new();
        for r in 0..self.dp_world {
            if r != dp_rank {
                match self.result_rxs[r].recv() {
                    Ok(StepResult::Prefill(Ok(_))) => {}
                    Ok(StepResult::Prefill(Err(err))) => {
                        padding_errors.push(format!(
                            "Kimi-K2 DP rank {r} padding prefill failed: {err:#}"
                        ));
                    }
                    Ok(StepResult::Decode(_)) => {
                        padding_errors.push(format!(
                            "Kimi-K2 DP rank {r} returned decode result during prefill"
                        ));
                    }
                    Err(_) => abort_dropped_result_channel(r, "prefill"),
                }
            }
        }

        let owner_report = match owner_result {
            Ok(StepResult::Prefill(Ok(report))) => report,
            Ok(StepResult::Prefill(Err(err))) => {
                error!("kimi-k2: DP rank {dp_rank} prefill failed: {err:#}");
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: format!("Kimi-K2 DP rank {dp_rank} prefill failed: {err:#}"),
                    prompt_tokens: prompt_len,
                    completion_tokens: 0,
                });
                return;
            }
            Ok(StepResult::Decode(_)) => {
                let message =
                    format!("Kimi-K2 DP rank {dp_rank} returned decode result during prefill");
                error!("kimi-k2: {message}");
                let _ = req.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: prompt_len,
                    completion_tokens: 0,
                });
                return;
            }
            Err(_) => abort_dropped_result_channel(dp_rank, "prefill"),
        };

        if !padding_errors.is_empty() {
            let message = padding_errors.join("; ");
            error!("kimi-k2: {message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            return;
        }

        let last_token = owner_report.local_next_token_global_id;
        if !req.params.ignore_eos && self.stop_token_ids.contains(&last_token) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            return;
        }
        if req
            .token_tx
            .send(TokenEvent::Token {
                id: last_token,
                logprob: owner_report.logprob,
            })
            .is_err()
        {
            return;
        }

        let completion_tokens = 1;
        if completion_tokens >= req.max_tokens {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens,
            });
            return;
        }

        let options = row_options(&req);
        self.ranks[dp_rank].slots[slot] = Some(RequestState {
            token_tx: req.token_tx,
            prompt_len,
            completion_tokens,
            max_tokens: req.max_tokens,
            last_token,
            options,
        });
    }

    fn synchronized_decode_admissions(&mut self, batch: Vec<Vec<DecodeAdmission>>) {
        let mut rows_by_rank = Vec::with_capacity(self.dp_world);
        for (dp_rank, rank_batch) in batch.into_iter().enumerate() {
            let seed = self.next_step_seed();
            let mut rows = self.ranks[dp_rank]
                .active_decode_inputs()
                .into_iter()
                .map(DecodeBatchRow::Active)
                .collect::<Vec<_>>();
            rows.extend(rank_batch.into_iter().map(DecodeBatchRow::Admission));
            if rows.len() > MAX_BATCH_PER_DP {
                let message = format!(
                    "Kimi-K2 DP rank {dp_rank} decode rows exceed arena capacity {MAX_BATCH_PER_DP}"
                );
                self.fail_decode_rows(dp_rank, rows, &message);
                send_step_command(
                    &self.step_txs[dp_rank],
                    dp_rank,
                    "decode admission overflow padding",
                    build_padding_decode_command(seed),
                );
                rows_by_rank.push(Vec::new());
                continue;
            }

            let cmd = if rows.is_empty() {
                build_padding_decode_command(seed)
            } else {
                build_decode_command_from_rows(&rows, seed)
            };
            send_step_command(&self.step_txs[dp_rank], dp_rank, "decode admission", cmd);
            rows_by_rank.push(rows);
        }

        for (dp_rank, rows) in rows_by_rank.into_iter().enumerate() {
            let result = match self.result_rxs[dp_rank].recv() {
                Ok(StepResult::Decode(result)) => result,
                Ok(StepResult::Prefill(_)) => {
                    let message = format!(
                        "Kimi-K2 DP rank {dp_rank} returned prefill result during decode admission"
                    );
                    error!("kimi-k2: {message}");
                    self.fail_decode_rows(dp_rank, rows, &message);
                    continue;
                }
                Err(_) => abort_dropped_result_channel(dp_rank, "decode admission"),
            };

            let reports = match result {
                Ok(reports) => reports,
                Err(err) if rows.is_empty() => {
                    error!(
                        "kimi-k2: fatal: DP rank {dp_rank} padding decode failed during decode admission: {err:#}"
                    );
                    std::process::abort();
                }
                Err(err) => {
                    error!("kimi-k2: DP rank {dp_rank} decode admission failed: {err:#}");
                    let message =
                        format!("Kimi-K2 DP rank {dp_rank} decode admission failed: {err:#}");
                    self.fail_decode_rows(dp_rank, rows, &message);
                    continue;
                }
            };

            if rows.is_empty() {
                continue;
            }

            for (row, report) in rows.into_iter().zip(reports) {
                match row {
                    DecodeBatchRow::Active(input) => {
                        self.ranks[dp_rank].process_decode_report(
                            input.slot,
                            &report,
                            &self.stop_token_ids,
                        );
                    }
                    DecodeBatchRow::Admission(admission) => {
                        self.install_decode_admission_result(dp_rank, admission, &report);
                    }
                }
            }
        }
    }

    fn install_decode_admission_result(
        &mut self,
        dp_rank: usize,
        admission: DecodeAdmission,
        report: &KimiOneTokenForwardReport,
    ) {
        let token_id = report.local_next_token_global_id;
        if !admission.req.params.ignore_eos && self.stop_token_ids.contains(&token_id) {
            let _ = admission.req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: admission.req.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return;
        }
        if admission
            .req
            .token_tx
            .send(TokenEvent::Token {
                id: token_id,
                logprob: report.logprob.clone(),
            })
            .is_err()
        {
            return;
        }

        let completion_tokens = 1;
        if completion_tokens >= admission.req.max_tokens {
            let _ = admission.req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: admission.req.prompt_tokens.len(),
                completion_tokens,
            });
            return;
        }

        let options = row_options(&admission.req);
        self.ranks[dp_rank].slots[admission.slot] = Some(RequestState {
            token_tx: admission.req.token_tx,
            prompt_len: admission.req.prompt_tokens.len(),
            completion_tokens,
            max_tokens: admission.req.max_tokens,
            last_token: token_id,
            options,
        });
    }

    fn fail_decode_rows(&mut self, dp_rank: usize, rows: Vec<DecodeBatchRow>, message: &str) {
        if rows
            .iter()
            .any(|row| matches!(row, DecodeBatchRow::Active(_)))
        {
            let err = anyhow::anyhow!(message.to_string());
            self.ranks[dp_rank].fail_all_active(&err);
        }

        for row in rows {
            let DecodeBatchRow::Admission(admission) = row else {
                continue;
            };
            let _ = admission.req.token_tx.send(TokenEvent::Error {
                message: message.to_string(),
                prompt_tokens: admission.req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
    }

    fn synchronized_prefill(
        &mut self,
        owning_rank: usize,
        prefill_slots: &[usize],
        req: &GenerateRequest,
    ) {
        let ep_max_seq_len = req.prompt_tokens.len();
        debug_assert_eq!(prefill_slots.len(), self.dp_world);
        for (dp_rank, slot) in prefill_slots
            .iter()
            .copied()
            .enumerate()
            .take(self.dp_world)
        {
            let seed = self.next_step_seed();
            let cmd = if dp_rank == owning_rank {
                StepCommand::Prefill {
                    input_ids: req.prompt_tokens.clone(),
                    slot,
                    decode_batch_size: MAX_BATCH_PER_DP,
                    ep_max_seq_len,
                    row: row_options(req),
                    seed,
                }
            } else {
                // All ranks run prefill so they traverse layers at the same
                // pace, making exactly 1 PPLX dispatch/combine per MoE layer.
                StepCommand::Prefill {
                    input_ids: vec![0],
                    slot,
                    decode_batch_size: MAX_BATCH_PER_DP,
                    ep_max_seq_len,
                    row: KimiRowOptions::default(),
                    seed,
                }
            };
            send_step_command(&self.step_txs[dp_rank], dp_rank, "prefill", cmd);
        }
    }

    fn synchronized_decode_step(&mut self) {
        // Build per-rank decode commands
        for dp_rank in 0..self.dp_world {
            let seed = self.next_step_seed();
            let cmd = self.ranks[dp_rank].build_decode_command(seed);
            send_step_command(&self.step_txs[dp_rank], dp_rank, "decode", cmd);
        }

        // Collect results from all ranks
        for dp_rank in 0..self.dp_world {
            let result = match self.result_rxs[dp_rank].recv() {
                Ok(StepResult::Decode(Ok(reports))) => reports,
                Ok(StepResult::Decode(Err(err))) => {
                    error!("kimi-k2: DP rank {dp_rank} decode failed: {err:#}");
                    self.ranks[dp_rank].fail_all_active(&err);
                    continue;
                }
                Ok(StepResult::Prefill(_)) => {
                    let err = anyhow::anyhow!(
                        "Kimi-K2 DP rank {dp_rank} returned prefill result during decode"
                    );
                    error!("kimi-k2: {err:#}");
                    self.ranks[dp_rank].fail_all_active(&err);
                    continue;
                }
                Err(_) => abort_dropped_result_channel(dp_rank, "decode"),
            };

            self.ranks[dp_rank].process_decode_results(result, &self.stop_token_ids);
        }
    }
}

impl DpRankState {
    fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub(in crate::runner) fn free_slot_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_none()).count()
    }

    pub(in crate::runner) fn has_free_slot(&self) -> bool {
        self.slots.iter().any(Option::is_none)
    }

    fn find_free_slot(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_none)
    }

    fn free_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.is_none().then_some(idx))
            .collect()
    }

    fn build_decode_command(&self, seed: u64) -> StepCommand {
        let inputs = self.active_decode_inputs();
        if inputs.is_empty() {
            return build_padding_decode_command(seed);
        }

        build_decode_command_from_inputs(&inputs, seed)
    }

    fn active_decode_inputs(&self) -> Vec<DecodeInput> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, state)| {
                state.as_ref().map(|req| DecodeInput {
                    token_id: req.last_token,
                    append_position: req.prompt_len + req.completion_tokens - 1,
                    slot,
                    options: req.options,
                })
            })
            .collect()
    }

    fn process_decode_results(
        &mut self,
        reports: Vec<KimiOneTokenForwardReport>,
        stop_token_ids: &[u32],
    ) {
        let active_slots: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| i))
            .collect();

        if active_slots.is_empty() {
            return;
        }

        for (slot_idx, report) in active_slots.into_iter().zip(reports) {
            self.process_decode_report(slot_idx, &report, stop_token_ids);
        }
    }

    fn process_decode_report(
        &mut self,
        slot_idx: usize,
        report: &KimiOneTokenForwardReport,
        stop_token_ids: &[u32],
    ) {
        let Some(req) = self.slots[slot_idx].as_mut() else {
            return;
        };

        let token_id = report.local_next_token_global_id;
        req.completion_tokens += 1;

        // EOS outranks the length limit; the stop token itself is not emitted
        // (same contract as the Qwen schedulers).
        if !req.options.sampling.ignore_eos && stop_token_ids.contains(&token_id) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.completion_tokens,
            });
            self.slots[slot_idx] = None;
            return;
        }

        if req
            .token_tx
            .send(TokenEvent::Token {
                id: token_id,
                logprob: report.logprob.clone(),
            })
            .is_err()
        {
            self.slots[slot_idx] = None;
            return;
        }

        if req.completion_tokens >= req.max_tokens {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.completion_tokens,
            });
            self.slots[slot_idx] = None;
        } else {
            req.last_token = token_id;
        }
    }

    fn fail_all_active(&mut self, err: &anyhow::Error) {
        let message = format!("{err:#}");
        for slot in &mut self.slots {
            if let Some(req) = slot.take() {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.completion_tokens,
                });
            }
        }
    }
}

/// Padding command for idle ranks: 1 dummy token so the rank participates
/// in EP collectives without producing real output.
fn build_padding_decode_command(seed: u64) -> StepCommand {
    StepCommand::Decode {
        token_ids: vec![0],
        positions: vec![0],
        slots: vec![0],
        decode_batch_size: MAX_BATCH_PER_DP,
        rows: vec![KimiRowOptions::default()],
        seed,
    }
}

fn build_decode_command_from_rows(rows: &[DecodeBatchRow], seed: u64) -> StepCommand {
    StepCommand::Decode {
        token_ids: rows.iter().map(DecodeBatchRow::token_id).collect(),
        positions: rows.iter().map(DecodeBatchRow::append_position).collect(),
        slots: rows.iter().map(DecodeBatchRow::slot).collect(),
        decode_batch_size: MAX_BATCH_PER_DP,
        rows: rows.iter().map(DecodeBatchRow::options).collect(),
        seed,
    }
}

fn build_decode_command_from_inputs(inputs: &[DecodeInput], seed: u64) -> StepCommand {
    StepCommand::Decode {
        token_ids: inputs.iter().map(|input| input.token_id).collect(),
        positions: inputs.iter().map(|input| input.append_position).collect(),
        slots: inputs.iter().map(|input| input.slot).collect(),
        decode_batch_size: MAX_BATCH_PER_DP,
        rows: inputs.iter().map(|input| input.options).collect(),
        seed,
    }
}

fn send_step_command(tx: &Sender<StepCommand>, dp_rank: usize, phase: &str, command: StepCommand) {
    if tx.send(command).is_err() {
        error!("kimi-k2: fatal: DP rank {dp_rank} forward thread dropped before {phase}");
        std::process::abort();
    }
}

fn abort_dropped_result_channel(dp_rank: usize, phase: &str) -> ! {
    error!("kimi-k2: fatal: DP rank {dp_rank} forward thread dropped during {phase}");
    std::process::abort();
}

fn rank_forward_loop(
    executor: Box<dyn ForwardExecutor + Send>,
    cmd_rx: Receiver<StepCommand>,
    res_tx: Sender<StepResult>,
) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            StepCommand::Decode {
                token_ids,
                positions,
                slots,
                decode_batch_size,
                rows,
                seed,
            } => {
                let result = executor.forward_decode_batch(
                    &token_ids,
                    &positions,
                    &slots,
                    decode_batch_size,
                    &rows,
                    seed,
                );
                let _ = res_tx.send(StepResult::Decode(result));
            }
            StepCommand::Prefill {
                input_ids,
                slot,
                decode_batch_size,
                ep_max_seq_len,
                row,
                seed,
            } => {
                let result = executor.forward_prefill(
                    &input_ids,
                    slot,
                    decode_batch_size,
                    ep_max_seq_len,
                    row,
                    seed,
                );
                let _ = res_tx.send(StepResult::Prefill(result));
            }
            StepCommand::Shutdown => break,
        }
    }
    drop(executor);
    drop(cmd_rx);
    drop(res_tx);
}

fn pick_rank_from_free_slots(free_slots: &[Vec<usize>]) -> Option<usize> {
    free_slots
        .iter()
        .enumerate()
        .filter(|(_, slots)| !slots.is_empty())
        .max_by_key(|(_, slots)| slots.len())
        .map(|(rank, _)| rank)
}

#[cfg(test)]
mod tests {
    use pegainfer_core::sampler::SamplingParams;

    use super::*;

    fn dummy_request(prompt_tokens: Vec<u32>, max_tokens: usize) -> GenerateRequest {
        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        }
    }

    fn dummy_state(prompt_len: usize, completion_tokens: usize, last_token: u32) -> RequestState {
        let (token_tx, _token_rx) = mpsc::unbounded_channel();
        RequestState {
            token_tx,
            prompt_len,
            completion_tokens,
            max_tokens: 16,
            last_token,
            options: KimiRowOptions::default(),
        }
    }

    fn dummy_report(token_id: u32) -> KimiOneTokenForwardReport {
        KimiOneTokenForwardReport {
            rank: 0,
            batch_slot: 0,
            input_token_id: 0,
            local_next_token_id: token_id,
            local_next_token_global_id: token_id,
            local_top_logit_f32: 0.0,
            vocab_start: 0,
            vocab_rows: 0,
            dense_layers_executed: 0,
            moe_layers_executed: 0,
            logprob: None,
        }
    }

    fn test_coordinator(dp_world: usize) -> DpCoordinator {
        DpCoordinator {
            dp_world,
            ranks: (0..dp_world)
                .map(|_| DpRankState {
                    slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
                })
                .collect(),
            executors: Vec::new(),
            step_txs: Vec::new(),
            result_rxs: Vec::new(),
            stop_token_ids: Vec::new(),
            rng: rand::SeedableRng::seed_from_u64(0),
        }
    }

    #[test]
    fn sparse_decode_slot_keeps_stable_arena_capacity() {
        let mut rank = DpRankState {
            slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
        };
        rank.slots[MAX_BATCH_PER_DP - 1] = Some(dummy_state(4, 3, 123));

        let StepCommand::Decode {
            token_ids,
            positions,
            slots,
            decode_batch_size,
            rows,
            seed,
        } = rank.build_decode_command(7)
        else {
            panic!("decode command expected");
        };

        assert_eq!(decode_batch_size, MAX_BATCH_PER_DP);
        assert_eq!(token_ids, vec![123]);
        assert_eq!(positions, vec![6]);
        assert_eq!(slots, vec![MAX_BATCH_PER_DP - 1]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].logprobs, 0);
        assert_eq!(seed, 7);
    }

    #[test]
    fn decode_rows_merge_active_decode_and_new_admission() {
        let mut sampling_req = dummy_request(vec![99], 8);
        sampling_req.params.temperature = 0.8;
        sampling_req.params.top_p = 0.9;
        let batch_rows = vec![
            DecodeBatchRow::Active(DecodeInput {
                token_id: 11,
                append_position: 5,
                slot: 3,
                options: KimiRowOptions {
                    logprobs: 4,
                    sampling: SamplingParams::default(),
                },
            }),
            DecodeBatchRow::Admission(DecodeAdmission {
                slot: 7,
                req: sampling_req,
            }),
        ];

        let StepCommand::Decode {
            token_ids,
            positions,
            slots,
            decode_batch_size,
            rows,
            seed,
        } = build_decode_command_from_rows(&batch_rows, 42)
        else {
            panic!("decode command expected");
        };

        assert_eq!(decode_batch_size, MAX_BATCH_PER_DP);
        assert_eq!(token_ids, vec![11, 99]);
        assert_eq!(positions, vec![5, 0]);
        assert_eq!(slots, vec![3, 7]);
        assert_eq!(rows[0].logprobs, 4);
        assert!(rows[0].sampling.is_greedy());
        assert_eq!(rows[1].logprobs, 0);
        assert!(!rows[1].sampling.is_greedy());
        assert!((rows[1].sampling.temperature - 0.8).abs() < f32::EPSILON);
        assert_eq!(seed, 42);
    }

    #[test]
    fn padding_decode_uses_stable_arena_capacity() {
        let StepCommand::Decode {
            token_ids,
            positions,
            slots,
            decode_batch_size,
            rows,
            seed: _,
        } = build_padding_decode_command(1)
        else {
            panic!("decode command expected");
        };

        assert_eq!(decode_batch_size, MAX_BATCH_PER_DP);
        assert_eq!(token_ids, vec![0]);
        assert_eq!(positions, vec![0]);
        assert_eq!(slots, vec![0]);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].sampling.is_greedy());
    }

    #[test]
    fn decode_report_finishes_with_stop_at_eos() {
        let mut rank = DpRankState {
            slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
        };
        let (token_tx, mut token_rx) = mpsc::unbounded_channel();
        rank.slots[0] = Some(RequestState {
            token_tx,
            prompt_len: 4,
            completion_tokens: 1,
            max_tokens: 16,
            last_token: 7,
            options: KimiRowOptions::default(),
        });

        rank.process_decode_report(0, &dummy_report(163_586), &[163_586]);

        assert!(rank.slots[0].is_none());
        let Ok(TokenEvent::Finished {
            finish_reason,
            completion_tokens,
            ..
        }) = token_rx.try_recv()
        else {
            panic!("expected Finished event");
        };
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(completion_tokens, 2);
        // The stop token itself is not emitted.
        assert!(token_rx.try_recv().is_err());
    }

    #[test]
    fn decode_report_honors_ignore_eos() {
        let mut rank = DpRankState {
            slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
        };
        let (token_tx, mut token_rx) = mpsc::unbounded_channel();
        rank.slots[0] = Some(RequestState {
            token_tx,
            prompt_len: 4,
            completion_tokens: 1,
            max_tokens: 16,
            last_token: 7,
            options: KimiRowOptions {
                logprobs: 0,
                sampling: SamplingParams {
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
            },
        });

        rank.process_decode_report(0, &dummy_report(163_586), &[163_586]);

        assert!(rank.slots[0].is_some());
        let Ok(TokenEvent::Token { id, .. }) = token_rx.try_recv() else {
            panic!("expected Token event");
        };
        assert_eq!(id, 163_586);
    }

    #[test]
    fn prefill_padding_slots_avoid_active_requests() {
        let mut coordinator = test_coordinator(2);
        coordinator.ranks[0].slots[0] = Some(dummy_state(4, 1, 10));

        let slots = coordinator
            .prefill_slots_for(1, 3)
            .expect("rank 0 has a free padding slot");

        assert_ne!(slots[0], 0);
        assert_eq!(slots[1], 3);
    }

    #[test]
    fn prefill_waits_when_any_padding_rank_is_full() {
        let mut coordinator = test_coordinator(2);
        for slot in 0..MAX_BATCH_PER_DP {
            coordinator.ranks[0].slots[slot] = Some(dummy_state(4, 1, slot as u32));
        }

        assert!(coordinator.prefill_slots_for(1, 3).is_none());
    }
}
