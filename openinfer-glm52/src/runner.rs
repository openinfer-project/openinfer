use std::{
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Result, anyhow, bail, ensure};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use half::bf16;
use openinfer_core::engine::{FinishReason, GenerateRequest, TokenEvent, unix_now_s};
use tokio::sync::mpsc;

use crate::decode::{GLM52_DECODE_MAX_CTX, Glm52StageDecode};
use crate::model::Glm52StageModel;
use crate::weights::{
    Glm52NonExpertWeightContractReport, Glm52RankGpuContext, Glm52StageLoadBundle,
    load_stage_sliced_weights_to_gpu,
};

/// Input to one pipeline stage's forward: the first stage embeds a token id, the
/// rest receive the previous stage's hidden `[HIDDEN]` (staged through host).
pub(crate) enum Glm52StageInput {
    Token(u32),
    Hidden(Vec<bf16>),
}

/// Output of one stage's forward: a middle stage hands its hidden `[HIDDEN]` to
/// the next; the last stage (which owns the lm_head) returns the vocab logits.
pub(crate) enum Glm52StageOutput {
    Hidden(Vec<bf16>),
    Logits(Vec<f32>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StagePlacement {
    pub(crate) stage: usize,
    pub(crate) device_ordinal: usize,
}

impl Glm52StagePlacement {
    pub(crate) fn new(stage: usize, device_ordinal: usize) -> Result<Self> {
        ensure!(stage < 8, "GLM5.2 stage must be < 8, got {stage}");
        Ok(Self {
            stage,
            device_ordinal,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StageWeightLoadReport {
    pub(crate) stage: usize,
    pub(crate) tensor_count: usize,
    pub(crate) total_bytes: usize,
    pub(crate) non_expert_weight_contract: Glm52NonExpertWeightContractReport,
    pub(crate) loaded_to_gpu: bool,
}

/// One command per GPU-owning stage thread: load its sliced weights, run one
/// decode step over its layers, or shut down. The coordinator drives the eight
/// stages serially (bs=1) — `Forward` blocks until that stage's step completes.
enum Glm52StageCommand {
    LoadSlicedWeights {
        model_path: PathBuf,
        resp: Sender<Result<Glm52StageWeightLoadReport>>,
    },
    Forward {
        input: Glm52StageInput,
        position: usize,
        resp: Sender<Result<Glm52StageOutput>>,
    },
    Shutdown,
}

pub(crate) struct Glm52StageWorker {
    tx: Sender<Glm52StageCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Glm52StageWorker {
    pub(crate) fn spawn(
        placement: Glm52StagePlacement,
        bundle: Glm52StageLoadBundle,
    ) -> Result<Self> {
        ensure!(
            bundle.load_plan.stage == placement.stage,
            "GLM5.2 stage load plan {} does not match placement {}",
            bundle.load_plan.stage,
            placement.stage
        );
        let (tx, rx) = unbounded();
        let (startup_tx, startup_rx) = bounded::<Result<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("glm52-stage-{}", placement.stage))
            .spawn(move || {
                let ctx = match Glm52RankGpuContext::new(placement.device_ordinal) {
                    Ok(ctx) => ctx,
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                        return;
                    }
                };
                let _ = startup_tx.send(Ok(()));
                stage_worker_loop(rx, Glm52StageThreadState::new(placement, ctx, bundle));
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 stage worker: {err}"))?;
        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 stage worker exited during startup"))??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    pub(crate) fn load_sliced_weights_async(
        &self,
        model_path: &Path,
    ) -> Result<Receiver<Result<Glm52StageWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52StageCommand::LoadSlicedWeights {
                model_path: model_path.to_path_buf(),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 stage worker channel closed"))?;
        Ok(resp_rx)
    }

    /// Run one decode step on this stage and block for its result. The
    /// coordinator chains the eight stages this way for one bs=1 token.
    pub(crate) fn forward(
        &self,
        input: Glm52StageInput,
        position: usize,
    ) -> Result<Glm52StageOutput> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52StageCommand::Forward {
                input,
                position,
                resp: resp_tx,
            })
            .map_err(|_| anyhow!("GLM5.2 stage worker channel closed"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow!("GLM5.2 stage worker dropped forward response"))?
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.request_shutdown()?;
        self.join()
    }

    fn request_shutdown(&self) -> Result<()> {
        self.tx
            .send(Glm52StageCommand::Shutdown)
            .map_err(|_| anyhow::anyhow!("GLM5.2 stage worker channel closed"))?;
        Ok(())
    }

    fn join(&mut self) -> Result<()> {
        if self.handle.is_none() {
            return Ok(());
        }
        let handle = self.handle.take().expect("GLM5.2 stage handle must exist");
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("GLM5.2 stage worker panicked"))?;
        Ok(())
    }
}

impl Drop for Glm52StageWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

struct Glm52StageThreadState {
    placement: Glm52StagePlacement,
    ctx: Glm52RankGpuContext,
    bundle: Glm52StageLoadBundle,
    loaded: Option<Glm52StageLoadedState>,
}

struct Glm52StageLoadedState {
    decode: Glm52StageDecode,
}

impl Glm52StageThreadState {
    fn new(
        placement: Glm52StagePlacement,
        ctx: Glm52RankGpuContext,
        bundle: Glm52StageLoadBundle,
    ) -> Self {
        Self {
            placement,
            ctx,
            bundle,
            loaded: None,
        }
    }

    fn load_sliced_weights(&mut self, model_path: &Path) -> Result<Glm52StageWeightLoadReport> {
        let loaded = load_stage_sliced_weights_to_gpu(&self.ctx, model_path, &self.bundle)?;
        let total_bytes = loaded.weights.total_bytes + loaded.expert_kernel_weights.total_bytes;
        ensure!(
            loaded.loaded_total_bytes == total_bytes,
            "GLM5.2 stage {} loaded bytes {} differ from resident raw {} + expert package {}",
            self.placement.stage,
            loaded.loaded_total_bytes,
            loaded.weights.total_bytes,
            loaded.expert_kernel_weights.total_bytes
        );
        let report = Glm52StageWeightLoadReport {
            stage: self.placement.stage,
            tensor_count: loaded.loaded_tensor_count,
            total_bytes,
            non_expert_weight_contract: loaded.non_expert_weight_contract,
            loaded_to_gpu: true,
        };
        // Drain the raw loader output into the typed decode model (strict: every
        // resident tensor + expert package must be consumed), then build the
        // decode runtime (per-layer KV caches + rotary tables).
        let ctx = self.ctx.as_device_context();
        let model = Glm52StageModel::build(
            &ctx,
            loaded.weights,
            loaded.expert_kernel_weights,
            &self.bundle.names,
        )?;
        let decode = Glm52StageDecode::new(&ctx, model)?;
        self.loaded = Some(Glm52StageLoadedState { decode });
        Ok(report)
    }

    fn forward(&mut self, input: Glm52StageInput, position: usize) -> Result<Glm52StageOutput> {
        let ctx = self.ctx.as_device_context();
        let decode = &mut self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow!("GLM5.2 stage {} forward before load", self.placement.stage))?
            .decode;
        // Position 0 is the first token of a fresh request: clear the KV caches.
        if position == 0 {
            decode.reset(&ctx)?;
        }
        let hidden_in = match input {
            Glm52StageInput::Token(token) => decode.embed(&ctx, token)?,
            Glm52StageInput::Hidden(hidden) => hidden,
        };
        if decode.owns_head() {
            Ok(Glm52StageOutput::Logits(
                decode.run_layers_and_head(&ctx, &hidden_in, position)?,
            ))
        } else {
            Ok(Glm52StageOutput::Hidden(
                decode.run_layers(&ctx, &hidden_in, position)?,
            ))
        }
    }
}

fn stage_worker_loop(rx: Receiver<Glm52StageCommand>, mut state: Glm52StageThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52StageCommand::LoadSlicedWeights { model_path, resp } => {
                let _ = resp.send(state.load_sliced_weights(&model_path));
            }
            Glm52StageCommand::Forward {
                input,
                position,
                resp,
            } => {
                let _ = resp.send(state.forward(input, position));
            }
            Glm52StageCommand::Shutdown => break,
        }
    }
}

/// Drive bs=1 decode across the eight pipeline stages. Each token runs the eight
/// stages serially (stage k's hidden feeds stage k+1, staged through host), the
/// last stage returns logits, and the coordinator greedily picks the next token.
pub(crate) fn run_pp_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    mut workers: Vec<Glm52StageWorker>,
    stop_token_ids: Vec<u32>,
) {
    while let Some(req) = submit_rx.blocking_recv() {
        send_scheduled(&req);
        if let Err(err) = run_request_decode(&workers, &req, &stop_token_ids) {
            log::error!("GLM5.2 decode failed: {err:?}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message: format!("GLM5.2 decode failed: {err:?}"),
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
    }
    if let Err(err) = shutdown_stage_workers(&mut workers) {
        log::error!("GLM5.2 stage worker shutdown failed: {err:?}");
    }
}

/// Greedy bs=1 decode for one request. Processes the prompt one position at a
/// time (decode-style prefill — reuses the validated single-token kernel) then
/// generates until a stop token, `max_tokens`, or the context cap.
fn run_request_decode(
    workers: &[Glm52StageWorker],
    req: &GenerateRequest,
    stop_token_ids: &[u32],
) -> Result<()> {
    ensure!(!workers.is_empty(), "GLM5.2 has no pipeline stages");
    let prompt_len = req.prompt_tokens.len();
    ensure!(prompt_len > 0, "GLM5.2 decode requires a non-empty prompt");
    ensure!(
        prompt_len < GLM52_DECODE_MAX_CTX,
        "GLM5.2 prompt length {prompt_len} exceeds decode context {GLM52_DECODE_MAX_CTX}"
    );

    let mut completion_tokens = 0usize;
    let mut last_token: Option<u32> = None;
    let mut position = 0usize;
    let finish_reason = loop {
        let token = if position < prompt_len {
            req.prompt_tokens[position]
        } else {
            last_token.expect("a generated token exists past the prompt")
        };
        let logits = pipeline_forward(workers, token, position)?;

        // Logits are produced at every position; only sample from the final
        // prompt token onward (the positions that predict new tokens).
        if position + 1 >= prompt_len {
            let next = argmax(&logits);
            completion_tokens += 1;
            let _ = req.token_tx.send(TokenEvent::Token {
                id: next,
                logprob: None,
            });
            last_token = Some(next);
            if stop_token_ids.contains(&next) {
                break FinishReason::Stop;
            }
            if completion_tokens >= req.max_tokens {
                break FinishReason::Length;
            }
        }
        position += 1;
        if position >= GLM52_DECODE_MAX_CTX {
            break FinishReason::Length;
        }
    };

    let _ = req.token_tx.send(TokenEvent::Finished {
        finish_reason,
        prompt_tokens: prompt_len,
        completion_tokens,
    });
    Ok(())
}

/// Run one token through all eight stages serially, returning the last stage's
/// vocab logits.
fn pipeline_forward(workers: &[Glm52StageWorker], token: u32, position: usize) -> Result<Vec<f32>> {
    let mut output = workers[0].forward(Glm52StageInput::Token(token), position)?;
    for worker in &workers[1..] {
        let hidden = match output {
            Glm52StageOutput::Hidden(hidden) => hidden,
            Glm52StageOutput::Logits(_) => {
                bail!("GLM5.2 non-final stage produced logits")
            }
        };
        output = worker.forward(Glm52StageInput::Hidden(hidden), position)?;
    }
    match output {
        Glm52StageOutput::Logits(logits) => Ok(logits),
        Glm52StageOutput::Hidden(_) => bail!("GLM5.2 final stage did not produce logits"),
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .expect("logits are non-empty")
}

fn send_scheduled(req: &GenerateRequest) {
    let scheduled_at = unix_now_s();
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s: req.queued_at_unix_s.unwrap_or(scheduled_at),
        scheduled_at_unix_s: scheduled_at,
        prompt_tokens: req.prompt_tokens.len(),
        cached_tokens: 0,
    });
}

fn shutdown_stage_workers(workers: &mut [Glm52StageWorker]) -> Result<()> {
    for worker in workers.iter() {
        worker.request_shutdown()?;
    }
    for worker in workers.iter_mut() {
        worker.join()?;
    }
    Ok(())
}
