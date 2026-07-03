use std::{
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Result, ensure};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use openinfer_core::engine::{GenerateRequest, TokenEvent, unix_now_s};
use tokio::sync::mpsc;

#[cfg(feature = "glm52")]
use anyhow::Context as _;
#[cfg(feature = "glm52")]
use openinfer_core::engine::FinishReason;

#[cfg(feature = "glm52")]
use crate::model::{GLM52_MAX_MODEL_LEN, Glm52ExpertRankModel, Glm52Rank0Model};
#[cfg(feature = "glm52")]
use crate::moe_ep8::Glm52MoeEp8State;
use crate::weights::{
    GLM52_EP_RANKS, Glm52RankGpuContext, Glm52RankGpuWeights, Glm52RankLoadBundle,
    load_rank_weights_to_gpu,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankPlacement {
    pub(crate) rank: usize,
    pub(crate) device_ordinal: usize,
}

impl Glm52RankPlacement {
    pub(crate) fn new(rank: usize, device_ordinal: usize) -> Result<Self> {
        ensure!(
            rank < GLM52_EP_RANKS,
            "GLM5.2 rank must be < {GLM52_EP_RANKS}, got {rank}"
        );
        Ok(Self {
            rank,
            device_ordinal,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankWeightLoadReport {
    pub(crate) rank: usize,
    pub(crate) loaded_tensor_count: usize,
    pub(crate) loaded_total_bytes: usize,
    pub(crate) resident_raw_bytes: usize,
    pub(crate) loaded_to_gpu: bool,
}

enum Glm52RankCommand {
    LoadWeights {
        model_path: PathBuf,
        resp: Sender<Result<Glm52RankWeightLoadReport>>,
    },
    /// Collective: the coordinator issues this to every rank concurrently
    /// (DeepEP context creation barriers across ranks).
    #[cfg(feature = "glm52")]
    BuildModel {
        unique_id: Box<[u8; 128]>,
        resp: Sender<Result<()>>,
    },
    /// Rank 0 only: one full-model step (75 MoE collectives inside).
    #[cfg(feature = "glm52")]
    Rank0Step {
        token: u32,
        position: usize,
        resp: Sender<Result<u32>>,
    },
    /// Ranks 1..7 only: replay one step's 75 MoE collectives.
    #[cfg(feature = "glm52")]
    ExpertStep {
        resp: Sender<Result<()>>,
    },
    Shutdown,
}

pub(crate) struct Glm52RankWorker {
    tx: Sender<Glm52RankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Glm52RankWorker {
    pub(crate) fn spawn(
        placement: Glm52RankPlacement,
        bundle: Glm52RankLoadBundle,
    ) -> Result<Self> {
        ensure!(
            bundle.plan.rank == placement.rank,
            "GLM5.2 rank load plan {} does not match placement {}",
            bundle.plan.rank,
            placement.rank
        );
        let (tx, rx) = unbounded();
        let (startup_tx, startup_rx) = bounded::<Result<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("glm52-rank-{}", placement.rank))
            .spawn(move || {
                let ctx = match Glm52RankGpuContext::new(placement.device_ordinal) {
                    Ok(ctx) => ctx,
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                        return;
                    }
                };
                let _ = startup_tx.send(Ok(()));
                rank_worker_loop(rx, Glm52RankThreadState::new(placement, ctx, bundle));
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 rank worker: {err}"))?;
        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker exited during startup"))??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    pub(crate) fn load_weights_async(
        &self,
        model_path: &Path,
    ) -> Result<Receiver<Result<Glm52RankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::LoadWeights {
                model_path: model_path.to_path_buf(),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    #[cfg(feature = "glm52")]
    pub(crate) fn build_model_async(&self, unique_id: [u8; 128]) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::BuildModel {
                unique_id: Box::new(unique_id),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    #[cfg(feature = "glm52")]
    fn rank0_step_async(&self, token: u32, position: usize) -> Result<Receiver<Result<u32>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::Rank0Step {
                token,
                position,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    #[cfg(feature = "glm52")]
    fn expert_step_async(&self) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::ExpertStep { resp: resp_tx })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.request_shutdown()?;
        self.join()
    }

    fn request_shutdown(&self) -> Result<()> {
        self.tx
            .send(Glm52RankCommand::Shutdown)
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(())
    }

    fn join(&mut self) -> Result<()> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker panicked"))?;
        Ok(())
    }
}

impl Drop for Glm52RankWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(feature = "glm52")]
enum Glm52RankModel {
    Rank0(Box<Glm52Rank0Model>),
    Expert(Glm52ExpertRankModel),
}

#[cfg(feature = "glm52")]
struct Glm52RankRuntime {
    model: Glm52RankModel,
    ep8: Glm52MoeEp8State,
}

struct Glm52RankThreadState {
    placement: Glm52RankPlacement,
    ctx: Glm52RankGpuContext,
    bundle: Glm52RankLoadBundle,
    loaded: Option<Glm52RankGpuWeights>,
    #[cfg(feature = "glm52")]
    runtime: Option<Glm52RankRuntime>,
}

impl Glm52RankThreadState {
    fn new(
        placement: Glm52RankPlacement,
        ctx: Glm52RankGpuContext,
        bundle: Glm52RankLoadBundle,
    ) -> Self {
        Self {
            placement,
            ctx,
            bundle,
            loaded: None,
            #[cfg(feature = "glm52")]
            runtime: None,
        }
    }

    fn load_weights(&mut self, model_path: &Path) -> Result<Glm52RankWeightLoadReport> {
        let loaded = load_rank_weights_to_gpu(&self.ctx, model_path, &self.bundle)?;
        ensure!(
            loaded.loaded_total_bytes == loaded.weights.total_bytes,
            "GLM5.2 rank {} loaded bytes {} differ from resident raw bytes {}",
            self.placement.rank,
            loaded.loaded_total_bytes,
            loaded.weights.total_bytes
        );
        let report = Glm52RankWeightLoadReport {
            rank: self.placement.rank,
            loaded_tensor_count: loaded.loaded_tensor_count,
            loaded_total_bytes: loaded.loaded_total_bytes,
            resident_raw_bytes: loaded.weights.total_bytes,
            loaded_to_gpu: true,
        };
        self.loaded = Some(loaded.weights);
        Ok(report)
    }

    #[cfg(feature = "glm52")]
    fn build_model(&mut self, unique_id: &[u8; 128]) -> Result<()> {
        let mut weights = self
            .loaded
            .take()
            .context("GLM5.2 build_model called before weights were loaded")?;
        let dev_ctx = self.ctx.device_context();
        let model = if self.placement.rank == 0 {
            Glm52RankModel::Rank0(Box::new(Glm52Rank0Model::build(&dev_ctx, &mut weights)?))
        } else {
            Glm52RankModel::Expert(Glm52ExpertRankModel::build(&dev_ctx, &mut weights)?)
        };
        ensure!(
            weights.expert_layers.is_empty(),
            "GLM5.2 rank {} left {} expert layers unconsumed after model build",
            self.placement.rank,
            weights.expert_layers.len()
        );
        // Non-expert leftovers are the MTP-layer tensors (out of scope) —
        // dropped with `weights` here.
        drop(weights);
        // Collective: every rank calls this concurrently.
        let ep8 = Glm52MoeEp8State::new(&dev_ctx, unique_id, GLM52_EP_RANKS, self.placement.rank)?;
        self.runtime = Some(Glm52RankRuntime { model, ep8 });
        Ok(())
    }

    #[cfg(feature = "glm52")]
    fn rank0_step(&mut self, token: u32, position: usize) -> Result<u32> {
        let dev_ctx = self.ctx.device_context();
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 step before build_model")?;
        match &mut runtime.model {
            Glm52RankModel::Rank0(model) => {
                model.decode_step(&dev_ctx, &mut runtime.ep8, token, position)
            }
            Glm52RankModel::Expert(_) => {
                anyhow::bail!(
                    "GLM5.2 Rank0Step sent to expert rank {}",
                    self.placement.rank
                )
            }
        }
    }

    #[cfg(feature = "glm52")]
    fn expert_step(&mut self) -> Result<()> {
        let dev_ctx = self.ctx.device_context();
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 step before build_model")?;
        match &mut runtime.model {
            Glm52RankModel::Expert(model) => model.expert_step(&dev_ctx, &mut runtime.ep8),
            Glm52RankModel::Rank0(_) => {
                anyhow::bail!("GLM5.2 ExpertStep sent to rank 0")
            }
        }
    }
}

fn rank_worker_loop(rx: Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadWeights { model_path, resp } => {
                let _ = resp.send(state.load_weights(&model_path));
            }
            #[cfg(feature = "glm52")]
            Glm52RankCommand::BuildModel { unique_id, resp } => {
                let _ = resp.send(state.build_model(&unique_id));
            }
            #[cfg(feature = "glm52")]
            Glm52RankCommand::Rank0Step {
                token,
                position,
                resp,
            } => {
                let _ = resp.send(state.rank0_step(token, position));
            }
            #[cfg(feature = "glm52")]
            Glm52RankCommand::ExpertStep { resp } => {
                let _ = resp.send(state.expert_step());
            }
            Glm52RankCommand::Shutdown => break,
        }
    }
    // The DeepEP context drop is collective — it runs here as every rank's
    // worker exits its loop after the coordinator broadcast Shutdown.
    drop(state);
}

#[cfg(feature = "glm52")]
/// One synchronized step across all ranks: rank 0 runs the full model, ranks
/// 1..7 replay the MoE collectives. Joins every rank so an error on any rank
/// surfaces immediately instead of via the DeepEP device timeout.
fn step_all_ranks(workers: &[Glm52RankWorker], token: u32, position: usize) -> Result<u32> {
    let expert_resps = workers[1..]
        .iter()
        .map(Glm52RankWorker::expert_step_async)
        .collect::<Result<Vec<_>>>()?;
    let rank0_resp = workers[0].rank0_step_async(token, position)?;
    let next = rank0_resp
        .recv()
        .map_err(|_| anyhow::anyhow!("GLM5.2 rank 0 dropped its step response"))??;
    for (idx, resp) in expert_resps.into_iter().enumerate() {
        resp.recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {} dropped its step response", idx + 1))??;
    }
    Ok(next)
}

#[cfg(feature = "glm52")]
/// Serial bs=1 coordinator: prefill rides decode token-by-token, then greedy
/// decode until eos/max_tokens. Batching, streaming scheduler, CUDA graphs:
/// PR5.
pub(crate) fn run_bs1_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52RankWorker>,
    eos_token_ids: Vec<u32>,
) {
    while let Some(req) = submit_rx.blocking_recv() {
        let prompt_tokens = req.prompt_tokens.len();
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens: 0,
        });

        if let Err(message) = validate_request(&req) {
            let _ = req.token_tx.send(TokenEvent::Rejected {
                message,
                prompt_tokens,
                completion_tokens: 0,
            });
            continue;
        }

        match run_request(&workers, &req, &eos_token_ids) {
            Ok((completion_tokens, finish_reason)) => {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                });
            }
            Err((completion_tokens, err)) => {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: format!("{err:#}"),
                    prompt_tokens,
                    completion_tokens,
                });
            }
        }
    }
}

#[cfg(feature = "glm52")]
fn validate_request(req: &GenerateRequest) -> std::result::Result<(), String> {
    if req.prompt_tokens.is_empty() {
        return Err("GLM5.2 requires a non-empty prompt".to_owned());
    }
    if req.max_tokens == 0 {
        return Err("GLM5.2 requires max_tokens > 0".to_owned());
    }
    // Position of the last generated token's forward step.
    let last_position = req.prompt_tokens.len() + req.max_tokens - 1;
    if last_position > GLM52_MAX_MODEL_LEN {
        return Err(format!(
            "GLM5.2 bring-up context cap: prompt {} + max_tokens {} exceeds {GLM52_MAX_MODEL_LEN}",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    if !req.params.is_greedy() {
        return Err("GLM5.2 bring-up supports greedy sampling only (temperature 0)".to_owned());
    }
    if req.logprobs > 0 || req.echo {
        return Err("GLM5.2 bring-up does not support logprobs/echo".to_owned());
    }
    if req.lora_adapter.is_some() {
        return Err("GLM5.2 does not support LoRA adapters".to_owned());
    }
    Ok(())
}

#[cfg(feature = "glm52")]
fn run_request(
    workers: &[Glm52RankWorker],
    req: &GenerateRequest,
    eos_token_ids: &[u32],
) -> std::result::Result<(usize, FinishReason), (usize, anyhow::Error)> {
    let mut completion = 0usize;

    // Prefill rides decode: feed prompt tokens one position at a time; the
    // last prompt token's step yields the first generated token.
    let mut next = 0u32;
    for (position, &token) in req.prompt_tokens.iter().enumerate() {
        next = step_all_ranks(workers, token, position).map_err(|err| (completion, err))?;
    }

    loop {
        completion += 1;
        let _ = req.token_tx.send(TokenEvent::Token {
            id: next,
            logprob: None,
        });
        if !req.params.ignore_eos && eos_token_ids.contains(&next) {
            return Ok((completion, FinishReason::Stop));
        }
        if completion >= req.max_tokens {
            return Ok((completion, FinishReason::Length));
        }
        let position = req.prompt_tokens.len() + completion - 1;
        next = step_all_ranks(workers, next, position).map_err(|err| (completion, err))?;
    }
}

/// Featureless builds stop at weight residency: requests are rejected after
/// scheduling (the `glm52` feature compiles the real engine).
#[cfg(not(feature = "glm52"))]
pub(crate) fn run_rejecting_load_only_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52RankWorker>,
) {
    let _workers = workers;
    while let Some(req) = submit_rx.blocking_recv() {
        let prompt_tokens = req.prompt_tokens.len();
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
        let scheduled_at_unix_s = unix_now_s();
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s,
            prompt_tokens,
            cached_tokens: 0,
        });
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: "GLM5.2 forward requires the glm52 cargo feature; this build is load-only"
                .to_owned(),
            prompt_tokens,
            completion_tokens: 0,
        });
    }
}
