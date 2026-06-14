use std::thread;

use anyhow::Result;
use crossbeam_channel as channel;

use crate::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::dflash::DFlashDraftModel;
use crate::speculative::{
    DraftResult as SpeculativeDraftResult, DraftStepItem as SpeculativeDraftStepItem,
    VerifyResult as SpeculativeVerifyResult, VerifyStepItem as SpeculativeVerifyStepItem,
};
use crate::weights::Qwen3Model;
use openinfer_core::engine::TokenLogprob;
use openinfer_core::kv_pool::KvLayout;
use openinfer_core::sampler::SamplingParams;
use openinfer_core::tensor::{DeviceVec, HiddenStates};
use openinfer_kv_cache::{KvBuffer, KvView};

use super::dflash_lane::DFlashLaneState;
use super::{
    CublasThreadGuard, DecodeResult, DecodeStepItem, PrefillResult, PrefillStepItem, RequestId,
    SamplingScratch, UnifiedResult, bind_model_thread, compute_logprobs_from_cpu,
    execute_step_on_lane, tune_decode_gemm_algos,
};

pub(super) struct LocalQwen3Lane {
    pub(super) model: Qwen3Model,
    kv_buffer: KvBuffer,
    layout: KvLayout,
    pub(super) bufs: BatchDecodeBuffers,
    pub(super) sample_scratch: SamplingScratch,
    pub(super) dflash: Option<DFlashLaneState>,
}

impl LocalQwen3Lane {
    pub(super) fn new(
        model: Qwen3Model,
        kv_buffer: KvBuffer,
        total_blocks: usize,
        padding_block_id: i32,
        dflash_model: Option<DFlashDraftModel>,
    ) -> Result<Self> {
        let buf_layout = kv_buffer.layout();
        let layout = KvLayout::new(
            buf_layout.num_layers,
            buf_layout.num_kv_heads,
            buf_layout.head_dim,
            buf_layout.page_size,
        );
        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let bufs = BatchDecodeBuffers::new(
            model.device_ctx(),
            model.config().hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config().vocab_size,
            max_bucket,
            total_blocks,
            padding_block_id,
            model.local_num_attention_heads(),
        )?;
        let sample_scratch =
            SamplingScratch::new(model.device_ctx(), model.config().vocab_size, max_bucket)?;
        Ok(Self {
            model,
            kv_buffer,
            layout,
            bufs,
            sample_scratch,
            dflash: dflash_model.map(DFlashLaneState::new),
        })
    }

    fn bind(&self) -> Result<CublasThreadGuard> {
        bind_model_thread(&self.model)?;
        tune_decode_gemm_algos(&self.model)?;
        if let Some(dflash) = &self.dflash {
            dflash.model.tune_gemm_algos(&self.model)?;
        }
        Ok(CublasThreadGuard)
    }

    /// Pick one token per logits column (batched argmax for greedy rows,
    /// per-row sampler otherwise). Grows the sampling scratch when a step
    /// is wider than the decode bucket it was sized for.
    pub(super) fn select_step_tokens(
        &mut self,
        logits: &HiddenStates,
        params: &[&SamplingParams],
        random_vals: &[f32],
    ) -> Result<Vec<u32>> {
        let scratch_rows = logits.seq_len.max(params.len());
        if scratch_rows > self.sample_scratch.row_indices.len() {
            self.sample_scratch = SamplingScratch::new(
                self.model.device_ctx(),
                self.model.config().vocab_size,
                scratch_rows,
            )?;
        }
        openinfer_core::ops::select_batch_tokens_into(
            self.model.device_ctx(),
            logits,
            params,
            random_vals,
            &mut self.sample_scratch.row_indices,
            &mut self.sample_scratch.argmax_partial_values,
            &mut self.sample_scratch.argmax_partial_indices,
            &mut self.sample_scratch.probs,
            &mut self.sample_scratch.top1_values,
            &mut self.sample_scratch.row_states,
            &mut self.sample_scratch.valid,
            &mut self.sample_scratch.out,
        )
    }

    pub(super) fn select_greedy_contiguous_tokens(
        &mut self,
        logits: &HiddenStates,
    ) -> Result<Vec<u32>> {
        if logits.seq_len > self.sample_scratch.row_indices.len() {
            self.sample_scratch = SamplingScratch::new(
                self.model.device_ctx(),
                self.model.config().vocab_size,
                logits.seq_len,
            )?;
        }
        openinfer_core::ops::argmax_batch_bf16_split_into(
            self.model.device_ctx(),
            logits,
            &mut self.sample_scratch.argmax_partial_values,
            &mut self.sample_scratch.argmax_partial_indices,
            &mut self.sample_scratch.top1_values,
            &mut self.sample_scratch.out,
        )?;
        self.sample_scratch
            .out_host
            .resize(self.sample_scratch.out.len(), 0);
        self.model
            .device_ctx()
            .stream
            .memcpy_dtoh(&self.sample_scratch.out, &mut self.sample_scratch.out_host)
            .map_err(|e| anyhow::anyhow!("D2H greedy argmax read failed: {}", e))?;
        self.model.device_ctx().sync()?;
        Ok(self
            .sample_scratch
            .out_host
            .iter()
            .take(logits.seq_len)
            .map(|&token| token as u32)
            .collect())
    }

    pub(super) fn extract_logprobs(
        &self,
        logits: &DeviceVec,
        sampled_token: u32,
        top_k: usize,
    ) -> Result<TokenLogprob> {
        let logits_f32 = logits.to_host(self.model.device_ctx())?;
        compute_logprobs_from_cpu(&logits_f32, sampled_token, top_k)
            .ok_or_else(|| anyhow::anyhow!("logprobs computation failed"))
    }

    pub(super) fn extract_prompt_logprobs(
        &self,
        all_logits: &HiddenStates,
        prev_pos: usize,
        target_token: u32,
        top_k: usize,
    ) -> Option<TokenLogprob> {
        openinfer_core::ops::extract_vec(self.model.device_ctx(), all_logits, prev_pos)
            .ok()
            .and_then(|logits_vec| {
                let logits_f32 = logits_vec.to_host(self.model.device_ctx()).ok()?;
                compute_logprobs_from_cpu(&logits_f32, target_token, top_k)
            })
    }

    pub(super) fn execute_prefill(
        &mut self,
        prompts: &[&[u32]],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
        echo: bool,
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<(HiddenStates, Option<HiddenStates>, Option<HiddenStates>)> {
        if let Some(capture_layer_ids) = capture_layer_ids {
            self.model.batch_prefill_with_hidden_capture(
                prompts,
                kv_views,
                lora_adapters,
                self.kv_buffer.buffer(),
                &self.layout,
                echo,
                Some(capture_layer_ids),
            )
        } else {
            let (logits, all_position_logits) = self.model.batch_prefill(
                prompts,
                kv_views,
                lora_adapters,
                self.kv_buffer.buffer(),
                &self.layout,
                echo,
            )?;
            Ok((logits, all_position_logits, None))
        }
    }

    pub(super) fn execute_decode(
        &mut self,
        token_ids: &[u32],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
    ) -> Result<()> {
        self.model.batch_decode(
            token_ids,
            kv_views,
            lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
            &mut self.bufs,
        )
    }

    pub(super) fn execute_unified(
        &mut self,
        prefill_prompts: &[&[u32]],
        prefill_views: &[KvView],
        prefill_lora_adapters: &[Option<&str>],
        decode_tokens: &[u32],
        decode_views: &[KvView],
        decode_lora_adapters: &[Option<&str>],
    ) -> Result<HiddenStates> {
        self.model.unified_step(
            prefill_prompts,
            prefill_views,
            prefill_lora_adapters,
            decode_tokens,
            decode_views,
            decode_lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
        )
    }

    fn load_lora_adapter(
        &mut self,
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
    ) -> Result<()> {
        let device_adapter =
            crate::lora::load_device_lora_adapter(self.model.device_ctx(), name, adapter)?;
        self.model
            .install_lora_adapter(device_adapter, load_inplace)
    }

    fn unload_lora_adapter(&mut self, name: &str) -> Result<()> {
        self.model.uninstall_lora_adapter(name)
    }

    fn discard_lora_adapter(&mut self, name: &str) -> Result<()> {
        self.model.discard_lora_adapter(name)
    }
}

#[derive(Clone)]
pub(super) enum StepCommand {
    Prefill {
        requests: Vec<PrefillStepItem>,
        kv_views: Vec<KvView>,
        echo: bool,
    },
    Decode {
        requests: Vec<DecodeStepItem>,
        kv_views: Vec<KvView>,
    },
    SpeculativeVerify {
        requests: Vec<SpeculativeVerifyStepItem>,
        kv_views: Vec<KvView>,
    },
    SpeculativeDraft {
        requests: Vec<SpeculativeDraftStepItem>,
    },
    Unified {
        prefill_requests: Vec<PrefillStepItem>,
        prefill_kv_views: Vec<KvView>,
        decode_requests: Vec<DecodeStepItem>,
        decode_kv_views: Vec<KvView>,
    },
}

impl StepCommand {
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Prefill { .. } => "prefill",
            Self::Decode { .. } => "decode",
            Self::SpeculativeVerify { .. } => "speculative_verify",
            Self::SpeculativeDraft { .. } => "speculative_draft",
            Self::Unified { .. } => "unified",
        }
    }
}

enum WorkerCommand {
    RunStep {
        step: StepCommand,
        collect_result: bool,
        resp: channel::Sender<Result<WorkerStepOutcome>>,
    },
    LoadLoraAdapter {
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
        resp: channel::Sender<Result<()>>,
    },
    UnloadLoraAdapter {
        name: String,
        resp: channel::Sender<Result<()>>,
    },
    DiscardLoraAdapter {
        name: String,
        resp: channel::Sender<Result<()>>,
    },
    DropDFlashRequest {
        request_id: RequestId,
        resp: channel::Sender<Result<()>>,
    },
    Shutdown,
}

pub(super) enum WorkerStepOutcome {
    Ack,
    Prefill(PrefillResult),
    Decode(DecodeResult),
    SpeculativeDraft(SpeculativeDraftResult),
    SpeculativeVerify(SpeculativeVerifyResult),
    Unified(UnifiedResult),
}

impl WorkerStepOutcome {
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Ack => "ack",
            Self::Prefill(_) => "prefill",
            Self::Decode(_) => "decode",
            Self::SpeculativeDraft(_) => "speculative_draft",
            Self::SpeculativeVerify(_) => "speculative_verify",
            Self::Unified(_) => "unified",
        }
    }
}

pub(super) struct RankWorker {
    tx: channel::Sender<WorkerCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RankWorker {
    pub(super) fn spawn(rank: usize, mut lane: LocalQwen3Lane) -> Result<Self> {
        let (tx, rx) = channel::unbounded();
        let (startup_tx, startup_rx) = channel::bounded(1);
        let handle = thread::Builder::new()
            .name(format!("qwen3-tp-rank-{rank}"))
            .spawn(move || {
                let startup = lane.bind();
                match startup {
                    Ok(_guard) => {
                        let _ = startup_tx.send(Ok(()));
                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                WorkerCommand::RunStep {
                                    step,
                                    collect_result,
                                    resp,
                                } => {
                                    let result =
                                        execute_step_on_lane(&mut lane, &step, collect_result);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::LoadLoraAdapter {
                                    name,
                                    adapter,
                                    load_inplace,
                                    resp,
                                } => {
                                    let result =
                                        lane.load_lora_adapter(name, adapter, load_inplace);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::UnloadLoraAdapter { name, resp } => {
                                    let result = lane.unload_lora_adapter(&name);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DiscardLoraAdapter { name, resp } => {
                                    let result = lane.discard_lora_adapter(&name);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DropDFlashRequest { request_id, resp } => {
                                    lane.drop_dflash_request(request_id);
                                    let _ = resp.send(Ok(()));
                                }
                                WorkerCommand::Shutdown => break,
                            }
                        }
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn tensor-parallel worker {rank}: {e}"))?;
        startup_rx.recv().map_err(|_| {
            anyhow::anyhow!("tensor-parallel worker {rank} exited during startup")
        })??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    pub(super) fn run_step(
        &self,
        step: StepCommand,
        collect_result: bool,
    ) -> Result<channel::Receiver<Result<WorkerStepOutcome>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::RunStep {
                step,
                collect_result,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker step channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn load_lora_adapter(
        &self,
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
    ) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::LoadLoraAdapter {
                name,
                adapter,
                load_inplace,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on LoRA load"))?;
        Ok(resp_rx)
    }

    pub(super) fn unload_lora_adapter(
        &self,
        name: String,
    ) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::UnloadLoraAdapter {
                name,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on LoRA unload"))?;
        Ok(resp_rx)
    }

    pub(super) fn discard_lora_adapter(
        &self,
        name: String,
    ) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DiscardLoraAdapter {
                name,
                resp: resp_tx,
            })
            .map_err(|_| {
                anyhow::anyhow!("tensor-parallel worker channel closed on LoRA discard")
            })?;
        Ok(resp_rx)
    }

    pub(super) fn drop_dflash_request(&self, request_id: RequestId) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DropDFlashRequest {
                request_id,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on DFlash drop"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker dropped DFlash drop response"))?
    }

    pub(super) fn shutdown(&mut self) {
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
