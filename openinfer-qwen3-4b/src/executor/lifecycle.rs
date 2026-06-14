use anyhow::Result;
use crossbeam_channel as channel;

use crate::config::TensorParallelConfig;
use crate::dflash::DFlashDraftModel;
use crate::weights::{ModelRuntimeConfig, Qwen3Model};
use crate::{Qwen3LoraOptions, Qwen3OffloadOptions, Qwen3SpeculativeOptions};
use openinfer_core::tensor::DeviceContext;
use openinfer_kv_cache::{KvBlockGuard, KvBuffer, KvCacheManager};
use openinfer_kv_offload::{OffloadConfig, OffloadEngine};

use super::worker::{LocalQwen3Lane, RankWorker, StepCommand, WorkerStepOutcome};
use super::{
    DecodePlan, DecodeResult, ModelExecutor, PrefillPlan, PrefillRequestResult, PrefillResult,
    PrefillStepItem, Qwen3Executor, Qwen3ExecutorMetadata, RequestId, SpeculativeDraftPlan,
    SpeculativeDraftResult, SpeculativeVerifyPlan, SpeculativeVerifyResult, UnifiedPlan,
    UnifiedResult, dflash_memory_reserve_bytes,
};

impl Qwen3Executor {
    fn single(
        model: Qwen3Model,
        offload_opts: &Qwen3OffloadOptions,
        speculative_options: Qwen3SpeculativeOptions,
    ) -> Result<Self> {
        let budget =
            model.kv_budget_with_reserved_bytes(dflash_memory_reserve_bytes(&speculative_options)?);
        let kv_mgr = KvCacheManager::new(
            &model.device_ctx().stream,
            budget.num_layers,
            budget.num_kv_heads,
            budget.head_dim,
            budget.block_size,
            budget.num_blocks,
        )?;
        let kv_buffer = kv_mgr.buffer().clone();
        // Build the offload engine while the model's stream is still in hand
        // (it moves into the RankWorker below). Registers the fused KV buffer.
        let offload = build_offload(offload_opts, &kv_mgr, model.device_ctx())?;
        let total_blocks = kv_mgr.pool().total_blocks();
        let padding_block_id = kv_mgr.pool().padding_block_id();
        let dflash = match speculative_options.dflash {
            Some(options) => {
                let path = options
                    .model_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("DFlash model path must be valid UTF-8"))?;
                Some(DFlashDraftModel::from_safetensors_for_target(
                    model.device_ctx(),
                    path,
                    &model,
                )?)
            }
            None => None,
        };
        let speculative_enabled = dflash.is_some();
        let max_context_tokens = effective_max_context_tokens(
            model.config().max_position_embeddings,
            dflash.as_ref().map(DFlashDraftModel::block_size),
        )?;
        let metadata = Qwen3ExecutorMetadata {
            block_size: budget.block_size,
            stop_token_ids: model.config().stop_token_ids.clone(),
            config: model.config().clone(),
            max_context_tokens,
        };
        if speculative_enabled {
            log::info!(
                "Qwen3 DFlash loaded; disabling prefix cache for hidden-state capture; max_context_tokens={}",
                max_context_tokens
            );
        }
        Ok(Self {
            metadata,
            kv_mgr,
            request_kvs: Default::default(),
            primary: RankWorker::spawn(
                0,
                LocalQwen3Lane::new(model, kv_buffer, total_blocks, padding_block_id, dflash)?,
            )?,
            workers: Vec::new(),
            loaded_lora_adapters: Default::default(),
            prefix_cache_enabled: !speculative_enabled,
            lora_options: Qwen3LoraOptions::default(),
            offload,
            saved_cursor: Default::default(),
            prefetch: Default::default(),
            l1_retention_disabled: false,
            speculative_enabled,
            dflash_ready_requests: Default::default(),
        })
    }

    pub fn from_runtime(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        Self::from_runtime_with_lora_options(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            Qwen3LoraOptions::default(),
            Qwen3OffloadOptions::disabled(),
        )
    }

    pub fn from_runtime_with_lora_options(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        lora_options: Qwen3LoraOptions,
        offload_options: Qwen3OffloadOptions,
    ) -> Result<Self> {
        Self::from_runtime_with_options(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            lora_options,
            offload_options,
            Qwen3SpeculativeOptions::disabled(),
        )
    }

    pub fn from_runtime_with_options(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        lora_options: Qwen3LoraOptions,
        offload_options: Qwen3OffloadOptions,
        speculative_options: Qwen3SpeculativeOptions,
    ) -> Result<Self> {
        let lora_options = lora_options.validate()?;
        anyhow::ensure!(
            !device_ordinals.is_empty(),
            "Qwen3 executor requires at least one device"
        );
        anyhow::ensure!(
            speculative_options.dflash.is_none()
                || (device_ordinals.len() == 1
                    && lora_options == Qwen3LoraOptions::default()
                    && !offload_options.enabled),
            "DFlash speculative decoding currently requires the single-GPU base Qwen3 path without LoRA or KV offload"
        );
        anyhow::ensure!(
            !offload_options.enabled || device_ordinals.len() == 1,
            "KV offload is only supported on the single-GPU path (tensor parallel \
             shards KV per rank); got {} devices",
            device_ordinals.len()
        );
        if device_ordinals.len() == 1 {
            let model = Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: None,
                    device_ordinal: device_ordinals[0],
                    max_loras: lora_options.max_loras,
                    max_lora_rank: lora_options.max_lora_rank,
                },
            )?;
            let mut executor = Self::single(model, &offload_options, speculative_options)?;
            executor.lora_options = lora_options;
            return Ok(executor);
        }

        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                    max_loras: lora_options.max_loras,
                    max_lora_rank: lora_options.max_lora_rank,
                },
            )?);
        }

        // Compute budget from first model (all ranks share geometry).
        let budget = models[0].kv_budget();

        // Create the centralized KvCacheManager on rank 0's stream.
        let kv_mgr = KvCacheManager::new(
            &models[0].device_ctx().stream,
            budget.num_layers,
            budget.num_kv_heads,
            budget.head_dim,
            budget.block_size,
            budget.num_blocks,
        )?;

        let metadata = Qwen3ExecutorMetadata {
            block_size: budget.block_size,
            stop_token_ids: models[0].config().stop_token_ids.clone(),
            config: models[0].config().clone(),
            max_context_tokens: models[0].config().max_position_embeddings,
        };

        // Create extra KvBuffers for ranks 1+ on their respective streams.
        let mut extra_kv_buffers = Vec::with_capacity(world_size - 1);
        for model in &models[1..] {
            extra_kv_buffers.push(KvBuffer::new(
                &model.device_ctx().stream,
                budget.num_layers,
                budget.num_kv_heads,
                budget.head_dim,
                budget.block_size,
                budget.num_blocks,
            )?);
        }

        let streams = models
            .iter()
            .map(|m| m.device_ctx().stream.clone())
            .collect();
        let comms = cudarc::nccl::safe::Comm::from_devices(streams)
            .map_err(|e| anyhow::anyhow!("failed to initialize NCCL comms: {e:?}"))?;
        for (model, comm) in models.iter_mut().zip(comms) {
            model.attach_tp_comm(comm);
        }

        let total_blocks = kv_mgr.pool().total_blocks();
        let padding_block_id = kv_mgr.pool().padding_block_id();

        // Primary rank gets the KvBuffer from the centralized manager.
        let primary_buffer = kv_mgr.buffer().clone();
        let mut models_iter = models.into_iter();
        let primary_model = models_iter.next().unwrap();
        let primary = RankWorker::spawn(
            0,
            LocalQwen3Lane::new(
                primary_model,
                primary_buffer,
                total_blocks,
                padding_block_id,
                None,
            )?,
        )?;

        // Worker ranks get their own extra KvBuffers.
        let workers = models_iter
            .zip(extra_kv_buffers)
            .enumerate()
            .map(|(index, (model, buffer))| {
                let lane =
                    LocalQwen3Lane::new(model, buffer, total_blocks, padding_block_id, None)?;
                RankWorker::spawn(index + 1, lane)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            metadata,
            kv_mgr,
            request_kvs: Default::default(),
            primary,
            workers,
            loaded_lora_adapters: Default::default(),
            prefix_cache_enabled: true,
            lora_options,
            // Offload is single-GPU only (asserted above); never built here.
            offload: None,
            saved_cursor: Default::default(),
            prefetch: Default::default(),
            l1_retention_disabled: false,
            speculative_enabled: false,
            dflash_ready_requests: Default::default(),
        })
    }

    pub fn block_size(&self) -> usize {
        <Self as ModelExecutor>::block_size(self)
    }

    pub fn max_request_blocks(&self) -> usize {
        <Self as ModelExecutor>::max_request_blocks(self)
    }

    pub fn available_blocks(&self) -> usize {
        <Self as ModelExecutor>::available_blocks(self)
    }

    pub fn is_stop_token(&self, token_id: u32) -> bool {
        <Self as ModelExecutor>::is_stop_token(self, token_id)
    }

    pub fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        <Self as ModelExecutor>::drop_request(self, request_id)
    }

    pub fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        <Self as ModelExecutor>::execute_prefill(self, plan)
    }

    pub fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        <Self as ModelExecutor>::execute_decode(self, plan)
    }

    pub fn execute_speculative_verify(
        &mut self,
        plan: SpeculativeVerifyPlan<'_>,
    ) -> Result<SpeculativeVerifyResult> {
        <Self as ModelExecutor>::execute_speculative_verify(self, plan)
    }

    pub fn execute_speculative_draft(
        &mut self,
        plan: SpeculativeDraftPlan<'_>,
    ) -> Result<SpeculativeDraftResult> {
        <Self as ModelExecutor>::execute_speculative_draft(self, plan)
    }

    pub fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        <Self as ModelExecutor>::execute_unified(self, plan)
    }

    /// Prefix caching is on by default; tests that assert bit-identical
    /// replay disable it (a cache hit changes prefill GEMM shapes, which
    /// drifts logits by bf16 ULPs).
    pub fn set_prefix_cache_enabled(&mut self, enabled: bool) {
        self.prefix_cache_enabled = enabled && !self.speculative_enabled;
    }

    /// vLLM-style `--no-prefix-cache`. Behaviour depends on whether offload is
    /// active:
    ///   * **No offload** — classic: disable prefix matching outright, so every
    ///     prefill recomputes the full prompt.
    ///   * **With offload** — pure-L2 mode: keep matching on (the host-tier
    ///     restore registers blocks and relies on `match_and_add_prefix` to pick
    ///     them up) but stop retaining completed blocks in HBM, so no request
    ///     ever serves its prefix from a cross-request L1 hit. Every reuse then
    ///     comes from the host tier, which is the point of the L2 benchmark.
    ///
    /// A resident HBM block and its host-tier copy share one content hash, so
    /// the cache cannot be told to prefer L2 for a block still in HBM — the only
    /// way to force the bytes from L2 is to not keep the HBM copy around.
    pub fn set_no_prefix_cache(&mut self, on: bool) {
        if self.speculative_enabled {
            self.prefix_cache_enabled = false;
            self.l1_retention_disabled = false;
            return;
        }
        if self.offload.is_some() {
            self.l1_retention_disabled = on;
        } else {
            self.prefix_cache_enabled = !on;
        }
    }

    /// Whether KV offload is active on this executor.
    pub fn offload_enabled(&self) -> bool {
        self.offload.is_some()
    }

    /// Flush pending offload saves into the host read cache so a following
    /// query can see them. A persistence barrier for handoff and tests; no-op
    /// without offload.
    pub fn flush_offload_saves(&self) {
        if let Some(offload) = &self.offload {
            offload.flush_saves();
        }
    }

    /// Drop every cached-but-unused GPU prefix block. With offload on, this
    /// forces a cold prefix to be restored from the host tier on its next
    /// request (rather than served from HBM).
    pub fn evict_cached_blocks(&self) {
        self.kv_mgr.pool().evict_inactive();
    }

    /// Begin an async CPU-tier KV prefetch for `request_id`; see the
    /// [`ModelExecutor`] hook. Public so admission drivers and tests can park a
    /// request on its load. Returns `true` when a load is in flight.
    pub fn begin_kv_prefetch(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        lora_adapter: Option<&str>,
        reserve_floor: usize,
    ) -> bool {
        <Self as ModelExecutor>::begin_kv_prefetch(
            self,
            request_id,
            prompt_tokens,
            lora_adapter,
            reserve_floor,
        )
    }

    /// Block until at least one in-flight prefetch settles, then sweep the
    /// rest; returns the settled request ids (now prefill-eligible).
    pub fn wait_ready_prefetch(&mut self) -> Vec<RequestId> {
        <Self as ModelExecutor>::wait_ready_prefetch(self)
    }

    // ── KV-offload SAVE ────────────────────────────────────────────────

    /// Save every block that sealed since this request's last save to the host
    /// tier (fire-and-forget). Safe to call right after `apply_prefill`/
    /// `apply_decode`: the producing step's token read-back has already
    /// synchronized the compute stream, so the sealed KV is fully written.
    pub(super) fn save_sealed_blocks(&mut self, request_id: RequestId) {
        if self.offload.is_none() {
            return;
        }
        let Some(rkv) = self.request_kvs.get(&request_id) else {
            return;
        };
        // `assigned_block_hashes` lists only sealed (registered) blocks; the
        // partial tail block has no hash and never appears here.
        let assigned = rkv.assigned_block_hashes();
        let prefix_matched = rkv.prefix_matched_blocks();
        let cursor = self
            .saved_cursor
            .entry(request_id)
            .or_insert(prefix_matched);
        if assigned.len() <= *cursor {
            return;
        }
        let fresh = &assigned[*cursor..];
        let block_ids: Vec<i32> = fresh.iter().map(|(id, _)| *id).collect();
        let block_hashes: Vec<Vec<u8>> = fresh.iter().map(|(_, h)| h.to_vec()).collect();
        // Pin exactly the blocks being saved (aligned 1:1 with `assigned`) for
        // the duration of the async D2H, so a finished request can't hand the
        // slot to a new request that overwrites it before the copy lands.
        let pins: Vec<KvBlockGuard> = rkv
            .assigned_block_guards()
            .into_iter()
            .skip(*cursor)
            .collect();
        *cursor = assigned.len();
        self.offload
            .as_ref()
            .expect("offload present")
            .save(&block_ids, &block_hashes, pins);
    }

    // ── Chunked prefill ────────────────────────────────────────────────

    /// Prepare one prefill step for `req`: create its `RequestKv` on the
    /// first chunk (matching the prefix cache), then clamp the scheduler's
    /// chunk budget to the prompt tokens actually remaining and allocate KV
    /// for them. Sets `chunk_start`/`chunk_tokens` on the item.
    pub(super) fn schedule_prefill_chunk(&mut self, req: &mut PrefillStepItem) -> Result<()> {
        if !self.request_kvs.contains_key(&req.request_id) {
            let mut rkv = self.kv_mgr.pool().new_request(
                req.prompt_tokens.clone(),
                req.max_output_tokens,
                req.lora_adapter.as_deref(),
            );
            // Echo needs logits for every prompt position; cached positions
            // are never forwarded, so echo requests prefill from scratch.
            if self.prefix_cache_enabled && !req.echo {
                req.cached_tokens = rkv.match_and_add_prefix(self.kv_mgr.pool())?;
            }
            self.request_kvs.insert(req.request_id, rkv);
            // match_and_add_prefix above already absorbed any CPU-prefetched
            // blocks (now held by the request's sequence), so release the
            // prefetch's separate hold.
            self.prefetch.remove(&req.request_id);
        }
        let rkv = self
            .request_kvs
            .get_mut(&req.request_id)
            .expect("inserted above");
        req.chunk_start = rkv.kv_position();
        let remaining = req.prompt_tokens.len() - req.chunk_start;
        // Echo must produce all-position logits in a single forward, so it is
        // exempt from chunking (the scheduler never splits echo requests).
        req.chunk_tokens = if req.echo {
            remaining
        } else {
            remaining.min(req.chunk_budget)
        };
        assert!(
            req.chunk_tokens > 0,
            "zero-token prefill chunk for {:?} (budget {})",
            req.request_id,
            req.chunk_budget
        );
        rkv.schedule_prefill(req.chunk_tokens, self.kv_mgr.pool())
            .map_err(|e| anyhow::anyhow!("schedule_prefill failed for {:?}: {e}", req.request_id))
    }

    /// Register a finished prefill step on the request's KV: the final chunk
    /// carries the first generated token, non-final chunks only advance the
    /// KV position.
    pub(super) fn apply_prefill_result(&mut self, result: &PrefillRequestResult) -> Result<()> {
        let rkv = self
            .request_kvs
            .get_mut(&result.request_id)
            .expect("request must exist after prefill");
        if result.completed {
            rkv.apply_prefill(result.first_token, self.kv_mgr.pool())
        } else {
            rkv.apply_prefill_chunk(self.kv_mgr.pool())
        }
    }

    /// Finalize one prefetch whose load returned `result`. On success the
    /// reserved blocks are staged + registered (held by the probe until the
    /// request prefills); on failure the state is dropped so the request
    /// prefills from scratch.
    pub(super) fn settle_prefetch(
        &mut self,
        id: RequestId,
        result: Result<(), openinfer_kv_offload::EngineError>,
    ) {
        if let Some(st) = self.prefetch.get_mut(&id) {
            st.handle = None;
        }
        match result {
            Ok(()) => {
                let reservation = self
                    .prefetch
                    .get_mut(&id)
                    .and_then(|st| st.reservation.take())
                    .expect("reservation present until commit");
                let st = self.prefetch.get_mut(&id).expect("prefetch present");
                self.kv_mgr
                    .pool()
                    .commit_loaded_blocks(&mut st.probe, reservation);
            }
            Err(e) => {
                log::warn!("KV offload load failed for {id:?} (prefill from scratch): {e}");
                self.prefetch.remove(&id);
            }
        }
    }

    fn wait_for_step_ack(
        pending: Vec<channel::Receiver<Result<WorkerStepOutcome>>>,
        op_name: &'static str,
    ) -> Result<()> {
        for recv in pending {
            match recv
                .recv()
                .map_err(|_| anyhow::anyhow!("tensor-parallel {op_name} worker dropped"))??
            {
                WorkerStepOutcome::Ack => {}
                other => {
                    return Err(anyhow::anyhow!(
                        "tensor-parallel {op_name} worker returned unexpected payload: {}",
                        other.kind()
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn run_step(&self, step: &StepCommand) -> Result<WorkerStepOutcome> {
        let primary = self.primary.run_step(step.clone(), true)?;
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            pending.push(worker.run_step(step.clone(), false)?);
        }
        let primary_result = primary
            .recv()
            .map_err(|_| anyhow::anyhow!("primary worker dropped step response"))??;
        Self::wait_for_step_ack(pending, step.kind())?;
        Ok(primary_result)
    }
}

fn effective_max_context_tokens(
    target_max_context_tokens: usize,
    speculative_block_size: Option<usize>,
) -> Result<usize> {
    match speculative_block_size {
        Some(block_size) => {
            anyhow::ensure!(
                target_max_context_tokens > block_size,
                "DFlash block_size {} leaves no usable context in target max_position_embeddings {}",
                block_size,
                target_max_context_tokens
            );
            Ok(target_max_context_tokens - block_size)
        }
        None => Ok(target_max_context_tokens),
    }
}

/// Build the KV-offload engine for the single-GPU path, or `None` when offload
/// is disabled. Registers the fused KV buffer with pegaflow against the model's
/// device/stream — must be called while that stream is still owned by the model
/// (before it moves into the `RankWorker`).
fn build_offload(
    opts: &Qwen3OffloadOptions,
    kv_mgr: &KvCacheManager,
    ctx: &DeviceContext,
) -> Result<Option<OffloadEngine>> {
    if !opts.enabled {
        return Ok(None);
    }
    let device_id = ctx.device_ordinal as i32;
    let config = OffloadConfig::new(
        format!("qwen3-4b-dev{device_id}"),
        device_id,
        opts.pinned_pool_bytes,
    );
    let engine = OffloadEngine::new(config, kv_mgr.buffer(), &ctx.stream)
        .map_err(|e| anyhow::anyhow!("KV offload engine init failed: {e}"))?;
    log::info!(
        "KV offload enabled on device {device_id} ({} MiB host tier)",
        opts.pinned_pool_bytes >> 20
    );
    Ok(Some(engine))
}

#[cfg(test)]
mod tests {
    use super::effective_max_context_tokens;

    #[test]
    fn dflash_context_limit_reserves_one_draft_block() {
        assert_eq!(effective_max_context_tokens(128, None).unwrap(), 128);
        assert_eq!(effective_max_context_tokens(128, Some(16)).unwrap(), 112);
        assert!(effective_max_context_tokens(16, Some(16)).is_err());
    }
}
