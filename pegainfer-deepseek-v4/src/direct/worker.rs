use std::{path::Path, thread, time::Instant};

use anyhow::{Context, Result, ensure};
use crossbeam_channel as channel;
use cudarc::driver::{CudaSlice, DevicePtrMut, DeviceRepr, result as cuda_result};
use log::info;

use crate::{
    Config, DeepSeekRopeCache, F32Logits, LayerDecodeCache, RankGpuContext, RankWeightView,
    TensorParallelConfig, all_reduce_hidden_in_place, build_moe_expert_ptr_cache,
    embedding_rank_local, final_logits_rank_local_bf16_hidden, hc_expand_bf16_hidden,
    load_rank_to_gpu, precompute_rope_cache,
    runtime::{
        AttentionAuxScratch, AttentionIndexScratch, AttentionOutputScratch,
        AttentionProjectionScratch, DecodeBatchMeta, DecodeEntryScratch, F32BatchLogits,
        FinalLogitsScratch, HcPostScratch, HcPreNormScratch, MoeAgRsScratch, PrefillWindowTopk,
        SharedExpertScratch, all_gather_logits, all_gather_logits_into,
        block_decode_rank_lane_bf16_hidden_batch_with_scratch,
        block_decode_rank_lane_bf16_hidden_with_scratch,
        block_prefill_rank_lane_bf16_hidden_with_decode_cache, embedding_rank_local_into,
        final_logits_rank_local_bf16_hidden_into, hc_expand_bf16_hidden_into, hc_head_bf16_hidden,
        rank_local_logits_from_hidden_all, rms_norm_bf16_hidden,
    },
};

type RankResult = (usize, Option<Vec<f32>>);
type RankBatchResult = (usize, Option<Vec<Vec<f32>>>);
const DIRECT_BATCH_DECODE_CAPACITY: usize = 2;

#[derive(Clone, Copy, Debug)]
pub(super) struct DirectBatchDecodeEntry {
    pub(super) token_id: u32,
    pub(super) start_pos: usize,
    pub(super) slot_id: usize,
}

pub(super) struct FullDirectRuntime {
    workers: Vec<RankWorker>,
    prefill_profile: bool,
}

enum RankCommand {
    // TODO: cache sizing/lifecycle should be decided by the scheduler, with all
    // 8 ranks applying the same plan. Direct runtime keeps this command only as
    // a temporary bridge while DeepSeek V4 still uses its direct generator.
    EnsureCaches {
        max_seq_len: usize,
        request_slots: usize,
        resp: channel::Sender<Result<()>>,
    },
    ResetCaches {
        resp: channel::Sender<Result<()>>,
    },
    Prefill {
        prompt_tokens: Vec<u32>,
        profile: bool,
        resp: channel::Sender<Result<RankResult>>,
    },
    Decode {
        token_id: u32,
        start_pos: usize,
        resp: channel::Sender<Result<RankResult>>,
    },
    #[allow(dead_code)] // PR A lands the runtime batch path before scheduler wiring.
    DecodeBatch {
        entries: Vec<DirectBatchDecodeEntry>,
        resp: channel::Sender<Result<RankBatchResult>>,
    },
    CloneCacheSlot {
        src_slot: usize,
        dst_slot: usize,
        resp: channel::Sender<Result<()>>,
    },
    #[cfg(test)]
    ResetCacheSlot {
        slot_id: usize,
        resp: channel::Sender<Result<()>>,
    },
    Shutdown,
}

struct RankWorker {
    tx: channel::Sender<RankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

struct OwnedRankComm(cudarc::nccl::safe::Comm);

// SAFETY: The communicator is moved into exactly one persistent rank worker and
// is only used by that worker thread for its owning CUDA stream/device.
unsafe impl Send for OwnedRankComm {}

impl OwnedRankComm {
    fn get(&self) -> &cudarc::nccl::safe::Comm {
        &self.0
    }
}

struct RankDecodeScratch {
    token_ids: CudaSlice<u32>,
    batch_token_ids: CudaSlice<u32>,
    start_pos: CudaSlice<i32>,
    src_rows: CudaSlice<i32>,
    window_dst_rows: CudaSlice<i32>,
    window_base: CudaSlice<i32>,
    compressed_base: CudaSlice<i32>,
    compressed_len: CudaSlice<i32>,
    start_pos_host: Vec<usize>,
    slot_ids_host: Vec<usize>,
    entry: DecodeEntryScratch,
    hc_post: HcPostScratch,
    final_logits: FinalLogitsScratch,
    hc_pre_norm: HcPreNormScratch,
    shared_expert: SharedExpertScratch,
    moe_ag_rs: MoeAgRsScratch,
    attention_projection: AttentionProjectionScratch,
    attention_output: AttentionOutputScratch,
    attention_index: AttentionIndexScratch,
    attention_aux: AttentionAuxScratch,
}

impl RankDecodeScratch {
    fn new(ctx: &RankGpuContext, config: &Config, world_size: usize) -> Result<Self> {
        ctx.set_current()?;
        let seq_capacity = DIRECT_BATCH_DECODE_CAPACITY;
        let token_ids = unsafe { ctx.stream.alloc(1)? };
        let batch_token_ids = unsafe { ctx.stream.alloc(seq_capacity)? };
        let start_pos = unsafe { ctx.stream.alloc(seq_capacity)? };
        let src_rows = unsafe { ctx.stream.alloc(seq_capacity)? };
        let window_dst_rows = unsafe { ctx.stream.alloc(seq_capacity)? };
        let window_base = unsafe { ctx.stream.alloc(seq_capacity)? };
        let compressed_base = unsafe { ctx.stream.alloc(seq_capacity)? };
        let compressed_len = unsafe { ctx.stream.alloc(seq_capacity)? };
        let entry = DecodeEntryScratch::new(ctx, config, seq_capacity)?;
        let hc_post = HcPostScratch::new(ctx, config, seq_capacity)?;
        let final_logits = FinalLogitsScratch::new(ctx, config, world_size, seq_capacity)?;
        let hc_pre_norm = HcPreNormScratch::new(ctx, config, seq_capacity)?;
        let shared_expert = SharedExpertScratch::new(ctx, config, seq_capacity)?;
        let moe_ag_rs = MoeAgRsScratch::new(ctx, config, world_size, seq_capacity)?;
        let attention_projection =
            AttentionProjectionScratch::new(ctx, config, world_size, seq_capacity)?;
        let attention_output = AttentionOutputScratch::new(ctx, config, world_size, seq_capacity)?;
        let attention_index = AttentionIndexScratch::new(ctx, config, seq_capacity)?;
        let attention_aux = AttentionAuxScratch::new(ctx, config, world_size, seq_capacity)?;
        Ok(Self {
            token_ids,
            batch_token_ids,
            start_pos,
            src_rows,
            window_dst_rows,
            window_base,
            compressed_base,
            compressed_len,
            start_pos_host: Vec::with_capacity(seq_capacity),
            slot_ids_host: Vec::with_capacity(seq_capacity),
            entry,
            hc_post,
            final_logits,
            hc_pre_norm,
            shared_expert,
            moe_ag_rs,
            attention_projection,
            attention_output,
            attention_index,
            attention_aux,
        })
    }

    fn token_ids(&mut self, ctx: &RankGpuContext, token_id: u32) -> Result<&CudaSlice<u32>> {
        ctx.set_current()?;
        ctx.stream.memcpy_htod(&[token_id], &mut self.token_ids)?;
        Ok(&self.token_ids)
    }
}

fn fill_decode_batch_metadata(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    slot_stride: usize,
    compressed_slots: usize,
    entries: &[DirectBatchDecodeEntry],
    token_ids: &mut CudaSlice<u32>,
    start_pos_slice: &mut CudaSlice<i32>,
    src_rows_slice: &mut CudaSlice<i32>,
    window_dst_rows_slice: &mut CudaSlice<i32>,
    window_base_slice: &mut CudaSlice<i32>,
    compressed_base_slice: &mut CudaSlice<i32>,
    compressed_len_slice: &mut CudaSlice<i32>,
    start_pos_host: &mut Vec<usize>,
    slot_ids_host: &mut Vec<usize>,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        !entries.is_empty(),
        "direct batch decode entries must not be empty"
    );
    ensure!(
        entries.len() <= DIRECT_BATCH_DECODE_CAPACITY,
        "direct batch decode capacity exceeded: entries={}, capacity={DIRECT_BATCH_DECODE_CAPACITY}",
        entries.len()
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "batch meta layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(
        slot_stride >= config.sliding_window,
        "batch decode slot stride {} smaller than sliding window {}",
        slot_stride,
        config.sliding_window
    );
    ensure!(
        ratio == 0 || compressed_slots > 0,
        "batch decode compressed layer {layer} has no compressed cache slots"
    );
    ensure!(
        ratio > 0 || compressed_slots == 0,
        "batch decode non-compressed layer {layer} unexpectedly has {compressed_slots} compressed slots"
    );
    ensure!(
        slot_stride == config.sliding_window + compressed_slots,
        "batch decode slot stride mismatch: stride={}, window={}, compressed={}",
        slot_stride,
        config.sliding_window,
        compressed_slots
    );
    let mut token_ids_host = Vec::with_capacity(entries.len());
    let mut start_pos = Vec::with_capacity(entries.len());
    let mut src_rows = Vec::with_capacity(entries.len());
    let mut window_dst_rows = Vec::with_capacity(entries.len());
    let mut window_base = Vec::with_capacity(entries.len());
    let mut compressed_base = Vec::with_capacity(entries.len());
    let mut compressed_len = Vec::with_capacity(entries.len());
    start_pos_host.clear();
    slot_ids_host.clear();
    for (row, entry) in entries.iter().enumerate() {
        ensure!(
            entry.slot_id < DIRECT_BATCH_DECODE_CAPACITY,
            "batch decode slot {} exceeds capacity {DIRECT_BATCH_DECODE_CAPACITY}",
            entry.slot_id
        );
        token_ids_host.push(entry.token_id);
        start_pos.push(entry.start_pos as i32);
        src_rows.push(row as i32);
        let base = entry.slot_id * slot_stride;
        window_base.push(base as i32);
        window_dst_rows.push((base + entry.start_pos % config.sliding_window) as i32);
        let c_base = base + config.sliding_window;
        compressed_base.push(c_base as i32);
        if ratio > 0 {
            ensure!(
                entry.start_pos / ratio < compressed_slots,
                "batch decode compressed dst out of range: start_pos={}, ratio={}, compressed_slots={}",
                entry.start_pos,
                ratio,
                compressed_slots
            );
        }
        compressed_len.push(if ratio > 0 {
            ((entry.start_pos + 1) / ratio) as i32
        } else {
            0
        });
        start_pos_host.push(entry.start_pos);
        slot_ids_host.push(entry.slot_id);
    }
    ctx.stream.memcpy_htod(&token_ids_host, token_ids)?;
    ctx.stream.memcpy_htod(&start_pos, start_pos_slice)?;
    ctx.stream.memcpy_htod(&src_rows, src_rows_slice)?;
    ctx.stream
        .memcpy_htod(&window_dst_rows, window_dst_rows_slice)?;
    ctx.stream.memcpy_htod(&window_base, window_base_slice)?;
    ctx.stream
        .memcpy_htod(&compressed_base, compressed_base_slice)?;
    ctx.stream
        .memcpy_htod(&compressed_len, compressed_len_slice)?;
    Ok(())
}

impl RankWorker {
    fn spawn(
        rank: usize,
        ctx: RankGpuContext,
        weights: RankWeightView<'static>,
        comm: cudarc::nccl::safe::Comm,
        moe_comm: cudarc::nccl::safe::Comm,
        config: &'static Config,
    ) -> Result<Self> {
        let (tx, rx) = channel::unbounded();
        let (startup_tx, startup_rx) = channel::bounded(1);
        let comm = OwnedRankComm(comm);
        let moe_comm = OwnedRankComm(moe_comm);
        let handle = thread::Builder::new()
            .name(format!("deepseek-v4-rank-{rank}"))
            .spawn(move || {
                super::affinity::pin_rank_worker_thread(rank, ctx.device_ordinal);
                let mut ropes = Vec::new();
                let mut caches = Vec::new();
                let mut max_cache_seq_len = 0usize;
                let mut cache_request_slots = 0usize;
                let startup = bind_rank_thread(&ctx).and_then(|()| {
                    let ptr_cache = build_moe_expert_ptr_cache(&ctx, config, &weights)?;
                    let decode_scratch =
                        RankDecodeScratch::new(&ctx, config, weights.world_size())?;
                    Ok((ptr_cache, decode_scratch))
                });
                match startup {
                    Ok((ptr_cache, mut decode_scratch)) => {
                        let _ = startup_tx.send(Ok(()));
                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                RankCommand::EnsureCaches {
                                    max_seq_len,
                                    request_slots,
                                    resp,
                                } => {
                                    let result = ensure_rank_worker_caches(
                                        &ctx,
                                        config,
                                        max_seq_len,
                                        request_slots,
                                        &mut caches,
                                        &mut ropes,
                                        &mut max_cache_seq_len,
                                        &mut cache_request_slots,
                                    );
                                    let _ = resp.send(result);
                                }
                                RankCommand::ResetCaches { resp } => {
                                    let result = reset_rank_decode_caches(&ctx, &mut caches);
                                    let _ = resp.send(result);
                                }
                                RankCommand::Prefill {
                                    prompt_tokens,
                                    profile,
                                    resp,
                                } => {
                                    let result = run_prefill_on_rank_lane(
                                        rank,
                                        &ctx,
                                        &weights,
                                        &ptr_cache,
                                        comm.get(),
                                        &ropes,
                                        config,
                                        &prompt_tokens,
                                        &mut caches,
                                        profile,
                                    )
                                    .map(|logits| (rank, logits));
                                    let _ = resp.send(result);
                                }
                                RankCommand::Decode {
                                    token_id,
                                    start_pos,
                                    resp,
                                } => {
                                    let result = run_decode_on_rank_lane(
                                        rank,
                                        &ctx,
                                        &weights,
                                        &ptr_cache,
                                        comm.get(),
                                        moe_comm.get(),
                                        &ropes,
                                        config,
                                        token_id,
                                        start_pos,
                                        &mut caches,
                                        &mut decode_scratch,
                                    )
                                    .map(|logits| (rank, logits));
                                    let _ = resp.send(result);
                                }
                                RankCommand::DecodeBatch { entries, resp } => {
                                    let result = run_decode_batch_on_rank_lane(
                                        rank,
                                        &ctx,
                                        &weights,
                                        &ptr_cache,
                                        comm.get(),
                                        moe_comm.get(),
                                        &ropes,
                                        config,
                                        &entries,
                                        &mut caches,
                                        &mut decode_scratch,
                                    )
                                    .map(|logits| (rank, logits));
                                    let _ = resp.send(result);
                                }
                                RankCommand::CloneCacheSlot {
                                    src_slot,
                                    dst_slot,
                                    resp,
                                } => {
                                    let result = clone_rank_decode_cache_slot(
                                        &ctx,
                                        config,
                                        &mut caches,
                                        src_slot,
                                        dst_slot,
                                    );
                                    let _ = resp.send(result);
                                }
                                #[cfg(test)]
                                RankCommand::ResetCacheSlot { slot_id, resp } => {
                                    let result = reset_rank_decode_cache_slot_for_test(
                                        &ctx,
                                        &mut caches,
                                        slot_id,
                                    );
                                    let _ = resp.send(result);
                                }
                                RankCommand::Shutdown => break,
                            }
                        }
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn DeepSeek rank worker {rank}: {err}"))?;
        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker {rank} exited during startup"))??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    fn decode(
        &self,
        token_id: u32,
        start_pos: usize,
    ) -> Result<channel::Receiver<Result<RankResult>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::Decode {
                token_id,
                start_pos,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker channel closed"))?;
        Ok(resp_rx)
    }

    #[allow(dead_code)] // PR A lands the runtime batch path before scheduler wiring.
    fn decode_batch(
        &self,
        entries: Vec<DirectBatchDecodeEntry>,
    ) -> Result<channel::Receiver<Result<RankBatchResult>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::DecodeBatch {
                entries,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker channel closed"))?;
        Ok(resp_rx)
    }

    fn ensure_caches(&self, max_seq_len: usize, request_slots: usize) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::EnsureCaches {
                max_seq_len,
                request_slots,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker channel closed on EnsureCaches"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker dropped EnsureCaches response"))?
    }

    fn reset_caches(&self) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::ResetCaches { resp: resp_tx })
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker channel closed on ResetCaches"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker dropped ResetCaches response"))?
    }

    fn clone_cache_slot(&self, src_slot: usize, dst_slot: usize) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::CloneCacheSlot {
                src_slot,
                dst_slot,
                resp: resp_tx,
            })
            .map_err(|_| {
                anyhow::anyhow!("DeepSeek rank worker channel closed on CloneCacheSlot")
            })?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker dropped CloneCacheSlot response"))?
    }

    #[cfg(test)]
    fn reset_cache_slot_for_test(&self, slot_id: usize) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::ResetCacheSlot {
                slot_id,
                resp: resp_tx,
            })
            .map_err(|_| {
                anyhow::anyhow!("DeepSeek rank worker channel closed on ResetCacheSlot")
            })?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker dropped ResetCacheSlot response"))?
    }

    fn prefill(
        &self,
        prompt_tokens: Vec<u32>,
        profile: bool,
    ) -> Result<channel::Receiver<Result<RankResult>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::Prefill {
                prompt_tokens,
                profile,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("DeepSeek rank worker channel closed on Prefill"))?;
        Ok(resp_rx)
    }

    fn shutdown(&mut self) {
        let _ = self.tx.send(RankCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for FullDirectRuntime {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

fn bind_rank_thread(ctx: &RankGpuContext) -> Result<()> {
    ctx.set_current()?;
    unsafe {
        pegainfer_kernels::ffi::cublas_init();
    }
    Ok(())
}

pub(super) fn load_full_direct_runtime(
    model_path: &Path,
    config: &'static Config,
    prefill_profile: bool,
) -> Result<FullDirectRuntime> {
    let mut contexts = Vec::with_capacity(8);
    for rank in 0..8 {
        contexts.push(RankGpuContext::new(rank)?);
    }
    let weights = contexts
        .iter()
        .enumerate()
        .map(|(rank, ctx)| {
            load_rank_to_gpu(ctx, model_path, config, TensorParallelConfig::mp8(rank))
        })
        .collect::<Result<Vec<_>>>()?;
    let weights: &'static [_] = Box::leak(weights.into_boxed_slice());
    let views = weights
        .iter()
        .map(|weights| weights.view(config))
        .collect::<Result<Vec<_>>>()?;
    let decode_streams = contexts
        .iter()
        .map(|ctx| ctx.stream.clone())
        .collect::<Vec<_>>();
    let worker_comms = cudarc::nccl::safe::Comm::from_devices(decode_streams)
        .map_err(|err| anyhow::anyhow!("decode NCCL comm creation failed: {err:?}"))?;
    let moe_streams = contexts
        .iter()
        .map(|ctx| {
            ctx.ctx
                .new_stream()
                .with_context(|| format!("create MoE NCCL stream for rank {}", ctx.device_ordinal))
        })
        .collect::<Result<Vec<_>>>()?;
    let moe_comms = cudarc::nccl::safe::Comm::from_devices(moe_streams)
        .map_err(|err| anyhow::anyhow!("decode MoE NCCL comm creation failed: {err:?}"))?;
    let mut workers = Vec::with_capacity(8);
    for ((((rank, ctx), view), comm), moe_comm) in contexts
        .iter()
        .cloned()
        .enumerate()
        .zip(views.iter().cloned())
        .zip(worker_comms.into_iter())
        .zip(moe_comms.into_iter())
    {
        workers.push(RankWorker::spawn(rank, ctx, view, comm, moe_comm, config)?);
    }
    Ok(FullDirectRuntime {
        workers,
        prefill_profile,
    })
}

fn allocate_rank_decode_caches(
    ctx: &RankGpuContext,
    config: &Config,
    max_seq_len: usize,
    request_slots: usize,
) -> Result<Vec<LayerDecodeCache>> {
    let mut caches = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        caches.push(LayerDecodeCache::zeros_with_max_seq_and_slots(
            ctx,
            config,
            layer,
            max_seq_len,
            request_slots,
        )?);
    }
    Ok(caches)
}

fn allocate_rank_rope_caches(
    ctx: &RankGpuContext,
    config: &Config,
    max_seq_len: usize,
) -> Result<Vec<DeepSeekRopeCache>> {
    let mut ropes = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        ropes.push(precompute_rope_cache(ctx, config, layer, max_seq_len)?);
    }
    Ok(ropes)
}

pub(super) fn ensure_direct_decode_caches(
    runtime: &mut FullDirectRuntime,
    config: &Config,
    max_seq_len: usize,
) -> Result<()> {
    ensure_direct_decode_caches_with_slots(runtime, config, max_seq_len, 1)
}

pub(super) fn ensure_direct_decode_batch_caches(
    runtime: &mut FullDirectRuntime,
    config: &Config,
    max_seq_len: usize,
) -> Result<()> {
    ensure_direct_decode_caches_with_slots(
        runtime,
        config,
        max_seq_len,
        DIRECT_BATCH_DECODE_CAPACITY,
    )
}

fn ensure_direct_decode_caches_with_slots(
    runtime: &mut FullDirectRuntime,
    config: &Config,
    max_seq_len: usize,
    request_slots: usize,
) -> Result<()> {
    ensure!(
        runtime.workers.len() == 8,
        "DeepSeek V4 direct runtime expects 8 rank workers"
    );
    ensure!(
        request_slots > 0,
        "DeepSeek V4 direct runtime cache request slots must be positive"
    );
    for (rank, worker) in runtime.workers.iter().enumerate() {
        worker
            .ensure_caches(max_seq_len, request_slots)
            .with_context(|| format!("ensure rank worker caches rank {rank}"))?;
        worker
            .reset_caches()
            .with_context(|| format!("reset rank worker caches rank {rank}"))?;
    }
    let _ = config;
    Ok(())
}

fn ensure_rank_worker_caches(
    ctx: &RankGpuContext,
    config: &Config,
    max_seq_len: usize,
    request_slots: usize,
    caches: &mut Vec<LayerDecodeCache>,
    ropes: &mut Vec<DeepSeekRopeCache>,
    max_cache_seq_len: &mut usize,
    cache_request_slots: &mut usize,
) -> Result<()> {
    if caches.len() != config.n_layers
        || *max_cache_seq_len < max_seq_len
        || *cache_request_slots != request_slots
    {
        *caches = allocate_rank_decode_caches(ctx, config, max_seq_len, request_slots)?;
        *ropes = allocate_rank_rope_caches(ctx, config, max_seq_len)?;
        *max_cache_seq_len = max_seq_len;
        *cache_request_slots = request_slots;
    }
    Ok(())
}

fn reset_rank_decode_caches(ctx: &RankGpuContext, caches: &mut [LayerDecodeCache]) -> Result<()> {
    for cache in caches {
        zero_cuda_slice(ctx, &mut cache.kv.data)?;
        if let Some(compressor) = cache.compressor.as_mut() {
            zero_cuda_slice(ctx, &mut compressor.kv)?;
            fill_f32_cuda_slice(ctx, &mut compressor.score, f32::NEG_INFINITY)?;
        }
        if let Some(indexer_kv) = cache.indexer_kv.as_mut() {
            zero_cuda_slice(ctx, &mut indexer_kv.data)?;
        }
        if let Some(indexer_compressor) = cache.indexer_compressor.as_mut() {
            zero_cuda_slice(ctx, &mut indexer_compressor.kv)?;
            fill_f32_cuda_slice(ctx, &mut indexer_compressor.score, f32::NEG_INFINITY)?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn reset_rank_decode_cache_slot_for_test(
    ctx: &RankGpuContext,
    caches: &mut [LayerDecodeCache],
    slot_id: usize,
) -> Result<()> {
    ensure!(
        slot_id < DIRECT_BATCH_DECODE_CAPACITY,
        "reset cache slot out of range: slot={slot_id}, capacity={DIRECT_BATCH_DECODE_CAPACITY}",
    );
    for (layer, cache) in caches.iter_mut().enumerate() {
        reset_slot_rows(
            ctx,
            &mut cache.kv.data,
            cache.kv.hidden_dim,
            cache.kv.slots / DIRECT_BATCH_DECODE_CAPACITY,
            slot_id,
            half::bf16::ZERO,
        )
        .with_context(|| format!("reset layer {layer} KV cache slot"))?;
        if let Some(compressor) = cache.compressor.as_mut() {
            let rows_per_slot = compressor.slots / DIRECT_BATCH_DECODE_CAPACITY;
            reset_slot_rows(
                ctx,
                &mut compressor.kv,
                compressor.hidden_dim,
                rows_per_slot,
                slot_id,
                0.0,
            )
            .with_context(|| format!("reset layer {layer} compressor KV slot"))?;
            reset_slot_rows(
                ctx,
                &mut compressor.score,
                compressor.hidden_dim,
                rows_per_slot,
                slot_id,
                f32::NEG_INFINITY,
            )
            .with_context(|| format!("reset layer {layer} compressor score slot"))?;
        }
        if let Some(indexer_kv) = cache.indexer_kv.as_mut() {
            reset_slot_rows(
                ctx,
                &mut indexer_kv.data,
                indexer_kv.hidden_dim,
                indexer_kv.slots / DIRECT_BATCH_DECODE_CAPACITY,
                slot_id,
                half::bf16::ZERO,
            )
            .with_context(|| format!("reset layer {layer} indexer KV slot"))?;
        }
        if let Some(indexer_compressor) = cache.indexer_compressor.as_mut() {
            let rows_per_slot = indexer_compressor.slots / DIRECT_BATCH_DECODE_CAPACITY;
            reset_slot_rows(
                ctx,
                &mut indexer_compressor.kv,
                indexer_compressor.hidden_dim,
                rows_per_slot,
                slot_id,
                0.0,
            )
            .with_context(|| format!("reset layer {layer} indexer compressor KV slot"))?;
            reset_slot_rows(
                ctx,
                &mut indexer_compressor.score,
                indexer_compressor.hidden_dim,
                rows_per_slot,
                slot_id,
                f32::NEG_INFINITY,
            )
            .with_context(|| format!("reset layer {layer} indexer compressor score slot"))?;
        }
    }
    Ok(())
}

fn clone_rank_decode_cache_slot(
    ctx: &RankGpuContext,
    config: &Config,
    caches: &mut [LayerDecodeCache],
    src_slot: usize,
    dst_slot: usize,
) -> Result<()> {
    ensure!(
        src_slot < DIRECT_BATCH_DECODE_CAPACITY && dst_slot < DIRECT_BATCH_DECODE_CAPACITY,
        "clone cache slot out of range: src={src_slot}, dst={dst_slot}, capacity={DIRECT_BATCH_DECODE_CAPACITY}",
    );
    ensure!(
        caches.len() == config.n_layers,
        "clone cache slot layer mismatch: have {}, need {}",
        caches.len(),
        config.n_layers
    );
    if src_slot == dst_slot {
        return Ok(());
    }
    for (layer, cache) in caches.iter_mut().enumerate() {
        clone_slot_rows(
            ctx,
            &mut cache.kv.data,
            cache.kv.hidden_dim,
            cache.kv.slots / DIRECT_BATCH_DECODE_CAPACITY,
            src_slot,
            dst_slot,
        )
        .with_context(|| format!("clone layer {layer} KV cache slot"))?;
        if let Some(compressor) = cache.compressor.as_mut() {
            let rows_per_slot = compressor.slots / DIRECT_BATCH_DECODE_CAPACITY;
            clone_slot_rows(
                ctx,
                &mut compressor.kv,
                compressor.hidden_dim,
                rows_per_slot,
                src_slot,
                dst_slot,
            )
            .with_context(|| format!("clone layer {layer} compressor KV slot"))?;
            clone_slot_rows(
                ctx,
                &mut compressor.score,
                compressor.hidden_dim,
                rows_per_slot,
                src_slot,
                dst_slot,
            )
            .with_context(|| format!("clone layer {layer} compressor score slot"))?;
        }
        if let Some(indexer_kv) = cache.indexer_kv.as_mut() {
            clone_slot_rows(
                ctx,
                &mut indexer_kv.data,
                indexer_kv.hidden_dim,
                indexer_kv.slots / DIRECT_BATCH_DECODE_CAPACITY,
                src_slot,
                dst_slot,
            )
            .with_context(|| format!("clone layer {layer} indexer KV slot"))?;
        }
        if let Some(indexer_compressor) = cache.indexer_compressor.as_mut() {
            let rows_per_slot = indexer_compressor.slots / DIRECT_BATCH_DECODE_CAPACITY;
            clone_slot_rows(
                ctx,
                &mut indexer_compressor.kv,
                indexer_compressor.hidden_dim,
                rows_per_slot,
                src_slot,
                dst_slot,
            )
            .with_context(|| format!("clone layer {layer} indexer compressor KV slot"))?;
            clone_slot_rows(
                ctx,
                &mut indexer_compressor.score,
                indexer_compressor.hidden_dim,
                rows_per_slot,
                src_slot,
                dst_slot,
            )
            .with_context(|| format!("clone layer {layer} indexer compressor score slot"))?;
        }
    }
    Ok(())
}

fn clone_slot_rows<T>(
    ctx: &RankGpuContext,
    data: &mut CudaSlice<T>,
    row_width: usize,
    rows_per_slot: usize,
    src_slot: usize,
    dst_slot: usize,
) -> Result<()>
where
    T: Copy + DeviceRepr,
{
    ctx.set_current()?;
    ensure!(row_width > 0, "clone cache row width must be positive");
    ensure!(
        rows_per_slot > 0,
        "clone cache rows_per_slot must be positive"
    );
    let slot_len = row_width * rows_per_slot;
    let src_start = src_slot * slot_len;
    let dst_start = dst_slot * slot_len;
    ensure!(
        src_start + slot_len <= data.len() && dst_start + slot_len <= data.len(),
        "clone cache slot copy out of range: len={}, slot_len={}, src={}, dst={}",
        data.len(),
        slot_len,
        src_slot,
        dst_slot
    );
    let mut host = ctx.stream.clone_dtoh(data)?;
    host.copy_within(src_start..src_start + slot_len, dst_start);
    ctx.stream.memcpy_htod(&host, data)?;
    Ok(())
}

#[cfg(test)]
fn reset_slot_rows<T>(
    ctx: &RankGpuContext,
    data: &mut CudaSlice<T>,
    row_width: usize,
    rows_per_slot: usize,
    slot_id: usize,
    fill: T,
) -> Result<()>
where
    T: Copy + DeviceRepr,
{
    ctx.set_current()?;
    ensure!(row_width > 0, "reset cache row width must be positive");
    ensure!(
        rows_per_slot > 0,
        "reset cache rows_per_slot must be positive"
    );
    let slot_len = row_width * rows_per_slot;
    let start = slot_id * slot_len;
    ensure!(
        start + slot_len <= data.len(),
        "reset cache slot out of range: len={}, slot_len={}, slot={}",
        data.len(),
        slot_len,
        slot_id
    );
    let mut host = ctx.stream.clone_dtoh(data)?;
    host[start..start + slot_len].fill(fill);
    ctx.stream.memcpy_htod(&host, data)?;
    Ok(())
}

fn zero_cuda_slice<T>(ctx: &RankGpuContext, slice: &mut CudaSlice<T>) -> Result<()> {
    ctx.set_current()?;
    let bytes = slice.num_bytes();
    if bytes == 0 {
        return Ok(());
    }
    let (ptr, _guard) = slice.device_ptr_mut(&ctx.stream);
    unsafe {
        cuda_result::memset_d8_async(ptr, 0, bytes, ctx.stream.cu_stream())?;
    }
    Ok(())
}

fn fill_f32_cuda_slice(ctx: &RankGpuContext, slice: &mut CudaSlice<f32>, value: f32) -> Result<()> {
    ctx.set_current()?;
    let host = vec![value; slice.len()];
    ctx.stream.memcpy_htod(&host, slice)?;
    Ok(())
}

fn run_decode_on_rank_lane(
    rank: usize,
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    ptr_cache: &crate::MoeGroupedPtrCache,
    comm: &cudarc::nccl::safe::Comm,
    moe_comm: &cudarc::nccl::safe::Comm,
    ropes: &[DeepSeekRopeCache],
    config: &Config,
    token_id: u32,
    start_pos: usize,
    caches: &mut [LayerDecodeCache],
    scratch: &mut RankDecodeScratch,
) -> Result<Option<Vec<f32>>> {
    ensure!(
        ropes.len() == config.n_layers,
        "rank {rank} rope cache layer mismatch: have {}, need {}",
        ropes.len(),
        config.n_layers
    );
    ensure!(
        caches.len() == config.n_layers,
        "rank {rank} decode cache layer mismatch: have {}, need {}",
        caches.len(),
        config.n_layers
    );

    ctx.set_current()?;
    scratch
        .token_ids(ctx, token_id)
        .with_context(|| format!("copy token_id to rank {rank}"))?;
    let token_ids = &scratch.token_ids;
    embedding_rank_local_into(
        ctx,
        config,
        weights,
        token_ids,
        1,
        &mut scratch.entry.embedding,
    )
    .with_context(|| format!("embedding rank {rank}"))?;
    all_reduce_hidden_in_place(&mut scratch.entry.embedding, comm)
        .with_context(|| format!("embedding all_reduce rank {rank}"))?;
    hc_expand_bf16_hidden_into(
        ctx,
        &scratch.entry.embedding,
        config.hc_mult,
        &mut scratch.entry.hc_expand,
    )
    .with_context(|| format!("hc_expand rank {rank}"))?;

    let mut current_hc_slot = None;
    for layer in 0..config.n_layers {
        let output_slot = current_hc_slot.map_or(0, |slot| 1 - slot);
        if let Some(input_slot) = current_hc_slot {
            let (attention_reduce_temp, attention_hc_out, layer_outputs) = (
                &mut scratch.hc_post.attention_reduce_temp,
                &mut scratch.hc_post.attention_out,
                &mut scratch.hc_post.layer_outputs,
            );
            let (hc_input, layer_out) =
                split_hc_input_output(layer_outputs, input_slot, output_slot)?;
            block_decode_rank_lane_bf16_hidden_with_scratch(
                ctx,
                weights,
                ptr_cache,
                comm,
                moe_comm,
                config,
                layer,
                hc_input,
                token_ids,
                &ropes[layer],
                start_pos,
                &mut caches[layer],
                &mut scratch.hc_pre_norm,
                &mut scratch.shared_expert,
                &mut scratch.moe_ag_rs,
                &mut scratch.attention_projection,
                &mut scratch.attention_output,
                &mut scratch.attention_index,
                &mut scratch.attention_aux,
                attention_reduce_temp,
                attention_hc_out,
                layer_out,
            )
            .with_context(|| format!("decode layer {layer} rank {rank}"))?;
        } else {
            let layer_out = scratch
                .hc_post
                .layer_outputs
                .get_mut(output_slot)
                .ok_or_else(|| anyhow::anyhow!("missing HC post output slot {output_slot}"))?;
            block_decode_rank_lane_bf16_hidden_with_scratch(
                ctx,
                weights,
                ptr_cache,
                comm,
                moe_comm,
                config,
                layer,
                &scratch.entry.hc_expand,
                token_ids,
                &ropes[layer],
                start_pos,
                &mut caches[layer],
                &mut scratch.hc_pre_norm,
                &mut scratch.shared_expert,
                &mut scratch.moe_ag_rs,
                &mut scratch.attention_projection,
                &mut scratch.attention_output,
                &mut scratch.attention_index,
                &mut scratch.attention_aux,
                &mut scratch.hc_post.attention_reduce_temp,
                &mut scratch.hc_post.attention_out,
                layer_out,
            )
            .with_context(|| format!("decode layer {layer} rank {rank}"))?;
        }
        current_hc_slot = Some(output_slot);
    }

    let final_hc = current_hc_slot
        .and_then(|slot| scratch.hc_post.layer_outputs.get(slot))
        .unwrap_or(&scratch.entry.hc_expand);
    final_logits_rank_local_bf16_hidden_into(
        ctx,
        config,
        weights,
        final_hc,
        &mut scratch.final_logits,
    )
    .with_context(|| format!("final logits rank {rank}"))?;
    gather_logits_for_sampling_into(rank, ctx, comm, weights, &mut scratch.final_logits)
        .with_context(|| format!("final logits all_gather rank {rank}"))
}

fn run_decode_batch_on_rank_lane(
    rank: usize,
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    ptr_cache: &crate::MoeGroupedPtrCache,
    comm: &cudarc::nccl::safe::Comm,
    moe_comm: &cudarc::nccl::safe::Comm,
    ropes: &[DeepSeekRopeCache],
    config: &Config,
    entries: &[DirectBatchDecodeEntry],
    caches: &mut [LayerDecodeCache],
    scratch: &mut RankDecodeScratch,
) -> Result<Option<Vec<Vec<f32>>>> {
    ensure!(
        !entries.is_empty(),
        "rank {rank} batch decode must have at least one entry"
    );
    ensure!(
        entries.len() <= DIRECT_BATCH_DECODE_CAPACITY,
        "rank {rank} batch decode entries={} exceeds capacity {DIRECT_BATCH_DECODE_CAPACITY}",
        entries.len()
    );
    ensure!(
        ropes.len() == config.n_layers,
        "rank {rank} rope cache layer mismatch: have {}, need {}",
        ropes.len(),
        config.n_layers
    );
    ensure!(
        caches.len() == config.n_layers,
        "rank {rank} decode cache layer mismatch: have {}, need {}",
        caches.len(),
        config.n_layers
    );

    ctx.set_current()?;
    let token_ids_host = entries
        .iter()
        .map(|entry| entry.token_id)
        .collect::<Vec<_>>();
    ctx.stream
        .memcpy_htod(&token_ids_host, &mut scratch.batch_token_ids)?;
    embedding_rank_local_into(
        ctx,
        config,
        weights,
        &scratch.batch_token_ids,
        entries.len(),
        &mut scratch.entry.embedding,
    )
    .with_context(|| format!("batch embedding rank {rank}"))?;
    all_reduce_hidden_in_place(&mut scratch.entry.embedding, comm)
        .with_context(|| format!("batch embedding all_reduce rank {rank}"))?;
    hc_expand_bf16_hidden_into(
        ctx,
        &scratch.entry.embedding,
        config.hc_mult,
        &mut scratch.entry.hc_expand,
    )
    .with_context(|| format!("batch hc_expand rank {rank}"))?;

    let mut current_hc_slot = None;
    for layer in 0..config.n_layers {
        let output_slot = current_hc_slot.map_or(0, |slot| 1 - slot);
        let slot_stride = caches[layer].kv.slots / DIRECT_BATCH_DECODE_CAPACITY;
        let compressed_slots = slot_stride
            .checked_sub(config.sliding_window)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "batch decode layer {layer} cache slots per request {} smaller than sliding window {}",
                    slot_stride,
                    config.sliding_window
                )
            })?;
        fill_decode_batch_metadata(
            ctx,
            config,
            layer,
            slot_stride,
            compressed_slots,
            entries,
            &mut scratch.batch_token_ids,
            &mut scratch.start_pos,
            &mut scratch.src_rows,
            &mut scratch.window_dst_rows,
            &mut scratch.window_base,
            &mut scratch.compressed_base,
            &mut scratch.compressed_len,
            &mut scratch.start_pos_host,
            &mut scratch.slot_ids_host,
        )?;
        let token_ids = &scratch.batch_token_ids;
        let batch_meta = DecodeBatchMeta {
            batch: entries.len(),
            compressed_slots,
            start_pos: &scratch.start_pos,
            src_rows: &scratch.src_rows,
            window_dst_rows: &scratch.window_dst_rows,
            window_base: &scratch.window_base,
            compressed_base: &scratch.compressed_base,
            compressed_len: &scratch.compressed_len,
            start_pos_host: &scratch.start_pos_host,
            slot_ids_host: &scratch.slot_ids_host,
        };
        if let Some(input_slot) = current_hc_slot {
            let (attention_reduce_temp, attention_hc_out, layer_outputs) = (
                &mut scratch.hc_post.attention_reduce_temp,
                &mut scratch.hc_post.attention_out,
                &mut scratch.hc_post.layer_outputs,
            );
            let (hc_input, layer_out) =
                split_hc_input_output(layer_outputs, input_slot, output_slot)?;
            block_decode_rank_lane_bf16_hidden_batch_with_scratch(
                ctx,
                weights,
                ptr_cache,
                comm,
                moe_comm,
                config,
                layer,
                hc_input,
                token_ids,
                &ropes[layer],
                &batch_meta,
                &mut caches[layer],
                &mut scratch.hc_pre_norm,
                &mut scratch.shared_expert,
                &mut scratch.moe_ag_rs,
                &mut scratch.attention_projection,
                &mut scratch.attention_output,
                &mut scratch.attention_index,
                &mut scratch.attention_aux,
                attention_reduce_temp,
                attention_hc_out,
                layer_out,
            )
            .with_context(|| format!("batch decode layer {layer} rank {rank}"))?;
        } else {
            let layer_out = scratch
                .hc_post
                .layer_outputs
                .get_mut(output_slot)
                .ok_or_else(|| anyhow::anyhow!("missing HC post output slot {output_slot}"))?;
            block_decode_rank_lane_bf16_hidden_batch_with_scratch(
                ctx,
                weights,
                ptr_cache,
                comm,
                moe_comm,
                config,
                layer,
                &scratch.entry.hc_expand,
                token_ids,
                &ropes[layer],
                &batch_meta,
                &mut caches[layer],
                &mut scratch.hc_pre_norm,
                &mut scratch.shared_expert,
                &mut scratch.moe_ag_rs,
                &mut scratch.attention_projection,
                &mut scratch.attention_output,
                &mut scratch.attention_index,
                &mut scratch.attention_aux,
                &mut scratch.hc_post.attention_reduce_temp,
                &mut scratch.hc_post.attention_out,
                layer_out,
            )
            .with_context(|| format!("batch decode layer {layer} rank {rank}"))?;
        }
        current_hc_slot = Some(output_slot);
    }

    let final_hc = current_hc_slot
        .and_then(|slot| scratch.hc_post.layer_outputs.get(slot))
        .unwrap_or(&scratch.entry.hc_expand);
    let local_logits = final_batch_logits_rank_local_bf16_hidden(ctx, config, weights, final_hc)
        .with_context(|| format!("batch final logits rank {rank}"))?;
    gather_batch_logits_for_sampling(rank, ctx, comm, weights, &local_logits)
        .with_context(|| format!("batch final logits all_gather rank {rank}"))
}

fn split_hc_input_output(
    layer_outputs: &mut [crate::HcHiddenStates],
    input_slot: usize,
    output_slot: usize,
) -> Result<(&crate::HcHiddenStates, &mut crate::HcHiddenStates)> {
    ensure!(
        input_slot != output_slot,
        "HC input/output slots must be distinct: input={input_slot}, output={output_slot}"
    );
    ensure!(
        input_slot < layer_outputs.len() && output_slot < layer_outputs.len(),
        "HC slot out of range: input={input_slot}, output={output_slot}, len={}",
        layer_outputs.len()
    );
    if input_slot < output_slot {
        let (left, right) = layer_outputs.split_at_mut(output_slot);
        Ok((&left[input_slot], &mut right[0]))
    } else {
        let (left, right) = layer_outputs.split_at_mut(input_slot);
        Ok((&right[0], &mut left[output_slot]))
    }
}

fn run_prefill_on_rank_lane(
    rank: usize,
    ctx: &RankGpuContext,
    weights: &RankWeightView<'_>,
    ptr_cache: &crate::MoeGroupedPtrCache,
    comm: &cudarc::nccl::safe::Comm,
    ropes: &[DeepSeekRopeCache],
    config: &Config,
    prompt_tokens: &[u32],
    caches: &mut [LayerDecodeCache],
    profile: bool,
) -> Result<Option<Vec<f32>>> {
    ensure!(
        !prompt_tokens.is_empty(),
        "rank {rank} prefill prompt must be non-empty"
    );
    ensure!(
        ropes.len() == config.n_layers,
        "rank {rank} rope cache layer mismatch: have {}, need {}",
        ropes.len(),
        config.n_layers
    );
    ensure!(
        caches.len() == config.n_layers,
        "rank {rank} prefill cache layer mismatch: have {}, need {}",
        caches.len(),
        config.n_layers
    );

    ctx.set_current()?;
    let seq_len = prompt_tokens.len();
    let mut profile_phases = Vec::new();
    let prefill_window_topk = PrefillWindowTopk::new(ctx, seq_len, config.sliding_window)
        .with_context(|| format!("prefill window topk rank {rank}"))?;

    let phase_start = Instant::now();
    let token_ids = ctx
        .stream
        .clone_htod(prompt_tokens)
        .with_context(|| format!("copy prompt tokens to rank {rank}"))?;
    record_prefill_profile_phase(
        ctx,
        profile,
        rank,
        &mut profile_phases,
        "token_h2d",
        None,
        None,
        phase_start,
    )?;

    let phase_start = Instant::now();
    let mut hidden = embedding_rank_local(ctx, config, weights, &token_ids, seq_len)
        .with_context(|| format!("embedding rank {rank}"))?;
    record_prefill_profile_phase(
        ctx,
        profile,
        rank,
        &mut profile_phases,
        "embedding",
        None,
        None,
        phase_start,
    )?;

    let phase_start = Instant::now();
    all_reduce_hidden_in_place(&mut hidden, comm)
        .with_context(|| format!("embedding all_reduce rank {rank}"))?;
    record_prefill_profile_phase(
        ctx,
        profile,
        rank,
        &mut profile_phases,
        "embedding_all_reduce",
        None,
        None,
        phase_start,
    )?;

    let phase_start = Instant::now();
    let mut hc = hc_expand_bf16_hidden(ctx, &hidden, config.hc_mult)
        .with_context(|| format!("hc_expand rank {rank}"))?;
    record_prefill_profile_phase(
        ctx,
        profile,
        rank,
        &mut profile_phases,
        "hc_expand",
        None,
        None,
        phase_start,
    )?;

    for layer in 0..config.n_layers {
        let phase_start = Instant::now();
        hc = block_prefill_rank_lane_bf16_hidden_with_decode_cache(
            ctx,
            weights,
            ptr_cache,
            comm,
            config,
            layer,
            &hc,
            &token_ids,
            &ropes[layer],
            0,
            &mut caches[layer],
            &prefill_window_topk,
        )
        .with_context(|| format!("prefill layer {layer} rank {rank}"))?;
        record_prefill_profile_phase(
            ctx,
            profile,
            rank,
            &mut profile_phases,
            "block_prefill",
            Some(layer),
            Some(config.compress_ratios[layer]),
            phase_start,
        )?;
    }

    let phase_start = Instant::now();
    let local_logits = final_logits_rank_local_bf16_hidden(ctx, config, weights, &hc)
        .with_context(|| format!("prefill final logits rank {rank}"))?;
    record_prefill_profile_phase(
        ctx,
        profile,
        rank,
        &mut profile_phases,
        "final_logits",
        None,
        None,
        phase_start,
    )?;

    let phase_start = Instant::now();
    let logits = gather_logits_for_sampling(rank, ctx, comm, weights, &local_logits)
        .with_context(|| format!("prefill final logits all_gather rank {rank}"))?;
    record_prefill_profile_phase(
        ctx,
        profile,
        rank,
        &mut profile_phases,
        "final_logits_all_gather",
        None,
        None,
        phase_start,
    )?;
    if profile && rank == 0 {
        info!(
            "pegainfer_prefill_profile {}",
            serde_json::json!({
                "rank": rank,
                "prompt_tokens": seq_len,
                "phases": profile_phases,
            })
        );
    }
    Ok(logits)
}

fn record_prefill_profile_phase(
    ctx: &RankGpuContext,
    profile: bool,
    rank: usize,
    phases: &mut Vec<serde_json::Value>,
    name: &str,
    layer: Option<usize>,
    compress_ratio: Option<usize>,
    started_at: Instant,
) -> Result<()> {
    if !profile || rank != 0 {
        return Ok(());
    }
    ctx.sync()?;
    phases.push(serde_json::json!({
        "name": name,
        "layer": layer,
        "compress_ratio": compress_ratio,
        "ms": started_at.elapsed().as_secs_f64() * 1000.0,
    }));
    Ok(())
}

fn final_batch_logits_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    input: &crate::HcHiddenStates,
) -> Result<F32BatchLogits> {
    ctx.set_current()?;
    let hidden = hc_head_bf16_hidden(
        ctx,
        config,
        input,
        &weights.hc_head_fn()?,
        &weights.hc_head_scale()?,
        &weights.hc_head_base()?,
    )?;
    let normed = rms_norm_bf16_hidden(ctx, &hidden, &weights.norm()?, config.rms_norm_eps)?;
    rank_local_logits_from_hidden_all(ctx, &normed, &weights.head()?)
}

fn gather_batch_logits_for_sampling(
    rank: usize,
    ctx: &RankGpuContext,
    comm: &cudarc::nccl::safe::Comm,
    weights: &RankWeightView<'_>,
    local_logits: &F32BatchLogits,
) -> Result<Option<Vec<Vec<f32>>>> {
    ctx.set_current()?;
    let world_size = weights.world_size();
    ensure!(world_size > 0, "batch logits world size must be positive");
    let gathered_len = local_logits.data.len() * world_size;
    let mut gathered = unsafe { ctx.stream.alloc(gathered_len)? };
    comm.all_gather(&local_logits.data, &mut gathered)
        .map_err(|err| anyhow::anyhow!("NCCL batch logits all-gather failed: {err:?}"))?;
    if rank != 0 {
        return Ok(None);
    }
    let host = ctx.stream.clone_dtoh(&gathered)?;
    ctx.sync()?;
    let local_vocab = local_logits.vocab_size;
    let seq_len = local_logits.seq_len;
    let mut rows = (0..seq_len)
        .map(|_| Vec::with_capacity(local_vocab * world_size))
        .collect::<Vec<_>>();
    for row in 0..seq_len {
        for rank in 0..world_size {
            let start = rank * seq_len * local_vocab + row * local_vocab;
            rows[row].extend_from_slice(&host[start..start + local_vocab]);
        }
    }
    Ok(Some(rows))
}

fn gather_logits_for_sampling(
    rank: usize,
    ctx: &RankGpuContext,
    comm: &cudarc::nccl::safe::Comm,
    weights: &RankWeightView<'_>,
    local_logits: &F32Logits,
) -> Result<Option<Vec<f32>>> {
    let gathered = all_gather_logits(ctx, comm, local_logits, weights.world_size())?;
    if rank == 0 {
        Ok(Some(gathered.to_host(ctx)?))
    } else {
        Ok(None)
    }
}

fn gather_logits_for_sampling_into(
    rank: usize,
    ctx: &RankGpuContext,
    comm: &cudarc::nccl::safe::Comm,
    weights: &RankWeightView<'_>,
    scratch: &mut FinalLogitsScratch,
) -> Result<Option<Vec<f32>>> {
    all_gather_logits_into(
        ctx,
        comm,
        &scratch.local_logits,
        weights.world_size(),
        &mut scratch.gathered_logits,
    )?;
    if rank == 0 {
        Ok(Some(scratch.gathered_logits.to_host(ctx)?))
    } else {
        Ok(None)
    }
}

pub(super) fn run_direct_decode_logits(
    runtime: &mut FullDirectRuntime,
    token_id: u32,
    start_pos: usize,
) -> Result<Vec<f32>> {
    ensure!(
        runtime.workers.len() == 8,
        "DeepSeek V4 direct decode expects 8 workers"
    );
    let rank_count = runtime.workers.len();
    let pending = runtime
        .workers
        .iter()
        .enumerate()
        .map(|(rank, worker)| {
            worker
                .decode(token_id, start_pos)
                .with_context(|| format!("dispatch decode rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut results = Vec::with_capacity(rank_count);
    for recv in pending {
        results.push(
            recv.recv()
                .map_err(|_| anyhow::anyhow!("DeepSeek rank worker dropped decode response"))??,
        );
    }
    results.sort_by_key(|(rank, _)| *rank);

    rank0_logits(results)
}

#[allow(dead_code)] // PR A lands the runtime batch path before scheduler wiring.
pub(super) fn run_direct_decode_batch_logits(
    runtime: &mut FullDirectRuntime,
    entries: &[DirectBatchDecodeEntry],
) -> Result<Vec<Vec<f32>>> {
    ensure!(
        runtime.workers.len() == 8,
        "DeepSeek V4 direct batch decode expects 8 workers"
    );
    ensure!(
        !entries.is_empty(),
        "DeepSeek V4 direct batch decode entries must not be empty"
    );
    ensure!(
        entries.len() <= DIRECT_BATCH_DECODE_CAPACITY,
        "DeepSeek V4 direct batch decode capacity exceeded: entries={}, capacity={DIRECT_BATCH_DECODE_CAPACITY}",
        entries.len()
    );
    let rank_count = runtime.workers.len();
    let pending = runtime
        .workers
        .iter()
        .enumerate()
        .map(|(rank, worker)| {
            worker
                .decode_batch(entries.to_vec())
                .with_context(|| format!("dispatch batch decode rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut results = Vec::with_capacity(rank_count);
    for recv in pending {
        results.push(recv.recv().map_err(|_| {
            anyhow::anyhow!("DeepSeek rank worker dropped batch decode response")
        })??);
    }
    results.sort_by_key(|(rank, _)| *rank);
    rank0_batch_logits(results)
}

pub(super) fn clone_direct_decode_cache_slot(
    runtime: &mut FullDirectRuntime,
    src_slot: usize,
    dst_slot: usize,
) -> Result<()> {
    ensure!(
        runtime.workers.len() == 8,
        "DeepSeek V4 direct cache clone expects 8 workers"
    );
    for (rank, worker) in runtime.workers.iter().enumerate() {
        worker
            .clone_cache_slot(src_slot, dst_slot)
            .with_context(|| format!("clone rank worker cache slot rank {rank}"))?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn reset_direct_decode_cache_slot_for_test(
    runtime: &mut FullDirectRuntime,
    slot_id: usize,
) -> Result<()> {
    ensure!(
        runtime.workers.len() == 8,
        "DeepSeek V4 direct cache slot reset expects 8 workers"
    );
    for (rank, worker) in runtime.workers.iter().enumerate() {
        worker
            .reset_cache_slot_for_test(slot_id)
            .with_context(|| format!("reset rank worker cache slot rank {rank}"))?;
    }
    Ok(())
}

pub(super) fn run_prefill_logits_and_seed_decode_cache(
    runtime: &mut FullDirectRuntime,
    config: &Config,
    prompt_tokens: &[u32],
) -> Result<Vec<f32>> {
    ensure!(
        runtime.workers.len() == 8,
        "DeepSeek V4 direct prefill expects 8 workers"
    );
    ensure!(
        prompt_tokens.len() <= config.max_position_embeddings,
        "prompt length {} exceeds max position embeddings {}",
        prompt_tokens.len(),
        config.max_position_embeddings
    );
    let pending = runtime
        .workers
        .iter()
        .enumerate()
        .map(|(rank, worker)| {
            worker
                .prefill(prompt_tokens.to_vec(), runtime.prefill_profile)
                .with_context(|| format!("dispatch prefill rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut results = Vec::with_capacity(runtime.workers.len());
    for recv in pending {
        results.push(
            recv.recv()
                .map_err(|_| anyhow::anyhow!("DeepSeek rank worker dropped prefill response"))??,
        );
    }
    results.sort_by_key(|(rank, _)| *rank);
    rank0_logits(results)
}

fn rank0_logits(results: Vec<RankResult>) -> Result<Vec<f32>> {
    results
        .into_iter()
        .find_map(|(rank, logits)| (rank == 0).then_some(logits))
        .flatten()
        .ok_or_else(|| anyhow::anyhow!("rank 0 did not return gathered logits"))
}

#[allow(dead_code)] // PR A lands the runtime batch path before scheduler wiring.
fn rank0_batch_logits(results: Vec<RankBatchResult>) -> Result<Vec<Vec<f32>>> {
    results
        .into_iter()
        .find_map(|(rank, logits)| (rank == 0).then_some(logits))
        .flatten()
        .ok_or_else(|| anyhow::anyhow!("rank 0 did not return gathered batch logits"))
}
