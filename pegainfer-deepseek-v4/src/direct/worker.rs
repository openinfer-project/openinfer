use std::{path::Path, thread};

use anyhow::{Context, Result, ensure};
use crossbeam_channel as channel;
use cudarc::driver::{CudaSlice, DevicePtrMut, result as cuda_result};

use crate::{
    Config, DeepSeekRopeCache, F32Logits, LayerDecodeCache, RankGpuContext, RankWeightView,
    TensorParallelConfig, all_reduce_hidden_in_place, block_decode_rank_lane_bf16_hidden,
    build_moe_expert_ptr_cache, embedding_rank_local, final_logits_rank_local_bf16_hidden,
    hc_expand_bf16_hidden, load_rank_to_gpu, precompute_rope_cache,
    runtime::{all_gather_logits, block_prefill_rank_lane_bf16_hidden_with_decode_cache},
};

type RankResult = (usize, Option<Vec<f32>>);

pub(super) struct FullDirectRuntime {
    workers: Vec<RankWorker>,
}

enum RankCommand {
    // TODO: cache sizing/lifecycle should be decided by the scheduler, with all
    // 8 ranks applying the same plan. Direct runtime keeps this command only as
    // a temporary bridge while DeepSeek V4 still uses its direct generator.
    EnsureCaches {
        max_seq_len: usize,
        resp: channel::Sender<Result<()>>,
    },
    ResetCaches {
        resp: channel::Sender<Result<()>>,
    },
    Prefill {
        prompt_tokens: Vec<u32>,
        resp: channel::Sender<Result<RankResult>>,
    },
    Decode {
        token_id: u32,
        start_pos: usize,
        resp: channel::Sender<Result<RankResult>>,
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

impl RankWorker {
    fn spawn(
        rank: usize,
        ctx: RankGpuContext,
        weights: RankWeightView<'static>,
        comm: cudarc::nccl::safe::Comm,
        config: &'static Config,
    ) -> Result<Self> {
        let (tx, rx) = channel::unbounded();
        let (startup_tx, startup_rx) = channel::bounded(1);
        let comm = OwnedRankComm(comm);
        let handle = thread::Builder::new()
            .name(format!("deepseek-v4-rank-{rank}"))
            .spawn(move || {
                pin_rank_worker_thread(rank);
                let mut ropes = Vec::new();
                let mut caches = Vec::new();
                let mut max_cache_seq_len = 0usize;
                let startup = bind_rank_thread(&ctx)
                    .and_then(|()| build_moe_expert_ptr_cache(&ctx, config, &weights));
                match startup {
                    Ok(ptr_cache) => {
                        let _ = startup_tx.send(Ok(()));
                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                RankCommand::EnsureCaches { max_seq_len, resp } => {
                                    let result = ensure_rank_worker_caches(
                                        &ctx,
                                        config,
                                        max_seq_len,
                                        &mut caches,
                                        &mut ropes,
                                        &mut max_cache_seq_len,
                                    );
                                    let _ = resp.send(result);
                                }
                                RankCommand::ResetCaches { resp } => {
                                    let result = reset_rank_decode_caches(&ctx, &mut caches);
                                    let _ = resp.send(result);
                                }
                                RankCommand::Prefill {
                                    prompt_tokens,
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
                                        &ropes,
                                        config,
                                        token_id,
                                        start_pos,
                                        &mut caches,
                                    )
                                    .map(|logits| (rank, logits));
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

    fn ensure_caches(&self, max_seq_len: usize) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::EnsureCaches {
                max_seq_len,
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

    fn prefill(&self, prompt_tokens: Vec<u32>) -> Result<channel::Receiver<Result<RankResult>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(RankCommand::Prefill {
                prompt_tokens,
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

#[cfg(target_os = "linux")]
fn pin_rank_worker_thread(rank: usize) {
    unsafe {
        let mut allowed: libc::cpu_set_t = std::mem::zeroed();
        let set_size = std::mem::size_of::<libc::cpu_set_t>();
        if libc::sched_getaffinity(0, set_size, &mut allowed) != 0 {
            log::warn!("failed to read CPU affinity mask for DeepSeek rank worker {rank}");
            return;
        }

        let mut cpus = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &allowed) {
                cpus.push(cpu);
            }
        }
        if cpus.is_empty() {
            log::warn!("empty CPU affinity mask for DeepSeek rank worker {rank}");
            return;
        }

        let cpu = cpus[rank % cpus.len()];
        let mut target: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut target);
        libc::CPU_SET(cpu, &mut target);
        let rc = libc::pthread_setaffinity_np(libc::pthread_self(), set_size, &target);
        if rc != 0 {
            log::warn!(
                "failed to pin DeepSeek rank worker {rank} to CPU {cpu}: pthread_setaffinity_np rc={rc}"
            );
        } else {
            log::info!("pinned DeepSeek rank worker {rank} to CPU {cpu}");
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_rank_worker_thread(_rank: usize) {}

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
    let mut workers = Vec::with_capacity(8);
    for (((rank, ctx), view), comm) in contexts
        .iter()
        .cloned()
        .enumerate()
        .zip(views.iter().cloned())
        .zip(worker_comms.into_iter())
    {
        workers.push(RankWorker::spawn(rank, ctx, view, comm, config)?);
    }
    Ok(FullDirectRuntime { workers })
}

fn allocate_rank_decode_caches(
    ctx: &RankGpuContext,
    config: &Config,
    max_seq_len: usize,
) -> Result<Vec<LayerDecodeCache>> {
    let mut caches = Vec::with_capacity(config.n_layers);
    for layer in 0..config.n_layers {
        caches.push(LayerDecodeCache::zeros_with_max_seq(
            ctx,
            config,
            layer,
            max_seq_len,
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
    ensure!(
        runtime.workers.len() == 8,
        "DeepSeek V4 direct runtime expects 8 rank workers"
    );
    for (rank, worker) in runtime.workers.iter().enumerate() {
        worker
            .ensure_caches(max_seq_len)
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
    caches: &mut Vec<LayerDecodeCache>,
    ropes: &mut Vec<DeepSeekRopeCache>,
    max_cache_seq_len: &mut usize,
) -> Result<()> {
    if caches.len() != config.n_layers || *max_cache_seq_len < max_seq_len {
        *caches = allocate_rank_decode_caches(ctx, config, max_seq_len)?;
        *ropes = allocate_rank_rope_caches(ctx, config, max_seq_len)?;
        *max_cache_seq_len = max_seq_len;
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
    ropes: &[DeepSeekRopeCache],
    config: &Config,
    token_id: u32,
    start_pos: usize,
    caches: &mut [LayerDecodeCache],
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
    let token_ids = ctx
        .stream
        .clone_htod(&[token_id])
        .with_context(|| format!("copy token_id to rank {rank}"))?;
    let mut hidden = embedding_rank_local(ctx, config, weights, &token_ids, 1)
        .with_context(|| format!("embedding rank {rank}"))?;
    all_reduce_hidden_in_place(&mut hidden, comm)
        .with_context(|| format!("embedding all_reduce rank {rank}"))?;
    let mut hc = hc_expand_bf16_hidden(ctx, &hidden, config.hc_mult)
        .with_context(|| format!("hc_expand rank {rank}"))?;

    for layer in 0..config.n_layers {
        hc = block_decode_rank_lane_bf16_hidden(
            ctx,
            weights,
            ptr_cache,
            comm,
            config,
            layer,
            &hc,
            &token_ids,
            &ropes[layer],
            start_pos,
            &mut caches[layer],
        )
        .with_context(|| format!("decode layer {layer} rank {rank}"))?;
    }

    let local_logits = final_logits_rank_local_bf16_hidden(ctx, config, weights, &hc)
        .with_context(|| format!("final logits rank {rank}"))?;
    gather_logits_for_sampling(rank, ctx, comm, weights, &local_logits)
        .with_context(|| format!("final logits all_gather rank {rank}"))
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
    let token_ids = ctx
        .stream
        .clone_htod(prompt_tokens)
        .with_context(|| format!("copy prompt tokens to rank {rank}"))?;
    let mut hidden = embedding_rank_local(ctx, config, weights, &token_ids, seq_len)
        .with_context(|| format!("embedding rank {rank}"))?;
    all_reduce_hidden_in_place(&mut hidden, comm)
        .with_context(|| format!("embedding all_reduce rank {rank}"))?;
    let mut hc = hc_expand_bf16_hidden(ctx, &hidden, config.hc_mult)
        .with_context(|| format!("hc_expand rank {rank}"))?;

    for layer in 0..config.n_layers {
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
        )
        .with_context(|| format!("prefill layer {layer} rank {rank}"))?;
    }

    let local_logits = final_logits_rank_local_bf16_hidden(ctx, config, weights, &hc)
        .with_context(|| format!("prefill final logits rank {rank}"))?;
    gather_logits_for_sampling(rank, ctx, comm, weights, &local_logits)
        .with_context(|| format!("prefill final logits all_gather rank {rank}"))
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
                .prefill(prompt_tokens.to_vec())
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
